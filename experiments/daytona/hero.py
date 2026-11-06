"""Continue the two hero runs (hero-1v1-cont1h, hero-ffa4) on Daytona from their Modal checkpoints.

    python experiments/daytona/hero.py plan                      # resources, $/h, expected samples and cost (offline)
    python experiments/daytona/hero.py hub                       # the shared CPU sandbox that holds /vol state
    python experiments/daytona/hero.py publish                   # git HEAD + engine wheel -> /vol/code/<sha>
    python experiments/daytona/hero.py launch RUN [--hours 8] [--actors N] [--dry-run]
    python experiments/daytona/hero.py keep RUN [--max-usd X]    # keeper loop: preemptions, ladders, budget, teardown
    python experiments/daytona/hero.py eval RUN [--final]        # one ladder round / four-seat table now
    python experiments/daytona/hero.py status RUN
    python experiments/daytona/hero.py stop RUN                  # early stop: the learner writes final.npz, actors exit
    python experiments/daytona/hero.py down RUN                  # delete every sandbox of the run
    python experiments/daytona/hero.py restart RUN               # hotfix: live sandboxes onto the published HEAD
    python experiments/daytona/hero.py sync-ckpt --src HEAD_NPZ --dest /results/...npz   # a Curio checkpoint -> the volume
    python experiments/daytona/hero.py pool-probe RUN            # trunk-block pooling bias vs convolution on self-play positions
    python experiments/daytona/hero.py tests --paths a.py,b.py   # pytest, one 4-vCPU sandbox per file, on the published HEAD

Layout (Daytona volume `hero-runs`, mounted at /vol; /results and /retained are symlinks into it):
  /vol/results/distributed/<run>/   the run folder (latest.npz with Adam state, critic, training.jsonl, snapshots),
                                    staged from Modal on 2026-10-28 and synced back by the learner every checkpoint
  /vol/results/..., /vol/retained/  ladder / table references, at their Modal paths
  /vol/code/<sha>/                  code.tar.gz + engine-rs wheel (one build, every sandbox pip-installs it)
  /vol/runs/<run>/                  plan.json, STOP, learner-done.json, ladder history
Networking: sandboxes have no private network (2026-10-28 probe: different hosts, NAT). Actors reach the learner
through an SSH-gateway local forward (ssh <token>@ssh.app.daytona.io, -L 7070:127.0.0.1:7070), ~120 MB/s per
pair measured; the runs need ~20 MB/s (1v1) and ~13 MB/s (4p) in total.
GPU sandboxes are deleted when they stop, spot ones can be preempted without warning: the keeper replaces actors and
resumes a lost learner from the volume. Every sandbox has autoStop off and a wall-clock TTL, so billing ends at the
TTL even if the keeper (this Mac) is asleep.
Auth: DAYTONA_API_KEY (+ DAYTONA_ORGANIZATION_ID) from the environment, else the `daytona login` profile.
"""
import argparse
import http.client as http_client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER = Path(os.environ.get('HERO_LEDGER') or ROOT / 'runs' / 'hero-daytona')   # hero-hub keeper: /workspace/ledger
KEY_FILE = Path(os.environ.get('DAYTONA_API_KEY_FILE', '/workspace/.daytona-api-key'))   # on hero-hub, written by the user
VOLUME = 'hero-runs'
HUB = 'hero-hub'
ENV_PREFIX = ('export PATH=/opt/venv/bin:/opt/cargo/bin:$PATH RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo '
              'PYTHONPATH=/workspace/src XLA_PYTHON_CLIENT_PREALLOCATE=false; ')
LEARNER_ENV = dict(POLYTOPIA_MATMUL_PRECISION='DEFAULT', POLYTOPIA_HEAD_PRECISION='HIGHEST',
                   POLYTOPIA_TRUNK_DTYPE='bfloat16', XLA_PYTHON_CLIENT_MEM_FRACTION='.95', JAX_PLATFORMS='cuda',
                   JAX_COMPILATION_CACHE_DIR='/workspace/jax-cache')
# (NCCL_P2P_DISABLE=1 was tried against the 4p KL ~1 problem, did not help, and a 1v1 learner running with it hung
# in a collective after 329 updates, 3 GPUs at 100% and one idle, 2026-10-28: the keeper now recycles stalled learners)
# Daytona list prices (daytona.io/pricing, read 2026-10-28), USD per hour, billed per second. GPU rates are the
# preemptible (spot) column; on-demand needs GPU credits in the Wallet (the org has none: create returns HTTP 400).
PRICE = dict(vcpu=0.0504, mem_gib=0.0162, disk_gib=0.000108,
             gpu={'B300': 4.08, 'B200': 3.59, 'H200': 2.61, 'H100': 2.27, 'RTX-PRO-6000': 1.74, 'RTX-5090': 0.74,
                  'RTX-4090': 0.57, 'MI355X': 3.44})
CHAMP = '/retained/checkpoints/distributed-final-v1/scratch-cont-2-final/final.npz'
BN4 = '/results/distributed/bn-main4/final.npz'
C3 = '/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz'
HERO1 = '/results/distributed/hero-1v1/final.npz'
# Sized from the 2026-10-28 Daytona smokes and benches (docs/hero-daytona-2026-10-28.md): RTX PRO 6000 hosts have
# Zen 5 CPUs, so their actor sandboxes run 1v1 at ~29.5k/s live (4 actors fed a 4x PRO 6000 learner 118k/s) and 4p at
# ~28k/s bench (~20k/s assumed live; 4p needs >128 GiB); RTX 4090 hosts (Zen 3) ran 4p at 6.3k/s live. Learner caps
# are live: 4x PRO 6000 trains a 262k 1v1 batch in 1.86 s (~125k/s with assembly) and a 131k 4p batch in 1.91 s.
PLANS = {
    'hero-1v1-cont1h': dict(
        config='experiments/hero/hero-1v1-cont1h.json', players=2, start_update=11650, start_elapsed=26853.99,
        learner=dict(gpu=4, gpu_type=['RTX-PRO-6000'], cpu=64, memory=128, disk=100),
        actors=5, actor=dict(gpu=1, gpu_type=['RTX-PRO-6000', 'RTX-5090'], cpu=16, memory=64, disk=40),
        processes=48, games=24, bench_per_actor=29500, learner_cap=125000,
        eval=dict(kind='ladder', interval=7200, maps=32, maps_per_box=8, final_maps=64, seed_base=520000,
                  references=dict(champion=CHAMP, bn4=BN4, c3=C3, hero1=HERO1,
                                  start='/results/distributed/ladder/hero-1v1-cont1h-update-11650.npz'))),
    'hero-ffa4': dict(
        config='experiments/hero/hero-ffa4.json', players=4, start_update=11550, start_elapsed=26051.61,
        learner=dict(gpu=4, gpu_type=['RTX-PRO-6000'], cpu=64, memory=128, disk=100),
        # 4p actors on RTX PRO 6000 hosts (Zen 5) produced records the learner disagreed with: KL ~1 from the first
        # update, four times on 2026-10-28, bf16 and fp32 serving alike (1v1 on the same hosts is fine). The one clean
        # 4p smoke used RTX 4090 actors (Zen 3 hosts, 6.3k/s each), so 4p runs on those.
        actors=10, actor=dict(gpu=1, gpu_type=['RTX-4090'], cpu=16, memory=112, disk=40),
        processes=48, games=24, bench_per_actor=6300, learner_cap=65000,
        eval=dict(kind='four', interval=7200, maps=100, parts=10, final_maps=200, seed_base=530000,
                  references=dict(START='/results/four-player/monitor/ffa4-big2/update-7650.npz',
                                  CHAMP='/results/four-player/parents/champ-ffa4-2x64/final.npz',
                                  PREV='/results/four-player/monitor/hero-ffa4/update-11550.npz'))),
}
FARSIGHT = '/results/distributed/ladder/farsight-latest.npz'   # hero.py sync-farsight copies Curio's latest here
# any other Curio run synced by hero.py sync-farsight --run NAME (bighero: the plain 4x128 conv hero on obs v2 that
# the Curio cluster resumed after farsight was stopped at update 5660 on 2026-11-01 18:26 CST)
CURIO_REF = '/results/distributed/ladder/curio-{run}-latest.npz'
PLANS['lookout-1v1'] = dict(
    # From scratch (no run folder, no start reference): a random-init hybrid transformer (2 conv blocks + 3 attention
    # blocks, width 128: hero's parameter count) on obs v3, written on the hub by node.py hero_random_parent. Its
    # ladder meets farsight (the conv + pooling + history run on Curio, synced by hand) when the file exists.
    config='experiments/hero/lookout-1v1.json', players=2,
    scratch=dict(size=14, blocks=2, width=128, attention=3, obs_version=3, seed=1),
    learner=dict(gpu=4, gpu_type=['RTX-PRO-6000'], cpu=64, memory=128, disk=100),
    actors=5, actor=dict(gpu=1, gpu_type=['RTX-PRO-6000', 'RTX-5090'], cpu=16, memory=64, disk=40),
    processes=48, games=24, bench_per_actor=29500, learner_cap=125000,
    eval=dict(kind='ladder', interval=7200, maps=32, maps_per_box=8, final_maps=64, seed_base=540000,
              references=dict(hero1=HERO1, champion=CHAMP, c3=C3, farsight=FARSIGHT, bighero=CURIO_REF.format(run='bighero')),
              # live references: another run's folder on this volume; its latest is frozen on the hub at round time
              live=dict(farsight_v2='/results/distributed/farsight-v2-1v1')))
BIGHERO_U3880 = '/results/distributed/ladder/bighero-u3880.npz'   # hero.py sync-ckpt from the Curio head's ckpt/bighero-u3880
PLANS['farsight-v2-1v1'] = dict(
    # From scratch: farsight's architecture with the trunk-block pooling redesigned (network.BLOCK_POOL_LAYOUTS: layout 2,
    # gated and normalized, blocks 1 and 3 of 4) on obs v3, the bighero recipe (experiments/curio/configs/
    # hero-1v1-scratch.json: no peace, a capped game is a loss for both seats, lr 2e-4 + kl_adapt_target 0.01) on
    # Daytona's platform knobs. Ladder: the Modal references, bighero's u3880 net (the obs-v2 control run, equal
    # updates) and farsight's synced checkpoint (layout 1). docs/obs-v3-history-2026-11-01.md.
    config='experiments/hero/farsight-v2-1v1.json', players=2,
    scratch=dict(size=14, blocks=4, width=128, block_pool=2, obs_version=3, seed=1),
    learner=dict(gpu=4, gpu_type=['RTX-PRO-6000'], cpu=64, memory=128, disk=100),
    actors=5, actor=dict(gpu=1, gpu_type=['RTX-PRO-6000', 'RTX-5090'], cpu=16, memory=64, disk=40),
    processes=48, games=24, bench_per_actor=29500, learner_cap=125000,
    eval=dict(kind='ladder', interval=7200, maps=32, maps_per_box=8, final_maps=64, seed_base=550000,
              references=dict(c3=C3, bn4=BN4, champion=CHAMP, hero1=HERO1, bighero_u3880=BIGHERO_U3880, farsight=FARSIGHT,
                              # equal-update bighero snapshots (synced Nov 2 from Curio runs/bighero/snapshots) for round 3 (~u5200) and the final (~u7700)
                              bighero_u5250='/results/distributed/ladder/bighero-u5250.npz',
                              bighero_u7750='/results/distributed/ladder/bighero-u7750.npz',
                              bighero_u10000='/results/distributed/ladder/bighero-u10000.npz')))
PLANS['armc'] = dict(
    # Arm C continued from Curio (2026-11-02 evening): bighero's from-scratch recipe with hero1 + hero_u11650 (old-rules
    # developers widened to obs v2, exact) as frozen opponents at 15% each; the run folder /vol/results/distributed/armc
    # is the Curio run dir synced after STOP-ALL at update 3920 (hero.py sync-run). Ladder: the Modal bots, bighero's
    # equal-update snapshots and its final net.
    config='experiments/hero/armc-daytona.json', players=2, start_update=3920,
    learner=dict(gpu=4, gpu_type=['RTX-PRO-6000'], cpu=64, memory=128, disk=100),
    actors=5, actor=dict(gpu=1, gpu_type=['RTX-PRO-6000', 'RTX-5090'], cpu=16, memory=64, disk=40),
    processes=48, games=24, bench_per_actor=29500, learner_cap=125000,
    eval=dict(kind='ladder', interval=7200, maps=32, maps_per_box=16, final_maps=64, seed_base=570000,
              references=dict(c3=C3, champion=CHAMP, hero1=HERO1, bighero_u3880=BIGHERO_U3880,
                              bighero_u5250='/results/distributed/ladder/bighero-u5250.npz',
                              bighero_u7750='/results/distributed/ladder/bighero-u7750.npz',
                              bighero_u10000='/results/distributed/ladder/bighero-u10000.npz')))
PLANS['farsight-v2-capfix'] = dict(
    # Arm B of the score-at-cap A/B (docs/capfix-arm-b-brief-2026-11-02.md): farsight-v2-1v1's architecture and recipe
    # with the cap rule changed (cap_score_reward 1.0, truncation_reward None), from the same random parent. Ladder:
    # equal-update references of the SAME architecture under the old rule (farsight-v2-1v1's frozen ladder copies),
    # bighero's snapshots, and the Modal bots; 16 maps per box (2 boxes/ref, 24 boxes) so a round fits the org's
    # 250-vCPU cap next to the learner (64) and 5 actors (80).
    config='experiments/hero/farsight-v2-capfix.json', players=2,
    scratch=dict(size=14, blocks=4, width=128, block_pool=2, obs_version=3, seed=1),
    learner=dict(gpu=4, gpu_type=['RTX-PRO-6000'], cpu=64, memory=128, disk=100),
    actors=5, actor=dict(gpu=1, gpu_type=['RTX-PRO-6000', 'RTX-5090'], cpu=16, memory=64, disk=40),
    processes=48, games=24, bench_per_actor=29500, learner_cap=125000,
    eval=dict(kind='ladder', interval=7200, maps=32, maps_per_box=16, final_maps=64, seed_base=560000,
              # segment 2 (Nov 3, resumed from u13014): the equal-sample question vs the plain trunk under the same cap
              # rule. hero1 = hero-1v1 final (3.9 B), hero1_u11500 = its continuation near 6.9 B (Arm B reaches ~6.9 B at
              # the end of this segment), hero1_final = the continuation's last checkpoint (u17840); self = Arm B at the
              # start of the segment. 8 references x 2 boxes = 16 boxes per round.
              references=dict(c3=C3, champion=CHAMP, hero1=HERO1,
                              hero1_u11500='/results/distributed/hero-1v1-cont1h/snapshots/update-11500.npz',
                              hero1_final='/results/distributed/hero-1v1-cont1h/final.npz',
                              bighero_u10000='/results/distributed/ladder/bighero-u10000.npz',
                              v2_final='/results/distributed/ladder/farsight-v2-1v1-update-9779-final.npz',
                              self_u13014='/results/distributed/ladder/farsight-v2-capfix-update-13014-final.npz')))
for _run in list(PLANS):   # smoke-<run>: a scratch copy of the run folder, 2 actors, no evaluation
    PLANS['smoke-' + _run] = dict(PLANS[_run], actors=2, **({} if PLANS[_run].get('scratch') else dict(source=_run)))
sys.path.insert(0, str(HERE))
import sweep   # noqa: E402  the iso-FLOP sweep's arms (experiments/daytona/sweep.py): fresh runs with their own config
PLANS.update(sweep.plans())
EVAL_BOX = dict(cpu=4, memory=8, disk=10)   # the org's CPU sandbox limit is 4 vCPU / 8 GiB (250 vCPU in all)
SERVE = dict(trunk_dtype='bfloat16', server_cores=2)   # main's launch default (2 pinned cores per inference server)
ACTOR_ENV = dict(JAX_PLATFORMS='cpu', JAX_COMPILATION_CACHE_DIR='/workspace/jax-cache')   # the server picks cuda itself


# ---- Daytona REST
def _auth():
    key = os.environ.get('DAYTONA_API_KEY') or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else '')
    if key:
        return ('https://app.daytona.io/api', key, os.environ.get('DAYTONA_ORGANIZATION_ID'))
    try:   # refreshes an expired CLI token; the CLI can hang (2026-10-28 22:00), then the stored token is tried as is
        subprocess.run(['daytona', 'organization', 'list'], capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        pass
    c = json.loads((Path.home() / 'Library/Application Support/daytona/config.json').read_text())
    p = next(p for p in c['profiles'] if p['id'] == c['activeProfile'])
    return p['api']['url'], p['api']['token']['accessToken'], p['activeOrganizationId']


def http(method, url, body=None, timeout=120, retries=5):
    base, token, org = _auth()
    url = url if url.startswith('http') else base + url
    headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
    if org:
        headers['X-Daytona-Organization-ID'] = org
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data, headers, method=method), timeout=timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            if e.code >= 500 and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f'{method} {url}: HTTP {e.code} {e.read()[:400]!r}')
        except (urllib.error.URLError, TimeoutError, ConnectionError, http_client.IncompleteRead):
            if attempt == retries - 1:   # the Mac's network drops TLS handshakes and truncates long replies
                raise
            time.sleep(5 * (attempt + 1))


def sandboxes(**labels):
    rows = http('GET', '/sandbox?labels=' + urllib.request.quote(json.dumps(dict(polytopia='hero', **labels))))
    return rows['items'] if isinstance(rows, dict) and 'items' in rows else rows


def find(name):
    return next((s for s in sandboxes() if s.get('name') == name), None)


def volume_id():
    return next(v['id'] for v in http('GET', '/volumes') if v['name'] == VOLUME)


def create(name, labels, cpu, memory, disk, ttl_minutes, gpu=0, gpu_type=None, spot=True, wait=1200):
    body = dict(name=name, buildInfo=dict(dockerfileContent=(HERE / 'Dockerfile').read_text(), contextHashes=[]),
                cpu=cpu, memory=memory, disk=disk, labels=dict(polytopia='hero', **labels), autoStopInterval=0,
                autoArchiveInterval=0, autoDeleteInterval=0, ttlMinutes=int(ttl_minutes),
                volumes=[dict(volumeId=volume_id(), mountPath='/vol')])
    if gpu:
        body.update(gpu=gpu, gpuType=list(gpu_type), spot=bool(spot))
    try:
        s = http('POST', '/sandbox', body)
    except RuntimeError as error:   # a dropped reply retried the POST and the first one had landed (2026-11-01)
        if 'HTTP 409' not in str(error) or find(name) is None:
            raise
        s = find(name)
    deadline = time.time() + wait
    while s.get('state') not in ('started', 'error', 'build_failed') and time.time() < deadline:
        time.sleep(5)
        s = http('GET', f"/sandbox/{s['id']}")
    if s.get('state') != 'started':
        raise RuntimeError(f"{name}: {s.get('state')} {s.get('errorReason')}")
    return s


def delete(s):
    try:
        http('DELETE', f"/sandbox/{s['id']}")
    except RuntimeError as e:
        print(f"delete {s.get('name')}: {e}", file=sys.stderr)


def sh(s, command, timeout=600, check=True, retries=5):
    """Run bash in the sandbox through the toolbox proxy; returns stdout+stderr. A dropped reply is retried, so
    `command` must be safe to run twice unless retries=1 (start() does that for anything that launches a process)."""
    url = f"{s['toolboxProxyUrl']}/{s['id']}/process/execute"
    out = http('POST', url, dict(command='bash -c ' + _quote(ENV_PREFIX + command), timeout=timeout), timeout=timeout + 60,
               retries=retries)
    if check and out.get('exitCode'):
        raise RuntimeError(f"{s.get('name')}: exit {out.get('exitCode')}: {command[:120]}\n{out.get('result', '')[-3000:]}")
    return out.get('result', '')


def _quote(text):
    return "'" + text.replace("'", "'\"'\"'") + "'"


def put(s, path, data: bytes, retries=4):
    """Write a file in the sandbox (toolbox multipart upload; a path under /vol lands on the volume)."""
    import uuid
    base, token, org = _auth()
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="f"\r\n'
            'Content-Type: application/octet-stream\r\n\r\n').encode() + data + f'\r\n--{b}--\r\n'.encode()
    headers = {'Authorization': 'Bearer ' + token, 'Content-Type': f'multipart/form-data; boundary={b}'}
    if org:
        headers['X-Daytona-Organization-ID'] = org
    target = f'/tmp/upload-{b}' if path.startswith('/vol/') else path   # the S3 mount takes no renames
    url = f"{s['toolboxProxyUrl']}/{s['id']}/files/upload?path=" + urllib.request.quote(target)
    sh(s, f'mkdir -p $(dirname {path})')
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, body, headers, method='POST'), timeout=600):
                pass
            if target != path:
                sh(s, f'cp {target} {path} && rm {target}')
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def ssh_token(s, minutes=1440):
    return http('POST', f"/sandbox/{s['id']}/ssh-access?expiresInMinutes={minutes}")['token']


def hourly(spec):
    gpu = spec.get('gpu', 0) * PRICE['gpu'][spec['gpu_type'][0]] if spec.get('gpu') else 0.
    return spec['cpu'] * PRICE['vcpu'] + spec['memory'] * PRICE['mem_gib'] + spec.get('disk', 10) * PRICE['disk_gib'] + gpu


def _git(*args):
    try:
        return subprocess.run(['git', *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except FileNotFoundError:   # hero-hub has no git: it runs from a code tarball
        return ''


def code_sha():   # off git: the bootstrap's CODE_SHA
    return _git('rev-parse', '--short', 'HEAD') or (ROOT / 'CODE_SHA').read_text().strip()


def engine_tree(sha='HEAD'):
    return _git('rev-parse', '--short', f'{sha}:engine-rs') or (ROOT / 'ENGINE_TREE').read_text().strip()   # keep-remote


def bootstrap(s, sha):
    wheels = f'/vol/code/wheels-{engine_tree(sha)}'
    sh(s, f'set -e; ln -sfn /vol/results /results; ln -sfn /vol/retained /retained; mkdir -p /workspace /tmp/polytopia; '
          f'test -f /vol/code/{sha}/code.tar.gz || {{ echo "no /vol/code/{sha}: run hero.py publish"; exit 3; }}; '
          f'cd /workspace && tar xzf /vol/code/{sha}/code.tar.gz && echo {sha} > CODE_SHA && '
          f'pip install -q --force-reinstall --no-deps {wheels}/*.whl', timeout=900)


def launch(s, command, name):
    """Run `command` detached in /workspace exactly once, as an async command of the toolbox session 'hero'
    (the synchronous exec waits on a backgrounded child until its timeout). A pidfile written by the command's
    own shell guards against a second copy (pgrep -f would match the launching shell itself), and the submit is
    single-shot: the Mac's network drops replies, and a blind retry used to start duplicate nodes (2026-10-28)."""
    pid = f'/workspace/{name}.pid'
    alive = f'[ -f {pid} ] && kill -0 $(cat {pid}) 2>/dev/null'
    script = (f'{ENV_PREFIX}mkdir -p /tmp/polytopia; if {alive}; then exit 0; fi; echo $$ > {pid}; '
              f'cd /workspace && exec {command} >> /tmp/polytopia/{name}.log 2>&1 < /dev/null')
    base = f"{s['toolboxProxyUrl']}/{s['id']}/process/session"
    try:
        http('POST', base, dict(sessionId='hero'), retries=1, timeout=60)
    except RuntimeError:
        pass   # the session exists
    for attempt in range(4):
        try:
            http('POST', f'{base}/hero/exec', dict(command='bash -c ' + _quote(script), runAsync=True), retries=1,
                 timeout=60)
            return 'started'
        except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError):
            time.sleep(10)
            if 'up' in sh(s, f'{alive} && echo up; true'):
                return 'started'
    raise RuntimeError(f"{s.get('name')}: could not start {name}")


def start(s, cmd, spec, env=None, log='node'):
    put(s, f'/workspace/{log}.json', json.dumps(spec).encode())
    envs = ' '.join(f'{k}={_quote(str(v))}' for k, v in (env or {}).items())
    return launch(s, f'env {envs} python experiments/daytona/node.py {cmd} /workspace/{log}.json', log)


# ---- commands
def cmd_plan(a):
    hours = a.hours
    rows, total = [], 0.
    for run, p in PLANS.items():
        learner, actor = hourly(p['learner']), hourly(p['actor'])
        rate = min(p['actors'] * p['bench_per_actor'] * .85, p['learner_cap'])
        per_hour = learner + p['actors'] * actor
        evals = (hours / (p['eval']['interval'] / 3600) + 1) * 16 * .3 * hourly(dict(EVAL_BOX))
        cost = per_hour * hours + evals
        total += cost
        rows.append(dict(run=run, learner=f"{p['learner']['gpu']}x {p['learner']['gpu_type'][0]} spot ${learner:.2f}/h",
                         actors=f"{p['actors']}x {p['actor']['gpu_type'][0]} spot ${actor:.2f}/h each",
                         usd_per_hour=round(per_hour, 2), samples_per_s=int(rate),
                         samples_b=round(rate * hours * 3600 / 1e9, 2), usd_per_b=round(per_hour / (rate * 3600 / 1e9), 1),
                         evals_usd=round(evals, 1), total_usd=round(cost, 0)))
    for r in rows:
        print(json.dumps(r))
    print(json.dumps(dict(hours=hours, both_runs_usd=round(total, 0))))


def cmd_hub(a):
    s = find(HUB) or create(HUB, dict(role='hub'), ttl_minutes=a.ttl_hours * 60, **EVAL_BOX)
    print(json.dumps({k: s.get(k) for k in ('id', 'name', 'state', 'target')}))


def hub():
    s = find(HUB)
    if s is None or s.get('state') != 'started':
        raise SystemExit('no hub sandbox: python experiments/daytona/hero.py hub')
    return s


def cmd_publish(a):
    """Code (git HEAD: src, scripts, experiments/daytona, experiments/hero) + the engine wheel, built on the hub."""
    sha, s = code_sha(), hub()
    wheels = f'/vol/code/wheels-{engine_tree()}'   # rebuilt only when engine-rs changes
    if 'ok' in sh(s, f'test -f /vol/code/{sha}/OK && echo ok', check=False):
        print(json.dumps(dict(published=sha, cached=True)))
        return
    tar = subprocess.run(['git', 'archive', '--format=tar.gz', 'HEAD', 'src', 'scripts', 'engine-rs', 'tests',
                          'experiments/daytona', 'experiments/hero', 'experiments/curio'], capture_output=True, check=True,
                         cwd=ROOT).stdout   # tests/ and experiments/curio for hero.py tests
    put(s, '/tmp/code.tar.gz', tar)
    sh(s, f'set -e; mkdir -p /vol/code/{sha} && cp /tmp/code.tar.gz /vol/code/{sha}/; test -f {wheels}/OK && exit 0; '
          f'rm -rf /tmp/build && mkdir -p /tmp/build && cd /tmp/build && tar xzf /tmp/code.tar.gz && '
          f'cd engine-rs && CARGO_TARGET_DIR=/tmp/target maturin build --release -q -o /tmp/wheels && '
          f'mkdir -p {wheels} && cp /tmp/wheels/*.whl {wheels}/ && echo ok > {wheels}/OK', timeout=3600)
    sh(s, f'echo ok > /vol/code/{sha}/OK')
    print(json.dumps(dict(published=sha, wheels=wheels, bytes=len(tar))))


def run_config(run, hours, start_elapsed):
    """The run's Modal config, extended for a resume: training_seconds bounds the run as a whole (the learner's clock
    resumes at the logged elapsed time), so it becomes elapsed + hours; LR runs at learning_rate again and decays
    to learning_rate_end over the last 25% of the new segment, as in each 8-h Modal segment."""
    p = PLANS[run]
    if 'run_config' in p:   # a sweep arm: training_updates stops it, training_seconds is only the backstop
        return dict(p['run_config'], training_seconds=int(start_elapsed + hours * 3600))
    cfg = json.loads((ROOT / p['config']).read_text())
    cfg.pop('_doc', None)
    total = int(start_elapsed + hours * 3600)
    cfg.update(training_seconds=total, lr_decay_fraction=round(.25 * hours * 3600 / total, 4),
               checkpoint_every=20,   # spot: a reclaimed learner loses at most 20 updates (~1 min)
               critic_path='')   # the resume loads latest-critic.npz; the Modal parent critic is not staged
    return cfg


def _actor_spec(run, index, until):
    p = PLANS[run]
    return dict(run=run, index=index, processes=p['processes'], games=p['games'], until_unix=until,
                serve_options=p.get('serve_options', SERVE), affinity=p.get('affinity', 16))


def healthy_box(name, labels, ttl, spec, sha, tries=3):
    """A GPU sandbox that passed experiments/daytona/gpu_check.py (all-reduce, matmul agreement, host->device
    bandwidth): a Daytona RTX PRO 6000 x4 host moved 0.2-1.6 GB/s to 3 of its GPUs (healthy: 13-16) and its learner
    trained with KL ~1 at 2-30x the step time (2026-10-28). A failing host is deleted and another one requested."""
    for attempt in range(tries):
        s = create(f'{name}-{int(time.time()) % 100000}' if attempt else name, labels, ttl_minutes=ttl, **spec)
        bootstrap(s, sha)
        min_h2d = 2 if labels.get('role') == 'learner' else .5   # actor uploads are small; 5090/4090 hosts do ~1 GB/s
        out = sh(s, 'cd /workspace && JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false '
                    f'python experiments/daytona/gpu_check.py {min_h2d} 2>&1 | tail -1; true', timeout=300)
        try:
            check = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            check = dict(ok=False, raw=out[-300:])
        print(json.dumps(dict(gpu_check=s['name'], **check)), flush=True)
        if check.get('ok'):
            return s
        delete(s)
    raise RuntimeError(f'{name}: no healthy host in {tries} tries')


def learner_parent(run):
    """A sweep arm starts from its random init (hero_learner resumes from the volume once the folder has latest.npz);
    the hero runs always resume."""
    return PLANS[run].get('parent') or f'/vol/results/distributed/{run}/latest.npz'


def launch_learner(run, plan, sha):
    ttl = (plan['until_unix'] - time.time()) / 60 + 60
    s = healthy_box(f"{run}-learner-{int(time.time()) % 100000}", dict(run=run, role='learner'), ttl, plan['learner'], sha)
    spec = dict(run=run, parent=learner_parent(run), config=plan['config'],
                fallback_parent=plan.get('scratch_parent', ''))
    start(s, 'hero_learner', spec, env=LEARNER_ENV, log='learner')
    return s


def launch_actor(run, plan, sha, index, token):
    ttl = (plan['until_unix'] - time.time()) / 60 + 30
    s = healthy_box(f'{run}-actor-{index}', dict(run=run, role='actor', index=str(index)), ttl, plan['actor'], sha)
    put(s, '/workspace/learner-token', token.encode())
    start(s, 'hero_actor', _actor_spec(run, index, plan['until_unix']), env=ACTOR_ENV, log='actor')
    return s


def cmd_launch(a):
    run, p = a.run, PLANS[a.run]
    if a.actors:
        p['actors'] = a.actors
    if a.learner_gpu:   # e.g. H100:2 (spot 4x H100 / 4x H200 were refused on 2026-10-28: no spot capacity)
        kind, n = a.learner_gpu.split(':')
        p['learner'] = dict(p['learner'], gpu=int(n), gpu_type=kind.split(','), cpu=16 * int(n))
    if a.on_demand:
        p['learner'] = dict(p['learner'], spot=False)
    if a.actor_gpu:   # e.g. RTX-PRO-6000,RTX-4090 (PRO 6000 hosts have Zen 5 CPUs: ~2-4x the actor rate, 2026-10-28)
        p['actor'] = dict(p['actor'], gpu_type=a.actor_gpu.split(','))
    if a.actor_memory:
        p['actor'] = dict(p['actor'], memory=a.actor_memory)
    sha = code_sha()
    h = hub()
    if p.get('source'):
        sh(h, f'test -d /vol/results/distributed/{run} || cp -r /vol/results/distributed/{p["source"]} '
              f'/vol/results/distributed/{run}', timeout=900)
    scratch_parent = ''
    if p.get('scratch'):   # a from-scratch run: its random-init parent on the volume (resumes use latest.npz)
        if 'ok' not in sh(h, f'test -f /vol/code/{sha}/OK && echo ok', check=False):
            raise SystemExit(f'code {sha} not published: hero.py publish')
        scratch_parent = f'/vol/parents/{run}/random-init.npz'
        if 'ok' not in sh(h, f'test -f {scratch_parent} && echo ok', check=False):
            bootstrap(h, sha)
            spec = dict(p['scratch'], players=p['players'], out=scratch_parent)
            print(sh(h, f'cd /workspace && JAX_PLATFORMS=cpu python experiments/daytona/node.py hero_random_parent '
                        f'{_quote(json.dumps(spec))} 2>&1 | tail -2', timeout=900).strip(), flush=True)
    log_line = sh(h, f'tail -1 /vol/results/distributed/{run}/training.jsonl 2>/dev/null; true', check=False).strip()
    elapsed = float(json.loads(log_line)['elapsed']) if log_line.startswith('{') else 0.
    plan = dict(run=run, sha=sha, hours=a.hours, start_elapsed=elapsed, t0=time.time(),
                until_unix=time.time() + a.hours * 3600 + 900, actors=p['actors'], config=run_config(run, a.hours, elapsed),
                learner=p['learner'], actor=p['actor'], scratch_parent=scratch_parent,
                usd_per_hour=round(hourly(p['learner']) + p['actors'] * hourly(p['actor']), 2))
    print(json.dumps({k: v for k, v in plan.items() if k != 'config'}))
    if a.dry_run:
        print(json.dumps(plan['config']))
        return
    if sandboxes(run=run):
        raise SystemExit(f'{run} already has sandboxes (hero.py status/down)')
    if 'ok' not in sh(h, f'test -f /vol/code/{sha}/OK && echo ok', check=False):
        raise SystemExit(f'code {sha} not published: hero.py publish')
    # the `start` reference of the ladder: the checkpoint this segment resumes from, frozen once
    stage = p.get('start_update')
    src = p.get('source', run)   # a smoke freezes the same reference, from the real run's (untouched) folder
    if stage is None:
        pass   # from scratch: no start reference
    elif src == 'hero-ffa4':
        sh(h, 'mkdir -p /vol/results/four-player/monitor/hero-ffa4 && for e in npz json; do '
              f'test -f /vol/results/four-player/monitor/hero-ffa4/update-{stage}.$e || cp /vol/results/distributed/hero-ffa4/latest.$e '
              f'/vol/results/four-player/monitor/hero-ffa4/update-{stage}.$e; done')
    else:
        sh(h, 'mkdir -p /vol/results/distributed/ladder && for e in npz json; do '
              f'test -f /vol/results/distributed/ladder/{src}-update-{stage}.$e || cp /vol/results/distributed/{src}/latest.$e '
              f'/vol/results/distributed/ladder/{src}-update-{stage}.$e; done')
    put(h, f'/vol/runs/{run}/plan.json', json.dumps(plan, indent=1).encode())
    LEDGER.mkdir(parents=True, exist_ok=True)
    (LEDGER / f'{run}-plan.json').write_text(json.dumps(plan, indent=1))
    learner = launch_learner(run, plan, sha)
    token = ssh_token(learner)
    print(json.dumps(dict(learner=learner['name'], gpu_type=learner.get('gpuType'))), flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(launch_actor, run, plan, sha, i, token) for i in range(p['actors'])]
        for f in futures:
            try:
                s = f.result()
                print(json.dumps(dict(actor=s['name'], gpu_type=s.get('gpuType'))), flush=True)
            except Exception as e:
                print(json.dumps(dict(actor_error=repr(e)[:400])), flush=True)   # the keeper fills the gap
    (LEDGER / f'{run}-state.json').unlink(missing_ok=True)   # a fresh segment: no final_eval / pending_evals / progress from the last one
    _state(run, dict(token=token, token_at=time.time(), learner=learner['id'], next_actor=p['actors'], replaced=0,
                     next_eval=time.time() + PLANS[run]['eval']['interval'], eval_round=0, usd=0., usd_at=time.time()))


def _state(run, update=None):
    path = LEDGER / f'{run}-state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    if update:
        state.update(update)
        path.write_text(json.dumps(state, indent=1))
    return state


def _alive(s):
    return s.get('state') in ('started', 'starting', 'creating', 'pending_build', 'building_snapshot') and not s.get('spotEvictedAt')


def _rate(s):
    return hourly(dict(cpu=s.get('cpu', 0), memory=s.get('memory', 0), disk=s.get('disk', 0), gpu=s.get('gpu', 0),
                       gpu_type=[s.get('gpuType') or 'RTX-4090']))


def cmd_keep(a):
    """Every minute: replace lost actors, resume a lost learner, refresh the SSH token, run ladder rounds,
    stop at --max-usd, and tear down once the learner has written final.npz."""
    run, p = a.run, PLANS[a.run]
    LEDGER.mkdir(parents=True, exist_ok=True)
    plan = json.loads((LEDGER / f'{run}-plan.json').read_text())
    pending_evals = list(_state(run).get('pending_evals', []))   # a restarted keeper finishes rounds it launched
    _state(run, dict(progress={}))   # a stall clock saved by an earlier keeper would recycle a healthy learner
    while True:
        try:
            pending_evals = _keep_once(a, run, p, plan, pending_evals)
        except SystemExit:
            raise
        except Exception as err:   # a Daytona API outage (TLS drops for minutes at a time) must not end the keeper
            print(json.dumps(dict(at=time.time(), keeper_error=repr(err)[:300])), flush=True)
            time.sleep(60)
            continue
        if pending_evals is None:
            return
        if a.once:
            return
        time.sleep(60)


def _keep_once(a, run, p, plan, pending_evals):
    """One keeper iteration. Returns the pending evals to carry (None once the run is finished and torn down)."""
    if True:
        state, now = _state(run), time.time()
        rows = sandboxes(run=run)
        state['usd'] = state.get('usd', 0.) + sum(_rate(s) for s in rows if _alive(s)) * (now - state['usd_at']) / 3600
        state['usd_at'] = now
        learners = [s for s in rows if s['labels'].get('role') == 'learner' and _alive(s)]
        actors = [s for s in rows if s['labels'].get('role') == 'actor' and _alive(s)]
        # a sandbox whose node process has exited (crash, or an actor that gave up) is deleted and replaced;
        # sandboxes younger than 15 min are still booting
        for s in _stalled(learners, state, now):   # no new update for 10 min: a hung collective, resume elsewhere
            print(json.dumps(dict(at=now, learner_stalled=s['name'])), flush=True)
            # keep the evidence on the volume: py-spy stacks of every thread, Python and native (faulthandler caps
            # its dump at 100 threads, which the ~245 connection threads fill before the main/assembler threads),
            # then the faulthandler dump (SIGUSR1) and the log tail
            tag = f'/vol/runs/{run}/stall-{s["name"]}-{int(now)}'
            sh(s, f'mkdir -p /vol/runs/{run}; P=$(cat /workspace/learner.pid); pip install -q py-spy >/dev/null 2>&1; '
                  f'(timeout 90 py-spy dump --pid $P; echo "=== native ==="; timeout 150 py-spy dump --native --pid $P) '
                  f'> /tmp/pyspy-stall.txt 2>&1; cp /tmp/pyspy-stall.txt {tag}-pyspy.txt; '
                  f'pkill -USR1 -f "node.py hero_[l]earner"; sleep 3; tail -c 600000 /tmp/polytopia/learner.log '
                  f'> /tmp/stall.log; cp /tmp/stall.log {tag}.log; true', check=False, timeout=400)
            delete(s)
            learners.remove(s)
        for s in _diverged(learners, run):   # KL > .3 over the last 5 updates: a bad host, resume elsewhere
            print(json.dumps(dict(at=now, learner_diverged=s['name'])), flush=True)
            delete(s)
            learners.remove(s)
        settling = now - state.get('actors_restarted_at', 0) < 300   # nodes just restarted: not dead
        for s in _dead_processes(learners + ([] if settling else actors), now):
            print(json.dumps(dict(at=now, process_gone=s['name'])), flush=True)
            delete(s)
            (learners if s in learners else actors).remove(s)
        for s in rows:   # preempted / failed sandboxes: delete so they stop counting and their names free up
            if not _alive(s) and s['labels'].get('role') in ('learner', 'actor'):
                print(json.dumps(dict(at=now, lost=s['name'], state=s.get('state'), evicted=s.get('spotEvictedAt'))), flush=True)
                delete(s)
        h = hub()
        done = 'ok' in sh(h, f'test -f /vol/runs/{run}/learner-done.json && echo ok', check=False)
        if state['usd'] > a.max_usd and not done:
            print(json.dumps(dict(at=now, budget_stop=round(state['usd'], 2))), flush=True)
            cmd_stop(argparse.Namespace(run=run))
        if done:
            for s in rows:   # tear the fleet down BEFORE the final round: its boxes need the fleet's share of the CPU cap
                if s['labels'].get('role') in ('learner', 'actor'):
                    delete(s)
            if not state.get('final_eval') and not a.no_eval:
                pending_evals.append(eval_round(run, plan, final=True))
                state['final_eval'] = now
            pending_evals = [e for e in pending_evals if not collect(run, e)]
            state['pending_evals'] = pending_evals
            _state(run, state)
            if not pending_evals:
                print(json.dumps(dict(at=now, finished=True, usd=round(state['usd'], 2))), flush=True)
                return None
            return pending_evals
        if not learners and now < plan['until_unix'] - 1200 and state.get('learner_restarts', 0) < 16:
            state['learner_restarts'] = state.get('learner_restarts', 0) + 1
            print(json.dumps(dict(at=now, resume_learner=state['learner_restarts'])), flush=True)
            try:
                learner = launch_learner(run, plan, plan['sha'])
            except Exception as error:   # no healthy host this time: log it and try again next tick (2026-10-28:
                print(json.dumps(dict(at=now, learner_launch_failed=repr(error)[:300])), flush=True)   # it crashed)
                _state(run, state)
                return pending_evals
            state.update(learner=learner['id'], token=ssh_token(learner), token_at=time.time(), learner_at=time.time())
            # actor processes connected to the old learner wedge on dead sockets (CLOSE_WAIT, 2026-10-28) instead of
            # reconnecting: restart their node processes in place (no new sandboxes, the compile cache is warm)
            restart_actor_nodes(run, plan, actors, state['token'])
            state.update(actors_restarted_at=time.time(), progress={})
        elif learners and state.get('kicked') != learners[0]['id'] and \
                state.get('progress', {}).get(learners[0]['id'], [1])[0] == 0 and \
                now - state.get('learner_at', now) > 900:   # a learner that never got data: the actors are wedged
            print(json.dumps(dict(at=now, kick_actors=learners[0]['name'])), flush=True)
            state['kicked'] = learners[0]['id']
            restart_actor_nodes(run, plan, actors, state['token'])
            state.update(actors_restarted_at=time.time(), progress={})
        elif learners and now - state['token_at'] > 20 * 3600:
            state.update(token=ssh_token(learners[0]), token_at=time.time())
            for s in actors:   # a gateway session to a deleted learner can stay up: drop it so the loop reconnects
                put(s, '/workspace/learner-token', state['token'].encode())
                sh(s, 'pkill -f "[s]sh -N"; true', check=False)
        missing = plan['actors'] - len(actors)
        while missing > 0 and state['replaced'] < 3 * plan['actors'] and now < plan['until_unix'] - 1800:
            index = state['next_actor']
            state['next_actor'] += 1
            missing -= 1
            try:
                s = launch_actor(run, plan, plan['sha'], index, state['token'])
                state['replaced'] += 1   # only a box that came up counts against the budget (a refused create does not)
                print(json.dumps(dict(at=now, replaced_actor=s['name'], gpu_type=s.get('gpuType'))), flush=True)
            except Exception as e:
                print(json.dumps(dict(at=now, actor_error=repr(e)[:300])), flush=True)
                if 'GPU credits' in repr(e):   # the org wallet is empty: stop hammering the API until the next loop
                    break
        if now > state['next_eval'] and not a.no_eval and pending_evals:   # a round next to a live one hits the 250-vCPU cap
            print(json.dumps(dict(at=now, eval_deferred=[e['tag'] for e in pending_evals])), flush=True)
            state['next_eval'] = now + 300
        elif now > state['next_eval'] and not a.no_eval:
            pending_evals.append(eval_round(run, plan, round_index=state['eval_round']))
            state.update(next_eval=now + p['eval']['interval'], eval_round=state['eval_round'] + 1)
        pending_evals = [e for e in pending_evals if not collect(run, e)]
        state['pending_evals'] = pending_evals
        _state(run, state)
        print(json.dumps(dict(at=now, learners=len(learners), actors=len(actors), usd=round(state['usd'], 2),
                              evals=len(pending_evals))), flush=True)
        return pending_evals


def restart_actor_nodes(run, plan, actors, token):
    """Kill each actor sandbox's node process and its tunnel, hand it the token, start the node again."""
    from concurrent.futures import ThreadPoolExecutor

    def one(s):
        try:
            # the node's spawned actor workers and inference server outlive it (ppid 1): kill them too, or each
            # restart stacks a fleet on the old one until the sandbox OOMs (52 OOM kills on 2026-10-28)
            sh(s, 'pkill -f "node.py hero_[a]ctor"; sleep 3; pkill -9 -f "node.py hero_[a]ctor"; '
                  'pkill -9 -f "[m]ultiprocessing"; pkill -f "[s]sh -N"; rm -f /workspace/actor.pid; sleep 2; true',
               check=False)
            put(s, '/workspace/learner-token', token.encode())
            start(s, 'hero_actor', _actor_spec(run, int(s['labels']['index']), plan['until_unix']), env=ACTOR_ENV,
                  log='actor')
            return s['name']
        except Exception as e:
            return f"{s['name']}: {e!r}"[:200]
    with ThreadPoolExecutor(8) as pool:
        print(json.dumps(dict(restarted_actors=list(pool.map(one, actors)))), flush=True)


def _stalled(learners, state, now, minutes=20, young=1800):
    """Learners whose update count has not moved for `minutes` (tracked in state['progress'] by sandbox id). A
    sandbox younger than `young` seconds is never stalled: a fresh learner waits for its actors' inference servers
    to compile (~11 min for the attention trunk), and the 10-minute rule killed lookout's first learner at update 1
    while they did (2026-11-01)."""
    from datetime import datetime
    out, progress = [], state.setdefault('progress', {})
    for s in learners:
        if now - datetime.fromisoformat(s['createdAt'].replace('Z', '+00:00')).timestamp() < young:
            continue
        try:
            n = int(sh(s, 'grep -a -c "\\"stage\\": \\"update\\"" /tmp/polytopia/learner.log; true', check=False,
                       timeout=60).split()[0])
        except Exception:
            continue
        seen = progress.get(s['id'])
        if seen is None or seen[0] != n:
            progress[s['id']] = [n, now]
        elif n > 0 and now - seen[1] > minutes * 60:
            out.append(s)
    return out


def _diverged(learners, run, kl=.3):
    out = []
    for s in learners:
        try:
            rows = [json.loads(l) for l in sh(s, 'grep -a "\\"stage\\": \\"update\\"" /tmp/polytopia/learner.log | tail -n 5',
                                              check=False, timeout=60).splitlines() if l.startswith('{')]
        except Exception:
            continue
        if len(rows) == 5 and min(r['approx_kl'] for r in rows) > kl:
            out.append(s)
    return out


def _dead_processes(rows, now):
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime

    def gone(s):
        age = now - datetime.fromisoformat(s['createdAt'].replace('Z', '+00:00')).timestamp()
        if age < 900:
            return False
        role = s['labels']['role']
        try:
            pattern = f'node.py hero_[{role[0]}]{role[1:]}'   # [x]: no match on this command's own bash
            return 'up' not in sh(s, f'pgrep -f "{pattern}" >/dev/null && echo up', check=False, timeout=60)
        except Exception:
            return False   # the toolbox did not answer: decide next minute
    with ThreadPoolExecutor(8) as pool:
        return [s for s, g in zip(rows, pool.map(gone, rows)) if g]


def eval_round(run, plan, round_index=0, final=False):
    """Freeze the run's latest (or final) checkpoint on the volume and start the arena sandboxes for one round.
    1v1: every reference x `maps` maps x 2 seats, `maps_per_box` maps per 4-vCPU sandbox. 4p: as four_monitor,
    CUR vs the references on seat-rotated tables, `parts` sandboxes."""
    e, h, sha = PLANS[run]['eval'], hub(), plan['sha']
    src = f'/vol/results/distributed/{run}/' + ('final' if final else 'latest')
    stage = json.loads(sh(h, f'cat {src}.json'))['stage']
    tag = f"{run}-{stage}{'-final' if final else ''}"
    frozen = f'/vol/results/distributed/ladder/{tag}.npz'
    sh(h, f'mkdir -p /vol/results/distributed/ladder && cp {src}.npz {frozen} && cp {src}.json {frozen[:-4]}.json')
    maps = e['final_maps'] if final else e['maps']
    seed_base = e['seed_base'] + 1000 * (99 if final else round_index)
    jobs = []
    ref_stages = {}
    if e['kind'] == 'ladder':
        per_box = e['maps_per_box'] * (2 if final else 1)   # both final rounds at once stay under 250 vCPU
        present = sh(h, 'for f in ' + ' '.join(e['references'].values()) + '; do test -f /vol$f && echo $f; done; true').split()
        references = dict(e['references'])
        for ref, folder in e.get('live', {}).items():   # freeze a live run's latest (json, npz, json; retry until consistent)
            frozen_live = f'/results/distributed/ladder/live-{ref}-{tag}.npz'
            script = (f'cd /vol{folder} || exit 3; test -f latest.npz || exit 4; for i in 1 2 3 4 5; do cp latest.json /tmp/a.json && '
                      f'cp latest.npz /vol{frozen_live} && cp latest.json /vol{frozen_live[:-4]}.json && '
                      f'cmp -s /tmp/a.json /vol{frozen_live[:-4]}.json && cat latest.json && exit 0; sleep 10; done; exit 7')
            try:
                meta = json.loads(sh(h, script, timeout=600))
                references[ref] = frozen_live
                present.append(frozen_live)
                ref_stages[ref] = meta.get('stage')
            except Exception as error:
                print(json.dumps(dict(eval=tag, live_reference_skipped=ref, error=repr(error)[:200])), flush=True)
        for ref, path in references.items():
            if path not in present:   # a reference synced by hand (farsight) that is not there yet
                print(json.dumps(dict(eval=tag, missing_reference=ref, path=path)), flush=True)
                continue
            for lo in range(0, maps, per_box):
                jobs.append(('hero_arena', dict(name=f'{tag}-vs-{ref}-{seed_base + lo}', candidate=frozen, opponent=path,
                                                seed_base=seed_base + lo, maps=min(per_box, maps - lo),
                                                map_mix='proposed-v1', size=14, games=4, processes=4, reference=ref,
                                                out=f'/vol/results/distributed/arena/{tag}-vs-{ref}-{seed_base + lo}.json')))
    else:
        names = list(e['references'])
        tables = [['CUR'] + [n] * 3 for n in names] + [['CUR', 'CUR'] + names[:2]]
        seeds = list(range(seed_base, seed_base + maps))
        parts = e['parts'] * (2 if final else 1)
        for q in range(parts):
            mine = seeds[q::parts]
            jobs.append(('hero_arena4', dict(entrants=dict(CUR=frozen, **e['references']), seeds=mine, seed=7000 + q,
                                             tables=[tables[(q + parts * j) % len(tables)] for j in range(len(mine))],
                                             lanes=16, out=f'/vol/results/four-player/monitor/parts/{tag}-{q}.json')))
    boxes = []
    from concurrent.futures import ThreadPoolExecutor

    def one(i, job):
        # the Daytona API drops TLS handshakes / truncates replies now and then: retry the whole box, deleting a
        # half-made one (an untracked box would still run and bill, and the round would wait on the missing part)
        for attempt in range(3):
            s = None
            try:
                s = create(f'{tag[:40]}-e{i}-{int(time.time()) % 10000}', dict(run=run, role='eval'), ttl_minutes=180,
                           **EVAL_BOX)
                bootstrap(s, sha)
                start(s, job[0], job[1], env=dict(JAX_PLATFORMS='cpu'), log='eval')
                return s
            except Exception as err:
                print(json.dumps(dict(eval_retry=i, attempt=attempt, error=repr(err)[:200])), flush=True)
                if s is not None:
                    try:
                        delete(s)
                    except Exception:
                        pass
                if attempt == 2:
                    raise
                time.sleep(10 * (attempt + 1))
    with ThreadPoolExecutor(8) as pool:
        for f in [pool.submit(one, i, j) for i, j in enumerate(jobs)]:
            try:
                boxes.append(f.result()['id'])
            except Exception as err:
                print(json.dumps(dict(eval_error=repr(err)[:300])), flush=True)
    if e['kind'] == 'ladder':   # which checkpoint each reference was at this round (farsight is re-synced by hand)
        for ref, path in e['references'].items():
            if path in present and ref not in ref_stages:
                try:
                    ref_stages[ref] = json.loads(sh(h, f'cat /vol{path[:-4]}.json', timeout=60)).get('stage')
                except Exception:
                    ref_stages[ref] = None
    print(json.dumps(dict(eval=tag, jobs=len(jobs), boxes=len(boxes), ref_stages=ref_stages)), flush=True)
    return dict(tag=tag, stage=stage, final=final, round=round_index, kind=e['kind'], boxes=boxes,
                outs=[j[1]['out'] for j in jobs], refs=[j[1].get('reference') for j in jobs], started=time.time(),
                ref_stages=ref_stages)


def collect(run, ev, timeout=180 * 60):   # 16-map boxes on a busy Daytona host can take > 100 min (Nov 3); the next round is deferred while one is pending
    """True once the round is written (or timed out): merge the parts, append to the history, delete the boxes."""
    h = hub()
    have = sh(h, 'for f in ' + ' '.join(ev['outs']) + '; do test -f $f && echo $f; done; true').split()
    if len(have) < len(ev['outs']) and time.time() - ev['started'] < timeout:
        return False
    row = dict(round=ev['round'], stage=ev['stage'], final=ev['final'], finished=time.time(), parts=len(have),
               expected=len(ev['outs']), ref_stages=ev.get('ref_stages', {}))
    if ev['kind'] == 'ladder':
        per = {}
        for out, ref in zip(ev['outs'], ev['refs']):
            if out in have:
                d = json.loads(sh(h, 'python -c "import json; d=json.load(open(\'%s\')); print(json.dumps({k: d.get(k, 0) for k in '
                                     '(\'wins\', \'losses\', \'draws_or_truncations\', \'games\')}))"' % out))
                t = per.setdefault(ref, dict(wins=0, losses=0, draws_or_truncations=0, games=0))
                for k in t:
                    t[k] += d.get(k, 0)
        for t in per.values():
            t['win_rate'] = round(t['wins'] / max(t['games'], 1), 3)
            t['score'] = round((t['wins'] + .5 * t['draws_or_truncations']) / max(t['games'], 1), 3)
        row['vs'] = per
    else:
        if 'ok' not in sh(h, 'test -f /workspace/experiments/daytona/node.py && echo ok', check=False):
            bootstrap(h, code_sha())   # the summary runs on the hub
        out = f"/vol/results/four-player/monitor/{ev['tag']}-summary.json"
        spec = dict(parts=have, anchor='START', out=out)
        sh(h, f"cd /workspace && echo {_quote(json.dumps(spec))} > /tmp/s4.json && JAX_PLATFORMS=cpu python "
              f"experiments/daytona/node.py hero_summary4 /tmp/s4.json", timeout=600)
        row['summary'] = json.loads(sh(h, f'cat {out}'))
    history_path = f'/vol/runs/{run}/eval-history.json'
    history = json.loads(sh(h, f'cat {history_path} 2>/dev/null || echo []'))
    put(h, history_path, json.dumps(history + [row], indent=1).encode())
    local = LEDGER / f'{run}-eval-history.json'
    local.write_text(json.dumps(history + [row], indent=1))
    for s in sandboxes(run=run, role='eval'):
        if s['id'] in ev['boxes']:
            delete(s)
    print(json.dumps(dict(eval_done=ev['tag'], **{k: v for k, v in row.items() if k != 'summary'})), flush=True)
    return True


def cmd_eval(a):
    plan = json.loads((LEDGER / f'{a.run}-plan.json').read_text())
    ev = eval_round(a.run, plan, round_index=90 + int(time.time()) % 9, final=a.final)
    while not collect(a.run, ev):
        time.sleep(60)


def cmd_status(a):
    rows = sandboxes(run=a.run)
    for s in rows:
        print(json.dumps({k: s.get(k) for k in ('name', 'state', 'gpu', 'gpuType', 'spot', 'cpu', 'memory', 'createdAt',
                                                 'spotEvictedAt')} | dict(usd_per_hour=round(_rate(s), 2))))
    print(json.dumps(dict(sandboxes=len(rows), usd_per_hour=round(sum(_rate(s) for s in rows if _alive(s)), 2),
                          state=_state(a.run))))
    learner = next((s for s in rows if s['labels'].get('role') == 'learner' and _alive(s)), None)
    if learner:
        print(sh(learner, f'tail -c 1500 /tmp/polytopia/learner.log; echo; tail -1 /workspace/run/{a.run}/training.jsonl '
                          f'2>/dev/null | head -c 1200', check=False))


def cmd_restart(a):
    """Hotfix: put the run's live sandboxes on the published HEAD and restart their node processes. The learner
    resumes from its local run folder (latest.npz), the actors reconnect."""
    run, sha = a.run, code_sha()
    plan = json.loads((LEDGER / f'{run}-plan.json').read_text())
    plan['sha'] = sha
    (LEDGER / f'{run}-plan.json').write_text(json.dumps(plan, indent=1))
    state = _state(run)
    roles = ('actor',) if a.actors_only else ('learner', 'actor')
    rows = [s for s in sandboxes(run=run) if _alive(s) and s['labels'].get('role') in roles]
    for s in sorted(rows, key=lambda s: s['labels']['role'] != 'learner'):   # learner first
        role = s['labels']['role']
        sh(s, f'pkill -f "node.py hero_[{role[0]}]{role[1:]}"; sleep 3; pkill -9 -f "node.py hero_[{role[0]}]{role[1:]}"; '
              'pkill -9 -f "[m]ultiprocessing"; pkill -f "[s]sh -N"; sleep 2; true')   # orphaned workers, old tunnel
        bootstrap(s, sha)
        if role == 'learner':
            start(s, 'hero_learner', dict(run=run, parent=learner_parent(run),
                                          config=plan['config'], fallback_parent=plan.get('scratch_parent', '')),
                  env=LEARNER_ENV, log='learner')
            state.update(learner=s['id'], token=ssh_token(s), token_at=time.time())
            _state(run, state)
        else:
            put(s, '/workspace/learner-token', state['token'].encode())
            start(s, 'hero_actor', _actor_spec(run, int(s['labels']['index']), plan['until_unix']),
                  env=ACTOR_ENV, log='actor')
        print(json.dumps(dict(restarted=s['name'], sha=sha)), flush=True)


def cmd_keep_remote(a):
    """Move RUN's keeper from this Mac onto hero-hub (non-spot CPU sandbox, no auto-stop), so it survives the laptop:
    stop the local keeper, copy the run's ledger files, install the published code on the hub, start the keeper
    there. Needs a Daytona API key in /workspace/.daytona-api-key on the hub (the user stores it; see the doc)."""
    run, h, sha = a.run, hub(), code_sha()
    if a.no_ledger:   # restart the hub keeper on the published HEAD, keeping the hub's own ledger
        org = _auth()[2]
        # retried as a whole: a dropped reply after the pkill used to leave the run with no keeper (2026-10-28)
        for attempt in range(5):
            try:
                sh(h, f'pkill -f "hero.py [k]eep {run}"; sleep 3; rm -f /workspace/keeper-{run}.pid; true', check=False)
                bootstrap(h, sha)
                sh(h, f'echo {engine_tree(sha)} > /workspace/ENGINE_TREE')
                print(launch(h, f'env HERO_LEDGER=/workspace/ledger DAYTONA_ORGANIZATION_ID={org} python -u '
                                f'experiments/daytona/hero.py keep {run} --max-usd {a.max_usd}', f'keeper-{run}'))
                return
            except Exception as error:
                print(f'keep-remote attempt {attempt + 1} failed: {error!r}'[:200], flush=True)
                time.sleep(15)
        raise SystemExit('keeper NOT restarted: check `hero.py keeper-log`')
    if 'ok' not in sh(h, f'test -s {KEY_FILE} && echo ok', check=False):
        raise SystemExit(f'no API key on hero-hub ({KEY_FILE}): create one in the Daytona dashboard and store it there first')
    if 'ok' not in sh(h, f'test -f /vol/code/{sha}/OK && echo ok', check=False):
        raise SystemExit(f'code {sha} not published: hero.py publish')
    subprocess.run(['pkill', '-f', f'hero.py keep {run}'])
    time.sleep(3)
    bootstrap(h, sha)
    sh(h, f'echo {engine_tree(sha)} > /workspace/ENGINE_TREE && mkdir -p /workspace/ledger')
    for f in sorted(LEDGER.glob(f'{run}-*.json')):
        put(h, f'/workspace/ledger/{f.name}', f.read_bytes())
    org = _auth()[2]
    print(launch(h, f'env HERO_LEDGER=/workspace/ledger DAYTONA_ORGANIZATION_ID={org} python -u '
                    f'experiments/daytona/hero.py keep {run} --max-usd {a.max_usd}', f'keeper-{run}'))
    time.sleep(20)
    print(sh(h, f'tail -n 3 /tmp/polytopia/keeper-{run}.log', check=False))


def cmd_keeper_log(a):
    """The hub keeper's log tail and whether it is running."""
    h = hub()
    print(sh(h, f'pgrep -f "hero.py [k]eep {a.run}" >/dev/null && echo "keeper: running on hero-hub" || echo "keeper: NOT running"; '
                f'tail -n {a.n} /tmp/polytopia/keeper-{a.run}.log', check=False))


def cmd_store_key(a):
    """Store a Daytona API key on hero-hub (KEY_FILE) for the hub keeper (keep-remote). The key is read from this
    shell only: the DAYTONA_HUB_KEY environment variable, else a hidden prompt; it is never printed or logged."""
    import getpass
    key = os.environ.get('DAYTONA_HUB_KEY') or getpass.getpass('Daytona API key for the hub keeper: ')
    key = key.strip()
    if len(key) < 20:
        raise SystemExit('that does not look like an API key')
    h = hub()
    put(h, str(KEY_FILE), (key + '\n').encode())
    sh(h, f'chmod 600 {KEY_FILE}')
    print(json.dumps(dict(stored=str(KEY_FILE), hub=h['name'], chars=len(key))))


def cmd_sync_farsight(a):
    """Copy farsight's newest checkpoint (the conv + pooling + history run on the Curio head, read-only over ssh)
    to the hub's volume as the ladder reference FARSIGHT (+ a stage-tagged copy), and record its stage."""
    import tempfile
    h = hub()
    head, remote = a.ssh, f'{a.root}/runs/{a.run}'
    with tempfile.TemporaryDirectory() as tmp:
        # a still copy on the head first (its run dir is only read): latest.npz is rewritten every 20 updates and a
        # direct scp over the slow link raced it (sha256 mismatch, 2026-11-01)
        still = f'/tmp/{a.run}-sync-{os.getpid()}'
        # json, npz, json again; the pair is consistent when both json copies match (a checkpoint lands every ~20
        # updates, ~1-2 min, and the learner writes npz then json)
        script = (f'mkdir -p {still}; for i in 1 2 3 4 5; do cp {remote}/latest.json {still}/a.json && '
                  f'cp {remote}/latest.npz {still}/latest.npz && cp {remote}/latest.json {still}/latest.json && '
                  f'cmp -s {still}/a.json {still}/latest.json && exit 0; sleep 5; done; exit 7')
        for attempt in range(8):   # the head's sshd drops sessions under load (load average ~100, 2026-11-01 23:30)
            r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', head, script], timeout=600)
            if r.returncode == 0:
                break
            time.sleep(15)
        else:
            raise SystemExit(f'could not make a still copy on {head} in 8 attempts')
        for ext in ('json', 'npz'):   # the link to the head is slow and drops: resumable copy, several attempts
            for attempt in range(6):
                r = subprocess.run(['rsync', '-q', '--partial', '--timeout=120', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=20',
                                    f'{head}:{still}/latest.{ext}', f'{tmp}/latest.{ext}'], timeout=1800)
                if r.returncode == 0:
                    break
                time.sleep(10)
            else:
                subprocess.run(['ssh', '-o', 'BatchMode=yes', head, f'rm -rf {still}'], check=False, timeout=120)
                raise SystemExit(f'could not copy latest.{ext} from {head} in 6 attempts')
        subprocess.run(['ssh', '-o', 'BatchMode=yes', head, f'rm -rf {still}'], check=False, timeout=120)
        meta = json.loads(Path(f'{tmp}/latest.json').read_text())
        if meta.get('sha256') and meta['sha256'] != _sha256(Path(f'{tmp}/latest.npz')):
            raise SystemExit('latest.npz changed under the copy (sha256 mismatch): run again')
        stage = meta['stage']
        latest = FARSIGHT if a.run == 'farsight' else CURIO_REF.format(run=a.run)
        tagged = latest.replace('latest', f'{stage}')
        for ext in ('npz', 'json'):
            put(h, f'/vol{tagged[:-4]}.{ext}', Path(f'{tmp}/latest.{ext}').read_bytes())
        sh(h, f'cp /vol{tagged} /vol{latest} && cp /vol{tagged[:-4]}.json /vol{latest[:-4]}.json')
    LEDGER.mkdir(parents=True, exist_ok=True)
    log = LEDGER / 'farsight-sync.jsonl'
    with log.open('a') as f:
        f.write(json.dumps(dict(at=time.time(), run=a.run, stage=stage, tagged=tagged, parameter_elements=meta.get('parameter_elements'))) + '\n')
    print(json.dumps(dict(run=a.run, stage=stage, reference=latest, tagged=tagged)))


def cmd_tests(a):
    """The test suite on Daytona (the user's choice over Modal, 2026-11-01): one 4-vCPU CPU sandbox per test file on the
    published HEAD (tests/ and experiments/curio are in the tarball), pytest output collected, boxes deleted. Prints
    each file's tail as it finishes and exits 1 if any failed."""
    from concurrent.futures import ThreadPoolExecutor
    sha, h = code_sha(), hub()
    if 'ok' not in sh(h, f'test -f /vol/code/{sha}/OK && echo ok', check=False):
        raise SystemExit(f'code {sha} not published: hero.py publish')
    paths = a.paths.split(',')

    def one(i, path):
        s = create(f'tests-{sha}-{i}-{int(time.time()) % 10000}', dict(role='tests', run='tests'), ttl_minutes=a.ttl_minutes,
                   **EVAL_BOX)
        try:
            bootstrap(s, sha)
            out = sh(s, f'cd /workspace && JAX_PLATFORMS=cpu python -m pytest -q -x --no-header -p no:cacheprovider {path} '
                        f'2>&1 | tail -n {a.tail}; echo "exit ${{PIPESTATUS[0]}}"', timeout=a.timeout, check=False)
        finally:
            delete(s)
        code = int(out.strip().rsplit('exit ', 1)[-1].split()[0]) if 'exit ' in out else 1
        return path, code, out
    failed = False
    with ThreadPoolExecutor(8) as pool:
        for path, code, out in pool.map(lambda t: one(*t), enumerate(paths)):
            failed |= bool(code)
            print(f'===== {path}: exit {code}\n{out.strip()}', flush=True)
    LEDGER.mkdir(parents=True, exist_ok=True)
    (LEDGER / f'tests-{sha}.txt').write_text(f'{a.paths}\nfailed={failed}\n')
    if failed:
        raise SystemExit(1)


def cmd_sync_ckpt(a):
    """Copy one checkpoint pair (npz + json) from the Curio head (read-only ssh) to the volume, e.g. the ladder
    reference bighero_u3880: --src /data/polytopia/ckpt/bighero-u3880/latest.npz --dest /results/distributed/ladder/bighero-u3880.npz."""
    import tempfile
    h = hub()
    src, dest = a.src, a.dest
    if not (src.endswith('.npz') and dest.endswith('.npz')):
        raise SystemExit('--src and --dest name .npz files')
    with tempfile.TemporaryDirectory() as tmp:
        for ext in ('json', 'npz'):
            subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', f'{a.ssh}:{src[:-4]}.{ext}', f'{tmp}/ckpt.{ext}'],
                           check=True, timeout=1800)
        meta = json.loads(Path(f'{tmp}/ckpt.json').read_text())
        if meta.get('sha256') and meta['sha256'] != _sha256(Path(f'{tmp}/ckpt.npz')):
            raise SystemExit('checkpoint changed under the copy (sha256 mismatch): run again')
        for ext in ('npz', 'json'):
            put(h, f'/vol{dest[:-4]}.{ext}', Path(f'{tmp}/ckpt.{ext}').read_bytes())
    print(json.dumps(dict(synced=dest, stage=meta.get('stage'), parameter_elements=meta.get('parameter_elements'),
                          config={k: meta['config'].get(k) for k in ('blocks', 'width', 'obs_version', 'block_pool')})))


def cmd_pool_probe(a):
    """Trunk-block pooling statistics of RUN's latest checkpoint (or --checkpoint, a volume path) on self-play
    positions, run on the hub (experiments/daytona/pool_probe.py): bias_rms / conv_rms per pooling layer, the gates,
    and what switching the pooling off changes. Appended to runs/hero-daytona/<run>-pool-probe.jsonl and written to
    /vol/runs/<run>/pool-probe/<stage>.json. The farsight stop rule: bias past ~0.5x the convolution."""
    h, sha = hub(), code_sha()
    if 'ok' not in sh(h, f'test -f /workspace/CODE_SHA && grep -q {sha} /workspace/CODE_SHA && echo ok', check=False):
        bootstrap(h, sha)
    ckpt = a.checkpoint or f'/vol/results/distributed/{a.run}/latest.npz'
    stage = json.loads(sh(h, f'cat {ckpt[:-4]}.json'))['stage']
    spec = dict(checkpoint=ckpt, games=a.games, steps=a.steps, every=a.every, seed=a.seed,
                out=f'/vol/runs/{a.run}/pool-probe/{stage}.json')
    out = sh(h, f'cd /workspace && JAX_PLATFORMS=cpu python experiments/daytona/node.py hero_pool_probe '
                f'{_quote(json.dumps(spec))} 2>&1 | tail -1', timeout=3600).strip()
    row = json.loads(out.splitlines()[-1])
    LEDGER.mkdir(parents=True, exist_ok=True)
    with (LEDGER / f'{a.run}-pool-probe.jsonl').open('a') as f:
        f.write(json.dumps(dict(row, at=time.time())) + '\n')
    print(json.dumps(row, indent=1))


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cmd_stop(a):
    h = hub()
    sh(h, f'mkdir -p /vol/runs/{a.run} && date > /vol/runs/{a.run}/STOP')
    for s in sandboxes(run=a.run):
        if s['labels'].get('role') in ('learner', 'actor') and _alive(s):
            sh(s, 'touch /workspace/STOP', check=False)
    print(json.dumps(dict(stop=a.run)))


def cmd_down(a):
    for s in sandboxes(run=a.run):
        delete(s)
        print('deleted', s['name'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('plan'); q.add_argument('--hours', type=float, default=8)
    q = sub.add_parser('hub'); q.add_argument('--ttl-hours', type=int, default=36)
    sub.add_parser('publish')
    q = sub.add_parser('launch'); q.add_argument('run', choices=PLANS); q.add_argument('--hours', type=float, default=8)
    q.add_argument('--actors', type=int); q.add_argument('--dry-run', action='store_true')
    q.add_argument('--learner-gpu', help='TYPE[,FALLBACK]:COUNT, e.g. H100:2'); q.add_argument('--on-demand', action='store_true')
    q.add_argument('--actor-gpu', help='TYPE[,FALLBACK...]'); q.add_argument('--actor-memory', type=int, help='GiB')
    q = sub.add_parser('keep'); q.add_argument('run', choices=PLANS); q.add_argument('--max-usd', type=float, default=400)
    q.add_argument('--once', action='store_true'); q.add_argument('--no-eval', action='store_true')
    q = sub.add_parser('eval'); q.add_argument('run', choices=PLANS); q.add_argument('--final', action='store_true')
    q = sub.add_parser('keep-remote'); q.add_argument('run', choices=PLANS); q.add_argument('--max-usd', type=float, default=400)
    q.add_argument('--no-ledger', action='store_true', help='restart the hub keeper keeping its ledger')
    q = sub.add_parser('keeper-log'); q.add_argument('run', choices=PLANS); q.add_argument('-n', type=int, default=20)
    sub.add_parser('store-key')
    q = sub.add_parser('sync-farsight'); q.add_argument('--run', default='farsight'); q.add_argument('--ssh', default='curio-gpu')
    q.add_argument('--root', default='/data/polytopia')
    q = sub.add_parser('tests'); q.add_argument('--paths', default='tests/test_block_pool.py,tests/test_distributed_rl.py')
    q.add_argument('--ttl-minutes', type=int, default=90); q.add_argument('--timeout', type=int, default=3600)
    q.add_argument('--tail', type=int, default=40)
    q = sub.add_parser('sync-ckpt'); q.add_argument('--src', required=True); q.add_argument('--dest', required=True)
    q.add_argument('--ssh', default='curio-gpu')
    q = sub.add_parser('pool-probe'); q.add_argument('run', choices=PLANS); q.add_argument('--checkpoint', default='')
    q.add_argument('--games', type=int, default=8); q.add_argument('--steps', type=int, default=1600)
    q.add_argument('--every', type=int, default=40); q.add_argument('--seed', type=int, default=700000)
    for name in ('status', 'stop', 'down', 'restart'):
        q = sub.add_parser(name); q.add_argument('run', choices=PLANS)
        if name == 'restart':
            q.add_argument('--actors-only', action='store_true')
    a = ap.parse_args()
    globals()['cmd_' + a.cmd.replace('-', '_')](a)
