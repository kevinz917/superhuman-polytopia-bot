"""Long Verda training runs whose control lives on Modal, not on the Mac. The Mac creates and provisions the learner,
then arms the run; from then on modal_controller.py (deployed, every 3 min) creates and replaces the actors (spot),
syncs checkpoints to /results/distributed/<run>/, and tears everything down at the deadline, the spend cap, a lost or
stalled learner, or the end of training. The Mac can sleep or be off after `arm`.

    python experiments/verda/bigrun.py init --run R --config CFG.json --parent /results/... [--critic ...] ...
    python experiments/verda/bigrun.py keys        # one-time: controller SSH key + the Modal secrets (prints steps)
    python experiments/verda/bigrun.py preflight --run R          # balance, prices, availability, plan $/h (free)
    python experiments/verda/bigrun.py up --run R                 # create the learner VM (starts billing)
    python experiments/verda/bigrun.py setup --run R              # provision it, build the engine wheel, upload bundle
    python experiments/verda/bigrun.py bench --run R --devices 0,1;0,1,2,3 --minibatches 4096,8192,16384
    python experiments/verda/bigrun.py start --run R              # start the learner service (resumes on crash)
    python experiments/verda/bigrun.py arm --run R                # hand the run to the Modal controller
    python experiments/verda/bigrun.py status --run R
    python experiments/verda/bigrun.py scale --run R --actors N [--cap USD]
    python experiments/verda/bigrun.py down --run R [--local]     # teardown via the controller (or the Mac's CLI)

Local state: runs/verda-<run>/plan.json. Modal volume polytopia-structured-rl (env staging): verda/<run>/ holds the
bundle, controller.json, overrides.json, controller-log.jsonl and actors/*.json; distributed/<run>/ the learner files.
Credentials: the Verda CLI's own store on the Mac; the controller's come from Modal secrets the user creates.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VOLUMES = {'/results/': 'polytopia-structured-rl', '/retained/': 'polytopia-structured-rl-retained'}
MAC_KEY = Path.home() / '.ssh/verda_polytopia'                       # the pilot's key, registered as polytopia-verda
CONTROLLER_KEY = Path.home() / '.ssh/verda_controller'               # bigrun.py keys; its private half -> verda-ssh
KEY_NAMES = ('polytopia-verda', 'polytopia-verda-controller')
REMOTE_ENV = 'PATH=/opt/venv/bin:$PATH PYTHONPATH=/workspace/src XLA_PYTHON_CLIENT_PREALLOCATE=false'


def run_dir(run):
    return ROOT / 'runs' / f'verda-{run}'


def load(run):
    return json.loads((run_dir(run) / 'plan.json').read_text())


def save(plan):
    d = run_dir(plan['run'])
    d.mkdir(parents=True, exist_ok=True)
    (d / 'plan.json').write_text(json.dumps(plan, indent=1))


def verda(*args, check=True):
    out = subprocess.run(['verda', '--agent', *map(str, args)], text=True, capture_output=True)
    if check and out.returncode != 0:
        raise SystemExit(f'verda {" ".join(map(str, args))} -> {out.returncode}: {(out.stderr or out.stdout)[-800:]}')
    try:
        return json.loads(out.stdout) if out.stdout.strip() else None
    except json.JSONDecodeError:
        return out.stdout.strip()


def records(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from records(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from records(v)


def ssh(plan, command, check=True, capture=False, timeout=None):
    args = ['ssh', '-i', str(MAC_KEY), '-o', 'StrictHostKeyChecking=accept-new', '-o',
            f'UserKnownHostsFile={run_dir(plan["run"]) / "known_hosts"}', '-o', 'ConnectTimeout=20',
            f'root@{plan["learner"]["ip"]}', command]
    return subprocess.run(args, check=check, text=True, capture_output=capture, timeout=timeout)


def scp(plan, src, dst):
    subprocess.run(['scp', '-q', '-i', str(MAC_KEY), '-o', 'StrictHostKeyChecking=accept-new', '-o',
                    f'UserKnownHostsFile={run_dir(plan["run"]) / "known_hosts"}', str(src), str(dst)], check=True)


def on_learner(plan, path):
    return f'root@{plan["learner"]["ip"]}:{path}'


def volume_get(path, local):
    vol = next(v for k, v in VOLUMES.items() if path.startswith(k))
    rel = path.split('/', 2)[2]
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['modal', 'volume', 'get', '--env', 'staging', '--force', vol, rel, str(local)], check=True)


def volume_put(local, rel):
    subprocess.run(['modal', 'volume', 'put', '--env', 'staging', '--force', 'polytopia-structured-rl', str(local),
                    rel], check=True)


def prices():
    return {r['instance_type']: r for r in records(verda('instance-types')) if 'instance_type' in r}


def hourly(plan, p=None):
    p = p or prices()
    actor = float(p[plan['actor_type']]['spot_price' if plan['spot'] else 'price_per_hour'])
    return float(p[plan['learner_type']]['price_per_hour']), actor


def key_ids(require_controller=True):
    """Verda SSH key ids put on every VM: the Mac's pilot key, and the controller's (needed once the run is armed;
    a learner created without it cannot be armed)."""
    ids = {k.get('name'): k['id'] for k in records(verda('ssh-key', 'list')) if 'id' in k and k.get('name')}
    missing = [n for n in KEY_NAMES[:2 if require_controller else 1] if n not in ids]
    if missing:
        raise SystemExit(f'SSH keys not registered on Verda: {missing} (see bigrun.py keys)')
    return [ids[n] for n in KEY_NAMES if n in ids]


# ---------------------------------------------------------------- commands

def cmd_init(a):
    config = json.loads(Path(a.config).read_text())
    config['training_seconds'] = int(a.hours * 3600)
    plan = dict(run=a.run, config=config, parent=a.parent, critic=a.critic, learner_type=a.learner_type,
                actor_type=a.actor_type, actors=a.actors, spot=a.spot, hours=a.hours, slack_hours=a.slack_hours,
                cap_usd=a.cap, location=a.location, games=a.games, processes=a.processes, ref=a.ref,
                learner_image=a.learner_image, actor_image=a.actor_image, created=time.time())
    save(plan)
    print(json.dumps(plan, indent=1))


def cmd_keys(a):
    if not CONTROLLER_KEY.exists():
        print(f'1. ssh-keygen -t ed25519 -N "" -C polytopia-verda-controller -f {CONTROLLER_KEY}')
    print(f'2. verda ssh-key add --name polytopia-verda-controller --public-key "$(cat {CONTROLLER_KEY}.pub)"')
    print(f'3. modal secret create --env staging verda-ssh VERDA_SSH_KEY="$(cat {CONTROLLER_KEY})"')
    print('4. modal secret create --env staging verda-api VERDA_CLIENT_ID=<id> VERDA_CLIENT_SECRET=<secret>'
          '   (from the Verda console; the same pair as `verda auth login`)')
    print('5. modal deploy --env staging experiments/verda/modal_controller.py')


def cmd_preflight(a):
    plan = load(a.run)
    print('balance', json.dumps(verda('cost', 'balance')))
    p = prices()
    for t in (plan['learner_type'], plan['actor_type']):
        print(t, 'price/h', p[t].get('price_per_hour'), 'spot', p[t].get('spot_price'), 'availability',
              json.dumps(verda('availability', '--location', plan['location'], '--type', t))[:300])
    learner, actor = hourly(plan, p)
    total = learner + plan['actors'] * actor
    hours = plan['hours'] + plan['slack_hours']
    print(f'plan: {plan["learner_type"]} ${learner:.2f}/h + {plan["actors"]}x {plan["actor_type"]} '
          f'{"spot" if plan["spot"] else "on-demand"} ${actor:.3f}/h = ${total:.2f}/h; '
          f'{hours:.1f} h worst case ${total * hours:.0f} (cap ${plan["cap_usd"]:.0f})')
    print('keys', key_ids())


def cmd_up(a):
    plan = load(a.run)
    if plan.get('learner', {}).get('id'):
        raise SystemExit(f'learner already exists: {plan["learner"]}')
    hostname = f'{plan["run"]}-learner-0'
    args = ['vm', 'create', '--kind', 'gpu', '--instance-type', plan['learner_type'], '--location', plan['location'],
            '--os', plan['learner_image'], '--os-volume-size', 200, '--hostname', hostname,
            '--description', f'polytopia {plan["run"]} learner', '--contract', 'pay_as_go', '--wait=false']
    keys = key_ids(require_controller=a.armable)
    for k in keys:
        args += ['--ssh-key', k]
    verda(*args)
    plan['learner'] = dict(hostname=hostname, requested=time.time(), ssh_key_ids=keys)
    save(plan)
    for _ in range(90):
        row = next((r for r in records(verda('vm', 'list')) if r.get('hostname') == hostname), {})
        print(hostname, row.get('status'), row.get('ip'), flush=True)
        if row.get('status') in ('error', 'no_capacity', 'installation_failed'):
            raise SystemExit('learner deploy failed; run down --local')
        if row.get('status') == 'running' and row.get('ip'):
            plan['learner'].update(id=row['id'], ip=row['ip'], running=time.time())
            save(plan)
            return
        time.sleep(20)
    raise SystemExit('learner not running after 30 min; run down --local')


def write_bundle(plan):
    """code.tar: src + engine-rs at plan['ref']; ckpt/: parent, critic, frozen opponents (paths rewritten to the
    node's /workspace/pilot/ckpt/); config.json."""
    out = run_dir(plan['run']) / 'bundle'
    (out / 'ckpt' / 'frozen').mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', '-C', str(ROOT), 'archive', '--format=tar', '-o', str(out / 'code.tar'), plan['ref'],
                    'src', 'engine-rs'], check=True)
    config = dict(plan['config'])
    fetch = [(plan['parent'], out / 'ckpt' / 'final.npz')]
    if plan.get('critic'):
        fetch.append((plan['critic'], out / 'ckpt' / 'critic.npz'))
        config['critic_path'] = '/workspace/pilot/ckpt/critic.npz'
    frozen = []
    for name, path, prob in config.get('frozen', []):
        fetch.append((path, out / 'ckpt' / 'frozen' / f'{name}.npz'))
        frozen.append([name, f'/workspace/pilot/ckpt/frozen/{name}.npz', prob])
    config['frozen'] = frozen
    for src, dst in fetch:
        volume_get(src, dst)
        volume_get(src[:-len('.npz')] + '.json', dst.with_suffix('.json'))
    (out / 'config.json').write_text(json.dumps(config, indent=1))
    return out


def pack(plan, out):
    names = ['ckpt', 'config.json'] + (['wheels'] if (out / 'wheels').exists() else [])
    subprocess.run(['tar', '-C', str(out), '-cf', str(out / 'pilot.tar'), *names, '-C', str(ROOT / 'experiments/verda'),
                    'node_learner.py', 'node_actor.py', 'setup_node.sh'], check=True)


def cmd_setup(a):
    plan = load(a.run)
    out = write_bundle(plan)
    pack(plan, out)
    ssh(plan, 'mkdir -p /workspace/pilot')
    scp(plan, out / 'code.tar', on_learner(plan, '/workspace/code.tar'))
    scp(plan, out / 'pilot.tar', on_learner(plan, '/workspace/pilot.tar'))
    r = ssh(plan, 'cd /workspace && tar -xf code.tar && tar -C pilot -xf pilot.tar && '
                  'bash pilot/setup_node.sh learner 2>&1 | tail -3', capture=True)
    print(r.stdout.strip())
    # the engine wheel once, here, so controller-provisioned actors install it instead of compiling (same x86/glibc)
    r = ssh(plan, 'curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable '
                  '>/dev/null && /root/.local/bin/uv pip install -q -p /opt/venv/bin/python "maturin>=1.7,<2" && '
                  'cd /workspace/engine-rs && PATH=/root/.cargo/bin:/opt/venv/bin:$PATH maturin build --release -q '
                  '-o /workspace/pilot/wheels && ls /workspace/pilot/wheels', capture=True)
    wheels = [w for w in r.stdout.split() if w.endswith('.whl')]
    if len(wheels) != 1:
        raise SystemExit(f'expected one engine wheel, got {r.stdout} {r.stderr[-500:]}')
    (out / 'wheels').mkdir(exist_ok=True)
    scp(plan, on_learner(plan, f'/workspace/pilot/wheels/{wheels[0]}'), out / 'wheels' / wheels[0])
    pack(plan, out)
    for name in ('code.tar', 'pilot.tar'):
        volume_put(out / name, f'verda/{plan["run"]}/bundle/{name}')
    plan['setup_done'] = time.time()
    save(plan)
    print('bundle uploaded:', [p.name for p in out.iterdir()])


def cmd_bench(a):
    plan = load(a.run)
    rows = []
    for devices in a.devices.split(';'):
        for mb in map(int, a.minibatches.split(',')):
            r = ssh(plan, f'cd /workspace/pilot && CUDA_VISIBLE_DEVICES={devices} JAX_PLATFORMS=cuda {REMOTE_ENV} '
                          f'python node_learner.py bench {a.epochs} {mb} 2>&1 | tail -4', check=False, capture=True)
            result = next((json.loads(l) for l in r.stdout.splitlines() if '"bench_result"' in l), None)
            row = dict(devices=devices, minibatch=mb, result=result, tail=None if result else r.stdout[-600:])
            if result:
                eps = [x['samples_per_second'] for x in result['rows'][1:]] or [result['rows'][0]['samples_per_second']]
                row['samples_per_second'] = sum(eps) / len(eps)
            print(json.dumps({k: v for k, v in row.items() if k != 'result'}), flush=True)
            rows.append(row)
    with (run_dir(plan['run']) / 'bench.jsonl').open('a') as f:
        for row in rows:
            f.write(json.dumps(dict(row, at=time.time())) + '\n')


def cmd_start(a):
    plan = load(a.run)
    ssh(plan, 'rm -rf /workspace/run /workspace/learner.log && systemctl enable --now polytopia-learner')
    for _ in range(90):
        if ssh(plan, 'ss -ltn | grep -q ":7070 "', check=False).returncode == 0:
            plan['learner']['started'] = time.time()
            save(plan)
            print('learner listening on 7070')
            return
        time.sleep(10)
    raise SystemExit('learner never listened on 7070; see /workspace/learner.log')


def cmd_arm(a):
    plan = load(a.run)
    if not plan.get('learner', {}).get('started'):
        raise SystemExit('start the learner first')
    if len(plan['learner'].get('ssh_key_ids', [])) < 2:
        raise SystemExit('the learner was created without the controller key (up --no-armable); it cannot be armed')
    p = prices()
    learner_rate, actor_rate = hourly(plan, p)
    bal = next((float(r[k]) for r in records(verda('cost', 'balance')) for k in ('amount', 'balance', 'available',
                'value') if isinstance(r.get(k), (int, float))), None)
    now = time.time()
    state = dict(run=plan['run'], status='armed', armed_at=now, learner=plan['learner']['hostname'],
                 learner_ip=plan['learner']['ip'], learner_started=plan['learner']['started'],
                 deadline=plan['learner']['started'] + (plan['hours'] + plan['slack_hours']) * 3600,
                 cap_usd=plan['cap_usd'], start_balance=bal, spent_estimate_usd=(now - plan['learner']['running'])
                 / 3600 * learner_rate, last_tick=now, hourly={plan['learner']['hostname']: learner_rate},
                 hourly_default=float(p[plan['actor_type']]['price_per_hour']), actor_hourly=actor_rate,
                 desired_actors=plan['actors'], actor_type=plan['actor_type'], actor_image=plan['actor_image'],
                 location=plan['location'], spot=plan['spot'], ssh_key_ids=key_ids(), games=plan['games'],
                 processes=plan['processes'], actors={})
    path = run_dir(plan['run']) / 'controller.json'
    path.write_text(json.dumps(state, indent=1))
    volume_put(path, f'verda/{plan["run"]}/controller.json')
    plan['armed'] = now
    save(plan)
    apps = subprocess.run(['modal', 'app', 'list', '--env', 'staging', '--json'], capture_output=True, text=True).stdout
    deployed = any(r.get('description') == 'verda-controller' and r.get('state') == 'deployed'
                   for r in json.loads(apps or '[]'))
    print(f'armed: deadline {time.ctime(state["deadline"])}, cap ${state["cap_usd"]}, start balance {bal}; '
          f'controller {"deployed" if deployed else "NOT DEPLOYED: modal deploy --env staging experiments/verda/modal_controller.py"}')


def cmd_status(a):
    plan = load(a.run)
    d = run_dir(plan['run'])
    for name in ('controller.json', 'controller-log.jsonl'):
        subprocess.run(['modal', 'volume', 'get', '--env', 'staging', '--force', 'polytopia-structured-rl',
                        f'verda/{plan["run"]}/{name}', str(d / f'remote-{name}')], capture_output=True)
    if (d / 'remote-controller.json').exists():
        s = json.loads((d / 'remote-controller.json').read_text())
        print({k: s.get(k) for k in ('status', 'stop_reason', 'spent_usd', 'cap_usd', 'balance', 'desired_actors')},
              {k: a['phase'] for k, a in s.get('actors', {}).items()}, 'replaced', len(s.get('replaced', [])))
    if (d / 'remote-controller-log.jsonl').exists():
        for line in (d / 'remote-controller-log.jsonl').read_text().splitlines()[-3:]:
            print(line[:400])
    r = ssh(plan, 'tail -n 1 /workspace/run/training.jsonl', check=False, capture=True, timeout=60)
    try:
        row = json.loads(r.stdout)
        print({k: row.get(k) for k in ('update', 'samples', 'seconds', 'train_seconds', 'ready_wait_seconds',
                                       'received_samples_per_second', 'connections', 'relays', 'learning_rate')})
    except json.JSONDecodeError:
        print('learner:', r.stdout[-300:], r.stderr[-300:])


def cmd_scale(a):
    plan = load(a.run)
    overrides = {'desired_actors': a.actors} if a.actors is not None else {}
    if a.cap is not None:
        overrides['cap_usd'] = a.cap
    if a.actor_options:                        # "TYPE:spot[:LOCATION],TYPE:ondemand" in preference order
        p = prices()
        overrides['actor_options'] = [[t, m == 'spot', float(p[t]['spot_price' if m == 'spot' else 'price_per_hour']),
                                       *loc] for t, m, *loc in (x.split(':') for x in a.actor_options.split(','))]
    path = run_dir(plan['run']) / 'overrides.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **overrides}))
    volume_put(path, f'verda/{plan["run"]}/overrides.json')
    print('overrides', {**old, **overrides})


def cmd_down(a):
    plan = load(a.run)
    if not a.local and plan.get('armed'):
        subprocess.run(['modal', 'run', '--env', 'staging', str(ROOT / 'experiments/verda/modal_controller.py') +
                        '::teardown', '--run', plan['run'], '--reason', 'bigrun down'], check=True)
    else:
        verda('vm', 'delete', '--all', '--hostname', f'{plan["run"]}-*', '--with-volumes', '--yes',
              '--wait-timeout', '10m', check=False)
        for _ in range(60):
            if not [r for r in records(verda('vm', 'list')) if str(r.get('hostname', '')).startswith(f'{plan["run"]}-')]:
                break
            time.sleep(10)
        for v in records(verda('volume', 'list', check=False)):
            if str(v.get('name', '')).startswith(f'{plan["run"]}-') and v.get('id'):
                verda('volume', 'delete', v['id'], '--yes', check=False)
    left = [r.get('hostname') for r in records(verda('vm', 'list')) if str(r.get('hostname', '')).startswith(f'{plan["run"]}-')]
    vols = [v.get('name') for v in records(verda('volume', 'list', check=False))
            if str(v.get('name', '')).startswith(f'{plan["run"]}-')]
    print('vms left', left, 'volumes left', vols)
    plan.setdefault('history', []).append(dict(down=time.time(), vms_left=left, volumes_left=vols))
    plan.pop('learner', None)
    save(plan)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['init', 'keys', 'preflight', 'up', 'setup', 'bench', 'start', 'arm', 'status',
                                        'scale', 'down'])
    ap.add_argument('--run')
    ap.add_argument('--config')
    ap.add_argument('--parent')
    ap.add_argument('--critic', default='')
    ap.add_argument('--learner-type', default='4RTXPRO6000.120V')
    ap.add_argument('--actor-type', default='CPU-TURIN.120V.480G')   # the pilot's measured CPU family
    ap.add_argument('--actors', type=int)
    ap.add_argument('--spot', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--hours', type=float, default=1.)
    ap.add_argument('--slack-hours', type=float, default=1.)
    ap.add_argument('--cap', type=float)
    ap.add_argument('--location', default='FIN-03')
    ap.add_argument('--games', type=int, default=24)
    ap.add_argument('--processes', type=int, default=0)
    ap.add_argument('--ref', default='HEAD')
    ap.add_argument('--learner-image', default='24.04.cuda12.9')   # driver for the pip CUDA 12 JAX wheels
    ap.add_argument('--actor-image', default='24.04.base')
    ap.add_argument('--devices', default='0,1;0,1,2,3')
    ap.add_argument('--minibatches', default='4096,8192,16384')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--local', action='store_true')
    ap.add_argument('--actor-options', default='')
    ap.add_argument('--armable', action=argparse.BooleanOptionalAction, default=True)   # up: put the controller key on
    a = ap.parse_args()
    globals()[f'cmd_{a.command}'](a)
