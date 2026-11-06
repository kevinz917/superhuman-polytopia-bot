"""Controller for the hero continuation on the Curio cluster, run from the Mac (docs/curio-hero-2026-10-31.md).

    python experiments/curio/curio.py check                      # read-only: reachability, GPUs, IB, disk, python
    python experiments/curio/curio.py stage                      # Modal -> Mac -> head -> workers: parent + references
    python experiments/curio/curio.py publish                    # code tarball + engine wheel to every box, venvs set up
    python experiments/curio/curio.py bench [--gpus 4,5,6,7] [--minibatch 16384]   # Learner.benchmark on the head GPUs
    python experiments/curio/curio.py launch --run hero-1v1-curio --hours 19        # learner + fleet + keeper (auto chain)
    python experiments/curio/curio.py status | logs [--lines 40] | stop [--all] | kill | fetch --run RUN

Nothing here runs on the Mac but ssh/scp/rsync and `modal volume get`. All cluster state lives under the head's root
(configs/cluster.json); the keeper (keeper.py) runs on the head so the Mac can sleep.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, hosts_from, load_plan, paths, learner_command, actor_command, LEARNER_ENV, ACTOR_ENV  # noqa: E402

REPO = HERE.parents[1]
STAGE = Path(os.environ.get('CURIO_STAGE', REPO / 'runs' / 'curio-hero' / 'stage'))
# Modal volume -> relative path under <root>/ckpt. 'results' = polytopia-structured-rl (staging), 'retained' = ...-retained.
CHECKPOINTS = {
    'hero-1v1-cont1h/latest': ('results', 'distributed/hero-1v1-cont1h/latest'),            # parent = u11650 (strongest)
    'hero-1v1-cont1h/latest-critic': ('results', 'distributed/hero-1v1-cont1h/latest-critic'),
    'hero-1v1/final': ('results', 'distributed/hero-1v1/final'),                            # hero1 reference (u14871)
    'bn-main4/final': ('results', 'distributed/bn-main4/final'),
    'scratch-cont-2-final/final': ('retained', 'checkpoints/distributed-final-v1/scratch-cont-2-final/final'),
    'scratch-cont-2-foghead/final': ('retained', 'checkpoints/distributed-final-v1/scratch-cont-2-foghead/final'),
    'c3-maps-ctl-update-4854/final': ('retained', 'checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final'),
}
VOLUMES = dict(results='polytopia-structured-rl', retained='polytopia-structured-rl-retained')
REFS = {'champion': 'scratch-cont-2-final/final.npz', 'champion_foghead': 'scratch-cont-2-foghead/final.npz',
        'bn4': 'bn-main4/final.npz', 'c3': 'c3-maps-ctl-update-4854/final.npz', 'hero1': 'hero-1v1/final.npz',
        'hero_u11650': 'hero-1v1-cont1h/latest.npz',   # the strongest 1v1 checkpoint so far
        'bighero_u3880': 'bighero-u3880/latest.npz',   # farsight's starting point before the history + pooling widen
        'bighero_u5250': 'bighero-u5250/latest.npz', 'bighero_u7750': 'bighero-u7750/latest.npz',   # equal-update
        'bighero_u10000': 'bighero-u10000/latest.npz', 'bighero_final': 'bighero-final/latest.npz'}  # control curve
START_REF = 'hero-1v1-cont1h/latest.npz'
PARENT, CRITIC = 'hero-1v1-cont1h/latest.npz', 'hero-1v1-cont1h/latest-critic.npz'


def sh(cmd, check=True, timeout=3600):
    print('$', cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, timeout=timeout, text=True)


def ctx(args):
    plan = load_plan(args.plan)
    hosts = hosts_from(plan)
    return plan, hosts, hosts['head'], paths(plan['head']['root'])


# ----- check ---------------------------------------------------------------------------------------------------------
def cmd_check(args):
    plan, hosts, head, p = ctx(args)
    print(json.dumps(dict(head=head.out('hostname; uptime; nvidia-smi --query-gpu=index,memory.used,utilization.gpu '
                                        '--format=csv,noheader; df -h / /data | tail -2; free -g | sed -n 2p; '
                                        f'ls {p["venv"]}/bin/python {p["code"]} 2>&1 | head -3; true', default='UNREACHABLE')),
                     indent=1))
    for w in plan['workers']:
        h = hosts[w['name']]
        wp = paths(w['root'])
        print(json.dumps({w['name']: h.out(f'hostname; uptime; free -g | sed -n 2p; df -h /home | tail -1; '
                                           f'ping -c1 -W2 {plan["head"]["data_ip"]} >/dev/null && echo ib_ok || echo ib_FAIL; '
                                           f'pgrep -c flagship-game; ls {wp["venv"]}/bin/python 2>&1; true', default='UNREACHABLE')},
                         indent=1))


# ----- stage ---------------------------------------------------------------------------------------------------------
def cmd_stage(args):
    plan, hosts, head, p = ctx(args)
    STAGE.mkdir(parents=True, exist_ok=True)
    for rel, (vol, remote) in CHECKPOINTS.items():
        for ext in ('.npz', '.json'):
            local = STAGE / (rel + ext)
            if local.exists() and not args.force:
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            sh(f'modal volume get --env staging --force {VOLUMES[vol]} {shlex.quote(remote + ext)} {shlex.quote(str(local))}')
    verify(STAGE)
    head.run(f'mkdir -p {p["ckpt"]}')
    sh(f'rsync -az -e {shlex.quote(" ".join(["ssh", "-o", "BatchMode=yes"]))} {shlex.quote(str(STAGE))}/ '
       f'{plan["head"]["ssh"]}:{p["ckpt"]}/')
    for w in plan['workers']:
        wp = paths(w['root'])
        hosts[w['name']].run(f'mkdir -p {wp["ckpt"]}')
        head.run(f'rsync -az -e "ssh -o BatchMode=yes" {p["ckpt"]}/ {w["ssh"]}:{wp["ckpt"]}/', timeout=1800)
    print(head.out(f'find {p["ckpt"]} -name "*.npz" | sort | xargs -I{{}} sh -c "echo {{}} $(stat -c %s {{}} 2>/dev/null)"'))


def verify(stage):
    """Every .npz has its .json and the sha256 in the json matches (what load_checkpoint checks)."""
    import hashlib
    for npz in sorted(stage.rglob('*.npz')):
        meta = json.loads(npz.with_suffix('.json').read_text())
        digest = hashlib.sha256(npz.read_bytes()).hexdigest()
        ok = meta.get('sha256') in (None, digest)
        print(f'{"ok " if ok else "BAD"} {npz.relative_to(stage)} stage={meta.get("stage")} ruleset={meta.get("ruleset")} '
              f'{npz.stat().st_size / 1e6:.1f} MB')
        if not ok:
            raise SystemExit(f'sha256 mismatch for {npz}')


# ----- publish -------------------------------------------------------------------------------------------------------
def cmd_publish(args):
    plan, hosts, head, p = ctx(args)
    sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    tree = subprocess.run(['git', 'rev-parse', 'HEAD:engine-rs'], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()[:12]
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'src', 'engine-rs', 'scripts', 'experiments/curio'],
                           cwd=REPO, capture_output=True, text=True).stdout.strip()
    if dirty and not args.allow_dirty:
        raise SystemExit('uncommitted changes in the published trees; commit first (or --allow-dirty):\n' + dirty)
    STAGE.parent.mkdir(parents=True, exist_ok=True)
    tar = STAGE.parent / f'code-{sha}.tar.gz'
    sh(f'cd {shlex.quote(str(REPO))} && git archive --format=tar --prefix=code/ HEAD src engine-rs scripts tests '
       f'experiments/curio experiments/hero pyproject.toml | gzip -1 > {shlex.quote(str(tar))}')
    head.run(f'mkdir -p {p["root"]}/code {p["wheels"]}')
    sh(f'scp -q -o BatchMode=yes {shlex.quote(str(tar))} {plan["head"]["ssh"]}:{p["root"]}/code/')
    unpack = (f'cd {p["root"]}/code && rm -rf {sha} && mkdir {sha} && tar xzf code-{sha}.tar.gz -C {sha} --strip-components=1 '
              f'&& echo {tree} > {sha}/ENGINE_TREE')
    switch = f' && ln -sfn {p["root"]}/code/{sha} {p["code"]}'
    if args.stage_only:
        # --stage-only: unpack beside the live code and build the engine wheel from the staged tree, but leave
        # code/current and the venv's installed engine alone (a live run keeps reading them; `switch` moves them)
        head.run(unpack, timeout=600)
        head.run(f'export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH; ls {p["wheels"]}/{tree}/*.whl >/dev/null 2>&1 || '
                 f'(cd {p["root"]}/code/{sha}/engine-rs && CARGO_TARGET_DIR={p["root"]}/cargo-target CARGO_BUILD_JOBS=32 '
                 f'{p["venv"]}/bin/maturin build --release -q -o {p["wheels"]}/{tree})', timeout=3600)
    else:
        head.run(unpack + switch, timeout=600)
        print(head.run(f'bash {p["curio"]}/setup_node.sh head {p["root"]}', timeout=3600, capture=True).stdout[-2000:])
    for w in plan['workers']:
        wp = paths(w['root'])
        h = hosts[w['name']]
        h.run(f'mkdir -p {wp["root"]}/code {wp["wheels"]}/{tree}')
        head.run(f'scp -q -o BatchMode=yes {p["root"]}/code/code-{sha}.tar.gz {w["ssh"]}:{wp["root"]}/code/ && '
                 f'scp -q -o BatchMode=yes {p["wheels"]}/{tree}/*.whl {w["ssh"]}:{wp["wheels"]}/{tree}/', timeout=600)
        wunpack = unpack.replace(p['root'], wp['root'])
        if args.stage_only:
            h.run(wunpack, timeout=600)
        else:
            h.run(wunpack + switch.replace(p['root'], wp['root']).replace(p['code'], wp['code']), timeout=600)
            print(h.run(f'bash {wp["curio"]}/setup_node.sh worker {wp["root"]}', timeout=3600, capture=True).stdout[-2000:])
    print(json.dumps(dict(published=sha, engine_tree=tree, staged_only=bool(args.stage_only))))


def cmd_switch(args):
    """Point code/current at a staged commit (publish --stage-only) on every box and install that tree's engine wheel
    into each venv. Run it with every process of the run stopped: a running process may still import lazily through
    code/current. --sha may also name an earlier commit (rollback)."""
    plan, hosts, head, p = ctx(args)
    sha = args.sha
    for name, h in hosts.items():
        hp = paths(h.root)
        if not h.exists(f'{hp["root"]}/code/{sha}/ENGINE_TREE'):
            raise SystemExit(f'{name}: {hp["root"]}/code/{sha} is not staged (curio.py publish --stage-only)')
        tree = h.read(f'{hp["root"]}/code/{sha}/ENGINE_TREE')
        if not h.exists(f'{hp["wheels"]}/{tree}'):
            raise SystemExit(f'{name}: no engine wheel for tree {tree}')
        h.run(f'export PATH=$HOME/.local/bin:$PATH; uv pip install -q -p {hp["python"]} --force-reinstall --no-deps '
              f'{hp["wheels"]}/{tree}/*.whl && ln -sfn {hp["root"]}/code/{sha} {hp["code"]}', timeout=600)
        print(name, h.out(f'readlink {hp["code"]}; {hp["python"]} -c "import polytopia_engine_rs as e; print(e.GameBatch(11, 1, open(\'{hp["src"]}/polytopia_bot/gpu/rules.json\').read()).view_width)"'))
    print(json.dumps(dict(switched=sha)))


# ----- bench ---------------------------------------------------------------------------------------------------------
def cmd_bench(args):
    plan, hosts, head, p = ctx(args)
    gpus = args.gpus or ','.join(map(str, plan['learner']['gpus']))
    cfg = f'{p["curio"]}/configs/{args.config}'
    envs = dict(LEARNER_ENV, CUDA_VISIBLE_DEVICES=gpus, PYTHONPATH=p['src'], JAX_COMPILATION_CACHE_DIR=p['cache'])
    for kv in args.env:   # K=V overrides, e.g. POLYTOPIA_HEAD_PRECISION=DEFAULT or XLA_FLAGS=...
        k, _, v = kv.partition('=')
        envs[k] = v
    env = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in envs.items())
    parent = args.parent or f'{p["ckpt"]}/{PARENT}'
    cmd = (f'cd {p["code"]} && env {env} {p["python"]} experiments/curio/node_learner.py bench --root {p["root"]} '
           f'--config {cfg} --parent {parent} --gpus {gpus} --epochs {args.epochs}'
           + (f' --minibatch {args.minibatch}' if args.minibatch else ''))
    res = head.run(cmd, check=False, timeout=3600)
    print(res.stdout[-6000:], res.stderr[-3000:])


# ----- launch --------------------------------------------------------------------------------------------------------
def cmd_launch(args):
    plan, hosts, head, p = ctx(args)
    run = args.run
    run_dir = f'{p["runs"]}/{run}'
    if head.exists(f'{run_dir}/latest.npz') and not args.resume:
        raise SystemExit(f'{run_dir} exists on the head; pass --resume to continue it or pick another --run')
    if head.pid_alive(f'{p["pids"]}/learner.pid') or head.pid_alive(f'{p["pids"]}/keeper.pid'):
        raise SystemExit('a learner or keeper is already running on the head (curio.py status / stop --all)')
    started = time.time()
    config = f'{p["curio"]}/configs/{args.config}'
    parent = args.parent or f'{p["ckpt"]}/{PARENT}'
    critic = '' if args.no_critic else (args.critic or f'{p["ckpt"]}/{CRITIC}')
    # --resumed-hours: training time already in the run's log (a copied run dir); the learner's clock resumes there and
    # --hours bounds the whole run, so the wall-clock end is that much sooner
    until = started + (args.hours - args.resumed_hours) * 3600
    fleet_until = until + args.fleet_extra_hours * 3600
    plan_out = dict(run=run, config=config, parent=parent, critic=critic, hours=args.hours, started_at=started,
                    until_unix=until, fleet_until=fleet_until, cluster=plan, refs=REFS,
                    start_ref=None if args.no_start_ref else START_REF, auto=not args.no_auto, eval=plan['eval'], search=plan['search'],
                    foghead=dict(config=f'{p["curio"]}/configs/hero-1v1-curio-foghead.json', hours=args.foghead_hours))
    head.write(f'{run_dir}/plan.json', json.dumps(plan_out, indent=1))
    for f in (parent, critic, config):
        if f and not head.exists(f):
            raise SystemExit(f'missing on the head: {f} (curio.py stage / publish)')
    env = dict(LEARNER_ENV, CUDA_VISIBLE_DEVICES=','.join(map(str, plan['learner']['gpus'])),
               JAX_COMPILATION_CACHE_DIR=p['cache'], PYTHONPATH=p['src'])
    print('learner:', head.start('learner', learner_command(plan, run, config, parent, critic, args.hours), env,
                                 p['code'], f'{p["logs"]}/learner-{run}.log', f'{p["pids"]}/learner.pid'))
    for _ in range(60):   # wait for the learner to serve (protocol.json appears when it binds) before the fleet
        if head.exists(f'{run_dir}/protocol.json'):
            break
        time.sleep(10)
    else:
        print('learner has not written protocol.json after 10 min; check logs. Starting the fleet anyway.')
    learner_ip = plan['head']['data_ip']
    for i, a in enumerate(plan['head_actors']):
        cmd = actor_command(plan, plan['head']['root'], i, learner_ip, a['processes'], a['games'], a['cores'], 'cuda',
                            fleet_until, f'{run_dir}/STOP-ALL', plan.get('serve_options'))
        aenv = dict(ACTOR_ENV, CUDA_VISIBLE_DEVICES=str(a['gpu']), JAX_COMPILATION_CACHE_DIR=p['cache'], PYTHONPATH=p['src'])
        print(f'head actor {i} (gpu {a["gpu"]}):', head.start(f'actor-{i}', cmd, aenv, p['code'], f'{p["logs"]}/actor-{i}.log',
                                                               f'{p["pids"]}/actor-{i}.pid'))
    wa = plan['worker_actors']
    for j, w in enumerate(plan['workers'] if wa.get('processes') else []):
        h, wp = hosts[w['name']], paths(w['root'])
        cmd = actor_command(plan, w['root'], 100 + j, learner_ip, wa['processes'], wa['games'], wa['cores'], 'cpu', fleet_until)
        aenv = dict(ACTOR_ENV, JAX_COMPILATION_CACHE_DIR=wp['cache'], PYTHONPATH=wp['src'])
        print(f'{w["name"]} actor {100 + j}:', h.start(f'actor-{100 + j}', cmd, aenv, wp['code'],
                                                       f'{wp["logs"]}/actor-{100 + j}.log', f'{wp["pids"]}/actor-{100 + j}.pid'))
    print('keeper:', head.start('keeper', f'{p["python"]} experiments/curio/keeper.py --root {p["root"]} --run {run}',
                                dict(PYTHONPATH=p['src']), p['code'], f'{p["logs"]}/keeper.log', f'{p["pids"]}/keeper.pid'))
    print(json.dumps(dict(run=run, started_at=started, training_ends=time.ctime(plan_out['until_unix']),
                          fleet_until=time.ctime(fleet_until), auto=plan_out['auto'])))


# ----- status / logs / stop / kill / fetch ---------------------------------------------------------------------------
def current_run(head, p):
    return head.out(f'ls -t {p["runs"]} 2>/dev/null | head -1')


def cmd_status(args):
    plan, hosts, head, p = ctx(args)
    run = args.run or current_run(head, p)
    rd = f'{p["runs"]}/{run}'
    print(f'== run {run}')
    print(head.out(f'cat {rd}/keeper-state.json 2>/dev/null | {p["python"]} -c "import json,sys,time; s=json.load(sys.stdin); '
                   f'print(\'phase\', s[\'phase\'], \'learner_restarts\', s[\'learner_restarts\'], \'ladder_round\', '
                   f's[\'ladder_round\'], \'next_ladder in\', int(s[\'next_ladder\']-time.time()), \'s\'); '
                   f'[print(e) for e in s[\'events\'][-6:]]"', default='(no keeper state)'))
    print('-- learner status:', head.read(f'{rd}/learner-status.json', default='(none)')[:1500])
    print('-- learner alive:', head.pid_alive(f'{p["pids"]}/learner.pid'), ' keeper alive:', head.pid_alive(f'{p["pids"]}/keeper.pid'))
    print('-- last updates:')
    print(head.out(f'grep -a \'"stage": "update"\' {p["logs"]}/learner-{run}.log | tail -3 | cut -c1-600', default=''))
    print('-- gpus:')
    print(head.out('nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'))
    for i in range(len(plan['head_actors'])):
        print(f'-- head actor {i}:', head.pid_alive(f'{p["pids"]}/actor-{i}.pid'), head.read(f'{p["logs"]}/actor-{i}.json', default='')[:300])
    for j, w in enumerate(plan['workers']):
        h, wp = hosts[w['name']], paths(w['root'])
        print(f'-- {w["name"]} actor:', h.pid_alive(f'{wp["pids"]}/actor-{100 + j}.pid'), h.read(f'{wp["logs"]}/actor-{100 + j}.json', default='')[:300])
    print('-- ladder history (last rows):')
    print(head.out(f'tail -c 1500 {p["ladder"]}/{run}-history.json 2>/dev/null', default='(none yet)'))


def cmd_logs(args):
    plan, hosts, head, p = ctx(args)
    run = args.run or current_run(head, p)
    for name in (f'learner-{run}', 'keeper', 'actor-0'):
        print(f'== {name}')
        print(head.out(f'tail -n {args.lines} {p["logs"]}/{name}.log | cut -c1-400', default='(no log)'))
    for j, w in enumerate(plan['workers']):
        wp = paths(w['root'])
        print(f'== {w["name"]} actor-{100 + j}')
        print(hosts[w['name']].out(f'tail -n {args.lines} {wp["logs"]}/actor-{100 + j}.log | cut -c1-400', default='(no log)'))


def cmd_stop(args):
    plan, hosts, head, p = ctx(args)
    run = args.run or current_run(head, p)
    marker = 'STOP-ALL' if args.all else 'STOP'
    head.run(f'date > {p["runs"]}/{run}/{marker}')
    print(f'{marker} written for {run}: ' + ('the keeper tears the fleet down and exits' if args.all else
                                           'the learner writes final.npz and the keeper continues the chain'))


def cmd_kill(args):
    plan, hosts, head, p = ctx(args)
    for name, h in hosts.items():
        hp = paths(h.root)
        h.run(f'for f in {hp["pids"]}/*.pid; do [ -f "$f" ] && kill -TERM -- -$(cat $f) 2>/dev/null; done; sleep 5; '
              f'pkill -9 -f "^{hp["python"]} "; '   # only our venv's interpreter: the boxes are shared
              f'rm -f {hp["pids"]}/*.pid; true', check=False, timeout=120)
        print(name, 'killed')


def cmd_restart(args):
    """Restart the keeper and the learner from the run's plan.json on the head (after a `publish`, so both pick up the
    new code and env). Actor nodes keep running and reconnect; the learner resumes latest.npz."""
    plan, hosts, head, p = ctx(args)
    run = args.run or current_run(head, p)
    rd = f'{p["runs"]}/{run}'
    stored = json.loads(head.read(f'{rd}/plan.json'))
    stored.update(cluster=plan, eval=plan['eval'], search=plan['search'])   # topology changes take effect on restart
    head.write(f'{rd}/plan.json', json.dumps(stored, indent=1))
    head.stop(f'{p["pids"]}/keeper.pid', grace=2)
    head.stop(f'{p["pids"]}/learner.pid', grace=10)
    train = stored.get('phase_run', run)
    env = dict(LEARNER_ENV, CUDA_VISIBLE_DEVICES=','.join(map(str, plan['learner']['gpus'])),
               JAX_COMPILATION_CACHE_DIR=p['cache'], PYTHONPATH=p['src'])
    print('learner:', head.start('learner', learner_command(plan, train, stored['config'], stored['parent'], stored['critic'],
                                                            stored['hours']), env, p['code'], f'{p["logs"]}/learner-{train}.log',
                                 f'{p["pids"]}/learner.pid'))
    print('keeper:', head.start('keeper', f'{p["python"]} experiments/curio/keeper.py --root {p["root"]} --run {run}',
                                dict(PYTHONPATH=p['src']), p['code'], f'{p["logs"]}/keeper.log', f'{p["pids"]}/keeper.pid'))


def cmd_fetch(args):
    plan, hosts, head, p = ctx(args)
    run = args.run or current_run(head, p)
    local = STAGE.parent / run
    local.mkdir(parents=True, exist_ok=True)
    files = ' '.join(f'{p["runs"]}/{run}/{f}' for f in ('summary.md', 'plan.json', 'keeper-state.json', 'training.jsonl',
                                                        'learner-done.json'))
    sh(f'scp -q -o BatchMode=yes {plan["head"]["ssh"]}:"{files} {p["ladder"]}/{run}-history.json" {shlex.quote(str(local))}/',
       check=False)
    sh(f'rsync -az -e "ssh -o BatchMode=yes" {plan["head"]["ssh"]}:{p["arena"]}/ {shlex.quote(str(local))}/arena/', check=False)
    if args.weights:
        for r in (run, f'{run}-foghead'):
            sh(f'rsync -az -e "ssh -o BatchMode=yes" --include="final*" --include="latest*" --exclude="*" '
               f'{plan["head"]["ssh"]}:{p["runs"]}/{r}/ {shlex.quote(str(local))}/{r}/', check=False)
    print('fetched to', local)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default=None, help='cluster json (default configs/cluster.json)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check').set_defaults(fn=cmd_check)
    s = sub.add_parser('stage'); s.add_argument('--force', action='store_true'); s.set_defaults(fn=cmd_stage)
    s = sub.add_parser('publish'); s.add_argument('--allow-dirty', action='store_true')
    s.add_argument('--stage-only', action='store_true', help='unpack + build the wheel, do not move code/current'); s.set_defaults(fn=cmd_publish)
    s = sub.add_parser('switch'); s.add_argument('--sha', required=True); s.set_defaults(fn=cmd_switch)
    s = sub.add_parser('bench'); s.add_argument('--gpus', default=''); s.add_argument('--minibatch', type=int, default=0)
    s.add_argument('--epochs', type=int, default=4); s.add_argument('--config', default='hero-1v1-curio.json')
    s.add_argument('--parent', default=''); s.add_argument('--env', action='append', default=[]); s.set_defaults(fn=cmd_bench)
    s = sub.add_parser('launch')
    s.add_argument('--run', required=True); s.add_argument('--hours', type=float, required=True)
    s.add_argument('--config', default='hero-1v1-curio.json'); s.add_argument('--parent', default='')
    s.add_argument('--critic', default=''); s.add_argument('--no-critic', action='store_true')
    s.add_argument('--no-auto', action='store_true', help='no fog head / search chain after training')
    s.add_argument('--foghead-hours', type=float, default=1.0)
    s.add_argument('--no-start-ref', action='store_true', help='no extra start reference in the final ladder (from scratch)')
    s.add_argument('--fleet-extra-hours', type=float, default=2.5, help='actors stay up this long past training')
    s.add_argument('--resumed-hours', type=float, default=0., help='training hours already logged in the run dir (with --resume)')
    s.add_argument('--resume', action='store_true'); s.set_defaults(fn=cmd_launch)
    for name, fn in (('status', cmd_status), ('logs', cmd_logs), ('stop', cmd_stop), ('fetch', cmd_fetch), ('restart', cmd_restart)):
        s = sub.add_parser(name); s.add_argument('--run', default='')
        if name == 'logs':
            s.add_argument('--lines', type=int, default=30)
        if name == 'stop':
            s.add_argument('--all', action='store_true')
        if name == 'fetch':
            s.add_argument('--weights', action='store_true')
        s.set_defaults(fn=fn)
    sub.add_parser('kill').set_defaults(fn=cmd_kill)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
