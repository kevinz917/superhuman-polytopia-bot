"""Verda pilot: the scratch-cont-2 recipe's real learner + actors on Verda instead of Modal, to measure samples/s
and $/sample. Runs from the Mac but only calls the Verda CLI and ssh/scp; all compute is on Verda.

    python experiments/verda/pilot.py preflight            # balance, prices, availability, images (free)
    python experiments/verda/pilot.py up                   # deploy learner + actors (starts billing)
    python experiments/verda/pilot.py setup                # firewall, venv, JAX, engine-rs, code + checkpoints
    python experiments/verda/pilot.py bench                # learner benchmark on 1 and 2 GPUs
    python experiments/verda/pilot.py run --seconds 1500   # real learner + actors
    python experiments/verda/pilot.py status
    python experiments/verda/pilot.py down                 # delete every pilot-* VM with its volumes
    python experiments/verda/pilot.py watchdog --minutes 120   # sleep, then down

Credentials: the Verda CLI's own store (`verda auth login`, ~/.verda/credentials); this script never reads them.
Teardown is `vm delete --with-volumes`: shutdown keeps billing, and a detached OS volume keeps billing too.
"""
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEY = Path.home() / '.ssh/verda_polytopia'
STATE_DIR = ROOT / 'runs/verda-pilot'
STATE = STATE_DIR / 'state.json'
LOCATION = 'FIN-03'
LEARNER_TYPE = '2RTXPRO6000.60V'
ACTOR_TYPE = 'CPU-TURIN.32V.128G'
PARENT = 'distributed/scratch-cont-2'       # on the Modal volume polytopia-structured-rl
LEDGER = ROOT / 'runs/scratch-cont-2/budget.json'
REMOTE_ENV = 'PATH=/opt/venv/bin:$PATH PYTHONPATH=/workspace/src XLA_PYTHON_CLIENT_PREALLOCATE=false'


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else dict(instances=[])


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))


def verda(*args, check=True):
    """`verda --agent ...` (JSON out, no prompts, structured errors) -> parsed JSON."""
    out = subprocess.run(['verda', '--agent', *map(str, args)], text=True, capture_output=True)
    if check and out.returncode != 0:
        raise SystemExit(f'verda {" ".join(map(str, args))} -> {out.returncode}: {(out.stderr or out.stdout)[-800:]}')
    try:
        return json.loads(out.stdout) if out.stdout.strip() else None
    except json.JSONDecodeError:
        return out.stdout.strip()


def records(obj):
    """Every dict in a CLI JSON reply (lists may sit under a wrapper key)."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from records(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from records(v)


def ssh(ip, command, check=True, capture=False):
    args = ['ssh', '-i', str(KEY), '-o', 'StrictHostKeyChecking=accept-new', '-o',
            f'UserKnownHostsFile={STATE_DIR / "known_hosts"}', '-o', 'ConnectTimeout=20', f'root@{ip}', command]
    return subprocess.run(args, check=check, text=True, capture_output=capture)


def scp(ip, local, remote):
    subprocess.run(['scp', '-q', '-i', str(KEY), '-o', 'StrictHostKeyChecking=accept-new', '-o',
                    f'UserKnownHostsFile={STATE_DIR / "known_hosts"}', str(local), f'root@{ip}:{remote}'], check=True)


def prices():
    return {r['instance_type']: r for r in records(verda('instance-types')) if 'instance_type' in r}


def plan_hourly(actors):
    p = prices()
    return float(p[LEARNER_TYPE]['price_per_hour']) + actors * float(p[ACTOR_TYPE]['price_per_hour'])


def cmd_preflight(a):
    print('balance', json.dumps(verda('cost', 'balance')))
    p = prices()
    for t in (LEARNER_TYPE, ACTOR_TYPE):
        print(t, 'price/h', p[t].get('price_per_hour'), 'spot', p[t].get('spot_price'), 'availability',
              json.dumps(verda('availability', '--location', LOCATION, '--type', t))[:300])
    print(f'plan: 1x {LEARNER_TYPE} + {a.actors}x {ACTOR_TYPE} = ${plan_hourly(a.actors):.2f}/h')
    for t, image in ((LEARNER_TYPE, a.learner_image), (ACTOR_TYPE, a.actor_image)):
        ok = any(r.get('image_type') == image for r in records(verda('images', '--type', t)))
        print('image', image, 'for', t, 'ok' if ok else 'MISSING')
    print('running now:', json.dumps(verda('vm', 'list'))[:500])


def ssh_key_id():
    public = KEY.with_suffix('.pub').read_text().strip()
    def find():
        return next((k['id'] for k in records(verda('ssh-key', 'list')) if 'id' in k and
                     (public in map(lambda v: str(v).strip(), k.values()) or k.get('name') == 'polytopia-verda')), None)
    if find() is None:
        verda('ssh-key', 'add', '--name', 'polytopia-verda', '--public-key', public)
    return find()


def deploy(kind, instance_type, image, key_id, index):
    hostname = f'pilot-{kind}-{index}'
    # CLI 1.9.0 requires --kind but only uses it for a prefix check that knows "CPU." and not "CPU-TURIN.";
    # it is not sent to the API, so CPU-TURIN types pass as "gpu"
    kind_flag = 'cpu' if instance_type.upper().startswith('CPU.') else 'gpu'
    verda('vm', 'create', '--kind', kind_flag, '--instance-type', instance_type,
          '--location', LOCATION, '--os', image, '--os-volume-size', 100, '--hostname', hostname,
          '--description', 'polytopia verda pilot', '--contract', 'pay_as_go', '--ssh-key', key_id,
          '--wait=false')
    return dict(kind=kind, index=index, hostname=hostname, type=instance_type)


def refresh(state):
    """Fill id/ip/status for this pilot's VMs from `vm list` (matched by hostname)."""
    rows = {r.get('hostname'): r for r in records(verda('vm', 'list')) if 'hostname' in r}
    for x in state['instances']:
        r = rows.get(x['hostname'], {})
        x.update(id=r.get('id', x.get('id')), ip=r.get('ip'), status=r.get('status', 'missing'))
    save_state(state)


def cmd_up(a):
    hourly = plan_hourly(a.actors)
    if hourly * a.hours > a.cap:
        raise SystemExit(f'plan ${hourly:.2f}/h x {a.hours} h exceeds the ${a.cap} cap')
    state = load_state()
    if state['instances']:
        raise SystemExit(f'state already has instances: {state["instances"]} (run down first)')
    key_id = ssh_key_id()
    state.update(started=time.time(), hourly=hourly)
    for kind, itype, i in [('learner', LEARNER_TYPE, 0)] + [('actor', ACTOR_TYPE, i) for i in range(a.actors)]:
        state['instances'].append(deploy(kind, itype, a.learner_image if kind == 'learner' else a.actor_image,
                                         key_id, i))
        save_state(state)
    print(f'requested {len(state["instances"])} VMs at ${hourly:.2f}/h; waiting for running + IP')
    deadline = time.time() + 1200
    while time.time() < deadline:
        refresh(state)
        print(' '.join(f'{x["hostname"]}={x["status"]}' for x in state['instances']), flush=True)
        if any(x['status'] in ('error', 'no_capacity', 'installation_failed') for x in state['instances']):
            raise SystemExit('a deploy failed; run down')
        if all(x['status'] == 'running' and x['ip'] for x in state['instances']):
            return
        time.sleep(20)
    raise SystemExit('timed out waiting for running; run down')


def write_bundle(seconds):
    """main's src + engine-rs (the code the Modal runs used), the node scripts, the run config and the parent."""
    out = STATE_DIR / 'bundle'
    (out / 'ckpt').mkdir(parents=True, exist_ok=True)
    subprocess.run(f'git -C {ROOT} archive --format=tar -o {out}/code.tar main src engine-rs', shell=True, check=True)
    for name in ('final.npz', 'final.json', 'final-critic.npz', 'final-critic.json'):
        if not (out / 'ckpt' / name).exists():
            subprocess.run(['modal', 'volume', 'get', '--env', 'staging', 'polytopia-structured-rl',
                            f'{PARENT}/{name}', str(out / 'ckpt' / name)], check=True)
    run = next(r for r in json.loads(LEDGER.read_text())['runs'] if r['run_id'] == 'scratch-cont-2')
    config = dict(run['config'], critic_path='/workspace/pilot/ckpt/final-critic.npz', seed=606,
                  training_seconds=seconds)
    (out / 'config.json').write_text(json.dumps(config, indent=1))
    subprocess.run(f'tar -C {out} -cf {out}/pilot.tar ckpt config.json -C {ROOT}/experiments/verda '
                   'node_learner.py node_actor.py setup_node.sh', shell=True, check=True)
    return out


def setup_one(x, learner_allow, bundle):
    ip = x['ip']
    ssh(ip, 'mkdir -p /workspace/pilot')
    scp(ip, bundle / 'code.tar', '/workspace/code.tar')
    scp(ip, bundle / 'pilot.tar', '/workspace/pilot.tar')
    allow = ' '.join(learner_allow) if x['kind'] == 'learner' else ''
    out = ssh(ip, f'cd /workspace && tar -xf code.tar && tar -C pilot -xf pilot.tar && '
                  f'bash pilot/setup_node.sh {x["kind"]} {allow} 2>&1 | tail -5', check=False, capture=True)
    return f'{x["kind"]}{x["index"]} {ip}: {out.stdout.strip()[-600:]}'


def cmd_setup(a):
    state = load_state()
    bundle = write_bundle(a.seconds)
    actor_ips = [x['ip'] for x in state['instances'] if x['kind'] == 'actor']
    with ThreadPoolExecutor(len(state['instances'])) as pool:
        for line in pool.map(lambda x: setup_one(x, actor_ips, bundle), state['instances']):
            print(line, flush=True)


def learner(state):
    return next(x for x in state['instances'] if x['kind'] == 'learner')


def cmd_bench(a):
    ip = learner(load_state())['ip']
    for devices in ('0', '0,1'):
        out = ssh(ip, f'cd /workspace/pilot && CUDA_VISIBLE_DEVICES={devices} JAX_PLATFORMS=cuda {REMOTE_ENV} '
                      f'python node_learner.py bench {a.epochs} 2>&1 | tail -3', check=False, capture=True)
        print(f'GPUs {devices}:', out.stdout.strip()[-1500:], flush=True)


def cmd_run(a):
    state = load_state()
    lip = learner(state)['ip']
    ssh(lip, f'cd /workspace/pilot && rm -rf /workspace/run && JAX_PLATFORMS=cuda {REMOTE_ENV} '
             f'tmux new -d -s learner "python node_learner.py run > /workspace/learner.log 2>&1"')
    for _ in range(90):
        if ssh(lip, 'ss -ltn | grep -q ":7070 "', check=False).returncode == 0:
            break
        time.sleep(10)
    else:
        raise SystemExit('learner never listened on 7070; see /workspace/learner.log')
    for x in state['instances']:
        if x['kind'] == 'actor':
            ssh(x['ip'], f'cd /workspace/pilot && JAX_PLATFORMS=cpu {REMOTE_ENV} tmux new -d -s actor '
                         f'"python node_actor.py {lip} {x["index"]} > /workspace/actor.log 2>&1"')
    state['run_started'] = time.time()
    save_state(state)
    print('learner listening, actors started')


def cmd_status(a):
    state = load_state()
    if state.get('started'):
        hours = (time.time() - state['started']) / 3600
        print(f'up {hours:.2f} h, ~${hours * state["hourly"]:.2f} so far at ${state["hourly"]:.2f}/h')
    out = ssh(learner(state)['ip'], 'grep \'"stage": "update"\' /workspace/learner.log | tail -n 4; '
                                    'grep -iE "error|traceback" /workspace/learner.log | tail -n 3', check=False,
              capture=True).stdout
    keys = ('update', 'seconds', 'assemble_seconds', 'train_seconds', 'ready_wait_seconds',
            'received_samples_per_second', 'connections', 'inbox')
    for line in out.splitlines():
        try:
            row = json.loads(line)
            print({k: round(row[k], 3) if isinstance(row.get(k), float) else row.get(k) for k in keys})
        except json.JSONDecodeError:
            print(line)
    for x in state['instances']:
        if x['kind'] == 'actor':
            print(f'actor{x["index"]}:', ssh(x['ip'], 'tail -n 1 /workspace/actor.log', check=False,
                                             capture=True).stdout.strip()[-400:])


def cmd_down(a):
    """Delete every pilot-* VM (not only the ones in state) with its volumes, then confirm none is left."""
    state = load_state()
    verda('vm', 'delete', '--all', '--hostname', 'pilot-*', '--with-volumes', '--yes', '--wait-timeout', '10m',
          check=False)
    for _ in range(60):
        live = {r.get('hostname'): r.get('status') for r in records(verda('vm', 'list'))
                if str(r.get('hostname', '')).startswith('pilot-')}
        print('remaining', live, flush=True)
        if not live:
            break
        time.sleep(10)
    else:
        raise SystemExit('pilot VMs still listed after 10 min; delete them in the console')
    # CLI 1.9.0: --with-volumes still left the OS volumes detached (and billing) in the Oct 23 pilot
    for v in records(verda('volume', 'list', check=False)):
        if str(v.get('name', '')).startswith('pilot-') and v.get('id'):
            verda('volume', 'delete', v['id'], '--yes', check=False)
    print('volumes left:', json.dumps(verda('volume', 'list', check=False))[:500])
    if state.get('started'):
        hours = (time.time() - state['started']) / 3600
        state.setdefault('history', []).append(dict(hours=hours, est_usd=hours * state['hourly'], ended=time.time()))
    state['instances'] = []
    save_state(state)


def cmd_watchdog(a):
    time.sleep(a.minutes * 60)
    cmd_down(a)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['preflight', 'up', 'setup', 'bench', 'run', 'status', 'down', 'watchdog'])
    ap.add_argument('--actors', type=int, default=2)
    ap.add_argument('--learner-image', default='24.04.cuda12.9')   # driver for the pip CUDA 12 JAX wheels
    ap.add_argument('--actor-image', default='24.04.base')
    ap.add_argument('--hours', type=float, default=2.)
    ap.add_argument('--cap', type=float, default=15.)
    ap.add_argument('--seconds', type=int, default=1500)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--minutes', type=float, default=120)
    a = ap.parse_args()
    globals()[f'cmd_{a.command}'](a)
