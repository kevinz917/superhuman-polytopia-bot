"""Modal-hosted controller for Verda training runs: deadline, spend cap, learner health, checkpoint sync and spot
actor replacement, every 3 minutes, independent of the Mac. Decisions are control.decide; this file executes them.

    modal deploy --env staging experiments/verda/modal_controller.py                 # the schedule (tick)
    modal run --env staging experiments/verda/modal_selftest.py                      # control.py tests
    modal run --env staging experiments/verda/modal_controller.py::once --run R --dry-run   # one tick, print only
    modal run --env staging experiments/verda/modal_controller.py::teardown --run R --reason manual

A run is armed by `bigrun.py arm`, which writes /results/verda/<run>/controller.json; the tick acts on every armed
run and stops acting once a teardown marks it stopped. Learner files land in /results/distributed/<run>/ in the
layout Modal runs use, so elo_ladder / arena read them unchanged.

Secrets, created by the user (never written by this code): `verda-api` with VERDA_CLIENT_ID and VERDA_CLIENT_SECRET
(read by the Verda CLI from the environment), `verda-ssh` with VERDA_SSH_KEY, the private half of the controller key
registered on the Verda account (bigrun.py keys).
"""
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
VERDA_TGZ = 'https://github.com/verda-cloud/verda-cli/releases/download/v1.9.0/verda_1.9.0_linux_amd64.tar.gz'
VERDA_SHA256 = '532bb4d53f009bfc882738d82ddad2fc703afed51c0e879da8494263bb93f82e'   # the Homebrew tap's pin
image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('openssh-client', 'curl', 'ca-certificates')
         .run_commands(f'curl -fsSL -o /tmp/verda.tgz {VERDA_TGZ} && echo "{VERDA_SHA256}  /tmp/verda.tgz" | '
                       'sha256sum -c - && mkdir /tmp/v && tar -C /tmp/v -xzf /tmp/verda.tgz && '
                       'install "$(find /tmp/v -type f -name verda | head -1)" /usr/local/bin/verda && verda --version')
         .add_local_file(HERE / 'control.py', '/ctl/control.py'))
app = modal.App('verda-controller')
results = modal.Volume.from_name('polytopia-structured-rl', create_if_missing=False)
SECRETS = [modal.Secret.from_name('verda-api'), modal.Secret.from_name('verda-ssh')]
RUNS = Path('/results/verda')
KEY = Path('/tmp/verda_controller_key')
SYNC_PATTERNS = ("-name 'latest*' -o -name 'training.jsonl' -o -name 'protocol.json' -o "
                 "-name 'training-complete.json' -o -path './snapshots/*'")


def _ctl():
    import sys
    sys.path.insert(0, '/ctl')
    import control
    return control


# ---------------------------------------------------------------- Verda CLI, SSH

def verda(*args, check=True):
    out = subprocess.run(['verda', '--agent', *map(str, args)], text=True, capture_output=True, timeout=120)
    if check and out.returncode != 0:
        raise RuntimeError(f'verda {" ".join(map(str, args))} -> {out.returncode}: {(out.stderr or out.stdout)[-600:]}')
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


def run_vms(run):
    return {r['hostname']: r for r in records(verda('vm', 'list')) if str(r.get('hostname', '')).startswith(f'{run}-')}


def balance():
    """Account balance in USD, or None if the reply has no recognisable amount."""
    for r in records(verda('cost', 'balance', check=False)):
        for k in ('amount', 'balance', 'available', 'value'):
            if isinstance(r.get(k), (int, float)):
                return float(r[k])
    return None


def ssh_args(ip, timeout=15):
    if not KEY.exists():
        KEY.write_text(os.environ['VERDA_SSH_KEY'].strip() + '\n')
        KEY.chmod(0o600)
    return ['ssh', '-i', str(KEY), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o',
            'UserKnownHostsFile=/tmp/known_hosts', '-o', f'ConnectTimeout={timeout}', f'root@{ip}']


def ssh(ip, command, timeout=120):
    try:
        return subprocess.run(ssh_args(ip) + [command], text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(e.cmd, 255, '', 'timeout')


def firewall(learner_ip, verb, ips):
    """ufw allow/delete for the actors' port on the learner. ufw rewrites its rules file non-atomically, so concurrent
    calls (24 parallel setup_actor) lose each other's rules on the next reload; flock serializes them on the learner."""
    if not ips:
        return
    rules = '; '.join(f'ufw {verb} from {ip} to any port 7070 proto tcp >/dev/null' for ip in ips)
    r = ssh(learner_ip, f"flock /run/polytopia-ufw.lock sh -c '{rules}'")
    if r.returncode:
        raise RuntimeError(f'ufw {verb} {ips}: {r.stderr[-300:]}')


def scp(ip, local, remote):
    args = ssh_args(ip)
    subprocess.run(['scp', '-q', *args[1:-1], str(local), f'{args[-1]}:{remote}'], check=True, timeout=600)


# ---------------------------------------------------------------- state

def load(run):
    return json.loads((RUNS / run / 'controller.json').read_text())


def save(run, state):
    path = RUNS / run / 'controller.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(path)


def note(run, row):
    with (RUNS / run / 'controller-log.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')


def probe(ip):
    """Learner health over SSH: None if it does not answer. log_age is the age of learner.log once it holds an
    update row (training.jsonl is only appended at checkpoints, every checkpoint_every updates)."""
    out = ssh(ip, 'systemctl is-active polytopia-learner; test -f /workspace/run/training-complete.json && '
                  'echo COMPLETE; echo NOW $(date +%s); echo LOG $('
                  'grep -q \'"stage": "update"\' /workspace/learner.log && stat -c %Y /workspace/learner.log || '
                  'echo none)', timeout=60)
    if out.returncode == 255:
        return None
    lines = out.stdout.split()
    now = float(lines[lines.index('NOW') + 1]) if 'NOW' in lines else time.time()
    log = lines[lines.index('LOG') + 1] if 'LOG' in lines else 'none'
    return dict(active=bool(lines) and lines[0] == 'active', complete='COMPLETE' in lines,
                log_age=None if log == 'none' else now - float(log))


def sync(run, state, ip, final=False):
    """Copy new or changed learner files into /results/distributed/<run>/ (one tar per tick, so the latest.* files of
    one checkpoint travel together); at the end also publish latest* as final*, as Modal runs do."""
    listing = ssh(ip, f"cd /workspace/run && find . -maxdepth 2 -type f \\( {SYNC_PATTERNS} \\) -printf '%P %s %T@\\n'")
    if listing.returncode != 0:
        return dict(error=listing.stderr[-300:])
    seen = {}
    for line in listing.stdout.splitlines():
        path, size, mtime = line.rsplit(' ', 2)
        seen[path] = [int(size), float(mtime)]
    synced = state.setdefault('synced', {})
    changed = sorted(p for p, v in seen.items() if synced.get(p) != v)
    dest = Path('/results/distributed') / run
    (dest / 'snapshots').mkdir(parents=True, exist_ok=True)
    if changed:
        files = ' '.join(shlex.quote(p) for p in changed)
        src = subprocess.Popen(ssh_args(ip) + [f'cd /workspace/run && tar -cf - {files}'], stdout=subprocess.PIPE)
        untar = subprocess.run(['tar', '-xf', '-', '-C', str(dest)], stdin=src.stdout, timeout=1200)
        src.wait(timeout=60)
        if src.returncode != 0 or untar.returncode != 0:
            return dict(error=f'tar ssh={src.returncode} local={untar.returncode}')
        synced.update({p: seen[p] for p in changed})
    if final:
        for a, b in (('latest.npz', 'final.npz'), ('latest.json', 'final.json'),
                     ('latest-critic.npz', 'final-critic.npz'), ('latest-critic.json', 'final-critic.json')):
            if (dest / a).exists():
                (dest / b).write_bytes((dest / a).read_bytes())
    results.commit()
    return dict(files=len(changed), bytes=sum(seen[p][0] for p in changed))


# ---------------------------------------------------------------- actions

def create_actor(state, hostname):
    """Request an actor VM, trying state['actor_options'] ([type, spot, hourly(, location)], in order; default the
    plan's one type) until one has capacity: spot supply for large CPU types runs out (Oct 23: one CPU.180V spot in
    FIN-02). An option's location may differ from the learner's: actors reach it over its public IP."""
    options = state.get('actor_options') or [[state['actor_type'], state['spot'], state['actor_hourly']]]
    errors = []
    for t, spot, rate, *where in options:
        location = where[0] if where else state['location']
        kind = 'cpu' if t.upper().startswith('CPU.') else 'gpu'   # CLI 1.9.0 only knows the "CPU." prefix
        args = ['vm', 'create', '--kind', kind, '--instance-type', t, '--location', location,
                '--os', state['actor_image'], '--os-volume-size', 60, '--hostname', hostname,
                '--description', f'polytopia {state["run"]} actor', '--wait=false']
        for key_id in state['ssh_key_ids']:
            args += ['--ssh-key', key_id]
        if spot:
            args += ['--contract', 'spot', '--is-spot', '--os-volume-on-spot-discontinue', 'delete_permanently']
        else:
            args += ['--contract', 'pay_as_go']
        try:
            verda(*args)
        except RuntimeError as e:
            message = str(e)
            errors.append(f'{t} {"spot" if spot else "on-demand"} {location}: {message[message.find("->"):][:240]}')
            continue                                          # no capacity or refused: next option
        state['hourly'][hostname] = rate
        state.setdefault('actor_kinds', {})[hostname] = [t, spot, location]
        return
    raise RuntimeError(' | '.join(errors))


def delete_vm(state, vms, hostname):
    vm = vms.get(hostname, {})
    ip = vm.get('ip') or next((a.get('ip') for a in state.get('actors', {}).values() if a['hostname'] == hostname),
                              None)
    if ip:
        firewall(state['learner_ip'], 'delete allow', [ip])
    if vm.get('id'):
        verda('vm', 'delete', vm['id'], '--with-volumes', '--yes', '--wait=false', check=False)
    state['hourly'].pop(hostname, None)


def sweep_volumes(run):
    """CLI 1.9.0 leaves OS volumes detached (and billing) after vm delete --with-volumes (Oct 23 pilot)."""
    for v in records(verda('volume', 'list', check=False)):
        if (str(v.get('name', '')).startswith(f'{run}-') and v.get('id')
                and str(v.get('status', '')).lower() in ('detached', 'available', '')):
            verda('volume', 'delete', v['id'], '--yes', check=False)


def teardown_run(run, state, reason, learner=None):
    if learner is not None:
        state['final_sync'] = sync(run, state, state['learner_ip'], final=True)
    verda('vm', 'delete', '--all', '--hostname', f'{run}-*', '--with-volumes', '--yes', '--wait-timeout', '10m',
          check=False)
    for _ in range(30):
        if not run_vms(run):
            break
        time.sleep(10)
    sweep_volumes(run)
    state.update(status='stopped', stop_reason=reason, stopped_at=time.time(), vms_left=sorted(run_vms(run)))
    return state


def tick_run(run, dry_run=False):
    control = _ctl()
    state = load(run)
    overrides = RUNS / run / 'overrides.json'          # bigrun.py scale/cap: the Mac never writes controller.json
    if overrides.exists():                             # after arming, so the two writers cannot race
        state.update({k: v for k, v in json.loads(overrides.read_text()).items()
                      if k in ('desired_actors', 'cap_usd', 'deadline', 'actor_options')})
    if state.get('status') != 'armed':
        return dict(run=run, status=state.get('status'))
    now = time.time()
    for marker in (RUNS / run / 'actors').glob('*.json'):
        m = json.loads(marker.read_text())
        state = control.setup_done(state, m['slot'], m['hostname'], m['ok'], m['at'])
    vms = run_vms(run)
    state['balance'] = balance()
    learner = probe(state['learner_ip'])
    state, actions = control.decide(state, vms, learner, now)
    row = dict(at=now, spent_usd=round(state['spent_usd'], 2), learner=learner, actions=actions,
               actors={k: a['phase'] for k, a in state.get('actors', {}).items()}, vms=len(vms))
    if dry_run:
        return row
    for action in actions:
        try:
            if action[0] == 'teardown':
                state = teardown_run(run, state, action[1], learner)
            elif action[0] == 'sync':
                state['last_sync'] = now
                row['sync'] = sync(run, state, state['learner_ip'])
            elif action[0] == 'create':
                try:
                    create_actor(state, action[2])
                except Exception:
                    state['actors'][action[1]]['since'] = -1e18   # past the request timeout: retried next tick
                    raise
            elif action[0] == 'delete':
                delete_vm(state, vms, action[1])
            elif action[0] == 'setup':
                slot, hostname = action[1], action[2]
                setup_actor.spawn(run, slot, hostname, state['actors'][slot]['ip'], state['actors'][slot]['gen'])
        except Exception as e:                    # one failed action must not stop the deadline / cap checks
            row.setdefault('errors', []).append(f'{action}: {e}'[:500])
    if any(a[0] == 'delete' for a in actions):
        sweep_volumes(run)
    if state.get('status') == 'armed':          # re-allow every live actor: a lost rule only bites at the next reconnect
        live = sorted({a['ip'] for a in state.get('actors', {}).values()
                       if a.get('ip') and a['phase'] in ('provisioning', 'running')})
        try:
            firewall(state['learner_ip'], 'allow', live)
        except Exception as e:
            row.setdefault('errors', []).append(f'firewall reconcile: {e}'[:500])
    save(run, state)
    note(run, row)
    results.commit()
    return row


# ---------------------------------------------------------------- Modal functions

@app.function(image=image, secrets=SECRETS, volumes={'/results': results}, timeout=900, max_containers=1,
              schedule=modal.Period(minutes=3))
def tick():
    results.reload()
    if not RUNS.exists():
        return []
    out = []
    for path in sorted(RUNS.glob('*/controller.json')):
        try:
            out.append(tick_run(path.parent.name))
        except Exception as e:
            out.append(dict(run=path.parent.name, error=str(e)[:500]))
    print(json.dumps(out))
    return out


@app.function(image=image, secrets=SECRETS, volumes={'/results': results}, timeout=2400)
def setup_actor(run, slot, hostname, ip, gen):
    """Provision one actor VM: allow it on the learner's firewall, copy the run's bundle, install, start the service.
    Writes /results/verda/<run>/actors/<hostname>.json, which the next tick applies (control.setup_done)."""
    results.reload()
    state = load(run)
    ok, tail = False, ''
    try:
        for _ in range(40):                                   # sshd comes up a little after "running"
            if ssh(ip, 'true', timeout=30).returncode == 0:
                break
            time.sleep(15)
        firewall(state['learner_ip'], 'allow', [ip])
        bundle = RUNS / run / 'bundle'
        ssh(ip, 'mkdir -p /workspace/pilot')
        scp(ip, bundle / 'code.tar', '/workspace/code.tar')
        scp(ip, bundle / 'pilot.tar', '/workspace/pilot.tar')
        index = int(slot) + 100 * gen                         # distinct game ids for every slot and generation
        out = ssh(ip, f'cd /workspace && tar -xf code.tar && tar -C pilot -xf pilot.tar && bash pilot/setup_node.sh '
                      f'actor {state["learner_ip"]} {index} {state["games"]} {state["processes"]} 2>&1 | tail -3 && '
                      'systemctl enable --now polytopia-actor && sleep 20 && systemctl is-active polytopia-actor',
                  timeout=2100)
        ok, tail = out.returncode == 0, (out.stdout + out.stderr)[-800:]
    except Exception as e:
        tail = str(e)[-800:]
    (RUNS / run / 'actors').mkdir(parents=True, exist_ok=True)
    (RUNS / run / 'actors' / f'{hostname}.json').write_text(json.dumps(
        dict(slot=slot, hostname=hostname, ip=ip, ok=ok, at=time.time(), tail=tail)))
    results.commit()
    return ok


@app.function(image=image, secrets=SECRETS, volumes={'/results': results}, timeout=900)
def once_fn(run, dry_run):
    results.reload()
    return tick_run(run, dry_run=dry_run)


@app.function(image=image, secrets=SECRETS, volumes={'/results': results}, timeout=1500)
def teardown_fn(run, reason):
    results.reload()
    state = load(run)
    learner = probe(state['learner_ip']) if state.get('learner_ip') else None
    state = teardown_run(run, state, reason, learner)
    save(run, state)
    note(run, dict(at=time.time(), actions=[('teardown', reason)], manual=True))
    results.commit()
    return dict(status=state['status'], vms_left=state['vms_left'], final_sync=state.get('final_sync'))


@app.local_entrypoint()
def once(run: str, dry_run: bool = False):
    print(json.dumps(once_fn.remote(run, dry_run), indent=1))


@app.local_entrypoint()
def teardown(run: str, reason: str = 'manual'):
    print(json.dumps(teardown_fn.remote(run, reason), indent=1))
