"""Session L launcher for V3 flagship stages and qualification tests (LAUNCH-PLAN.md sections 3, 5, 7).

    launcher.py validate MANIFEST
    launcher.py deploy                              # modal deploy of modal_flagship.py (records the source sha)
    launcher.py seed-maps                           # copy E's classic map table into the L volumes
    launcher.py launch MANIFEST [--dry-run]         # reserve, clear pause flag, spawn, persist the call id
    launcher.py pause RUN_ID [--reason TEXT]        # graceful checkpoint + stop (control Dict flag)
    launcher.py resume RUN_ID --decision FILE       # paused stage continues (new launch id); flagship needs a decision file
    launcher.py status RUN_ID

FLAGSHIP launches are refused unless /Users/Shared/PolytopiaV3/flagship/GO-stage<N>.json holds Jack's own words
(relayed by the infra session) and the sha256 of this exact manifest. Qualification launches spend only from L's
qualification ledger (cap $60). Every launch reserves its bound first; a reserve that would cross a cap is refused.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FLAGSHIP_DIR = Path(os.environ.get('V3L_FLAGSHIP_DIR', str(REPO / 'runs' / 'plato-control')))
STATE_DIR = FLAGSHIP_DIR / 'state'
QUAL_LEDGER = Path(os.environ.get('V3L_QUAL_LEDGER', str(FLAGSHIP_DIR / 'qualification-budget.json')))
FLAG_LEDGER = Path(os.environ.get('V3L_FLAG_LEDGER', str(FLAGSHIP_DIR / 'budget.json')))
MODAL = os.environ.get('V3L_MODAL', shutil.which('modal') or str(Path(sys.executable).with_name('modal')))
PY = sys.executable
APP = os.environ.get('V3L_APP', 'claude-v3l-flagship')
CONTROL_DICT = os.environ.get('V3L_CONTROL_DICT', 'claude-v3l-control')   # session M reuses the stack with its own
# stage_worker also runs the CNN control (mode 'control')
WORKERS = {'stage_worker': ('flagship', 8), 'stage_worker_16': ('flagship', 16), 'qual_worker_16': ('qualification', 16),
           'qual_worker_8': ('qualification', 8), 'qual_worker_1': ('qualification', 1)}
WORKER_TIMEOUT = {'stage_worker': 86400, 'stage_worker_16': 86400, 'qual_worker_16': 5400, 'qual_worker_8': 7200,
                  'qual_worker_1': 2700}
RATES = {1: 3.95 + 4 * 0.0473 + 32 * 0.008, 8: 8 * 3.95 + 32 * 0.0473 + 128 * 0.008}   # USD/h at the workers' sizes
RATES[16] = 2 * RATES[8]                                                               # 2 clustered 8xH100 nodes


def pt(t=None):
    return datetime.fromtimestamp(time.time() if t is None else t,
                                  ZoneInfo(os.environ.get('TZ', 'America/Los_Angeles'))).strftime('%Y-%m-%d %H:%M %Z')


def no_proxy_env():
    return {k: v for k, v in os.environ.items() if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}


# ---------------------------------------------------------------- manifest
def manifest_sha(m):
    body = {k: v for k, v in m.items() if k != 'manifest_sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _arg(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def git_head():
    sha = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--', 'src', 'experiments', 'engine-cu'],
                           capture_output=True, text=True).stdout.strip()
    return sha + ('-dirty' if dirty else '')


def validate(m, *, check_source=True):
    """Problems with a manifest (empty list = valid)."""
    bad = []
    need = ('schema', 'mode', 'run_id', 'stage', 'stage_cap_usd', 'usd_per_hour', 'worker', 'trainer_args')
    bad += [f'missing {k}' for k in need if k not in m]
    if bad:
        return bad
    if m['schema'] != 'v3l-manifest-1':
        bad.append('schema must be v3l-manifest-1')
    if m['mode'] not in ('flagship', 'control', 'qualification'):
        bad.append('mode must be flagship, control or qualification')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,60}', m['run_id']):
        bad.append('run_id: lowercase letters, digits, dashes')
    if m['worker'] not in WORKERS:
        bad.append(f'worker must be one of {list(WORKERS)}')
    else:
        mode, gpus = WORKERS[m['worker']]
        if mode != m['mode'] and not (mode == 'flagship' and m['mode'] == 'control'):
            bad.append(f'worker {m["worker"]} is for {mode} manifests')
        if abs(float(m['usd_per_hour']) - RATES[gpus]) > 0.01:
            bad.append(f'usd_per_hour {m["usd_per_hour"]} != list rate {RATES[gpus]:.4f} for {gpus} GPU worker')
        if int(m.get('timeout_s', 0)) != WORKER_TIMEOUT[m['worker']]:
            bad.append(f'timeout_s must equal the {m["worker"]} timeout {WORKER_TIMEOUT[m["worker"]]}')
        devices = int(_arg(m['trainer_args'], '--devices', 1))
        procs = int(m.get('processes', 1))
        if procs > 1 and ('--devices' in m['trainer_args'] or '--num-processes' in m['trainer_args']):
            bad.append('per-process manifests leave --devices/--num-processes to the supervisor')
        if devices * procs != gpus:
            bad.append(f'--devices {devices} x processes {procs} != worker GPUs {gpus}')
        if int(m.get('nodes', 1)) != max(1, gpus // 8):
            bad.append(f'nodes {m.get("nodes", 1)} != worker GPUs {gpus} / 8')
    a = m['trainer_args']
    if _arg(a, '--run-root') != '/flagship/runs':
        bad.append('--run-root must be /flagship/runs')
    if _arg(a, '--run-id') != m['run_id']:
        bad.append('--run-id must equal run_id')
    if '--parent' in a and _arg(a, '--parent'):
        bad.append('flagship/qualification runs start from scratch (--parent "")')
    arch = _arg(a, '--architecture', 'normalized-v2')
    width, blocks = int(_arg(a, '--width', 128)), int(_arg(a, '--blocks', 4))
    if 'transformer' in arch and (blocks < 6 or width < 128):
        bad.append(f'Jack: no network smaller than 6x128 ({arch} {blocks}x{width})')
    if m['mode'] == 'flagship':
        if not arch.startswith('transformer-v3'):
            bad.append('flagship must be a transformer-v3 family network (a CNN flagship needs a separate Jack decision)')
        if m.get('faults'):
            bad.append('flagship manifests cannot inject faults')
        if not m.get('leak_check', True):
            bad.append('flagship needs the leakage check')
        if int(m['stage']) not in (1, 2, 3, 4):
            bad.append('Jack 2026-11-08 ~08:50 PT: stages 1-4 only ($1,500 cumulative); stage 5 is NOT authorized')
        bounds = {1: 100, 2: 300, 3: 1000, 4: 1500}
        if float(m['stage_cap_usd']) > bounds.get(int(m['stage']), 0):
            bad.append(f'stage {m["stage"]} cap above the ${bounds.get(int(m["stage"]))} boundary')
        if int(m.get('max_resumes_per_stage', 3)) != 3:
            bad.append('Jack: max 3 auto-resumes per stage')
    elif m['mode'] == 'control':
        if arch != 'normalized-v2' or width != 128 or blocks != 4:
            bad.append('the CNN control is the hero normalized-v2 4x128 (Jack ~08:55 PT)')
        if m.get('faults'):
            bad.append('control manifests cannot inject faults')
        if int(m['stage']) != 1 or float(m['stage_cap_usd']) > 100:
            bad.append('CNN control: one stage, cap <= $100 (Jack ~09:10 PT resized it to ~$100)')
    else:
        if float(m['stage_cap_usd']) > 25:
            bad.append('a single qualification test is capped at $25')
    if '--seconds' not in a:
        bad.append('--seconds (the LR schedule length) must be explicit')
    if check_source and m.get('source_sha') and m['source_sha'] != git_head():
        bad.append(f'source_sha {m["source_sha"]} != worktree HEAD {git_head()}')
    if m.get('source_sha', '').endswith('-dirty'):
        bad.append('source tree is dirty: commit first')
    if m.get('manifest_sha256') and m['manifest_sha256'] != manifest_sha(m):
        bad.append('manifest_sha256 does not match the body')
    return bad


def go_file_ok(m):
    p = FLAGSHIP_DIR / (f'GO-stage{m["stage"]}.json' if m['mode'] == 'flagship' else 'GO-control.json')
    if not p.exists():
        return False, f'{p} missing: no launch without Jack\'s explicit go relayed by infra'
    go = json.loads(p.read_text())
    approved = [go.get('manifest_sha256')] + list(go.get('manifest_sha256s', []))
    if manifest_sha(m) not in approved:
        return False, f'{p} approves manifests {approved}, this is {manifest_sha(m)}'
    if not go.get('jack_words') or not go.get('relayed_by'):
        return False, f'{p} lacks jack_words / relayed_by'
    return True, str(p)


# ---------------------------------------------------------------- ledger
def ledger_committed(entries):
    return sum(e.get('actual_usd', e['reserved_usd']) for e in entries)


def ledger_reserve(path, name, usd, note):
    data = json.loads(path.read_text())
    have = ledger_committed(data['entries'])
    if have + usd > float(data['cap_usd']) + 1e-9:
        raise SystemExit(f'REFUSED by ledger {path}: {have:.2f} + {usd:.2f} > cap {data["cap_usd"]}')
    data['entries'].append(dict(name=name, reserved_usd=round(usd, 4), reserved_at=pt(), note=note))
    data['spent_usd'] = round(ledger_committed(data['entries']), 4)
    path.write_text(json.dumps(data, indent=1))
    return have + usd


def ledger_reconcile(path, name, usd, note):
    data = json.loads(path.read_text())
    rows = [e for e in data['entries'] if e['name'] == name]
    if not rows:
        raise SystemExit(f'no ledger entry {name}')
    rows[-1].update(actual_usd=round(usd, 4), reconciled_at=pt(), reconcile_note=note)
    data['spent_usd'] = round(ledger_committed(data['entries']), 4)
    path.write_text(json.dumps(data, indent=1))


# ---------------------------------------------------------------- modal
def modal_fn(name):
    os.environ.pop('HTTP_PROXY', None), os.environ.pop('HTTPS_PROXY', None)
    os.environ.pop('http_proxy', None), os.environ.pop('https_proxy', None)
    os.environ.setdefault('MODAL_ENVIRONMENT', 'staging')
    import modal
    return modal.Function.from_name(APP, name, environment_name='staging')


def control():
    os.environ.setdefault('MODAL_ENVIRONMENT', 'staging')
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        os.environ.pop(k, None)
    import modal
    return modal.Dict.from_name(CONTROL_DICT, environment_name='staging', create_if_missing=True)


def state_path(run_id):
    return STATE_DIR / f'{run_id}.json'


def load_state(run_id):
    p = state_path(run_id)
    return json.loads(p.read_text()) if p.exists() else None


def save_state(st):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = state_path(st['run_id']).with_suffix('.tmp')
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, state_path(st['run_id']))


def live_call(st):
    """True when the last persisted call of this run is still running (or unknown)."""
    if not st or not st.get('launches'):
        return False
    import modal
    if st['launches'][-1].get('cancelled'):
        return False       # a cancelled input can stay PENDING in Modal's call graph (Q1 09:23 PT); its pause flag is set
    cid = st['launches'][-1]['call_id']
    try:
        modal.FunctionCall.from_id(cid).get(timeout=0)
        return False
    except TimeoutError:
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- commands
def cmd_validate(args):
    m = json.loads(Path(args.manifest).read_text())
    bad = validate(m)
    print(json.dumps(dict(manifest=args.manifest, sha256=manifest_sha(m), problems=bad), indent=1))
    return 1 if bad else 0


def cmd_deploy(args):
    head = git_head()
    if STATE_DIR.exists():
        import modal  # noqa: F401
        for sp in STATE_DIR.glob('*.json'):
            st = json.loads(sp.read_text())
            if st.get('active') and live_call(st):
                raise SystemExit(f'refusing to deploy while {st["run_id"]} is live: a resume would start on the new '
                                 f'image and pause on a source mismatch')
    if head.endswith('-dirty'):
        raise SystemExit(f'refusing to deploy a dirty tree ({head}); commit first')
    p = subprocess.run([MODAL, 'deploy', '--env', 'staging', str(HERE / 'modal_flagship.py')], cwd=REPO,
                       env=no_proxy_env(), capture_output=True, text=True)
    print(p.stdout[-3000:], p.stderr[-3000:])
    FLAGSHIP_DIR.mkdir(parents=True, exist_ok=True)
    rec = dict(t=pt(), source_sha=head, rc=p.returncode)
    with open(FLAGSHIP_DIR / 'deploys.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    return p.returncode


def cmd_seed_maps(args):
    print(json.dumps(modal_fn('seed_map_table').remote(), indent=1))


def launch(m, *, dry_run=False, launch_note=''):
    bad = validate(m)
    if bad:
        raise SystemExit('manifest invalid: ' + '; '.join(bad))
    m = dict(m, manifest_sha256=manifest_sha(m))
    last = (FLAGSHIP_DIR / 'deploys.jsonl').read_text().splitlines()[-1] if (FLAGSHIP_DIR / 'deploys.jsonl').exists() else '{}'
    if json.loads(last).get('source_sha') != m.get('source_sha'):
        raise SystemExit(f'deployed source {json.loads(last).get("source_sha")} != manifest source {m.get("source_sha")}: deploy first')
    age = time.time() - (FLAGSHIP_DIR / 'deploys.jsonl').stat().st_mtime
    if age < 90:      # Q0e 08:48 PT: a spawn 30 s after deploy landed on a warm container of the previous image
        raise SystemExit(f'REFUSED: last deploy {age:.0f} s ago; wait 90 s so no warm old-image container takes the call')
    st = load_state(m['run_id']) or dict(run_id=m['run_id'], mode=m['mode'], launches=[], relayed=[])
    if m['mode'] in ('flagship', 'control'):
        ok, why = go_file_ok(m)
        if not ok:
            raise SystemExit(f'REFUSED: {why}')
        ledger = FLAG_LEDGER
    else:
        ledger = QUAL_LEDGER
    if not dry_run and live_call(st):
        raise SystemExit(f'REFUSED: run {m["run_id"]} has a live call {st["launches"][-1]["call_id"]}')
    stage_budget = float(m['stage_cap_usd']) - float(m.get('campaign_spent_before_usd', 0.))
    # the supervisor stops at budget - stop reserve ($3 flagship/control); qualification tests keep a +$2 margin.
    # A relaunch of the same run and stage reserves only what earlier launches of it have not already booked (the
    # supervisor's budget clock counts every attempt of the stage, so the stage can never spend more than its cap).
    booked = 0.
    if ledger.exists():
        entries = {e['name']: e for e in json.loads(ledger.read_text())['entries']}
        for old in st['launches']:
            e = entries.get(old['launch_id'])
            if e and int(old.get('stage', 0)) == int(m['stage']):
                booked += float(e.get('actual_usd', e['reserved_usd']))
    reserve = max(0., stage_budget + (2.0 if m['mode'] == 'qualification' else 0.0) - booked)
    launch_id = f'{m["run_id"]}-s{m["stage"]}-{time.strftime("%m%d%H%M%S")}-{uuid.uuid4().hex[:6]}'
    if dry_run:
        print(json.dumps(dict(dry_run=True, launch_id=launch_id, reserve_usd=reserve, ledger=str(ledger),
                              manifest_sha256=m['manifest_sha256']), indent=1))
        return None
    ledger_reserve(ledger, launch_id, reserve, f'{m["worker"]} {m["run_id"]} stage {m["stage"]} cap {stage_budget:.2f} '
                                               f'at {m["usd_per_hour"]:.4f}/h; {launch_note}')
    ctl = control()
    ctl.pop(f'{m["run_id"]}/pause', None)
    fn = modal_fn(m['worker'])
    call = fn.spawn(m, launch_id)
    rec = dict(launch_id=launch_id, call_id=call.object_id, t=time.time(), t_pt=pt(), worker=m['worker'],
               stage=m['stage'], manifest_sha256=m['manifest_sha256'], reserve_usd=reserve, ledger=str(ledger))
    st['launches'].append(rec)
    st['manifest'] = m
    st['active'] = True
    save_state(st)
    (FLAGSHIP_DIR / 'manifests').mkdir(parents=True, exist_ok=True)
    (FLAGSHIP_DIR / 'manifests' / f'{launch_id}.json').write_text(json.dumps(m, indent=1))
    print(json.dumps(rec, indent=1))
    return rec


def cmd_launch(args):
    m = json.loads(Path(args.manifest).read_text())
    m.setdefault('source_sha', git_head())
    launch(m, dry_run=args.dry_run, launch_note=args.note or '')


def cmd_pause(args):
    control().put(f'{args.run_id}/pause', dict(reason=args.reason or 'operator pause', by='launcher', t=time.time()))
    print(f'pause requested for {args.run_id}: the supervisor checkpoints and stops within ~1 poll + 1 update')


def cmd_cancel(args):
    """Stop a call that is queued or running: pause flag first (a queued input that still starts pauses at once), then
    cancel the call and its containers, then mark the run inactive. State on the volume is kept."""
    import modal
    st = load_state(args.run_id)
    control().put(f'{args.run_id}/pause', dict(reason=f'cancelled: {args.reason or "operator"}', by='launcher', t=time.time()))
    launch = st['launches'][-1]
    modal.FunctionCall.from_id(launch['call_id']).cancel(terminate_containers=True)
    launch.update(cancelled=True, cancelled_pt=pt(), cancel_reason=args.reason or 'operator')
    st['active'] = False
    save_state(st)
    print(f'cancelled {launch["call_id"]} ({args.run_id}); pause flag left set; reconcile the ledger entry {launch["launch_id"]}')


def cmd_resume(args):
    st = load_state(args.run_id)
    if not st:
        raise SystemExit('unknown run')
    m = dict(st['manifest'])
    m.pop('manifest_sha256', None)
    if m['mode'] in ('flagship', 'control'):
        if not args.decision or not Path(args.decision).exists():
            raise SystemExit('REFUSED: a flagship resume needs --decision FILE recording who decided and why')
    launch(m, launch_note=f'resume; decision {args.decision or "qualification"}')


def cmd_status(args):
    st = load_state(args.run_id)
    ctl = control()
    out = dict(state=st and {k: st[k] for k in ('run_id', 'mode', 'active') if k in st},
               launches=st and st['launches'][-3:], heartbeat=ctl.get(f'{args.run_id}/heartbeat'),
               status=ctl.get(f'{args.run_id}/status'), pause=ctl.get(f'{args.run_id}/pause'))
    if st and st.get('launches'):
        out['call_live'] = live_call(st)
    print(json.dumps(out, indent=1, default=str))


def main(argv=None):
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):   # Modal from the mini needs a direct route
        os.environ.pop(k, None)
    os.environ.setdefault('MODAL_ENVIRONMENT', 'staging')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('validate'); s.add_argument('manifest'); s.set_defaults(fn=cmd_validate)
    s = sub.add_parser('deploy'); s.set_defaults(fn=cmd_deploy)
    s = sub.add_parser('seed-maps'); s.set_defaults(fn=cmd_seed_maps)
    s = sub.add_parser('launch'); s.add_argument('manifest'); s.add_argument('--dry-run', action='store_true')
    s.add_argument('--note'); s.set_defaults(fn=cmd_launch)
    s = sub.add_parser('pause'); s.add_argument('run_id'); s.add_argument('--reason'); s.set_defaults(fn=cmd_pause)
    s = sub.add_parser('cancel'); s.add_argument('run_id'); s.add_argument('--reason'); s.set_defaults(fn=cmd_cancel)
    s = sub.add_parser('resume'); s.add_argument('run_id'); s.add_argument('--decision'); s.set_defaults(fn=cmd_resume)
    s = sub.add_parser('status'); s.add_argument('run_id'); s.set_defaults(fn=cmd_status)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main() or 0)
