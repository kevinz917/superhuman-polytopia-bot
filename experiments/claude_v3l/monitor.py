"""Session L independent monitor for V3 flagship stages and qualification runs (LAUNCH-PLAN.md section 3).

    monitor.py --once                 # one pass over every active run in /Users/Shared/PolytopiaV3/flagship/state
    monitor.py --loop [--every 120]   # nohup / launchd

It CANNOT launch or resume. Per pass and run it:
  * reads the control Dict (heartbeat, events, status) and polls the persisted FunctionCall;
  * relays every new INFO / PAUSE / ALERT event once to the infra session (brain.py msg claude, subject
    "[V3L <SEVERITY>] ..."), INFO batched per pass, PAUSE / ALERT one message each;
  * recomputes spend from the heartbeat and the persisted launch times; above the stage budget + $5 it requests a
    PAUSE, and above budget + $10 (or the heartbeat stale for 30 min while the call still runs) it cancels the call;
  * alerts when the heartbeat is stale for 20 min or Modal is unreachable for 15 min;
  * on a finished call: records the outcome, reconciles the ledger estimate and marks the run inactive.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launcher as L  # noqa: E402

# Optional integration with the original coordinator; public checkouts log locally by default.
BRAIN = shlex.split(os.environ.get('V3L_BRAIN', ''))
MON_DIR = L.FLAGSHIP_DIR / 'monitor'
STALE_ALERT_S, STALE_CANCEL_S, UNREACHABLE_ALERT_S = 20 * 60, 30 * 60, 15 * 60
OVER_PAUSE_USD, OVER_CANCEL_USD = 5.0, 10.0


def log(msg):
    MON_DIR.mkdir(parents=True, exist_ok=True)
    line = f'{L.pt()} {msg}'
    print(line, flush=True)
    with open(MON_DIR / 'monitor.log', 'a') as f:
        f.write(line + '\n')


STATUS_FILE = Path(os.environ.get('V3L_STATUS_FILE', str(L.FLAGSHIP_DIR / 'STATUS.md')))


def status_line(text):
    """Coordinator 09:01 PT: every flagship/control event as one timestamped PT line (the shared inbox gets archived)."""
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, 'a') as f:
            f.write(f'- {L.pt()} {text}\n')
    except OSError as exc:
        log(f'status file write failed: {exc!r}')


def notify(severity, subject, body, *, dry=False, mode=None):
    subj = f'[V3L {severity}] {subject}'
    if mode in ('flagship', 'control') and not dry:
        status_line(f'[{severity}] {subject} :: {body[:400]}'.replace('\n', ' | '))
    text = (f'For the INFRA session (relay to Jack when severity is PAUSE or ALERT). From the Session L monitor '
            f'(automatic, {L.pt()}):\n{body}')
    log(f'NOTIFY {subj} :: {body[:300]}')
    if dry or os.environ.get('V3L_NOTIFY_DRY') or not BRAIN:
        return True
    p = subprocess.run([*BRAIN, 'msg', 'claude', text, '--agent', 'claude', '--subject', subj], capture_output=True, text=True)
    if p.returncode:
        log(f'brain.py msg failed rc={p.returncode}: {p.stderr[-300:]}')
    return p.returncode == 0


def describe(ev):
    keep = {k: v for k, v in ev.items() if k not in ('id', 'severity', 'run_id', 'tail') and v is not None}
    keep['t'] = L.pt(ev.get('t'))
    return json.dumps(keep, default=str)[:800]


def poll_call(cid):
    """'running' | ('done', result) | ('failed', error)."""
    import modal
    try:
        return 'done', modal.FunctionCall.from_id(cid).get(timeout=0)
    except TimeoutError:
        return 'running', None
    except Exception as exc:
        return 'failed', repr(exc)[:1500]


def one_run(st, ctl, *, dry=False):
    run_id, m = st['run_id'], st['manifest']
    relayed = set(st.get('relayed', []))
    sev_rank = {'DEBUG': 0, 'INFO': 1, 'PAUSE': 2, 'ALERT': 3}
    events = ctl.get(f'{run_id}/events') or []
    hb = ctl.get(f'{run_id}/heartbeat') or {}
    info = []
    for ev in events:
        if ev['id'] in relayed:
            continue
        relayed.add(ev['id'])
        sev = ev.get('severity', 'INFO')
        if sev_rank.get(sev, 1) >= 2:
            notify(sev, f'{run_id}: {ev["kind"]} {ev.get("reason", "")}'.strip(), describe(ev), dry=dry, mode=m.get('mode'))
        elif sev == 'INFO':
            info.append(ev)
    if info:
        notify('INFO', f'{run_id}: ' + ', '.join(sorted({e['kind'] for e in info})),
               '\n'.join(describe(e) for e in info[-12:]), dry=dry, mode=m.get('mode'))
    st['relayed'] = sorted(relayed)[-5000:]
    launch = st['launches'][-1]
    state, result = poll_call(launch['call_id'])
    now = time.time()
    budget = float(m['stage_cap_usd']) - float(m.get('campaign_spent_before_usd', 0.))
    spent = float(hb.get('spent_usd') or 0.)
    # independent spend estimate: wall-clock since this launch at the list rate (upper bound while the call runs)
    est = (now - launch['t']) * float(m['usd_per_hour']) / 3600. if state == 'running' else spent
    st['last'] = dict(t=now, call=state, spent_hb=spent, spent_wall_upper=round(est, 3), hb_t=hb.get('t'),
                      update=hb.get('update'), sps=hb.get('samples_per_second'))
    if state == 'running':
        # 12:27 PT fix: a call with no heartbeat yet is QUEUED for a GPU node (8xH100 waits of 10-30 min seen), not
        # hung: never cancel it; alert once after 60 min. Staleness counts only after a first heartbeat of this launch.
        seen_hb = bool(hb.get('t')) and float(hb['t']) >= launch['t'] and hb.get('launch_id', launch['launch_id']) == launch['launch_id']
        if not seen_hb:
            if now - launch['t'] > 3600 and not st.get('queue_alert_sent'):
                st['queue_alert_sent'] = True
                notify('ALERT', f'{run_id}: still queued for GPUs after {(now - launch["t"]) / 60:.0f} min',
                       'no container yet; not cancelled', dry=dry, mode=m.get('mode'))
            return st
        hb_age = now - float(hb['t'])
        if spent > budget + OVER_CANCEL_USD or hb_age > STALE_CANCEL_S:
            why = f'spent {spent:.2f} > budget {budget:.2f} + {OVER_CANCEL_USD}' if spent > budget + OVER_CANCEL_USD \
                else f'heartbeat stale {hb_age / 60:.0f} min'
            notify('ALERT', f'{run_id}: monitor CANCELLED the call ({why})',
                   f'call {launch["call_id"]}; state kept on the volume; no relaunch by the monitor', dry=dry, mode=m.get('mode'))
            if not dry:
                import modal
                modal.FunctionCall.from_id(launch['call_id']).cancel(terminate_containers=True)
        elif spent > budget + OVER_PAUSE_USD:
            if not st.get('over_pause_sent'):
                st['over_pause_sent'] = True
                ctl.put(f'{run_id}/pause', dict(reason='monitor: budget overrun', by='monitor', t=now))
                notify('ALERT', f'{run_id}: budget overrun, PAUSE requested', f'spent {spent:.2f} budget {budget:.2f}',
                       dry=dry, mode=m.get('mode'))
        elif hb_age > STALE_ALERT_S:
            if st.get('stale_alert_for') != hb.get('t'):
                st['stale_alert_for'] = hb.get('t')
                notify('ALERT', f'{run_id}: heartbeat stale {hb_age / 60:.0f} min',
                       f'last update {hb.get("update")}; call still running; will cancel at 30 min', dry=dry, mode=m.get('mode'))
    else:
        st['active'] = False
        launch['outcome'] = result if state == 'done' else dict(failed=result)
        launch['ended_pt'] = L.pt(now)
        # no heartbeat (paused before the first one): the launch-to-now wall-clock at the list rate, an upper bound
        actual = spent or (now - launch['t']) * float(m['usd_per_hour']) / 3600.
        # the heartbeat's spend is the STAGE's cumulative (every launch of it); book only this launch's share
        earlier = [l for l in st['launches'][:-1] if int(l.get('stage', 0)) == int(m['stage'])]
        if spent and earlier:
            led = {e['name']: e for e in json.loads(Path(launch['ledger']).read_text())['entries']}
            actual = max(0., spent - sum(float(led[l['launch_id']].get('actual_usd', 0.)) for l in earlier
                                         if l['launch_id'] in led))
        try:
            L.ledger_reconcile(Path(launch['ledger']), launch['launch_id'], actual,
                               f'monitor estimate: supervisor clock (list rate x container wall + allowance); '
                               f'final Modal billing pending')
        except SystemExit as exc:
            log(f'reconcile: {exc}')
        sev = 'INFO' if state == 'done' and (result or {}).get('state') == 'complete' else 'PAUSE'
        if state == 'failed':
            sev = 'ALERT'
        notify(sev, f'{run_id}: call {state}' + (f' ({(result or {}).get("state")}: {(result or {}).get("reason")})'
                                                 if state == 'done' else ''),
               f'outcome {json.dumps(launch["outcome"], default=str)[:1200]}\nestimated cost {actual:.2f} USD '
               f'(ledger {launch["ledger"]}); final Modal billing pending', dry=dry, mode=m.get('mode'))
    return st


def one_pass(*, dry=False):
    MON_DIR.mkdir(parents=True, exist_ok=True)
    reach = MON_DIR / 'reach.json'
    r = json.loads(reach.read_text()) if reach.exists() else dict(ok_t=time.time(), alerted=False)
    runs = [json.loads(p.read_text()) for p in sorted(L.STATE_DIR.glob('*.json'))] if L.STATE_DIR.exists() else []
    active = [s for s in runs if s.get('active')]
    if not active:
        return 0
    try:
        ctl = L.control()
        ctl.get('ping')
        r.update(ok_t=time.time(), alerted=False)
    except Exception as exc:
        if time.time() - r['ok_t'] > UNREACHABLE_ALERT_S and not r['alerted']:
            notify('ALERT', 'monitor cannot reach Modal', f'{exc!r}; active runs: {[s["run_id"] for s in active]}', dry=dry, mode='flagship')
            r['alerted'] = True
        reach.write_text(json.dumps(r))
        return len(active)
    reach.write_text(json.dumps(r))
    for st in active:
        try:
            st = one_run(st, ctl, dry=dry)
            L.save_state(st)
        except Exception as exc:
            log(f'{st["run_id"]}: monitor error {exc!r}')
    return len(active)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--every', type=float, default=120.)
    ap.add_argument('--dry', action='store_true', help='log notifications instead of sending them')
    ap.add_argument('--exit-when-idle', action='store_true')
    a = ap.parse_args()
    if a.once or not a.loop:
        one_pass(dry=a.dry)
        return
    while True:
        n = one_pass(dry=a.dry)
        if n == 0 and a.exit_when_idle:
            log('no active runs; monitor exits')
            return
        time.sleep(a.every)


if __name__ == '__main__':
    main()
