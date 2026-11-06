"""Overnight guard for scratch-cont-2 (polls Modal every 5 min; never computes anything heavy locally).

Record of the script that ran the night of 2026-10-22/23 (docs/scratch-cont-2-2026-10-23.md); app ids, paths and the
launch time are hard-coded for that run. It ran from a scratch folder holding refs.json, rr-apps.txt and its state/log
files, with the main checkout's .venv python (numpy) under caffeinate. Lessons: schedule finals on Modal, not here.

- Budget brake: stops the training app when estimated spend + remaining evaluation would pass the $200 cap.
- Logs each hourly ladder round (win rates, per-map pairs, land/water, seat) and refits Bradley-Terry Elo.
- After training: launches the best hourly checkpoint vs the 4 references on the final 256-map block
  (the final model itself is played by the scratch-cont-2-final ladder on Modal), then writes the final report.
Events go to guard.log (one line each, prefixed EVENT for the session's waiter); the report to REPORT.
"""
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

S = Path(__file__).resolve().parent
LAUNCH_DIR = Path('/Users/K.Z./Desktop/projects/polytopia-bot/.claude/worktrees/scratch-cont-launch')
sys.path.insert(0, str(LAUNCH_DIR / 'scripts'))
from ladder_elo import fit  # noqa: E402

MODAL = str(Path.home() / '.local/share/uv/tools/modal/bin/modal')
ENV = dict(__import__('os').environ, MODAL_ENVIRONMENT='staging')
F = str(LAUNCH_DIR / 'experiments/modal_distributed_rl.py')
REPORT = Path('/Users/K.Z./Library/Application Support/Claude/side-session-notes/'
              'local_7fb9784c-e4c4-4b71-90da-a35a86497ed7/scratch-cont-2-report.md')
RUN = 'scratch-cont-2'
TRAIN_APP = 'ap-2pDHF9fitS6azUsO345Grb'
# The first hourly ladder hung after round 2 (02:52, no copy, no history write) and was cancelled at 03:30 with the
# scheduled final ladder; ap-7dsg runs the remaining 3 rounds into the same history, and the guard plays the finals.
LADDER_APP, FINAL_APP = 'ap-7dsgN89CTky8jpWVzkuN7H', 'ap-19zXWtgQNEx6XvdoQOUOhv'
OLD_APPS = ['ap-axeIurVpCKZ13doEk16V7b']
RR_APPS = (S / 'rr-apps.txt').read_text().split()
REFS = json.loads((S / 'refs.json').read_text())
LAUNCH = 1792736879.0          # 2026-10-22 23:27:59 PDT
CAP, RATE = 200.0, 22.5         # $ total, $/h for the training app (scratch-cont-1 billed ~$21/h)
EVAL_PLAN = dict(rr=3.0, ladder=10.0, final=4.5, best=4.5)
FINAL_SEEDS, FINAL_MAPS = 392000, 256
WATER = {'archipelago', 'waterworld', 'lakes'}
state = json.loads((S / 'guard-state.json').read_text()) if (S / 'guard-state.json').exists() else dict(
    rounds_seen=0, train_done_at=None, braked=False, best=None, best_apps=[], final_written=False, alerts=[])


def log(msg, event=False):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {'EVENT ' if event else ''}{msg}"
    with open(S / 'guard.log', 'a') as f:
        f.write(line + '\n')


def save():
    (S / 'guard-state.json').write_text(json.dumps(state, indent=1))


def sh(args, timeout=180):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=ENV).stdout
    except Exception as error:
        log(f'command failed {args[:4]}: {error!r}')
        return ''


def vol_get(remote, local):
    sh([MODAL, 'volume', 'get', '--env', 'staging', '--force', 'polytopia-structured-rl', remote, str(local)])
    return Path(local).exists()


def vol_exists(remote):
    return bool(sh([MODAL, 'volume', 'ls', '--env', 'staging', 'polytopia-structured-rl', remote]).strip())


def billed():
    out = sh([MODAL, 'billing', 'report', '--start', '2026-10-23', '-r', 'h', '--json'])
    cost = collections.defaultdict(float)
    try:
        for row in json.loads(out):
            if row.get('environment') == 'staging':
                cost[row['object_id']] += float(row['cost'])
    except Exception:
        pass
    return cost


def containers():
    """Containers per app, or None when the listing failed (never read a failed call as 'no containers')."""
    try:
        listing = json.loads(sh([MODAL, 'container', 'list', '--env', 'staging', '--json']))
        return collections.Counter(c.get('App ID') or c.get('app_id') for c in listing) if listing else None
    except Exception:
        return None


def spend(cost, now):
    end = state['train_done_at'] or now
    train = max(cost.get(TRAIN_APP, 0.), RATE * (end - LAUNCH) / 3600)
    evals = sum(cost.get(a, 0.) for a in [LADDER_APP, FINAL_APP, *OLD_APPS, *RR_APPS, *state['best_apps']])
    rounds_left = max(0, 6 - state['rounds_seen'])
    remaining = rounds_left * EVAL_PLAN['ladder'] / 6 + EVAL_PLAN['final'] + (0 if state['best_apps'] else EVAL_PLAN['best'])
    return train, evals, remaining


def history(tag):
    local = S / f'{tag}-history.json'
    if vol_get(f'distributed/ladder/{tag}-history.json', local):
        try:
            return [dict(r, tag=tag) for r in json.loads(local.read_text()) if 'error' not in r]
        except Exception:
            return []
    return []


def details(name):
    local = S / 'arena' / f'{name}.json'
    local.parent.mkdir(exist_ok=True)
    if not local.exists():
        vol_get(f'distributed/arena/{name}.json', local)
    try:
        return json.loads(local.read_text())['details']
    except Exception:
        return None


def breakdown(rows):
    if not rows:
        return '-'
    pairs = collections.defaultdict(dict)
    for r in rows:
        pairs[r['seed']][r['seat']] = r['reward'] > 0
    two = sum(v.get(1) and v.get(2) for v in pairs.values())
    zero = sum(not v.get(1) and not v.get(2) for v in pairs.values())
    rate = lambda xs: f"{sum(x['reward'] > 0 for x in xs) / max(len(xs), 1):.0%}"
    water = [r for r in rows if r.get('map') in WATER]
    land = [r for r in rows if r.get('map') not in WATER]
    return (f"pairs {two}/{len(pairs) - two - zero}/{zero} · land {rate(land)} · water {rate(water)} · "
            f"Blue {rate([r for r in rows if r['seat'] == 1])} · Red {rate([r for r in rows if r['seat'] == 2])} · "
            f"capped {sum(r['truncated'] for r in rows)}")


def rr_rows():
    rows = []
    names = list(REFS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = details(f'{RUN}-rr-{a}-vs-{b}')
            if d:
                w, l = sum(r['reward'] > 0 for r in d), sum(r['reward'] < 0 for r in d)
                rows.append(dict(run_id=a, stage='fixed', reference=b, wins=w, losses=l, draws_or_truncations=len(d) - w - l))
    return rows


def final_rows():
    rows = []
    for stage in state.get('finals', []):
        for ref in REFS:
            d = details(f'{RUN}-finalgame-{stage}-vs-{ref}')
            if d:
                w, l = sum(r['reward'] > 0 for r in d), sum(r['reward'] < 0 for r in d)
                rows.append(dict(run_id=RUN, stage=stage, reference=ref, wins=w, losses=l,
                                 draws_or_truncations=len(d) - w - l, round='final', _details=d))
    return rows


def write_report(hourly, final, best, cost_line, done):
    rows = rr_rows() + hourly + final + best
    lines = [f"# scratch-cont-2 — {'final' if done else 'live'} report ({time.strftime('%Y-%m-%d %H:%M %Z')})", '',
             f'Training app {TRAIN_APP} (7 h, 8x32 actors, proposed-v1). {cost_line}', '']
    try:
        table = fit([r for r in rows if r.get('wins') is not None], 'c3')
        lines += ['## Elo (Bradley-Terry, c3 = 0, all mixed-map games so far)', '', '| player | Elo | ± | W-L-D |', '|---|---|---|---|']
        lines += [f"| {t['player']} | {t['elo']:+.0f} | {t['se']:.0f} | {'-'.join(map(str, t['record']))} |" for t in table]
    except Exception as error:
        lines += [f'(Elo fit failed: {error!r})']
    lines += ['', '## Per checkpoint (200 games per opponent; final 512)', '',
              '| round | checkpoint | opponent | win % | per-map pairs 2-0/split/0-2 · land · water · seat · capped |', '|---|---|---|---|---|']
    for r in hourly + final + best:
        d = r.get('_details')
        if d is None:
            d = details(f"{r['tag']}-{RUN}-{r['stage']}-vs-{r['reference']}-{r['seed_base']}") if 'seed_base' in r else None
        games = r['wins'] + r['losses'] + r['draws_or_truncations']
        lines.append(f"| {r.get('round')} | {r['stage']} | {r['reference']} | {r['wins'] / max(games, 1):.0%} ({r['wins']}/{games}) | {breakdown(d)} |")
    REPORT.write_text('\n'.join(lines) + '\n')


def main():
    log('guard start', event=True)
    while True:
        now = time.time()
        cost = billed()
        cont = containers()
        listed = cont is not None
        cont = cont or collections.Counter()
        train, evals, remaining = spend(cost, now)
        cost_line = f'Spend est ${train + evals:.0f} (training ${train:.0f}, eval ${evals:.0f}), still planned ${remaining:.0f}, cap ${CAP:.0f}.'
        log(f"status listed={listed} containers train={cont.get(TRAIN_APP, 0)} ladder={cont.get(LADDER_APP, 0)} final={cont.get(FINAL_APP, 0)} | {cost_line}")
        if cont.get(TRAIN_APP, 0) > 12 and 'dup' not in state['alerts']:
            state['alerts'].append('dup')
            log(f"ALERT training app has {cont.get(TRAIN_APP)} containers (expected 10): possible orchestrator replay", event=True)
        # training finished?
        if state['train_done_at'] is None and (vol_exists(f'distributed/{RUN}/training-complete.json')
                                               or (listed and now - LAUNCH > 1800 and cont.get(TRAIN_APP, 0) == 0)):
            state['train_done_at'] = now
            log('training finished', event=True)
        # budget brake
        if state['train_done_at'] is None and not state['braked'] and train + evals + remaining >= CAP - 2:
            sh([MODAL, 'app', 'stop', '--env', 'staging', TRAIN_APP])
            state['braked'] = True
            state['train_done_at'] = now
            log(f'BRAKE stopped training at est ${train + evals:.0f} + planned ${remaining:.0f} (latest.npz kept)', event=True)
        hourly = history(RUN)
        if len(hourly) > 4 * state['rounds_seen'] and len(hourly) % 4 == 0:
            state['rounds_seen'] = len(hourly) // 4
            last = hourly[-4:]
            log(f"ROUND {state['rounds_seen']} {last[0]['stage']}: " + ', '.join(
                f"{r['reference']} {r['wins']}/{r['wins'] + r['losses'] + r['draws_or_truncations']}" for r in last), event=True)
            write_report(hourly, [], [], cost_line, False)
        # hourly ladder stall watch (the first ladder hung silently)
        if len(hourly) != state.get('hist_len'):
            state['hist_len'], state['hist_changed_at'] = len(hourly), now
        elif state['train_done_at'] is None and now - state.get('hist_changed_at', now) > 85 * 60 and 'stall' not in state['alerts']:
            state['alerts'].append('stall')
            log('ALERT hourly ladder history unchanged for 85 min', event=True)
        # final tournament: final model + best hourly checkpoint vs the 4 references on the final block (seeds 392000+)
        if state['train_done_at'] and not state['best_apps'] and now - state['train_done_at'] > 300:
            score = collections.defaultdict(list)
            for r in hourly:
                score[r['stage']].append(r['wins'] / max(r['wins'] + r['losses'] + r['draws_or_truncations'], 1))
            best = max(score, key=lambda s: sum(score[s]) / len(score[s])) if score else None
            state['best'] = best
            final_path = f'/results/distributed/{RUN}/final.npz'
            if not vol_exists(f'distributed/{RUN}/final.npz'):
                final_path = f'/results/distributed/{RUN}/latest.npz'
            plan = [('final', final_path)]
            if best and train + evals + EVAL_PLAN['final'] + EVAL_PLAN['best'] <= CAP:
                plan.append((best, f'/results/distributed/ladder/{RUN}-{best}.npz'))
            elif best:
                log(f'best checkpoint {best} final games skipped: budget', event=True)
            state['finals'] = [stage for stage, _ in plan]
            for stage, path in plan:
                for ref, ref_path in REFS.items():
                    out = S / f'final-{stage}-{ref}.log'
                    subprocess.Popen([MODAL, 'run', '--detach', '--env', 'staging', f'{F}::arena_match', '--name',
                                      f'{RUN}-finalgame-{stage}-vs-{ref}', '--candidate', path, '--opponent', ref_path,
                                      '--seed-base', str(FINAL_SEEDS), '--maps', str(FINAL_MAPS), '--map-mix', 'proposed-v1'],
                                     stdout=open(out, 'w'), stderr=subprocess.STDOUT, env=ENV)
                    time.sleep(3)
            time.sleep(90)
            state['best_apps'] = sorted({w for f in S.glob('final-*.log') for w in f.read_text().split() if w.startswith('ap-')}
                                        ) or ['launched']
            log(f"final tournament launched: {state['finals']} ({final_path}) vs 4 refs, 256 maps x 2 seats ({len(state['best_apps'])} apps)", event=True)
        save()
        final = final_rows()
        if state.get('finals') and len(final) == 4 * len(state['finals']) and not state['final_written']:
            write_report(hourly, final, [], cost_line, True)
            state['final_written'] = True
            save()
            log('DONE final report written', event=True)
            return
        if now - LAUNCH > 11 * 3600:
            write_report(hourly, final, [], cost_line, True)
            log('TIMEOUT guard stopping after 11 h; report written with what exists', event=True)
            return
        time.sleep(300)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import traceback
        log(f'CRASH {error!r} {traceback.format_exc()[-800:]}', event=True)
        raise
