"""Session L stage checks for the V3 flagship (LAUNCH-PLAN.md section 6, Jack's GATE CHANGE ~08:45 PT).

    stage_check.py prereg STAGE RUN_ID CANDIDATE PREVIOUS [--control CONTROL_RUN] [--out DIR]   # freeze the panel
    stage_check.py run PREREG_JSON                                                              # arena on Modal
    stage_check.py decide PREREG_JSON RESULTS_JSON                                              # apply the rules

Panel per stage (frozen before any result is seen; sha256 recorded):
  (i)  improvement: CANDIDATE vs PREVIOUS (stage 1: the run's own u250 snapshot; stage 2: the stage-1 paused ckpt);
  (ii) fixed references: vs hero1 u11650 (benchmark/references/hero-1v1.npz) and vs the champion (friend-champion);
  (+)  information only: flagship vs CNN control at each matched cumulative-dollar mark both runs have.
1,024 paired maps (2,048 games) per comparison, both seats, T=1, full horizon (max_ticks 10000), fresh map seeds per
comparison outside the training table. Intervals: 99% per comparison from the arena's map-clustered SE.
Rules: continue only if (i) lower99 > .5; (ii) is reported (progress vs fixed references), not a stop rule except a
supported loss to the previous stage. A plateau ((i) not supported) pauses for review. Stage 3 is not authorized.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launcher as L  # noqa: E402

HERO1 = '/workspace/benchmark/references/hero-1v1.npz'
CHAMPION = '/r/checkpoints/friend-champion/final.npz'
Z95, Z99 = 1.959964, 2.575829


def ff(run_id, rel):
    return f'/ff/runs/{run_id}/{rel}'


def prereg(a):
    stage = int(a.stage)
    if stage not in (1, 2):
        raise SystemExit('stage 3 is not authorized (Jack ~09:10 PT)')
    base = 500000 + 10000 * stage
    matches = [dict(name=f'{a.run_id}-s{stage}-improve', a=a.candidate, b=a.previous, role='improvement'),
               dict(name=f'{a.run_id}-s{stage}-hero1', a=a.candidate, b=HERO1, role='fixed_reference'),
               dict(name=f'{a.run_id}-s{stage}-champion', a=a.candidate, b=CHAMPION, role='fixed_reference')]
    for usd in (a.marks or []):
        matches.append(dict(name=f'{a.run_id}-s{stage}-vs-cnn-usd{usd}', a=ff(a.run_id, f'snapshots/usd-{usd}.npz'),
                            b=ff(a.control, f'snapshots/usd-{usd}.npz'), role='cnn_control_information'))
    for i, m in enumerate(matches):
        m.update(maps=1024, seed_base=base + 1024 * i, max_ticks=10000, timeout=2400)
    body = dict(schema='v3l-stage-check-1', stage=stage, run_id=a.run_id, frozen_pt=L.pt(), matches=matches,
                rules=dict(interval='99% map-clustered', improvement='lower99 > 0.5 to continue',
                           fixed_reference='reported; progress tracked', cnn_control='information only'))
    body['sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    out = Path(a.out or L.FLAGSHIP_DIR / 'stage-checks') / f'{a.run_id}-s{stage}-prereg.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f'{out} exists: a frozen panel is never rewritten')
    out.write_text(json.dumps(body, indent=1))
    print(out, body['sha256'])


def run(a):
    body = json.loads(Path(a.prereg).read_text())
    fn = L.modal_fn('arena_worker')
    matches = [{k: v for k, v in m.items() if k != 'role'} for m in body['matches']]
    t = time.time()
    res = fn.remote(matches)
    res['client_wall_seconds'] = round(time.time() - t, 1)
    res['prereg_sha256'] = body['sha256']
    out = Path(a.prereg).with_name(Path(a.prereg).name.replace('-prereg', '-results'))
    out.write_text(json.dumps(res, indent=1))
    print(out)


def interval99(summary):
    lo, hi = summary['ci95']
    se = (hi - lo) / (2 * Z95)
    return summary['score'] - Z99 * se, summary['score'] + Z99 * se


def decide(a):
    body = json.loads(Path(a.prereg).read_text())
    res = json.loads(Path(a.results).read_text())
    by = {r['name']: r for r in res['rows']}
    lines, ok = [], True
    for m in body['matches']:
        r = by.get(m['name'])
        if not r or 'score_at_cap' not in r:
            lines.append(f"{m['name']}: MISSING ({(r or {}).get('error', 'no row')[:200]})")
            if m['role'] == 'improvement':
                ok = False
            continue
        s = r['terminal_only'] if r.get('unfinished', 0) == 0 else r['score_at_cap']
        lo, hi = interval99(r['score_at_cap'])
        tlo, thi = interval99(r['terminal_only'])
        lines.append(f"{m['name']} [{m['role']}]: cap-score {r['score_at_cap']['score']:.3f} 99% [{lo:.3f}, {hi:.3f}]; "
                     f"terminal-only {r['terminal_only']['score']:.3f} [{tlo:.3f}, {thi:.3f}]; unfinished {r.get('unfinished')}")
        if m['role'] == 'improvement':
            # both scorings must support improvement: a cap-score lead must not hide unresolved games
            if not (lo > .5 and tlo > .5):
                ok = False
    verdict = 'CONTINUE (supported improvement)' if ok else 'PAUSE FOR REVIEW (improvement not supported: plateau or missing)'
    print(json.dumps(dict(stage=body['stage'], prereg_sha256=body['sha256'], verdict=verdict, lines=lines), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('prereg')
    s.add_argument('stage'); s.add_argument('run_id'); s.add_argument('candidate'); s.add_argument('previous')
    s.add_argument('--control'); s.add_argument('--marks', nargs='*', type=int); s.add_argument('--out')
    s.set_defaults(fn=prereg)
    s = sub.add_parser('run'); s.add_argument('prereg'); s.set_defaults(fn=run)
    s = sub.add_parser('decide'); s.add_argument('prereg'); s.add_argument('results'); s.set_defaults(fn=decide)
    a = ap.parse_args()
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        os.environ.pop(k, None)
    a.fn(a)


if __name__ == '__main__':
    main()
