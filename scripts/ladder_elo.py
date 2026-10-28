"""Bradley-Terry Elo from an Elo-ladder history (registry `{tag}/history` or a JSON file).

    modal volume get --env staging polytopia-structured-rl distributed/ladder/TAG-history.json /tmp/h.json
    .venv/bin/python scripts/ladder_elo.py TAG --file /tmp/h.json [--anchor u3142]
(without --file it reads the registry Dict, which needs numpy and modal in one interpreter).

Draws and turn-cap truncations count as half a win for each side. Players are the fixed references by
name and ladder snapshots as `run@update-N`. Elo is anchored at 0 for --anchor; ± is one standard error
from the inverse Hessian of the likelihood (assumes independent games, so it is a little optimistic).
"""
import argparse
import json

import numpy as np


def player(row):
    return row['run_id'] if row.get('stage') == 'fixed' else f"{row['run_id']}@{row['stage']}"


def fit(rows, anchor):
    rows = [r for r in rows if 'error' not in r and r.get('wins') is not None]
    names = sorted({player(r) for r in rows} | {r['reference'] for r in rows})
    if anchor not in names:
        raise SystemExit(f'anchor {anchor} not among players {names}')
    index = {n: i for i, n in enumerate(names)}
    a = np.array([index[player(r)] for r in rows])
    b = np.array([index[r['reference']] for r in rows])
    score = np.array([r['wins'] + .5 * r['draws_or_truncations'] for r in rows], float)
    games = np.array([r['wins'] + r['losses'] + r['draws_or_truncations'] for r in rows], float)
    free = np.array([i != index[anchor] for i in range(len(names))])
    k = np.log(10) / 400
    elo = np.zeros(len(names))
    for _ in range(200):
        p = 1 / (1 + np.exp(-k * (elo[a] - elo[b])))
        g = np.zeros(len(names))
        np.add.at(g, a, k * (score - games * p))
        np.add.at(g, b, -k * (score - games * p))
        w = k * k * games * p * (1 - p)
        h = np.zeros((len(names), len(names)))
        np.add.at(h, (a, a), -w)
        np.add.at(h, (b, b), -w)
        np.add.at(h, (a, b), w)
        np.add.at(h, (b, a), w)
        h -= 1e-6 * np.eye(len(names))  # weak prior keeps players with all-win records finite
        step = np.zeros(len(names))
        step[free] = np.linalg.solve(h[np.ix_(free, free)], -g[free])
        elo += np.clip(step, -200, 200)
        if np.abs(step).max() < 1e-6:
            break
    se = np.zeros(len(names))
    se[free] = np.sqrt(np.diag(np.linalg.inv(-h[np.ix_(free, free)])))
    record = {n: [0, 0, 0] for n in names}
    for r in rows:
        record[player(r)] = [x + y for x, y in zip(record[player(r)], (r['wins'], r['losses'], r['draws_or_truncations']))]
        record[r['reference']] = [x + y for x, y in zip(record[r['reference']], (r['losses'], r['wins'], r['draws_or_truncations']))]
    return [dict(player=n, elo=round(float(elo[i]), 1), se=round(float(se[i]), 1), record=record[n])
            for i, n in sorted(enumerate(names), key=lambda x: -elo[x[0]])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag')
    parser.add_argument('--anchor', default='u3142')
    parser.add_argument('--file')
    args = parser.parse_args()
    if args.file:
        rows = json.load(open(args.file))
    else:
        import modal
        rows = modal.Dict.from_name('polytopia-distributed-rl-registry', environment_name='staging').get(
            f'{args.tag}/history', [])
    for r in rows:
        if 'error' in r:
            print('error', json.dumps(r))
    for row in fit(rows, args.anchor):
        print(f"{row['player']:42s} {row['elo']:+7.1f} ± {row['se']:5.1f}   W-L-D {row['record']}")
    print()
    for r in rows:
        if 'error' not in r and r.get('stage') != 'fixed':
            mix = r.get('candidate_unit_mix') or {}
            top = ', '.join(f'{u} {v:.0%}' for u, v in sorted(mix.items(), key=lambda x: -x[1])[:3])
            print(f"round {r['round']} {player(r):34s} vs {r['reference']:8s} {r['wins']}-{r['losses']}-"
                  f"{r['draws_or_truncations']}  types/game {r.get('candidate_unit_types_per_game')}  {top}")


if __name__ == '__main__':
    main()
