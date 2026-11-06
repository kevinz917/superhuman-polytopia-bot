"""Depth curve of the leak-free turn search on c5-omni-s1-final vs raw c3 (2026-10-21).

    uv run --no-project --with numpy python scripts/analyze_search_depth.py runs/c5/search/c5-search-v1 runs/c5/search/c5-search-depth-v1

Arms: win-k1-vs-c3 (raw) and win-d1det-ownval-vs-c3 (depth 1) from tag c5-search-v1, omni-d2det-vs-c3 and
omni-d3det-vs-c3 from tag c5-search-depth-v1 (experiments/modal_test_time_search.py::sweep, spec
docs/search-depth/depth-spec.json). All on the same 100 proposed-v1 14x14 maps x 2 seats (seeds 388000+), K=16,
T=1/T=1, determinized. Elo differences use a paired-map bootstrap (4000 resamples, seed 0), as analyze_c5_search.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_test_time_search import by_map, elo, points  # noqa: E402

ORDER = ['win-k1-vs-c3', 'win-d1det-ownval-vs-c3', 'omni-d2det-vs-c3', 'omni-d3det-vs-c3']
LABEL = {'win-k1-vs-c3': 'raw (no search)', 'win-d1det-ownval-vs-c3': 'leak-free d=1',
         'omni-d2det-vs-c3': 'leak-free d=2', 'omni-d3det-vs-c3': 'leak-free d=3'}


def boot(fn, *maps, n=4000):
    keys = sorted(set.intersection(*(set(m) for m in maps)))
    arrays = [np.array([m[k] for k in keys]) for m in maps]
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(keys), len(keys))
        out.append(fn(*(a[i].mean() for a in arrays)))
    return fn(*(a.mean() for a in arrays)), tuple(np.percentile(out, [2.5, 97.5])), len(keys)


def gain(a, b):
    return boot(lambda x, y: elo(x) - elo(y), a, b)


def fmt(g):
    v, (lo, hi), _ = g
    return f'{v:+.0f} [{lo:+.0f}, {hi:+.0f}]'


def main(paths):
    arms = {}
    for path in paths:
        for p in sorted(Path(path).glob('*.json')):
            if p.stem != 'index':
                d = json.loads(p.read_text())
                if d.get('rows'):
                    arms[p.stem] = d
    arms = {n: arms[n] for n in ORDER if n in arms}
    S = {n: by_map(a['rows']) for n, a in arms.items()}
    keys = sorted(set.intersection(*(set(m) for m in S.values())))
    H = {k: .5 for k in keys}
    print('| arm | depth | games | W–L–cap | score [95% CI] | of decisive | cap rate | Elo vs raw c3 [95% CI] | '
          'Elo minus d=1 [95% CI] | s / searched turn | lane steps / turn | incomplete lanes | override | mean turns | $ | chunks |')
    print('|---|---:|---:|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|')
    d1 = S.get('win-d1det-ownval-vs-c3')
    for n, a in arms.items():
        s, c, rows = a['summary'], a['config'], a['rows']
        g = fmt(gain(S[n], H))
        dd = fmt(gain(S[n], d1)) if d1 is not None and n != 'win-d1det-ownval-vs-c3' and c['lanes'] > 1 else '—'
        dec = s['wins'] / max(s['wins'] + s['losses'], 1)
        plans = sum(r['plans'] for r in rows)
        inc = sum(r['incomplete_lanes'] for r in rows) / max(plans * c['lanes'], 1)
        print(f"| {LABEL.get(n, n)} | {c.get('depth', 0) if c['lanes'] > 1 else '—'} | {s['games']} | "
              f"{s['wins']}–{s['losses']}–{s['truncations']} | {s['score']:.3f} [{s['score_ci95'][0]:.2f}, {s['score_ci95'][1]:.2f}] | "
              f"{100 * dec:.0f}% | {100 * s['truncations'] / s['games']:.0f}% | **{g}** | {dd} | "
              f"{s.get('seconds_per_search_turn', 0):.2f} | {s.get('lane_steps_per_turn', 0):.0f} | {100 * inc:.1f}% | "
              f"{s.get('override_rate', 0):.2f} | {s['mean_turns']:.1f} | {a.get('cost_usd', 0):.2f} | {a.get('chunks')} |")
    print('\nPaired comparisons (Elo, same maps and seats, map bootstrap):')
    pairs = [('win-d1det-ownval-vs-c3', 'win-k1-vs-c3'), ('omni-d2det-vs-c3', 'win-k1-vs-c3'), ('omni-d3det-vs-c3', 'win-k1-vs-c3'),
             ('omni-d2det-vs-c3', 'win-d1det-ownval-vs-c3'), ('omni-d3det-vs-c3', 'win-d1det-ownval-vs-c3'),
             ('omni-d3det-vs-c3', 'omni-d2det-vs-c3')]
    for x, y in pairs:
        if x in S and y in S:
            g = gain(S[x], S[y])
            print(f'- {LABEL[x]} minus {LABEL[y]}: {fmt(g)} (maps {g[2]})')
    print('\nScore by map type (games):')
    kinds = sorted({r.get('map_type', '?') for a in arms.values() for r in a['rows']})
    print('| arm | ' + ' | '.join(kinds) + ' |')
    print('|---|' + '---:|' * len(kinds))
    for n, a in arms.items():
        per = {}
        for r in a['rows']:
            per.setdefault(r.get('map_type', '?'), []).append(points(r))
        print(f'| {LABEL.get(n, n)} | ' + ' | '.join(f'{np.mean(per[k]):.2f} ({len(per[k])})' if k in per else '—' for k in kinds) + ' |')
    print('\nCritic calibration of the greedy lane (first plan of each searched turn) and chosen-minus-greedy score:')
    for n, a in arms.items():
        if a['config']['lanes'] > 1:
            v = np.array([t[1] for r in a['rows'] for t in r['trace']])
            ch = np.array([t[2] for r in a['rows'] for t in r['trace']])
            z = np.array([r['reward'] for r in a['rows'] for t in r['trace']])
            if len(v) > 1 and v.std() and z.std():
                print(f'- {LABEL[n]}: mean greedy value {v.mean():+.3f}, chosen {ch.mean():+.3f} (gap {(ch - v).mean():+.3f}), '
                      f'outcome {z.mean():+.3f}, corr(greedy, outcome) {np.corrcoef(v, z)[0, 1]:.3f}, '
                      f'corr(chosen, outcome) {np.corrcoef(ch, z)[0, 1]:.3f}, mse {np.mean((v - z) ** 2):.3f} (n={len(v)})')
    print(f"\nmaps in every arm: {len(keys)}; fallbacks per game: " + ', '.join(
        f"{LABEL[n]} {np.mean([r.get('fallbacks', 0) for r in a['rows']]):.2f}" for n, a in arms.items() if a['config']['lanes'] > 1))


if __name__ == '__main__':
    main(sys.argv[1:])
