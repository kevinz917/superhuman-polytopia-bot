"""Summarize the search-breadth sweep: leak-free d=1 robust search with 64 vs 256 sampled opponent replies on top of
the 6-h omniscient-critic final (c5-omni-s1-final) vs raw c3, paired on the same maps and seats as the c5-search-v1
baselines (raw, and single-reply d=1).

    uv run --no-project --with numpy python scripts/analyze_search_breadth.py \
        runs/c5/search/c5-search-breadth-v1 docs/c5-omni-long/search

The second directory supplies the paired baselines win-k1-vs-c3 (raw) and win-d1det-ownval-vs-c3 (one reply).
Elo differences use a paired-map bootstrap (4000 resamples, seed 0), as in analyze_c5_search.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_test_time_search import by_map, elo, points  # noqa: E402

LABEL = {
    'win-k1-vs-c3': 'raw s1-final (no search)',
    'win-d1det-ownval-vs-c3': 's1-final + leak-free d=1, 1 reply',
    'omni-d1det-r64-vs-c3': 's1-final + leak-free d=1, 64 replies (robust)',
    'omni-d1det-r128-vs-c3': 's1-final + leak-free d=1, 128 replies (robust)',
    'omni-d1det-r256-vs-c3': 's1-final + leak-free d=1, 256 replies (robust)',
}
ORDER = list(LABEL)


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
    v, (lo, hi), n = g
    return f'{v:+.0f} [{lo:+.0f}, {hi:+.0f}] (n={n} maps)'


def main(paths):
    arms = {}
    for path in paths:
        for p in sorted(Path(path).glob('*.json')):
            if p.stem in LABEL and p.stem not in arms:
                arms[p.stem] = json.loads(p.read_text())
    arms = {n: arms[n] for n in ORDER if n in arms}
    S = {n: by_map(a['rows']) for n, a in arms.items()}
    total = 0.
    print('| arm | W–L–cap | score [95% CI] | % of decisive | Elo over raw c3 [95% CI] | s / searched turn | '
          'lane-steps / turn | override | mean turns | $ | chunks |')
    print('|---|---|---|---:|---|---:|---:|---:|---:|---:|---|')
    for n, a in arms.items():
        s = a['summary']
        g = fmt(gain(S[n], {k: .5 for k in S[n]}))
        dec = s['wins'] / max(s['wins'] + s['losses'], 1)
        cost = a.get('cost_usd', 0)
        total += cost if n.startswith('omni-') else 0
        print(f"| {LABEL[n]} | {s['wins']}–{s['losses']}–{s['truncations']} | "
              f"{s['score']:.3f} [{s['score_ci95'][0]:.2f}, {s['score_ci95'][1]:.2f}] | {100 * dec:.0f}% | **{g}** | "
              f"{s.get('seconds_per_search_turn', 0):.2f} | {s.get('lane_steps_per_turn', 0):.0f} | "
              f"{s.get('override_rate', 0):.2f} | {s['mean_turns']:.1f} | {cost:.2f} | {a.get('chunks')} |")
    print('\nPaired comparisons (Elo, same maps and seats, map bootstrap):')
    pairs = [('win-d1det-ownval-vs-c3', 'win-k1-vs-c3', '1 reply over raw'),
             ('omni-d1det-r64-vs-c3', 'win-k1-vs-c3', '64 replies over raw'),
             ('omni-d1det-r128-vs-c3', 'win-k1-vs-c3', '128 replies over raw'),
             ('omni-d1det-r256-vs-c3', 'win-k1-vs-c3', '256 replies over raw'),
             ('omni-d1det-r64-vs-c3', 'win-d1det-ownval-vs-c3', '64 replies over 1 reply'),
             ('omni-d1det-r128-vs-c3', 'win-d1det-ownval-vs-c3', '128 replies over 1 reply'),
             ('omni-d1det-r256-vs-c3', 'win-d1det-ownval-vs-c3', '256 replies over 1 reply'),
             ('omni-d1det-r128-vs-c3', 'omni-d1det-r64-vs-c3', '128 replies over 64 replies'),
             ('omni-d1det-r256-vs-c3', 'omni-d1det-r64-vs-c3', '256 replies over 64 replies')]
    for x, y, what in pairs:
        if x in S and y in S:
            print(f'- {what}: {fmt(gain(S[x], S[y]))}')
    print('\nScore by map type (games):')
    kinds = sorted({r.get('map_type', '?') for a in arms.values() for r in a['rows']})
    print('| arm | ' + ' | '.join(kinds) + ' |')
    print('|---|' + '---:|' * len(kinds))
    for n, a in arms.items():
        per = {}
        for r in a['rows']:
            per.setdefault(r.get('map_type', '?'), []).append(points(r))
        print(f'| {LABEL[n]} | ' + ' | '.join(f'{np.mean(per[k]):.2f} ({len(per[k])})' if k in per else '—' for k in kinds) + ' |')
    print('\nCritic calibration of the greedy lane (searched arms, first plan of each turn):')
    for n, a in arms.items():
        if a['config']['lanes'] > 1:
            v = np.array([t[1] for r in a['rows'] for t in r['trace']])
            z = np.array([r['reward'] for r in a['rows'] for t in r['trace']])
            if len(v) > 1 and v.std() and z.std():
                print(f'- {n}: mean greedy value {v.mean():+.3f}, mean outcome {z.mean():+.3f}, '
                      f'corr {np.corrcoef(v, z)[0, 1]:.3f}, mse {np.mean((v - z) ** 2):.3f} (n={len(v)})')
    for n, a in arms.items():
        if a.get('errors'):
            print(f'\n{n}: {len(a["errors"])} chunk error(s): {a["errors"][0][:300]}')
    print(f'\nbreadth arms $ {total:.2f}')


if __name__ == '__main__':
    main(sys.argv[1:])
