"""Summarize the c5 search sweep: leak-free d=1 search on top of the 6-h omniscient-critic final (WIN).

    uv run --no-project --with numpy python scripts/analyze_c5_search.py runs/c5/search/c5-search-v1

Arms (experiments/modal_test_time_search.py::sweep, spec runs/c5/search-spec.json, 100 proposed-v1 maps x 2 seats,
seeds 388000+): win-k1-vs-c3 (raw WIN vs raw c3), win-d1det-ownval-vs-c3 (WIN + search, own value head),
win-d1det-c3val-vs-c3 (WIN + search, c3's value head scores the lanes), win-d1det-ownval-vs-win (mirror: WIN + search vs
raw WIN; baseline 0.5). Elo differences use a paired-map bootstrap (4000 resamples, seed 0), as in analyze_omni_search.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_test_time_search import by_map, elo, points  # noqa: E402

LABEL = {
    'win-k1-vs-c3': 'raw WIN (no search)',
    'win-d1det-ownval-vs-c3': 'WIN + leak-free d=1, own value head',
    'win-d1det-c3val-vs-c3': 'WIN + leak-free d=1, c3 value scorer',
    'win-d1det-ownval-vs-win': 'WIN + leak-free d=1, own value head (mirror)',
}


def boot(fn, *maps, n=4000):
    keys = sorted(set.intersection(*(set(m) for m in maps)))
    arrays = [np.array([m[k] for k in keys]) for m in maps]
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(keys), len(keys))
        out.append(fn(*(a[i].mean() for a in arrays)))
    return fn(*(a.mean() for a in arrays)), tuple(np.percentile(out, [2.5, 97.5]))


def gain(a, b):
    return boot(lambda x, y: elo(x) - elo(y), a, b)


def fmt(g):
    (v, (lo, hi)) = g
    return f'{v:+.0f} [{lo:+.0f}, {hi:+.0f}]'


def main(path):
    arms = {p.stem: json.loads(p.read_text()) for p in sorted(Path(path).glob('*.json')) if p.stem != 'index'}
    S = {n: by_map(a['rows']) for n, a in arms.items()}
    keys = sorted(set.intersection(*(set(m) for m in S.values())))
    H = {k: .5 for k in keys}
    total = 0.
    print('| arm | opponent | scorer | W–L–cap | score [95% CI] | % of decisive | Elo over the raw baseline [95% CI] | '
          's / searched turn | override | mean turns | $ | chunks |')
    print('|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|')
    for n, a in arms.items():
        s, c = a['summary'], a['config']
        g = fmt(gain(S[n], H))   # raw c3 vs raw c3 and raw WIN vs raw WIN are both exactly 0.5 on every map
        scorer = '—' if c['lanes'] == 1 else ('c3 (fog-only) head' if c.get('value') else 'own head')
        dec = s['wins'] / max(s['wins'] + s['losses'], 1)
        total += a.get('cost_usd', 0)
        print(f"| {LABEL.get(n, n)} | raw {n.split('-vs-')[1]} | {scorer} | {s['wins']}–{s['losses']}–{s['truncations']} | "
              f"{s['score']:.3f} [{s['score_ci95'][0]:.2f}, {s['score_ci95'][1]:.2f}] | {100 * dec:.0f}% | **{g}** | "
              f"{s.get('seconds_per_search_turn', 0):.2f} | {s.get('override_rate', 0):.2f} | {s['mean_turns']:.1f} | "
              f"{a.get('cost_usd', 0):.2f} | {a.get('chunks')} |")
    print('\nPaired comparisons (Elo, same maps and seats):')
    pairs = [('win-d1det-ownval-vs-c3', 'win-k1-vs-c3', 'search gain on WIN (own head), vs raw c3'),
             ('win-d1det-c3val-vs-c3', 'win-k1-vs-c3', 'search gain on WIN (c3 scorer), vs raw c3'),
             ('win-d1det-c3val-vs-c3', 'win-d1det-ownval-vs-c3', 'c3 scorer minus own head (WIN + search vs raw c3)')]
    for x, y, what in pairs:
        if x in S and y in S:
            print(f'- {what}: {fmt(gain(S[x], S[y]))}')
    if 'win-d1det-ownval-vs-win' in S:
        print(f'- search gain on WIN (own head), mirror: {fmt(gain(S["win-d1det-ownval-vs-win"], H))}')
    print('\nScore by map type (games):')
    kinds = sorted({r.get('map_type', '?') for a in arms.values() for r in a['rows']})
    print('| arm | ' + ' | '.join(kinds) + ' |')
    print('|---|' + '---:|' * len(kinds))
    for n, a in arms.items():
        per = {}
        for r in a['rows']:
            per.setdefault(r.get('map_type', '?'), []).append(points(r))
        print(f'| {LABEL.get(n, n)} | ' + ' | '.join(f'{np.mean(per[k]):.2f} ({len(per[k])})' if k in per else '—' for k in kinds) + ' |')
    print('\nCritic calibration of the greedy lane (searched arms, first plan of each turn):')
    for n, a in arms.items():
        if a['config']['lanes'] > 1:
            v = np.array([t[1] for r in a['rows'] for t in r['trace']])
            z = np.array([r['reward'] for r in a['rows'] for t in r['trace']])
            if len(v) > 1 and v.std() and z.std():
                print(f'- {n}: mean greedy value {v.mean():+.3f}, mean outcome {z.mean():+.3f}, '
                      f'corr {np.corrcoef(v, z)[0, 1]:.3f}, mse {np.mean((v - z) ** 2):.3f} (n={len(v)})')
    print(f'\nmaps in every arm: {len(keys)}; total $ {total:.2f}')


if __name__ == '__main__':
    main(sys.argv[1])
