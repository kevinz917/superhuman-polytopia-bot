"""Summarize the omni-search-eval sweep (search on top of the omniscient-critic checkpoint).

    python scripts/analyze_omni_search.py runs/omni-search-eval/<tag>

Arms (experiments/modal_test_time_search.py::sweep, spec runs/omni-search-eval/spec.json, all on the same
100 proposed-v1 maps x 2 seats): the opponent is always a raw greedy bot, so every arm's gain is measured
against the raw version of the *candidate's lineage* playing the same opponent on the same maps:
  vs raw C3:   the raw-C3-vs-raw-C3 baseline is exactly 0.5 on every map (same deterministic policy both seats);
  vs raw OMNI: the OMNI mirror baseline is 0.5; for C3 + search vs raw OMNI the baseline raw C3 vs raw OMNI is the
               omni-k1-vs-c3 arm with the sign flipped (the same deterministic games, seats swapped).
Elo differences use a paired-map bootstrap (4000 resamples, seed 0).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_test_time_search import by_map, elo, points  # noqa: E402

LABEL = {
    'omni-k1-vs-c3': 'raw OMNI (no search)',
    'c3-d1det-vs-c3': 'C3 + leak-free d=1',
    'omni-d1det-c3val-vs-c3': 'OMNI + leak-free d=1, C3 value scorer',
    'omni-d1det-ownval-vs-c3': 'OMNI + leak-free d=1, own (stale) value head',
    'omni-d1det-c3val-vs-omni': 'OMNI + leak-free d=1, C3 value scorer',
    'c3-d1det-vs-omni': 'C3 + leak-free d=1',
}
BLOCKS = [('vs raw C3', ['omni-k1-vs-c3', 'c3-d1det-vs-c3', 'omni-d1det-c3val-vs-c3', 'omni-d1det-ownval-vs-c3']),
          ('vs raw OMNI', ['omni-d1det-c3val-vs-omni', 'c3-d1det-vs-omni'])]


def scores(arm, flip=False):
    m = by_map(arm['rows'])
    return {k: 1. - v for k, v in m.items()} if flip else m


def half(keys):
    return {k: .5 for k in keys}


def boot(fn, *maps, n=4000):
    """fn(*mean scores of each map dict over a resample of maps) -> statistic; returns (point, (lo, hi))."""
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
    keys = sorted(set.intersection(*(set(by_map(a['rows'])) for a in arms.values())))
    total = 0.
    raw_c3_vs_omni = scores(arms['omni-k1-vs-c3'], flip=True) if 'omni-k1-vs-c3' in arms else None
    for title, names in BLOCKS:
        names = [n for n in names if n in arms]
        if not names:
            continue
        print(f'\n### {title}\n')
        base_label = 'raw C3' if title == 'vs raw C3' else 'raw lineage (OMNI mirror / raw C3 vs raw OMNI)'
        print(f'| arm | scorer | W–L–cap | score [95% CI] | % of decisive | Elo over {base_label} [95% CI] | '
              f's / searched turn | plans / turn | override | mean turns | $ | chunks |')
        print('|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|')
        for n in names:
            a, s, c = arms[n], arms[n]['summary'], arms[n]['config']
            base = raw_c3_vs_omni if n == 'c3-d1det-vs-omni' else half(keys)
            g = fmt(gain(scores(a), base))
            scorer = '—' if c['lanes'] == 1 else ('C3 (fog-only) head' if c.get('value') else 'own head')
            dec = s['wins'] / max(s['wins'] + s['losses'], 1)
            total += a.get('cost_usd', 0)
            print(f"| {LABEL.get(n, n)} | {scorer} | {s['wins']}–{s['losses']}–{s['truncations']} | {s['score']:.3f} "
                  f"[{s['score_ci95'][0]:.2f}, {s['score_ci95'][1]:.2f}] | {100 * dec:.0f}% | **{g}** | "
                  f"{s['seconds_per_search_turn']:.2f} | {s['replans_per_turn']:.2f} | {s['override_rate']:.2f} | "
                  f"{s['mean_turns']:.1f} | {a.get('cost_usd', 0):.2f} | {a.get('chunks')} |")

    S = {n: scores(a) for n, a in arms.items()}
    H = half(keys)
    print('\n### Paired comparisons (Elo, same maps and seats)\n')
    pairs = [
        ('omni-d1det-c3val-vs-c3', 'omni-k1-vs-c3', 'search gain on OMNI (C3 scorer), vs raw C3'),
        ('omni-d1det-ownval-vs-c3', 'omni-k1-vs-c3', 'search gain on OMNI (own stale head), vs raw C3'),
        ('omni-d1det-c3val-vs-c3', 'omni-d1det-ownval-vs-c3', 'C3 scorer minus own head (OMNI + search vs raw C3)'),
        ('omni-d1det-c3val-vs-c3', 'c3-d1det-vs-c3', 'OMNI + search minus C3 + search, both vs raw C3'),
        ('omni-d1det-ownval-vs-c3', 'c3-d1det-vs-c3', 'OMNI + search (own head) minus C3 + search, both vs raw C3'),
        ('omni-d1det-c3val-vs-omni', 'c3-d1det-vs-c3', 'search gain on OMNI (mirror) minus search gain on C3 (mirror)'),
    ]
    for x, y, what in pairs:
        if x in S and y in S:
            print(f'- {what}: {fmt(gain(S[x], S[y]))}')
    if {'omni-d1det-c3val-vs-c3', 'omni-k1-vs-c3', 'c3-d1det-vs-c3'} <= set(S):
        d = boot(lambda a, b, c: (elo(a) - elo(b)) - (elo(c) - elo(.5)),
                 S['omni-d1det-c3val-vs-c3'], S['omni-k1-vs-c3'], S['c3-d1det-vs-c3'])
        print(f'- search gain on OMNI (C3 scorer, vs raw C3) minus search gain on C3 (vs raw C3): {fmt(d)}')
    if {'omni-d1det-ownval-vs-c3', 'omni-k1-vs-c3', 'c3-d1det-vs-c3'} <= set(S):
        d = boot(lambda a, b, c: (elo(a) - elo(b)) - (elo(c) - elo(.5)),
                 S['omni-d1det-ownval-vs-c3'], S['omni-k1-vs-c3'], S['c3-d1det-vs-c3'])
        print(f'- search gain on OMNI (own head, vs raw C3) minus search gain on C3 (vs raw C3): {fmt(d)}')
    if 'c3-d1det-vs-omni' in S and raw_c3_vs_omni:
        print(f'- C3 + search vs raw OMNI, gain over raw C3 vs raw OMNI: {fmt(gain(S["c3-d1det-vs-omni"], raw_c3_vs_omni))}')
        print(f'- C3 + search vs raw OMNI, Elo over 0.5: {fmt(gain(S["c3-d1det-vs-omni"], H))}')

    print('\n### Score by map type (games)\n')
    kinds = sorted({r.get('map_type', '?') for a in arms.values() for r in a['rows']})
    print('| arm | ' + ' | '.join(kinds) + ' |')
    print('|---|' + '---:|' * len(kinds))
    for n, a in arms.items():
        per = {}
        for r in a['rows']:
            per.setdefault(r.get('map_type', '?'), []).append(points(r))
        print(f'| {LABEL.get(n, n)} ({n.split("-vs-")[1]}) | ' + ' | '.join(
            f'{np.mean(per[k]):.2f} ({len(per[k])})' if k in per else '—' for k in kinds) + ' |')

    print('\n### Critic calibration of the greedy lane (searched arms, first plan of each turn)\n')
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
