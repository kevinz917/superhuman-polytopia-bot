"""Round-robin readout from arena summaries: score table, Bradley-Terry Elo, per-pair residuals, 3-cycles.

    python scripts/roundrobin_elo.py docs/ablation-v1/diag-18x18/*.json --anchor grown [--order grown,r2u220,...]

Each file is one `arena` result (/results/distributed/arena/<name>.json) named `<prefix>-<a>-vs-<b>.json`, with
<a> the candidate. Score = (W + 0.5 * cap) / games, so turn-cap truncations count as half a win (as in
ladder_elo.py, whose fit this reuses). A pair's standard error clusters the two seats of a map (per-map mean
scores, both seats on the same layout), so residual z-scores account for map-level correlation.
"""
import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

from ladder_elo import fit


def load(paths):
    pairs = {}
    for path in paths:
        a, b = re.match(r'.*?-([^-]+)-vs-([^-]+)\.json$', Path(path).name).groups()
        r = json.loads(Path(path).read_text())
        per_map = {}
        for g in r.get('details', []):
            per_map.setdefault(g['seed'], []).append(1. if g['reward'] > 0 else 0. if g['reward'] < 0 else .5)
        means = np.array([np.mean(v) for v in per_map.values()])
        se = float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else float('nan')
        pairs[a, b] = dict(wins=r['wins'], losses=r['losses'], caps=r['draws_or_truncations'], games=r['games'],
                           mean_turns=r.get('mean_turns'), se=se)
    return pairs


def view(pairs, a, b):
    """W, L, cap, games and score from a's side, whichever way the match was played; None if not played."""
    if (a, b) not in pairs and (b, a) not in pairs:
        return None
    if (a, b) in pairs:
        p = pairs[a, b]
        w, l = p['wins'], p['losses']
    else:
        p = pairs[b, a]
        w, l = p['losses'], p['wins']
    return w, l, p['caps'], p['games'], (w + .5 * p['caps']) / p['games'], p['se']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--order', help='comma-separated row order of the table (default: by Elo)')
    args = parser.parse_args()
    pairs = load(args.files)
    rows = [dict(run_id=a, stage='fixed', reference=b, wins=p['wins'], losses=p['losses'],
                 draws_or_truncations=p['caps']) for (a, b), p in pairs.items()]
    ratings = fit(rows, args.anchor)
    elo = {r['player']: r['elo'] for r in ratings}
    names = args.order.split(',') if args.order else [r['player'] for r in ratings]

    print('| row vs col | ' + ' | '.join(names) + ' | mean score |')
    print('|---' * (len(names) + 2) + '|')
    for a in names:
        cells, scores = [], []
        for b in names:
            if a == b or view(pairs, a, b) is None:
                cells.append('—' if a == b else 'not played')
                continue
            w, l, c, n, s, _ = view(pairs, a, b)
            scores.append(s)
            cells.append(f'{w}-{l}-{c}<br>{s:.1%} · dec {(w + l) / n:.0%}')
        print(f'| **{a}** | ' + ' | '.join(cells) + f' | {np.mean(scores):.1%} |')

    print('\n| player | Elo (anchor ' + args.anchor + ') | ± se | W-L-cap |')
    print('|---|---|---|---|')
    for r in ratings:
        print(f"| {r['player']} | {r['elo']:+.1f} | {r['se']:.1f} | {'-'.join(map(str, r['record']))} |")

    k = np.log(10) / 400
    chi2 = 0.
    print('\n| pair | observed | BT predicted | residual | pair se | z |')
    print('|---|---|---|---|---|---|')
    for a, b in itertools.combinations(names, 2):
        if view(pairs, a, b) is None:
            continue
        _, _, _, _, s, se = view(pairs, a, b)
        p = 1 / (1 + np.exp(-k * (elo[a] - elo[b])))
        z = (s - p) / se
        chi2 += z * z
        print(f'| {a} vs {b} | {s:.1%} | {p:.1%} | {100 * (s - p):+.1f} pts | {100 * se:.1f} pts | {z:+.2f} |')
    dof = len(pairs) - (len(names) - 1)
    print(f'\nchi2 = {chi2:.2f} on {dof} dof (pairs - free ratings)')

    print('\nDirected 3-cycles (each edge score > 50%; strict = every edge more than 2 se above 50%):')
    found = False
    for x, y, z_ in itertools.permutations(names, 3):
        if x != min(x, y, z_):
            continue  # each cycle once, starting from its smallest name
        edges = [view(pairs, u, v) for u, v in ((x, y), (y, z_), (z_, x))]
        if all(e is not None and e[4] > .5 for e in edges):
            found = True
            strict = all(e[4] - .5 > 2 * e[5] for e in edges)
            print(f'  {x} > {y} > {z_} > {x}: ' + ', '.join(f'{e[4]:.1%}' for e in edges) +
                  (' STRICT' if strict else ' (within noise on at least one edge)'))
    if not found:
        print('  none')


if __name__ == '__main__':
    main()
