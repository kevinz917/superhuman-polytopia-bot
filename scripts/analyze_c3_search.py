"""Summarize the c3 (c3-maps-ctl u4854) test-time search sweep fetched by modal_test_time_search.py::fetch.

    python scripts/analyze_c3_search.py runs/test-time-search-c3/c3-v1

Blocks are arm-name prefixes: c3-classic / c3-mix (mirror vs raw c3: the raw K=1 mirror scores exactly 0.5 on
every map, because both seats play the same deterministic policy) and u3142-classic (baseline arm
u3142-classic-k1). Elo gains are paired by map with a bootstrap CI; pairwise rows compare search variants.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_test_time_search import paired_elo_gain, points  # noqa: E402

BLOCKS = {'c3-classic': 'vs raw c3, classic maps', 'u3142-classic': 'vs raw u3142, classic maps',
          'c3-mix': 'vs raw c3, proposed-v1 map mix'}
LABEL = {'k1': 'raw (no search)', 'd1': 'leaky d=1 K=16', 'd1-det': 'leak-free d=1 K=16',
         'r64': 'leaky robust K=16, 64 replies', 'r64-det': 'leak-free robust K=16, 64 replies'}
PAIRS = [('d1-det', 'd1'), ('r64-det', 'r64'), ('r64', 'd1'), ('r64-det', 'd1-det')]


def mirror_rows(rows):
    return [dict(seed=r['seed'], reward=0.) for r in rows]


def main(path):
    arms = {p.stem: json.loads(p.read_text()) for p in sorted(Path(path).glob('*.json')) if p.stem != 'index'}
    total = 0.
    for prefix, title in BLOCKS.items():
        names = {n[len(prefix) + 1:]: n for n in arms if n.startswith(prefix + '-')}
        if not names:
            continue
        base = arms[names['k1']]['rows'] if 'k1' in names else None
        print(f'\n### {title}\n')
        print('| arm | W–L–trunc | score [95% CI] | Elo vs raw c3 [95% CI] | s / searched turn | plans / turn | override | mean turns | $ | chunks |')
        print('|---|---|---|---|---:|---:|---:|---:|---:|---|')
        for tag in [t for t in LABEL if t in names]:
            a = arms[names[tag]]
            s, rows = a['summary'], a['rows']
            ref = base if base is not None else mirror_rows(rows)
            gain = '—'
            if tag != 'k1':
                g, (lo, hi) = paired_elo_gain(rows, ref)
                gain = f'**{g:+.0f}** [{lo:+.0f}, {hi:+.0f}]'
            total += a.get('cost_usd', 0)
            print(f"| {LABEL[tag]} | {s['wins']}–{s['losses']}–{s['truncations']} | {s['score']:.3f} "
                  f"[{s['score_ci95'][0]:.2f}, {s['score_ci95'][1]:.2f}] | {gain} | {s['seconds_per_search_turn']:.2f} | "
                  f"{s['replans_per_turn']:.2f} | {s['override_rate']:.2f} | {s['mean_turns']:.1f} | "
                  f"{a.get('cost_usd', 0):.2f} | {a.get('chunks')} |")
        for x, y in PAIRS:
            if x in names and y in names:
                g, (lo, hi) = paired_elo_gain(arms[names[x]]['rows'], arms[names[y]]['rows'])
                print(f'- {LABEL[x]} minus {LABEL[y]}: {g:+.0f} [{lo:+.0f}, {hi:+.0f}]')
        if prefix == 'c3-mix':
            for tag in [t for t in LABEL if t in names]:
                kinds = {}
                for r in arms[names[tag]]['rows']:
                    kinds.setdefault(r.get('map_type', '?'), []).append(points(r))
                print(f'- {LABEL[tag]} by map type: ' + ', '.join(
                    f'{k} {np.mean(v):.2f} (n={len(v)})' for k, v in sorted(kinds.items())))
    print(f'\ntotal $ {total:.2f}')


if __name__ == '__main__':
    main(sys.argv[1])
