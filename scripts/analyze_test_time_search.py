"""Summarize test-time search arms fetched by experiments/modal_test_time_search.py::fetch.

    python scripts/analyze_test_time_search.py runs/test-time-search-v1/v1 [more dirs]

Prints a markdown table per opponent (score, Elo vs the K=1 arm on the same maps with a paired-map
bootstrap CI, compute per searched turn), the critic's calibration by game phase, and draws by K.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


def elo(score):
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / score - 1)


def points(r):
    return 1. if r['reward'] > 0 else .5 if r['reward'] == 0 else 0.


def by_map(rows):
    out = {}
    for r in rows:
        out.setdefault(r['seed'], []).append(points(r))
    return {k: np.mean(v) for k, v in out.items()}


def paired_elo_gain(rows, base_rows, n=4000):
    a, b = by_map(rows), by_map(base_rows)
    keys = sorted(set(a) & set(b))
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n):
        i = rng.integers(0, len(keys), len(keys))
        boots.append(elo(x[i].mean()) - elo(y[i].mean()))
    return elo(x.mean()) - elo(y.mean()), np.percentile(boots, [2.5, 97.5])


def calibration(rows, edges=(0, 10, 20, 40, 60, 80, 101)):
    """Greedy-lane critic score at each searched turn vs the game's final outcome (searcher view)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v, z = [], []
        for r in rows:
            for turn, greedy, chosen, mean in r['trace']:
                if lo <= turn < hi:
                    v.append(greedy)
                    z.append(r['reward'])
        if v:
            v, z = np.array(v), np.array(z)
            out.append(dict(turns=f'{lo}-{hi - 1}', n=len(v), mean_value=float(v.mean()), mean_outcome=float(z.mean()),
                            mse=float(np.mean((v - z) ** 2)), corr=float(np.corrcoef(v, z)[0, 1]) if v.std() and z.std() else float('nan')))
    return out


def series_key(name, c):
    opp = name.split('-k')[0]
    parts = [f"d={c.get('depth', 0)}"]
    if c.get('temperature', 1.) != 1.:
        parts.append(f"T={c['temperature']}")
    if c.get('root', 'sequence') != 'sequence':
        parts.append(f"root={c['root']}")
    if c.get('depth', 0):
        if c.get('rollout_temperature', 0.):
            parts.append(f"random replies (T={c['rollout_temperature']})")
        else:
            parts.append('greedy replies' + (' = the opponent' if opp == 'u3142' else ''))
    if c.get('replies'):
        parts.append('averaged + halving')
    if c.get('after_end'):
        parts.append('scored after END')
    return f"vs {opp}, " + ', '.join(parts)


def svg_chart(points, path):
    """points: [(series, K, seconds/turn, gain, lo, hi)]. Log-x seconds per searched turn, y Elo vs K=1."""
    W, H, L, R, T, B = 1040, 460, 70, 380, 30, 50
    xs = [p[2] for p in points]
    x0, x1 = math.log10(min(xs) / 1.5), math.log10(max(xs) * 1.5)
    y0, y1 = min(-100, min(p[4] for p in points) - 20), max(p[5] for p in points) + 20
    X = lambda v: L + (math.log10(v) - x0) / (x1 - x0) * (W - L - R)
    Y = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)
    colors = ['#2a6fdb', '#d9480f', '#2b8a3e', '#7048e8', '#c2255c', '#0b7285', '#868e96', '#e67700', '#5c940d',
              '#364fc7', '#a61e4d', '#1098ad']
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    for v in range(int(y0 // 100) * 100, int(y1) + 1, 100):
        if y0 <= v <= y1:
            out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{"#444" if v == 0 else "#ddd"}"/>'
                       f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#333">{v:+d}</text>')
    for v in (0.1, 0.3, 1, 3, 10):
        if x0 <= math.log10(v) <= x1:
            out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T}" y2="{H - B}" stroke="#eee"/>'
                       f'<text x="{X(v):.1f}" y="{H - B + 16}" text-anchor="middle" fill="#333">{v:g}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 10}" text-anchor="middle" fill="#333">seconds of one CPU core per searched turn (log)</text>')
    out.append(f'<text transform="translate(16,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle" fill="#333">Elo gain over K=1 (same opponent, same maps)</text>')
    for i, name in enumerate(sorted({p[0] for p in points})):
        c = colors[i % len(colors)]
        pts = sorted({p for p in points if p[0] == name}, key=lambda p: p[2])
        out.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (c, ' '.join(f'{X(p[2]):.1f},{Y(p[3]):.1f}' for p in pts)))
        for _, k, sec, g, lo, hi in pts:
            out.append(f'<line x1="{X(sec):.1f}" x2="{X(sec):.1f}" y1="{Y(lo):.1f}" y2="{Y(hi):.1f}" stroke="{c}" opacity=".5"/>'
                       f'<circle cx="{X(sec):.1f}" cy="{Y(g):.1f}" r="4" fill="{c}"/>'
                       f'<text x="{X(sec) + 6:.1f}" y="{Y(g) - 6:.1f}" fill="{c}" font-size="11">K={k}</text>')
        out.append(f'<rect x="{W - R + 14}" y="{T + i * 20}" width="12" height="12" fill="{c}"/>'
                   f'<text x="{W - R + 32}" y="{T + i * 20 + 10}" fill="#333">{name}</text>')
    out.append('</svg>')
    Path(path).write_text('\n'.join(out) + '\n')


def main(dirs, svg=None):
    points = []
    arms = {}
    for d in dirs:
        for p in sorted(Path(d).glob('*.json')):
            if p.stem != 'index':
                arms[p.stem] = json.loads(p.read_text())
    groups = {}
    for name, a in arms.items():
        groups.setdefault(name.split('-k')[0], []).append(name)
    for opp, names in groups.items():
        names.sort(key=lambda n: (arms[n]['config'].get('depth', 0), arms[n]['config'].get('temperature', 1.),
                                  arms[n]['config']['lanes']))
        base = next((n for n in names if arms[n]['config']['lanes'] == 1), None)
        print(f'\n### vs {opp}\n')
        print('| arm | K | d | T | games | W–L–trunc | score [95% CI] | Elo vs K=1 [95% CI] | s / searched turn | lane steps / turn | override | mean turns | $ |')
        print('|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|')
        for n in names:
            a, s, c = arms[n], arms[n]['summary'], arms[n]['config']
            gain = '—'
            if base and n != base:
                g, (lo, hi) = paired_elo_gain(a['rows'], arms[base]['rows'])
                gain = f'{g:+.0f} [{lo:+.0f}, {hi:+.0f}]'
                points.append((series_key(n, c), c['lanes'], s['seconds_per_search_turn'], g, lo, hi))
                b = arms[base]['summary']['seconds_per_search_turn']
                points.append((series_key(n, c), 1, b, 0., 0., 0.))
            print(f"| {n} | {c['lanes']} | {c.get('depth', 0)} | {c.get('temperature', 1.)} | {s['games']} | "
                  f"{s['wins']}–{s['losses']}–{s['truncations']} | {s['score']:.3f} [{s['score_ci95'][0]:.2f}, {s['score_ci95'][1]:.2f}] | "
                  f"{gain} | {s['seconds_per_search_turn']:.2f} | {s['lane_steps_per_turn']:.0f} | {s['override_rate']:.2f} | "
                  f"{s['mean_turns']:.1f} | {a.get('cost_usd', 0):.2f} |")
        for n in names:
            if arms[n]['config']['lanes'] in (1, max(arms[m]['config']['lanes'] for m in names)) and arms[n]['config'].get('depth', 0) == 0:
                print(f'\ncalibration {n}: ' + json.dumps(calibration(arms[n]['rows'])))
    print('\ntotal $', round(sum(a.get('cost_usd', 0) for a in arms.values()), 2))
    if svg and points:
        # root=first collapses to about -400 Elo (docs explain why); plotting it would flatten the rest.
        # league96 sits at a 93% ceiling (CIs of +-150 Elo) and is left to the tables.
        svg_chart([p for p in points if 'root=first' not in p[0] and 'league96' not in p[0]], svg)


if __name__ == '__main__':
    args = sys.argv[1:]
    svg = args[args.index('--svg') + 1] if '--svg' in args else None
    main([a for a in args if a != '--svg' and a != svg], svg)
