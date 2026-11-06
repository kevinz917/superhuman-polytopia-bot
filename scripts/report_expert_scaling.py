"""Fit and plot the expert-SFT data-scaling curve from runs/<run>/summary-small.json (no model compute).

  python3 scripts/report_expert_scaling.py runs/expert-sft-scaling-v1 [out_dir]

Writes docs/expert-sft/scaling-curve-summary.json and docs/expert-sft/scaling-curve.svg.
Fit: per-game logistic regression win ~ a + b*log2(examples seen) over the four SFT checkpoints,
with a map-cluster bootstrap (resample the 15 maps) for the slope and the extrapolations.
"""
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SFT = ['ckpt157', 'ckpt314', 'ckpt471', 'final']


def logistic_fit(xs, ys, iters=50):
    a, b = 0.0, 0.0
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(xs, ys):
            p = 1/(1+math.exp(-(a+b*x)))
            w = p*(1-p)
            ga += y-p; gb += (y-p)*x
            haa += w; hab += w*x; hbb += w*x*x
        det = haa*hbb-hab*hab + 1e-9
        a += (hbb*ga-hab*gb)/det; b += (haa*gb-hab*ga)/det
    return a, b


def predict(a, b, n):
    return 1/(1+math.exp(-(a+b*math.log2(n))))


def main(run_dir, out_dir=ROOT/'docs/expert-sft'):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    s = json.loads((run_dir/'summary-small.json').read_text())
    p = json.loads((run_dir/'protocol.json').read_text())
    curve, wins = s['curve'], s['per_job_wins']
    seed_of = {j['id']: j['seed'] for j in p['jobs']}
    maps = sorted(set(seed_of.values()))
    rows = [(seed_of[j], math.log2(curve[a]['examples_seen']), w) for a in SFT for j, w in wins[a].items()]
    a, b = logistic_fit([r[1] for r in rows], [r[2] for r in rows])
    full = curve['final']['examples_seen']
    targets = {'1x': full, '2x': 2*full, '4x': 4*full}
    rng, boots = random.Random(1701), []
    by_map = {m: [r for r in rows if r[0] == m] for m in maps}
    for _ in range(4000):
        sample = [r for m in rng.choices(maps, k=len(maps)) for r in by_map[m]]
        if len({r[2] for r in sample}) < 2:
            continue
        ba, bb = logistic_fit([r[1] for r in sample], [r[2] for r in sample])
        boots.append((bb, {k: predict(ba, bb, n) for k, n in targets.items()}))
    def ci(vals):
        vals = sorted(vals)
        return [vals[int(.025*len(vals))], vals[int(.975*len(vals))-1]]
    fit = dict(model='win ~ logistic(a + b*log2(examples_seen)), 4 SFT checkpoints x 30 games', a=a, b=b,
               slope_odds_ratio_per_doubling=math.exp(b), slope_bootstrap_95=ci([x[0] for x in boots]),
               slope_positive_fraction=sum(x[0] > 0 for x in boots)/len(boots), bootstrap_reps=len(boots),
               predicted_win_rate={k: dict(examples=n, point=predict(a, b, n), map_cluster_bootstrap_95=ci([x[1][k] for x in boots]))
                                   for k, n in targets.items()},
               caveat='2x/4x are extrapolations of one cosine-LR run; treat as directional only.')
    table = [dict(arm=k, step=v['step'], examples_seen=v['examples_seen'], fraction=v.get('fraction', 0.0),
                  wins=v['wins'], losses=v['losses'], unfinished=v['unfinished'], games=v['games'],
                  win_rate=v['win_rate'], median_round=v['median_round'], median_round_wins=v['median_round_wins'],
                  median_model_moves=v['median_model_moves'], mean_peak_cities=v['mean_peak_cities'],
                  median_peak_cities=v['median_peak_cities'], max_peak_cities=v['max_peak_cities'],
                  learning_rate=v.get('learning_rate'), train_loss_last20=v.get('train_loss_last20'),
                  statuses=v['statuses']) for k, v in curve.items()]
    out = dict(schema='expert-sft-scaling-curve-v1', run=p['run_id'], source_run=p['reference']['run'],
               jobs='expert-sft-v1 dev games: 15 validation maps (100691-100705) x 2 seats vs scripted balanced',
               adapters={k: dict(path=v['adapter'], adapter_files=v['adapter_files']) for k, v in p['arms'].items()},
               table=table, paired=s['paired'], fit=fit,
               doc='docs/expert-sft-scaling-curve-2026-10-19.md')
    (out_dir/'scaling-curve-summary.json').write_text(json.dumps(out, indent=2)+'\n')
    (out_dir/'scaling-curve.svg').write_text(svg(table, a, b, fit))
    print(json.dumps(dict(table=[{k: r[k] for k in ('arm', 'examples_seen', 'wins', 'losses', 'unfinished', 'median_round',
                                                    'mean_peak_cities')} for r in table],
                          paired={k: [round(v['a_minus_b'], 3)] + [round(x, 3) for x in v['map_cluster_bootstrap_95']]
                                  for k, v in s['paired'].items()},
                          fit={k: fit[k] for k in ('b', 'slope_bootstrap_95', 'slope_positive_fraction', 'predicted_win_rate')}), indent=1))


def svg(table, a, b, fit):
    W, H, L, R, T, B = 640, 380, 64, 24, 36, 56
    xmax = 42000
    X = lambda n: L+(W-L-R)*n/xmax
    Y = lambda r: T+(H-T-B)*(1-r)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif" font-size="12">',
             '<style>.ax{stroke:#888;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.lbl{fill:#555}'
             '.pt{fill:#2a6fdb}.fit{stroke:#2a6fdb;stroke-width:1.5;fill:none;stroke-dasharray:5 4}.ci{stroke:#2a6fdb;stroke-width:1.5}'
             '@media (prefers-color-scheme: dark){.lbl{fill:#bbb}.grid{stroke:#333}.ax{stroke:#777}}</style>',
             f'<text x="{L}" y="20" class="lbl" font-size="14" font-weight="600">Expert-SFT win rate vs expert moves seen (30 dev games each)</text>']
    for r in (0, .25, .5, .75, 1):
        parts.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{Y(r):.1f}" y2="{Y(r):.1f}"/>'
                     f'<text class="lbl" x="{L-8}" y="{Y(r)+4:.1f}" text-anchor="end">{int(r*100)}%</text>')
    for n in (0, 10000, 20000, 30000, 40000):
        parts.append(f'<text class="lbl" x="{X(n):.1f}" y="{H-B+18}" text-anchor="middle">{n//1000}k</text>')
    parts.append(f'<line class="ax" x1="{L}" x2="{W-R}" y1="{Y(0)}" y2="{Y(0)}"/><line class="ax" x1="{L}" x2="{L}" y1="{T}" y2="{Y(0)}"/>')
    parts.append(f'<text class="lbl" x="{(L+W-R)/2}" y="{H-12}" text-anchor="middle">expert moves seen (one epoch; final = 39,626)</text>')
    pts = ' '.join(f'{X(n):.1f},{Y(predict(a, b, n)):.1f}' for n in range(6000, 41001, 500))
    parts.append(f'<polyline class="fit" points="{pts}"/>')
    for r in table:
        n, k, g = r['examples_seen'], r['wins'], r['games']
        ph = k/g; z = 1.96
        den = 1+z*z/g; c = (ph+z*z/(2*g))/den; h = z*math.sqrt(ph*(1-ph)/g+z*z/(4*g*g))/den
        parts.append(f'<line class="ci" x1="{X(n):.1f}" x2="{X(n):.1f}" y1="{Y(c-h):.1f}" y2="{Y(min(1, c+h)):.1f}"/>'
                     f'<circle class="pt" cx="{X(n):.1f}" cy="{Y(ph):.1f}" r="5"/>'
                     f'<text class="lbl" x="{X(n)+8:.1f}" y="{Y(ph)-8:.1f}">{k}/{g}</text>')
    parts.append(f'<text class="lbl" x="{W-R}" y="{T+12}" text-anchor="end">dashed: logistic fit in log2(examples); bars: Wilson 95%</text>')
    parts.append('</svg>')
    return '\n'.join(parts)+'\n'


if __name__ == '__main__':
    main(*(sys.argv[1:] or [ROOT/'runs/expert-sft-scaling-v1']))
