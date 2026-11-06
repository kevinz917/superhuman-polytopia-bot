"""Tables for docs/search-distillation-2026-10-19.md.

    .venv/bin/python scripts/analyze_search_distill.py data runs/search-distill/d1/chunks.json
    .venv/bin/python scripts/analyze_search_distill.py eval runs/search-distill/eval/e1 [--pair armA,armB]
    .venv/bin/python scripts/analyze_search_distill.py train runs/search-distill/train/t1/train.json
    .venv/bin/python scripts/analyze_search_distill.py critic runs/search-distill/eval/e2/c3-d1det-vs-raw-c3.json ...

`critic`: how well the searcher's own critic ranks candidates, from the per-turn traces of search-arm rows
(turn, greedy-lane score, chosen-lane score, mean lane score): Pearson correlation of the greedy lane's score (and of
the chosen lane's) with the game outcome over all search turns, with a bootstrap CI over games, plus the
correlation restricted to turns 10-40 (the opening and the decided endgame are easy for every critic).

Arms split into seed blocks (<arm>-b0, <arm>-b1, ...) are merged. Raw arms are cpu_arena summaries (rows: seed, seat, reward); search arms are search_arena rows. A game scores
1 / 0.5 / 0 for win / truncation / loss; Elo CIs are a paired-map bootstrap (4000 resamples over map seeds).
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


def elo(score):
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / score - 1)


def per_map(rows):
    by = {}
    for r in rows:
        by.setdefault(int(r['seed']), []).append(1. if r['reward'] > 0 else .5 if r['reward'] == 0 else 0.)
    return by


def boot(seeds, fn, n=4000):
    rng = np.random.default_rng(0)
    seeds = np.asarray(seeds)
    vals = np.array([fn(seeds[rng.integers(0, len(seeds), len(seeds))]) for _ in range(n)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def arm_rows(d):
    return d.get('details') or d.get('rows') or []


def arm_line(name, rows):
    by = per_map(rows)
    seeds = sorted(by)
    score = float(np.mean([np.mean(by[s]) for s in seeds]))
    lo, hi = boot(seeds, lambda ss: np.mean([np.mean(by[s]) for s in ss]))
    w = sum(r['reward'] > 0 for r in rows)
    l = sum(r['reward'] < 0 for r in rows)
    return dict(name=name, games=len(rows), w=w, l=l, t=len(rows) - w - l, score=score, score_ci=(lo, hi),
                elo=elo(score), elo_ci=(elo(lo), elo(hi)))


def pair_line(a, b, ra, rb):
    """Elo(a) - Elo(b) on the maps both played (each against the same opponent)."""
    ba, bb = per_map(ra), per_map(rb)
    seeds = sorted(set(ba) & set(bb))
    diff = lambda ss: elo(np.mean([np.mean(ba[s]) for s in ss])) - elo(np.mean([np.mean(bb[s]) for s in ss]))
    return dict(pair=f'{a} - {b}', maps=len(seeds), delta=diff(seeds), ci=boot(seeds, diff))


def cmd_eval(root, pairs):
    root = Path(root)
    arms = {}
    for p in sorted(root.glob('*.json')):
        if p.stem == 'index':
            continue
        d = json.loads(p.read_text())
        rows = arm_rows(d)
        if rows:   # arms split into seed blocks (<arm>-b0, -b1, ...) are merged
            arms.setdefault(re.sub(r'-b\d+$', '', p.stem), []).extend(rows)
    print('| arm | W–L–trunc | score [95% CI] | Elo [95% CI] |')
    print('|---|---|---|---:|')
    for name, rows in arms.items():
        a = arm_line(name, rows)
        print(f"| {name} | {a['w']}–{a['l']}–{a['t']} | {a['score']:.3f} [{a['score_ci'][0]:.2f}, {a['score_ci'][1]:.2f}] "
              f"| {a['elo']:+.0f} [{a['elo_ci'][0]:+.0f}, {a['elo_ci'][1]:+.0f}] |")
    for pair in pairs:
        a, b = pair.split(',')
        p = pair_line(a, b, arms[a], arms[b])
        print(f"{p['pair']}: {p['delta']:+.0f} [{p['ci'][0]:+.0f}, {p['ci'][1]:+.0f}] over {p['maps']} maps")
    # by map type
    for name, rows in arms.items():
        kinds = {}
        for r in rows:
            kinds.setdefault(r.get('map') or r.get('map_type', '?'), []).append(
                1. if r['reward'] > 0 else .5 if r['reward'] == 0 else 0.)
        print(name, ' '.join(f'{k} {np.mean(v):.2f} ({len(v)})' for k, v in sorted(kinds.items())))


def cmd_data(path):
    chunks = json.loads(Path(path).read_text())
    metas = [m for c in chunks.values() for m in c.get('metas', [])]
    seconds = sum(c.get('seconds', 0) for c in chunks.values())
    done = [m for m in metas if m['complete']]
    dec = sum(m['decisions'] for m in metas)
    print(f"containers {len(chunks)}  games {len(metas)} (complete {len(done)})  decisions {dec:,}  "
          f"plans {sum(m['plans'] for m in metas):,}  override rate "
          f"{sum(m['overrides'] for m in metas) / max(sum(m['plans'] for m in metas), 1):.3f}")
    print(f"container-hours {seconds / 3600:.1f}  decisions per search core-second "
          f"{dec / max(sum(m['search_seconds'] for m in metas), 1):.2f}  fallbacks {sum(m['fallbacks'] for m in metas)}")
    if done:
        print(f"seat-1 score {np.mean([(m['rewards'][0] + 1) / 2 for m in done]):.3f}  truncated "
              f"{sum(m['truncated'] for m in done)}/{len(done)}  mean turns {np.mean([m['turns'] for m in done]):.1f}  "
              f"decisions/game {np.mean([m['decisions'] for m in done]):.0f}")
    kinds = {}
    for m in metas:
        k = kinds.setdefault(m.get('map_type'), [0, 0, 0])
        k[0] += 1
        k[1] += m['decisions']
        k[2] += m['truncated']
    for k, (g, d, t) in sorted(kinds.items(), key=lambda kv: -kv[1][1]):
        print(f'  {k:12s} games {g:5d} decisions {d:10,d} ({d / dec:.1%}) truncated {t}')
    errors = [c.get('error') for c in chunks.values() if c.get('error')]
    print('errors', len(errors), errors[:1])


def cmd_critic(paths):
    print('| arm | search turns | corr(greedy score, outcome) [95% CI] | corr(chosen, outcome) | corr turns 10-40 | '
          'MSE greedy | mean greedy score / outcome |')
    print('|---|---:|---|---|---|---:|---|')
    for path in paths:
        d = json.loads(Path(path).read_text())
        rows = arm_rows(d)
        games = []
        for r in rows:
            tr = np.asarray(r['trace'], np.float64)
            if len(tr):
                games.append((tr[:, 0], tr[:, 1], tr[:, 2], np.full(len(tr), float(r['reward']))))
        turn, greedy, chosen, out = (np.concatenate([g[i] for g in games]) for i in range(4))
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float('nan')

        def boot_corr(ss):
            g = np.concatenate([games[i][1] for i in ss])
            o = np.concatenate([games[i][3] for i in ss])
            return corr(g, o)
        lo, hi = boot(np.arange(len(games)), boot_corr, n=1000)
        mid = (turn >= 10) & (turn <= 40)
        print(f"| {Path(path).stem} | {len(turn)} | {corr(greedy, out):.3f} [{lo:.3f}, {hi:.3f}] | {corr(chosen, out):.3f} "
              f"| {corr(greedy[mid], out[mid]):.3f} (n={int(mid.sum())}) | {float(np.mean((greedy - out) ** 2)):.3f} "
              f"| {greedy.mean():+.3f} / {out.mean():+.3f} |")


def cmd_train(path):
    d = json.loads(Path(path).read_text())
    for h in d['history.json']:
        t = h['held']['teacher']
        print(f"step {h['step']} epoch {h.get('epoch')} teacher acc {t['correct']:.4f} (ovr {t['acc_override']:.4f} "
              f"grd {t['acc_greedy']:.4f}) ce {t['ce']:.3f} ent {t['entropy']:.3f} vloss {t['vloss']:.4f}")
        for name, m in h['held']['variants'].items():
            print(f"   {name:14s} acc {m['correct']:.4f} (ovr {m['acc_override']:.4f} grd {m['acc_greedy']:.4f}) "
                  f"ce {m['ce']:.3f} kl {m['kl']:.4f} ent {m['entropy']:.3f} vloss {m['vloss']:.4f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('eval', 'data', 'train', 'critic'))
    ap.add_argument('path', nargs='+')
    ap.add_argument('--pair', action='append', default=[])
    a = ap.parse_args()
    {'eval': lambda: cmd_eval(a.path[0], a.pair), 'data': lambda: cmd_data(a.path[0]),
     'train': lambda: cmd_train(a.path[0]), 'critic': lambda: cmd_critic(a.path)}[a.what]()
