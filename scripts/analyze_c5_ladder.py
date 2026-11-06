"""Boards of the c5 shared ladder (modal_distributed_rl.py::select outputs), pure Python.

    python scripts/analyze_c5_ladder.py runs/c5/c5-ladder-c3.json [runs/c5/c5-ladder-omni.json ...]

Per candidate: W-L-cap, win rate of all games with the arena's paired-map bootstrap 95% CI, win share of decisive
games, mean turns, per-map-type W-L-cap, recruit mix. The pre-registered pass for a 6-h final vs c3: >= 65% of all
games with the CI lower bound > 58%; seeds within 5 points of each other at 3 h and at 6 h.
"""
import json
import sys


def board(path):
    data = json.load(open(path))
    print(f'## {path}')
    print('| Candidate | W-L-cap | Win rate [95% CI] | Of decisive | Mean turns | Warrior share |')
    print('|---|---|---|---|---|---|')
    rates = {}
    for name, r in data.items():
        if 'error' in r:
            print(f'| {name} | error: {r["error"][:80]} |')
            continue
        w, l, cap = r['wins'], r['losses'], r['truncations']
        n = w + l + r['draws_or_truncations']
        lo, hi = r['ci95']
        rates[name] = (100 * r['win_rate'], 100 * lo, 100 * hi)
        print(f"| {name} | {w}-{l}-{cap} | {100 * r['win_rate']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] | "
              f"{100 * w / max(w + l, 1):.1f}% | {r['mean_turns']:.0f} | {100 * r['candidate_unit_mix'].get('Warrior', 0):.0f}% |")
    print()
    print('Per map type (W-L-cap):')
    types = sorted({t for r in data.values() if 'by_map' in r for t in r['by_map']})
    print('| Candidate | ' + ' | '.join(types) + ' |')
    print('|---|' + '---|' * len(types))
    for name, r in data.items():
        if 'by_map' not in r:
            continue
        cells = [f"{m['wins']}-{m['losses']}-{m['truncations']}" for t in types for m in [r['by_map'].get(t, dict(wins=0, losses=0, truncations=0))]]
        print(f'| {name} | ' + ' | '.join(cells) + ' |')
    print()
    print('Recruit mix (top 5):')
    for name, r in data.items():
        if 'candidate_unit_mix' in r:
            top = list(r['candidate_unit_mix'].items())[:5]
            print(f"  {name}: " + ', '.join(f'{u} {100 * p:.0f}%' for u, p in top))
    print()
    return rates


def main():
    boards = {p: board(p) for p in sys.argv[1:]}
    for p, rates in boards.items():
        if 'c3' not in p:
            continue
        for s in ('s1', 's2'):
            r = rates.get(f'{s}-final')
            if r:
                ok = r[0] >= 65 and r[1] > 58
                print(f"{s}-final vs c3: {r[0]:.1f}% [{r[1]:.1f}, {r[2]:.1f}] -> pass (>=65%, CI lower > 58%): {'YES' if ok else 'NO'}")
        for stage in ('3h', 'final'):
            a, b = rates.get(f's1-{stage}'), rates.get(f's2-{stage}')
            if a and b:
                print(f"seeds at {stage}: {a[0]:.1f}% vs {b[0]:.1f}% -> within 5 points: {'YES' if abs(a[0] - b[0]) <= 5 else 'NO'}")


if __name__ == '__main__':
    main()
