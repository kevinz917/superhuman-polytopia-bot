"""Win rates from an arena result file: python3 plato-t3/arena/summarize.py runs/plato-t3/<run>.json

Score = (wins + draws/2) / games for the first-named side. A game that reaches the 200-round cap is a draw.
Intervals are 95%, from a bootstrap over boards (each board's seat-swapped games stay together); with fewer than
two boards no interval is printed. A side is named bot:decode, T3 first.
"""
import collections
import json
import random
import sys


def report(rows, out=sys.stdout):
    bad = [r for r in rows if r['outcome'] in ('error', 'step_cap')]
    cells = collections.defaultdict(lambda: collections.defaultdict(list))  # (panel, a, b) -> board -> scores for a
    extra = collections.defaultdict(collections.Counter)
    for r in rows:
        if r['outcome'] in ('error', 'step_cap'):
            continue
        sides = [f'{b}:{d}' for b, d in zip(r['bots'], r['decodes'])]
        a, b = sorted(sides, key=lambda s: (not s.startswith('t3:joint'), not s.startswith('t3:'), s))
        seat = sides.index(a)
        r0, r1 = r['rewards']
        score = .5 if r['outcome'] == 'truncated' or r0 == r1 else float((r0 > r1) == (seat == 0))
        key = (r['panel'], a, b)
        cells[key][r['board']].append(score)
        e = extra[key]
        e['games'] += 1
        e['wins'] += score == 1
        e['draws'] += score == .5
        e['losses'] += score == 0
        e['capped'] += r['outcome'] == 'truncated'
        e['turns'] += r['turn']
        e['first_seat_wins'] += r0 > r1
        e['refused_a'] += r['refused'][seat]
        e['refused_b'] += r['refused'][1 - seat]
        e['decisions_a'] += r['decisions'][seat]
        e['decisions_b'] += r['decisions'][1 - seat]
    rng = random.Random(0)
    print(f'{len(rows)} games, {len(bad)} unusable', file=out)
    for r in bad[:5]:
        print('  unusable:', r['id'], r['outcome'], r.get('error'), file=out)
    print(f'{"panel":10} {"pairing":34} {"games":>6} {"W-D-L":>14} {"score":>6} {"95% interval":>15} {"capped":>7} {"seat1":>6} {"turns":>6}  refused proposals', file=out)
    for key in sorted(cells):
        boards = list(cells[key].values())
        e = extra[key]
        mean = sum(map(sum, boards)) / e['games']
        interval = f'{"-":>15}'
        if len(boards) > 1:
            boots = []
            for _ in range(2000):
                pick = [boards[rng.randrange(len(boards))] for _ in boards]
                boots.append(sum(map(sum, pick)) / sum(map(len, pick)))
            boots.sort()
            interval = f'[{boots[50]:.3f}, {boots[1949]:.3f}]'
        panel, a, b = key
        ref = ', '.join(f'{n} {100 * e[f"refused_{k}"] / max(1, e[f"decisions_{k}"]):.2f}%' for n, k in ((a, 'a'), (b, 'b')))
        print(f'{panel:10} {a + " vs " + b:34} {e["games"]:6d} {e["wins"]:5d}-{e["draws"]:3d}-{e["losses"]:4d} {mean:6.3f} {interval} '
              f'{e["capped"]:7d} {e["first_seat_wins"] / e["games"]:6.2f} {e["turns"] / e["games"]:6.1f}  {ref}', file=out)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    report(json.load(open(sys.argv[1])))
