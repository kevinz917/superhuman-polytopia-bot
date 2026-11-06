"""Win rate of the candidate by map type x seat, plus per-map pair outcomes (2-0 / split / 0-2), from an arena json.

    modal volume get --env staging polytopia-structured-rl distributed/arena/NAME.json /tmp/
    python scripts/arena_by_type_seat.py /tmp/NAME.json
"""
import collections
import json
import sys

s = json.load(open(sys.argv[1]))
rows = s['details']
print(s['name'], s['map_mix'], 'games', len(rows), 'wins', sum(r['reward'] > 0 for r in rows),
      'truncated', sum(r['truncated'] for r in rows), 'win_rate %.3f' % s['win_rate'], 'ci95', s.get('ci95'))
WATER = {'archipelago', 'waterworld', 'lakes'}
groups = collections.defaultdict(list)
for r in rows:
    kind = r.get('map', 'classic')
    groups[kind].append(r)
    groups['(water: archipelago/waterworld/lakes)' if kind in WATER else '(land: everything else)'].append(r)
    groups['(all)'].append(r)
print(f'{"map type":40} {"maps":>4} {"all":>9} {"as Blue":>9} {"as Red":>9}  pairs 2-0/split/0-2  turns')
for kind in sorted(groups, key=lambda k: (k.startswith('('), k)):
    rs = groups[kind]
    rate = lambda xs: f'{sum(x["reward"] > 0 for x in xs)}/{len(xs)}' if xs else '-'
    pairs = collections.defaultdict(list)
    for r in rs:
        pairs[r['seed']].append(r['reward'] > 0)
    split = collections.Counter(sum(v) for v in pairs.values() if len(v) == 2)
    turns = sum(r.get('turn', 0) for r in rs) / len(rs)
    print(f'{kind:40} {len(pairs):>4} {rate(rs):>9} {rate([r for r in rs if r["seat"] == 1]):>9} '
          f'{rate([r for r in rs if r["seat"] == 2]):>9}  {split[2]:>3}/{split[1]:>3}/{split[0]:>3}  {turns:5.1f}')
