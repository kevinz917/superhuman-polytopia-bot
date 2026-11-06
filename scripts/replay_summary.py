"""Summarize Studio replays: per-player trains, research order, builds, captures, city counts by turn.

    python scripts/replay_summary.py artifacts/studio-d3142/replays/<id>.json.gz [...]
"""
import collections
import gzip
import json
import sys

for path in sys.argv[1:]:
    d = json.load(gzip.open(path))
    frames = d['frames']
    print('=' * 8, d['title'], f'({len(frames)} frames)')
    trains, builds, research, misc = (collections.defaultdict(collections.Counter) for _ in range(4))
    order = collections.defaultdict(list)
    first = {}
    cities_by_turn = {}
    for f in frames:
        e = f['event'] or {}
        p, turn = e.get('player'), e.get('turn')
        label = (e.get('action') or {}).get('label', '')
        words = label.split(' ')
        if words[0] == 'Train':
            trains[p][words[1]] += 1
            first.setdefault((p, 'train ' + words[1]), turn)
        elif words[0] == 'Build' or words[0] == 'Clear':
            builds[p][' '.join(words[:2])] += 1
        elif words[0] == 'Research':
            order[p].append(f'{words[1]}@{turn}')
        elif words[0] in ('Capture', 'Offer', 'Super', 'Explorer', 'Population', 'Workshop', 'Resources', 'Promote',
                          'Recover', 'Harvest', 'Attack'):
            misc[p][words[0]] += 1
            if words[0] == 'Capture':
                first.setdefault((p, 'capture'), turn)
        vs = f['views']['all']['visible_state']
        if turn is not None:
            cities_by_turn[turn] = collections.Counter(c['owner'] for c in vs['cities'])
    last = frames[-1]['event'] or {}
    print('result: winner', max(last['rewards'], key=last['rewards'].get), 'turn', last.get('turn'), last.get('reason'))
    for p in (1, 2):
        print(f'-- P{p}')
        print('  train', dict(trains[p].most_common()))
        print('  build', dict(builds[p].most_common()))
        print('  misc ', dict(misc[p].most_common()))
        print('  research', ' '.join(order[p]))
        print('  firsts', {k[1]: v for k, v in first.items() if k[0] == p and k[1] in ('capture', 'train Scoutship',
                                                                                    'train Rider', 'train Archer')})
    turns = sorted(cities_by_turn)
    print('  cities (P1/P2) by turn:', ' '.join(f'{t}:{cities_by_turn[t][1]}/{cities_by_turn[t][2]}'
                                               for t in turns if t % 4 == 0 or t == turns[-1]))
