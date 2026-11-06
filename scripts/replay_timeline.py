"""Turn-by-turn timeline of a Studio replay: economy, army, cities, value estimates, city flips.

    curl -s http://localhost:3000/api/meta > /tmp/meta.json   # unit/terrain names from a running Studio
    python scripts/replay_timeline.py artifacts/studio-d3142/replays/<id>.json.gz /tmp/meta.json
"""
import collections
import gzip
import json
import sys

d = json.load(gzip.open(sys.argv[1]))
meta = json.load(open(sys.argv[2]))
UNIT = {u['id']: u['name'] for u in meta['rules']['units']}
TERR = {int(k): v for k, v in meta['terrain'].items()}
STRUCT = {s['id']: s['name'] for s in meta['rules']['structures']} if isinstance(meta['rules']['structures'], list) else {}
frames = d['frames']
print(d['title'])

vs0 = frames[0]['views']['all']['visible_state']
size = vs0['board_size']
terr = collections.Counter(TERR[t['terrain']] for t in vs0['tiles'])
print('terrain', dict(terr.most_common()))
caps = {c['owner']: c['tile'] for c in vs0['cities'] if c['owner']}
print('capitals', {p: divmod(t, size) for p, t in caps.items()})
print('villages', sum(1 for c in vs0['cities'] if not c['owner']))

owner = {c['tile']: c['owner'] for c in vs0['cities']}
values = collections.defaultdict(list)
acts = collections.defaultdict(collections.Counter)
for i, f in enumerate(frames):
    e = f['event'] or {}
    p, turn = e.get('player'), e.get('turn')
    label = (e.get('action') or {}).get('label', '')
    if p:
        values[(turn, p)].append(e['metrics']['value'])
        acts[(turn, p)][label.split(' ')[0] + ('' if label.split(' ')[0] not in ('Train', 'Build', 'Research') else ' ' + label.split(' ')[1])] += 1
    vs = f['views']['all']['visible_state']
    for c in vs['cities']:
        if owner.get(c['tile']) != c['owner']:
            print(f'  t{turn} city {divmod(c["tile"], size)} lvl{c["level"]}: {owner.get(c["tile"])} -> {c["owner"]} ({label})')
            owner[c['tile']] = c['owner']
    if label == 'End turn':
        v = f['views'][str(p)]
        st = v['visible_state']
        units = collections.Counter(UNIT.get(u['kind'], u['kind']) for u in vs['units'] if u['owner'] == p)
        enemy_units = sum(1 for u in vs['units'] if u['owner'] == 3 - p)
        cities = [c for c in vs['cities'] if c['owner'] == p]
        terr_n = sum(len(x) for x in v['hud']['territory'].values())
        val = values[(turn, p)]
        a = acts[(turn, p)]
        a.pop('End', None)
        print(f't{turn:>2} P{p} val {val[0]:+.2f}->{val[-1]:+.2f} stars {st["stars"]:>3} inc {v["hud"]["income"]:>3} '
              f'score {v["hud"]["score"]:>5} cities {len(cities):>2} pop {sum(c["population"] for c in cities):>3} '
              f'terr {terr_n:>3} techs {len(st["technology"]):>2} units {sum(units.values()):>2} (enemy {enemy_units:>2}) '
              f'{dict(units.most_common())} | {dict(a.most_common(6))}')
