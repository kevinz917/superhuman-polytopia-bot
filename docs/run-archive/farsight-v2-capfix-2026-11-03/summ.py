import json, glob, collections, sys
tag = sys.argv[1]
by = collections.defaultdict(lambda: [0, 0, 0, 0])
keys = None
for f in sorted(glob.glob(f'/vol/results/distributed/arena/{tag}-vs-*.json')):
    d = json.load(open(f))
    if keys is None:
        keys = sorted(d.keys()); print('keys', keys[:40])
    ref = f.split('-vs-')[1].rsplit('-', 1)[0]
    w = d.get('candidate_wins', d.get('wins')); l = d.get('opponent_wins', d.get('losses')); dr = d.get('draws', 0); n = d.get('games')
    if w is None and 'results' in d:
        r = d['results']; w = sum(1 for x in r if x.get('winner') == 'candidate'); l = sum(1 for x in r if x.get('winner') == 'opponent'); dr = len(r) - w - l; n = len(r)
    b = by[ref]; b[0] += w or 0; b[1] += l or 0; b[2] += dr or 0; b[3] += n or 0
for ref, (w, l, dr, n) in by.items():
    print(f'{ref:15s} W {w:3d} L {l:3d} D {dr:3d} / {n:3d}  win% {100*w/max(1,n):5.1f}  score% {100*(w+0.5*dr)/max(1,n):5.1f}')
