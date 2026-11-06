import json, glob, sys, collections
tag = sys.argv[1]
acc = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(f'/vol/results/distributed/arena/{tag}-vs-*.json')):
    d = json.load(open(f)); ref = f.split('-vs-')[1].rsplit('-', 1)[0]
    g = d.get('growth') or {}
    for k, v in g.items():
        if v is not None: acc[ref][k].append(v)
    acc[ref]['mean_turns'].append(d.get('mean_turns') or 0)
for ref in sorted(acc):
    m = {k: sum(v) / len(v) for k, v in acc[ref].items()}
    if 'cand_builds' not in m: print(f'{ref:15s} no growth fields'); continue
    print(f"{ref:15s} builds {m['cand_builds']:5.1f} vs {m['opp_builds']:5.1f} | harvests {m['cand_harvests']:4.1f} vs {m['opp_harvests']:4.1f} | t16 cities {m['cand_cities_t16']:.2f} vs {m['opp_cities_t16']:.2f} level {m['cand_level_t16']:.2f} vs {m['opp_level_t16']:.2f} | end cities {m['cand_cities_end']:.2f} vs {m['opp_cities_end']:.2f} level {m['cand_level_end']:.2f} vs {m['opp_level_end']:.2f} | turns {m['mean_turns']:.0f}")
