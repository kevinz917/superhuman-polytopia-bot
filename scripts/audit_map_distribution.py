#!/usr/bin/env python3
"""Read-only map-distribution audit of the pinned source generator."""
import json
from collections import Counter
from polytopia_bot.oracle import Oracle
from polytopia_bot.gpu import ENGINE_REVISION

with Oracle() as oracle:
    rows=[]
    for size in (11,14):
        for seed in range(1,17):
            raw=oracle.request(op='reset',size=size,seed=seed,raw=True)['state']
            counts=Counter(tile['type'] for tile in raw['tiles'].values())
            rows.append(dict(size=size,seed=seed,terrain_counts=dict(counts),
                land_fraction=sum(counts[k] for k in (3,4,5))/(size*size)))
print(json.dumps(dict(engine_revision=ENGINE_REVISION,
    scope='Pinned generator output before post_load. Not a native-map corpus.',
    source_land_target=.95,
    finding='dotnet_rng RngCore emits only 31 bits per u32; rand bounded/float distributions require uniform full-width bits. Drylands sampling cannot reach its target land ratio.',
    maps=rows),indent=2))
