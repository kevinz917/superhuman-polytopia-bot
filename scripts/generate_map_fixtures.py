#!/usr/bin/env python3
"""Corrected-sampling reference layouts, never commercial-map goldens."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from polytopia_bot.gpu import ENGINE_REVISION
root=Path(__file__).resolve().parents[1]
binary=root/'vendor/Polyfish/polyfish-rs/target/debug/mapgen-oracle'
cases=[]
with subprocess.Popen([str(binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as process:
    for size in (11,14,18):
        for seed in (*range(1,17),0,-1,2147483647,-2147483648):
            process.stdin.write(json.dumps(dict(size=size,seed=seed))+'\n');process.stdin.flush()
            raw=json.loads(process.stdout.readline())
            tiles=[raw['tiles'][str(i)] for i in range(size*size)]
            cases.append(dict(size=size,seed=seed,terrain=[t['type'] for t in tiles],
                structure=[(raw['structures'].get(str(i)) or {}).get('type',0) for i in range(size*size)],
                resource=[(raw['resources'].get(str(i)) or {}).get('type',0) for i in range(size*size)],
                capitals=[next(i for i,t in enumerate(tiles) if t.get('capitalOf')==p) for p in (1,2)]))
    process.stdin.close();process.wait()
data=dict(engine_revision=ENGINE_REVISION,source_sha256=hashlib.sha256((root/'vendor/Polyfish/polyfish-rs/src/mapgen.rs').read_bytes()).hexdigest(),
    scope='Pinned Drylands/Bardur layout algorithm with explicit .NET Next/NextDouble sampling, not original bug parity or commercial map equivalence.',cases=cases)
out=root/'tests/fixtures/corrected-drylands.json.gz'
out.write_bytes(gzip.compress(json.dumps(data,separators=(',',':')).encode(),mtime=0))
print(f'{len(cases)} layouts saved to {out}')
