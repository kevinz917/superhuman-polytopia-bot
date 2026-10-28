#!/usr/bin/env python3
"""Long sequential differential replays; never used as an environment backend."""
import argparse
from collections import defaultdict
from functools import partial
import json
from pathlib import Path
import random
import jax
import jax.numpy as jnp
import numpy as np
from polytopia_bot.oracle import Oracle
from polytopia_bot.gpu.core import step,action
from polytopia_bot.gpu.importer import from_polyfish
from polytopia_bot.gpu.rules import load_rules


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--games',type=int,default=10)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--size',type=int,default=11)
    args=parser.parse_args()
    rules=load_rules('polyfish')
    advance=jax.jit(partial(step,rules=rules,size=args.size))
    rng=random.Random(8191)
    counts=defaultdict(int)
    completed=0
    with Oracle() as oracle:
        for seed in range(1,args.games+1):
            response=oracle.request(op='reset',size=args.size,seed=seed)
            s=from_polyfish(response['state']).state
            for index in range(args.steps):
                groups=defaultdict(list)
                for a in response['legal']:
                    k=a['moveType'];p=a.get('type',0)
                    if k==3 and p not in (1,2,3,4,7,8,9,16):continue
                    if k==4 and not rules.unit_supported[p]:continue
                    if k==6 and p==37:continue
                    if k not in (1,2,3,4,5,6,7,8,9,10):continue
                    groups[k].append(a)
                if not groups:break
                kinds=list(groups)
                weights=[{1:8,2:12,3:1,4:4,5:6,6:3,7:3,8:20,9:40,10:2}[k] for k in kinds]
                kind=rng.choices(kinds,weights)[0]
                a=rng.choice(groups[kind])
                before=response['state']
                result=advance(s,action(kind,source=a.get('src',-1),target=a.get('target',-1),parameter=a.get('type',0)),s.state_version)
                response=oracle.request(op='step',state=before,action=a)
                imported=from_polyfish(response['state'])
                expected=imported.state._replace(actions=s.actions+1,state_version=s.state_version+1)
                mismatches=[]
                for (path,value),wanted in zip(jax.tree.flatten_with_path(result.state)[0],jax.tree.leaves(expected)):
                    if not np.array_equal(value,wanted):mismatches.append(str(path))
                if result.status!=0 or imported.unsupported or mismatches:
                    dest=Path('runs/trajectory-counterexample.json');dest.parent.mkdir(exist_ok=True)
                    dest.write_text(json.dumps(dict(seed=seed,index=index,before=before,action=a,after=response['state'],status=int(result.status),unsupported=imported.unsupported,mismatches=mismatches),indent=2))
                    raise AssertionError(dict(seed=seed,index=index,action=a,status=int(result.status),mismatches=mismatches,unsupported=imported.unsupported,artifact=str(dest)))
                s=result.state;counts[kind]+=1
                if s.terminated or s.truncated:
                    completed+=1;break
            print(json.dumps(dict(seed=seed,steps=index+1,terminated=bool(s.terminated),truncated=bool(s.truncated))),flush=True)
    print(json.dumps(dict(games=args.games,finished=completed,transitions=sum(counts.values()),action_counts=counts)),flush=True)

if __name__=='__main__':main()
