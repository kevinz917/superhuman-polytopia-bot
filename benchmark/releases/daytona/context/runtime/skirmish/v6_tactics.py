"""Small engine-valid tactical fixtures, generated before testing and scored separately."""
import numpy as np
from .game import reset_map
from .common import write_json


def make_suite():
    from polytopia_bot.studio.environment import RustEnvironment
    from polytopia_bot.learning.memory import PublicMemory
    fixtures=[]
    for kind in ['last_city_capture','one_hit_kill','knight_chain_start']:
        env=RustEnvironment(size=14,max_rounds=200,max_actions=40000)
        try:
            reset_map(env,750000001,'drylands')
            d={k:np.array(v,copy=True) for k,v in env.games.batch.export().items()}
            for k in d:
                if k.startswith('units.'):d[k][0]=0
            d['tiles.explored'][:]=True
            d['current_player'][0]=1
            if kind=='last_city_capture':
                target=int(np.flatnonzero((d['tiles.capital'][0]==2)&(d['cities.owner'][0]==2))[0])
                d['units.owner'][0,target]=1;d['units.kind'][0,target]=2;d['units.health'][0,target]=10
                predicate=dict(kind=8,source=target)
            else:
                source,target=100,101
                for tile in [source,target,102]:
                    d['tiles.terrain'][0,tile]=3;d['tiles.structure'][0,tile]=0;d['tiles.resource'][0,tile]=0
                    d['cities.owner'][0,tile]=0;d['tiles.owner'][0,tile]=0
                d['units.owner'][0,source]=1;d['units.kind'][0,source]=4 if kind=='knight_chain_start' else 2;d['units.health'][0,source]=10
                d['units.owner'][0,target]=2;d['units.kind'][0,target]=2;d['units.health'][0,target]=1
                if kind=='knight_chain_start':
                    d['units.owner'][0,102]=2;d['units.kind'][0,102]=2;d['units.health'][0,102]=1
                predicate=dict(kind=2,target=target)
            env.games.batch.import_({k:np.ascontiguousarray(v) for k,v in d.items()});env._sync();env.memory=PublicMemory();env._cached={}
            obs=env.observe(1)
            assert any(all(a.command.get(k)==v for k,v in predicate.items()) for a in obs.legal_actions),(kind,predicate)
            fixtures.append(dict(name=kind,snapshot=env.snapshot().decode(),predicate=predicate,description='single-decision tactical diagnostic; not a full-game strength score'))
        finally:env.close()
    return fixtures

if __name__=='__main__':write_json('/results/tactical-fixtures.json',make_suite())
