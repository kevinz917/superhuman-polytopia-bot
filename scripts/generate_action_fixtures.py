#!/usr/bin/env python3
"""Independent Rust golden transitions for military and economy mechanics."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random
from polytopia_bot.oracle import Oracle
from polytopia_bot.gpu import ENGINE_REVISION

ROOT = Path(__file__).resolve().parents[1]

def coord(i,n):
    return dict(idx=i,x=i%n,y=i//n)


def main():
    cases=[]
    rng=random.Random(173)
    with Oracle() as o:
        rules=o.request(op='rules')
        stats={u['id']:u for u in rules['units']}
        def save(name,s,select):
            legal=o.request(op='query',state=s)['legal']
            picked=[a for a in legal if select(a)]
            if not picked:
                raise AssertionError((name,'no selected actions',legal))
            transitions=[dict(action=a,after=o.request(op='step',state=s,action=a)['state']) for a in picked]
            cases.append(dict(name=name,before=deepcopy(s),legal=legal,transitions=transitions))
        for size in (11,14):
            base=o.request(op='reset',size=size,seed=1)['state']
            for p in base['tribes'].values():
                p['stars']=100
                p['knownPlayers']=[3-p['id']]
                p['tech_vanilla']=[dict(type=t['id'],discovered=True,discoveredTurn=0) for t in rules['technologies'] if t['id'] in (*range(25),38,44)]
            base['settings']['turn']=5
            for t in base['tiles'].values():
                t['explorers']=[1,2]
            for j in range(64):
                s=deepcopy(base)
                center=size*(size//2)+size//2
                for tile in s['tiles'].values():
                    tile.pop('unitOwnerId',None)
                for player,pos in ((1,center),(2,center+1+(size if j%4==0 else 0))):
                    u=s['tribes'][str(player)]['units'][0]
                    k=rng.choice((2,3,4,5,8,9,11,12))
                    u.update(type=k,health=float(rng.randint(1,int(stats[k]['health']))),veteran=j%7==0,kills=j%4,
                             moved=False,attacked=False,effects=[1] if j%9==0 else ([2] if j%6==0 else []),coords=coord(pos,size))
                    t=s['tiles'][str(pos)]
                    t['type']=rng.choice((3,4,5));t['unitOwnerId']=player
                save(f'{size}-combat-{j}',s,lambda a:a['moveType']==2)
            # Empty capital: every affordable ordinary unit, capacity enforced separately.
            s=deepcopy(base)
            unit=s['tribes']['1']['units'].pop()
            s['tiles'][str(unit['coords']['idx'])].pop('unitOwnerId',None)
            save(f'{size}-train',s,lambda a:a['moveType']==4 and a['type'] in (2,3,4,5,8,9,10,11))
            # Every resource harvest and forest action available on actual owned territory.
            s=deepcopy(base)
            city=s['tribes']['1']['cities'][0]
            tiles=city['territory']
            for j,i in enumerate(tiles):
                s['structures'].pop(str(i),None)
                s['tiles'][str(i)]['type']=5 if j%2 else 3
                s['resources'][str(i)]={'type':(1,3,6)[j%3]}
            save(f'{size}-economy',s,lambda a:a['moveType']==5 or (a['moveType']==3 and a.get('type') in (1,2,3)))
            # Promotion remains legal after moving, disband removes poisoned units correctly.
            for moved in (False,True):
                s=deepcopy(base)
                u=s['tribes']['1']['units'][0]
                u.update(kills=3,health=3.0,moved=moved,attacked=moved,effects=[1])
                save(f'{size}-unit-abilities-{moved}',s,lambda a:a['moveType']==3 and a.get('type') in (8,16))
            # Building, destruction and income cases include every ordinary land
            # structure and monument, plus independent source-generated afterstates.
            s=deepcopy(base)
            c=s['tribes']['1']['cities'][0]
            c.update(level=5,rewards=[3,1,8,2],production=5)
            s['tribes']['1']['kills']=10
            s['tribes']['1']['pacifistTurns']=5
            for j,i in enumerate(c['territory']):
                s['structures'].pop(str(i),None)
                s['tiles'][str(i)]['type']=(3,5,4,1)[j%4]
                resource=2 if j%4==0 else 5 if j%4==2 else 0
                if resource:
                    s['resources'][str(i)]={'type':resource}
                else:
                    s['resources'].pop(str(i),None)
            # Multipliers need adjacent production buildings.
            for j,kind in enumerate((5,12,21)):
                i=c['territory'][j]
                s['structures'][str(i)]={'type':kind,'level':1,'founded':0}
            save(f'{size}-buildings',s,lambda a:a['moveType']==6 and a['type'] not in (8,48,37))
            built=[a for a in o.request(op='query',state=s)['legal'] if a['moveType']==6 and a['type'] not in (8,48,37)]
            for k,a in enumerate(built):
                after=o.request(op='step',state=s,action=a)['state']
                # Resolve pending population rewards before testing destroy/income.
                for c2 in after['tribes']['1']['cities']:
                    while len(c2['rewards'])<c2['level']-1:
                        c2['rewards'].append(2)
                if a['type']!=3:
                    save(f'{size}-destroy-{k}',after,lambda b:b['moveType']==3 and b.get('type')==4 and b['target']==a['target'])
                after['settings']['currentPlayerTurnId']=2
                after['settings']['turn']=14
                save(f'{size}-income-{k}',after,lambda b:b['moveType']==10)
            # Neutral capture, enemy city transfer, and last-city elimination.
            for enemy in (False,True):
                s=deepcopy(base)
                src=s['tribes']['2']['cities'][0]['idx'] if enemy else next(int(k) for k,v in s['structures'].items() if v and v['type']==1 and s['tiles'][k]['owner']==0)
                for t in s['tiles'].values(): t.pop('unitOwnerId',None)
                u=s['tribes']['1']['units'][0]
                u['coords']=coord(src,size)
                s['tiles'][str(src)]['unitOwnerId']=1
                other=s['tribes']['2']['units'][0]
                other['coords']=coord(size*size//2,size)
                s['tiles'][str(other['coords']['idx'])]['unitOwnerId']=2
                save(f'{size}-capture-{enemy}',s,lambda a:a['moveType']==8 and a['src']==src)
                if enemy:
                    after=o.request(op='step',state=s,action={'moveType':8,'src':src})['state']
                    save(f'{size}-native-terminal',after,lambda a:a['moveType']==10)
            # Multi-hop movement: roads, hidden terrain and a friendly blocker.
            for variant in range(4):
                s=deepcopy(base)
                u=s['tribes']['1']['units'][0]
                source=u['coords']['idx']
                u.update(type=3,health=10.0)
                for key,t in s['tiles'].items():
                    i=int(key)
                    t['type']=3 if i%7 else 5
                    t['hasRoad']=variant%2==0
                    if max(abs(i%size-source%size),abs(i//size-source//size))>1:
                        t['explorers']=[2]
                # A visible corridor makes multi-hop moves legal while adjacent fog reveals vary.
                for dx in range(5):
                    i=source+dx
                    if i<size*size: s['tiles'][str(i)]['explorers']=[1,2]
                if variant>=2:
                    friend=deepcopy(u);friend['coords']=coord(source+1,size)
                    s['tribes']['1']['units'].append(friend);s['tiles'][str(source+1)]['unitOwnerId']=1
                save(f'{size}-path-{variant}',s,lambda a:a['moveType']==1 and a['src']==source)
            for seed in range(8):
                s=o.request(op='reset',size=size,seed=seed)['state']
                s['tribes']['1']['cities'][0]['level']=2
                save(f'{size}-explorer-{seed}',s,lambda a:a['moveType']==9 and a['type']==4)
            for kind in (41,42,43,44,45):
                for scenario in ('move','attack','upgrade'):
                    s=deepcopy(base)
                    center=(size//2)*size+size//2
                    for tile in s['tiles'].values():
                        tile.pop('unitOwnerId',None)
                        tile['type']=1
                        tile['explorers']=[1,2]
                    for who,idx in ((1,center),(2,center+2)):
                        u=s['tribes'][str(who)]['units'][0]
                        u.update(type=kind if who==1 else 44,passengerType=5,health=15.0,coords=coord(idx,size))
                        s['tiles'][str(idx)]['unitOwnerId']=who
                    if scenario=='move':
                        # Legal land disembark plus ocean unlocked by Sailing.
                        s['tiles'][str(center+size)]['type']=3
                        s['tiles'][str(center-size)]['type']=2
                        save(f'{size}-naval-move-{kind}',s,lambda a:a['moveType']==1)
                    elif scenario=='attack':
                        u=s['tribes']['2']['units'][0]
                        s['tiles'][str(center+2)].pop('unitOwnerId',None)
                        u['coords']=coord(center+1,size);s['tiles'][str(center+1)]['unitOwnerId']=2
                        # Neighbor enemies exercise Bomber splash / Juggernaut stomp.
                        friend=deepcopy(u);friend['coords']=coord(center+size+1,size);friend['health']=3.0
                        s['tribes']['2']['units'].append(friend);s['tiles'][str(center+size+1)]['unitOwnerId']=2
                        save(f'{size}-naval-attack-{kind}',s,lambda a:a['moveType']==2)
                    elif kind==43:
                        save(f'{size}-naval-upgrade',s,lambda a:a['moveType']==4 and a['src']==center)
            for kind in (2,3,5,10,12):
                s=deepcopy(base)
                u=s['tribes']['1']['units'][0];source=u['coords']['idx'];target=source+1
                u.update(type=kind,health=float(stats[kind]['health']))
                s['structures'][str(target)]={'type':8,'level':1,'founded':0}
                s['tiles'][str(target)]['type']=1;s['tiles'][str(target)]['owner']=1
                save(f'{size}-embark-{kind}',s,lambda a:a['moveType']==1 and a['target']==target)
            for rng_seed in range(32):
                s=deepcopy(base)
                if rng_seed % 2 == 0:
                    s['tribes']['1']['tech_vanilla']=[dict(type=k,discovered=True,discoveredTurn=0) for k in (0,15)]
                target=next(int(k) for k,v in s['structures'].items() if v and v['type']==2 and s['tiles'][k]['type']>2)
                u=s['tribes']['1']['units'][0]
                s['tiles'][str(u['coords']['idx'])].pop('unitOwnerId',None)
                u['coords']=coord(target,size);s['tiles'][str(target)]['unitOwnerId']=1
                s['settings']['seed']=rng_seed
                if rng_seed % 3 == 0:
                    for k,t in s['tiles'].items():
                        t['explorers']=[1,2] if max(abs(int(k)%size-target%size),abs(int(k)//size-target//size))<=1 else [2]
                save(f'{size}-ruin-{rng_seed}',s,lambda a:a['moveType']==8 and a['src']==target)
                # Same seeded choices on water exercise rammer rewards and pushes.
                s['tiles'][str(target)]['type']=1
                u.update(type=43,passengerType=2)
                save(f'{size}-water-ruin-{rng_seed}',s,lambda a:a['moveType']==8 and a['src']==target)
            for kind in (8,48):
                s=deepcopy(base)
                c=s['tribes']['1']['cities'][0]
                target=next(i for i in c['territory'] if i!=c['idx'] and 0<i%size<size-1 and 0<i//size<size-1)
                for i in (target-1,target+1,target-size,target+size):
                    s['tiles'][str(i)]['type']=3
                s['tiles'][str(target)]['type']=1
                s['structures'].pop(str(target),None)
                s['resources'].pop(str(target),None)
                save(f'{size}-build-naval-{kind}',s,lambda a:a['moveType']==6 and a['type']==kind and a['target']==target)
            for slot in range(1,6):
                s=deepcopy(base)
                c=s['tribes']['1']['cities'][0]
                c.update(level=slot+1,rewards=[3,1,8,2][:slot-1],progress=0,production=slot+1)
                save(f'{size}-reward-{slot}',s,lambda a:a['moveType']==9 and True)
    output=ROOT/'tests/fixtures/polyfish-actions.json.gz'
    output.write_bytes(gzip.compress(json.dumps(dict(revision=ENGINE_REVISION,cases=cases),sort_keys=True,separators=(',',':')).encode(),mtime=0))
    print(dict(cases=len(cases),transitions=sum(len(c['transitions']) for c in cases)))

if __name__=='__main__': main()
