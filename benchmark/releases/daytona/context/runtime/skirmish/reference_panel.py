"""Frozen public-observation reference styles and explicitly handicapped hero blends.

Strength labels are empirical calibration results, never inferred from these names.
The reference implementation is trusted and never mounted in coder containers.
"""
import hashlib
import math
from .game import ReferenceBot


class StyleBot:
    def __init__(self, style='cavalry'):
        if style not in ('cavalry','naval','siege'):raise ValueError('unknown reference style')
        self.style=style

    def close(self):pass

    def select(self, obs):
        v=obs.visible_state;size=v['board_size'];p=obs.player
        tiles=v['tiles'];units={u['tile']:u for u in v['units']};cities={c['tile']:c for c in v['cities']}
        enemies=[t['tile'] for t in tiles if t.get('known') and t.get('owner') not in (None,0,p)]
        objectives=[c['tile'] for c in v['cities'] if c['owner']!=p]+[t['tile'] for t in tiles if not t['known']]
        if not objectives:objectives=enemies or [size*size//2]
        def dist(a,b):return max(abs(a%size-b%size),abs(a//size-b//size))
        def nearest(t):return min(dist(t,q) for q in objectives)
        techs=set(v['technology']);army=sum(u['owner']==p for u in units.values())
        def score(a):
            c=a.command;k=c['kind'];src=c['source'];target=c['target'];param=c['parameter']
            tie=int.from_bytes(hashlib.sha256(a.action_id.encode()).digest()[:4],'big')/2**32*.001
            if k==11:return -1e6
            if k==10:return -100
            if k==9:return 1000 + (12 if param in (1,3,5,7) else 0)
            if k==8:return 950
            if k==2:
                victim=units.get(target,{})
                return 200+20*(victim.get('health',10)<=5)+15*(target in cities)-victim.get('health',10)+tie
            if k==5:return 80+tie
            if k==6:
                # Favor inexpensive growth/income and ports for the naval style.
                return 45+(25 if param==8 and self.style=='naval' else 0)+(10 if param in (5,6,7) else 0)+tie
            if k==7:
                priority={'cavalry':[1,4,2,3], 'naval':[10,13,44,14], 'siege':[16,17,20,23]}[self.style]
                return (90-priority.index(param)*8 if param in priority else 10)+min(v['stars']/8,10)+tie
            if k==4:
                unit=param;preferred={'cavalry':{3:55,4:65,2:10,5:5},'naval':{2:20,3:25,5:15,9:15},'siege':{8:55,5:35,9:30,2:10}}[self.style]
                return 65+preferred.get(unit,-10)-max(0,army-len(cities)*3)*15+tie
            if k==1:
                u=units.get(src,{});gain=nearest(src)-nearest(target)
                tile=tiles[target];bonus=20 if tile.get('structure') in (1,2) and tile.get('owner')!=p else 0
                return 20+gain*15+bonus+(10 if not tile.get('known') else 0)+tie
            if k==3:
                if param==8:return -1e6  # Never discard the army as routine development.
                # Recovery and boat upgrades after combat; avoid unbounded low-value actions.
                return (25 if param in (7,8,9,16) else 5)+tie
            return tie
        return max(obs.legal_actions,key=score).action_id


class HybridReference(ReferenceBot):
    def __init__(self,path,digest,style,hero_turn_fraction):
        super().__init__(path,digest)
        self.style_bot=StyleBot(style);self.fraction=hero_turn_fraction
        if not 0<=self.fraction<=1:raise ValueError('invalid frozen hero fraction')

    def select(self,obs):
        # Entire turns use a single policy; immutable observable key, no private state.
        key=f'{obs.player}:{obs.visible_state["turn"]}'.encode()
        u=int.from_bytes(hashlib.sha256(key).digest()[:8],'big')/2**64
        return super().select(obs) if u<self.fraction else self.style_bot.select(obs)
