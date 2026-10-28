"""Procedural 1v1 Bardur Drylands reset, entirely on device.

Layout algorithm: pinned Apache-2.0 Polyfish mapgen.rs, Drylands/Bardur branch.
Sampling explicitly follows .NET Next/NextDouble, correcting the source's
31-bit RngCore adapter. This is a versioned reference-derived distribution,
not a claim of commercial-map seed compatibility. See docs/map-generation.md.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from . import random
from .state import empty_state
from .kernels import distances, neighbors, update_exploration

# v2 adds the 18x18 (Large) board; 11x11 and 14x14 layouts are bitwise those of v1.
GENERATOR_REVISION='bardur-drylands-dotnet-v2'
RUINS={11:4,14:5,18:9}  # reference mapgen.rs ruin_count: Tiny 4, Small 5, Large 9


class Layout(NamedTuple):
    terrain: jax.Array
    structure: jax.Array
    resource: jax.Array
    capitals: jax.Array


def _choose(rng,mask):
    rng,draw=random.bounded(rng,jnp.maximum(jnp.sum(mask,dtype=jnp.int32),1))
    return rng,jnp.argmax(mask & (jnp.cumsum(mask)-1==draw)).astype(jnp.int32)


def layout(seed,*,size,diagnostics=False):
    if size not in RUINS:
        raise ValueError('Bardur procedural reset supports 11, 14 or 18')
    n=size*size
    ids=jnp.arange(n)
    x,y=ids%size,ids//size
    d=distances(size)
    rng=random.initialize(seed)
    available=jnp.ones(4,jnp.bool_)
    caps=jnp.zeros(2,jnp.int32)
    quadsize=size//2
    centers=jnp.arange(4)
    centers_x=(centers%2)*quadsize+quadsize//2
    centers_y=(centers//2)*quadsize+quadsize//2
    def capital(i,carry):
        rng,available,caps=carry
        sq=(centers_x-caps[0]%size)**2+(centers_y-caps[0]//size)**2
        maximum=jnp.max(jnp.where(available,sq,-1))
        allowed=available & ((i==0)|(sq>=maximum//2))
        rng,q=_choose(rng,allowed)
        sx=jnp.minimum((q%2)*quadsize+2,size-3)
        ex=jnp.minimum(jnp.maximum(((q%2)+1)*quadsize-2,sx+1),size-2)
        sy=jnp.minimum((q//2)*quadsize+2,size-3)
        ey=jnp.minimum(jnp.maximum(((q//2)+1)*quadsize-2,sy+1),size-2)
        rng,cx=random.bounded(rng,ex-sx)
        rng,cy=random.bounded(rng,ey-sy)
        return rng,available.at[q].set(False),caps.at[i].set((cy+sy)*size+cx+sx)
    rng,_,caps=jax.lax.fori_loop(0,2,capital,(rng,available,caps))
    land=jnp.zeros(n,jnp.bool_).at[caps].set(True)
    def grow_cond(v):return (jnp.sum(v[1])<int(n*.95))&(v[2]<n*10)
    def grow(v):
        rng,land,attempt=v
        rng,at=random.bounded(rng,n)
        return rng,land.at[at].set(True),attempt+1
    rng,land,_=jax.lax.while_loop(grow_cond,grow,(rng,land,jnp.int32(0)))
    rng_land=random.next_int(rng)[1]
    villages=jnp.zeros(n,jnp.bool_).at[caps].set(True)

    # Both tribes are Bardur, but the reference biome traversal consumes draws.
    # Retain its two active arrays and swap-removal to preserve the RNG stream.
    active=jnp.zeros((2,n),jnp.int32).at[:,0].set(caps)
    counts=jnp.ones(2,jnp.int32)
    done=villages
    def biome_round(v):
        rng,active,counts,done,_=v
        def one(i,c):
            def advance(c):
                rng,active,counts,done,changed=c
                rng,index=random.bounded(rng,counts[i])
                at=active[i,index]
                candidates=(d[at]<=1)&~done
                preferred=candidates&land
                candidates=jnp.where(jnp.any(preferred),preferred,candidates)
                def expand(c):
                    rng,active,counts,done,changed=c
                    rng,chosen=_choose(rng,candidates)
                    return rng,active.at[i,counts[i]].set(chosen),counts.at[i].add(1),done.at[chosen].set(True),jnp.bool_(True)
                def prune(c):
                    rng,active,counts,done,changed=c
                    return rng,active.at[i,index].set(active[i,counts[i]-1]),counts.at[i].add(-1),done,changed
                return jax.lax.cond(jnp.any(candidates),expand,prune,(rng,active,counts,done,changed))
            return jax.lax.cond(counts[i]>0,advance,lambda c:c,c)
        return jax.lax.fori_loop(0,2,one,(rng,active,counts,done,jnp.bool_(False)))
    rng,_,_,_,_=jax.lax.while_loop(lambda v:v[4],biome_round,(rng,active,counts,done,jnp.bool_(True)))
    rng_biomes=random.next_int(rng)[1]
    nb,in_board=neighbors(size)
    orthogonal=jnp.array([False,True,False,True,True,False,True,False])
    coast=~land & jnp.any(land[nb]&in_board&orthogonal[None,:],axis=-1)
    terrain=jnp.where(land,3,jnp.where(coast,1,2)).astype(jnp.int32)
    def biome(i,v):
        def choose(v):
            rng,terrain=v
            rng,value=random.unit_float(rng)
            kind=jnp.where(value<jnp.float32(.14),4,jnp.where(value<jnp.float32(.14)+jnp.float32(.38)*jnp.float32(1.5),5,3))
            return rng,terrain.at[i].set(kind)
        return jax.lax.cond(land[i]&~villages[i],choose,lambda v:v,v)
    rng,terrain=jax.lax.fori_loop(0,n,biome,(rng,terrain))
    rng_terrain=random.next_int(rng)[1]
    edge=jnp.minimum(jnp.minimum(x,size-1-x),jnp.minimum(y,size-1-y))
    def village_candidates(villages):
        return land&~villages&(terrain!=4)&(edge>=2)&(edge!=3)&~jnp.any((d<3)&villages[None,:],axis=1)
    def village(v):
        rng,villages,terrain=v
        rng,at=_choose(rng,village_candidates(villages))
        return rng,villages.at[at].set(True),terrain.at[at].set(3)
    rng,villages,terrain=jax.lax.while_loop(lambda v:jnp.any(village_candidates(v[1])),village,(rng,villages,terrain))
    # Above values below 100 are structures; 100 + resource ID are resources.
    above=jnp.where(villages,1,0).astype(jnp.int32)
    def guaranteed(i,v):
        def animal(_,v):
            rng,terrain,above=v
            mask=(d[caps[i]]<=1)&(ids!=caps[i])&(above==0)&(terrain!=2)
            def add(v):
                rng,terrain,above=v
                rng,at=_choose(rng,mask)
                return rng,terrain.at[at].set(5),above.at[at].set(101)
            return jax.lax.cond(jnp.any(mask),add,lambda v:v,v)
        return jax.lax.fori_loop(0,2,animal,v)
    rng,terrain,above=jax.lax.fori_loop(0,2,guaranteed,(rng,terrain,above))

    def resources(city,v):
        rng,above=v
        count=jnp.sum((d[city]<=1)&(above==101),dtype=jnp.int32)
        def radius(rad,v):
            def tile(at,v):
                rng,above,count=v
                kind=terrain[at]
                eligible=(d[city,at]<=rad)&(above[at]==0)&(kind!=2)
                def add(v):
                    rng,above,count=v
                    rng,value=random.unit_float(rng)
                    probability=jnp.where(kind==3,jnp.where(rad==1,.18,.06),
                        jnp.where(kind==5,jnp.where((rad==1)&(count>=3),0,jnp.where(rad==1,.38,.12)),
                        jnp.where(kind==4,jnp.where(rad==1,.11,.03),.5)))
                    resource=jnp.where(kind==3,106,jnp.where(kind==5,101,jnp.where(kind==4,105,103)))
                    added=value<probability
                    return rng,above.at[at].set(jnp.where(added,resource,0)),count+(added&(kind==5)&(rad==1))
                return jax.lax.cond(eligible,add,lambda v:v,(rng,above,count))
            return jax.lax.fori_loop(0,n,tile,v)
        rng,above,_=jax.lax.fori_loop(1,3,radius,(rng,above,count))
        return rng,above
    rng,above=jax.lax.fori_loop(0,n,lambda i,v:jax.lax.cond(villages[i],lambda v:resources(i,v),lambda v:v,v),(rng,above))

    resource_value=jnp.where((above==101)|(above==103)|(above==106),2,jnp.where(above==105,3,0))+(terrain==5)
    scores=jnp.sum((d[caps]<=2)*resource_value[None,:],axis=-1,dtype=jnp.int32)
    best=jnp.max(scores)
    def fairness(i,v):
        def improve(v):
            rng,above=v
            mask=d[caps[i]]<=2
            order=jnp.argsort(jnp.where(mask,ids,n))
            count=jnp.sum(mask,dtype=jnp.int32)
            def swap(_,v):
                rng,order=v
                rng,a=random.bounded(rng,count)
                rng,b=random.bounded(rng,count)
                return rng,order.at[a].set(order[b]).at[b].set(order[a])
            rng,order=jax.lax.fori_loop(0,count,swap,(rng,order))
            def inject(v):
                at,above,score=v
                idx=order[at]
                add=(above[idx]==0)&(terrain[idx]==5)
                return at+1,above.at[idx].set(jnp.where(add,101,above[idx])),score+2*add
            _,above,_=jax.lax.while_loop(lambda v:(v[0]<count)&(v[2]+4<best),inject,(jnp.int32(0),above,scores[i]))
            return rng,above
        return jax.lax.cond(scores[i]+4<best,improve,lambda v:v,v)
    rng,above=jax.lax.fori_loop(0,2,fairness,(rng,above))
    def place_ruin(v):
        rng,above,placed,attempt=v
        rng,at=random.bounded(rng,n)
        nearby=d[at]<=1
        valid=(above[at]==0)&(terrain[at]>2)&~jnp.any(nearby&((above==2)|villages))
        return rng,above.at[at].set(jnp.where(valid,2,above[at])),placed+valid,attempt+1
    rng,above,_,_=jax.lax.while_loop(lambda v:(v[2]<RUINS[size])&(v[3]<2000),place_ruin,(rng,above,jnp.int32(0),jnp.int32(0)))
    def place_starfish(v):
        rng,above,placed,attempt=v
        rng,at=random.bounded(rng,n)
        valid=(terrain[at]<=2)&(above[at]==0)&~jnp.any((d[at]<=1)&((above==108)|(above==47)|(above==1)))
        return rng,above.at[at].set(jnp.where(valid,108,above[at])),placed+valid,attempt+1
    _,above,_,_=jax.lax.while_loop(lambda v:(v[2]<n//25)&(v[3]<1000),place_starfish,(rng,above,jnp.int32(0),jnp.int32(0)))
    above=above.at[jnp.array([0,size-1,n-size,n-1])].set(47)
    if diagnostics:
        return jnp.stack([rng_land,rng_biomes,rng_terrain])
    return Layout(terrain,jnp.where(above<100,above,0),jnp.where(above>=100,above-100,0),caps)


def reset(seed,*,size,max_turns=None):
    """Generate a fresh native-candidate game without a host initial-state pool.
    max_turns=None uses the ruleset round limit for the board size."""
    from . import round_limit
    max_turns=round_limit(size) if max_turns is None else max_turns
    board=layout(seed,size=size)
    s=empty_state(size)
    n=size*size
    ids=jnp.arange(n)
    caps=board.capitals
    owners=jnp.array([1,2],jnp.int32)
    area=distances(size)[caps]<=1
    owner=jnp.max(jnp.where(area,owners[:,None],0),axis=0)
    city=jnp.where(owner>0,caps[jnp.clip(owner-1,0,1)],-1)
    s=s._replace(seed=jnp.asarray(seed,jnp.uint32),initial_seed=jnp.asarray(seed,jnp.uint32),reference_max_turns=jnp.int32(max_turns),
        tiles=s.tiles._replace(terrain=board.terrain,structure=board.structure,resource=board.resource,owner=owner,city=city,
            capital=s.tiles.capital.at[caps].set(owners)),
        cities=s.cities._replace(owner=s.cities.owner.at[caps].set(owners),level=s.cities.level.at[caps].set(1),
            production=s.cities.production.at[caps].set(1),border=s.cities.border.at[caps].set(1),
            territory=s.cities.territory.at[caps].set(area),order=s.cities.order.at[caps].set(1)),
        units=s.units._replace(owner=s.units.owner.at[caps].set(owners),kind=s.units.kind.at[caps].set(2),
            health=s.units.health.at[caps].set(10.),home=s.units.home.at[caps].set(caps+1),
            prev_x=s.units.prev_x.at[caps].set(caps%size),prev_y=s.units.prev_y.at[caps].set(caps//size)),
        players=s.players._replace(stars=jnp.array([5,5],jnp.int32),tech=s.players.tech.at[:,0].set(True).at[:,15].set(True),
            score=jnp.array([390,390],jnp.int32)))  # city 100 + territory 180 + Hunting 100 + Warrior 10
    s=update_exploration(s,1,size=size)
    return update_exploration(s,2,size=size)
