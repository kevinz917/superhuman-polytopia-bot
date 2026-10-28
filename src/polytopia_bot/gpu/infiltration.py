"""Corrected native Cloak infiltration; reference dispatcher is incomplete."""
import jax
import jax.numpy as jnp
from . import military
from .kernels import distances, combat, max_health
from .discovery import hash_words, hash_word


def spawn_order(s,target,*,size):
    """Historical native placement priorities with seed/coordinate tie noise.

    Isolated uniform-land ordering is executable-cross-checked. Current-build
    equivalence and resource/terrain priorities still need native replay cases.
    """
    t=s.tiles
    ids=jnp.arange(size*size)
    tech=s.players.tech[s.current_player-1]
    water=t.terrain<=2
    eligible=(t.city==target)&(s.units.owner==0)
    eligible &= ((t.terrain!=4)|tech[20]) & ((t.terrain!=1)|tech[10]) & ((t.terrain!=2)|tech[13])
    defense=((t.terrain==4)&tech[20]) | ((t.terrain==5)&tech[18]) | (water&tech[12])
    known_resource=(t.resource>0)&((t.resource!=2)|tech[6]|tech[8]|tech[9])&((t.resource!=5)|tech[20]|tech[23]|tech[24])
    noise=hash_word(s.initial_seed,hash_words(s.initial_seed,(ids%size,ids//size,s.turn))).astype(jnp.float32)*jnp.float32(.1)/jnp.float32(4294967296.)
    priority=noise+(t.structure==1)*2.-water*1.6+known_resource*.5+defense
    return jnp.argsort(jnp.where(eligible,-priority,jnp.inf),stable=True),eligible


def execute(s,source,target,rules,*,size):
    owner=s.current_player
    c=s.cities
    innate=c.production[target]+(s.tiles.capital[target]==c.owner[target])
    innate=jnp.maximum(innate+jnp.minimum(c.progress[target],0),0)
    # Retain pre-action level; consumption and damage may open spawn squares.
    count=jnp.minimum(c.level[target],5)
    s=military.remove(s,source,rules)
    enemy=(s.units.owner[target]>0)&(s.units.owner[target]!=owner)
    damage=combat(rules.unit_attack[37],rules.unit_health[37],rules.unit_health[37],
        rules.unit_defense[s.units.kind[target]],s.units.health[target],max_health(s,rules)[target],
        military.defense_bonuses(s,rules)[target])[0]
    s=s._replace(units=s.units._replace(health=s.units.health.at[target].add(jnp.where(enemy,-damage,0.))))
    dead=enemy&(s.units.health[target]<=0)
    s=jax.lax.cond(dead,lambda x:military.remove(x,target,rules),lambda x:x,s)
    s=s._replace(players=s.players._replace(kills=s.players.kills.at[owner-1].add(dead.astype(jnp.int32)),
                    stars=s.players.stars.at[owner-1].add(innate)),
        cities=s.cities._replace(infiltrated=s.cities.infiltrated.at[target].set(s.turn*2+owner),
                    riot=s.cities.riot.at[target].set(True)))
    t=s.tiles
    water=t.terrain<=2
    order,eligible=spawn_order(s,target,size=size)
    def spawn(i,x):
        at=order[i]
        def yes(x):
            x=military.summon(x,at,jnp.where(water[at],40,37),rules,size=size,costs=False)
            return x._replace(units=x.units._replace(home=x.units.home.at[at].set(0)))
        return jax.lax.cond((i<count)&eligible[at],yes,lambda x:x,x)
    return jax.lax.fori_loop(0,5,spawn,s)
