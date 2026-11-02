"""Factored, capacity-free legal action space for device-side rollouts.

Every slot has a stable meaning; no truncation to a fixed number of legal moves.
Flattening the masks is optional. The semantic action decoder stays on device.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .core import Action, action, pending_reward
from .kernels import movement_targets, research_mask, max_health, distances
from . import military, economy, structures, capture, diplomacy


class Masks(NamedTuple):
    move: jax.Array
    attack: jax.Array
    summon: jax.Array
    build: jax.Array
    research: jax.Array
    harvest: jax.Array
    forest: jax.Array
    reward: jax.Array
    capture: jax.Array
    recover: jax.Array
    promote: jax.Array
    disband: jax.Array
    heal: jax.Array
    destroy: jax.Array
    diplomatic: jax.Array
    end: jax.Array
    resign: jax.Array


def legal_masks(s,rules,*,size):
    n=size*size
    ids=jnp.arange(n)
    u=s.units
    own=u.owner==s.current_player
    idle=own&~u.moved&~u.attacked
    hp=max_health(s,rules)
    reward=economy.reward_mask(s)
    heal=idle&(u.kind==10)&jnp.any((distances(size)==1)&(own&(u.health<hp))[None,:],axis=1)
    other=3-s.current_player
    capital=jnp.argmax((s.tiles.capital==other)&(s.cities.owner==other))
    # Five abilities, with response reject/accept using separate slots.
    abilities=jnp.array([18,19,19,20,21,22])
    targets=jnp.array([-1,0,1,capital,-1,-1])
    diplomatic=jax.vmap(lambda k,t:diplomacy.legal(s,Action(jnp.int32(3),other,t,k)))(abilities,targets)&rules.native
    masks=Masks(
        jax.vmap(lambda i:movement_targets(s,i,rules,size=size))(ids),
        jax.vmap(lambda i:military.attack_mask(s,i,rules,size=size))(ids),
        military.summon_mask(s,rules)&rules.unit_supported[None,:],
        structures.build_mask(s,rules,size=size)&(jnp.arange(128)!=37),
        research_mask(s,rules), economy.harvest_mask(s), economy.forest_masks(s), reward,
        capture.capture_mask(s,rules), idle&(u.health<hp), own&~u.veteran&(u.kills>=3)&~rules.static[u.kind],
        idle&s.players.tech[s.current_player-1,2],heal,structures.destroy_mask(s,rules),diplomatic,jnp.bool_(True),rules.native)
    active=s.core_supported&~s.terminated&~s.truncated
    pending=pending_reward(s)
    masks=jax.tree.map(lambda x:x&active&~pending,masks)
    return masks._replace(reward=reward&active,resign=rules.native&active)


def flatten(masks):
    return jnp.concatenate([a.reshape(-1) for a in masks])


def decode(s,index,*,size):
    """Decode a flattened mask index without extracting an action list to host."""
    n=size*size
    widths=(n*n,n*n,n*128,n*128,128,n,n*3,n*9,n,n,n,n,n,n,6,1,1)
    offsets=jnp.cumsum(jnp.array((0,)+widths))
    group=jnp.sum(index>=offsets[1:])
    at=index-offsets[jnp.clip(group,0,len(widths)-1)]
    other=3-s.current_player
    capital=jnp.argmax((s.tiles.capital==other)&(s.cities.owner==other))
    def ability(k):
        return lambda _:Action(jnp.int32(3),at,jnp.int32(-1),jnp.int32(k))
    return jax.lax.switch(group,(
        lambda _:Action(jnp.int32(1),at//n,at%n,jnp.int32(0)),
        lambda _:Action(jnp.int32(2),at//n,at%n,jnp.int32(0)),
        lambda _:Action(jnp.int32(4),at//128,jnp.int32(-1),at%128),
        lambda _:Action(jnp.int32(6),jnp.int32(-1),at//128,at%128),
        lambda _:Action(jnp.int32(7),jnp.int32(-1),jnp.int32(-1),at),
        lambda _:Action(jnp.int32(5),jnp.int32(-1),at,jnp.int32(0)),
        lambda _:Action(jnp.int32(3),jnp.int32(-1),at//3,at%3+1),
        lambda _:Action(jnp.int32(9),jnp.int32(-1),at//9,at%9),
        lambda _:Action(jnp.int32(8),at,jnp.int32(-1),jnp.int32(0)),
        ability(7),ability(16),ability(8),ability(9),
        lambda _:Action(jnp.int32(3),jnp.int32(-1),at,jnp.int32(4)),
        lambda _:Action(jnp.int32(3),other,jnp.array([-1,0,1,capital,-1,-1])[at],jnp.array([18,19,19,20,21,22])[at]),
        lambda _:action(10),lambda _:action(11)),None)
