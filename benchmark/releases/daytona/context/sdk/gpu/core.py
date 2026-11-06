"""Device transition dispatcher for the frozen Bardur research ruleset.

The action ABI uses Polyfish MoveType IDs. Unsupported actions return status=2
without changing any state. No CPU oracle or fallback appears in this module.
The native profile follows docs/ruleset-v1.md; reference parity is separate.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .kernels import distances, max_health, movement_targets, research_costs, research_mask, update_exploration
from .rules import Rules, INFILTRATE
from .state import State
from . import economy, military, structures, capture, diplomacy, infiltration
from .pathfinding import path_reveal, path_details, reveal_path

OK, INVALID_ACTION, UNSUPPORTED_RULE, STALE_VERSION, EPISODE_DONE = range(5)


class Action(NamedTuple):
    kind: jax.Array
    source: jax.Array
    target: jax.Array
    parameter: jax.Array


def action(kind: int, *, source=-1, target=-1, parameter=0) -> Action:
    return Action(*(jnp.int32(v) for v in (kind, source, target, parameter)))


class StepResult(NamedTuple):
    state: State
    status: jax.Array


def pending_reward(state):
    c = state.cities
    return jnp.any((c.owner == state.current_player) & (jnp.sum(c.rewards, axis=-1) < c.level - 1))


def _meet(state):
    p = state.current_player - 1
    other = 1 - p
    sees = jnp.any(state.tiles.explored[p] & (state.units.owner == other + 1) & ~state.units.invisible)
    new = state.fog & sees & ~state.players.known[p, other]
    score = state.players.score[other]
    gift = 1 + 2 * jnp.where(score > 0, jnp.minimum((score - 1) // 1000 + 1, 5), 0)
    return state._replace(players=state.players._replace(
        stars=state.players.stars.at[p].add(jnp.where(new, gift, 0)),
        known=state.players.known.at[p, other].set(state.players.known[p, other] | new),
    ))


def _research(state, a, rules):
    p = state.current_player - 1
    k = jnp.clip(a.parameter, 0, 127)
    cost = research_costs(state, rules)[k]
    return _meet(state._replace(players=state.players._replace(
        stars=state.players.stars.at[p].add(-cost),
        score=state.players.score.at[p].add(rules.tech_tier[k] * 100),
        tech=state.players.tech.at[p, k].set(True),
        tech_turn=state.players.tech_turn.at[p, k].set(state.turn),
    )))


def _recover(state, a, rules):
    src = jnp.clip(a.source, 0, state.units.owner.size - 1)
    u = state.units
    territory_owner=state.tiles.owner[src]
    friendly=(territory_owner==state.current_player)|(rules.native & (territory_owner>0) & state.players.peace[state.current_player-1,jnp.clip(territory_owner-1,0,1)])
    amount = jnp.where(friendly, 4, 2)
    hp = jnp.where(u.poison[src], u.health[src], jnp.minimum(u.health[src] + amount, max_health(state, rules)[src]))
    return _meet(state._replace(units=u._replace(
        health=u.health.at[src].set(hp), poison=u.poison.at[src].set(False),
        moved=u.moved.at[src].set(True), attacked=u.attacked.at[src].set(True),
    )))


def _move_adjacent(state, a, rules, size):
    mask,parent,rank,found=path_details(state,a.source,a.target,rules,size=size)
    hidden=mask & state.units.invisible & (state.units.owner!=state.current_player)
    encounter=rules.native & found & jnp.any(hidden)
    contact=jnp.argmax(jnp.where(hidden,rank,-1))
    # A path may cross friendly units. Stop at the last empty tile before a
    # hidden contact, never overwrite a friendly unit occupying its parent.
    can_stop=mask & (rank>rank[contact]) & ((state.units.owner==0)|(jnp.arange(size*size)==a.source))
    before_contact=jnp.argmin(jnp.where(can_stop,rank,jnp.iinfo(jnp.int32).max))
    stop=jnp.where(encounter,before_contact,a.target)
    traveled=jnp.where(encounter,mask&(rank>rank[contact]),mask)
    reveal=reveal_path(state,a.source,traveled,rules,size=size)
    state=jax.lax.cond(encounter,lambda s:s._replace(units=s.units._replace(invisible=s.units.invisible.at[contact].set(False))),lambda s:s,state)
    state=jax.lax.cond(stop!=a.source,lambda s:military.relocate(s,a.source,stop,rules,size=size,reveal=reveal),lambda s:s,state)
    return _meet(state)


def _end_turn(state, rules, size):
    u, p, c, t = state.units, state.players, state.cities, state.tiles
    owner, old = state.current_player, state.current_player - 1
    idle = (u.owner == owner) & ~u.moved & ~u.attacked
    friendly=(t.owner==owner)|(rules.native & (t.owner>0) & p.peace[old,jnp.clip(t.owner-1,0,1)])
    amount = jnp.where(friendly, 4, 2)
    hp = jnp.where(idle & ~u.poison, jnp.minimum(u.health + amount, max_health(state, rules)), u.health)
    dead = (p.killed_turn > 0) | (p.resigned_turn > 0)
    new_owner = jnp.where(dead[2 - owner], owner, 3 - owner)
    turn = state.turn + (new_owner == 1)
    pacifist = jnp.where(p.attacked[old], 0, p.pacifist[old] + p.tech[old, 21])
    out = state._replace(
        units=u._replace(health=hp, poison=u.poison & ~idle),
        players=p._replace(pacifist=p.pacifist.at[old].set(pacifist), attacked=p.attacked.at[old].set(False)),
        current_player=new_owner, turn=turn,
    )
    # Polyfish flags its safety turn limit as gameOver. Here it is a truncation,
    # not a native victory. Like the source, skip new-turn income after the cap.
    native = jnp.any(dead)
    limit = turn > state.reference_max_turns

    def start(s):
        s = jax.lax.cond(rules.native, lambda s: diplomacy.start_turn(s, new_owner), lambda s:s, s)
        s = _meet(update_exploration(s, new_owner, size=size))
        s = structures.start_turn(s, new_owner,rules)
        income = structures.income(s, new_owner, rules, size=size)
        income = jnp.where((turn > 0) & (income > 0), income, 0)
        new_units = u.owner == new_owner
        return s._replace(
            cities=s.cities._replace(riot=s.cities.riot & (s.cities.owner != new_owner)),
            players=s.players._replace(stars=s.players.stars.at[new_owner - 1].add(income)),
            units=s.units._replace(moved=jnp.where(new_units, u.frozen, s.units.moved),
                                    attacked=jnp.where(new_units, u.frozen, s.units.attacked),
                                    attacks_performed=jnp.where(new_units & ~u.frozen, 0, s.units.attacks_performed),
                                    frozen=u.frozen & ~new_units),
        )
    return jax.lax.cond(limit | native, lambda s: s._replace(truncated=limit & ~native, terminated=native), start, out)


def _resign(s):
    p=s.current_player-1
    return s._replace(terminated=jnp.bool_(True),players=s.players._replace(
        resigned_turn=s.players.resigned_turn.at[p].set(jnp.maximum(s.turn,1)),
        killer=s.players.killer.at[p].set(3-s.current_player)))


def terminal_rewards(s):
    """Seat-indexed native outcome; truncation never invents a draw label."""
    dead=(s.players.killed_turn>0)|(s.players.resigned_turn>0)
    winner=(~dead)&dead[::-1]
    return jnp.where(s.terminated,jnp.where(winner,1.,jnp.where(dead,-1.,0.)),0.)


def step(state: State, a: Action, expected_version, rules: Rules, *, size: int, max_actions=10_000):
    """Apply one atomic action; every rejection is transactionally inert."""
    n = size * size
    src, dst = jnp.clip(a.source, 0, n - 1), jnp.clip(a.target, 0, n - 1)
    param = jnp.clip(a.parameter, 0, 127)
    u = state.units
    source_valid = (a.source >= 0) & (a.source < n)
    target_valid = (a.target >= 0) & (a.target < n)
    param_valid = (a.parameter >= 0) & (a.parameter < 128)
    own = source_valid & (u.owner[src] == state.current_player)
    idle = own & ~u.moved[src] & ~u.attacked[src]
    research = a.kind == 7
    recover = (a.kind == 3) & (a.parameter == 7)
    move = a.kind == 1
    end = a.kind == 10
    attack = a.kind == 2
    summon = a.kind == 4
    harvest = a.kind == 5
    forest = (a.kind == 3) & (a.parameter >= 1) & (a.parameter <= 3)
    promote = (a.kind == 3) & (a.parameter == 16)
    disband = (a.kind == 3) & (a.parameter == 8)
    reward = a.kind == 9
    build = a.kind == 6
    destroy = (a.kind == 3) & (a.parameter == 4)
    cap = a.kind == 8
    heal = (a.kind == 3) & (a.parameter == 9)
    diplomatic = (a.kind == 3) & (a.parameter >= 18) & (a.parameter <= 22) & rules.native
    resign = (a.kind == 11) & rules.native
    supported_kind = research | recover | move | end | attack | summon | harvest | forest | promote | disband | reward | build | destroy | cap | heal | diplomatic | resign
    pending = pending_reward(state)
    reward_implemented = (a.parameter == 6) | (a.parameter == 4) | (a.parameter == 1) | (a.parameter == 2) | (a.parameter == 3) | (a.parameter == 5) | (a.parameter == 7) | (a.parameter == 8)
    supported = state.core_supported & supported_kind & (~pending | reward | resign) & (~reward | reward_implemented)
    supported &= ~summon | rules.unit_supported[param]
    supported &= ~build | (param != 37)
    valid = (end | resign
        | (research & param_valid & research_mask(state, rules)[param])
        | (recover & idle & (u.health[src] < max_health(state, rules)[src]))
        | (move & source_valid & target_valid & movement_targets(state, src, rules, size=size)[dst])
        | (attack & source_valid & target_valid & military.attack_mask(state, src, rules, size=size)[dst])
        | (summon & source_valid & param_valid & military.summon_mask(state, rules)[src, param])
        | (harvest & target_valid & economy.harvest_mask(state)[dst])
        | (forest & target_valid & economy.forest_masks(state)[dst, jnp.clip(a.parameter - 1, 0, 2)])
        | (promote & own & ~u.veteran[src] & (u.kills[src] >= 3) & ~rules.static[u.kind[src]])
        | (disband & idle & state.players.tech[state.current_player - 1, 2])
        | (build & target_valid & param_valid & structures.build_mask(state, rules, size=size)[dst, param])
        | (destroy & target_valid & structures.destroy_mask(state,rules)[dst])
        | (heal & idle & (u.kind[src] == 10) & jnp.any((distances(size)[src] == 1) & (u.owner == state.current_player) & (u.health < max_health(state, rules))))
        | (cap & source_valid & capture.capture_mask(state, rules)[src])
        | (diplomatic & diplomacy.legal(state, a))
        | (reward & target_valid & (a.parameter >= 0) & (a.parameter < 9) & economy.reward_mask(state)[dst, jnp.clip(a.parameter, 0, 8)]))
    status = jnp.where(~supported, UNSUPPORTED_RULE, jnp.where(valid, OK, INVALID_ACTION))
    status = jnp.where(expected_version != state.state_version, STALE_VERSION, status)
    status = jnp.where(state.terminated | state.truncated, EPISODE_DONE, status)

    def apply(s):
        s = jax.lax.cond(rules.native,lambda x: structures.remember_monuments(x,rules,size=size),lambda x:x,s)
        kinds = jnp.stack((research, recover, move, end, attack, summon, harvest, forest, promote, disband, reward, build, destroy, cap, heal, diplomatic, resign))
        branch = jnp.argmax(kinds)
        new = jax.lax.switch(branch, (
            lambda x: _research(x, a, rules), lambda x: _recover(x, a, rules),
            lambda x: _move_adjacent(x, a, rules, size), lambda x: _end_turn(x, rules, size),
            lambda x: jax.lax.cond(rules.native & rules.skills[x.units.kind[src],INFILTRATE],
                lambda x: infiltration.execute(x,src,dst,rules,size=size),
                lambda x: jax.lax.cond(rules.native & (x.units.kind[src] == 10),
                    lambda x: military.convert(x, src, dst, rules,size=size), lambda x: military.attack(x, src, dst, rules, size=size), x), x),
            lambda x: jax.lax.cond(x.units.kind[src] == 43, lambda x: military.upgrade(x, src, param, rules, size=size), lambda x: military.summon(x, src, param, rules, size=size), x),
            lambda x: economy.harvest(x, dst), lambda x: economy.forest(x, dst, a.parameter),
            lambda x: military.promote(x, src, rules), lambda x: military.disband(x, src, rules),
            lambda x: economy.reward(x, dst, a.parameter, rules, size=size),
            lambda x: structures.build(x, dst, param, rules, size=size),
            lambda x: structures.destroy(x, dst, rules, size=size),
            lambda x: jax.lax.cond(x.tiles.structure[src] == 1,
                lambda x: capture.city(x, src, rules, size=size), lambda x: jax.lax.cond(x.tiles.structure[src] == 2,
                    lambda x: capture.ruin(x, src, rules, size=size), lambda x: capture.starfish(x, src), x), x),
            lambda x: military.heal_others(x, src, rules, size=size),
            lambda x: diplomacy.execute(x, a, rules, size=size), _resign), s)
        new = jax.lax.cond(rules.native & attack, diplomacy.hostile, lambda x:x, new)
        new = _meet(new)
        new = jax.lax.cond(rules.native, lambda x: diplomacy.capital_vision(x, size=size), lambda x:x, new)
        new = jax.lax.cond(rules.native, lambda x: structures.remember_monuments(x,rules,size=size),lambda x:x,new)
        count = new.actions + 1
        return new._replace(state_version=new.state_version + 1, actions=count,
                            truncated=(new.truncated | (count >= max_actions)) & ~new.terminated)
    return StepResult(jax.lax.cond(status == OK, apply, lambda s: s, state), status)
