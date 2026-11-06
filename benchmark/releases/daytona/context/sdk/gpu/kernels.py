"""JIT/vmap-compatible Polyfish rule primitives, without host callbacks.

Source: functions.rs, moves/mod.rs, settings/technology.rs,
actions/units.rs::calculate_combat and actions/mod.rs::update_exploration.
"""
import jax
import jax.numpy as jnp

from .rules import Rules, FLOAT, CARRY, SCOUT, CREEP, HIDE, INDEPENDENT, SURPRISE, SPLASH, STOMP, INFILTRATE
from .state import State


def combat(attacker_attack, attacker_health, attacker_max_health,
           defender_defense, defender_health, defender_max_health, defense_bonus):
    """Reference f32 damage formula, including Rust's half-away-from-zero round.

    Inputs are nonnegative game stats; jnp.round uses ties-to-even and would
    disagree with the Rust oracle. Returned retaliation is raw formula damage;
    range/Stiff/Surprise are handled by an eventual attack transition.
    """
    aa, ah, am, dd, dh, dm, db = [jnp.asarray(x, jnp.float32) for x in (
        attacker_attack, attacker_health, attacker_max_health,
        defender_defense, defender_health, defender_max_health, defense_bonus)]
    af = aa * (ah / jnp.maximum(am, 1))
    df = dd * (dh / jnp.maximum(dm, 1)) * db
    total = af + df
    divisor = jnp.where(total > 0, total, 1)
    attack = jnp.floor(jnp.where(total > 0, (af / divisor) * aa * 4.5, 0) + jnp.float32(0.000001) + 0.5)
    defense = jnp.floor(jnp.where((dh > attack) & (total > 0), (df / divisor) * dd * 4.5, 0) + jnp.float32(0.000001) + 0.5)
    return jnp.stack((attack, defense, attack * 0.5), axis=-1)


def distances(size: int):
    idx = jnp.arange(size * size)
    return jnp.maximum(jnp.abs(idx[:, None] % size - idx[None, :] % size),
                       jnp.abs(idx[:, None] // size - idx[None, :] // size))


def neighbors(size: int):
    idx = jnp.arange(size * size)
    dx = jnp.array([-1, 0, 1, -1, 1, -1, 0, 1])
    dy = jnp.array([-1, -1, -1, 0, 0, 1, 1, 1])
    x, y = idx[:, None] % size + dx, idx[:, None] // size + dy
    valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    return jnp.clip(y * size + x, 0, size * size - 1), valid


def max_health(state: State, rules: Rules):
    return rules.unit_health[jnp.where(state.units.passenger > 0, state.units.passenger, state.units.kind)] + state.units.veteran * 5


def research_costs(state: State, rules: Rules):
    p = state.current_player - 1
    cities = jnp.sum(state.cities.owner == state.current_player)
    cost = 4 + cities * rules.tech_tier
    return jnp.where(state.players.tech[p, 22], jnp.ceil(cost.astype(jnp.float32) * 0.66).astype(jnp.int32), cost)


def research_mask(state: State, rules: Rules):
    p = state.current_player - 1
    discovered = state.players.tech[p]
    reachable = jnp.any(discovered[:, None] & rules.tech_edges, axis=0)
    pending = jnp.any((state.cities.owner == state.current_player) & (jnp.sum(state.cities.rewards, axis=-1) < state.cities.level - 1))
    active = ~pending & ~state.terminated & ~state.truncated
    return active & reachable & ~discovered & rules.tech_exists & (research_costs(state, rules) <= state.players.stars[p])


def movement_data(s, source, rules, *, size):
    t,u=s.tiles,s.units
    owner=u.owner[source]
    p=jnp.clip(owner-1,0,1)
    tech=s.players.tech[p]
    kind=u.kind[source]
    aquatic=rules.skills[kind,FLOAT]
    creep=rules.skills[kind,CREEP]
    bridge=t.structure==48
    water=(t.terrain==1)|(t.terrain==2)
    terrain_ok=((t.terrain!=1)|tech[10]) & ((t.terrain!=2)|tech[13]|tech[25]) & ((t.terrain!=4)|tech[20])
    terrain_ok |= bridge
    own_port=(t.structure==8)&(t.owner==owner)
    land_access=~water|bridge|own_port
    hidden_enemy=rules.native & u.invisible & (u.owner!=owner)
    passable=t.explored[p]&terrain_ok&(aquatic|land_access)&((u.owner==0)|(u.owner==owner)|hidden_enemy)
    nb,in_board=neighbors(size)
    peaceful=rules.native & s.players.peace[p,jnp.clip(u.owner-1,0,1)]
    enemy=(u.owner!=0)&(u.owner!=owner)&~u.invisible&~peaceful&(~rules.native|t.explored[p])
    zoc=jnp.any(enemy[nb]&in_board,axis=-1)&~(creep|rules.skills[kind,INFILTRATE]|u.invisible[source])
    friendly_road=(t.owner==0)|(t.owner==owner)|(rules.native & s.players.peace[p,jnp.clip(t.owner-1,0,1)])
    road=(t.road|(t.structure==1)|(s.cities.owner!=0)|bridge)&friendly_road
    terminal=zoc|(t.terrain==4)|((t.terrain==5)&~road&~creep)| (own_port & ~aquatic) | (rules.skills[kind,CARRY]&~water)
    budget=(rules.unit_movement[kind]+u.boosted[source])*2
    return passable,terminal,road,budget


def movement_targets(state: State, source, rules: Rules, *, size: int):
    """Complete destination set for ordinary land units under the core guard.

    Sparse min-plus relaxation replaces CPU Dijkstra. Costs are exact half-star
    movement units (integers). A unit may enter a tile with any positive budget
    left, even if that step consumes more than the remainder. Friendly units
    permit transit but not stopping. Rough terrain and enemy ZOC stop expansion.
    """
    n = size * size
    source_valid = (source >= 0) & (source < n)
    src = jnp.clip(source, 0, n - 1)
    owner = state.units.owner[src]
    p = jnp.clip(owner - 1, 0, 1)
    kind = state.units.kind[src]
    nb, in_board = neighbors(size)
    t = state.tiles
    u = state.units
    passable, terminal, road, budget = movement_data(state, src, rules, size=size)
    cost = jnp.where(road[:, None] & road[nb] & ~rules.skills[kind, CREEP], 1, 2)
    inf = jnp.int32(1_000_000)
    d = jnp.full(n, inf, jnp.int32).at[src].set(0)

    def relax(_, dist):
        # At target i, consider all predecessor neighbors j. All edges are
        # symmetric in this declared subset; terminal status is on predecessor.
        expand = (dist[nb] < budget) & ((nb == src) | ~terminal[nb])
        candidate = jnp.where(in_board & expand, dist[nb] + cost, inf)
        best = jnp.minimum(dist, jnp.min(candidate, axis=-1))
        return jnp.where(passable | (jnp.arange(n) == src), best, inf)

    # Maximum supported movement = knight 3 + boost 1; roads cost 1/2.
    d = jax.lax.fori_loop(0, 8, relax, d)
    pending = jnp.any((state.cities.owner == state.current_player) & (jnp.sum(state.cities.rewards, axis=-1) < state.cities.level - 1))
    active = source_valid & (owner == state.current_player) & ~u.moved[src] & rules.unit_supported[kind] & ~pending & ~state.terminated & ~state.truncated
    empty=(u.owner==0)|(rules.native & u.invisible & (u.owner!=owner))
    return active & (d < inf) & empty & (jnp.arange(n) != src)


def update_exploration(state: State, player, *, size: int):
    """Reference cumulative visibility, including capital corner exclusion."""
    p = player - 1
    d = distances(size)
    t, c, u = state.tiles, state.cities, state.units
    city_range = jnp.where(t.capital > 0, 2, c.border)
    from_city = jnp.any((c.owner == player)[:, None] & (d <= city_range[:, None]), axis=0)
    ids = jnp.arange(size * size)
    corners = (ids == 0) | (ids == size - 1) | (ids == size * (size - 1)) | (ids == size * size - 1)
    from_city &= ~corners | (c.owner == player)
    unit_range = jnp.where((t.terrain == 4) | (u.kind == 42) | (u.kind == 38) | (u.kind == 39), 2, 1)
    from_unit = jnp.any((u.owner == player)[:, None] & (d <= unit_range[:, None]), axis=0)
    explored = t.explored[p] | from_city | from_unit | ~state.fog
    gain = jnp.sum(explored & ~t.explored[p], dtype=jnp.int32) * 5
    return state._replace(
        tiles=t._replace(explored=t.explored.at[p].set(explored)),
        players=state.players._replace(score=state.players.score.at[p].add(gain)),
    )


def observe(state: State, player):
    """Allowlist policy view; no seed, other player's bank/tech, or save data.

    Invisible enemies appear empty. Detect only exposes the native nearby-enemy
    indicator. Crop/metal visibility follows the technology reveal requirements.
    """
    p = player - 1
    visible = state.tiles.explored[p] | ~state.fog
    hide = lambda a: jnp.where(visible, a, -1)
    tech = state.players.tech[p]
    resource_visible = ((state.tiles.resource != 2) | tech[6] | tech[8] | tech[9]) & (
        (state.tiles.resource != 5) | tech[20] | tech[23] | tech[24])
    units_visible=visible & ((state.units.owner==player)|~state.units.invisible)
    unit_view=lambda a:jnp.where(visible,jnp.where(units_visible,a,0),-1)
    # Detect exposes a boolean on each friendly unit, not the hidden location.
    n=state.units.owner.size
    size=int(n**.5)
    hidden=(state.units.owner!=0)&(state.units.owner!=player)&state.units.invisible
    detected=(state.units.owner==player)&jnp.any((distances(size)==1)&hidden[None,:],axis=1)
    return {
        "player": player, "turn": state.turn, "state_version": state.state_version,
        "stars": state.players.stars[p], "tech": tech,
        "explored": visible, "terrain": hide(state.tiles.terrain),
        "owner": hide(state.tiles.owner), "structure": hide(state.tiles.structure),
        "road": hide(state.tiles.road.astype(jnp.int32)),
        "capital": hide(state.tiles.capital),
        "structure_level": hide(state.tiles.structure_level),
        "resource": hide(jnp.where(resource_visible, state.tiles.resource, 0)),
        "unit_owner": unit_view(state.units.owner), "unit_type": unit_view(state.units.kind),
        "unit_health": unit_view(state.units.health), "detect": detected,
        "unit_veteran": unit_view(state.units.veteran.astype(jnp.int32)),
        "own_unit_moved": (state.units.owner==player)&state.units.moved,
        "own_unit_attacked": (state.units.owner==player)&state.units.attacked,
        "own_unit_kills": jnp.where(state.units.owner==player,state.units.kills,0),
        "own_unit_home": jnp.where(state.units.owner==player,state.units.home,0),
        "own_unit_passenger": jnp.where(state.units.owner==player,state.units.passenger,0),
        "own_unit_poison": (state.units.owner==player)&state.units.poison,
        "own_unit_frozen": (state.units.owner==player)&state.units.frozen,
        "own_unit_boosted": (state.units.owner==player)&state.units.boosted,
        "own_unit_invisible": (state.units.owner==player)&state.units.invisible,
        "own_city_level": jnp.where(state.cities.owner==player,state.cities.level,0),
        "own_city_progress": jnp.where(state.cities.owner==player,state.cities.progress,0),
        "own_city_rewards": jnp.where((state.cities.owner==player)[:,None],state.cities.rewards,0),
        "own_city_population": jnp.where(state.cities.owner==player,state.cities.population,0),
        "own_city_production": jnp.where(state.cities.owner==player,state.cities.production,0),
        "own_city_connected": (state.cities.owner==player)&state.cities.connected,
        "own_city_riot": (state.cities.owner==player)&state.cities.riot,
        "own_monuments": state.players.monuments[p], "own_built": state.players.built[p],
        "known_players": state.players.known[p],
        "peace": state.players.peace[p], "embassies_sent": state.players.embassy[p],
        "embassies_received": state.players.embassy[:,p], "peace_offers": state.players.peace_offer[p],
    }
