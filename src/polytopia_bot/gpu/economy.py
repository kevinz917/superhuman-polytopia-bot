"""Device-side population, resource and ordinary economic actions.

Semantics follow the pinned Rust actions, including its incremental score and
production accounting. These functions are called only after legality checks.
"""
import jax
import jax.numpy as jnp
from .kernels import distances


def add_population(s, index, amount):
    """Vectorized over all cities, including negative population/level changes."""
    n = s.cities.owner.size
    amount = jnp.where((jnp.arange(n) == index) & (s.cities.owner == s.current_player), amount, 0)
    return add_population_vector(s, amount)


def add_population_vector(s, amount, owner=None, *, native=False):
    owner=s.current_player if owner is None else owner
    c, t = s.cities, s.tiles
    active = (c.owner == owner) & (amount != 0)
    amount = jnp.where(active, amount, 0)
    progress = c.progress + amount
    def up_cond(v):
        level, progress, production, structure_level = v
        return jnp.any(active & (progress >= level + 1))
    def up(v):
        level, progress, production, sl = v
        m = active & (progress >= level + 1)
        return level + m, progress - jnp.where(m, level + 1, 0), production + m, sl + (m & (t.structure > 0))
    values = jax.lax.while_loop(up_cond, up, (c.level, progress, c.production, t.structure_level))
    def down_cond(v):
        level, progress, production, sl = v
        return ~jnp.asarray(native) & jnp.any(active & (progress < 0) & (level > 1))
    def down(v):
        level, progress, production, sl = v
        m = active & (progress < 0) & (level > 1)
        return (level - m, progress + jnp.where(m, level, 0),
                jnp.where(m, jnp.maximum(production - 1, 1), production),
                jnp.where(m & (t.structure > 0), jnp.maximum(sl - 1, 1), sl))
    level, progress, production, sl = jax.lax.while_loop(down_cond, down, values)
    score = jnp.sum(amount * 5 + (level - c.level) * 50, dtype=jnp.int32)
    return s._replace(cities=c._replace(population=c.population + amount, progress=progress, level=level, production=production),
                      tiles=t._replace(structure_level=sl),
                      players=s.players._replace(score=s.players.score.at[owner - 1].add(score)))


def in_territory(s):
    return jnp.any(s.cities.territory & (s.cities.owner == s.current_player)[:, None], axis=0)


def economic_tiles(s):
    return in_territory(s) & ((s.units.owner == 0) | (s.units.owner == s.current_player) | s.units.invisible)


def harvest_mask(s):
    r = s.tiles.resource
    tech = s.players.tech[s.current_player - 1]
    unlocked = ((r == 1) & tech[15]) | ((r == 3) & tech[10]) | ((r == 6) & tech[6])
    return economic_tiles(s) & (s.tiles.structure == 0) & unlocked & (s.players.stars[s.current_player - 1] >= 2)


def harvest(s, target):
    city = jnp.clip(s.tiles.city[target], 0, s.cities.owner.size - 1)
    s = s._replace(tiles=s.tiles._replace(resource=s.tiles.resource.at[target].set(0)),
                   players=s.players._replace(stars=s.players.stars.at[s.current_player - 1].add(-2)))
    return add_population(s, city, 1)


def forest_masks(s):
    tech, stars = s.players.tech[s.current_player - 1], s.players.stars[s.current_player - 1]
    free = economic_tiles(s) & ((s.tiles.structure == 0) | (s.tiles.structure == 3))
    forest, field = s.tiles.terrain == 5, s.tiles.terrain == 3
    return jnp.stack((free & forest & tech[9] & (stars >= 3), free & forest & tech[16],
                      free & field & tech[19] & (stars >= 5)), axis=-1)


def forest(s, target, ability):
    t = s.tiles
    terrain = jnp.where(ability == 3, 5, 3)
    resource = jnp.where(ability == 1, 2, jnp.where(ability == 2, 0, t.resource[target]))
    # v115: clear yields one star, burn costs three, grow costs five.
    stars = jnp.where(ability == 1, -3, jnp.where(ability == 2, 1, -5))
    return s._replace(tiles=t._replace(terrain=t.terrain.at[target].set(terrain), resource=t.resource.at[target].set(resource)),
                      players=s.players._replace(stars=s.players.stars.at[s.current_player - 1].add(stars)))


def discover(s, reveal, player=None):
    owner = s.current_player if player is None else player
    p = owner - 1
    new = reveal & ~s.tiles.explored[p]
    s = s._replace(tiles=s.tiles._replace(explored=s.tiles.explored.at[p].set(s.tiles.explored[p] | reveal)),
                   players=s.players._replace(score=s.players.score.at[p].add(jnp.sum(new, dtype=jnp.int32) * 5)))
    # get_capital_city picks the first capital in city insertion order.
    c = s.cities
    capital = (c.owner == owner) & (s.tiles.capital == owner)
    own = c.owner == owner
    preferred = jnp.where(jnp.any(capital), capital, own)
    index = jnp.argmin(jnp.where(preferred, c.order, jnp.iinfo(jnp.int32).max))
    count = jnp.sum(new & (s.tiles.structure == 47), dtype=jnp.int32)
    return add_population(s, index, jnp.where(jnp.any(own), count, 0))


def reward_mask(s):
    c = s.cities
    count = jnp.sum(c.rewards, axis=-1)
    pending = (c.owner == s.current_player) & (count < c.level - 1)
    slot = count + 1
    kind = jnp.arange(9)
    allowed = jnp.where((slot == 1)[:, None], (kind == 3) | (kind == 4),
              jnp.where((slot == 2)[:, None], (kind == 1) | (kind == 7),
              jnp.where((slot == 3)[:, None], (kind == 5) | (kind == 8), (kind == 2) | (kind == 6))))
    return pending[:, None] & allowed


def claim(s, city, mask, rules=None):
    t, c, p = s.tiles, s.cities, s.players
    # Structure score is added by the building port; initial structures have zero.
    changed = mask & (t.owner != s.current_player)
    if rules is None:
        tile_value = jnp.full(t.owner.shape, 20, jnp.int32)
    else:
        from .structures import scores
        tile_value = 20 + scores(s, rules)
    gain = jnp.sum(jnp.where(changed, tile_value, 0), dtype=jnp.int32)
    loss = jnp.sum(jnp.where(changed & (t.owner == 3 - s.current_player), tile_value, 0), dtype=jnp.int32)
    return s._replace(tiles=t._replace(owner=jnp.where(mask, s.current_player, t.owner), city=jnp.where(mask, city, t.city)),
                      cities=c._replace(territory=c.territory.at[city].set(c.territory[city] | mask)),
                      players=p._replace(score=p.score.at[s.current_player - 1].add(gain).at[2 - s.current_player].add(-loss)))


def reward(s, target, kind, rules, *, size):
    c = s.cities
    c = c._replace(rewards=c.rewards.at[target, kind].add(1))
    s = s._replace(cities=c)
    def production(x):
        return x._replace(cities=x.cities._replace(production=x.cities.production.at[target].add(1)),
                          players=x.players._replace(score=x.players.score.at[x.current_player - 1].add(jnp.where(kind == 2, 250, 0))))
    def stars(x):
        return x._replace(players=x.players._replace(stars=x.players.stars.at[x.current_player - 1].add(5)))
    def borders(x):
        # get_adjacent_indices excludes the center.
        area = (distances(size)[target] <= 2) & (jnp.arange(size * size) != target)
        x = x._replace(cities=x.cities._replace(border=x.cities.border.at[target].add(1)))
        return discover(claim(x, target, area & (x.tiles.owner == 0), rules), area)
    def explore(x):
        from .discovery import explorer
        return discover(x, explorer(x, target, size=size))
    def giant(x):
        from .military import superunit
        return superunit(x, target, rules, size=size)
    branch = jnp.where((kind == 2) | (kind == 3), 1, jnp.where(kind == 7, 2, jnp.where(kind == 8, 3, jnp.where(kind == 5, 4, jnp.where(kind == 4, 5, jnp.where(kind == 6, 6, 0))))))
    return jax.lax.switch(branch, (lambda x: x, production, stars, lambda x: add_population(x, target, 3), borders, explore, giant), s)
