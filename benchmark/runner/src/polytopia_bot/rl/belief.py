"""Determinization for leak-free turn search: sample a world consistent with the searcher's own view.

search.TurnSearcher(determinize=True) maps the true engine state through `determinize` before the lanes are
created (one sampled world per plan, shared by all lanes), and replays the chosen plan on the true game with its
guarded replay, which stops right after the first sub-action whose outcome the searcher's observation shows
differently from the sampled world (a move into fog, a ruin, a hidden unit). The searcher then re-plans
from the new true state with a fresh sample, so it only ever commits to actions planned on what it could know.

Invariant (tested): kernels.observe(determinize(s, key), me) == kernels.observe(s, me).

What the sample replaces, for the searcher `me` and the enemy `e` (seen = explored[me] | ~fog):
  - Unseen tiles get terrain / structure / resource from a fresh mapgen.layout with a random seed, with no
    owner, road or improvement. Unseen tiles that a kept enemy city rules keep their owner, but their
    improvements are also replaced by the sample (villages and ruins there are dropped).
  - Seen tiles where crop or metal is hidden by the searcher's missing tech take the sample's crop/metal if
    the sample has that tile's terrain, else nothing.
  - Enemy units on unseen tiles, and invisible enemy units, are removed. An invisible unit next to one of the
    searcher's units is kept, because the observation's `detect` flag already reveals it, and so is one on
    the searcher's own city tile, because the true legal-action mask (which the raw policy also gets) shows
    that the city cannot train.
  - Enemy cities whose tile and territory are all unseen move to random sampled village sites in the fog (the
    capital first, preferring the sample's capital sites), with territory re-claimed around the new site;
    with no free site the city is dropped. Enemy cities with some seen territory stay where they are.
  - Enemy stars = the searcher's stars minus its own income (the opponent is assumed to have banked as much as
    the searcher did; the income comes from engine-rs, the caller passes it in); enemy tech = the searcher's tech.
  - The enemy's explored set = the searcher's seen tiles plus the enemy's current vision in the sample.
  - The engine seeds (ruin rewards, explorer paths) are redrawn.

Approximations that still use hidden information (residual leaks, all small next to the ones removed): the
number and internals (level, population, rewards) of enemy cities, internals of seen enemy units (kills,
passenger, status effects, home city), the enemy's score, monuments and diplomacy flags. The prior is crude:
no enemy units are imagined in the fog, and the sampled map is not conditioned on the seen tiles beyond
copying them.

The sampling itself is array code on the State layout (gpu/state.py) and reuses the map generator and the
exploration helper of gpu/ (map generation and vision, not game rules); the game itself is engine-rs.
"""
import jax
import jax.numpy as jnp

from polytopia_bot.gpu import mapgen
from polytopia_bot.gpu.kernels import distances, update_exploration
from polytopia_bot.gpu.state import empty_state

CROP, METAL = 2, 5


def _where_rows(mask, new, old):
    return jax.tree.map(lambda a, b: jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - 1)), a, b), new, old)


def determinize(state, key, income, *, size):
    """One world sampled for the player to move (`income`: the stars it earns per turn); see the module docstring."""
    n = size * size
    d = distances(size)
    ids = jnp.arange(n)
    me = state.current_player
    p, q, enemy = me - 1, 2 - me, 3 - me
    empty = empty_state(size)
    t, u, c, pl = state.tiles, state.units, state.cities, state.players
    seen = t.explored[p] | ~state.fog
    fog = ~seen
    k_map, k_sites, k_seed = jax.random.split(key, 3)
    board = mapgen.layout(jax.random.randint(k_map, (), 0, 2 ** 31 - 1, dtype=jnp.int32), size=size)

    # Cities: keep the searcher's and every enemy city with some seen territory; relocate the rest.
    ruled = (t.city[None, :] == ids[:, None]) | c.territory
    move = (c.owner == enemy) & fog & ~jnp.any(ruled & seen[None, :], axis=1)
    kept_city = (c.owner > 0) & ~move
    kept_rule = (t.city >= 0) & kept_city[jnp.clip(t.city, 0, n - 1)] & (t.owner > 0)

    # Unseen tiles: the sampled layout, unowned, except the territory of kept cities.
    village_or_ruin = (board.structure == 1) | (board.structure == 2)
    structure = jnp.where(kept_city, t.structure, jnp.where(kept_rule & village_or_ruin, 0, board.structure))
    tiles = t._replace(
        terrain=jnp.where(fog & ~kept_city, board.terrain, t.terrain),
        structure=jnp.where(fog, structure, t.structure),
        resource=jnp.where(fog & ~kept_city, board.resource, t.resource),
        owner=jnp.where(fog & ~kept_rule, 0, t.owner),
        city=jnp.where(fog & ~kept_rule, -1, t.city),
        capital=jnp.where(fog & ~kept_city, 0, t.capital),
        road=jnp.where(fog, False, t.road),
        structure_level=jnp.where(fog & ~kept_city, 0, t.structure_level),
        structure_turn=jnp.where(fog & ~kept_city, 0, t.structure_turn))

    # Crop/metal on seen tiles the searcher's tech cannot see: redraw from the sample.
    tech = pl.tech[p]
    crop_hidden, metal_hidden = ~(tech[6] | tech[8] | tech[9]), ~(tech[20] | tech[23] | tech[24])
    shown = jnp.where((t.resource == CROP) & crop_hidden | (t.resource == METAL) & metal_hidden, 0, t.resource)
    redraw = seen & (shown == 0) & (crop_hidden | metal_hidden)
    sampled = jnp.where((board.terrain == t.terrain) & (((board.resource == CROP) & crop_hidden)
                                                         | ((board.resource == METAL) & metal_hidden)),
                        board.resource, 0)
    tiles = tiles._replace(resource=jnp.where(redraw, sampled, tiles.resource))

    # Units: drop enemies the searcher cannot see (an invisible one next to its units is flagged by detect).
    mine = u.owner == me
    detected = jnp.any((d == 1) & mine[None, :], axis=1) | (c.owner == me)
    drop = (u.owner == enemy) & (fog | (u.invisible & ~detected))
    units = _where_rows(drop, empty.units, u)
    s = state._replace(tiles=tiles, units=units, cities=_where_rows(move, empty.cities, c))

    # Relocate hidden enemy cities (capital first) to free sampled village sites in the fog.
    site = (board.structure == 1) & fog & ~jnp.any((d <= 1) & seen[None, :], axis=1)
    is_cap_site = jnp.zeros(n, bool).at[board.capitals].set(True)
    order = jnp.argsort(jnp.where(move, jnp.where(t.capital > 0, 0, 1), 2) * n + ids)

    def relocate(k, carry):
        s, free = carry
        i = order[k]

        def go(carry):
            s, free = carry
            ok = free & (s.tiles.owner == 0) & (s.units.owner == 0)
            pool = jnp.where((t.capital[i] > 0) & jnp.any(ok & is_cap_site), ok & is_cap_site, ok)
            noise = jax.random.uniform(jax.random.fold_in(k_sites, i), (n,))
            j = jnp.argmax(jnp.where(pool, noise, -1.))
            found = jnp.any(pool)

            def place(s):
                area = (d[j] <= c.border[i]) & fog & (s.tiles.owner == 0)
                area = area.at[j].set(True)
                cs = jax.tree.map(lambda cur, orig: cur.at[j].set(orig[i]), s.cities, c)
                cs = cs._replace(territory=cs.territory.at[j].set(area))
                ts = s.tiles._replace(
                    owner=jnp.where(area, enemy, s.tiles.owner), city=jnp.where(area, j, s.tiles.city),
                    structure=s.tiles.structure.at[j].set(1), terrain=s.tiles.terrain.at[j].set(t.terrain[i]),
                    resource=s.tiles.resource.at[j].set(0), capital=s.tiles.capital.at[j].set(t.capital[i]),
                    structure_level=s.tiles.structure_level.at[j].set(t.structure_level[i]))
                home = jnp.where(s.units.home == i + 1, j + 1, s.units.home)
                return s._replace(cities=cs, tiles=ts, units=s.units._replace(home=home))

            def drop_city(s):
                return s._replace(units=s.units._replace(home=jnp.where(s.units.home == i + 1, 0, s.units.home)))

            s = jax.lax.cond(found, place, drop_city, s)
            return s, free & ~(found & (d[j] <= 2))
        return jax.lax.cond(move[i], go, lambda x: x, (s, free))

    s, _ = jax.lax.fori_loop(0, n, relocate, (s, site))

    # Enemy economy and knowledge: symmetric priors from the searcher's own position.
    players = s.players._replace(stars=s.players.stars.at[q].set(jnp.maximum(pl.stars[p] - income, 0)),
                                 tech=s.players.tech.at[q].set(tech))
    score = players.score
    s = s._replace(players=players, tiles=s.tiles._replace(explored=s.tiles.explored.at[q].set(seen)))
    s = update_exploration(s, enemy, size=size)
    s = s._replace(players=s.players._replace(score=score))
    seeds = jax.random.bits(k_seed, (2,), jnp.uint32)
    return s._replace(seed=seeds[0], initial_seed=seeds[1])


def build_belief(*, size):
    return jax.jit(lambda state, key, income: determinize(state, key, income, size=size))
