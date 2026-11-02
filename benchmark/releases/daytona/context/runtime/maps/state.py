"""Frozen-engine starting state from any map layout (variant generator or classic).

initial_state(...) repeats gpu.mapgen.reset after its layout step, so a classic
layout gives a state bitwise identical to mapgen.reset(seed) (tests/test_maps.py).
The engine sources are not edited; only the map that seeds a game changes.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from polytopia_bot.gpu import round_limit
from polytopia_bot.gpu.kernels import distances, update_exploration
from polytopia_bot.gpu.state import empty_state


class MapBatch(NamedTuple):
    """Per-lane maps for RustGames: seed [G], terrain/structure/resource [G, n], capitals [G, 2]."""
    seed: object
    terrain: object
    structure: object
    resource: object
    capitals: object


def initial_state(seed, terrain, structure, resource, caps, *, size, max_turns=None):
    """max_turns=None: the ruleset round limit for the board size (100 on 11/14, 150 on 18)."""
    max_turns = round_limit(size) if max_turns is None else max_turns
    s = empty_state(size)
    owners = jnp.array([1, 2], jnp.int32)
    area = distances(size)[caps] <= 1
    owner = jnp.max(jnp.where(area, owners[:, None], 0), axis=0)
    city = jnp.where(owner > 0, caps[jnp.clip(owner - 1, 0, 1)], -1)
    s = s._replace(seed=jnp.asarray(seed, jnp.uint32), initial_seed=jnp.asarray(seed, jnp.uint32),
        reference_max_turns=jnp.int32(max_turns),
        tiles=s.tiles._replace(terrain=terrain, structure=structure, resource=resource, owner=owner, city=city,
            capital=s.tiles.capital.at[caps].set(owners)),
        cities=s.cities._replace(owner=s.cities.owner.at[caps].set(owners), level=s.cities.level.at[caps].set(1),
            production=s.cities.production.at[caps].set(1), border=s.cities.border.at[caps].set(1),
            territory=s.cities.territory.at[caps].set(area), order=s.cities.order.at[caps].set(1)),
        units=s.units._replace(owner=s.units.owner.at[caps].set(owners), kind=s.units.kind.at[caps].set(2),
            health=s.units.health.at[caps].set(10.), home=s.units.home.at[caps].set(caps + 1),
            prev_x=s.units.prev_x.at[caps].set(caps % size), prev_y=s.units.prev_y.at[caps].set(caps // size)),
        players=s.players._replace(stars=jnp.array([5, 5], jnp.int32),
            tech=s.players.tech.at[:, 0].set(True).at[:, 15].set(True),
            score=jnp.array([390, 390], jnp.int32)))
    s = update_exploration(s, 1, size=size)
    return update_exploration(s, 2, size=size)


def from_batch_row(row, *, size, max_turns=None):
    return initial_state(row.seed, row.terrain, row.structure, row.resource, row.capitals, size=size,
                         max_turns=max_turns)


def stack(rows):
    """List of (seed, Layout-like) -> MapBatch of numpy arrays."""
    return MapBatch(np.array([s for s, _ in rows], np.int32),
                    np.stack([np.asarray(l.terrain, np.int32) for _, l in rows]),
                    np.stack([np.asarray(l.structure, np.int32) for _, l in rows]),
                    np.stack([np.asarray(l.resource, np.int32) for _, l in rows]),
                    np.stack([np.asarray(l.capitals, np.int32) for _, l in rows]))
