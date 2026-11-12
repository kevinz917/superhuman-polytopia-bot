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


TRIBE_SHIFT = 16   # engine-rs tribes.rs / engine-cu core.h: a packed capital is tile | tribe << 16 (tribe 0 = Bardur)
# the tribes the engines have (engine-rs tribes::start): increment R, the twelve regular tribes
ENGINE_TRIBES = (2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
# the special tribes (Aquarion, Elyrion, Polaris, Cymanti): work in progress (tribes::wip), packed only when
# POLYTOPIA_GEN_WIP_TRIBES lists them, exactly as the engines offer them (tribes::offered, gpu_engine.tribe_offered)
SPECIAL_TRIBES = (3, 5, 16, 17)


def _opted_in(tribe):
    import os
    import re
    return any(re.fullmatch(r"[+-]?[0-9]+", x.strip()) and int(x.strip()) == tribe
               for x in os.environ.get("POLYTOPIA_GEN_WIP_TRIBES", "").split(","))


def pack_capitals(capitals, tribes=None):
    """Capitals [..., players] for the engines: tile | tribe << 16. tribes=None leaves them plain
    (Bardur), so Bardur tables stay bit for bit what they were."""
    caps = np.asarray(capitals).astype(np.int32)
    if tribes is None:
        return caps
    tribes = np.asarray(tribes).astype(np.int32)
    bad = sorted(t for t in set(np.unique(tribes).tolist())
                 if t not in ENGINE_TRIBES and not (t in SPECIAL_TRIBES and _opted_in(t)))
    if bad:
        # engine-cu's device-resident resets cannot refuse a tribe: refuse the table here (fail closed)
        raise ValueError(f'tribes {bad} are not in the engines yet (have {ENGINE_TRIBES})')
    return caps | (tribes << TRIBE_SHIFT)


def table_capitals(z):
    """A map table's (np.load) capitals packed with its `tribes` column when it has one."""
    return pack_capitals(z['capitals'], z['tribes'] if 'tribes' in z.files else None)


def stack(rows):
    """List of (seed, Layout-like) -> MapBatch of numpy arrays."""
    return MapBatch(np.array([s for s, _ in rows], np.int32),
                    np.stack([np.asarray(l.terrain, np.int32) for _, l in rows]),
                    np.stack([np.asarray(l.structure, np.int32) for _, l in rows]),
                    np.stack([np.asarray(l.resource, np.int32) for _, l in rows]),
                    np.stack([np.asarray(l.capitals, np.int32) for _, l in rows]))
