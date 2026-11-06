"""Fixed-shape PyTrees. A batch is tree.map(stack, states), with no Python games.

Unit/city fields are indexed by tile, avoiding arbitrary entity capacities:
there is at most one unit and one city per tile in the supported rules.
Player indices are 0/1; owner values retain Polyfish's 0=neutral, 1/2=players.
This is a port of the rule-relevant core, not a lossless Polyfish save format.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class Tiles(NamedTuple):
    terrain: Array
    owner: Array
    capital: Array
    city: Array  # ruling city tile, -1 if absent
    road: Array
    structure: Array
    resource: Array
    explored: Array  # [2, tiles], cumulative exploration as in the reference
    structure_level: Array
    structure_turn: Array


class Units(NamedTuple):
    owner: Array
    kind: Array
    health: Array
    veteran: Array
    moved: Array
    attacked: Array
    kills: Array
    poison: Array
    frozen: Array
    boosted: Array
    home: Array  # index + 1; zero means independent/no home
    created: Array
    converted: Array
    attacks_performed: Array
    prev_x: Array
    prev_y: Array
    last_attack: Array  # index + 1; zero means absent
    passenger: Array
    invisible: Array
    direction: Array  # facing, GridDirection 0..7 (SW, W, NW, N, NE, E, SE, S); TryPushUnit starts from it


class Cities(NamedTuple):
    owner: Array
    level: Array
    population: Array
    progress: Array
    production: Array
    border: Array
    rewards: Array  # [tiles, 9] counts (parks can repeat)
    territory: Array  # reference membership, not inferred from tile ownership
    connected: Array
    order: Array  # stable insertion ordering, used for capital fallback
    infiltrated: Array  # action-turn serial + 1; prevents repeat infiltration
    riot: Array  # blocks the next income collection, then clears


class Players(NamedTuple):
    stars: Array
    score: Array
    tech: Array  # [2, 128]
    tech_turn: Array
    known: Array  # [2, 2]
    kills: Array
    casualties: Array
    pacifist: Array
    attacked: Array

    built: Array  # [2, 128] ever-built unique improvements
    killed_turn: Array
    resigned_turn: Array
    killer: Array
    peace: Array  # symmetric [2, 2]
    embassy: Array  # sender -> capital owner, separate from tile structures
    peace_offer: Array  # recipient -> sender
    truce_breaker: Array  # [p, q] != 0: p's relation to q is in cool-down after a broken peace (breaker's id)
    hostile_until: Array  # [attacker, victim] action-turn serial of the last attack (engine-rs: per pair; this JAX
                          # engine writes the whole attacker row; engine-rs also imports an older [2] per attacker)
    monuments: Array  # [2, 128] earned tasks persist after their condition ends
    peace_broken: Array  # [p, q] turn p's peace with q was last broken (engine-rs: ends each side's cool-down)


class History(NamedTuple):
    """Per seat, what it remembers of its own past views (engine-rs history.rs; observation version 3). Carried
    through export/import so a loaded position keeps its history; None where a State was built without one (the
    engine then starts it empty). The retired JAX engine never updates it."""
    seen_unit: Array  # [np, 3, n] a visible enemy unit stood here at the end of the seat's turn k + 1 turns ago
    prev_code: Array  # [np, n] structure * 64 + level * 2 + road of the tiles seen at the end of the seat's last turn, -1 unseen
    built: Array  # [np, n] an enemy structure/road/level appeared or changed here since the seat's last turn
    trained: Array  # [np, n] a visible enemy unit stands on an enemy city tile where none stood at the seat's last turn
    enemy_tech: Array  # [np, 128] techs the enemy must have, from what the seat has seen


class State(NamedTuple):
    tiles: Tiles
    units: Units
    cities: Cities
    players: Players
    current_player: Array  # 1/2
    turn: Array
    version: Array  # rules version
    state_version: Array  # action revision, never a hidden-state hash
    actions: Array
    terminated: Array
    truncated: Array
    fog: Array
    reference_max_turns: Array
    # Import guard for fields not represented yet. Unsupported imports remain
    # inspectable, but step() refuses to change them.
    core_supported: Array
    seed: Array
    initial_seed: Array
    history: History | None = None


def empty_state(size: int) -> State:
    if size < 2:
        raise ValueError("board size must be at least 2")
    n = size * size
    z = lambda *shape: jnp.zeros(shape or (n,), jnp.int32)
    b = lambda *shape: jnp.zeros(shape or (n,), jnp.bool_)
    return State(
        Tiles(z(), z(), z(), jnp.full(n, -1, jnp.int32), b(), z(), z(), b(2, n), z(), z()),
        Units(z(), z(), jnp.zeros(n, jnp.float32), b(), b(), b(), z(), b(), b(), b(),
              z(), z(), b(), z(), z(), z(), z(), z(), b(), z()),
        Cities(z(), z(), z(), z(), z(), z(), z(n, 9), b(n, n), b(), z(), z(), b()),
        Players(z(2), z(2), b(2, 128), z(2, 128), b(2, 2), z(2), z(2), z(2), b(2), b(2, 128), z(2), z(2), z(2),
                b(2, 2), b(2, 2), b(2, 2), z(2, 2), z(2, 2), b(2, 128), z(2, 2)),
        jnp.int32(1), jnp.int32(0), jnp.int32(115), jnp.int32(0), jnp.int32(0),
        jnp.bool_(False), jnp.bool_(False), jnp.bool_(True), jnp.int32(45), jnp.bool_(True), jnp.uint32(0), jnp.uint32(0),
    )


def stack_states(states: list[State]) -> State:
    if not states:
        raise ValueError("empty batch")
    return jax.tree.map(lambda *a: jnp.stack(a), *states)


def reset_from_pool(pool: State, indices: Array) -> State:
    """Gather imported initial states entirely on device (not GPU mapgen)."""
    return jax.tree.map(lambda a: a[indices], pool)


def masked_reset(states: State, replacements: State, mask: Array) -> State:
    """Reset selected batch lanes without touching continuing games."""
    return jax.tree.map(
        lambda old, new: jnp.where(mask.reshape(mask.shape + (1,) * (old.ndim - mask.ndim)), new, old),
        states, replacements,
    )
