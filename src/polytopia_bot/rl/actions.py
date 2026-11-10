"""Complete hierarchical policy over the simulator's stable action catalog."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from polytopia_bot.gpu.actions import Masks

FAMILIES = Masks._fields


class Choice(NamedTuple):
    family: jax.Array
    source: jax.Array
    target: jax.Array


class PathMasks(NamedTuple):
    family: jax.Array
    source: jax.Array
    target: jax.Array


# Catalog layout versions (engine-rs core::family_widths). Every version keeps the 17 families in the same order and
# every older entry at its (family, source, target): new entries are appended at the end of a family's rows, so a
# network's heads (family logits, per-tile source logits, per-family target embeddings) index the same entries in
# every layout and older checkpoints play on as they are (docs/rules-parity-2-actions.md).
#   1: the layout before 2026-10-31 (the retired JAX engine's gpu.actions catalog, the frozen learning line's demos).
#   2: recover rows (n, 2): target 1 = DisembarkCommand (a boat on a bridge lands in place); diplomatic (1, 18): one
#      block of 6 per other seat by turn-order offset from the mover (two seats: block 0 only, the version-1 entries).
ACTION_VERSION = 2
RECOVER, DIPLOMATIC, END, RESIGN = 9, 14, 15, 16
DISEMBARK_TARGET = 1          # recover family target of the disembark entry
DIPLOMACY_BLOCK, DIPLOMACY_BLOCKS = 6, 3


def shapes(size, version=ACTION_VERSION):
    n = size * size
    if version == 1:
        return ((n, n), (n, n), (n, 128), (n, 128), (1, 128),
                (n, 1), (n, 3), (n, 9), *((n, 1),) * 6, (1, 6), (1, 1), (1, 1))
    if version != ACTION_VERSION:
        raise ValueError(f'unknown action layout version {version}')
    return ((n, n), (n, n), (n, 128), (n, 128), (1, 128),
            (n, 1), (n, 3), (n, 9), (n, 1), (n, 2), *((n, 1),) * 4, (1, DIPLOMACY_BLOCK * DIPLOMACY_BLOCKS),
            (1, 1), (1, 1))


def leaf_shapes(size, version=ACTION_VERSION):
    """Shapes of the engine's legal-mask leaves (rust_engine.split_masks; gpu.actions.Masks fields): the family
    matrices of `shapes` with a one-row family as a vector, a one-column per-tile family as a vector, and the
    one-entry families as scalars."""
    out = []
    for a, b in shapes(size, version):
        out.append(() if a * b == 1 else (b,) if a == 1 else (a,) if b == 1 else (a, b))
    return tuple(out)


def offsets(size, version=ACTION_VERSION):
    return np.cumsum([0] + [a * b for a, b in shapes(size, version)])


def flatten_choice(choice, size):
    widths = jnp.array([b for _, b in shapes(size)], jnp.int32)
    return jnp.asarray(offsets(size), jnp.int32)[choice.family] + choice.source * widths[choice.family] + choice.target


def unflatten(index, size, version=ACTION_VERSION):
    bounds = offsets(size, version)
    if not 0 <= index < bounds[-1]:
        raise ValueError("action index outside catalog")
    f = int(np.searchsorted(bounds, index, side="right") - 1)
    a, b = divmod(int(index - bounds[f]), shapes(size, version)[f][1])
    return Choice(f, a, b)


def matrices(masks, size):
    return tuple(m.reshape(shape) for m, shape in zip(masks, shapes(size), strict=True))


def family_mask(masks):
    return jnp.stack([jnp.any(m) for m in masks])


def source_mask(masks, family, size):
    n = size * size
    rows = [jnp.pad(jnp.any(m, axis=1), (0, n - m.shape[0])) for m in matrices(masks, size)]
    return jnp.stack(rows)[family]


def target_mask(masks, family, source, size):
    n = max(size * size, 128)
    # Dynamic branch avoids padding all N*N catalogs into a much larger tensor.
    branches = tuple((lambda m: lambda s: jnp.pad(m[jnp.minimum(s, m.shape[0] - 1)],
                       (0, n - m.shape[1])))(m) for m in matrices(masks, size))
    return jax.lax.switch(family, branches, source)


def path_masks(masks, choice, size):
    return PathMasks(family_mask(masks), source_mask(masks, choice.family, size),
                     target_mask(masks, choice.family, choice.source, size))


def from_catalog(indices, selected, size, version=ACTION_VERSION):
    """Offline demos use stable simulator catalog IDs, never prompt-local IDs. `version`: the catalog layout the IDs
    were recorded in (1 for the JAX engine's demonstrations); the returned choice and path masks do not depend on it."""
    full = np.zeros(offsets(size, version)[-1], bool)
    for i in indices:
        if not 0 <= int(i) < len(full):
            raise ValueError("invalid catalog index")
        full[int(i)] = True
    if not full[selected]:
        raise ValueError("demonstration action is not legal")
    choice = unflatten(selected, size, version)
    bounds = offsets(size, version)
    arrays = [full[bounds[i]:bounds[i + 1]].reshape(s) for i, s in enumerate(shapes(size, version))]
    # Resignation remains a legal policy action, as in the engine. No action pruning.
    fm = np.array([m.any() for m in arrays])
    sm = np.pad(arrays[choice.family].any(1), (0, size * size - arrays[choice.family].shape[0]))
    tm = np.pad(arrays[choice.family][choice.source], (0, max(size * size, 128) - arrays[choice.family].shape[1]))
    return choice, PathMasks(fm, sm, tm)
