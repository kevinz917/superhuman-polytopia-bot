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


def shapes(size):
    n = size * size
    return ((n, n), (n, n), (n, 128), (n, 128), (1, 128),
            (n, 1), (n, 3), (n, 9), *((n, 1),) * 6, (1, 6), (1, 1), (1, 1))


def offsets(size):
    return np.cumsum([0] + [a * b for a, b in shapes(size)])


def flatten_choice(choice, size):
    widths = jnp.array([b for _, b in shapes(size)], jnp.int32)
    return jnp.asarray(offsets(size), jnp.int32)[choice.family] + choice.source * widths[choice.family] + choice.target


def unflatten(index, size):
    bounds = offsets(size)
    if not 0 <= index < bounds[-1]:
        raise ValueError("action index outside catalog")
    f = int(np.searchsorted(bounds, index, side="right") - 1)
    a, b = divmod(int(index - bounds[f]), shapes(size)[f][1])
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


def from_catalog(indices, selected, size):
    """Offline demos use stable simulator catalog IDs, never prompt-local IDs."""
    full = np.zeros(offsets(size)[-1], bool)
    for i in indices:
        if not 0 <= int(i) < len(full):
            raise ValueError("invalid catalog index")
        full[int(i)] = True
    if not full[selected]:
        raise ValueError("demonstration action is not legal")
    choice = unflatten(selected, size)
    bounds = offsets(size)
    arrays = [full[bounds[i]:bounds[i + 1]].reshape(s) for i, s in enumerate(shapes(size))]
    # Resignation remains a legal policy action, as in the engine. No action pruning.
    fm = np.array([m.any() for m in arrays])
    sm = np.pad(arrays[choice.family].any(1), (0, size * size - arrays[choice.family].shape[0]))
    tm = np.pad(arrays[choice.family][choice.source], (0, max(size * size, 128) - arrays[choice.family].shape[1]))
    return choice, PathMasks(fm, sm, tm)
