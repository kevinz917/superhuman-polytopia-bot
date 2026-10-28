"""Shared helpers for the Rust A/B scripts (experiments/modal_rust_ab.py): load several builds of
polytopia_engine_rs into one process, sample random legal choices from dense masks, unflatten them.
"""
import glob
import importlib.machinery
import importlib.util
from importlib.resources import files

import numpy as np

COMBAT_FAMILIES = (1, 8, 9, 10, 12)   # attack, capture, recover, promote, heal


def load_rev(name, root='/workspace/revs'):
    """The polytopia_engine_rs extension built for revision `name`, imported as `<name>.polytopia_engine_rs`."""
    so = sorted(glob.glob(f'{root}/{name}/site/polytopia_engine_rs/*.so'))
    if not so:
        raise FileNotFoundError(f'no build for revision {name!r}')
    full = f'rev_{name}.polytopia_engine_rs'
    loader = importlib.machinery.ExtensionFileLoader(full, so[0])
    spec = importlib.util.spec_from_loader(full, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def rules_json():
    return files('polytopia_bot.gpu').joinpath('rules.json').read_text()


def shapes(size):
    n = size * size
    return ((n, n), (n, n), (n, 128), (n, 128), (1, 128), (n, 1), (n, 3), (n, 9), *((n, 1),) * 6, (1, 6), (1, 1), (1, 1))


def offsets(size):
    return np.cumsum([0] + [a * b for a, b in shapes(size)])


def sample(masks, size, rng, end_boost=0.12, policy='random'):
    """One legal (family, source, target) per game from dense [G, mask_width] 0/1 masks (tests/test_rust_engine.py's
    random and combat samplers, minus the JAX flatten)."""
    bounds = offsets(size)
    rows = np.array([b for _, b in shapes(size)])
    end_at = bounds[15]
    fam, src, tgt = [], [], []
    for g in range(masks.shape[0]):
        legal = np.flatnonzero(masks[g])
        pick = None
        if policy == 'combat' and rng.random() < 0.6:
            combat = legal[np.isin(np.searchsorted(bounds, legal, side='right') - 1, COMBAT_FAMILIES)]
            if len(combat):
                pick = int(combat[rng.integers(len(combat))])
        if pick is None:
            if masks[g, end_at] and rng.random() < end_boost:
                pick = int(end_at)
            else:
                pick = int(legal[rng.integers(len(legal))])
        f = int(np.searchsorted(bounds, pick, side='right') - 1)
        a, b = divmod(int(pick - bounds[f]), int(rows[f]))
        fam.append(f); src.append(a); tgt.append(b)
    return tuple(np.array(x, np.int32) for x in (fam, src, tgt))
