"""Variant map generator: engine-start equivalence, layout invariants, CPU-engine play."""
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from polytopia_bot.gpu import mapgen
from polytopia_bot.maps.state import initial_state, stack
from polytopia_bot.maps.stream import MapSource, TYPES, parse_mix
from polytopia_bot.maps.variants import (MIN_CAPITAL_DISTANCE, RUINS, PRESETS, generate, _cheb, _components, SHALLOW, OCEAN, FIELD, MOUNTAIN,
                                         FOREST, VILLAGE, LIGHTHOUSE, ANIMAL, FISH, METAL, FRUIT, STARFISH)

SIZE = 14


def _same(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_classic_layout_start_is_bitwise_mapgen_reset():
    start = jax.jit(lambda seed, t, s, r, c: initial_state(seed, t, s, r, c, size=SIZE))
    for seed in (200000, 200001, 305000, 1810000001):
        lay = mapgen.layout(jnp.int32(seed), size=SIZE)
        _same(start(jnp.int32(seed), lay.terrain, lay.structure, lay.resource, lay.capitals),
              mapgen.reset(jnp.int32(seed), size=SIZE))


@pytest.mark.parametrize('size', [14, 18])
def test_variant_layout_invariants_and_determinism(size):
    SIZE = size
    d = _cheb(SIZE)
    corners = [0, SIZE - 1, SIZE * SIZE - SIZE, SIZE * SIZE - 1]
    started = time.perf_counter()
    count = 0
    for name, preset in PRESETS.items():
        for seed in range(60):
            lay, info = generate(seed, name, size=SIZE)
            count += 1
            t, st, r, caps = lay.terrain, lay.structure, lay.resource, lay.capitals
            assert set(np.unique(t)) <= {SHALLOW, OCEAN, FIELD, MOUNTAIN, FOREST}
            assert caps[0] != caps[1] and d[caps[0], caps[1]] >= MIN_CAPITAL_DISTANCE[SIZE]
            assert (st == 2).sum() <= RUINS[SIZE]
            for c in caps:
                assert st[c] == VILLAGE and np.all(t[d[c] <= 1] >= FIELD) and np.all(t[d[c] <= 1] != MOUNTAIN)
            villages = np.flatnonzero(st == VILLAGE)
            assert np.all(t[villages] == FIELD)
            assert np.all(st[corners] == LIGHTHOUSE)
            for res, ok in ((ANIMAL, (FOREST,)), (FISH, (SHALLOW,)), (METAL, (MOUNTAIN,)), (FRUIT, (FIELD,)),
                            (STARFISH, (SHALLOW, OCEAN))):
                assert np.all(np.isin(t[r == res], ok)), (name, seed, res)
            assert np.all(st[r > 0] == 0)
            # two animals next to each capital (the frozen start guarantees them too)
            for c in caps:
                assert np.sum((d[c] == 1) & (r == ANIMAL)) >= 1
            if preset.shared_landmass:
                label, _ = _components(t.reshape(SIZE, SIZE) >= FIELD, SIZE)
                assert label[caps[0]] == label[caps[1]], (name, seed)
            again, _ = generate(seed, name, size=SIZE)
            for x, y in zip(lay, again):
                np.testing.assert_array_equal(x, y)
    per_map_ms = 1e3 * (time.perf_counter() - started) / (2 * count)
    print(f'variant generation {per_map_ms:.2f} ms/map')
    assert per_map_ms < 50


def test_mix_parsing_and_type_is_a_function_of_seed():
    assert parse_mix('proposed-v1') and set(parse_mix('uniform')) == set(TYPES)
    source = MapSource('uniform')
    kinds = [source.type_for(s) for s in range(400)]
    assert set(kinds) == set(TYPES)
    assert kinds == [MapSource('uniform').type_for(s) for s in range(400)]


def test_variant_maps_on_v1_boards_are_unchanged():
    """bardur-variants-v2 only adds 18x18: the 11x11/14x14 maps hash like bardur-variants-v1 (commit 72d69a3)."""
    import hashlib
    h = hashlib.sha256()
    for size in (11, 14):
        for preset in (*PRESETS, 'mixed'):
            for seed in range(12):
                lay, _ = generate(seed, preset, size=size)
                for a in lay:
                    h.update(np.asarray(a).astype(np.int32).tobytes())
    assert h.hexdigest() == '36de45e8dc7704068b84429172255f2f8e900b5356a8a259eddbfdb91a487e1d'
