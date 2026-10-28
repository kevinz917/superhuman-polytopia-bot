# Procedural GPU reset

`gpu.mapgen.reset(seed, size=11|14)` generates a complete initial 1v1 Bardur state on device. It does not gather an imported board. `jit(vmap(reset))` generates independent lanes, and `masked_reset` can replace finished lanes. The distribution is versioned as **`bardur-drylands-dotnet-v1`**.

The layout is the pinned Polyfish Drylands/Bardur algorithm with corrected random sampling: quadrant capitals, a 95% target land fraction, biome expansion, separated villages, starting animals, resources, a capital fairness pass, ruins, starfish and four corner lighthouses. Initialization supplies two Warriors, Basic/Hunting, five stars, city ownership and visibility, a zero action revision, and an explicit round cap. Layout-generation RNG is local to reset; game-event RNG is retained in the resulting state.

This is a **reference-derived map distribution**, not a claim that a seed reproduces the commercial game. Polyfish's village-placement and fairness heuristics are themselves approximations. Freeze this generator revision in research manifests and evaluate seat/resource balance on the resulting boards before collecting demonstrations.

## Confirmed reference defect

Polyfish's `dotnet_rng.rs::internal_sample` returns a number below `2^31-1`. Its `RngCore::next_u32` forwards that value unchanged, leaving the top bit permanently zero. `mapgen.rs` calls Rust `rand` distributions through this adapter instead of the explicit .NET `next_range`/`next_float` methods. This violates [`RngCore`'s uniform-bit contract](https://docs.rs/rand_core/0.9.3/rand_core/trait.RngCore.html). It is different from the documented [`System.Random.Next`](https://learn.microsoft.com/en-us/dotnet/api/system.random.next?view=net-8.0) interface.

A diagnostic build of the unmodified layout algorithm showed seed 7, size 11 reaching only 61 land cells before biome generation despite its 95% target. Bounded draws cannot reach enough distinct indices before the attempt cap. A 32-map output audit is saved in `reference-counterexamples/drylands-rng-bias.json`. The vendor checkout is unchanged.

`gpu.random` implements the seeded .NET stream directly, including signed-seed wrapping and integer bounds without enabling global JAX 64-bit mode. Eight Microsoft-runtime seeds provide 1,024 raw samples, 1,024 bounded samples and 1,024 float samples. All match locally, including `int.MinValue` and `int.MaxValue` seeds.

## Validation reference

`tools/mapgen-oracle/build.rs` reads the pinned Apache-2.0 source at build time and changes the random adapter to explicit .NET sampling. The rest of the layout algorithm is retained. Optional diagnostic logging has no state effects. This is a corrected-source validation oracle, not an independent commercial-game oracle.

Forty layouts (20 seeds at each size, including signed boundary seeds) match the GPU generator exactly in terrain, resources, structures and capital positions. Additional tests check initialized Bardur units, technologies, ownership, visibility, starting resources and legal actions. CUDA validation and reset timings are recorded separately in the mixed-action experiment receipt.

```sh
CARGO_TARGET_DIR=vendor/Polyfish/polyfish-rs/target cargo build --locked --manifest-path tools/mapgen-oracle/Cargo.toml
.venv/bin/python scripts/generate_map_fixtures.py
.venv/bin/python -m pytest tests/test_mapgen.py tests/test_random.py -q
```

Seed-for-seed layout agreement with the corrected source is not a claim of balance, native commercial-map equivalence, or full-game transition correctness. Those are separate validation questions.
