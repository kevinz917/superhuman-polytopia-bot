# Research rules `bardur-1v1-v2`: v1 plus the 18×18 board

> **Retired 2026-10-27.** These research rules (and the JAX engine that defined them) are no longer used by anything that trains, evaluates or serves games. The game is now Polytopia's own rules (2.17.3), the only ruleset: see [rules-alignment-2026-10-27.md](rules-alignment-2026-10-27.md).

Created 2026-10-19 for the 18×18 self-play run ([rl-18x18-2026-10-19.md](rl-18x18-2026-10-19.md)). v2 is
[`bardur-1v1-v1`](ruleset-v1.md) with one addition: a third board size. Every v1 rule, table and transition is
unchanged; nothing about 11×11 or 14×14 games changed.

| | v1 | v2 |
|---|---|---|
| Board sizes | 11, 14 | 11, 14, **18** (Polytopia "Large", 324 tiles) |
| Round limit (truncation, never a win) | 100 | 100 on 11/14, **150 on 18** (`gpu.ROUND_LIMITS`) |
| Action limit | 10,000 | 10,000 |
| Classic generator | `bardur-drylands-dotnet-v1` | `bardur-drylands-dotnet-v2`: v1 + Large; ruins 4 / 5 / **9** as the reference `mapgen.rs` (`Tiny 4, Small 5, Large 9`) |
| Variant generator (outside `gpu/`) | `bardur-variants-v1` | `bardur-variants-v2`: v1 + 18×18 scaling (below) |

## What changed in code

- `gpu/mapgen.py`: `layout` accepts size 18 (ruin count from the reference table); `reset(max_turns=None)` uses the
  size's round limit. Quadrant capitals, land growth, biomes, villages, resources, fairness, starfish and
  lighthouses are the same code for every size.
- `gpu/__init__.py`: `RULESET_ID = "bardur-1v1-v2"`, `BOARD_SIZES`, `ROUND_LIMITS`, `round_limit(size)`.
- `gpu/snapshot.py`: `COMPATIBLE_DIGESTS` lists the v1 engine digest (`7b76e65b…`). Checkpoints and snapshots
  recorded under v1 still load (`runner.load_checkpoint`, `snapshot.loads`, league/bootstrap inputs); any other
  digest is still refused. This is safe because v1 games are bitwise unchanged (receipts below).
- `maps/variants.py` at 18×18 only: capitals ≥ 8 apart (6 on 11/14); 9 ruins; ridge maps get one more pass
  (3–4); pangea and continents scale their land-shape noise with the board (×18/14) and lakes by √(18/14);
  islands, villages (spacing 3), mountains and forests keep their size in tiles, so their count grows with the
  area (about 1.65×). Everything is gated on `size == 18`; the random stream on 11/14 is untouched.

## Receipts

| Fact | Evidence |
|---|---|
| 11×11 / 14×14 classic layouts unchanged | `tests/test_mapgen.py`: the 40 corrected-oracle layouts (fixture rebuilt on Modal reproduced all 40 exactly) |
| 18×18 classic layouts match the reference algorithm | 20 Large layouts from the corrected-sampling Polyfish oracle (`experiments/modal_mapgen_oracle.py`, pinned `a9f90ab5`, `MapSize::Large`) match `gpu.mapgen.layout` bitwise in terrain, structures, resources and capitals |
| 11×11 / 14×14 variant maps unchanged | `tests/test_maps.py::test_variant_maps_on_v1_boards_are_unchanged` (digest of 10 presets × 12 seeds × 2 sizes equals the v1 generator's, computed from commit 72d69a3 on Modal) |
| 18×18 starts are complete Bardur starts | `tests/test_mapgen.py` and `tests/test_rl_18x18.py` (units, techs, ownership, lighthouses, 9 ruins, round limit 150, legal actions) |
| 18×18 variant invariants | `tests/test_maps.py` at size 18: capitals ≥ 8 apart and on 3×3 land, resources on legal terrain, shared landmass where required |
| Fast CPU engine = frozen engine at 18×18 | `tests/test_maps.py::test_cpu_engine_plays_variant_maps_like_the_frozen_engine[18]` (states, masks, rewards bitwise through resets) |
| Random legal traces at 18×18 keep invariants | `docs/benchmarks/2026-10-19-cpu-native-episodes-18.json` (`gpu.validation.run_native_validation`, size 18, round limit 150) |
| Policy games at 18×18 finish with no illegal action | per-type smoke in the run report (frozen engine, all 10 map types) |

Scope is the same as v1: an internally consistent research environment, not commercial-game equivalence. The
18×18 board is Polytopia's Large size; the commercial game's Large-map generation is not claimed.
