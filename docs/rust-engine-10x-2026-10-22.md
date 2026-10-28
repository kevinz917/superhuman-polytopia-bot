# Rust actor engine toward 10x per tick: levers, exactness, where it stops (2026-10-22)

**Question.** How close can the Rust actor engine (`engine-rs/`, merged Oct 21 as the only training engine) get to
10x faster per `GameBatch.tick` without changing a single output bit, keeping the 3-step policy and the dense mask
output as the default?

**Answer.** Not 10x: **about 6x with the dense masks and 6.5-7.4x with packed masks, bit for bit.** On
one Modal host, one pinned core, 24 games, mid-game states (mean turn 46), all revisions interleaved: the steady
24-game tick on 14x14 went from 3.57 ms to 0.600 ms with dense masks (5.96x) and 0.545 ms with
`packed_masks=True` (6.55x); the report's one-shot tick from 3.59 to 0.644 ms (5.6x). On 18x18: 5.22 -> 0.872 ms
(5.99x) dense, 0.711 ms (7.35x) packed. The target of 0.35-0.5 ms on 14x14 is met by the packed path only at its
upper end on a faster host (0.39-0.41 ms on the fastest host seen, where the baseline ran 2.7 ms), not by the
dense default. Exactness: the JAX lockstep harness passes the report's whole matrix on HEAD (9,408,000 steps),
13.1 M further Rust-vs-baseline lockstep steps cover every lever commit (with 2% random illegal actions mixed in
from lever 5a on), the full pytest suite passes, and two deliberate mutants were caught by the Rust harness.

Branch `claude/rust-engine-10x-1b12b7` (cut from `main` d5a5550), one commit per lever. Nothing under `gpu/`, no
policy or action-space change, no mask-semantics change. The dense `[G, 130868]` u8 mask output is still the
default and bit-identical; `tick(..., packed_masks=True)` / `legal_masks(packed=True)` add the inference-server
branch's packed format (same signature and numpy.packbits layout as `claude/inference-server` e26291a, not a fork).

## How it was measured

Everything ran on Modal (staging): no build, test or benchmark ran on the Mac.

* `experiments/modal_rust_ab.py` builds several engine-rs revisions side by side in one image (each a git commit
  extracted with `git archive`, plus `work`), and `scripts/bench_rust_ab.py` loads all of them into one process
  pinned to one core of a 4-vCPU container, warms 24 games with 400 ticks of random legal play (mean turn 46, ~8
  units per game, the report's state), imports that state into every revision and times them **interleaved call by
  call**, so every row of a table below is the same host at the same time. Modal does not report the CPU model
  (`model name: unknown`) and hosts differ by up to 1.5x between runs, so only rows of one run are compared.
* Metrics, ms per 24-game tick: **tick** = one `GameBatch.tick` (step, auto-reset, masks, both views) on the warmed
  state, restored before each call, median of 25 (the report's method; caches are cold after the restore);
  **play** = the steady state an actor sees: 25 consecutive ticks of random play from the warmed state, the
  revisions taking turns tick by tick, three passes, median of the 75 per-tick times (dense masks, and with
  `packed_masks=True`); **step**, **masks**, **observe x2** = the separate entry points, median of 25;
  **mask compute** = `legal_masks` into a counting sink (no memory traffic).
* `GameBatch.profile_phases(True)` / `phase_profile()` (new, off by default) time tick and step phases, step
  validation and execution per action kind; `profile_masks` times each mask family.

## Results

Times are ms per 24-game tick, one host for all rows of a table (see "How it was measured"); "speed-up" is
the baseline's play time over the row's. Rows are cumulative in commit order; the rows from "6 territory bitsets"
to "decode capital lazily" still carry lever 6, which "3c" reverts (the revert commit alone does not build).
The per-family profile's harvest/forest/destroy entry includes the build setup that lists the economy tiles.

### 14x14 (mean turn 46.3, 8.3 units/game)

| lever (cumulative) | tick | play dense | play packed | step | masks | masks packed | observe x2 | mask compute | speed-up (play dense / packed) |
|---|---|---|---|---|---|---|---|---|---|
| baseline (main d5a5550) | 3.586 | 3.574 | - | 0.556 | 2.478 | - | 0.469 | 2.309 | 1.00x |
| 1 scratch buffers | 3.617 | 3.593 | - | 0.565 | 2.511 | - | 0.448 | 2.385 | 0.99x |
| 2 frontier Bellman-Ford | 2.954 | 2.971 | - | 0.504 | 1.960 | - | 0.466 | 1.843 | 1.20x |
| 3 candidate kinds/tiles | 1.309 | 1.323 | - | 0.301 | 0.566 | - | 0.450 | 0.417 | 2.70x |
| 4 sparse sink + packed | 1.135 | 1.148 | 1.0896 | 0.289 | 0.370 | 0.3014 | 0.458 | 0.296 | 3.11x / 3.28x |
| 7a observe loops | 0.964 | 0.975 | 0.9207 | 0.300 | 0.364 | 0.3 | 0.291 | 0.246 | 3.67x / 3.88x |
| 3b build bitset algebra | 0.948 | 0.953 | 0.898 | 0.301 | 0.354 | 0.2875 | 0.289 | 0.219 | 3.75x / 3.98x |
| 7b move path/reveal, no-op skips | 0.844 | 0.851 | 0.7914 | 0.199 | 0.347 | 0.2838 | 0.284 | 0.222 | 4.20x / 4.52x |
| 7c observe AVX2 dispatch | 0.753 | 0.750 | 0.6829 | 0.193 | 0.358 | 0.2809 | 0.181 | 0.219 | 4.77x / 5.23x |
| 7d uninit views, settled monuments | 0.717 | 0.696 | 0.6383 | 0.193 | 0.342 | 0.2688 | 0.194 | 0.205 | 5.13x / 5.60x |
| 6 territory bitsets | 0.719 | 0.701 | 0.6366 | 0.188 | 0.346 | 0.2676 | 0.186 | 0.202 | 5.10x / 5.61x |
| 5a cached validation | 0.724 | 0.677 | 0.6178 | 0.183 | 0.339 | 0.2711 | 0.184 | 0.202 | 5.28x / 5.78x |
| decode capital lazily | 0.719 | 0.671 | 0.6116 | 0.181 | 0.330 | 0.2705 | 0.192 | 0.207 | 5.33x / 5.84x |
| 3c economy-tile list | 0.690 | 0.649 | 0.5909 | 0.188 | 0.318 | 0.2473 | 0.197 | 0.179 | 5.51x / 6.05x |
| 3d rule bitsets per tech/stars | 0.656 | 0.619 | 0.564 | 0.181 | 0.278 | 0.2115 | 0.191 | 0.141 | 5.77x / 6.34x |
| 7e observe own fields, fused passes | 0.644 | 0.600 | 0.5454 | 0.178 | 0.278 | 0.2158 | 0.182 | 0.148 | 5.96x / 6.55x |

Final per-family mask compute (counting sink, ms per 24 games): {'reward': 0.0029, 'move_rows': 0.0479, 'attack_rows': 0.0073, 'summon': 0.0136, 'economic_tiles': 0.0079, 'build': 0.0504, 'research': 0.0062, 'harvest_forest_destroy': 0.0272, 'units': 0.0045, 'pending_reward': 0.0035, 'full_count': 0.148, 'full': 0.1941, 'observe': 0.0602}

Final phase profile over the 25-tick play sequence (dense masks, ms per tick, profiler on):

```
  tick.masks: 0.2012 (600 calls)
  tick.step: 0.1208 (600 calls)
  tick.alloc: 0.0875 (25 calls)
  tick.observe: 0.0737 (600 calls)
  tick.observe_opp: 0.0707 (600 calls)
  execute.move: 0.0349 (246 calls)
  execute.build: 0.0330 (138 calls)
  step.post: 0.0169 (600 calls)
  execute.end_turn: 0.0124 (114 calls)
  execute.diplomacy: 0.0071 (22 calls)
  tick.output: 0.0035 (25 calls)
  validate.move: 0.0026 (246 calls)
```
`export()` (what `RustGames.states()` pays every tick), ms: {'base': 0.1461, 'l6': 0.7867, 'work': 0.1441}

### 18x18 (mean turn 46.8, 8.2 units/game)

| lever (cumulative) | tick | play dense | play packed | step | masks | masks packed | observe x2 | mask compute | speed-up (play dense / packed) |
|---|---|---|---|---|---|---|---|---|---|
| baseline (main d5a5550) | 5.285 | 5.221 | - | 0.795 | 3.888 | - | 0.688 | 3.520 | 1.00x |
| 1 scratch buffers | 5.255 | 5.224 | - | 0.778 | 3.889 | - | 0.670 | 3.584 | 1.00x |
| 2 frontier Bellman-Ford | 4.288 | 4.292 | - | 0.681 | 3.011 | - | 0.660 | 2.706 | 1.22x |
| 3 candidate kinds/tiles | 1.964 | 1.999 | - | 0.469 | 0.897 | - | 0.655 | 0.634 | 2.61x |
| 4 sparse sink + packed | 1.644 | 1.664 | 1.4946 | 0.476 | 0.556 | 0.426 | 0.664 | 0.442 | 3.14x / 3.49x |
| 7a observe loops | 1.384 | 1.394 | 1.2481 | 0.460 | 0.549 | 0.4165 | 0.421 | 0.329 | 3.75x / 4.18x |
| 3b build bitset algebra | 1.362 | 1.385 | 1.2274 | 0.457 | 0.540 | 0.409 | 0.422 | 0.303 | 3.77x / 4.25x |
| 7b move path/reveal, no-op skips | 1.242 | 1.223 | 1.0688 | 0.307 | 0.543 | 0.4036 | 0.415 | 0.303 | 4.27x / 4.89x |
| 7c observe AVX2 dispatch | 1.086 | 1.085 | 0.9273 | 0.306 | 0.537 | 0.3984 | 0.288 | 0.304 | 4.81x / 5.63x |
| 7d uninit views, settled monuments | 1.025 | 1.013 | 0.8554 | 0.281 | 0.509 | 0.3658 | 0.296 | 0.275 | 5.15x / 6.10x |
| 6 territory bitsets | 1.021 | 1.006 | 0.8493 | 0.281 | 0.491 | 0.368 | 0.294 | 0.270 | 5.19x / 6.15x |
| 5a cached validation | 1.073 | 0.987 | 0.8222 | 0.285 | 0.508 | 0.3779 | 0.281 | 0.267 | 5.29x / 6.35x |
| decode capital lazily | 1.055 | 0.984 | 0.8185 | 0.274 | 0.493 | 0.3813 | 0.295 | 0.271 | 5.31x / 6.38x |
| 3c economy-tile list | 1.011 | 0.943 | 0.7898 | 0.287 | 0.467 | 0.3419 | 0.292 | 0.235 | 5.54x / 6.61x |
| 3d rule bitsets per tech/stars | 0.940 | 0.900 | 0.7427 | 0.275 | 0.431 | 0.2991 | 0.289 | 0.190 | 5.80x / 7.03x |
| 7e observe own fields, fused passes | 0.906 | 0.872 | 0.7105 | 0.265 | 0.434 | 0.2962 | 0.277 | 0.186 | 5.99x / 7.35x |

Final per-family mask compute (counting sink, ms per 24 games): {'reward': 0.0045, 'move_rows': 0.0782, 'attack_rows': 0.0113, 'summon': 0.0191, 'economic_tiles': 0.0126, 'build': 0.0567, 'research': 0.0068, 'harvest_forest_destroy': 0.0332, 'units': 0.0069, 'pending_reward': 0.0055, 'full_count': 0.1863, 'full': 0.2977, 'observe': 0.0967}

Final phase profile over the 25-tick play sequence (dense masks, ms per tick, profiler on):

```
  tick.masks: 0.2440 (600 calls)
  tick.step: 0.1576 (600 calls)
  tick.alloc: 0.1516 (25 calls)
  tick.observe: 0.1038 (600 calls)
  tick.observe_opp: 0.1004 (600 calls)
  execute.move: 0.0470 (235 calls)
  execute.build: 0.0435 (144 calls)
  step.post: 0.0238 (600 calls)
  execute.end_turn: 0.0178 (110 calls)
  tick.output: 0.0036 (25 calls)
  execute.diplomacy: 0.0031 (8 calls)
  validate.forest: 0.0024 (40 calls)
  validate.move: 0.0021 (235 calls)
```
`export()` (what `RustGames.states()` pays every tick), ms: {'base': 0.2399, 'l6': 2.0298, 'work': 0.2358}


## The levers

Order is commit order; each row of the tables is cumulative.

1. **Scratch buffers allocated once** (`MaskScratch`, `MoveScratch` in the step `Scratch`): no measurable gain on
   its own (small `vec!`s are cheap); kept because later levers build on the buffers.
2. **Frontier Bellman-Ford** for movement reachability. `kernels.movement_targets` runs 8 synchronous relaxation
   sweeps over every tile; `reach()` runs the same 8 synchronous (Jacobi) rounds over the frontier only: each round
   relaxes from the tiles whose distance changed in the previous round, reading their previous-round distances.
   A tile whose distance did not change offers nothing new, so the result is the reference's min over walks of at
   most 8 edges under the same budget / terminal / road-cost / `j == source` rules. The unit-independent tile
   terms (passable, land access, road, zone of control, mountain, forest, own port, water) are packed into one
   byte per tile once per owner (`movement_flags`) and specialised per unit by its skills.
3. **Build, summon, research visit candidates only.** Build rows evaluate the four terms of
   `structures.build_mask` only for kinds a term can hold; then (3b) as bitset algebra over all 128 kinds at
   once: (a) unlocked & terrain & empty & adjacency & city room, (b) the neutral road, (c) the tile's resource
   building, (d) earned monuments on free land, then the bridge orientation test, with derived rule bitsets
   (`terrain_bits`, `adjacent_rev`, ...); (3c) only economy tiles get full rows, other tiles only the neutral-road
   term; (3d) per-call setup from tables indexed by discovered tech and by stars (`CostIndex`) instead of 128-kind
   scans. Summon and research the same way. Step validates Build and Summon from one row instead of n x 128
   buffers.
4. **Masks emit only their true entries** through a `Sink`: `Dense` (0/1 bytes into a zeroed buffer) or `Packed`
   (bits, numpy.packbits order). The masks hold ~60 true entries per game out of 130,868; the old code zeroed and
   rewrote all of them. The output is allocated zeroed and never filled.
5. **(5a) Step validation from the previous tick's masks.** Each game keeps the packed bits of the masks `tick()`
   returned while they describe its state (`masks::Tee` records them as they are written; reset, `reset_layouts`,
   `step()` and `import_` forget them). The next step validates Move, Build and Summon by one lookup. The status
   is provably the same: wherever a mask entry and step's own predicate can differ (a pending city reward, the
   kind-37 build, an unsupported unit kind, `core_supported=False`) the action is `UNSUPPORTED` regardless of
   validity. The diff harness injects 2% uniformly random catalog entries so rejections are compared too.
   **Full incremental masks (dirty tracking keyed to what `step` touched) were not built**, see below.
6. **Territory as u64 bitsets: measured null, reverted.** Interleaved A/B: build family 0.1035 vs 0.1016 ms,
   play 0.740 vs 0.739 ms on 14x14, 18x18 flat; the per-city scans it replaced are cheap byte loops. It also made
   `export()` 3x slower (0.33 -> 1.07 ms: expanding 24 x n^2 bits), which `RustGames.states()` pays every tick;
   that was fixed (5f125b1) and then the lever reverted (5d5edda) rather than keep a dual representation. The
   revert commit 5d5edda does not build on its own (its `structures.rs` half landed in the next commit, 1715a52);
   1715a52 and every later commit build and pass, so it is left in history rather than rewritten.
7. **Step and observe.** (7a) `observe` as one tight select loop per packed-view field over precomputed
   visibility/ownership masks; (7c) the same code compiled twice and picked at run time with
   `is_x86_feature_detected!("avx2")` (the i32 -> i16 selects vectorise only with AVX2; the wheel still runs on
   baseline x86-64; integer-only, identical bits); (7e) own-city/own-unit fields as a zero fill plus writes at the
   seat's cities and units. (7b) Move execution: the reference Dijkstra (same heap, tie order, neighbour order)
   in reusable buffers returning the path as a list; contact, stop tile and traveled set from path ranks (distinct
   along a path, so the reference's argmax/argmin pick the same tiles); reveal by the boxes around traveled tiles
   instead of an n-scan per path tile. Exact no-op skips: `discover` stops after the exploration score when no
   lighthouse was revealed (`add_population` of 0), `meet` returns when the players already know each other,
   `capital_vision` when its gate is closed, `adjust_building_population` when every amount is zero, `income`'s
   territory bonus when there is no market. (7d) the two views are written into uninitialised numpy arrays (every
   element is written); `State.monuments_settled` (engine-internal, not exported) marks a state that
   `remember_monuments` was already applied to, so the next step's leading call (an OR of the same earned set) is
   skipped; `update_exploration` visits Chebyshev balls (`Geom::for_ball`). `decode` looks up the opponent capital
   only for diplomacy actions.

## Why not full incremental masks (lever 5 proper)

After levers 2-4 and 3b-3d the whole mask computation is 0.148 ms per 24 games on 14x14 (about
6 us per game), spread over ten families whose per-call setup dominates; the biggest left are move
rows and build rows at roughly 1-1.5 us per game each. A per-family cache with dirty tracking would need a
dependency spec for every family against every side effect of `step` (a move reveals tiles, which changes zone of
control, bridge legality and the explorer monument; `meet` and ruins change stars, which changes every
affordability bit; captures change territory and owners), for at most ~0.05 ms per tick. The one cross-tick reuse
that is exact by construction and pays, validating the next action from the masks already computed for the same
state, is lever 5a.

## Exactness evidence

**JAX lockstep harness** (`tests/test_rust_engine.py` via `experiments/modal_rust_engine.py::equality`, HEAD
ee8b9a0, against `CpuGames`: all 72 State fields, all 17 mask families, both packed views, status, done /
terminated / truncated, rewards, `final_views`, policy features and `final_features` after every tick):

| board | maps | play | reference | processes x ticks | steps | games finished | by elimination/resign | by round cap | maps seen |
|---|---|---|---|---|---|---|---|---|---|
| 14x14 | classic | random | fast | 16 x 10,000 | 3,840,000 | 4,706 | 1,702 | 3,004 | 5,090 |
| 18x18 | classic | random | fast | 8 x 8,000 | 1,536,000 | 1,263 | 664 | 599 | 1,455 |
| 11x11 | classic | random | fast | 4 x 6,000 | 576,000 | 688 | 263 | 425 | 784 |
| 14x14 | `uniform` variant mix (map_batch path) | random | fast | 16 x 4,000 | 1,536,000 | 1,773 | 678 | 1,095 | 2,157 |
| 14x14 | classic | heuristic | fast | 16 x 4,000 | 1,536,000 | 1,546 | 1,544 | 2 | 1,930 |
| 14x14 | classic | combat | **frozen (fast=False)** | 8 x 2,000 | 384,000 | 328 | 197 | 131 | 520 |

All 68 processes passed; 9,408,000 steps. The heuristic run alone covers 132,204 attacks, 8,165 enemy-city
captures, 1,868 promotions, 164 Mind Bender conversions, 269 Cloak infiltrations, 356 Stomps and 39 Splashes.

**Rust vs Rust lockstep** (`scripts/rust_engine_diff.py` via `modal_rust_ab.py::diff`): every lever commit was
played in lockstep against the certified baseline wheel (d5a5550) in one process, comparing status, done /
terminated / truncated, rewards, final views, dense masks, both views and every exported field after every tick;
odd ticks through `tick(packed_masks=True)` and even ticks through `legal_masks(packed=True)` against
`numpy.packbits` of the baseline's dense masks; an `export`/`import_` round trip and out-of-band
`legal_masks`/`observe` calls every 251 ticks (cache invalidation); from 9b7dca5 on, 2% uniformly random catalog
entries (about 15,000 rejected actions per 768k steps, equal status and untouched state). 13,094,400 steps in
total over 11x11, 14x14 and 18x18, classic maps and the `uniform` variant mix, random and combat-biased play, all
equal; branch-coverage counters equal between the two engines in every run.

| commit | runs (board, maps, play, procs x ticks) | steps |
|---|---|---|
| 97cf71e lever 2 | 14x14 classic random 16x1000 | 384,000 |
| c1ff632 lever 3 | 14x14 classic random 16x1500; 18x18 classic random 16x800; 14x14 uniform combat 16x1000 | 1,267,200 |
| 2474fb8 lever 4 | 14x14 classic random 16x1500; 18x18 classic combat 16x800 (packed checked every tick) | 883,200 |
| f99fd6f lever 7a | 14x14 classic combat 16x1500 | 576,000 |
| 80ad2c9 levers 3b + 7b | 14x14 classic random 16x2000; 18x18 classic combat 16x1000; 14x14 uniform combat 16x1500 | 1,728,000 |
| 6cfab7c levers 7c + 7d | 14x14 classic random 16x2000; 18x18 classic combat 16x1000; 11x11 uniform combat 16x2000 | 1,920,000 |
| da95a3b lever 6 | 14x14 classic combat 16x2000; 18x18 uniform random 16x1000 | 1,152,000 |
| 09d5df5 lever 5a + decode (with random illegal actions from here on) | 14x14 classic random 16x2000; 18x18 classic combat 16x1000 | 1,152,000 |
| 1715a52 revert 6 + lever 3c | 14x14 classic combat 16x2000; 18x18 uniform random 16x1000 | 1,152,000 |
| c090610 lever 3d | 14x14 classic combat 16x2000; 18x18 uniform random 16x1000 | 1,152,000 |
| ee8b9a0 lever 7e (HEAD) | 14x14 classic random 16x2000; 18x18 classic combat 16x1000; 14x14 uniform combat 16x1500 | 1,728,000 |

**The harness catches mistakes.** Two deliberate mutants, run once each and discarded: road cost from the source
tile only (caught at reset, 99-122 mask entries differ in each of 4 processes) and no lighthouse population for a
single lighthouse (caught within 5 ticks in all 16 processes, through the packed view's city population).

**Repository tests** (`modal_rust_engine.py::tests_sharded`, every `tests/test_*.py` in its own pytest process
with the wheel installed): 75 files, 2,331 tests passed, 7 skipped, 0 failures (6 files collect no tests in this image, as before).

## The actor side

`RustGames.split_masks` now reinterprets the engine's 0/1 uint8 masks as bool (`view(bool)`) instead of copying
them (`astype(bool)`): the same `Masks`, still views into a fresh per-tick buffer. With the report's own bench
script (`scripts/bench_rust_engine.py`, Rust only) on HEAD: 

| 14x14, ms per 24 games | main d5a5550 (host A) | HEAD (host B) |
|---|---|---|
| `GameBatch.tick` | 2.75 | 0.494 |
| step / masks / observe x2 | 0.43 / 1.98 / 0.33 | 0.11 / 0.18 / 0.13 |
| tick + `split_masks` | 5.87 | 0.594 |
| `export()` | 0.09 | 0.12 |
| `RustGames.tick` incl. JAX `view_features` | 6.66 | 2.09 |
| JAX `encode_view` alone | 1.45 | 1.80 |

(two hosts: the ratios inside a column are the comparison; 18x18: tick 4.43 -> 0.668, `RustGames.tick` 6.37 -> 3.73.)
The wheel grew from 471 to 517 KB.

The engine is no longer the actor's CPU cost: `RustGames.tick` spends most of its time in the JAX `encode_view`
of the packed views (the policy features), and `states()` exports all 72 fields every tick although the actor
reads six of them (`current_player`, `players.stars`, `players.score`, `units.owner`, `tiles.owner`,
`cities.owner`). Making `Tick.states` lazy would change what callers that tree-map or index it get, so it was
left as is.

## Was 10x reached?

No. The steady-state tick is ~6x faster with the dense masks and 6.5-7.4x with packed masks; on 14x14 that is
0.60 / 0.55 ms on the measurement host (0.45 / 0.39 ms on the fastest host seen) against a 0.35-0.5 ms target.
What blocks the rest:

1. **The dense output itself.** The default contract is a fresh `[24, 130868]` u8 array per tick (7.2 MB on 18x18).
   Allocating it zeroed and touching the pages the ~60 true entries per game land on costs ~0.09 ms on 14x14 and
   ~0.15 ms on 18x18 (`tick.alloc` plus the masks phase's page faults: dense `masks` 0.278 vs packed 0.216 vs
   compute-only 0.148), whatever the engine does. Reusing one buffer across ticks would change aliasing for
   callers. The packed format removes most of it; the actor could only use it through the inference-server path.
2. **No dominant piece left.** Per 14x14 game the tick is now ~25 us: masks ~6 us of compute spread over ten
   families (move and build rows ~2 us each, the rest well under 1 us), step ~5 us (~3.5 us per move, ~6 us per
   build, ~2.7 us per end turn, ~0.7 us post-step bookkeeping per action), observe ~3 us per seat, output allocation ~3.5 us.
   Each further lever found was worth 2-5%.
3. **What would be needed for 10x**: full incremental masks (dirty tracking per family, see above; at most ~0.05
   ms), an incremental observe (two 15 KB views per game, mostly unchanged between ticks: ~0.1 ms at most, same
   dependency-tracking risk), and compiling the whole crate for x86-64-v3 (AVX2 everywhere, not only observe) with
   an import-time CPU check. Together plausibly 8-9x on packed output; dense stays bounded by (1).

For the actor, the engine is no longer the CPU cost: JAX `encode_view` (1.8 ms) is now ~3.5x the whole Rust tick
inside `RustGames.tick`. Porting `encode_view` to Rust (or running it on the GPU with the network) is the lever
that moves actor throughput; a further engine speed-up does not.

## Spend

Modal staging, billed at the time of writing: $3.03 (A/B builds, Rust-vs-Rust diffs and benches on
`polytopia-rust-ab` $0.44; JAX equality matrix, pytest and the report-script benches on `polytopia-rust-engine`
$2.59). Billing lags up to an hour and the last JAX run was still in that window, so the total is roughly
$4-5, well under the $15 budget. Image builds are not billed.

## Merged-tree gate (main 86554c9 merged in, 1f71a03)

Before fast-forwarding `main`, `main` (86554c9: the A6 learner batch assembly) was merged into this branch (no
conflicts; the two share no files) and the merged tree was gated on Modal:

- **pytest**, `modal_rust_engine.py::tests_sharded` (every `tests/test_*.py` including the new
  `tests/test_a6_assembly.py`): 76 files, 2,335 tests passed, 7 skipped, 0 failures (`test_a6_assembly.py` 4
  passed, `test_rust_engine.py` passed); the first container was preempted by Modal and the restarted one completed.
- **JAX lockstep**, 14x14 classic, 16 x 4,000 ticks: all 16 processes equal, 1,536,000 steps, 1,770 games
  finished (686 by elimination/resignation, 1,084 at the round cap).
- **10-minute training smoke** `rust10x-merged-smoke` (app ap-mZPq5tc0PIZhe7oHg9td5y, ledger `runs/rust-engine`): the
  `a6-h100-smoke` configuration (c5-omni recipe from `c5-omni-s1-final` and its critic, epochs 1, seed 403, 1xH100,
  TF32 convs + fp32 heads, 6 actor containers x 16 vCPU / 24 GiB, 16 processes x 24 games, A6 assembly as on main),
  600 s of training. 66 updates, 17.3 M samples in 605 s.

| | a6-h100-smoke (main + A6, Oct 22) | rust10x-merged-smoke (this branch + main) |
|---|---|---|
| samples/s steady | 43.8k (updates 21-197) | **53.6k** (updates 26-66, all 6 actors up; 51.7k over 21-66) |
| learner GPU train per update | 1.65 s | 1.65 s |
| actor processes connected / `engine` | 96/96, rust | 96/96, rust (all 6 containers) |
| errors / restarts / reconnects / illegal-action raises | 0 | 0 / 0 / 0 / 0 |
| KL, clip fraction (late updates) | 0.004, 0.046 | 0.003-0.004, 0.035-0.047 |
| entropy update 1 -> 66 | 0.84 -> 2.46 | 1.23 -> 2.65 |
| stale rows, mean lag | 0, 1.1 | 0, 0.8 |

Actor containers delivered 5.6-9.5k samples/s each, **364-614 environment steps/s per process** (by container:
614, 467, 364, 570, 373, 551; the earlier Rust smoke measured ~525 and Stage A JAX 399; the spread is the same
host variation seen before, and the later-starting containers ran shorter so their averages carry more startup).
The fleet is still actor-bound (the learner's own assembly per batch stays ~1.2 s under 1.65 s of training), and
the actor's cost is now mostly the policy forward and the JAX feature encode, not the engine. One set of function
calls in the registry and 8 containers at peak (orchestrator, learner, 6 actors): no orchestrator replay; all
containers stopped on their own at the end of training.

Gate spend: $2.83 billed at the time of writing (smoke $1.83, pytest $0.80 including the preempted attempt,
equality $0.20); about $3 once the last hour settles. Branch total including the 10x work: roughly $7-8.

## How to run

```bash
RUST_REVS=base=d5a5550 RUST_WORK=<commit> modal run --env staging experiments/modal_rust_ab.py::diff --size 14 --ticks 2000 --procs 16 --policy combat
RUST_REVS=base=d5a5550,l3d=c090610 modal run --env staging experiments/modal_rust_ab.py::bench --sizes 14,18
modal run --env staging experiments/modal_rust_ab.py::cargo --cmd "check --release"
modal run --env staging experiments/modal_rust_engine.py::equality --size 14 --games 24 --ticks 10000 --procs 16
modal run --env staging experiments/modal_rust_engine.py::tests_sharded
```

`RUST_REVS` lists `name=git-rev` builds to load next to `work` (the working tree, or the commit in `RUST_WORK`);
`::cargo` runs `cargo check`/`cargo test` on the working tree with a cached target directory.
