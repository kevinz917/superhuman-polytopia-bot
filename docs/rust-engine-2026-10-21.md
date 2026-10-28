# Rust port of the actor engine: cost and exactness vs the JAX CPU engine (2026-10-21)

**Question.** What does a hand-written step cost versus the XLA one the actors run today
(`rl/cpu_engine.py::CpuGames`, fast=True), and does it reproduce the frozen rules exactly?

**Answer.** On the same Modal CPU host, 24 games on 14x14, one pinned core, mid-game states
(mean turn 46): the Rust full tick takes **3.5–3.7 ms** against **45–60 ms** for the JAX fast
CPU engine (**12–17x**). The step alone is 28–41x faster, the legal masks 8–12x, observe+pack
2–4x. The port is **bitwise equal** to the JAX engine on every State field, every legal mask
and the packed public views after every one of **12.5 million lockstep steps** across
15,000 maps on 11x11, 14x14 and 18x18 boards (classic generator and the variant-map mix),
under uniform-random, combat-biased and public-heuristic play, against both the fast CPU build
and the frozen engine, through elimination, resignation and the round cap, with auto-reset. No rule
divergence was found; the only two bugs the harness caught were in the Python glue
(a packed-view width constant and the mask reshape).

Code: `engine-rs/` (crate + PyO3 bindings, 3.9k lines of Rust), `src/polytopia_bot/rl/rust_engine.py`
(CpuGames-shaped wrapper), `tests/test_rust_engine.py` (lockstep harness, also a CLI),
`scripts/bench_rust_engine.py`, `experiments/modal_rust_engine.py`. Nothing under `gpu/` changed.

## What was ported

The native 1v1 ruleset the actors use (`load_rules()` default profile; the `polyfish`
differential profile is not ported, every `rules.native` branch is specialised to True):

| JAX source | Rust module | Notes |
|---|---|---|
| `core.step`, `_end_turn`, `_meet`, `_research`, `_recover`, `_move_adjacent`, `_resign`, `terminal_rewards`; `actions.decode` after `rl.actions.flatten_choice` | `core.rs` | status codes identical (OK/INVALID/UNSUPPORTED/STALE/DONE) |
| `actions.legal_masks` (with `fast_engine._legal_masks_with_move`'s move rows) | `masks.rs` | one flat bool buffer in catalog order; `resign` forced False as in CpuGames(no_resign=True) |
| `kernels`: combat, max_health, research, movement_data/targets, update_exploration, observe | `kernels.rs` | observe writes the packed int16 view directly (`cpu_engine.pack_view` layout, sorted keys) |
| `military`, `economy`, `structures`, `capture`, `diplomacy`, `infiltration`, `pathfinding`, `discovery` | same names | including the reference heap ordering in path replay, the .NET-style push tie-breaks, the seeded explorer and ruin draws, Cloak infiltration spawn order |
| `mapgen.layout/reset`, `random` (.NET System.Random), `discovery.hash_*` (xxHash32), `maps.state.initial_state` (map_batch path) | `mapgen.rs`, `hash.rs` | seeds and per-lane layouts both supported |
| `rules.load_rules` | `rules.rs` | parses the same `gpu/rules.json` (serde), checks `ENGINE_REVISION` |

API (`polytopia_engine_rs.GameBatch`, wrapped by `RustGames`): `reset(seeds)` /
`reset_layouts(...)`, `tick(family, source, target, seeds, both_views, layouts=None)` returning
masks `[G, 130868]` u8, packed views `[G, 7649]` int16 for the mover and the other seat, done /
terminated / truncated, native rewards `[G, 2]`, status, and both seats' packed views of a
truncated final state (the learner's `final_features` are `view_features` of those);
`step`, `legal_masks`, `observe`, `terminal_rewards`; `export()` / `import_()` move every State
field as NumPy arrays with the JAX layout; `profile_masks`, `coverage` for diagnostics.

Left in JAX on purpose: `encode_view` (packed view to float features; 0.8–1.6 ms per 24 games,
now the largest remaining piece of a Rust-engine tick) and the policy itself. Not ported: the
`polyfish` rules profile, `fog=False` imports (`fog` is honoured but never False in actor
games), `core_supported=False` imports.

Integer and float semantics follow XLA: int32 wrapping arithmetic, the f32 combat formula in the
same operation order (`af/divisor*aa*4.5 + 1e-6 + 0.5` then floor), `.at[-1]` wrap-around on the
two player-index gathers that can see owner 0, floor division where JAX uses `//`. Movement
reachability runs the same 8 synchronous relaxation sweeps (stopping early only once a sweep
changes nothing, which is exact). Every array access is bounds-checked (plain Rust indexing;
`overflow-checks` off in release like XLA).

## Speed (Modal CPU container, one pinned core, 24 games, median of 25)

JAX = `CpuGames(fast=True, both_views=True)` and its pieces timed with the same `lax.map` shape
as `scripts/bench_tick_phases.py`. Both engines were run on the same states (warmed with 400
ticks of random legal play, mean turn 46, ~8 units per game). Two runs landed on two different
hosts (Modal reports no CPU model), so the ratios within a row are the comparison:

| 14x14, ms per 24-game tick | Rust | JAX (host A) | JAX (host B) | ratio |
|---|---|---|---|---|
| step (`core.step` after decode) | 0.53–0.56 | 15.8 | 21.7 | 28–41x |
| legal masks | 2.4–2.6 | 20.6 | 29.5 | 8–12x |
| observe + pack, both seats | 0.31–0.44 | 0.89 | 1.65 | 2–4x |
| **full tick** (step, auto-reset, masks, both views) | **3.5–3.7** | **44.8** | **60.1** | **12–17x** |
| full tick + JAX `encode_view` from the packed view | 4.6–5.5 | (included above) | | 8–13x |
| 2026-10-21 reference (`docs/throughput-research-2026-10-21.md`) | | 47 (step 18, masks 23, observe+encode+pack 2.6) | | |

| 18x18, ms per 24-game tick | Rust | JAX (host A) | JAX (host B) | ratio |
|---|---|---|---|---|
| step | 0.71–0.85 | 33.0 | 46.3 | 39–65x |
| legal masks | 3.8–3.9 | 44.9 | 58.6 | 11–16x |
| observe + pack, both seats | 0.49–0.65 | 2.0 | 2.7 | 4x |
| **full tick** | **5.3–5.4** | **92.9** | **122.4** | **17–23x** |

Where the Rust tick goes (14x14, `profile_masks`, ms per 24 games): build mask 0.76 (was 1.29
before skipping tiles outside the mover's economy), move rows 0.32–0.45, research 0.13–0.16
(per-tech cost recomputes the city count), observe 0.16–0.19, attack rows 0.02, summon 0.03,
harvest/forest/reward 0.04–0.06; the mask families sum to ~1.5 ms and `legal_masks` as a
whole to 2.3–2.5 ms, the difference being the 131 KB-per-game output buffer (zeroing plus
per-entry writes); step 0.5 ms; auto-reset (a full map generation) is amortised over ~800 ticks.
Host-side extras per tick: NumPy mask split/reshape 0.45 ms, `export()` 0.1 ms, `import_()`
3–4 ms (harness only).

Size: the abi3 wheel is 471 KB (release, fat LTO). The crate is size-generic at runtime
(11/14/18 supported by the map generator; any size for stepping).

## Equality evidence

`tests/test_rust_engine.py` plays both engines in lockstep: actions are sampled uniformly from
the JAX legal masks (end-turn boosted to 12 % so games reach the round cap; one forced
resignation every 97 ticks), both engines take the same choices and the same reset seeds or
layouts, and after every tick the harness asserts bitwise equality of status, done /
terminated / truncated, native rewards, all 72 State fields (dtype and value), all 17 mask
families, the mover's and the other seat's packed views, and (allclose, 1e-6) the
`final_features` of truncated games. The first difference is reported with field, index and
both values.

Runs on Modal (`experiments/modal_rust_engine.py::equality`, 16 vCPU containers, one process per
core, 24 games per process; all passed):

| board | maps | processes x ticks | steps | games finished | by elimination/resign | by round cap | maps seen |
|---|---|---|---|---|---|---|---|
| 14x14 | classic (mapgen.reset) | 16 x 10,000 | 3,840,000 | 4,708 | 1,689 | 3,019 | 5,092 |
| 18x18 | classic | 8 x 8,000 | 1,536,000 | 1,256 | 661 | 595 | 1,448 |
| 11x11 | classic | 4 x 6,000 | 576,000 | 697 | 273 | 424 | 793 |
| 14x14 | `uniform` variant mix, map_batch path | 4 x 1,500 | 96,000 | 104 | 60 | 44 | 168 |
| 14x14 | `uniform` variant mix, map_batch path | 16 x 4,000 | 1,536,000 | 1,759 | 671 | 1,088 | 2,143 |

Second round (policy-driven play, same harness; `--policy heuristic` takes the public heuristic
opponent's choice 60 % of the time and a combat-biased draw otherwise, `--policy combat` draws an
attack/capture/recover/promote/heal 60 % of the time when one is legal; all passed):

| board | maps | play | reference | processes x ticks | steps | games finished | by elimination/resign | by round cap | maps seen |
|---|---|---|---|---|---|---|---|---|---|
| 14x14 | classic | heuristic | fast | 16 x 4,000 | 1,536,000 | 1,520 | 1,517 | 3 | 1,904 |
| 14x14 | `uniform` | heuristic | fast | 16 x 3,000 | 1,152,000 | 936 | 916 | 20 | 1,320 |
| 18x18 | classic | heuristic | fast | 8 x 3,000 | 576,000 | 284 | 284 | 0 | 476 |
| 14x14 | classic | combat | **frozen (fast=False)** | 8 x 2,000 | 384,000 | 326 | 187 | 139 | 518 |
| 14x14 | classic | random, end-turn never boosted, no resigns | fast | 4 x 10,500 | 1,008,000 | 649 | 21 | 628 | 745 |
| 11x11 | `uniform` | combat | fast | 4 x 3,000 | 288,000 | 282 | 177 | 105 | 378 |

Coverage under heuristic play on 14x14 classic (1.54 M steps): 133,081 attacks, 18,459 attackers
killed by retaliation, 31,253 advances after a kill, 20,519 captures of which 8,113 enemy cities,
861 eliminations, 1,811 promotions, 1,183 heals, 283 Cloak infiltrations, 163 Mind Bender
conversions, 409 Stomps, 115 Splash attacks, 1,677 ruin super units, 4,373 diplomacy actions,
656 resignations. The variant-map heuristic run adds 2,314 Stomps, 929 Splashes and 154 kills by
area damage; the frozen-engine run covers 11,630 attacks, 128 conversions and 75 infiltrations
against `CpuGames(fast=False)`.

Action mix on 14x14 classic (3.84 M steps): move 1.57 M, end turn 799 k, build 596 k, forest
207 k, research 180 k, summon 119 k, diplomacy 99 k, disband 79 k, destroy 71 k, harvest 45 k,
city reward 40 k, attack 24 k, capture 5.6 k, recover 4.9 k, resign 1.6 k, promote 33, heal 33.

Branch coverage counters from the Rust step on the 1.54 M-step variant-map run (every counter
below was also compared bitwise against JAX in that run): move 643,726; end turn 328,616;
build 190,985; forest 81,745; research 76,076; summon 57,105; disband 38,375; diplomacy 35,308
(all five abilities reachable through the mask); destroy 31,150; harvest 17,736; city reward
17,295 (explorer 3,261, super unit 1,379); embark/disembark 13,583; attack 9,056; capture
2,635 (enemy cities 53, eliminations 15); recover 1,919; raft upgrade 1,916; round-cap
truncations 1,088; advance after a kill 917; Stomp 916; push 885; attacker killed by
retaliation 814; resign 656; ruins 1,524 (population 373, unit 366, tech 350, stars 330,
explore 105); hidden-unit encounter on a move path 139; Mind Bender convert 67; starfish 46;
Cloak infiltration 33; promote 17; Splash 17; heal others 14; kills by area damage 5.

Not reached in any run (so equal only by construction, not by evidence): truncation by the
10,000-action cap. Even with end-turn never boosted and no resignations, uniform play ends every
14x14 game at the round cap after about 1,550 actions, so the action cap is unreachable without
changing `core.step`'s default, which `CpuGames` does not expose. Random uniform play attacks
rarely (0.6 % of actions); a policy-driven soak (actions from a trained
checkpoint instead of uniform sampling) would cover combat-heavy paths at a far higher rate and
is the obvious next harness run.

## Repository test suite with the wheel installed

`experiments/modal_rust_engine.py::tests_sharded` runs every `tests/test_*.py` as its own pytest
process inside the image that has the Rust wheel: 2,318 tests passed, 7 files skipped entirely
(trainer/serving tests that need their own fixtures), 0 failures. The first attempt showed three
image gaps unrelated to the port (missing `zstandard`, `experiments/`, `configs/` and `scripts/`
on the path), which also made `test_distributed_rl.py::test_wire_roundtrip` hang: its sender
thread died on the missing compression import while the receiver blocked on the socket. With
those mounted the four affected files pass (39 tests). `tests/test_rust_engine.py` skips cleanly
when the wheel is not installed, so a plain `pytest` on a machine without the crate is unaffected.

## Default actor engine (merged on top of Stage A)

`rl/rust_engine.py::RustGames` is a drop-in for the post-Stage-A `CpuGames` contract: `reset(seeds | MapBatch)`,
`tick(states, choices, seeds, check=False) -> (Tick, status)` with `Tick.states` (NumPy export of every field,
what the actor reads for score and shaping), `prepared.features` (the JAX `view_features` of the packed view,
the same encode the learner uses), masks, both packed views, `final_views` and, when a game truncates,
`final_features`. Every action is fully validated in Rust, so `check` is accepted and ignored (the trusted step
is a JAX-side cost that Rust does not have). `LearnerConfig.engine='rust'` is the default and the only
training engine: an actor fails fast if the `polytopia_engine_rs` wheel is missing (the actor container reports
`engine` in its summary). `'jax'` (`CpuGames`) is retired for training and kept as the frozen reference for the
lockstep harness, evaluation and A/B debugging. `experiments/modal_distributed_rl.py`'s actor image installs a Rust toolchain and builds the wheel with
maturin at image-build time; the learner image is unchanged.

Gates run on the merged tree (Stage A branch + main + this branch), all on Modal:

- lockstep harness against the Stage A `CpuGames(fast=True)` (trusted step with the full check on every 7th tick,
  build rows, both views), heuristic play on 14x14 classic (8 x 1,500 ticks, 288,000 steps, 30 game endings) and random play on the
  variant mix (4 x 1,000 ticks, 96,000 steps, 40 round-cap truncations): states, masks, views, `final_views`,
  policy features and `final_features` all equal. (The first attempt caught one drop-in detail: CpuGames leaves
  the final features of lanes that did not truncate at zero, not the encode of a zero view; fixed.)
- the whole pytest suite, one process per file: STAGEA_PYTEST
- a 10-minute end-to-end training smoke of the c5 recipe (`engine='rust'`, the default, 1xH100 learner, 2x16-vCPU actor
  containers of 16 processes x 24 games, proposed-v1 maps, omniscient critic, ledger `runs/rust-engine`):
  run `rust-engine-smoke-1` (Modal app `ap-enJgGbWh42qqMHr2bGV8DW`, $7.11/h): both actor containers report
  `engine: rust`, 32 processes connected and compiled, 39 learner updates (10.2 M samples) in 627 s at a
  steady 17.7 k samples/s from two containers (Stage A fed 37 k from twelve), 10.66 M environment steps in
  ~635 s = ~525 steps/s per process (Stage A production-shape 399, c5 180; the policy forward is now the
  actor's main cost), no errors, no illegal-action raise, no restarts; the learner's KL, entropy and value
  statistics look like the c5 recipe's first updates. Spend about $1.3.

## Lines of code

| | lines |
|---|---|
| Rust engine (`engine-rs/src`, 17 modules incl. bindings and coverage) | 3,952 |
| JAX modules ported (`gpu/{core,actions,kernels,military,economy,structures,capture,diplomacy,infiltration,pathfinding,discovery,mapgen,random,rules,state}.py`) | 2,160 |
| Python glue (`rl/rust_engine.py`, harness, bench, Modal runner) | 497 |

## Recommendation

Put it behind the actor's engine contract as an opt-in (`engine='rust'` next to
`cpu_engine.CpuGames fast=True`), gated on the lockstep harness, but do not switch training to it
yet:

1. **Today the win is bounded.** The 2026-10-21 actor profile put the engine at about a third
   of the actor tick with the policy and critic forwards taking the rest; replacing 47 ms with
   ~5 ms (Rust tick plus JAX feature encoding) speeds the CPU actor by roughly 1.5x. The sibling
   JAX fix (one build mask instead of two, ~13 ms engine) captures about half of that without a
   second engine to maintain.
2. **After network inference moves to the GPU** the engine is the whole CPU cost of an actor,
   and 12–17x per core (17–23x on 18x18) is the difference between ~0.4 and ~5 ms of CPU per
   24-game decision: the same actor fleet feeds an order of magnitude more games to the GPU, or
   the fleet shrinks accordingly. That is when the port pays for itself.
3. **Before switching:** (a) optionally repeat the soak with a trained checkpoint's actions (the
   public heuristic already covers combat, promotion, conversion and infiltration at realistic
   rates); (b) decide where
   `encode_view` lives (JAX from the packed view at ~1 ms, a Rust port, or on the GPU with the
   network); (c) add the harness to the Modal check set so any rules change in `gpu/` trips it;
   (d) the Studio/WebAssembly build is a separate target (no PyO3) and needs the same crate
   compiled without the bindings module, which the `rlib` crate type already allows.

Remaining Rust levers if needed: bit-packed or caller-provided mask buffers (~0.8 ms of the
3.5 ms tick is writing the 131 KB mask array per game), a per-player cached city count in the
research mask, and incremental territory/economy sets; none was needed for this answer.

## How to run

```bash
modal run --env staging experiments/modal_rust_engine.py::equality --size 14 --games 24 --ticks 10000 --procs 16
modal run --env staging experiments/modal_rust_engine.py::equality --size 14 --games 24 --ticks 4000 --procs 16 --maps uniform
modal run --env staging experiments/modal_rust_engine.py::bench --sizes 14,18 --games 24 --warm 400 --repeats 25
```

The image installs a Rust toolchain, builds the crate with `maturin build --release` (abi3
wheel) and runs the harness or the bench pinned to one core. Locally, `pytest tests/test_rust_engine.py`
needs the wheel installed (`cd engine-rs && maturin develop --release`); the pytest default is
a short run (8 games, 150 ticks).
