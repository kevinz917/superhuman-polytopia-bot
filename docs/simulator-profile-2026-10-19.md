# Simulator step profile and lane-utilization fix (2026-10-19)

Follow-up to `docs/simulator-throughput-diagnosis-2026-10-18.md`. Goal: faster single-GPU game
generation for structured-RL PPO. All GPU numbers: one Modal L40S, JAX 0.11.2, parent checkpoint
`diverse-bc-dagger-v1/final.npz` (the parent of the live diverse league runs), opponent = same
network, native 14×14 maps from the 2048-map training pool. No multi-GPU work.

**Bottom line**

1. **Lane utilization fix (collector only, ready to use): 3.4× faster PPO updates at 512 lanes**
   in a real `repair.run` (105.5 → 31.1 s per 32,768-decision update; utilization 0.30 → 0.99),
   with per-lane PPO data bitwise identical to the old collector. In the live runs, whose old tail
   is shorter (~250 instead of ~450 steps), expect ~1.9× (60 → ~32 s/update).
2. **The step is not a fixed latency floor at ≥512 lanes. It is compute spent on discarded
   work.** `vmap(core.step)` runs every one of the 17 action branches for every lane, every
   step, then keeps one. At 512 lanes, `advance` is 186 of ~214 ms; diplomacy (42 ms) and
   destroy (11 ms) run every step although the policy never chose them in the sampled step.
   The network is ~5 ms. XLA flags do not help.
3. **Frozen-engine proposals (prototyped, bitwise-identical on CPU tests and on the GPU, NOT
   deployed, need approval): 5.2× faster engine step** (142 → 27 ms at 512 lanes) from vectorized
   unit removal, a no-op path search for non-moving lanes, lane-compacted step dispatch, compacted
   move masks and float32 build-mask products. Together with (1): **5,158–6,038 learner
   decisions/s at 512 lanes and 9,718/s at 2048 lanes, versus 312/s for the current collector
   on the same states (16–31×)**. Details in §5.

## 1. Per-stage profile of one collection step

Steady-state states (two warm rollouts, mean turn 14.7). Milliseconds per batched call, each stage
jitted alone, `block_until_ready`, 5 repetitions after warm-up
(`scripts/profile_collector.py stages`; run `profile-stages-v1`).

| Stage | 128 lanes | 512 lanes | 2048 lanes |
|---|---:|---:|---:|
| `Runtime.prepare` (observe + encode + legal masks) | 5.6 | 22.8 | 400.1 |
| · legal masks: move (196 sources × 8 relaxations) | 3.7 | 14.7 | 58.5 |
| · legal masks: build (whole-board) | 2.1 | 8.2 | 330.5 |
| · observe + encode | 0.6 | 0.6 | 1.1 |
| Network sample, one forward (`Runtime.act`) — called twice per step | 0.8 | 1.9 | 9.5 |
| Next-state seat value (observe + trunk) | 0.7 | 1.7 | 7.3 |
| **Engine `advance` (`vmap(core.step)`)** | **104.0** | **185.6** | **465.0** |
| Rewards, selects, buffer writes | <1 | <1 | <1 |
| Measured collector step (whole tick, in the loop) | — | 201.7 | 733–750 |

What `advance` spends its time on (each piece timed alone, vmapped over all lanes with that
step's real actions). Under `vmap`, `lax.switch`/`lax.cond` become "compute all, select one", so
every row below is paid every step regardless of the chosen action:

| `core.step` piece | 128 | 512 | 2048 | lanes that took it (of 512, sampled step) |
|---|---:|---:|---:|---:|
| diplomacy (`diplomacy.execute`: 196-step unit-removal loop + 2× road BFS) | 18.8 | 41.6 | 128.4 | 0 |
| capture (`capture.city`: 196-step elimination loop + 2× road BFS) | 16.9 | 36.8 | 113.8 | 11 |
| move (`_move_adjacent`: heap Dijkstra of nested while-loops) | 19.5 | 27.4 | 44.9 | 197 |
| · of which path search alone | 7.9 | 13.3 | 26.6 | |
| attack (incl. splash/stomp 196-step removal loops) | 23.9 | 27.2 | 34.8 | 54 |
| reward (city level-up while-loops) | 13.7 | 15.6 | 21.4 | 26 |
| build (+ road BFS) | 2.8 | 12.4 | 44.1 | 58 |
| destroy (+ road BFS) | 2.6 | 11.3 | 46.2 | 0 |
| validity: whole-board build mask for one entry | 2.2 | 7.9 | 331.1 | |
| research, summon, end turn, monuments, vision, meet, … | ~5 | ~7 | ~15 | |

Compiled structure (`collector_hlo`, `vmap(core.step)` HLO): 84 while-loops per tick (67 with
data-dependent trip counts, 7 fixed 196-iteration unit-removal loops), ~1,900 fusions. A device
trace of the collector shows ~33,000 GPU kernel/memcpy events per step and the GPU busy ~70% of
wall time; the largest single kernel is a state-transpose fusion (~25 ms/step).

Interpretation:

- **Network evaluation is not the problem.** Each step runs the network three times on every lane
  (learner sample, rival sample, next-state value): 1.9 + 1.9 + 1.7 ≈ 5.5 ms of ~214 ms at 512
  lanes (2.6%; 2.5% at 2048).
- **Small batches are latency-bound, large batches compute-bound.** 128 lanes cost 0.86 ms per
  lane-step, 512 lanes 0.41, 2048 lanes 0.42 (0.37 for `advance`, plus `build_mask` and move-mask
  blow-ups). Beyond 512 lanes there is nothing left to amortize, so "more lanes" does not raise
  decisions/s; the diagnosis' "90 ms fixed + 0.27 ms/lane" fit only holds up to ~512.
- The compute is mostly **branches whose results are thrown away** (diplomacy/destroy never taken;
  capture taken by 2% of lanes but computed for 100%), plus `build_mask`'s int32 matmuls, which
  scale super-linearly (8 ms → 330 ms from 512 → 2048 lanes) and run twice per step.

### XLA flags (no source change)

Run `profile-xlaflags-v1`, 512 lanes, 3 rollouts of 16 decisions from fresh states:

| XLA_FLAGS | ms/step | bit-identical |
|---|---:|---|
| default (command buffers for fusions/cuBLAS on) | 142.9 | — |
| `--xla_gpu_enable_command_buffer=` (off) | 286.5 | yes |
| + `CONDITIONAL,WHILE` command buffers | 146.1 | yes |
| + `CONDITIONAL,WHILE` and `--xla_gpu_graph_min_graph_size=1` | 146.1 | yes |
| `--xla_gpu_enable_while_loop_unrolling=AUTO_UNROLL` | 143.6 | yes |

The default already captures the launch-overhead win; nothing further from flags.

## 2. Lane-utilization fix: `collector_mode='decision-budget'`

`src/polytopia_bot/rl/async_league.py` (`decision_budget`, `lane_capacity`), wired through
`repair.run(collector_mode='decision-budget', lane_capacity=None)`. Collector-level only; no
pinned file changed.

- Every lane keeps playing until the lanes **together** commit exactly `environments × horizon`
  learner transitions (32,768 for the live recipe). No lane waits for the slowest one.
- A lane stores at most `lane_capacity` rows (default `2 × horizon`; observed max 123 of 128 at
  512 lanes). Rows per lane stay one contiguous stretch of that lane's game.
- GAE runs on the zero-padded `[rows, lanes]` buffer: unfilled rows are all zero, so they add
  nothing to any lane's recurrence (test proves equality with per-lane GAE on the unpadded
  segments, scalar and per-turn discounting). `compact_rollout` then gathers exactly the 32,768
  committed rows; minibatching, advantage normalization, replay audit and every `training.json`
  row field are computed on those rows as before.
- The last tick may complete more transitions than the budget needs; lower lane indices win and
  the rest are dropped. A learner decision whose opponent response is still running when the budget
  fills is played but not recorded (`batch.dropped_pending`). Its game continues in the next
  rollout, exactly like an ordinary rollout boundary; the previous transition bootstraps from the
  critic as before. Measured: 1.1% of decisions at 512 lanes, 5.8% at 2048.
- Zero-sum sign handling is unchanged (learner-seat rewards, `same_player = 1`), truncation
  bootstrap unchanged (pre-reset seat view), per-lane episode continuity unchanged.
- New stats in `batch`: `dropped_pending`, `saturated_lanes`, `max_lane_transitions`; all existing
  fields kept. `protocol.json` records the budget rule.
- Also: `league.check_collection` now tests finiteness on device instead of copying the whole
  rollout (3.5 GB of features at 512×64) to host every update.
- Also (decision-budget only): when the rival is the current policy, one policy forward per tick
  serves both seats; each lane samples with the key it would have used anyway, so actions are
  bitwise unchanged (test). Saves ~1.9 ms/step at 512 lanes.

Correctness tests (`tests/test_decision_budget.py`): bitwise-prefix equality with the per-lane
collector on real 11×11 native maps (features, choices, masks, logp, values, rewards, flags), exact
budget fill, GAE padding isolation, behavior log-prob replay ≤ 1e-5 on compacted rows, end-to-end
`repair.run` in both modes with matching `training.json` fields, shared-forward equivalence.

### Collector benchmark (one L40S, 32,768 learner decisions per rollout)

`scripts/profile_collector.py lanes`; runs `bench-lanes-512-1024-v1`, `bench-lanes-2048-4096-v1`.
Mean of 2 rollouts after one warm rollout; states warmed from the same checkpoint.

| Lanes | Mode | s/rollout | steps | ms/step | utilization | learner dec/s | all dec/s | dropped |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | per-lane 64 (current) | 105.1 | 496 | 212 | 0.30 | 312 | 735 | 0 |
| 512 | **decision budget** | **28.1** | 133 | 212 | **0.99** | **1,166** | 2,400 | 1.1% |
| 1024 | per-lane 32 | 183.5 | 367 | 500 | 0.21 | 179 | 432 | 0 |
| 1024 | decision budget | 31.4 | 67 | 472 | 0.99 | 1,044 | 2,146 | 2.3% |
| 2048 | per-lane 16 | 159.5 | 218 | 733 | 0.17 | 205 | 479 | 0 |
| 2048 | decision budget | 26.3 | 35 | 751 | 0.97 | 1,247 | 2,657 | 5.8% |
| 4096 | per-lane 8 | 148.7 | 103 | 1,450 | 0.18 | 220 | 495 | 0 |

(4096 decision-budget not reached within the 20-minute job budget.)

Reading: with the current engine, 512 lanes is the right width. Per-step cost grows linearly
beyond it (0.41 ms per lane), so wider batches only add dropped decisions and shorter per-lane
segments (16 rows per lane at 2048). The baseline here has a longer tail (~496 steps) than the
live runs (~250), because the warm states are earlier in the game; the budget mode's step count
(~133 = 32,768 / (512 lanes × 0.48 learner decisions per lane-step)) does not depend on the tail.
No uncredited terminal outcomes were observed in the budget rollouts.

## 3. End-to-end PPO updates (real `repair.run`, live recipe minus frozen rivals)

Run `profile-train-v1`: 512 lanes, horizon 64 (32,768 learner decisions per update), lr 1e-4,
entropy .005, gamma .995, lambda .99, minibatch 256, 2 epochs, current/historical self-play, 7
minutes of training per mode. Medians over updates 2+.

| Collector | Updates in 420 s | s/update | collection s | atomic steps | learner dec/s |
|---|---:|---:|---:|---:|---:|
| `independent-lanes` (current) | 4 | 105.5 | 104.2 | 455 | 311 |
| `decision-budget` | 12 | **31.1** | 30.0 | 122 | **1,052** |

All `training.json` row fields are present in both modes (identical key sets). The batch's
`dropped_pending` was a median of 415 of 32,768 decisions (1.3%). In **both** modes the flat replay
audit landed at 1.0–1.9e-5 on every update and the existing chunked re-check passed exactly
(0.0), so the stored `logp_error` is 0. This happens with the old collector too and comes from the
checkpoint and the flat-batch arithmetic, not from the collector change. Each such audit writes a
diagnostic sample and policy under `replay-audits/`. That costs volume space in live runs and is
worth a separate look.

## 4. Recommendation for live runs

- Switch new PPO runs to `collector_mode='decision-budget'` at 512 lanes: ~1.9× per update in the
  live recipe, no engine change, no change to the PPO batch definition (still
  environments × horizon learner transitions). The launcher (`experiments/modal_structured_scale.py`)
  still hard-codes `independent-lanes`; exposing the mode there is a one-line change I have not made,
  so no running or future campaign job changes without your decision.
- Do not go wider than 512 lanes with the current engine. With the engine proposals, 1024–2048
  lanes do pay off (6,556/s at 1024, 9,718/s at 2048), at the cost of shorter per-lane segments and
  more dropped decisions (2.3% / 5.8%).
- Multi-GPU is still an N× multiplier on top of this, now from a 2–10× higher single-GPU base.

## 5. Frozen-engine proposals (prototypes only; need the user's approval)

`src/polytopia_bot/proposals/fast_engine.py` holds opt-in replacements that take effect only inside
`fast_engine.patched(...)` while functions are being traced. No frozen file is edited
(`scripts/verify_ruleset.py`: all 22 hashes match), nothing in `rl/` imports the module, and no campaign run
uses it. Adopting any of them means editing the frozen sources (and the ruleset manifest) the same
way, which needs your explicit approval.

| Proposal | What changes | Why it is exact |
|---|---|---|
| `removal` | `capture.city` (elimination), `military.area_damage` (splash/stomp) and `diplomacy.execute` (broken peace) remove units in one vectorized step instead of a 196-iteration `fori_loop` of `cond(remove)` | each removal only adds integers to score/casualties/kills and writes its own tile; no removal reads another removed tile |
| `path` | `_move_adjacent` gets source = target = 0 on lanes whose action is not a move | the heap search then ends on its first pop; those lanes' branch output is discarded by the select anyway |
| `dispatch` / `compact` | a batch-level copy of `core.step`'s dispatcher runs each branch (and the whole-board build validity mask) only if some lane takes it; `compact` also gathers just those lanes into a 1/32, 1/8 or 1/2 bucket, then scatters back | per-lane branch arithmetic is the unchanged frozen function; only work `vmap` would have discarded is skipped |
| `masks` | move masks are computed only for tiles holding the mover's unmoved units (bucket 16/32/64, else all 196) | `movement_targets` returns an all-False row for every other source |
| `buildmask` | `build_mask`'s three 0/1 int32 matmuls run in float32 | sums of at most 196 ones are exact in float32, so `> 0` is identical |

Verification: `tests/test_fast_engine.py` (18 tests, CPU, 11×11 native maps from random play)
compares states, observations and masks bitwise over 120-step trajectories with every action
family, plus targeted cases (forced eliminations, lethal splash, broken peace, mask buckets and
the all-tile fallback, dense random territories). On the GPU, `profile_collector.py flags`
compares SHA-256 digests of three full collector rollouts (states and collected data) per
variant: **all variants identical to the original engine.**

Measured, one L40S, 512 lanes, 16-decision rollouts from fresh games (runs `profile-engine-v1/-v2`):

| Engine variant | ms/step | speedup |
|---|---:|---:|
| original | 142–143 | 1.0× |
| `buildmask` | 137.0 | 1.04× |
| `dispatch` | 127.8 | 1.1× |
| `compact` | 112.5 | 1.3× |
| `removal` + `path` | 54.9 | 2.6× |
| `removal` + `path` + `compact` + `masks` | 35.7 | 4.0× |
| **all five** | **27.3** | **5.2×** |

Combined with the decision-budget collector and the shared-forward option (runs
`bench-lanes-proposals-v1/-v2`, same warm states as §2):

| Lanes | Proposals | s/rollout | ms/step | learner dec/s | vs current collector |
|---:|---|---:|---:|---:|---:|
| 512 | four (no buildmask) | 6.35 | 47.9 | 5,158 | 16.5× |
| 512 | all five | 5.43 | 41.0 | 6,038 | 19.4× |
| 1024 | four | 5.00 | 75.2 | 6,556 | 37× (vs 179) |
| 2048 | all five | 3.37 | 96.3 | 9,718 | 47× (vs 205) |

Step counts and dropped decisions match the unpatched budget runs exactly (132.5 / 375.5 at 512),
as expected from bit-identical trajectories. At ~1.2 s of gradient work per update, all five
proposals plus the budget collector would put a 512-lane update near 6.6 s (vs 60 s live today).

Compile time grows with the compacted dispatch (collector compile 31 s → 122–273 s), paid once
per run.

## 6. Reproducing

- `scripts/profile_collector.py {stages,lanes,flags,train}`. It runs locally on CPU with `--size 11
  --pool 4 --warm-lanes 4`, and on Modal through `experiments/modal_collector_profile.py`
  (staging only, reserves $7 per job through `rl/budget.py`, 1,200 s budget, 7,200 s hard cap,
  run ids `profile-*` / `bench-lanes-*`).
- Modal runs from this study (all terminally accounted in the ledger, about $14.50 in total):
  `profile-stages-v1`, `profile-xlaflags-v1`, `bench-lanes-512-1024-v1`, `bench-lanes-2048-4096-v1`,
  `profile-engine-v1`, `profile-engine-v2`, `bench-lanes-proposals-v1`, `bench-lanes-proposals-v2`,
  `profile-train-v1`. Raw JSON is on the `polytopia-structured-rl` volume under each run id.
- The multi-GPU session independently fit ~0.10 s fixed + 0.25 ms per game per step from its
  128- and 512-game runs; that agrees with §1 between 128 and 512 lanes. The 2048-lane point
  shows the per-lane term dominating beyond 512.

## 7. Adoption for training (2026-10-18 evening): fast engine end to end

The user chose collector + fast engine **for training only**. Wiring (main checkout, uncommitted):
`repair.run(engine='frozen'|'fast')` builds a second `Runtime` inside
`fast_engine.patched(*TRAINING_PROPOSALS)` (all five) and traces/compiles only the training
collectors in that context. The probe, health checks and every evaluation keep the frozen
`Runtime`. `protocol.json` records `engine` (name `fast-training-v1`, proposal list, source hash,
equivalence evidence) and copies `proposals/` into `source/`. The launcher adds `--engine`
(default `fast` for new PPO runs; bootstrap runs stay frozen). No frozen or pinned file changed;
`scripts/verify_ruleset.py` still passes. New test: same-seed `repair.run` with `fast` and `frozen` gives
identical `training.json` rows and identical evaluations. `repair.run` rows now also carry
`phase_seconds` (collection, advantages, replay_audit, gradient, diagnostics), and checkpoint
writes are logged with their duration.

### End-to-end, real diverse recipe (one L40S)

Parent `diverse-bc-dagger-v1`; 25% frozen rivals (the 8 live-recipe rivals), otherwise 50/50
current/historical self-play; lr 1e-4, entropy .005, gamma .995, lambda .99, minibatch 256,
2 epochs, 32,768 learner decisions per update, decision-budget collector. Medians over updates 2+
(runs `profile-train-diverse-v1`, `profile-train-diverse-wide-v1`).

| Engine | Lanes | s/update | collection | replay audit | gradient | learner dec/s | collector compile | dropped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| live runs before today (old collector, frozen) | 512 | 60–67 | 56–61 | — | — | ~500 | ~60 s | 0 |
| frozen | 512 | 30.7 | 29.7 | 0.75 | 0.26 | 1,066 | 56 s | 1.1% |
| **fast** | 512 | **7.2** | **6.5** | 0.14 | 0.34 | **4,538** | 144 s | 1.3% |
| fast | 1024 | 5.5 | 3.4 | 0.63 | 0.48 | 5,998 | 168 s | 2.1% |
| fast | 2048 | 5.4 | 3.1 | 1.64 | 0.69 | 6,117 | 187 s | 3.8% |

No invalid actions or loop exhaustions. Frozen rivals rotated in as configured. Replay audit
passed everywhere (max stored `logp_error` 9.9e-6; on most updates the flat replay is 1.0–1.9e-5
and the chunked re-check is exactly 0, as with the frozen engine). Checkpoint writes take ~0.2 s.
Evaluation ran on the frozen engine.

### GPU bit-identity with frozen rivals in the loop

Run `profile-digest-rivals-v1`: from the same warmed 512-lane states and key, nine consecutive
decision-budget rollouts — one against each of the 8 frozen rivals, then current-policy
self-play with the shared forward — produced **identical SHA-256 digests** (final states, seats,
collected data and stats) on the frozen and fast engines. Rollout time grew with game length:
frozen 24 → 41 s, fast 4.8 → 11 s (5.1× early, 3.7× later; 4.3× over the nine).

### What is left at the new operating point (512 lanes, fast engine)

Run `profile-stages-fast-v1`, plus the phase timings above. Per update (7.2 s): collection
6.5 s (90%), gradient 0.34 s, replay audit 0.14–1.6 s, advantages 0.1 s, checkpoint ~0.2 s every
16 updates.

Inside a collection step (36.7 ms measured in the loop at turn ~15; ~52 ms averaged over training, 6.5 s / ~125 steps):

- **The GPU is now idle more than half the time.** A device trace shows ~9,800 kernel/memcpy
  events per step (was ~33,400) with the GPU busy 45% of wall time (was ~70%). Collection is
  again bound by launch latency and serial loops, not arithmetic.
- The ~25 ms state-transpose fusion is down to ~3 ms/step; the network's convolutions ~2.8 ms/step;
  while-loop predicate copies (`memcpy32_post`) ~2.7 ms/step.
- Remaining serial work: the heap-ordered path search for move lanes (path search alone 14.8 ms in
  isolation at 512 lanes), road-network BFS inside capture/build/destroy, and the compacted
  dispatcher's batch-level conditionals, each with a host round trip.
- Network: three passes per step (learner, rival, next-state value), ~3–4 ms each at 512 lanes
  in isolation, under ~15% of a step. The shared forward removes one when the rival is the
  current policy.
- Steps get slower as games grow (more units, larger buckets): fast rollouts doubled from turn ~10
  to later play.

### Ranked next bottlenecks (estimates)

| # | Item | Est. gain | Effort / risk |
|---|---|---|---|
| 1 | **Multi-GPU data parallel** (branch `claude/practical-boyd-62b6f5`, not merged): one collector per GPU, gradients averaged with `pmean` | ~N× decisions/s: 4×L40S ≈ 15–20k learner dec/s at 512–1024 lanes/GPU (it measured 3.69× at 4×512 with the slow engine) | medium: port decision-budget + fast-engine tracing into its pmap path; cost per decision unchanged |
| 2 | **Use 1024 lanes** with the fast engine | +32% (4.5k → 6.0k dec/s) | trivial; 2.1% dropped decisions and 32 rows per lane per update |
| 3 | **Replay audit**: the flat replay lands just above 1e-5 on most updates and writes a diagnostic sample plus policy checkpoint each time (0.6–1.6 s at 1024–2048 lanes, up to 30% of an update; also volume space); the chunked (collection-shaped) replay is exact | 10–30% per update at ≥1024 lanes | small; an audit-policy decision (e.g. chunked replay as the primary check, or keep diagnostics but write them rarely) |
| 4 | Engine latency: an exact parallel replacement for the heap path search; incremental road-connectivity instead of BFS per capture/build/destroy | maybe 1.5–2× collection | high: must reproduce Rust BinaryHeap tie-breaking bit for bit |
| 5 | Two collectors interleaved per GPU (different streams/processes) to fill the 55% idle time | up to ~1.5–1.8× | medium; needs a learner-side merge |
| 6 | Compile time 144–187 s per launch (fast engine) | minutes per launch only | small: persistent compilation cache on the volume |
| 7 | Frozen-rival switching | none measured | rival params are arguments; no recompiles |

Modal jobs for §7 (all terminally accounted): `profile-train-diverse-v1` $1.62,
`profile-train-diverse-wide-v1` $1.58, `profile-digest-rivals-v1` $1.61, `profile-stages-fast-v1` $1.79.
