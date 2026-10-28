# Where the 6-hour omniscient-critic run spent its time, and what bounds the sampling rate (2026-10-21)

Diagnosis of `c5-omni-s1` (the 6-h 14×14 run of 2026-10-20/21: 1×H100 learner, 12 × 16-vCPU actor containers of
16 processes × 24 games, omniscient critic, `proposed-v1` maps, 262,144-sample batches, 3 epochs). Sources: the
run's `training.json` (2,212 per-update rows, from the retained checkpoint folder), `bench_learner` on one H100
with and without the critic, and a production-shaped per-phase actor profile (`scripts/bench_actor_profile.py`,
new, run through `experiments/modal_actor_profile.py`; both copied into this checkout uncommitted; they need `main`'s omniscient-critic code to run).
Diagnostic spend ≈ $1.5 on Modal staging. Nothing in the training path was changed.

## 1. The run was actor-bound: the GPU sat idle 37% of the time

Steady state (updates 61–2,212, after the 60 critic warm-up updates):

| Quantity | Value |
|---|---|
| Wall time per update | 9.72 s (9.35 s timed + 0.37 s untimed bookkeeping) |
| · GPU training (`train_seconds`, 3 epochs, incl. host→device copy) | 5.78 s (59%) |
| · learner waiting for a batch (`ready_wait_seconds`) | 3.4 s (35%); 4.4–4.7 s in the first hour, 3.2 s in the last |
| Inbox occupancy | 0 in every status row (never backed up) |
| Age of a batch when training starts | 0.01 s (consumed the moment it is assembled) |
| Samples consumed | 262,144 / 9.72 s ≈ 27k/s (580M in 6 h) |
| Samples produced (192 processes, 30.9k steps/s, 90.8% learner decisions) | ≈ 28k/s |
| Policy lag of trained samples | mean 1.46 updates, max 5–7 (max_lag 10), ~100 stale samples dropped per update |
| Epoch early-stops on `target_kl` | none (3 epochs in all 2,152 updates) |
| Per-actor rate | 2,575 steps/s per 16-vCPU container = 161 steps/s per vCPU = 149 ms per 24-game tick |
| Checkpoint + volume commit every 10 updates | hidden inside the wait (0.09 s visible) |

The learner never had a second batch waiting; it finished an update and then waited ~3.4 s for the actors to
finish the next one. The actor fleet, not the GPU, set the sampling rate. This is the opposite of the 18×18 pilot
(`docs/rl-18x18-bottlenecks-2026-10-20.md`), which was learner-bound at fp32 on a grown network: TF32 on the
learner and the smaller 14×14 network moved the wall to the actors.

Two consequences:

- The GPU could consume 262,144 / 5.78 s ≈ 45k samples/s as configured, so 1.65× more actor throughput would
  be absorbed by the same learner with no change. Beyond that the learner becomes the wall again (section 3).
- The 60 warm-up updates cost 11.2 s each (full learner pass, policy gradient zeroed): 11 min = 3% of the run.

## 2. Actors: three network forwards are two thirds of every tick; the engine is one third

Per-phase profile of one 24-game tick in the production shape (omniscient critic, `proposed-v1` maps, a pool
opponent on 20% of lanes), 16 pinned processes on a 16-vCPU container (`bench_actor_profile.py`, 45–60 s per arm):

| Arm (16 processes, 45 s, two containers agreeing within 2%) | Container steps/s | per vCPU | ms per tick | policy fwd | critic fwd (+ opp-view decode) | rival fwd | engine tick | bookkeeping |
|---|---|---|---|---|---|---|---|---|
| classic maps, no critic, no rival (the old bench shape) | 4,630 | 290 | 83 | 30.1 | — | — | 48.3 | 2.9 |
| + `proposed-v1` maps | 4,500 | 282 | 85 | 30.0 | — | — | 50.5 | 3.3 |
| + omniscient critic | 3,085 | 193 | 124 | 30.1 | 38.6 | — | 50.6 | 3.6 |
| **+ pool rival on 20% of lanes (production shape)** | **2,695** | **169** | **141** | **30.7 (22%)** | **39.4 (28%)** | **15.9 (11%)** | **51.3 (36%)** | **3.5 (3%)** |
| production shape, 48 games per process | 2,700 | 169 | 284 | 58.7 | 76.3 | 47.7 | 94.9 | 4.3 |
| production shape, 96 games per process | 2,395 | 150 | 643 | 120 | 152 | 178 | 183 | 5.8 |

The production-shape arm reproduces the run (161 steps/s per vCPU, 149 ms per tick in the run; 169 and 141 here).
Per sample that is ~5.9 ms of core time: policy 1.3 ms, critic 1.6 ms, rival 0.7 ms, engine 2.1 ms, bookkeeping
0.15 ms. The three forwards are 60% of the tick, the engine 36%.

Reading of the shares (they are stable across hosts; absolute speeds are not, see below):

1. **The omniscient critic forward is the single largest piece, bigger than the policy forward itself** (+39 ms
   on an 85 ms tick, −32% throughput): it runs the full critic trunk (stem with doubled input channels) on all 24
   lanes every tick, and first decodes the opponent's int16 view into float features on the CPU (`opp_features`),
   work the engine already did for the own view. Observing and packing the second seat's view inside the engine
   costs nothing measurable (50.5 → 50.6 ms). The ablation readout saw the same loss from the outside (22k vs 35k
   samples/s for the control on the same fleet).
2. **The pool-rival forward costs half a policy forward for a tenth of the decisions** (+16 ms; the rival moves
   on ~2–3 of 24 lanes per tick): it is a separate jitted call on a 4-lane bucket, so the per-call cost (dispatch,
   weight reads, tiny matmuls) dominates. `proposed-v1` map generation costs 2 ms per tick (2%).
3. **The policy forward** on 24 lanes is ~1.2 ms per sample: 89 MFLOP per sample at ~70 GFLOP/s on one core,
   i.e. close to the single-core fp32 peak. CPU inference is compute-bound; there is no 2× left in it on CPU.
4. **The engine tick** (step + legal masks + observe + encode) is ~2.1 ms per game. From the 18×18 per-piece
   profile the structures `build_mask` is ~31% of it at 14×14 and `core.step` ~34%.
6. **Games per process is not a lever**: 48 games per process give the same container throughput as 24 (every
   phase scales linearly, so the per-call overhead that would amortize is already small), and 96 are 11% worse
   (the rival call grows superlinearly). Memory is 2.4–2.8 GiB per process regardless.
5. Python bookkeeping (row dicts, GAE segments, map stacking, pickling + zstd of each 512-row message) is
   ~3% of the tick; the compressed stream is 60–90 bytes per step, i.e. ~2–3 MB/s into the learner for the whole
   fleet. Transport is not a bottleneck at this scale and will not be below ~30×.

**Host variance is large when arms run on separate containers.** In a first pass with one container per arm, the
same 16×24 production arm measured 1,565 steps/s on one container and 2,997 on another (both "16 vCPU", CPU model
hidden); the earlier reports saw 4.9k–8.6k per 32-core container between launches. The table above therefore ran
every arm sequentially inside one container, on two containers, which agreed within 2%. Fleet-level numbers
(and any future bench) should be read with that ±40% placement spread in mind.

### Why 10× cannot come from the actors as they are

Per vCPU the production tick is ~141 ms for 24 samples: ~5.9 ms of core time per sample, of which ~3.6 ms is
neural-network forward passes and ~2.1 ms engine. Removing every forward from the CPU (serving policy + critic +
rivals from a GPU) leaves the engine's ~2.1 ms plus bookkeeping: a hard ceiling of ~2.6× per vCPU from inference
alone, ~3.5× with the build-mask work trimmed. The rest of a 10× has to come from more vCPUs (money) or from a
faster engine.

## 3. Learner: the critic doubles the epoch, and the batch copy is a sixth of the update

`bench_learner`, one H100, production batch/minibatch (262,144 / 16,384), TF32 convolutions (`precision=DEFAULT`;
the bench also runs the heads at TF32, production ran them fp32, a few percent):

| Config | Epoch over 262,144 samples | Sample-epochs/s | Host→device copy per batch | 3-epoch update |
|---|---|---|---|---|
| policy only (no omniscient critic) | 0.82 s | 320k | 0.45 s | ≈ 2.9 s |
| omniscient critic (as run) | 1.61 s | 163k | 0.90 s | ≈ 5.7 s (measured in the run: 5.78 s) |

So of the 5.78 s per update: ~2.4 s policy forward/backward over 3 epochs, ~2.4 s critic forward/backward over
3 epochs, ~0.9 s copying the batch (262,144 × two int16 views ≈ 8 GB) onto the GPU, during which nothing trains.
Everything is recomputed per epoch, including the int16→float feature decode of both views inside every minibatch.

Learner ceiling today: 45k samples/s (1 H100, 3 epochs, critic). The levers, all measured or exact:

| Lever | Effect on the 5.78 s | Evidence |
|---|---|---|
| 1 epoch instead of 3 (ablation arm "1 epoch + filter + EMA" was neutral at 1 h, +4 pts) | −3.2 s (→ ~2.5 s) | `docs/ablation-v1/readout-2026-10-20.md` |
| `advantage_keep` 0.25 on top (same arm) | → ~1.3 s | same |
| overlap the host→device copy with the previous update's epochs (double-buffer) | −0.9 s | measured copy time |
| decode features once per batch instead of per minibatch per epoch | part of the 2 × 2.4 s; unmeasured | code reading |
| bf16 trunk with the same parity gate as TF32 | unmeasured, plausibly 1.5–2× on the 4.8 s | SFT trainer precedent |
| 2 H100s | linear (measured 1→2 earlier: 165k → 320k) | `docs/distributed-rl-2026-10-19.md` |

With 1 epoch + overlapped copy the learner takes ~1.6 s per 262k batch (≈ 160k samples/s) on one H100 without
touching precision; two H100s or bf16 take it past 300k/s. The learner is not where 10× is hard.

## 4. The bottleneck chain for a 10× sampling rate

Today: 27k samples/s for ~$22/h (billed; 1×H100 + 192 vCPUs). The walls, in the order they are hit:

| Scale | Wall | What removes it |
|---|---|---|
| 1× (today) | actor CPU: 6 ms of core time per sample, 64% of it in three network forwards | GPU inference for the actors (policy + critic + rivals in one batched service, or one small GPU per actor container); trim the critic path (reuse encoded own features, decode the opponent view once); drop the separate rival call by folding rival lanes into the policy batch where weights allow |
| 1.65× | learner: 5.78 s per 262k batch | 1 epoch (ablation-neutral), overlapped batch copy, decode features once; then bf16 or a second H100 |
| 2.6–3.5× per vCPU | engine: ~2.1 ms per game step at 14×14 (`build_mask` ~31%, `core.step` ~34%) | build-mask proposal restricted to own territory (bitwise-tested like the other proposals); beyond that a non-JAX step (Numba/C) as the PufferLib note priced it: weeks |
| 10× | money and placement: ~600 vCPUs at the improved efficiency, ~1,900 at today's | 8/16-vCPU containers place faster and deliver more per dollar (1.3–1.66× in the 18×18 study); Modal replaced 4 actor containers in the first hour of each arm |
| ≥ 30× | transport: one TCP tunnel into one learner process (80–110 MB/s per connection measured), 60–90 B/step compressed | not before 30× |

Rough plan that reaches ~10× at roughly 2× today's hourly cost:

1. **Actors: GPU inference** — one L4/L40S per 2–3 actor containers, or an inference service, serving policy,
   critic and rival forwards in one batched call (network forwards are 60% of actor CPU; policy + critic + rival
   for 27k samples/s is ~7 TFLOP/s, well inside one L40S even at fp32). Expected ~2.3× samples per vCPU. The
   cheap partial version: drop the critic forward from the actors entirely and bootstrap from the learner-side
   critic (segments of 32 decisions already carry their bootstrap value; it could be computed on the GPU when
   the batch is assembled), which alone is +45% actor throughput (141 → 100 ms per tick).
2. **Actors: engine trims** — build-mask proposal, ~1.2×. Together ~2.8× per vCPU.
3. **Fleet: 3.5× the vCPUs** in 8- or 16-vCPU containers (~700 vCPUs, ~$40–45/h billed) → ~10× samples.
4. **Learner: 1 epoch + overlapped copy** (free), then bf16 or a second H100 → ≥ 300k samples/s.

What is not a lever (measured): more games per process (24 → 48 gives +0%, 96 gives −11%), larger
`send_samples` or lighter compression (packing and transport are 3% of actor time), or the vmapped GPU engine
(970 steps/s per L40S, 7.6k per H100 in the Oct 19 comparison: an order of magnitude below a CPU container).

Open question worth one 1-h run before building any of it: does 1 epoch stay neutral at 6 h (it was tested only
at 1 h)? It is the cheapest learner lever by far and gates how much actor throughput one GPU can absorb.
