# Simulator throughput diagnosis (2026-10-18)

Question: why does one training run produce ~100× fewer samples per second than Duper's
Curio setup, and is the fix multi-GPU training or rebuilding the simulator?

**Short answer:** training is limited by game generation, and game generation is limited by
the batched simulator step. Each step carries a large fixed cost (~90 ms) that barely shrinks with
a faster GPU, and only 30–50% of game lanes do useful work per step. The network and gradient
steps are minor. Multi-GPU multiplies the simulator but keeps both problems; fixing the step
cost and lane utilization attacks the root cause, and multi-GPU then multiplies the improved rate.

Data: every update row of 22 past runs (cloud `training.json` + `protocol.json`), produced by
`runs/structured-ppo-v3/throughput-evidence.py`; full table in
`runs/structured-ppo-v3/throughput-evidence.json`. Steady-state rows only (first three updates
excluded where a run had more than six).

## 1. Collection dominates every run

| Run (L40S unless noted) | Lanes | Seconds/update | Collection share |
|---|---:|---:|---:|
| scale-champion128-v1 | 128 | 41.6 | 96% |
| scale-champion512-v1 | 512 | 79.5 | 93% |
| scale-champion96-lowrate-v1 | 512 | 73.4 | 92% |
| low256-selfplay-v1 | 512 | 60.3 | 93% |
| low256-league-rivals-v1 | 512 | 67.2 | 91% |
| champion-a1001024-v1 (A100) | 1024 | 166.3 | 92% |

Gradient updates are 3–8 s of a 40–170 s update. Faster learning code cannot help much.

## 2. The cost is the simulator, not the network

Per batched step (one atomic action advanced in every lane):

| Actor in the loop | Lanes | ms per step |
|---|---:|---:|
| Scripted teacher only, no network acting (bootstraps, roll-in 1.0) | 128 | 104–118 |
| Network acts half the lanes (bootstraps, roll-in 0.5) | 128 | 119–132 |
| Network learner + network opponent (PPO) | 128 | 123–128 |
| PPO vs learned opponent / vs scripted opponent (same run) | 512 | 224 / 223 |

Removing the network entirely only takes the step from ~125 ms to ~110 ms. The engine
advance plus observation/legal-mask preparation is ~85–90% of a step.

## 3. A large fixed cost per step

| Lanes (L40S) | ms per step | ms per lane-step |
|---:|---:|---:|
| 128 | 123–128 | ~1.0 |
| 512 | 222–245 | ~0.45 |
| 1024 (A100) | 350 | ~0.34 |

Fits ~90 ms fixed + ~0.27 ms per lane. A step is a long chain of small dependent operations, so
latency rather than arithmetic sets its floor.

## 4. Half or more of the lanes idle

Lane utilization = (learner + opponent decisions) / (batched steps × lanes):
0.30–0.51 across all PPO runs (0.30 at 1,024 lanes, 0.37–0.51 at 512). Lanes collect a fixed
64 learner decisions and then wait for the slowest lane; more lanes make that tail worse. This
is why the A100 with 1,024 lanes gave no more decisions per second (817/s) than L40S with 512
(754–1,085/s): fixed cost was amortized, but utilization fell from ~0.45 to 0.30.

## 5. Faster GPUs move the step only modestly

| GPU (same low256 recipe) | ms per collection step | Seconds/update |
|---|---:|---:|
| L40S | 223 | 60.3 |
| H100 | 202 (−9%) | 82.2* |
| B200 | 164 (−27%) | 70.3* |

*Only 5–6 updates; their update totals still include warm-up/compilation in the optimization
phase, so update times overstate the cost. Per-step collection is the fair comparison: H100
about 10% faster, B200 about 25–35% faster, at 1.8–2.7× the price. The earlier "H100 slower"
summary reflected those warm-up update totals.

## What this means

- **Throughput per GPU ≈ lanes × utilization ÷ step time** ≈ 512 × 0.45 ÷ 0.225 s ≈ 1,000
  decisions/s (learner + opponent), matching observed 750–1,085/s.
- **Fix lane utilization** (collect a fixed total decision budget per update instead of 64 per
  lane, keep finished lanes playing or refill them): up to ~2× at 512 lanes, and it unlocks
  scaling to 1,024–2,048 lanes per GPU, where the fixed cost is amortized further.
- **Cut the fixed per-step cost:** profile the step (JAX profiler) to split engine rules vs
  observation vs legal masks vs sampling. The ~90 ms floor suggests recomputing large dense
  structures (per-tile masks, observations) every atomic action and evaluating every rule branch
  for every lane. Plausible 2–5× from incremental masks/observations and fewer serial ops; this
  is an estimate until profiled.
- **Multi-GPU data parallel** is orthogonal: ~N× on top of whatever single-GPU rate exists. It
  does not fix the step cost or idle lanes, so it is the multiplier after these fixes, not instead
  of them.
- **A full rebuild** (Duper-style native CPU simulator plus GPU inference servers) is the biggest
  long-run gain but the largest effort. Recommended only if profiling shows the JAX step cannot
  be made several times faster.

Recommended order: (1) profile the step; (2) fix lane utilization; (3) optimize the hottest
step components; (4) then multi-GPU. Re-measure after each with this script.

## Addendum: measured multi-GPU scaling (branch claude/practical-boyd-62b6f5)

| Setup (L40S) | Lanes | Seconds/update | Learner decisions/update | Learner decisions/s | vs 1 GPU |
|---|---:|---:|---:|---:|---:|
| 1 GPU (low256-selfplay-v1) | 512 | 60.2 | 32,768 | 545 | 1.0× |
| 4 GPUs × 128 lanes | 512 | 31.9 | 32,768 | 1,028 | 1.89× |
| 4 GPUs × 512 lanes | 2,048 | 65.3 | 131,072 | 2,008 | 3.69× |

Splitting the same 512 lanes across four GPUs gives only 1.89×, because each GPU still pays the
fixed per-step cost. Keeping 512 lanes per GPU scales almost linearly in samples per second (3.69×),
but updates stay at about one per minute with 4× larger batches. Replay error stayed below 1e-5.
Multi-GPU is a real multiplier for sample throughput, but it does not make individual updates faster,
which confirms the step-cost and lane-utilization fixes as the first priority.
