# TF32 precision gate and fleet numbers (Stage 1), 2026-10-20

**Verdict: TF32 passes.** All three precision modes pass the gate on 14×14 and 18×18. The recommended
setting for the training arms is TF32 in the convolutions with fp32 heads, set **on the learner container
only**:

    POLYTOPIA_MATMUL_PRECISION=DEFAULT  POLYTOPIA_HEAD_PRECISION=HIGHEST

On one H100 this runs the PPO epoch **1.92× faster at 14×14** and **3.27× faster at 18×18** than fp32. It
gives up nothing against full TF32 (−1%), and its actor/learner log-prob mismatch is about half as large.

Code: `POLYTOPIA_HEAD_PRECISION` in `src/polytopia_bot/rl/network.py` (defaults to
`POLYTOPIA_MATMUL_PRECISION`, which defaults to HIGHEST, so nothing changes when neither is set),
`experiments/modal_precision_gate.py` and `experiments/precision_gate_worker.py`. Raw results:
`docs/ablation-v1/precision-gate-20261020T132107.json`, and `/results/precision-gate/` on the
`polytopia-structured-rl` volume (rows, gate and bench JSON).

## Method

- **Rows.** These are real actor rows, two checkpoints × 20,480 rows. c3
  (`c3-maps-ctl-update-4854`) plays at 14×14 and grown c3 (`c3-u4854-grown18x2`) at 18×18. The rows come
  from sampled self-play on `proposed-v1` maps, produced the way `_actor_process` does it:
  `CpuGames(fast=True, map_batch=True)`, `network.sample` in fp32 on CPU, and a lane's map refilled
  only after its game ends. Each row is in the transport format: int16 view, packed path-mask bits,
  choice and actor logp. The actor ran the **float16-rounded** weights it receives at version 0
  (`Learner._leaves`). There were 16 processes × 16 games × 640 ticks, recording every 8th tick.
- **Gate.** One H100 recomputes the log-probs the way `Learner._build_trainer.to_batch` does:
  `view_features` from the int16 view, `PathMasks` from the unpacked bits, `network.evaluate`, in
  4096-row minibatches. Each mode runs in a fresh subprocess with its environment set, because the
  network reads precision at import. The gate compares against the same f16-rounded weights the actor
  ran, so it measures precision alone. log r = lp_learner − lp_actor. Pass: mean k3 = r − log r − 1 below
  1e-4 **and** max |log r| below 0.02.
- **Bench.** This is the unmodified `Learner.benchmark`: a synthetic all-legal batch, one H100,
  batch 65,536, minibatch 4,096 (16 steps per epoch), 4 epochs. The table reports the median of epochs
  1–3, so the compile epoch is excluded. No anchor or reference network is used.

## Gate (epoch-0 importance ratio, 20,480 real actor rows per board)

| Mode (convs / heads) | Board | mean r | mean k3 | max \|log r\| | p99.9 \|log r\| | mean \|log r\| | \|log r\| > 1e-3 | Gate |
|---|---|---|---|---|---|---|---|---|
| fp32 / fp32 (control) | 14 | 1.0000000 | 1.8e-12 | 1.7e-5 | 1.1e-5 | 1.2e-6 | 0% | pass |
| fp32 / fp32 (control) | 18 | 1.0000000 | 2.1e-12 | 1.9e-5 | 1.2e-5 | 1.3e-6 | 0% | pass |
| TF32 / TF32 | 14 | 1.0000057 | 4.8e-7 | 9.0e-3 | 7.3e-3 | 5.7e-4 | 17.8% | pass |
| TF32 / TF32 | 18 | 1.0000103 | 4.4e-7 | 1.09e-2 | 6.1e-3 | 5.7e-4 | 18.4% | pass |
| **TF32 / fp32** | 14 | 1.0000058 | 1.8e-7 | 6.1e-3 | 3.8e-3 | 3.6e-4 | 9.1% | **pass** |
| **TF32 / fp32** | 18 | 1.0000060 | 1.9e-7 | 6.5e-3 | 3.6e-3 | 3.9e-4 | 9.9% | **pass** |

No row in any mode falls outside the PPO clip (|r − 1| > 0.2).

**Context: the float16 weight transport is already a mismatch of the same size.** The learner holds
fp32 weights, but actors run the f16 copy. So the ratio a training run actually sees at epoch 0 is the
learner's fp32 weights against the actor rows:

| Mode | Board | mean k3 | max \|log r\| |
|---|---|---|---|
| fp32 / fp32 (today's production) | 14 / 18 | 3.0e-7 / 3.2e-7 | 6.8e-3 / 6.1e-3 |
| TF32 / TF32 | 14 / 18 | 5.2e-7 / 4.9e-7 | 1.54e-2 / 1.28e-2 |
| TF32 / fp32 | 14 / 18 | 2.9e-7 / 2.9e-7 | 7.2e-3 / 8.7e-3 |

TF32 in the torso adds almost nothing to the mismatch the runs already have. Full TF32 roughly doubles
it, and at 14×14 its worst row (1.54e-2) is at 77% of the 0.02 limit.

## Learner throughput (one H100, sample-epochs/s; epoch seconds in brackets)

| Mode (convs / heads) | c3, 14×14 | grown c3, 18×18 |
|---|---|---|
| fp32 / fp32 | 161,946 (0.405 s) | 40,427 (1.613 s) |
| TF32 / TF32 | 313,640 (0.209 s) — 1.94× | 133,645 (0.490 s) — 3.31× |
| **TF32 / fp32** | **311,179 (0.211 s) — 1.92×** | **132,526 (0.495 s) — 3.28×** |

The epoch times were stable to ±0.5% across epochs 1–3. The heads cost about 1% of the epoch, so
keeping them in fp32 is free. The 18×18 gain is larger than the 1.5–2.5× estimated in
`docs/rl-18x18-bottlenecks-2026-10-20.md`: that network has two more residual blocks and 1.65× the tiles,
so the convolutions dominate it even more. These benches leave out the anchor forward and multi-GPU
`pmean`, so the speedup transfers to a full update but the absolute seconds per update do not.

## Actor fleet (`bench_actor`, pinned processes × 24 games, 60 s, two policy forwards per tick)

| Board | Container vCPUs | Processes | Container steps/s (runs) | Steps/s per paid vCPU |
|---|---|---|---|---|
| 14×14 | 8 | 8 | 1,928; 1,876 | 241; 234 |
| 14×14 | 16 | 16 | 3,779; 3,890 | 236; 243 |
| 14×14 | 32 | 32 | 6,915 (bottlenecks doc: 8,550) | 216 (267) |
| 18×18, grown | 8 | 8 | 638; 827; 516 (bottlenecks doc: 1,110) | 80; 103; 65 (139) |
| 18×18, grown | 16 | 16 | 1,619; 1,001 (bottlenecks doc: 1,724) | 101; 63 (108) |
| 18×18, grown | 32 | 32 | (bottlenecks doc: 2,669) | (83) |

- **At 14×14 the container size barely matters.** 8- and 16-vCPU containers both deliver 234–243
  steps/s per paid vCPU, repeatable to ±3%. A 32-vCPU container gave 216 today and 267 earlier, so it is
  no worse to within host variance. The 1.3–1.66× advantage of small containers measured earlier was an
  18×18 result.
- **At 18×18 the numbers vary too much between identical runs to rank container sizes.** Identical
  configurations differed by up to 1.6× from one run to the next. Most of the spread is in the policy
  forward (141–235 ms per tick) and the engine (89–145 ms), which suggests different host CPUs or memory
  bandwidth contention; `/proc/cpuinfo` reports "unknown" for the model. All of today's 18×18 runs came
  in at or below the earlier ones. Settling the 18×18 fleet shape would take several repeats per size,
  because single benches cannot.
- The container sees more cores than it pays for: 8 vCPU sees 24, 16 sees 32, 32 sees 48. Processes are
  pinned one per core, so they use the paid count.
- `BOARD_SIZE` / `EXTRA_BLOCKS` do reach the container: `bench_actor --size 18 --extra-blocks 2` passes
  them through `actor_bench`'s subprocess environment.

For the 14×14 arms, **16-vCPU containers** (16 processes × 24 games, as planned in Stage 2) are the
choice. They are as efficient per dollar as 8-vCPU ones, need half as many containers, and give
~3.8k steps/s each.

## Recommendation for the training arms

Set `POLYTOPIA_MATMUL_PRECISION=DEFAULT` and `POLYTOPIA_HEAD_PRECISION=HIGHEST` **in the learner
container, before `polytopia_bot.rl.network` is imported**. For example, add `.env({...})` on the learner
image, or set `os.environ` at the top of `learner()` the way `learner_bench` already does. The `learner`
function in `experiments/modal_distributed_rl.py` does not set the variables today; it is another
session's file, so this doc only states the requirement. Leave the actors at the default (fp32). The
gate measured exactly that pairing: CPU fp32 actors against a TF32 learner.

Reasoning:
1. Both TF32 modes pass with margin. Torso-only TF32 has 2.3–2.6× lower mean k3 than full TF32 and its
   worst row is 60–68% of full TF32's. Against today's real epoch-0 ratio, which includes the f16 weight
   transport, it is indistinguishable from fp32 (max 7–9e-3 vs 6–7e-3).
2. Keeping the heads in fp32 costs about 1% of throughput. Full TF32 buys that 1% by moving the
   worst-case row to 77% of the gate at 14×14, which leaves little room for drift once training moves the
   weights.
3. The learner epoch is 1.92× faster at 14×14 and 3.28× faster at 18×18. That roughly halves (14×14) or
   thirds (18×18) the learner GPU cost per sample, or lets the same learner absorb a larger actor fleet.

Caveats:
- The rows cover the first ~640 decisions of each lane, which is early and mid game: 70 of 256 lanes
  finished a game at 14×14 and 10 at 18×18. Late-game positions with large unit counts are
  under-represented.
- The gate tests one checkpoint per board at epoch 0. Watch the first updates of each arm: approx_kl and
  clip fraction at update 1 should match an fp32 run's.

## Cost

Ledger row `precision-gate` in `runs/ablation-v1/budget.json` ($12 reserved). It holds the gate app
`ap-6uHZNTe9aGAL8TsedeaMIS` (2 × 16-CPU row makers, 3 × H100 for gate and bench) and nine `bench_actor`
apps. Modal billing had not posted any of them when this was written (it lags up to an hour). The
estimate from container wall times is **≈ $2**: about 0.25 H100-hours ≈ $1.0, the row makers ≈ $0.3, the
actor benches ≈ $0.5. Reconcile with `python scripts/ledger_spend.py runs/ablation-v1 --since 2026-10-20 --write`
once the hours settle.
