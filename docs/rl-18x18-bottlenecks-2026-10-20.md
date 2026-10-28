# Where the 18×18 run spends its time (2026-10-20)

Diagnosis of run `l18-turnclock-v1` (the 40-minute turn-clock pilot: 4×H100 learner, 16 × 32-vCPU actor
containers, 262,144-sample batches, 3 epochs, KL anchor to grown c3). Measurements only; nothing was
changed in the training path. Sources: the run's `training.json` and registry rows, `bench_actor`,
`bench_learner`, and a per-piece engine profile (scratch script, not in the repo). Diagnostic spend
≈ $2 on Modal staging (not in a ledger).

## 1. The learner is the wall-clock bottleneck; the actors are the cost bottleneck

Steady state (updates 20–260):

| Quantity | Value |
|---|---|
| Wall time per update | 7.51 s |
| · GPU training (`train_seconds`, incl. host→device copy) | 6.67 s (89%) |
| · untimed bookkeeping after an ordinary update (registry status put, actor-stat merge, log) | 0.28 s |
| · after every 10th update (save `latest.npz` + volume commit) | 5.9 s (7.9 s when a snapshot is also written) |
| · amortised overhead | 0.83 s/update (11%) |
| Trainer waiting for a batch (`ready_wait_seconds`) | 0.00 s in every update after the first |
| Age of a batch when training starts (`batch_age_seconds`) | 10–11 s |
| Inbox occupancy | 128/128 (full) in every status row |
| Policy lag of trained samples | mean 5.5–6.1 updates (max_lag 10); ~900 stale samples dropped per update (0.35%) |
| Samples consumed | 262,144 / 7.51 s ≈ 35k/s |
| Samples produced (16 containers × ~2,450 steps/s × 0.917 learner-decision share) | ≈ 36k/s |

The learner never waits, the inbox is always full, and the assembled batch waits a whole update before
it is trained: the GPU side sets the pace, and backpressure throttles the actors by roughly 4–8% (their
measured capacity is 2.56–2.67k steps/s per container; they ran at ~2.45k). The two sides are within
10% of each other, so a faster learner alone would starve immediately, and more actors alone would only
raise lag.

The first update took 172 s (42 s learner compile + 130 s waiting for the actors to compile at 18×18:
80–136 s per container). Two of the 16 actor containers (7 and 14) were placed by Modal 28 minutes after
the others (no restarts, no errors), so the run had 14 containers for most of its life.

### Learner: fp32 math on CUDA cores, anchor forward repeated per epoch

`bench_learner`, one H100, production per-device shape (4,096-sample minibatch, 65,536-sample batch):

| Config | Epoch over 65,536 samples | Sample-epochs/s per H100 | Host→device copy |
|---|---|---|---|
| as run (anchor_coef 0.05) | 2.107 s | 31.1k | 0.25 s |
| anchor off | 1.605 s | 40.8k | 0.15–0.19 s |

Projected to the run: 3 epochs × 2.107 s = 6.32 s + ~0.3 s copy ≈ the measured 6.67 s. Of the 6.67 s:

| Piece | Seconds | Share |
|---|---|---|
| PPO forward+backward, 3 epochs | 4.8 | 72% |
| anchor (frozen grown c3) forward, recomputed in every epoch | 1.5 | 22% |
| host→device copy of the batch (262,144 × 24.9 KB ≈ 6.5 GB) | ~0.3 | 5% |

The grown network costs 243 MFLOP per forward at 18×18 (147 without the two extra blocks; 89 at 14×14).
Without the anchor the learner does ~730 MFLOP per sample-epoch × 40.8k/s ≈ 30 TFLOP/s per H100, about
45% of the fp32 CUDA-core peak: it is compute-bound on fp32 `Precision.HIGHEST` convolutions and
matmuls, with the tensor cores unused. The 2 epochs' worth of anchor forwards recomputed per update are
pure repetition: the anchor is frozen, so its log-probs for a batch are identical in every epoch.

### Actors: forward 40%, engine 60%, and the legal-structures mask is almost half of the engine

`bench_actor`, 32 pinned processes × 24 games on a 32-vCPU container, one policy forward per tick as in
production (two runs):

| Board | Container steps/s | ms per tick (24 games) | policy forward | engine tick | post |
|---|---|---|---|---|---|
| 18×18, grown net | 2,561–2,669 | 288–300 | 105–129 (36–43%) | 169–179 (57–60%) | 2–4 |
| 14×14, c3 net | 8,550 | 90 | 33 | 54 | 2 |

Per-piece engine profile (one process, uncontended core, mid-game states, 24 games, `fast=True`;
proportions are what matter, absolute times are lower than in the loaded container):

| Piece, 18×18 | ms | Piece, 14×14 | ms |
|---|---|---|---|
| full tick | 93.7 | | 71.2 |
| `core.step` + decode | 44.9 | | 24.2 |
| legal masks total (fast proposals) | 55.0 | | 25.5 |
| · of which `build_mask` (structures) | 46.6 | | 22.0 |
| · of which bucketed move rows | 10.1 | | 6.8 |
| · of which attack mask, all tiles | 2.3 | | 1.1 |
| observe + encode + pack view | 4.0 | | 3.0 |
| frozen (non-fast) legal masks, for reference | 465.6 | | 191.4 |

The structures build mask is ~45% of the engine tick at 18×18 (it was 31% at 14×14; its float32
products scale with the square of the tile count), computed for every game on every decision whether or
not a build is plausible. `core.step` is the other ~45%; move rows ~10%.

### Actors: the 32-vCPU containers scale badly

`bench_actor` at 18×18, 24 games per process, pinned:

| Container vCPUs | Processes | Container steps/s | Steps/s per process | Steps/s per paid vCPU |
|---|---|---|---|---|
| 32 | 8 | 1,257 | 157 | 39 |
| 32 | 16 | 2,237 | 140 | 70 |
| 32 | 24 | 2,438 | 102 | 76 |
| 32 | 32 | 2,669 | 83 | 83 |
| 16 | 16 | 1,724 | 108 | 108 |
| 8 | 8 | 1,110 | 139 | 139 |
| 4 (one process, profile) | 1 | 256 | 256 | — |

On a 32-vCPU box the second 16 processes add 19% throughput; per-process speed halves from 8 to 32
processes (hyperthread siblings and shared memory bandwidth on a 48-vCPU host). Since Modal bills per
requested vCPU, an 8-vCPU container delivers 1.66× the samples per actor dollar of a 32-vCPU one and a
16-vCPU container 1.29× (single benches; host variance ±5–10%). Memory cost is per process either way
(~2.5 GiB at 18×18).

### Cost structure of the run

| Item | $/h | Share |
|---|---|---|
| learner (4×H100 + 16 vCPU + 64 GiB) | 17.1 | 33% |
| 16 actor containers (32 vCPU + 80 GiB each) | 34.4 | 67% |
| total | 51.5 | ≈ $0.40 per million samples (actors 0.27, learner 0.13) |

Time budget of the 40-minute run: 172 s first batch (7%), 80 critic warm-up updates × 7.5 s = 10 min
(25%, full forward/backward with the policy gradient zeroed), ~11% bookkeeping, the rest PPO.

## 2. Levers (not applied; ranked by gain per effort)

Learner (wall clock):

1. **Tensor-core precision on the learner** (`POLYTOPIA_MATMUL_PRECISION=DEFAULT`, i.e. TF32; or bf16):
   the 4.8 s PPO core is fp32 CUDA-core bound; expected 1.5–2.5× on the epoch time. Needs an
   actor/learner log-prob parity gate (ratio distribution at epoch 0; the SFT trainer's rounding-path
   check is the precedent). A no-op-by-default env hook is in `rl/network.py` (uncommitted).
2. **Anchor log-probs once per batch**, not per epoch: exact, −1.0 s of 6.67 (−15%).
3. **Checkpoint/status off the training thread**: save+commit every 10 updates costs 5.9 s, the
   registry put 0.28 s per update; together 11% of wall time.
4. **Cheaper critic warm-up**: stop the gradient at the trunk while `policy_weight == 0` (the trunk's
   gradients are zeroed anyway), ~3× cheaper warm-up updates, −6–7 min per 40-min run; or warm up once and
   reuse the critic across pilots.
5. Overlap the host→device copy with training (0.3 s, 5%).

Actors (cost):

6. **Fleet shape**: 8- or 16-vCPU containers instead of 32: 1.3–1.66× more samples per actor dollar
   from a launch flag; likely faster placement than the 32-vCPU boxes too (unverified).
7. **Build-mask proposal in `proposals/fast_engine.py`** (frozen rules untouched, bitwise test as for
   the other proposals): the structures mask is ~45% of the engine tick at 18×18; restricting it to tiles
   that can hold a build (own territory) or caching it within a turn would cut roughly 20–25% of actor
   time.
8. **Network size**: the two grown blocks are 40% of forward FLOPs on both sides (243 vs 147 MFLOP);
   a run without them is a modelling choice, worth −15–20% actor time and −30% learner time.
9. Later, GPU inference service for actors (forward is ~40% of actor time; one L40S could serve the
   whole fleet, bounded by tunnel bandwidth): large effort, only after 6–7.

Pipeline: the inbox (128 messages) plus the ready queue hold ~1.3 updates of samples, which is where
the 5.5–6 update lag comes from; shrink once the learner is faster (quality, not speed).

Rough combination: levers 1–3 bring the learner to ~3 s per update (≈ 87k samples/s needed); lever 6
supplies that from ~690 paid vCPUs in 8-vCPU containers (~$46/h) → ~$63/h for 2.2× today's samples per
hour, about $0.20 per million samples instead of $0.40.
