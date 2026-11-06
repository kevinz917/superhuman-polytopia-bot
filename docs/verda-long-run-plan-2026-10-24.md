# Plan: a ~40-h continuation of the 4×128 net on Verda (bn-main2), 2026-10-24

**Goal.** Spend ~$390 of the $441 Verda balance training the big 4×128 net for as long as it buys, ending with a
learning-rate polish, and leave a checkpoint that can be measured against the scratch champion. Not launched yet:
waiting for approval.

## Where it starts

- **bn-main1** (Oct 23 23:10 to Oct 24 09:10 PT): 10 h from random init, 1.42B samples, $96.32. Its final checkpoint
  (u5400) lost to the small net at matched samples (Scratch 7h u5250): 84–138–34, 32.8%, about −85 Elo.
- **Confound.** bn-main1 ran at minibatch 4,096 (the A1 OOM workaround); every small-net run used 16,384. That is 4×
  more Adam steps per batch at the same learning rate, and the policy moved about 1.5× further per update (KL 0.0073 vs
  0.0049). The small-net recipe's minibatch is the obvious fix and the only recipe change in this plan.
- **Throughput from bn-main1's log.** Steady 40.1k samples/s from update 500 to the end. The learner trains a batch in
  5.56 s (ceiling ≈ 47k samples/s) and waited about 1.0 s per update for data. So it was slightly actor-short on 20
  boxes; 24 boxes should fill it. The bench showed minibatch barely changes learner speed (2× PRO 6000: 50–52k).

## Recipe (experiments/verda/configs/bignet-4x128-long.json)

Same as bn-main1 except:

| knob | bn-main1 | bn-main2 |
|---|---|---|
| parent | random-init-b4w128 | bn-main1 final.npz + final-critic.npz |
| minibatch | 4,096 | **16,384** (8,192 per GPU on 2 GPUs, the size the bench ran) |
| seed | 702 | 703 |
| LR | 3e-4 flat | 3e-4, then linear to **1e-4 over the last 12%** (~4.8 h). Polish B1 gained +41 Elo this way. |
| training_seconds | 10 h | 40 h |

Everything else (batch 262k, 1 epoch, omniscient critic, proposed-v1 maps, entropy 0.01, historical pool) is unchanged.
No untested code (KataGo aux heads, GPU actor inference) goes into a $390 run.

## Hardware and cost

| option | learner | actors | $/h | 40 h | samples (est.) |
|---|---|---|---|---|---|
| **A (preferred)** | 2× RTX PRO 6000, FIN-03 (measured last night) | 24× Turin 16V, FIN-03 | $9.64 | **$386** | ~47k/s → **6.8B** |
| B (fallback) | 1× H200 141 GB, FIN-02 (unmeasured) | 24× Turin 16V, FIN-03 over public IP | $10.56 | 36 h = $380 | ~6.1B if it matches A |

- **Spend cap $400** in the controller, so the balance never drops below ~$41. A zero balance deletes Verda volumes.
- After this run the 4×128 net has ~8.2B samples, about the small champion's total (7.5B: 8,500 + 9,000 + 11,250 updates over its three runs).
- Evals run on Modal (~$2 per 256-game match, ~$20 total). They don't touch the Verda balance.
- **Availability at 13:00 PT today:** no RTX PRO 6000 in any location. Turin actors are available in FIN-03. H200 is
  available in FIN-02, B200 in FIN-03 ($8.44, overkill: 24 boxes can't feed it within the 512-vCPU quota).
  Option A means waiting for capacity. Option B can start now, after a 15-min bench (~$1.50) that confirms minibatch
  16,384 fits on one GPU and the learner does ≥ 45k samples/s.

## Gates during the run

The Modal controller syncs a snapshot every 250 updates (~25 min). Matches use the bn-main1 protocol (128
proposed-v1 maps × 2 seats, deterministic, seeds 396000+).

| when | match | continue if | otherwise |
|---|---|---|---|
| 30 min | health: samples/s, KL, clip fraction, entropy, actors connected | ≥ 42k/s, KL ≤ 0.01, entropy not collapsing | fix actors or stop |
| ~6 h (~1B samples) | latest vs bn-main1 final | ≥ 55% | stop: ~$330 left for a different plan |
| ~16 h, ~28 h | latest vs bn-main1 final and vs the champion (scratch-cont-2) | win rate vs the champion rising | lower the cap and end early |
| end | final vs the champion, 512 seat-paired games, + vs OMNI c3 | — | — |

Ending early is done with `bigrun.py scale --cap`. The run's clock sets when the LR decay starts, so an early stop
skips the polish. If a gate fails late, the cheaper move is to let the decay finish.

## Risks

- **Learner capacity.** The main risk for option A. Verda refused 4 of 6 creates with 503 even when it reported
  stock. If the learner dies mid-run, the controller tears down after 3 missed checks. Resuming from latest.npz
  (≤ 15 min lost) is a manual relaunch from the Mac.
- **Cross-DC (option B).** Actors in FIN-03 send ~10 MB/s of samples to FIN-02. Verda doesn't publish egress prices
  (third parties say free). Check the balance drop against the estimate in the first hour.
- **The recipe change is untested on 4×128.** The 6-h gate is where we'd find out.
- **Fresh optimizer and opponent pool.** The continuation restarts Adam state and the historical pool. The first ~30
  min may show a KL bump.
- **Expectations.** At 1.42B the big net was −85 Elo behind the small one at matched samples. Matching the champion's
  sample count is not the same as beating it. The 6-h gate tests whether the minibatch fix closes the gap.

## Launch steps (once approved)

1. `bigrun.py init` (already written as plan `runs/verda-bn-main2`, H200 variant; re-init for option A), `preflight`.
2. `up` → `setup` → `bench --devices 0[,1] --minibatches 16384 --epochs 2` → `start` → `arm`.
3. Check the 30-min health gate, then the remaining gates from synced snapshots.
