# One additional epoch: movement checkpoint comparison

The extra supervised epoch improved training-set agreement and the non-movement control sample, but did not establish a meaningful improvement in held-out movement. The key held-out score increased from **13/458 to 19/458**, only **1.31 percentage points**; its paired map-bootstrap 95% interval is **−0.60 to +3.24 points**. Ordinary teacher-demonstration training positions remain almost as weak as held-out positions. Prioritize a focused movement learning probe before another full-corpus pass or RL.

## What ran

Continued the exact `speed-v2/end-to-end-v1` adapter through **one additional pass over the same 10,000 examples**, with the original prompt, labels, prepared tokens, model revision, LoRA configuration, batch size, and optimized training runtime unchanged. The epoch covered every example once, with second-epoch shuffle seed 18 (base seed 17 plus epoch offset 1). Training completed all **157 optimizer updates** in **634.33 seconds (10.57 minutes)** on four B200 GPUs.

The previous run did not save optimizer state. This continuation therefore used **fresh AdamW**, the same peak learning rate of `1e-4`, five warmup steps, and a restarted 157-step cosine schedule. It is not equivalent to an uninterrupted two-epoch optimizer trajectory. The new run saves optimizer state for future work.

Saved checkpoints after 2,560, 5,056, 7,552, and 10,000 additional examples. Each checkpoint and a freshly evaluated starting adapter received the same ordered set of **2,161 positions**:

- All **1,298 movement-labeled training positions** on 402 training maps.
- All **575 held-out movement positions** on 98 disjoint validation maps.
- **288 held-out non-movement controls**, sampled deterministically without consulting predictions: 32 from each action family 2–10.

All five evaluations completed. They used the same prompt, complete legal-action catalog, vLLM version, greedy decoding, batch size 64, position order, and two-repair policy. Prompt identities and token counts matched across checkpoints. No full-game evaluation, RL, final-test access, or automatic policy promotion occurred.

## Primary movement result

Tie-aware agreement accepts any action with the globally maximum frozen teacher score. The primary subset requires **every globally top-scored action to be movement**. Thus, choosing end turn or another family cannot earn credit merely by tying with movement.

| Additional epoch completed | Optimizer step | All training, 1,037 positions | Teacher-demonstration training, 632 positions | Held out, 458 positions |
|---|---:|---:|---:|---:|
| Baseline | 0 | 96 = **9.26%** | 13 = **2.06%** | 13 = **2.84%** |
| 25.60% | 40 | 128 = **12.34%** | 30 = **4.75%** | 15 = **3.28%** |
| 50.56% | 79 | 132 = **12.73%** | 28 = **4.43%** | 18 = **3.93%** |
| 75.52% | 118 | 130 = **12.54%** | 26 = **4.11%** | 19 = **4.15%** |
| 100% | 157 | 146 = **14.08%** | 26 = **4.11%** | 19 = **4.15%** |

The final all-training gain is **+4.82 points**, with paired map-bootstrap interval **+2.70 to +7.19**. The final held-out gain is **+1.31 points**, with interval **−0.60 to +3.24**. On held-out positions, 13 previously wrong choices became teacher-best and 7 previously correct choices became wrong: a net gain of six positions. Checkpoints 118 and 157 tie for the highest primary score; there is no evidence here that one is a clearly better movement checkpoint.

The mixed training population should not be mistaken for a matched train/held-out comparison. Its 405 movement-required corrective examples improved from **20.49% to 29.63%**. These are generally easier states than ordinary teacher trajectories. Ordinary teacher-demonstration training positions finished at **4.11%**, essentially the same as held-out **4.15%**. This is persistent weak learning on seen states, alongside some generalization gap in the mixed population; it does not isolate a representation defect by itself.

Uncertainty uses 5,000 paired map-cluster resamples, seed 17, preserving each position's before/after pair and resampling entire maps. The primary held-out subset spans 97 maps. These are exploratory percentile intervals, without adjustment for examining multiple checkpoints or inference-runtime variability.

## Broader agreement and non-movement controls

The full movement-labeled set includes positions where other action families tie with movement. It gives a more favorable score, but cannot substitute for the movement-required check.

| Additional epoch | Training tie-aware, 1,298 | Held-out tie-aware, 575 | Training exact, 1,298 | Held-out exact, 575 | Non-movement exact, 288 |
|---|---:|---:|---:|---:|---:|
| Baseline | 21.34% | 18.26% | 6.01% | 1.57% | 63.89% |
| 25.60% | 26.35% | 20.52% | 8.01% | 1.74% | 60.76% |
| 50.56% | 25.89% | 20.87% | 8.63% | 2.09% | 66.67% |
| 75.52% | 26.96% | 20.87% | 8.01% | 1.57% | 68.40% |
| 100% | 27.58% | 20.35% | 9.24% | 1.39% | 69.79% |

Full held-out tie-aware agreement improved **2.09 points** (95% interval **+0.18 to +4.01**). The non-movement control sample improved **5.90 points**, from 184/288 to 201/288 (interval **+1.39 to +10.83**). There was an early control regression at the quarter checkpoint, followed by recovery. The final control gain supports retaining the new adapter as a candidate; it does not establish improved win rate or universal improvement across skills.

| Non-movement family | Baseline correct / 32 | Final correct / 32 |
|---|---:|---:|
| Attack | 24 | 22 |
| Ability | 14 | 18 |
| Summon/recruit | 19 | 23 |
| Harvest | 22 | 19 |
| Build | 6 | 7 |
| Research | 12 | 20 |
| Capture | 27 | 29 |
| City reward | 32 | 32 |
| End turn | 28 | 31 |

This balanced 288-position sample is a regression screen, not a prevalence-weighted estimate of overall policy performance. Attack and harvest declined in this small sample. Final first-attempt validity was 2,157/2,161 (99.81%); one training position remained invalid after all repairs, versus none at baseline. High validity does not explain away poor movement selection.

## What still fails

On movement-required held-out positions, the model chose movement more often: **60/458 (13.10%) → 101/458 (22.05%)**. But only **13 → 19** choices attained the teacher's best score. The remaining breakdown changed from 398 non-movement choices and 47 lower-scored moves to **357 non-movement choices and 82 lower-scored moves**. More willingness to move did not translate proportionally into better moves.

Late-game movement remains almost absent: final teacher-best movement was **1/183** on ordinary training demonstrations and **1/141** held out. Opening performance was about 8.1% on both sets; exploration performance was 3.1% training and 3.6% held out. Both deciding when to move and selecting a useful destination remain problems.

## Recommended next step

Do not make another undifferentiated full-corpus epoch the main movement intervention. The extra epoch had measurable supervised-learning value, but the movement-required held-out curve flattened around the middle of the pass and ordinary training movement remains poor.

Run a small **movement-only overfit probe** on existing training examples, stratified across phase and legal-catalog complexity, keeping the prompt fixed. Test whether focused supervision can achieve near-perfect tie-aware fit on those seen states. This distinguishes a problem with the amount or emphasis of movement supervision from difficulty fitting movement even when it is the only task.

Then choose one controlled intervention: movement-balanced sampling/loss emphasis if focused training fits well, or an input/action representation experiment with explicit coordinates and source–destination relationships if fitting remains difficult after checking optimization and labels. Keep held-out maps separate and rerun the same movement-required metrics and non-movement controls. A representation improvement remains a hypothesis, not a result of this experiment. Defer RL and expensive full-game evaluation until a convincing movement improvement appears.

## Integrity, execution, and artifacts

Checks verified identical data and training configuration, exact loading of all starting adapter tensors, unchanged frozen-base hash, changed adapter weights, all 157 steps, every planned checkpoint, and equality of checkpoint 157 with the final adapter. The original adapter is intact. The final adapter is retained locally with safetensors SHA256 `a248b8e2ad713072db10dd0cebcbc9e1bb63a012eb03da423e0538604e238e8f`.

The fresh baseline differs slightly from the earlier diagnostic: 244/1,298 training choices and 109/575 held-out choices changed under a different ordering/batching arrangement, with unchanged prompt token counts. Aggregate full tie-aware agreement changed only +0.08 and −0.35 points, respectively; primary held-out agreement changed −0.66 points. All gains in this report use the fresh baseline and common batch order. This reduces a known comparison confound but does not quantify all runtime variability. Teacher scores are frozen heuristic preferences, not proven optimal actions.

Total wall time, overlapping training and evaluations, was **22.72 minutes**. Training occupied the four-B200 container for 709.19 seconds; optimizer work took 634.33 seconds. Five A100 evaluation invocations occupied about **52.18 GPU-minutes in total**. The CPU coordinator was preempted after training had completed; recovery reconnected to the exact same training and evaluation call IDs. No training or evaluation was repeated. The coordinator now supports this recovery path. Both run apps are stopped.

Validation: 23 focused continuation/runtime tests and 2 paired-summary tests passed; all five real checkpoint evaluations and the recovered coordinator completed. Python compilation and Git whitespace checks passed. Existing simulator behavior was not changed.

- [Full numerical summary and paired intervals](../runs/movement-epoch2-v1/summary.json)
- [Frozen protocol](../runs/movement-epoch2-v1/protocol.json)
- [Completed run and execution receipts](../runs/movement-epoch2-v1/status.json)
- [Training integrity checks](../runs/movement-epoch2-v1/training-integrity.json)
- [Historical baseline sensitivity](../runs/movement-epoch2-v1/historical-baseline-comparison.json)
- [Final adapter configuration](../runs/movement-epoch2-v1/adapter-final/adapter_config.json)
- [Final prediction report](../runs/movement-epoch2-v1/report-157.json)

Reproduce the summary with `.venv/bin/python scripts/summarize_movement_continuation.py runs/movement-epoch2-v1`. Full inference traces, all intermediate adapters, the final adapter, and `optimizer-final.pt` remain on the staging `polytopia-sft0-training` volume under `movement-epoch2-v1/`. Local run artifacts remain excluded from Git by the existing rules. The original app was `ap-rTjjSD7NReRpquXJRgOnOm`; the recovery coordinator used `ap-MqhepDKJm0iL4ftWuPIer1`.
