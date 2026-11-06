# Proportional 40k experiment — stopped at user request

The run was stopped immediately on request after **31,856 examples / 500 optimizer updates**. Modal app `ap-d26Nfs9PWOmeUuiZYbqG7G` is **stopped with zero tasks**. No further GPU work was launched. The 10k, 20k, and 30k adapters were saved; only the baseline, 10k, and 20k evaluations finished. The 30k evaluation was interrupted after 128 of 1,475 decisions, and no 40k checkpoint was produced. The incomplete prefix is not used as a checkpoint score.

**20k is the strongest completed checkpoint on the two aggregate diagnostics.** It improves movement-required agreement by 3.49 percentage points and non-movement agreement by 3.67 points relative to the fresh main baseline. This supports continuing to take data quantity seriously, but movement remains weak and recruitment regresses. The main checkpoint remains unchanged.

## Completed checkpoint comparison

Tie-aware agreement accepts any globally highest-scoring action from the frozen teacher. The primary movement score uses 458 held-out positions where every globally top action is a move. Non-movement uses 900 held-out controls, 100 per action family. All completed evaluations scored the same ordered 1,475 positions, including 575 movement-labeled positions.

| Additional examples | Movement tie-aware / 458 | Movement exact / 458 | Non-movement tie-aware / 900 | Non-movement exact / 900 |
|---:|---:|---:|---:|---:|
| 0 | 20 (4.37%) | 7 (1.53%) | 741 (82.33%) | 596 (66.22%) |
| 10,000 | 13 (2.84%) | 5 (1.09%) | 770 (85.56%) | 622 (69.11%) |
| 20,000 | 36 (7.86%) | 16 (3.49%) | 774 (86.00%) | 624 (69.33%) |
| 30,000 | Evaluation interrupted | — | Evaluation interrupted | — |
| 40,000 | Not produced | — | Not produced | — |

At 20k versus baseline, paired map-bootstrap 95% intervals are **+0.70 to +6.37 points** for movement and **+1.68 to +5.65 points** for non-movement. From 10k to 20k, movement improves **+5.02 points** (interval **+2.18 to +8.00**); non-movement changes **+0.44 points** (**−1.70 to +2.41**). Thus the second 10k clearly improves the observed movement score, while its additional aggregate non-movement gain is uncertain. These are exploratory intervals, not adjusted for checkpoint selection or repeated development-set use.

## Other skills are not uniformly preserved

| Action family | Main baseline | 10k | 20k |
|---|---:|---:|---:|
| Attack | 68.00% | 78.00% | 71.00% |
| Unit abilities | 57.00% | 63.00% | 71.00% |
| Recruitment | 75.00% | 72.00% | 65.00% |
| Harvest | 97.00% | 96.00% | 95.00% |
| Build | 85.00% | 91.00% | 82.00% |
| Research | 71.00% | 84.00% | 94.00% |
| Capture | 95.00% | 96.00% | 99.00% |
| City reward | 99.00% | 97.00% | 99.00% |
| End turn | 94.00% | 93.00% | 98.00% |

At 20k, research rises **71% → 94%** and unit abilities **57% → 71%**, but recruitment falls **75% → 65%**. Build and harvest also decline slightly. The inherited diagnostic flags require a loss greater than 3 points overall or greater than 10 points for one family, so recruitment at exactly −10 does not trigger a flag. An empty flag list does not establish lossless retention.

## What improved in movement?

| Additional examples | Chooses movement / 458 | Agreement among chosen moves | Correct on the fixed 96 states where baseline moved |
|---:|---:|---:|---:|
| 0 | 96 (20.96%) | 20.83% | 20/96 |
| 10,000 | 73 (15.94%) | 17.81% | 10/96 |
| 20,000 | 171 (37.34%) | 21.05% | 17/96 |

The 20k improvement primarily accompanies a higher rate of choosing movement: **20.96% → 37.34%**. Agreement conditional on choosing a move is nearly unchanged (**20.83% → 21.05%**), although that comparison uses changing subsets. On the fixed 96 baseline-movement states, correctness goes from 20 to 17; its paired change interval is **−13.27 to +6.67 points**. This does not demonstrate improved destination selection. Movement agreement at 20k remains only **14.19% opening, 5.92% exploration, and 3.55% late-game**.

## What this says about scaling data

The completed results support the practical benefit of adding examples and training longer with the original action mix: by 20k, both aggregate diagnostics improve. They do not show that data quantity alone solves movement, nor establish that a new representation is necessary. More unique examples, more optimizer updates, and a changing learning rate occur together. There was no compute-matched repeated-data control.

The earlier movement-enriched 10k run reached 13.10% movement / 81.33% non-movement at its retention-focused checkpoint, or 14.63% / 80.00% at its final checkpoint. This proportional-data run reaches 7.86% / 86.00% at 20k: stronger aggregate non-movement retention but weaker movement. These are descriptive cross-experiment comparisons, not a controlled equal-compute test.

Keep 20k as a separate candidate, with the unchanged main model as reference. The current comparison supports retaining the broader-data gains while investigating movement choice/destination errors and recruitment retention. The stopped 30k checkpoint cannot be ranked without finishing an evaluation; no such evaluation has been restarted.

## Data, execution, and limits

- Started from `movement-epoch2-v1` final, with its exact adapter hashes; same frozen original prompt, model revision, LoRA configuration, and response-only objective. The original 10,000 prepared token sequences are byte-for-byte unchanged.
- Prepared 40,000 distinct examples: all original 10,000 plus 30,000 new semantic inputs. Every action-family count is exactly quadrupled, including 5,192 movement examples (12.98%) and 34,808 other examples. The overall 32,000 teacher / 8,000 corrective-source split is exact.
- **Source-by-action and phase proportions are not exact.** The existing corrective pool lacks enough examples to quadruple several source/action cells. Allocation preserves the action-family and source totals under those capacities; phase matching is approximate. The movement late-game share rises from about one-third to 42.22%. This is an additional limitation when attributing changes solely to sample count.
- All 40,000 teacher commands and complete score lists reproduced; semantic deduplication and train/validation separation passed. Movement examples cover 499 training maps and 973 episodes; the full corpus covers the same 500 training maps. Validation uses disjoint maps.
- Used fresh AdamW, unchanged peak LR 1e-4, 19 warmup updates, and cosine decay planned over 628 updates. Short 16-example batches enable exact 10k boundaries. Stopping did not retroactively change this schedule. The 20k checkpoint is midway through the original schedule.
- Longest input was 15,818 tokens; its real four-GPU forward/backward preflight passed without optimizer updates. Training logged approximately 33.0 minutes before cancellation.
- Thirty-nine distinct focused training/allocation/guard tests passed. Token and warm-start identities and the epoch order were verified. Termination prevented the normal end-of-run frozen-weight hash and final adapter receipt; this is not represented as a completed 40k run.
- All three completed evaluations eventually produced valid choices for all 1,475 positions. The fresh baseline differs slightly from prior runs despite identical initial adapter hashes; this run uses its own paired baseline. Bootstrap intervals exclude inference-runtime variability.
- These reused development maps measure imitation of a heuristic teacher, not gameplay strength or untouched final-test generalization. No RL, full games, final-test access, or automatic checkpoint promotion occurred.

## Artifacts

- [Complete available metrics and uncertainty](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/summary.json)
- [Stop receipt and checkpoint availability](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/stop-receipt.json)
- [Frozen experiment protocol](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/protocol.json)
- [Replay and checkpoint composition audit](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/replay-integrity.json)
- [Dataset composition](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/data-summary.json)
- [Last persisted training receipt](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/stopped-training-receipt.json)
- [Stopped app state](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/app-final-state.json)
- [Hash-verified 20k adapter configuration](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/adapter-checkpoint-314/adapter_config.json)

All saved adapters remain in the staging `polytopia-sft0-training` volume under `movement-scale40k-v1/training/model/`. The 20k adapter was downloaded and hash-verified separately; the main adapter was not replaced. Cloud coordinator status may record cancellation as failure, and the last trainer receipt retains `training`; the explicit stop receipt and app state document the user-requested termination.

Reproduce available metrics with `training/.venv/bin/python scripts/summarize_movement_mixed.py runs/movement-scale40k-v1`. This performs local analysis only.
