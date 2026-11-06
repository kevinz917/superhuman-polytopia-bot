# Why movement dipped at the 10k checkpoint

The clearest observed mechanism is a **temporary shift from movement toward construction**, especially roads. It is not evidence that adding 10,000 examples inherently hurts movement, and it is not a demonstrated destination-selection collapse. The saved artifacts support a specific behavioral diagnosis; they cannot isolate the optimization cause without a controlled comparison.

This investigation used saved predictions, all 40,000 tokenized examples and raw labels, reconstructed batch order, training logs, and a second existing evaluation of the same main adapter. **No training or GPU evaluation was restarted.**

## 1. The drop is seven net decisions, with substantial uncertainty

| Checkpoint | Correct movement / 458 | Chooses movement / 458 | Chooses build / 458 | Chooses road / 458 | Non-movement agreement / 900 |
|---|---:|---:|---:|---:|---:|
| Main | 20 (4.37%) | 96 | 45 | 13 | 741 (82.33%) |
| 10k | 13 (2.84%) | 73 | 104 | 53 | 770 (85.56%) |
| 20k | 36 (7.86%) | 171 | 43 | 19 | 774 (86.00%) |

The headline 4.37% → 2.84% is **−1.53 percentage points**, not a large absolute collapse. Sixteen previously correct decisions became wrong, nine previously wrong decisions became correct, and 433 retained their correctness status. Losses span 13 maps and gains eight maps.

The paired map-bootstrap 95% interval for the accuracy change is **−3.94 to +0.95 points**. It includes no change, so this run does not establish a repeatable reduction in underlying movement accuracy. These intervals are exploratory and do not adjust for checkpoint selection or runtime variation.

The behavioral shift is clearer: the movement-choice rate falls **5.02 points** (interval **−8.24 to −2.03**), while build choices rise **12.88 points** (**+8.88 to +16.74**) and road choices rise **8.73 points** (**+5.76 to +11.70**). On these 458 positions, only movement can receive a globally top teacher score.

## 2. Exactly where the seven lost successes went

| Fixed cohort defined by baseline and 10k choices | Positions | Baseline correct | 10k correct | Net change |
|---|---:|---:|---:|---:|
| Both checkpoints move | 54 | 10 | 10 | 0 |
| Baseline moves; 10k does not | 42 | 10 | 0 | −10 |
| Baseline does not move; 10k moves | 19 | 0 | 3 | +3 |
| Neither checkpoint moves | 343 | 0 | 0 | 0 |

Thus **all of the net −7 is accounted for by changing whether to move**. Among the 54 positions where both checkpoints moved, six correct moves became incorrect and six incorrect moves became correct. There are individual destination errors, but they cancel in the aggregate on this fixed cohort.

Of the 16 lost successes, ten switched to a different action family: six build, two end turn, one recruit, and one research. The remaining six still moved but chose a teacher-suboptimal move. Twenty-three of all 42 move-to-non-move switches went to building.

Conditional accuracy among each checkpoint's chosen moves falls from 20.83% to 17.81%, but those denominators contain different states. The fixed-cohort accounting above is a stronger explanation of the net loss than treating that conditional change as proof of worse spatial reasoning.

Concrete examples from the saved predictions:

| Position | Main choice | 10k choice | 20k choice |
|---|---|---|---|
| `100626-1-362` | Move 78 → 63; teacher score 12 | Build a road on 191; score 4 | Move 78 → 63; score 12 |
| `100631-2-176` | Move 85 → 71; score 12 | Build a road on 71; score 4 | Move 85 → 71; score 12 |
| `100571-1-3` | Move 156 → 141; score 12 | Move 156 → 170; score −12 | Move 156 → 141; score 12 |

These are legal but suboptimal actions under the frozen heuristic teacher. They are not invalid JSON, catalog mistakes, or tie-breaking penalties. Teacher scores are heuristic preferences, not game returns. All 25 correctness-changing cases are saved in the linked case file.

## 3. The first 10k was not starved of movement

I reconstructed the actual shuffled order, verified its hash against the training receipt, and matched every raw record to its sampling metadata.

| Training quantity | Original 10k corpus | First 10k of this run | Second 10k of this run |
|---|---:|---:|---:|
| Movement labels | 1,298 | 1,336 | 1,293 |
| Labels where movement is strictly required for top score | 1,037 | 1,098 | 1,065 |
| Build labels | 1,114 | 1,124 | 1,127 |
| Road labels | 438 | 445 | 462 |
| Teacher-trajectory / corrective-source examples | 8,000 / 2,000 | 8,009 / 1,991 | 7,998 / 2,002 |
| Movement share of supervised response tokens | 12.49% | 12.95% | 12.52% |

Every effective batch in the first 10k included at least two movement examples. The last 1,000 examples included 124 moves and 114 builds. The final short batch had 16 examples: two moves and only one build. There was no construction-heavy final batch or unexplained disappearance of movement supervision.

Road labels rose by only seven relative to the original 10k corpus, while road choices on movement-required evaluation states rose by 40. That is inconsistent with a simple explanation based on a large accidental road oversampling.

There are real distribution differences below the action-family totals. The first 10k contained **2,539 original examples and 7,461 new examples**; it was not another pass over exactly the original corpus. Its movement examples contained 549 late-game states versus 433 originally, and 417 corrections versus 456 originally. Mean legal-move count was 41.05 versus 38.08. Source-by-action and phase proportions were only approximately preserved because some source pools were too small. These differences remain possible contributors; the experiment was not a perfectly isolated quantity-only comparison.

The fraction of movement labels with multiple globally top actions was similar: 56.74% in the first 10k versus 57.40% originally. More ambiguous ties are not an obvious explanation for this dip either.

## 4. The 10k optimization conditions were materially different

The run loaded the main adapter exactly, then intentionally started **fresh AdamW with peak learning rate 1e-4**. The same peak rate does not mean the same schedule:

| Schedule property | Previous standalone 10k pass | This run's first 10k |
|---|---:|---:|
| Planned optimizer updates | 157 | 628 for the full 40k |
| Warmup updates | 5 | 19 |
| Learning rate after update 157 | 0 | 0.00008786 |
| Sum of learning rates used in the first 157 updates | 0.00785 | 0.01414 |

At 10k, this run was still at **87.9% of peak learning rate**, whereas the previous standalone 10k schedule had decayed to zero. The sum of step learning rates was approximately **1.80× larger**. This sum is only a schedule comparison, not a measurement of parameter displacement or a proof of excessive updates.

At 20k the next-update learning rate had declined to 0.00005245, with another 10,000 examples processed. The movement score recovered to 36/458. More data, additional updates, and lower learning rates changed together, so the recovery does not identify which factor caused it.

The global token-averaged imitation objective provides no guarantee that every action family's held-out accuracy improves at every checkpoint. At 10k, build control accuracy improved **85% → 91%**, while building was applied too often on movement-required states. That is consistent with interference between skills or a shift in action preference under the high-rate restart. It is a plausible explanation, not a measured gradient-level cause.

I should therefore not frame the 10k checkpoint as an otherwise-identical repeat of the previous 10k run. It is an early, still-high-rate snapshot of a longer schedule over mostly new inputs. Likewise, calling this “overfitting” would go beyond the evidence: per-family training losses and gradients were not logged, and this audit did not run new model evaluations.

## 5. Implementation and evaluation checks

- The actual epoch-order hash matches the prelaunch reconstruction. Batch coverage, checkpoint boundaries, token-file hash, and raw-record/metadata alignment passed.
- The warm-start adapter hash matches the preserved main model. Frozen prompt/tokenizer checks and original-token replay checks passed. The evaluator and relevant prompt/serving source hashes match the prior mixed run.
- The loss uses the global response-token denominator; gradients are summed across ranks. The code does not accidentally average an already globally normalized gradient again.
- First-10k losses and gradient norms were finite. Only two of 157 updates exceeded the clipping threshold of 1; the maximum pre-clipping norm was 1.386. The longest-input preflight made zero optimizer updates and cleared gradients before training. No observed numerical failure explains the dip.
- All 458 primary movement outputs were valid on their first attempt at baseline, 10k, and 20k. Invalid-output repair cannot explain this movement regression.
- The exact same main adapter, inputs, prompt/serving source, configuration, and batch size had an earlier saved baseline evaluation. It scored 21/458 versus 20/458 now. Thirty-four strict movement commands differed, but only three changed correctness (one gain, two losses). This demonstrates repeat-evaluation variability; two samples are insufficient to characterize its full distribution. Comparing 10k against that earlier baseline still gives an uncertain accuracy drop: −1.75 points, interval −4.10 to +0.69.
- Cancellation left no normal final frozen-weight hash; this audit does not claim the interrupted 40k run passed an end-of-run check that never occurred.

## Conclusion and the most informative next test

**Observed:** the 10k checkpoint temporarily preferred construction too much. The net movement accuracy loss came from moving less often, rather than a net loss among states where both models moved. The measured accuracy drop itself is uncertain.

**Leading explanation to test:** the high-learning-rate restart and stretched decay schedule shifted action preferences while the model adapted to mostly new mixed data. The slightly different state/source distribution may also contribute. Neither cause is isolated by the saved artifacts.

**Not supported:** a claim that more examples generally hurt, that movement supervision was accidentally omitted, that the model suffered numerical collapse, or that representation must already be changed.

The cleanest next controlled experiment would reuse **exactly the same first 10,000 examples in the same order**, start from the same main adapter, retain fresh AdamW and every other setting, and change only the learning-rate schedule to finish decaying by 10k. Compare its movement/build behavior and non-movement retention with the existing 10k checkpoint. This separates schedule effects from additional unique data much more cleanly than scaling again or changing the prompt simultaneously. A lower-peak-rate variant is a separate subsequent test, not another factor to combine into the same comparison. No proposed run has been launched.

The 20k checkpoint's aggregate recovery also does not restore every old decision: only six of the 16 lost baseline successes return, and only three of the nine new 10k successes remain. The policy continues to change; aggregate improvement is not evidence of monotonic retention of each skill or example.

## Reproducible evidence

- [Prediction transitions, uncertainty, and repeat-baseline comparison](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/investigation-10k/prediction-audit.json)
- [All 25 changed decisions](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/investigation-10k/changed-decisions.json)
- [Training composition and schedule audit](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/investigation-10k/training-audit.json)
- [Every logged batch and its actual learning rate](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/investigation-10k/batch-exposure.json)
- [Evaluation source/configuration integrity](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-scale40k-v1/investigation-10k/evaluation-integrity.json)

[Analysis script](/Users/K.Z./Desktop/projects/polytopia-bot/scripts/investigate_scale10k.py). Run with `predictions` or `training`; both commands operate locally on saved files. The training audit verifies the entire 40k prepared corpus, while exposure analysis uses the logged prefix.
