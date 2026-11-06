# Movement-enriched mixed training, unchanged prompt

Completed one pass over **10,000 distinct examples: 4,000 movement and 6,000 non-movement replay**, starting from the preserved `movement-epoch2-v1` main checkpoint. Optimizer training took **11.52 minutes** on four B200s. The prompt, model revision, LoRA settings, and response-only objective remained unchanged.

**Checkpoint 79 is the retention-focused recommendation among the checkpoints without predeclared point-estimate flags.** Movement-required tie-aware agreement rises from 4.59% to 13.10%; non-movement tie-aware agreement changes from 82.56% to 81.33%. This is a candidate for further testing, not evidence that every other skill was preserved. The main checkpoint has not been replaced.

## Checkpoint comparison

Both tie-aware columns accept any globally highest-scored action from the frozen heuristic teacher. Movement is scored on the **458 validation positions requiring movement**; non-movement on **900 controls, 100 per action family**. All 575 movement-labeled positions were evaluated, with the full group available in the JSON summary.

| Additional examples | Movement tie-aware / 458 | Movement exact / 458 | Non-movement tie-aware / 900 | Non-movement exact / 900 | Retention flags |
|---:|---:|---:|---:|---:|---|
| 0 | 21 (4.59%) | 8 (1.75%) | 743 (82.56%) | 597 (66.33%) | None |
| 2,560 | 60 (13.10%) | 28 (6.11%) | 712 (79.11%) | 577 (64.11%) | overall >3 pp; Build >10 pp |
| 5,056 | 60 (13.10%) | 25 (5.46%) | 732 (81.33%) | 601 (66.78%) | None |
| 7,552 | 75 (16.38%) | 28 (6.11%) | 713 (79.22%) | 569 (63.22%) | overall >3 pp; Unit abilities >10 pp; Research >10 pp |
| 10,000 | 67 (14.63%) | 29 (6.33%) | 720 (80.00%) | 583 (64.78%) | None |

Before launch, retention flags were defined as an overall non-movement tie-aware loss **greater than 3 percentage points**, or a single-family loss **greater than 10 points**. These are diagnostic thresholds, not proof of non-inferiority. In particular, a loss of exactly 10 points does not trigger the latter flag.

At checkpoint 79, movement improves **+8.52 points** (paired map-bootstrap 95% interval **+5.74 to +11.50**). Non-movement changes **-1.22 points** (**-2.91 to +0.56**). These intervals are exploratory and not adjusted for selecting among checkpoints.

The final checkpoint has the highest movement score among the two checkpoints without point-estimate retention flags, but its non-movement loss is **−2.56 points** (nominal paired 95% interval **−4.76 to −0.35**). The halfway checkpoint loses **1.22 points** (interval **−2.91 to +0.56**). Preferring halfway here is a **post-analysis recommendation prioritizing retention**, not a pre-registered checkpoint-selection rule. Neither result proves that the other skills are unchanged.

## Which other skills changed?

| Family | Main baseline | Checkpoint 79 | Checkpoint 118 | Final 157 |
|---|---:|---:|---:|---:|
| Attack | 69.00% | 68.00% | 79.00% | 74.00% |
| Unit abilities | 58.00% | 65.00% | 46.00% | 51.00% |
| Recruit | 75.00% | 68.00% | 66.00% | 67.00% |
| Harvest | 97.00% | 97.00% | 96.00% | 96.00% |
| Build | 85.00% | 81.00% | 82.00% | 81.00% |
| Research | 72.00% | 62.00% | 60.00% | 68.00% |
| Capture | 94.00% | 96.00% | 94.00% | 94.00% |
| City reward | 99.00% | 100.00% | 100.00% | 100.00% |
| End turn | 94.00% | 95.00% | 90.00% | 89.00% |

Each family has 100 held-out positions. Equal family weighting makes this a retention diagnostic rather than an estimate of natural gameplay action frequency. Exact scores, paired per-family uncertainty, and rates of mistakenly choosing movement are saved in the complete summary.

## Movement willingness versus destination quality

| Checkpoint | Chooses movement / 458 | Best-score agreement among chosen moves | Uniform-move expectation on those same selected states | Correct on the 98 states where the baseline already moved |
|---|---:|---:|---:|---:|
| 0 | 98 (21.40%) | 21.43% | 17.42% | 21/98 |
| 40 | 286 (62.45%) | 20.98% | 15.10% | 16/98 |
| 79 | 287 (62.66%) | 20.91% | 15.45% | 22/98 |
| 118 | 343 (74.89%) | 21.87% | 14.78% | 21/98 |
| 157 | 304 (66.38%) | 22.04% | 14.80% | 24/98 |

Uniformly choosing a legal move on every movement-required position has analytic expected teacher agreement of **14.26%**. Checkpoint 79 differs from that reference by **-1.15 points** (interval **-4.18 to +1.92**). This is an action-selection reference on saved positions, not a gameplay baseline.

On the fixed 98 states where the main baseline already moved, checkpoint 79 changes correctness by **+1.02 points** (interval **-8.00 to +10.00**). Thus the aggregate improvement should not be read as a demonstrated broad improvement in destination selection. Conditional comparisons on each checkpoint's chosen-movement states are descriptive and use changing subsets; they do show that its selected moves are not simply uniformly random.

The matched uniform-movement and fixed-baseline-cohort uncertainty breakdowns were added during analysis; they did not select training data or alter the fixed stopping schedule.

At the selected compromise checkpoint, movement agreement is **19.59% opening**, **11.83% exploration**, and **7.80% late-game**. Late-game movement remains especially weak.

## Data and protocol

The 4,000 movement examples span **500 training maps and 968 episodes**. **3,418** are new semantic inputs relative to the main checkpoint’s original corpus; 582 are replay. Opening, exploration, and late-game counts are 1,333, 1,334, and 1,333. Of the 4,000, **3,055 on 499 maps require movement for any globally top teacher score**. The other examples retain the teacher’s original choice among cross-family ties.

The 6,000 non-movement examples are replay from the original corpus: 5,658 teacher-trajectory examples and 342 model-collected corrections. Sampling balances action families and phases while preferring less-represented maps. It does not preserve the entire original 8,702 non-movement set, so this is a test of one replay mixture, not an exhaustive retention recipe. No model predictions or held-out outcomes selected the training examples.

All 10,000 teacher commands and complete score lists reproduced exactly. Public-input, split, semantic-deduplication, tokenizer, and frozen-prompt checks passed. Longest input: 14,934 tokens, within the already stress-tested runtime range. Total processed tokens: 45,456,631. Prepared tokens were compressed from 491,257,873 to 40,845,531 bytes for upload; decompression preserved their audited hash.

Used fresh AdamW intentionally for the changed mixture, the unchanged peak learning rate 1e-4, effective batch 64, five warmup updates, and cosine decay over 157 updates. Every example appears exactly once. The main checkpoint already had two passes over its prior corpus; these percentages refer only to this new mixed pass. This experiment does not isolate enrichment, new examples, and replay allocation as separate causal factors.

All five evaluations use the same ordered 1,475 inputs, prompt tokens, greedy runtime configuration, and batch size 64. The primary movement group covers 97 validation maps; the control group covers 100. Training and validation maps are disjoint. The fresh baseline differs slightly from older evaluations (4.59% versus 4.15% movement); comparisons here use this run’s common input/batch order and unchanged initial adapter hash. Runtime variation is not included in the bootstrap intervals.

## Decision and next useful work

Keep checkpoint 79 as a **separate candidate**, and retain the main checkpoint as the reference. Mixed supervision avoids the catastrophic non-movement collapse of the 128-example movement-only probe, but the measured family regressions prevent claiming lossless improvement. Continuing this same run longer is not supported by its retention curve.

The next diagnostic should focus on **choosing movement destinations and retaining the weaker non-movement families**. A controlled spatial/action-representation comparison can use this exact corpus, budget, and held-out tests. Separately, a smaller-update or full-non-movement-replay experiment can test retention; avoid changing representation and replay policy together if the aim is to identify the cause. Neither test needs full RL. The current result motivates these comparisons but does not prove that representation alone is the bottleneck.

## Execution and artifacts

All 157 optimizer updates and four checkpoint publications completed. Initial adapter tensors match the main checkpoint exactly, the frozen base stayed unchanged, and checkpoint 157 matches the final adapter. All five evaluations completed all 1,475 positions. Cloud workflow wall time was **22.07 minutes**, excluding local data auditing/tokenization and upload. Training container occupancy was **12.66 minutes on four B200s**; total evaluation occupancy was **37.52 A100 GPU-minutes**.

The Modal app `ap-ms4SNwVHax7Hva6S5KuWFK` is stopped with zero tasks.

Forty-one focused selection, warm-start integrity, scoring, and training-runtime tests passed. The full pipeline additionally verified all raw teacher labels, compressed-token identity, training parameter integrity, and evaluation input identities. No full games, RL, final-test evaluation, or automatic promotion occurred. Repeated use of these validation maps makes this development evidence, not a final generalization claim.

- [Complete metrics and uncertainty](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-mixed-v1/summary.json)
- [Frozen protocol](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-mixed-v1/protocol.json)
- [Training integrity](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-mixed-v1/training-integrity.json)
- [Dataset composition](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-mixed-v1/data-summary.json)
- [Completed execution](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-mixed-v1/cloud-status.json)
- [Checkpoint 79 adapter configuration](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-mixed-v1/adapter-checkpoint-79/adapter_config.json)
- [Final adapter configuration](/Users/K.Z./Desktop/projects/polytopia-bot/runs/movement-mixed-v1/adapter-final/adapter_config.json)

Prediction validity (first-attempt / eventually valid, each out of 1,475): checkpoint 0: 1473 / 1475; checkpoint 40: 1471 / 1475; checkpoint 79: 1472 / 1475; checkpoint 118: 1474 / 1475; checkpoint 157: 1474 / 1475. The score differences are therefore primarily behavioral, not invalid-output failures.

All intermediate/final adapters, optimizer state, and full evaluation attempts remain on the staging `polytopia-sft0-training` volume under `movement-mixed-v1/`. The local checkpoint-79 and final-adapter downloads were hash-verified and are separate from the main adapter. Reproduce analysis with `training/.venv/bin/python scripts/summarize_movement_mixed.py runs/movement-mixed-v1`.
