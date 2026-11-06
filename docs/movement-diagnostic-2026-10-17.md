# Movement diagnosis: training versus held-out positions

The deployed checkpoint has not learned the movement behavior well even on its actual training examples. This is not primarily the pattern of high training accuracy followed by a large held-out collapse. The comparison supports testing additional supervised optimization before investing in a representation redesign, but cannot establish that more epochs alone will solve the problem.

## Measured results

Evaluated every movement-labeled example in the exact 10,000-example training corpus (1,298 positions on 402 maps). Reused the same checkpoint's 575 held-out movement predictions on 98 disjoint validation maps. All 1,873 teacher targets reproduced exactly. The original prepared training prompts, data/token hashes, tokenizer, adapter, and held-out inputs were verified. No model weights, training labels, prompts, or rules changed; no RL, gameplay rollouts, or final-test evaluation ran.

“Movement-labeled” means the teacher's selected action is movement. Tie-aware agreement accepts any action with the global maximum teacher score, including other action families when they tie. It does not mean optimal play or win rate.

| Metric | All training movement examples | Teacher-demonstration training subset | Held-out movement positions |
|---|---:|---:|---:|
| Positions | 1,298 | 842 | 575 |
| Exact teacher agreement | 72/1,298 = **5.55%** | 15/842 = **1.78%** | 11/575 = **1.91%** |
| Tie-aware agreement | 276/1,298 = **21.26%** | 158/842 = **18.76%** | 107/575 = **18.61%** |
| Model chooses any movement | 405/1,298 = **31.20%** | 112/842 = **13.30%** | 66/575 = **11.48%** |
| First-attempt validity | **99.92%** | **99.88%** | **100%** |

The overall train-minus-held-out tie-aware gap is **2.65 percentage points**. An independent map-cluster bootstrap (5,000 resamples, seed 17) gives a descriptive 95% interval of **−3.34 to +8.46 points**. The teacher-demonstration-only difference is **0.16 points**, with interval **−5.21 to +5.09**. These intervals account for within-map correlation, not every possible source of distribution shift or repeated evaluation.

The other 456 training positions are teacher corrections of states visited by the older model. They achieve 25.88% tie-aware agreement and 64.25% movement selection. They have shorter prompts and smaller legal-action catalogs (mean 29 actions, versus 92 in teacher demonstrations and 94 held out). Their higher score cannot be attributed solely to the value of corrective data: state distribution and complexity differ.

## Ties conceal a larger action-selection problem

In some positions, movement ties with ending the turn or another action family. Separate the cases where **every globally top-scored action is movement**:

| Metric, movement strictly preferred by teacher | All training | Teacher-demonstration training | Held out |
|---|---:|---:|---:|
| Positions | 1,037 | 632 | 458 |
| Tie-aware agreement | 87/1,037 = **8.39%** | 12/632 = **1.90%** | 16/458 = **3.49%** |
| Chooses any movement | 329/1,037 = **31.73%** | 73/632 = **11.55%** | 49/458 = **10.70%** |
| Chooses a non-movement action | **708** | **559** | **409** |
| Chooses a lower-scored movement | **242** | **61** | **33** |

Among the 458 held-out cases, the model researches 153 times and recruits 126 times. Together these account for 279 of the 409 wrong-family decisions. The immediate failure is often **when to move**, before the question of where to move. When it does move in these cases, 16/49 choices attain the top score; direction/target selection also needs work.

For ordinary teacher demonstrations, the model chooses a non-movement action in **559/632 movement-required training positions**. This makes the failure visible directly in examples that received movement supervision. The issue is not explained away by arbitrary teacher tie-breaking or invalid JSON.

As an illustrative held-out case, position `100576-2-548` has a unique teacher-best move from tile 115 to 86, scored 136. The model recruits at tile 67, scored −4 by the same teacher. The scores are heuristic preferences, not measured returns. Additional descriptive examples are in the diagnostic output; none are added to training.

## Distribution and ambiguity checks

- All-training and held-out phase proportions are almost identical: approximately one third opening, exploration, and late game. Reweighting training to held-out phase frequencies changes tie-aware agreement from 21.263% to 21.260%. Teacher-only training changes from 18.765% to 18.766%.
- On movement-required late-game teacher demonstrations, agreement is **0/183 training and 0/141 held out**. Late-game movement selection is poor in both sets.
- A single best teacher action exists in 553 training and 184 held-out positions. Agreement is **7.59% training and 2.72% held out** on those unambiguous cases. Ambiguous ties are not the whole problem.
- With at least 128 legal actions, the model chooses movement in **4.28% of training and 1.02% of held-out movement-labeled positions**. This is consistent with action-catalog complexity being relevant, but phase, source and state complexity are confounded; it is not proof of a representation cause.

## Interpretation and next experiment

**Finding:** the current checkpoint underfits movement behavior as measured by its deployed greedy policy, especially ordinary teacher demonstrations. Training-set success is far too low to characterize this as merely memorizing training moves and failing on new maps. There is some train/held-out difference, particularly in corrective states and unambiguous cases, but both populations remain weak.

**Unresolved:** insufficient optimization, the relative influence of movement examples, and difficulty extracting spatial/action-value information can all produce low training accuracy. A train-versus-held-out comparison cannot causally distinguish them. The earlier mixed 128-example fit check demonstrated memorization/plumbing, not that a single epoch of the full corpus learned a reusable movement policy.

**Recommended next test:** continue the same adapter for one additional pass over the unchanged 10k corpus and unchanged prompt, with a deliberately specified continuation learning-rate schedule. Evaluate the same movement diagnostics at intermediate checkpoints, including the movement-required subset and non-movement skill controls. Do not change data and representation simultaneously.

- If training and held-out movement both improve, additional SFT is helping.
- If training improves while held-out stays poor, investigate generalization and representation/data coverage.
- If training movement remains poor, use a small movement-only overfit probe to localize the learning problem, then compare explicit spatial features or clearer action targets in a controlled interface experiment.

No continuation or representation change has been launched by this diagnostic. Full-game evaluation should follow a promising behavioral change, rather than run after every diagnostic checkpoint. Existing acceptance gates remain unchanged.

## Execution and evidence

One A100-80GB inference invocation evaluated the 1,298 training positions in **362.26 occupied seconds** (87.23 seconds model load, 271.11 seconds evaluation; remaining time subprocess overhead). Held-out predictions required no new GPU inference. The app `ap-20BdEuTZlqyoupVFM4LdXw` is stopped with zero tasks. Three focused scoring tests pass.

The model is the final adapter from `speed-v2/end-to-end-v1`; its safetensors SHA256 is `e935f3d62667fe5264b252fe312fd58baf2eb10183bbc92e5a2d5624a548202f`. Both splits use the pinned Qwen3 model, vLLM 0.10.2, greedy decoding, batch size 64, the complete legal catalog and the existing two-repair policy. Training needed one repair; every final action was legal. Held-out predictions were produced in the earlier full evaluation, so batching-dependent numerical variation is not excluded. The result concerns this deployed checkpoint/backend, not hypothetical teacher-forced training accuracy.

- [Complete diagnostic summary and per-position results](../runs/movement-diagnostic-v1/summary.json)
- [Protocol and input identities](../runs/movement-diagnostic-v1/protocol.json)
- [Input audit](../runs/movement-diagnostic-v1/input-audit.json)
- [Training inference receipt](../runs/movement-diagnostic-v1/execution.json)
- [Training predictions](../runs/movement-diagnostic-v1/training-report.json)
- [Held-out error examples](../runs/movement-diagnostic-v1/held-out-error-examples.json)

Reproduce scoring with `.venv/bin/python scripts/diagnose_movement.py summarize runs/movement-diagnostic-v1`. The preparation and Modal entrypoints require a fresh output ID for a new inference run. Full attempts remain on the `polytopia-sft0-training` volume under `movement-diagnostic-v1/training-results/attempts.jsonl.gz`. Machine-readable run artifacts remain outside Git under the existing ignore rules; this report and the diagnostic code can be versioned normally.
