# Movement-only overfit probe, unchanged prompt

**The current model and prompt can fit movement examples:** the probe reached **127/128 exact teacher matches (99.22%)** after 20 focused passes. It first passed the predeclared fit target at the ten-pass checkpoint: 125/128 tie-aware and 124/128 exact matches.

**That capacity did not produce a strong general movement policy.** Held-out movement peaked at 17.25% after five passes, then finished at 14.63%, close to the 14.26% expected from uniformly choosing a legal movement action on these positions. Non-movement control accuracy collapsed from 66.67% to 13.89%. Retain this as a diagnostic adapter; the main checkpoint remains unchanged.

## Experiment

Started from the final adapter of `movement-epoch2-v1`, not a newly initialized adapter. Selected **128 ordinary teacher-demonstration positions** from the exact original 10k training corpus. Every globally best teacher action in each selected state is movement. Selection used deterministic hash ordering and round-robin sampling across phase and legal-catalog size, with one state per map and no selection based on model predictions.

The probe contains 19 opening, 59 exploration, and 50 late-game states; 50 have fewer than 64 legal actions, 39 have 64–127, and 39 have at least 128. Its longest training sequence is 11,928 tokens. This deliberately broad small sample is not weighted to population prevalence.

The **prompt, full public observation, full legal catalog, teacher labels, and prepared token sequences were unchanged**. Raw examples were audited again, and the trainer compared every selected token/label sequence directly with the original parent corpus. The frozen base, model revision, LoRA configuration, response-only objective, peak learning rate, and optimized runtime stayed the same.

For this fitting-capacity test, used fresh AdamW, batch size 8, 20 passes, learning rate `1e-4`, ten warmup updates and cosine decay across **320 updates / 2,560 example presentations**. This changes repetition and batch size as well as data concentration; it does not independently identify their causal contributions. Training took **234.60 seconds (3.91 minutes)** on four B200 GPUs.

Evaluated the starting adapter and checkpoints after 5, 10, and 20 passes on a fixed ordered set of **858 positions**, using the same greedy vLLM inference and batch size 64:

- 128 probe training examples on 128 maps.
- 128 other ordinary teacher movement examples on 128 different training maps. The parent already saw these; they were excluded only from focused training and are **not genuinely held out**.
- All 458 movement-required validation positions on 97 maps never used for training.
- 144 non-movement validation controls, 16 per action family 2–10, chosen without inspecting predictions.

The three movement groups have disjoint maps. All scored movement states require movement for a globally top teacher score, so a non-movement tie cannot count as success. The fit target was at least **95% tie-aware** and **90% exact** agreement on the 128 probe states. No RL, representation change, full games, final-test access, or automatic policy promotion occurred.

## Checkpoint results

Tie-aware agreement accepts any globally top-scored teacher choice. Exact agreement requires the teacher's particular selected command. Non-movement controls use exact agreement.

| Focused passes | Probe tie-aware / 128 | Probe exact / 128 | Other parent-training tie-aware / 128 | Held-out tie-aware / 458 | Non-movement exact / 144 |
|---|---:|---:|---:|---:|---:|
| 0 | 5 = 3.91% | 4 = 3.13% | 5 = 3.91% | 19 = 4.15% | 96 = 66.67% |
| 5 | 88 = 68.75% | 84 = 65.63% | 20 = 15.63% | 79 = 17.25% | 30 = 20.83% |
| 10 | 125 = 97.66% | 124 = 96.88% | 15 = 11.72% | 66 = 14.41% | 25 = 17.36% |
| 20 | 127 = 99.22% | 127 = 99.22% | 17 = 13.28% | 67 = 14.63% | 20 = 13.89% |

The final fit includes all 50 late-game probe states and 38/39 states with at least 128 legal actions. The model's inability to learn the original movement examples was therefore not an absolute inability to fit their existing representation.

Held-out agreement improved relative to the fresh baseline: **+13.10 percentage points** at five passes (paired map-bootstrap 95% interval **+9.74 to +16.56**), and **+10.48 points** at twenty passes (**+6.20 to +15.02**). These gains require interpretation because movement-only supervision changes the action-family prior dramatically.

The final non-movement decline is **−52.78 points**, with interval **−60.43 to −45.52**. All final predictions were legal on the first attempt (858/858), so this is a behavioral regression rather than a formatting failure. One ten-pass prediction remained invalid after repairs; the other checkpoints were fully valid.

## Movement bias versus choosing useful moves

On held-out movement-required states, movement selection rose from **87/458 (19.00%)** to **455/458 (99.34%)**. Only 67 of those final moves attained the teacher's best score: **14.73% conditional agreement**. The final errors comprise 388 lower-scored moves and only 3 non-movement choices. The probe largely removed reluctance to move without learning reliable destination selection on unseen states.

An analytic reference helps separate those effects. For each held-out position, a uniform random legal move has expected agreement equal to the number of teacher-best moves divided by the number of legal moves. Averaging across the fixed 458 states gives **14.26%**. This is a saved-position teacher-agreement reference, not a gameplay baseline or win-rate claim.

| Checkpoint | Held-out agreement minus uniform-movement expectation | Exploratory 95% map-bootstrap interval |
|---|---:|---:|
| Five passes | +2.99 points | +0.42 to +5.63 |
| Ten passes | +0.16 points | −2.77 to +3.18 |
| Twenty passes | +0.37 points | −3.36 to +4.37 |

There is a modest early signal beyond simply choosing movement, but that advantage is not maintained by further fitting. These exploratory intervals are not adjusted for examining multiple checkpoints or runtime variation; the random-movement comparison was added during analysis, not used to select the training examples or stop training.

A comparison on fixed states gives the same caution. On the 87 held-out states where the baseline already moved, teacher-best choices changed **19 → 15 → 18 → 16**. The remaining 371 states, where it previously chose another action family, account for the final gains: **0 → 64 → 48 → 51**. Thus the overall held-out increase should not be described as a demonstrated improvement in destination selection wherever movement was already occurring.

The same bias appears in non-movement controls: movement choices increased from **1/144 to 93/144**, while exact agreement fell from 96 to 20. Training exclusively on movement is unsuitable as the next production recipe.

## Interpretation and next step

This clears the basic fitting-capacity check: the model can learn the original movement targets, including difficult catalogs and late-game states, without a prompt or representation change. It does **not** establish that the existing representation supports good spatial generalization. Repeating a tiny set mostly memorized it; held-out performance peaked earlier and the model forgot other behaviors.

The next useful test is **movement-enriched mixed SFT using a larger, more diverse set of existing teacher movement examples**, while retaining non-movement supervision. Start from the preserved parent checkpoint, not this specialized probe. Evaluate early checkpoints on held-out movement-required states, destination quality relative to the uniform-movement reference, and non-movement controls. Select for the tradeoff rather than training-set fit.

Keep the current prompt as that experiment's baseline. If balanced, diverse training still fails to improve held-out destination choice beyond the movement-prior effect, compare an explicit spatial/action representation against it using the same data and training budget. The probe rules out neither data-coverage nor representation limitations for generalization. It does rule out treating more repetitions of these 128 examples as a solution. RL remains unnecessary for this next diagnostic.

## Integrity, execution, and artifacts

The run completed all 320 updates and 20 full permutations of the selected examples. The initial adapter tensor hash matches the parent's final tensor hash; the frozen base stayed unchanged; the probe adapter changed; all three checkpoints were published; checkpoint 320 matches the final adapter. The original parent adapter is intact. The probe adapter is saved separately with safetensors SHA256 `f7ff8dfdac5b153b65ce4f947bfef4add9e78a3d31f212b65e58447c6cc42112`.

All four evaluations completed all 858 positions with matching input identities and prompt token counts. Total workflow wall time was **610.34 seconds (10.17 minutes)**. Training occupied the four-B200 container for 293.23 seconds; the four A100 evaluations occupied approximately **23.65 GPU-minutes in total**. The app `ap-D8AHdwC4pOOTIzRA8bxwjG` is stopped. Twenty-five focused selection, subset-integrity, continuation, and runtime tests passed, plus one test covering the fractional uniform-movement reference and map-cluster calculation. Python compilation and whitespace checks passed.

- [Complete summary, phase/catalog breakdowns, and uncertainty](../runs/movement-overfit-v1/summary.json)
- [Frozen protocol and sampling design](../runs/movement-overfit-v1/protocol.json)
- [Completed execution receipts](../runs/movement-overfit-v1/status.json)
- [Training integrity checks](../runs/movement-overfit-v1/training-integrity.json)
- [Selected examples and parent-token provenance](../runs/movement-overfit-v1/prepared/manifest.json)
- [Final predictions](../runs/movement-overfit-v1/report-320.json)
- [Separate probe adapter configuration](../runs/movement-overfit-v1/adapter-final/adapter_config.json)

Reproduce the analysis with `.venv/bin/python scripts/summarize_movement_probe.py runs/movement-overfit-v1`. Full attempts, intermediate adapters, final adapter, and optimizer state remain under `movement-overfit-v1/` on the staging `polytopia-sft0-training` volume. Local run artifacts remain excluded from Git under the existing rules.
