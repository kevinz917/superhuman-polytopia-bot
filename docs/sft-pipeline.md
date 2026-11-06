# SFT pipeline and public-state policies

**Execution update (2026-10-16):** SFT-0 work has started. See the [execution log](sft-execution.md) for the live adapter, public memory, reserved maps, qualification attempts and model profiling. No trained game checkpoint is claimed.

Implemented on 2026-10-16 and extended after the frozen GPU simulator handoff. This is runnable training infrastructure and an initial heuristic teacher, not SFT-0 or evidence of full-game competence. APEX setup is deferred until after SFT, as requested.

## Implemented components

- `src/polytopia_bot/learning/interface.py`: strict public observation schema, deterministic complete catalog rendering, public rule glossary, response parsing, stale-version checks, and three-attempt policy inference with preserved attempts and no fallback action.
- `teacher.py`: deterministic balanced, economy, and pressure heuristic variants. Every decision includes semantic commands, scores, explanations, a stable tie-break, and a teacher source/rule hash. The teacher uses public observations and static rule tables only; it never imports JAX state or a simulator branch.
- `collection.py`: episode collection against the existing `Environment` protocol, separate next-player observations, complete traces, preservation of losing/truncated episodes, and a readiness gate for real collection. The live adapter is implemented in `adapter.py` against the frozen JAX simulator.
- `data.py`: observation-to-command records, training-split checks, semantic deduplication, conflicting-label rejection, fixture exclusion, explicit response/EOS masks, and padding-safe collation.
- `training.py`: streaming preparation, indexed JSONL loading, immutable tokenizer/model revision, single-GPU BF16 LoRA, checkpoint saves at 25%, 50%, and 100%, run receipts, and qualification checks before real training.
- `inference.py`: base/adapter inference with exact prompt/response token logging, matching base revision, and context-budget enforcement. `serving.py` adds batched requests with per-request token logs; `gameplay.py` coordinates independent full games.
- `adapter.py`, `memory.py`, and `encoding.py`: live frozen-engine transport, seat-isolated factual memory, and lossless compact-v3 prompts with explicit action IDs.
- `splits.py`, `research.py`, and `prompt_search.py`: reserved maps, teacher qualification, and bounded prompt-screen preparation.
- `corpus.py`, `selection.py`, and `freeze.py`: qualified raw episodes, deterministic stratified sampling, paired prompt selection, and a verified prompt manifest required before real training.
- `smoke.py`: local real-gradient test using a tiny randomly initialized Qwen3 model. It verifies loss/accuracy improvement, frozen base weights, changed adapters, scheduled checkpoints, and identical logits after adapter reload.

The training implementation uses **Transformers Trainer + PEFT with precomputed labels**. It does not require TRL to reformat or mask the dataset. No prompt token, padding token, or text after the assistant end-of-message marker contributes to the loss. Packing and silent truncation are disabled. `<|im_end|>` is supervised even if padding uses the same token ID.

## Run the local checks

Run commands from the repository root. The ML runtime is separate from the simulator's JAX environment and is pinned for Python 3.12; `uv.lock` includes Linux CUDA dependencies as well as the local macOS dependencies.

```sh
uv sync --project training --locked --python 3.12
uv run --project training --locked python -m pytest tests/test_learning.py tests/test_learning_training.py -q
uv run --project training --locked python scripts/sft.py tiny-smoke --output runs/my-sft-smoke
```

Output directories must be new. The tiny smoke downloads no pretrained model weights, uses CPU, and creates its own tiny tokenizer and model. The optional pinned-Qwen tokenizer test skips if the tokenizer is not already cached. To exercise actual Qwen tokenization without downloading 4B weights:

```sh
uv run --project training --locked python scripts/sft.py fixture --output runs/my-fixtures.jsonl
uv run --project training --locked python scripts/sft.py prepare \
  --input runs/my-fixtures.jsonl --output runs/my-prepared-fixtures --allow-fixtures
```

Preparation downloads only the tokenizer at the pinned model revision. Each prepared directory contains exact messages and semantic targets (`examples.jsonl`), token IDs and labels (`tokens.jsonl`), the tokenizer, and a manifest with source hashes, data audit, token counts, and template/tokenizer hashes. An interrupted or rejected preparation has no completed manifest and cannot train.

## Public observation schema

The existing `Observation` envelope has `player` (1 or 2), monotonic `state_version`, `visible_state`, `remembered_state`, and `legal_actions`. This layer accepts only a new allowlisted public schema; it does not accept raw GPU state or omniscient snapshots.

`visible_state` fields:

| Field | Shape |
| --- | --- |
| `board_size`, `turn`, `stars` | Integers; stars belong to the acting player. |
| `technology` | Acting player's known technology IDs. |
| `tiles` | Every board tile once. Unknown tiles contain only `tile` and `known: false`. Known tiles may include public `terrain`, `owner`, `resource`, and `structure` IDs. |
| `units` | Visible units: `tile`, `owner`, `kind`, `health`; optional public `veteran`, `moved`, `attacked`. Invisible enemies must already be omitted by the adapter. |
| `cities` | Public `tile` and `owner`; own cities may additionally include `level`, `population`, `production`, `progress`. |

`polytopia-public-v2` supports the current adapter's own-unit/city flags, diplomacy and achievements. Opponent private fields remain rejected. `remembered_state` uses `polytopia-memory-v1`: five executed own actions and up to 32 timestamped historical entity facts absent from current public observations. Current observations supersede history. The runner updates each seat independently, resets memory on every episode and restores it with private snapshots. Tests cover hidden-state invariance, seat isolation, stale timestamps, idempotency, replay and record caps. Full conversation history and model-written plans remain outside SFT-0.

Prompts use the lossless `polytopia-compact-v3` transport and a 16,384-token context allowance, expanded before training after measured overflow and ID-comprehension checks. No legal action is pruned. The canonical semantic observation is retained separately for audits and labels.

Every `LegalAction` has an upstream `action_id` and exactly four integer command fields: `kind`, `source`, `target`, `parameter`, matching the existing GPU ABI. Rendering sorts semantic commands and assigns prompt-local `a0`, `a1`, etc. Parsing maps that response back to the original engine action. The engine must recheck `expected_state_version` when executing it. Static rule definitions provide readable unit, technology, and structure names; IDs never encode hidden state.

The caller must supply the complete catalog. This layer rejects duplicates, empty catalogs and oversize prompts, but cannot prove that an engine omitted no legal move or derived legality without a hidden-state leak. Unknown-field rejection is likewise not a proof of correct redaction. Those guarantees require the adapter's hidden-state invariance tests before collecting real data.

Any further change to information content or memory requires a new interface experiment and renewed comparisons. Unrecognized extra fields fail explicitly. The public schema is the frozen research observation contract, not a claim to reproduce every statistic in the commercial UI.

## Teacher and collection

`ScriptedTeacher("balanced")` prioritizes mandatory rewards, captures, useful combat/recovery, population/income, visible research opportunities, recruitment demand, and movement toward visible settlements or exploration. It can end its turn instead of spending on low-value actions and strongly disfavors resignation. Combat scoring is an approximate preference, not an exact damage or guaranteed-kill oracle. Naval/diplomatic specialization, broader tactics, and full-game teacher strength remain unvalidated. Economy and pressure variants share the same implementation and are diversity candidates, not independently qualified opponents.

`teacher.select(observation)` returns the chosen original engine action, prompt-local JSON response, all scores, and the teacher hash. `WeakBaseline()` supplies a separate, unstrategic deterministic legal-action opponent; it avoids resignation but is not yet calibrated as B-lite. `demonstrate(...)` creates an SFT record. `collect_episode(...)` plugs into an implementation of `Environment`; it accepts `full_game_ready=True` for real collection only after certification, or `fixture=True` for explicitly synthetic integration tests. Do not set the gate based on these unit tests.

The collector closes its environment after each episode and emits two artifacts: learner-side demonstration records and a complete episode receipt containing both policies' actions, rewards, terminal/truncation flags, and the reason. Retain both, including losing and truncated episodes. Deduplication and phase/action-family sampling follow collection; the 50,000-example corpus is not generated by this implementation step.

## Real training after environment and teacher qualification

Before rendering the real smoke/main corpus, complete the [starting-prompt optimization and freeze stage](sft-checkpoint-plan.md#starting-prompt-optimization-before-sft). The current `SYSTEM` prompt is provisional. The saved-position screen, full-game runner, paired selection and frozen-prompt artifact loading are implemented. Their live execution status is recorded in the execution log; implementation alone does not satisfy the gate.

The frozen split manifest is a JSON object with `train`, `validation`, and `test` arrays of board hashes. Preparation checks disjointness and requires every record's board hash to be in `train`. Symmetry-equivalent board detection remains a map-generation/split-audit responsibility.

The readiness JSON must contain:

- `full_game_ready: true` and `teacher_qualified: true` only after their documented gates pass.
- `teacher_hashes`, exactly matching the prepared corpus.
- `split_manifest_hash`, using `learning.interface.digest` on the split manifest.
- `environment_receipt` and `teacher_receipt`, each an object with a readable `path` and its `sha256`.
- Prepared data must additionally contain a verified frozen prompt manifest supplied through `prepare --prompt-manifest`; it binds instructions, model/tokenizer context settings, sources and completed selection evidence.

These bind the run to external qualification evidence; they do not automatically establish that the evidence satisfies the research plan. Review and retain the actual gate reports. The fit check requires exactly 128 qualified examples. Smoke training requires 5,000–10,000 unique examples; main training requires 50,000. Synthetic fixtures are rejected by both stages. Preparation streams examples and training reads token records by file offset to avoid retaining the whole corpus in RAM.

```sh
uv run --project training --locked python scripts/sft.py prepare \
  --input data/train.jsonl --output data/prepared-main --split-manifest data/splits.json --prompt-manifest data/frozen-prompt.json
uv run --project training --locked python scripts/sft.py train \
  --prepared data/prepared-main --output runs/sft-seed17 --seed 17 --stage main --readiness data/readiness.json
uv run --project training --locked python scripts/sft.py train \
  --prepared data/prepared-main --output runs/sft-seed29 --seed 29 --stage main --readiness data/readiness.json
```

The last two commands require a CUDA BF16 GPU and fetch the pinned 4B weights. They have **not** been run. Start with the smaller fixture/fit and measured-memory checks from the research plan before main training. CUDA fit, throughput, 128-example overfit, and complete Polytopia evaluation remain outstanding. QLoRA, distributed training and automated checkpoint selection remain outstanding. Batched serving and bounded Modal inference/gameplay launchers are implemented. The optional locked `simulation` dependency group enables CPU JAX alongside GPU model inference. Checkpoint selection still uses Polytopia validation; no APEX data is loaded by this code.

Run receipts include configuration, seed, source/package versions, data/tokenizer hashes, hardware, timing, peak GPU memory, metrics, and final adapter hashes. Successful training saves optimizer/scheduler checkpoints and `adapter-final`; failure writes a failed receipt. Existing outputs are never overwritten automatically.

## Validation evidence

See [local plumbing evidence](benchmarks/2026-10-16-sft-plumbing.json). The scope is CPU interface/teacher tests, real response masking with the pinned Qwen tokenizer, and a tiny random-model LoRA update/reload. It is not a 4B training result, GPU benchmark, teacher win-rate measurement, or completed SFT checkpoint.
