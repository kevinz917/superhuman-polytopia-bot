# Configurable expert-SFT pipeline (2026-10-19)

One JSON config now describes an expert-SFT run end to end: which RL bot to imitate, against which opponents, on which maps, how examples are selected, the prompt, the LoRA recipe and the evaluation. One Modal app runs it as five restartable stages. The next large run on the new RL bot is a config change (§6).

| | |
|---|---|
| Launcher / coordinator | [`experiments/modal_sft_pipeline.py`](../experiments/modal_sft_pipeline.py) (Modal app `polytopia-sft-pipeline`, env `staging`) |
| Config schema, prompt registry, players, recording, selection, estimate | [`src/polytopia_bot/learning/sft_pipeline.py`](../src/polytopia_bot/learning/sft_pipeline.py) |
| Selection + tokenization + corpus | [`scripts/sft_pipeline_prepare.py`](../scripts/sft_pipeline_prepare.py) |
| Evaluation arm (wraps `benchmark_expert_sft.py`) | [`scripts/sft_pipeline_eval.py`](../scripts/sft_pipeline_eval.py) |
| Evaluation summary | [`scripts/sft_pipeline_summarize.py`](../scripts/sft_pipeline_summarize.py) |
| Configs | [`configs/sft-pipeline/`](../configs/sft-pipeline/): `smoke.json` (end to end), `next-expert-160k.json` (template for the large run), `league96-v1-repro.json` (v1 as a config), `eval-only-smoke.json` (evaluate existing adapters), `restart-test.json` (CPU-only restart test) |

The pipeline reuses, unchanged, the v1 pieces: `expert_trainer.train_expert` (4 × B200 packed LoRA trainer, same corpus gate), `data.tokenize_example`, `expert_data.read_game`/`FAMILIES`, `rl.policy.StructuredPolicy`, `teacher.ScriptedTeacher`, and the evaluation harness `benchmark_expert_sft.gameplay`. No existing file was edited.

## 1. Running it

```bash
# Smoke: launches, then follows status until the run ends (prints FINAL status)
modal run --detach --env staging -m experiments.modal_sft_pipeline --action smoke --config configs/sft-pipeline/smoke.json --run-id sftp-smoke-N

# Real run: launches the coordinator and returns
modal run --detach --env staging -m experiments.modal_sft_pipeline --config configs/sft-pipeline/NAME.json --run-id RUN

# Status (reads RUN/status.json from the volume; triggers no Modal function)
modal run --env staging -m experiments.modal_sft_pipeline --action status --run-id RUN

# Resume after anything stopped it (skips verified stages, reattaches to live calls)
modal run --detach --env staging -m experiments.modal_sft_pipeline --action resume --run-id RUN [--retry STAGE]

# Cost/time estimate (local, no Modal call)
python -m experiments.modal_sft_pipeline estimate configs/sft-pipeline/NAME.json
```

Use the Modal SDK python (`~/.local/share/uv/tools/modal/bin/python`) for the estimate. Always launch with `--detach`. In detached mode Modal keeps only the last triggered function alive after the local client disconnects, which is the coordinator. For that reason `status` and the smoke follower read the volume and never call a function.

## 2. Config fields

Unknown fields are rejected. Defaults are in `sft_pipeline.DEFAULTS`.

### Top level
- `schema`: `"sft-pipeline-config-v1"`.
- `name`, `description`: free text, recorded in the protocol.
- `stages`: the subset of `generate, prepare, train, evaluate, summarize` to run. For example, an eval-only run of existing adapters uses `["evaluate", "summarize"]`.

### `players`
A name → spec registry shared by generation and evaluation.
- RL checkpoint: `{"kind": "rl", "path": "/retained/…npz", "sha256": "…", "deterministic": true}`.
  - The path is on volume `polytopia-structured-rl-retained` and the hash is checked once per container.
  - A non-deterministic RL player is reseeded per game from the job id, so games stay reproducible. This is only allowed in generation, because the benchmark plays RL opponents deterministically.
- Scripted: `balanced`, `pressure` and `economy` resolve to `ScriptedTeacher(style)` without a definition. Aliases use `{"kind": "scripted", "style": "pressure"}`.

### `prompt`
- `renderer`: a name in `sft_pipeline.RENDERERS`. Today that is `readable-v1`, the v1 prompt and grammar.
  - A renderer entry has four parts: `render(obs)` (returns messages + catalog), `schema(messages)` (the evaluation grammar), `response(label)` (the target string) and `harness` (its name).
  - Registered: `readable-v1` and `compact-v2`.
    - `compact-v2` is the prompt-v2 session's `compact_prompt.render_compact_decision` + `action_schema`, with the same `{"action_id"}` target.
    - It is registered but not yet exercised by this pipeline's smoke.
  - Another prompt plugs in with `register_renderer(...)`, or a new dict entry in `RENDERERS`, and `"renderer": "<name>"`.
  - The same renderer is used for labels at generation time and, through `sft_pipeline_eval.py`, at evaluation time.
  - The game files record their renderer, and prepare refuses to mix renderers.

### `generation`
- `experts`: RL player names to imitate. Several experts produce the cross product of jobs.
- `map_manifest`: the split manifest. Default `runs/sft0/splits-v1/maps.json`.
- `map_pool`: an ordered list of sources.
  - `{"split": "train", "offset": 0, "count": 500}`.
  - `{"fresh_seeds": {"start": 2000000, "count": 1500}}`: maps beyond the 500 training maps.
    - Fresh seeds must be above every manifest seed.
    - Each worker computes the map's board and symmetry hash. A map whose symmetry hash equals any non-train split map (validation, test, prompt or teacher-dev, 740 maps) is recorded as `excluded` and never played.
- `seats`: default `[1, 2]`.
- `opponents`: a list of `{"name", "maps"?, "map_offset"?, "seats"?}`. Each opponent plays `pool[map_offset : map_offset+maps]` × seats. This lets different opponents cover different maps.
- `games_per_call` (default 8): games per worker call. One call is the unit of call-ID persistence and reattachment.
- `max_failed_games`: a fraction (<1) or a count. If more games still have no result after `max_rounds` (default 3) retry rounds, the stage fails.
- `record_trajectory` (default true): every game file also stores the full action-id sequence, so it can be replayed and re-rendered with a future prompt without the RL bots.
- `reuse_run`: a run id whose verified games are used instead of generating. It must have the same renderer. Use it to re-prepare one game set with different selection rules or recipes.

### `data` (selection, applied per game, so it does not depend on sharding)
Rules run in this order:
1. `outcomes`: keep games whose expert result is in `win, loss, draw, truncated`. Default: all.
2. `skip_forced` (default true): drop decisions where every legal action is `end_turn`/`resign`. The recorder stores a `forced` flag and the legal-family counts per decision.
3. `max_move_fraction` (default null): cap plain `move` examples per game at this fraction of the kept examples, choosing which moves to keep at random with a fixed seed.
4. `examples_per_game`: an integer (v1 used 40) or `"all"`.
5. `upweight`: `{family: weight}` with weight in [0, 10].
   - A weight of 2 writes each such example twice. A weight of 1.5 writes it once, plus once more with probability 0.5. A weight of 0 drops the family.
   - Families: `move attack ability recruit harvest build research capture city_reward end_turn resign`.

Corpus-level settings:
- `sample_seed` (17): the seed for every random choice, keyed by `seed:job_id`.
- Identical states (same system + user text) keep their first label. Duplicates and label conflicts are counted.
- `max_examples`: a budget on the final row count, with upweight copies included. Unique states enter in a fixed hash order, so the subset doesn't depend on the order in which shards finished. This also makes data-scaling subsets cheap: re-prepare one `reuse_run` with several budgets.
- `min_examples`: prepare fails below this.
- `games_per_shard` (25): games per tokenization container.
- Examples longer than `training.overrides.max_length` are skipped and counted, never truncated (same as v1).

### `training`
- `base`: a protocol whose `training.config` and qualified runtime `benchmark` are the starting point. Default `runs/movement-scale40k-v1/protocol.json`, the v1 recipe:
  - LoRA r32/α64 on all projections;
  - learning rate 1e-4 with cosine decay and 3% warmup;
  - effective batch 64, response-only loss, packed cast-free segmented SDPA kernel.
- `overrides`: any of `lora_rank lora_alpha lora_dropout target_modules learning_rate warmup_ratio epochs max_length max_new_tokens effective_batch_size max_grad_norm microbatch_size`.
  - Integer epochs only (a trainer rule).
  - `max_length` above 24,576 warns, because only 24,576 passed the 4 × B200 memory stress.
- `gpu_type` (B200), `gpus` (1/2/4/8): applied with `Function.with_options` and checked against the GPUs the container actually has.
  - The packed-kernel benchmark was qualified on 4 × B200. Other counts keep the effective batch but are not separately qualified.
- `seed`, `checkpoint_every`.
- `timeout_hours` (≤ 24): Modal's per-call limit. A lost train call resumes from its newest checkpoint on `--retry train` (§4), so `checkpoint_every` bounds the work lost (157 steps ≈ 30 min at 11.3 s/step).

### `evaluation`
- `harness_source`: the protocol to copy the harness from (model, revision, sampling, serving, concurrency, context). Default `runs/expert-sft-v1/protocol.json`, so new results are directly comparable with v1 and the held-out runs.
- `gpu`: default H100.
- `maps`: `{"split", "offset", "count"}` from the manifest. It may not be `train`, and overlap with the generation pool (seed or symmetry hash) is refused.
- `seats`, `opponents`: player names. RL opponents must be deterministic.
- `arms`: `{name: {"adapter": …}}`, where the adapter is one of:
  - `null`: the base model;
  - `"trained"`: this run's adapter;
  - `{"path", "files"}`: hashes given;
  - `{"path", "training_receipt"}`: hashes read from a completed `run.json`.
- `shard_by_opponent` (default true): one H100 container per arm × opponent (`name@opponent`) for parallelism. The summary merges shards back by arm.
- `rl_reference`: RL players that play the same evaluation jobs on CPU through the same public interface, reported as `rl:<name>`.
- `pairs`: `[[a, b], …]` for map-clustered paired bootstraps (20,000 resamples, as in v1). An entry is an arm name or `rl:<name>`.
- `start_independent_arms_early` (default true): arms that don't need this run's adapter start when the coordinator starts, overlapping generation and training.
- `serving` (default `{}`, i.e. the v1 engine): opt-in engine settings for the `benchmark_expert_sft.py` speedups (`max_num_batched_tokens`, `enable_prefix_caching`, `gpu_memory_utilization`; see `docs/sft-eval-speedup-2026-10-19.md`). The 160k template uses the recommended 32768 / true / 0.90.

## 3. Stages and outputs

Everything is under `/workspace/runs/RUN/` on volume `polytopia-sft0-training`. Local copies of the protocol files are in `runs/RUN/`.

| Stage | Work | Output (verified → skipped on resume) | Throughput logged |
|---|---|---|---|
| launch (local) | validate config, build jobs, check overlaps, hash sources | `protocol.json`, `eval/protocol.json`, `source/source.tar.gz` | estimate |
| generate | `generate_chunk` CPU calls, ≤200 containers; per game: `generate/games/<id>.jsonl.gz` + `generate/results/<id>.json` (failures in `generate/failures/`) | `generate/manifest.json` with the generation fingerprint | games/min, expert decisions/min, container-hours, $ |
| prepare | `prepare_shard` (select + tokenize, ≤100 containers) → `prepare/shards/shard-NNNN.{jsonl,json}`; `combine_corpus` → `prepare/corpus/` (written to a temporary directory, then renamed) | `prepare/corpus/manifest.json` with the prepare fingerprint (schema `polytopia-expert-prepared-v1`, accepted by the v1 trainer) | examples/min, tokenize tokens/s, combine seconds |
| train | `train` on `gpu_type:gpus` → `training/model/` (`run.json`, `adapter-final/`, checkpoints), `training/execution.json` | `run.json` status `completed` + `execution.json` | s/step, tokens/s, examples/s, GPU-hours, $ |
| evaluate | one `evaluate` H100 call per arm (`eval/<arm>/`), `reference_chunk` CPU calls (`eval/reference-<player>/`) | `eval/<arm>/report.json`, one reference file per job | per arm: prompt tokens/s, decisions/min, load seconds |
| summarize | `summarize_eval` | `eval/summary.json`, `eval/summary-small.json` | — |

**Status.**
- `status.json` holds:
  - the overall state and current stage;
  - for each stage: state, first-start/finish times, wall seconds, live progress (games done, shards done, training step/loss/ETA, games finished per arm), call counts, summary and throughput;
  - a list of every coordinator life, flagged when it was a platform restart.
- It is rewritten at every stage transition and at most once a minute in between.
- `calls.json` holds every spawned call ID.

**Integrity.**
- The protocol records the SHA-256 of every file in `src/`, `scripts/` and `configs/` (262 files today), the config path and hash, the launcher hash, and each RL checkpoint hash.
- At launch the same three directories are snapshotted into `source/source.tar.gz`. Every worker, including the trainer and the benchmark subprocess, first overlays that snapshot onto `/workspace` and re-checks every file hash.
- So a resumed run executes exactly the launch-time code, even if this shared checkout has changed. Only the orchestration code in `experiments/` comes from the current checkout, and each coordinator life records its hash.
- Fingerprints chain generate → prepare → train. A stage output with a different fingerprint is never reused.

## 4. Restart behaviour

- **Persisted call IDs.**
  - The coordinator writes each call ID to `calls.json` (and commits the volume) right after spawning, before it waits on anything.
  - Fan-out stages (generate, prepare shards, RL references) store their round plan: which games or shards are in which call.
  - A restarted coordinator resumes the last recorded round. It polls the recorded calls (`FunctionCall.get(timeout=0)`: still running → keep waiting, finished → use the result, failed or expired → terminal) and spawns only items that were never spawned.
  - Only after every call of a round is terminal does it plan a new round from what is still missing on the volume. It never spawns a second copy of a worker that may still be running.
- **One coordinator per run.**
  - `status.json.coordinator_call_id` names the owner.
  - A new coordinator refuses to start while that call is still running.
  - A Modal platform restart re-runs the same call ID, which is recognized and recorded as `platform_restart`.
- **Idempotent workers.** This also guards the window between a spawn and the `calls.json` write.
  - A game counts as done when its result file exists, and game files are written under process-unique temporary names and renamed.
  - A shard counts as done when its marker has the right fingerprint and file size.
  - The corpus directory appears only by atomic rename.
- **Non-idempotent workers** (train, evaluation arms) are never retried automatically:
  - The trainer refuses an existing `training/`, and the benchmark refuses an existing arm directory.
  - If such a call failed or was lost, the stage fails with a message, and status lists what to inspect.
  - `--action resume --retry train|evaluate|prepare|generate` first checks that neither the coordinator nor any of that stage's calls is running. It then archives the dead call records (`calls.json.archived`) and moves partial outputs aside (`training.failed-<stamp>/`, `eval/<arm>.failed-<stamp>/`), so the resumed coordinator redoes only that part.
- **Training resumes from its newest checkpoint.** A lost train call no longer restarts from step 0 (about 8 h for 160k examples on 4 × B200).
  - **What each `checkpoint-N/` holds:** the adapter and tokenizer as before, plus `trainer_state.pt` and `trainer_state.json`.
    - `trainer_state.pt`: the AdamW state, the LR-scheduler state, every rank's RNG states (Python, NumPy, torch CPU and CUDA), and the position: the epoch and in-epoch offset of the next batch. The sampler order is a pure function of seed and epoch, so the offset is enough.
    - `trainer_state.json`: that position, the SHA-256 of `trainer_state.pt`, the adapter file hashes, the hash of the adapter tensors, and the run identity: config hash, seed, world size, runtime-variant hash, and the corpus `tokens_sha256`, `config_hash` and record count.
    - The `checkpoint-N.ready.json` marker is written only after both files. The train worker commits the volume as soon as a new marker appears, not only once a minute, so a checkpoint survives a call that dies right after it.
    - Cost: `trainer_state.pt` is 2× the adapter (529 MB next to a 264 MB adapter for Qwen3-4B, LoRA r32), so a checkpoint is about 0.8 GB. At `checkpoint_every: 157` a 2,500-step run keeps about 13 GB.
    - Everything else is unchanged without `resume_from`: the `run.json`, `.ready.json`, log and checkpoint-entry keys match the pre-change `sftp-smoke-3` receipt, `input.json` has no `resume_from`, and the steps are the same.
  - **`train_expert(..., resume_from=<checkpoint dir>)`** verifies before it trains:
    - the identity matches the current corpus, config, seed, runtime and world size;
    - the adapter files match the hashes in `trainer_state.json`, the `.ready.json` marker and the old `run.json`;
    - `trainer_state.pt` matches its hash;
    - the loaded adapter tensors match the checkpointed tensor hash;
    - the frozen base matches the old run's `frozen_before`;
    - the restored scheduler is at step N, and the epoch's sample order matches the old run's `epoch_order_hashes`.
  - **Then it continues** at step N+1 with the restored optimizer, scheduler and RNG states, skipping the batches already trained.
    - It writes a new output directory. Its `run.json` starts from the old receipt's logs and checkpoints up to step N and appends from there. `elapsed_training_seconds` continues from the checkpoint.
    - A `resumes` entry records the checkpoint path, the step, and the previous status, error and corpus stress. The long-input stress test runs again on the new container.
  - **The pipeline:** `--action resume --retry train` moves the dead attempt to `training.failed-<stamp>/` as before.
    - It then looks for the newest checkpoint with trainer state across every archived attempt of the run whose `input.json` matches the current one (corpus, config, seed, benchmark). It prints that checkpoint as `"resume"` in the `RETRY` line and records it in `calls.json.archived`.
    - The new train call passes it as `resume_from`. `training/input.json` and `execution.json.resumed_from` record it, and the stage summary shows `resumed_at_steps`.
    - A second failure after the resume is handled the same way. The newest checkpoint wins, whether it comes from the first attempt or the resumed one.
    - Checkpoints written by a trainer without resume support (adapter only) are ignored, so such an attempt still restarts from step 0. Runs launched before this change also keep their launch-time trainer.
    - If a checkpoint fails verification, the train call fails with the reason. To start over instead, move the `training.failed-*` directories out of the run.
  - **Throughput caveat:** after a resume, `execution.json.occupied_seconds` and the stage's `gpu_hours` and `$` count only the last call, not the GPU time lost in the dead attempt.
  - **Tested on Modal** with `configs/sft-pipeline/resume-test.json`: the `smoke.json` data (games reused from `sftp-smoke-3`), `max_examples` 320, which is 5 updates on 4 × B200, and `checkpoint_every` 2. Both runs prepared the same corpus (same `tokens_sha256`).
    - **`sftp-resume-b`:** uninterrupted, as the reference.
    - **`sftp-resume-a`:** the train call was cancelled with `terminate_containers=True` 3 s after `checkpoint-2.ready.json` was committed.
      - The container still logged steps 3–4 and died while writing checkpoint-4. That checkpoint has `trainer_state.pt` but no `trainer_state.json` and no `.ready` marker.
      - `--action resume --retry train` archived the attempt and picked `checkpoint-2`, ignoring the incomplete checkpoint-4.
      - The new call resumed at step 3 (epoch 0, offset 128) and completed.
      - In its `run.json`, logs 1–2 and the checkpoint-2 entry are the dead attempt's verbatim, the attempt's steps 3–4 were dropped, and `resumes: [{step: 2, previous_logged_steps: 4}]` is recorded.

    | step | loss B (uninterrupted) | loss A, attempt 1 (no resume) | loss A, resumed at 2 | \|A1−B\|/B | \|A resumed−B\|/B |
    |---|---|---|---|---|---|
    | 1 | 1.81747544 | 1.81747544 | (attempt 1) | 0 | 0 |
    | 2 | 1.45916367 | 1.45916367 | (attempt 1) | 0 | 0 |
    | 3 | 0.78319979 | 0.78029847 | **0.78029847** | 3.7e-3 | 3.7e-3 |
    | 4 | 0.46126530 | 0.45303512 | 0.45548934 | 1.8e-2 | 1.3e-2 |
    | 5 | 0.35557735 | — | 0.35719919 | — | 4.6e-3 |

    - **Reading:**
      - The resumed step 3 loss is **bit-identical** to the dead attempt's own step 3. It is the forward pass of the step-2 adapter on the batch at offset 128, so the adapter and sampler position were restored exactly.
      - From step 3 on, the runs diverge only by the runtime's own nondeterminism. Grad norms already differ at step 1 (15.8542 vs 15.8508) between two uninterrupted runs. The resumed-vs-reference gap (≤ 1.3e-2) is within the gap between two uninterrupted runs (1.8e-2 at step 4).
      - A missing optimizer state would show up at step 4: a fresh AdamW's first step has a different magnitude. The learning rates match at every step (1e-4 → 0).
      - Adapters are therefore not byte-identical between any two runs, resumed or not. The frozen base was unchanged in both.
- **Tested on Modal** with `configs/sft-pipeline/restart-test.json` (6 games, then prepare), driven by a script that cancels the coordinator and relaunches `--action resume` twice:
  - **`sftp-restart-1`** (1 game per call): the coordinator was cancelled after the first results. The resumed coordinator reattached to the recorded round. `calls.json` still holds exactly one generate round with the original 6 call IDs, and each of the 6 games was played once.
  - **`sftp-restart-2`** (3 games per call): the coordinator was cancelled right after both chunk calls were recorded. At resume both chunk calls were still running with 0 results, and the new coordinator waited on those same 2 calls without spawning anything. The result was the same single round, 6 games, and prepare done.
  - **Concurrent resume:** in both tests a second `resume` started while the first resumed coordinator was running. It was refused (`coordinator fc-… is still running this run; not starting a second one`) and left no trace in `status.json`.
  - **Reproducibility:** both runs produced identical data (2,285 expert decisions, 50 forced). Generation is deterministic for deterministic players.
  - **Cancellation:** cancelling the coordinator does not cancel the workers it spawned.
  - **A real platform restart** happened during `sftp-smoke-3`: Modal preempted the coordinator at 14:37 PDT while the train call was queued for a B200, and restarted it with the same call ID. The restarted copy recorded `platform_restart: true`, re-verified generate and prepare from their manifests, and reattached to the same queued train call. `calls.json` still has a single `train` entry. This is the scenario that produced duplicate workers in the v1 held-out run.
- **GPU capacity waits.** A spawned train call waits in Modal's queue when no B200 is free (seen during the smoke). The coordinator just keeps waiting on the recorded call, and `status.json` shows `train.progress.phase = waiting_for_gpu` with the minutes waited. A coordinator restart during the wait reattaches to the queued call.
- **Shared volume backpressure.** The shared volume sometimes answers commits with `ResourceExhaustedError: too many layers` (seen at the first smoke upload).
  - Every commit, reload and upload retries with backoff for up to about 20 minutes.
  - The coordinator commits at most once a minute between transitions, and subprocess workers once a minute.

## 5. Scale: cost and time for ~160k examples

`python -m experiments.modal_sft_pipeline estimate configs/sft-pipeline/next-expert-160k.json` uses the expert-sft-v1 throughput and Modal list prices (B200 $6.25/h, CPU core $0.047/h, memory $0.008/GiB-h).

**Setup:** 2,000 maps × 2 seats = 5,000 games (500 train-split maps + 1,500 fresh seeds), against `balanced`/`pressure`/`economy` and the learned `league96`, with 32 non-forced decisions per game.

| Stage | Estimate | Basis |
|---|---|---|
| generate 5,000 games | ~69 container-hours, **≈ $11**, **≈ 25 min** wall at 200 containers | v1: 1,000 games in 8.3 min on ≤100 × 2-CPU containers (~50 s/game). Games against an RL opponent run two policies and are somewhat slower |
| prepare 160k examples | ≈ 4 container-hours, **≈ $1.5**, **≈ 15–25 min** wall (200 shards of about 1.5 min each + about 10–12 min combine of a ~30 GB token file) | smoke: 18 examples/s per 4-CPU shard container (10/s assumed); v1: 39.6k examples in 5.5 min end to end |
| train 160k × 9.3k tokens ≈ 1.49B tokens, 2,500 steps | **≈ 8.0 h on 4 × B200 ≈ 32 B200-hours ≈ $200** | v1: 369.7M tokens in 6,997 s, about 52.8k tokens/s and 11.3 s/step |
| **total before evaluation** | **≈ $215, ≈ 9 h wall** (plus any wait for B200 capacity) | |
| evaluation | about 1.5–3 H100-hours per 40-game arm shard for an SFT model ($6–12 each) | v1 development arm: 95 min for 30 games |

**Notes:**
- **Training dominates**, and its cost is linear in prompt tokens. The compact prompt work (prompt v2) is the main lever. Generation and prepare together cost under $15. At a quarter of today's 9.3k-token mean, the same 160k examples would train in about 2 h for about $50.
- **8 × B200** would bring training to about 4 h at the same cost if scaling is linear, but that is not qualified. 4 × B200 at ~8 h fits in one 23.5 h call with room to spare.
- **All decisions** (`examples_per_game: "all"`) gives about 335 examples per game after skipping forced moves, so 160k needs only about 480 games, but consecutive decisions from one game are highly correlated. The template prefers more maps and opponents with 32 per game, which costs the same and has more diverse states.

## 6. The next large run with the new RL bot

1. **Get the checkpoint ready:**
   - copy it to `polytopia-structured-rl-retained` (under `/retained/checkpoints/...`, next to its `.json` config);
   - compute its sha256 on Modal, not locally.
2. **Copy the config:** `cp configs/sft-pipeline/next-expert-160k.json configs/sft-pipeline/<bot>-160k.json`, then:
   - add the bot under `players`: `{"kind": "rl", "path": …, "sha256": …, "deterministic": true}`;
   - set `generation.experts: ["<bot>"]` and `evaluation.rl_reference: ["<bot>"]`;
   - optionally add `u3142` as a generation opponent (learned-opponent data) and as an evaluation opponent;
   - adjust `pairs`.
3. **Estimate:** `python -m experiments.modal_sft_pipeline estimate configs/sft-pipeline/<bot>-160k.json`.
4. **Smoke the new config:** copy `smoke.json` with the new bot's player entry to check the checkpoint loads and plays. It costs about $5 and about 25 min.
5. **Launch:** `modal run --detach --env staging -m experiments.modal_sft_pipeline --config configs/sft-pipeline/<bot>-160k.json --run-id <bot>-sft-160k-v1`.
6. **Follow:** use `--action status`. If anything stops the run, use `--action resume`, adding `--retry <stage>` only after reading the failure in `status.json`.

## 7. Smoke result: `sftp-smoke-3` passed every stage

Run with `configs/sft-pipeline/smoke.json`.

| Stage | Result | Throughput |
|---|---|---|
| generate | 2 games: league96 vs `balanced` on train map 100041 (won, round 37), and league96 vs the **learned u3142 bot** on fresh-seed map 2000000 (lost). 936 expert decisions, 32 forced. The fresh map passed the held-out check | 51.5 container-s per game |
| prepare | all decisions → 32 forced skipped → 904 sampled → 1,052 rows after upweighting attack ×2 and capture ×1.5 → `max_examples` budget of **128 rows** (111 unique states). Mean 9,533 tokens, max 18,995. Move cap 0.45 didn't bind | 18 examples/s per 4-CPU shard container, 91k tokens/s |
| train | 4 × B200, **2 updates**, loss 1.84 → 1.46. Longest-input stress passed at 118 GB peak. Frozen base unchanged, adapter saved | 12.5 s/step, 48.8k tokens/s (v1: 11.3 s/step, 52.8k) |
| evaluate | trained adapter on 2 validation games (vs `balanced` and vs u3142), plus the league96 RL reference | 8.9k prompt tokens/s, 107 decisions/min, 82 s model load |
| summarize | sft 0/2, league96 1/2, paired difference computed | — |

**Incidents during the smoke:**
- **B200 wait:** the train call waited about 55 minutes in Modal's queue for B200 capacity.
- **Platform restart:** Modal preempted and restarted the coordinator during that wait. It reattached correctly (§4).

**Earlier attempts:**
- **`sftp-smoke-1`:** the upload hit "too many layers", so the retry/backoff was added.
- **`sftp-smoke-2`:** the shard directory wasn't created. It failed cleanly with an actionable `StageFailed` message, and a new snapshot fixed it.

**Extra checks:**
- **`sftp-evalonly-1`** (`configs/sft-pipeline/eval-only-smoke.json`) ran an existing adapter, expert-sft-v1, with hashes taken from its training receipt. It used two opponent-sharded H100 arms started early, plus the reference and summary. `sft_v1@balanced`: 20k prompt tokens/s.
- **`sftp-restart-1`/`-2`** are the restart tests in §4.
- **Final regression on the last code:**
  - `sftp-restart-3` (generate + prepare) completed.
  - `sftp-evalonly-2` ran with `evaluation.serving` set to the eval-speedup recommendation. The engine receipt confirms 32768 batched tokens, prefix caching and 0.90 memory. `sft_v1@balanced` reached 24k prompt tokens/s, against 20k without.

**Cost of the smoke:**
- about 0.14 B200-hours of training (about $0.9);
- about 0.1 H100-hours;
- negligible CPU.

## 8. Known limits and follow-ups

- `benchmark_expert_sft.py` (owned by the eval-speedup session) hard-codes the readable renderer and grammar.
  - `sft_pipeline_eval.py` rebinds `render_readable_decision` and `action_schema` for other renderers.
  - Cleaner long-term: the benchmark reads `p['prompt_renderer']` from its protocol.
  - The wrapper relies on `gameplay(root, arm, preflight_only=…)` keeping its signature.
- Labels are rendered at generation time. A new prompt needs new games (cheap: about $11 for 5,000) or a replay tool over the recorded `actions` trajectories.
- Fresh-seed maps can repeat each other (rare, about 1%). They are deduplicated only against held-out maps, not against each other.
