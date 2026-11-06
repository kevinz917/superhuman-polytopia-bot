# Expert-SFT evaluation speed-up (2026-10-19)

Goal: run the expert-SFT evaluation harness (`scripts/benchmark_expert_sft.py`) faster without changing its decisions. Everything was measured on Modal H100s using real saved prompts from `expert-sft-v1`.

## Recommendation

| Setting | expert-sft-v1 | Recommended | Where |
|---|---|---|---|
| `serving.max_num_batched_tokens` | 8192 | **32768** | protocol `serving` (opt-in) |
| `serving.enable_prefix_caching` | false | **true** | protocol `serving` (opt-in) |
| `serving.gpu_memory_utilization` | 0.80 | **0.90** | protocol `serving` (opt-in) |
| Shards per arm | 1 | **ceil(games / 15)**, e.g. 4 for a 60-game arm | `experiments/modal_eval_shards.py` |

These settings leave unchanged:
- sampling;
- per-decision seeds;
- the xgrammar legal-action grammar;
- LoRA (`max_lora_rank=32`, `max_loras=1`);
- `max_num_seqs=64`;
- CUDA-graph capture (≤64, full decode coverage);
- `max_model_len=262144`;
- chunked prefill.

**Per H100:**
- Prompt throughput is **+10.7%** (39.2k → 43.4k prompt tokens/s).
- Mean request time is **−8.5%**.
- Median TTFT at 30 concurrent games drops from **7.0 s to 2.5 s**.

**Effect on decisions:** none beyond the noise that the old engine already has from one run to the next (see Equivalence).

**Wall-clock time:** almost all of the saving comes from sharding. Sharding uses the same GPU-hours plus about 1.5 min of engine load per shard.

**Projection for a 60-game arm** (this has not been run end to end):
- The 177 min measured today becomes about 160 min with the new engine settings.
- With 4 shards it becomes about 40–50 min.
- It cannot go below the time the longest game needs on its own: about 600–1,300 sequential decisions.

## Why the GPU-side gain is small

- **One H100 is compute-saturated on prefill well below 30 games.**
  - The smoke run already reached 36.8k prompt tokens/s with only 8 concurrent games.
  - At 30 and 60 games, throughput is the same (39.2k).
  - Every move sends about 11k prompt tokens for 8–10 output tokens.
- **The only engine-side levers are:**
  - computing fewer prompt tokens: prefix caching skips the shared system prompt, which is 9.9% of prompt tokens;
  - slightly more efficient larger prefill steps: about +1–2%.
- **Raising `max_num_batched_tokens` changes latency much more than throughput.**
  - Each prompt is prefilled in one step instead of two, so TTFT halves.
  - The 8–10 decode tokens then wait behind bigger prefill steps, so decode time rises from 1.5 s to 5.4 s.
  - **Request time, not TTFT, sets the game rate.** It improves only 8%.
- **No preemptions occurred in any config**, so KV cache capacity is not the limit at 60 games. `gpu_memory_utilization=0.90` adds only about +1–2%.
- **The LoRA adapter itself costs about 15%.** The same engine with no adapter attached runs 46.0k tokens/s.
  - The base arm already runs this way.
  - Merging the adapter into the weights would recover the 15% for SFT arms. It changes bf16 rounding, so it is not done here (see follow-ups).
- **Bigger wins must come from the prompt (fewer tokens per move) or from more GPUs (sharding).**

The vLLM 0.19.0 optimization guide (docs.vllm.ai/en/v0.19.0/configuration/optimization) agrees:
- With chunked prefill, which is on by default in V1, it recommends `max_num_batched_tokens > 8192` for throughput, especially for small models on large GPUs.
- A larger budget improves TTFT and a smaller one improves inter-token latency, which is exactly the trade-off measured here.
- The prefix-caching design doc says cached blocks are keyed by LoRA ID as well as by tokens, so adapters never share KV.
- The reproducibility doc says vLLM does not guarantee batch-independent results by default.

## Benchmark (H100, real prompts)

### Method

**Run:**
- run id `eval-speedup-v1`;
- `experiments/modal_eval_bench.py`;
- `scripts/eval_replay_bench.py`;
- one H100 per config, all started at the same time.

**Prompts:**
- 960 real decisions from the 30 `expert-sft-v1/expert_sft` games: 32 per game, evenly spaced over each game.
- In total 10.96M prompt tokens, median 9.9k and mean 11.4k per prompt.
- 8 further decisions are used as warm-up.

**Checks on every request** (all 960 passed in every config):
- It is re-tokenized through the chat template and must equal the saved `prompt_token_ids`.
- It uses the original seed (`seed + rng_index·100000 + decision`, checked against the trace).
- It uses the expert-SFT adapter, verified against the training receipt's hashes.
- It uses the same legal-action JSON grammar and sampling options.

**Load pattern:**
- Closed loop with no think time: each of C workers replays a contiguous, game-ordered slice of prompts.
- C = 30 means one worker per game; C = 60 means one worker per half game.
- The prefix cache is reset before each phase.
- `prompt tok/s` is all prompt tokens divided by phase wall time.
- `computed` excludes prefix-cache hits.

### Results

| config | C | prompt tok/s | computed tok/s | TTFT p50 | TTFT p90 | request p50 | request p90 | request mean | decode p50 |
|---|---|---|---|---|---|---|---|---|---|
| **v1** (8192, no PC) | 30 | 39,235 | 39,235 | 7.03 | 11.53 | 8.63 | 13.23 | 8.48 | 1.52 |
| v1b (identical copy) | 30 | 39,566 | 39,566 | 7.00 | 11.44 | 8.55 | 13.14 | 8.41 | 1.52 |
| pc (8192 + PC) | 30 | 41,827 | 37,680 | 6.47 | 10.89 | 8.09 | 12.70 | 7.93 | 1.58 |
| b16k | 30 | 39,816 | 39,816 | 5.53 | 9.64 | 8.43 | 13.08 | 8.40 | 3.01 |
| b16k-pc | 30 | 41,822 | 37,675 | 4.89 | 9.03 | 8.00 | 12.62 | 7.97 | 3.20 |
| b32k-pc | 30 | 42,569 | 38,355 | 2.56 | 5.35 | 8.04 | 12.46 | 7.91 | 5.45 |
| b64k-pc | 30 | 43,121 | 38,857 | 3.22 | 5.43 | 7.85 | 12.04 | 7.78 | 4.53 |
| **b32k-pc-gmu90** | 30 | **43,417** | 39,101 | **2.52** | **5.23** | 7.92 | 11.95 | **7.76** | 5.41 |
| v1-nolora (no adapter) | 30 | 46,043 | 46,043 | 5.97 | 9.83 | 7.34 | 11.43 | 7.20 | 1.34 |
| **v1** | 60 | 39,230 | 39,230 | 14.74 | 19.87 | 16.29 | 21.51 | 16.66 | 1.55 |
| v1b | 60 | 39,395 | 39,395 | 14.70 | 19.80 | 16.22 | 21.47 | 16.60 | 1.55 |
| pc | 60 | 41,778 | 37,667 | 13.63 | 18.69 | 15.19 | 20.43 | 15.62 | 1.62 |
| b16k | 60 | 39,768 | 39,768 | 12.95 | 17.91 | 16.06 | 21.18 | 16.51 | 3.09 |
| b16k-pc | 60 | 42,115 | 37,972 | 11.97 | 16.99 | 15.19 | 20.44 | 15.60 | 3.25 |
| b32k-pc | 60 | 42,811 | 38,604 | 8.92 | 13.26 | 14.92 | 20.09 | 15.52 | 6.40 |
| b64k-pc | 60 | 43,472 | 39,203 | 7.98 | 13.03 | 14.90 | 19.88 | 15.23 | 6.70 |
| **b32k-pc-gmu90** | 60 | **43,459** | 39,172 | 8.76 | 13.01 | 14.68 | 19.77 | **15.24** | 6.31 |
| v1-nolora | 60 | 45,879 | 45,879 | 12.58 | 16.95 | 13.94 | 18.45 | 14.21 | 1.37 |

All times are in seconds.

**Run-to-run noise:** v1 and v1b differ by 0.8% in throughput.

**vLLM counters:**
- Prefix-cache hits are 9.8–9.9% of prompt tokens in every PC config.
- There were 0 preemptions anywhere.

**Comparison with the real game run:**
- The replay of v1 reaches 39.2k tokens/s, against about 37–38k in the real game run. The real run has CPU gaps for environment steps and opponent turns.
- Replay TTFT at C = 30 is 7.0 s, against 5.3 s in-game, for the same reason: fewer requests are actually in flight during a real game.

**Choosing between the top configs:**
- b64k-pc and b32k-pc-gmu90 are tied within noise.
- 32k is recommended because it has the better TTFT tail at 30 games and keeps activation memory smaller.
- `gpu_memory_utilization=0.90` is safe here: JAX runs on the CPU (`JAX_PLATFORMS=cpu`), so vLLM is the only GPU user.

## Equivalence

**Question:** for the same prompts and seeds, are the answers identical between the old and new settings?

**Answer:** they are not bit-identical, but neither is the old engine against itself. The new settings change answers exactly as often as simply re-running the old engine.

**Pairwise disagreement between runs**, out of 960 decisions; tokens and actions agree or disagree together:

| pair | differing decisions |
|---|---|
| v1 vs v1b (same settings, same C = 30) | 28 (2.9%) |
| old vs old: v1/v1b × C30/C60, all pairs | mean 29, range 26–32 |
| old vs new: v1/v1b vs each new config/phase | mean 27.6, range 21–36 |
| new vs new | mean 26.3, range 17–36 |
| each replay vs the original expert-sft-v1 game output | 20–34 |

**Per-prompt stability:**
- Across all 17 executions (16 replays plus the original game), **884 of 960 prompts (92%) give the same answer every time**.
- The other 76 prompts are near-ties: two or three different actions appear, and the majority answer takes a median 76% of runs.
- These prompts are the long late-game ones: median 12.9k tokens, against 9.9k overall.

**Why they differ:**
- Floating-point reduction order in attention and GEMM kernels depends on the batch composition and on how chunked prefill splits a prompt.
- A tiny logit change can flip one sampled token at temperature 1.0 when two actions are almost equally likely.
- With a fixed seed the sampler's random draw is the same, so only those near-ties flip.
- Changing the batch budget or reusing cached prefix KV changes the kernel shapes. Changing concurrency, or which other games happen to share a step, does the same thing.

**Consequence for games:** a game can never be replayed bit-for-bit under any of these settings, including the v1 settings.
- In the shard smoke (below), the same 4 games run on the same engine — once as 2 shards of 2 games and once as 1 unsharded run of 4 — diverged into different games:
  - 2,089 against 3,144 model decisions;
  - different winners on 2 of the 4 games (map 100742, both seats).
- Evaluation results are therefore statistical either way. That is already how the reports treat them: one sampling sequence, paired bootstrap.
- The new settings add no new kind of variation.
- If bit-reproducible games are ever needed, vLLM's batch-invariant mode is the tool for it. It has its own speed cost and different numerics, and it was not tested here.

Outputs and receipts are on the volume:
- `eval-speedup-v1/summary.json`;
- `eval-speedup-v1/bench/<config>/{runtime.json,summary.json,c30.jsonl,c60.jsonl}`.

Each config's receipt records:
- `engine_args`;
- the CUDA-graph capture sizes, which are identical in every config;
- `num_gpu_blocks`;
- package versions: vLLM 0.19.0, torch 2.10.0, xgrammar 0.2.8.

## Sharding

**How it works:**
- `scripts/benchmark_expert_sft.py <root> <arm> --shard I/N` plays only `jobs[I::N]` of the arm, into `<arm>/shards/II-of-NN/`.
  - The strided split keeps map, seat and opponent mixes balanced across shards.
- Seeds depend only on `rng_index` and the decision index. Sharding therefore changes batch composition — the same kind of variation as concurrency — and no seed, prompt or setting.
- Each shard runs the normal per-arm checks before playing:
  - source, map and protocol hashes;
  - adapter file hashes;
  - preflight.

**Merging** (`scripts/benchmark_expert_sft.py merge <root> <arm> N`):
1. It refuses to merge unless every shard agrees on:
   - `engine_args`, `serving`, packages, model and revision;
   - tokenizer and chat-template hashes;
   - adapter path and adapter file hashes;
   - device, `max_num_seqs`, CUDA-graph ceiling and harness.
2. It refuses unless the shards cover the arm's jobs exactly once.
3. It moves traces, checkpoints and streams into the usual layout:
   - `<arm>/runtime.json`, which adds `sharding` plus every shard's preflight;
   - `<arm>/games/{games.json, jobs.json, report.json, progress.json, *.jsonl.gz, private/}`;
   - `<arm>/streams/`;
   - `<arm>/generation-timing.jsonl`;
   - `<arm>/report.json`, which adds `sharding`.
4. It checks that the number of stream files equals the number of timed decisions.

**Launcher:** `experiments/modal_eval_shards.py` provides `evaluate_sharded(run_id, protocol_sha, arm, shards)`.
- It is a drop-in replacement for the launchers' `evaluate`: it writes the same `<arm>-execution.json` and `<arm>/report.json`.
- A launcher adopts it with `app.include(experiments.modal_eval_shards.app)`.
- It can also be run by hand on any existing run directory:

```bash
modal run --detach --env staging -m experiments.modal_eval_shards --action arm --run-id <run> --arm <arm> --shards 4
```

**Smoke test** (`eval-shards-smoke-1792442961`):
- Setup: 4 held-out pressure games (maps 100741–100742, both seats) on the expert-SFT arm, run as 2 shards of 2 games and also unsharded.
- Both shards passed the adapter-hash and engine-equality checks.
- The merged arm has the same file set as the unsharded arm; the only extra key is `sharding` in `runtime.json` and `report.json`.
- Rows in `games.json` have identical keys.
- Stream-file count = timing rows = model decisions (2,089).
- The summarizers' `trace_metrics` reads both arms.
- Merge time is 23 s (moving files). An earlier copy-based version took about 7.5 min and was replaced.
- Shard wall times were 583 s and 643 s, against 1,464 s unsharded. The games themselves differed, as explained in Equivalence.

**Choosing the shard count:**
- One GPU stays saturated down to about 8 live games.
- Each game is a sequential chain of about 400–1,300 decisions, so very small shards spend their tail waiting on one long game.
- About 15 games per shard keeps the GPU busy for most of the run.
- Total GPU-hours stay about the same; add about 1.5 min of engine load per shard.

## How to opt in

**Engine settings:**
- Put the three keys in the protocol's `serving` block:

```json
"serving": {"enforce_eager": false, "max_cudagraph_capture_size": 64, "require_full_decode_graph_coverage": true,
            "log_stats": true, "max_num_batched_tokens": 32768, "enable_prefix_caching": true,
            "gpu_memory_utilization": 0.90}
```

- Without these keys, `engine_kwargs()` reproduces the expert-sft-v1 `engine_args`: key for key and in the same order. This was checked against `expert-sft-v1/expert_sft/runtime.json`.
- Existing launchers copy `serving` from the source protocol, so a new protocol (or the configurable pipeline) must set the keys explicitly.

**Sharding:** use `evaluate_sharded` (above).

**Caveat:** `scripts/benchmark_expert_sft.py` changed, so its sha256 changed.
- New protocols hash the file at launch and are unaffected.
- Re-launching an existing run (for example `expert-sft-heldout-v1`) from today's checkout would fail that run's own source-hash check. That is true of any edit to a hashed file.

## Cost

- Benchmark: 9 H100s × about 13 min.
- Shard smoke: 3 H100s × 10–25 min.
- Smoke runs: 2 H100s × about 3 min.
- Total: about 3 H100-hours.

## Follow-ups (not done)

- **Merge the LoRA into the base weights for SFT arms**, worth up to +17%.
  - It changes bf16 rounding, so it needs the same equivalence test.
  - It should then be compared against the 2.9% noise floor, not against bit-identity.
- **A compact prompt.** Prompt tokens per move are the throughput bottleneck. The prompt-v2 sibling owns this; prefix caching gains more as more of the prompt is shared.
- **A full 60-game sharded arm** to confirm the projected wall time.
