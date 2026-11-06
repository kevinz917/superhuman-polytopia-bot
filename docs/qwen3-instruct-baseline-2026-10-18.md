# Qwen3-4B-Instruct-2507 baseline on the Qwen3.5 thinking games (2026-10-18)

Third baseline for `qwen35-thinking-30paired-v1`. The repository's original model is **`Qwen/Qwen3-4B-Instruct-2507`** at revision `cdbee75f17c01a7cc42f958dc650907174af0554` (from `configs/sft.json`). It ran unadapted, with no LoRA, on exactly the same 30 games. Everything ran on Modal (env `staging`, volume `polytopia-sft0-training`).

**Bottom line:** all 30 games were native losses (30 eliminations). There were 0 wins, 0 forfeits and 0 operational failures. The finished Qwen3.5 thinking-off arm also lost 30/30. Neither arm won a game, so **nothing can be concluded about which model is better**. The map-clustered win difference is exactly 0 on all 15 maps.

## Run

| | |
|---|---|
| Run ID | `qwen3-instruct-2507-30paired-v2` (cloud dir of the same name; receipts in `runs/qwen3-instruct-2507-30paired-v2/`) |
| Modal app | `ap-W0MDg7AckqQIYRJWzqMI8N` (worker, coordinator and analysis, all finished) |
| Protocol SHA256 | `5bdcebb9611410d61be197a8282bcfb5db0138ef0ce109349ea910b2ea986fb1` |
| Source protocol | `qwen35-thinking-30paired-v1/protocol.json`, SHA256 `4a056d63…d2a5e7` (checked) |
| Code (new files only) | `experiments/modal_qwen3_instruct_baseline.py`, `scripts/benchmark_qwen3_instruct.py`, `scripts/summarize_qwen3_instruct_baseline.py` |
| Earlier attempt | `-v1` (`ap-ozwBVUjo1HUKroN4P7pFoE`) stopped in my preflight gate before any game: the preflight job dict lacked `rng_index`. No game decision was made. The fix went into a fresh `-v2` protocol with identical jobs. See `runs/qwen3-instruct-2507-30paired-v1/attempt-failed.json`. |

**Same as the Qwen3.5 run:**
- the 30 jobs: 15 validation maps (seeds 100691–100705) × seats 1 and 2, with the same job IDs, board hashes and `rng_index`. On Modal, every board hash was regenerated and checked before play;
- the `balanced` scripted opponent;
- per-decision seeds `17 + rng_index*100000 + decision`, checked on every one of the 2,803 calls;
- the frozen round-100 / 10,000-action horizon;
- the named-field prompt (`thinking_readable.render_readable_decision`) and the legal-action JSON grammar (`action_schema`, xgrammar);
- no `max_tokens` cap (explicit `None`) and the same exact-repetition guard;
- the vLLM 0.19.0 / Transformers 4.57.6 / torch 2.10.0 image from `experiments/modal_thinking_repaired.py`, plus xgrammar 0.2.8;
- the H100 GPU;
- all 70 source hashes from the source protocol, re-verified on Modal.

The only change to the job list is the `candidate` label. It is not used in gameplay.

**Sampling** is copied from the Qwen3.5 run so that only the model changes: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, repetition penalty 1.0. *The Instruct-2507 model card recommends different settings: temperature 0.7, top-p 0.8, top-k 20, min-p 0 (presence penalty 0–2 optional).* This run does not use them.

**Token budget:** no `max_tokens` cap. Under the grammar, every answer was 8–9 tokens (`{"action_id":"aN"}` plus EOS). Mean 8.28, max 9. Every call finished with `stop`.

### Runtime receipt (what changed for Qwen3)
The full receipt is in `runs/qwen3-instruct-2507-30paired-v2/runtime.json`.
- **No `enable_thinking` kwarg and no Qwen3 reasoning parser.** The Qwen3.5 off arm used `reasoning_parser=qwen3, enable_in_reasoning=False` with `enable_thinking=False`. As a result, its prompt ended in an empty closed `<think></think>` block, and vLLM applied the grammar from the first generated token; its streams show 8-token answers. Instruct-2507 never emits `<think>`. With the qwen3 parser, the grammar would wait for a `</think>` that never comes. Here the structured-outputs config is `backend='xgrammar'` with an empty reasoning parser, so the grammar constrains the whole response from token 0.
- **Dropped `gdn_prefill_backend='triton'`.** Qwen3 is a dense transformer with no Gated DeltaNet layers.
- **Dropped `limit_mm_per_prompt`.** The model is text-only.
- **Kept everything else:** native context 262,144 (`max_model_len` confirmed by vLLM), compiled engine (no `enforce_eager`), CUDA graph capture sizes 1…64 covering scheduler capacity 64 (full-coverage check passed), chunked prefill, no prefix caching, `gpu_memory_utilization` 0.80.
- **One H100 replica** ran all 30 games concurrently. The Qwen3.5 arms used two replicas per mode with 14–16 games each.
- **Preflight gate:** 4 samples on the opening position of job 0, with seeds 990000–990003 that are never used in games. All 4 were valid 8–9-token answers. The rendered prompt ends in `<|im_start|>assistant\n` with no think markers.

## Results

| Metric | Qwen3-Instruct-2507 | Qwen3.5-4B thinking off |
|---|---:|---:|
| Status counts | native 30 | native 30 |
| Native wins / losses / draws | 0 / 30 / 0 | 0 / 30 / 0 |
| How games ended | elimination 30 | elimination 22, **resignation 8** (model chose the legal resign action) |
| Invalid-action forfeits | 0 | 0 |
| Operational failures (unfinished/context/infra/abort) | 0 | 0 |
| Valid moves (= model calls) | 2,803 | 3,795 |
| Output tokens (total / mean per call) | 23,220 / 8.28 | 31,285 / 8.24 |
| Median request latency | 1.79 s (30 games on 1 H100) | 1.26 s (14–16 games per H100) |
| Median final round | 22 | 22.5 |
| Gameplay wall time | 598 s | 471 s (slower of 2 shards) |
| Summed GPU-worker seconds | 662 s (1 worker, includes 53 s model load and preflight) | 1,230 s (2 workers) |

Latency is not comparable in isolation. The two arms ran different numbers of concurrent games per GPU.

### Action-family mix (model actions)

| Family | Instruct-2507 | Qwen3.5 off |
|---|---:|---:|
| move | 1,004 (35.8%) | 1,671 (44.0%) |
| end_turn | 615 (21.9%) | 641 (16.9%) |
| ability | 280 (10.0%) | 243 (6.4%) |
| build | 261 (9.3%) | 279 (7.4%) |
| recruit | 214 (7.6%) | 286 (7.5%) |
| research | 205 (7.3%) | 226 (6.0%) |
| harvest | 113 (4.0%) | 160 (4.2%) |
| attack | 62 (2.2%) | 176 (4.6%) |
| city_reward | 48 (1.7%) | 72 (1.9%) |
| capture | 1 (0.0%) | 33 (0.9%) |
| resign | 0 | 8 (0.2%) |

Instruct-2507 ended turns more often, attacked less than half as often, and captured almost nothing (1 capture against 33).

## Per-map comparison (15 maps, both seats grouped)

Each cell shows the final round and model decisions. Every game in both arms was a native loss.

| Map seed | Instruct seat 1 | Instruct seat 2 | Q3.5-off seat 1 | Q3.5-off seat 2 |
|---|---|---|---|---|
| 100691 | r28 · 143 | r19 · 94 | r20 · 117 | r12 · 74 |
| 100692 | r20 · 94 | r12 · 48 | r27 · 166 | r24 · 139 |
| 100693 | r9 · 43 | r29 · 116 | r19 · 106 | r23 · 162 |
| 100694 | r13 · 69 | r9 · 36 | r28 · 176 | r17 · 93 |
| 100695 | r22 · 94 | r22 · 110 | r20 · 137 | r20 · 105 |
| 100696 | r21 · 100 | r30 · 133 | r13 · 62 | r27 · 142 |
| 100697 | r21 · 119 | r9 · 50 | r31 · 196 | r23 · 115 |
| 100698 | r23 · 112 | r9 · 37 | r20 · 110 | r9 · 39 |
| 100699 | r27 · 91 | r24 · 121 | r24 · 135 | r23 · 146 |
| 100700 | r23 · 104 | r22 · 80 | r30 · 206 | r25 · 158 |
| 100701 | r24 · 109 | r23 · 93 | r12 · 100 | r26 · 127 |
| 100702 | r25 · 127 | r22 · 110 | r28 · 218 | r15 · 88 |
| 100703 | r13 · 69 | r21 · 96 | r11 · 69 | r22 · 172 |
| 100704 | r22 · 111 | r23 · 119 | r27 · 132 | r24 · 147 |
| 100705 | r20 · 99 | r15 · 76 | r11 · 58 | r19 · 100 |

**Native win difference (Instruct minus Qwen3.5-off), map-clustered:** 0.00 on every map. The bootstrap 95% interval is [0, 0], and 15/15 maps are tied. With zero wins on both sides, this run says **nothing** about which model is better at winning.

**Survival length (descriptive only, not a win metric):** Instruct survived 1.0 round less on average per map. The map-cluster bootstrap 95% interval is [−4.1, +1.8]. Instruct lasted longer on 9 maps and Qwen3.5-off on 6. This difference is not distinguishable from noise.

## Qwen3.5 thinking-on arm

Not included. `qwen35-thinking-30paired-v1/summary.json` did not exist when this analysis ran. A read-only look at its progress at 11:33 PDT (worker elapsed 1.71 h) showed:
- 2/30 games finished: one native loss and one `generation_incomplete` (exact-repetition abort with period 1, an operational failure);
- the other 28 games still active, with about 80–110 decisions each;
- 3,018 calls so far, all but that one closing thinking normally. That is not a loop, just long reasoning: a mean of 6,209 thinking tokens per move (median 6,066, max 17,414) and a median of 57.5 s per move. Both are rising as prompts grow, from about 46 s per move in decisions 1–20 to about 70 s in decisions 81–100.

Its workers have a 4 h Modal timeout (`timeout=14400`). Games that need more than about 200 model decisions may not finish. The off arm had 3 games above 190 decisions. When that run publishes its final summary, rerun the analysis with:
```
modal run --env staging -m experiments.modal_qwen3_instruct_baseline --run-id qwen3-instruct-2507-30paired-v2 --analyze-only
```
The analysis picks the on arm up automatically and adds an `instruct_minus_qwen35_on` comparison.

## Caveats
- One sampling sequence. The positions reached diverge between policies, so per-move comparisons are not paired positions.
- These are development validation maps, not an untouched test split.
- Sampling is the Qwen3.5 recipe, not the Instruct card's recommendation. A run with the card's settings could behave differently.
- The full summary, with per-game traces, statistics and runtime, is on the volume at `qwen3-instruct-2507-30paired-v2/summary.json`. The small copy is `runs/qwen3-instruct-2507-30paired-v2/summary-small.json`.
