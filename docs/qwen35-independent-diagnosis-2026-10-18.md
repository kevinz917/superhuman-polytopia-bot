# Independent Qwen3.5-4B latency diagnosis

The user cancelled the recurring monitor on October 18 and requested an independent diagnosis of thinking latency, informed by published serving guidance. Automation `finish-qwen-thinking-comparison` was deleted through the Codex automation API. No replacement automation was created.

The existing comparison uses vLLM 0.19.0 `AsyncLLM`, one Modal H100 per mode, 100 asynchronous games per mode, and `max_num_seqs=40`. CUDA graphs, chunked prefill and asynchronous scheduling are enabled. Model weights, simulations and checkpoints remain in Modal. There is no local model server. The independent diagnostic is separate from the immutable gameplay run.

## Evidence before intervention

At approximately 16:13 UTC, thinking had completed 776 natural responses but zero whole games. Non-thinking had completed 21 games, all native losses, with no forfeits. Thinking's median input/output lengths were 2,810/5,165 tokens. Median request time was 196.165 seconds; median time to first token was 118.152 seconds. These are concurrent workload measurements and the two medians must not be subtracted as a per-request decomposition. For the most recent recorded request, 115.427 of 186.712 seconds elapsed before its first token; 5,515 tokens then completed naturally.

Two H100 snapshots at 16:14:55 UTC showed 96% and 99% GPU utilization. High utilization does not prove optimal throughput, but contradicts a serial, idle-GPU explanation. One scheduler admits only 40 of 100 pending requests, so waiting is expected. The diagnostic measures whether more admitted sequences increase throughput without disabling CUDA graphs or exhausting the cache.

Saved text shows avoidable reconsideration and decoding of numeric fields, rather than the old endless numeric echo. The 13,056-token opening response repeatedly debates researching now versus later and invents uncertainty about upkeep. A 2,071-token response repeatedly verifies that only end-turn and resign are available. Another response misreads a supplied distance and then debates the imagined discrepancy. These are generated reasoning failures; the inspected distance fact itself is consistent with its coordinates.

## Published guidance checked

- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/README.md): current sampling matches its general-thinking recommendation. A separate exploratory profile tests the card's precise-coding sampling; that is not assumed optimal for gameplay. Thinking is enabled through the chat template, and prior thinking is not included in history.
- [vLLM Qwen3.5 recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md): MTP is primarily a low-concurrency latency option and can reduce throughput under load. It is not automatically the right fix for 100 active games. Hybrid prefix caching remains experimental in the recipe.
- [vLLM 0.19 optimization guide](https://docs.vllm.ai/en/v0.19.0/configuration/optimization/): measure queueing, preemption and token throughput; use model replicas to scale independent requests. Larger prefill token batches are another candidate for small models on large GPUs.
- [vLLM 0.19 multimodal configuration](https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/config/multimodal.py): language-model-only is equivalent to setting every supported modality limit to zero. This run already disables image and video inputs; merely renaming that setting is not an established optimization.

## Independent experiment

`runs/qwen35-thinking-independent-v1`, app `ap-FasBcPz7TuActqd1uqdNfz`, uses one additional H100, bounded to one hour. It reuses the pinned checkpoint from the cloud cache. It compares identical sets of 100 synthetic 2,048-output-token requests at scheduler capacities 100 and 40. These fixed-length probes are serving measurements only; no game actions execute. Four captured game positions then compare baseline instructions, a focused request placed after the observation, and an alternate published sampling recipe. Two simple controls check whether long reasoning occurs without the game observation. Natural-completion probes have no thinking/output token cap and retain the complete text.

## Results and applied repair

All measurements below ran on Modal H100s with the same pinned BF16 weights. Startup/load/compile time is excluded from the speed probes. A single fixed batch per setting is a diagnostic measurement, not a stable full-game speed estimate.

| Scheduler capacity | Requests | Output tokens per request | Total seconds | Output tokens/sec |
|---|---:|---:|---:|---:|
| 40 | 100 | 2,048 | 67.93 | 3,014.81 |
| 100 (graph ceiling rounded to 96) | 100 | 2,048 | 77.16 | 2,654.27 |
| 128 (graph-compatible) | 100 | 2,048 | 43.58 | 4,699.79 |
| 128 | 100 | 8,192 | 267.34 | 3,064.25 |
| 64 | 50 | 8,192 | 130.12 | 3,147.91 |

The 128-capacity short probe is 55.9% faster than capacity 40. The longer probe is the more relevant warning: 100 long responses do not retain the short-probe throughput. Fifty long requests on one replica complete in about half the wall time with similar aggregate token throughput. This supports distributing the 100 games over two independent replicas, each with 50 active games and scheduler/graph capacity 64. It does not establish an exact full-game speedup. Both sustained tests showed zero waiting requests after admission; logged cache utilization peaked around 75% for 100 requests and 38% for 50, with no preemption warnings observed.

The four natural baseline positions averaged 5,160 tokens and 30.80 seconds at low concurrency. A focused reminder appended after the observation averaged 5,478.5 tokens and 30.85 seconds: **6.2% more tokens**, so it was rejected. The simple arithmetic control naturally closed after 135 tokens in 0.69 seconds. Even a 46-token game prompt with only end-turn versus resign generated 1,157 tokens in 5.65 seconds. Verbosity is therefore partly model behavior, not solely the large state representation.

The alternate temperature 0.6 / presence penalty 0 recipe was much worse. One position naturally completed with 8,526 tokens. Three requests entered exact repeated-paragraph loops and were still unfinished beyond 30,000 tokens. Their actual saved text repeatedly rechecks the same JSON action. The diagnostic app was stopped after preserving the streams. These interrupted diagnostics are not actions, completed responses, or game losses. General-thinking sampling and the existing prompt remain unchanged in the comparison.

The existing repetition detector had a demonstrated blind spot: it searched only periods 1–16. Retokenizing the three actual saved streams on Modal found periods 96, 51, 31; the old detector missed all three. The repaired detector checks periods up to 128 across 1,024 tokens, requiring at least eight identical cycles. Nonperiodic thinking remains uncapped. It detects all three saved failures and does not fire on any checked prefix of the eight naturally completed baseline/focused controls. Five prompt/repetition checks and five shard-accounting tests also passed on Modal. The serving launcher now checks actual CUDA graph coverage and periodically logs vLLM running/waiting counts and cache usage.

Receipts: `runs/qwen35-thinking-independent-v1/independent-summary.json`, `stop-receipt.json`, `loop-reproduction.json`; capacity128 short probe in `qwen35-thinking-independent-v2`; long probes in `qwen35-thinking-independent-v3` and `v4`. All diagnostic apps are finished or stopped; their results are retained.

## Replacement paired comparison

The queue-limited run was stopped after preserving its reports and source receipts. At that snapshot, thinking-off had 72 completed games, all native losses; thinking-on had no completed games. Unfinished games are not losses or draws, and no result from that partial run is mixed into its successor.

`qwen35-thinking-100paired-v2` starts fresh 100-vs-100 on the identical 50 maps/seats/jobs (seeds 100691–100740), with unchanged weights, prompt, grammar, sampling, opponent and per-decision seed formula. Each mode uses two H100/vLLM replicas with 50 games each, capacity 64 and compiled graphs. There are four cloud GPU workers total. A cloud coordinator owns progress and final aggregation; local scripts fetch only small logs/reports. The recurring Codex automation remains deleted.

App: https://modal.com/apps/duper/staging/ap-zrY0e3Y5FD4skmIopv7Ila

Coordinator: `fc-01M57XP07EK181CNJHK3TWR9RT`. Protocol SHA256: `3020972e1fadd47cd70217baeac08947f2954fa457796e0de32b9999a4add6e9`. Final results remain pending. The shard collector rejects overlapping jobs, changed identities, mismatched runtimes, missing final coverage and duplicate timings. Cloud tests passed; relative trace links were also tested on the actual Modal volume.

Initial replacement-run verification: all four workers loaded the pinned model with actual scheduler/graph capacities 64/64. A subsequent small-log snapshot contained 2,437 naturally stopped non-thinking calls and 82 naturally stopped thinking calls, all 82 with closed reasoning. Initial median thinking first-token/request times were 11.33/93.30 seconds; these startup measurements are not a controlled speedup estimate. First-actions receipt: `runs/qwen35-thinking-100paired-v2/first-actions-verification.json`. Full-game results remain pending.
