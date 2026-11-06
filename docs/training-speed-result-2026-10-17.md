# Training speed implementation and measured results — October 17, 2026

**The complete validation workflow finished in 48.72 minutes**, including fresh fit, training, 15,000 full saved-position evaluations, 50 base and 50 trained games, early diagnostics, and CPU replay/adapter verification. The prior fit-to-summary interval was 158.29 minutes; that historical workflow had different control work and policy trajectories, so this is an observed iteration-time comparison, not a controlled compute-speed multiplier.

The same 10,000-example Qwen3-4B training workload now takes **10.77 optimizer minutes instead of 62.15 minutes: 5.77x faster**. It uses four B200s instead of one, but optimizer GPU-seconds still fall by **30.7%**. The fresh fit, serving benchmark, full position comparisons, and paired gameplay were executed; final replay verification is recorded in the completed receipt linked below.

The earlier approximately 5% gain was a wall-time improvement from 65.23 to 62.15 optimizer minutes. It was not a measure of how much the model learned. That change retained the main training and inference execution paths. This implementation changes those paths and qualifies them against unchanged inputs.

| Stage | Earlier measurement | New measurement | Scope |
|---|---:|---:|---|
| 10k optimizer | 62.15 min, 1 B200 | 10.77 min, 4 B200 | 5.77x wall speedup; 30.7% fewer optimizer GPU-seconds |
| Fresh 128-example fit invocation | 16.85 min | 6.30 min | Includes startup, training, accuracy and reload checks |
| Fixed 128-prompt inference | 154.12 sec | 23.89 sec | 6.45x; same A100, exact input token sequences |
| 5k base positions | 19.46 min, 1 A100 reference | 5.88 min, 4 A100 shards | 3.31x in the new runtime |
| 5k trained positions | 22.05 min, 1 A100 reference | 8.23 min, 4 A100 shards | 2.68x in the new runtime |
| 50 trained games | 76.04 min, one time-budget incomplete | 22.49 min, all 50 evaluated | 3.38x observed wall speedup; trajectories differ |
| First 512-position diagnostic | Previously no equivalent early diagnostic | 7.18 min after training invocation starts | Also during training, rather than waiting for full games |
| Four paired 200-action prefixes | Previously no equivalent early diagnostic | 7.83 min after training invocation starts | Base and trained; not full-game outcomes |

The historical four-A100 saved-position evaluation took 15.54 minutes. The new four-shard trained evaluation therefore takes about half that time, although the model and serving numerics differ. The controlled same-adapter sharding comparison is 22.05 to 8.23 minutes. Sharding increases GPU-seconds versus a single-GPU position evaluation; it buys lower latency, not lower resource use in that stage.

## What changed

Training uses token-budget microbatches, isolated packed attention segments, fewer redundant LoRA casts, and no activation recomputation where the longest-input memory test permits it. The same global groups of 64 examples are split across four GPUs by token count. Response loss uses the global response-token denominator and gradients are summed. Model revision, training data, prompt, teacher, optimizer schedule, and simulator rules remain fixed. Single-GPU matched throughput improved 1.53x; the additional gain comes from four-GPU execution. A FlashAttention candidate that failed gradient parity was rejected.

Serving uses vLLM continuous batching and a shared request queue across CPU game workers. Games take jobs from a common queue, checkpoint per-game state, and retain durable completion records. Full position summaries publish independently of full-game completion. The cloud coordinator owns dependent stages even if the local terminal disconnects.

The default full-position path now uses four frozen shards after a separate quality and speed qualification. Early reports add action-family agreement and public-state progress indicators. A sample early game showed 26 consecutive movement reversals among 38 movement actions; another captured a city. No-discovery streaks and reversals are descriptive, not automatic failure labels.

## Learning quality

On the same 5,000 saved positions under the new serving backend, the base scores **14.76%**, the new adapter **57.00%**, and the previous adapter **57.16%** teacher agreement. The new and previous adapters differ by 0.16 percentage points in this run. New-adapter first-attempt validity is **99.68%**. This is evidence that the training acceleration retained the measured imitation gain; it is not a multi-seed equivalence result.

The previous reported 59.20% used the old serving backend. Greedy generated actions are not bitwise identical across these runtimes. The previous-adapter control avoids attributing that entire difference to the new trainer. The four-shard candidate scored 57.56% with 99.76% first-attempt validity, passed all predeclared quality/speed limits, and used the exact same saved adapter and input identities.

The full trained evaluation produced **6 wins, 4 losses, and 40 research-horizon truncations**, with zero forfeits and zero time-budget incompletions. The base produced 32 native losses and 18 forfeits. The trained checkpoint therefore fails the unchanged 80% native-completion gate (10/50 = 20%); it is not promoted. The historical trained run had 7 wins, 7 losses, 35 horizons and one time-budget incompletion. These 50-game results do not establish a gameplay improvement.

Native game completion is neither win rate nor a general learning-efficiency percentage: resignations and losses also count as native outcomes. Report wins, losses, horizons, forfeits, and time-budget incompletions separately. The unchanged smoke gate still requires 80% native completion; speed improvements alone do not establish a stronger or accepted policy.

## Verification

All 100 full game traces passed independent CPU replay: **44,530 transitions**, including 12,881 trained-model decisions. Replay checked public before/after observations, prompts and repair handling, selected commands, outcomes, and counters. All 504 saved adapter tensors are finite and LoRA-only, nonzero B weights are present, and the downloaded adapter hashes match the trained artifact. The fresh fit reached 99.22% accuracy and passed reload-output identity. The focused suite passed **26 tests** (20 training, 4 gameplay/recovery, 2 public-progress diagnostics); all 22 frozen simulator source hashes remain unchanged.

Every owned Modal app is stopped with zero remaining tasks. No GPU allocation is left running. The gameplay acceptance gate remains failed, independently of successful runtime/integrity verification.

## Evidence and reproduction

- [Completed workflow summary](../runs/speed-v2/end-to-end-v1/summary.json), [cloud stage status](../runs/speed-v2/end-to-end-v1/status.json), and [protocol](../runs/speed-v2/end-to-end-v1/protocol.json).
- [Training benchmark](../runs/speed-v2/training-benchmark-v4/execution.json), [distributed qualification](../runs/speed-v2/distributed-benchmark-v1/execution.json), and [inference benchmark](../runs/speed-v2/inference-benchmark-v1/execution.json).
- [Four-shard qualification](../runs/speed-v2/sharded-positions-v1/execution.json), [early public progress](../runs/speed-v2/end-to-end-v1/early-progress-diagnostics.json), and [local verification](../runs/speed-v2/local-validation.json).
- [Runtime instructions](speed-runtime-v2.md) and [downloaded adapter](../runs/speed-v2/end-to-end-v1/adapter-final/adapter_config.json).

The measured first workflow used single-GPU full-position evaluation. Its observed elapsed time must not be replaced with a projection incorporating the later sharding improvement. Final traces and replay snapshots remain on the `polytopia-sft0-training` Modal volume under `speed-v2/end-to-end-v1`; only compact results and the final adapter were downloaded. Startup/cache variants did not establish a repeatable improvement and were not adopted. One-time engineering and qualification runs are separate from recurring iteration time. No main50k or final-test run was launched, and no policy was promoted.
