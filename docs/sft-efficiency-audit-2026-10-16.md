# SFT efficiency audit — 2026-10-16

This audit covers the completed original smoke, the training-only repair collection, the fresh 10k repaired training run, saved-position evaluations and the interrupted/recovered evaluation. It changes no research settings or active GPU work. The current 50-game recovery remains running; its final throughput and outcome counts are not yet available.

## What “one sample at a time” means here

Training is one sequence per GPU microbatch, on one B200. Gradients accumulate over 64 microbatches before an optimizer update; 10,000 examples produced 157 updates. This is statistically a batch of 64 but does not obtain the hardware parallelism of a true batch of 64. Training is single-GPU, with gradient checkpointing, no sequence packing and no explicit length bucketing.

Live inference is already batched: four A100 workers, each with at most eight active games and one pending decision per game. Thus the cap is 32 concurrently active games, not a single game overall. Each individual game's next observation depends on its previous action, so that trajectory remains sequential. Host simulator/opponent/serialization work is processed in a Python loop before the next synchronous inference batch.

## Largest observed bottlenecks

| Priority | Bottleneck | Evidence | Implication |
|---|---|---|---|
| 1 | Expensive sequential rollout loop | Original smoke: 50 games, 13,156 model requests, 48,116 total simulator transitions; longest worker 43.19 minutes on four A100s. 62.4% of summed occupied worker time is inside model-call paths, 37.6% outside. | Repeated inference plus CPU preparation/logging are the largest recurring cost. More training GPU speed does not remove this loop. |
| 2 | Conservative training batching | Fresh 10k smoke: 65.23 minutes of optimization on B200, 2.56 examples/s, 10,479 processed tokens/s, about 39.95 GB peak tensor allocation. Microbatch 1, accumulation 64, gradient checkpointing enabled. | Larger true microbatches and length-aware batching are unbenchmarked opportunities. Peak allocation alone does not establish utilization, capacity for all long examples, or a speedup. |
| 3 | Repeated long inputs for tiny action outputs | Training processed 41,012,277 tokens for 87,623 supervised suffix tokens: ~4,101 total tokens/example and 8.76 supervised tokens/example. Original live evaluation processed 34.25M input tokens for 115,774 output tokens (~296:1). | Each action rereads the full public state/catalog. KV caching within one generation is distinct from reusing work across decisions; the current caller does not retain cross-call prefix caches. Context is necessary, so the 0.214% supervised-token share is not a claim that all other computation is useless. |
| 4 | Low behavioral return from training examples | Exact validation agreement: base 14.94%, first 5k smoke 53.4%, repaired 10k smoke 58.68%. Repaired movement: 6/575 exact choices; 17.74% even allowing equally top-scored teacher actions. Original smoke: 49/50 horizon truncations, only 119 movement and 49 attack decisions across 13,156 model actions. | Throughput and learning efficiency are separate. Weak movement/long episodes inflate rollout cost; making the same training faster does not establish stronger gameplay. Dataset size, coverage and corrections changed together, so their individual effects are unidentified. |
| 5 | Fragile orchestration and lost partial work | Local-client disconnect interrupted four game workers and one position worker: 5,258.12 summed worker seconds (~87.64 GPU-worker minutes), $4.11 lower compute estimate. The abandoned parallel invocation lasted ~21.5 elapsed minutes. | Cloud-owned coordination and resumable shards would avoid repeating unfinished work. Current detached local entrypoints only guarantee survival of their last triggered function, per the CLI warning. |

These are engineering priorities, not an exclusive decomposition: long input processing contributes to both the training and rollout totals.

## Measured rollout batching

The completed original smoke used 2,282 inference batches for 13,156 requests. Mean batch size was 5.77; time-weighted mean was 6.76. Full batches of eight accounted for 66.8% of model-call time. There were 309 singleton batches, largely a limitation of fixed small shard queues and worker tails; the trace audit does not separately attribute every singleton's cause.

The implementation left-pads every request to its batch's longest input. Across these batches, 34,250,080 real input tokens occupied 40,181,831 padded input slots: 14.76% padding. This is material, but not evidence of an order-of-magnitude batching loss. Padding slots are not an exact FLOP or wall-time measure.

The GPU-capable simulator is not being exercised as a vectorized GPU rollout engine here. The current evaluation launcher sets `JAX_PLATFORMS=cpu`; the adapter steps a single environment at a time. Simulator steps, scripted opponent actions, legal-catalog extraction, both-seat public observations and gzip/JSON trace serialization all happen between synchronous model calls. The measured 37.6% outside model calls includes those tasks plus load/artifact work; there are no finer spans to attribute it precisely. Model-call time also includes CPU tokenization/decoding, so 62.4% must not be called pure GPU compute time.

## Recent stage timing

| Completed stage | Elapsed work evidence | Caveat |
|---|---:|---|
| Original 5k smoke training | 55.05 min optimization | H100; different dataset/context from repaired run |
| Original 50-game smoke evaluation | 43.19 min longest shard | 153.33 aggregate GPU-worker min |
| Training-only 50-game repair collection | 38.53 min longest successful shard | 142.95 aggregate GPU-worker min; preemption recovery added delay |
| Repair replay, relabel, sample, audit and prepare | ~16.80 min sequential driver stages | Sampling alone took 8.51 min; no paid GPU required for these stages |
| Repaired 10k smoke training | 65.23 min optimization; 67.55 min local invocation | B200; 66.41 min occupied remote function including archive |
| Fresh base 5k positions | 12.59 min longest shard | 49.65 aggregate GPU-worker min |
| Repaired 5k positions | 15.16 min longest successful shard | 59.41 aggregate GPU-worker min; interruption/recovery extended elapsed completion |

Parallel worker time must not be added and presented as elapsed waiting time. Likewise the H100/B200 training runs above are not a matched hardware speed comparison because their datasets differ.

## What to investigate next

1. Profile finer rollout spans and GPU utilization, then overlap CPU environment preparation with model inference and feed workers from a larger queue. Benchmark batch sizes/length grouping on a separately versioned, paired evaluation rather than silently changing the live comparison.
2. Benchmark larger training microbatches while holding effective batch 64 fixed; compare checkpointing and packing only with verified memory, response masks and example boundaries. There is no measured speedup yet.
3. Move run ownership to a cloud coordinator and persist resumable shard state so a local disconnect cannot discard the batch.
4. Address the weak movement and stalled-game behavior before committing to larger training volume. The current evidence identifies the weakness, not its exact cause or a proven curriculum fix.

Additional CPU data-preparation/serialization optimization is worthwhile, but its measured ~17 minutes is smaller than repeated model training and full-game rollout costs. The audit does not justify shortening evaluation horizons, dropping bad outcomes, or changing current gates after seeing results.

Evidence: `runs/sft0/batch-efficiency-v1.json`, `smoke-game-runtime-v1.json`, `repaired-smoke-result-v2/summary.json`, `prepared-repaired-smoke-v2/manifest.json`, `repaired-position-tie-diagnostic-v2.json`, `repair-driver-status-v1.json`, `disconnect-evidence-v2/preservation-receipt.json`, and the per-run `execution.json` receipts. Source paths: `src/polytopia_bot/learning/training.py`, `serving.py`, `gameplay.py`, `adapter.py`, and `experiments/modal_prompt_games.py`.
