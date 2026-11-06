# Speed runtime v2

This is a separate runtime for the frozen Qwen3-4B SFT experiment. It retains the pinned model revision, prepared examples, public prompt, response-only loss, global effective batches, optimizer schedule, and frozen simulator. The original runners remain available.

## Execution

The cloud coordinator in `experiments/modal_speed_workflow.py` owns the dependent stages, so losing the local terminal does not lose orchestration. It consumes the saved training and inference benchmark receipts, runs a fresh 128-example fit/reload gate, trains the same 10k examples, and evaluates the base and new adapter on the same 5k saved positions and 50 allocated games. A third position evaluation uses the previous adapter to distinguish serving effects from training effects.

```sh
modal run --detach --env staging experiments/modal_speed_workflow.py --benchmark-distributed
modal run --detach --env staging experiments/modal_speed_workflow.py --run-id speed-v2/end-to-end-v2
```

The benchmark command is for initial qualification; its receipt already exists. For another full run, choose a fresh run ID (the example uses `end-to-end-v2`). Each output root is exclusive. Do not repeat a command against an existing run ID or launch a duplicate while its cloud call is active. Local `call.json` files identify the cloud-owned calls. The training volume is `polytopia-sft0-training`; final traces and private replay snapshots stay there.

## Training

`fast_training.py` provides token-budget microbatches, optional segmented attention with isolated examples, selective activation recomputation, and optional removal of redundant LoRA input casts under BF16 autocast. Adapter parameters and optimizer state remain FP32. Every candidate must pass loss/gradient checks against serial SDPA and the longest-input stress check before selection. The upstream FlashAttention packed path failed the current parity check and is not eligible.

The distributed runner partitions each unchanged effective batch among four GPUs by input length. Each rank divides response loss by the global response-token count; gradients are then **summed**, not averaged. The scaling benchmark checks this normalization on real model gradients before a distributed full run is eligible. Hardware count, wall time, and GPU-seconds are separate measures.

## Serving and feedback

`fast_serving.py` uses vLLM continuous batching, chunked prefill, and prefix caching. It feeds identical token IDs to the pinned base model and adapter. Greedy outputs can differ across implementations, so backend replacement requires fresh paired quality evaluation. The fixed 128-prompt benchmark provides the controlled speed comparison.

At step 16, the trainer publishes an immutable adapter and launches a fixed 512-position diagnostic plus four paired game prefixes capped at 200 total actions each. Position summaries include action-family agreement. Game summaries count model action kinds. These prefixes reveal early behavior; they do not count as completed full games or replace the 50-game evaluation.

Full position results publish independently of game completion. Games draw from a shared queue and run in independent CPU processes; pending decisions feed one continuous inference engine per GPU. Game state, counters, and trace offsets are checkpointed every 16 transitions, with periodic volume commits. Recovery truncates the uncommitted trace tail before resuming. Completed games have durable result records. This is per-game recovery support, not automatic retry of failed cloud allocations.

Full saved-position evaluations now use the four frozen 1,250-position shards. The paired qualification receipt is `runs/speed-v2/sharded-positions-v1/execution.json`: all 5,000 input identities matched; base wall time fell from 1,167 to 353 seconds, and trained from 1,323 to 494 seconds. Both passed the predeclared limits of at most one percentage point agreement loss, half a point validity loss, at least 95% trained validity, and at least 2x wall speedup. The coordinator requires this passing receipt. The early 512-position diagnostic remains a single job. At most eight position GPU workers run concurrently; the previous-adapter control may queue behind base and trained.

Early game reports also include public-state movement reversals, newly discovered tiles, captures, city counts, and consecutive end turns without discovery or capture. These are descriptive indicators, not automatic failure labels: waiting or developing an economy can be useful. This makes a concrete movement loop visible without waiting for the full game horizon. CPU replay begins after full games finish and overlaps any remaining position evaluation.

The first end-to-end validation used the original single-GPU full-position path; sharding was qualified separately on its exact saved adapter. Its measured time must not be substituted into the first workflow's observed elapsed time. Later diagnostic and orchestration additions likewise do not retroactively change that run's immutable source snapshot.

## Verification

Focused tests cover response-loss/gradient equivalence, packed attention isolation, global batch partitioning and summed normalization, asynchronous response routing, repair/forfeit behavior, bounded diagnostics, and interrupted-game recovery. GPU measurements additionally gate the actual runtime. Final full-game traces are independently replayed on CPU, and adapter hashes, finite values, and LoRA-only tensors are checked.

No prompt compression, teacher/data change, smaller-model substitution, main50k training, or final-test evaluation is included in this runtime experiment. Compact prompts remain a separate learning experiment because lossless serialization alone does not establish equivalent policy behavior.
