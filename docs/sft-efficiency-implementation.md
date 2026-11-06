# SFT runtime efficiency continuation

The user selected priorities 1 and 2 from the efficiency audit: overlap rollout CPU work with inference, and batch SFT training. This continuation preserves the frozen simulator, teacher, public observation interface, audited repaired 10k examples, effective training batch and response-token objective. Main training and final-test evaluation still require their original gates.

## Rollout pipeline

`efficient_gameplay.py` runs independent CPU game lanes. Each retains the original `play_games` state machine, job order and inference batch boundaries. One persistent inference process handles queued public prompts; private engine state stays in the CPU workers. Full-game runs use four A100 80GB workers, two lanes per worker, and a durable cloud coordinator.

The training-map prefix benchmark compared the same 256 model requests and 466 engine transitions. Serial execution took 160.355 seconds; pipelining took 107.023 seconds: **1.4983× faster**. Every prompt and generated output hash matched. This is an engineering pilot, not a guarantee of full-game throughput or evidence of gameplay quality. Receipt: `runs/sft0/efficiency-rollout-benchmark-v1/execution.json`.

Full validation uses a new common base/SFT lane allocation because splitting a shard changes batch composition. Parent dispatch logs count actual model time once. Lane batch times include queue waits and cannot be added as GPU time. The new base-model 50-game run and independent replay audit passed.

## Training implementation

`efficient_training.py` computes causal cross entropy only at supervised response positions. Context tokens still participate in the transformer and receive gradients through attention. Ignored prompt positions no longer pass through the vocabulary projection. Length sorting stays within each effective-batch group; a comparison of the actual original and optimized Trainer loaders confirmed identical group membership, including the final partial group.

Each microbatch contributes its summed response loss divided by the full effective batch's supervised-token count. The direct backbone call explicitly enters BF16 CUDA autocast, matching the original Trainer path. Twelve local tests cover loss and LoRA-gradient equivalence, accumulation, batch membership, real Trainer normalization, unsafe padding, and the actual SDPA mask/causal arguments.

Strictly right-padded inputs use an all-valid attention mask. Causality prevents real tokens from attending the future padded tail, and padding labels must be ignored. Left, interior and nonbinary padding, and supervised padding, are rejected. An explicit all-valid 2D mask avoids Transformers 4.57.1's packed-sequence detection, which otherwise materializes a redundant 4D mask. The GPU probe verifies that no mask reaches causal SDPA.

## Benchmark corrections and correctness evidence

Attempts remain separate and their cost is not assumed zero:

- v1 failed on a remote import before benchmarking.
- v2 omitted production CUDA autocast and was stopped.
- v3 used production precision, but padded attention made larger batches slower.
- v4 showed that passing no mask still triggered a packed-sequence mask in Transformers.
- v5 reached mask-free causal attention, but failed comparison against a padded dense probe. No training launched from it.
- v6 compared against the actual production reference: unpadded microbatch 1, accumulated under one shared response-token denominator. It retained the padded probe as a separate numerical diagnostic.

For v6, the reference loss was 0.9839137495 and candidate loss 0.9839137197. Relative gradient error was 1.3976%, with cosine similarity 0.9999124. Both passed the existing thresholds of less than 3% and greater than 0.999; no tolerance was relaxed. Padded dense gradients differed substantially from the actual serial reference on this BF16 stack. Full-precision local tests establish mathematical objective equivalence; the GPU comparison establishes agreement with the actual original training path within the recorded tolerance.

## Selected runtime

The completed v6 benchmark used 128 examples, 524,665 real tokens and effective batch 64:

| Variant | Seconds |
| --- | ---: |
| Original dense head, microbatch 1 | 51.0547 |
| Response-only head, microbatch 1 | 49.2914 |
| Response-only head, microbatch 4 | **47.2926** |
| Response-only head, microbatch 8 | 51.4766 |
| Response-only head, microbatch 16 | 61.4084 |

Microbatch 4 was selected: **1.07955× pilot throughput**. Its longest-record stress test passed at 15,232 tokens and 35.87 GB peak allocation. Longer padded batches added enough transformer work to erase the benefit. This is a modest gain, not a batching multiplier. Receipt: `runs/sft0/efficiency-training-benchmark-v6/execution.json`.

The benchmark occupied 373.20 seconds, approximately $0.681 in published-rate compute. Invalidated attempts are separately recorded in `runs/sft0/cost-ledger.json`. Estimates exclude startup/idle, builds, storage and egress; they are not invoices.

## Fresh fit and checkpoint continuation

`efficient-fit-result-v1` passed: exact response accuracy rose from 12.5% to 99.21875%; frozen base fingerprints stayed unchanged; LoRA fingerprints changed; reloaded greedy responses were identical. Optimization took 623.7513 seconds and the complete function/archive took 967.5340 seconds, approximately $1.765.

This small effective-batch-8 fit was slower than the old 515.4928-second fit because short grouping windows incurred more padding. It is a correctness gate, not the full-run speed estimate. The same audited 10k smoke has now completed with microbatch 4 and effective batch 64. Optimizer time fell from 3,913.8456 to 3,728.9747 seconds (65.23 to 62.15 minutes): 1.0496× throughput. Full function/archive time fell from 3,984.5994 to 3,855.0987 seconds. Peak GPU allocation was 35.88 GB; all 504 saved LoRA tensors passed integrity checks. Published-rate compute was approximately $7.031 versus $7.268 for the earlier run. The measured full-run gain is smaller than the pilot estimate.

The optimized checkpoint completed all 5,000 saved-position decisions: 59.2% exact teacher agreement and 100% first-attempt validity. The bounded game allocation ended with 14 native outcomes (7 wins, 7 losses), 35 research-horizon truncations and 1 budget-incomplete prefix. Native completion is 28%, below the required 80%; its best possible value after the unfinished game is 30%. Independent prefix replay verified all 45,839 engine transitions and 15,953 model decisions. The full-game audit correctly rejected the incomplete allocation; its historical driver status remains `stopped_on_error`. The separate negative-result package preserves this distinction and grants no acceptance.

The new run averaged 4.55 requests over 3,504 inference batches. Parent dispatch measured 12,302.36 seconds in the model-call path against 12,882.12 occupied worker seconds (95.5%); singleton batches consumed only 4.3% of model-call time. Model-call time includes tokenization/decoding and is not hardware utilization. CPU overlap is working, while prompt processing and long episodes dominate the remaining cost. Full-game policies and batch compositions differ, so their total-time ratio is not a controlled pipeline speedup. Game compute was approximately $10.371 including its CPU coordinator; all research allocations have ended.

Final report: `docs/sft-checkpoint-result.md`; structured evidence: `runs/sft0/efficient-checkpoint-result-v1.json`. Main and final-test gates remain closed.

## Earlier repaired checkpoint result

The earlier repaired 10k adapter scored 58.68% exact teacher agreement and 100% first-attempt validity on 5,000 saved positions. Its game evaluation recorded 12 native outcomes (5 wins, 7 losses), 33 horizon truncations and 5 budget-incomplete prefixes. All 40,762 recorded transitions replayed successfully. Even if all five unfinished games ended natively, native completion would reach only 34%, below the required 80%.

The prefix audit cannot satisfy the complete-game gate. Evidence: `runs/sft0/repaired-smoke-research-result-v2.json`. All recorded model decisions include 2,211 moves, 78 captures and 412 attacks; those counts include unfinished prefixes and are descriptive diagnostics, not independent samples or proof of improved generalization.
