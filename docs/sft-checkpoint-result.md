# First SFT checkpoint: efficiency continuation

The same-data 10k smoke checkpoint is trained and its saved adapter passed integrity checks. The smoke learning/gameplay diagnostic **failed**. This is an experimental smoke adapter, not an accepted main gameplay checkpoint. No main training or final-test evaluation ran.

Adapter directory: `/Users/K.Z./Desktop/projects/polytopia-bot/runs/sft0/efficient-smoke-result-v1/training-run/model/adapter-final`. Load it as a PEFT adapter on `Qwen/Qwen3-4B-Instruct-2507`, pinned revision `cdbee75f17c01a7cc42f958dc650907174af0554`. The frozen prompt, tokenizer, context and inference settings are recorded in `runs/sft0/efficient-smoke-result-v1/allocation.json`.

## Runtime changes

- Rollout CPU lanes overlap with one persistent model worker per GPU. The fixed-prefix pilot was **1.50× faster**, with identical prompts, outputs and transitions. Full-game evaluation uses a fresh common base/SFT pipeline allocation and cloud coordinators.
- Training uses microbatch **4**, effective batch **64**, response-only vocabulary projection and length sorting within each unchanged effective batch. GPU loss/gradient checks, longest-input stress and fit/reload checks passed.
- Full 10k optimizer time: **65.23 → 62.15 minutes**, or **1.050× throughput**. The small benchmark measured 1.080×. These are throughput results, not evidence of better sample efficiency.
- In the new game run, inference averaged **4.55 requests per batch**. The model-call path, including tokenization and decoding, occupied **95.5%** of worker time; singleton batches used **4.3%** of model-call time. This is not a GPU hardware-utilization measurement. Long prompts and many decisions per game remain the major runtime costs. Different policies and batch compositions prevent treating the old/new full-game elapsed-time ratio as a pipeline speedup.

## Checkpoint results

- Saved positions: **59.20%** exact teacher agreement versus **14.94%** for the base model; **100.00%** first-attempt validity.
- Live games: **7 wins**, **7 native losses**, status counts `{'native': 14, 'truncated': 35, 'budget_incomplete': 1}` across all 50 allocated games. Native completion was **28.00%**, against the required 80%.
- All recorded transitions and public observations passed independent prefix replay. One game hit the fixed runtime ceiling; full-game evaluation and its execution gate remain incomplete. Horizon truncations and budget-incomplete games are not native draws.
- The bounded game evaluation cost approximately **$10.37** at recorded published compute rates. This excludes startup/idle, builds, storage and egress. No research GPU allocations remain active.

The next research issue is closed-loop gameplay quality, especially the large number of episodes reaching the horizon. Any further data or teacher change needs a new versioned research protocol; these validation traces must remain diagnostic. Runtime improvements do not waive the failed completion gate.

Authoritative result: `runs/sft0/efficient-checkpoint-result-v1.json`. Detailed implementation and failed benchmark attempts: `docs/sft-efficiency-implementation.md`. Cost evidence: `runs/sft0/cost-ledger.json`.
