# Qwen3.5-4B: current-harness gameplay pilot

The non-thinking mode played games but won none. Thinking mode exhausted the 4,096-token response budget on every opening decision without returning a final action. This establishes a budget/response-completion problem; it does **not** establish which mode plays better when thinking has enough output budget.

No training or adapters were used. The existing `public-action-facts-v1` harness and game prompt were unchanged. No planning fields, new memory, repair calls, or fallback actions were introduced.

## Results

| Measure | Thinking off | Thinking on |
|---|---:|---:|
| Allocated games | 4 | 4 |
| Native wins | 0 | 0 |
| Native losses | 3 | 0 |
| Invalid-action forfeits | 1 | 0 |
| Unfinished games due to response budget | 0 | 4 |
| Valid executed model actions | 513 / 514 calls | 0 / 4 calls |
| Mean generated tokens per call | 8.31 | 4,096 |
| Total generated tokens | 4,270 | 16,384 |
| Gameplay wall time, four games batched | 162.87 s | 162.43 s |

Thinking's four native outcomes are **unobserved**, not draws or losses. All four responses ended with `finish_reason=length`, exactly 4,096 generated tokens, and no closing thinking delimiter or final action. The parser did not execute proposed actions mentioned within unfinished reasoning.

| Map seed / model seat | Thinking off | Thinking on |
|---|---|---|
| 100691 / 1 | Eliminated, round 18; 122 actions | Opening decision hit token cap |
| 100691 / 2 | Invalid-action forfeit, round 19; 203 valid actions | Opening decision hit token cap |
| 100692 / 1 | Model resigned, round 8; 33 actions | Opening decision hit token cap |
| 100692 / 2 | Eliminated, round 24; 155 actions | Opening decision hit token cap |

The invalid output was `{"action_id":"a147"}` with only 146 catalog entries (`a0`–`a145`). That game had reached four owned cities and eleven units; it was stopped by the single-call output contract, not a native defeat. The resignation was a legal action explicitly selected by the model while it still owned two cities and four units. The other two native losses were eliminations.

## Timing interpretation

Non-thinking used 204 inference batches totaling 89.79 seconds; the median batch took 0.419 seconds, with batch sizes varying from four to one as games ended. Thinking used one four-request batch taking 146.00 seconds, yielding no executable decisions. These are synchronous batch times, **not** measured individual-request latency or a controlled steady-state speed ratio. The first calls may include initialization overhead. vLLM did not return individual request timing in this configuration.

Model initialization took 122.35 seconds off and 138.30 seconds on. Successful worker occupancy was 303.69 and 311.44 seconds respectively, totaling about 10.25 H100 worker-minutes. Both processes exited successfully and the Modal app subsequently reported `stopped`, zero tasks.

## Frozen comparison

- Model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; identical weights, tokenizer and chat template, no Polytopia adapter.
- Toggle: native chat-template `enable_thinking`; final-action parser recognizes the thinking delimiter only in the enabled arm.
- Identical sampling: temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, repetition penalty 1.0; deterministic per-request seed sequence starting at 17. This controls the setting, not an identical random trajectory after modes diverge.
- Two held-out validation maps, both seats, against the same deterministic balanced scripted teacher. No teacher-agreement scoring or final-test maps.
- Frozen 14×14 Bardur-1v1-v1 simulator. Maximum 256 model decisions per game and 30 minutes of gameplay per arm; neither limit caused these outcomes.
- Maximum context 32,768; total generated response limit 4,096, including reasoning. No prompt truncation or context overflow occurred.
- Two identical H100 80 GB workers; BF16, vLLM 0.19.0, Torch 2.10.0, Transformers 4.57.6, JAX 0.11.2. Explicit Triton GDN prefill backend; eager execution, prefix caching disabled.

The four games per arm span only two maps and one sampling seed sequence. They are a basic pilot, not a statistically reliable ranking. Equal token caps constrain thinking disproportionately: this measures usability at that budget and cannot establish its unconstrained gameplay strength.

## What this supports next

Keep non-thinking as the usable baseline for this exact budget. Before spending on another thinking gameplay suite, measure whether a larger response allowance lets it finish the four opening decisions, using the same prompt and a separately recorded protocol. Only then run matched complete games to compare playing strength. The present evidence supports neither more training nor a claim that thinking inherently plays worse.

## Artifacts and validation

- Protocol: `runs/qwen35-thinking-games-v2/protocol.json`
- Aggregate: `runs/qwen35-thinking-games-v2/summary.json`
- Per-arm results/runtime/trace archives: `runs/qwen35-thinking-games-v2/{off,on}/` and `{off,on}.tar`
- Runtime log: `runs/qwen35-thinking-games-v2/modal.log`
- Final worker state: `runs/qwen35-thinking-games-v2/modal-apps-final.json`
- Modal app: https://modal.com/apps/duper/staging/ap-bmrFp0a91nXeKD73uAi4fz
- Reproduce aggregation: `.venv/bin/python scripts/summarize_thinking_games.py runs/qwen35-thinking-games-v2`

Aggregation verified all expected map/seat/opponent assignments, matching model/tokenizer/runtime identity, no adapters, and agreement between trace and inference token totals. The parser/gameplay/harness test selection passed all 10 tests.

An earlier infrastructure-only attempt is retained at `runs/qwen35-thinking-games-v1`: both workers failed before generating any model actions because the default FlashInfer prefill path required an unavailable CUDA compiler. The successful v2 run changed both arms to the supported Triton backend. Those failures are excluded from gameplay scores; their combined worker occupancy was approximately 4.32 H100 worker-minutes, in addition to the successful attempt above.
