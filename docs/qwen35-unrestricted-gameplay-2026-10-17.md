# Qwen3.5-4B unrestricted-output gameplay comparison

Status: **stopped at the user's request on October 17, 2026**, for a complete fresh restart. Both the old GPU app and its collector were verified stopped with zero tasks. Its artifacts remain intact. The replacement is `qwen35-thinking-games-unrestricted-v2`; see [current benchmark](qwen35-current-benchmark.md). The material below records the historical v1 protocol, not a running job or completed comparison.

## Allocation and controls

- **80 games total: 40 thinking on, 40 thinking off.** Twenty paired validation maps, seeds 100691–100710, with the model playing both seats on each map. The first two maps were in the earlier pilot; eighteen maps are additional.
- Unadapted `Qwen/Qwen3.5-4B`, pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. No training, adapters or RL.
- Current `public-action-facts-v1` harness and prompt, unchanged. Same deterministic balanced scripted opponent. One model call per action, no repair calls or fallback actions.
- Same sampling in both arms: temperature 1.0, top-p .95, top-k 20, presence penalty 1.5, repetition penalty 1.0. The seed is derived from the paired job and decision number, independent of concurrent execution order.
- Same H100 hardware and pinned vLLM/runtime versions. Two GPU workers, one per mode, each admitting up to 40 concurrent requests.

## No benchmark token allowance

Both arms explicitly pass `max_tokens=None` and `thinking_token_budget=None`, allowing generation to stop naturally at EOS. No custom stop string, forced thinking closure, repetition cutoff or prompt truncation is used. Omitting `max_tokens` would be incorrect because its vLLM default is finite.

The model's [native configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json) has a 262,144-token context window, covering prompt plus generated response. That is the engine's configured context, rather than the former 32,768-token serving context. vLLM's [input processor](https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/engine/input_processor.py#L244-L249) interprets `max_tokens=None` as the available native context remaining after the prompt. This finite model boundary cannot be removed without changing the model/context configuration.

The former 256-model-decision and 30-minute gameplay cutoffs are removed. The frozen simulator retains its existing round-100/action-10,000 research horizon; reaching it is a truncation, not a draw. Modal has a 24-hour infrastructure timeout; that is not a benchmark game-loss condition.

## Execution and scoring

Games advance independently through AsyncLLM. A long response in one game does not hold completed actions in other games until an entire synchronous batch finishes. The model-visible harness is unchanged; this changes only inference scheduling.

The runner streams token counts for progress, saves each response and executed transition, and persists a private environment snapshot after each action. It distinguishes native wins/losses, invalid-output forfeits, simulator truncations, native-context exhaustion and infrastructure errors. Naturally ending without a valid final action is an invalid-output forfeit, not a token-budget failure. No action from unfinished reasoning is executed.

The comparison reports gameplay outcomes, valid-action rate, token use and actual per-request latency. Paired-map bootstrap uncertainty is available when all paired outcomes are resolved; the two seats of a map are treated as one cluster. One sampling sequence and twenty maps still constrain generalization.

## Tracking

- Run: `runs/qwen35-thinking-games-unrestricted-v1`
- Frozen protocol: `runs/qwen35-thinking-games-unrestricted-v1/protocol.json`
- App: https://modal.com/apps/duper/staging/ap-9WKXDZN7sKDnuIsSZbYpf7
- Remote call IDs: `runs/qwen35-thinking-games-unrestricted-v1/calls.json`
- Local log: `runs/qwen35-thinking-games-unrestricted-v1/modal.log`
- Read persisted progress without launching jobs:
  `/Users/K.Z./.local/share/uv/tools/modal/bin/python scripts/fetch_thinking_games.py runs/qwen35-thinking-games-unrestricted-v1`
- After both archives are downloaded:
  `.venv/bin/python scripts/summarize_unrestricted_games.py runs/qwen35-thinking-games-unrestricted-v1`
- The initial local collector/finalizer later exited while the GPU continued. A CPU-only remote collector now waits on the existing calls: https://modal.com/apps/duper/staging/ap-72vrXh8Ie1FOn4AHiztd8E. It writes `collection.json`, `summary.json`, and `result.md` on the run volume, with no additional GPU jobs. The fetch script copies these to the local `live/` directory. Its call receipt is `collector-call.json`.
- Interim results are recorded in `docs/qwen35-unrestricted-gameplay-progress-2026-10-17.md`; they are not a completed strength comparison.

The worker verifies frozen simulator source hashes, benchmark source hashes, and each board hash before loading the model. Fourteen focused tests passed, covering existing harness/parser behavior, independent progress across slow/fast games, a response exceeding 4,096 tokens, explicit disabled sampling budgets, deterministic per-job seeds, and distinct native-context versus invalid-output outcomes.
