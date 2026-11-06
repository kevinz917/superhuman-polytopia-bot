# Current Qwen3.5 gameplay benchmark

**Final status (October 18, 18:47 PDT): stopped by the user.** See the [final three-arm comparison](qwen35-thinking-comparison-final-2026-10-18.md).

| | Wins | Losses | Failures | Unfinished |
|---|---:|---:|---:|---:|
| Thinking on | 2 | 16 | 1 | 11 |
| Thinking off | 0 | 30 | 0 | 0 |
| Qwen3-4B-Instruct-2507 | 0 | 30 | 0 | 0 |

The thinking arm expanded about 3× as fast and outlasted thinking off in 23 of 30 games. It cost about 74.8M thinking tokens and about 17.9 H100-hours. Its workers hit Modal's 4-hour limit, so unfinished games were resumed from checkpoints in two continuation segments (identical runtime, same seeds, exactness tests passed on Modal). No Modal app for this benchmark is still running. Everything below is history.

**October 18 run (history):** `qwen35-thinking-30paired-v1`, reduced at the user’s request to **30 games per mode on the same 15 maps**, seeds 100691–100705, both seats. Maps selected in ascending seed order, independent of outcomes. Two Modal H100/vLLM replicas per mode run 16 and 14 games. Thinking remains uncapped; all inference and simulations remain in Modal.

[Live Modal app](https://modal.com/apps/duper/staging/ap-yetFSdf60hrtaYAVLUtb57). Protocol: `runs/qwen35-thinking-30paired-v1/protocol.json`. The previous 100-game worker allocation could not be changed live, so it was stopped and all results preserved separately. This is a fresh paired run; no old outcomes are reused. All model, sampling, prompt, grammar and seed settings are retained. Cloud checks passed before dispatch. Results pending; this chat is monitoring directly, with no recurring automation.

At 10:17 Pacific the CPU collector was recovered after its container restarted and rejected the existing dispatch. Recovery reattached to the exact four original worker call IDs; GPU games continued without restart. The current collector is [here](https://modal.com/apps/duper/staging/ap-8XK8G8psQqD7n0ggHUN5G0). Non-thinking completed 30/30 native losses; thinking remains in progress.

**Superseded 100-game run:** `qwen35-thinking-100paired-v2` compares 100 fresh thinking-on games with 100 fresh thinking-off games on exactly the same 50 reserved validation maps (seeds 100691–100740), both seats. Each mode now uses **two Modal H100/vLLM replicas**, 50 concurrent games per replica, scheduler capacity 64 and full CUDA graph coverage. Four GPU workers total. Weights, prompt, grammar, sampling, opponent and per-decision seeds match the prior run. Thinking has no output-token cap; exact repetitive-generation failures remain separate from native losses.

[Live Modal app](https://modal.com/apps/duper/staging/ap-zrY0e3Y5FD4skmIopv7Ila). Protocol: `runs/qwen35-thinking-100paired-v2/protocol.json`. Pairing/test gate: `runs/qwen35-thinking-100paired-v2/validation-gate.json`. Active pointer: `runs/qwen35-thinking-active.json`. All four model workers passed startup verification (same pinned weights, capacity 64 and graph 64, no output/thinking cap). The cloud coordinator owns progress and final aggregation; results are pending.

**Independent diagnosis completed:** the previous median thinking request spent 118 of 196 seconds before its first token. Larger graph-compatible batches improved a short fixed-output probe, but sustained long responses favored splitting games between replicas. Additional conciseness instructions did not help; an alternate low-temperature recipe produced prose loops and was rejected. The loop guard now catches demonstrated 31/51/96-token paragraph periods that its old 16-token limit missed. See [measurements, sources and repair](qwen35-independent-diagnosis-2026-10-18.md).

The recurring automation was **deleted** at the user's request. No replacement automation exists. All model inference, simulations and checkpoint handling remain in Modal.

The superseded `qwen35-thinking-100paired-v1` app is stopped. Its preserved pre-stop snapshot has 72 non-thinking native losses and 0 completed thinking games. Unfinished games are not losses or draws. No partial outcomes are mixed into v2. Receipt: `runs/qwen35-thinking-100paired-v1/serving-migration-receipt.json`.

The earlier numeric-echo diagnosis and original CUDA-graph repair are documented in [the earlier investigation](qwen35-thinking-debug-2026-10-18.md). The material below is preserved history, not current status.

Run: **`qwen35-thinking-games-unrestricted-v2`**. Dispatched as a complete restart on October 17, 2026.

Startup verified: both models loaded successfully, both modes began fresh games, and both report disabled output/thinking token budgets. Evidence is saved in `runs/qwen35-thinking-games-unrestricted-v2/startup-verification.json`. The active-run pointer is `runs/qwen35-thinking-active.json`.

Interim milestone: the non-thinking arm completed all 40 games in 955.49 seconds of gameplay: zero wins, 23 elimination losses, five resignations, and twelve invalid-action forfeits. Trace validation confirmed all forfeits were unknown action IDs, with 4,849 valid actions out of 4,861 calls. Evidence: `runs/qwen35-thinking-games-unrestricted-v2/off-analysis.json`. Thinking is still running; this is not yet a completed paired comparison.

Thinking milestone recorded October 18 at 01:02 Pacific: after 10,851.9 seconds of gameplay, all 40 games remained active, with 274 valid actions and rounds ranging from zero to three. The longest pending response had 102,200 tokens; pending responses totaled 2,161,199 tokens, excluding completed responses. These are interim counts, not final outcomes. Evidence: the timestamped `runs/qwen35-thinking-games-unrestricted-v2/monitor.jsonl`.

A separate 45-second per-request check around minute 137 found that 22 of 40 pending responses gained tokens and 18 did not. This suggests scheduling contributes to wall-clock latency, but does not identify its cause or establish that unfinished text is repetitive. Evidence: `runs/qwen35-thinking-games-unrestricted-v2/on-request-progress-137min.json`. No inference settings were changed. Final latency results must be described as measurements under this concurrent workload, rather than isolated-request latency.

App: https://modal.com/apps/duper/staging/ap-wQXmCGWOZORYVfcZouQ9AW

- Both modes start from initial game states: 40 thinking-on games and 40 thinking-off games, twenty identical validation maps with both seats.
- Same unadapted Qwen3.5-4B revision, current harness, opponent, sampling and per-game seeds as v1. No adapters or training. No old game state, generated response, or outcome is reused.
- `max_tokens=None`, `thinking_token_budget=None`. No benchmark model-decision or gameplay-time cap. The model's native context boundary, frozen simulator research horizon and hosting platform timeout remain as documented in the protocol.
- A remote CPU coordinator owns both H100 workers and the final analysis. It remains active independently of the local desktop collector.
- The old v1 GPU app and separate collector were stopped and verified at zero tasks before dispatch. Existing artifacts are preserved separately.

Protocol: `runs/qwen35-thinking-games-unrestricted-v2/protocol.json`.

Equivalence checks: `runs/qwen35-thinking-games-unrestricted-v2/restart-equivalence.json`.

Old stop receipt: `runs/qwen35-thinking-games-unrestricted-v1/stop-receipt.json`.

Refresh progress and fetch final report when available:

```sh
/Users/K.Z./.local/share/uv/tools/modal/bin/python scripts/fetch_thinking_games.py runs/qwen35-thinking-games-unrestricted-v2
```

Remote final artifacts will be `summary.json`, `result.md`, and `collection.json` under this run ID. The fetch script downloads these into the local run's `live/` directory. A completed result must not be inferred from dispatch or from either arm completing alone.
