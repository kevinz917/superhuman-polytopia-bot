# Qwen3.5-4B thinking: runaway-generation diagnosis and repair

The stalled run was generating a numeric repetition loop, not useful 218k-token game analysis. The original comparison is incomplete and must not be used to rank thinking versus non-thinking gameplay.

## Evidence captured from the running process

The old runner persisted token counts but saved response text only after EOS. A diagnostic snapshot extracted the actual token lists and text from all 40 pending generation coroutines before the run was stopped. The artifacts are under `runs/qwen35-thinking-debug-v1/captured/`.

- 28 of the 40 pending requests had an exact two-token repeating suffix: token IDs 4838 and 16, decoding to the continuing `-1,-1,-1,...` sequence.
- The snapshot contained 3,464,284 pending output tokens; 3,365,794 (97.2%) were in those exact repeated suffixes.
- The longest response had 218,265 tokens and no closing `</think>` delimiter. Its text starts with game analysis, repeatedly revisits a chosen action, then attempts to transcribe the dense terrain array and continues unknown-tile values indefinitely.
- The other 12 pending requests were not all looping: some had ordinary reasoning prefixes and one had not generated a token. The simultaneous long contexts also consume serving capacity, but a scheduling diagnosis alone cannot explain the repeated text.
- All 274 previously completed thinking responses stopped naturally and produced valid actions; the longest was 10,245 tokens. Several confused unit `home` with `tile`, treated neutral ownership as an enemy, or doubted legal harvesting because no unit occupied the resource. These are additional prompt-comprehension problems, not evidence that the game engine rejected those actions.

The native thinking toggle, EOS handling, and final-action delimiter worked on completed responses. The original protocol already used temperature 1.0, top-p .95, top-k 20, presence penalty 1.5, and repetition penalty 1.0. This matches the model card's general thinking recommendation; simply adding the already-present presence penalty is not a fix. The model card also notes that presence penalties can reduce endless repetitions, not guarantee their absence. See [Qwen's model card](https://huggingface.co/Qwen/Qwen3.5-4B#best-practices).

The captured text supports dense-array transcription as the trigger in inspected cases. It does not prove that all long reasoning on all inputs shares that cause. Presence penalty is a one-time penalty per seen token and does not keep growing as a two-token loop repeats.

## Repair and diagnostic design

1. Encode terrain as `size`, `default: -1`, and explicit known `[tile, terrain]` pairs. This preserves every public field and the complete action catalog while removing long unknown-value runs. All 40 captured inputs pass exact round-trip reconstruction.
2. Test that transport alone, and separately with concise decision instructions and explanations of action-versus-turn, `home` versus `tile`, neutral ownership, and harvesting. Both thinking modes will receive the same selected prompt.
3. Persist unfinished text and tokens every 512 generated tokens. Future runaway outputs can be inspected immediately rather than after EOS.
4. Detect exact periodic suffixes. The diagnostic aborts after a 256-token exact loop; gameplay uses a more conservative 1,024-token suffix (period 1–16). A failed generation executes no action and is recorded as `generation_incomplete`, never a native loss or draw. There is no forced thinking closure, fabricated answer, retry, or teacher fallback.

The probe uses eight fixed captured positions: the four longest pending generations and the four shortest. It evaluates the original prompt, sparse terrain alone, and sparse terrain plus concise instructions, with the same checkpoint, tokenizer, sampling parameters, and per-position seed. Concurrency is eight, so this is not an exact replay of the original concurrency-40 run. Its 16,384-token diagnostic ceiling prevents another unbounded debug job; this ceiling does not apply to the fresh gameplay comparison.

The historical GPU run and its coordinator were stopped and verified at zero tasks. Its artifacts and off-mode results remain preserved; no unfinished on-mode games are converted to losses. No training or RL was launched.

## Status and artifacts

The controlled probe completed. Original prompts produced 7/8 valid natural stops and one exact-repetition abort. Sparse terrain and sparse terrain plus brief instructions each produced 8/8 valid natural stops, with no diagnostic-cap finishes. The selected sparse-plus-brief variant averaged 5,273 output tokens and 129.6 seconds per request at concurrency eight. This small completion check does not establish a normal-response speedup or game strength; the original variant's shorter mean includes its aborted response. The 256-token diagnostic guard can catch finite copying, so that particular probe abort is not itself proof of an endless loop.

Fresh paired gameplay is running on Modal as `qwen35-thinking-games-repaired-v1`: 40 games per mode, the same sparse-plus-brief prompt in both modes, identical weights/maps/sampling, no output/thinking token cap, and a 1,024-token exact-repetition guard. Both H100 workers loaded and started games. A durable cloud coordinator collects and summarizes results. This is a development comparison; probe positions overlap development maps. No winner is established yet.

Live app: https://modal.com/apps/duper/staging/ap-YbeIuJq6q5OTtyEQvj0yuA

- [Captured-output analysis](../runs/qwen35-thinking-debug-v1/diagnosis.json)
- [Exact transport round trips](../runs/qwen35-thinking-debug-v1/transport-validation.json)
- [Diagnostic protocol](../runs/qwen35-thinking-debug-v1/protocol.json)
- [Old run stop verification](../runs/qwen35-thinking-debug-v1/stop-receipt.json)
- [Probe dispatch](../runs/qwen35-thinking-debug-v1/probe-dispatch.json)
- [Probe results](../runs/qwen35-thinking-debug-v1/probe-summary.json)
- [Fresh comparison protocol](../runs/qwen35-thinking-games-repaired-v1/protocol.json)
- [Fresh comparison progress](../runs/qwen35-thinking-games-repaired-v1/latest-progress.json)

Local validation: 14 focused tests pass, covering checkpoint retention, local-execution guards, named-field transport, action schemas, and unchanged legal catalogs, exact public-state round trips, partial-thinking rejection, and failure accounting that executes no action and assigns no native outcome.

## Follow-up: remaining latency and invalid-action causes

The first repaired gameplay run reached 535 naturally completed thinking calls without an exact-repetition abort or context-limit finish. The longest completed response was 11,183 tokens. This is strong evidence that the observed 218k-token numeric echo was addressed, but the calls still averaged about 5.4k tokens. Inspected outputs repeatedly reopen decisions and confuse table columns and numeric meanings. Full games had not yet completed in the thinking arm at that snapshot.

The non-thinking arm completed 40 games: 25 native losses and 15 invalid-response forfeits. Eight forfeits copied the instruction's illustrative `a17` even though that ID was absent from the current catalog. Four other failures used nonexistent IDs, and three emitted prose. These are distinct from the numeric-loop failure. The simulator's `home` field is a city tile index plus one (zero means independent); the earlier prompt described it only as a reference.

A controlled cloud H100 serving test used eight identical prompts, a warm-up batch, and then exactly 512 generated tokens per request. Eager execution took 14.849 seconds (275.85 aggregate tokens/s); compiled CUDA graphs took 2.943 seconds (1,391.75 aggregate tokens/s), a 5.05x improvement. The generated length is fixed only for this speed diagnostic, never for gameplay. This is not yet a measured full-game speedup. Both configurations use the same model revision, dtype, native context, attention backend and sampling settings. Receipts: `runs/qwen35-thinking-serving-v2/speed-{eager,graphs}.json`. CUDA graph capture size is limited to the configured concurrency to avoid oversizing the hybrid model's state cache.

The next candidate replaces entity/rule/action positional rows with named records, removes the misleading illustrative action ID, and explains the home encoding. All sixteen captured diagnostic positions round-trip exactly, including optional fields, nulls, rules and every legal action. A reasoning-aware final-answer grammar permits exactly the current action IDs; it does not constrain the thinking segment or rank actions. Both modes receive the same grammar. This grammar and the final prompt require live cloud validation before promotion.

The first serving diagnostic failed at module import and was stopped. Its replacement completed the speed comparison; its obsolete prompt phase was stopped and is not used as validation. The corrected prompt/grammar diagnostic is `qwen35-thinking-serving-v3`. No checkpoint inference or downloads run on the Mac. The separate structured-checkpoint cloud service has completed native-outcome development matches successfully.


### Validated follow-up and expanded comparison

The corrected `qwen35-thinking-serving-v3` diagnostic completed 24/24 valid natural stops: eight captured non-thinking failures now produce legal JSON (mean 8.125 tokens, 0.687 s), eight sparse thinking controls (mean 7,768.5 tokens, 47.871 s), and eight named-field thinking cases (mean 5,965.375 tokens, 36.734 s). All sixteen thinking calls closed naturally; none hit the diagnostic limit or repetition guard. The named-field variant reduced mean generated tokens by 23.2% on this diagnostic set. Thinking remains multi-thousand-token model behavior; the repairs do not claim to make reasoning free or guarantee good strategy.

At the user's request, the new run is `qwen35-thinking-100paired-v1`: **100 games per mode on the exact same 50 maps, both seats**. All boards match the reserved validation manifest and are distinct under its symmetry hash. The earlier 40-game configurations were stopped; outcomes are not mixed across configurations. Both modes share the same prompt, grammar, compiled engine, seeds and opponent. The grammar does not constrain thinking. Output and thinking budgets remain unset; the exact-repetition guard and native context limit remain documented. Uncertainty is clustered by map, not by treating both seats as independent maps. The first 20 maps overlap prior development diagnostics; the remaining 30 were added for this comparison, so the full result is still a development benchmark.

The detached cloud coordinator now outlives its GPU worker timeout, cancels sibling workers if a worker fails, and labels unresolved outcomes explicitly. The local launcher exits after dispatch; no local inference or checkpoint download watcher is required. Final gameplay results are pending.
