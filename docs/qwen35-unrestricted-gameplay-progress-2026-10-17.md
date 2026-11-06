# Unrestricted gameplay: interim results

Historical snapshot: this v1 run was subsequently stopped at the user's request and replaced by `qwen35-thinking-games-unrestricted-v2`. See [current benchmark](qwen35-current-benchmark.md).

Checked 2026-10-17 09:55 PM PDT. Thinking remains active. This is not a completed comparison.

| Measure | Thinking off | Thinking on |
|---|---:|---:|
| Allocated games | 40 | 40 |
| Finished games | 40 | 0 |
| Native wins | 0 | Unobserved |
| Native losses | 29 | Unobserved |
| Invalid-action forfeits | 11 | 0 so far |
| Valid executed actions | 4779 | 318 so far |
| Completed model calls | 4790 | 318 so far |

Non-thinking completed in 907 seconds of gameplay (15.1 minutes). Its native losses comprised 26 eliminations and three model-selected resignations. All eleven forfeits were valid JSON with an action ID outside the current legal catalog; they were not JSON formatting errors. Median request time was 2.45 seconds under the concurrent workload. Every output ended naturally; none hit a token limit.

Thinking has run for 7.62 hours. All forty games remain active, currently spanning rounds 0–5. The longest currently generating response has produced 183,534 tokens. The longest completed response had 13,504 tokens. Completed outputs ended naturally; no length finishes have been recorded. Pending generation is excluded from completed-call averages.

Two persisted progress snapshots sixty seconds apart showed the longest response increase from 183,220 to 183,534 tokens. The GPU is still generating; the job is not simply dead. We have not inspected the content of unfinished generations, so these counts alone do not prove a repetition loop. Long inference is delaying the benchmark, and forty simultaneously active long contexts may also affect throughput; GPU cache-pressure metrics were not captured, so that remains a hypothesis.

The existing no-cap protocol remains unchanged. Neither context-incomplete nor unfinished games are assigned loss/draw labels. There is not yet an observed thinking win rate to compare with non-thinking.

The original desktop collector is no longer running. A separate CPU-only remote collector has been dispatched to await the existing GPU call, validate results, and save `summary.json` and `result.md` on the shared run volume. It launches no new GPU jobs. The read-only fetch script downloads these artifacts to this run's `live/` directory when available.

Artifacts: `runs/qwen35-thinking-games-unrestricted-v1/off-analysis.json`, `off/report.json`, and `live/on/games/progress.json`.
