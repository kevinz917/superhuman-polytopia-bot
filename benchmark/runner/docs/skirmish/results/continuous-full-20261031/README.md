# Polytopia Skirmish: complete continuous benchmark

**12 models × five independently authored bots × 360 final games = 21,600 outcomes.** Each author had up to 40 active minutes, including public practice, compilation and trace inspection. All submissions were frozen before final judging.

Primary score is 100 × (wins + half of draws) / 360. The table reports the mean and sample SD across five authored bots. Every bot faced the same six-opponent, three-family, ten-layout, both-seat schedule.

| Model | Panel points /100, mean ± SD | Champion wins | Recorded API USD |
|---|---:|---:|---:|
| openai/gpt-6-astra | 74.94 ± 6.02 | 2.7% | Subscription / unavailable |
| openai/gpt-6.1-sol | 74.39 ± 2.92 | 3.0% | Subscription / unavailable |
| google/gemini-4-argon | 64.36 ± 5.80 | 2.7% | $12.05 |
| anthropic/claude-opus-5.5 | 43.42 ± 3.59 | 1.0% | Subscription / unavailable |
| anthropic/claude-fable-5-1 | 31.89 ± 5.26 | 0.3% | Subscription / unavailable |
| moonshotai/kimi-k3.1-pro | 31.06 ± 14.41 | 1.3% | $7.37 |
| z-ai/glm-5.3 | 30.53 ± 19.06 | 0.3% | $7.85 |
| qwen/qwen3.8-max-0902 | 26.56 ± 16.93 | 0.7% | $5.88 |
| openai/gpt-6-luna | 19.58 ± 14.66 | 0.7% | Subscription / unavailable |
| deepseek/deepseek-v4.1-flash-1017 | 18.36 ± 14.82 | 0.3% | $0.47 |
| deepseek/deepseek-v4-pro-0813 | 15.19 ± 12.88 | 0.3% | $1.76 |
| minimax/minimax-m3.1-flash | 13.08 ± 11.28 | 0.3% | $0.88 |

All selected outcomes passed the native replay audit. Original sources and independently compiled Wasm hashes were checked. Full exports were byte-verified; lossless replay exports were checked against the original private action-log hashes and chains, native audited outcomes, and preserved public-observation hashes. The replay decoder reproduced all original public/private hashes in a 36-game qualification.

One Gemini authoring attempt failed on an API response and was replaced by a fresh full attempt. Grading-only recoveries reused frozen source and available audited checkpoints, with no additional authoring or LLM calls. The first complete validated grading export was selected for each source; selection never used its reward. Raw infrastructure failures and recovery provenance remain in the result manifest.

Recorded successful completed-response OpenRouter charges: **$36.26**. The failed authoring attempt recorded $1.21. These exclude potentially billed interrupted calls and setup probes. Proxy subscription and settled Daytona dollars are unavailable, not measured zero.

Local ARM and cloud x86_64 execution used the same declared limits; hardware and concurrent endpoint latency remain practical comparability limits. The displayed SD covers five authored bots on this fixed panel.

[Complete compact results and provenance](results.json). Final Rust sources are under `sources/`, and immutable validation receipts are under `validation_receipts/`. Conversations, practice records and final replays for every submission are archived under [`benchmark/runs/`](../../../../../runs/); container images are not included.

Portable archive checks can be repeated from the repository root with `python3 docs/skirmish/results/continuous-full-20261031/recovery_sources/verify_final_benchmark.py`. These checks validate the archived receipts and calculations; they do not rerun the games. Native replay validation was performed before acceptance.
