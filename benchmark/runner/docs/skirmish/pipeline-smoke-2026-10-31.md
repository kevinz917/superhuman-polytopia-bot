# Polytopia benchmark v5 — mini-pipeline-smoke-20261031a

Smoke integration test; these are not leaderboard measurements.

Status: completed; smoke: True.

Protocol: polytopia-mini-v5.1.
mean points (win=1, draw=0.5, loss=0), all attempts, equal maps/seats/opponents; compile failure=0; unresolved blocks point estimate.
Evaluation revision: `referee-900-stream-v1`; referee guard 900 seconds. Guest decision/fuel/memory limits follow the authoring protocol.


Smoke intervals are integration output only; one build cannot estimate variation between builds.

Shared hidden map digest: `f317d8f92e7720158c28044193d8359bd190fa3246f7d278cfc7f27f831f55af`. All stages and tracks use the same layouts.

Scheduled results: 192; primary 180; peer 12.
Replay audits: {'passed': True, 'actual_games': 192, 'virtual_compile_failure_losses': 0, 'actions': 119175}.

## Fixed evaluation-only references

- reference-weak: mini-top10-20261030a-qwen38-oneshot-20m; digest `4439287019367bcf44445ce7d29e0e1bd05e82398ef06a2c8a910519ac017184`.
- reference-intermediate: mini-top10-20261030a-deepseek-flash-oneshot-20m; digest `ce3cf260ef9f95bbfe2920adb29940890535776333da69e00d9fc85940cf226e`.
- hero-1v1: released hero-1v1 expert; digest `e6f4e1977e7f9cdcfbbe17358d87e49d638d82150496a6d4099e0cd389b0934f`.

## feedback-3r-40m primary ranking

| Model | Mean points /100 | 95% bootstrap interval | All attempt scores | Submission quality |
|---|---:|---|---|---|
| openai/gpt-6-luna | 13.89 | 13.89–13.89 | 13.89 | {'custom_submission': 1} |
| deepseek/deepseek-v4.1-flash-1017 | 5.56 | 5.56–5.56 | 5.56 | {'custom_submission': 1} |

## independent-40m primary ranking

| Model | Mean points /100 | 95% bootstrap interval | All attempt scores | Submission quality |
|---|---:|---|---|---|
| openai/gpt-6-luna | 8.33 | 8.33–8.33 | 8.33 | {'unchanged_starter': 1} |
| deepseek/deepseek-v4.1-flash-1017 | 2.78 | 2.78–2.78 | 2.78 | {'custom_submission': 1} |

## Final-stage outcomes against each fixed reference

Each cell shows points per 100, then wins–losses–draws–unresolved across all builds.

| Model | Track | reference-weak | reference-intermediate | hero-1v1 |
|---|---|---|---|---|
| openai/gpt-6-luna | feedback-3r-40m | 33.33 (2–4–0–0) | 8.33 (0–5–1–0) | 0.00 (0–6–0–0) |
| deepseek/deepseek-v4.1-flash-1017 | feedback-3r-40m | 16.67 (1–5–0–0) | 0.00 (0–6–0–0) | 0.00 (0–6–0–0) |
| openai/gpt-6-luna | independent-40m | 16.67 (0–4–2–0) | 8.33 (0–5–1–0) | 0.00 (0–6–0–0) |
| deepseek/deepseek-v4.1-flash-1017 | independent-40m | 0.00 (0–6–0–0) | 8.33 (0–5–1–0) | 0.00 (0–6–0–0) |

## Revision curves on identical hidden tests

| Model | Build | Initial | Revision 1 | Revision 2 | Revision 3 | Final − initial |
|---|---:|---:|---:|---:|---:|---:|
| deepseek/deepseek-v4.1-flash-1017 | 1 | 8.33 | 11.11 | 5.56 | 5.56 | -2.78 |
| openai/gpt-6-luna | 1 | 2.78 | 2.78 | 13.89 | 13.89 | 11.11 |

## independent-40m secondary Elo

Pooled matched-replicate peer games plus final-stage fixed-reference games. Failed compilations stay zeroes in the primary score and have no invented Elo. Ratings depend on this roster; they are not comparable with archived v4 ratings.

| Entrant | Elo | W–L–D–U |
|---|---:|---|
| hero-1v1 | 0.00 | 12–0–0–0 |
| reference-intermediate | -205.48 | 10–0–2–0 |
| reference-weak | -205.48 | 10–0–2–0 |
| deepseek/deepseek-v4.1-flash-1017 | -541.95 | 2–17–5–0 |
| openai/gpt-6-luna | -578.13 | 0–17–7–0 |

## feedback-3r-40m secondary Elo

Pooled matched-replicate peer games plus final-stage fixed-reference games. Failed compilations stay zeroes in the primary score and have no invented Elo. Ratings depend on this roster; they are not comparable with archived v4 ratings.

| Entrant | Elo | W–L–D–U |
|---|---:|---|
| hero-1v1 | 0.00 | 12–0–0–0 |
| reference-intermediate | -127.89 | 11–0–1–0 |
| reference-weak | -388.04 | 9–3–0–0 |
| openai/gpt-6-luna | -521.68 | 4–16–4–0 |
| deepseek/deepseek-v4.1-flash-1017 | -606.06 | 2–19–3–0 |

## Authoring timing and reliability

| Run | Status | Quality | Coding minutes | API wait seconds | Tool seconds | Saved API USD |
|---|---|---|---:|---:|---:|---:|
| mini-pipeline-smoke-20261031a-deepseek-flash-r1-feedback-3r-40m | completed | custom_submission | 0.85 | 41.4 | 8.6 | $0.0045 |
| mini-pipeline-smoke-20261031a-deepseek-flash-r1-independent-40m | completed | custom_submission | 0.50 | 19.4 | 9.9 | $0.0022 |
| mini-pipeline-smoke-20261031a-luna-high-r1-feedback-3r-40m | completed | custom_submission | 2.00 | 110.4 | 9.1 | $0.0120 |
| mini-pipeline-smoke-20261031a-luna-high-r1-independent-40m | completed | unchanged_starter | 0.50 | 20.2 | 8.8 | $0.0001 |

Costs are optional completed-call receipts. Interrupted requests, compatibility probes, infrastructure exclusions and Modal compute can add charges. No dollar-based limit is enforced.

Sources, Wasm, complete model trajectories, private referee chains and immutable plans remain on Modal volume `polytopia-skirmish`, under `mini-v5/mini-pipeline-smoke-20261031a` and its referenced `mini-runs` directories.

This cohort uses 1 build(s) per model and track and 3 hidden layouts. It measures performance under the declared deadline and provider routing; it does not establish a universal model ordering or isolate a causal feedback effect.

## Recorded authoring API receipts by model

| Model | Scored builds | Excluded infrastructure | All saved authoring calls | Mean per scored pair |
|---|---:|---:|---:|---:|
| deepseek/deepseek-v4.1-flash-1017 | $0.0067 | $0.0000 | $0.0067 | $0.0067 |
| openai/gpt-6-luna | $0.0121 | $0.0000 | $0.0121 | $0.0121 |

saved completed-call receipts for all authoring attempts; probes, in-flight requests and Modal compute additional; no dollar gate.

Saved compatibility-probe API receipts: $0.0006, additional to the authoring table above.
