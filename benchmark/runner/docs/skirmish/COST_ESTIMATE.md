# Full continuous benchmark cost estimate, 2026-10-31

The main benchmark has one continuous 40-minute authoring session per independently built bot, followed by 360 held-out games (six opponents × three map families × ten layouts × both seats). Public practice, compilation and trace inspection fit inside the same authoring budget. Final compilation, game execution and replay auditing run locally without model calls. Do not multiply by historical authoring tracks.

## Current split execution update

The [split launcher](SPLIT_EXECUTION.md) now connects proxy-supported models to local Harbor environments and the other seven roster models to Daytona/OpenRouter. Five attempts/model is its prepared default: 25 local plus 35 cloud attempts. Both paths passed actual-model integration smokes; the full standard campaign has not run. Plan approximately $8–10 base Daytona compute for the cloud fraction, plus historical inference of about $41; $20–25 Daytona and $75–100 API allowances provide provisional headroom. These estimates are not enforced caps.

The local-only serial-throughput table below records the earlier two-trial planning assumption. It does not describe the current split scheduler, which overlaps 25 local model sessions, bounds local CPU work, and schedules 35 cloud trials. See the split guide for the current concurrency and qualification receipts.

## Incremental cash cost

Assume the local machine and existing CLI-proxy subscriptions are already paid for. Proxy-backed inference adds no separately budgeted API charge here, but consumes subscription quota; the receipts report unknown subscription dollars, not measured zero cost. There is no Modal/cloud-VM bill or GPU requirement for this pipeline. Local electricity, hardware amortization and storage are additional.

For a broader twelve-model comparison using the earlier roster, route its OpenAI/Anthropic models through the proxy. The other seven models have historical saved completed-call costs from three independent 40-minute builds each:

| Paid provider model | Mean historical USD / build |
|---|---:|
| Gemini Argon | $2.07 |
| Kimi K3.1 Pro | $1.22 |
| Qwen 3.8 Max | $1.10 |
| GLM 5.3 | $3.12 |
| DeepSeek Pro | $0.48 |
| DeepSeek Flash | $0.13 |
| MiniMax M3.1 Flash | $0.16 |
| **Seven-model pass** | **$8.28** |

The [cost basis](results/cost-basis-20261031.json) is extracted from saved run receipts, not current price quotes. Five independently authored bots per model imply about **$41.38 historical paid inference** for that seven-model subset. Budget **$75–100** for the broader twelve-model/five-build campaign as a provisional allowance, including room for retries and larger contexts. This is not a cap or a new continuous-protocol measurement: on-demand practice can change conversation length, cached-input accounting differs, and interrupted requests may be billed. The paid-provider path is now connected through the native Daytona adapter; routes are explicit and never silently fall back to paid APIs.

A proxy-only cohort has **$0 separately budgeted API/cloud charges**, excluding existing subscription fees/quota, local power and storage. If all desired models are available through the proxy, the seven-model paid-provider estimate above does not apply.

## Local runtime

[Measured native Harbor timestamps](results/continuous-proxy-demo-20261031/runtime-measurements.json) show 284.33 seconds (Astra) and 370.37 seconds (Opus) to grade 36 demo games each, including verifier work and audits. Scaling those grader phases by ten gives about 47–62 minutes for 360 games. Add up to 40 authoring minutes and small setup overhead: plan **90–105 minutes per complete build** on the tested local setup.

This is an extrapolation from two demo submissions, not a measured standard-suite run or a statistical confidence interval. Different bots/maps, contention and proxy limits can materially change it. Image creation is a one-time extra step not measured here. Large retry/backoff delays or quota resets can extend scheduling; time spent waiting for inference inside an attempt consumes its active budget.

At the tested two concurrent trials:

| Models × independent builds | Total authored bots | Final games | Approximate elapsed time |
|---|---:|---:|---:|
| 2 × 1 | 2 | 720 | 1.5–1.75 hours |
| 2 × 5 | 10 | 3,600 | 7.5–8.75 hours |
| 4 × 5 | 20 | 7,200 | 15–17.5 hours |
| 7 × 5 | 35 | 12,600 | 27–32 hours |
| 12 × 5 | 60 | 21,600 | 45–53 hours |

General formulas: final games = models × builds × 360; conservative wall time ≈ ceil(models × builds / 2) × (1.5–1.75 hours), plus initial image builds and exceptional waits. Five builds/model is a planning scenario for measuring authoring variance, not an additional benchmark track or the launcher's default (one build).

Electricity can be estimated as wall hours × average machine watts / 1,000 × local price/kWh. For illustration only, 100 W average at $0.20/kWh adds about $0.90–1.06 over a 45–53-hour campaign; actual power and tariffs are not measured. Complete demo artifacts currently occupy about 1.9 GiB for two builds; retained traces and practice records require local disk space. No paid storage upload is assumed.

## What this Git snapshot saves

Source code, Harbor task files, configs, documentation, final demo bot sources, compilation/capture receipts and compact current/historical result summaries are versioned. Full conversations, match/practice traces, VM/Docker images, checkpoints and private reference bundles remain ignored local artifacts. This Git push is not a backup of those large/private assets.
