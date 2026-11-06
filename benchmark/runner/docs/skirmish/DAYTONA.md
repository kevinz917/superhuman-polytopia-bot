# Daytona setup and run estimate, 2026-10-31

Daytona is a viable execution target for the continuous benchmark. The CLI is already authenticated to **Curiosity Research Inc.**, using Daytona v0 (`https://app.daytona.io/api`) and default region `us`. The installed CLI is 0.226.1; this setup check uses Python SDK 0.226.0 alongside Harbor 0.25.0. No account upgrade, credential creation, credit purchase or benchmark campaign was performed.

## Observed account resources

A live authenticated organization-usage query returned the US container quota of **250 vCPUs, 500 GiB RAM and 2,000 GiB disk**. Organization defaults allow up to 4 vCPUs, 8 GiB RAM and 10 GiB disk per sandbox. The organization is active, not suspended. Its existing `polytopia-deps-v1` snapshot is for a separate RL stack and is not a qualified benchmark image.

For 50 simultaneous attempts, coder containers need 100 vCPUs and 100 GiB RAM; grader containers need 100 vCPUs and 200 GiB RAM. Even a temporary overlap of all coders and graders fits those CPU/RAM quotas. With a 10-GiB reservation per sandbox, 100 overlapping containers use 1,000 GiB of disk quota. Quota fit does not guarantee available runner capacity or provider inference concurrency.

The [saved setup receipt](results/daytona-setup-20261031.json) records a disposable stock-snapshot platform probe: CLI/SDK authentication, remote execution, explicit root execution, upload/download integrity, empty credential environment names and a failed outbound TCP connection with platform network blocking enabled. The stock toolbox defaults to uid 1001 despite a root create request; trusted root commands require explicit privilege selection. A production coder image must ensure the unprivileged coder cannot use that privilege path. The probe sandbox was deleted and post-cleanup usage was checked. This is **not** verification of the benchmark's cutoff or complete Harbor execution.

The installed CLI does not expose a wallet-balance command, and its SDK does not expose the needed wallet endpoint. The remaining credit balance and expiration have **not** been verified. Do not infer free cash cost from these setup checks.

## Estimated credit consumption

Use the pricing for the installed **Daytona v0 account**, not the separate v2 alpha product: [v0 pricing](https://www.daytona.io/pricing), [v0 billing semantics](https://www.daytona.io/docs/billing/). Published list rates are $0.0504/vCPU-hour, $0.0162/GiB-hour RAM, and $0.000108/GiB-hour disk beyond the free allowance. Running sandboxes reserve and bill resources while waiting for inference as well as while executing tools.

For **10 models × 5 independent attempts = 50 attempts**, at 2 vCPUs per sandbox:

| Phase | Resources per attempt | Duration per attempt | Total compute credits |
|---|---|---:|---:|
| Authoring including practice | 2 vCPUs, 2 GiB RAM | Up to 40 minutes | $4.44 |
| Held-out grading and audit | 2 vCPUs, 4 GiB RAM | Estimated 47–62 minutes | $6.54–8.52 |
| **Total** | | | **$10.98–12.96** |

Allow **$20–25 of Daytona credits** for initial setup, source/toolchain image builds, controller overhead and retries. This is an estimate, not a measured cloud benchmark cost or spending cap. Images and persistent snapshots can add storage charges; export artifacts and delete temporary resources after verification. The rough linear base estimate is $0.22–0.26/attempt, so 12 models × 5 attempts would be $13.18–15.55 before overhead.

The grading durations are extrapolated from [two local demo trials](results/continuous-proxy-demo-20261031/runtime-measurements.json); A later 36-game x86_64 fixture took approximately five minutes to grade; full 360-game contention has not been measured. With all 50 attempts concurrent, plan roughly **1.5–2 hours after images are ready**, conditional on runner capacity and proxy/provider limits. No GPU is required. Model inference is separate: existing CLI-proxy subscription costs/quota are excluded, and non-proxy provider API calls remain payable outside Daytona credits.

## Implemented benchmark integration

The split launcher now uses Harbor's native Daytona environment with separate private x86_64 coder/verifier snapshots and the same hidden suite as the local package. Host-only inference and controller credentials stay outside the offline coder; trusted source cutoff, unprivileged tools, practice, trace inspection, independent compilation and replay audits are implemented. Root-owned log permissions protect controller receipts.

Both deterministic valid/broken-source Harbor fixtures and a real DeepSeek Flash/OpenRouter end-to-end smoke completed on Daytona. An eight-second deadline control also verified cancellation before a delayed source write. The corresponding local Luna/proxy smoke completed. These are integration checks, not model rankings or a complete standard-suite campaign. See the [split execution guide](SPLIT_EXECUTION.md) and [saved verification](results/split-setup-20261031/README.md) for the current qualification status, concurrency controls and native launch configs.

The current twelve-model campaign puts five proxy-supported models locally and seven models on Daytona/OpenRouter: 25 local and 35 cloud attempts at five attempts/model. The cloud fraction's base compute estimate is approximately $8–10, with $20–25 of Daytona credits reserved as planning headroom. No automatic top-ups or spending-cap changes are enabled.

## Reproduce the setup check

```sh
uv pip install --python artifacts/harbor-runtime/bin/python 'daytona==0.226.0'
artifacts/harbor-runtime/bin/python scripts/check_skirmish_daytona.py \
  --probe --output artifacts/daytona-setup.json
```

Without `--probe`, the script performs read-only organization/usage queries. With it, it creates and deletes one small private sandbox with network blocking and a 15-minute TTL. Credentials are loaded from explicit Daytona environment variables or the existing macOS CLI profile, never written to the report or sent into the sandbox. The SDK may need a refreshed CLI login after token expiry. No automatic provider fallback or full campaign launch is included in this check.
