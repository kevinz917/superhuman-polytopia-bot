# Split local/proxy and Daytona/OpenRouter execution

One benchmark, one continuous 40-minute authoring session per attempt, one final 360-game held-out suite. The execution route is selected before authoring and recorded in the receipts. A failed local proxy never silently becomes a paid OpenRouter benchmark run.

The existing twelve-model roster in `configs/skirmish/local-v1-top12.json` supplies only model IDs and provider parameters; its historical tracks/replicate/map settings are not used. Current proxy-catalog matches are Astra, Sol, Luna, Opus and Fable. The seven other roster models (Gemini, Kimi, Qwen, GLM, DeepSeek Pro, DeepSeek Flash and MiniMax) use OpenRouter with Daytona. No additional models are added. Five attempts/model is the prepared campaign default and can be changed explicitly.

## Execution and isolation

The local launcher permits 32 simultaneous authoring sessions by default (25 for five models × five attempts). API waits overlap. Compilation and public practice use six shared CPU slots; graders use those same slots and at most four simultaneous grader containers, retaining capacity for coding tools. This bounds CPU/memory pressure while allowing many model sessions. Local CPU queue waits are recorded and excluded from the 2,400-second active budget. The six-hour outer agent guard accommodates large tool queues; the 12-hour environment-start guard accommodates queued local graders before their separate two-hour execution guard begins. Neither changes the coding allowance.

Daytona runs up to 64 independent trials (35 for this roster/default), with 2 vCPUs/2 GiB per coder and 2 vCPUs/4 GiB per grader. Sandbox startup is admitted four at a time through agent setup; all 35 prepared cloud sessions may then remain active together. Remote command submissions have eight concurrent HTTP acknowledgements and a 120-second transport allowance; the original remote command deadline and independent coding watchdog remain unchanged. Long-lived authoring controllers use a dedicated thread pool so they cannot starve SDK file transfers. Separate private snapshots enforce the grader boundary; source and artifact transfer is native Harbor. The host Mini-SWE worker reaches only its Harbor-allocated environment through a private host Unix socket. It is never mounted inside the coder. API keys and Daytona credentials remain on the host. Sandboxes have blocked outbound networking and a three-hour absolute lifetime in addition to normal Harbor cleanup.

Claude proxy models use the native Anthropic Messages wire format while retaining Mini-SWE's existing bash action parser, prompts, observations and agent loop. Signed thinking/redacted blocks are preserved in subsequent tool turns and in saved receipts. High effort uses adaptive thinking with `output_config.effort=high`, and the output ceiling remains 32,768 tokens. Other proxy models and OpenRouter use their compatible chat wire format. See [Anthropic effort documentation](https://platform.claude.com/docs/en/build-with-claude/effort).

Candidate access to controller/verifier log mounts is disabled; the host retains ownership. Native Docker cleanup removes per-trial containers and volumes while retaining immutable package images.

Both packages share exactly the same hidden layouts, opponent hashes and runtime contract. The verifier rebuilds captured source independently and audits each final game. Finished final traces are losslessly gzip-compressed after auditing to fit remote disk limits. Export validation hashes decompressed bytes against the original trace digest. Rewards are written after durable grading receipts; infrastructure errors remain errors.

## Setup and validation

Install `.[skirmish-harbor,skirmish-daytona]` or use the checked-in `harbor/requirements.lock`. Existing local CLI proxy and OpenRouter key must be available on the host. Daytona credentials are loaded from the existing CLI profile or explicitly provided host environment variables. Do not place credentials in task/job JSON or images.

```sh
colima start --profile polytopia-skirmish
PYTHONPATH=src artifacts/harbor-runtime/bin/python scripts/build_skirmish_harbor.py \
  --package artifacts/harbor-packages/UNIQUE_LOCAL
PYTHONPATH=src artifacts/harbor-runtime/bin/python scripts/build_skirmish_daytona.py \
  --package artifacts/harbor-packages/UNIQUE_DAYTONA \
  --shared-package artifacts/harbor-packages/UNIQUE_LOCAL
```

The Daytona builder uses explicit public/private archives and self-contained Dockerfiles; private registry snapshot references may not be pullable as Dockerfile base images. Snapshot names are source/suite-versioned. Source drift requires a new package, not rewriting old receipts.

Run actual native Harbor integration fixtures without model calls:

```sh
PYTHONPATH=src artifacts/harbor-runtime/bin/python scripts/check_skirmish_harbor.py \
  --package artifacts/harbor-packages/UNIQUE_LOCAL --name UNIQUE_LOCAL_FIXTURE
PYTHONPATH=src artifacts/harbor-runtime/bin/python scripts/check_skirmish_harbor.py \
  --package artifacts/harbor-packages/UNIQUE_DAYTONA --name UNIQUE_DAYTONA_FIXTURE --backend daytona
```

The valid fixture compiles, plays practice, inspects a trace, submits source and receives independently audited final rewards. The broken fixture confirms that current broken source receives zero and is not silently replaced by a prior working source. Actual model tool-continuation probes are separate; they do not count as benchmark attempts.

## Prepare or run the campaign

```sh
PYTHONPATH=src artifacts/harbor-runtime/bin/python scripts/run_skirmish_split.py \
  --name UNIQUE_CAMPAIGN --directory artifacts/harbor-campaigns/UNIQUE_CAMPAIGN \
  --local-package artifacts/harbor-packages/UNIQUE_LOCAL \
  --daytona-package artifacts/harbor-packages/UNIQUE_DAYTONA \
  --attempts 5 --local-concurrency 32 --daytona-concurrency 64 --probe
```

This writes validated native `local.json` and `daytona.json`, explicit `routes.json`, actual model transport receipts and `status.json`. Add `--run` to a new uniquely named invocation to launch both complete jobs concurrently. Preparation alone does not start 40-minute authoring attempts. `--demo` keeps the 40-minute budget but uses 36 final games and cannot establish a ranking.

For the standard twelve-model/five-attempt plan: 25 local attempts, 35 cloud attempts, and 21,600 final outcomes. Daytona base compute is roughly $8–10 for its 35 attempts at the currently published v0 rates; reserve $20–25 of credits for setup/retries. Historical seven-provider inference was about $41 for five attempts each; a $75–100 API allowance is provisional rather than a cap. Proxy subscriptions/quota, local electricity and retained artifacts are additional. See [cost basis](COST_ESTIMATE.md) and [Daytona account setup](DAYTONA.md).

The cloud side can complete in roughly 1.5–2 hours if capacity and provider limits permit. Local final grading remains bounded by the existing machine; highly parallel model inference does not make its 9,000 final games free of CPU work. No new account limits, subscriptions, automatic top-ups or paid fallback policies are enabled by this setup.

The [2026-10-31 verification receipt](results/split-setup-20261031/README.md) records 25 local plus 35 cloud controller sessions passing concurrently, two actual-model end-to-end smokes, twelve live transport probes, and 89 passing unit checks. The preserved setup is ready; the full benchmark campaign has not been started.
