# Polytopia Skirmish on Harbor

The main benchmark is one continuous 40-minute coding-agent attempt. The agent chooses when to compile, play public practice matches, inspect traces, and revise. A fresh verifier independently rebuilds the current source captured at submission/cutoff and grades it against a fixed six-opponent panel on held-out maps. There are no forced feedback cycles or public authoring tracks.

## Local setup

Requires Docker/Compose and the existing Colima profile. Controllers and the inference proxy run on the host; coder/verifier containers run offline in the local VM. No Modal execution is used. The original local launcher has no paid API fallback; the split launcher explicitly selects Daytona/OpenRouter for models outside the local proxy catalog.

```sh
uv venv artifacts/harbor-runtime --python 3.12
uv pip install --python artifacts/harbor-runtime/bin/python -r harbor/requirements.lock
colima start --profile polytopia-skirmish
```

Provide the existing CLIProxyAPI client key through its standard local key file or `CLIPROXY_API_KEY`; use `CLIPROXY_BASE_URL` to override the default localhost endpoint. Credentials never enter the coding container. The frozen reference bundle must be present in `artifacts/skirmish-references`; its required hashes are in `configs/skirmish/continuous-v1.json`. The builder validates it before creating images.

## Build a versioned local task package

```sh
PYTHONPATH=src artifacts/harbor-runtime/bin/python scripts/build_skirmish_harbor.py \
  --package artifacts/harbor-packages/MY_UNIQUE_PACKAGE
```

The checked-in `tasks/polytopia-skirmish` is the standard Harbor source format: `task.toml`, `instruction.md`, agent environment and separate verifier definition. The builder bundles the original Rust toolchain, public practice SDK, trusted evaluator and digest-checked references into separate images. It generates a shared held-out suite once, and writes a ready-to-run task with immutable image IDs, manifest/map hashes and a source archive. Never reuse a package/job name for a modified experiment.

## Run through Harbor

```sh
PYTHONPATH=src artifacts/harbor-runtime/bin/python scripts/run_skirmish_harbor.py \
  --package artifacts/harbor-packages/MY_UNIQUE_PACKAGE --name MY_UNIQUE_RUN \
  --model openai/gpt-6-astra --model anthropic/claude-opus-5.5
```

The launcher writes a native Harbor job config and invokes `harbor run --config ...`, using `polytopia_bot.skirmish.harbor_agent:SkirmishAgent` and local Docker. It does not emulate Harbor results. Standard final evaluation has 360 matches per bot. `--demo` keeps the same 2,400-second coding allowance and uses 36 final matches per bot; it is an explicitly labeled integration check, not a second authoring benchmark or a model ranking. `--attempts N` repeats complete independently authored bots, not individual evaluation games.

The official external agent preserves the existing Mini-SWE-agent 2.4.6 parsing/loop, host CLI-proxy transport, atomic conversation receipts, deadline cancellation, scheduler accounting, process cleanup and root-owned source capture. It uses the task's single instruction in one continuous conversation. Harbor's 21,600-second agent timeout is only an outer infrastructure guard; the controller independently enforces 2,400 active seconds. Alternate Harbor agents need an adapter preserving this clock and trusted cutoff contract before their results are comparable.

## Results and integrity

Harbor results live under `artifacts/harbor-runs/RUN/`. Each trial retains its original conversation, timing/usage receipts, root-owned captured source, independent compilation receipt, evaluation plan, complete traces, replay audits and numeric `verifier/reward.json`.

The main `reward` is mean match points (win=1, draw=0.5, loss/guest fault/compilation failure=0); display it multiplied by 100. Champion win rate/points, runtime faults, compilation losses, evaluation count and demo status are separate metrics. Infrastructure errors produce an error rather than an invented model zero. Subscription dollar cost is unavailable, not zero. All scheduled final outcomes count. Repeated-build uncertainty must be reported before making general ranking claims; many games from one authored bot do not estimate build variance.

Cap sweeps, transfer/tactical qualification and peer tournaments remain separate diagnostics and do not delay ordinary final scoring. Historical v5/v6 cohorts retain their original protocols and scores. This new format is `polytopia-skirmish-continuous-v1`.

## Recorded validation and run budget

The [completed two-model integration demo](../docs/skirmish/results/continuous-proxy-demo-20261031/README.md) includes saved final sources and summary receipts. See the [full-run cost estimate](../docs/skirmish/COST_ESTIMATE.md) for measured demo timings, explicit full-suite extrapolations and proxy versus paid-provider assumptions.

## Split local / Daytona execution

The existing twelve-model roster can be prepared using `scripts/run_skirmish_split.py`: exact proxy-catalog matches use the local VM; other models use OpenRouter in Daytona. Both routes use the same 40-minute authoring contract, shared held-out suite and source pinning. Local model sessions run concurrently while compilation/practice and graders share six CPU slots, with at most four graders. The long outer timeout accommodates explicitly excluded local queue waits; it grants no additional active coding time. See [split execution](../docs/skirmish/SPLIT_EXECUTION.md) for setup, fixtures and launch commands.
