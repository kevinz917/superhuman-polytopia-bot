# Recovering a frozen cohort

A model/API interruption is an infrastructure failure, not a loss against an opponent. The ordinary coordinator permits one fresh attempt after an infrastructure failure. It never retries a final compilation failure to improve the score.

Modal cancellation can leave a result wait without an output. A controller can also lose its return value after writing its artifacts. `scripts/finalize_skirmish_cohort.py` is an evaluation-only recovery path for these cases. It makes no model requests and preserves the original authoring configuration, image, source files, rules, guest move protocol and scoring functions. The final evaluation revision below changes only the referee guard and trace I/O.

First inspect the saved artifacts without changing anything:

```sh
MODAL_DISABLE_API_PROXY=1 modal run --env staging \
  scripts/finalize_skirmish_cohort.py::finish \
  --prefix COHORT_PREFIX --dry-run
```

The command validates every completed or failed-compilation attempt against the immutable authoring plan, checks all stage source and Wasm hashes, checks that cutoffs belong to the current stage clocks, and requires every slot to have a complete frozen attempt. Artifact-validation failures are recorded; they cannot become game scores. When more than one valid artifact survived a controller interruption, the earliest valid attempt by fixed attempt index is selected before scoring. Later duplicates are recorded as exclusions. A compile failure remains a valid zero-scoring submission.

If authoring is incomplete, it returns `needs_authoring` without drawing hidden maps, writing scores, launching models, or retiring apps. Provider failures require completing the affected authoring slots first. Do not substitute incomplete attempts or zeroes for a provider outage.

Before grading an official cohort, rebuild the frozen sources in fresh offline sandboxes:

```sh
MODAL_DISABLE_API_PROXY=1 modal run --env staging \
  scripts/rebuild_skirmish_sources.py::verify \
  --prefix COHORT_PREFIX --output rebuilds.json --all-attempts
```

Compiled outputs must match byte for byte; invalid submissions must remain invalid. The evaluator requires current proof receipts for every selected stage. A changed source hash triggers a new rebuild; earlier proofs remain in a history directory. These checks add no model opportunities.

Once all slots and rebuild proofs are ready, finish with the IDs of the authoring apps belonging to that cohort:

```sh
MODAL_DISABLE_API_PROXY=1 modal run --env staging --detach \
  scripts/finalize_skirmish_cohort.py::finish \
  --prefix COHORT_PREFIX --retire-apps ap-OLD_AUTHOR_APP,ap-RESUMED_AUTHOR_APP
```

Only after the readiness check does the command stop those authoring apps. An explicit CLI acknowledgment that an app is already stopped is accepted as a terminal receipt; other stop errors still abort the handoff. The evaluator cancels stranded calls, archives available unused workspaces, closes coder sandboxes, records its own source digest and artifact-selection receipt, and allocates the shared hidden maps. Evaluation uses the original engine transitions, fixed reference panel, aggregation and secondary Elo fit. Every action, public observation, final state and private hash chain is checked in a streaming replay audit. Existing digest-checked game results are reusable after another interruption. Evaluated cohorts cannot be overwritten.

Full sources and trajectories stay on the Modal volume. Export only compact status/results through the ordinary `status --output ... --watch` entrypoint. API costs remain optional receipts; this recovery path adds no dollar gate.

The read-only recovery check passed for all four attempts in `mini-v51-smoke-20261030a`; that smoke cohort's existing 192 game outcomes were unchanged. A read-only check is not an additional end-to-end evaluation run.

## Evaluation revision: `referee-900-stream-v1`

An additional 192-game recovery smoke test completed with all 110,144 actions replay-verified, but three games reached the 180-second referee guard. That report remains provisional and is retained under `mini-v51-recovery-smoke-20261030a`.

Before allocating the official cohort's hidden maps, the final evaluator was revised uniformly to use a 900-second referee guard. Guest limits remain 2 seconds per decision plus the original one-time startup allowance, 256 MiB memory, and one trillion fuel per game. Engine 100-round and 10,000-action caps remain announced draws. Referee timeouts remain unresolved and block a complete point estimate. The authoring environment and its feedback-match configuration retain their original v5.1 receipts.

The final evaluation plan and every job hash explicitly bind the new referee limit and evaluation revision. Existing 180-second plans cannot be overwritten by this evaluator. Reports display authoring schema and evaluation revision separately. Streaming file hashes and line-by-line replay preserve all existing integrity checks while avoiding whole-trace allocations; a smoke match produced a 339 MB public trace.

A negative fixture also verifies that a cutoff from an older invocation cannot be accepted under a newer stage clock. The fixture restores the valid clock and confirms readiness, without allocating hidden maps or making model requests.

## Additional infrastructure recovery

`scripts/retry_skirmish_authoring.py` allocates an explicit fresh attempt using the immutable original model configuration, coding image and budget. It refuses a slot with a valid frozen submission, including a compilation failure. It also refuses an attempt ID already started or owned by another active call. Failed launch allocations with no authoring directory can be archived and reclaimed without any model run.

A controller must be terminal before its slot can be recovered. An abandoned live sandbox is archived as an unqualified interruption and terminated; a valid returned artifact must be recovered instead of rerunning the model. Refreshing the volume before the original authoring-directory guard prevents a cloud input replay from silently overwriting the same attempt. Sources, prior status, controller state, call IDs and recovery-code digest are retained in the operator ledger.

The original coordinator allows one infrastructure retry. Additional operator recoveries are explicit, apply to any affected slot, and are logged before hidden maps or scores exist. They use fresh conversations and sandboxes; the model receives no earlier failed bot or hidden-game results. The October 30 cohort used an additional recovery at low API concurrency after repeated OpenRouter 402 responses and terminal controllers. This is error recovery, with no credit balance query or dollar threshold in the harness.

The batch entrypoint accepts comma-separated cell IDs and explicit attempt indices and runs at most two recovery authors per recovery app. Several operator apps may overlap; this is not a global two-call limit. It skips already allocated or completed slots. The original coding time and revision controls remain unchanged. These retries ran later and at lower concurrency, so API latency is recorded rather than assumed to be controlled hardware.

The evaluator's own source bytes and digest are frozen on the Modal volume when grading begins. Every evaluation job binds that digest, runtime revision and referee limit; resuming with different evaluator code is rejected.

## Stable CPU controllers

The long-running authoring controllers, main coordinator and final evaluator use `nonpreemptible=True`. Modal may replay preempted CPU function inputs even when ordinary exception retries are disabled. An observed October 31 controller received SIGTERM and was replayed while its independent coding sandbox survived; the replay's directory guard correctly refused to overwrite the original attempt. See [Modal's preemption documentation](https://modal.com/docs/guide/preemption) and [container crash semantics](https://modal.com/docs/guide/retries).

The stable-controller change does not alter the model's offline coding sandbox CPU/memory, guest limits, coding time, or revision opportunities. Previously completed artifacts remain eligible. Modal applies a compute-price multiplier to non-preemptible CPU functions; API receipts remain separate from Modal billing. Queued legacy call IDs are retained rather than overwritten when input cancellation cannot be confirmed through the graph API.

## Terminal reports and operator checks

A saved report blocks both dry-run readiness and actual finalization, even if a mutable status file is reset to `needs_attention`. The negative fixture in `terminal-guard-2026-10-31.json` preserves all four sentinel files and makes no model calls or games. The launcher separately rejects completed/provisional cohorts before probes.

App retirement uses explicit `--yes` for the cohort-owned IDs supplied by the operator. A live-worker test verified noninteractive stop, safe repeat acknowledgments and the already-stopped path; unrelated CLI failures remain errors. Compact evidence is in `retirement-guard-2026-10-31.json`.

The official October 30–31 cohort completed using the evaluator source retained in commit `9e2cb59` and on Modal at `mini-v5/mini-v51-20261030a/operator-recoveries/frozen-evaluator.py`. Its source digest is `cdf8ceeb667f9cbcdf20c36abfd892fc010e76d594490f9dc237205b8469fadc`. Terminal/retirement guards were hardened after that campaign completed; the grading records and plan were not changed. Restore a cohort's exact frozen evaluator when recovering an incomplete older campaign. Never silently substitute the current helper code into an existing frozen evaluation plan.
