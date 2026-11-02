# Polytopia benchmark v5 — frozen decisions

The complete eleven-model campaign is available in [V51_RESULTS.md](V51_RESULTS.md).

> Execution update: new cohorts use the [complete pipeline](PIPELINE.md), with non-preemptible CPU controllers, independent source rebuilds, and the explicitly versioned 900-second final referee guard. The authoring/feedback receipts below retain v5.1; older 180-second result sets are not silently rescored.

The new protocol is separate from archived v4 results. It implements the six
review priorities: shared hidden revision tests; five layouts per map family;
public-only compact feedback; ready/start timing and API/tool telemetry; explicit
benchmark draws for engine round/action caps; durable source capture and failure
classification. All APIs use OpenRouter. Dollars are optional reporting metadata.

## Cohort

`configs/skirmish/mini-v5-top11.json`: the original ten models plus GPT-6 Astra,
three independent builds each, both equal-budget 40-minute tracks (66 authoring
cells). Providers and high reasoning are pinned. A compatibility probe exercises
actual tools and continuation before authoring begins. No model-dependent tuning,
best-of selection, public opponent, or automatic unlimited practice is provided.

The reference panel is fixed BEFORE authoring: archived v4 Qwen one-shot (weak),
archived v4 DeepSeek Flash one-shot (intermediate), and the digest-pinned hero-1v1
expert. These are empirical strength labels within the previous cohort, not claims
of comprehensive strategy coverage. Source/Wasm hashes and provenance are recorded.
Weights and opponent code never enter the coding sandbox.

## Tests and scores

After ALL 66 cells finish (including declared infrastructure replacements), draw
one hidden suite: five layouts each of drylands/lakes/continents, both seats.
Every frozen stage faces all three references on these exact 30 positions: 90
scheduled results per stage. Compile failures are scheduled zeroes, not skipped
or reverted. There are 165 stage submissions and 14,850 scheduled primary results.
The final-stage model score averages all three attempts. Unchanged starters are
flagged, but scored according to actual games. No best attempt is selected.

For secondary peer comparisons, each model's replicate N plays other models'
replicate N on one shared layout per family, both seats. This produces at most
1,980 additional games. Invalid final submissions do not get invented Elo ratings;
they remain zeroes in the primary score and appear in reliability statistics.

A victory receives 1, an announced engine-cap draw 0.5, a loss 0. Native 100-round
and 10,000-action caps are benchmark draws. A referee timeout is an unresolved
result, NOT a draw or a loss, and blocks a complete primary score. Frozen v5.1
authoring feedback uses 180 seconds; final evaluation uses the explicit
`referee-900-stream-v1` revision with a uniform 900-second guard.
Bounds show what unresolved results could change. Guest runtime faults forfeit.
Matched map blocks and both seats reduce seat/map effects. Bootstrap uncertainty
resamples builds and map blocks within map families. Three builds remain a small
sample. Fitted peer Elo is secondary and changes when the roster changes.

All game traces and sources stay on Modal; every played scored game is replayed
through the real engine before its result is accepted. Virtual compilation losses
are clearly identified and have no fictitious replay. Revision curves compare all
stages on the same hidden tests and never flow back into authoring.

## Failure and recovery

The controller persists sandbox ID and stage-start receipts. A root-owned watchdog
kills coder processes and captures source independently at cutoff. Source and hash
are committed to the results volume before compilation. Compilation can be retried
against this immutable source without another model call. Earlier working code is
never substituted for a broken final file. Lost authoring trials are infrastructure
exclusions, not model losses; at most one fresh full-budget retry is automatic.
Every original attempt and replacement is preserved. Repeated infrastructure
failures stop the cohort for inspection, rather than silently reducing its roster.
Evaluation game summaries are idempotent and match a frozen job digest. Partial
attempts retain their own trace directories. Resumption reuses the same maps,
artifacts and completed summaries; it must never allocate a new favorable map set.

Coding time includes API latency, model retries, tool calls and tests. Worker
initialization is excluded by a ready/start handshake. Metadata-only timing events
separate API, tool, startup and overhead time and identify in-flight calls cut at
deadline. Costs are partial completed-call receipts, never an eligibility check.

## Run

```sh
modal run --env staging experiments/modal_skirmish.py::test
modal run --env staging experiments/modal_skirmish.py::mini_preflight
modal run --env staging --detach scripts/run_skirmish_cohort.py::launch --prefix NEW_COHORT_ID
modal run --env staging scripts/run_skirmish_cohort.py::inspect_results --prefix NEW_COHORT_ID --output /tmp/v5-summary.json --watch
```

Before official authoring, run a smaller explicitly marked smoke manifest through
both tracks and the complete scoring/audit pipeline. Smoke output cannot be mixed
with official results. Long-running execution must remain detached from local
polling; absence of a fresh local log is not proof that a worker is alive or dead.

## v5.1 transport correction

The first official allocation `mini-v5-20261030a` was invalidated before scoring:
a legitimate 76,112-byte shell command exceeded Modal's 65,536-byte argv limit.
The whole cohort was stopped rather than mixing transport implementations. Its
available frozen stages, trajectories and invalidation receipt remain on Modal.
The corrected implementation transfers command payloads as files and runs bash
scripts rather than placing full commands in either Modal or OS argv. Live
regressions cover a 300 KB command, Unicode/quoted literals and a 250 KB source.
The independent/revision budgets, model settings, reference panel and score do not
change. Worker-error exits also preserve frozen source before terminating a sandbox.
