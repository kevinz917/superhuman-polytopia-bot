# Polytopia Skirmish

> **Current main benchmark:** [continuous 40-minute bot building](../docs/skirmish/CONTINUOUS.md), packaged in [Harbor format](../harbor/README.md). On-demand practice and trace inspection share one coding budget; one frozen final bot receives one held-out panel score. The staged v6 protocols below are historical research conditions.

> **Archived local v6 pilot (October 31):** the [v6 protocol](../docs/skirmish/V6_DESIGN.md) used three controlled authoring conditions and broader opponent/cap/transfer tests. Its results and the [completed v5.1 results](../docs/skirmish/V51_RESULTS.md) remain frozen. The historical driver notes below retain their original rules.

The historical Mini benchmark used [mini-SWE-agent](mini/README.md), with a 20-minute
one-shot track and an initial five-minute build plus three five-minute feedback
revisions, through OpenRouter. See the
[tested v3 results](../docs/skirmish/mini-2026-10-30.md) and
[calibration evidence](../docs/skirmish/calibration-2026-10-30.md).

The sections below preserve historical Mini, custom-loop and pressure-study
notes for reproducibility. Their Modal/OpenRouter instructions and resource limits
do not describe the current local v6 harness.

A private extension of this repository: models write Rust bots under a fixed
resource budget, practice against a frozen reference, inspect public traces,
and revise. Frozen submissions then play seat-paired 1v1 tournaments and get a
Bradley–Terry Elo ladder. Controlled studies vary prompt framing separately
from actual time/revision limits.

- [Experimental framework](../docs/skirmish/pressure-study.md)
- [First implementation pilot and calibration findings](../docs/skirmish/pilot-2026-10-30.md)
- [Stable model-facing manual](PROMPT.md)
- Controller: `experiments/modal_skirmish.py`

## Game and reference

The referee is the existing `engine-rs` through
`studio.environment.RustEnvironment`; no substitute game implementation.
Current track: Bardur, 14×14 classic maps, `polytopia-2.17.3`, two seats.
All compilation, model API calls, games and artifacts run on Modal staging.

The fixed reference is **hero-1v1**, greedy deterministic, released update 11650,
SHA-256 `e6f4e1977e7f9cdcfbbe17358d87e49d638d82150496a6d4099e0cd389b0934f`.
It is the strongest released raw 1v1 policy in the inspected qualification;
this does not establish superiority to every search configuration.

## Identical model-facing tools

- `read_file(path)`: starter, rules and Rust engine reference source.
- `write_source(source)`: replace `src/main.rs`; changed content consumes one revision.
- `compile()`: locked, offline Cargo build to `wasm32-wasip1`.
- `campaign(maps)`: practice against hero, both seats; returns outcomes and trace IDs.
- `review(game_id, offset, limit)`: page through the bot's own public observations/actions.
- `budget_status()`: neutral factual remaining time/revisions/practice allowance.
- `finish()`: freeze the current compiled bot when ready.

The coding sandbox has `block_network=True`, cached Rust dependencies and only
the starter/engine documentation. It has no checkpoint, model API credential,
other submissions or final maps. There is no arbitrary host shell, browser or
package-install tool. Only Rust `std` and `serde_json` are initially available.

Generated bots run in Wasmtime with WASI stdin/stdout, no preopened filesystem,
no environment/network, 256 MiB memory, 2 seconds per decision (plus a one-time
5-second startup allowance) and 100 billion fuel units per game. The referee
has a 180-second game cap. Numerical limits are disclosed to the model. The
old v1 2-billion-fuel track is historical; its longer games exposed fuel
exhaustion inside JSON parsing. V1/v2 ratings must not be mixed.

## Resource and treatment contract

`Budget` supports time, revision and combined profiles. A changed-source write
is a revision even if it does not compile; identical writes are free and the
starter is revision 0. After the last edit the agent can compile/test/review and
finish. Read/compile/practice also consume wall time and their own request/game
ceilings. A revision-primary profile retains a generous administrative clock.

The framing catalog includes neutral, evaluative, supportive and challenge
wording. It changes evaluation tone, not permissions, objective, scoring or real
limits. Each protocol records the exact framing, full prompt, resource vector,
source/build hashes and model/decoding settings. The game manual is separate
from this operator README so documentation edits do not change model prompts.

Model HTTP requests have a killable parent deadline. The fixed reference is
loaded and warmed before the authoring clock. Code/builds arriving after the
deadline are not accepted; controller wall time is recorded separately from
the bounded authoring window.

## Evaluation and study execution

Practice and final maps use separate partitions. The trusted controller draws
fresh hidden final seeds only after entrants freeze. Every map is played in
both seats. Runtime faults are forfeits, reported separately from native wins
and losses. Simulator/referee cutoffs stay unresolved and are excluded from the
Elo fit. The symmetric prior makes separated fits finite; ratings are relative
to this roster, and map intervals condition on the frozen bots.

A study randomizes and saves all assignments before dispatch, limits writing
runs to two at a time, and evaluates each replicate block on common maps. Up to
eight bounded match workers process at most eight games apiece. Reports retain
failed allocations, submission validity, actual resource use, treatment labels
and descriptive paired **authoring-run** contrasts. One short run per condition
cannot establish a pressure effect.

Full source, Wasm, Cargo lockfiles, model/tool transcripts, source archives,
protocols and traces stay on the `polytopia-skirmish` Modal volume. Small status
and analytical reports can be fetched locally. Run/study IDs are immutable;
failed attempts are not silently overwritten or resubmitted. Different writing
budgets/framings can share a tournament only when game/runtime/source tracks agree.

## Commands

The OpenRouter key comes from `OPENROUTER_API_KEY` in the launcher environment,
then an ephemeral secret reaches only the trusted controller. No key is written
to source, command arguments, transcripts, manifests or coding sandboxes.

```sh
modal run --env staging experiments/modal_skirmish.py::test
modal run --env staging experiments/modal_skirmish.py::smoke
modal run --detach --env staging experiments/modal_skirmish.py::launch \
  --run-id my-run-001 --config configs/skirmish/default.json
modal run --env staging experiments/modal_skirmish.py::status --run-id my-run-001
modal run --env staging experiments/modal_skirmish.py::inspect --run-id my-run-001
modal run --detach --env staging experiments/modal_skirmish.py::ladder \
  --tournament-id my-ladder-001 --run-ids my-run-001,another-run-001 --maps 16
modal run --env staging experiments/modal_skirmish.py::audit --tournament-id my-ladder-001
modal run --detach --env staging experiments/modal_skirmish.py::study \
  --spec configs/skirmish/pressure-smoke.json
```

`default.json` allows one hour and 32 held-out maps. Pilot/study configs are
separate explicit dispatches. Stage 1 and Stage 2 are study proposals; the CLI
does not automatically advance to them. A study's assignments and results live
under `studies/<study_id>/`; individual runs and tournaments have their own roots.

OpenRouter is exercised live. Compatible Chat Completions, OpenAI Responses and
Anthropic Messages adapters are included; the latter two have protocol tests,
not live trials. Exact model IDs are configured; the controller does not
substitute models automatically. Provider references:
[OpenRouter](https://openrouter.ai/blog/tutorials/tool-calling/),
[OpenAI](https://developers.openai.com/api/docs/guides/function-calling),
[Anthropic](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview).

## Primary coding benchmark

New runs use [mini-SWE-agent](mini/README.md): a 20-minute one-shot track and an
initial five-minute build plus three five-minute feedback revisions. The custom
loop described above is retained for historical pilot reproducibility.
