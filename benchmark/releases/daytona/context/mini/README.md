# Polytopia coding benchmark

The current protocol is **polytopia-mini-v5.1**, implemented with mini-SWE-agent
2.4.6 and OpenRouter. See [the frozen v5 design](../../docs/skirmish/V5_DESIGN.md),
[wire/timing rules](PROTOCOL.md), [authoring tools](TOOLS.md) and
[agent quick start](QUICKSTART.md).

The official manifest `configs/skirmish/mini-v5-top11.json` includes eleven models
(including GPT-6 Astra), three independent builds per model, and two tracks:

- `independent-40m`: up to forty continuous coding minutes, no expert feedback.
- `feedback-3r-40m`: initial ten minutes, then three ten-minute revisions; six
  expert games after each of stages 0, 1 and 2. Reading feedback consumes coding
  time; producing it does not.

All agents get identical offline sandboxes (two CPU cores, 2 GiB), Rust 1.90,
std + serde_json, nine public maps, the full rule book and the real-engine CLI.
No opponent code/weights or automatic practice-match runner is provided. Local
self-written tests consume time. There is **no dollar gate or spending limit**.
The API key stays in the trusted controller, never in the coding sandbox.

The primary score averages all three builds against an evaluation-only fixed
panel on fifteen common hidden maps, both seats, after every agent is frozen.
Every intermediate revision uses the same hidden tests; hidden results never
return to the agent. Compile failures score zero; unchanged starters are labeled.
All played scored games receive a full deterministic engine replay audit.
A secondary peer ladder uses matched replicate numbers and three shared maps,
both seats. Do not compare its Elo values across different rosters or protocols.

```sh
modal run --env staging experiments/modal_skirmish.py::test
modal run --env staging experiments/modal_skirmish.py::mini_preflight
# OPENROUTER_API_KEY already exists in the trusted process environment.
modal run --env staging --detach experiments/modal_skirmish_v5.py::run \
  --prefix mini-v5-smoke-YYYYMMDD --manifest configs/skirmish/mini-v5-smoke.json --smoke
modal run --env staging --detach experiments/modal_skirmish_v5.py::run \
  --prefix mini-v5-YYYYMMDD
modal run --env staging experiments/modal_skirmish_v5.py::status \
  --prefix mini-v5-YYYYMMDD --output /tmp/v5-summary.json
```

Run the small marked smoke through compilation, scheduled feedback, cutoff,
hidden scoring and replay audits before the official cohort. Smoke output is
never an official score. All generated source, Wasm, trajectories and game logs
remain on Modal staging volume `polytopia-skirmish`; only compact reports leave it.

The root-owned watchdog captures source at cutoff independently of controller
transport. Source is committed before trusted compilation, never reconstructed by
another model. A declared infrastructure failure may receive one fresh full-budget
retry; failed compilations and unchanged starters never receive a selective retry.
Original attempts and immutable evaluation plans are preserved. Duplicate cohort
allocation is rejected; evaluation resumption reuses its frozen maps and jobs.

Archived v1–v4 experiments and the earlier 20-minute configuration files remain
for provenance. Their results keep their original timing/cap rules and must not
be merged into v5 rankings. The v4 ten-model pilot is in
`docs/skirmish/top10-2026-10-30.md`; it is not a v5 result.

Rebuild the complete rule book after protocol edits:

```sh
python3 scripts/build_skirmish_rules.py
```
