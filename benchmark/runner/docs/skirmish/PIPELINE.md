# Run the complete Polytopia coding benchmark

Use `scripts/run_skirmish_cohort.py` for new cohorts. It runs Mini-SWE-agent 2.4.6 through OpenRouter, freezes every source, rebuilds it in a fresh offline sandbox, and evaluates all revisions and final peer matches. The older `experiments/modal_skirmish_v5.py::run` entrypoint is retained for the original 180-second evaluation protocol and existing receipts.

The coding conditions are unchanged: three independent builds per model; forty continuous minutes without supplied game feedback, or four ten-minute coding stages with six expert traces after each of the first three stages. All model-visible sandboxes use two CPU cores and 2 GiB memory, Rust/std/serde_json, a detailed rule book, nine public maps and real-engine inspection/protocol tools. There is no supplied public opponent, internet access, unlimited expert-practice endpoint, model-call cap or dollar gate. Self-written tests consume coding time.

Final evaluation uses the explicit `referee-900-stream-v1` revision. The referee guard is 900 seconds; guest limits remain 2 seconds per decision plus the original startup allowance, 256 MiB memory and one trillion fuel per game. Native 100-round and 10,000-action caps are draws. A referee timeout remains unresolved and blocks a complete point estimate. Authoring feedback matches retain the frozen v5.1 180-second guard. Every evaluation job binds its referee configuration and evaluator-source digest.

Long CPU authoring controllers and coordinators use non-preemptible workers. This changes infrastructure scheduling, not the model-visible sandbox or coding budget. The normal pipeline polls results with bounded waits, reconciles complete artifacts before retrying a controller, and can recover a lost controller return from a complete set of frozen non-error stage receipts without making another model call. The default permits one fresh infrastructure retry, never a selective retry of a valid compilation failure. Extra operator recoveries are explicit and recorded.

Run commands from the repository root, with `OPENROUTER_API_KEY` already present only in the trusted operator environment. All research execution and raw artifacts remain on Modal staging.

```sh
MODAL_DISABLE_API_PROXY=1 modal run --env staging --detach \
  scripts/run_skirmish_cohort.py::launch \
  --prefix mini-smoke-YYYYMMDD \
  --manifest configs/skirmish/mini-v5-smoke.json --smoke

MODAL_DISABLE_API_PROXY=1 modal run --env staging \
  scripts/run_skirmish_cohort.py::inspect_results \
  --prefix mini-smoke-YYYYMMDD --output /tmp/smoke.json --watch
```

A smoke uses short coding stages and one build per model/track. Its scores are integration-test output, not leaderboard measurements. The complete entrypoint was exercised with four real DeepSeek Flash/Luna authoring runs, source rebuilds and 192 games; all 119,175 replay actions passed and no games remained unresolved. Compact evidence is in `docs/skirmish/pipeline-smoke-2026-10-31.json`.

After the smoke passes, launch the official manifest:

```sh
MODAL_DISABLE_API_PROXY=1 modal run --env staging --detach \
  scripts/run_skirmish_cohort.py::launch --prefix mini-YYYYMMDD \
  --manifest configs/skirmish/mini-v5-top11.json

MODAL_DISABLE_API_PROXY=1 modal run --env staging \
  scripts/run_skirmish_cohort.py::inspect_results \
  --prefix mini-YYYYMMDD --output /tmp/benchmark.json --watch
```

The eleven-model manifest allocates 66 authoring slots. All 165 frozen stages get the same fifteen hidden layouts, both seats, against three evaluation-only references: 14,850 scheduled primary outcomes, including virtual zeroes for invalid compilations. Up to 1,980 final-stage peer games form the secondary ladder. Hidden maps are allocated only after every slot has a valid frozen attempt and current independent source-rebuild proofs. Hidden results never return to an author.

The primary ranking is mean match points (win 1, draw 0.5, loss 0), with all three builds included and build/map bootstrap intervals. Reports separately show each fixed opponent, revision curves, submission quality, peer Elo, authoring timing and preserved API receipts. Elo is roster-dependent; three builds remain a preliminary comparison. API receipts are not a billing reconciliation: unsaved/replayed or in-flight requests, probes and Modal compute can add charges.

An existing cohort requires explicit `--resume` with the same source, manifest, image and host-code fingerprints, after the old coordinator is terminal. Existing hidden maps and digest-checked game results are reused. Do not edit a frozen evaluator and silently rescore a cohort. See [recovery procedures](RECOVERY.md) for an interrupted legacy cohort or an explicitly authorized additional infrastructure attempt.

The final evaluator permits up to 64 concurrent match workers, each with the same two-CPU allocation and guest limits. A 192-game concurrency smoke passed every replay audit (110,187 actions) and matched all outcomes, turn/action counts and public trace hashes from the 32-worker baseline. Private traces matched in 191 of 192 games: the expert selected six different private actions near the end of one match. The cause of that inference variation is not established. Replay determinism is verified from recorded actions; repeated policy inference is not claimed to be bitwise identical across workers. See `parallel64-comparison-2026-10-31.json`.

The launch command rejects resuming a completed/provisional cohort or any cohort with a saved report before probes or status writes. A negative integration check against the completed full-pipeline smoke left its status, report and both plans unchanged and made no model calls; see `resume-guard-2026-10-31.json`. Changing a status label cannot make an existing report resumable.

`inspect_results` exports selected-artifact progress after authoring closes and preserves superseded attempt states separately. Its compact export keeps the scoring report unchanged, omits raw compiler diagnostics, and separates comprehensive authoring receipts from probe receipts. The legacy `status` entrypoint remains available.

The full eleven-model campaign is complete: [results and interpretation](V51_RESULTS.md), [full tables and curves](v51-2026-10-30.md), and [final integrity checks](v51-2026-10-30-integrity.json). All 16,830 scheduled outcomes are recorded and 12,254,391 actions passed replay audits.
