# Continuous Harbor integration demo, 2026-10-31

[Results](summary.md) and [machine-readable summary](summary.json) are copied from the completed native Harbor run. Each trial directory preserves the final authored Rust source, capture receipt, independent compilation receipt, and verifier summary. [Measured verifier timing](runtime-measurements.json) comes from native Harbor trial phase timestamps.

These are integration demos: one authored bot/model, 40-minute coding allowance, 36 final matches/model. The standard suite schedules 360 final matches/model. Do not use this small demo as a model ranking.

The complete local run remains at `artifacts/harbor-runs/continuous-proxy-demo-20261031` (about 1.9 GiB): conversations, practice traces, final match traces, replay audits and native job/trial files. Those large ignored artifacts, local VM images and private reference assets are not included in this Git snapshot. Summary provenance paths refer to the local run.

Validation before this snapshot: 82 unit tests passed; 16 legacy-only tests skipped. Real Harbor fixture trials covered valid and broken Rust, practice/trace inspection and source transfer. A separate deadline fixture confirmed cancellation of an overlong tool command. Both live models completed with no Harbor errors; all 72 final outcomes and 64,252 replay actions were verified.
