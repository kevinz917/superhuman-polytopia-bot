# Cloud comparison migration — October 18, 2026

Qwen inference was already running on Modal H100s. Separate structured-policy watchers were downloading checkpoints and evaluating them on this Mac. Those local watchers and active evaluators were stopped; exact process arguments are preserved in `runs/cloud-comparison-migration-20261018/stopped-local-processes.json`. No training job was stopped. The superseded cloud retention-only service was stopped after its replacement started.

## Current cloud work

- Qwen3.5-4B thinking on/off: [Modal app](https://modal.com/apps/duper/staging/ap-YbeIuJq6q5OTtyEQvj0yuA), run `qwen35-thinking-games-repaired-v1`, 40 games per mode. Both workers loaded and began play. Its durable CPU coordinator gathers and summarizes results; no local inference is required. See generation diagnosis and repaired protocol.
- Structured checkpoints: [Modal app](https://modal.com/apps/duper/staging/ap-sxb7cXXxImeRcDEgiSo5dP), run `cloud-screens-20261018`. One service with a hard limit of six CPU cores and 16 GiB runs retention plus three bounded development evaluators (teacher, champion12, champion96). It reads the training volume read-only, retains snapshots in `polytopia-structured-rl-retained`, and screens the latest pending snapshot per run. All 15 roster runs have retained cloud snapshots. Maximum service duration is 11 hours within a 12-hour platform timeout; $12 was reserved in the existing campaign ledger, not reported as actual spend.
- Local watcher and evaluator CLI entry points now reject execution without the explicit cloud environment marker. Normal operation uses `experiments/modal_structured_comparisons.py`. Status collection downloads small reports, not checkpoints.

The current service is already dispatched; do not launch another copy. Cloud status and logs are under `cloud-evaluations/cloud-screens-20261018/` on the retained volume. Watcher status is under `checkpoints/`. Local dispatch/status receipts are in `runs/cloud-comparison-migration-20261018/`.

## Local cleanup and recovery

369 regular checkpoint files (335 unique content hashes) were uploaded to the retained Modal volume. A separate cloud function read and SHA-256 verified every file. Local files were then rehashed before removal. Cleanup removed 369 files totaling 1,337,703,417 bytes (1.34 GB) and 332 links pointing to those files. This is logical file size, not a measurement of physical APFS space reclaimed.

Preserved locally: historical results, replay traces, provenance/source metadata, and qualified parent bundles. No unique checkpoint was discarded. The migration directory contains:

- `manifest.json`: original relative paths, byte sizes, SHA-256 hashes, and remote paths.
- `verified.json`: the cloud verification receipt for every file.
- `cleanup-receipt.json`: exact removed files and symlink targets.

Recovery uses volume `polytopia-structured-rl-retained` in environment `staging`. For any removed file, fetch its recorded `remote_path` from `migration-20261018/checkpoints/<sha256>.npz`, verify its SHA-256, and restore the recorded relative path under `runs/structured-ppo-v3`. Restore recorded symlinks only after their targets exist. Routine comparisons should continue using the cloud copies.

## Blender observation

The observed processes were 13 idle `blender-mcp` connector processes and their launchers, parented by the Codex backend. They were not Blender rendering processes and were not used by these comparisons. Their duplicate startup cause was not established; this migration did not disable unrelated connectors.

These are execution and storage changes, not new model-strength results. The Qwen paired comparison and structured development screens are still running.
