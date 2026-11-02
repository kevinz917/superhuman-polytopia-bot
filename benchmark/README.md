# Polytopia Bench

How good are frontier models at writing Polytopia bots? Each model gets one 40-minute
session with a rule book, starter code, nine practice maps and a CLI to compile and
play practice games, with no internet. Its Rust bot sees fog of war, picks one legal
action per call with 2 s per move, and is graded on held-out maps against six
opponents, up to a self-play RL bot.

This directory is a self-contained archive of the 2026-10-31 run: **12 models × five
independently authored bots × 360 final games = 21,600 outcomes**, with the harness,
every conversation, every submitted bot and every final replay. Write-up:
[*Introducing Polytopia Bench*](https://thekevinz.com/blog/polytopia-bench).

## Results

Score is 100 × (wins + half of draws) / 360, mean ± sample SD over each model's five bots.

| Model | Score | Wins vs RL bot |
|---|---:|---:|
| openai/gpt-6-astra | 74.94 ± 6.02 | 2.7% |
| openai/gpt-6.1-sol | 74.39 ± 2.92 | 3.0% |
| google/gemini-4-argon | 64.36 ± 5.80 | 2.7% |
| anthropic/claude-opus-5.5 | 43.42 ± 3.59 | 1.0% |
| anthropic/claude-fable-5-1 | 31.89 ± 5.26 | 0.3% |
| moonshotai/kimi-k3.1-pro | 31.06 ± 14.41 | 1.3% |
| z-ai/glm-5.3 | 30.53 ± 19.06 | 0.3% |
| qwen/qwen3.8-max-0902 | 26.56 ± 16.93 | 0.7% |
| openai/gpt-6-luna | 19.58 ± 14.66 | 0.7% |
| deepseek/deepseek-v4.1-flash-1017 | 18.36 ± 14.82 | 0.3% |
| deepseek/deepseek-v4-pro-0813 | 15.19 ± 12.88 | 0.3% |
| minimax/minimax-m3.1-flash | 13.08 ± 11.28 | 0.3% |

## Where everything is

| Content | Location |
|---|---|
| Leaderboard, methodology and costs | [Full results](runner/docs/skirmish/results/continuous-full-20261031/README.md) |
| Complete per-submission scores and provenance | [results.json](runner/docs/skirmish/results/continuous-full-20261031/results.json) |
| All 60 final Rust programs | [sources/](runner/docs/skirmish/results/continuous-full-20261031/sources/) |
| Source/Wasm compilation and validation receipts | [Result archive](runner/docs/skirmish/results/continuous-full-20261031/) |
| Interactive-article data, including unit choices | [article-data.json](runner/docs/skirmish/results/continuous-full-20261031/article-data.json) |
| Harness, agent, referee and grader | [runner/src/polytopia_bot/skirmish/](runner/src/polytopia_bot/skirmish/) |
| Frozen Rust game engine and shared Python dependencies | [runner/engine-rs/](runner/engine-rs/), [runner/src/](runner/src/) |
| Harbor task, offline environments and requirements | [runner/harbor/](runner/harbor/) |
| Local VM, CLI proxy and Daytona launch/build scripts | [runner/scripts/](runner/scripts/) |
| Configurations and benchmark tests | [configs](runner/configs/skirmish/), [tests](runner/tests/) |
| Historical pilots, comparisons and setup notes | [runner/docs/skirmish/](runner/docs/skirmish/) |
| Conversations, practice records and final replay files | [runs/](runs/) |
| Frozen RL reference weights and reference Wasm bots | [references/](references/) |
| Exact local/Daytona release manifests, maps and source bundles | [releases/](releases/) |
| Original campaign registry and recovery history | [campaign/](campaign/) |
| Monitoring, checkpointing, recovery and finalization scripts | [tools/campaign/](tools/campaign/) |
| Export inventory and file SHA256s | [archive-manifest.json](archive-manifest.json) |

`runner/` preserves the original relative paths and all 207 source pins used by
the recorded campaign. It contains the benchmark work from commit `0ef94bb` and
its pinned engine dependencies. Keeping this snapshot together prevents changes
to the live training engine elsewhere in the repository from changing historical
benchmark behavior. The original worktree is no longer required to read or verify
this archive. Old absolute paths in native receipts are retained as provenance.
[`runner/README.md`](runner/README.md) is the pinned repository README from that
commit; its links point into the original research repository, not this archive.
One Daytona organization ID was redacted from the build receipts before publication;
their manifest hashes were updated to match.

## Verify the saved data

From the repository root, using only Python's standard library:

```sh
python3 benchmark/tools/verify_archive.py
```

This checks file hashes, frozen source pins, all 60 source/Wasm identities,
authoring clocks, the common 360-game schedule, outcomes, aggregate means and SDs,
and practice-log integrity. The 33 originally accepted compact exports retain
their exact blob hashes. The other 27 exports reconstruct every original private
action-log hash and chain from their newly stored replay entries. The check does
not rerun games or call model APIs. Its receipt is [archive-validation.json](archive-validation.json).

## Conversations and replays

Each `runs/<model>/<trial>/` directory contains:

- `authoring.tar.gz`: the original model trajectory, tool/clock/protocol records,
  captured Rust source, and available native task receipts. It is an exact-byte
  archive, apart from any credential redactions explicitly listed in the manifest.
- `final-replay.json.gz`: all 360 original native game receipts, complete recorded
  action entries, original public/private trace hashes, evaluation plan, rewards,
  compilation receipt and compiled Wasm bytes. It uses the already-qualified
  `skirmish-compact-replay-v1` format.
- `practice.tar.gz`: exact private action logs and practice reports, plus an index
  with original public-log hashes. Completed practice observations can be
  reconstructed using the frozen engine and each report's public map/seat. For
  interrupted attempts without a report, the exact public bytes are included too.

There are 60 scored authoring sessions, plus the failed Gemini API attempt under
`runs/failed-authoring/`; the latter is preserved but excluded from scores.
The official submissions retain 1,425 practice directories: 1,404 have audited
reports and 21 are incomplete or empty. See each archive's `index.json`.

For example, list or read a recorded conversation without extracting it:

```sh
tar -tzf benchmark/runs/openai__gpt-6-astra/task__B7Csrvr/authoring.tar.gz
tar -xOzf benchmark/runs/openai__gpt-6-astra/task__B7Csrvr/authoring.tar.gz agent/trajectory.json
```

The [original replay decoder](runner/docs/skirmish/results/continuous-full-20261031/recovery_sources/restore_compact_replay.py)
reconstructs public and private logs using the pinned native engine and checks
their original hashes. Run it inside a verifier environment; it writes `/results`.
Public observations are stored as lossless native-engine replay data rather than
duplicating tens of gigabytes of expanded observation JSON. They passed native
auditing before acceptance; the decoder qualification is saved with the results.

## Build and run the harness

Run the original tools from the frozen runner root. The local path requires
Docker/Compose, Colima, Rust build prerequisites and a configured CLI proxy:

```sh
cd benchmark/runner
uv venv artifacts/harbor-runtime --python 3.12
uv pip install --python artifacts/harbor-runtime/bin/python -r harbor/requirements.lock
colima start --profile polytopia-skirmish
export DOCKER_CONTEXT=colima-polytopia-skirmish
PYTHONPATH=src artifacts/harbor-runtime/bin/python scripts/build_skirmish_harbor.py \
  --references ../references --package artifacts/harbor-packages/new-local-release
```

The builder generates a fresh shared map suite. Use `--shared-suite ../releases/local`
only to reproduce the historical suite. The historical maps and reference weights
are now published; they were held out during the recorded run, and should not be
called unseen maps for future attempts that can access this archive.

See the [Harbor guide](runner/harbor/README.md), [single benchmark protocol](runner/docs/skirmish/CONTINUOUS.md)
and [split local/Daytona instructions](runner/docs/skirmish/SPLIT_EXECUTION.md).
Credentials must be provided through private configuration or environment
variables. No credential, cloud account configuration, container image, VM disk,
virtual environment, or build cache belongs in this directory.

The exporter is [tools/export_local_archive.py](tools/export_local_archive.py).
It reads retained original artifacts, so rerunning the exporter requires the
original worktree; consuming the exported data does not. Unrelated and discarded
pilot runtime directories remain local; their checked-in reports and code are
preserved under `runner/`.
