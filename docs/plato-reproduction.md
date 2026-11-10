# Running and auditing Plato

The released `weights/plato/final.npz` is runnable from this checkout. This release adds the V3 builder, its V1/V2
attention primitives, shared-encoder privileged critic, PPO integration, GPU-resident trainer, C/CUDA simulator,
map-table generator, paired arena and the historical stage supervisor. No private module is needed for inference.

## Install, verify, play

From the repository root, with Python 3.11+ and a Rust toolchain:

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e '.[test,train]'
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop --release --manifest-path engine-rs/Cargo.toml
.venv/bin/python scripts/plato.py verify
./scripts/studio.sh --checkpoint weights/plato/final.npz
```

`verify` checks the policy and critic with the normal hash/ruleset/shape/finiteness loader, then makes eight legal
decisions through `StructuredPolicy` and the Studio's Rust environment. The policy has **3,212,922** parameters;
the critic has **400,805**. Inference needs JAX and NumPy; torch, CUDA, Tokamax and Modal are only used by the GPU
training/deployment paths. The original float32 attention implementation is the default on a fresh environment.
The V3 policy's value return is a placeholder zero: use raw policy play, not the champion's value-based turn search.
The privileged critic is for training; it must never be supplied to the player-facing policy.

## Evaluate on both seats

A two-game smoke check on one fresh map, sampled at temperature 1:

```bash
.venv/bin/python scripts/plato.py arena --maps 1 --output runs/plato-smoke.json
```

A larger panel against the released references, with peace legal and disabled:

```bash
.venv/bin/python scripts/plato.py arena --maps 1024 --seed-base 500000 --output runs/plato-hero-peace.json
.venv/bin/python scripts/plato.py arena --maps 1024 --seed-base 500000 --no-peace --output runs/plato-hero-nopeace.json
.venv/bin/python scripts/plato.py arena --maps 1024 --seed-base 502048 --opponent weights/champion/final.npz --output runs/plato-champion.json
```

The default `--backend rust` uses engine-rs on CPU. `--backend cpu` runs the C oracle after
`make -C engine-cu cpu SIZE=14`; `--backend cuda` uses the CUDA library after building it with `nvcc`. Both policies
act on the mover's redacted view, with resignation masked out. Every map is played twice with seats exchanged.
The output separates elimination-only results from score-at-cap results, includes truncations and unfinished
games, and reports map-clustered 95% intervals when there is more than one map. A panel with any unfinished games
is incomplete. The historical report used its own frozen seeds and 99% intervals; these example commands are fresh
checks, not a claim to regenerate those exact numbers. A one-map smoke has no confidence interval.

## Source map

| Component | Public source |
|---|---|
| Hybrid stem, patch tokens, policy and privileged critic | [`rl/transformer_v3.py`](../src/polytopia_bot/rl/transformer_v3.py) |
| Attention blocks, relative position bias, kernel choices | [`rl/transformer.py`](../src/polytopia_bot/rl/transformer.py) |
| Sampling, architecture dispatch, checkpoint templates, PPO | [`rl/network.py`](../src/polytopia_bot/rl/network.py), [`rl/distributed.py`](../src/polytopia_bot/rl/distributed.py), [`rl/ppo.py`](../src/polytopia_bot/rl/ppo.py) |
| Production GPU-resident PPO and CLI | [`rl/gpu_resident.py`](../src/polytopia_bot/rl/gpu_resident.py), [`gpu_resident_train.py`](../experiments/gpu_resident_train.py) |
| C/CUDA simulator and Python ABI | [`engine-cu/`](../engine-cu), [`rl/gpu_engine.py`](../src/polytopia_bot/rl/gpu_engine.py) |
| Map tables and vector actor utilities | [`rl/vector_actor.py`](../src/polytopia_bot/rl/vector_actor.py), [`scripts/plato.py`](../scripts/plato.py) |
| Stage-4 pools, cap loss, leak/illegal-action/collusion checks | [`flagship_harness.py`](../experiments/claude_v3l/flagship_harness.py) |
| Checkpoint repair, retry/stop/budget supervision, multi-node launch | [`claude_v3l/`](../experiments/claude_v3l) |
| Frozen launch image and dependency pins | [`modal_flagship.py`](../experiments/claude_v3l/modal_flagship.py), [`modal_gpu_engine.py`](../experiments/modal_gpu_engine.py) |
| Paired arena | [`gpu_arena.py`](../experiments/claude_v3/gpu_arena.py) |
| Architecture experiments, configs and results | [`claude_v3/`](../experiments/claude_v3), [`claude_v3h/`](../experiments/claude_v3h), [`v3-compute/`](v3-compute), [`v3-hybrid/`](v3-hybrid) |

The export is from the stage-4 development snapshot **`1212facef0e3c5e4b6fabd3535e1a89926e7e2aa`**. The
[source manifest](plato-source-manifest.json) records each imported file's original hash and the exported hash,
including portability changes. Later stage-5 and observation-version-4 experiments are separate work and are not
substituted for the observation-version-3 Plato release. This is the recovered stage-4 source snapshot, not a claim
that all historical training workers used that final source revision for every update.

## Train from scratch or fine-tune

The entire algorithm is public. GPU training needs a Linux NVIDIA host, CUDA toolkit, torch 2.6.0 and
`jax[cuda12]==0.11.2`. The historical fast path also used `jax-triton==0.5.0`, `triton==3.8.0`,
`nvidia-cudnn-cu12==9.28.0.42` and Tokamax commit `1dc0149a34b09d7c559c04533029eb550e2c7027`
(its URL and archive hash are pinned in `modal_flagship.py`). That image also records the `ptxas` replacement and
kernel environment. CPU verification and evaluation do not need these optional packages.

Generate the deterministic training table, which needs no private download:

```bash
.venv/bin/python scripts/plato.py maps
```

The interval is `[200000, 300000)`, 100,000 classic 14×14 two-player maps. Small `--lo`/`--hi` intervals are useful
for checking the generator; training requires the table interval to match the learner config exactly.

Stages 1–3 used the early recipe embedded in [`configs/plato-recipe.json`](../configs/plato-recipe.json), also
published as `experiments/claude_v3/hero-1v1-classic-nopeace-warm150.json`. For a new run on a CUDA host:

```bash
make -C engine-cu cuda SIZE=14 SM=sm_90
PYTHONPATH=src .venv/bin/python experiments/gpu_resident_train.py \
  --config experiments/claude_v3/hero-1v1-classic-nopeace-warm150.json \
  --architecture transformer-v3-p2-h4 --width 128 --blocks 6 \
  --map-table runs/maps/table-classic-14-2p-200000-300000.npz \
  --run-root runs --run-id plato-new --games 4096 --sets 4 \
  --devices 1 --grad-accum 16 --seconds 600
```

This is a short new training run, not the original 8/16-H100 campaign or a promise of the same score. Set
`SM` for the target GPU. `--parent weights/plato/final.npz --critic-path weights/plato/final-critic.npz` instead
initializes a fine-tune from the released tensors. The release stripped Adam state, so it cannot resume the exact
optimizer trajectory. Changing the device count/precision, JAX backend or pool changes the numerical trajectory.

For the actual stage-4 curriculum, [`configs/plato.json`](../configs/plato.json) is the base learner config,
**not the whole launch**: its `no_peace: true` is overridden per game set by `flagship_harness.py`'s
`stage4.peace_sets`. The final configuration uses three peace-legal sets of four, separate opponent pools per
rule, historical snapshots, and `stage4.terminal_rule: cap_loss_both`. The corresponding learner settings are
`cap_score_reward: 0` and `truncation_reward: -1`; both seats lose at the round cap and the critic does not bootstrap.
The stage-by-stage learning rates, pool probabilities and monitor thresholds are in `plato-recipe.json`.
[`configs/plato-stage4-harness.example.json`](../configs/plato-stage4-harness.example.json) shows how those fields
map to the actual harness API. Supply real paths for every listed historical opponent before running it.

The historical Modal launchers are included for audit, with original app/volume names and budget/approval logic.
Their default local control directory is now `runs/plato-control`; `V3L_FLAGSHIP_DIR`, `V3L_QUAL_LEDGER`,
`V3L_FLAG_LEDGER` and `V3L_MODAL` override local paths. `make_manifest.py` only writes a manifest; deployment and
launch are separate explicit operations. The launcher's old stage dollar limits and approvals describe that
campaign, not an ongoing training run. Its referenced Modal volumes must be populated in the operator's account;
the direct training command above does not depend on those volumes.
The monitor logs locally by default; its original coordinator integration is optional via `V3L_BRAIN` and
`V3L_STATUS_FILE`. Earlier session-E/H shell launchers and notebooks retain their historical absolute paths for
audit; use the direct command above or configure the stage-L launcher for a new deployment.

## Availability and verification limits

All source needed to load/play Plato, generate training maps, run PPO, implement its final curriculum and supervise
the training is now in this repository. The final policy and critic, configs, aggregate benchmark/strategy data
and earlier public reference bots are included.

The original intermediate Plato snapshots (`u23270`, `u37641`, `u72889`, etc.), CNN control, Cloak/x3d exploiter
checkpoints, per-update optimizer states, complete worker logs, original per-stage manifests and raw game traces
have **not** been recovered into this checkout. They remain a gap for exact historical replay and the identical
stage-4 opponent mixture. Their names in a recipe are not downloadable artifacts. Reading the original Modal
archive failed during this export because the configured client could not connect; no training job was launched.
The deterministic map table can be regenerated, but its original stored bytes/hash have not been recovered either.
Do not describe this release as a complete archive of the historical training run or as a bit-for-bit reproduction.

Local release validation covers real policy/critic loading and legal Studio decisions, V3 shapes and gradients,
zero privileged-input policy gradients, PPO equivalence, checkpoint repair, supervisor fault handling, vector
actors, older CNN/attention paths and Rust/C engine parity. CUDA/Tokamax training and a new full 337-H100-hour run
were not executed for this source export. Historical performance claims remain the published measurements.
The [release verification record](benchmarks/2026-11-10-plato-source-smoke.json) contains the test suites, engine
parity receipts and matching Rust/C smoke results. These small panels check the release, not playing strength.
