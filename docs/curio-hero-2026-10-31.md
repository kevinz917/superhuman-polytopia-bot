# hero-1v1 on the Curio cluster: runbook (2026-10-31)

**Status, 2026-10-31 afternoon.** The continuation launched at 11:09 CST was stopped: the 4-GPU learner hung after
1-2 updates every time. Bisected with `experiments/curio/trial.py`: `stream_views=true` (DeviceRows page streaming,
device writes from the receiver threads) hangs with zero updates; `stream_views=false` runs 13+ updates at 6.4 s per
262k batch. The synthetic `Learner.benchmark` and a bare `shard_map` all-reduce probe (`nccl_probe.py`) both pass, so
collectives are fine on this host; the deadlock needs the streaming path. All Curio configs now run with
`stream_views` off. Earlier suspects that were NOT the cause: NCCL P2P / SHM / IB transport, the send thread, launch
concurrency (a `serialize_gpu` lock is on the branch, harmless, default on). The CUDA-graph crash on the first start was
real and is fixed by `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (pip CUDA 12.9 libraries on the 550 / CUDA 12.4
driver). The user then changed the goal: **train a 1v1 hero from scratch on the real rules (main after PR #11)**,
head only (the cluster owner asked that the two CPU servers stay unused), with infrastructure that can be rerun.
The sections below describe the harness; the continuation-specific parts are kept for reference.

## Throughput (2026-10-31 evening): 37k -> ~77k samples/s

Trials with `experiments/curio/trial.py` (full fleet, explicit pinning, steady-state averages) and
`curio.py bench` (training step alone):

| setup | samples/s |
|---|---|
| first launch (sockets-only NCCL, dense views, learner on far cores) | 37k |
| coded views + default NCCL (t2) | 66k |
| + value pass on GPUs 0-3, each group decodes its own copy of the coded views (t19-t23) | **~77k** |
| 8-GPU learner (any variant) | 43-50k |
| training step alone: 1 / 2 / 4 / 8 GPUs | 28k / 56k / 112k / 215k |

An in-process probe (`POLYTOPIA_PROBE_AT`) shows the live epoch runs at exactly the benchmark's 2.35 s on GPUs 4-7; the
remaining ~1 s per update is the training GPUs' own decode (~0.35 s on real data) plus uploads. Untried: a newer
NVIDIA driver (555+) so CUDA graphs work (the 550 driver rejects the graph memset nodes of the pip CUDA 12.9 libraries).

## From-scratch run (the current goal)

**Launched 2026-10-31 12:35 CST** as `hero-1v1-scratch-curio` (`--hours 21`; training ends Nov 1 09:35 CST, fleet
until 12:05, then fog head + evals). First 5 minutes: 47 updates at ~7.0 s each, ~37k samples/s received, KL 0.005-0.02,
4 actor nodes x 96 processes all connected, zero errors. Expected ~2.8 B samples by the end of training.

Transferable lessons from the Flagship setup on the same head (local copy `~/Desktop/projects/duper/rl_ray`, Curio
UCX outage report): same split (inference GPUs 0-3, trainers 4-7); one trainer process per GPU there, one JAX process
over 4 GPUs here (the threaded page-streaming path deadlocked, now off); their worst outage was workers alive but not
producing, so the keeper now restarts actor nodes whose reports go stale or whose sample count stops for 10 min.
Learner cost: the 4-GPU policy step alone is 2.4 s per 262k batch (110k samples/s); the omniscient critic's pass
brings the live step to ~6 s, so the learner is the wall (~40-45k/s). A future lever is an 8-GPU learner sharing
GPUs with the inference servers (untested).

```bash
python experiments/curio/curio.py launch --run hero-1v1-scratch-curio --config hero-1v1-scratch.json \
  --parent /data/polytopia/ckpt/scratch-init/hero-1v1-init-4x128-s1.npz --no-critic --no-start-ref --hours 20
```
Recipe `configs/hero-1v1-scratch.json` = the Oct 26 hero-1v1 recipe (4x128, omniscient critic from scratch, LR 3e-4 ->
1e-4 over the last 25%, turn clock, proposed-v1 maps, seed 801) with `stream_views` off and checkpoints every 20
updates. Parent: `random_parent.py` (network.init, seed 1, 1.97 M parameters). Topology: learner GPUs 4-7 (the
minibatch of 16384 must divide by the device count, so 4 or 8, not 6), four GPU-served actor nodes on GPUs 0-3, ladders
on cores 224-239, search after the fleet stops on cores 32-223. Measured: one actor node delivers ~41k samples/s, the
4-GPU step takes 5.7 s per 262k batch (~45k/s), so the learner is the wall at ~40-45k/s: about 3 B samples in 20 h.
hero-1v1 reached champion parity at ~3.9 B, so expect the 20-h checkpoint to land a little below the champion; extend
by restarting with a larger `--hours` (the learner resumes `latest.npz` on the same clock) if the window allows.
Ladder references: champion, bn4, c3, hero1 (hero-1v1 final) and hero_u11650 (the strongest 1v1 bot).

Restore of the cluster afterwards: `rm -rf /data/polytopia ~/.cargo ~/.rustup` on the head and
`rm -rf /home/ps/polytopia` on each worker; nothing else was changed (the two rustup lines in `~/.bashrc` /
`~/.profile` are already reverted, backups `~/.bashrc.bak-polytopia-1031`, `~/.profile.bak-polytopia-1031`).

---

# (Original plan) hero-1v1 continuation: 24-hour runbook

Everything below is staged and ready; nothing has been launched. One command starts the whole chain and a keeper on
the cluster head carries it to the end without the Mac. Code: `experiments/curio/` on branch `claude/curio-hero`.

## Decision: continue the strongest checkpoint, hero-1v1-cont1h u11650

| checkpoint | net | samples | vs champion (128 games) | note |
|---|---|---|---|---|
| hero-1v1 final u14871 | 4x128 | 3.9 B | 52% | from scratch on the real rules |
| **hero-1v1-cont1h u11650** (Modal, 4x H100) | 4x128 | 6.95 B | **68% (+131 Elo)**, 60% vs bn4, 63% vs B1 | **strongest 1v1 bot; the parent here** |
| hero-1v1-cont1h u17840 (Daytona continuation) | 4x128 | 8.6 B | 59%; **45% vs its own start u11650** | lost ground: 8 learner resumes, a stale pool, LR reset; not used |

Why continue rather than restart: the from-scratch lineage took ~3.9 B samples (a full night) just to reach champion
parity, and this 24-hour window yields roughly 6-9 B samples. A restart would spend most of the window re-earning what
u11650 has; every well-formed continuation so far gained (hero-1v1 -> cont1h: +131 Elo). hero already found the
Riding -> Chivalry knight line that beats the champion on the public site (docs/site-games-analysis-2026-10-31.md).

"Bigger network" = the hero 4x128 net (the champion is 2x64). A larger trunk (6x160 / 8x192) has no checkpoint and
cannot beat 7 B samples of 4x128 inside 24 hours; that question stays with the iso-FLOP sweep plan.

Rules: hero trained on the real 2.17.3 engine (PR #7, Oct 27). PR #11 (Oct 31) moved the engine closer to the real
game (landing units cannot attack, integer damage, ruins/explorer/push, diplomacy, four-seat population, and the real
Drylands map generator for `classic` maps). Training keeps the `proposed-v1` map mix hero was trained on: 85% of its
maps are the Python variant presets (unchanged), 15% `classic` (now the real generator, no water). Expect a dip in the
first hour or two while the policy adapts, then gains. All ladder references play under the same new rules.

## What runs where

Cluster (read-only probe 2026-10-31 09:36 CST): head `ubun` 2x Xeon 8581C 240 threads, 754 GiB, 8x RTX 4090 48 GB,
driver 550 / CUDA 12.4, Ubuntu 22.04, `/data` 6.7 TB free; workers `.83`/`.84` 2x EPYC 9754 512 threads, 944/1007 GiB,
1.7 TB free each. All three share the 100 Gb IB fabric (10.0.1.x): the learner port is reached over it. Internet
(PyPI, GitHub) works from all three. **The cluster is busy right now: Flagship holds all 8 GPUs (4 at 100%, 4 at 25 GB)
and ~290 load on each worker.** Nothing below may start until their run is off the GPUs.

| role | box | resources | config |
|---|---|---|---|
| learner | head GPUs 4-7, cores 0-31 | 4x 4090 bf16 trunk, per-device minibatch 4096 | `configs/hero-1v1-curio.json` |
| GPU-served actors | head GPUs 0-3, cores 32-223 | 4 nodes x 96 processes x 24 games, in-node inference server (2 pinned cores each) | `configs/cluster.json: head_actors` |
| CPU actors | w83, w84 cores 0-255 | 256 processes x 48 games each, policy on the CPU | `worker_actors` |
| ladders | w84 cores 256-511 | 128 processes, greedy both seats, `proposed-v1` 14x14 | `eval` |
| search arena, head check | w83 cores 256-511 | 128 processes | `search` |
| keeper | head | one Python process, every 60 s | `keeper.py` |

Memory: head actors ~480 GiB + learner ~60 GiB of 754; workers ~320-640 GiB of 944. Connections to the learner:
384 + 512 = 896 (the Oct 28 fixes ran 816 at 208k samples/s).

Expected rates (to be confirmed by `bench` and the first 10 minutes): 4x 4090 learner capacity ~100-150k samples/s
(4x H100 did 355k/s; 4090 has ~1/3 the bf16 throughput and bandwidth); head GPU actors ~4 x 25-35k/s; worker CPU
actors ~15-20k/s each (4x128 on Zen 4c, ~60 samples/s per process). The learner is probably the wall at ~100k/s:
**~7 B samples in 19 h** (hero-1v1 + cont1h together were 6.95 B). If the bench shows the learner far above the fleet,
move a GPU from the learner to the actors (`learner.gpus` and `head_actors` in cluster.json) before launch.

## Timeline (24 h window, T0 = engineer's go)

| when | what | by |
|---|---|---|
| T0 | `check` (GPUs free, load down), `bench` 5 min | you |
| T0+0:10 | `launch --run hero-1v1-curio --hours 19` | you |
| T0+0:10 .. T0+19:10 | training; ladder every 2 h (32 maps x 2 seats vs champion, bn4, c3, hero1) | keeper |
| T0+19:10 | final.npz; final ladder 64 maps x 2 vs champion, bn4, c3, hero1, start (u11650) | keeper |
| T0+19:12 .. T0+20:15 | fog-of-war value head: same fleet, `hero-1v1-curio-foghead` (policy frozen, 1 h) | keeper |
| T0+20:15 .. T0+21:30 | head check (AUC vs c3 and the champion head) + search arena hero+search vs champion+search (64 maps x 2) | keeper |
| ~T0+21:30 | `runs/hero-1v1-curio/summary.md` on the head; `fetch --weights` to the Mac | keeper / you |
| slack | 2.5 h for restarts, a late bench, or extending `--hours` | |

The actors stay up `--fleet-extra-hours 2.5` past training for the fog-head phase. If the window is shorter, lower
`--hours`; the fog head needs ~1 h and the evals ~1.25 h after it.

## Commands (from the Mac, in the `claude/curio-hero` checkout)

```bash
python experiments/curio/curio.py check
```
```bash
python experiments/curio/curio.py bench
```
```bash
python experiments/curio/curio.py launch --run hero-1v1-curio --hours 19
```
```bash
python experiments/curio/curio.py status
```
```bash
python experiments/curio/curio.py logs --lines 40
```
```bash
python experiments/curio/curio.py stop
```
```bash
python experiments/curio/curio.py fetch --run hero-1v1-curio --weights
```

- `stage` and `publish` are already done (see "Staged" below); rerun `publish` after any code change (it builds the
  engine wheel once per engine-rs tree and refreshes `code/current` on every box; needs a clean commit).
- `stop` ends the current learner gracefully (final.npz) and the keeper continues the chain (fog head, evals).
  `stop --all` tears the fleet down and the keeper exits. `kill` is the emergency hammer on every box.
- `launch --no-auto` trains only. A fog head by hand afterwards:
  `launch --run hero-1v1-curio-foghead --config hero-1v1-curio-foghead.json --parent /data/polytopia/runs/hero-1v1-curio/final.npz --no-critic --hours 1 --no-auto`.
- Elo from a history file on the Mac: `python scripts/ladder_elo.py hero-1v1-curio --file runs/curio-hero/hero-1v1-curio/hero-1v1-curio-history.json --anchor c3`.

## Go / no-go gates

1. `check`: all three boxes answer, `ib_ok` on both workers, GPUs 0-7 show < 1 GB used, worker load < 50,
   `venv/bin/python` present everywhere.
2. `bench`: a `bench_result` row on 4 GPUs at minibatch 16384 with no OOM. If it OOMs, set `"grad_accum": 2` in
   `configs/hero-1v1-curio.json` (exact, halves the per-device microbatch) and `publish` again.
3. After launch, within 10 min: `status` shows `"stage": "update"` rows, `approx_kl` ~0.005-0.02, received
   samples/s climbing as actors connect, all 6 actor nodes alive. A KL ~1 from update 1 means something is wrong with
   the served weights (the Daytona 4p symptom): `stop --all`, read the actor logs.
4. First ladder (T0+2 h): win rate vs the champion may sit below u11650's 68% while the policy adapts to the new
   rules. A steady decline over three rounds is the signal to stop and reconsider.

## What the keeper does (experiments/curio/keeper.py)

- Restarts a dead learner (it resumes `latest.npz` with Adam, log and pool on the same clock); dumps and restarts a
  learner whose status has not advanced for 10 min; restarts on 5 consecutive updates with KL > .3. Restarts dead actor
  nodes on all boxes; the nodes themselves reconnect when a learner comes back.
- Freezes `latest.npz` every 2 h to `ladder/<run>-update-N.npz`, copies it to w84, runs one `arena.py match` per
  reference on cores 256-511, and appends rows to `ladder/<run>-history.json` (same row format as the Modal ladder).
- At the end of training: final 64-map ladder; stops the fleet; starts the fog-head learner on `final.npz`; restarts
  the fleet; after that run: `head_check.py` (which parameters moved, AUC of the new head vs c3 and the champion fog
  head) on w84 and two search arms on w83 (hero fog head + search vs champion fog head + search, and vs champion raw);
  then writes `summary.md` (ladder table, Bradley-Terry Elo anchored at c3, search results).
- Markers: `runs/<run>/STOP` (graceful end of the current learner), `runs/<run>/STOP-ALL` (teardown).
  State: `runs/<run>/keeper-state.json`; log: `logs/keeper.log`.

## Changes to the core (this branch)

- `LearnerConfig.send_thread` (default False): the DeviceRows send thread is the strongly suspected cause of the
  multi-GPU learner hangs on Daytona (6 in 4.5 h with it, 0 in 3 h without). Off costs ~20% of assembly throughput on
  H100s; irrelevant here if the GPUs are the wall.
- Snapshot cadence fix: snapshots are written at every multiple of `snapshot_every` even when `checkpoint_every` does
  not divide it (the Daytona resumes rebuilt half-empty pools because of this). This run uses checkpoint 20 / snapshot
  250 / historical 250.

## Run configuration (configs/hero-1v1-curio.json)

The cont1h recipe with: parent `hero-1v1-cont1h/latest.npz` (u11650) and its omniscient critic loaded through
`critic_path` (no fresh-critic reset: every c6 continuation that reset the critic lost to its parent); LR 2e-4 ->
5e-5 over the last 30% (the parent ended at 1e-4 after its decay; Daytona restarted at 3e-4 and lost); fresh Adam
and a fresh historical pool (parent + hist snapshots every 250 updates, keep 8); turn clock γ .99 / λ .95, cap-score
reward 1.0, advantage floor .05, `proposed-v1` maps, seed 903. Fog head: `hero-1v1-curio-foghead.json` = the
scratch-fog-head-1 / bn4-foghead recipe (no omniscient critic, `critic_warmup` 1e7 so only `critic_hidden`/`value`
train, pure self-play, LR 3e-4, 1 h).

Not in this run (candidates for a second day): a Chivalry-biased or scripted cavalry-rush exploiter in the pool
and a Diplomacy/Cloak exploiter for the land-map blind spot; `training_updates` (stop by update count).

## Staged on the cluster

Under `/data/polytopia` on the head and `/home/ps/polytopia` on the workers: `venv` (uv, Python 3.12, jax 0.11.2,
CUDA 12 wheels on the head, CPU on the workers), `code/current` (git archive of this commit), `wheels/<engine tree>`
(engine-rs built on the head, installed everywhere), `ckpt/` (parent, critic, references, sha256-verified):

| name | file | role |
|---|---|---|
| parent | `ckpt/hero-1v1-cont1h/latest.npz` + `latest-critic.npz` | u11650 + its omniscient critic; also the `start` reference in the final ladder |
| hero1 | `ckpt/hero-1v1/final.npz` | u14871 |
| bn4 | `ckpt/bn-main4/final.npz` | 4x128, Verda lineage |
| champion | `ckpt/scratch-cont-2-final/final.npz` | the public site's bot (2x64, legacy rules digest) |
| champion_foghead | `ckpt/scratch-cont-2-foghead/final.npz` | search opponent / head-check reference |
| c3 | `ckpt/c3-maps-ctl-update-4854/final.npz` | Elo anchor |

Local copies: `runs/curio-hero/stage/`. Logs of the staging: `runs/curio-hero/{stage,publish}.log`.

## Verified on 2026-10-31 (without touching the GPUs or the Flagship run)

- `check`: all three boxes answer through the hops; `ib_ok` from both workers to 10.0.1.4.
- `stage`: 7 checkpoints sha256-verified on the Mac, rsynced to the head and both workers.
- `publish`: venvs built (uv Python 3.12.16, jax 0.11.2, CUDA wheels on the head), engine wheel built on the head
  (tree 93f1df698a5c) and installed everywhere; `setup_ok ... engine=ok` on all three.
- Head, CPU only: both learner configs parse with the real `LearnerConfig`; every staged checkpoint loads
  (`checkpoint_template` + `load_checkpoint`, legacy digests included).
- w84: `arena.py match` hero u11650 vs champion, 2 maps x 2 seats on 2 pinned cores: 2-2, 27 s (so a 128-game
  ladder round per reference is ~15 min on 16 cores; the keeper has 256).
- Wire smoke: a CPU learner (`configs/smoke-cpu.json`, cores 224-239 on the head) and one 8-process actor node on w83
  over IB: hello, relay, 8 engine processes compiled, ~100k samples uploaded and the first batch assembled in 62 s.
  The CPU learner then sat in the first training step (4x128 on 16 CPU cores, ~1,100% CPU) and was stopped after
  15 min; the GPU step is what `bench` measures at T0. Run directory removed.
- Detached start / pidfile / stop plumbing exercised on the head and a worker (found and fixed a `cd && cmd &`
  precedence bug that backgrounded the whole list).

Shared boxes: every kill in the harness matches our venv's interpreter path (`/data/polytopia/venv/bin/python`,
`/home/ps/polytopia/venv/bin/python`), never a bare `python` or `multiprocessing`, so the Flagship team's processes
cannot be hit. Nothing in the harness touches their files, Ray, PM2 or systemd.

## After the run: shipping to the site

`fetch --weights` brings `runs/hero-1v1-curio-foghead/final.npz` to the Mac. Serving it needs the browser bot
rebuilt on the current engine-rs (the site's WASM is the pre-alignment ruleset with a fixed 2x64 net port):
generalise `engine-wasm/net.rs` to 4x128, export the weights, re-check decision parity, then push `external-website`.
