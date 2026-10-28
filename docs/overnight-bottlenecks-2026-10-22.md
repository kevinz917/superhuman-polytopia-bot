# What limits training speed now: a read of `scratch-overnight-1` (2026-10-22)

**Question.** After Stage A, the Rust engine and A6, where does the time and money go in a real 7-h run, and what
are the next levers?

**Short answer.** The overnight run spent most of its time **actor-bound**: 12 actor containers made ~86k
samples/s against a learner that could take ~118k. For the last 1.7 h, 15 containers ran and the learner was the
wall at ~117k. Both sides are now close to each other, so the next 2× needs work on both. The biggest surprise is
in the bill, not the profile: **30% of the run's Modal bill ($50 of $169) was network egress.** About two thirds
of that is the learner sending a fresh 1 MB copy of the policy to every one of the 192-243 actor *processes*
every 2 updates.

Read-only analysis: no runs launched. Sources: `training.json` (8,724 per-update rows) and `protocol.json` from
`/results/distributed/scratch-overnight-1/` on the `polytopia-structured-rl` volume; the run's registry entries
(`scratch-overnight-1/actor/*`); `modal billing report --start 2026-10-22 -r h --show-resources` (staging). The
tables come from `scripts/analyze_learner_bottlenecks.py` (new, reads the JSON only). Spend for this analysis is
a few cents of volume and registry reads.

## 1. What code it ran

Branch `claude/scratch-selfplay-32b9ec`. Its merge base with `main` is c298e13 (the A6 write-up), so it has
**Stage A + the first Rust engine + A6** (2af8865: streamed host pages, zero-copy wire). It does **not** have the
Rust 10x engine work (c090610, ee8b9a0, merged to main as 240b060 later on Oct 22). `src/polytopia_bot/rl/distributed.py`
is identical on the branch and on `main` today, so every learner finding below still applies to `main`.

Config: 1×H100 learner (16 CPU), 12 actor containers × `cpu=16` (`os.cpu_count()` = 32) × 16 processes × 24
games, 24 GiB each; batch 262,144, minibatch 16,384, **epochs 1**, `publish_every` 2, `max_lag` 10, omniscient
critic with learner-side values (actors run no critic), `stream_views`, `assembly_threads` 4.

## 2. Throughput over the night

Per hour of learner clock. "Consumed" is batch × updates ÷ wall time (what the learner actually trained on).

| hour | updates | consumed samples/s | actor steps/s | connections | steps/s per process | ready_wait | inbox (of 128) | train | gap/update | decisions per game |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-1 | 1,169 | 85.2k | 88.6k | 192 | 462 | 1.20 s | 4.9 | 1.66 s | 0.21 s | 1,976 |
| 1-2 | 1,147 | 83.5k | 87.5k | 192 | 456 | 1.25 | 3.7 | 1.65 | 0.22 | 1,887 |
| 2-3 | 1,187 | 86.4k | 90.5k | 192 | 472 | 1.14 | 4.6 | 1.66 | 0.23 | 1,920 |
| 3-4 | 1,212 | 88.3k | 92.8k | 192 | 483 | 1.07 | 5.3 | 1.66 | 0.24 | 1,837 |
| 4-5 | 893 | 65.0k | 68.6k | 32-208 | 386 | 2.12 | 4.1 | 1.65 | 0.25 | 1,906 |
| 5-6 | 1,521 | 110.8k | 116.8k | 144-243 | 498 | 0.44 | 86.5 | 1.66 | 0.26 | 1,972 |
| 6-7 | 1,594 | 116.1k | 122.1k | 243 | 503 | 0.30 | 107.8 | 1.67 | 0.29 | 1,923 |

- **No slowdown as play improved.** Decisions per self-play game stayed at 1,840-1,980 all night, and per-process
  speed went slightly *up* (456 → 483 steps/s) over hours 0-4. The 20-minute check's drift (61k → 55k) was the
  move away from random play in the first minutes; after that, games got shorter in rounds but not more expensive
  per decision.
- **Hour 4-5 is the preemption wave** (04:37 orchestrator replay, see the scratch write-up): about 15 minutes of
  near-zero actor supply, roughly 3.5% of the run's samples and ~$3.
- **The change at hour 5 is the fleet, not the code.** After the replay recovery 15 containers ran (3
  replacements plus 3 originals that restarted late). The inbox went from empty to full and the rate jumped to
  the learner's ceiling.

Overall: 2.287 B samples in 25,201 s = **90.7k samples/s**.

## 3. The learner: what each update costs

The learner has two threads. The **assembler** pulls actor messages from the inbox (max 128 messages, ~65k
rows), streams the views to the GPU, runs the critic over the batch (values + GAE), and hands over one batch. The
**training thread** waits for that batch (`ready_wait`), trains one epoch (`train`), then does bookkeeping (the
gap between rows: status write, log line, checkpoint).

Medians, actor-bound phase (0.2-5.25 h, 5,725 updates) vs learner-bound phase (5.25-7 h, 2,806 updates):

| per update | actor-bound | learner-bound | what it is |
|---|---|---|---|
| `seconds` (wait + device batch + train) | 2.88 s | 1.98 s | |
| `train_seconds` | 1.65 | 1.66 | one PPO epoch on the H100 (= 158k samples/s if it were the only cost) |
| `ready_wait_seconds` | 1.22 | 0.31 | training thread waiting for the assembler |
| gap outside `seconds` | 0.14 + checkpoints | 0.14 + checkpoints | status put to the Modal Dict + log line every update; checkpoint every 50 |
| `asm_wait_seconds` | 1.79 | 0.004 | assembler waiting for actor messages |
| `asm_stream_seconds` | 0.65 | 0.96 | copy views into host pages + async device puts (4 threads) |
| `asm_views_seconds` | 0.12 | 0.32 | wait for those puts to land |
| `asm_value_seconds` | 0.36 | 0.46 | critic forward over the batch on the same GPU |
| other asm (concat, GAE, normalize) | 0.04 | 0.04 | |
| **assembler work** (total − wait) | **1.24** | **2.0-2.1** | |
| `inbox` | 0 | 128 (full) | |
| `stale_dropped` rows | 31 | 357 | 0.01% / 0.14% of a batch |
| `mean_lag` (updates) | 1.7 | 2.6 | |

**Verdict per phase.**

- **Hours 0-4.5: actor-bound.** The inbox was empty, the assembler spent 1.8 s per update waiting for data, and
  the H100 trained 1.66 s out of every ~2.9 s (57% duty).
- **Hours 5.25-7: learner-bound, and inside the learner the assembler is the wall, not the GPU.** The inbox
  stayed full and `asm_wait` was ~0. The assembler needs ~2.0-2.1 s of work per batch against 1.66 s of
  training, so the training thread still waits 0.31 s per update. Assembly work grows under load: stream 0.65 →
  0.96 s and views 0.12 → 0.32 s. Two things compete with it then: the receiver threads (1.8 s/update of zstd
  across threads) and the training thread, for the GIL and for the PCIe/GPU queue.
- **The bookkeeping gap costs 8.5% of the learner's wall clock** (2,147 of 25,201 s). It has two parts. First,
  ~0.14 s every update, mostly the synchronous `registry[f'{run}/status'] = row` (a network call) on the training
  thread. Second, a checkpoint stall every 50 updates that **grows with the run**: 3.6 s near the start, 6.6 s at
  the end. The cause is that `training.json` (27 MB by the end) is rewritten whole at each checkpoint, so the cost
  is O(n²) over a run. A 24-h run would stall ~15 s per checkpoint. A stall also backs up the actors once the
  128-message inbox fills (under 1 s of actor output at 88k/s).

Realized learner ceiling: 262,144 / (1.98 + 0.14 + 6/50) ≈ **117k samples/s**, matching A6's bench (116.9k).
Removing the gap alone gives ~131k (then the assembler's 2.0 s is the cycle). Getting the assembler under the
1.66 s epoch gives ~155k, where one H100 at epochs 1 and today's precision tops out.

## 4. The actors

Per-container totals from the registry (`sent_samples`, `steps`, `elapsed` since the container started,
including ~100 s of compile). No errors, 0 restarts and 0 reconnects reported in any surviving summary. Indices 5,
7 and 8 were overwritten by containers that restarted again in the last 8 minutes, so they are left out.

| container | life | steps/s over its life |
|---|---|---|
| 0 (original, never preempted) | 7.0 h | **5.1k** |
| 3 (original, never preempted) | 7.0 h | 7.9k |
| 9 (restarted 04:54) | 2.1 h | 6.2k |
| 1, 2, 4, 6, 10, 11 (restarted 04:54-05:07) | 1.8-2.1 h | 9.0-9.6k |
| 20, 21, 22 (replacements, 05:01-05:12) | 1.8-2.0 h | 9.4-9.6k |

- **Fleet average while actor-bound: 7.4k steps/s per container, 462 steps/s per process** (16 processes on 32
  CPUs). That is in line with the Stage A + Rust tick model: ~47 ms per 24-game tick, of which the CPU policy
  forward is ~30 ms.
- **Host lottery: a 1.9× spread between containers** (5.1k to 9.6k). The containers placed at ~05:00 were all
  fast, even while throttled by a full inbox. The e2e smoke (2 of 6 containers ~25% slower) and the Rust smoke
  (364-614 steps/s per process) showed the same thing. The slowest quarter of the fleet costs the same per hour
  for about 60% of the output.
- **Half of each actor container's CPUs are idle.** Production runs 16 processes on a `cpu=16` container that
  exposes 32 CPUs. The inference-server bench measured the same container shape with the Rust engine and local
  policy: **16 × 24 = 8,922 decisions/s, 32 × 24 = 14,348 (1.61×)**.
- **What the actor tick is made of now.** Stage A's production profile was policy 29.6 ms, rival 4.7, engine 20.5,
  rows/pack 4.3, giving 60.8 ms. The Rust port replaced the engine's 20.5 ms with `RustGames.tick` at ~6.7 ms
  (incl. the JAX `encode_view`), giving ~47 ms. Rust 10x cut `RustGames.tick` again, to ~2.1 ms (0.6 ms Rust + ~1.5-1.8
  ms `encode_view`), giving **~42 ms, of which the CPU policy forward is ~70%**. So `encode_view` dominates
  `RustGames.tick`, not the actor tick. Porting it to Rust saves ≤1.5 ms of ~42 (≤4%) on the local path, and
  nothing on the served path, where the actor already skips the encode.

## 5. Cost: what Modal actually billed

`modal billing report`, staging, hourly, by resource (UTC hours; the run is 07:00-14:00 UTC):

| app | CPU | memory | H100 | **network egress** | total |
|---|---|---|---|---|---|
| run `ap-6mk7N4…` (learner + 12 actors + orchestrator) | $71.05 | $19.35 | $28.53 | **$50.34** | $169.27 |
| 3 replacement actors `ap-prhfDm…` | $4.65 | $1.11 | – | $1.32 | $7.08 |
| Elo ladder `ap-6NAuSY…` | $3.82 | $1.11 | – | 0 | $4.93 |
| **total** | | | | | **$181.28** |

Per steady hour of the run app: CPU $10.3-10.5, memory $2.82, H100 $3.95. These match the launcher's rates (CPU
~7% above the reservation). Egress was **$6.4-6.8/h with 12 actors and $9.0-9.8/h with 15**. This is the whole
"1.33× billing ratio" seen since Oct 21: compute bills at the launcher's price, and egress is the rest.

**Where the egress comes from (fitted, not metered per stream).** The replacement-actor app carries only actor
uploads: $0.65/h for ~28k samples/s, i.e. **~$0.0063 per M samples uploaded**. Taking that out of the run app
leaves ~$4.7/h with 12 actors. That fits the learner's weight broadcast. Every `publish_every` = 2 updates, the
learner sends the full policy (519,267 parameters in fp16 = 1.04 MB, `compress=False`) separately to *each* actor
process: 192 connections × ~600 publishes/h ≈ 120 GB/h. The same fit predicts $9.9/h for the 15-container phase
(243 connections, 0.44 updates/s), against $9.0-9.8 billed. So roughly **two thirds of egress is the weight
broadcast and one third is the sample upload**. The broadcast is 16× redundant: all 16 processes in a container
download the same bytes. Actors run no critic, so `_critic_leaves()` is already `None` and no critic weights are sent.

**$/M samples, billed:**

| run | samples/s | $/h billed | $/M samples |
|---|---|---|---|
| c5 recipe (Oct 20-21) | 26.9k | $21.7 | $0.225 |
| e2e smoke, main + 6 actors (Oct 21) | 41.1k | $12.4 | ~$0.09 |
| A6 smoke, 6 actors (Oct 22) | 43.8k | ~$12.4 (est.) | ~$0.08 |
| **overnight, 12 actors, hours 1-4** | 86k | $23.9 | **$0.077** |
| **overnight, 15 actors, last full hour** | 117k | $30.6 | **$0.073** |
| overnight, whole run incl. replay and replacements | 90.7k | $176.35 | $0.077 |
| overnight with egress removed (hypothetical) | | | $0.048-0.056 |

So the overnight run was 3.2-4.3× the c5 speed at a third of the cost per sample. Actors are ~70% of the
compute bill (12 × $0.95/h vs the learner's $5.2/h).

## 6. Ranked bottlenecks and levers

Ranked by gain per unit of cost and risk. "Exact" means the training math is unchanged bit for bit.

| # | lever | what it fixes | expected gain | effort / cost | risk |
|---|---|---|---|---|---|
| 1 | **Run 32 actor processes per `cpu=16` container** (all 32 CPUs). Memory 24 → ~40 GiB | Half the actor CPUs idle | **~1.5-1.6× samples per actor container** (bench 1.61×); actor $/sample −35-40% | Config only. 15-min smoke ~$6 to confirm the rate, memory and that the CPU billing line stays near the reservation | Modal bills CPU used above the reservation. The bench showed the throughput, not the bill. Verify both in the smoke |
| 2 | **Weight relay per container** (one connection per container fetches weights and shares them with its processes), or as a stopgap `publish_every` 4 | ~$4.7-7.9/h of egress, 20-28% of the bill | **−20-25% $/sample**, no speed change. Stopgap: −10%, one config line | Relay ~½ day in `run_actor_container` + test that processes see identical weights, then a 10-min smoke ~$5 read from the billing egress line. Exact | Stopgap adds ~1 update of policy lag (mean 1.7 → 2.7, max_lag 10) |
| 3 | **Launch from `main`** (Rust 10x engine) | Engine share of the actor tick | +10-20% per actor (smoke 53.6k vs 43.8k on 6 actors; tick model ~+12%) | Zero: already merged | None (bitwise equal) |
| 4 | **Bookkeeping off the training thread**: status put and checkpoint in a background thread, `training.json` appended (JSONL or per-chunk) instead of rewritten | 8.5% of learner wall clock; growing O(n²) checkpoint stall that also stalls actors | **Learner ceiling 117k → ~131k (+12%)**; more on longer runs | ~½ day, exact; one 10-min smoke | Low (checkpoint consistency: snapshot params before handing them to the thread) |
| 5 | **Assembler under the epoch**: more copy threads (learner has 16 CPUs, uses 4), value pass in its own stream or reduced precision, receivers off the GIL | Assembler 2.0 s vs train 1.66 s when saturated | Ceiling ~131k → **~155k (+18%)**: GPU-bound at epochs 1 | 1-2 days with `assembly_bench.py` on the H100 (~$2 per bench) | Low-medium; the A6 bitwise gate covers it |
| 6 | **Drop slow hosts**: launch N+2 actor containers, cancel the 2 slowest after 5 minutes (or continuously replace containers below ~70% of the median) | 1.9× host spread | ~+10-15% actor output per dollar | ~½ day in the orchestrator | Interacts with the replay hazard: cancel by call id, never the orchestrator |
| 7 | **Learner scale-up past ~155k**: 2×H100 data-parallel (code shards over local GPUs) or a bf16 trunk with the parity gate | GPU epoch 1.66 s | ~1.7-1.9× ceiling with 2 GPUs if the assembler scales with them; bf16 maybe 1.3-1.5× | 2×H100 +$3.95/h; ~2-3 days incl. the assembler's device sharding and a gate | Medium; only worth it after 1-5, when ~12 fast containers outrun one H100 |
| 8 | **GPU inference server** (`claude/inference-server`) | CPU policy forward = ~70% of the actor tick | Today 1.19× vs 32 local processes (server capped by its serial Python loop). With the no-gather and overlap levers the doc estimates 2-3× per container, unmeasured | 2-3 days + ~$1 bench reruns; adds an L4 (~$0.8/h) per container | Medium; pays only if it beats lever 1 by more than the L4's price |
| 9 | `encode_view` in Rust | ≤4% of the local actor tick | ≤ +4% | 1-2 days + bitwise gate | Low gain; skip unless lever 8 is dropped and everything else is done |
| 10 | Curio cluster (8×4090 48 GB + ~1,000 EPYC threads, zero marginal cost) | Actor CPU and the whole bill | At ~450 decisions/s per thread, ~1,000 threads ≈ 4× the current learner's appetite; the 4090s could host learners and inference servers | Needs your explicit authorization and a LAN launcher (`distributed.py` has no Modal dependency) | Not touched here |

**What this adds up to (estimates).** Levers 1-4 are cheap and mostly config or small, exact changes. With them,
one H100 (~131k samples/s) needs only about 10 actor containers (~13k each after levers 1 and 3), not 12-15.
Cost would be roughly learner $5.2/h + 10 actors at ~$1.1/h + ~$3/h of upload and relay egress ≈ $19/h. That is
**~$0.04/M samples at ~131k samples/s**: about 1.5× the overnight run's 12-actor speed at about half its cost
per sample. Lever 5 takes the same setup to ~155k with ~12 containers. Past that, the learner needs a second GPU
(lever 7), and only then is the inference server (lever 8) worth building.

## 7. Smaller observations

- Stale drops and lag are not a problem at either phase (≤0.14% of a batch, mean lag ≤2.6 of 10).
- `received_samples_per_second` in the log is measured over the training thread's `seconds`, which leaves out
  the gap. It reads ~6% below the consumed rate. Use batch × updates ÷ elapsed (as above) for throughput.
- The registry's per-actor summaries are last-writer-wins per index, so a container restarted after a replay
  overwrites the earlier summary. Per-container rates over a whole run need the actor log or a per-start key.
- The checkpoint also writes a 27 MB `training.json` to the volume and commits it 175 times. Besides the stall,
  that is ~2.4 GB of volume writes per run, which the append fix (lever 4) removes as well.

## How to reproduce

```bash
modal volume get polytopia-structured-rl distributed/scratch-overnight-1/training.json training.json
python scripts/analyze_learner_bottlenecks.py training.json --window 3600 --split 18900
modal billing report --start 2026-10-22 -r h --json --show-resources
```
