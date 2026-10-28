# Cheap levers and egress fixes, then a small smoke (2026-10-22)

**Question.** The overnight bottleneck read (`docs/overnight-bottlenecks-2026-10-22.md`) ranked cheap levers:
use all 32 CPUs of each actor container, stop sending the policy to every actor process, take bookkeeping off the
learner's training thread, and look at upload egress and private networking. Are they built, exact, and what
does a small run with them show before a big run?

**Short answer.**

- All five are done. Levers 1-4 are in the code and pass their gates on Modal. Lever 5 (private networking)
  does not work here and is not worth paying for.
- Smoke (3 actor containers × 32 processes, H100 learner, 11 min): **~19.4k samples/s per container**, against
  7.4k in the overnight run (16 processes, older engine): 2.6×. The relay uses one weight connection per
  container (3 relays, not 96 processes), and weight egress fell from ~120 GB/h to ~1.2 GiB/h. The
  bookkeeping gap fell from 0.14 s per update (and 3.6-6.6 s at checkpoints) to under 1 ms (and 11-20 ms).
  Memory high-water: 20 GiB (cgroup), 30 GiB PSS, inside the 40 GiB reservation.
- **The catch is the CPU bill.** Modal bills every busy vCPU, not the 16 reserved cores. A 32-process container
  is billed ~31.8 cores, a 16-process one ~17.3. So 32 processes give ~1.7× the samples for ~1.8× the container
  price, and per sample the two shapes cost the same: ~$0.026 vs ~$0.024-0.027 per M. Lever 1 saves no money
  per sample. It halves the containers needed, which means fewer preemptions and connections.
- **Egress is now just the upload: $0.0056 per M samples, against $0.022 overnight (−75%).** The smoke's
  whole egress line ($0.22) equals its uploaded bytes × $0.04/GiB; the weight broadcast is too small to see.
- **Recommendation: ready for a big run.** Use one H100 and **8 actor containers × 32 processes**. That is
  ~155k samples/s of supply for a learner ceiling of ~131k (projected), at about **$21/h ≈ $0.044 per M
  samples**. The overnight run cost $0.077.

Branch `claude/levers-egress-3dfe46` (from `main` 240b060, with the bottleneck report b832cc2 and the orchestrator
replay guard bcc5279 merged in). Not merged to `main`. No big run launched.

## What was built

| # | lever | change | exact? |
|---|---|---|---|
| 1 | 32 actor processes per `cpu=16` container | `run_actor_container(processes=0)` starts one pinned process per CPU in the container's affinity set (32 on `cpu=16`). The launcher defaults are now `actor_cpu=16`, all CPUs, 24 games, 1.25 GiB per process (40 GiB). The container summary reports the affinity size and memory (cgroup + PSS, current and peak). The actor report carries the host's CPU model. | yes |
| 2 | Weight relay per container | `WeightRelay`: one learner connection per container long-polls for new weights and pool. The learner answers as soon as it publishes, else every 5 s. The relay writes `weights.pkl` / `pool.pkl` to `/dev/shm` by atomic replace. Processes say `relayed=True` in their hello. The learner then sends them no arrays (just version, session, and pool spec/version when it changes), and `RelayReader` fills the reply in from the relay's files. A process waits until the relay holds at least the version the learner announced, so it adopts a policy at the same reply as before (or a newer one if the relay already has it). A `session` id stops a process from taking a replaced learner's stale files. Sample uploads keep one connection per process. `publish_every` stays 2. | yes (bit-identical arrays) |
| 3 | Bookkeeping off the training thread | One bookkeeping thread runs the status put (Modal Dict) and the checkpoints (log append, saves, volume commit) in order. Status puts coalesce to the newest row. The training thread takes a host snapshot of params / Adam / critic / EMA (~10 ms) and hands it over; the queue is bounded at 4 jobs. The training log is now `training.jsonl`, appended at each checkpoint. `training.json` is written once at the end, so readers of finished runs are unchanged. `_resume` reads either file (skipping a torn last line), keeps rows up to `latest.npz`, and rewrites the `.jsonl` to match. New `training_log` module (no jax). The analysis scripts read both formats; `archive` copies the `.jsonl` too. | yes |
| 4 | Upload codec | `LearnerConfig.upload_zstd_level = 7` (was 1). Lossless, decoded the same way. | yes |
| 5 | Private networking | Probe only (`experiments/modal_i6pn_probe.py`); transport unchanged (Modal TCP tunnel). | n/a |

New per-update fields in the training log: `relays`, `weight_bytes`, `reply_bytes`, `received_bytes` (bytes on
the learner's sockets that update) and `bookkeeping_seconds` (the previous update's post-row work on the
training thread).

## Gates (all on Modal, `run_tests`)

| test file | result |
|---|---|
| `tests/test_levers.py` (new) | 4 passed |
| `tests/test_distributed_rl.py` | 15 passed |
| `tests/test_a6_assembly.py` | 4 passed |
| `tests/test_rust_engine.py` | 1 passed |

- **Relay gate** (`test_relayed_processes_see_the_direct_path_weights_pool_and_versions`). This runs a real
  learner socket server, a `WeightRelay`, one direct client and two relayed clients. Over 6 publishes (3 of them
  with a pool change), every client holds byte-identical weights, omniscient-critic leaves and pool after every
  exchange, at the same version, which is the learner's published one. The relayed connections carry zero arrays,
  and the learner counts exactly one relay. A second test shows a relay file from another learner session never
  satisfies a reader.
- **Resume gate** (`test_background_checkpoints_resume_to_the_same_state_and_log`). A CPU learner trains on
  synthetic messages for 150 s with checkpoints every 2 updates on the bookkeeping thread. Then:
  `training.jsonl` = final `training.json` = the returned history. The status callback saw the last row. A
  learner restarted in the folder resumes params, Adam state and critic **bitwise** equal to the training thread's
  snapshot at latest's update, with the log cut to that update. The same holds from an older run's
  `training.json` alone, and from a `.jsonl` that runs past `latest.npz` and ends in a torn line.
- **Codec gate**: a level-7 frame decodes to the same arrays as level 1.
- In the smoke itself, `training.jsonl` and the final `training.json` hold the same 140 rows.

## Lever 4: upload codec (measured, adopted)

`::upload_codec` on a Modal `cpu=16` container, timed on one pinned core like an actor process. The sets were
A6's 64 early-game messages, plus 64 new **mid-game** messages: real Rust actors on the trained overnight policy,
captured after the first 2,000 messages, ~34k rows. Files: `runs/levers/upload-codec-1.json`, `-2.json`.

| variant (mid-game set) | wire bytes per row | encode ms per 512-row message | learner decode ms |
|---|---|---|---|
| zstd 1 (was) | 186.5 | 5.8 | 1.8 |
| zstd 3 / 4 | 183.8 | 3.7-4.7 | 1.4-1.8 |
| zstd 5 | 174.0 | 7.3 | 2.2 |
| zstd 6 | 156.2 | 12-18 | 1.6-1.9 |
| **zstd 7 (adopted)** | **150.4** | 12.6 | 1.3 |
| byte planes + zstd 1/3 | 220-232 | 12-16 | 41-44 |
| int8 narrowing | no gain (views reach 347; never fits) | | |

Early-game messages behave the same way: 148 → 118 bytes per row at level 7. Level 7 costs ~13 µs more actor CPU
per row. An actor process makes a row every ~1.6-2 ms, so that is **~0.6-0.8% of its CPU**. In return it cuts
upload bytes by **19%**, and the learner's decompression gets no slower. The smoke measured **151 bytes per
sample** on the wire, matching the bench. At ~131k samples/s that is ~66 GiB/h ≈ $2.6/h of egress, against
$3.3/h at level 1.

## Lever 5: private networking (probe, not adopted)

`experiments/modal_i6pn_probe.py`: 1 server + 4 clients per arm, `i6pn=True`, cpu=1, 1 GiB per client. Spend was
about $0.03.

| arm | where Modal placed the containers | i6pn connections |
|---|---|---|
| `region='us'` (accepted by this workspace) | server Azure us-east; clients GCP us-central, Azure us-east, AWS us-west, GCP us-east | 0 of 4 (timeouts; even Azure us-east → Azure us-east) |
| no region | server Azure ap-south; clients GCP eu-west, Azure us-east, Azure ap-northeast, Azure us-east | 0 of 4 |

- `region=` works on this workspace; there is no `cloud=` option to pin a provider. A broad region still spans
  three clouds, and i6pn did not connect even between two containers on the same cloud and region.
- Pricing (modal.com): region selection multiplies **all** compute by **1.15× (broad) to 1.75× (narrow)**. After
  the relay, the egress left is the upload, ~$2.6/h at 131k. That is less than 15% of the ~$16/h of compute, so
  pinning cannot pay for itself even if i6pn worked. The transport stays the public TCP tunnel.
- Side observation: unpinned containers land on four continents (eu-west, ap-south, ap-northeast, us-east). The
  tunnel works from anywhere, but this may explain some of the host-to-host speed spread.

## Smoke

Config: the overnight protocol (`scratch-overnight-1/protocol.json`), continued from its `final.npz` with its
`final-critic.npz` (`critic_path`), epochs 1, batch 262,144, `publish_every` 2, learner precision as overnight.
Smoke-only changes: seed 503, `checkpoint_every` 10 (to exercise the bookkeeping thread), `historical_every` 25
(to send pool changes through the relay). Ledger `runs/levers`.

- `levers-smoke-1`: H100 + **3 × (cpu=16, 32 processes × 24 games, 40 GiB)**, 660 s of training, 140 updates.
  App `ap-yJ9v74ioWABmkhS26N1NOq`.
- `levers-smoke-2-p16` (added after the first billing row; see Cost): the same with **16 processes, 24 GiB**,
  480 s, 52 updates. App `ap-uLnfzX0ROjVqQm3OUonx6i`.

### Throughput per container

Steady state from the registry summaries (`scripts/levers_smoke_monitor.py`, after 4 min of container life):

| run | container | processes | steps/s | per process | memory peak (cgroup / PSS) |
|---|---|---|---|---|---|
| overnight (16 proc, older engine) | fleet mean | 16 | 7.4k | 462 | (24 GiB reserved) |
| smoke-1 | 0 | 32 | 18.9k | 591 | 20.0 / 29.8 GiB |
| smoke-1 | 1 | 32 | 19.0k | 593 | 20.3 / 30.1 GiB |
| smoke-1 | 2 | 32 | 20.4k | 638 | 20.1 / 30.0 GiB |
| smoke-2 | 0 | 16 | 11.3k | 704 | 9.7 / 14.5 GiB |
| smoke-2 | 1 | 16 | 8.0k | 503 | 10.0 / 14.8 GiB (slow host) |
| smoke-2 | 2 | 16 | 11.7k | 731 | 9.9 / 14.7 GiB |

- **32 vs 16 processes: ~1.7× per container** on the fast hosts (19.4k vs 11.5k), close to the bench's 1.61×.
  The per-process rate drops from ~720 to ~610 as the two hyperthreads of each core are both used.
- **vs the overnight's 7.4k: 2.6×.** Of that, ~1.5× is the Rust 10x engine on `main` (per process 462 → ~720 at 16
  processes) and ~1.7× is the 32 processes.
- Learner consumed **58.1k samples/s** in smoke-1 (3 containers) and 30.7k in smoke-2. Both were actor-bound: the
  inbox was empty, `ready_wait` 2.8 s and 6.9 s per update, and train 1.65 s. Policy lag 1.1 and 0.8 updates; 0
  stale rows. No errors, restarts or reconnects in any container. Exit codes are −15 everywhere: the container
  terminates its processes when the learner stops.
- Memory: 32 processes peak at **20 GiB** by the container's cgroup and 30 GiB summed PSS, against a 40 GiB
  reservation. 40 GiB is a safe default. 32 GiB would probably fit, but saves only ~$0.06/h per container.

### Relay and bookkeeping

| per update (median) | overnight (12 × 16 proc) | smoke-1 | smoke-2 |
|---|---|---|---|
| learner connections | 192 | 99 (96 processes + 3 relays) | 51 (48 + 3) |
| `relays` | – | 3 | 3 |
| weight bytes sent | ~100 MB (fit) | 1.68 MB | – |
| weight egress | ~120 GB/h, ~$4.7/h | **1.24 GiB/h, ~$0.05/h** | 0.65 GiB/h |
| upload bytes per sample | ~165 (fit, zstd 1) | **151** (zstd 7) | 150 |
| gap outside `seconds` | 0.14 s + 3.6-6.6 s per checkpoint | **0.3 ms; 11-20 ms at checkpoints** | 0.4 ms |
| `bookkeeping_seconds` after a checkpoint | (the stall) | 11-23 ms (host snapshot) | |

- Each relay fetched every published version (71 in smoke-1 = 140 updates ÷ 2 + the first) and all 6 pool
  changes: 79 MB per container in 11 min. The old path would have sent the same bytes once per process, i.e. 32×
  as much per container.
- With one relay per container, weight egress no longer grows with processes. At 131k samples/s (0.5 updates/s)
  with 10 containers it is ~9 GB/h ≈ **$0.35/h**, against ~$4.7/h overnight.

### Cost (Modal billing, `--show-resources`, staging)

Settled rows, read 18:23 UTC. Container times come from the H100 line (÷ $3.95/h) for the learner and from the
registry for the actors.

| app | CPU | memory | H100 | egress | total |
|---|---|---|---|---|---|
| smoke-1 (3 × 32 proc, 11.5 min of learner) `ap-yJ9v74…` | $1.027 | $0.298 | $0.766 | $0.220 | **$2.31** |
| smoke-2 (3 × 16 proc, 8.6 min) `ap-uLnfzX…` | $0.460 | $0.157 | $0.565 | $0.081 | **$1.26** |

**CPU is billed by use above the reservation, per busy vCPU.** smoke-1 billed 21.8 core-hours. Take out the
learner's 16 reserved cores (3.1 core-h) and the 0.25-core orchestrator, and **each 32-process actor container
billed ~31.8 cores**, not its 16. smoke-2's 16-process containers billed ~17.3 cores, matching the overnight run's
~7% over reservation. Memory billed at the reservations (~42 and ~24 GiB per container).

| per actor container | billed $/h (CPU + memory) | samples/s | $ per M samples |
|---|---|---|---|
| overnight: 16 proc, older engine | ~$1.00 | 7.4k | $0.038 |
| 16 proc, today's engine | ~$1.01 | 11.5k (fast hosts) / 10.3k (all 3) | $0.024 / $0.027 |
| 32 proc, today's engine | ~$1.84 | 19.4k | $0.026 |

So lever 1 is a wash per sample. Its value is fewer containers per learner: ~7-8 instead of ~13, so fewer
preemptions, connections and relay fetches. The speed gain per dollar since the overnight run (−30% actor $/sample)
comes from the Rust 10x engine on `main`.

**Egress = the upload.** smoke-1 uploaded ~5.6 GiB (learner `received_bytes`, 29.4 GiB/h × 0.19 h). At $0.04/GiB
that is $0.22, the whole egress line. The weight bytes (1.24 GiB/h) are lost in rounding. Per sample: 151 B →
**$0.0056 per M samples**, against $0.022 per M for the overnight run's total egress ($50.34 / 2.287 B samples).
That splits into ~$0.0063 of upload at zstd 1 and the rest the broadcast.

## Ready for a big run? What fleet size?

**Yes.** The levers are exact, gated, and ran clean with real actors. What the smoke could not show is the
learner at full load: both arms were actor-bound. The bookkeeping fix should take the ceiling from ~117k to
~131k samples/s, because the 0.14 s + checkpoint gap is gone (median 0.3 ms). The A6 assembler (~2.0 s of work
per 1.66 s epoch under load) is the next wall, so expect 125-135k.

**Fleet for one H100: 8 containers × 32 processes (cpu=16, 40 GiB, 24 games).**

- 8 × ~19.4k ≈ 155k of supply, ~15% above the ceiling. That covers one slow host (the 1.9× host lottery: one of
  the three 16-process containers here ran at 70% of the others) or a preempted container. Seven would leave no
  margin.
- Because CPU bills per busy vCPU, the surplus is cheap. Actors that wait on a full inbox are idle, and an idle
  container bills only its 16-core floor.
- Expected cost at ~131k: learner $5.2 + actor CPU ~$10.1 (busy share) + actor memory $2.7 + egress ~$2.9
  (uploads $2.65, relay $0.3) ≈ **$21/h ≈ $0.044 per M samples** (overnight: $0.077). 12-13 × 16 processes comes
  out the same (~$21/h) with twice the containers, so 32 stays the default.
- Not measured: the learner's own CPU line at full load. It shows 32 CPUs on a 16-core reservation, and its
  receiver and copy threads may bill above 16 cores (at most ~$0.8/h more).
- Launch from this branch (it carries the replay guard, bcc5279) or after merging it to `main`. Defaults are
  already 8-container-ready: `--actor-cpu 16`, processes = all CPUs, 24 games, 40 GiB.

## Spend

Ledger `runs/levers` (cap $15), spend from the settled billing rows:

| item | spend |
|---|---|
| upload codec benches (capture + 2 runs) | $0.09 |
| i6pn probe | $0.03 |
| smoke-1 (3 × 32) | $2.31 |
| smoke-2 (3 × 16) | $1.26 |
| `run_tests` (4 batches, not ledgered) | ~$0.35 |
| **total** | **~$4.0** (smoke cap was $8) |

## How to reproduce

```bash
modal run --env staging experiments/modal_distributed_rl.py::tests --paths tests/test_levers.py,tests/test_distributed_rl.py,tests/test_a6_assembly.py,tests/test_rust_engine.py --parallel
modal run --env staging experiments/modal_distributed_rl.py::upload_codec --parent /results/distributed/scratch-overnight-1/final.npz --config-json '{...}'
modal run --env staging experiments/modal_i6pn_probe.py --arms us,none --clients 4 --mb 1024
modal run --detach --env staging experiments/modal_distributed_rl.py::launch --run-id levers-smoke-1 \
  --parent /results/distributed/scratch-overnight-1/final.npz --config-json "$(cat smoke-config.json)" \
  --actors 3 --actor-cpu 16 --games 24 --seconds 660 --ledger runs/levers \
  --learner-env-json '{"POLYTOPIA_MATMUL_PRECISION":"DEFAULT","POLYTOPIA_HEAD_PRECISION":"HIGHEST"}'
python scripts/levers_smoke_monitor.py --summarize runs/levers/levers-smoke-1-monitor.jsonl --skip 240
modal billing report --start 2026-10-22 -r h --json --show-resources
```
