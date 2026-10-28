# Hero runs, night of Oct 27–28: where the samples/s went

Two runs, both launched from this worktree (branch `claude/gpu-inference`, uncommitted learner and server code):

| | hero-1v1-cont1h | hero-ffa4 |
|---|---|---|
| Task | 1v1 14x14, continuation of hero-1v1 | 4p 18x18 FFA, from random weights |
| Learner | 4x H100 + 32 CPU | 4x H200 + 32 CPU |
| Actors | 19 A10G containers x 48 procs | 6 A10G containers x 48 procs |
| Batch | 262,144 (stream_views) | 131,072 (coded_views) |
| Ran | 7.46 h (cancelled 06:01 CST, ~32 min early) | 7.24 h (stopped ~06:00 CST, ~45 min early) |
| Samples | 3.05 B (total 6.95 B with parent) | 1.51 B |
| Consumed rate | **114k/s** average, 120–125k/s steady | **58k/s** average, 56–62k/s steady |
| Ledger $/h | $67.66 | $35.92 |
| $/B samples | **~$165** (plan: $81) | **~$172** (plan: $93) |

Both runs cost about twice the plan's $/B. The two runs are limited by different things. 1v1 is limited by the
learner's batch assembler. 4p is limited by the actors' inference-server loop.

Data: `training.jsonl` from `/results/distributed/<run>/` (11,650 and 11,550 update rows) and the A10G inference
servers' `stage: server` log lines. Steady state means updates after 200.

## hero-1v1-cont1h: learner-bound in the assembler, with surplus actors

Per update (median, 2.19 s for 262k rows):

| phase | seconds |
|---|---|
| GPU train (4x H100) | 0.73 → 357k/s capacity, **33% busy** |
| learner waits for the next batch (`ready_wait`) | 1.44 (67% of wall time) |
| assembler total | 2.19 → **120k/s cap** |
| · waiting on the inbox | 0.007 (never starved) |
| · stream views to the devices | 0.53 |
| · views finish + value pass + GAE + normalize | 0.33 |
| · **untimed, inside the dequeue loop** | **1.30 (2.15 ms per message, ~600 messages)** |
| receiver zstd decompress, summed over threads | 2.2 (releases the GIL; 32 cores) |

- The inbox is pinned at its 128-message cap for the whole run. 931 connections are open: 912 actor processes
  connect directly, and 19 relays carry weights only. So 931 receiver threads are blocked in `inbox.put`.
- The actors produce more than the learner consumes: 145k/s received vs 120k/s used. The surplus waits in socket
  buffers until it is too old: mean lag 8.0 versions against `max_lag` 10, and **19% of received rows are dropped as
  stale**. The A10G servers sit idle 10% of the time, at 68 lanes per call, because the actors are blocked on sends.
- GPU type does not matter here. hero-1v1 on 4x B200 hit the same wall at ~135k/s.
- The untimed 1.3 s is the whole gap. In ffa4 the same loop costs 0.18 ms per message. Two suspects, and a profile
  will tell which one it is:
  1. **GIL convoy.** With the inbox full, every `get` wakes one of hundreds of runnable receiver threads, and the
     assembler waits up to the 5 ms switch interval to get the GIL back. ffa4 has 296 connections and an empty
     inbox, and does not show this.
  2. **Stale filtering copies the views.** `batch = {k: v[fresh] ...}` copies every key, the dense views included,
     for each partly stale message before `rows.add` pops the views. At 19% stale rows that is a lot of
     GIL-held memcpy.

### Levers, best first
1. **Get the assembler from 2.2 s to ~0.9 s per batch (code change, ~2.5x samples/s).** First profile it: a 15-min
   Modal smoke with `py-spy dump`/`record --gil` on the learner, or `assembly_bench` with ~900 connections. Then:
   - Filter stale rows by index after streaming, not by copying views (`DeviceRows` takes the `fresh` mask). This
     is small.
   - Move data to the per-container relay: 19 data connections carrying 48x bigger messages instead of 912. This
     removes the thread convoy and per-message overhead at its source. It reuses the relay that already exists for
     weights.
   - Or move receive, decompress and unpickle into 2–4 worker processes that write into shared host pages. The
     assembler then only does bookkeeping.

   With ~0.9 s of assembly overlapped with 0.73 s of train, the 4x H100 learner gives **~290–300k/s**. The current
   19 containers can feed that if they reach ~16k/s each; the fleet smoke measured 17.4k/s. That works out to
   **~$63/B, down from $165**, and 3 B samples take ~3 h instead of 7.5 h.
2. **Right-size the fleet now (no code).** While the learner is capped at 120k/s, 19 containers are about 7 too
   many. The extra ones only make stale rows. 12 containers would cut the cost from $67.7/h to ~$49/h (**~$114/B**)
   at the same speed, with lower lag. Check it with a 15-min run at 12 actors: consumed rate should stay ≥115k/s.
   Do this only if the assembler fix isn't ready before the next run.
3. Smaller items: a smaller inbox and bigger `send_samples` cut the per-message count; `checkpoint_every` 50 → 100
   (146 bookkeeping stalls, <0.2% of time); keep the learner on H100, since B200 bought nothing here.

## hero-ffa4: actor-bound; the learner sits idle 60% of the time

Per update (median, 2.22 s for 131k rows):

| phase | seconds |
|---|---|
| GPU train (4x H200) | 0.87 → **151k/s capacity**, 39% busy |
| assembler, not waiting | 0.75 → 176k/s capacity |
| assembler waiting on the inbox | **1.46** (inbox empty) |
| stale dropped | 4% (lag 4.3) |

- The actors deliver **10.3k samples/s per container**, half of the 19.1k/s the fleet smoke measured. The fleet
  was sized at 6 containers from that smoke figure and a 93k/s learner bench. With the real per-container rate, it
  delivers 62k/s.
- The cap inside each container is the **single-process inference server loop**, not the GPU. The server was idle
  0.6% of the time. Its time split: numpy gather (`np.concatenate` of views and masks from 48 shared-memory slots)
  **50%**, dispatch 13%, finish/device wait 13–19%, the rest loop and pipe overhead. Gather costs **44 µs/lane**,
  vs 20 µs in the smoke. The lanes are big: the 18x18 four-seat view plus packed masks, mostly the two 324x324 bit
  planes, come to ~60 KB per lane. The server process also shares 32 vCPUs with 48 busy actor processes.

### Levers, best first
1. **More actor containers (no code, 2.3x).** 6 → 14 containers gives ~145k/s, close to the learner's 151k/s train
   cap. Cost goes $35.9/h → ~$57/h, but $/B goes **$172 → ~$109**. The learner is a fixed ~$20/h that currently
   idles 60% of the time. If A10G capacity is short, mix in L4 containers.
2. **Fix the server gather (code, up to ~1.8x per container).** Any of these, cheapest first:
   - Run two server processes per container, each serving 24 actors on the same GPU. GPU device wait is only
     ~12–19% of server time.
   - Do the gather in a thread pool or with `np.take`/`copyto` into a preallocated pinned batch buffer, with no
     per-call `concatenate` and `repeat` padding.
   - Zero-copy: actors write their requests straight into a server-owned contiguous arena, so a full batch needs no
     gather at all.

   At the smoke's 19k/s per container, **8 containers saturate the learner: ~150k/s at ~$41/h ≈ $76/B**.
3. **After that, the learner sets the limit (151k/s).** The next levers are the 0.39 s/update learner value pass
   (it is already in the assembler, so this matters only once the actors are fixed) and a smaller learner. 2x H200
   would serve today's 62k/s at the same speed.

## Summary

| run | binding limit | quickest fix | expected | code fix | expected |
|---|---|---|---|---|---|
| 1v1 | learner assembler, 2.2 s/batch (1.3 s untimed loop overhead) | 19 → 12 actors | same 120k/s, ~$114/B | stale filter by index + data through relays, or multi-process receiver | ~300k/s, ~$63/B |
| 4p | inference-server gather in actor containers | 6 → 14 actors | ~145k/s, ~$109/B | 2 servers per container / zero-copy gather | ~150k/s on 8 actors, ~$76/B |

The sizing rule needs one change: size fleets from per-container rates measured in a real run, not from the fleet
smoke. For 4p that rate is 10.3k/s. For 1v1 it is still unmeasured: the containers ran at 7.9k/s only because the
learner was pushing back, so the 12-actor check doubles as that measurement.

## Fixes and measurements (same day)

Commits on `claude/gpu-inference`: e327e0f (stale index, MSG_WAITALL, server gather and pinning, bench modes) and
b8bb62b (DeviceRows send thread, launch defaults). Tests on Modal: distributed 17/17, view codec 3/3, four-player
13/13, inference server 3/3 (new: mixed parameter sets with two clients). The A6 bitwise equality gate (base + stale)
passes after each learner change.

**Learner (1v1).** The profile showed the untimed loop time came from the stale filter, which copied every partly stale
message's dense views on the assembler thread. Changes:
1. Stale rows are filtered by index. `DeviceRows.add(views, keep)` gathers the kept rows straight into the host page
   with `np.take` on the copy threads: one copy, off the GIL.
2. Frames are read with `recv_into` + `MSG_WAITALL`: one GIL round trip per message instead of one per ~64 KB
   network chunk across 931 receiver threads. A localhost bench cannot show this; the live smoke can.
3. DeviceRows' page uploads run on a dedicated send thread, in order.

Bench: 4x H100, real hero-1v1 messages, `--lag 2,8` restamps versions so ~18% of rows go stale, as in the run.

| learner code | 96 connections | 912 connections |
|---|---|---|
| as run last night | 180k/s | 165k/s |
| + stale by index | 201k/s | 182k/s |
| + send thread | 214k/s | **221k/s** |

**Inference server (4p actors).** Bench: one A10G container, cpu=16, 48 processes, hero-ffa4 trained weights,
coded views, zstd 7.

| server | samples/s per container | gather per lane |
|---|---|---|
| as run last night | 14.1k | 40 µs |
| preallocated gather buffers (one copy, no padded concat) | 19.5k | 15 µs |
| + `server_cores: 2` (pinned, actors kept off those cores) | **20.3k** | 14.6 µs |
| 2 servers pinned | 21.0k (GPU 75%, device time per lane x3) | 14.1 µs |
| 3 servers pinned | 14.5k | 20 µs |

`server_cores: 2` is now the default in `experiments/hero/launch.sh`. `servers` > 1 stays available for bigger GPUs.

**Live smokes.** 20 min, continued from last night's checkpoints. They ran fixes 1, 2 and the server fix, without the
send thread. Rates are after a 5-min warm-up.

| | last night | smoke | |
|---|---|---|---|
| 1v1, 4x H100 + 19 A10G | 120–125k/s | **208k/s** (+70%); received 259k/s = 13.6k per container; untimed loop 1.30 → 0.31 s | stale 20%: ~17 containers are enough |
| 4p, 4x H200 + 8 A10G (night: 6) | 58k/s | **92k/s** (+58%); 14.7k per container (+42%) | now GPU-bound: value pass 0.87 s + train 1.13 s per batch |

Cost of the day's checks: smokes ~$63 (ledger runs/hero-fix), learner benches ~$30, server arms ~$3.

### What is left
- **1v1:** with the send thread the bench says ~220k/s. The assembler (1.2 s) is still the limit; GPU train
  alone could do 330k/s. The next lever is the host→GPU upload itself (pinned host pages, or coded views for 1v1
  as 4p already uses). Launch defaults: 4x H100, 17 actors.
- **4p:** the 4x H200 learner is GPU-bound at ~92k/s. The omniscient critic's value pass (0.87 s/batch) is the
  biggest piece, so the next gain is model-side (a cheaper critic stem, or bf16 in the value pass). Launch
  defaults: 4x H200, 7 actors.
