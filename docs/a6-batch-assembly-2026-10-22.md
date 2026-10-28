# A6: learner batch assembly off the training path, 2026-10-22

**Result:** the learner's host-side batch assembly is no longer the training wall. On one H100 with the c5-omni
recipe (epochs 1, batch 262,144), the learner now absorbs **116.9k samples/s** when fed without limit from real
actor messages, against **59.2k** for the old path in the same container (44.8–49.1k with the old wire format, the
code on main). The GPU trains 75% of each update (1.68 of 2.24 s), against 26% on main. The new path's device
training batch and batch statistics are **bitwise equal** to the old path's: on CPU (incl. 2 devices) and on the
H100 with real 262k-row batches. The 6-actor H100 smoke is now **actor-bound**: 43.8k samples/s steady (42.9k over the whole 20 min, vs 41.1k / 39.2k for main), the inbox empty, and the learner idle 4.8 s per update waiting for actor data. **About 16 actor containers would saturate the new learner** (~117k at ~7.3k samples/s per container).

Branch `claude/a6-batch-assembly-018d0c` (not merged). Ledger `runs/a6/budget.json` (copy `docs/a6/budget.json`):
~$9.0 of the $13 folder cap ($4.8 smoke, estimated at launcher rates × 1.14 until Modal billing settles; $4.1 benches; $0.1 capture).

## What was slow (profile of main 7f66add)

Per-phase timings now land in every update row (`asm_*_seconds`, `receiver_*_seconds`, `device_batch_seconds`).
Bench: `assembly_bench` (below), same H100 container, unthrottled replay of 64 real actor messages, ~487 messages
per batch:

| phase, per 262k-row update | main (bench-2 host path) | A6 (bench-3) |
|---|---|---|
| update | 5.86 s | **2.24 s** |
| learner waiting for the next batch (`ready_wait`) | 3.99 s | **0.56 s** |
| GPU training (`train_seconds`, incl. contention with the value pass) | 1.85 s | 1.68 s |
| host concatenation of every field (8.6 GB of views) | 2.64 s | 0.01 s (small fields only) |
| views to the GPU (`_put_views` / page copies + puts) | 0.20 s + the put's tail inside the value pass | 0.90 s page copies (4 threads) + 0.37 s last page / transfer tail |
| critic value pass + fetch (`asm_value`) | 1.16 s | 0.60 s |
| segment GAE + normalization | 0.02 s | 0.02 s |
| receiver unpickle (GIL held, summed over 96 threads) | 1.61–1.91 s | **0.06 s** |
| receiver zstd decompress (GIL released, summed) | 2.1–12.8 s | 2.5 s |
| samples/s | 44.8k | **116.9k** |

Findings:

- **The host concatenation was the biggest single cost**: `np.concatenate` of ~487 messages into a fresh 8.6 GB
  array per batch (both packed views, 7,649 int16 each per row), first-touching 2 M fresh pages, on the assembler
  thread, while 96 receiver threads contended for the GIL.
- **The pageable put is cheaper than expected** (0.2 s issue + ~1 s of transfer hidden in the value pass): it was
  not the wall the hypothesis said.
- **Receiver unpickling held the GIL for 1.6–1.9 s per update**: in-band protocol-5 pickles copy every array out
  of the decompressed bytes. That starves the assembler thread and would cap ingestion at ~140k samples/s by itself.
- Messages are ~512–640 rows (`send_samples` 512 is the config default; the actor's 2048 fallback never applies),
  so a batch is ~487 messages, not 128.
- The value pass, GAE and normalization were never the problem (GAE 10–20 ms).

## What changed (exact)

1. **`DeviceRows` (stream_views=True, default)**: with `learner_values`, each dequeued message's `view`/`opp_view`
   (and the segment table's boundary views) are copied into reusable host pages of `VALUE_CHUNK` = 8,192 rows
   (already touched, so no page faults) by a 4-thread copy pool (`assembly_threads`; numpy copies release the
   GIL). A full page goes to the GPU as one async `device_put` plus one donated fixed-shape
   `dynamic_update_slice` into a per-device buffer, so uploads overlap the wait for the next messages and the
   previous batch's training, and there are ~2 dispatches per page instead of thousands. Page j lives on device
   j % D; each device's buffer is its pages in order, so every value-pass chunk is exactly one page and holds the
   same rows as the host path's chunk. `position()` maps rows to that layout (the identity on one GPU); the
   training gather and the value fetch go through it. Host pages return to the pool after the value pass.
   `stream_views=False` keeps the old host path as the reference.
2. **Wire kind 2**: `send_msg` now compresses `pickle.dumps(protocol=5, buffer_callback=...)` plus the raw
   out-of-band buffers with an index, so the learner's numpy arrays are read-only views of the one decompressed
   bytes object. Kinds 0 (uncompressed replies) and 1 (old frames, e.g. captured files) still decode.
3. Per-phase timings in every update row; `scripts/distributed_status.py` prints them and no longer crashes on
   the Rust engine's `first_compile_seconds='rust'`.

First try (bench-1, kept in the history): writing each message straight to the device in power-of-two pieces (no
host copy at all). Exact, but ~2,400 puts + jitted writes per batch under GIL contention cost 4.35 s: only 54k.
The page design replaced it.

## Verification gate (passed)

`assembly_bench.gate`: the same message sequence through the host path and the streamed path of one `Learner`;
every field of the device training batch compared bitwise (floats as their bit patterns) plus sharding, and every
non-timing `info` statistic compared for identity.

- **CPU** (`tests/test_a6_assembly.py`, VALUE_CHUNK patched to 64 so batches span pages and messages straddle page
  boundaries): base, turn clock (`turn_gamma`/`turn_lambda`), UPGO + PBRS `phi`, stale drops (`max_lag`),
  `advantage_keep` < 1 (subset gather), the policy's own value head (no omniscient critic), all also **on 2
  devices** (forced host devices, pages alternate between devices); page growth and reuse with stale tails zeroed;
  wire kinds 0/1/2 round trip and zero-copy views.
- **H100, real messages, 262,144-row batches** (all 3 benches, every code version): base (487 messages, 8,206
  segments), stale (997 messages, 274,537 stale rows dropped), turn clock, UPGO + phi: **all equal**.
- Test files on Modal (`run_tests`): `test_a6_assembly.py` 4, `test_distributed_rl.py` 15, `test_rl_18x18.py` 15,
  `test_rust_engine.py` 1: **35 passed** on the final code.

## Benches

`modal run experiments/modal_distributed_rl.py::assembly_capture --config-json <smoke config + parent>` writes 64
real data messages (Rust actors on a CPU container against an untrained learner, first ~20 s of play) to
`/results/a6/messages`. `::assembly_bench` loads them in one H100 container (16 vCPU, 64 GiB, the learner's spec),
runs the gate, then `Learner.run` per variant for 240 s with 6 client processes × 16 connections replaying the
messages over localhost (re-framed in the current wire format; unthrottled, so the inbox stays full). Results:
`docs/a6/bench-{1,2,3}.json` (summaries + every update row).

| bench | code | host path samples/s | streamed samples/s |
|---|---|---|---|
| 1 | per-piece device writes | 49.1k | 54.0k |
| 2 | host pages, 1 copy thread | 44.8k | 68.2k |
| 3 | host pages, 4 threads, wire kind 2 | 59.2k (wire only) | **116.9k** |

Caveat: the captured messages are early-game positions, which compress ~200×; mid-game views compress less, so
real decompress time per message is higher (it releases the GIL and runs on 96 threads, so it should not
bind before the GPU does).

## Smoke: a6-h100-smoke

Same config as `e2e-h100-smoke` (c5-omni recipe from `c5-omni-s1-final` + its critic, epochs 1, seed 402, 1×H100,
6 actor containers × 16 vCPU / 24 GiB, 16 processes × 24 games, 1200 s, TF32 convs + fp32 heads), on this branch
(commit 2af8865). Training log `docs/a6/a6-h100-smoke-training.json`.

| updates | samples/s | s/update | learner waits | of which waiting for actor data | own assembly work | GPU train | inbox |
|---|---|---|---|---|---|---|---|
| 1–25 | 38.2k | 6.60 | 4.82 | 4.57 | 2.21 (first-batch compiles) | 1.77 | 8 |
| 26–50 | 47.5k | 5.26 | 3.60 | 4.29 | 1.22 | 1.65 | 0.1 |
| 51–100 | 41.5–46.9k | 5.30–5.41 | 3.64–3.75 | 4.47–5.08 | 1.23 | 1.65 | 0.2–0.4 |
| 101–197 | 42.2–44.6k | 5.62–5.89 | 3.96–4.22 | 4.78–5.08 | 1.21–1.22 | 1.65 | 0–6 |
| **steady (21–197)** | **43.8k** | 5.55 | 3.89 | 4.76 | 1.22 | 1.65 | 0.8 |

(`asm_wait` is the assembler blocked on the empty inbox; it overlaps training, so it exceeds `ready_wait`.)

- **The 6 actors are now the limit.** main's smoke had a 22–58-message backlog while the learner assembled for
  4.3 s; here the inbox is empty and the learner's own work per batch is 1.22 s (0.64 s page copies, 0.37 s value
  pass, the rest bookkeeping), under the 1.65 s of training it overlaps with.
- Actors delivered 41.8–48.7k samples/s, ~7.3k per container (two of the six were ~50% faster than the slowest, the
  same host variation as before). Receiver unpickle 0.06 s/update under live traffic; 0 stale rows; mean lag 1.1.
- **Fleet sizing:** the learner absorbs ~117k (bench) to ~159k (training alone, if assembly overlapped perfectly),
  so **~16 containers of 16 vCPU saturate it**; at ~$0.95/h (launcher) per container that is ~$20/h for the whole
  run at ~$0.05/M samples vs ~$0.07/M here and $0.09/M on main.
- No orchestrator replay (one set of function calls in the registry), 96/96 processes connected, KL 0.004, clip 0.046.

## What's left

- At 117k the remaining assembly wait is ~0.56 s/update: the page copies (0.9 s, memory bandwidth) and the value
  pass sharing the GPU with training (0.6 s here vs 0.36 s alone). Levers if a bigger fleet needs them: pinned host
  pages (skip PJRT's staging copy), decompressing straight into pages, more copy threads. Above ~150k the H100 is
  the wall (1.62 s/epoch alone); then a second GPU on the learner node or bf16.
- Ingestion is still one process: zstd decompression scales across threads, but every Python step per message
  still takes the GIL. A per-node aggregator (A7) or receiver processes become worth it past ~150k.
