# Training 10× faster: what that means here, and a plan to get there (2026-10-21)

Goal as stated: reach a given strength in one tenth of the wall-clock. Baseline is `c5-omni-s1`: from c3 to
77% of all games vs c3 (+255 Elo) in 6 h on 1 H100 + 192 vCPUs, 580M samples, 2,212 updates. "10×" means that
strength in ≈ 36 min, or, equivalently, a 6-h run that reaches what today would take 60 h. Vendor-agnostic; every
number below is per core, per GPU or per sample so it transfers to Modal, the Curio cluster or anything else.

Wall-clock to a strength = (samples needed) ÷ (samples per second). Both factors are ours to move, and the plan is
built so each is measured separately against the c5 curve:

| Factor | Today | Plan target | Evidence for the target |
|---|---|---|---|
| samples/s per actor core | 170 (5.9 ms per sample) | 1,000–1,200 (0.8–1.0 ms) | per-phase profile: duplicated masks + duplicated critic + CPU forwards ([throughput-research](throughput-research-2026-10-21.md)) |
| samples/s per learner GPU | 48k (3 epochs + copy) | 160k+ (1 epoch, overlapped copy); 2 GPUs or bf16 beyond | measured epoch 1.46 s; 1-epoch arm neutral at 1 h |
| samples to +255 Elo | 580M | 150–250M | curve shape (52→75% in the first 190M samples, then slow); batch/epoch/opponent levers untested |
| **wall-clock to +255 Elo** | **6 h** | **~40 min** | 5–7× throughput × 2–3× efficiency |

Throughput alone cannot do it: 10× samples/s would need the learner at 270k samples/s (two H100s at 1 epoch, plus a
rewritten ingestion path) and the strength curve says the second 390M samples bought less than the first 190M. The
efficiency half is cheaper and has never been scanned.

## Part 1 — throughput: 5–7× samples per second

### Stage A (week 1): remove duplicated work on the actors, 2.3× on the same cores, exact

From the per-phase profile. All four are testable with the existing bitwise-equality test pattern
(`tests/test_cpu_engine.py`) except the critic move, which changes which critic snapshot scores GAE (the current
one instead of a ≤ 10-update-stale copy; the standard choice).

| Step | Change | ms per 24-game tick (from 141) |
|---|---|---:|
| A1 | Learner-side values: actors stop running the omniscient critic; the learner runs one critic forward over the assembled batch (≈ 0.5 s per 262k on an H100) and computes GAE there. Rows ship the segment boundary's view so the bootstrap value is available | 102 |
| A2 | Trusted-action `core.step` for actors: skip the validity recompute when the action came from this state's legal masks; keep `status` for version/done; keep the full check in evaluation and the public interface, and a sampled 1-in-N full check in actors | 84 |
| A3 | Candidate-tile `build_mask` in `prepare` (rows only for economic, free, own-territory tiles; bucketed like the move rows) | 72 |
| A4 | Pool rivals per process, not per lane: one full-width rival batch on half the ticks instead of a 4-lane call every tick | 62 |

Learner changes that Stage A forces (the learner becomes the wall at 48k/s otherwise):

| Step | Change | Effect |
|---|---|---|
| A5 | 1 epoch per batch + overlap the 1.1 s host→device copy with training (double-buffer) | 5.5 s → ≈ 1.6 s per 262k (≈ 160k samples/s on one H100) |
| A6 | Ingestion workers: N processes decompress + unpickle into shared-memory arrays; the main process slices | 41k rows/s per core, GIL-bound today → 300k+ |
| A7 | One connection per actor node (an in-node aggregator) instead of one per process | fewer threads, fewer small messages |

Pass criterion: re-run the c5 recipe for 1 h on the same fleet; samples/s ≥ 60k; the training-time record vs c3 at
equal *samples* matches c5 within noise (A2–A4 exact; A1 and A5 are recipe changes and get this check).

### Stage B (weeks 2–3): GPU policy inference, to ≈ 1,000 samples/s per core

**What "GPU policy inference" is.** Today every actor process runs the policy network on its own CPU core for its
24 games, 1.1 ms per sample, which after Stage A is half of the remaining 62 ms tick. A GPU runs the same forward
at 7–20 µs per sample when it is given thousands of states at once. The design (SEED RL, Espeholt et al. 2019;
Sample Factory, Petrenko et al. 2020) is: actor processes keep stepping the engine on CPU, but instead of running
the network they hand their states to **one inference server per node** that batches every process's lanes into one
forward on the node's GPU, samples the actions there, and hands back the choice, its log-probability and the path
masks. The actor's per-sample cost drops to the engine (≈ 0.55 ms after Stage A), bookkeeping (0.15 ms) and the
hand-off (≈ 0.1 ms over shared memory). So ≈ 0.8 ms per sample, ≈ 1,200 samples/s per core, about 7× today's 170.

What it needs, concretely:

1. **An in-node server** (one process with the GPU) and a shared-memory ring per actor process: the actor writes
   the packed int16 view (15.3 KB) and the packed legal masks (16.4 KB) for its lanes, the server decodes both on
   the GPU, runs `network.sample` and the rival forward for pool lanes, writes back `(choice, logp, path masks)`.
   At 1,200 samples/s/core × 48 cores that is 1.8 GB/s per node over shared memory: fine in-node, not over a network,
   which is why the server is per node and not central.
2. **Double-buffered actors**: each process holds two lane groups (e.g. 2 × 24 games); while group A's forward is on
   the GPU the process steps group B's engine. Without this the actor idles for the round trip (today's 24-game tick
   is 141 ms of pure CPU, so no idling was possible; after Stage B the engine is ~13 ms per group and the round trip
   would show).
3. **Measured GPU cost** (`scripts/bench_gpu_inference.py`, decode + masks + sample, synthetic inputs):

   | GPU | batch 1,024 | 4,096 | 16,384 | samples/s at 16k | precision |
   |---|---:|---:|---:|---:|---|
   | L4 | 42 ms | 97 ms | 361 ms | 45k | fp32 HIGHEST |
   | H100 | 30 ms | 51 ms | 118 ms | 139k | fp32 HIGHEST |
   | H100 | 38 ms | 36 ms | 84 ms | 195k (233k at 65k batch) | TF32 DEFAULT |
   | L4 | out of memory at 65k; TF32 at ≤16k not completed | | | | TF32 DEFAULT |

   At TF32 one H100 serves ≈ 230k samples/s, i.e. ≈ 190 actor cores at 1,200/s each: one H100-class GPU per
   ~190-core node, or one L4-class (45k/s measured at fp32, ~60–70k expected at TF32) per ~40–50 cores. Two
   details matter for the design: (a) the host→device copy is a third of the call (26 ms of 84 at 16k; 31.7 KB per
   sample, half of it packed masks), so the server should keep the mask unpack and view decode on the GPU and
   write inputs straight from shared memory, and (b) there is a fixed ≈ 35 ms per call regardless of batch (1k and
   4k cost the same), so calls must be large (8–16k lanes) and the actors must keep several lane groups in flight:
   with the engine at ≈ 13 ms per 24-game group after Stage A, a process needs ~4–6 groups (96–144 games) to hide
   a 60–80 ms round trip. Memory per process is ~2.5 GiB regardless of games, so that is affordable.
4. **Node shape**: 32–64 vCPUs + one small GPU per node, each node self-contained (engine + inference), nodes
   stream rows to the learner. On Modal that is a GPU container with many CPUs; on the Curio head it is one 4090
   per ~60 threads; on any cloud it is the standard "g" instance shape.
5. **Not bitwise**: GPU convolutions change sampled actions slightly versus CPU. The CPU path stays for evaluation
   and tests; training is already stochastic.

After Stage B the actor side delivers ≈ 1,200 samples/s per core: 192 cores → ≈ 230k/s (8.5×); the learner must be
at 1 epoch on 2 GPUs (or bf16 trunk with the parity gate, measured 1.9× on the SFT trainer) to absorb it. The next
engine wall is the step itself (13 ms per 24 games, `build_mask` for prepare and `core.step`'s apply) and the only
lever past that is a non-XLA step (Numba/C), weeks of work, not in this plan.

## Part 2 — sample efficiency: 2–3× fewer samples to the same strength

The c5 curve says the recipe is far from its efficient frontier: 52→75% in the first 2 h, 75→82% in the next 4 h,
against a *frozen* c3 opponent it beats 85% of the time by the end (the signal saturates). None of the following has
been scanned on the distributed PPO; each is a $30–130 arm with a pre-registered read-out (training-time record vs
frozen c3 per 100 updates, then a 128-map ladder vs c3 at matched *samples*, not matched hours).

| # | Lever | Hypothesis | Arm | Cost |
|---|---|---|---|---|
| C1 | **Batch size / update count.** 262,144 samples per update is very large for a 500k-parameter policy (OpenAI Five's 1M+ was for 20k-step games). Today's 6 h = 2,212 updates. If the critical batch is nearer 65k, the same samples give 4× the updates | strength at matched samples rises with smaller batch until noise dominates | batch 65,536 and 131,072 vs 262,144, same fleet, 2 h each, lr held then scaled √(batch) | $45 × 3 |
| C2 | **1 epoch + top-25% \|A\| + EMA at 6 h.** Neutral at 1 h on 1/3 the learner work; also gates Stage A5 | neutral at 6 h too | one 6-h arm | $130 |
| C3 | **Opponent signal.** Training record vs c3 saturates at 85% of decisive; self-play share 50% + historical 15% + c3 5%. Learning against a target you already beat 85% of the time wastes samples | PFSP over the historical pool (weight by loss rate) raises the slope after hour 2 | 3-h arm from the c5 3-h snapshot, resumed with critic | $70 |
| C4 | **Resume with critic and Adam** (c6 showed a fresh critic costs more than an hour of training). Needed for any multi-stage schedule, and it removes the 60-update warm-up (3% of a 6-h run, 11% of a 1-h one) | continuation holds ≥ parent at 1 h | already partly built (`Learner._resume`); 1-h check | $25 |
| C5 | **Turn clock γ = 1.0 / λ 0.95, advantage floor 0.01**, clean paired test from c3 (c6: +8–10 pts over per-decision discount, confounded by the floor) | +5–10 pts at matched samples | 2 × 2-h paired arms | $90 |
| C6 | **Learning-rate scan** once C1 picks the batch (2e-4 today; never scanned here) | 2× lr holds at smaller batch | 2 arms | $90 |
| C7 | **Auxiliary critic targets** (KataGo: final ownership per tile, final score margin) on the omniscient critic: a denser learning signal for the value, which is the lever that worked | faster critic convergence, steeper early curve | 2 days of code + 1 arm | $45 |
| C8 | **Search-improved targets done right** (soft targets from the leak-free d=1 searcher, mixed into the PPO loss, not hard-label distillation; the earlier failure was the hard-label form) | +Elo per sample at 2× actor cost per sample | after Stage B only (search needs the throughput) | later |

C1 and C3 are the two I would bet on; C2 and C4 are prerequisites for the throughput work and cost almost nothing;
C5/C6 are tuning. The literature anchor for C1/C6 is McCandlish et al. 2018 (critical batch size scales with the
gradient-noise scale, which is small early in training and grows later), which also predicts that the *right* batch
schedule starts small and grows, something a fixed 262k batch cannot do.

## Part 3 — order, schedule, budget

Two tracks in parallel; the efficiency arms use the existing stack and do not wait for the engineering.

| Week | Engineering (Stage A/B) | Efficiency arms (Part 2) | Spend |
|---|---|---|---|
| 1 | A1–A4 with bitwise tests; A5 (1 epoch + overlapped copy) behind a flag; 1-h parity run | C4 resume check; C1 batch scan (3 × 2 h); C2 6-h 1-epoch arm | ≈ $350 |
| 2 | A6–A7 ingestion; Stage B server + double-buffered actors on one node; TF32 inference re-measure | C3 PFSP arm; C5 turn-clock paired test | ≈ $300 |
| 3 | Stage B on the full fleet; learner at 2 GPUs or bf16 with parity gate; 1-h throughput run ≥ 200k samples/s | C6 lr scan; C7 auxiliary heads if C1/C3 land | ≈ $350 |
| 4 | **Verification run**: the c5 recipe + every lever that passed, 1 h, same fleet. Pass = ≥ 77% vs c3 at 1 h (what took 6 h), i.e. ≥ 6×; stretch 10× = 77% at 36 min | | ≈ $150 |

Total ≈ $1,150 of compute over four weeks; the engineering is roughly three person-weeks.

## Part 4 — what to expect and what could stop it

- **Most likely outcome**: 5–7× throughput from Stages A+B (the arithmetic is measured), 1.5–3× from C1/C3/C5
  (hypotheses), landing at 8–15× on "time to +255 Elo". The verification run in week 4 is the only number that counts.
- **Risk 1, staleness**: at 230k samples/s and 1 epoch the policy updates every ~1.2 s; actors on a 24-lane tick see
  weights 2–3 updates old (today 1.5). PPO tolerates this; V-trace corrections exist if it does not.
- **Risk 2, the efficiency half fails** (C1 shows 262k is already near-critical, PFSP does not steepen the curve):
  then 10× needs 10× throughput, which means two learner GPUs, the ingestion rewrite and ~400 actor cores. Still
  feasible, roughly 2× the hourly cost.
- **Risk 3, GPU inference is slower than the learner-epoch arithmetic suggests**: the TF32 re-measurement (§5)
  settles it before any server code is written.
- **Risk 4, strength measured against c3 stops being informative** once the bot wins 90%+: switch the ladder to the
  c5 final as the reference and keep c3 only for continuity.

## 5. Open measurements (cheap, before week 2)

1. TF32 GPU inference on an L4 at ≤ 16k batch, and the fixed ~35 ms per call (profile the decode + sample graph; it should be < 5 ms).
2. `build_mask` internals (which matmul is the 15 ms) — decides how small A3 can go.
3. One batch-scan arm (C1 at 65,536) is the first thing to run on the existing stack: it answers the biggest
   unknown in this plan for $45.

References: Espeholt et al. 2019 (SEED RL); Petrenko et al. 2020 (Sample Factory); Espeholt et al. 2018 (IMPALA,
V-trace); McCandlish et al. 2018 (An Empirical Model of Large-Batch Training); Berner et al. 2019 (OpenAI Five);
Vinyals et al. 2019 (AlphaStar, PFSP); Wu 2019 (KataGo, auxiliary targets); Hessel et al. 2021 (Podracer).
