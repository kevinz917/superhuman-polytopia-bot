# What limits samples per second in distributed PPO, and what removes it (2026-10-21)

Vendor-agnostic audit of the training loop that produced `c5-omni-s1` (14×14, omniscient critic, 1 GPU learner +
192 actor vCPUs, 27k samples/s). Sources: a code read of `rl/distributed.py`, `rl/cpu_engine.py`, `rl/network.py`,
`gpu/core.py`, `gpu/actions.py`, `gpu/structures.py`; the run's own timing rows (summarised in
rl-sampling-bottlenecks-2026-10-21.md); and three new measurements run today
on Modal staging (≈ $2): a per-phase split of one actor tick (`scripts/bench_tick_phases.py`,
`experiments/modal_tick_phases.py`), the learner's ingestion cost per row (`scripts/bench_ingest.py`), and the H100
epoch at three minibatch sizes (`modal_distributed_rl.py::bench_learner`). Nothing in the training path was changed.

## 1. The answer in one paragraph

One actor core spends ~5.9 ms per sample. Measured today, 3.6 ms of that is neural-network forwards on the CPU and
2.0 ms is the engine, and **inside the engine, two thirds is the same whole-board build mask computed twice per
tick**: once in `prepare` to give the policy its legal moves, and again inside `core.step`, which re-validates the
chosen action by recomputing every mask from scratch. Of the 3.6 ms of forwards, 1.6 ms is the omniscient critic,
which the learner recomputes anyway (its value is used only for GAE on the actor), and 0.7 ms is the pool rival,
which runs as a 4-lane call for 2–3 lanes. None of these is a hardware limit. Removing the duplicated work takes
the tick from 141 ms to ≈ 60 ms (2.3×) with no accuracy change and no new hardware; the next walls after that are
the learner's single-process ingestion (≈ 40k rows/s per core, GIL-bound) and the 3-epoch learner (48k samples/s on
one H100), both of which have cheap fixes. Beyond ≈ 4× the policy forward itself is what remains on the actor, and
that is where accelerator inference or a faster board representation would come in.

## 2. Where one tick goes today (measured)

One pinned core, 24 games per process, 14×14, template weights, mid-game states (mean turn 44–48). Median of 15
repeats, ms per 24-game tick. "16 procs" = the same measurement with 16 processes on a 16-vCPU container (the
production shape), which exposes memory-bandwidth contention. Host: AVX2 only, no AVX-512 (model hidden).

| Phase | 1 proc | 16 procs | What it is |
|---|---:|---:|---|
| **engine tick (production, fast=True, both views)** | **47.1** | **48.4** | `CpuGames.tick`: step + auto-reset + prepare |
| · `core.step` alone (lax.map over 24 games) | 17.9 | 17.8 | decode + validity + apply one action |
| · · masks `core.step` recomputes for validity (`validity_like`) | 17.9 | 17.0 | build, summon, harvest, forest, reward, capture, destroy, research + 1 move row + 1 attack row |
| · · · of which whole-board `structures.build_mask` | 15.3 | 15.4 | [196 tiles × 128 types], every step |
| · `prepare` legal masks (fast path, bucketed move rows) | 23.2 | 21.1 | the masks the policy samples from |
| · · bucketed move rows | 5.1 | 5.1 | vs 256 ms for the unbucketed frozen version |
| · · attack rows (all 196 sources) | 0.8 | 0.8 | |
| · · build mask (again) | 15.3 | 15.4 | |
| · observe + encode (own view) | 1.6 | 1.4 | |
| · observe + pack both int16 views | 1.0 | 1.0 | |
| engine tick, frozen engine (fast=False) | 262 | 258 | what evaluation uses; the proposals already buy 5.5× here |
| **policy forward, 24 lanes (sample)** | **27.3** | **26.6** | 1.1 ms per sample |
| · trunk only | 28.2 | 28.3 | the heads are free; it is all 3×3 convs |
| · with the production `jnp.take` gather | 28.4 | 30.6 | |
| · with `--xla_cpu_enable_fast_math` | 25.7 | — | 6%, not worth the numerics |
| **omniscient critic, 24 lanes** | **33.6** | **42.9** | 1.4 ms/sample alone; +28% under 16-way contention |
| · opponent-view int16 → features decode | 1.4 | 1.5 | small; the earlier guess that this was expensive was wrong |
| host copy of the packed view | 0.02 | 0.02 | |

The production profile from the run (rl-sampling-bottlenecks §2) adds the
rival forward (15.9 ms on a 4-lane bucket for ~2.5 lanes) and bookkeeping (3.5 ms), for 141 ms per tick.

Three things the split shows that the earlier phase profile could not:

1. **`core.step` is its own validity check.** `step` and `validity_like` cost the same 17.9 ms: applying the action
   is nearly free on the branching CPU build (only the taken branch runs), and the whole step is the
   `valid = ... | (build & build_mask(state)[dst, param]) | ...` expression, which evaluates every whole-board mask
   eagerly (it is a boolean OR, not a `lax.cond`). The actor already sampled that action from `prepare`'s masks on
   the identical state, so this is a second, redundant computation of the same masks. Evaluation and the public
   interface need it; the actor loop does not.
2. **`build_mask` is 15 ms per tick, twice.** That is 30 of the 47 ms engine tick (65%) and 21% of the whole
   production tick. It is a [196 × 128] mask with three integer matmuls (`territory @ one_hot(structure)`,
   `members.T @ (~present | ~per_city)`, `neighbors_own @ adjacent`), monument conditions and the bridge check,
   computed for all 196 tiles although at most a handful are buildable (economic, free, in own territory). The
   float32-product proposal is already applied in the fast path and it is still 15 ms; a candidate-tile version
   (compute rows only for `econ & free & in_territory` tiles, zeros elsewhere) is the bitwise-equal fix.
3. **The critic forward is the most expensive thing on the actor and it is duplicated work.** The learner trains
   the critic on `full_features` from the same rows (`to_batch` decodes both views), so the only use of the actor's
   value is GAE and the truncation bootstrap. It also runs on all 24 lanes every tick, then is indexed to the
   learner's lanes; and it suffers the largest contention penalty of any phase (its doubled-channel stem is the
   most memory-bound op).

What is *not* a lever on the actor (measured today or earlier): XLA fast-math (6%), games per process (24 → 48:
0%), the opponent-view decode (1.4 ms), host copies, map generation (2 ms), packing (0.6 ms per tick; 25 µs per row).
The policy forward is at ~50 GFLOP/s on an AVX2 core (≈ 50% of fp32 peak); XLA's CPU convs are not the problem.

## 3. The learner and the wire (measured)

**Epoch on one H100** (`bench_learner`, omniscient critic, 262,144 batch, TF32 trunk, synthetic batch):

| minibatch | epoch | samples/s | host→device put |
|---|---:|---:|---:|
| 16,384 (production) | 1.46 s | 180k | 1.1 s |
| 32,768 | 1.62 s | 162k | 1.1 s |
| 65,536 | out of memory | | |

So a 3-epoch update is ≈ 4.4 s + 1.1 s copy = 5.5 s per 262k samples (the run logged 5.78 s), a ceiling of
**≈ 48k samples/s** on one H100 at 3 epochs, 1.75× today's actor rate. Minibatch is not a lever. The copy is 1.1 s
because a sample is 30.6 KB raw (two int16 views of 15.3 KB each): 7.5 GiB per batch.

**Ingestion** (`bench_ingest.py`, real rows, 512-row messages): a message is 15.7 MB raw and 157 KB compressed
(307 B per row, 100×). Decompress + unpickle costs **24 µs per row = 41k rows/s per core**. The learner receives on
one Python thread per actor connection, so this is GIL-bound (zstd releases the GIL, `pickle.loads` of the array
bodies largely does not): expect 1–2 cores' worth, i.e. a wall at **≈ 40–80k samples/s**, right where the actor
fixes below would put us. Assembly is another 1.6 s of `np.concatenate` per 262k batch (in a thread, overlapped
today) plus 1.9 s for the stale-row boolean copy when any chunk has stale rows.

Neither is a wall at 27k/s, which is why the run showed `inbox=0` and the GPU waiting; both become walls at 2–3×.

## 4. The levers, in order

Ranked by samples-per-second gained per unit of work; "exact" means bitwise-identical training data, testable with
the existing `tests/test_cpu_engine.py` equality test pattern.

| # | Lever | Where | Tick ms (from 141) | Cumulative | Exactness / risk | Work |
|---|---|---|---:|---:|---|---|
| 1 | **Drop the critic forward from actors**; learner computes values for GAE when it assembles the batch (one critic forward over 262k samples ≈ 0.5 s on the H100; GAE moves to the learner, which already has segment order and bootstrap rows) | actor + learner | −39 | 102 (1.4×) | Not bitwise: values come from the *current* critic instead of the ≤ 10-update-stale actor copy. That is the standard (and slightly better) choice; AlphaStar/IMPALA compute values learner-side. GAE segments need the next-segment bootstrap value: keep the 32-decision segment but ship the boundary row's view too | 2 days |
| 2 | **Trusted-action step for actors**: a `core.step` variant that skips the validity recompute when the action was drawn from this state's legal masks (keep `status` for version/done checks; keep the full check in evaluation and the public interface) | engine proposal | −18 | 84 (1.7×) | Exact for legal actions by construction; the actor already raises on `status != 0`, so keep a sampled 1-in-N full check in debug mode | 1 day |
| 3 | **Candidate-tile `build_mask`** in `prepare`: compute the [tiles × 128] rows only for economic/free/own-territory tiles (bounded bucket like the move rows), zeros elsewhere | engine proposal | −12 | 72 (2.0×) | Exact; same test pattern as the bucketed move rows | 1–2 days |
| 4 | **Rival games per process, not per lane**: dedicate whole processes (or whole 24-lane blocks) to a pool opponent so the rival forward is one full-width batch on half the ticks instead of a 4-lane call every tick | actor | −10 | 62 (2.3×) | Exact; changes nothing but which lanes host which opponent | 1 day |
| 5 | **Learner: 1 epoch** (ablation-neutral at 1 h; untested at 6 h) and **overlap the 1.1 s batch copy** with training (double-buffer) | learner | learner 5.5 s → ~1.6 s per 262k (≈ 160k samples/s) | | 1 epoch is a training-recipe change: run it as an arm | 1 day |
| 6 | **Learner ingestion in worker processes**: N decoder processes decompress + unpickle into shared-memory arrays; the main process only slices | learner | ingestion 40k → 300k+ rows/s | | Exact | 2 days |
| 7 | **Ship 1 view per row** where possible: the opponent's view at the same tick is the own view of the lane's other seat one decision later; or store both views once per *tick* rather than per row (halves the 30 KB row and the 1.1 s copy) | wire + learner | halves bytes per sample | | Exact but fiddly bookkeeping | 2 days |
| 8 | **Policy inference off the CPU** (GPU inference service, or one small GPU per N actor containers) | actor | −27 of the remaining 62 → ~35 (4×) | | Not exact (different conv kernels change sampled actions slightly; fine for RL, breaks the bitwise tests); adds a network hop per tick | 1–2 weeks |
| 9 | **Non-XLA engine step** (Numba/C for `step` + masks) | engine | engine 47 → ~5 | | Large, weeks (PufferLib note) | weeks |

Levers 1–4 need no new hardware, no recipe change except where noted, and together take the actor from 27k to ≈
62k samples/s on the same 192 vCPUs. At that point lever 5 (1 epoch, overlapped copy) and lever 6 (ingestion
workers) are required, or the learner becomes the wall again at 48k/s and ingestion at ~40–80k/s. Levers 1–6 are
about a week of work and give ≈ 2.3× samples per dollar on any vendor; lever 8 is the only one that needs an
accelerator for inference and is the next 2× after that.

## 5. Why this was invisible before

The run-level numbers (`ready_wait_seconds`, inbox occupancy) said "actor-bound" and the per-phase profile said "three
forwards + engine". Both were right and neither could see inside `core.step` or `prepare`. The duplication only
shows when the step is timed without `prepare` and the validity masks are timed alone: the two numbers coincide, and
`build_mask` appears in both halves. Likewise the critic looked like "a bigger network" rather than "work the learner
repeats". The one earlier hypothesis that did not survive is the opponent-view decode (1.4 ms, not a lever).

## 6. What to measure next (cheap)

- `build_mask` internals (which of the three matmuls / monument conditions / bridge check is the 15 ms); 20 minutes
  on a CPU container with the same script pattern. The candidate-tile fix is right regardless, but this decides
  whether the whole-board version can also be made cheap for evaluation.
- 1 epoch at 6 h (one $130 arm) before adopting lever 5; it gates how much actor throughput one GPU absorbs.
- The contention penalty: the critic forward is +28% slower at 16 processes per 16 vCPUs; after lever 1 the policy
  forward's penalty (+8%) is what remains. 8-vCPU containers measured 1.3–1.66× more samples per actor dollar on
  18×18 for the same reason.

Scripts: `scripts/bench_tick_phases.py`, `scripts/bench_ingest.py`, `experiments/modal_tick_phases.py` (all new,
uncommitted; run from a checkout of `main`, they need the omniscient-critic code). Raw outputs in this session's
scratchpad (`tick_phases.log`, `ingest.log`, `learner_mb*.log`).
