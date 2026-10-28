# Fast leak-free robust turn search (2026-10-19)

Leak-free robust search for c3-maps-ctl u4854 (`TurnSearcher(lanes=16, depth=1, rollout_temperature=1,
replies=64, determinize=True)`, +117 Elo on the proposed-v1 map mix, docs/test-time-search-c3-2026-10-19.md)
cost ≈10.5 s per searched turn on one Modal core. `TurnSearcher(..., parallel=True)` now plays **the same
turns, bit for bit, in 0.95–1.2 s per turn on a 16-core Modal CPU container** (median 0.6–0.8 s), 8–10×
faster. The < 1 s target is met on the median but only on some hosts for the mean: Modal hosts vary, and one
32-core host ran the same code at 1.8 s. A GPU does not help (L4: 14.7 s/turn).

## Where the time went (one pinned Modal core, 8 turn-start positions)

| piece | share | note |
|---|---:|---|
| `plan` (the K=16 lanes + 64 replies) | 98.7% | 2.5 plans/turn after reveals, ≈3,500 lane-steps/turn, ≈3.0 ms per lane-step |
| guarded replay on the true game | 0.8% | 0.08 s/turn |
| belief sample (determinize) | 0.1% | 0.01 s/turn |

Per lane-step, measured as standalone jitted calls: network forward ≈1.3–1.7 ms (5 3×3 convs of width 64 on
14×14, already ≈50 GFLOP/s on one core; batching 16 or 64 lanes on one core only saves 13–20%), legal masks
≈1.9 ms, observe+encode ≈0.3 ms, engine step ≈5 ms (standalone calls include copying the state; inside the
lane loop the whole lane-step is 3.0 ms). So no single piece dominates, and none can be made much cheaper
without changing results. What is slow is the structure: one XLA call steps every lane one after another
(`lax.map`), so a plan costs the *sum* of all ≈1,400 lane-steps on one core.

## What changed

`search.ParallelPlanner` (used by `TurnSearcher(parallel=True, workers=N)`):

- Every lane is its own single-lane while-loop executable built from the same `advance` as the lockstep
  runner (refactored into `_lane_kit`, so the reference planner's code is unchanged), dispatched from a
  thread pool. XLA releases the GIL, so lanes run on separate cores; one-thread async dispatch does not
  run them concurrently (measured 1.0×).
- Each worker thread is pinned to its own CPU (stride 2 when there are spare CPUs). On Modal, 16 unpinned
  threads gave 6.8× over one lane, pinned 13–14× (32 threads: 10× vs 22–24×).
- Lane keys, the open-candidate test, the sequential-halving bookkeeping (scatter-adds, stable argsort of
  survivors) and the final pick are small jits with the same arithmetic as `build_planner`'s `robust_plan`.
- A candidate's first-round reply lanes (all candidates survive round 1) start as soon as that candidate's
  own turn finishes, removing one wait-for-the-slowest barrier (1.02 → 0.95 s/turn on the same code path).
- A plan now costs about the longest chain of stage A + round 1 plus the longest lane of rounds 2–4:
  ≈280 critical lane-steps per turn instead of ≈3,700 total. More than 16 cores cannot help: the next
  phase needs the previous phase's means.

Also: `build_planner(vmap=True)` / `TurnSearcher(vmap=True)` batches the lockstep runner with vmap (for the
GPU measurement), and `search_arena` passes `parallel` / `workers` from the arm config.

## Equivalence

Against the lockstep planner, same sampled worlds and keys, c3 u4854 on proposed-v1 positions (seeds
366001–366003, turns 2–20, both seats), `experiments/modal_search_speed.py::equivalence`:

- 32 single plans (4 runs): actions, chosen lane, per-lane steps and per-lane scores **bitwise identical**.
- 84 whole turns with re-planning (4 runs): resulting game state and every committed plan identical.
- Turn digests (hash of the resulting state and all committed sub-actions) are identical across 16/8/4
  workers, 16 vs 32 cores, XLA Eigen threading on or off, and before/after the round-1 chaining.
- `tests/test_parallel_search.py` (robust+determinized, d=1, root=first with template params; plans, scores
  to 1e-6 and states) plus `tests/test_search.py` and `tests/test_belief.py`: 11 passed on Modal; the parallel
  tests passed again (3/3) after the round-1 chaining.

## Benchmark (s per searched turn, 24 turn-start positions unless noted)

| hardware | searcher | mean | p50 | p90 | max |
|---|---|---:|---:|---:|---:|
| 1 Modal core (pinned), 8 positions | lockstep (today) | 10.36 | — | — | — |
| 1 Modal core, same 24 positions | lockstep (today) | 9.39–10.72 | 6.5–7.0 | 17.5–20.0 | 38.6–43.9 |
| 16-core Modal CPU | parallel, 4 workers | 2.72 | 1.68 | 5.26 | 9.37 |
| 16-core Modal CPU | parallel, 8 workers | 1.69 | 1.06 | 3.32 | 5.63 |
| 16-core Modal CPU (2 hosts) | parallel, 16 workers | **0.95 / 1.17** | 0.60 / 0.80 | 1.84 / 2.06 | 2.69 / 3.57 |
| 16-core Modal CPU (3 hosts), before round-1 chaining | parallel, 16 workers | 1.02–1.32 | 0.70–0.82 | 1.90–2.41 | 2.78–4.17 |
| 32-core Modal CPU (1 slow host) | parallel, 16 workers | 1.77 | 1.13 | 3.41 | 5.26 |
| 32-core Modal CPU, before chaining (2 hosts) | parallel, 16 / 32 workers | 1.12–1.32 | 0.74–0.93 | 2.2–2.5 | 3.2–3.9 |
| NVIDIA L4 (Modal), 3 early turns (turn 4–8) | lockstep, vmap-batched | 14.72 | 14.32 | 16.44 | 16.44 |

The positions average 2.8 plans and ≈3,700–3,900 lane-steps per turn (late-game turns with many units are
the long tail). Same-code repeats on different hosts varied a lot (0.95–1.77 s for the final code); the
lockstep reference measured on the same host as the 0.95 s run took 9.39 s (9.9×). The L4 row ran only
early-game turns (≈1,700 lane-steps/turn), i.e. ≈8.6 ms per lane-step against ≈3 ms on one CPU core, and
took 7.5 min to compile: under vmap every `lax.cond`/`switch` of the engine runs all branches, and a step is
a long chain of small kernels with a ≈90 ms fixed latency already measured at 128 lanes on an L40S
(docs/simulator-throughput-diagnosis-2026-10-18.md). With only 16–64 lanes per phase there is nothing to
amortize that over.

Cost: a 16-core container at Modal's rates (≈$0.047/core-h + memory) is ≈$0.9/h; at ~1 s/turn that is
≈$0.00025 per searched turn. Compile (the two lane executables) takes ≈80–125 s at container start, once.

## Serving notes

- Use `TurnSearcher(params, lanes=16, depth=1, rollout_temperature=1., replies=64, determinize=True,
  parallel=True)` in a process that owns ≥16 cores (no other pinned workers on them); `workers` defaults to
  the process's CPU affinity, capped at 16.
- Keep `XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'` (Eigen threading did
  not help, 1.12–1.32 s vs 0.95–1.02 s, and only adds variance).
- Call `searcher.compile(state, key)` at startup; the first plan otherwise compiles on demand.

## Possible next steps

- Exact: skip the masks and policy heads on a lane's final (value-only) step and start the greedy
  candidate's known reply slots of rounds 2–4 early; each is worth a few percent.
- Serving: pin to a known-fast instance or measure s/turn at container start; host variance is now larger
  than the remaining gap to 1 s.
- Not exact: run all 64 replies in one phase instead of 4 halving rounds (one barrier instead of four, at
  the cost of spending replies on weak candidates) — would need its own Elo check.

Reproduce: `modal run --env staging experiments/modal_search_speed.py::profile`, `::threads --cores 32`,
`::equivalence --cores 16 --plan-checks 8 --play-turns 24 --workers 16`, `::gpu --gpu L4`.
