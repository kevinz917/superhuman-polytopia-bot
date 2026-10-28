# Hero runs plan: two from-scratch 4x128 runs on the real rules (2026-10-26)

Branch `claude/gpu-inference` (eedcad8 = 43274ba + main 77bb1e3, one ruleset `polytopia-2.17.3`). Modal staging,
app `polytopia-distributed-rl`. Everything after launch runs with the Mac closed.

## Goal

| Run id | Mode | Start | Recipe |
|---|---|---|---|
| `hero-1v1` | 1v1 14x14, `proposed-v1` maps | random-init 4x128 (2-seat) | bn-main recipe: batch 262,144, minibatch 16,384, 1 epoch, γ .995 / λ .99, turn clock γ .99 λ .95, omniscient critic, advantage floor .05, entropy .01, seed 801 |
| `hero-ffa4` | 4-player 18x18, `four-v1` maps | random-init 4x128 (4-seat) | ffa4-big recipe: batch 65,536, minibatch 4,096 (grad_accum 1 on B200, 2 if it OOMs), rank rewards, advantage floor .01, no resign, coded views, seed 802 |

Both: LR 3e-4 decaying to 1e-4 over the last 25% (`learning_rate_end` 1e-4, `lr_decay_fraction` .25), bf16 trunk on the
learner (`POLYTOPIA_TRUNK_DTYPE=bfloat16`, `POLYTOPIA_MATMUL_PRECISION=DEFAULT`, `POLYTOPIA_HEAD_PRECISION=HIGHEST`),
GPU-inference actors (A10G default; L4 when schedulable), bf16 serving.

## Timeline (T0 = launch, after phase 0)

| Time | Event | Runs on |
|---|---|---|
| T−90 min → T0 | Phase 0 (below). **Mac must be open.** | laptop + Modal |
| T0 | Launch both runs, ladders, monitors, stop backstops, fleet keepers. Confirm samples flowing. Close the lid. | Modal |
| T0 + 2/4/6/8 h | 1v1 ladder rounds (32 maps x 2 seats) vs champion, bn-main4, c3. 4p tables (100 maps) vs ffa4-big2, converted champion. | Modal |
| T0 + 9 h 00 | Both learners stop on `training_seconds` = 32,400; write `final.npz`. Actors exit. | Modal |
| T0 + 9 h 05 | 1v1 final round (64 maps x 2 seats = 128 games/pair, 5 refs). 4p final table (200 maps, 40 parts). | Modal |
| T0 + 9 h 15 | `stop_run_at` backstop: marks runs stopped and cancels any learner still alive. | Modal |
| T0 + ~9 h 50 | Final tables done; everything idle. Hard timeouts kill anything left by T0 + 9 h 40 (training) / tables' own timeouts. | Modal |

Total detached window: **~10 h**. LR decay ends exactly at 9 h, so the scheduled stop must not cut training early.

## Throughput and samples (measured inputs, projected fleet)

Measured: 1x B200 learner bench, old rules, replayed real messages: **1v1 100–105k/s, 4p 39–51k/s**. GPU-inference
actor containers (A10G, cpu=16, 48 procs, bf16): **1v1 17.4k/s, 4p 19.1k/s per container**. Multi-GPU learner
scaling is **untested**; phase 0 decides the size.

| Learner | 1v1 samples/s | 1v1 actors | 1v1 samples in 9 h | 4p samples/s | 4p actors | 4p samples in 9 h |
|---|---|---|---|---|---|---|
| 4x B200 (if scaling holds) | ~250k | 16 | **~8.1B** | ~170k | 10 | **~5.5B** |
| 2x B200 | ~180k | 11 | ~5.8B | ~90k | 6 | ~2.9B |
| 1x B200 | ~100k | 7 | ~3.2B | ~50k | 4 | ~1.6B |

Actor count = learner rate / per-container rate, +1 spare. An idle 4x B200 learner costs ~$26/h, a spare actor $2.6/h,
so err toward one extra actor (the stale-drop waste is cheaper than learner idle).

For scale: bn-main1-4 (the big net that beat the champion) totalled ~7.9B samples over four segments; ffa4-big2's
lineage had ~1.2B 4p samples on top of the 1v1 champion.

## Cost

Rates from `GPU_RATE` / `CPU_CORE_RATE` / `MEM_GIB_RATE` in `experiments/modal_distributed_rl.py`:
B200 $6.25/h, A10G $1.10, CPU core $0.0472, memory $0.008/GiB. Learner host = 16 cores + 64 GiB ($1.27/h).
Actor container = A10G + 16 cores + 96 GiB = **$2.62/h**. Egress ~$5.6/B samples (1v1), assume ~$8/B (4p).

| Item (4x B200 plan) | $/h | Hours | Expected |
|---|---|---|---|
| hero-1v1: learner $26.3 + 16 actors $41.9 + egress $5.0 | $73 | 9.0 | **$660** |
| hero-ffa4: learner $26.3 + 10 actors $26.2 + egress $4.9 | $57 | 9.0 | **$520** |
| Ladders + monitors + final tables (with 2x preemption replay) | | | $50 |
| Phase 0 (tests, captures, 1/2/4-GPU benches, 15-min smokes) | | | $70 |
| **Expected total** | | | **~$1,300** |

Worst case by construction: every training container lives to its hard timeout (`seconds + 2400` = 9 h 40 m) →
1v1 $705 + 4p $555; fleet keeper capped at +4 replacements per run (~$95); evals $80; phase 0 $70 → **~$1,500**.

**Cost cap: $1,500 hard.** If phase 0 picks 2x B200: expected ~$820, cap $1,000. 1x B200: ~$520, cap $650.

Per sample: 1v1 ~$81/B, 4p ~$93/B (was ~$170 and ~$295 before GPU-inference actors + B200 bf16).

## Safeguards for an unattended 10 h

Modal has no per-app spend cap, so every piece is bounded in time instead:

1. **Training stops itself**: `training_seconds` 32,400; learner/actor function timeout `seconds + 2400`.
2. **`stop_run_at` backstop** at T0 + 9 h 15 per run: marks the run stopped (actors exit at their next report) and
   cancels the learner call with `terminate_containers=True`.
3. **Orchestrator replay guard** (already on main): a preempted `orchestrate` adopts its fleet from
   `{run}/calls` instead of spawning a second one. Never cancel a replayed orchestrator.
4. **Fleet keeper (new, build in phase 0)**: Modal function, every 5 min reads `{run}/actor/{i}` report timestamps;
   if fewer than the planned count reported in the last 10 min, spawns replacements via `_spawn_actors` with new
   indexes. Max 4 replacements per run, stops when the run is marked stopped or at T0 + 9 h. Without it, a preempted
   actor (Modal preempted 6/10 arena containers on Oct 20) stays lost for the rest of the night.
5. **Learner death = clean stop**: if a learner is preempted, actors exit on disconnect and the keeper sees the run
   stopped, so billing stops. The run would need a manual resume from `latest.npz` in the morning (accepted risk).
6. **Evaluation is bounded**: `elo_ladder` has a fixed round count, `four_monitor` has `max_rounds`, every arena
   call has a timeout.
7. **Workspace backstop (user action)**: set the Modal workspace budget/spend alert in the dashboard to
   current month-to-date + $1,500 if the plan exposes one.

## Phase 0 (Mac open, ~90 min, ~$70)

1. **Tests on Modal** on eedcad8: inference server, view codec, assembly, distributed, four_player, real rules,
   rust engine. Gate: all pass.
2. **Random-init parents** (4x128): `hero-1v1-init` (2-seat, as `random-init-b4w128`) and `hero-ffa4-init`.
   `random_init_fn` only builds 2-seat inputs today: add a `players` argument (4-seat features + 4-view critic stem).
   Gate: near-uniform policy, loads in `learner_bench`.
3. **Capture messages under the real rules** (both modes) and **bench the learner at B200:1, :2, :4** for each
   mode (`bench_learner --gpus B200:1,B200:2,B200:4`, learner_cpu 32 for the 4-GPU arms). Gate: pick the largest size
   that gives ≥1.6x per doubling; record samples/s, memory, assembly wait.
4. **Build + selftest the fleet keeper** (replay-guard-style check: seed a fake run, kill one actor, see it replaced,
   see the cap hold).
5. **15-minute smoke of each run at full fleet**. Gates: samples/s within 15% of the bench; stale drops < 10%;
   actor memory < 85% of the container (4p processes reached ~2.5 GiB each locally; served may be lower; set
   `memory_gib` from this); entropy stable; 4p games finish by elimination; logp mismatch (bf16 serve vs fp32
   learner) clip fraction < 1%. Then stop the smokes (`stop_run_at` now) and launch the real runs fresh.

## Launch commands (fill N from phase 0)

```
modal run --detach --env staging experiments/modal_distributed_rl.py::launch --run-id hero-1v1 \
  --parent /results/hero/hero-1v1-init.npz --gpu B200:4 --learner-cpu 32 --actors 16 --actor-cpu 16 \
  --actor-gpu A10G --processes 48 --seconds 32400 --ledger runs/hero-1026 \
  --serve-options-json '{"trunk_dtype": "bfloat16"}' \
  --learner-env-json '{"POLYTOPIA_TRUNK_DTYPE":"bfloat16","POLYTOPIA_MATMUL_PRECISION":"DEFAULT","POLYTOPIA_HEAD_PRECISION":"HIGHEST"}' \
  --config-json "$(cat configs/hero-1v1.json)"
modal run --detach ... ::launch --run-id hero-ffa4 --parent /results/hero/hero-ffa4-init.npz --gpu B200:4 \
  --learner-cpu 32 --actors 10 ... --config-json "$(cat configs/hero-ffa4.json)"
modal run --detach ... ::start_elo_ladder --tag hero-1v1 --run-ids hero-1v1 --interval 7200 --rounds 4 \
  --first-delay 7200 --maps 32 --map-mix proposed-v1 \
  --references-json '{"champion":"/retained/checkpoints/distributed-final-v1/scratch-cont-2-final/final.npz","bn4":"<bn-main4 final>","c3":"/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz"}'
modal run --detach ... ::start_elo_ladder --tag hero-1v1-final --run-ids hero-1v1 --rounds 1 --first-delay 32700 \
  --maps 64 --map-mix proposed-v1 --seed-base 460000 --references-json '{... + "b1-polish", "small-6.1B"}'
modal run --detach ... ::start_four_monitor --tag hero-ffa4 --run-id hero-ffa4 --interval 7200 --max-rounds 4 \
  --maps 100 --parts 20 --final-maps 200 \
  --references-json '{"START":"/results/four-player/monitor/ffa4-big2/update-7650.npz","CHAMP":"/results/four-player/parents/champ-ffa4-2x64/final.npz"}'
modal run --detach ... ::schedule_stop --run-id hero-1v1 --at-unix <T0+33300>
modal run --detach ... ::schedule_stop --run-id hero-ffa4 --at-unix <T0+33300>
```

The final 1v1 ladder must use `latest.npz` = final stage (the learner writes both at the end); first_delay 32,700 s
gives the learner 5 min to finish. Note START for the 4p monitor is ffa4-big2 (a 2x64 net), not the run's parent,
since a random-init START would make every Elo meaningless.

## Expected outcome

- **1v1 at ~8B samples (4x B200)**: around bn-main4 level (it had ~7.9B, with LR polish only on its last segment), and on
  the real rules from the first sample. At 2x B200 (~5.8B): around the champion.
- **4p at ~5.5B**: ~4.5x ffa4-big2's 4p samples, but from random weights and a 4x128 net. Expect to pass the converted
  champion by mid-run; matching ffa4-big2 is plausible, not certain. Watch the first table at T0 + 2 h.

## Morning checklist

`/results/distributed/ladder/hero-1v1*-history.json`, `/results/four-player/monitor/hero-ffa4-status.md`, ledger
`runs/hero-1026`, Modal billing for the app (rows lag ~1 h), then `scripts/ladder_elo.py` for the BT table.
