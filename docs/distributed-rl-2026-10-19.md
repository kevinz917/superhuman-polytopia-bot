# Distributed CPU-actor PPO campaign (2026-10-19)

**Result.** Delivered bot: `dist-c460div-v1` update 3142 (sha256 `04cbb4745a4e…`), archived at
`/retained/checkpoints/distributed-final-v1/dist-c460div-v1-update-3142/final.npz` (volume
`polytopia-structured-rl-retained`, staging). Deterministic, frozen engine, both seats:

| Match | Maps | W–L–trunc | Win rate [95% CI] |
|---|---|---|---|
| **vs league96 — untouched final maps** | 400032–400095 (64 × 2) | **117–10–1** | **91.4% [0.87, 0.96]** |
| vs league96, development (selection) | 306100+64 | 122–6–0 | 95.3% |
| same games on the batched GPU arena | 306100+64 | 122–6–0 | identical outcomes; 2/2 public replay audits passed |
| vs previous diverse bot (lr3e5 u176) | 306200+32 | 62–2–0 | 96.9% |
| vs anchored diverse bot (u256) | 306300+32 | 62–2–0 | 96.9% |
| vs sibling diversity run E3 final | 306000+64 | 56–36–36 | 61% of decisive games |

For reference, the previous diverse bot scored 21–43 against league96 (33%). Diversity: on the final
maps the bot recruited a non-Warrior unit in 126/128 games (Warrior 63%, Catapult 30%, Cloak 4%,
Defender 2%, Rider 1%; 2.9 unit types/game) versus league96's 99% Warriors; against strong learned
opponents it fields 3.5–3.8 unit types/game (Catapult ~42%, Warrior ~37%, Defender 10–15%, Cloak 4–7%).
It is less varied than the earlier diverse-teacher bots (Swordsman/Archer/Rider mostly dropped out).
Lineage: league96 → A (300 updates) → C (460) → D (3142), about 1.0 billion learner samples
(262,144 per update), learned-only self-play league; imitation ancestry through league96 remains.
Final-map use is recorded in `runs/structured-ppo-v3/budget.json` (partitions 400032 and 400064).

Goal set by the user: (1) generate games on a fast, branching CPU build of the
environment instead of the vmapped GPU simulator; (2) a Duper-style multi-GPU
architecture; (3) an aggressive ~8-hour, few-hundred-dollar campaign producing a
bot far stronger than the last baseline (league96) that keeps the diverse unit
strategy, fixing the earlier diverse runs' weakness against league96.

## Architecture

```
 CPU actor containers (Modal, 32 cores each, N per run)          GPU learner (Modal, H100:2)
 ┌───────────────────────────────────────────┐                ┌──────────────────────────────┐
 │ 32 processes, each pinned to one core     │  segments      │ receiver threads (zstd+pickle)│
 │  24 games: CpuGames (frozen rules, lax.map │ ─────────────▶ │ bounded inbox (backpressure)  │
 │  → real branching), local policy forward   │  TCP tunnel    │ assembler thread → 262k batch │
 │  on the lanes each policy controls         │                │ drop samples > max_lag old    │
 │  per-seat GAE segments (32 decisions)      │ ◀───────────── │ shard_map PPO over all GPUs   │
 │  sender thread, reconnect + respawn        │  fp16 weights, │  (pmean grads), publish every │
 └───────────────────────────────────────────┘  opponent pool  │  2 updates, pool snapshots    │
                                                               └──────────────────────────────┘
 orchestrator fn (spawns learner + actors) · ladder fn (snapshots vs league96, frozen CPU engine)
```

Files:
- `src/polytopia_bot/rl/cpu_engine.py` — single-game CPU build of the frozen engine. Games of one
  process are stepped with `lax.map`, so `lax.cond`/`switch` and data-dependent loops execute only the
  taken branch on XLA CPU (vmap computes all 17 action branches for every lane). `fast=True` adds the
  exact removal/path/buildmask proposals and move masks computed only for tiles with unmoved units.
  `tests/test_cpu_engine.py` checks states, masks and rewards bitwise (features to 1 ulp) against the
  frozen batched engine through random trajectories with resignations and auto-resets.
- `src/polytopia_bot/rl/distributed.py` — actor process, per-seat GAE segments, wire format, learner.
- `src/polytopia_bot/rl/cpu_arena.py` — paired-seat matches on the frozen engine (fast=False) with
  unit-mix / opening statistics.
- `experiments/modal_distributed_rl.py` — Modal app: `launch`, `more_actors`, `start_ladder`, `arena`,
  `tests`, `bench_actor`, `bench_learner`.
- `scripts/distributed_status.py` — live status from the registry Dict.

Differences from Duper (`duper/rl_ray_deploy`): no separate GPU predictor service — the policy is a
0.5M-parameter CNN, and CPU inference on the gathered lanes costs about as much as the engine step, so
a network hop per decision would only add latency. GAE is computed on the actor (as in Duper), but
over 32-decision segments with critic bootstrap rather than whole episodes, to bound policy lag.
Samples older than `max_lag` policy versions are dropped (Duper has no explicit bound). PPO runs 2
epochs over each 262,144-sample batch (Duper: one step per batch).

Self-play games record both seats; games against pool opponents record the learner seat only.

## Measurements

| Item | Result |
|---|---|
| CPU engine, M-series single core | 0.66 ms per game step incl. masks (frozen unbatched: 1.5 ms) |
| Modal CPU, one process | 2–3 ms per game step; inference ~0.4 ms/sample |
| 32-core container, unpinned, 2 full-batch forwards | ~3,000–4,500 steps/s |
| pinned processes | ~7,400–7,800 steps/s (1.7×) |
| + forward only on the lanes each policy controls | 4–8k steps/s per container in production (host variance) |
| Learner, synthetic batch | 165k samples/s/epoch on 1×H100, 320k on 2×H100 (linear), ~30 s compile |
| Tunnel | 80–110 MB/s per connection; i6pn unusable (cross-cloud placement) |
| Arena check | div176 vs league96 21–43 on 304000–304031, matching the earlier GPU ladder (19–45) |

## Campaign

Opponent mix in every run: 50% self-play (both seats learn), 15% historical snapshots (every 100
updates, last 8), 35% frozen rivals including league96. Learned-only; no scripted opponents or labels.
The diversity term is an exact KL(pi || reference) over the masked unit distribution at summon
decisions only (reference: diverse lr3e5 update 176); it never touches research, movement or the
decision whether to recruit.

| Run | Parent | Unit KL | lr / epochs | Fate |
|---|---|---|---|---|
| A `dist-l96div-v1` | league96 | 0.1 | 1e-4 / 2 | stopped 03:46 (plateau 75%) |
| B `dist-div176-v1` | diverse u176 | 0.05 | 1e-4 / 2 | stopped 02:07 (55% at 1 h) |
| C `dist-a300div-v1` | A update 300 | 0.3 | 1e-4 / 2 | stopped 04:33 (plateau 82%) |
| D `dist-c460div-v1` | C update 460 | 0.5 | 2e-4 / 3 | main line, to 08:47 |
| E `dist-d380div-v1` | D update 380 | 1.0 | 2e-4 / 3 | stopped 05:21 (no diversity gain, 86%) |
| E2 `dist-d880ent-v1` | D update 880 | 0.5 + unit entropy 0.05 | 2e-4 / 3 | stopped 06:08 (entropy too weak) |
| E3 `dist-e340ent-v1` | E2 update 340 | 0.5 + unit entropy 0.3 | 2e-4 / 3 | diversity line, to 08:45 |

Ladder vs league96 (deterministic both sides, frozen engine on CPU, fresh development maps, both seats):

| Time | Run, snapshot | Maps | W–L–D | Win rate [95% CI] | Warrior share | Unit types/game |
|---|---|---|---|---|---|---|
| baseline | diverse u176 (previous diverse bot) | 304000+32 | 21–43–0 | 33% [0.22, 0.44] | 57% | 2.8 |
| 02:05 | A u300 | 305000+32 | 48–16–0 | 75% [0.66, 0.84] | 70% | 3.3 |
| 02:05 | B u300 | 305000+32 | 35–27–2 | 55% [0.45, 0.66] | 64% | 3.0 |
| 02:55 | A u760 | 305100+32 | 46–17–1 | 72% [0.64, 0.80] | 78% | 2.3 |
| 02:55 | C u220 | 305100+32 | 52–10–2 | 81% [0.73, 0.89] | 69% | 2.8 |
| 03:42 | A u1180 | 305200+64 | 96–32–0 | 75% [0.67, 0.83] | 69% | 2.2 |
| 03:42 | C u460 | 305200+64 | 105–22–1 | 82% [0.76, 0.88] | 71% | 2.3 |
| 04:29 | C u740 | 305300+64 | 105–21–2 | 82% [0.75, 0.88] | 68% | 2.4 |
| 04:29 | D u380 | 305300+64 | 117–11–0 | 91% [0.87, 0.95] | 71% | 2.2 |
| 05:15 | D u880 | 305400+64 | 119–8–1 | 93% [0.88, 0.97] | 63% | 2.2 |
| 05:15 | E u280 | 305400+64 | 110–14–4 | 86% [0.80, 0.91] | 69% | 2.1 |
| 06:05 | D u1380 | 305500+64 | 117–9–2 | 91% [0.87, 0.96] | 64% | 2.0 |
| 06:05 | E2 u340 | 305500+64 | 115–12–1 | 90% [0.85, 0.95] | 65% | 2.0 |
| 06:50 | D u1880 | 305600+64 | 117–11–0 | 91% [0.87, 0.96] | 62% | 2.2 |
| 06:50 | E3 u280 | 305600+64 | 114–13–1 | 89% [0.84, 0.94] | 61% | 2.2 |

D u880 played sampled vs sampled league96 (305900+64): 115–13 (90%), 2.5 unit types/game.

Second ladder vs our own D u880 (both deterministic, 32 maps × 2 seats; "trunc" = engine turn cap, no winner):

| Time | Run, snapshot | Seeds | W–L–trunc | Unit mix (top) | Unit types/game |
|---|---|---|---|---|---|
| 06:50 | D u1880 | 307000+32 | 28–11–25 | Catapult 44%, Warrior 36%, Defender 10%, Cloak 5% | 3.7 |
| 06:50 | E3 u280 | 307000+32 | 36–15–13 | Catapult 43%, Warrior 37%, Defender 6%, Swordsman 5% | 3.6 |

Against league96 the learned bots win quickly with Warriors plus Catapults; against a strong learned
opponent the same policies field 3.6–3.7 unit types per game.

GPU cross-check of A u300 on 305000+32 with the established batched `DeterministicArena`: 48–16,
identical game outcomes on all 64 games, public-interface replay audits of one win and one loss passed.

Late rounds (vs league96 64 maps × 2 / vs D u880 32 maps × 2): D u2360 118–10 / 32–9–23, E3 u600
113–14–1 / 31–8–25; D u2860 — / 34–9–21, E3 u920 — / 38–10–16 (the 08:10 league96 round did not
report). Selection used only development blocks 306000/306100; the final partition was played once.

What mattered: (1) throughput — ~35–50k learner samples/s per run versus ~1–6k before; (2) a larger
step size with 3 epochs (D's jump from 82% to 91% within 40 minutes); (3) training against frozen
league96 and the earlier diverse bots inside a historical-snapshot league. The unit-choice KL and the
unit-entropy bonus kept recruit mixes away from all-Warrior play in training but did not, by
themselves, increase deterministic unit variety against league96.

## Cost (estimate from app lifetimes × launcher hourly rates, not an invoice)

About $330–360 in total: ~$15 smokes/benchmarks/tests, ~$300 for the training runs (two learners on
2×H100 plus 11–13 actor containers of 32 cores for ~7.5 h), ~$15 ladders, arenas and GPU cross-checks.
CPU usage above the requested cores (Modal's soft limit) may add a little.

## Caveats

- league96 is close to saturated as a yardstick (≥90%); progress after 05:00 was measured against our
  own snapshot D u880. Strong-vs-strong games often hit the engine turn cap (truncation, no winner).
- Opponents of the final bot during training included frozen league96 (10–15% of games), so part of
  the margin may be specific to that opponent; generalization against the diverse bots was 62–2.
- Research ruleset, no human evaluation. Not from-scratch RL (league96 had imitation ancestry).
