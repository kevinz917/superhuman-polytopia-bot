# Speed comparison: distributed CPU-actor setup vs the original single-L40S setup (2026-10-19)

The question: how much faster is the Duper-style setup (CPU actor containers plus a 2×H100
learner, `docs/distributed-rl-2026-10-19.md`) than the original setup, a single L40S running
before the optimizations of 2026-10-18/19? No new jobs were run. Every number below comes from
recorded `training.json` rows on the `polytopia-structured-rl` volume (staging env),
`runs/structured-ppo-v3/budget.json` (GPU type and $/h of the old runs), and
`runs/distributed-rl-v1/budget.json`. Snapshots of the two live runs (`dist-c460div-v1`,
`dist-e340ent-v1`) were taken at ~08:28 PDT.

## The three setups

| | Baseline: L40S before the optimizations | Intermediate: H100 after the single-GPU optimizations | New: distributed |
|---|---|---|---|
| Hardware | 1×L40S, 8 cores, 32 GiB ($2.58/h, ledger) | 1×H100, 8 cores, 32 GiB ($4.58/h) | learner H100:2, 16 cores, 64 GiB ($9.17/h) + 5–7 actor containers with 32 cores each ($1.77/h each) |
| Simulator | frozen engine, `vmap` over lanes (computes all 17 action branches) | fast engine (exact proposals), still `vmap` | per-process CPU engine with `lax.map` (real branching) + fast proposals |
| Collector | `independent-lanes`: 512 lanes × 64 decisions, lane utilization ~0.3–0.5 | decision-budget, 2048 lanes × 32 | actors stream 32-decision GAE segments |
| PPO | 32,768 samples/update, minibatch 256, 2 epochs | 65,536 samples/update, minibatch 256, 2 epochs | 262,144 samples/update, minibatch 16,384, 2–3 epochs |
| Runs used | `low256-league-rivals-v1` (281 updates, 5.2 h; it produced league96); `diverse-{dagger-lr1e4,dagger-lr3e5,bc-lr1e4}-league-v1` (~1 h each, the last runs before the optimizations) | `diverse-h100-anchor-v1`, `diverse-h100-lr3e5-v1` | all seven `dist-*` runs |

Rates are steady state. Old runs: the sum of row `seconds` for updates 2 onward; the first update,
which includes the compile, is left out. New runs: from the first update's `elapsed` to the last
row. The new actor-container count is time-weighted from `connections` divided by 32.

## Results

Ranges for the baseline: low256-league / diverse-league runs. Ranges for the new setup: all 7 runs
(main line D `dist-c460div-v1` is the high end).

| Metric | Baseline: L40S, pre-optimization | Intermediate: optimized 1×H100 | New: distributed | New ÷ baseline |
|---|---|---|---|---|
| (1) Game steps/s (atomic env steps, both seats) | **970** / 475–543 | 7,585 (anchor) / 4,866 (lr3e5) | **29.5k–57.9k** | **30–60×** (54–122× vs diverse-league) |
| · per device | one L40S: ~970 steps/s at $2.58/h | one H100: ~7.6k steps/s at $4.58/h | one 32-core container: 4.9k–8.6k steps/s at $1.77/h | each container: 5–9× the steps of an L40S at 69% of its price |
| (2) Training samples/s (decisions that enter PPO) | **488** / 215–264 | 3,780 / 2,679 | **23.2k–46.3k** | **48–95×** (88–215× vs diverse-league) |
| (3a) PPO updates per hour | 54 / 24–29 (32,768 samples each) | 208 / 147 (65,536 each) | 319–635 (262,144 each) | 6–12× |
| (3b) Training samples per hour | **1.76M** / 0.77–0.95M | 13.6M / 9.6M | **84M–167M** | **48–95×** |
| (3c) Optimizer steps per hour | 13.8k / 6.1k–7.4k | 106k / 75k | 10k–31k | 0.7–2.2× |
| (4a) $ per hour | $2.58 | $4.58 | $18.0–21.6 | 7.0–8.4× |
| (4b) $ per M training samples | **$1.47** / $2.72–3.34 | $0.336 / $0.475 | **$0.129–0.228** | **6.4–11.4× cheaper** (12–26× vs diverse-league) |
| (4c) $ per M game steps | $0.74 / $1.32–1.51 | $0.168 / $0.261 | $0.103–0.184 | 4.0–7.2× cheaper |
| (5) Strength vs league96 | league96 itself = this setup's output: 96 updates of `low256-league-rivals-v1` (1.85 h, 3.1M samples, from parent low256). The diverse-league runs were stopped after ~1 h with no ladder. | best 33% after 1.3 h (lr3e5 u176) | 75% after 1.0 h; **91% after ~3.2 h and 297M samples** (A u300 → C u460 → D u380); 93% after ~4.0 h and 428M samples | no measured ratio (see caveats) |

Per-run detail for the new setup (steady state):

| Run | Epochs | Containers | Updates/h | Samples/s | Steps/s | Steps/s per container (p10–p90) | Waiting for data | $/M samples |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| A `dist-l96div-v1` | 2 | 7.0 | 457 | 33,290 | 41,825 | 6,216 (5,942–6,433) | 58% | 0.180 |
| B `dist-div176-v1` | 2 | 5.0 | 319 | 23,242 | 29,533 | 7,077 (4,961–7,432) | 69% | 0.215 |
| C `dist-a300div-v1` | 2 | 6.2 | 336 | 24,440 | 30,259 | 4,904 (4,311–5,936) | 70% | 0.228 |
| D `dist-c460div-v1` | 3 | 7.0 | 635 | 46,272 | 57,865 | 8,337 (7,873–8,691) | **4%** | **0.129** |
| E `dist-d380div-v1` | 3 | 6.0 | 473 | 34,466 | 42,363 | 7,058 (6,872–7,274) | 36% | 0.159 |
| E2 `dist-d880ent-v1` | 3 | 6.0 | 571 | 41,615 | 51,378 | 8,558 (8,284–8,804) | 20% | 0.132 |
| E3 `dist-e340ent-v1` | 3 | 6.0 | 438 | 31,861 | 40,245 | 6,719 (6,324–7,081) | 38% | 0.173 |

"Waiting for data" is the learner's `ready_wait_seconds` as a share of wall time.

**How the ~50–95× breaks down (samples/s, using the low256 baseline):**

- **Single-GPU optimizations, L40S → H100, ~7.7×** (488 → 3,780). This step bundles three
  changes: the decision-budget collector (~2–3.4× at 512 lanes), the fast engine (~4×), and the
  move to an H100 at 2048 lanes.
- **Distributed setup, H100 → 2×H100 + CPU actors, ~6–12×** (3,780 → 23–46k).

## Caveats

- **Samples are counted differently.** Both old setups record only the learner seat, at
  ~2.0 game steps per training sample. In current-policy self-play they play the opponent's
  seat and throw it away. The new actors record both seats in self-play, at ~1.25 steps per
  sample. Game steps/s (30–60×) is the seat-neutral comparison. Samples/s (48–95×) includes the
  extra seat.
- **Two baselines.** `low256-league-rivals-v1` (488 samples/s, 65.8 s/update) matches the "~500
  dec/s, 60–67 s/update" that `docs/simulator-profile-2026-10-19.md` §7 quotes for live runs
  before the optimizations. The diverse-league runs ran slower (215–264/s, 120–154 s/update). They
  are short and early, and their games are longer, so idle lanes waited longer for each lane's
  64-decision quota. The profile's controlled measurement on the same states gave 311/s
  (`profile-train-v1`).
- **Learner-bound vs actor-bound.** D is learner-bound. Its learner waited for data only 4% of
  the time, and 3 epochs over 262,144 samples take ~4.8 s, a cap of ~54k samples/s. Its actor
  rate is held back by inbox backpressure, so it is a lower bound. The other new runs are
  actor-bound: their learners waited 20–70% of the time. The baseline was bound by game
  collection, which took 91–99% of every update.
- **Host variance.** Medians per container range from 4.9k to 8.6k steps/s between runs, while
  p10–p90 within a run are tight. Most of the spread comes from container placement at launch and
  from game length, not from noise within a run.
- **Optimizer steps.** The new setup takes about the same number of optimizer steps per hour as
  the baseline (0.7–2.2×), with 64× larger minibatches. The 50–95× more data per hour mostly buys
  larger, lower-noise gradient steps, not more of them.
- **Time to strength is not a controlled comparison.** Parents, opponent mix, unit-KL term,
  learning rate and epochs all differ. The baseline produced league96 (the 50% reference) in
  1.85 h of training from its parent, but that parent's own strength against league96 was never
  measured. A sample-count equivalence: the 297M samples behind D u380 (91%) would take ~169 h
  (~7 days) at the baseline's 1.76M samples/h. That assumes equal sample efficiency, which was
  not measured. Chain hours count training time only, not the minutes between relaunches.
- Stale samples dropped by `max_lag` were under 0.2% of what was received in every new run.

## Bottom line

Against the original single-L40S setup, the distributed setup feeds PPO **48–95× more training
samples per hour**: 23k–46k/s against 488/s, and 84M–167M/h against 1.8M/h. It generates
**30–60× more game steps/s**; recording both seats in self-play accounts for the rest. It costs
7–8× more per hour but is still **6–11× cheaper per training sample** ($0.13–0.23/M vs $1.47/M).
Each $1.77/h CPU container generates 5–9× the game steps of the $2.58/h L40S.
About 7.7× of the gain came from the single-GPU optimizations (decision-budget collector, fast
engine, H100); the distributed setup added another 6–12× on top. Time to strength was not
measured head-to-head. The 297M samples behind the 91%-vs-league96 checkpoint would have taken
about a week at the baseline's rate, assuming equal sample efficiency.

## Sources

- Baseline: `/low256-league-rivals-v1/training.json`,
  `/diverse-dagger-league-lr1e4-v1/training.json`,
  `/diverse-dagger-league-lr3e5-v1/training.json`, `/diverse-bc-league-lr1e4-v1/training.json`
  plus their `protocol.json` (`collector_mode: independent-lanes`, 512 × 64, no `engine`, i.e.
  frozen). GPU type and rate (`L40S`, $2.584/h) from `runs/structured-ppo-v3/budget.json`. Game
  steps per update = 32,768 + `batch.opponent_decisions`.
- Intermediate: `/diverse-h100-anchor-v1/training.json`, `/diverse-h100-lr3e5-v1/training.json`.
  Game steps per update = 65,536 + `batch.dropped_pending` + `batch.opponent_decisions`. In lr3e5
  the gradient phase took 8.8 s per update throughout, against 0.3–2.5 s for anchor; the cause is
  unexplained. `docs/simulator-profile-2026-10-19.md` §2, §3, §7.
- New: `/distributed/<run>/training.json` (`elapsed`, `samples`, `train_seconds`,
  `ready_wait_seconds`, `connections`, `actors.steps`, `actors.learner_decisions`) and
  `protocol.json`; the ladder table in `docs/distributed-rl-2026-10-19.md`;
  `runs/distributed-rl-v1/budget.json` for hourly rates. `actors.steps` adds one per game per
  engine tick (`distributed.py` actor loop), the same atomic step the old collectors count as
  learner plus opponent decisions.
