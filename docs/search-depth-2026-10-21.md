# Search depth curve: leak-free lookahead at d=1, 2, 3 on `c5-omni-s1-final` vs raw c3 (2026-10-21)

Question: the leak-free one-turn lookahead (`TurnSearcher(lanes=16, depth=1, determinize=True)`) adds +83 to +100 Elo on
top of the 6-h omniscient-critic final ([c5-omni-long-2026-10-21.md](c5-omni-long-2026-10-21.md), "Search on the
stronger final"). Does playing two or three rounds forward before the critic judges the candidate turn help, hurt, or
do nothing? No depth above 1 had ever been run. A positive d=3 would justify a turn-level tree search; the expectation
going in was flat at d=2 and negative at d=3 as rollout noise and fog-sample drift take over.

## Setup

Tag `c5-search-depth-v1` (`experiments/modal_test_time_search.py::sweep`, spec [search-depth/depth-spec.json](search-depth/depth-spec.json)),
14 × 32-core Modal containers (32 games per container), launched 14:03 PDT. Candidate
`/retained/checkpoints/distributed-final-v1/c5-omni-s1-final/final.npz`, opponent raw greedy c3, `proposed-v1` 14×14,
100 maps × 2 seats, seeds 388000–388099, lanes 16, temperature 1.0, rollout temperature 1.0, determinized (leak-free),
scorer the candidate's own value head, no `replies`. The paired baselines are the `c5-search-v1` arms on the same seeds and
settings: `win-k1-vs-c3` (raw) and `win-d1det-ownval-vs-c3` (depth 1), re-fetched with their per-game rows from the volume
into `runs/c5/search/c5-search-v1/` (the copies under `docs/c5-omni-long/search/` carry summaries only). A sibling session
ran a reply-breadth sweep on the same checkpoint and seeds at the same time under tag `c5-search-breadth-v1`.

### What depth does in `search.py`

- A lane plays the candidate turn in the root (lane 0 greedy, lanes 1–15 sampled at T=1), then both sides keep playing
  on the lane's own copy of the world with the searcher's network. `changes` counts mover changes; the lane is scored
  by the critic at `changes == 2·depth`, i.e. at the searcher's first decision after `depth` opponent turns
  (`make_advance`, `horizon`). With `rollout_temperature` 1.0 **every** non-root step is sampled at T=1, the searcher's
  own rollout turns included (`greedy = rollout_temperature <= 0` outside the root), so at d=2 and d=3 the candidate's
  later turns are noisy samples too, not greedy play.
- Step budget `max_steps = max_turn_actions · (1 + 2·depth)` = 288 / 480 / 672 for d = 1 / 2 / 3, i.e. exactly 96
  sub-actions for each of the 1 + 2d turns a lane can play, so a lane runs out of budget only when every turn in it hits
  the per-turn cap. A lane out of budget is still scored (the critic's value at its last state, from the searcher's side);
  the `complete` flag only records it. The summary exposes `incomplete_lanes` as a count; the per-lane rate below is
  that count over plans × 16 lanes.
- Lanes that end the game inside the horizon score the native outcome (±1); a lane that hits the game's turn cap scores 0.
- Determinization is unchanged by depth: the root world is sampled from the searcher's own view (`belief.py`), the plan is
  replayed on the true game only up to the first revealed difference (`build_guarded_replay`), and after 9 rejected
  samples one sub-action is committed from the true state (`fallbacks`). Deeper rollouts run entirely inside the sampled
  world, so fog drift compounds over 2·depth turns but never leaks: what is committed is still only root actions.

### Smoke (2 games per arm, seed 386000, both seats; `search-depth/depth-smoke.json`)

| Arm | W–L–cap | s / searched turn | lane steps / turn | incomplete lanes | compile | $ |
|---|---|---:|---:|---:|---:|---:|
| d=2 | 1–1–0 | 4.11 | 1274 | 0 | 227 s | 0.17 |
| d=3 | 1–1–0 | 5.23 | 1550 | 0 | 257 s | 0.19 |

Lane steps per turn scale 1.23× and 1.50× over d=1's 1036 (two processes on a 32-core container, so the s/turn numbers
are lighter than under the sweep's full load). From the d=1 arm ($2.14, 4 containers × 17.5 min) the estimate was ≈ $3
and $3.5 nominal per arm, $6–12 with preemption replay, and ≈ 40 min wall for d=3, so both arms were launched.

## Results (14:03–14:49 PDT, 46 min wall, 7 of 7 chunks per arm, no errors, no preemption replays)

`scripts/analyze_search_depth.py runs/c5/search/c5-search-v1 runs/c5/search/c5-search-depth-v1`
(full output: [search-depth/analysis.md](search-depth/analysis.md)). All four arms on the same 100 maps × 2 seats.

| Arm | W–L–cap | Score [95% CI] | Of decisive | Cap rate | Elo vs raw c3 [95% CI] | Paired Elo vs d=1 [95% CI] | s / searched turn | Lane steps / turn | Incomplete lanes | Override | Mean turns | $ (nominal) |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| raw (no search) | 151–26–23 | 0.812 [0.77, 0.85] | 85% | 12% | **+255 [+212, +305]** | — | 0.08 | 13 | — | — | 42.9 | 0.70 |
| leak-free d=1 | 167–17–16 | 0.875 [0.84, 0.91] | 91% | 8% | **+338 [+285, +407]** | 0 (reference) | 4.72 | 1036 | 0.0% | 0.87 | 40.0 | 2.14 |
| leak-free d=2 | 170–17–13 | 0.882 [0.84, 0.92] | 91% | 6% | **+350 [+295, +424]** | **+12 [−66, +94]** | 8.27 | 1843 | 0.0% | 0.89 | 38.2 | 5.34 |
| leak-free d=3 | 168–23–9 | 0.863 [0.82, 0.90] | 88% | 4% | **+319 [+263, +387]** | **−19 [−94, +53]** | 11.18 | 2688 | 0.0% | 0.89 | 38.0 | 6.06 |

Paired search gains over the raw bot (same maps and seats, 4,000-resample map bootstrap): d=1 **+83 [+21, +153]**,
d=2 **+96 [+38, +164]**, d=3 **+64 [+3, +135]**; d=3 minus d=2 **−31 [−105, +41]**.

- **Incomplete lanes: 0 of 271,664 (d=2) and 0 of 269,120 (d=3)** lane-plans, so the 480 / 672 step budgets never bind
  (lanes use 115 / 168 steps on average). Determinization never fell back to the true state (0 fallbacks per game at
  every depth); the one-sub-action-at-a-time commit on revealed differences is depth-independent, as expected.
- **Compute** per searched turn under full load (32 games per 32-core container): 4.7 s → 8.3 s → 11.2 s, i.e.
  1.75× and 2.37× d=1, tracking lane steps per turn (1.78× / 2.59×). Games take 327 s (d=2) and 438 s (d=3) of wall
  each; compile 6 min.
- **Cap games fall with depth** (12% raw → 8% → 6% → 4%): deeper rollouts push the bot to finish games (the critic sees
  the ±1 outcome inside the horizon more often, and a cap scores 0). At d=3 the games saved from the cap went to the
  loss column as much as the win column (losses 17 → 23), so the score did not improve.
- **Critic calibration** of the greedy lane is the same at every depth (corr 0.62–0.64 with the outcome, MSE
  0.35–0.38); the chosen-minus-greedy value gap grows with depth (+0.037 → +0.061 → +0.080), which is the
  optimizer's-curse signature of selecting the max over 16 noisier estimates rather than a better plan.
- Per map type the arms move within noise (8–28 games per type); d=2 is best on highlands and ridge (0.96), d=3 worst on
  pangea (0.79) and ridge (0.85).

### Cost

| Item | Modal app | Nominal (container-seconds × rate) | Billed at 14:50 PDT |
|---|---|---:|---:|
| smoke (2 arms × 2 games) | ap-kq3lOLv1xkX5OwXjSSpVjp | 0.36 | 0.22 |
| fetch of the `c5-search-v1` baselines + fetch of this tag | ap-7DsS50udKrjbroZAKm5H0I, ap-uIZWLfkFQJg3RHqrs92ZZo | 0.00 | 0.00 |
| `c5-search-depth-v1` (2 arms × 200 games, 14 × 32-core containers, 22,423 container-s) | ap-NeRXTeRNXMjicXDF1dqF5U | 11.40 | not yet in the billing report |
| **Total** | | **11.76** | reserved $12 in `runs/c5/search/budget.json` (row `c5-search-depth-v1`) |

Billing lags the hour; by the `c5-search-v1` precedent (nominal $9.00, billed $5.08) the final bill should land near
$6–8. `scripts/ledger_spend.py --write` was not run because the sibling breadth sweep had live rows in the same ledger.

## Verdict

Depth is worth nothing beyond d=1. The curve is flat at d=2 (+12 Elo paired, CI −66 to +94) and turns down at d=3
(−19 paired vs d=1, −31 vs d=2, both CIs spanning zero but the point estimates ordered as predicted), while the
compute per searched turn grows 1.75× and 2.4×. Every mechanism check comes out clean, so the null is not an
artefact: no lane ran out of steps, the determinized commit never fell back, calibration of the critic is unchanged,
and the only systematic shifts with depth are fewer cap games and a growing chosen-minus-greedy value gap, which
together say the deeper rollout mostly adds variance to the estimate that the max over 16 lanes then exploits. With
`rollout_temperature` 1.0 the searcher's own later turns are also sampled, so d=2/d=3 judge a candidate by two or
three rounds of noisy play from both sides on a sampled world; a greedy own-rollout variant (rollout temperature 0 for
the searcher's turns, sampled only for the opponent) was not run and is the one cheap follow-up if anyone wants to
close the question completely. The result does **not** justify a turn-level tree search project: a tree would buy
exactly the horizon that this experiment shows is already saturated at one opponent reply, at several times the
cost. The leak-free d=1 search (4.7 s per turn on 32 cores, +83–100 Elo) remains the ship configuration; breadth
(the sibling `c5-search-breadth-v1` sweep: more sampled opponent replies per candidate at d=1) is the only remaining
search axis with a plausible upside.

Files: spec, per-arm summaries (rows stripped), smoke, launch row and the analysis printout in `docs/search-depth/`;
full per-game rows in `runs/c5/search/c5-search-depth-v1/` (shared, uncommitted) and on the volume under
`test-time-search/c5-search-depth-v1/`.
