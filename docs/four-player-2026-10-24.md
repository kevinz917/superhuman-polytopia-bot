# Four-player 18×18 (2026-10-24)

The next checkpoint after the 14×14 1v1 scratch champion is a free-for-all bot for **Large (18×18) with four
players**. This branch makes the Rust engine, maps, features, network and trainer handle four seats. It then
runs the smoke tests, a 1-hour pilot and a two-arm network comparison, and stops before the long run.

## Ruleset (`bardur-ffa4-v1`)

| Rule | Choice | Why |
|---|---|---|
| Board | 18×18, 150-round cap (the 18×18 ruleset's) | Large; 81 tiles per player vs 98 in our 14×14 1v1 |
| Seats | 4, Bardur mirror | One tribe keeps the rules surface the engine already proves |
| Diplomacy | Off: no peace, embassies or treaties; the per-pair tables stay false | Treaties open kingmaking and stalling |
| Resign | Off | A resigned seat's pieces would stay on the board |
| Elimination | Capturing a seat's last city removes its units; the turn order skips it; the game ends at one seat left | Real-game domination |
| Reward | Rank: 1st +1, 2nd +1/3, 3rd −1/3, 4th −1 (zero-sum). Seats that left take the places from last upward in the order they left; the seats still in share the top places equally. At the round cap every survivor ties | Survival is worth something even when the win is gone; the tie at the cap removes any reward for farming score |
| Maps | `four-v1`: classic .25, drylands .15, lakes .2, pangea .15, continents .1, highlands .08, jungle .07; one capital per quadrant | Land-heavy; ridge and the water worlds are two-player or naval shapes |
| Meeting | Each seat meets every other seat on first sight (star gift per met seat) | The two-player rule, applied per pair |
| Capital vision | Writing reveals the capitals of every met seat | Same |

## Engine (`engine-rs`)

- **Seat count:** `State.np` holds it (2 or 4). Per-seat arrays hold `MAXP = 4` rows, and rows ≥ np stay zero.
- **Two-seat games stay the reference:** every generalized branch reduces to the old code at np = 2, and the
  end-of-turn rule keeps the reference's exact two-seat formula. `tests/test_rust_engine.py` (bitwise equal to
  the JAX engine) still passes.
- **New pieces:**
  - `out_order`, the leaving order.
  - `core::leave` and `core::place_reward`.
  - Rank rewards in `terminal_rewards`, which also fire at truncation for four seats.
  - "Any enemy" in attack and infiltration targets.
  - Per-seat loss of score when territory is claimed.
  - `mapgen::layout_players`: the classic generator with one capital per quadrant.
- **Packed view for four seats:**
  - The five relation fields have one entry per seat, and `alive` is added (`kernels::view_fields`,
    `view_width_np`).
  - `opp_view` is `[games, 3, width]`: the other seats in turn order after the mover.
  - Rewards are `[games, 4]` and final views `[games, 4, width]`.

## Features and network

- **Features (`observation.encode_view`):** with four seats, "enemy" means any other seat. The new inputs come
  after the two-seat ones:
  - 9 board channels: the owner, unit-owner and capital one-hots by seat offset in turn order.
  - 14 global entries: the third and fourth seats' relations, plus the four alive flags.
  
  Two-seat features are unchanged (`test_two_seat_features_unchanged`).
- **Warm start (`rl/four_player.convert`):**
  - Zero weights for every new input.
  - The critic keeps its own-view and next-seat-view weights and starts blind to seats 3 and 4.
  - The board is grown to 18×18 (identity) and, optionally, identity residual blocks are added.
  
  Checked on four-seat positions to give the parent's exact choices and values
  (`test_widened_networks_compute_the_parent_function`; `four_convert` repeats the check on the converted
  checkpoint).
- **Global pooling (`network.init/grow global_pool`):** each extra block adds a per-channel bias from the mean and
  max of its first convolution over the board, as in KataGo. It starts as the identity, like every grown block.
- **Dropped from the plan: auxiliary heads** (final ownership, score margin). Rows leave the actors every
  `segment` decisions, long before the game ends, so final-outcome targets would need rows held until the game
  ends. That conflicts with the staleness limit (`max_lag`).

## Trainer

- **`LearnerConfig.players = 4`** requires the Rust engine and `learner_values`, and rules out PBRS,
  `cap_score_reward` and `truncation_reward`. Rank rewards end every game, including at the cap.
- **Trajectories:** the per-seat trajectory lists, the segment boundaries and the pool lanes cover four seats.
  A pool lane gives the learner one random seat and the pool opponent the other three.
- **Omniscient critic:** it reads all four views (own first, then the others in turn order;
  `distributed.critic_features`), with 4× the input channels.
- **Actor stats:** they add `finished`, `turns` and `seats_out`, from which game length and elimination pace
  follow.

## Evaluation

- **`rl/arena4.py`:** each table of four entrants is played on every map with the seats rotated four times.
- **Results:** mean rank reward, first-place rate, mean place, reward by seat, and pairwise scores (A beats B in
  a game when A's reward is higher; ties count half).
- **Ratings:** Bradley–Terry fitted on the Elo scale, with a half-game prior.
- **Entry point:** `modal run experiments/modal_distributed_rl.py::arena4`.

## Results

### Tests (Modal, `experiments/modal_distributed_rl.py::tests`)

| Suite | Result | What it covers |
|---|---|---|
| `test_rust_engine.py` | pass | Two-seat Rust vs JAX engine, bitwise, after the refactor |
| `test_four_player.py` | 13/13 | Four-seat maps, random games, eliminations and rank rewards, round cap, export/import, two-seat features and maps unchanged, function-preserving widening, segment collector, gradient accumulation, **seat relabeling symmetry**, arena and ratings |
| `test_distributed_rl.py` | 16/16 | Trainer |
| `test_maps.py` | 7/7 | Map generators |
| `test_rl_18x18.py` | 15/15 | 18×18 ruleset. Also fixes five stale `train_epoch` calls that predate `entropy_coef`; these failed on main before this branch |

- **Seat relabeling test:** rotating every seat label of a mid-game state gives the mover the same legal
  actions and features, and the same successor state.
- **Bug it found:** `structures::remember_monuments` only evaluated seats 1 and 2, so seats 3 and 4 could never
  unlock a monument. Fixed.

### Converted parents (`four_convert`)

- Two parents, both made from the scratch champion `scratch-cont-2/final.npz` (with its critic):
  - **Arm A** `champ-ffa4-2x64`: the champion's 2×64 network.
  - **Arm B** `champ-ffa4-4x64gp`: 2×64 plus 2 identity residual blocks with global pooling.
- Over 60 four-seat positions, each converted network gives the same deterministic choices as the champion
  reading the two-seat part of the features. Log-prob and critic-value differences are 0.0.

### Smoke tests

- **Smokes 1 and 2:** the learner ran out of GPU memory on its first update, with 22 and 28 GiB allocations
  in `jit_epoch`. The 4-view critic stem has about 620 input channels, and its im2col buffer at a minibatch of
  4,096 on 18×18 is about 30 GB.
- **Fix:** `LearnerConfig.grad_accum`, an exact mean over micro-batches (tested), at batch 65,536 / minibatch
  4,096 / grad_accum 2. The value pass runs in 1,024-row sub-chunks for four seats.
- **Smoke 3 (15 min, 2 actor containers):** 194 updates and no errors. Games finish by elimination: 3 seats
  out per finished game, mean 52–98 rounds.
  - **Problem:** the raw advantage std fell to about 0.013, under the 0.05 normalization floor. That scaled
    the policy gradient down about 4× against the entropy bonus, and entropy climbed 0.5 → 4.3. This is the
    old 18×18 failure mode.
  - **Fix:** `advantage_floor` 0.01. The champion's own runs sat above the floor, so this restores its
    effective regime.
- **Throughput (fp32):** the learner trains in 2.4 s per 65k update plus a 4 s value pass, for about
  16k samples/s. With TF32 convolutions and fp32 heads on the learner only (the previously gated
  `POLYTOPIA_MATMUL_PRECISION=DEFAULT`), that drops to 0.86 s train + 0.35 s value. The learner now
  waits on actors.
- **Actors:** about 5.8–6.5k samples/s per 32-process container. Memory plateaus at about 69 GiB per
  container (2.4 GiB per process), so runs request 80 GiB.

### Seat effect (converted champion in every seat, seat-rotated)

| Engine / features | Games | Cap rate | Mean rounds | First-place share, seats 1–4 |
|---|---|---|---|---|
| Raw seat index (2/3 for seats 3/4: never seen in 1v1) | 64 | 23% | 78 | 38 / 34 / 3 / 2% |
| Seat index scaled to [0, 1] | 128 | 19% | 73 | 43 / 37 / 2 / 0% |
| + monuments fix | 192 | 14% | 69 | 45 / 36 / 2 / 3% |

- **What causes it:** the engine is symmetric under relabeling. A trace with the network's seat input
  zeroed turns seat 4's collapse (flat at about 3.2 cities from round 7) into a smooth turn-order gradient.
  At round 14, seats 1→4 have 13.0 / 10.9 / 9.4 / 7.5 units.
- **So:** the champion's seat-specific 1v1 habits plus turn order. Training has to learn around both, and
  evaluation rotates seats, so tables stay fair.

### Pilot (1 h, arm A network, 4 actor containers; ran on the engine before the monuments fix)

| Updates | Entropy | Raw adv. std | Value std | Learner vs 3 parents | Finished games | Mean rounds | Cap rate |
|---|---|---|---|---|---|---|---|
| 1–349 | 3.17 | .038 | .35 | −0.026 (659) | 3,804 | 67 | 1.2% |
| 350–698 | 2.99 | .028 | .29 | −0.065 (1,013) | 8,173 | 56 | 5.9% |
| 699–1,047 | 2.91 | .033 | .31 | −0.149 (793) | 8,397 | 52 | 1.5% |
| 1,048–1,397 | 2.93 | .030 | .31 | −0.186 (459) | 7,582 | 55 | 1.5% |

- **Scale:** 91.5M samples in 1 hour (about 28k/s), actor-bound, with the TF32 learner.
- **Entropy:** settles at the champion's own 1v1 training level (2.92 at its final update), so the entropy bonus
  is not running away.
- **Games:** short and decisive, with nearly 3 seats out per game.
- **The problem:** the learner seat falls further behind three copies of the parent every quarter.

### Network comparison arms (3 h each, fixed engine, launched together)

Arm A: 2×64. Arm B: 2×64 + 2 global-pooling blocks. Same recipe as the pilot (lr 3e-4, critic warm-up 10).
Both repeat the pilot's decline in the first 35 minutes (learner vs 3 parents: A +0.025 → −0.235, B +0.067 →
−0.163).

**Arena check** (`armA-u750-vs-parent-mix`): arm A at update 750 (L) vs the parent (P), 192 sampled games, seats
rotated.

| Table | L mean reward | P mean reward | P first-place rate |
|---|---|---|---|
| L P P P | −0.286 | +0.095 | 22% |
| L L P P | −0.263 | +0.263 | 33% |
| L L L P | −0.073 | +0.219 | 33% |

Bradley–Terry: L −111 Elo vs P. The learner is worse than its parent in every mix. Even three copies of it lose
to the one parent, so this is real regression, not an FFA effect where aggression only works among copies. It is
the continuation-drift pattern from the 18×18 and c6 runs: the critic has to relearn rank values from a 1v1
critic, and the policy moves (KL about 0.008 per update) on noisy advantages meanwhile.

### Remedy pilot (`ffa4-pilot2-anchor`, 1 h)

The pilot recipe changed in four ways: KL anchor to the parent (`anchor_coef` 0.05), lr 1e-4, critic warm-up
150 updates, and the frozen parent at 20% of lanes.

**Training-time record vs 3 parent seats** (mean rank reward per 250 updates; the dip is a transient):

| Run | First 250 | Worst | Latest |
|---|---|---|---|
| Arm A (lr 3e-4) | +0.025 | −0.245 (u1001–1250) | −0.04 (u2501–2600) |
| Arm B (lr 3e-4, global pooling) | +0.067 | −0.168 (u501–750) | +0.08 to +0.10 (u1501–2250) |
| Remedy pilot (anchor 0.05, lr 1e-4, warm-up 150) | +0.033 | −0.050 (u251–500) | +0.03 to +0.04 (u751–1391) |

- **Arm B throughput:** global pooling costs about 20% (147M vs 170M samples at 1 h 44 min).
- **Remedy recipe:** halves the per-update KL (0.0027 vs 0.007) and keeps the dip shallow.

**Interim arena** (`interim-A3250-B3000-R-P`, 400 sampled games, 5 seat-rotated table mixes): A at u3250, B at
u3000, R = remedy pilot final, P = parent.

| Entrant | Elo (P = 0) | Mean rank reward | First-place rate | Mean place |
|---|---|---|---|---|
| B | +44 | +0.139 | 28% | 2.29 |
| P | 0 | −0.012 | 23% | 2.52 |
| R | 0 | −0.013 | 25% | 2.52 |
| A | −19 | −0.083 | 17% | 2.62 |

Pairwise: B beats A 60% (160 comparisons) and P 55.5% (480); A vs P 46.7% (480).

### Final (3 h arms)

**Final training-time record vs 3 parent seats:**

| Run | Samples | Updates | Last 1,000 updates |
|---|---|---|---|
| Arm A | 302M | 4,616 | +0.035 to +0.138 |
| Arm B | 268M | 4,086 | +0.137 to +0.242 |

**Final arena** (`final-A-B-R-P`, 800 sampled games, 5 seat-rotated table mixes, seeds 386000+; records in
`runs/four-player-v1/final-A-B-R-P.json`):

| Entrant | Elo (P = 0) | Mean rank reward | First-place rate | Mean place |
|---|---|---|---|---|
| B (arm B final) | **+63** | +0.100 | 26% | 2.35 |
| A (arm A final) | +47 | +0.041 | 24% | 2.44 |
| R (remedy pilot final, 1 h) | +15 | −0.063 | 24% | 2.59 |
| P (converted champion) | 0 | −0.072 | 22% | 2.61 |

**Pairwise:**

| Pair | Score | Comparisons |
|---|---|---|
| B vs P | 0.589 | 960 |
| A vs P | 0.572 | 960 |
| B vs A | 0.526 | 960 |
| A vs R | 0.531 | 160 |
| B vs R | 0.562 | 160 |

**Per table (mean rank reward):**

| Table | Results |
|---|---|
| A P P P | A +0.162, P −0.054 |
| B P P P | B +0.208, P −0.069 |
| A B P P | B +0.131, A +0.081, P −0.106 |
| A A B B | B +0.025, A −0.025 |

Games average 69 rounds, and 6% hit the cap.

## Conclusions and the long run

1. **The four-seat stack works end to end.** Engine (seat-symmetric, two-seat bitwise unchanged), maps,
   features, warm start, trainer and evaluation. 18×18 four-seat training runs at about 28–31k samples/s on
   1×H100 (TF32 learner) + 4×32 actor processes, about $11/h nominal.
2. **Three hours from the champion beat it at four-seat play.** Arm A +47 Elo and arm B +63 Elo over the
   converted champion, after a transient dip in the first 1–1.5 h while the critic relearns rank values.
3. **Global pooling (arm B) is the network to scale.** It edges arm A head-to-head (52.6%, 960 comparisons:
   within noise) on 12% fewer samples, recovered from the dip about 40 minutes sooner, and leads every table
   mix. Its learner is about 20% slower per sample.
4. **The remedy recipe (KL anchor 0.05, lr 1e-4, warm-up 150)** keeps the dip shallow (−0.05 vs −0.17/−0.25).
   But it moves the policy half as fast, and its 1-hour final (+15) is below where the arms ended. For a long
   run, the arms' recipe at lr 3e-4 is the better default. The anchor is the fallback if a long run's
   vs-parent record keeps falling past 1.5 h.
5. **Seat effect.** In parent self-play, seats 1–2 win about 80% of games. This is turn order plus the
   champion's 1v1 seat habits, not the engine. Tables rotate seats, so ratings are fair. Watch whether
   training flattens it.

**Recommended long run:**
- Arm B's final (`/results/distributed/ffa4-armB/final.npz` + `final-critic.npz` as `critic_path`), the arms'
  recipe, `critic_warmup` 0.
- Batch 65,536 / minibatch 4,096 / grad_accum 2, TF32 learner env.
- More actor containers: 6–8 (the learner has headroom at about 31k/s).
- Hourly seat-rotated tables vs the converted champion and arm A.

## Big run `ffa4-big1b` (2026-10-25, 15:29–22:41 PT)

**Setup:**
- Started from arm B's final checkpoint and critic, with the arms' recipe.
- Frozen pool: arm B's start at 10% of lanes.
- 1×H100 (TF32 learner) + 6×32 actor processes, about 33k samples/s.
- Stopped early at 7.13 h (update 10,750, 704M samples) to finish by 23:00. The latest checkpoint is frozen at
  `/results/four-player/monitor/ffa4-big1b/update-10750.npz` (critic: `distributed/ffa4-big1b/latest-critic.npz`).
- A first launch (`ffa4-big1`) crashed at startup: its frozen pool held the 2×64 champion, but pool opponents
  must share the learner's network. The learner now refuses that with a clear error.

**Monitor** (`four_monitor`, 400 seat-rotated games per table: CUR vs START = arm B final, CHAMP = converted 1v1
champion):

| Checkpoint | Hours | Samples | Elo vs START | Elo vs CHAMP | CUR mean rank reward | CUR first-place rate |
|---|---|---|---|---|---|---|
| u5700 | 3.78 | 374M | +29 | +84 | +0.133 | — |
| u9150 | 6.09 | 600M | +72 | +107 | +0.210 | — |
| u10750 (table 1) | 7.13 | 704M | +53 | +85 | +0.163 | 35% |
| u10750 (table 2, other seeds) | 7.13 | 704M | +51 | +104 | +0.181 | 34% |

**Final checkpoint vs each reference** (pairwise scores, 672 comparisons each table):

| Reference | Pairwise score | Combined Elo |
|---|---|---|
| START (arm B final) | 0.575–0.585 | +52 |
| CHAMP (converted champion) | 0.611–0.642 | +95 |

**Training:** the learner seat's record vs 3 START seats rose from +0.04 to about +0.18. Entropy held at about
3.1, games at about 67–68 rounds, and 2–5% capped. There was no dip, since the run started from arm B.
The curve was still rising at the stop, so a continuation from `latest.npz` + `latest-critic.npz` should keep
gaining.

**Fixes made during the run:**
- `four_monitor` re-created its round clock every loop, so its 2-hourly tables never ran. Fixed by persisting the
  clock; the corrected monitor `big1b-mon2` ran from 3.8 h.
- `stop_run_at` (graceful early stop on Modal) added for the 23:00 finish.

## Second big run `ffa4-big2` (2026-10-25 23:48 – 2026-10-26 07:28 PT)

**Setup:** continued from `ffa4-big1b` update 10,750 + critic. Same recipe. Last night's checkpoint ("prev") at
10% of pool lanes. 1×H100 + 8×32 actors. The stop was scheduled with `stop_run_at` so every table would finish
inside 8 h; the final table landed at 08:09, about 20 minutes late, after a straggler evaluation container.

**Monitor** (`big2-mon`, 400 seat-rotated games per table; START = ffa4-big1b u10750, CHAMP = converted 1v1
champion):

| Checkpoint | Hours | Samples | Elo vs START | Elo vs CHAMP | CUR mean rank reward | CUR first-place rate |
|---|---|---|---|---|---|---|
| u2000 | 2.0 | 131M | +24 | +145 | +0.192 | — |
| u4100 | 4.15 | 269M | +31 | +149 | +0.205 | — |
| u6300 | 6.3 | 413M | +13 | +134 | +0.169 | — |
| u7650 (final) | 7.62 | 501M | +57 | +158 | +0.244 | 37% |

**Final pairwise:** CUR beats START 58.1% (672 comparisons) and CHAMP 71.3% (660).

**Throughput:** 8 actors did not help. The learner's batch assembly (about 3.5 s per update) capped the run at
about 26–27k samples/s. The inbox stayed full and about 27k samples per update were dropped as stale, so 6 actors
would have trained as fast for less.

**Reading:** the 6.3 h dip and the 7.6 h rise are within table noise (about ±25 Elo). Over two runs the bot is
about +110 Elo over the 7.1 h checkpoint's parent chain start (arm B) and +150 over the converted champion.
Gains per hour are falling.

**Checkpoint:** `/results/four-player/monitor/ffa4-big2/update-7650.npz` (critic: `distributed/ffa4-big2/latest-critic.npz`).
