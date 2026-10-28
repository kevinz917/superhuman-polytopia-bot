# Campaign c6-credit readout: credit assignment on 14×14 (2026-10-21)

Three one-hour continuations of the strongest 14×14 checkpoint, identical except for how the win reward is credited
back to earlier decisions. Motivation and literature in [credit-assignment-review-2026-10-21.md](credit-assignment-review-2026-10-21.md);
per-arm reports with run logs, curves, ladders and ledgers in [ctl](c6-credit/ctl-2026-10-21.md),
[tc99](c6-credit/tc99-2026-10-21.md) and [tc100](c6-credit/tc100-2026-10-21.md). Total billed ≈ $69 of $105 in caps.

## Question

`c5-omni-s1` (the ship candidate, 77% vs c3) discounts per decision: γ 0.995, λ 0.99 over ~785 decisions per seat,
so the opening sees ≈ 2% of the terminal ±1. The 18×18 arm that first beat its parent used the *turn clock* instead
(`turn_gamma`/`turn_lambda`: discount only at the seat's own end-turn decisions, 1 within a turn). Every published
sparse-win-reward system (AlphaZero, DeepNash, AlphaStar) uses γ = 1 on the outcome. Does the clock matter on 14×14,
and is γ = 1 usable?

## Design

| | A `c6-credit-ctl` | B `c6-credit-tc99` | C `c6-credit-tc100` |
|---|---|---|---|
| credit | per decision γ 0.995, λ 0.99 (c5's) | turn clock γ 0.99, λ 0.95 | turn clock γ 1.0, λ 0.95 |
| win visible to the opening | ≈ 2% | ≈ 60–70% (0.99^40–50) | 100% |
| seed / ladder seeds | 301 / 389000–389127 | 302 / 389200–389327 | 303 / 389400–389527 |

Shared: parent `c5-omni-s1-final`, fresh omniscient critic with 60 critic-only warm-up updates, lr 2e-4, entropy
0.005, 3 epochs, batch 262,144, minibatch 16,384, segment 32, advantage floor 0.05, 14×14 `proposed-v1` maps,
opponents 80% self-play / 5% frozen parent / 15% own snapshots, 1×H100 + 12 × 16-vCPU actors, 3,600 s of learner
clock, then a deterministic ladder vs the parent on 128 fresh maps × 2 seats. Pre-registered reading: an arm
"holds" at ≥ 50% of all games vs the parent, "wins" if the CI lower bound clears 50%, "loses" if the upper bound is
under 50%. γ = 1 was checked end to end before launch (nothing divides by 1 − γ; truncated games bootstrap as
before) and got a unit test, `test_turn_clock_undiscounted_gamma_one`.

## Results

Ladder, final checkpoint vs `c5-omni-s1-final`, deterministic both sides, 256 games per arm:

| Arm | W–L–cap | all games [95% CI] | of decisive | capped | first seat (all / dec.) | second seat (all / dec.) |
|---|---|---|---|---|---|---|
| A per-decision γ 0.995 | 81–107–68 | **31.6%** [27.0, 35.9] | 43.1% | 27% | 39% / 53% | 24% / 33% |
| B turn clock γ 0.99 | 101–94–61 | **39.5%** [35.2, 43.8] | 51.8% | 24% | 47% / 59% | 32% / 44% |
| C turn clock γ 1.0 | 107–92–57 | **41.8%** [37.1, 46.5] | 53.8% | 22% | 48% / 60% | 35% / 47% |

Training-time sampled record vs the frozen parent (5% of games), share of decisive games per 50-update window:

| updates | A | B | C |
|---|---|---|---|
| 51–100 | 48% | 47% | 50% |
| 101–150 | 48% | 52% | 56% |
| 151–200 | 47% | 55% | 55% |
| 201–250 | 42% | 58% | 57% |
| 251–300 | 46% | 50% | 52% |
| final window | 46% | 52% | 45% |

Critic and optimiser statistics at the end of the hour:

| | A | B | C | c5-omni-s1 (6 h) |
|---|---|---|---|---|
| value std | 0.28 | 0.51 | 0.62 | 0.29–0.30 |
| raw advantage std (floor 0.05) | 0.039–0.048 | 0.082–0.087 | 0.085–0.115 | 0.039–0.044 |
| entropy | 2.5 | 2.2 | 2.2 | 2.3 |
| approx KL / update | 0.007–0.008 | 0.008–0.009 | 0.008 | 0.008 |
| updates / samples | 308 / 81M | 332 / 87M | 316 / 83M | — |
| billed | $28.09 (incl. ≈ $8 duplicate fleet) | $20.51 | $20.55 | — |

## Findings

1. **The turn clock beats per-decision discounting.** Both turn-clock arms finish 8–10 points of all games above the
   control with non-overlapping confidence intervals, and they are at or above parity in decided games where the
   control is clearly behind (43%). The training-time records agree: A drifts down to 42–46% of decisive while B and C
   sit at 52–58% from the second window on. B and C are indistinguishable from each other; γ = 1 is at least as good
   as 0.99 and gives the highest decisive share and lowest cap rate.
2. **No arm held.** All three lose to their parent on the all-games rule, including the control that continued c5's
   own recipe. One more hour of training from a *fresh* omniscient critic and fresh Adam state sets the policy back
   further than the hour recovers; this restart cost is shared by all arms and is larger than the between-arm effect.
   Implication for every continuation run: resume the parent's `final-critic.npz` (and optimiser state) instead of
   warming a new critic up from scratch. The learner already has `_resume` for its own checkpoints; it needs the
   same for a parent's critic.
3. **Mirror ladders need the decisive-game reading.** Two near-equal bots stall each other: 22–27% of games reached
   the 100-round cap against 8–10% when c5 played c3. The all-games rule scores a cap as a loss for the candidate,
   so an arm at true parity shows ≈ 38–40% of all games. The pre-registered 50% bar was too strict for a
   continuation vs its own parent; the between-arm comparison is the informative one.
4. **γ = 1 is numerically fine.** Returns stay in [−1, 1] (λ 0.95 mixes outcome and critic), the critic fits its wider
   targets (value std 0.62, equal to the return std, value loss 0.003–0.006), entropy and KL per update match c5's,
   and self-play cap rate did not rise (16–21%).
5. **The advantage floor is a confound and a defect.** c5 and the control run with raw advantage std 0.039–0.044,
   *below* the floor of 0.05, so their normalised advantages have std ≈ 0.8 and their effective policy step is
   15–25% smaller than the turn-clock arms', whose raw std (0.08–0.12) clears the floor. Part of B/C's edge may be
   that larger step. The 18×18 recipe already uses floor 0.01; 14×14 should too.
6. **Ops.** Arm A's orchestrator was preempted and replayed at 11:06, spawning a duplicate learner and 12 actors on
   the same run id for 21 min (≈ $8, five actor processes lost, update-100/150 snapshots overwritten). Third
   occurrence; the launcher fix (idempotent `orchestrate`, learner `finally` only marks its own registry entry) is
   still pending. Modal publishes an hour's billing row after the hour closes, so the first in-run rate check sees a
   partial row. Full-fleet rate was $21–22/h per arm, +31% over the launcher's estimate, as in c5.

## Recommendations

- **Default credit for new runs:** turn clock, γ 1.0, λ 0.95, advantage floor 0.01. Carry this into the 16–20
  board-size campaign; it is the setting that removed the horizon problem at 18×18 and did not hurt at 14×14.
- **Continuations resume the parent critic.** Done: `LearnerConfig.critic_path` loads the parent's `final-critic.npz`
  and `critic_warmup` 0 skips the warm-up ([critic-path-2026-10-21.md](c6-credit/critic-path-2026-10-21.md)). A 25-min
  check from c5-omni-s1 starts with a trained critic (value std 0.21 at update 1) and holds 52–53% of decisive sampled
  games vs the parent where the fresh-critic control sat at 48%; the paired update-100 ladder is parity vs parity, so
  whether it also prevents the control's later decline needs the one-hour version.
- **Confirm with a clean paired test before calling it the default for 14×14:** control vs γ 1.0 from c3
  (`abl-omni-2` start), 2–3 h each, shared ladder seeds, so the critic reset hits both sides equally. ≈ $140–200.
- **Ladder rule for mirror evaluations:** report decisive share and cap rate alongside the all-games score; score
  caps as ½ when ranking continuations against their parent.

## Artefacts

Branches `claude/c6-credit-ctl-3adf08`, `claude/c6-credit-tc99-b14e12`, `claude/c6-credit-tc100-2754d3`, merged here.
Checkpoints `/retained/checkpoints/distributed-final-v1/c6-credit-{ctl,tc99,tc100}-final/` (`final.npz` +
`final-critic.npz`). Per-arm `training.json`, specs, ladder boards and ledgers under `docs/c6-credit/<arm>/`.
`scripts/analyze_c5_training.py` now takes `--opponent` (alias `--frozen`) to name the frozen opponent.
