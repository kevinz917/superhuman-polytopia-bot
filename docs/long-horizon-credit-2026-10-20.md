# Long-horizon credit on 18×18: research and the next run (2026-10-20)

Follow-up to [rl-18x18-2026-10-19.md](rl-18x18-2026-10-19.md), where three PPO self-play runs from grown c3 did
not beat their parent. Questions: did AlphaZero-style systems hit the same problem, what should the discount be,
would search in training help or just cost, and how does Duper's action space compare.

## Our numbers (measured)

| | 14×14 (c3) | 18×18 (grown c3) |
|---|---|---|
| decisions per player per game (mean / median) | 890 / ~550 | 2,400 / ~1,850 |
| turns per game | 56 | 88 |
| decisions per player-turn (mean, p90) | 13.7, 21 | 23.7, 37 |
| legal actions per decision (median / mean / p90 / max) | 50 / 64 / 140 / 381 | 80 / 93 / 195 / 446 |
| flat catalog size (all families) | 130,868 | 299,188 |
| chosen family: move / attack / summon / end turn | 54% / 11% / 10% / 8% | 59% / 10% / 10% / 6% |

(480 sampled self-play games per size for lengths; 24,000 decisions per size for legal counts,
`experiments/modal_distributed_rl.py::action_stats`.)

The catalog is huge but the real choice is not. Each decision has 50–80 legal options (65% of them moves), and
the policy factors them into family → source tile → target, so no softmax is wider than one board. The problem
is depth, not breadth: ~2,400 decisions per player on 18×18, 24 per turn, against a per-decision discount.
With γ = 0.995 the final ±1 reaches the opening at 0.995^2400 ≈ 6e-6 (14×14: 0.995^890 ≈ 0.01). Per turn,
0.995^24 ≈ 0.89, a horizon of ~9 turns in an 88-turn game. On 14×14 the same γ gives 0.995^13.7 ≈ 0.93 per turn,
a horizon of ~15 turns in a 56-turn game. The discount clock counts unit moves, so bigger armies shorten the
horizon.

## What strong game systems do

| System | Game length | Discount / value target | Policy signal |
|---|---|---|---|
| AlphaZero ([paper](https://ar5iv.labs.arxiv.org/html/1712.01815)) | chess ~70, Go ~211 plies | none: value regresses the outcome z | MCTS visit counts (800 sims), a local per-move target |
| MuZero ([paper](https://ar5iv.labs.arxiv.org/html/1911.08265)) | board games | "discount assumed to be 1", bootstrap to game end (Atari: γ 0.997, 10-step) | MCTS |
| OpenAI Five ([paper](https://ar5iv.labs.arxiv.org/html/1912.06680)) | ~20k steps | horizon set in **game seconds**, annealed 60 s → 840 s; λ 0.95; heavy shaping. A sparse-reward ablation needed a 1 h horizon and was far less sample efficient | PPO |
| AlphaStar ([paper](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)) | thousands of actions | terminal reward without discount, TD(λ) value, V-trace + UPGO | KL to the supervised policy |
| DeepNash ([paper](https://ar5iv.labs.arxiv.org/html/2206.15378)) | Stratego, up to ~2,000 moves | full-trajectory estimates, no bootstrapping | R-NaD: −η·log(π/π_reg), η 0.2, reference reset periodically |
| Generals.io, superhuman ([abstract](https://arxiv.org/abs/2606.23348)) | ~730 turns, one move per turn | sparse win/loss, plain policy-gradient self-play (the agent's read of the full text: γ 1.0, λ 0.9, HL-Gauss value; not verified) | trains only on the highest-advantage samples; parameter EMA |
| Turn-PPO ([abstract](https://arxiv.org/abs/2512.17008)) | multi-turn LLM agents, many tokens per turn | advantages estimated **per turn**, not per token | PPO |
| Semi-MDP / options ([Sutton, Precup, Singh](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)) | — | discount by elapsed time, γ^τ, not per primitive action | — |

So yes: AlphaZero and MuZero face long games but avoid the problem. Their value head just predicts the outcome
(no discount), and the policy is never trained from a long-horizon advantage; search gives it a per-move
target. Systems trained by policy gradient on long games set the discount in game time (OpenAI Five), do not
discount (AlphaStar, DeepNash), or estimate advantages per turn (Turn-PPO). We are the outlier: a per-atomic-action
discount in a game whose atomic-action count grows with army size.

**Duper** (local `rl_ray/`) is not Polytopia: a 7×7 card and territory game of 12–24 rounds whose widest head
has 100 logits. Its credit design still matters here. Each player's decisions are split into seven trajectories
by decision type, each with its own value head and γ (0.97–0.99), and λ drops to 0.7–0.8 at end-turn steps
(`flow/utils.py:167-229`, `single_env.py:242-260`). An earlier branch (`1012-in-turn-gamma-1`) used γ = λ = 1
inside a turn and discounted only at end-turn. The team swept γ/λ repeatedly, and its branch history shows it
kept splitting decisions into separate value groups to shorten chains. No search.

## Search in training: does it pay?

- At play time, yes. On 14×14, leak-free d=1 turn search adds +104 Elo, robust +117
  ([test-time-search-c3](test-time-search-c3-2026-10-19.md)), at ~1 s/turn on 16 cores
  ([search-speed](search-speed-2026-10-19.md)).
- In the training loop it is expensive. Searched decisions run at 4.6 per core-second (search-distillation
  run) against ~156 raw decisions per core-second in our actors, ~34× fewer samples per dollar on 14×14. 18×18
  has more decisions per turn and heavier steps, so expect ~40×. Campaign 3's 1.3B samples would become ~30M
  searched samples for the same money.
- One round already failed. The sibling search-distillation run (23.4M search-labelled decisions, ~$110) left
  the raw network exactly as strong as c3 (±3 Elo). Search overrides greedy 87% of the time, but mostly within
  its own lane-score noise (median gain 0.023 vs lane spread 0.018).
- AlphaZero works because its search (800 simulations) produces a much better target than the network. Our
  d=1, K=16 search produces a slightly better, noisy one. Gumbel-style root selection, which guarantees policy
  improvement with few simulations, is the version worth trying if search training comes back.

Verdict: keep search as a play-time booster. As a training signal it is ~35–40× more expensive per sample and
has not yet shown a raw gain. The cheaper fix is the discount clock.

## Constraint from our architecture

An 18×18 game lasts ~25 minutes of actor wall time, about 300 learner updates. AlphaZero-style targets that wait
for the outcome would be hundreds of policy versions stale, so the actors emit 32-decision GAE segments with a
critic bootstrap and drop samples more than 10 versions old. Whatever replaces the discount must work within
segments, i.e. through the critic. That makes a well-scaled critic (warm-up, turn clock) the main lever.

## Recommended next run (ranked)

1. **Turn-clock discounting.** Apply γ and λ only at the acting seat's end-turn decision; γ = λ = 1 for the other
   decisions of a turn (the Duper `1012` variant; SMDP discounting; Turn-PPO). Pick γ_turn ≈ 0.97–0.98 and
   λ_turn ≈ 0.9–0.95, a horizon of 33–50 turns. The horizon no longer depends on army size or board, and 14×14
   and 18×18 can share a critic. Change: `gae()` takes per-step γ/λ from an `end_turn` flag the actors already
   know (family 15). Segments should end at turn boundaries rather than every 32 decisions.
2. **Critic warm-up, longer** (already built: `critic_warmup`). Use 100–200 updates, until the value correlates
   with outcomes, since the parent critic was trained under a different clock. Optionally a separate critic λ of
   1 within the segment (VC-PPO's decoupled GAE).
3. **KL anchor to the parent.** `ppo.loss` already supports `anchor_coef` (k3 estimator); the learner needs the
   parent's log-probs per minibatch, one extra forward pass, ~25% of learner throughput. Every run so far drifted
   below its parent after a peak; AlphaStar, DeepNash and Diplomacy's piKL all regularize toward a reference.
   Start at 0.05–0.1 and keep the parent fixed.
4. **Kill rule and selection by the training-time "vs frozen parent" record**, per 50 updates (it predicted
   every collapse within ~30 minutes), with deterministic ladder confirmation.
5. Later: potential-based shaping on end-turn steps (score/city differential, annealed to zero), a 14×14 → 18×18
   curriculum (needs per-size learner batches), and parameter EMA for the deployed policy.

Cost: like run 3 (4×H100 + 19 actors), billed ~$64/h including the memory overage (request 2.5 GiB per actor
process). A 3 h run plus evaluation is about $220. A 45-minute pilot on the kill rule is about $50, and would
show by update ~300 whether the turn clock moves the critic and the vs-parent record.

## Pilot result: turn clock + KL anchor (2026-10-20, `l18-turnclock-v1`)

Setup as run 3 (lr 1e-4, entropy 0.002, advantage floor 0.01, 15% frozen grown c3, 10% own snapshots) plus:
- turn clock: γ_turn 0.97, λ_turn 0.92, applied only at the seat's own end-turn decisions; 1 inside a turn;
- KL anchor to grown c3, `anchor_coef` 0.05;
- 80 critic-only warm-up updates;
- actor memory requested at 80 GiB per 32-process container.

It ran 40 min of training: ~310 updates, ~80M samples, 4×H100, 16 actor containers (2 lost mid-run, no
throughput effect because the learner is the bottleneck at ~39k samples/s with the anchor's extra forward pass).

| Updates | vs frozen grown c3, sampled training games (decisive share) | value std | entropy |
|---|---|---|---|
| 0–49 (warm-up, policy frozen) | 48% | 0.22 | 1.8 |
| 50–99 (warm-up until 80) | 54% | 0.28 | 2.4 |
| 100–149 | 50% | 0.26 | 2.6 |
| 150–199 | 56% | 0.25 | 2.8 |
| 200–249 | 55% | 0.25 | 2.8 |
| 250–310 | 53% | 0.26 | 2.8 |

Deterministic vs grown c3 on fresh development maps (370800+48 × 2 seats, frozen engine): update 200
37–39–20 (49% of decisive games), final 37–34–25 (52%).

Reading:
- The turn clock did what it was meant to for the critic. Value std was 0.25–0.28 throughout, above every
  earlier 18×18 run (0.15–0.27) and close to c3's 0.28 on 14×14.
- The anchor did what it was meant to for stability. KL to the parent grew slowly (0.008 → 0.019 per sample),
  and the final checkpoint did not slide below parity the way runs 1–3 did.
- But the policy did not improve. Sampled play vs the parent plateaued at 53–56%, about run 3's level, and
  deterministic play is at parity.
- So discounting by decisions was a real defect (run 1's collapse, run 2's flat critic), not the only one.
  With a healthy critic and no drift, 300 updates of self-play from grown c3 still find no clear improvement on
  18×18.

Not recommended: the ~$220 full run of this configuration. Its training curve was flat for the last 150 updates.

Candidates for the next attempt, cheapest first:
1. Potential-based shaping on end-turn steps (score / city / star differential, γΦ(s′) − Φ(s), annealed to zero).
   Generals.io reports +149 Elo early from such a term; it gives the policy a per-turn signal the outcome cannot.
2. A 14×14 → 18×18 curriculum, which needs per-size learner batches. c3 improves quickly on 14×14 (65% vs its
   parent within 100 updates), so mixing sizes may carry over skill rather than relearn it.
3. Advantage filtering (train on the top-|A| quarter, as Generals.io does) to raise the signal-to-noise of each
   batch. Cheap to try with the current learner.

Cost: pilot ~$40 (wall-time bound; billing settling), tests and checks ~$3; ledger `runs/rl-18x18-v2`
($55 cap).
