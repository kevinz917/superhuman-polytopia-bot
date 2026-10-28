# Why the first pure-RL continuation failed

The experiment mostly learned to avoid resigning and change its action priors.
It did not learn to expand and fight. The strongest findings are poor reward
coverage, a training-induced collapse of the global representation, and an
incomplete entropy gradient. The earlier legality/probability tests did not
establish that the representation or learning recipe was healthy.

This analysis added local diagnostic scripts and reports only. It did not alter
the trained policy, learner, or simulator, or launch another cloud training run.

## Replayed behavior

The parent and final policies were replayed on all 16 games each against the
fixed random opponent, using the original map seeds, seat allocation, and RNG
schedule. All 32 CPU replays matched the recorded GPU game lengths, terminal
flags, and outcomes. Original full GPU action traces were not saved, so this is
outcome/length agreement rather than a claim of checking every original action.

| Learner behavior across 16 games | Parent | Final |
|---|---:|---:|
| Learner decisions | 1,722 | 2,506 |
| End-turn actions | 1,388 (80.6%) | 1,596 (63.7%) |
| Moves | 63 (3.7%) | 85 (3.4%) |
| Attacks | 1 | 1 |
| Recruits | 26 | 78 |
| Disbands | 5 | 80 |
| Research actions | 110 | 384 |
| Capture-family actions | 2 | 2 |
| Mean final owned cities | 1.0625 | 1.0 |
| Mean final units | 2.31 | 0.81 |
| Games ending with zero units | 0 | 5 |

The final policy never owned more than one city in any of these games. The
capture family also includes ruins, so two capture-family actions must not be
described as two city captures. All final games finished with 26 technologies,
while the mean unspent bank was 135 stars. It repeatedly acted without building
a productive army: 78 recruits and 80 disbands is expenditure of decisions,
not evidence of military progress. These counts do not establish that each
recruited unit was immediately disbanded.

Legality was not the obstacle: movement was available at 1,856 learner decisions,
yet selected only 85 times. It ended the turn 1,156 times when movement was legal.
There were 15 capture opportunities; seven were followed by ending the turn.
Attacks were available at only 16 decisions, consistent with rarely reaching
combat, and only one attack was selected.

All 15 final-versus-random truncations occurred at **turn 101**, after 399–596
atomic actions. The implementation stops when `turn > reference_max_turns`,
with a configured limit of 100. None reached the 10,000-action guard. These
games were not lengthy strategic battles stopped just short of victory.

## 1. The reward distribution taught the wrong first lesson

The initial pure-RL smoke produced 431 native completions, all resignations.
The continuation produced 112 native completions, again all resignations, and
394 engine truncations. Thus it observed no elimination-based win as a reward
for strategic play. Sixteen of 64 continuation batches had no native completion
at all; six of the final 16 batches had none.

With resignation treated as an ordinary action family, a nearly uniform initial
policy can lose immediately. Avoiding resignation supplies a much easier lesson
than expanding, recruiting, reaching an enemy, and capturing its cities.
Passing also gives a weak opponent more chances to resign. This is an inference
about the incentive structure, supported by the reward and replay records; no
controlled resignation-mask ablation was run.

The historical pool contained variants of this same weak strategy. It supplied
opponent variety but no demonstrated competent adversary to punish inactivity.
Terminal win/loss is a reasonable eventual objective; full-map cold-start
self-play did not provide useful examples of reaching that objective here.

## 2. The global representation saturated during training

`network.trunk()` passes a learned projection through tanh, shared by the
action-family head and critic. During actual evaluation decisions:

| Global representation diagnostic | Parent | Final |
|---|---:|---:|
| Activations with absolute value above 0.99 | 99.52% | 97.30% |
| Mean per-feature standard deviation across positions | 0.000442 | 0.000491 |
| Critic mean | 0.001835 | 0.203215 |
| Critic standard deviation | 0.000036 | 0.000073 |

The global vector was almost constant across positions. The family head
therefore behaved largely like a fixed preference list filtered by legal masks,
and the critic supplied nearly the same value for very different boards.
The spatial source/target paths are not identical to this global bottleneck;
these measurements do not prove that every network feature was constant.

A separate fixed probe used 64 existing public teacher observations **only for
analysis**, without fitting to them. On exactly those inputs, the random
initial model had 0% saturated global activations and mean feature standard
deviation 0.11245. The parent had 99.39% saturation and standard deviation
0.000609; the final checkpoint had 98.80% and 0.000697. Mean absolute pre-tanh
activation rose from 0.143 to 4.315 and then 5.018. This establishes that the
saturation developed during training rather than being present at random
initialization. It was already present in the parent after the first smoke.

The exact causal contribution of shared actor/critic gradients, initialization,
learning rate, and absent normalization still needs an ablation. Normalization
and activation/gradient monitoring are justified fixes to test; simply claiming
the model is too small is not supported by these measurements.

## 3. Bootstrapping plus normalization can amplify weak information

In a terminal-free segment with gamma 1 and a constant critic C, the TD target
is C again. Small value loss then measures agreement with the model's own
bootstrap, not successful prediction of victories. The last training batch had
zero terminal rewards and value loss 2.32e-9, while its reported pre-clipping
gradient norm remained 0.498.

The learner always standardizes advantages by their batch standard deviation.
For 69 complete, zero-reward, 32-learner-decision chunks extracted from the
final-versus-random replay, the median raw advantage standard deviation was
1.79e-5. Applying the implemented normalization gives a median multiplier of
about **55,765**. These are diagnostic evaluation chunks, not saved historical
training batches: raw training advantages were not logged, so the original
run's exact amplification cannot be reconstructed.

Advantage normalization is not inherently a bug. Here it can turn very small,
weakly grounded differences in bootstrap predictions into sizeable policy
updates. Entropy also continues to produce gradients. The evidence does not
identify how much each term caused the final policy change.

The 32-decision horizon and lambda 0.95 further make long delayed outcomes depend
heavily on the critic. A TD residual 20 steps away receives weight 0.95^20 ≈
0.358; beyond the collected segment, information must come through bootstrap
values. This does not mean PPO can never learn long games, but it is a poor
combination with this nearly constant critic. See the
[GAE paper](https://arxiv.org/abs/1506.02438) for the estimator's bias/variance
tradeoff; the run-specific diagnosis comes from the implementation and probes.

## 4. Confirmed entropy-gradient implementation error

The entropy bonus sums family entropy and the conditional source/target
entropies along a sampled action path. Its expected value at the behavior
policy can equal joint entropy, but differentiating that expression with fixed
sampled branches omits the gradient of the branch probabilities multiplying
downstream entropies. PPO's main joint action log-probability is a separate
calculation and passed its replay checks.

An exhaustive four-action probe of the actual network produced:

| Probe | Exact joint entropy | Implemented sampled-path objective |
|---|---:|---:|
| Expected entropy value | 1.2130076 | 1.2130076 |
| Gradient into move-family bias | +0.2599302 | 0 |
| Gradient into end-family bias | −0.2599302 | 0 |

This is a reproducible mathematical error relative to a joint-entropy bonus,
not just a hypothetical tuning issue. It needs an exhaustive small-space
gradient test and an appropriate estimator, including treatment of behavior
versus current-policy samples across PPO epochs. The distinction between entropy
and entropy-gradient estimators is covered by
[Efficient Entropy for Policy Gradient with Multidimensional Action Space](https://arxiv.org/abs/1806.00589).
Its contribution to this run's strength failure is not isolated by an ablation.

## What the checks do and do not rule out

Probability replay, legal moves, finite parameters, checkpoint reload, and the
frozen simulator hashes passed. A fresh run of the existing scripted native
capture episode also passed: capture can terminate a game with +1/−1 rewards.
That rules out a blanket inability to produce capture outcomes, not every
possible simulator defect or a learning-capacity limitation.

The trained model had only the short initial smoke and this 65,536-learner-
decision continuation. It was not a meaningful test of converged self-play.
However, the reward distribution and representation problems are reasons to
repair the experiment before spending substantially more compute on it.

## Prioritized next experiment

1. Correct the joint-entropy gradient and test it against exhaustive enumeration.
2. Prevent and monitor representation saturation; measure state sensitivity,
   raw advantage scale, critic targets, and component gradient norms. Start from
   a healthy initialization for that comparison, not blindly from this parent.
3. Qualify learning on short, solvable tactical starts using the same frozen
   engine: immediate wins, short captures, and then progressively longer games.
   Keep terminal rewards if preserving pure outcome-based RL is the goal.
4. Mask voluntary resignation during the learning curriculum and pair this with
   actual chances to win. Removing resignation alone will not create positive
   strategic examples. Version and report the policy action restriction.
5. Compare a competent fixed opponent and stronger imitation warm start against
   pure RL, while evaluating on unchanged full maps and reporting city growth,
   army retention, combat, captures, and outcomes—not just loss and legality.

No fix or new training allocation was executed as part of this analysis.

## Evidence

All diagnostics are in `runs/structured-ppo-v2/postmortem/`:

- `replay.py`, both `*-replay.json` reports and `*-trace.npz` arrays.
- `entropy_probe.py` and `entropy-gradient.json`.
- `embedding_probe.py` and `embedding-probe.json`.
- `advantage-probe.json` and `capture-test.log`.

The original protocol, metrics, checkpoints, and evaluations remain in
`runs/structured-ppo-v2/artifacts/pure-league-v1/`.
