# Pure-RL continuation with checkpoint opponents — October 17, 2026

The continuation completed **64 updates, 65,536 learner decisions, and 64,392
opponent moves**. It demonstrates working optimizer continuation, compiled
learner-only rollouts, historical opponents, and larger-batch training. It does
**not** demonstrate strong strategic play: no training or evaluation game ended
by elimination, and the final policy still won zero games against random.

## Implementation and run

The [league runner](structured-rl-league.md) continued the pure-RL checkpoint
`pure-smoke-v2/final.npz`, without demonstrations. It restored Adam step 253 and
finished at step 765. Games and RNG were restarted; this was not an exact
full-state resume. The simulator and its 22 frozen source hashes were unchanged.

Opponents were selected once per rollout: 37 batches used the current policy and
27 used historical snapshots. Historical versions actually sampled were the
parent and updates 16, 32, and 48. Update 64 was saved for future use. The pool
retains four versions and can change opponents at batch boundaries within an
ongoing game. Opponent actions are environment responses, never PPO samples.

Allocation: one L40S, eight CPU cores, 32 GiB RAM, 32 environments, a 32-decision
learner horizon, minibatches of 256, two PPO epochs, and 128 reserved training
maps. The 900-second soft training budget stopped after update 64, at 906.1
seconds including compilation, checks, and checkpoint writes. All rewards
remained native terminal ±1; truncations bootstrap rather than become draws.

## Throughput

| Measurement | Learner decisions/second |
|---|---:|
| 8-lane collection profile | 36.72 |
| 32-lane collection profile | 93.50 |
| Full training, including first-batch compilation and checks | 72.33 |
| Updates after the first compilation, measured batch times | 81.29 |

The 32-lane early-game collection profile was **2.55×** the 8-lane profile, so
the run selected it. These short profiles used three timed eight-decision
rollouts after compilation. They are not GPU saturation benchmarks, and their
learner decisions include intervening opponent actions. Do not directly compare
them with the original smoke's both-seats atomic-decision throughput.

## Outcomes

Training produced 112 native completions, all resignations (69 learner wins and
43 learner losses), plus 394 engine truncations. There were no illegal actions,
opponent-loop failures, or pre-decision uncredited completions. Near the end,
some entire batches had no terminal reward.

Evaluation used validation seeds 100541–100548 with swapped seats: 16 games per
matchup. Parent and final versus random used matched maps and sampling seeds.
The random opponent excludes resignation; both learned policies retain it.

| Matchup | Wins | Losses | Engine truncations |
|---|---:|---:|---:|
| Parent vs fixed random | 0 | 5 | 11 |
| Final vs fixed random | 0 | 1 | 15 |
| Final vs parent | 2 | 3 | 11 |

Every decided game ended by resignation. All 48 evaluation games had zero
illegal actions and zero evaluation-cap cutoffs. Truncated games are censored,
not draws. This small, single-training-seed comparison does not establish a
strength gain, nor isolate the effect of the opponent pool from longer training
and the changed collector.

## Opening diagnostic

The same eight validation maps were inspected at their initial seat-1 positions:

| Mean action-family probability | Parent | Update 32 | Final |
|---|---:|---:|---:|
| End turn | 85.72% | 61.98% | 44.26% |
| Research | 7.08% | 31.63% | 48.94% |
| Move | 4.16% | 2.76% | 3.31% |
| Harvest | 2.83% | 3.58% | 3.48% |
| Resign | 0.215% | 0.051% | 0.015% |

The model changed its opening behavior mainly toward research, with movement
still rare. This is an initial-state diagnostic, not a whole-game action census.
Together with the absence of elimination outcomes, it points to exploration
and credit assignment as the next questions. A tactical starting-position
curriculum or a stronger demonstration warm start should be tested against
unchanged full-game evaluation before assuming a longer version of this run
will produce strong play. Neither intervention was added to this experiment.

## Verification, cost, and artifacts

- **42 tests passed** across structured PPO, the new collector/continuation
  cases, and native rules. All 22 frozen simulator hashes still match.
- Behavior probabilities were checked before every PPO update; maximum replay
  error was **3.34e-6**, below the 1e-5 threshold.
- The final policy and Adam state reloaded exactly. Local verification matched
  all **30 source hashes and seven checkpoint hashes**, checked finite tensors,
  and executed 12 legal decisions through the existing host bot interface.
- Policy parameter L2 change from the continued parent was 3.32293. This
  confirms changed weights, not improved playing strength.
- The Modal app is stopped with zero tasks. Its wall lifetime was 1,357 seconds
  (22m 37s), including profiling. At the October 17
  [Modal base rates](https://modal.com/pricing) used in the earlier estimate,
  $2.584224/hour for this allocation gives **about $0.97**. This is an app-wall
  resource-cost proxy, not an invoice; build, storage, plan charges, and overage
  are not included.

Modal app: [ap-9bvWnhctmresyRKxNvOtpV](https://modal.com/apps/duper/staging/ap-9bvWnhctmresyRKxNvOtpV).
The staging volume is `polytopia-structured-rl`, directory `/pure-league-v1`.
Downloaded artifacts are in `runs/structured-ppo-v2/artifacts/pure-league-v1/`.
That directory contains initial/final/latest checkpoints, four snapshots,
protocol/source hashes, per-update metrics, profiling, evaluations, the summary,
and local verification. Final checkpoint SHA-256:

```text
c5c0ff28d0c00c4d4d5c97818e217d15f89be694bcfa8f10fddbbc702d8bd79f
```

Additional receipts are `runs/structured-ppo-v2/final-tests.log`,
`opening-diagnostic.json`, `opening_diagnostic.py`, and `apps-final.json`.
To repeat the downloaded-checkpoint verification:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_structured_checkpoint.py \
  runs/structured-ppo-v2/artifacts/pure-league-v1
```
