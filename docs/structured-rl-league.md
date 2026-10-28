# Structured PPO continuation and checkpoint opponents

The first follow-up has completed; see its
[training, evaluation, and cost results](structured-rl-league-result-2026-10-17.md).

`polytopia_bot.rl.league` extends the initial shared-policy self-play runner.
It loads policy weights and Adam moments from an integrity-checked checkpoint,
starts fresh games and a new random stream, and trains against a mixture of the
current policy and historical snapshots. This is optimizer continuation, not
bitwise resumption of the earlier rollout/RNG state.

For each rollout, opponent selection is 50% the current behavior-policy snapshot
and 50% a uniformly sampled historical snapshot. The initial pool contains the
continued parent. A new snapshot is added every 16 updates, retaining the newest
four. All lanes share the selected opponent during that rollout. Opponents may
change at rollout boundaries even for ongoing games; they remain frozen within
each on-policy batch. The learner seat is randomized on reset, with initially
balanced seats.

## Learner-only transitions

A training transition consists of one learner decision and any subsequent
opponent moves until the learner acts again or the episode ends. Only the
learner action, masks, and behavior probability enter PPO. Opponent actions are
environment responses; they never enter the policy loss.

Reward is the native terminal outcome from the learner seat's perspective.
There is no intermediate shaping. GAE operates per learner decision, with
gamma 1 and lambda 0.95, without the sign changes needed by the original
both-seats atomic-action collector. A terminal has no bootstrap. An engine
truncation bootstraps the learner's public view of the final position before
reset, even when the opponent made the last move, and cuts the trace.

If an opponent ends a new game before the learner has taken any action, the
collector records that outcome separately as uncredited and starts a new game.
It does not fabricate a learner action. Opponent loops and reset retries are
bounded; exhaustion or an illegal action aborts the batch rather than inventing
a game outcome. A JAX scan collects the batch on device. Every batch's behavior
log probabilities are replay-checked before its PPO update.

The same frozen `bardur-1v1-v1` simulator, public observation encoder, 481,379
parameter model, and full legal action catalog are used. Resignation remains
available. Native completions must therefore be separated into resignations
and eliminations when interpreting training progress.

## Running it

The bounded Modal follow-up profiles 8 and 32 lanes, chooses the faster measured
collection rate, and continues `pure-smoke-v2/final.npz` from the existing
`polytopia-structured-rl` staging volume:

```bash
modal run --env staging experiments/modal_structured_league.py --run-id my-league-run
```

The allocation is one L40S, eight CPU cores, 32 GiB RAM, no retries, and a one-hour
hard timeout. Training stops at 128 updates or a 900-second soft budget checked
at batch boundaries. Profiling, compilation, and evaluation share the hard
timeout. It uses 128 training maps and eight validation maps in paired seats:
16 games each for parent versus random, final versus random, and final versus
parent. The evaluation action cap is 10,000, matching the frozen engine's action
safety limit, so the earlier 512-action evaluation cutoff is removed. Engine
truncations still remain distinct from wins, losses, and evaluation cutoffs.

Local continuation accepts an explicit checkpoint:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_structured_league.py \
  --input runs/structured-ppo-v1/pure-input \
  --checkpoint runs/structured-ppo-v1/artifacts/pure-smoke-v2/final.npz \
  --output runs/structured-ppo-v2/local --environments 8 --updates 8
```

The final checkpoint saves optimizer state. Parent hashes, source hashes,
opponent selection, per-batch outcomes, sampled snapshot IDs, throughput,
evaluations, and exact final policy/optimizer reload checks are retained beside
it. A run directory must be new. The initial smoke results remain separate.

Tests in `test_structured_league.py` cover opponent terminal credit, exclusion
of opponent actions from PPO, learner-view truncation bootstrap, bounded-loop
failure, pre-decision game completion, and identical next Adam updates after
checkpoint reload.
