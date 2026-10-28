# Structured-state PPO bot

**Historical v1 description.** For the current normalized policy, corrected PPO
learner, resignation restriction, independent-lane self-play and usable
checkpoints, start with [the current guide](structured-rl-current.md).
The experiments below preceded the [failure analysis](structured-rl-postmortem-2026-10-17.md)
and are retained for provenance; they are not the current training recipe.

This is a separate small neural policy in `polytopia_bot.rl`, alongside the
language-model research pipeline. Both use the same frozen `bardur-1v1-v1`
simulator. No simulator rules, existing demonstrations, or language-model
checkpoints are changed. The initial experiment is 14×14 Bardur versus Bardur,
with fog, two seats, and the existing reserved map splits.

## Architecture

Inspired by Duper's structured observations, separate policy action heads,
value prediction, and PPO learner. This implementation uses JAX throughout so
the existing simulator and policy stay on one device. It does not import Duper
code or require its Ray/UCX deployment.

- A 64-channel convolutional encoder with two residual blocks reads public
  board features. A global projection includes own stars, technologies,
  achievements, and public relations.
- Three masked categorical choices form one action: **family → source tile →
  target tile/parameter**. Source is conditioned on family; target is conditioned
  on both. The probability is the product of all three choices.
- All 17 simulator families remain available, including mandatory city rewards,
  diplomacy, and resignation. There is no top-K pruning. The source/target heads
  also represent research, harvest, and other non-movement commands.
- A scalar critic predicts the current player's outcome from the same public
  features. Neither network receives the private engine state. Version one is
  feedforward; it has no recurrent history or search.

## PPO and rewards

Both seats use the same behavior-policy snapshot for each rollout. PPO trains
both seats, then collects the next batch with updated weights. Defaults are
learning rate 3e-4, clip 0.2, gamma 1, GAE lambda 0.95, two optimization epochs,
value coefficient 0.5, entropy coefficient 0.01, and gradient norm cap 0.5.
An approximate-KL threshold stops further minibatches/epochs on that batch.
The original smoke runner uses current-policy self-play. The subsequent
[continuation runner](structured-rl-league.md) adds historical training opponents
and optimizer continuation with a separate learner-only rollout collector.

Only native terminal outcomes supply rewards: **+1 win, −1 loss**. No income,
exploration, combat, score, or survival shaping is added. Resignation is a native
outcome and is counted separately in training; early resignation-heavy results
must not be described as strategic victories.

Values and GAE change sign when the next decision belongs to the other player.
Several actions by the same player retain their sign. Native terminals have no
bootstrap; engine truncations bootstrap the pre-reset final position and cut
the GAE trace. A rollout boundary also bootstraps. Truncations never become draws
or score-based victories. The actor never sees an auto-reset position as the
bootstrap for the preceding episode.

The learner saves the exact behavior log probabilities and conditional masks.
It checks their replay before its first PPO update. The summed conditional
entropies are sampled-path estimates, not an enumeration of every action branch.
Convolutions and matrix products explicitly use highest FP32 precision. A GPU
qualification caught a 0.00154 log-probability discrepancy with default arithmetic
between rollout and learner batch shapes; the runner rejected that batch before
any PPO update. The current implementation requires replay error below 1e-5.

## Optional demonstration warm start

`scripts/prepare_structured_rl.py` takes existing semantic teacher records from
the SFT corpus, checks their training split against the reserved map manifest,
and samples a bounded reservoir across the corpus. It reconstructs the same
tensor observations from public data and supervises the three action choices.
Values receive no imitation targets. Input hashes, teacher hashes, episode IDs,
decision indices, and selected map hashes are retained.

The smoke default uses 2,048 examples for two BC epochs, then resets Adam moments
and runs outcome-only PPO. This is **BC + RL**, not RL from random initialization.
Set `--bc-epochs 0` to disable imitation gradients and train from random weights.
Training-set demonstration loss is a fit diagnostic, not a held-out measure.
Prepare with `--count 0` to produce a split-only input directory without reading
or requiring any demonstration file. Use that directory with BC disabled.
For the Modal from-scratch control, prepare the split-only directory at
`runs/structured-ppo-v1/pure-input` and pass `--bc-epochs 0` with a new run ID.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_structured_rl.py \
  --output runs/structured-ppo-v1/input --count 2048
.venv/bin/python -m pytest tests/test_structured_rl.py tests/test_native_rules.py -q
.venv/bin/python scripts/verify_ruleset.py
modal run --env staging experiments/modal_structured_rl.py --run-id my-bc-run

# Independent pure-RL control, without demonstration input:
PYTHONPATH=src .venv/bin/python scripts/prepare_structured_rl.py \
  --output runs/structured-ppo-v1/pure-input --count 0
modal run --env staging experiments/modal_structured_rl.py \
  --run-id my-pure-run --bc-epochs 0
```

The default model contains **481,379 parameters**. The reserved validation maps
are used for development evaluation; the existing final-test partition remains
untouched.

Use a new run ID for every allocation. Existing result directories are never
overwritten. The Modal entry point is staging-only, one L40S, eight CPU cores,
32 GiB RAM, no retries, no service, and a 3,600-second hard timeout. The PPO phase
has a 900-second soft limit checked at update boundaries. The default target is
32 updates × 32 steps × 8 environments = **8,192 self-play atomic decisions**.
Startup, compilation, warm start, and evaluation share the one-hour allocation.
At the October 17, 2026 base rates this is approximately $2.58 per allocated hour,
excluding image builds, storage, plan charges and any resource overage.

Checkpoints, optimizer arrays, source hashes, config, per-update metrics, and
evaluations persist on the separate `polytopia-structured-rl` Modal volume. The
runner verifies exact checkpoint reload. The continuation runner can restore
policy and optimizer state; a complete rollout/RNG resume is not implemented.

Local execution is also possible (the simulator selects CPU on the Mac):

```bash
PYTHONPATH=src .venv/bin/python scripts/train_structured_rl.py \
  --input runs/structured-ppo-v1/input --output runs/structured-ppo-v1/local \
  --updates 4 --bc-epochs 0
```

## Playing the checkpoint

```python
from polytopia_bot.learning.adapter import GPUEnvironment
from polytopia_bot.rl.policy import StructuredPolicy

env = GPUEnvironment(size=14)
obs = env.reset(seed=100541, settings={})
bot = StructuredPolicy("runs/structured-ppo-v1/artifacts/smoke-v2/final.npz")
decision = bot.select(obs)
transition = env.step(action_id=decision.action.action_id,
                      expected_state_version=obs.state_version)
```

The host adapter is for compatibility with the existing bot interface. The
training runner uses the direct batched device path.
The example checkpoint is from the completed smoke run; see the
[October 17 results and artifact paths](structured-rl-smoke-2026-10-17.md).

## Evaluation and interpretation

Warm-start and post-PPO policies play a frozen, family-uniform random opponent
on the same reserved validation maps, with swapped seats and fixed sampling
seeds. Post-PPO also plays the frozen warm-start policy. The random opponent
excludes resignation; the trained policy does not. Each smoke evaluation has
two maps × two seats and a 512-action evaluation cap, reported separately from
the engine's own truncation. No final-test maps are used.

These tiny evaluations can validate legal play and detect gross failures. They
cannot establish expert play, generalization, or a significant strength gain.
Sparse rewards may mostly teach the initial policy not to resign; low initial
completion and little strategic learning are plausible. A subsequent experiment
should use more maps, full games, fixed strategic opponents, and multiple seeds,
and compare the BC-only checkpoint against BC+PPO at matched evaluation settings.
