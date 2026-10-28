# First structured self-play PPO runs — October 17, 2026

The first version trains and plays through the shared frozen simulator. Two
bounded Modal runs completed: pure RL from random weights, and PPO after a small
demonstration warm start. Both performed 32 updates and 8,192 self-play atomic
decisions with zero illegal actions. **This validates the implementation, not
strategic strength.** Every observed native completion was a resignation;
neither run demonstrated an elimination-based win.

## What was built

`src/polytopia_bot/rl/` contains a 481,379-parameter JAX CNN policy and critic,
masked family/source/target action heads covering all 17 simulator families,
clipped PPO, alternating-player GAE, optional behavioral cloning, checkpoints,
fixed-opponent evaluation, and a public-observation bot adapter. It uses the
same `bardur-1v1-v1` rules and 14×14 Bardur maps as the other bot.

Both players use one frozen behavior-policy snapshot during a rollout. The
learner updates it from both players' experience before collecting again.
Rewards are native terminal ±1, with zero intermediate shaping. Engine
truncations bootstrap and cut the trace; they are not counted as draws or wins.
Version one has current-policy self-play, without a historical-opponent league.
See [architecture and reproduction](structured-rl.md).

## Allocation and data

Each run used one L40S, eight CPU cores, 32 GiB RAM, eight environments, 32-step
rollouts, two PPO epochs, and 64-example minibatches. Both used seed 20261017 and
the same 32 training maps. Evaluation used reserved validation seeds 100541 and
100542 in both seats. No final-test maps were evaluated.

- **BC + PPO (`smoke-v2`):** 2,048 audited teacher decisions from the existing SFT
  training corpus, two imitation epochs, then fresh optimizer moments for PPO.
  The training probe's negative log likelihood fell from 2.8862 to 2.2194 during
  BC. This is a training fit diagnostic, not held-out accuracy.
- **Pure PPO (`pure-smoke-v2`):** random initialization, no BC, and a split-only
  input manifest with zero examples and no demonstration data file. No
  demonstration loss was computed.

## Observed results

| Measurement | BC + PPO | Pure PPO |
|---|---:|---:|
| Self-play decisions | 8,192 | 8,192 |
| PPO updates | 32 | 32 |
| Native training completions | 217 | 431 |
| Training resignations | 217 | 431 |
| Training eliminations | 0 | 0 |
| Engine training truncations | 7 | 8 |
| Illegal training actions | 0 | 0 |
| PPO phase, including its compilation/checks | 257.6 s | 262.6 s |
| Measured training decisions/second | 31.80 | 31.19 |
| Full runner, including initialization and evaluations | 394.0 s | 392.4 s |
| Maximum behavior log-probability replay error | 9.54e-7 | 4.77e-7 |
| L2 parameter change during PPO | 2.4313 | 2.5479 |

Each evaluation row below contains four games, with stochastic sampling and
paired seats. The fixed random opponent samples legal action families uniformly
and excludes resignation. The learned policies retain resignation.

| Evaluation | Wins | Losses | Engine truncations |
|---|---:|---:|---:|
| BC checkpoint vs random, before PPO | 0 | 4 | 0 |
| BC + PPO vs random | 0 | 3 | 1 |
| Random initialization vs random, before PPO | 0 | 4 | 0 |
| Pure PPO vs random | 0 | 1 | 3 |
| BC + PPO vs its frozen BC checkpoint | 4 | 0 | 0 |
| Pure PPO vs its frozen initial checkpoint | 4 | 0 | 0 |

All wins and losses in this table came from resignations. There were no
evaluation action-cap cutoffs or illegal actions. Before PPO, the four games
against random lasted 27 total decisions for BC and 18 for random initialization;
after PPO they lasted 1,470 and 1,283 respectively. This supports learning to
avoid immediate resignation, not learning strong economy, tactics, or captures.
Four games from one training seed cannot establish that pure PPO outperforms
BC + PPO.

## Validation and an issue caught on GPU

The combined structured-RL and native-rules suites passed **37 tests**. They cover
joint action probabilities and complete masks, hidden-state exclusion,
demonstration/public observation parity, turn-sign changes, terminal and
truncation handling, the PPO gradient direction, checkpoint reload, and the
existing bot interface. All **22 frozen rule source hashes** still match.

An initial GPU attempt (`smoke-v1`) failed the behavior-probability replay check
before any PPO update: default matrix/convolution precision produced a 0.00154
difference between rollout and learner batch shapes. Explicit highest FP32
precision fixed this; both v2 runs passed the 1e-5 runtime threshold. The first
pure-control attempt was stopped to apply the same correction.

Both completed runs report finite parameters and exact warm-start checkpoint
reload. Downloaded checkpoints were checked against their recorded SHA-256
hashes, and each run's 29 source hashes were checked against the current RL and
GPU modules. The downloaded final policies also load through `StructuredPolicy`
and select legal actions through the existing `GPUEnvironment` interface.

## Cost and scope of the compute estimate

Using [Modal's base pricing](https://modal.com/pricing), the allocation is
approximately **$2.5842/hour**: $1.9512 for one L40S, $0.37728 for eight CPU cores,
and $0.255744 for 32 GiB RAM. The two successful runners' measured 786.3 seconds
correspond to approximately **$0.56** at those rates. This is a resource-time
estimate, not an invoice; it excludes build, storage, plan charges, and overage.
Pricing every app's entire wall lifetime at that allocation rate, including the
failed and canceled qualification attempts, gives approximately **$0.80**. That
is an accounting proxy, not measured billing.

A full day at this one-GPU allocation would be about **$62.02**. No day-long run
was launched. The measured ~31 decisions/second comes from a deliberately tiny,
host-coordinated eight-environment smoke configuration with compilation overhead.
It is not a GPU saturation benchmark and cannot support a claim that one day
produces a strong bot or that throughput scales linearly to a Duper-sized
cluster. Larger batched rollouts should be profiled before sizing that run.

All four apps used for this work are stopped with zero tasks. Curio and Duper
were reference-only; their training deployment was not changed.

## Artifacts

| Artifact | Location |
|---|---|
| BC + PPO Modal app | [ap-N3tFv9DoUjg9p0YXnFA8ap](https://modal.com/apps/duper/staging/ap-N3tFv9DoUjg9p0YXnFA8ap) |
| Pure PPO Modal app | [ap-stKCvVkgtu9KY7dJjRYD2P](https://modal.com/apps/duper/staging/ap-stKCvVkgtu9KY7dJjRYD2P) |
| Persistent volume | `polytopia-structured-rl`, staging, directories `/smoke-v2` and `/pure-smoke-v2` |
| BC + PPO local artifacts | `runs/structured-ppo-v1/artifacts/smoke-v2/` |
| Pure PPO local artifacts | `runs/structured-ppo-v1/artifacts/pure-smoke-v2/` |
| Warm-start provenance | `runs/structured-ppo-v1/input/input.json` |
| Pure-RL provenance | `runs/structured-ppo-v1/pure-input/input.json` |
| Test receipt | `runs/structured-ppo-v1/final-tests.log` |
| Download verification script | `runs/structured-ppo-v1/verify_downloads.py` |
| App shutdown receipt | `runs/structured-ppo-v1/modal-apps-final.json` |

Each artifact directory contains `final.npz` with its integrity/config JSON,
initial and warm-start checkpoints, `protocol.json` with source hashes,
`training.json`, `summary.json`, and `local-verification.json`. Binary artifacts
are retained locally and in Modal, not committed to Git.

The next experiment should measure strategic completion on more held-out maps
and training seeds, use fixed strategic baselines, and compare pure RL, BC-only,
and BC + PPO. A historical-checkpoint opponent pool can reduce cycling. Those
are subsequent experiments, not capabilities or results claimed by this smoke.
