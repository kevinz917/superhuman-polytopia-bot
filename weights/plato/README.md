# Plato

`final.npz` holds the policy of **Plato**, the strongest 1v1 bot so far. It is a 3.2M-parameter hybrid
convolution + transformer network trained **from random weights by self-play reinforcement learning (PPO)** under the
game's own rules (`polytopia-2.17.3`, 1v1, 14×14 classic maps). Training took four stages and 85,270 updates
(about 25.6 billion samples, ~337 H100-hours, ~$1,498 including evaluations). The last stage taught it to play with
**peace treaties legal**. In that stage a game that reaches the round cap without an elimination counts as a loss for
**both** sides.

With peace legal it beats hero-1v1 **89%** and the champion **97%**, and it won **48–0** against a replica of the
built-in Crazy AI. Almost every win is an elimination: it takes the opponent's last city.

`final.json` records the training config, the parameter count (3,212,922), the checkpoint's SHA-256 and the ruleset.
Loaders refuse the file if either does not match. Optimizer state was stripped for release. `final-critic.npz` is
the run's privileged critic (400,805 parameters, it reads both seats' views); it is only needed to continue training.

- Ruleset: `polytopia-2.17.3`, 14×14, classic maps, 1v1, resignation disabled
- Architecture: `transformer-v3-p2-h4`, 6 transformer blocks × 128 channels, 4 attention heads, observation version 3
- SHA-256 (`final.npz`): `aced9e4ea74527631b63236a0e0697bff37896549d7349083698d672c15e9420`
- SHA-256 (`final-critic.npz`): `5d16daf0340b93ea542857d7f3d6adb2064028b355ff659c2bf6b9456f6adf99`
- Training update 85,270

## Results

Preregistered final check: raw policies, sampled actions, no search, 1,024 fresh maps, each played twice with the
seats swapped (2,048 games per row; 512 for the exploiters). Ranges are 99% intervals. "Capped" games hit the
200-round harness limit; scoring those as draws instead of score wins changes no row by more than .014.

| Plato vs | Peace legal | No peace | Elo gap (peace legal) | Capped (peace legal) |
|---|---|---|---|---|
| hero-1v1 ([`weights/hero-1v1`](../hero-1v1)) | **.892** [.874, .909] | .853 [.834, .873] | +366 | 5 / 2,048 |
| the champion ([`weights/champion`](../champion), raw) | **.972** [.962, .981] | .984 [.977, .991] | +614 | 57 / 2,048 |
| its own $1,000 checkpoint (end of stage 3) | .981 [.973, .989] | .528 [.506, .550] | +685 | 4 / 2,048 |
| a CNN control (hero-1v1 architecture, same recipe, $100) | 1.000 | .934 | — | 0 |
| Cloak exploiter (built to beat an earlier Plato) | .996 | .961 | — | 0 / 512 |
| x3d peace exploiter (built to beat an earlier Plato) | 1.000 (512–0) | — | — | 0 / 512 |

All 57 capped games against the champion, and all 5 against hero-1v1, were lost on score. Training scores a capped
game as a loss for both sides, so Plato has no reason to lead on score at the cap. By elimination alone it is
1,990–1 against the champion.

Against a **replica** of Polytopia 2.17.3's built-in AI at Crazy difficulty (re-implemented on engine-rs, not the
retail game), the raw policy won **48–0**: 24 maps, both seats, wins on turns 9–20 (median 16), zero rejected commands.

The full report, with every earlier checkpoint, the learning curve, the peace-collusion monitors and the
negatives, is [docs/plato-2026-11-09.md](../../docs/plato-2026-11-09.md). The machine-readable results are in
[docs/benchmarks/2026-11-09-plato.json](../../docs/benchmarks/2026-11-09-plato.json).

## Training in brief

| Stage | Updates | Hardware | Batch | Rules in training | Spend |
|---|---|---|---|---|---|
| 1 | 0 → 8,169 | 8 H100 | 262,144 | no peace, score at cap | $97 |
| 2 | → 23,270 | 8 H100 | 262,144 | no peace, score at cap | $183 |
| 3 | → 72,889 | 16 H100 | 262,144 | no peace, score at cap | $698 |
| 4 pilots | → 73,870 | 8 H100 | 524,288 | 2 of 4 game sets peace-legal | $41 |
| 4a | → 79,150 | 16 H100 | 524,288 | 2 of 4 sets peace-legal, peace self-play .30 | $187 |
| 4b | → 85,270 | 16 H100 | 524,288 | 3 of 4 sets peace-legal, peace self-play .30, **cap = loss for both** | $230 |

Training cost ≈ $1,436 and evaluations ≈ $62, so the total is $1,497.84. These are list-price estimates from the
run supervisor's clock (H100 at about $4.27 per GPU-hour); the provider's invoice may differ.

The final recipe:

- **Batch:** the "b2" bigger batch, 524,288 samples and 32,768 minibatch, with the learning rate scaled by √2.
- **Game sets:** each GPU runs four sets of 1,024 games. Three of the four sets allow peace and one forbids it.
- **Peace-legal opponents:** self-play 30%, historical snapshots 10%, and fixed opponents 60%. The fixed opponents are
  hero-1v1, the champion, the CNN control, three earlier Plato checkpoints and the two exploiters. Fixed opponents
  don't collude.
- **Terminal reward:** a game ends at elimination (+1 / −1), and a game reaching the 200-round cap ends **−1 for both
  sides** with no critic bootstrap.
- **Collusion monitors** warn or pause the run on capped self-play, steps spent at peace, and late-round play.

Configs: [`configs/plato.json`](../../configs/plato.json) (the final learner config) and
[`configs/plato-recipe.json`](../../configs/plato-recipe.json) (stages, learning-rate schedules, opponent pools,
monitors and totals).

## Loading it

The file follows this folder's format: arrays `p0` … `p105`, the leaves of the network's JAX parameter tree in
`jax.tree.flatten` order, next to `final.json`. Check integrity with
`shasum -a 256 weights/plato/final.npz` or this repository's `polytopia_bot.rl.runner.load_checkpoint`. That loader
checks the hash, the ruleset, every tensor shape and finiteness.

**Not yet runnable from this repository.** The `transformer-v3` network builder that turns these arrays into a
playable policy is not part of this repository yet; `network.init` here knows only `legacy` and `normalized-v2`. So
the Studio (`./scripts/studio.sh --checkpoint …`) and the arena cannot play Plato until that code is added. The
weights, configs and results are complete. Every result above was produced with that network code on engine-rs.

## Limitations

- **Rules engine, not the retail game.** All games ran on engine-rs, this repository's implementation of the
  2.17.3 rules. Plato has not played the retail game. The 48–0 is against a replica of the built-in AI, unlike the
  champion's 48–0 against the real game's Crazy AI.
- **Narrow setting.** Training and testing used only 1v1 on 14×14 classic maps, with a 200-round cap and
  resignation disabled. Other map types, sizes, four-player games and the game's own turn limits are untested.
- **Diplomacy is learned, not solved.** Plato still makes about 17 diplomatic actions per game. In the last training
  window 4% of its self-play games with peace legal still reached the cap, and 10% of game steps were played at
  peace.
- **Exploitability of Plato itself is untested.** Both exploiters were trained against the $300 checkpoint. No new
  exploiter was trained against Plato.
- **No search.** Every result is the raw policy. The critic reads the opponent's view, so it cannot guide turn
  search at play time. Plato has no fog-of-war value head.
- **Not certified.** The training rule changed mid-stage, so no ablation separates the effect of the cap = loss rule
  from time and learning-rate annealing. Costs are estimates.
