# hero-1v1 checkpoint

`final.npz` holds the policy of `hero-1v1`, the strongest 1v1 bot: self-play PPO from random weights under the
game's own rules (`polytopia-2.17.3`), in two runs of about 8 h and 7.5 h (6.95 billion samples, update 11,650).
It is a larger network than the champion and has **no fog-of-war value head yet**, so it plays without turn search.
`final.json` records the training config, the parameter count (1,972,387), the checkpoint's SHA-256 and the ruleset
it was trained under. Loaders refuse the file if either does not match. Optimizer state was stripped for release.

- Ruleset: `polytopia-2.17.3`, 14×14, `proposed-v1` map mix
- Architecture: `normalized-v2`, 4 residual blocks × 128 channels
- SHA-256: `e6f4e1977e7f9cdcfbbe17358d87e49d638d82150496a6d4099e0cd389b0934f`

Final round, raw policies, 64 maps × 2 seats = 128 games per opponent:

| hero-1v1 vs | Win rate | Elo gap |
|---|---|---|
| its first run's final checkpoint | 61% | +80 |
| bn-main4 (champion architecture, longer training) | 60% | +74 |
| B1 (low-LR polish of the champion) | 63% | +97 |
| the champion | 68% | +131 |
| c3, imitation-bootstrapped | 97% | — |

Play it in the Studio with `./scripts/studio.sh --checkpoint weights/hero-1v1/final.npz` (no `--search`).
Run configs: [`experiments/hero/hero-1v1.json`](../../experiments/hero/hero-1v1.json) and
[`experiments/hero/hero-1v1-cont1h.json`](../../experiments/hero/hero-1v1-cont1h.json); the plan and the throughput
work behind them are in [docs/hero-runs-plan-2026-10-26.md](../../docs/hero-runs-plan-2026-10-26.md) and
[docs/hero-throughput-2026-10-28.md](../../docs/hero-throughput-2026-10-28.md).
