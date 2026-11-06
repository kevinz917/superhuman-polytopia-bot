# Checkpoints

Every folder holds `final.npz` (the policy) and its `final.json`: the training config, the update it was saved at,
the parameter count, the ruleset it was trained under and the file's SHA-256. Loaders refuse a file whose hash does
not match. Most folders also hold the run's critic (`final-critic.npz`), which is only needed to continue training.
Optimizer state was stripped from every policy.

Three rulesets appear below. `polytopia-2.17.3` is the game's own rules, the only ruleset engine-rs plays.
`bardur-1v1-v2` and `bardur-1v1-v1` are the earlier research rules; those checkpoints play on the legacy engine in
[`legacy/ruleset-v2`](../legacy/ruleset-v2) (v1 differs from v2 in a few rules).

## The headline bots

| Folder | What it is | Network | Ruleset |
|---|---|---|---|
| [`champion/`](champion) | The site's bot: from-scratch self-play plus a fog-of-war value head for turn search | 2x64 | bardur-1v1-v2 |
| [`hero-1v1/`](hero-1v1) | Strongest 1v1 bot on the game's rules: 68% vs the champion | 4x128 | polytopia-2.17.3 |

## 1v1 lineage, research rules

| Folder | Update | What it is | Result | Notes |
|---|---|---|---|---|
| `c3/` | 4,854 | Imitation-bootstrapped PPO, the first strong baseline | the champion beats it 92% | [docs](../docs/campaign-log-2026-10-19-20.md) |
| `omni-c5/` | 2,212 | 6-h continuation of c3 with an omniscient critic | 77% vs c3 (+255 Elo); previous best | [docs](../docs/c5-omni-long-2026-10-21.md) |
| `scratch-overnight/` | 8,724 | 7 h of self-play from random weights | 83% vs c3, 52–58% vs omni-c5 | [docs](../docs/scratch-check-2026-10-22.md) |
| `b1-polish/` | 2,886 | Low-learning-rate polish of the champion | +41 Elo vs the champion (no LR control) | [docs](../docs/pilots-2026-10-23.md) |
| `bn-main4/` | 7,300 | The champion recipe at 4x128, longer training | hero-1v1 beats it 60% | [docs](../docs/verda-big-run-2026-10-23.md) |

## 1v1, the game's rules

| Folder | Update | What it is | Result | Notes |
|---|---|---|---|---|
| `bighero/` | 18,840 | 26 h from random weights on a self-hosted cluster, 4x128 | plateaued: no mid-game city growth | [docs](../docs/curio-hero-2026-10-31.md) |
| `farsight-v2/` | 9,779 | obs v3 with history and gated global pooling, 12 h | 80% vs bighero at equal updates; 21% vs the champion | [docs](../docs/farsight-v2-arm-b-analysis-2026-11-03.md) |
| `capfix-arm-b/` | 28,957 | farsight-v2 with score-at-cap rewards, 7.6 B samples | 56% vs the champion | [docs](../docs/farsight-v2-arm-b-analysis-2026-11-03.md) |
| `arm-c/` | 5,180 | bighero recipe against frozen developer opponents | 57–39 vs bighero u5250; 4% vs the champion | [run archive](../docs/run-archive/farsight-v2-capfix-2026-11-03) |
| `lookout/` | 13,931 | Hybrid attention trunk, 2x128 | trails farsight-v2 at equal samples (0.31) | [docs](../docs/lookout-2026-11-01.md) |

`hero-1v1/final-critic.npz` is that run's critic, for continuing it.

## Four players, 18x18

| Folder | Update | What it is | Result | Ruleset | Notes |
|---|---|---|---|---|---|
| `ffa4-big2/` | 7,650 | Four-seat FFA, continued from the converted champion, 2x64 | +158 Elo over the converted champion | bardur-1v1-v2 rules, 4 seats | [docs](../docs/four-player-2026-10-24.md) |
| `hero-ffa4/` | 20,250 | Four-seat FFA from random weights, 4x128 | no final table recorded | polytopia-2.17.3 | [docs](../docs/hero-ffa4-levers-2026-10-28.md) |

Results are seat-paired arena win rates from the linked docs, on fresh maps.
