| arm | W–L–cap | score [95% CI] | % of decisive | Elo over raw c3 [95% CI] | s / searched turn | lane-steps / turn | override | mean turns | $ | chunks |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| raw s1-final (no search) | 151–26–23 | 0.812 [0.77, 0.85] | 85% | **+255 [+212, +305] (n=100 maps)** | 0.08 | 13 | 0.00 | 42.9 | 0.70 | 4/4 |
| s1-final + leak-free d=1, 1 reply | 167–17–16 | 0.875 [0.84, 0.91] | 91% | **+338 [+285, +407] (n=100 maps)** | 4.72 | 1036 | 0.87 | 40.0 | 2.14 | 4/4 |
| s1-final + leak-free d=1, 64 replies (robust) | 173–19–8 | 0.885 [0.84, 0.93] | 90% | **+354 [+295, +436] (n=100 maps)** | 12.84 | 3142 | 0.85 | 36.5 | 6.05 | 7/7 |
| s1-final + leak-free d=1, 256 replies (robust) | 30–2–0 | 0.938 [0.84, 1.00] | 94% | **+470 [+293, +1200] (n=16 maps)** | 43.11 | 11095 | 0.87 | 34.7 | 2.32 | 1/7 |

Paired comparisons (Elo, same maps and seats, map bootstrap):
- 1 reply over raw: +83 [+21, +153] (n=100 maps)
- 64 replies over raw: +100 [+34, +173] (n=100 maps)
- 256 replies over raw: +197 [+56, +927] (n=16 maps)
- 64 replies over 1 reply: +16 [-65, +100] (n=100 maps)
- 256 replies over 1 reply: +177 [+45, +885] (n=16 maps)
- 256 replies over 64 replies: +76 [+0, +729] (n=16 maps)

Score by map type (games):
| arm | archipelago | classic | continents | drylands | highlands | jungle | lakes | pangea | ridge | waterworld |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw s1-final (no search) | 0.92 (26) | 0.77 (26) | 0.79 (28) | 0.69 (8) | 0.82 (14) | 0.89 (14) | 0.75 (20) | 0.70 (28) | 0.92 (24) | 0.88 (12) |
| s1-final + leak-free d=1, 1 reply | 0.94 (26) | 0.90 (26) | 0.88 (28) | 0.88 (8) | 0.75 (14) | 0.86 (14) | 0.82 (20) | 0.89 (28) | 0.90 (24) | 0.83 (12) |
| s1-final + leak-free d=1, 64 replies (robust) | 0.92 (26) | 0.83 (26) | 0.88 (28) | 0.81 (8) | 1.00 (14) | 0.96 (14) | 0.93 (20) | 0.77 (28) | 0.92 (24) | 0.92 (12) |
| s1-final + leak-free d=1, 256 replies (robust) | 0.50 (2) | 1.00 (4) | 1.00 (6) | 1.00 (2) | 1.00 (2) | 1.00 (2) | — | 0.75 (4) | 1.00 (10) | — |

Critic calibration of the greedy lane (searched arms, first plan of each turn):
- win-d1det-ownval-vs-c3: mean greedy value +0.304, mean outcome +0.651, corr 0.620, mse 0.356 (n=8169)
- omni-d1det-r64-vs-c3: mean greedy value +0.352, mean outcome +0.751, corr 0.564, mse 0.383 (n=7476)
- omni-d1det-r256-vs-c3: mean greedy value +0.379, mean outcome +0.902, corr 0.374, mse 0.454 (n=1142)

breadth arms $ 8.37
