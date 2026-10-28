| arm | depth | games | W–L–cap | score [95% CI] | of decisive | cap rate | Elo vs raw c3 [95% CI] | Elo minus d=1 [95% CI] | s / searched turn | lane steps / turn | incomplete lanes | override | mean turns | $ | chunks |
|---|---:|---:|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| raw (no search) | — | 200 | 151–26–23 | 0.812 [0.77, 0.85] | 85% | 12% | **+255 [+212, +305]** | — | 0.08 | 13 | 0.0% | 0.00 | 42.9 | 0.70 | 4/4 |
| leak-free d=1 | 1 | 200 | 167–17–16 | 0.875 [0.84, 0.91] | 91% | 8% | **+338 [+285, +407]** | — | 4.72 | 1036 | 0.0% | 0.87 | 40.0 | 2.14 | 4/4 |
| leak-free d=2 | 2 | 200 | 170–17–13 | 0.882 [0.84, 0.92] | 91% | 6% | **+350 [+295, +424]** | +12 [-66, +94] | 8.27 | 1843 | 0.0% | 0.89 | 38.2 | 5.34 | 7/7 |
| leak-free d=3 | 3 | 200 | 168–23–9 | 0.863 [0.82, 0.90] | 88% | 4% | **+319 [+263, +387]** | -19 [-94, +53] | 11.18 | 2688 | 0.0% | 0.89 | 38.0 | 6.06 | 7/7 |

Paired comparisons (Elo, same maps and seats, map bootstrap):
- leak-free d=1 minus raw (no search): +83 [+21, +153] (maps 100)
- leak-free d=2 minus raw (no search): +96 [+38, +164] (maps 100)
- leak-free d=3 minus raw (no search): +64 [+3, +135] (maps 100)
- leak-free d=2 minus leak-free d=1: +12 [-66, +94] (maps 100)
- leak-free d=3 minus leak-free d=1: -19 [-94, +53] (maps 100)
- leak-free d=3 minus leak-free d=2: -31 [-105, +41] (maps 100)

Score by map type (games):
| arm | archipelago | classic | continents | drylands | highlands | jungle | lakes | pangea | ridge | waterworld |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw (no search) | 0.92 (26) | 0.77 (26) | 0.79 (28) | 0.69 (8) | 0.82 (14) | 0.89 (14) | 0.75 (20) | 0.70 (28) | 0.92 (24) | 0.88 (12) |
| leak-free d=1 | 0.94 (26) | 0.90 (26) | 0.88 (28) | 0.88 (8) | 0.75 (14) | 0.86 (14) | 0.82 (20) | 0.89 (28) | 0.90 (24) | 0.83 (12) |
| leak-free d=2 | 0.90 (26) | 0.87 (26) | 0.86 (28) | 0.88 (8) | 0.96 (14) | 0.86 (14) | 0.85 (20) | 0.84 (28) | 0.96 (24) | 0.88 (12) |
| leak-free d=3 | 0.96 (26) | 0.87 (26) | 0.86 (28) | 0.88 (8) | 0.82 (14) | 0.89 (14) | 0.88 (20) | 0.79 (28) | 0.85 (24) | 0.83 (12) |

Critic calibration of the greedy lane (first plan of each searched turn) and chosen-minus-greedy score:
- leak-free d=1: mean greedy value +0.304, chosen +0.341 (gap +0.037), outcome +0.651, corr(greedy, outcome) 0.620, corr(chosen, outcome) 0.624, mse 0.356 (n=8169)
- leak-free d=2: mean greedy value +0.327, chosen +0.388 (gap +0.061), outcome +0.657, corr(greedy, outcome) 0.639, corr(chosen, outcome) 0.647, mse 0.347 (n=7806)
- leak-free d=3: mean greedy value +0.346, chosen +0.426 (gap +0.080), outcome +0.620, corr(greedy, outcome) 0.622, corr(chosen, outcome) 0.635, mse 0.383 (n=7764)

maps in every arm: 100; fallbacks per game: leak-free d=1 0.00, leak-free d=2 0.00, leak-free d=3 0.00
