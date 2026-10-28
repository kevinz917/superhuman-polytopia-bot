# Test-time search on top of the omniscient-critic checkpoint (2026-10-20)

Question: after one hour of training c3 with an omniscient critic produced `abl-omni-2-final` ("OMNI",
[ablation readout](ablation-v1/readout-2026-10-20.md): 54% of all games / 65% of decisive vs c3), is leak-free
test-time search ([c3 study](test-time-search-c3-2026-10-19.md): +104 Elo on c3) still the dominant gain, and how do
the two best bots compare?

## Answer

- **No — training is now the bigger lever, and search stacks on top of it.** One hour of omniscient-critic
  training is worth **+125 Elo [+92, +160]** over raw C3 on these boards (raw OMNI beats raw C3 112–43–45, 67% of all
  games, 72% of decisive), while leak-free d=1 search adds **+70 [+31, +111]** to C3 on the same boards (+104 in
  yesterday's study on other maps). The ranking has flipped: C3 + search is no longer better than the new raw bot
  (68–81–51 vs raw OMNI, −23 Elo [−56, +10]).
- **Search still helps OMNI, undiminished.** OMNI + leak-free d=1 beats raw C3 140–32–28, **+210 Elo [+166, +260]**;
  the search gain on OMNI is +85 [+28, +143] against raw C3 and **+139 [+102, +179]** in the mirror (vs raw OMNI).
  It is not smaller than on C3: mirror gain OMNI minus C3 = +69 [+14, +123]; against the common C3 opponent the
  difference is +14 [−52, +83]. The distillation students' loss of search gain does not transfer here — the trunk
  changed, but the value ranking of candidate turns survived.
- **The scorer does not matter.** OMNI's own stale value head (no gradient during the omni run) scores lanes as
  well as C3's calibrated fog-only network: +205 [+166, +249] vs +210, difference +5 [−45, +53]; both rank the
  greedy lane's value against the outcome equally (corr 0.607 vs 0.610). The stale head is biased low (mean +0.13
  vs outcome +0.42) but ranking is what search uses. No scorer swap is needed to serve OMNI + search.
- **Best bot today: OMNI + leak-free d=1** (≈ 3.4–3.9 s per searched turn on one Modal core, ≈ 1.2 s on an
  M-series core; the parallel searcher cuts latency further). OMNI + search minus C3 + search against the same
  opponent: **+139 [+85, +197]**.
- Implication for the next run: gains from training and from search add roughly linearly here (+125 + 85 ≈ +210),
  so a longer omniscient-critic run is the priority, and search remains a free ≈ +80–140 Elo on whatever it produces.
  Cost: $12.7 nominal of the $40 budget (`runs/omni-search-eval/budget.json`).

## Setup

- Checkpoints. C3 = `/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz`.
  OMNI = `/retained/checkpoints/distributed-final-v1/abl-omni-2-final/final.npz` (policy). OMNI's omniscient critic
  (`final-critic.npz`) sees both players' views, so it is never used by search here: scoring lanes with it would
  leak the hidden state.
- Search, every searched arm: leak-free d=1 — K=16 lanes, depth 1, candidate temperature 1.0, rollout temperature
  1.0, `determinize=True` (belief sampling, re-plan at the first revealed difference; the +104 Elo setting on c3).
  No reply averaging. Opponents always play the raw greedy policy.
- Boards: 100 fresh `proposed-v1` 14×14 development maps, seeds 386000–386099, both seats, every arm on the same
  boards and seats. Frozen engine, 100-turn cap (capped = ½ point). Harness
  `experiments/modal_test_time_search.py::sweep` → `rl/search_arena.py`, one pinned game process per core on
  Modal staging; spec `runs/omni-search-eval/spec.json`, results `runs/omni-search-eval/omni-v1/`.
- Elo. The opponent is a fixed raw bot, so the paired baseline is the candidate's own raw lineage against that
  opponent on the same maps: raw C3 vs raw C3 and raw OMNI vs raw OMNI are exactly 0.5 on every map (one
  deterministic policy in both seats); raw C3 vs raw OMNI is the raw-OMNI-vs-C3 arm with the sign flipped (the
  identical games, seats swapped). CIs: paired-map bootstrap, 4000 resamples
  (`scripts/analyze_omni_search.py`).

### Which value head scores OMNI's lanes

`rl/distributed.py` (loss, "The policy's own value head is unused; the omniscient critic regresses the
returns") confirms the design point: with `omniscient_critic=True` the policy network's value head receives no
gradient (`value_coef=0`), so OMNI's `final.npz` carries the value head of its c3 warm start, now attached to a
trunk that moved for 307 updates. `search.TurnSearcher` scores a lane with `network.trunk(params, …)[2]`, the
candidate's own head, so "OMNI + search" needs a scorer choice:

| scorer | what it is | arms |
|---|---|---|
| (a) own head | OMNI's stale c3 value head on OMNI's trunk | `omni-d1det-ownval-vs-c3` |
| (b) C3 network | `TurnSearcher(value_params=C3)`: OMNI's policy generates the K candidate turns and the rollouts; C3's full network (its trunk + value head, a calibrated fog-only critic) scores each lane at the horizon | `omni-d1det-c3val-vs-c3`, `omni-d1det-c3val-vs-omni` |

`value_params=` (new, `rl/search.py`) passes a (policy, scorer) pair through the planners; the scorer's trunk
runs only where a lane is scored (the d=1 horizon, END for d=0, or a lane out of budget), and the plain-parameter
path is byte-for-byte the previous code (lanes == 1 ignores the scorer). `tests/test_search.py` checks that a
searcher with `value_params=params` plays the default searcher's plans and that a different scorer plays legally
and changes scores; `test_search.py`, `test_parallel_search.py`, `test_belief.py` all pass on Modal (12 passed, `runs/omni-search-eval/tests.log`).
Smoke (`runs/omni-search-eval/smoke-arms.json`): 2 games per arm, all legal, 1.3–2.9 s per searched turn on one
Modal core.

## Results

All arms 200 games (100 maps × 2 seats), 0 worker errors, 0 determinization fallbacks. "% of decisive" = wins /
(wins + losses). Elo is paired by map against the raw lineage baseline described above.

### vs raw C3 (seeds 386000–386099, proposed-v1 mix)

| arm | scorer | W–L–cap | score [95% CI] | % of decisive | Elo over raw C3 [95% CI] | s / searched turn | plans / turn | override | mean turns | $ |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| raw OMNI (no search) | — | 112–43–45 | 0.672 [0.63, 0.71] | 72% | **+125 [+92, +160]** | 0.08 | 1.00 | 0.00 | 56.4 | 0.70 |
| C3 + leak-free d=1 | own head | 106–66–28 | 0.600 [0.55, 0.65] | 62% | **+70 [+31, +111]** | 3.25 | 1.89 | 0.87 | 54.8 | 2.16 |
| OMNI + leak-free d=1 | (b) C3 network | 140–32–28 | 0.770 [0.72, 0.82] | 81% | **+210 [+166, +260]** | 3.87 | 1.99 | 0.88 | 49.4 | 2.05 |
| OMNI + leak-free d=1 | (a) own stale head | 133–27–40 | 0.765 [0.72, 0.81] | 83% | **+205 [+166, +249]** | 3.43 | 1.97 | 0.87 | 52.5 | 1.98 |

### vs raw OMNI (same seeds)

| arm | scorer | W–L–cap | score [95% CI] | % of decisive | Elo over the raw lineage [95% CI] | Elo over 0.5 | s / searched turn | override | mean turns | $ |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|
| OMNI + leak-free d=1 (mirror) | (b) C3 network | 114–38–48 | 0.690 [0.64, 0.74] | 75% | **+139 [+102, +179]** (vs OMNI mirror = 0.5) | +139 | 4.68 | 0.88 | 58.7 | 2.24 |
| C3 + leak-free d=1 | own head | 68–81–51 | 0.468 [0.42, 0.52] | 46% | **+102 [+58, +149]** (vs raw C3 vs raw OMNI = 0.328) | **−23 [−56, +10]** | 3.85 | 0.88 | 61.0 | 2.25 |

### Paired comparisons (same maps and seats; Elo, bootstrap 95% CI)

| comparison | Elo |
|---|---|
| (3 − 1) search gain on OMNI, C3 scorer, vs raw C3 | **+85 [+28, +143]** |
| (4 − 1) search gain on OMNI, own stale head, vs raw C3 | +80 [+36, +128] |
| (3 − 4) C3 scorer minus own head | +5 [−45, +53] |
| (3 − 2) OMNI + search minus C3 + search, both vs raw C3 | **+139 [+85, +197]** |
| (5) search gain on OMNI in the mirror minus (2) search gain on C3 in the mirror | **+69 [+14, +123]** |
| (3 − 1) minus (2 − 0.5): search gain on OMNI minus search gain on C3, common opponent raw C3 | +14 [−52, +83] |
| (6) C3 + search vs raw OMNI, over raw C3 vs raw OMNI | +102 [+58, +149] |

Search's gain on OMNI measured against raw C3 (+85) is smaller than in the mirror (+139) for the usual reason:
against a weaker opponent the raw bot already wins 72% of decisive games, so the same improvement moves the score
less. Against its own lineage the gain is the largest we have measured for leak-free d=1 (c3 mirror: +104 on
seeds 366000+, +70 here).

### Score by map type (games in parentheses)

| arm (opponent) | archipelago | classic | continents | drylands | highlands | jungle | lakes | pangea | ridge | waterworld |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw OMNI (c3) | 0.69 (18) | 0.73 (24) | 0.75 (16) | 0.81 (8) | 0.67 (18) | 0.67 (30) | 0.63 (30) | 0.56 (24) | 0.68 (20) | 0.67 (12) |
| C3 + search (c3) | 0.69 (18) | 0.58 (24) | 0.50 (16) | 0.75 (8) | 0.64 (18) | 0.65 (30) | 0.50 (30) | 0.48 (24) | 0.75 (20) | 0.58 (12) |
| OMNI + search, C3 scorer (c3) | 0.86 (18) | 0.75 (24) | 0.91 (16) | 0.75 (8) | 0.75 (18) | 0.73 (30) | 0.75 (30) | 0.67 (24) | 0.82 (20) | 0.79 (12) |
| OMNI + search, own head (c3) | 0.89 (18) | 0.73 (24) | 0.84 (16) | 0.81 (8) | 0.67 (18) | 0.72 (30) | 0.77 (30) | 0.67 (24) | 0.82 (20) | 0.88 (12) |
| OMNI + search, C3 scorer (omni) | 0.83 (18) | 0.69 (24) | 0.69 (16) | 0.62 (8) | 0.50 (18) | 0.70 (30) | 0.75 (30) | 0.65 (24) | 0.68 (20) | 0.75 (12) |
| C3 + search (omni) | 0.56 (18) | 0.50 (24) | 0.50 (16) | 0.62 (8) | 0.58 (18) | 0.30 (30) | 0.40 (30) | 0.40 (24) | 0.53 (20) | 0.58 (12) |

OMNI + search is at or above 0.67 on every map type against raw C3 (the two leak-free arms agree within the
per-type noise of ±0.2); raw OMNI's weakest types are pangea and lakes, and search lifts both.

### Value scorer calibration (greedy lane's score at the first plan of each searched turn vs the game outcome)

| arm | mean value | mean outcome | corr | mse | n |
|---|---:|---:|---:|---:|---:|
| C3 + search vs C3 (C3 head) | +0.051 | +0.150 | 0.626 | 0.482 | 11,095 |
| OMNI + search vs C3, C3 scorer | +0.118 | +0.468 | 0.610 | 0.443 | 10,044 |
| OMNI + search vs C3, own stale head | +0.128 | +0.423 | 0.607 | 0.377 | 10,660 |
| OMNI + search vs OMNI, C3 scorer | +0.073 | +0.318 | 0.616 | 0.378 | 11,877 |

Both scorers under-estimate OMNI's winning positions by ~0.3 (they were calibrated on c3 self-play), and both rank
them equally well; the stale head's 122 incomplete lanes (out of ≈ 340k lane evaluations) vs 20 for the C3 scorer
is the only visible difference.

### Caveats

- Raw OMNI's margin here (67% of all games, 72% of decisive) is larger than the ablation ladder's (54% / 65% on 128
  other maps, seeds 371300+); the score CIs barely overlap ([0.63, 0.71] vs [0.50, 0.58]). Same deterministic
  protocol, different boards — map-set variance is the only explanation we have, and it says the ±4% CIs of these
  100-map sweeps understate between-set variance. The paired *gains* within this sweep are not affected.
- Opponents are raw greedy policies from our own lineages; no external anchor.
- One scorer design was not tried: OMNI's omniscient critic (it would need the opponent's view — a leak by
  construction).
- Compute: the sweep's 32-core containers exposed 48 CPUs, so `search_arena` ran 48 game processes each; s/turn
  (3.3–4.7 s) matches the earlier study's 3.7 s and smoke's 1.3–2.9 s on 2 processes, so timings are not inflated.
  Spend: $11.38 sweep + $1.17 smoke + tests, nominal $12.7 of $40 (`runs/omni-search-eval/budget.json`,
  launches in `launches.json`); billing report at the time of writing: smoke $1.02, tests $0.03, sweep not yet posted.
