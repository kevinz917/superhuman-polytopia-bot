# Search breadth: do more imagined opponent replies keep paying? (2026-10-21)

**Question.** Leak-free one-turn lookahead (`TurnSearcher(lanes=16, depth=1, determinize=True)`, one sampled
opponent reply per candidate) adds +83 to +100 Elo on top of `c5-omni-s1-final`, the strongest 14×14 bot
(docs/c5-omni-long-2026-10-21.md). On 2026-10-19, on the older u3142 bot, averaging each candidate over 64 sampled
replies with sequential halving (the robust planner, `replies=64`) was worth +72 Elo over a single reply
(docs/test-time-search-2026-10-19.md). Nobody had measured replies on the OMNI bot, or whether the breadth gain keeps
scaling past 64. This sweep answers both on the same 100 proposed-v1 maps × 2 seats (seeds 388000–388099) as the
October 21 search study, so every comparison is paired.

**Arms** (`docs/search-breadth/spec.json`, `experiments/modal_test_time_search.py::sweep`, tag
`c5-search-breadth-v1`): candidate `c5-omni-s1-final`, opponent raw `c3`, lanes 16, depth 1, temperature 1.0,
rollout temperature 1.0, determinized (leak-free), root `sequence`, own value head as scorer:

- `omni-d1det-r64-vs-c3`: `replies` 64 (sequential halving 16 → 8 → 4 → 2 survivors, 16 reply lanes per round);
- `omni-d1det-r256-vs-c3`: `replies` 256 (same schedule, 64 reply lanes per round).

Paired baselines from `c5-search-v1` (same seeds, re-fetched with per-game rows into `runs/c5/search/c5-search-v1/`):
`win-k1-vs-c3` (raw, no search) and `win-d1det-ownval-vs-c3` (one sampled reply per candidate).

**Run.** Smoke (`smoke_arms`, 2 games each, app `ap-AO7bsfiBCbYLcNSSref9xj`, $0.57): both arms legal, 7.85 s (r64) and
16.4 s (r256) per searched turn with 2 processes on an idle 32-core container. Sweep launched 14:06 PDT (app
`ap-HpevTcFKjnBnjOrjHcKbl4`, call `fc-01M5G4K7F8G095B7TM0BSD02N4`), 14 × 32-core containers, `per_container` 32, the
lockstep planner with one pinned process per core (the `parallel=True` planner is ~10× faster per turn but needs 16
cores per game, so it costs more per game; it is the serving configuration, not the evaluation one). **Cancelled at
16:03 PDT on the user's stop-all request** with three r256 containers still playing their cap games. Ledger row
`c5-search-breadth-v1` in `runs/c5/search/budget.json` (reserved $24, recorded $31; `modal billing report` showed
$30.37 for the sweep app at 16:04 with the last hour still open, plus $0.57 smoke).

**What landed.** r64: all 7 chunks, 200 games. r256: **1 of 7 chunks, 32 games (maps 388000–388015, both seats)**.
Four r256 containers had finished by the cancel, but `orchestrate` collects chunks in launch order and chunk 2's
container was one of the stragglers, so chunks 3–5 (96 finished games) were never written to the volume and are lost
with the cancel. The missing r256 games are the slow ones: under full-container load r256 ran at 43 s per searched
turn (11,100 lane-steps per turn, 3.5× r64's 3,142), a 101-turn cap game needs 6,000+ s of search, and the chunks
that did not land are exactly the ones holding those games. **The landed r256 score is therefore biased upward**: it
contains no cap game at all, while r64 drew 8 caps in 200 games and the single-reply arm 16.

## Results (c5-omni-s1-final vs raw c3, proposed-v1 14×14, seeds 388000+, both seats)

| Arm | W–L–cap | Score [95% CI] | Of decisive | Elo over raw c3 [95% CI] | s / searched turn | Lane-steps / turn | Override | $ (containers) | Games |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| raw s1-final (no search) | 151–26–23 | 0.812 [0.77, 0.85] | 85% | **+255 [+212, +305]** | 0.08 | 13 | — | 0.70 | 200 |
| + leak-free d=1, 1 reply | 167–17–16 | 0.875 [0.84, 0.91] | 91% | **+338 [+285, +407]** | 4.7 | 1,036 | 0.87 | 2.14 | 200 |
| + leak-free d=1, 64 replies (robust) | 173–19–8 | 0.885 [0.84, 0.93] | 90% | **+354 [+295, +436]** | 12.8 | 3,142 | 0.85 | 6.05 | 200 |
| + leak-free d=1, 256 replies (robust), partial | 30–2–0 | 0.938 [0.84, 1.00] | 94% | +470 [+293, +1200] | 43.1 | 11,095 | 0.87 | 2.32 of ≈ 22 | **32 (16 maps)** |

The first two rows are the October 21 `c5-search-v1` baselines (same seeds). Dollar figures are container-seconds at
the ledger rate ($1.83 / container-hour); the billed total for the sweep app was $30.37 because the straggler
containers ran 1.5–2 h each before the cancel.

Paired on the same maps and seats (Elo, 4,000-resample map bootstrap, `scripts/analyze_search_breadth.py`):

| Comparison | Maps | Paired Elo gain [95% CI] |
|---|---:|---|
| 1 reply over raw | 100 | +83 [+21, +153] |
| 64 replies over raw | 100 | **+100 [+34, +173]** |
| **64 replies over 1 reply** | 100 | **+16 [−65, +100]** |
| 256 replies over raw (landed maps only) | 16 | +197 [+56, +927] |
| 256 replies over 1 reply (landed maps only) | 16 | +177 [+45, +885] |
| **256 replies over 64 replies (landed maps only)** | 16 | **+76 [+0, +729]** |
| 64 replies over 1 reply, on the same 16 maps | 16 | +101 [−71, +806] |

On the 16 landed maps the other arms score: raw 25–4–3 (0.828), 1 reply 25–3–4 (0.844), 64 replies 29–3–0 (0.906),
256 replies 30–2–0 (0.938). These 16 maps are an easy subset (every arm scores above its 100-map average there), and
the 16-map CIs are an order of magnitude wider than the 100-map ones.

Per map type, r64 vs the single reply: r64 is better on highlands (1.00 vs 0.75), jungle (0.96 vs 0.86), lakes (0.93
vs 0.82) and waterworld (0.92 vs 0.83) and worse on classic (0.83 vs 0.90), pangea (0.77 vs 0.89) and drylands
(0.81 vs 0.88), i.e. the two configurations trade maps rather than one dominating. r64 halves the cap rate (8 vs 16
cap games) at the same loss count (19 vs 17), so its gain is mostly converting caps into wins.

Critic calibration (greedy lane's score vs outcome, first plan per turn): 1 reply corr 0.62, r64 0.56, r256 0.37
(n = 1,142); averaging more replies raises the mean greedy value (+0.30 → +0.35 → +0.38) without tightening it.

## Verdict

Breadth does not keep paying on the OMNI bot the way it did on u3142. Going from one sampled reply to 64 (the robust
planner) is worth **+16 Elo [−65, +100]** on 200 paired games, against +72 on the older bot in October 19's study:
the single-reply leak-free lookahead already captures almost all of the search gain (+83 of the +100 that r64
reaches over raw), at 2.7× less compute per turn (4.7 vs 12.8 s on one core). The 256-reply arm is **inconclusive**:
its 32 landed games are 30–2–0 and +76 Elo over r64 on the same 16 maps, but the CI touches zero, the sample is the
easiest sixth of the map set, and it is missing precisely the cap games that cost r64 and the single-reply arm their
points, so the true r256 gain is smaller than the landed number. At 43 s per searched turn on one core (≈ 3 s/turn
with the 16-core parallel planner, by the 9–10× speedup measured on October 19), 256 replies are 3.4× r64's cost for a
gain that 200 games could not have resolved either. **For the demo bot, run `c5-omni-s1-final` with
`TurnSearcher(lanes=16, depth=1, determinize=True, rollout_temperature=1.0)` and one sampled reply (the current
configuration): it gives +83 Elo at under 1 s/turn on 16 cores, and `replies=64` buys at most a few dozen Elo for 2.7×
the compute (mostly fewer cap games).** If a breadth follow-up is wanted, the open question is whether r256 beats
single-reply, not whether it beats r64: rerun r256 alone on the full 100 maps with `per_container` 16 and the
`parallel=True` planner so cap games cannot stall a chunk, and write chunks as they finish rather than in launch order.

## Files

- `docs/search-breadth/spec.json`: the two arms; `docs/search-breadth/omni-d1det-r64-vs-c3.json`,
  `omni-d1det-r256-vs-c3.json` (per-game rows, 200 and 32 games), `smoke-arms.json`, `analysis.md`.
- `scripts/analyze_search_breadth.py <breadth dir> <baseline dir>`: the tables above (paired-map bootstrap).
- `runs/c5/search/c5-search-breadth-v1/` (shared, git-excluded): the same JSON, `launches.json`, `smoke.log`,
  `launch.log`; `runs/c5/search/c5-search-v1/` now holds the c5-search-v1 baselines with per-game rows (the copies
  under `docs/c5-omni-long/search/` carry only summaries).
