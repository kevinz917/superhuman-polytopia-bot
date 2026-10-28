# Test-time search on the OMNI bot: depth and breadth are tapped out (2026-10-21)

Question: `c5-omni-s1-final` with leak-free one-turn lookahead is the strongest bot (+338 Elo over raw c3,
[c5-omni-long-2026-10-21.md](c5-omni-long-2026-10-21.md)). Does spending more inference compute on the search
buy more, either by looking further ahead (depth) or by imagining more opponent replies per candidate (breadth)?

Two sweeps ran in parallel on Modal, both on the same 100 maps x 2 seats (seeds 388000-388099, `proposed-v1` 14x14,
opponent raw greedy c3, lanes 16, T=1, rollout T=1, determinized so nothing leaks through fog):

- [search-depth-2026-10-21.md](search-depth-2026-10-21.md): d=2 and d=3 with one sampled rollout, tag `c5-search-depth-v1`.
- [search-breadth-2026-10-21.md](search-breadth-2026-10-21.md): 64 and 256 averaged replies at d=1 (the robust planner,
  sequential halving), tag `c5-search-breadth-v1`. The 256 arm was cancelled at the user's request with 32 of 200 games
  landed; three finished chunks were lost in the cancel.

Both are paired against the October 21 `c5-search-v1` rows (raw, and d=1 with one reply) on the same seeds.

## Answer

**No. Nothing beyond d=1 with a single sampled reply is worth its cost. The configuration to keep is unchanged:**
`TurnSearcher(lanes=16, depth=1, determinize=True)` with the candidate's own value head, about 4.7 s per searched
turn on a 32-core container. That is what the play-omni Studio port serves.

| Configuration | W-L-cap | Elo vs raw c3 [95% CI] | Paired Elo vs d=1 single reply [95% CI] | s / searched turn | Compute vs d=1 | Games |
|---|---|---|---|---:|---:|---|
| raw, no search | 151-26-23 | +255 [+212, +305] | | 0.08 | | 200 |
| d=1, 1 reply (ship config) | 167-17-16 | +338 [+285, +407] | reference | 4.7 | 1x | 200 |
| d=2, 1 reply | 170-17-13 | +350 [+295, +424] | +12 [-66, +94] | 8.3 | 1.75x | 200 |
| d=3, 1 reply | 168-23-9 | +319 [+263, +387] | -19 [-94, +53] | 11.2 | 2.4x | 200 |
| d=1, 64 replies | 173-19-8 | +354 [+295, +436] | +16 [-65, +100] | 12.8 | 2.7x | 200 |
| d=1, 256 replies (partial) | 30-2-0 | not usable | | 43.1 | 9x | 32 |

The 256-reply row is 16 maps with the long cap games missing, so it is biased upward and should not be quoted.

Cost: depth about $12, breadth $30.94 (the straggler containers ran 1.5-2 h on cap games before the cancel), total
about $43 of a $60 cap. Ledger rows are in `runs/c5/search/budget.json`.

## Why it did not work: what the per-game data shows

The analysis below uses the per-game rows and per-turn traces of the five full arms (`rows[*].trace` holds, for
each searched turn, the greedy lane's critic score, the chosen lane's score and the mean lane score).

**1. One-reply search already fixed what was fixable; the remaining losses are coin flips.**

| Arm | Losses | Raw losses it rescued (of 26) | New losses it created |
|---|---:|---:|---:|
| d=1 | 17 | 16 | 7 |
| d=2 | 17 | 16 | 7 |
| d=3 | 23 | 17 | 14 |
| 64 replies | 19 | 16 | 9 |

Every configuration rescues the same 16 raw losses: the tactical blunders that one opponent reply exposes. Deeper or
wider search catches the same ones and no more. The remaining losses are not a fixed set of hard maps: the union of
lost games across the four search arms is 44, but only 4 games are lost by all four. Each configuration loses a
different handful. 73 of 200 games change outcome between arms and the changes net to zero.

**2. Depth makes the chosen turn look better by exactly the added noise.**

| Arm | Median chosen-minus-greedy critic gap | 90th percentile |
|---|---:|---:|
| d=1, 1 reply | 0.026 | 0.088 |
| d=2 | 0.044 | 0.141 |
| d=3 | 0.059 | 0.185 |
| d=1, 64 replies | 0.018 | 0.064 |
| d=1, 256 replies | 0.020 | 0.065 |

Depth doubles the apparent advantage of the chosen lane while winning no more games: the optimizer's curse, since
noisier lane scores make the maximum look better without being better. Breadth shrinks the gap by a third (averaging
removes the noise) and also wins no more games. Together these say the single-reply d=1 picks were already about as
good as de-noised picks; the noise was rarely changing which turn got chosen.

**3. The candidates barely differ.** The median gap between the chosen and the greedy lane is 0.026 on a -1..+1
value scale, and between the chosen and the average lane 0.034. Sixteen turns sampled at T=1 from a strong policy are
near-variants of one plan. Search chooses among near-equals, and the few turns where a candidate is genuinely better
are the tactical ones d=1 already finds.

**4. c3 is a ceiling.** Every search arm scores 0.875-0.885 against c3. A true +30 Elo is about 5 games in 200, inside
the +-60 Elo interval. Games are decided early: in the d=1 arm every game where the critic read above +0.5 by turn 6
was won. The only measurement with headroom is the mirror (OMNI-search vs OMNI-search, +81 on October 21), which this
sweep did not run.

**Mechanism.** The critic was trained on millions of game outcomes, so its score after one reply is already an average
over the rest of the game. A longer single rollout replaces that learned average with one noisy sample (the
searcher's own later turns are sampled at T=1 too, not greedy), and the determinized world drifts further from the
real one with every imagined turn. More replies de-noise the one-reply estimate, but the estimate was good enough
already. This is the same lesson as AlphaGo's value network beating rollouts, and the same as the earlier distillation
results here: on this bot, further Elo comes from training, not from inference-time search.

## Recommendation

- Keep the demo and ship configuration as it is: d=1, 16 lanes, one sampled reply, determinized, own value head.
- Do not build a turn-level tree search; the depth curve gives it no support.
- Spend future budget on training (longer omni runs, the critic), not on search compute.
- If a search question comes up again, measure it in the mirror, not against c3.

## Files

- Reports: `docs/search-depth-2026-10-21.md`, `docs/search-breadth-2026-10-21.md`.
- Specs, summaries and analysis scripts: `docs/search-depth/`, `docs/search-breadth/`, `scripts/analyze_search_depth.py`,
  `scripts/analyze_search_breadth.py`.
- Per-game rows (not committed, shared `runs/` folder): `runs/c5/search/c5-search-depth-v1/`, `runs/c5/search/c5-search-breadth-v1/`.
