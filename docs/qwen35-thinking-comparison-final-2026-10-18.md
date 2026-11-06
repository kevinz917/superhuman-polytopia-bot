# Thinking vs. non-thinking on 30 paired Polytopia games (final, stopped 2026-10-18)

Three policies played the same 30 games: 15 validation maps (seeds 100691–100705) × both seats, all against the scripted `balanced` opponent. Each arm used the same named-field prompt, legal-action grammar, sampling settings (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5), per-decision seeds and simulator horizon. Every inference, simulation and analysis ran on Modal.

| Arm | Model | Run |
|---|---|---|
| **Thinking on** | `Qwen/Qwen3.5-4B` @ `851bf6e8`, `enable_thinking=True`, no token cap | `qwen35-thinking-30paired-v1` (on arm) |
| **Thinking off** | Same weights, `enable_thinking=False` | `qwen35-thinking-30paired-v1` (off arm) |
| **Instruct** | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f` (non-thinking model, the repository's original base) | `qwen3-instruct-2507-30paired-v2` ([details](qwen3-instruct-baseline-2026-10-18.md)) |

**The user stopped the thinking-on arm at 18:47 PDT.** At that point 19 of its 30 games had finished, and 11 were still running after about 9 hours. Those 11 are reported as *unfinished*: not wins, losses or draws.

## Bottom line

- **Outcomes:** thinking on won **2** games, lost 16, had 1 operational failure and left 11 unfinished. Thinking off and Instruct each lost **30/30**.
- **Win difference:** thinking on minus either non-thinking arm, clustered by map, is **+0.067** (2 wins over 15 maps). The bootstrap 95% interval is [0, 0.167], counting unfinished games as non-wins. Excluding unresolved games (13 maps remain), it is +0.115 [0, 0.308].
- **What we can and can't conclude:**
  - These are the only wins observed in any arm. But the evidence is thin: one sampling sequence, development maps, and 11 unresolved games.
  - The unresolved games include 4 where thinking was clearly ahead: 10 cities against 1, 8 against 3, 6 against 3, and 5 against 2. So the final win count could have been higher.
- **Where thinking clearly wins:** play quality short of winning. It expands about **3×** as fast, fields **3–10×** as many units, and outlasts thinking off in 23 of 30 games and Instruct in 27 of 30.
- **What it costs:** about **74.8 million thinking tokens** and **about 17.9 H100-hours**. The two non-thinking arms used about 31k and 23k output tokens, and **0.34** and **0.18** H100-hours.

## Outcomes

| | Thinking on | Thinking off | Instruct |
|---|---:|---:|---:|
| Native wins | **2** (rounds 23 and 24) | 0 | 0 |
| Native losses | 16 (rounds 15–42) | 30 (22 eliminations, 8 resignations) | 30 (all eliminations) |
| Operational failures | 1 (exact-repetition abort, round 10) | 0 | 0 |
| Unfinished at stop | 11 (rounds 22–38) | 0 | 0 |
| Median last round | 29 | 22.5 | 22 |
| Model moves | 10,029 | 3,795 | 2,803 |
| Model moves per turn | 11.3 | 5.8 | 4.5 |

**The two wins:**
- `seed-100697-seat-1`: elimination in round 23. Peak of 8 cities, 2 enemy cities taken, 8 kills. Thinking off lost this game in round 31.
- `seed-100702-seat-1`: elimination in round 24, from 9 cities against 1. Thinking off lost this game in round 28.

**State of the 11 unfinished games** (cities and units, model / opponent):

| Game | Round | Cities | Units |
|---|---:|---:|---:|
| `seed-100699-seat-1` | 24 | 10 / 1 | 30 / 4 |
| `seed-100692-seat-1` | 24 | 8 / 3 | 25 / 10 |
| `seed-100691-seat-2` | 22 | 6 / 3 | 23 / 11 |
| `seed-100704-seat-1` | 28 | 6 / 4 | 15 / 10 |
| `seed-100694-seat-2` | 38 | 5 / 2 | 24 / 9 |
| `seed-100705-seat-1` | 29 | 5 / 3 | 14 / 11 |
| `seed-100701-seat-2` | 26 | 5 / 6 | 15 / 13 |
| `seed-100693-seat-2` | 29 | 4 / 6 | 11 / 12 |
| `seed-100701-seat-1` | 29 | 4 / 6 | 11 / 14 |
| `seed-100693-seat-1` | 30 | 2 / 8 | 6 / 19 |
| `seed-100700-seat-1` | 35 | 1 / 8 | 5 / 25 |

## Same game, same turn (games where all three arms were still alive)

| Turn | Games | Our cities (on / off / Instruct) | Opponent cities | Our units | Our city levels (summed) |
|---:|---:|---:|---:|---:|---:|
| 6 | 30 | 1.53 / 1.17 / 1.00 | 1.73 / 1.53 / 1.60 | 4.3 / 3.5 / 3.1 | 3.2 / 2.6 / 2.3 |
| 9 | 30 | 2.27 / 1.27 / 0.87 | 2.23 / 2.23 / 2.27 | 6.7 / 3.9 / 2.6 | 5.3 / 3.0 / 2.2 |
| 12 | 24 | 3.17 / 1.42 / 0.96 | 3.00 / 2.92 / 3.00 | 9.3 / 4.3 / 2.4 | 7.7 / 3.5 / 2.5 |
| 15 | 19 | 4.05 / 1.53 / 1.00 | 3.58 / 4.05 / 4.00 | 10.8 / 4.2 / 1.4 | 9.8 / 3.8 / 2.5 |
| 18 | 17 | 4.65 / 1.47 / 1.06 | 4.06 / 4.88 / 5.41 | 13.7 / 4.1 / 1.4 | 12.4 / 3.9 / 2.7 |

- Peak cities per game averaged 4.7 for thinking on, 1.5 for thinking off and 1.0 for Instruct. Instruct never got past its capital in 29 of 30 games.
- The opponent grows at about the same rate in all three arms. The difference comes from what our side does.

## Strategy comparison

| | Thinking on | Thinking off | Instruct |
|---|---:|---:|---:|
| Stars spent per turn | **10.2** | 5.0 | 4.2 |
| Stars from city rewards / captures | 374 / 150 | 32 / 45 | 25 / 0 |
| Captures (villages + cities) | 198 | 33 | **1** |
| Enemy cities taken / own cities lost | **27** / 86 | 0 / 36 | 0 / 31 |
| Attacks · kills · units lost | 959 · 202 · 315 | 176 · 33 · 82 | 62 · 10 · 50 |
| Units recruited | 925 (857 Warriors, 36 Rammerships) | 286 (230 Warriors, 50 Riders) | 214 (206 Warriors) |
| **Own units disbanded for stars** | 29 (3% of recruits) | **115 (40%)** | **149 (70%)** |
| Healing actions | **255** | 19 | 70 |
| Builds | 614 roads, 95 lumber huts, 45 ports, 11 Park of Fortune, 10 Gate of Power | 252 roads, 10 ports | 239 roads, 16 ports |
| First research | Riding 17, Climbing 7, Organization 4 | Riding 25 | Riding 26 |
| Turn ends with ≥5 stars unspent | 82 of 866 (9%) | 123 of 641 (19%) | 14 of 615 (2%) |
| Turn ends while research/build/recruit was available | 16 / 70 / 20 | **94 / 101 / 62** | 0 / 0 / 0 |
| Picks the first-listed option in its action family (chance rate) | 31% (13%) | 40% (16%) | **48% (19%)** |

### Thinking on: an expansion economy that rarely finishes games
1. **It expands and grows.** It captures villages early, harvests so cities level up, collects city rewards (374 stars) and reinvests nearly all its income. Its star income is about twice either non-thinking arm's.
2. **It builds an army and fights.** It recruited 925 units, healed damaged units 255 times and disbanded almost none. It attacked 959 times and took 27 enemy cities. In the late game it also built Rammerships and ports.
3. **It doesn't close games out.** In the dominating games it kept capturing villages and fighting units near the enemy capital instead of taking the last city. `seed-100699-seat-1` sat at 10 cities against 1 for about 7 rounds without winning.
4. **It still loses many games late.** The opponent keeps expanding and adds Giants. Thinking lost 86 cities overall, mostly in games where it had peaked at 2–4 cities.

### Thinking off: passive, and it dismantles itself
1. It stays on one city: peak 1.5 cities, 33 captures in 30 full games.
2. It spends its stars on research (42%) and roads (252 builds), which do little for a one-city empire.
3. **It disbands its own army for cash:** 115 units scrapped against 286 recruited, after taking FreeSpirit (the tech that allows disbanding) in 25 games.
4. It ends turns with useful options still available (94 times while it could research, 101 while it could build) and resigned 8 games.

### Instruct (Qwen3-4B-Instruct-2507): even more passive
1. **It essentially never expands:** 1 capture in 30 games, average peak of 1.0 city.
2. **It disbands most of what it builds:** 149 of 214 recruits. Its army shrinks from 3.1 units at turn 6 to 1.4 by turn 15.
3. It rarely attacks: 62 attacks and 10 kills in 30 games.
4. It shows the strongest bias toward the first-listed option: 48%, against a chance rate of 19%.
5. It does end turns only once nothing useful is left. That's because it has almost nothing to do, not because it plays efficiently.

## The reasoning behind the moves (thinking on, all 10,029 responses)

- **Volume:** a median of about 7,000–8,000 thinking tokens per move, and every response closed its thinking normally. The only exception was the one repetition abort, a collapse into `,None` while it was reciting a rules table.
- **Self-correction:** 99.8% of responses contain "Wait", averaging **25 per response**. Most of it is re-checking action IDs, costs, legality and the output format (97.5% of responses discuss the answer format).
- **What it reasons about** (keyword heuristic, not a quality score): enemies or threats in 99% of responses, the economy in 98%, expansion in 99.8%, defense in 98%. The enemy capital or taking a city comes up in only 54%. "Next turn" or "later" phrasing appears often, but the reasoning almost never commits to a plan spanning several turns.
- **Where the tokens go:**
  - 53% of all thinking tokens were spent on plain unit moves.
  - 436 decisions where the only legal options were end turn or resign used **1.47 million tokens**, about 3,400 each, to decide not to resign.
- **What the reasoning gets right:** local tactics. It checks threat and safety fields ("`visible_enemy_in_range` is 0 for all tiles... So capturing is safe"), weighs star costs and prefers resolving threats before expanding.
- **Typical failures:**
  - It invents or doubts rules ("Tier 2 might require Tier 1 unlocked?").
  - It mixes up its own unit types; in the round-20 sample of the first win it calls a Warrior a Rider.
  - It breaks ties arbitrarily ("Let's choose `a133`").
  - It gets confused about the turn structure ("whether I've already acted twice").
- **Overall:** each move is a careful, local, one-step analysis. Thinking finds the good immediate action much more often than the non-thinking arms do. That is enough to expand and fight, but not to plan the kill.

The non-thinking arms produce no reasoning text: 8-token answers, all `stop`. Their strategy can only be read from behavior. The behavior is consistent with picking a plausible, often first-listed action each step with no notion of value: roads with nowhere to go, disbanding units for cash, and resigning.

## Cost

| | Thinking on | Thinking off | Instruct |
|---|---:|---:|---:|
| Output tokens | 74.8M (median 7–8k per move, max 20.8k) | 31,285 | 23,220 |
| Time per move | 44 s early, about 75 s late | 1.3 s | 1.8 s |
| Wall time | about 9 h, stopped unfinished | 471 s | 598 s |
| GPU time | about 64,400 H100-seconds (17.9 h) | 1,230 s | 662 s |

The thinking-on GPU total is approximate. It counts 2 workers × 14,400 s for the original segment and for continuation 1, plus about 2 × 3,400 s for continuation 2. Timed-out workers leave no exit receipt. No dollar figures are given, because no billing data was read.

**Why the thinking arm is slow:** each game waits for its own answer, so each H100 decodes only about 9–15 streams at about 105 tokens per second each. That's 1,100–1,500 tokens per second per GPU, well below the 3,000–4,700 tokens per second measured at high concurrency. More GPUs wouldn't help much. Speculative decoding (MTP), FP8 weights, or skipping the model on forced moves would cut wall time in a future run.

## Run integrity

- **4-hour limit:** the original thinking workers hit Modal's 4-hour limit at 13:47. The unfinished games were resumed from engine checkpoints in two continuation segments: [`async_resume.py`](../src/polytopia_bot/learning/async_resume.py), [`continue_thinking_games.py`](../scripts/continue_thinking_games.py) and [`modal_thinking_continuation.py`](../experiments/modal_thinking_continuation.py).
- **Same seeds and runtime:** decision numbers, and so per-decision seeds, continued unchanged. Each resumed worker's runtime receipt (model, tokenizer, chat template, packages, serving settings) was identical to the original's.
- **Exactness tests on Modal:** a single-resume test reproduced uninterrupted games row for row ([`test_async_resume.py`](../tests/test_async_resume.py)). A two-interruption, three-segment test reproduced them through the unchanged merge and summary ([`test_continuation_chain.py`](../tests/test_continuation_chain.py)).
- **Reconciliation:** at each cutoff every checkpoint was reconciled against its trace and timing log. No completed decision was dropped. Requests still in flight at a cutoff were lost and regenerated from the same seed.
- **Not finalized:** because the run was stopped, the standard finalizer (which needs every game finished) was not run. This report comes from [`analyze_thinking_comparison.py`](../scripts/analyze_thinking_comparison.py), a read-only Modal pass that joins each game's segments. Evidence: `runs/qwen35-thinking-30paired-v1/evidence-final-2026-10-18.json`.

## Caveats

- One sampling sequence. Positions diverge after the first different action, so per-move comparisons are not on paired positions.
- These are development validation maps, and prompt repairs used positions from them. This is not untouched-test generalization.
- The Instruct arm used the Qwen3.5 sampling recipe, not its model card's settings (temperature 0.7, top-p 0.8).
- The keyword statistics on reasoning are heuristics that show what the model talks about, not how well it reasons.
- Wins are tiny counts (2 of 30). The win interval reflects only map-level resampling of this run.
