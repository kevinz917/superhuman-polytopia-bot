# Unit diversity investigation (2026-10-18)

Question: do our strongest RL bots play with only basic troops instead of using the
tech tree to field stronger units?

**Answer: yes.** Low176, the strongest fully qualified checkpoint and the parent of
every current learned-only candidate, recruits Warriors in about 99% of its summons. It
never recruited a Rider, Archer, Defender, Catapult, Knight, Swordsman or Mind Bender in
any audited game, even though those units were legal and affordable in most games. The
behavior is inherited from the scripted teacher used for the imitation bootstrap.

## Data

- Source: every public trace from `low176-cloud-qualification-v1` (288 games: 64 teacher,
  32 random, 64 vs incumbent96, 64 vs low144, 32 vs native96; plus 32 vs gamma48 and
  32 vs noentropy64 not included in this pass).
- Each trace records the acting player's public observation, legal-action list and
  chosen command for every decision. Summon = command kind 4 (parameter = unit id),
  research = kind 7 (parameter = tech id); ids map through `src/polytopia_bot/gpu/rules.json`.
- Analysis ran in Modal (`experiments/modal_trace_analysis.py` with
  `scripts/analyze_unit_composition.py`). Results:
  `runs/structured-ppo-v3/unit-composition-low176-v3-results.json` (per-game and aggregate).

## Findings

Low176 vs teacher and learned opponents (256 games):

| Unit | Games where it was summonable at least once | Share of low176 summons |
|---|---:|---:|
| Warrior | 256 / 256 | ~99% |
| Catapult | 238 | 0 |
| Rider | 227 | 0 |
| Archer | 223 | 0 |
| Defender | 108 | 0 |
| Scoutship | 106 | ~1% |
| Cloak | 70 | 0 |
| Bomber | 35 | 0 |
| Knight / Mind Bender / Swordsman | 7–9 each | 0 |

- **Research happens, but doesn't turn into units.** Low176 researches about 9–12 techs per
  game against learned or teacher opponents and owns 12–15 at the end, including
  Riding, Archery, Mathematics, Strategy and Diplomacy, which unlock the units above.
  The research appears to serve the economy and city growth: low176 ends with 5–8 cities.
- **Warrior volume is high.** About 41–53 summons per game against learned opponents
  (games last about 25–27 turns). Its only non-Warrior units are Giants, which come from
  city-upgrade rewards rather than recruiting, plus occasional Scoutships.
- **Opponents are the same.** The incumbent (champion512 update 96), low144, native96 and
  the scripted teacher also recruit 98–100% Warriors. The uniform-random opponent is the
  only one that recruits Riders, Archers, Defenders and Swordsmen, so the engine allows them.

## Why

`src/polytopia_bot/learning/teacher.py:119-122` scores a summon as

```
20 * demand + 20 * threat - 2 * unit_cost
```

The only unit-dependent term is the cost penalty, so the teacher always recruits the
cheapest legal unit, the Warrior (cost 2; Rider/Archer/Defender 3, Swordsman/Mind Bender 5,
Catapult/Knight 8, per `rules.json`). Every strong checkpoint descends from imitation of
this teacher (bootstrap-native96 → champion512 → low176/low256). Roughly 9M later PPO
decisions never moved off it, for three plausible reasons:
1. Warrior-only play already beats every opponent in the training mix.
2. A costlier unit is a delayed, uncertain payoff under a terminal-only reward.
3. The autoregressive summon head assigns very low probability to other unit ids,
   so they are rarely sampled. Entropy coefficient 0.005 is small.

The learned-only continuations (low256 league, per-turn discount) start from the
same Warrior-only parents. Their strength plateau (ladder screens: 55–61% then about 50%
against low176 update 16) is consistent with self-play refining one narrow strategy.

## Research directions (proposed, not run)

1. **Measure first, cheaply:** rerun the same analysis on the current winner's
   qualification traces, and on the summon head's probabilities over legal non-Warrior
   options, to confirm the probability collapse. Cost: cents.
2. **Diversity-aware bootstrap:** replace the teacher's recruit score with a tech-aware
   one (e.g., counter-units, value per cost, role coverage), re-run imitation, then
   learned-only PPO. Clean but restarts the lineage; the new teacher needs its own
   strength check.
3. **Exploration without labels:** raise entropy only on the summon unit-type head, or
   add a per-head entropy floor, so PPO actually samples Archers/Riders/Catapults.
   Keeps native win/loss reward; cheapest to try on the existing low256 league setup.
4. **League pressure:** add frozen rivals or exploiters that punish Warrior spam (for
   example, a rival trained or scripted to field Archers/Catapults/Defenders), so
   diversity is rewarded through wins instead of shaping.
5. **Auxiliary signals (critic-only):** following Duper's social value heads, add
   value-loss-only prediction heads for unit composition. They don't change the reward
   but may give the trunk useful features.

Recommended order: 1 → 3 (cheap, keeps the current lineage) → 4 → 2 if the policy stays locked in.
