# How Plato plays (2026-11-09)

What strategy does [Plato](../weights/plato) use, and what changed since its $300 checkpoint (update 23,270)? Both
checkpoints played hero-1v1 and the champion on the **same 128 maps × both seats** (256 games per pairing), under
both rules. Each also played itself on 128 maps with peace legal. Decisions are sampled at temperature 1, as in
training, on engine-rs (the 2.17.3 rules, 14×14 classic maps). Every game records every decision and a state
summary at the start of every turn. That covers income, cities, units by type, techs, builds, city rewards,
attacks, captures and diplomatic actions. The $300 records come from an earlier run of the same tool on the same
seeds, so every map and seat is paired. Data: [benchmarks/2026-11-09-plato-strategy.json](benchmarks/2026-11-09-plato-strategy.json).

## Five findings

1. **The answer to the Knight rush is its own Knights, earlier.** Plato now researches Riding (round 13), Free Spirit
   (14) and Chivalry (**16**, in 91% of games), five rounds before hero-1v1's Chivalry (21). The $300 checkpoint got
   there at round 33–37 in fewer than half its games. Knights are 17–22% of Plato's trained units; at $300 they were
   2–5%.
   - Against hero-1v1, Plato trains 11.6 Knights by round 25 to hero-1v1's 4.3.
   - hero-1v1 trained 8.6 by round 25 against the $300 checkpoint.
   - In the 86 games where hero-1v1 still rushes (≥ 5 Knights by round 25), Plato scores **.744**. The $300
     checkpoint scored .053 in 150 such games.
2. **It wins fast, by elimination.** Median game length:
   - vs hero-1v1: **34 rounds** (was 45);
   - vs the champion: **28 rounds** (was 64).

   It takes 4.3–4.9 enemy cities per game (was 1.7 against hero-1v1). Every win against both opponents was an
   elimination. Its 9 losses to the champion with peace legal were all games that reached the round cap; the
   champion never eliminated it.
3. **The early economy is still the CNNs' weak spot for it, but it now catches up by round 10.**
   - Plato still earns less at round 3 (2.0 vs 3.0 stars per turn) and round 5 (4.9 vs 5.5).
   - By round 10 it out-earns hero-1v1 (12.2 vs 11.7) and the champion (12.8 vs 11.2). The $300 checkpoint trailed
     hero-1v1 there (11.1 vs 12.0).
   - By round 20 the gap is wide: 22.2 vs 19.3 (hero-1v1) and 23.5 vs 16.2 (the champion). By then its opponents
     are losing cities.
4. **The economic habits did not change: Explorer over Workshop, no early Lumber Huts.**
   - At city level 2 it still takes **Explorer** (4.8 per game) far more than **Workshop** (1.5–1.7), though Workshop
     doubled from 0.75–1.0.
   - It builds 0.4–0.5 Lumber Huts by round 10 (was 0.36), against about 3 for hero-1v1 and the champion.
   - What changed is the mid-game investment. Roads, Mines and Forges mostly disappeared: against the champion, 0.7
     Roads per game (was 29), 1.8 Mines (was 13) and 0.6 Forges (was 7.6). Mining is now researched in only 15–29% of games against
     these opponents (was 90–97%).
   - It spends on Knights and Temples instead, and ends games before a mining economy would pay back.
5. **Diplomacy shrank from a liability to a tool.** With peace legal, Plato makes 13.6 diplomatic actions per game
   against hero-1v1 (was 27.0) and 5.8 against the champion (was 35.9). The $300 checkpoint had never trained with
   peace and wasted its turns on it. In Plato-vs-Plato games with peace legal, 18.6 actions per side per game (was
   27) and only 7 of 256 games reached the cap.

   Under the no-peace rule its play is nearly identical:
   - the same openings, unit mix and income to two decimals through round 20;
   - scores .859 and .977 against .887 and .965 with peace legal.

   Peace is now a small part of how it wins.

## Opening and tech

| | Plato | $300 checkpoint | hero-1v1 | the champion |
|---|---|---|---|---|
| first two units | Warrior, Warrior (100%) | Warrior, Warrior (100%) | Warrior, Warrior (98%) | Warrior, Warrior (99%) |
| first four techs (most common) | Organization → Forestry → Archery → Climbing (51–56%) | same (47–50%) | Forestry → Archery → Climbing → Organization | Forestry → Archery → Climbing → Organization |
| Riding / Free Spirit / Chivalry (median round, share of games) | 13 / 14 / **16 (91–98%)** | 25 / 30 / 33–37 (43–46%) | 17 / 20 / 21 (80–89%) | — (Riding in 31%) |
| Mining / Smithery | 14–16 (15–29%) / 21 (≤ 21%) | 15 (90–97%) / 20 (88%) | — | 28 (27%) / — |
| first four builds (most common) | Park of Fortune → Eye of God → Gate of Power → Temple | Park of Fortune → Eye of God → Gate of Power → Mine | Lumber Hut × 4 (69%) | Lumber Hut × 4 (61%) |

## Income by round

Income is the summed star production of a side's cities per turn, averaged over games still running at that round.
Past round 20 the averages are survivor-biased, because most of Plato's games are over by round 30.

| Round | Plato vs hero-1v1 | $300 vs hero-1v1 | Plato vs the champion | $300 vs the champion |
|---|---|---|---|---|
| 3 | 2.0 / 3.0 | 2.0 / 3.0 | 2.0 / 3.0 | 2.0 / 3.0 |
| 5 | 4.9 / 5.5 | 4.5 / 5.6 | 4.8 / 5.5 | 4.3 / 5.5 |
| 10 | **12.2 / 11.7** | 11.1 / 12.0 | **12.8 / 11.2** | 11.3 / 11.8 |
| 15 | 18.3 / 18.7 | 17.1 / 19.6 | 18.7 / 16.5 | 17.9 / 18.1 |
| 20 | **22.2 / 19.3** | 21.3 / 22.4 | **23.5 / 16.2** | 23.3 / 20.7 |

Being behind on income at round 10 no longer decides the game. Plato wins 76% of its games against hero-1v1 when
behind at round 10 and 98% when ahead. The $300 checkpoint won 3% and 11%.

## Units and attacks

| | Plato vs hero-1v1 | $300 vs hero-1v1 | Plato vs the champion | $300 vs the champion |
|---|---|---|---|---|
| trained units, mix | Archer 41%, Knight 22%, Warrior 17%, Rider 10%, Swordsman 7% | Archer 36–42%, Swordsman 16–27%, Warrior 17–21%, Defender 8–12% | Archer 43%, Warrior 21%, Knight 17%, Rider 13% | (same as left) |
| army on the board at round 20 | Archer 58%, Knight 15%, Warrior 12%, Rider 9% | Archer 69%, Warrior 16%, Swordsman 6% | Archer 54%, Knight 16%, Warrior 14%, Rider 12% | Archer 69%, Warrior 18%, Swordsman 6% |
| first attack / first enemy city (median round) | 6 / 18 | 5 / 17 | 6 / 16 | 6 / 19.5 |
| enemy cities taken per game, Plato / opponent | **4.9 / 1.4** | 1.7 / 6.7 | **4.3 / 0.4** | 4.3 / 7.1 |
| median game length (rounds) | 34 | 45 | 28 | 64 |
| games over by round 30 | 35% | 27% | 63% | 5% |

The attack timing is the same as at $300: first attack at round 5–6, first city taken around round 16–18. The
difference is what the pressure turns into. With Knights in the army from round 16, the attacks keep taking cities,
and the opponent's income falls after round 15.

## Diplomacy and how games end

| Diplomatic actions per game | Plato | $300 checkpoint |
|---|---|---|
| vs hero-1v1, peace legal (Plato / opponent) | 13.6 / 14.1 | 27.0 / 20.5 |
| vs the champion, peace legal | 5.8 / 5.9 | 35.9 / 22.9 |
| vs itself, peace legal (per side) | 18.6 | 26.8–27.0 |
| any opponent, no peace | 0 | 0.3–0.6 |

With peace legal: 0 of 256 games against hero-1v1 reached the 200-round cap, 9 of 256 against the champion, and 7
of 256 against itself. The rest ended by elimination.

## Against the built-in AI

Against a replica of the built-in Crazy AI (re-implemented on engine-rs, not the retail game), Plato won 48–0 in 15.5
rounds on average (median 16; the $300 checkpoint: 48–0 in 16.8). Per game it averaged 19.5 units trained, 12.5
attacks and 2.9 captures.

## What did not change, and what to watch

- Plato still starts slower: below hero-1v1 and the champion on income at rounds 3–5, with few Lumber Huts and
  Explorer over Workshop. A bot that survives the round-16 Knight push could test that. Plato has not faced one.
- Its mid-game economy is now thin: few Roads, Mines and Forges. That works because games end by round 30–35. In
  a long game, for example against a strong defender, the $300 checkpoint's mining economy might have been the
  better plan.
- All of this is against two CNN opponents and itself on 14×14 classic maps. These are 256-game samples, so
  per-pairing rates carry about ±6 points (95%).
