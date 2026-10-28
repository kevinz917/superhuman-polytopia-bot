# Champion vs the game's Crazy AI (2026-10-24)

**Result: 74–0–0** (78–0–0 with the 4 recorded replay games). Our strongest bot at the time
(`scratch-cont-2-foghead` + the leak-free turn search) beat the built-in **Crazy** AI of Polytopia 2.17.3 in all 48
match games: 24 per seat, fresh Small Dryland 1v1 Bardur maps, the game's own rules. Every win came from capturing
the last city (Domination); none reached the turn cap. The raw policy with no search also won all 24 of its games,
and the two pilot games were wins too. Crazy is not a measuring stick for this bot. 48/48 puts the win rate at
≥ 92.6% (Wilson 95%), and more Crazy games cannot sharpen that into anything useful.

## Setup

- **Opponent:** the game's own AI at the Crazy handicap, playing on the game's own maps and rules (game version 123).
  In this version Crazy plays like Hard.
- **Game settings:** Domination, Small 14×14, Dryland, Bardur vs Bardur.
- **Champion:**
  - Seat 1 is the champion moving first, with the AI playing seat 2; seat 2 is the reverse.
  - **Search arm:** the public site's settings (`lanes=16, depth=1, replies=64, rollout_temperature=1,
    determinize=True`), on one 16-core container per game.
  - **Raw arm:** the greedy policy (`lanes=1`).
  - Each turn is planned in our engine from the champion's view of the game and executed one command at a time;
    any difference between the game's state and our engine's prediction triggers a re-plan.
- **Fog:** search sees exactly what it sees on the site. Lanes plan on worlds sampled from the champion's own view.
  Nothing about hidden enemy units or stars reaches the policy.
- **Turn cap:** 100 rounds, as in training. No game came near it.

## Results

| Arm | Games | W–L–cap | Win rate (Wilson 95%) | Seat 1 | Seat 2 | Mean win turn (min–max) |
|---|---|---|---|---|---|---|
| Champion + search | 48 | **48–0–0** | 100% (92.6–100%) | 24–0–0 | 24–0–0 | 20.7 (14–53) |
| Champion raw (no search) | 24 | **24–0–0** | 100% (86.2–100%) | 12–0–0 | 12–0–0 | 18.7 (9–25) |
| Pilot (search) | 2 | 2–0–0 | — | 1–0 | 1–0 | 40.5 (39–42) |

### Economy

Medians over the 48 search games, at the champion's turn:

| Turn | Champion cities / pop / units / stars / techs | Crazy AI cities / pop / units / stars / techs | Champion ahead / behind in cities |
|---|---|---|---|
| 5 | 2 / 5 / 5 / 13 / 4 | 2 / 7 / 5 / 1 / 5 | 17 / 8 |
| 10 | 5 / 22.5 / 14 / 25 / 6 | 3 / 9 / 7 / 1 / 7 | 37 / 2 |
| 15 | 8 / 43 / 26 / 34.5 / 9 | 3 / 11 / 4 / 1 / 9 | 46 / 0 |

- The AI keeps pace for about five turns: slightly more population and techs, because it spends to zero every
  turn.
- From turn 10 the champion has several times its population and twice its army, while banking 25–35 stars.
- The AI matches the champion in tech count, but its army collapses between turns 10 and 15.

### Unit and tech mix (search arm)

**Units:** Archer 46%, Cloak 26%, Warrior 15%, Catapult 5%, Defender 3%, Knight 3%, Rider 1%.
- The archer-first opening matches what we saw in our own engine.
- Cloaks are trained heavily here (Diplomacy researched in 47 of 48 games).
- Without search, the raw policy trains more catapults (17%) and fewer cloaks (18%).

**Techs** (games researched): Forestry, Archery and Climbing 48/48; Organization 47; Diplomacy 47; Shields 43;
Mining 27; Farming 25; Meditation 21; Fishing 15.

### Re-planning (search arm)

- 1,680 plan calls (35 per game, 3.25 s per call), 30,265 game commands executed, 639 re-plans (13 per game),
  19 greedy-policy fallbacks, 0 crashes.
- Most re-plans were expected randomness (explorer rewards, ruins outcomes) or population timing; the rest were
  rules gaps in our engine at the time, listed below and since closed by the
  [rules alignment](rules-alignment-2026-10-27.md).
- Wall time: 2.4 min per game at 20 games in parallel; the whole 48-game match took about 7 minutes.

## Failure modes and rules gaps found

1. **Slow close-out.**
   - Three games ran long: g107 to turn 53, g146 to 36 and g145 to 33.
   - In g107 the champion held 10 cities to the AI's 2 from about turn 20, but the AI's two cities held more
     population, and the game took 30 more turns to finish.
   - The first pilot game was the same: the AI was down to one city with no units on turn 20, but the champion
     needed until turn 42 to find and take it, banking 633 stars meanwhile.
   - Finishing a beaten opponent, especially a hidden last city, is the weakest part of its play. Unspent stars
     pile up instead of turning into scouts or attackers.
2. **Seat 2 starts behind.** The AI moving first leads in cities at turn 5 more often in seat-2 games. It never
   mattered for the result.
3. **Rules gaps between our engine and the game** (at the time; triggers for re-planning, not bot weaknesses):
   Grow Forest adding an animal, star rewards after kills, some attacks and moves the game rejects, destroys the
   game does not offer, super-unit placement next to an occupied city, Cloak exploration, and roads on unowned
   villages and unexplored tiles.

## What would measure the bot now

Crazy is the strongest setting the game has, and it loses every game, so further strength tests need a stronger
opponent:

- a human player;
- Crazy given a star handicap;
- one champion against two Crazy AIs;
- the game's AI on larger maps, though the champion was trained only on 14×14 1v1 Bardur.
