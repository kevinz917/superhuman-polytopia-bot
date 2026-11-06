# Public-site games: how humans beat the bot (Oct 29–31)

A visitor asked whether we log games played on the site, and whether the bot's losses come from too little scale or
from a broken assumption in the code. This note answers that from the logs.

## Data

- Source: Modal volume `polytopia-site-games` (staging), `games/<date>/<id>.json`, schema `polytopia-site-game-v1`
  (seed, seat, action ids, result, bot turn timings, device). The browser posts the whole game at every turn boundary.
- 65 games from 49 browsers (26 on phones), Oct 29 22:38 to Oct 31 06:07 (Mac local time, UTC+8). The bot is the scratch-cont-2 champion
  with its fog value head (sha 90618f5a), ruleset `bardur-1v1-v2`, leak-free d=1 turn search with 16 lanes, halved
  down to 4 on slow devices. The human is always seat 1 and moves first.
- Every game was rebuilt from its seed and actions in the site's own `engine_wasm.wasm`. All 65 replayed without an
  error. Scripts: session scratchpad `replay.ts` (per-turn score, cities, units, techs, actions) and `combat.ts`
  (attacker and victim of every attack, and the bot's value at the start of each of its turns).
- Coverage caveat: browsers that block beacons (Brave, GPC) may not log. 65 games is a lower bound.

## Results

| | games |
|---|---|
| logged | 65 |
| reached turn 8 | 31 |
| reached turn 15 | 5 |
| finished | 3: human 2, bot 1 |
| unfinished, bot ahead on score at the last logged turn | 25 of 28 that reached turn 8 |

Most people stop between turns 8 and 14, usually behind by 1–4 cities. Only three humans ever got ahead and stayed
ahead:

| game | device | turns | result | human line |
|---|---|---|---|---|
| 80dc2c26 | US desktop, 24 cores | 26 | **human eliminated the bot**, 9 cities to 0 | Riding t4, Roads t8, Free Spirit t14, Chivalry t15; 12 riders, 10 knights |
| 1452535d | DE phone | 21 | **human eliminated the bot**, 9 cities to 0 | Riding t10, Roads t11, Free Spirit t16, Chivalry t17; 8 riders, 4 knights |
| 6ea71178 | IL phone | 32 (quit) | human ahead 5 cities to 4, +1,875 score | Riding t1, Free Spirit t18, Chivalry t20; 12 riders, 27 knights, 11 catapults |

All three played the same plan, and it is the one the bot never plays.

## What happened

**1. The bot plays a single opening.** In all 31 games that reached turn 8 the bot researched Forestry, then Archery
(turns 4–7), then Climbing, then sometimes Organization, Mathematics or Strategy. It never researched Riding before
turn 21. Its army is mostly archers: 22 of 38 units trained in 80dc2c26, 18 of 33 in 1452535d. That matches the earlier
unit-mix study ([champion-unit-mix]: Archery on turns 3–5 in every game, Chivalry only in long games).

**2. Cavalry beats massed archers.** Archers have 1 defence. Knights keep attacking after each kill (persist).

- Across the three human-win games, human knights attacked 74 times and killed 58 units. In the other 28 games,
  knights got 4 kills.
- In 80dc2c26, human turn 16: two knights killed 8 archers, a catapult and a warrior in one turn. The bot went from
  16 units to 8 and never recovered.
- In 1452535d, riders did the work: the human took 3 bot cities on turns 13–14 and then picked archers off in the open
  (4 rider kills of archers on turn 17 alone).

**3. The bot didn't see it coming.** Its own value estimate at the start of each of its turns:

| game | before the collapse | after |
|---|---|---|
| 80dc2c26 | t15: **+0.21** (one knight visible to it) | t16: −0.68, t18: −0.96 |
| 1452535d | t10: **+0.66**, t16: +0.18 | t17: −0.49, t18: −0.90 |

A swing of about 0.9 in one human turn means the value head has not learned what a knight threat is worth. The knight
was not hidden by fog: the bot could see one of the two.

**4. Search can't make up for it.** The turn search is d=1 and leak-free. It tries 16 candidate turns, samples one
opponent reply for each **from the bot's own policy**, and scores the result with the same value head. Its model of the
opponent is the bot itself, and the bot doesn't charge with knights. So the search inherits the blind spot instead of
covering it.

## Scale or a broken assumption?

There's no bug in the usual sense. Replays are bit-exact, the rules engine matched JAX on 9,695 of 9,695 decisions, and
the bot beats most visitors. The losses come from **self-play monoculture**: training assumes the opponent plays like
the bot. That assumption breaks in three places at once:

- **Policy.** The champion's lineage almost never met cavalry rushes, so it never learned the counters: defenders and
  walls on the approach, archers kept on forest or mountain tiles or behind a screen, or an early Riding of its own.
- **Value head.** It prices positions the way they resolve in self-play, so it reads a mass of archers facing knights
  as fine.
- **Search.** The opponent model is the bot's own policy, so the search can't imagine a line the policy never plays.

Scale helps, but only partly:

- The later hero-1v1 run (about 7 B samples on the real 2.17.3 rules) **did** find knights on its own. It researches
  Chivalry in 71% of seats, and those seats win 65% (strategy probe, Oct 29). More self-play compute found the strategy.
- That bot is not on the site. The site still serves the Oct 23 champion, because hero has no fog value head and the
  site's WASM engine is from before the rules alignment.
- A bot that plays knights itself has not necessarily learned to defend against a human who commits to them earlier and
  harder. AlphaStar is the standard lesson here: self-play converges, and you need exploiter agents and a league so the
  main agent faces the strategies it would never pick.

## Smaller factors

- **Seat.** The human always plays seat 1. In bot-vs-bot 1v1, seat 1 wins 59%. That's a real handicap for the bot,
  though it doesn't explain why every win used the same plan.
- **Phone search.** In 2 of the 3 human wins, the bot searched with fewer lanes for much of the game: 6ea71178 had 4 or
  8 lanes on 29 of 32 turns, and 1452535d on 8 of 21. Over all games, only 73 of 504 bot turns had fewer lanes. The
  desktop win (80dc2c26) had full search on every turn, so fewer lanes aren't needed to lose this way. They may make it
  easier.
- **Cavalry isn't automatic.** Five humans reached Chivalry and three won. 3c405cb9 got knights on turn 6 with a weak
  economy and was 2 cities to 6 by turn 12. Riders and roads first, then knights when the bot's archers are spread out
  around turns 15–20, is the plan that worked.

## Possible fixes, cheapest first

1. **Measure it.** Build a scripted cavalry-rush opponent, or replay the three human openings as fixed prefixes, and
   play the champion and hero-1v1 against it on Modal. That shows whether hero already holds.
2. **Ship hero to the site.** Train it a fog value head and rebase the WASM engine on the aligned rules. That gives
   +131 Elo raw over the champion, and hero plays knights itself.
3. **Exploiter in the training pool.** Add a frozen opponent biased toward Riding and Chivalry (style-biased frozen
   opponents were already in the Oct 21 diversity plan), and train the main agent against it plus self-play.
4. **Search opponent model.** Sample opponent replies from a mix of policies (self plus the exploiter) so search can
   see lines the main policy doesn't play.
5. **Site.** Let players choose their seat, or alternate seats, and log `lanes` per turn so we know when phone search
   was weaker.

## Case study: game 80dc2c26 (the player's screen recording)

A player sent an 8-minute screen recording of this game (`MOV_9653.mov`, a replay played back from turn 0 to turn 26).
To match it, I read seven (turn, score) pairs off the HUD (4/1,600, 8/2,655, 11/3,060, 12/3,515, 13/4,030, 15/5,250,
16/5,240) and checked them against the human's score after every logged action in every game. All seven occur in
80dc2c26 and in no other game. The video's last frame (turn 26, score 9,185, 164 stars) is the logged final state.
Game details: seed 1810897034, US desktop with 24 cores, full 16-lane search on all 26 bot turns, 14×14 land map
(102 forest, 51 field, 33 mountain, 10 shallow water), 7 villages between the two capitals.

### How it went

| phase | turns | what happened |
|---|---|---|
| Expansion | 0–10 | The human took Riding on turn 4 and trained 6 riders by turn 8. Riders took villages on turns 6, 8 and 9, giving the human 5 cities to the bot's 3 by turn 10. The bot spent its early stars clearing 9 forests and on Archery. |
| Archer war | 10–15 | The bot massed archers (11 by turn 12) and won the skirmishes, with 18 kills to the human's 13 over turns 0–15. Its value estimate rose to +0.37. The human traded cheap units and built up: 11 roads on turns 8–13, lumber huts, Gate of Power, Park of Fortune, and 2 Giants from city upgrades. |
| The strike | 14–16 | The human banked stars (19, then 23, then 36), then spent 49 in two turns: Free Spirit (14) on turn 14, Chivalry (19) and two knights on turn 15. One knight stood in the human's capital, where the bot could see it. The other stood in the captured village at [2,4], hidden by fog. |
| Collapse | 16 | The visible knight moved 3 tiles and chain-killed 3 archers. The **hidden** knight moved 5 tiles along the human's roads to [7,7] and chain-killed 7 units in one sweep (4 archers, a catapult, a warrior and one more archer) through the whole bot front. The bot went from 16 units to 6. |
| Mop-up | 17–26 | Knights took the bot's cities on turns 17, 19 and 20, then the capital on turn 22. The bot offered peace 4 times. A bot Giant raided and took [2,4] on turn 21, and riders retook it on turn 26, eliminating the bot. Over turns 16–26 the human made 26 kills and the bot 5. |

### What the player did well

- **Fast expansion with riders.** Riding on turn 4 is cheap, and riders reach villages before warriors. The human was a
  city ahead from turn 7 and two ahead from turn 9, and never fell behind on score.
- **Didn't fight on the bot's terms.** During the archer war the human lost more units but spent on economy: roads,
  lumber huts, monuments and Giants. They had the higher income from turn 12 on (16 vs 12, then 21 vs 18).
- **Saved up for a power spike.** They banked stars for two turns, then bought Free Spirit, Chivalry and four knights
  in two turns. The bot saw no knights until the turn before they struck.
- **Picked the right counter.** Knights have high attack and persist (after a kill they move in and attack again).
  The bot's army was a connected line of 1-defence archers, which is the ideal target for a chain. 10 of the 11 kills
  on turn 16 were knight chain kills.
- **Used fog and roads.** The knight that did the most damage started out of the bot's sight and covered 5 tiles in one
  move on roads built three to eight turns earlier.
- **Followed through.** They kept training knights (10 in total), took a city almost every turn, and didn't give the bot
  time to rebuild.

### Why the bot lost

1. **One plan, no counter-plan.** Forestry → Archery → Climbing, then archers and catapults. It didn't research
   Organization (Defender) or Strategy until turns 17 and 18, after the line had already broken.
2. **A chain-killable formation.** Eleven archers and two catapults stood in a contiguous front, on adjacent field,
   forest and mountain tiles, with no defender units screening them. Every kill opened the next target.
3. **Greed on the critical turn.** On turn 15, with a knight in view and Chivalry researched, it spent 15 of its 21
   stars on five lumber huts. It also spent 8 attacks taking a Giant from 40 to 7 HP without killing it.
4. **Fog plus a one-sample search.** At the start of turn 15 the search valued its plans at **+0.52 to +0.57**, higher
   than the raw +0.21, so it expected to be winning after the human's reply. That reply is sampled once per candidate
   from a determinized world in which the bot has to guess hidden units. The knight that killed 7 was hidden.
5. **Not a tactics gap.** From the start of the human's turn 16, I let the bot's own policy play the human's side 30
   times. It killed 9 bot units in the deterministic rollout and 9–13 in most sampled ones, so it knows the knight
   chain. What it lacks is defensive anticipation: it doesn't picture that attack coming at it, especially from a
   unit it can't see.

**Verdict for this game:** not a scale problem in the narrow sense, since the bot can execute the same tactic itself.
What failed is that a self-play agent rarely faces knight rushes, so its value head, its formations and its belief
about hidden enemy units are all tuned for archer-vs-archer games. A cavalry-heavy exploiter in the training pool would
target exactly these gaps. So would a search that samples several opponent replies and several determinizations of
hidden units, instead of one of each.

## Cavalry probe: is cavalry an exploit when the bot itself plays it?

Script: `experiments/cavalry_probe.py` on the `claude/strategy-probe` worktree (uncommitted). Raw rows:
`runs/cav-1031.json`. Setup: engine-rs (real 2.17.3 rules, not the site's older ruleset), classic 14×14 maps, seeds
520000–520255, each seed played twice with seats swapped, deterministic raw policies with no search. About $2 of
Modal CPU.

The attacker is a defender's own network with its legal-move masks steered into the site winners' plan:

- No Archers, no Catapults, no Archery and no Mathematics.
- A city that can afford a Knight trains only a Knight or a Defender. A city that can afford a Rider trains no Warrior.
- The three arms differ only in when Riding, Free Spirit and Chivalry become the only researchable techs: from turn 0
  (`cav`), from turn 10 (`late`), or never (`free`).

| arm (512 games each) | attacker wins | attacker captures | defender captures | attacker knights/game | attacker Chivalry turn (median) |
|---|---|---|---|---|---|
| champion-cav vs champion | 7.2% | 5.2 | 10.4 | 6.2 | 9 |
| champion-late vs champion | 14.6% | 6.2 | 10.1 | 9.1 | 13 |
| champion-free vs champion | 13.7% | 6.1 | 10.2 | 6.8 | 12 |
| hero-cav vs hero | 2.9% | 4.6 | 10.1 | 4.0 | 9 |
| hero-late vs hero | 5.5% | 5.5 | 9.9 | 5.1 | 13 |
| hero-free vs hero | 5.1% | 5.4 | 9.9 | 4.1 | 13 |
| reference: champion vs hero (unsteered) | 33.8% | 6.9 | 8.8 | 2.0 | 23 |

The reference arm reproduces the known gap: hero wins 66% here, against 68% in its final ladder.

**Reading.** Played by the networks themselves, the cavalry plan is not a strong strategy. It is much worse than their
natural archer-based play. The steered side loses the expansion race (about 5–6 captures to the defender's 10) and
loses 85–97% of games. That holds even with no tech forcing, so most of the cost comes from giving up archers and
paying 8 stars for knights. Hero is the less exploitable of the two: its steered copy wins 3–6%, against 7–15% for the
champion's. When the steered side does field 8 or more knights it wins 30–40% (champion) or 14–26% (hero). That's
partly selection, since games that run long enough to build that many knights are games the attacker didn't lose early.

**What this means for the site losses.** The knight chain is not a general hole the bot falls into whenever someone
builds cavalry. The humans who won executed a whole plan: rider expansion, economy, then a saved-up, fog-covered knight
strike when the archer line was exposed. That is the kind of coordinated, opponent-specific plan a masked network can't
improvise. The probe says three things:

- The site losses aren't a sign that cavalry is broken or that the bot is fundamentally weak.
- A hand-steered copy of the bot is too crude to stand in for a skilled human, so masks can't measure the exploit. A
  cavalry exploiter has to be **trained**: PPO from the champion against a frozen champion, optionally with a
  Riding/Chivalry reward bias, until it beats the champion. Whatever plan it finds is the bot's real weakness, and
  training the main agent against it is the fix.
- Hero holds up better against both steered cavalry and the champion, which is one more reason to ship it with a fog
  head.

Limits: real rules rather than the site's ruleset, raw policies (the site adds search), deterministic play, classic
maps only.
