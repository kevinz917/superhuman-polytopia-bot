# Introducing Polytopia Bench

I gave 12 models 40 minutes each to write a Rust bot for _The Battle of Polytopia_. Here's how they did:

> [Figure 1: interactive leaderboard]
> Each dot is one independently authored bot. Scores are the mean ± sample standard deviation over five bots. Filter by opponent or map, then select a model to inspect its five submissions.

Code + results: [Polytopia Bench on GitHub ↗](https://github.com/kevinz917/polytopia-bot/tree/50280698b5d663b2a1ff59c133b577e5fa70dc9c/docs/skirmish/results/continuous-full-20261031)

## Why

After [training the self-play bot](/blog/polytopia-bot) I naturally asked: how strong can bots be by just writing code one-shot? Given that I already had all the RL environments built, I thought this was a pretty light follow up question. I was also curious what strategies, accounting methods, and long term thinking frontier models would apply to games. Polytopia rewards long-horizon economic planning with fog of war. Should be fun!

## Task design

Quite straight forward: each model gets one continuous session with a 40-minute budget of active time and a sandbox containing the rule book, the bot protocol, a starter `main.rs`, nine public maps and a small CLI to write code and practice the result it produces. It can compile, play practice matches against a public greedy baseline, and step through the trace of any practice game. It decides what to do with the time and in what order. There is no internet access.

> [Figure 2: authoring loop diagram]
> The authoring loop. Reading, editing, compiling, practice play and trace inspection all draw down the same 40 active minutes. API latency counts; sandbox setup and the final grading do not.

The bot itself is a Rust program that reads a JSON observation on stdin and writes back the id of one legal action. It sees only what its tribe can see, keeps whatever memory it wants within a game, and starts fresh for the next one. There is no rollback and no picking the best of several attempts, so the model has to manage its own risk near the deadline.

I ran the same protocol for all 12 models, five independent attempts each. Each attempt starts from the same starter code with a fresh conversation, so the five submissions are five different programs.

## Verifier design

Grading happens in a separate container and plays against a fixed schedule:

1. **Six opponents.** Two earlier frozen code bots, a scripted siege player, two blends that hand 25% or 50% of turns to the RL champion, and the full champion. _(panel pinned by digest before any model coded)_
2. **Three map families × ten held-out layouts.** Drylands, lakes and continents, none of them in the public practice set. _(same layouts for every bot)_
3. **Both seats.** Every layout is played from each starting position, since the seat matters on many boards. _(6 × 3 × 10 × 2 = 360 games per bot)_
4. **Replay audit.** Every game's action chain is replayed in the engine and must reproduce the recorded outcome before it counts. _(21,600 games, all audited)_

Guests run under hard limits: two seconds per decision and 256 MiB of memory. A bot that crashes or exceeds them loses. Games that reach 200 rounds or 40,000 actions are draws. Infrastructure failures are recorded as errors and regraded, so they never show up as a model's loss. A win is one point and a draw half a point. The displayed score is the percentage of available points, averaged over the five submitted bots.

> [Figure 3: model × opponent heatmap]
> Points per opponent, averaged over five bots. Opponents are ordered from the weakest frozen bot to the full champion. Hover a cell for the win/draw/loss split.

In the heatmap shown, you can see that every model beats the weak bot most of the time (the lowest mean is 42 points) and nearly nobody beats the champion (41 wins in 3,600 games across all 12 models). The spread comes from the middle of the panel, where the intermediate bot, the siege script and the two champion blends separate programs that merely work from programs that play well.

## How they played

- **The first eight turns decide most games.** The bad bots froze themselves at the start with one wrong constant. One scored research above recruiting, so every star went to tech and the starting warrior fought alone. One scored harvesting above research and never left the basic techs. One capped its army at the capital's level, which meant zero recruits. One penalized leaving the capital so hard that the warrior stayed home and blocked the city from training anything. All of this was visible in the bot's own practice traces by turn five. The good bots recruit a Rider on turn one and keep every spending option in a narrow score band so the mix shifts with the game.
- **Good bots fight carefully.** The top bots charge every move, recruit and attack with a danger term: how hard the enemy could hit that tile next turn. They pull wounded units back and add a bonus when a second unit can finish a kill. Most of the field has no danger term on movement at all, so units walk into range one at a time and die one at a time.
- **Good bots know where they are going.** Every top bot runs a real shortest-path search over land and water to villages, ruins and enemy cities, then hands each unit one target so they don't all chase the same village. Fewer than a quarter of the bottom bots have any pathfinding. Their units measure straight-line distance through mountains and lakes, or sit still because nothing scores above ending the turn.

Nobody beat the champion. The top two won 17 of 600 games against it, and across all 12 models it won every one of its 1,200 drylands games. The points that separate a 75 from a 45 come from the middle of the panel, and most of that is water. Against the naval blend, Astra's bots lost 88 of 300 games. They build a port on any watery map after turn six and park a Defender next to it. Opus lost 241, because it only builds a port once no land target is left. Getting onto the water early is the biggest single lever below the top.

The armies look different too. Astra and Sol both lean on Riders (39% and 36% of recruits) but fill out differently: Sol adds Swordsmen and Archers, Astra spends 10% on Catapults. Opus went cavalry-heavy, with Knights at 32%. Luna built almost nothing but Riders and Warriors (96% together). A simple unit preference can be fine; the opening and the pathfinding are what made the difference.

> [Figure 4: army mix by model]
> Share of recruited units per model over its 1,800 final games, ordered by panel score. Longer games recruit more, and I'd need matched edits to the same bot to say which choices caused a better result.

One more thing from the sessions. The good runs tested on every map and both seats, in parallel, built a small tool to print why the bot picked one action over another, and threw away edits that made things worse. The bad runs tested on one map, patched symptoms with bigger and bigger edits, and shipped a last-minute change they never ran. More practice games did not help on their own; Opus played the most and finished mid-table.

## Strategy highlights

Here are interesting systems bots came up with:

### The systems

- **Combat as an exchange rate.** The best bots price every attack like a trade. Astra credits the attack with the share of the target's value it destroys, plus a bonus for a kill, and debits the share of its own unit's value it loses to retaliation, plus a bigger penalty if the attacker dies. Sol's engine-pasting bot goes further: it copies the game state, plays the attack, and compares the total unit value on the board before and after. Weaker bots score a kill or a hit as a flat bonus and never price what they give up.
- **A punishment for standing in the wrong place.** Astra keeps a threat map of how hard the enemy can hit each tile next turn. A move, a recruit or an attack that ends on a dangerous tile pays for it. The strongest enemy hit counts fully and the next ones count for 85%, 65% and 30%, since the enemy rarely lands all of them. Sol adds a nice detail: an enemy that would die to retaliation if it attacked only counts as 65% of a threat.
- **A job market for units.** Astra runs an auction. It lists jobs (villages, ruins, enemy cities, patches of fog, enemy units, threatened home cities) and hands them out one unit at a time. Every unit already heading for a village makes that village almost three times less attractive to the next one, so the army spreads out instead of piling onto one target. Gemini pairs units with villages one to one. Most of the field has no allocation at all, and five units chase the same ruin.
- **A savings account.** Astra picks one technology it wants, adds up the cost, and holds that many stars in reserve, discounting anything else that would dip into it. Sol had the opposite problem: in self-play its bot banked 44 stars by turn 20 and stopped growing, so it added a wealth factor that makes spending cheaper the richer it gets. Kimi's good bot does a tiny version: it refuses to spend its first five stars on anything except Riding.
- **Beliefs about what it can't see.** Fog hides most of the enemy, so some bots infer it. Sol works out which technologies the enemy must have from the units it has seen, guesses what is on an enemy boat from the boat's health, and detects hidden City Walls when a hit does less damage than it predicted. Gemini and Opus guess the enemy capital by mirroring their own across the map.
- **Memory against dithering.** A greedy scorer happily walks a unit back and forth forever. The top bots remember where each unit has been, with a visit count that fades by about 15% a turn, and charge a small penalty for going back. Luna's 45-line bot does the same thing by reading the last few moves from the observation.
- **Guardrails.** Sol forces the turn to end after 140 actions and Opus after 300, so a scoring bug can't loop forever. Kimi's good bot scores destroying its own buildings at −10,000 after catching itself in a build-and-destroy loop.
- **Score tiers, and how they break.** Some bots sort actions into fixed tiers: rewards beat captures, captures beat attacks, attacks beat building, and so on. That is easy to write and easy to break. Qwen put building at 30,000, recruiting at 26,000, harvesting at 20,000 and research at 15,000, so research could never win and the bot never left the starting techs. Opus kept every economic choice (recruit, harvest, build, research) inside one narrow band between about 250 and 520, so small situational bonuses could tip the balance. The narrow band ended mid-table. The rigid tiers ended near the bottom.
- **Punishments that backfire.** The worst bugs came from penalties that were too big. Luna's worst bot charged −230 for leaving the capital in the first eight turns against an end-turn score of 0, so the starting Warrior never moved. MiniMax charged −2,000 for any move that didn't step strictly closer to the unit's goal, and its goal was a tile another unit already stood on. A penalty larger than everything else on the table turns into "do nothing".

### Model by model

Some moments from the sessions and the code, from the clever to the self-sabotaging.

- **GPT-6 Astra: the best general.** Its top bot hands every unit one job (a village, a ruin, an enemy city, a patch of fog) through a little auction, and penalizes sending too many units to the same place. On water maps it builds a port early and parks a Defender next to it. While writing, it made the bot print its top 15 candidate moves, spotted that it never built ports on lake maps, fixed it, and deleted the debug code before submitting.
- **GPT-6.1 Sol: pasted in the whole game.** One of Sol's bots copied the entire 160 KB game engine into its own source so it could simulate every attack exactly. It even inferred hidden City Walls: if a hit on an enemy city did less damage than predicted, it assumed a wall was there. That bot scored 76.2, a hair below Sol's hand-written version at 76.9. Exact simulation did not buy much.
- **Gemini 4 Argon: the scholar.** It read the game's source code for 17 minutes before writing a line, then wrote 70 KB in two huge passes. Some of it is clever: it buys research right before capturing a village, because tech gets more expensive with every city you own, and it guesses the enemy capital by mirroring its own. It ran out of time before it could submit, so the clock captured whatever was on disk.
- **Claude Opus 5.5: the most practice, the earliest exit.** Its best session found a bug the smart way: it saved a game state from a lost practice match, fed it into the bot with debug prints on, and saw every move scored −1 because its pathfinding thought the map was blocked. Another session played 113 practice games, won 36 in a row against the easy practice bot, and submitted with nearly 19 minutes left. It had tuned itself against an opponent that never pushed back.
- **Kimi K3.1 Pro: caught itself in a loop.** The good bot saves its first five stars for Riding instead of spending them on turn one. Its session noticed the bot building a port and then destroying it, over and over, and banned destroying buildings. The bad session boosted Chivalry research 90 seconds before the deadline, lost the one test game, and submitted it anyway.
- **GLM 5.3: one rule, two outcomes.** The good bot's units stood still in practice because the only target in range was a ruin on top of a mountain they could not climb. GLM found it by replaying one turn and fixed the pathfinding. The bad bot capped its army at the total level of its cities, which at the start of the game means zero recruits.
- **GPT-6 Luna: smallest bot, biggest swing.** Its best bot is 45 lines long and scored 41, ahead of bots twenty times its size. Its worst bot penalized leaving the capital so hard for the first eight turns that the starting Warrior never moved. With the Warrior standing on the city, the city could not train anything either. It lost nearly every game with zero cities left.
- **Qwen 3.8 Max: never left the Stone Age.** Its bot always preferred growing a city to researching, so it never got past the starting techs. A mix-up in its combat code used a unit's health where it meant the unit's cost, which made it refuse any fair trade.
- **DeepSeek V4 Pro: all tech, no army.** Research scored roughly twice as high as recruiting, so every star went to technology. The starting Warrior marched off alone in a straight line. The session saw this in its own traces twice and reordered the tech list instead of fixing the score.
- **MiniMax M3.1 Flash: the longest bot and the weakest.** 1,049 lines. Most of its units were assigned to guard the capital, but the capital tile was already occupied, so every move scored as forbidden and they stood still. It lost to the weakest opponent on the panel 50 times out of 60.

## Evaluation details and limits

These are 1v1 games on 14×14 boards. Five bots per model is a small sample, and the SD describes the spread across those five bots on this fixed panel. API latency counts toward the active coding budget, so a slow endpoint is a real handicap. Local runs used ARM machines and cloud runs x86, under the same declared game limits. These results describe this harness and budget.

The task is packaged for Harbor; the [result manifest](https://github.com/kevinz917/polytopia-bot/blob/50280698b5d663b2a1ff59c133b577e5fa70dc9c/docs/skirmish/results/continuous-full-20261031/results.json) records the protocol, sources, hashes and recovery history.

[Full results and frozen bot sources ↗](https://github.com/kevinz917/polytopia-bot/tree/50280698b5d663b2a1ff59c133b577e5fa70dc9c/docs/skirmish/results/continuous-full-20261031) · [Harbor task ↗](https://github.com/kevinz917/polytopia-bot/tree/codex/polytopia-skirmish/harbor)
