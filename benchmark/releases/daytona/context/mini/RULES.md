Active protocol: /reference/PROTOCOL.md supersedes embedded historical defaults.

# Complete Polytopia Skirmish rule book

This is the implementation-defined rule book for polytopia-2.17.3, Bardur 1v1,
14x14, drylands/lakes/continents. It contains the bot protocol, current rules
alignment explanation, every rule-table entry, full Rust implementation, map
variants, and public observation serializer. The legal action catalog is always
authoritative for what is available now. The engine source and tables resolve
mechanical edge cases; this benchmark does not substitute older research rules.

## How to read this book

Start with the protocol and alignment's gameplay sections, then consult the tables
for numeric IDs/stats. Rust masks define legality; core dispatch defines actions;
military/pathfinding cover damage, retaliation and movement; economy/structures/
capture cover population, income, construction and city rewards; discovery and
infiltration describe exploration, ruins and hidden units; diplomacy describes
relations and peace. All these sources appear below without truncation.

Bardur starts with Hunting, one capital, a Warrior and five stars. Expand by
exploring villages and capturing them, grow cities through resources and buildings,
research unlocks, recruit armies within city capacity, and defeat the opponent by
capturing their last city. End turn collects income and refreshes units. Fog limits
observations; do not assume unseen enemy resources, units, or technologies.

Feedback/final map seeds are controlled by the referee. Layout generation is an
approximation of map distributions, separate from transition rules. No checkpoint,
opponent policy or held-out map allocation appears in this book.


## Gameplay mechanics reference

### Actions, economy and cities

One observation is supplied per atomic action, not per whole turn. You may take many
actions before end turn. Mandatory city rewards block ordinary actions until selected.
Stars pay for technologies, units, resources and construction. Harvesting an animal,
fish or fruit costs two stars and adds one population when the required technology
is known. Farms/mines and population multipliers use the exact costs and prerequisites
in the tables and structure implementation below. Population progress to the next
level is current city level + 1. Level growth increases production and triggers a
reward. City capacity and occupied city tiles constrain recruitment.

City reward command.parameter codes: 1 City Wall, 2 Park, 3 Workshop, 4 Explorer,
5 Border Growth, 6 Giant, 7 Resources (+5 stars), 8 Population (+3).
At level 2 choose Workshop/Explorer; level 3 Wall/Resources; level 4 Border/Population;
subsequent levels Park/Giant. Consult the actual legal catalog for each pending choice.
Capture commands transfer villages/cities; see capture.rs for unit action flags,
territory transfer, home-unit capacity and last-city victory. Capturing and recruitment
are separate from moving; arriving at a city alone does not execute capture.

Research price is 4 + ownedCityCount * technologyTier. Philosophy multiplies by
0.66666 with ceiling. Only reachable, unknown and affordable technologies appear.
Researching before expansion can therefore cost less. Technologies expose unit,
building and ability unlocks, but some table entries belong to tribes not playable
in this Bardur track. Follow the successor graph from known Basic/Hunting and the
legal catalog; a listed unit is not necessarily recruitable in this game.

### Combat and unit state

Health affects combat strength. Let af=attackStat*attackerHP/attackerMaxHP and
df=defenseStat*defenderHP/defenderMaxHP*defenseBonus. Normal damage is rounded
(af/(af+df))*attackStat*4.5; retaliation, if the defender survives and can retaliate,
uses (df/(af+df))*defenseStat*4.5. Exact rounding, range, terrain/city defense,
Surprise, Convert, splash and advance rules are in kernels.rs/military.rs.
Veterancy adds five maximum health; promotion heals. Recovery restores four health
in friendly territory and two elsewhere; Heal Others is a separate action.
Dash permits attacking after a move; Escape permits moving after attacking; Persist
allows kill chains. Non-Dash units generally cannot move then attack. Newly recruited,
embarked, healed or exhausted units follow the explicit moved/attacked flags.
Mind Bender conversion is not ordinary zero-damage combat. Attack outcomes include
retaliation and possible occupation of the defeated unit's tile.

### Geography, fog and navy

Distance is Chebyshev on the square grid (diagonals count). Movement uses terrain,
roads, mountains, water, bridges, occupied tiles and visible zone of control.
Entering a forest ends movement unless traveling road-to-road; mountain entry and
water require their unlocks. Exact reachability comes from the supplied legal moves.
Forests/mountains and city fortification also affect combat defense. Exploration
reveals tiles, public units/resources and contacts; public memory persists while
current visibility and remembered facts remain separate. Never infer that an unknown
tile is empty. Ruins, explorers, giant pushes and infiltration have seeded outcomes.

Ports are built in eligible owned shallow water. Moving a land unit into a port
embarks it; upgrades require the appropriate tech, funds and friendly territory.
Naval types have their own movement/attack/skills; carried land-unit health and cost
are tracked. Disembarking restores the passenger. Continents typically requires
naval transport; lakes mixes land and water. Diplomacy and peace affect hostility,
movement, embassies and cooldowns as specified by diplomacy.rs.

The following readable tables mirror every entry, including unreachable special-tribe
entries for completeness. The full JSON and source appendices retain every field.


## Units

| id | name | cost | health | attack | defense | movement | range | skills |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | None | -1 | 1.0 | -1.0 | 0.0 | 1 | 1 | [] |
| 2 | Warrior | 2 | 10.0 | 2.0 | 2.0 | 1 | 1 | ['Dash', 'Fortify'] |
| 3 | Rider | 3 | 10.0 | 2.0 | 1.0 | 2 | 1 | ['Dash', 'Escape', 'Fortify'] |
| 4 | Knight | 8 | 10.0 | 3.5 | 1.0 | 3 | 1 | ['Dash', 'Fortify', 'Persist'] |
| 5 | Defender | 3 | 15.0 | 1.0 | 3.0 | 1 | 1 | ['Fortify'] |
| 8 | Catapult | 8 | 10.0 | 4.0 | 0.0 | 1 | 3 | ['Stiff'] |
| 9 | Archer | 3 | 10.0 | 2.0 | 1.0 | 1 | 2 | ['Dash', 'Fortify'] |
| 10 | MindBender | 5 | 10.0 | 0.0 | 1.0 | 1 | 1 | ['Convert', 'HealOthers', 'Stiff'] |
| 11 | Swordsman | 5 | 15.0 | 3.0 | 3.0 | 1 | 1 | ['Dash'] |
| 12 | Giant | 10 | 40.0 | 5.0 | 4.0 | 1 | 1 | ['Static'] |
| 15 | Polytaur | 2 | 15.0 | 3.0 | 1.0 | 1 | 1 | ['Dash', 'Fortify', 'Independent', 'Static'] |
| 17 | DragonEgg | 10 | 10.0 | -1.0 | 2.0 | 1 | 1 | ['Fortify', 'Grow', 'Static', 'Stiff'] |
| 18 | BabyDragon | 10 | 15.0 | 3.0 | 3.0 | 2 | 1 | ['Dash', 'Escape', 'Fly', 'Grow', 'Scout', 'Static'] |
| 19 | FireDragon | 10 | 20.0 | 4.0 | 3.0 | 3 | 2 | ['Dash', 'Fly', 'Scout', 'Splash', 'Static'] |
| 20 | Amphibian | 3 | 10.0 | 2.0 | 1.0 | 2 | 1 | ['Amphibious', 'Dash', 'Escape', 'Fortify'] |
| 21 | Tridention | 8 | 10.0 | 2.5 | 1.0 | 2 | 2 | ['Amphibious', 'Dash', 'Persist'] |
| 22 | Mooni | 5 | 10.0 | 0.0 | 1.0 | 1 | 1 | ['AutoFreeze', 'Skate', 'Static', 'Stiff'] |
| 23 | BattleSled | 5 | 15.0 | 3.0 | 2.0 | 2 | 1 | ['Dash', 'Escape', 'Skate'] |
| 24 | IceFortress | 15 | 20.0 | 4.0 | 3.0 | 1 | 2 | ['Escape', 'Scout', 'Skate', 'Static'] |
| 25 | IceArcher | 3 | 10.0 | 1.0 | 1.0 | 1 | 2 | ['Dash', 'Fortify', 'Freeze', 'Stiff'] |
| 26 | Crab | 10 | 40.0 | 4.0 | 5.0 | 2 | 1 | ['Amphibious', 'AutoFlood', 'Escape', 'Static'] |
| 27 | Gaami | 10 | 30.0 | 4.0 | 3.0 | 1 | 1 | ['AutoFreeze', 'FreezeArea', 'Static'] |
| 28 | Hexapod | 3 | 5.0 | 3.0 | 1.0 | 2 | 1 | ['Creep', 'Dash', 'Escape', 'Sneak'] |
| 29 | Doomux | 10 | 20.0 | 3.5 | 2.0 | 3 | 1 | ['Creep', 'Dash', 'Explode', 'Static'] |
| 30 | Phychi | 3 | 5.0 | 0.699999988079071 | 1.0 | 2 | 2 | ['Dash', 'DoubleAttack', 'Fly', 'Poison', 'Surprise'] |
| 31 | Kiton | 3 | 15.0 | 1.0 | 3.0 | 1 | 1 | ['Creep', 'Poison'] |
| 32 | Exida | 8 | 10.0 | 3.0 | 1.0 | 1 | 3 | ['Creep', 'Poison', 'Splash'] |
| 33 | Centipede | 10 | 20.0 | 4.0 | 3.0 | 2 | 1 | ['Creep', 'Dash', 'Eat', 'Static'] |
| 34 | Segment | 10 | 10.0 | 2.0 | 2.0 | 1 | 1 | ['Creep', 'Dash', 'Explode', 'Independent', 'Static', 'Stiff'] |
| 35 | Raychi | 5 | 10.0 | 3.0 | 2.0 | 3 | 1 | ['Dash', 'Water'] |
| 36 | Shaman | 10 | 10.0 | 1.0 | 1.0 | 1 | 1 | ['Convert', 'Static', 'Swarm'] |
| 37 | Dagger | 2 | 10.0 | 2.0 | 2.0 | 1 | 1 | ['Dash', 'Independent', 'Static', 'Surprise'] |
| 38 | Cloak | 8 | 5.0 | 0.0 | 0.5 | 2 | 1 | ['Creep', 'Dash', 'Hide', 'Infiltrate', 'Scout', 'Static', 'Stiff'] |
| 39 | Dinghy | 2 | 5.0 | 0.0 | 0.5 | 2 | 1 | ['Carry', 'Dash', 'Float', 'Hide', 'Infiltrate', 'Scout', 'Static', 'Stiff'] |
| 40 | Pirate | 2 | 10.0 | 2.0 | 2.0 | 2 | 1 | ['Carry', 'Dash', 'Float', 'Independent', 'Static', 'Surprise'] |
| 41 | Bomber | 15 | 10.0 | 3.0 | 2.0 | 2 | 3 | ['Carry', 'Float', 'Splash', 'Static', 'Stiff'] |
| 42 | Scoutship | 5 | 10.0 | 2.0 | 1.0 | 3 | 2 | ['Carry', 'Dash', 'Float', 'Scout', 'Static'] |
| 43 | Raft | 0 | 10.0 | 0.0 | 1.0 | 2 | 2 | ['Carry', 'Float', 'Static', 'Stiff'] |
| 44 | Rammership | 5 | 10.0 | 3.0 | 3.0 | 3 | 1 | ['Carry', 'Dash', 'Float', 'Static'] |
| 45 | Juggernaut | 10 | 40.0 | 4.0 | 4.0 | 2 | 1 | ['Carry', 'Float', 'Static', 'Stiff', 'Stomp'] |
| 56 | Boomchi | 5 | 10.0 | 3.0 | 3.0 | 2 | 1 | ['Amphibious', 'Dash', 'Explode', 'Stiff'] |
| 57 | LivingIsland | 20 | 20.0 | 4.0 | 4.0 | 2 | 1 | ['Algae', 'Poison', 'Static', 'Stomp', 'Water'] |
| 59 | Mantis | 5 | 15.0 | 3.0 | 3.0 | 1 | 1 | ['Creep', 'Dash'] |
| 60 | InsectEgg | 0 | 10.0 | 2.0 | 3.0 | 0 | 1 | ['Explode', 'Grow', 'Static', 'Stiff'] |
| 61 | Moth | 5 | 10.0 | 2.0 | 0.10000000149011612 | 2 | 1 | ['Dash', 'Fly', 'Infiltrate', 'Poison', 'Sneak', 'Static', 'Stiff'] |
| 62 | Larva | 0 | 10.0 | 2.0 | 2.0 | 1 | 1 | ['Creep', 'Dash', 'Grow', 'Independent', 'Static', 'Surprise'] |


## Technologies

| id | name | tier | requires | next | units | structures |
| --- | --- | --- | --- | --- | --- | --- |
| -1 | BeyondComprehension | 1 | None | [] | [] | [] |
| 0 | Basic | 0 | None | [1, 6, 20, 10, 15] | [2] | [] |
| 1 | Riding | 1 | None | [2, 4] | [3] | [] |
| 2 | FreeSpirit | 2 | 1 | [3] | [] | [17] |
| 3 | Chivalry | 3 | 2 | [] | [4] | [] |
| 4 | Roads | 2 | 1 | [5] | [] | [3, 48, 39] |
| 5 | Trade | 3 | 4 | [] | [] | [50, 41] |
| 6 | Organization | 1 | None | [7, 8] | [] | [] |
| 7 | Strategy | 2 | 6 | [38] | [5] | [] |
| 8 | Farming | 2 | 6 | [9] | [] | [5, 37] |
| 9 | Construction | 3 | 8 | [] | [] | [6] |
| 10 | Fishing | 1 | None | [13, 44] | [43] | [8] |
| 12 | Aquatism | 3 | 44 | [] | [] | [19] |
| 13 | Sailing | 2 | 10 | [14] | [42] | [] |
| 14 | Navigation | 3 | 44 | [] | [41] | [] |
| 15 | Hunting | 1 | None | [18, 16] | [] | [] |
| 16 | Forestry | 2 | 15 | [17] | [] | [12, 32] |
| 17 | Mathematics | 3 | 16 | [] | [8] | [13] |
| 18 | Archery | 2 | 15 | [19] | [9] | [] |
| 19 | Spiritualism | 3 | 18 | [] | [] | [18] |
| 20 | Climbing | 1 | None | [23, 21] | [] | [] |
| 21 | Meditation | 2 | 20 | [22] | [] | [20] |
| 22 | Philosophy | 3 | 21 | [] | [10] | [] |
| 23 | Mining | 2 | 20 | [24] | [] | [21] |
| 24 | Smithery | 3 | 23 | [] | [11] | [22] |
| 25 | FreeDiving | 1 | None | [3] | [] | [] |
| 26 | Spearing | 1 | None | [] | [] | [] |
| 28 | ForestMagic | 1 | None | [18, 16] | [] | [] |
| 30 | Frostwork | 1 | None | [32] | [] | [33] |
| 31 | PolarWarfare | 1 | None | [] | [] | [] |
| 32 | Polarism | 1 | None | [] | [] | [35] |
| 35 | ShockTactics | 1 | None | [] | [] | [] |
| 36 | Recycling | 1 | None | [] | [] | [] |
| 37 | Hydrology | 1 | None | [14] | [] | [38] |
| 38 | Diplomacy | 3 | 7 | [] | [38] | [70] |
| 41 | Sledding | 1 | None | [31] | [] | [] |
| 42 | IceFishing | 1 | None | [32] | [] | [] |
| 44 | Ramming | 2 | 10 | [12] | [44] | [] |
| 48 | Waterways | 1 | 1 | [] | [] | [] |
| 49 | Oceantology | 1 | None | [] | [] | [] |
| 120 | Synergy | 1 | None | [] | [] | [] |
| 121 | Rituals | 2 | 20 | [22] | [] | [] |


## Structures

| id | name | cost | population | stars | terrain | resource | adjacent | per_city | per_tribe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | None | None | 0 | 0 | [] | None | [] | False | False |
| 1 | Village | None | 0 | 0 | [] | None | [] | False | False |
| 2 | Ruin | None | 0 | 0 | [] | None | [] | False | False |
| 3 | Road | 3 | 0 | 0 | [3, 5, 7, 8] | None | [] | False | False |
| 5 | Farm | 5 | 2 | 0 | [3] | 2 | [] | False | False |
| 6 | Windmill | 5 | 1 | 0 | [3] | None | [5] | True | False |
| 8 | Port | 7 | 1 | 0 | [1] | None | [] | False | False |
| 12 | LumberHut | 3 | 1 | 0 | [5] | None | [] | False | False |
| 13 | Sawmill | 5 | 1 | 0 | [3] | None | [12] | True | False |
| 17 | Temple | 20 | 1 | 0 | [3] | None | [] | False | False |
| 18 | ForestTemple | 15 | 1 | 0 | [5] | None | [] | False | False |
| 19 | WaterTemple | 20 | 1 | 0 | [1, 2] | None | [] | False | False |
| 20 | MountainTemple | 20 | 1 | 0 | [4] | None | [] | False | False |
| 21 | Mine | 5 | 2 | 0 | [4] | 5 | [] | False | False |
| 22 | Forge | 5 | 2 | 0 | [3, 5] | None | [21] | True | False |
| 23 | AltarOfPeace | None | 3 | 0 | [1, 3, 6] | None | [] | False | False |
| 24 | TowerOfWisdom | None | 3 | 0 | [1, 3, 6] | None | [] | False | False |
| 25 | GrandBazaar | None | 3 | 0 | [1, 3, 6] | None | [] | False | False |
| 26 | EmperorsTomb | None | 3 | 0 | [1, 3, 6] | None | [] | False | False |
| 27 | GateOfPower | None | 3 | 0 | [1, 3, 6] | None | [] | False | False |
| 28 | ParkOfFortune | None | 3 | 0 | [1, 3, 6] | None | [] | False | False |
| 29 | EyeOfGod | None | 3 | 0 | [1, 3, 6] | None | [] | False | False |
| 32 | Sanctuary | 5 | 0 | 1 | [3, 4, 5] | None | [] | True | False |
| 33 | Outpost | 5 | 1 | 0 | [6] | None | [] | False | False |
| 34 | IceBank | 20 | 0 | 0 | [3, 6] | None | [] | False | True |
| 35 | IceTemple | 20 | 1 | 0 | [6] | None | [] | False | False |
| 37 | Fungi | 5 | 1 | 0 | [3] | 7 | [] | False | False |
| 38 | Algae | 5 | 1 | 0 | [1, 2] | None | [] | False | False |
| 39 | Mycelium | 5 | 0 | 0 | [1, 2, 3, 4, 5] | None | [] | True | False |
| 41 | Clathrus | 5 | 0 | 1 | [1, 2] | None | [] | False | False |
| 47 | Lighthouse | None | 0 | 0 | [] | None | [] | False | False |
| 48 | Bridge | 5 | 0 | 0 | [1] | None | [] | False | False |
| 50 | Market | 5 | 0 | 1 | [3] | None | [6, 13, 22] | True | False |
| 70 | Embassy | 5 | 0 | 0 | [] | None | [] | False | False |
| 123 | ChurchOfConverts | None | 3 | 0 | [1, 3, 5] | None | [] | False | False |


# Polytopia Skirmish: continuous-v1

One task, one continuous coding-agent session, one final Rust bot and one fixed-panel score. The official coding allowance is 2,400 active seconds. There are no staged authoring tracks, forced feedback cycles, automatic repair rounds or separate one-shot leaderboard.

The agent decides when to inspect the engine/rules, edit, compile, run public practice, inspect traces, and revise. Public practice is limited to nine named development maps and the documented public-greedy-v1 baseline. It contains no champion weights or hidden map seeds. The final six-opponent panel is held in a separate verifier. Public practice feedback is never a hidden grading result.

API latency/retries, model reasoning, editing, compilation, practice play, practice replay checks, and trace reading consume the same clock. Setup, final grading and explicitly recorded CPU scheduler waits are excluded. The agent can submit early. At submission or deadline, coder processes stop and the trusted controller atomically captures current main.rs. There is no previous-valid-source fallback. A fresh verifier compiles that exact source with the original locked manifest and replays every played evaluation game.

The official adapter is a host-side Mini-SWE-agent 2.4.6 loop using explicitly selected local CLI-proxy or Daytona/OpenRouter routing, with no automatic paid fallback. Coding commands execute as the coder user in an offline Docker or Daytona environment with 2 CPUs and 2 GiB RAM. The verifier has 2 CPUs and 4 GiB. Credentials remain on the host. Each response has a 32,768-token ceiling including reasoning; tokens and timings are reported. Subscription dollar costs are unavailable, not zero. Harbor's 21,600-second agent timeout is an outer infrastructure guard, not extra coding time; the adapter enforces the 2,400-second active deadline and separately records scheduler waits.

Game ABI: persistent WASI stdin/stdout, one JSON observation per line, one current legal action_id per response. No guest filesystem, networking or environment. Bardur 1v1 with fog, board 14x14. Exactly 200 full rounds, 40,000 shared actions, 4 trillion fuel per guest/game, 256 MiB Wasm memory, 2 seconds per decision and a one-time 5-second startup allowance. Both practice and evaluation use this runtime; practice also stops at the remaining coding deadline. The 1,800-second per-game referee guard is an infrastructure limit.

Primary reward is mean match points over the declared fixed panel: win=1, cap draw=0.5, loss/guest fault/compilation failure=0. The displayed score is reward times 100. Referee errors or timeouts leave evaluation unresolved and produce a Harbor error rather than a model zero. Standard evaluation: six opponents, ten held-out maps in each of three families, both seats, 360 matches. The local integration demo keeps the same 40-minute authoring protocol and uses one map/family, 36 matches; its results are labeled demo and do not establish a model ranking.

One Harbor trial is one authored bot. Evaluation matches are nested measurements, not independent authoring attempts. Repeated trials are averaged with appropriate build/layout uncertainty; hidden scores never select the best authored bot. Champion metrics are separate from primary panel points. Peer leagues and exhaustive horizon/transfer/tactical qualification are optional postprocessing, not additional main authoring tracks or mandatory gates delaying ordinary scores.

Historical v5/v6 protocols, cohorts and source archives remain unchanged records of their own experiments. Their earlier staged/no-feedback results are not silently relabeled as continuous-v1 results.

## docs/rules-alignment-2026-10-27.md

Source SHA-256: 909dcdddc48da33c450a22686368ff24c63cbb262e64f99b05ca24092ec26ffa

````markdown
# Rules alignment with the real game: one ruleset, `polytopia-2.17.3` (2026-10-27)

**What this is.** The game every bot here trains and plays now follows the real Polytopia game (Steam 2.17.3, game
version 123, GameLogicData v28): Bardur, 1v1 on 11/14/18 boards and four-seat free-for-all on 18×18 (diplomacy off
with four seats). There is **one ruleset**, `polytopia-2.17.3` (`src/polytopia_bot/ruleset.py`), implemented by
engine-rs. The research rules the earlier runs used (`bardur-1v1-v1/v2`, the four-seat research rules) and the JAX
engine that defined them are retired: nothing that trains, evaluates, searches or serves games can choose them.

The work happened in two steps on this branch. First the real rules were added behind a ruleset switch
(`bardur-1v1-v3` / `bardur-ffa4-v2` next to the research rules) and checked against the real game. Then, at the
user's request, the switch was removed: the real rules are simply the engine's rules. The parity tables below compare
the research rules ("before") with the final engine ("after").

**Result** (final engine, `runs/rules-align/one-ruleset`):

| check | research rules (before) | real rules (after) |
|---|---|---|
| 27,785 real transitions, identical in every compared field | 19,193 | **19,619** |
| real commands our engine rejects | 25 | **12** (11 follow a differently-rolled ruins reward, 1 peace offer) |
| recorded real positions with the real game's legal moves, apart from the fog exception | 380 of 693 | **685 of 686** |
| positions our engine generated (1v1 and four seats), same | 227 of 373 | **367 of 367** |

Every remaining transition difference has a named cause: the AI difficulty handicap (Crazy earns +3/turn; self-play
seats are both human, so it stays), random outcomes (ruins, explorer, Giant push, Dagger spawns), flag/representation
conventions with no gameplay effect, and three small open items (below). The one deliberate exception: the real game
accepts roads and bridges on tiles the mover has never explored; ours keeps requiring the tile to be explored.

## How to use it

- Nothing to choose. `RustGames(size, games, players=...)` plays the real rules; so do the training actors
  (`rl/distributed.py`, engine-rs only), the arenas (`rl/cpu_arena.py`, `arena_match`, `rl/arena4.py`, the ladders and
  `four_monitor`), the turn search (`rl/search.py`), Studio (`studio/environment.py`), the real-game bridge
  (`experiments/real_game/`) and the replica arenas.
- Checkpoints record `ruleset: "polytopia-2.17.3"` and `simulator_sha256 = ruleset.rules_digest()` (the name and the
  rule table). Old checkpoints (the research rules) still load as evaluation references and play under the new
  rules: `ruleset.LEGACY_DIGESTS` lists the two digests every one of the 2,527 checkpoints on the result volumes
  carries (`7b76e65b…` = bardur-1v1-v1, `e4b656d3…` = bardur-1v1-v2 and the four-seat research runs), including the
  champion (scratch-cont-2 final and its fog head), bn-main4 final, bn4-foghead, ffa4-big2 u7650 and c3 (u4854).
- `LearnerConfig.engine` must be `rust`; `trusted_step` / `trusted_check_every` / `build_rows` are kept only so older
  run configs parse (they were knobs of the JAX actor engine).

## What was removed, and what remains of the JAX engine

Removed (77 files; git history keeps them):
- The ruleset switch: `Rules::real`, `RULESETS`, `for_ruleset`, `RustGames(ruleset=)`, `LearnerConfig.ruleset`, the
  arenas' `ruleset` arguments and `checkpoint_ruleset`.
- The JAX actor engine `cpu_engine.CpuGames` (and the `engine='jax'` actor option) and the JAX turn search
  (`build_planner`, `ParallelPlanner`, `build_replay`, the guarded JAX replay).
- The old single-machine RL stack: `rl/{league, async_league, bootstrap, repair, curriculum, opponents, reward,
  replay_audit, acceptance, arena, diverse_heuristic, public_heuristic, distill_data, distill_train}.py`, the
  `Runtime` / `evaluate` / `run` PPO loop of `rl/runner.py`, `proposals/`, `gpu/{validation, benchmark, rollout}.py`.
- Their experiments and scripts (`modal_structured_*`, `modal_gpu`, `modal_rollout`, `modal_native_episodes`,
  `modal_search_distill`, `modal_search_speed`, `modal_policy_probes`, `modal_actor_profile`, the JAX actor
  benchmarks, the Rust-vs-JAX diff tools, `gpu_crosscheck`, `native_validation`), and their tests, including
  `tests/test_rust_engine.py` (engine-rs vs JAX bitwise, which cannot hold any more) and the JAX engine's own rule
  tests (`test_gpu`, `test_actions`, `test_action_space`, `test_native_rules`, `test_native_boundary`,
  `test_cpu_engine`, `test_fast_engine`, `test_rollout`).

Remains, for reasons that are not game rules:
- `gpu/state.py` (the State layout engine-rs exports), `gpu/rules.json` (engine-rs's rule table, now with the
  GameLogicData corrections), `gpu/importer.py` (Polyfish / real-game worlds), `gpu/mapgen.py` (map layouts for the
  determinized search and the map variants), `gpu/kernels.observe` (read once for the packed-view schema), and
  `gpu/snapshot.py` (state files for Studio).
- `rl/cpu_engine.py` keeps only the packed-view layout and feature encoding (`view_layout`, `view_features`) that the
  actors and the learner share, and the `Prepared` / `Tick` records.
- **Frozen, not maintained:** `polytopia_bot/learning/` (the expert-SFT / language-model line and its scripts) still
  plays the retired rules on the JAX engine (`learning.adapter.GPUEnvironment`); its module docstring says so. Studio
  reuses only its rules-free serializer and memory. `tests/test_learning_adapter.py`, `test_mapgen.py`
  and `test_structured_rl.py` (network/PPO/encoder tests that build fixture positions with `gpu/`) remain;
  `test_snapshot.py` now round-trips engine-rs states.

## How the rules are tested now

1. **Transition parity** on 27,785 recorded real transitions (`experiments/game_ai/transition_parity.py`, classified
   by `parity_classify.py --real`).
2. **Legal-move parity** on recorded real positions (`legal_parity.py`): the real legal set is built exactly as the
   headless host's `tools/il2cpp-host/legal.py` builds it from the running game, with the replica's ports of
   `CommandUtils` / `IsValid` (which reproduce the game's AI on 28,640 of 28,641 real decisions).
3. **Legal-move parity on positions our engine generates** (`legal_parity4.py`), for four seats (no real four-seat
   games exist) and for more 1v1 positions: the replica AI plays all seats in engine-rs.
4. **Unit tests** per rule (`tests/test_real_rules.py`): each rule on one hand-edited position with a control, plus
   random 1v1 and four-seat games that must never reject a legal action. `tests/test_search.py` covers the search.

`modal run --detach --env staging experiments/game_ai/modal_transition_parity.py::main` runs 1–3 as 46 containers
(one per recorded game, one per generated game; about 15 minutes); outputs land on the `polytopia-rules-align`
volume; `experiments/game_ai/parity_report.py` makes the tables.

## Every rule changed (research rules → real game)

Evidence is the decompile (`third_party/polytopia-2.17.3/decompiled/<Class>.c`), the parsed game data
(`gamedata/runtime-gld28.json`) and the replica's ports (`replica/rules.py`, which cite the same functions).

### Combat and units

| # | Rule | Research rules | Real game | Evidence |
|---|---|---|---|---|
| 1 | Cloak and Dinghy retaliation | retaliate (Cloak dealt 1 HP) | never: both are `Stiff` | GameLogicData v28 `unitAbilities` of units 38/39 |
| 2 | Retaliation needs sight | any defender in range retaliates | only if the defender's owner has explored the attacker's tile (the attacker must be one of the defender's attack options) | `BattleHelpers.CanDefenderRetaliate` (v ≥ 120) → `UnitDataExtensions.GetAttackOptionsAtPosition` |
| 3 | Kill XP | every killer counts the kill | none for a vehicle (`Carry`: every boat) or a `Static` unit; a passenger keeps the xp it embarked with | `AttackAction.Execute` (v ≥ 95 branches: `IsVehicle`, `HasAbility(Static)`) |
| 4 | Promotion at sea | a boat is Static, so its passenger can never be promoted | judged by the passenger: a Warrior/Archer/… with 3 xp is promoted on a boat | `UnitDataExtensions.CanBePromoted` (reads `passengerUnit`) |
| 5 | Knight Persist | `attacked` cleared after every kill | cleared only if another attack option remains from the new position | `AttackCommand.ExecuteV104` (Persist block) |
| 6 | Wiped player | units removed, no refund | every unit disbanded with its refund (`WipePlayerAction` → `DisbandUnitAction`); explains the earlier "Defenders vanish with +1 star" captures | `WipePlayerAction.Execute` |

Checked and **equal**: one promotion level in both (`CanBePromoted` requires `promotionLevel == 0`; 896,025 unit
observations in 14 real games never exceed level 1, xp kept through all 156 real promotions); promotion heals to the
new maximum (+5) and costs no action (`PromoteAction.Execute`); Heal Others gives a fixed +4 (`HealOthersAction`:
`HealAction` with 40 tenths) and Recover 4 in friendly / 2 elsewhere (`ActionUtils.GetHealAmount`); unit stats
otherwise equal GameLogicData (the embarked Dinghy/Pirate/Juggernaut store the passenger's health and cost in ours,
which comes out the same).

### Movement and sight

| # | Rule | Research rules | Real game | Evidence |
|---|---|---|---|---|
| 7 | Forest entry | a forest tile with a road can be crossed from anywhere | entering a forest ends the move unless the step runs road to road | `TileData.GetMovementCost` (`HasRoadTo` before the forest rule) |
| 8 | Hide units (Cloak, Dinghy) | blocked by other players' units | may pass tiles held by other players' units (never end there) | `PathFinder.IsTileAccessible` |
| 9 | Peace partners' units | block movement | may be passed (never ended on) | `PathFinder.IsTileAccessible` |
| 10 | Ending on cities | anywhere free | an Infiltrate unit never ends on an opponent's city; no unit ends on a peace or cool-down partner's city | `PathFinder.IsAllowedToFinishOnTile` |
| 11 | Bridges, crossing | any direction | without water movement (Fishing) a bridge cannot be entered straight across its span | `PathFinder.IsTileAccessible`, `TileData.GetNonAllowedBridgeDirection` |
| 12 | Training sight | radius 1 (2 on mountains) | `GetSightRange`: 2 for Scout units (Cloak) | `TrainAction.Execute` → `ActionUtils.ExploreFromTile` |
| 13 | Turn-start exploration | re-explores around every own city (radius 2 for capitals) and unit | none | `StartTurnAction.ExecuteDefault` explores nothing |

### Economy and building

| # | Rule | Research rules | Real game | Evidence |
|---|---|---|---|---|
| 14 | Away stars | meeting pays `1 + 2·min(⌈score/1000⌉, 5)` | +2 more when, with two players, the met player is player 1 (seat 2's first meeting) | `MeetAction.GetReward` / `ShouldAddAwayStars` (v ≥ 123, `PlayerCount == 2`) |
| 15 | Siege and markets | a sieged city pays nothing, markets included | a sieged city's own work stops; every owned market still pays | `ResourceDataUtils.CalculateIncomeFor` sums `TileDataExtensions.CalculateWork` per tile |
| 16 | What is a siege | a visible foreign unit on the city | any other player's unit, hidden ones included | `CalculateWork` (`tile.unit.owner != tile.owner`) |
| 17 | Negative progress | reduces the city's income including its markets | reduces only the city's own work | `CalculateWork` (`min(xp, 0)` inside the city branch) |
| 18 | Roads on villages | allowed on an unowned village | never on a City tile, villages included | `TileData.HasRoad` (true for City/Bridge) in `GameLogicData.CanBuild` |
| 19 | Bridges, building | only inside own borders; span from explored tiles | also on neutral water (`Freelance`, like roads); span from the true terrain of the opposite neighbours | `GameLogicData.CanBuild`, `TileData.GetPossibleBridgeDirection` |
| 20 | Lighthouse | can be destroyed | not destructible | `ImprovementDataExtensions.IsDestructable` (bitmask excludes City, Ruin, LightHouse) |
| 21 | Peace partner on a tile | blocks building, harvesting, forest actions, destroying, neutral roads | does not block | `GameLogicData.CanBuild`, `TileData.CanDestroy` (`u.owner != player && !peace`) |
| 22 | Workable tiles | the tiles of an own city's territory row | additionally the tile itself must be owned by the player | `GameLogicData.CanBuild` (`tile.owner == player`) |
| 23 | Mountain Temple | unlocked by Philosophy (22) | unlocked by Meditation (21) | GameLogicData tech 21 `improvementUnlocks` |
| 24 | Monument terrain | Field, Water, Forest | Field, Water, Ice | improvements 23–29 `terrainRequirements` |
| 25 | Monument uniqueness | blocked by a monument standing in own territory (a captured one too) | blocked only by having built it | `ImprovementAbility.Unique` → `PlayerState.builtUniqueImprovements` |
| 26 | Grand Bazaar | Roads tech and 5 connected cities | 5 connected cities (task enabled by the first connection) | `NetworkTask.IsTaskCompleted`, `ConnectCityAction` |
| 27 | Grow Forest | keeps the field's resource | the resource is removed | `BuildAction.ExecuteDefault` (a `creates` terrain clears the resource; `Consumed`) |
| 28 | Star Fishing | allowed under an improvement (a port) | only on a tile without improvement | `GameLogicData.CanBuild` |
| 29 | Literacy research price | `ceil(price · 0.66)` | `ceil(price · 0.66666f)` (differs from about 50 stars) | `GameLogicData.GetTechPrice` |

Checked and **equal**: the rest of the tech tree, unit unlocks and the ability techs (Free Spirit disband, Chivalry
destroy, Shields peace, Diplomacy embassy/capital vision); the research price formula; per-city limits (`Limited`);
Pacifist / Wealth / Killer / Metropolis / Explorer monument conditions; temple growth; combat damage (no health
difference in thousands of real attacks); flag timing; ruins, explorer, Giant push and Dagger RNG procedures.

### Parity tooling fixes

- `transition_parity.load` now imports diplomacy (peace, cool-down, embassies). Without it, Easy games (which sign
  peace) produced hundreds of false attack/move/build differences.
- `legal_parity4` compares the position without the replica agent's Stay parking (the agent marks parked units as moved
  in its own World only).

## Parity before and after

### Transitions: 27,785 real transitions (all four golden files)

Each recorded command (plus the city-reward choice it raises) is applied to the imported real world before it, and 37
rule-relevant fields are compared with the real world after it.

| ruleset | transitions | identical in every field | rejected by our engine |
|---|---|---|---|
| research rules (before) | 27,785 | 19,193 (69.1%) | 25 |
| real rules (after) | 27,785 | 19,619 (70.6%) | 12 |

The identical share is bounded by conventions, not rules: 4,596 field differences are "ours marks a moved unit that
cannot attack as attacked", 2,827 are the end-of-turn flag reset timing, 1,861 the real StayCommand (a no-op in ours),
846 a structure-level number our engine never reads. Differences per cause, before → after (`parity_classify.py`):

| cause | before | after |
|---|---|---|
| conventions and representation (above, plus lighthouse level) | 10,168 | 10,172 |
| difficulty handicap income (Crazy +4 / Normal +1 / Easy 0 vs ours +1; kept on purpose) | 935 | 958 |
| random outcomes: ruins, explorer walk, Giant push, Dagger spawns | 938 | 942 |
| kill xp of Static units (#3) | 303 | 0 |
| Knight Persist flag (#5) | 29 | 0 |
| away stars (#14) | 7 | 0 (paid by ours now) |
| Cloak retaliation (#1) | 4 | 0 |
| robbed-city import artifact | 6 | 0 |
| unexplained before, gone after: unit kills 69 (vehicle/Static xp), end-turn stars 39 (siege/markets, turn-start exploration), wipe refunds 5, exploration and Cloak-training sight 8 | 121 | 0 |
| **open items** (below): stale market level 52, capture population/connection 50, 1 move, 1 end turn | 104 | 104 |

The 12 remaining rejections: 11 city-reward choices that follow a ruins reward our RNG resolved differently (the real ruins
levelled up a city, ours did not, so the real reward is not pending in ours) and 1 peace offer (diplomacy is imported
only approximately: cool-down owner).

### Legal moves: recorded real positions (every 40th recorded decision)

| ruleset | positions | identical legal sets | identical but for the fog exception | other differing actions |
|---|---|---|---|---|
| research rules | 693 | 214 | 380 | ours-only: build 434, destroy 285, move 57, attack 1; real-only: build 353, move 18, harvest 16, forest 15 |
| real rules (final, sampled per game) | 686 | 349 | **685** | ours-only: attack 1 |

The single remaining case is a hidden Cloak whose `attacked` flag our importer cannot read from the real world.

### Legal moves: positions our engine generated (replica AI on every seat)

| ruleset | positions | identical but for the fog exception | other differing actions | replica commands rejected |
|---|---|---|---|---|
| research rules, 1v1, 4 games 14×14 | 133 | 75 | ours-only: build 220, forest 45, harvest 8, capture 4; real-only: build 281, promote 15 | 284 of 2,650 |
| real rules, 1v1, 4 games 14×14 | 130 | **130** | none | 0 of 2,626 |
| research rules, four seats, 6 games 18×18 | 240 | 152 | ours-only: build 146, capture 8, move 6; real-only: build 171 | 0 of 6,031 |
| real rules, four seats, 6 games 18×18 | 237 | **237** | none | 0 of 5,956 |

Result folders (git-excluded): final engine `runs/rules-align/one-ruleset/{transition,legal,engine2,engine4}`;
research rules `runs/rules-align/r5/{parity,legal}-v2`, `r3/legal{4-v1,2e-v2}`; tables by
`experiments/game_ai/parity_report.py`.

## Remaining known differences

| Difference | Size | Why it stays |
|---|---|---|
| **Roads and bridges on unexplored tiles.** The real game accepts a road (or neutral bridge) command on any tile, explored or not; ours requires the tile to be explored. | 3,273 legal actions in 693 real positions; never played by the real AI | Deliberate. The real UI cannot target an unexplored tile, and allowing it would leak hidden terrain to the policy through the legal mask. |
| **Difficulty handicap.** The real capital pays +handicap (Easy 0, Normal/human 1, Hard 2, Crazy 4). | 958 transitions vs AI seats | Self-play seats are both human-equivalent (+1), as before. |
| **Random outcomes.** Ruins rewards, the Explorer reward walk, the tile a city's occupant is pushed to for a Giant, Dagger spawn tiles. | 942 | Same procedures, different dice (the real game hashes game state). Distributions, not rules. |
| **Flag conventions.** End-of-turn vs turn-start flag reset; ours marks a moved non-Dash unit as attacked; StayCommand. | 9,284 | No gameplay effect (a moved non-Dash unit cannot attack in either engine). |
| **Stale market level.** The real game updates improvement levels at turn start in tile order from neighbours' *stored* levels, so a market can lag one turn behind a sawmill/forge that just changed. | 52 field differences, about 1 star for 1 turn | Tile-order bookkeeping; not ported. |
| **Capital connections after captures.** In 50 field differences (mostly long Easy games with repeated recaptures) the real game drops a road link that ours keeps, costing the city its connected flag and 1 population. | rare | Route rule not pinned down in the decompile. |
| **Resignation in four-seat games.** The real wipe also disbands a resigning player's units; ours keeps them as obstacles. | none in self-play | Training runs with resignation off (`no_resign`). |
| One exploration and one end-turn star difference. | 2 transitions | Not investigated further. |

## Four seats

Four-seat games follow every rule above that applies to four seats (away stars do not: the game pays
them only with two players). Diplomacy stays off, as in v1.

Four-seat check (`legal_parity4.py`): six 18×18 games on the `four-v1` map mix, the replica AI on all four seats
(5,956 commands, none rejected by our engine), every 25th decision compared. The engine equals the real legal set
in all 237 positions apart from the fog exception; the research rules differed in 88 of 240 (Bridges, monuments,
Star Fishing on ports, roads on another player's tile, moves). The rules that only show up with more than two players
were checked in the decompile: away stars need `PlayerCount == 2`; a capture that takes a player's last city wipes them
(`WipePlayerAction`, units disbanded with refunds), as the engine now does; zone of control counts every non-hidden enemy,
seen or not, which matters more with three opponents.

**Cannot be verified for four seats**, because no recorded four-seat real game exists and the replica has no command
executor (only the game's predicates): transitions (income, captures, meetings, eliminations, turn order after
eliminations) are checked only through the shared two-seat code and the four-seat invariant tests
(`tests/test_four_player.py`); the legal-move check relies on the replica arenas' world conversion
(`EngineAgentN.world_dict`).

### What enabling diplomacy in four-seat games would take

1. **Action space.** The diplomatic family has 6 entries for the one opponent (`core::decode`, family 14: break peace,
   accept/decline, embassy, offer, destroy embassy) and is hard-wired to `3 - current_player`
   (`diplomacy::legal_with`, `execute`, `opponent_capital`). Four seats need one block per opponent (18 entries), a new
   catalog width (`family_widths`, `rl/actions.py`), and therefore a new policy head: a widened network can be grown from
   the current four-seat checkpoints, as was done for the 2→4 conversion.
2. **Engine rules per pair.** The state already has per-pair tables (`peace`, `embassy`, `peace_offer`,
   `truce_breaker`, `[MAXP][MAXP]`); `diplomacy::execute` must act on the chosen opponent, embassy income must sum over
   all pairs (`structures::income` already does), breaking peace must expel units from that one player's territory,
   capital vision must reveal each met embassy partner, and `hostile()` must stay per pair.
3. **Real-game rules to match**, all in the decompile: `PeaceTreatyCommand` / `PeaceRequestResponseCommand` (offers
   are messages answered on the receiver's turn, a pending trigger in the real game), `BreakPeaceCommand` and the
   `CoolDown` state (`StartTurnAction.CheckDiplomacy` ends it), `EstablishEmbassyCommand` cost
   (`get_embassy_cost_for_player`), `PlayerDiplomacyExtensions.GetIncomeFromEmbassy`, and the movement/build/attack
   effects already implemented above (#9, #10, #21).
4. **Observations.** The per-seat relation fields exist in the four-seat view (`kernels::VIEW_FIELDS`, one entry per
   seat for the five relation fields), so no new features are needed.
5. **Tests.** Parity on 1v1 Easy games covers peace (they sign treaties); four-seat diplomacy has no real data, so it
   would rest on the replica's predicates (`get_diplomacy_commands_for_opponent`, `is_valid`) and invariant tests.

Rough size: 2–3 days of engine and network work plus a short grown-network smoke, no training budget beyond that.

## Digests and checkpoint compatibility

- New checkpoints: `ruleset = "polytopia-2.17.3"`, `simulator_sha256 = ruleset.rules_digest()` = `77cc7d6a712953a0b33c3e6c8d30c8129eba84cfa2b9ad691b65d84a692a0101`
  (sha256 of the ruleset name and `gpu/rules.json`).
- Loading (`runner.load_checkpoint` → `ruleset.compatible`): a checkpoint of this ruleset, or one whose digest is in
  `ruleset.LEGACY_DIGESTS` (the research rules; weights load unchanged and play under the real rules). The JAX
  engine's own `gpu/snapshot.py COMPATIBLE_DIGESTS` now matters only to the frozen learning line and to Studio's
  state files.
- Engine code digest (`scripts/engine_rs_digest.py`: sha256 over `engine-rs/src/*.rs`, `Cargo.toml`, `Cargo.lock`,
  `gpu/rules.json`): `96c47275e307f685065127455fdabc5e9502a4ec3fbf662a6b16982b7bd29e7c`.

## Tests (Modal, staging)

All 51 test files, one container each (`experiments/modal_distributed_rl.py::tests --parallel`, log
`runs/rules-align/tests-one-ruleset-2.log`; `tests/test_real_rules.py` rerun after the last fix,
`runs/rules-align/tests-real-rules-3.log`): **329 passed, 7 skipped, 0 failed**, plus one file that cannot be collected.

- `tests/test_real_rules.py` (40): every rule with a control position, random 1v1 and four-seat play, one ruleset only.
- `tests/test_search.py` (4): search on engine-rs — raw turns legal and greedy, robust determinized search, belief
  keeps the searcher's view.
- `tests/test_four_player.py`, `test_rl_18x18.py`, `test_distributed_rl.py`, `test_snapshot.py` (rewritten on
  engine-rs states), `test_maps.py`, `test_mapgen.py` and the rest pass.
- Six files collect nothing on a CPU container (GPU-only or opt-in) and exit 5, as before.
- `tests/test_thinking_shards.py` fails to import `merge_thinking_shards` (a script, not on the pytest path). This was
  already broken before this change; it belongs to the frozen learning line and was left alone.
- The Modal test image now mounts `scripts/` too: a dozen learning-line tests load scripts by path and had been
  failing on Modal only because the files were missing.
- Deleted with the code they tested: `test_infiltration_order` (JAX infiltration hash), `test_checkpoint_retention`
  (old watcher script), `test_structured_qualification_plan` (old plan script), alongside the files listed under "What was removed".

## Spend

About $2.5 of Modal (staging) for the whole task, both phases: $2.41 billed across 46 apps when this was written
(the last hour of billing lags). The largest items were the sharded parity runs (~$0.30 each) and the full test
suite (~$0.20). No training was launched.

````


## src/polytopia_bot/gpu/rules.json

Source SHA-256: 9158d807142afe0ea19eef9482daf847d474d11608609417321231e5450f342f

````json
{
  "engine_revision": "a9f90ab54607f5633556d02422d7457f548cc58f",
  "structures": [
    {
      "adjacent": [],
      "cost": null,
      "id": 0,
      "name": "None",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 1,
      "name": "Village",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 2,
      "name": "Ruin",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 3,
      "id": 3,
      "name": "Road",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        3,
        5,
        7,
        8
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 5,
      "name": "Farm",
      "per_city": false,
      "per_tribe": false,
      "population": 2,
      "resource": 2,
      "score": 0,
      "stars": 0,
      "terrain": [
        3
      ],
      "tribe": null
    },
    {
      "adjacent": [
        5
      ],
      "cost": 5,
      "id": 6,
      "name": "Windmill",
      "per_city": true,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        3
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 7,
      "id": 8,
      "name": "Port",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        1
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 3,
      "id": 12,
      "name": "LumberHut",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        5
      ],
      "tribe": null
    },
    {
      "adjacent": [
        12
      ],
      "cost": 5,
      "id": 13,
      "name": "Sawmill",
      "per_city": true,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        3
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 20,
      "id": 17,
      "name": "Temple",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        3
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 15,
      "id": 18,
      "name": "ForestTemple",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        5
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 20,
      "id": 19,
      "name": "WaterTemple",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        1,
        2
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 20,
      "id": 20,
      "name": "MountainTemple",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        4
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 21,
      "name": "Mine",
      "per_city": false,
      "per_tribe": false,
      "population": 2,
      "resource": 5,
      "score": 0,
      "stars": 0,
      "terrain": [
        4
      ],
      "tribe": null
    },
    {
      "adjacent": [
        21
      ],
      "cost": 5,
      "id": 22,
      "name": "Forge",
      "per_city": true,
      "per_tribe": false,
      "population": 2,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        3,
        5
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 23,
      "name": "AltarOfPeace",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 24,
      "name": "TowerOfWisdom",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 25,
      "name": "GrandBazaar",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 26,
      "name": "EmperorsTomb",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 27,
      "name": "GateOfPower",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 28,
      "name": "ParkOfFortune",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 29,
      "name": "EyeOfGod",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 32,
      "name": "Sanctuary",
      "per_city": true,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 1,
      "terrain": [
        3,
        4,
        5
      ],
      "tribe": 5
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 33,
      "name": "Outpost",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        6
      ],
      "tribe": 16
    },
    {
      "adjacent": [],
      "cost": 20,
      "id": 34,
      "name": "IceBank",
      "per_city": false,
      "per_tribe": true,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        3,
        6
      ],
      "tribe": 16
    },
    {
      "adjacent": [],
      "cost": 20,
      "id": 35,
      "name": "IceTemple",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        6
      ],
      "tribe": 16
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 37,
      "name": "Fungi",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": 7,
      "score": 0,
      "stars": 0,
      "terrain": [
        3
      ],
      "tribe": 17
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 38,
      "name": "Algae",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        1,
        2
      ],
      "tribe": 17
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 39,
      "name": "Mycelium",
      "per_city": true,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        1,
        2,
        3,
        4,
        5
      ],
      "tribe": 17
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 41,
      "name": "Clathrus",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 1,
      "terrain": [
        1,
        2
      ],
      "tribe": 17
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 47,
      "name": "Lighthouse",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 48,
      "name": "Bridge",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        1
      ],
      "tribe": null
    },
    {
      "adjacent": [
        6,
        13,
        22
      ],
      "cost": 5,
      "id": 50,
      "name": "Market",
      "per_city": true,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 1,
      "terrain": [
        3
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 70,
      "name": "Embassy",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 123,
      "name": "ChurchOfConverts",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        5
      ],
      "tribe": null
    }
  ],
  "technologies": [
    {
      "id": -1,
      "name": "BeyondComprehension",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 0,
      "name": "Basic",
      "next": [
        1,
        6,
        20,
        10,
        15
      ],
      "requires": null,
      "structures": [],
      "tier": 0,
      "units": [
        2
      ]
    },
    {
      "id": 1,
      "name": "Riding",
      "next": [
        2,
        4
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": [
        3
      ]
    },
    {
      "id": 2,
      "name": "FreeSpirit",
      "next": [
        3
      ],
      "requires": 1,
      "structures": [
        17
      ],
      "tier": 2,
      "units": []
    },
    {
      "id": 3,
      "name": "Chivalry",
      "next": [],
      "requires": 2,
      "structures": [],
      "tier": 3,
      "units": [
        4
      ]
    },
    {
      "id": 4,
      "name": "Roads",
      "next": [
        5
      ],
      "requires": 1,
      "structures": [
        3,
        48,
        39
      ],
      "tier": 2,
      "units": []
    },
    {
      "id": 5,
      "name": "Trade",
      "next": [],
      "requires": 4,
      "structures": [
        50,
        41
      ],
      "tier": 3,
      "units": []
    },
    {
      "id": 6,
      "name": "Organization",
      "next": [
        7,
        8
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 7,
      "name": "Strategy",
      "next": [
        38
      ],
      "requires": 6,
      "structures": [],
      "tier": 2,
      "units": [
        5
      ]
    },
    {
      "id": 8,
      "name": "Farming",
      "next": [
        9
      ],
      "requires": 6,
      "structures": [
        5,
        37
      ],
      "tier": 2,
      "units": []
    },
    {
      "id": 9,
      "name": "Construction",
      "next": [],
      "requires": 8,
      "structures": [
        6
      ],
      "tier": 3,
      "units": []
    },
    {
      "id": 10,
      "name": "Fishing",
      "next": [
        13,
        44
      ],
      "requires": null,
      "structures": [
        8
      ],
      "tier": 1,
      "units": [
        43
      ]
    },
    {
      "id": 12,
      "name": "Aquatism",
      "next": [],
      "requires": 44,
      "structures": [
        19
      ],
      "tier": 3,
      "units": []
    },
    {
      "id": 13,
      "name": "Sailing",
      "next": [
        14
      ],
      "requires": 10,
      "structures": [],
      "tier": 2,
      "units": [
        42
      ]
    },
    {
      "id": 14,
      "name": "Navigation",
      "next": [],
      "requires": 44,
      "structures": [],
      "tier": 3,
      "units": [
        41
      ]
    },
    {
      "id": 15,
      "name": "Hunting",
      "next": [
        18,
        16
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 16,
      "name": "Forestry",
      "next": [
        17
      ],
      "requires": 15,
      "structures": [
        12,
        32
      ],
      "tier": 2,
      "units": []
    },
    {
      "id": 17,
      "name": "Mathematics",
      "next": [],
      "requires": 16,
      "structures": [
        13
      ],
      "tier": 3,
      "units": [
        8
      ]
    },
    {
      "id": 18,
      "name": "Archery",
      "next": [
        19
      ],
      "requires": 15,
      "structures": [],
      "tier": 2,
      "units": [
        9
      ]
    },
    {
      "id": 19,
      "name": "Spiritualism",
      "next": [],
      "requires": 18,
      "structures": [
        18
      ],
      "tier": 3,
      "units": []
    },
    {
      "id": 20,
      "name": "Climbing",
      "next": [
        23,
        21
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 21,
      "name": "Meditation",
      "next": [
        22
      ],
      "requires": 20,
      "structures": [
        20
      ],
      "tier": 2,
      "units": []
    },
    {
      "id": 22,
      "name": "Philosophy",
      "next": [],
      "requires": 21,
      "structures": [],
      "tier": 3,
      "units": [
        10
      ]
    },
    {
      "id": 23,
      "name": "Mining",
      "next": [
        24
      ],
      "requires": 20,
      "structures": [
        21
      ],
      "tier": 2,
      "units": []
    },
    {
      "id": 24,
      "name": "Smithery",
      "next": [],
      "requires": 23,
      "structures": [
        22
      ],
      "tier": 3,
      "units": [
        11
      ]
    },
    {
      "id": 25,
      "name": "FreeDiving",
      "next": [
        3
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 26,
      "name": "Spearing",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 28,
      "name": "ForestMagic",
      "next": [
        18,
        16
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 30,
      "name": "Frostwork",
      "next": [
        32
      ],
      "requires": null,
      "structures": [
        33
      ],
      "tier": 1,
      "units": []
    },
    {
      "id": 31,
      "name": "PolarWarfare",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 32,
      "name": "Polarism",
      "next": [],
      "requires": null,
      "structures": [
        35
      ],
      "tier": 1,
      "units": []
    },
    {
      "id": 35,
      "name": "ShockTactics",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 36,
      "name": "Recycling",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 37,
      "name": "Hydrology",
      "next": [
        14
      ],
      "requires": null,
      "structures": [
        38
      ],
      "tier": 1,
      "units": []
    },
    {
      "id": 38,
      "name": "Diplomacy",
      "next": [],
      "requires": 7,
      "structures": [
        70
      ],
      "tier": 3,
      "units": [
        38
      ]
    },
    {
      "id": 41,
      "name": "Sledding",
      "next": [
        31
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 42,
      "name": "IceFishing",
      "next": [
        32
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 44,
      "name": "Ramming",
      "next": [
        12
      ],
      "requires": 10,
      "structures": [],
      "tier": 2,
      "units": [
        44
      ]
    },
    {
      "id": 48,
      "name": "Waterways",
      "next": [],
      "requires": 1,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 49,
      "name": "Oceantology",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 120,
      "name": "Synergy",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 121,
      "name": "Rituals",
      "next": [
        22
      ],
      "requires": 20,
      "structures": [],
      "tier": 2,
      "units": []
    }
  ],
  "units": [
    {
      "attack": -1.0,
      "cost": -1,
      "defense": 0.0,
      "health": 1.0,
      "id": 0,
      "movement": 1,
      "name": "None",
      "range": 1,
      "skills": []
    },
    {
      "attack": 2.0,
      "cost": 2,
      "defense": 2.0,
      "health": 10.0,
      "id": 2,
      "movement": 1,
      "name": "Warrior",
      "range": 1,
      "skills": [
        "Dash",
        "Fortify"
      ]
    },
    {
      "attack": 2.0,
      "cost": 3,
      "defense": 1.0,
      "health": 10.0,
      "id": 3,
      "movement": 2,
      "name": "Rider",
      "range": 1,
      "skills": [
        "Dash",
        "Escape",
        "Fortify"
      ]
    },
    {
      "attack": 3.5,
      "cost": 8,
      "defense": 1.0,
      "health": 10.0,
      "id": 4,
      "movement": 3,
      "name": "Knight",
      "range": 1,
      "skills": [
        "Dash",
        "Fortify",
        "Persist"
      ]
    },
    {
      "attack": 1.0,
      "cost": 3,
      "defense": 3.0,
      "health": 15.0,
      "id": 5,
      "movement": 1,
      "name": "Defender",
      "range": 1,
      "skills": [
        "Fortify"
      ]
    },
    {
      "attack": 4.0,
      "cost": 8,
      "defense": 0.0,
      "health": 10.0,
      "id": 8,
      "movement": 1,
      "name": "Catapult",
      "range": 3,
      "skills": [
        "Stiff"
      ]
    },
    {
      "attack": 2.0,
      "cost": 3,
      "defense": 1.0,
      "health": 10.0,
      "id": 9,
      "movement": 1,
      "name": "Archer",
      "range": 2,
      "skills": [
        "Dash",
        "Fortify"
      ]
    },
    {
      "attack": 0.0,
      "cost": 5,
      "defense": 1.0,
      "health": 10.0,
      "id": 10,
      "movement": 1,
      "name": "MindBender",
      "range": 1,
      "skills": [
        "Convert",
        "HealOthers",
        "Stiff"
      ]
    },
    {
      "attack": 3.0,
      "cost": 5,
      "defense": 3.0,
      "health": 15.0,
      "id": 11,
      "movement": 1,
      "name": "Swordsman",
      "range": 1,
      "skills": [
        "Dash"
      ]
    },
    {
      "attack": 5.0,
      "cost": 10,
      "defense": 4.0,
      "health": 40.0,
      "id": 12,
      "movement": 1,
      "name": "Giant",
      "range": 1,
      "skills": [
        "Static"
      ]
    },
    {
      "attack": 3.0,
      "cost": 2,
      "defense": 1.0,
      "health": 15.0,
      "id": 15,
      "movement": 1,
      "name": "Polytaur",
      "range": 1,
      "skills": [
        "Dash",
        "Fortify",
        "Independent",
        "Static"
      ]
    },
    {
      "attack": -1.0,
      "cost": 10,
      "defense": 2.0,
      "health": 10.0,
      "id": 17,
      "movement": 1,
      "name": "DragonEgg",
      "range": 1,
      "skills": [
        "Fortify",
        "Grow",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 3.0,
      "cost": 10,
      "defense": 3.0,
      "health": 15.0,
      "id": 18,
      "movement": 2,
      "name": "BabyDragon",
      "range": 1,
      "skills": [
        "Dash",
        "Escape",
        "Fly",
        "Grow",
        "Scout",
        "Static"
      ]
    },
    {
      "attack": 4.0,
      "cost": 10,
      "defense": 3.0,
      "health": 20.0,
      "id": 19,
      "movement": 3,
      "name": "FireDragon",
      "range": 2,
      "skills": [
        "Dash",
        "Fly",
        "Scout",
        "Splash",
        "Static"
      ]
    },
    {
      "attack": 2.0,
      "cost": 3,
      "defense": 1.0,
      "health": 10.0,
      "id": 20,
      "movement": 2,
      "name": "Amphibian",
      "range": 1,
      "skills": [
        "Amphibious",
        "Dash",
        "Escape",
        "Fortify"
      ]
    },
    {
      "attack": 2.5,
      "cost": 8,
      "defense": 1.0,
      "health": 10.0,
      "id": 21,
      "movement": 2,
      "name": "Tridention",
      "range": 2,
      "skills": [
        "Amphibious",
        "Dash",
        "Persist"
      ]
    },
    {
      "attack": 0.0,
      "cost": 5,
      "defense": 1.0,
      "health": 10.0,
      "id": 22,
      "movement": 1,
      "name": "Mooni",
      "range": 1,
      "skills": [
        "AutoFreeze",
        "Skate",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 3.0,
      "cost": 5,
      "defense": 2.0,
      "health": 15.0,
      "id": 23,
      "movement": 2,
      "name": "BattleSled",
      "range": 1,
      "skills": [
        "Dash",
        "Escape",
        "Skate"
      ]
    },
    {
      "attack": 4.0,
      "cost": 15,
      "defense": 3.0,
      "health": 20.0,
      "id": 24,
      "movement": 1,
      "name": "IceFortress",
      "range": 2,
      "skills": [
        "Escape",
        "Scout",
        "Skate",
        "Static"
      ]
    },
    {
      "attack": 1.0,
      "cost": 3,
      "defense": 1.0,
      "health": 10.0,
      "id": 25,
      "movement": 1,
      "name": "IceArcher",
      "range": 2,
      "skills": [
        "Dash",
        "Fortify",
        "Freeze",
        "Stiff"
      ]
    },
    {
      "attack": 4.0,
      "cost": 10,
      "defense": 5.0,
      "health": 40.0,
      "id": 26,
      "movement": 2,
      "name": "Crab",
      "range": 1,
      "skills": [
        "Amphibious",
        "AutoFlood",
        "Escape",
        "Static"
      ]
    },
    {
      "attack": 4.0,
      "cost": 10,
      "defense": 3.0,
      "health": 30.0,
      "id": 27,
      "movement": 1,
      "name": "Gaami",
      "range": 1,
      "skills": [
        "AutoFreeze",
        "FreezeArea",
        "Static"
      ]
    },
    {
      "attack": 3.0,
      "cost": 3,
      "defense": 1.0,
      "health": 5.0,
      "id": 28,
      "movement": 2,
      "name": "Hexapod",
      "range": 1,
      "skills": [
        "Creep",
        "Dash",
        "Escape",
        "Sneak"
      ]
    },
    {
      "attack": 3.5,
      "cost": 10,
      "defense": 2.0,
      "health": 20.0,
      "id": 29,
      "movement": 3,
      "name": "Doomux",
      "range": 1,
      "skills": [
        "Creep",
        "Dash",
        "Explode",
        "Static"
      ]
    },
    {
      "attack": 0.699999988079071,
      "cost": 3,
      "defense": 1.0,
      "health": 5.0,
      "id": 30,
      "movement": 2,
      "name": "Phychi",
      "range": 2,
      "skills": [
        "Dash",
        "DoubleAttack",
        "Fly",
        "Poison",
        "Surprise"
      ]
    },
    {
      "attack": 1.0,
      "cost": 3,
      "defense": 3.0,
      "health": 15.0,
      "id": 31,
      "movement": 1,
      "name": "Kiton",
      "range": 1,
      "skills": [
        "Creep",
        "Poison"
      ]
    },
    {
      "attack": 3.0,
      "cost": 8,
      "defense": 1.0,
      "health": 10.0,
      "id": 32,
      "movement": 1,
      "name": "Exida",
      "range": 3,
      "skills": [
        "Creep",
        "Poison",
        "Splash"
      ]
    },
    {
      "attack": 4.0,
      "cost": 10,
      "defense": 3.0,
      "health": 20.0,
      "id": 33,
      "movement": 2,
      "name": "Centipede",
      "range": 1,
      "skills": [
        "Creep",
        "Dash",
        "Eat",
        "Static"
      ]
    },
    {
      "attack": 2.0,
      "cost": 10,
      "defense": 2.0,
      "health": 10.0,
      "id": 34,
      "movement": 1,
      "name": "Segment",
      "range": 1,
      "skills": [
        "Creep",
        "Dash",
        "Explode",
        "Independent",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 3.0,
      "cost": 5,
      "defense": 2.0,
      "health": 10.0,
      "id": 35,
      "movement": 3,
      "name": "Raychi",
      "range": 1,
      "skills": [
        "Dash",
        "Water"
      ]
    },
    {
      "attack": 1.0,
      "cost": 10,
      "defense": 1.0,
      "health": 10.0,
      "id": 36,
      "movement": 1,
      "name": "Shaman",
      "range": 1,
      "skills": [
        "Convert",
        "Static",
        "Swarm"
      ]
    },
    {
      "attack": 2.0,
      "cost": 2,
      "defense": 2.0,
      "health": 10.0,
      "id": 37,
      "movement": 1,
      "name": "Dagger",
      "range": 1,
      "skills": [
        "Dash",
        "Independent",
        "Static",
        "Surprise"
      ]
    },
    {
      "attack": 0.0,
      "cost": 8,
      "defense": 0.5,
      "health": 5.0,
      "id": 38,
      "movement": 2,
      "name": "Cloak",
      "range": 1,
      "skills": [
        "Creep",
        "Dash",
        "Hide",
        "Infiltrate",
        "Scout",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 0.0,
      "cost": 2,
      "defense": 0.5,
      "health": 5.0,
      "id": 39,
      "movement": 2,
      "name": "Dinghy",
      "range": 1,
      "skills": [
        "Carry",
        "Dash",
        "Float",
        "Hide",
        "Infiltrate",
        "Scout",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 2.0,
      "cost": 2,
      "defense": 2.0,
      "health": 10.0,
      "id": 40,
      "movement": 2,
      "name": "Pirate",
      "range": 1,
      "skills": [
        "Carry",
        "Dash",
        "Float",
        "Independent",
        "Static",
        "Surprise"
      ]
    },
    {
      "attack": 3.0,
      "cost": 15,
      "defense": 2.0,
      "health": 10.0,
      "id": 41,
      "movement": 2,
      "name": "Bomber",
      "range": 3,
      "skills": [
        "Carry",
        "Float",
        "Splash",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 2.0,
      "cost": 5,
      "defense": 1.0,
      "health": 10.0,
      "id": 42,
      "movement": 3,
      "name": "Scoutship",
      "range": 2,
      "skills": [
        "Carry",
        "Dash",
        "Float",
        "Scout",
        "Static"
      ]
    },
    {
      "attack": 0.0,
      "cost": 0,
      "defense": 1.0,
      "health": 10.0,
      "id": 43,
      "movement": 2,
      "name": "Raft",
      "range": 2,
      "skills": [
        "Carry",
        "Float",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 3.0,
      "cost": 5,
      "defense": 3.0,
      "health": 10.0,
      "id": 44,
      "movement": 3,
      "name": "Rammership",
      "range": 1,
      "skills": [
        "Carry",
        "Dash",
        "Float",
        "Static"
      ]
    },
    {
      "attack": 4.0,
      "cost": 10,
      "defense": 4.0,
      "health": 40.0,
      "id": 45,
      "movement": 2,
      "name": "Juggernaut",
      "range": 1,
      "skills": [
        "Carry",
        "Float",
        "Static",
        "Stiff",
        "Stomp"
      ]
    },
    {
      "attack": 3.0,
      "cost": 5,
      "defense": 3.0,
      "health": 10.0,
      "id": 56,
      "movement": 2,
      "name": "Boomchi",
      "range": 1,
      "skills": [
        "Amphibious",
        "Dash",
        "Explode",
        "Stiff"
      ]
    },
    {
      "attack": 4.0,
      "cost": 20,
      "defense": 4.0,
      "health": 20.0,
      "id": 57,
      "movement": 2,
      "name": "LivingIsland",
      "range": 1,
      "skills": [
        "Algae",
        "Poison",
        "Static",
        "Stomp",
        "Water"
      ]
    },
    {
      "attack": 3.0,
      "cost": 5,
      "defense": 3.0,
      "health": 15.0,
      "id": 59,
      "movement": 1,
      "name": "Mantis",
      "range": 1,
      "skills": [
        "Creep",
        "Dash"
      ]
    },
    {
      "attack": 2.0,
      "cost": 0,
      "defense": 3.0,
      "health": 10.0,
      "id": 60,
      "movement": 0,
      "name": "InsectEgg",
      "range": 1,
      "skills": [
        "Explode",
        "Grow",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 2.0,
      "cost": 5,
      "defense": 0.10000000149011612,
      "health": 10.0,
      "id": 61,
      "movement": 2,
      "name": "Moth",
      "range": 1,
      "skills": [
        "Dash",
        "Fly",
        "Infiltrate",
        "Poison",
        "Sneak",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 2.0,
      "cost": 0,
      "defense": 2.0,
      "health": 10.0,
      "id": 62,
      "movement": 1,
      "name": "Larva",
      "range": 1,
      "skills": [
        "Creep",
        "Dash",
        "Grow",
        "Independent",
        "Static",
        "Surprise"
      ]
    }
  ]
}

````


## engine-rs/src/capture.rs

Source SHA-256: 002b790e9de5ded2fe6c89a5da5d65c035b26051799c01ac1d98d56fab8b8940

````rust
//! gpu/capture.py: village/city capture, starfish and ruins (native profile).

use crate::diplomacy::protected;
use crate::discovery::explorer;
use crate::economy::{add_population, claim, discover};
use crate::geom::Geom;
use crate::hash::hash_word;
use crate::military::{exhaust, promote, push, summon};
use crate::rules::{Rules, INDEPENDENT};
use crate::state::State;
use crate::structures::{adjust_building_population, capital_index, connections, population_contribution};

pub fn capture_mask(s: &State, out: &mut [bool]) {
    let u = &s.units;
    let cp = s.current_player;
    for i in 0..s.n {
        let idle = u.owner[i] == cp && !u.moved[i] && !u.attacked[i];
        out[i] = idle && capturable(s, i);
    }
}

/// capture_mask's tile condition (a foreign unprotected village/city, a ruin, or a starfish with Navigation).
#[inline]
pub fn capturable(s: &State, i: usize) -> bool {
    let t = &s.tiles;
    let cp = s.current_player;
    // Star Fishing is a build (GameLogicData.CanBuild), so not on a tile that has an improvement
    (t.structure[i] == 1 && t.owner[i] != cp && !protected(s, cp, t.owner[i])) || t.structure[i] == 2
        || (t.resource[i] == 8 && s.players.tech[s.p()][14] && t.structure[i] == 0)
}

pub fn city(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let before = population_contribution(s, r, g);
    let owner = s.current_player;
    let old = s.tiles.owner[source];
    let enemy = old > 0;
    let old_order = s.cities.order[source];
    let old_territory: Vec<bool> = s.cities.territory[source * n..source * n + n].to_vec();
    let order = (0..n).filter(|&i| s.cities.owner[i] == owner).map(|i| s.cities.order[i]).max().unwrap_or(0).max(0) + 1;
    let area: Vec<bool> = (0..n).map(|t| g.d(source, t) <= 1).collect();
    let membership: Vec<bool> = if enemy { old_territory.clone() } else { area.clone() };
    let transfer = 100 + (s.cities.level[source] - 1) * 50 + s.cities.population[source] * 5 + s.reward(source, 2) * 250;
    let score = if enemy { transfer } else { 100 };
    let old_owners: Vec<i32> = s.cities.owner.clone();
    {
        let c = &mut s.cities;
        c.owner[source] = owner;
        if !enemy {
            c.level[source] = 1;
            c.population[source] = 0;
            c.progress[source] = 0;
            c.production[source] = 1;
            c.border[source] = 1;
            for k in 0..9 {
                c.rewards[source * 9 + k] = 0;
            }
        }
        c.connected[source] = enemy && c.connected[source];
        c.territory[source * n..source * n + n].copy_from_slice(&membership);
        for i in 0..n {
            if enemy && old_owners[i] == old && c.order[i] > old_order {
                c.order[i] -= 1;
            }
        }
        c.order[source] = order;
    }
    let po = (owner - 1) as usize;
    s.players.score[po] = s.players.score[po].wrapping_add(score);
    let oo = s.pclip(old);
    if enemy {
        s.players.score[oo] = s.players.score[oo].wrapping_sub(score);
    }
    let capital_here = s.tiles.capital[source] > 0;
    if enemy && capital_here {
        for k in 0..s.np {
            s.players.embassy[k][oo] = false;
        }
    }
    let eliminated = enemy && !(0..n).any(|i| s.cities.owner[i] == old);
    if eliminated {
        crate::cov::hit(30);
        let oi = (old - 1) as usize;
        s.players.killed_turn[oi] = s.turn.max(1);
        s.players.killer[oi] = owner;
        crate::core::leave(s, oi);
        for i in 0..n {
            if s.units.owner[i] == old {
                // WipePlayerAction: every unit of the wiped player is disbanded, refund included
                crate::military::disband_of(s, i, r, oi);
            }
        }
    }
    // two seats: the reference's end at the first elimination; more: when one seat is left
    s.terminated |= eliminated && (s.np == 2 || s.alive_count() <= 1);
    let claimed: Vec<bool> = (0..n)
        .map(|t| if enemy { membership[t] || t == source } else { area[t] && s.tiles.owner[t] == 0 })
        .collect();
    claim(s, source, &claimed, r);
    adjust_building_population(&before, s, r, g);
    discover(s, if enemy { &old_territory } else { &area }, owner);
    connections(s, owner, g, true);
    if enemy {
        connections(s, old, g, true);
    }
    exhaust(s, source);
    s.units.invisible[source] = false;
    for i in 0..n {
        if s.units.owner[i] != owner && s.units.home[i] == source as i32 + 1 {
            s.units.home[i] = 0;
        }
    }
    let k = s.units.kind[source] as usize;
    s.units.home[source] = if r.skills[k][INDEPENDENT] { 0 } else { source as i32 + 1 };
}

pub fn starfish(s: &mut State, source: usize) {
    exhaust(s, source);
    s.units.invisible[source] = false;
    s.tiles.resource[source] = 0;
    let p = s.p();
    s.players.stars[p] += 8;
}

pub fn ruin(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let p = s.p();
    let cp = s.current_player;
    let tech = s.players.tech[p];
    let mut eligible = [false; 128];
    for k in 0..128 {
        let req = r.free_requires[k];
        let prerequisite = if req >= 0 { tech[req.clamp(0, 127) as usize] } else { r.tech_tier[k] == 1 };
        eligible[k] = r.free_tech[k] && !tech[k] && prerequisite;
    }
    let (cap, has_cap) = capital_index(s, cp);
    let can_explore = s.tiles.terrain[source] != 4
        && (0..n).any(|t| g.d(source, t) <= 2 && t != source && !s.tiles.explored[p][t]);
    let options = [true, eligible.iter().any(|&e| e), has_cap, can_explore, true];
    let seed = hash_word(s.initial_seed, s.seed);
    let total = options.iter().filter(|&&o| o).count() as u32;
    let draw = (seed % total) as i32;
    let mut choice = 0usize;
    let mut seen = 0;
    for (i, &o) in options.iter().enumerate() {
        if o {
            if seen == draw {
                choice = i;
                break;
            }
            seen += 1;
        }
    }
    exhaust(s, source);
    s.units.invisible[source] = false;
    s.seed = seed;
    s.tiles.structure[source] = 0;
    s.tiles.structure_level[source] = 0;
    s.tiles.structure_turn[source] = 0;
    crate::cov::hit(18 + choice);
    match choice {
        0 => s.players.stars[p] += 10,
        1 => {
            let seed = hash_word(s.initial_seed, s.seed);
            let count = eligible.iter().filter(|&&e| e).count() as u32;
            let draw = (seed % count.max(1)) as i32;
            let mut kind = 0usize;
            let mut seen = 0;
            for k in 0..128 {
                if eligible[k] {
                    if seen == draw {
                        kind = k;
                        break;
                    }
                    seen += 1;
                }
            }
            s.seed = seed;
            s.players.tech[p][kind] = true;
            s.players.tech_turn[p][kind] = s.turn;
            s.players.score[p] = s.players.score[p].wrapping_add(100 * r.tech_tier[kind]);
        }
        2 => add_population(s, cap, 3),
        3 => {
            let reveal = explorer(s, source, g);
            discover(s, &reveal, cp);
        }
        _ => {
            let water = s.tiles.terrain[source] <= 2;
            push(s, source, r, g);
            summon(s, source, if water { 44 } else { 11 }, r, g, false);
            promote(s, source, r);
            s.units.passenger[source] = if water { 2 } else { 0 };
            s.units.home[source] = 0;
        }
    }
}

````


## engine-rs/src/core.rs

Source SHA-256: a9cb100b905897f3207ed0097a8e0ef34468690dfdf1a75252c74012aea97a49

````rust
//! gpu/core.py: the atomic action dispatcher, end of turn, terminal rewards; gpu/actions.py::decode.

use crate::capture;
use crate::cov;
use crate::diplomacy;
use crate::economy;
use crate::geom::Geom;
use crate::infiltration;
use crate::kernels::{max_health, movement_targets, research_cost, research_mask, MoveScratch};
use crate::military;
use crate::prof;
use crate::pathfinding::{reveal_tiles, search, PathScratch};
use crate::rules::{Rules, INFILTRATE};
use crate::state::{State, MAXP};
use crate::structures;

pub const OK: i32 = 0;
pub const INVALID_ACTION: i32 = 1;
pub const UNSUPPORTED_RULE: i32 = 2;
pub const STALE_VERSION: i32 = 3;
pub const EPISODE_DONE: i32 = 4;

#[derive(Clone, Copy, Debug)]
pub struct Action {
    pub kind: i32,
    pub source: i32,
    pub target: i32,
    pub parameter: i32,
}

/// Catalog widths of the 17 mask families (gpu/actions.py::decode) for board tiles n.
pub fn family_widths(n: usize) -> [usize; 17] {
    [n * n, n * n, n * 128, n * 128, 128, n, n * 3, n * 9, n, n, n, n, n, n, 6, 1, 1]
}

/// Row widths used by rl/actions.py::flatten_choice (second dim of `shapes`).
pub fn family_row_widths() -> [usize; 17] {
    [0, 0, 128, 128, 128, 1, 3, 9, 1, 1, 1, 1, 1, 1, 6, 1, 1]
}

/// rl.actions.flatten_choice followed by gpu.actions.decode.
pub fn decode(s: &State, family: i32, source: i32, target: i32) -> Action {
    let n = s.n as i64;
    let widths = family_widths(s.n);
    let mut rows = family_row_widths();
    rows[0] = s.n;
    rows[1] = s.n;
    let mut offsets = [0i64; 18];
    for i in 0..17 {
        offsets[i + 1] = offsets[i] + widths[i] as i64;
    }
    let f = family.clamp(0, 16) as usize;
    let flat = offsets[f] + source as i64 * rows[f] as i64 + target as i64;
    let group = (1..=17).filter(|&g| flat >= offsets[g]).count().min(16);
    let at = flat - offsets[group];
    let other = 3 - s.current_player;
    let a = |kind: i32, source: i64, target: i64, parameter: i64| Action {
        kind,
        source: source as i32,
        target: target as i32,
        parameter: parameter as i32,
    };
    match group {
        0 => a(1, at / n, at % n, 0),
        1 => a(2, at / n, at % n, 0),
        2 => a(4, at / 128, -1, at % 128),
        3 => a(6, -1, at / 128, at % 128),
        4 => a(7, -1, -1, at),
        5 => a(5, -1, at, 0),
        6 => a(3, -1, at / 3, at % 3 + 1),
        7 => a(9, -1, at / 9, at % 9),
        8 => a(8, at, -1, 0),
        9 => a(3, at, -1, 7),
        10 => a(3, at, -1, 16),
        11 => a(3, at, -1, 8),
        12 => a(3, at, -1, 9),
        13 => a(3, -1, at, 4),
        14 => {
            let k = (at.clamp(0, 5)) as usize;
            let (capital, _) = diplomacy::opponent_capital(s);
            let targets = [-1i64, 0, 1, capital as i64, -1, -1];
            let abilities = [18i64, 19, 19, 20, 21, 22];
            a(3, other as i64, targets[k], abilities[k])
        }
        15 => a(10, -1, -1, 0),
        _ => a(11, -1, -1, 0),
    }
}

fn meet(s: &mut State) {
    if !s.fog {
        return; // `new` below is false: nothing changes
    }
    let p = s.p();
    let n = s.n;
    // every other seat in order; with two seats exactly the reference's single opponent
    for other in (0..s.np).filter(|&q| q != p) {
        if s.players.known[p][other] {
            continue;
        }
        let sees = (0..n).any(|t| s.tiles.explored[p][t] && s.units.owner[t] == other as i32 + 1 && !s.units.invisible[t]);
        let score = s.players.score[other];
        let mut gift = 1 + 2 * if score > 0 { ((score - 1).div_euclid(1000) + 1).min(5) } else { 0 };
        // game version >= 123 (MeetAction.GetReward / ShouldAddAwayStars): in a two-player game,
        // meeting player 1 pays 2 more ("away stars" for the seat that moves second)
        if s.np == 2 && other == 0 {
            gift += 2;
        }
        if sees {
            s.players.stars[p] += gift;
            s.players.known[p][other] = true;
        }
    }
}

fn research(s: &mut State, a: Action, r: &Rules) {
    let p = s.p();
    let k = a.parameter.clamp(0, 127) as usize;
    let cost = research_cost(s, r, k);
    s.players.stars[p] -= cost;
    s.players.score[p] = s.players.score[p].wrapping_add(r.tech_tier[k] * 100);
    s.players.tech[p][k] = true;
    s.players.tech_turn[p][k] = s.turn;
    meet(s);
}

fn recover(s: &mut State, src: usize, r: &Rules) {
    let cp = s.current_player;
    let territory_owner = s.tiles.owner[src];
    let friendly = territory_owner == cp || (territory_owner > 0 && s.players.peace[s.p()][s.pclip(territory_owner)]);
    let amount = if friendly { 4.0 } else { 2.0 };
    let mh = max_health(s, r, src);
    let u = &mut s.units;
    let hp = if u.poison[src] { u.health[src] } else { (u.health[src] + amount).min(mh) };
    u.health[src] = hp;
    u.poison[src] = false;
    u.moved[src] = true;
    u.attacked[src] = true;
    meet(s);
}

fn move_adjacent(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom, ps: &mut PathScratch) {
    let cp = s.current_player;
    let found = search(s, source, target, r, g, ps);
    // The reference's path mask is {source, target} + ps.path with rank = index in ps.path (source: len, target: 0).
    // A hidden enemy on the path stops the move on the stoppable path tile just before the first one met from the
    // source side (`contact`, the hidden tile of highest rank); ranks along a path are distinct, so the reference's
    // argmax/argmin pick the same tiles.
    let hidden = |i: usize| s.units.invisible[i] && s.units.owner[i] != cp;
    let contact = if found { ps.path.iter().rposition(|&i| hidden(i)) } else { None };
    let (stop, traveled_from) = match contact {
        Some(c) => {
            // tiles of rank > c: path[c+1..] and the source; the first stoppable one by increasing rank
            let stop = ps.path[c + 1..].iter().copied().find(|&i| s.units.owner[i] == 0).unwrap_or(source);
            (stop, c + 1)
        }
        None => (target, 0),
    };
    // traveled: the source plus path tiles of rank >= traveled_from; without a path the mask is {source, target}
    let extra = if !found && target != source { Some(target) } else { None };
    reveal_tiles(s, source, traveled_from, extra, r, ps);
    if let Some(c) = contact {
        cov::hit(33);
        let tile = ps.path[c];
        s.units.invisible[tile] = false;
    }
    if stop != source {
        military::relocate(s, source, stop, r, g, false, Some(&ps.reveal));
    }
    meet(s);
}

fn end_turn(s: &mut State, r: &Rules, g: &Geom) {
    let n = s.n;
    let owner = s.current_player;
    let old = s.p();
    for i in 0..n {
        let u = &s.units;
        let idle = u.owner[i] == owner && !u.moved[i] && !u.attacked[i];
        if idle {
            if !u.poison[i] {
                let to = s.tiles.owner[i];
                let friendly = to == owner || (to > 0 && s.players.peace[old][s.pclip(to)]);
                let amount = if friendly { 4.0 } else { 2.0 };
                let mh = max_health(s, r, i);
                s.units.health[i] = (s.units.health[i] + amount).min(mh);
            }
            s.units.poison[i] = false;
        }
    }
    let p = &s.players;
    let (new_owner, turn, native) = if s.np == 2 {
        let dead = [p.killed_turn[0] > 0 || p.resigned_turn[0] > 0, p.killed_turn[1] > 0 || p.resigned_turn[1] > 0];
        let new_owner = if dead[(2 - owner) as usize] { owner } else { 3 - owner };
        (new_owner, s.turn + (new_owner == 1) as i32, dead[0] || dead[1])
    } else {
        // the next seat still in the game, in seat order; a round ends when the order wraps
        let np = s.np as i32;
        let new_owner = (1..=np).map(|k| (owner - 1 + k) % np + 1).find(|&o| s.alive((o - 1) as usize)).unwrap_or(owner);
        (new_owner, s.turn + (new_owner <= owner) as i32, s.alive_count() <= 1)
    };
    let pacifist = if p.attacked[old] { 0 } else { p.pacifist[old] + p.tech[old][21] as i32 };
    s.players.pacifist[old] = pacifist;
    s.players.attacked[old] = false;
    s.current_player = new_owner;
    s.turn = turn;
    let limit = turn > s.reference_max_turns;
    if limit || native {
        if limit && !native { cov::hit(31); }
        s.truncated = limit && !native;
        s.terminated = native;
        return;
    }
    let frozen: Vec<bool> = s.units.frozen.clone();
    diplomacy::start_turn(s, new_owner);
    // StartTurnAction explores nothing: tiles are explored when units move, train, upgrade or convert and when
    // borders grow
    meet(s);
    structures::start_turn(s, new_owner);
    let mut income = structures::income(s, new_owner, r, g);
    if !(turn > 0 && income > 0) {
        income = 0;
    }
    for i in 0..n {
        if s.cities.riot[i] && s.cities.owner[i] == new_owner {
            s.cities.riot[i] = false;
        }
    }
    let no = (new_owner - 1) as usize;
    s.players.stars[no] += income;
    for i in 0..n {
        if s.units.owner[i] == new_owner {
            s.units.moved[i] = frozen[i];
            s.units.attacked[i] = frozen[i];
            if !frozen[i] {
                s.units.attacks_performed[i] = 0;
            }
            s.units.frozen[i] = false;
        }
    }
}

fn resign(s: &mut State) {
    let p = s.p();
    s.players.resigned_turn[p] = s.turn.max(1);
    if s.np == 2 {
        s.terminated = true;
        s.players.killer[p] = 3 - s.current_player;
    } else {
        // the seat leaves (its pieces stay as obstacles; the turn order skips it); the game ends at one seat left
        leave(s, p);
        s.terminated = s.alive_count() <= 1;
    }
}

/// Record that seat `k` has just left a multi-seat game (killed or resigned): its place in the leaving order.
pub fn leave(s: &mut State, k: usize) {
    if s.np > 2 && s.players.out_order[k] == 0 {
        s.players.out_order[k] = s.players.out_order.iter().filter(|&&o| o > 0).count() as i32 + 1;
    }
}

/// Reward of finishing place `place` (0 = first) among `np` seats: evenly spaced from +1 down to -1, zero-sum.
#[inline]
pub fn place_reward(place: usize, np: usize) -> f32 {
    1.0 - 2.0 * place as f32 / (np - 1) as f32
}

/// Seat-indexed final rewards (rows >= np are 0). Two seats: the reference's win +1 / loss -1, 0 unless terminated.
/// More seats: rank rewards (place_reward), set at termination and at truncation. Seats that left take the
/// places from last upward in the order they left; the seats still in share the top places equally (a seat
/// still in at the round cap ties with every other one still in).
pub fn terminal_rewards(s: &State) -> [f32; MAXP] {
    let p = &s.players;
    let mut out = [0f32; MAXP];
    if s.np == 2 {
        let dead = [p.killed_turn[0] > 0 || p.resigned_turn[0] > 0, p.killed_turn[1] > 0 || p.resigned_turn[1] > 0];
        if s.terminated {
            for k in 0..2 {
                let winner = !dead[k] && dead[1 - k];
                out[k] = if winner { 1.0 } else if dead[k] { -1.0 } else { 0.0 };
            }
        }
        return out;
    }
    if !(s.terminated || s.truncated) {
        return out;
    }
    let np = s.np;
    let alive = s.alive_count();
    let shared = (0..alive).map(|k| place_reward(k, np)).sum::<f32>() / alive.max(1) as f32;
    for k in 0..np {
        out[k] = if s.alive(k) { shared } else { place_reward(np - p.out_order[k].clamp(1, np as i32) as usize, np) };
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Research, Recover, Move, End, Attack, Summon, Harvest, Forest, Promote, Disband, Reward, Build, Destroy, Cap,
    Heal, Diplomatic, Resign, Unsupported,
}

fn classify(a: Action) -> Kind {
    match (a.kind, a.parameter) {
        (7, _) => Kind::Research,
        (3, 7) => Kind::Recover,
        (1, _) => Kind::Move,
        (10, _) => Kind::End,
        (2, _) => Kind::Attack,
        (4, _) => Kind::Summon,
        (5, _) => Kind::Harvest,
        (3, 1..=3) => Kind::Forest,
        (3, 16) => Kind::Promote,
        (3, 8) => Kind::Disband,
        (9, _) => Kind::Reward,
        (6, _) => Kind::Build,
        (3, 4) => Kind::Destroy,
        (8, _) => Kind::Cap,
        (3, 9) => Kind::Heal,
        (3, 18..=22) => Kind::Diplomatic,
        (11, _) => Kind::Resign,
        _ => Kind::Unsupported,
    }
}

/// Scratch buffers reused across steps.
pub struct Scratch {
    pub row: Vec<bool>,
    pub mv: MoveScratch,
    pub build: structures::BuildCtx,
    pub summon: military::SummonCtx,
    pub path: PathScratch,
}

impl Scratch {
    pub fn new(n: usize) -> Scratch {
        Scratch { row: vec![false; n * 9], mv: MoveScratch::new(n), build: structures::BuildCtx::new(n),
                  summon: military::SummonCtx::new(n), path: PathScratch::new(n) }
    }
}

/// core.step: apply one atomic action; rejection leaves the state untouched. Returns the status code.
pub fn step(s: &mut State, a: Action, expected_version: i32, r: &Rules, g: &Geom, max_actions: i32, sc: &mut Scratch) -> i32 {
    step_with(s, a, expected_version, r, g, max_actions, sc, None)
}

/// step, given optionally the packed legal masks (masks::Packed layout) computed for exactly this state: Move,
/// Build and Summon are then validated by one mask lookup. The status is the same either way: where a mask entry
/// and step's own predicate can differ (a pending city reward, the kind-37 build, an unsupported unit kind, a
/// core_supported=False state) the action is unsupported regardless of validity.
#[allow(clippy::too_many_arguments)]
pub fn step_with(s: &mut State, a: Action, expected_version: i32, r: &Rules, g: &Geom, max_actions: i32, sc: &mut Scratch,
                 legal: Option<&[u8]>) -> i32 {
    let n = s.n as i32;
    let legal = if s.core_supported { legal } else { None };
    let o = crate::masks::family_offsets(s.n);
    let bit = |b: &[u8], i: usize| b[i >> 3] >> (7 - (i & 7)) & 1 == 1;
    let src = a.source.clamp(0, n - 1) as usize;
    let dst = a.target.clamp(0, n - 1) as usize;
    let param = a.parameter.clamp(0, 127) as usize;
    let source_valid = a.source >= 0 && a.source < n;
    let target_valid = a.target >= 0 && a.target < n;
    let param_valid = a.parameter >= 0 && a.parameter < 128;
    let cp = s.current_player;
    let kind = classify(a);
    let own = source_valid && s.units.owner[src] == cp;
    let idle = own && !s.units.moved[src] && !s.units.attacked[src];
    let pending = s.pending_reward();
    let reward = kind == Kind::Reward;
    let reward_implemented = matches!(a.parameter, 1..=8);
    let mut supported = s.core_supported && kind != Kind::Unsupported && (!pending || reward || kind == Kind::Resign)
        && (!reward || reward_implemented);
    if kind == Kind::Summon {
        supported &= r.unit_supported[param];
    }
    if kind == Kind::Build {
        supported &= param != 37;
    }
    let valid = prof::time(prof::VALIDATE + kind as usize, || match kind {
        Kind::End | Kind::Resign => true,
        Kind::Research => {
            let mut m = [false; 128];
            research_mask(s, r, &mut m);
            param_valid && m[param]
        }
        Kind::Recover => idle && s.units.health[src] < max_health(s, r, src),
        Kind::Move => source_valid && target_valid && match legal {
            Some(b) => bit(b, o[0] + src * s.n + dst),
            None => {
                movement_targets(s, src, r, g, &mut sc.mv, &mut sc.row);
                sc.row[dst]
            }
        },
        Kind::Attack => {
            military::attack_mask(s, src, r, g, &mut sc.row);
            source_valid && target_valid && sc.row[dst]
        }
        Kind::Summon => source_valid && param_valid && match legal {
            Some(b) => bit(b, o[2] + src * 128 + param),
            None => {
                sc.summon.prepare(s, r);
                military::summon_row(s, &sc.summon, src) >> param & 1 == 1
            }
        },
        Kind::Harvest => {
            economy::harvest_mask(s, &mut sc.row);
            target_valid && sc.row[dst]
        }
        Kind::Forest => {
            economy::forest_masks(s, &mut sc.row);
            target_valid && sc.row[dst * 3 + (a.parameter - 1).clamp(0, 2) as usize]
        }
        Kind::Promote => own && military::promotable(s, r, src),
        Kind::Disband => idle && s.players.tech[s.p()][2],
        Kind::Build => target_valid && param_valid && match legal {
            Some(b) => bit(b, o[3] + dst * 128 + param),
            None => structures::build_legal(s, r, g, &mut sc.build, dst, param),
        },
        Kind::Destroy => {
            structures::destroy_mask(s, &mut sc.row);
            target_valid && sc.row[dst]
        }
        Kind::Heal => {
            idle && s.units.kind[src] == 10
                && (0..s.n).any(|i| g.d(src, i) == 1 && s.units.owner[i] == cp && s.units.health[i] < max_health(s, r, i))
        }
        Kind::Cap => {
            capture::capture_mask(s, &mut sc.row);
            source_valid && sc.row[src]
        }
        Kind::Diplomatic => diplomacy::legal(s, a.source, a.target, a.parameter),
        Kind::Reward => {
            economy::reward_mask(s, &mut sc.row);
            target_valid && a.parameter >= 0 && a.parameter < 9 && sc.row[dst * 9 + a.parameter.clamp(0, 8) as usize]
        }
        Kind::Unsupported => false,
    });
    let mut status = if !supported { UNSUPPORTED_RULE } else if valid { OK } else { INVALID_ACTION };
    if expected_version != s.state_version {
        status = STALE_VERSION;
    }
    if s.terminated || s.truncated {
        status = EPISODE_DONE;
    }
    if status != OK {
        return status;
    }
    if !s.monuments_settled {
        prof::time(prof::STEP_PRE, || structures::remember_monuments(s, r, g));
    }
    prof::time(prof::EXECUTE + kind as usize, || match kind {
        Kind::Research => { cov::hit(0); research(s, a, r) }
        Kind::Recover => { cov::hit(1); recover(s, src, r) }
        Kind::Move => { cov::hit(2); move_adjacent(s, src, dst, r, g, &mut sc.path) }
        Kind::End => { cov::hit(3); end_turn(s, r, g) }
        Kind::Attack => {
            let uk = s.units.kind[src] as usize;
            if r.skills[uk][INFILTRATE] {
                cov::hit(6);
                infiltration::execute(s, src, dst, r, g);
            } else if uk == 10 {
                cov::hit(5);
                military::convert(s, src, dst, r, g);
            } else {
                cov::hit(4);
                military::attack(s, src, dst, r, g);
            }
        }
        Kind::Summon => {
            if s.units.kind[src] == 43 {
                cov::hit(8);
                military::upgrade(s, src, param, r, g);
            } else {
                cov::hit(7);
                military::summon(s, src, param, r, g, true);
            }
        }
        Kind::Harvest => { cov::hit(9); economy::harvest(s, dst) }
        Kind::Forest => { cov::hit(10); economy::forest(s, dst, a.parameter) }
        Kind::Promote => { cov::hit(11); military::promote(s, src, r) }
        Kind::Disband => { cov::hit(12); military::disband(s, src, r) }
        Kind::Reward => { cov::hit(13); if a.parameter == 6 { cov::hit(38); } if a.parameter == 4 { cov::hit(39); } economy::reward(s, dst, a.parameter, r, g) }
        Kind::Build => { cov::hit(14); structures::build(s, dst, param, r, g) }
        Kind::Destroy => { cov::hit(15); structures::destroy(s, dst, r, g) }
        Kind::Cap => {
            let st = s.tiles.structure[src];
            if st == 1 {
                cov::hit(16);
                if s.tiles.owner[src] > 0 { cov::hit(17); }
                capture::city(s, src, r, g);
            } else if st == 2 {
                capture::ruin(s, src, r, g);
            } else {
                cov::hit(23);
                capture::starfish(s, src);
            }
        }
        Kind::Heal => { cov::hit(24); military::heal_others(s, src, r, g) }
        Kind::Diplomatic => { cov::hit(25); diplomacy::execute(s, a.target, a.parameter, r, g) }
        Kind::Resign => { cov::hit(26); resign(s) }
        Kind::Unsupported => unreachable!(),
    });
    prof::time(prof::STEP_POST, || {
        if a.kind == 2 {
            diplomacy::hostile(s);
        }
        meet(s);
        diplomacy::capital_vision(s);
        structures::remember_monuments(s, r, g);
    });
    s.monuments_settled = true;
    let count = s.actions + 1;
    s.state_version += 1;
    s.actions = count;
    if count >= max_actions && !s.terminated && !s.truncated { cov::hit(32); }
    s.truncated = (s.truncated || count >= max_actions) && !s.terminated;
    OK
}

````


## engine-rs/src/cov.rs

Source SHA-256: 11985a1a7060782ad77e6a7d3e6be0ef656593105b224cdf73f0a2fabed5e8ec

````rust
//! Branch-coverage counters (thread-local, diagnostic only): which rule paths a run exercised.

use std::cell::RefCell;

pub const NAMES: [&str; 40] = [
    "research", "recover", "move", "end_turn", "attack", "convert", "infiltrate", "summon", "upgrade", "harvest",
    "forest", "promote", "disband", "reward", "build", "destroy", "capture_city", "capture_enemy_city", "ruin_stars",
    "ruin_tech", "ruin_population", "ruin_explore", "ruin_unit", "starfish", "heal_others", "diplomacy", "resign",
    "splash", "stomp", "push", "elimination", "truncated_round_cap", "truncated_max_actions", "move_hidden_encounter",
    "retaliation_kill", "advance_after_kill", "embark_disembark", "area_damage_kill", "reward_giant", "reward_explore",
];

thread_local! {
    static COV: RefCell<[u64; 40]> = const { RefCell::new([0; 40]) };
}

#[inline]
pub fn hit(i: usize) {
    COV.with(|c| c.borrow_mut()[i] += 1);
}

pub fn snapshot() -> [u64; 40] {
    COV.with(|c| *c.borrow())
}

pub fn reset() {
    COV.with(|c| *c.borrow_mut() = [0; 40]);
}

````


## engine-rs/src/diplomacy.rs

Source SHA-256: 9eea31c7a81e2b0637fcb54b1f2114565be972b18a3b9a312f698d27f5db6cc6

````rust
//! gpu/diplomacy.py: native two-player diplomacy. Games with more seats have no diplomacy (no ability is legal);
//! their per-pair tables stay false, so the peace terms elsewhere are inert.

use crate::economy::discover;
use crate::geom::Geom;
use crate::military::remove;
use crate::rules::Rules;
use crate::state::State;
use crate::structures::connections;

#[inline]
pub fn protected(s: &State, owner: i32, other: i32) -> bool {
    let (p, q) = (s.pclip(owner), s.pclip(other));
    other > 0 && (owner == other || s.players.peace[p][q] || s.players.truce_breaker[p][q] != 0)
}

/// Opponent capital tile (first match) and whether one exists; shared by legal_masks and decode.
pub fn opponent_capital(s: &State) -> (usize, bool) {
    if s.np != 2 {
        return (0, false);
    }
    let other = 3 - s.current_player;
    for i in 0..s.n {
        if s.tiles.capital[i] == other && s.cities.owner[i] == other {
            return (i, true);
        }
    }
    (0, false)
}

/// diplomacy.legal for Action(3, source, target, ability).
pub fn legal(s: &State, source: i32, target: i32, ability: i32) -> bool {
    legal_with(s, source, target, ability, opponent_capital(s))
}

/// diplomacy.legal given opponent_capital(s).
pub fn legal_with(s: &State, source: i32, target: i32, ability: i32, (capital, has_capital): (usize, bool)) -> bool {
    if s.np != 2 {
        return false;
    }
    let p = s.p();
    let q = (2 - s.current_player) as usize;
    let pl = &s.players;
    let other = q as i32 + 1;
    let eligible = source == other && pl.known[p][q] && pl.killed_turn[q] == 0 && pl.resigned_turn[q] == 0;
    eligible
        && ((ability == 18 && pl.peace[p][q])
            || (ability == 19 && pl.peace_offer[p][q] && (target == 0 || target == 1))
            || (ability == 20 && pl.tech[p][38] && !pl.embassy[p][q] && has_capital && target == capital as i32
                && pl.stars[p] >= 5 && s.turn * 2 + p as i32 >= pl.hostile_until[p])
            || (ability == 21 && pl.tech[p][7] && !pl.peace[p][q] && !pl.peace_offer[q][p] && pl.truce_breaker[p][q] == 0)
            || (ability == 22 && pl.embassy[p][q]))
}

pub fn hostile(s: &mut State) {
    let p = s.p();
    s.players.embassy = Default::default();
    s.players.attacked[p] = true;
    s.players.peace_offer = Default::default();
    s.players.hostile_until[p] = s.turn * s.np as i32 + p as i32 + 2;
}

pub fn capital_vision(s: &mut State) {
    let n = s.n;
    let p = s.p();
    if !s.players.tech[p][38] {
        return; // discover of an empty reveal changes nothing
    }
    // the capitals of every met seat (with two seats: the opponent's, once met)
    let met: Vec<i32> = (0..s.np).filter(|&q| q != p && s.players.known[p][q]).map(|q| q as i32 + 1).collect();
    if met.is_empty() {
        return;
    }
    let reveal: Vec<bool> = (0..n).map(|t| met.contains(&s.tiles.capital[t])).collect();
    discover(s, &reveal, s.current_player);
}

pub fn start_turn(s: &mut State, owner: i32) {
    for a in 0..s.np {
        for b in 0..s.np {
            if s.players.truce_breaker[a][b] == owner {
                s.players.truce_breaker[a][b] = 0;
            }
        }
    }
}

pub fn execute(s: &mut State, target: i32, ability: i32, r: &Rules, g: &Geom) {
    let n = s.n;
    let cp = s.current_player;
    let p = s.p();
    let q = (2 - cp) as usize;
    match ability {
        18 => {
            s.players.peace[p][q] = false;
            s.players.peace[q][p] = false;
            s.players.truce_breaker[p][q] = cp;
            s.players.truce_breaker[q][p] = cp;
            let doomed: Vec<usize> = (0..n).filter(|&i| s.units.owner[i] == cp && s.tiles.owner[i] == q as i32 + 1).collect();
            for i in doomed {
                remove(s, i, r, None, true);
            }
            for i in 0..n {
                if s.units.owner[i] == cp {
                    s.units.moved[i] = true;
                    s.units.attacked[i] = true;
                }
            }
        }
        19 => {
            let accepted = target == 1;
            s.players.peace[p][q] = accepted;
            s.players.peace[q][p] = accepted;
            s.players.peace_offer[p][q] = false;
            s.players.peace_offer[q][p] = false;
        }
        20 => {
            s.players.embassy[p][q] = true;
            s.players.stars[p] -= 5;
            s.players.known[q][p] = true;
            let t = target.clamp(0, n as i32 - 1) as usize;
            let reveal: Vec<bool> = (0..n).map(|i| g.d(t, i) <= 1).collect();
            discover(s, &reveal, cp);
        }
        21 => {
            s.players.peace_offer[q][p] = true;
        }
        _ => {
            s.players.embassy[p][q] = false;
        }
    }
    connections(s, cp, g, true);
    connections(s, 3 - cp, g, true);
}

````


## engine-rs/src/discovery.rs

Source SHA-256: dfff5ec6c1c16472bfa360c2255fbd7b4ffbf5ef15a907369c55b90d59fffcc8

````rust
//! gpu/discovery.py::explorer, the reference's deterministic city explorer.

use crate::geom::Geom;
use crate::hash::hash_word;
use crate::state::State;

pub fn explorer(s: &State, source: usize, g: &Geom) -> Vec<bool> {
    let n = s.n;
    let p = s.p();
    let initial = &s.tiles.explored[p];
    let tech = &s.players.tech[p];
    let mut visible: Vec<bool> = initial.clone();
    let mut current = source;
    let mut previous = -1i32;
    let mut score = vec![0i32; n];
    let mut spread = vec![0i32; n];
    for _ in 0..12 {
        for t in 0..n {
            if visible[t] {
                score[t] = 1000;
            } else {
                let mut count = 0;
                let mut light = false;
                for j in 0..n {
                    if g.d(t, j) <= 2 && !visible[j] {
                        count += 1;
                        if s.tiles.structure[j] == 47 {
                            light = true;
                        }
                    }
                }
                let count = count.clamp(1, 4);
                score[t] = 183 - count * 10 - if light { 33 } else { 0 };
            }
        }
        for _ in 0..3 {
            for t in 0..n {
                let mut best = 1000;
                for &j in &g.adj[t] {
                    best = best.min(score[j as usize] + 100);
                }
                spread[t] = score[t].min(best);
            }
            std::mem::swap(&mut score, &mut spread);
        }
        let mut allowed = [false; 8];
        let mut values = [10000i32; 8];
        let mut best = 10000i32;
        for k in 0..8 {
            if !g.nb_valid[current * 8 + k] {
                continue;
            }
            let idx = g.nb[current * 8 + k] as usize;
            let terrain = s.tiles.terrain[idx];
            let passable = terrain != 0 && (terrain != 4 || tech[20]) && (terrain != 1 || tech[13]) && (terrain != 2 || tech[14]);
            allowed[k] = !visible[idx] || passable;
            if allowed[k] {
                values[k] = score[idx] + (idx as i32 == previous) as i32;
                best = best.min(values[k]);
            }
        }
        let candidates: Vec<usize> = (0..8).filter(|&k| allowed[k] && values[k] == best).collect();
        let count = candidates.len() as u32;
        if count == 0 {
            break;
        }
        let draw = hash_word(s.seed, current as u32) % count;
        let target = g.nb[current * 8 + candidates[draw as usize]] as usize;
        if target == current {
            break;
        }
        for j in 0..n {
            if g.d(target, j) <= 1 {
                visible[j] = true;
            }
        }
        previous = current as i32;
        current = target;
    }
    (0..n).map(|t| visible[t] && !initial[t]).collect()
}

````


## engine-rs/src/economy.rs

Source SHA-256: ea2a6db4596b5756e0cae9056cedd0bcd5708ec0c328a58e49a312e3eacf81b1

````rust
//! gpu/economy.py: population, territory, harvest/forest actions, discovery and city rewards.

use crate::geom::Geom;
use crate::rules::Rules;
use crate::state::State;

pub fn add_population(s: &mut State, index: usize, amount: i32) {
    let n = s.n;
    let cp = s.current_player;
    let mut v = vec![0i32; n];
    if index < n && s.cities.owner[index] == cp {
        v[index] = amount;
    }
    add_population_vector(s, &v, cp, false);
}

/// economy.add_population_vector: level-ups (and, when !native, level-downs) per city, score to `owner`.
pub fn add_population_vector(s: &mut State, amount: &[i32], owner: i32, native: bool) {
    let n = s.n;
    let mut score: i32 = 0;
    for i in 0..n {
        let active = s.cities.owner[i] == owner && amount[i] != 0;
        let a = if active { amount[i] } else { 0 };
        let old_level = s.cities.level[i];
        let c = &mut s.cities;
        c.progress[i] += a;
        if active {
            while c.progress[i] >= c.level[i] + 1 {
                c.progress[i] -= c.level[i] + 1;
                c.level[i] += 1;
                c.production[i] += 1;
                if s.tiles.structure[i] > 0 {
                    s.tiles.structure_level[i] += 1;
                }
            }
            if !native {
                while c.progress[i] < 0 && c.level[i] > 1 {
                    c.progress[i] += c.level[i];
                    c.level[i] -= 1;
                    c.production[i] = (c.production[i] - 1).max(1);
                    if s.tiles.structure[i] > 0 {
                        s.tiles.structure_level[i] = (s.tiles.structure_level[i] - 1).max(1);
                    }
                }
            }
        }
        c.population[i] += a;
        score += a * 5 + (c.level[i] - old_level) * 50;
    }
    let o = (owner - 1) as usize;
    s.players.score[o] = s.players.score[o].wrapping_add(score);
}

pub fn in_territory(s: &State, out: &mut [bool]) {
    let n = s.n;
    out[..n].fill(false);
    for c in 0..n {
        if s.cities.owner[c] == s.current_player {
            let row = &s.cities.territory[c * n..c * n + n];
            for t in 0..n {
                out[t] |= row[t];
            }
        }
    }
}

pub fn economic_tiles(s: &State, out: &mut [bool]) {
    in_territory(s, out);
    for t in 0..s.n {
        out[t] = out[t] && unit_allows_work(s, t) && s.tiles.owner[t] == s.current_player;
    }
}

/// No unit blocks work on tile `t`: none, the mover's, a hidden one, or (GameLogicData.CanBuild /
/// TileData.CanDestroy) a unit of a player the mover has peace with.
#[inline]
pub fn unit_allows_work(s: &State, t: usize) -> bool {
    let o = s.units.owner[t];
    o == 0 || o == s.current_player || s.units.invisible[t] || s.players.peace[s.p()][s.pclip(o)]
}

pub fn harvest_mask(s: &State, out: &mut [bool]) {
    let mut econ = vec![false; s.n];
    economic_tiles(s, &mut econ);
    harvest_mask_with(s, &econ, out);
}

/// harvest_mask given economic_tiles(s).
pub fn harvest_mask_with(s: &State, econ: &[bool], out: &mut [bool]) {
    let n = s.n;
    let tech = &s.players.tech[s.p()];
    let stars = s.players.stars[s.p()];
    for t in 0..n {
        let r = s.tiles.resource[t];
        let unlocked = (r == 1 && tech[15]) || (r == 3 && tech[10]) || (r == 6 && tech[6]);
        out[t] = econ[t] && s.tiles.structure[t] == 0 && unlocked && stars >= 2;
    }
}

pub fn harvest(s: &mut State, target: usize) {
    let city = s.tiles.city[target].clamp(0, s.n as i32 - 1) as usize;
    s.tiles.resource[target] = 0;
    let p = s.p();
    s.players.stars[p] -= 2;
    add_population(s, city, 1);
}

/// out[t*3 + k]
pub fn forest_masks(s: &State, out: &mut [bool]) {
    let mut econ = vec![false; s.n];
    economic_tiles(s, &mut econ);
    forest_masks_with(s, &econ, out);
}

/// forest_masks given economic_tiles(s).
pub fn forest_masks_with(s: &State, econ: &[bool], out: &mut [bool]) {
    let n = s.n;
    let tech = &s.players.tech[s.p()];
    let stars = s.players.stars[s.p()];
    for t in 0..n {
        let free = econ[t] && (s.tiles.structure[t] == 0 || s.tiles.structure[t] == 3);
        let forest = s.tiles.terrain[t] == 5;
        let field = s.tiles.terrain[t] == 3;
        out[t * 3] = free && forest && tech[9] && stars >= 3;
        out[t * 3 + 1] = free && forest && tech[16];
        out[t * 3 + 2] = free && field && tech[19] && stars >= 5;
    }
}

pub fn forest(s: &mut State, target: usize, ability: i32) {
    let terrain = if ability == 3 { 5 } else { 3 };
    // BuildAction: a `creates` terrain clears the resource (Consumed improvements destroy it), so Grow
    // Forest leaves no resource behind
    let resource = if ability == 1 { 2 } else { 0 };
    let stars = if ability == 1 { -3 } else if ability == 2 { 1 } else { -5 };
    s.tiles.terrain[target] = terrain;
    s.tiles.resource[target] = resource;
    let p = s.p();
    s.players.stars[p] += stars;
}

/// economy.discover(s, reveal, player): exploration gain for `player`; lighthouse population to the
/// preferred city of `player` (applied only if that city belongs to the player to move).
pub fn discover(s: &mut State, reveal: &[bool], player: i32) {
    let n = s.n;
    let p = (player - 1) as usize;
    let mut new = 0i32;
    let mut count = 0i32;
    for t in 0..n {
        if reveal[t] && !s.tiles.explored[p][t] {
            new += 1;
            if s.tiles.structure[t] == 47 {
                count += 1;
            }
            s.tiles.explored[p][t] = true;
        }
    }
    s.players.score[p] = s.players.score[p].wrapping_add(new * 5);
    if count == 0 {
        return; // add_population(.., 0) below changes nothing
    }
    let c = &s.cities;
    let any_capital = (0..n).any(|i| c.owner[i] == player && s.tiles.capital[i] == player);
    let any_own = (0..n).any(|i| c.owner[i] == player);
    let mut index = 0usize;
    let mut best = i32::MAX;
    for i in 0..n {
        let preferred = if any_capital { c.owner[i] == player && s.tiles.capital[i] == player } else { c.owner[i] == player };
        if preferred && c.order[i] < best {
            best = c.order[i];
            index = i;
        }
    }
    add_population(s, index, if any_own { count } else { 0 });
}

/// out[t*9 + k]
pub fn reward_mask(s: &State, out: &mut [bool]) {
    let n = s.n;
    for t in 0..n {
        let count = s.reward_count(t);
        let pending = s.cities.owner[t] == s.current_player && count < s.cities.level[t] - 1;
        let slot = count + 1;
        for k in 0..9i32 {
            let allowed = match slot {
                1 => k == 3 || k == 4,
                2 => k == 1 || k == 7,
                3 => k == 5 || k == 8,
                _ => k == 2 || k == 6,
            };
            out[t * 9 + k as usize] = pending && allowed;
        }
    }
}

pub fn claim(s: &mut State, city: usize, mask: &[bool], r: &Rules) {
    let n = s.n;
    let cp = s.current_player;
    let (mut gain, mut loss) = (0i32, [0i32; crate::state::MAXP]);
    for t in 0..n {
        if mask[t] && s.tiles.owner[t] != cp {
            let value = 20 + crate::structures::score_of(s, r, t);
            gain += value;
            if State::foreign(s.tiles.owner[t], cp) {
                loss[(s.tiles.owner[t] - 1) as usize] += value;
            }
        }
    }
    for t in 0..n {
        if mask[t] {
            s.tiles.owner[t] = cp;
            s.tiles.city[t] = city as i32;
            s.cities.territory[city * n + t] = true;
        }
    }
    let p = s.p();
    s.players.score[p] = s.players.score[p].wrapping_add(gain);
    // with two seats exactly the reference's single subtraction from the opponent (also when its loss is 0)
    for q in (0..s.np).filter(|&q| q != p) {
        s.players.score[q] = s.players.score[q].wrapping_sub(loss[q]);
    }
}

pub fn reward(s: &mut State, target: usize, kind: i32, r: &Rules, g: &Geom) {
    let n = s.n;
    s.cities.rewards[target * 9 + kind as usize] += 1;
    match kind {
        2 | 3 => {
            s.cities.production[target] += 1;
            if kind == 2 {
                let p = s.p();
                s.players.score[p] += 250;
            }
        }
        7 => {
            let p = s.p();
            s.players.stars[p] += 5;
        }
        8 => add_population(s, target, 3),
        5 => {
            let mut area = vec![false; n];
            for t in 0..n {
                area[t] = g.d(target, t) <= 2 && t != target;
            }
            s.cities.border[target] += 1;
            let claimed: Vec<bool> = (0..n).map(|t| area[t] && s.tiles.owner[t] == 0).collect();
            claim(s, target, &claimed, r);
            discover(s, &area, s.current_player);
        }
        4 => {
            let reveal = crate::discovery::explorer(s, target, g);
            discover(s, &reveal, s.current_player);
        }
        6 => crate::military::superunit(s, target, r, g),
        _ => {}
    }
}

````


## engine-rs/src/geom.rs

Source SHA-256: b991366bb031f5294c38226687a45828a5977f356b552cb280eab4b46a3a1ce9

````rust
//! Board geometry shared by every module: Chebyshev distances and the 8-neighbour table
//! (same order as gpu/kernels.py: dx=[-1,0,1,-1,1,-1,0,1], dy=[-1,-1,-1,0,0,1,1,1]).

pub const DX: [i32; 8] = [-1, 0, 1, -1, 1, -1, 0, 1];
pub const DY: [i32; 8] = [-1, -1, -1, 0, 0, 1, 1, 1];

#[derive(Clone)]
pub struct Geom {
    pub size: usize,
    pub n: usize,
    /// dist[i*n+j] Chebyshev distance.
    pub dist: Vec<u8>,
    /// nb[i*8+k] neighbour index (clipped into the board like the JAX table) and valid flag.
    pub nb: Vec<u16>,
    pub nb_valid: Vec<bool>,
    /// Per tile: list of in-board neighbour indices (compact form).
    pub adj: Vec<Vec<u16>>,
    pub corners: [usize; 4],
}

impl Geom {
    pub fn new(size: usize) -> Geom {
        let n = size * size;
        let mut dist = vec![0u8; n * n];
        for i in 0..n {
            let (xi, yi) = ((i % size) as i32, (i / size) as i32);
            for j in 0..n {
                let (xj, yj) = ((j % size) as i32, (j / size) as i32);
                dist[i * n + j] = ((xi - xj).abs().max((yi - yj).abs())) as u8;
            }
        }
        let mut nb = vec![0u16; n * 8];
        let mut nb_valid = vec![false; n * 8];
        let mut adj = vec![Vec::with_capacity(8); n];
        let s = size as i32;
        for i in 0..n {
            let (x, y) = ((i % size) as i32, (i / size) as i32);
            for k in 0..8 {
                let (nx, ny) = (x + DX[k], y + DY[k]);
                let valid = nx >= 0 && nx < s && ny >= 0 && ny < s;
                let idx = (ny * s + nx).clamp(0, (n - 1) as i32) as u16;
                nb[i * 8 + k] = idx;
                nb_valid[i * 8 + k] = valid;
                if valid {
                    adj[i].push(idx);
                }
            }
        }
        Geom { size, n, dist, nb, nb_valid, adj, corners: [0, size - 1, n - size, n - 1] }
    }

    #[inline]
    pub fn d(&self, i: usize, j: usize) -> i32 {
        self.dist[i * self.n + j] as i32
    }

    /// Calls f(t) for every tile t with d(center, t) <= radius (row-major order), visiting only that box.
    #[inline]
    pub fn for_ball(&self, center: usize, radius: i32, mut f: impl FnMut(usize)) {
        if radius < 0 {
            return;
        }
        let size = self.size as i32;
        let (cx, cy) = ((center % self.size) as i32, (center / self.size) as i32);
        for y in (cy - radius).max(0)..=(cy + radius).min(size - 1) {
            for x in (cx - radius).max(0)..=(cx + radius).min(size - 1) {
                f((y * size + x) as usize);
            }
        }
    }
}

````


## engine-rs/src/hash.rs

Source SHA-256: 6f8248bd33bced250496a628211d2adfb3e9db281c88707fc6963adb8f3c9a21

````rust
//! xxHash32 short-input path (gpu/discovery.py) and the .NET System.Random stream (gpu/random.py).

pub fn hash_words(seed: u32, words: &[u32]) -> u32 {
    let mut h = seed.wrapping_add(374761393).wrapping_add(4 * words.len() as u32);
    for &w in words {
        h = h.wrapping_add(w.wrapping_mul(3266489917));
        h = h.rotate_left(17).wrapping_mul(668265263);
    }
    h = (h ^ (h >> 15)).wrapping_mul(2246822519);
    h = (h ^ (h >> 13)).wrapping_mul(3266489917);
    h ^ (h >> 16)
}

#[inline]
pub fn hash_word(seed: u32, word: u32) -> u32 {
    hash_words(seed, &[word])
}

const MAX: i32 = 2147483647;

pub struct DotNetRandom {
    values: [i32; 56],
    current: usize,
    lag: usize,
}

impl DotNetRandom {
    pub fn new(seed: i32) -> DotNetRandom {
        let absolute = if seed == i32::MIN { MAX } else { seed.abs() };
        let first = 161803398i32.wrapping_sub(absolute);
        let mut values = [0i32; 56];
        values[55] = first;
        let (mut previous, mut next_value) = (first, 1i32);
        for i in 1..55usize {
            let at = (21 * i) % 55;
            values[at] = next_value;
            let delta = previous.wrapping_sub(next_value);
            previous = next_value;
            next_value = if delta < 0 { delta.wrapping_add(MAX) } else { delta };
        }
        for _ in 0..4 {
            for i in 1..56usize {
                let delta = values[i].wrapping_sub(values[1 + (i + 30) % 55]);
                values[i] = if delta < 0 { delta.wrapping_add(MAX) } else { delta };
            }
        }
        DotNetRandom { values, current: 0, lag: 21 }
    }

    pub fn next_int(&mut self) -> i32 {
        let i = if self.current == 55 { 1 } else { self.current + 1 };
        let j = if self.lag == 55 { 1 } else { self.lag + 1 };
        let mut value = self.values[i].wrapping_sub(self.values[j]);
        if value == MAX {
            value -= 1;
        }
        if value < 0 {
            value = value.wrapping_add(MAX);
        }
        self.values[i] = value;
        self.current = i;
        self.lag = j;
        value
    }

    /// Exact Next(0, upper) with the reference's 32-bit limb arithmetic.
    pub fn bounded(&mut self, upper: i32) -> i32 {
        let a = self.next_int() as u32;
        let b = upper as u32;
        let lower = (a & 65535).wrapping_mul(b);
        let middle = (a >> 16).wrapping_mul(b).wrapping_add(lower >> 16);
        let high = middle >> 16;
        let low = (middle << 16) | (lower & 65535);
        let mut quotient = (high << 1) | (low >> 31);
        let remainder = (low & 2147483647).wrapping_add(quotient);
        quotient = quotient.wrapping_add((remainder >= 2147483647) as u32);
        quotient as i32
    }

    /// Float32 cast of NextDouble, including the upward-shifted halfway cases.
    pub fn unit_float(&mut self) -> f32 {
        let value = self.next_int();
        let bits = value as u32;
        let shift = (8 - bits.leading_zeros() as i32).max(0) as u32;
        let quantum = 1u32 << shift;
        let tie = shift > 0 && (bits & (quantum - 1)) == (quantum >> 1) && ((bits / quantum) & 1) == 0;
        let result = (value as f32) * (1.0f32 / 2147483648.0f32);
        if tie {
            f32::from_bits(result.to_bits() + 1)
        } else {
            result
        }
    }
}

````


## engine-rs/src/infiltration.rs

Source SHA-256: 218af83367b65a673064aa21a75946ab63442505b4a9d8a096479aff221e6d8a

````rust
//! gpu/infiltration.py: native Cloak infiltration.

use crate::geom::Geom;
use crate::hash::{hash_word, hash_words};
use crate::kernels::{combat, max_health};
use crate::military::{defense_bonus, remove, summon};
use crate::rules::Rules;
use crate::state::State;

/// Spawn tiles for `target`'s city in priority order (eligible only), with the reference's seeded tie noise.
pub fn spawn_order(s: &State, target: usize, g: &Geom) -> Vec<usize> {
    let (n, size) = (s.n, s.size);
    let t = &s.tiles;
    let tech = &s.players.tech[s.p()];
    let mut keyed: Vec<(f32, usize)> = Vec::new();
    for i in 0..n {
        let water = t.terrain[i] <= 2;
        let mut eligible = t.city[i] == target as i32 && s.units.owner[i] == 0;
        eligible &= (t.terrain[i] != 4 || tech[20]) && (t.terrain[i] != 1 || tech[10]) && (t.terrain[i] != 2 || tech[13]);
        if !eligible {
            continue;
        }
        let defense = (t.terrain[i] == 4 && tech[20]) || (t.terrain[i] == 5 && tech[18]) || (water && tech[12]);
        let known_resource = t.resource[i] > 0 && (t.resource[i] != 2 || tech[6] || tech[8] || tech[9])
            && (t.resource[i] != 5 || tech[20] || tech[23] || tech[24]);
        let words = [(i % size) as u32, (i / size) as u32, s.turn as u32];
        let noise = (hash_word(s.initial_seed, hash_words(s.initial_seed, &words)) as f32) * 0.1f32 / 4294967296.0f32;
        let priority = noise + (t.structure[i] == 1) as i32 as f32 * 2.0 - water as i32 as f32 * 1.6 + known_resource as i32 as f32 * 0.5
            + defense as i32 as f32;
        keyed.push((-priority, i));
    }
    // stable argsort on the float key, ties by index
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let _ = g;
    keyed.into_iter().map(|(_, i)| i).collect()
}

pub fn execute(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom) {
    let owner = s.current_player;
    let po = s.p();
    let c = &s.cities;
    let mut innate = c.production[target] + (s.tiles.capital[target] == c.owner[target]) as i32;
    innate = (innate + c.progress[target].min(0)).max(0);
    let count = c.level[target].min(5) as usize;
    remove(s, source, r, None, true);
    let enemy = s.units.owner[target] > 0 && s.units.owner[target] != owner;
    let tk = s.units.kind[target] as usize;
    let damage = combat(r.unit_attack[37], r.unit_health[37], r.unit_health[37], r.unit_defense[tk], s.units.health[target],
                        max_health(s, r, target), defense_bonus(s, r, target)).0;
    s.units.health[target] += if enemy { -damage } else { 0.0 };
    let dead = enemy && s.units.health[target] <= 0.0;
    if dead {
        remove(s, target, r, None, true);
    }
    s.players.kills[po] += dead as i32;
    s.players.stars[po] += innate;
    s.cities.infiltrated[target] = s.turn * s.np as i32 + owner;
    s.cities.riot[target] = true;
    let order = spawn_order(s, target, g);
    for (i, &at) in order.iter().enumerate() {
        if i >= count.min(5) {
            break;
        }
        let water = s.tiles.terrain[at] <= 2;
        summon(s, at, if water { 40 } else { 37 }, r, g, false);
        s.units.home[at] = 0;
    }
}

````


## engine-rs/src/kernels.rs

Source SHA-256: e0adfa629b1a76c8758133f02d207e1cff0484a40d77f70b65cf5fba29439cbf

````rust
//! gpu/kernels.py: combat formula, unit health, research, movement reachability, exploration and the observation.

use crate::geom::Geom;
use crate::rules::{Rules, CARRY, CREEP, FLOAT, HIDE, INFILTRATE};
use crate::state::State;

pub const INF: i32 = 1_000_000;

/// Reference f32 damage formula with Rust's half-away-from-zero rounding; (attack, defense, attack*0.5).
#[inline]
pub fn combat(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    let divisor = if total > 0.0 { total } else { 1.0 };
    let attack = ((if total > 0.0 { (af / divisor) * aa * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    let defense = ((if dh > attack && total > 0.0 { (df / divisor) * dd * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    (attack, defense, attack * 0.5)
}

#[inline]
pub fn max_health(s: &State, r: &Rules, i: usize) -> f32 {
    let u = &s.units;
    let kind = if u.passenger[i] > 0 { u.passenger[i] } else { u.kind[i] };
    r.unit_health[(kind as usize).min(127)] + (u.veteran[i] as i32 * 5) as f32
}

pub fn research_cost(s: &State, r: &Rules, k: usize) -> i32 {
    let cp = s.current_player;
    let cities = s.cities.owner.iter().filter(|&&o| o == cp).count() as i32;
    research_cost_with(s, r, k, cities)
}

/// research_cost given the mover's city count.
#[inline]
pub fn research_cost_with(s: &State, r: &Rules, k: usize, cities: i32) -> i32 {
    let cost = 4 + cities * r.tech_tier[k];
    // GameLogicData.GetTechPrice: Literacy multiplies by 0.66666f
    if s.players.tech[s.p()][22] { ((cost as f32) * 0.66666f32).ceil() as i32 } else { cost }
}

pub fn research_mask(s: &State, r: &Rules, out: &mut [bool]) {
    let active = !s.pending_reward() && !s.terminated && !s.truncated;
    let bits = if active { research_bits(s, r) } else { 0 };
    for k in 0..128 {
        out[k] = bits >> k & 1 == 1;
    }
}

/// research_mask without its `active` gate, as a bitset: techs reachable from a discovered one, not discovered,
/// existing and affordable (one city count, predecessor bitsets).
pub fn research_bits(s: &State, r: &Rules) -> u128 {
    let p = s.p();
    let discovered = &s.players.tech[p];
    let stars = s.players.stars[p];
    let known = crate::rules::bits128(discovered);
    let mut cand = 0u128;
    let mut rest = known;
    while rest != 0 {
        cand |= r.tech_succ[rest.trailing_zeros() as usize];
        rest &= rest - 1;
    }
    cand &= !known & r.tech_exists_bits;
    if cand == 0 {
        return 0;
    }
    let cp = s.current_player;
    let cities = s.cities.owner.iter().filter(|&&o| o == cp).count() as i32;
    let mut out = 0u128;
    while cand != 0 {
        let k = cand.trailing_zeros() as usize;
        cand &= cand - 1;
        out |= ((research_cost_with(s, r, k, cities) <= stars) as u128) << k;
    }
    out
}

pub struct MovementData {
    pub passable: Vec<bool>,
    pub terminal: Vec<bool>,
    pub road: Vec<bool>,
    pub enemy: Vec<bool>,
    pub budget: i32,
}

impl MovementData {
    pub fn new(n: usize) -> MovementData {
        MovementData { passable: vec![false; n], terminal: vec![false; n], road: vec![false; n], enemy: vec![false; n], budget: 0 }
    }
}

pub fn movement_data(s: &State, source: usize, r: &Rules, g: &Geom) -> MovementData {
    let mut md = MovementData::new(s.n);
    movement_data_into(s, source, r, g, &mut md);
    md
}

/// kernels.movement_data into reusable buffers.
pub fn movement_data_into(s: &State, source: usize, r: &Rules, g: &Geom, md: &mut MovementData) {
    let n = s.n;
    let (t, u) = (&s.tiles, &s.units);
    let owner = u.owner[source];
    let p = s.pclip(owner);
    let tech = &s.players.tech[p];
    let kind = u.kind[source] as usize;
    let aquatic = r.skills[kind][FLOAT];
    let creep = r.skills[kind][CREEP];
    let (passable, enemy, road, terminal) = (&mut md.passable, &mut md.enemy, &mut md.road, &mut md.terminal);
    for i in 0..n {
        let bridge = t.structure[i] == 48;
        let water = t.terrain[i] == 1 || t.terrain[i] == 2;
        let terrain_ok = ((t.terrain[i] != 1 || tech[10]) && (t.terrain[i] != 2 || tech[13] || tech[25]) && (t.terrain[i] != 4 || tech[20])) || bridge;
        let own_port = t.structure[i] == 8 && t.owner[i] == owner;
        let land_access = !water || bridge || own_port;
        let hidden_enemy = u.invisible[i] && u.owner[i] != owner;
        passable[i] = t.explored[p][i] && terrain_ok && (aquatic || land_access) && (u.owner[i] == 0 || u.owner[i] == owner || hidden_enemy);
        let peaceful = s.players.peace[p][s.pclip(u.owner[i])];
        enemy[i] = u.owner[i] != 0 && u.owner[i] != owner && !u.invisible[i] && !peaceful;
        let friendly_road = t.owner[i] == 0 || t.owner[i] == owner || s.players.peace[p][s.pclip(t.owner[i])];
        road[i] = (t.road[i] || t.structure[i] == 1 || s.cities.owner[i] != 0 || bridge) && friendly_road;
    }
    let no_zoc = creep || r.skills[kind][INFILTRATE] || u.invisible[source];
    for i in 0..n {
        let zoc = !no_zoc && g.adj[i].iter().any(|&j| enemy[j as usize]);
        let water = t.terrain[i] == 1 || t.terrain[i] == 2;
        let own_port = t.structure[i] == 8 && t.owner[i] == owner;
        terminal[i] = zoc || t.terrain[i] == 4 || (t.terrain[i] == 5 && !road[i] && !creep) || (own_port && !aquatic) || (r.skills[kind][CARRY] && !water);
    }
    md.budget = (r.unit_movement[kind] + u.boosted[source] as i32) * 2;
}

// Unit-independent movement terms of one owner, one byte per tile (movement_flags).
pub const MV_PASS: u8 = 1; // explored, terrain allowed by tech (or bridge), no blocking unit
pub const MV_LAND: u8 = 2; // land access: not water, or a bridge, or an own port
pub const MV_ROAD: u8 = 4; // friendly road network tile
pub const MV_ZOC: u8 = 8; // next to a visible, non-peaceful, explored enemy unit
pub const MV_MOUNTAIN: u8 = 16;
pub const MV_FOREST: u8 = 32;
pub const MV_OWN_PORT: u8 = 64;
pub const MV_WATER: u8 = 128;

/// kernels.movement_data's per-tile terms for units of `owner`, which only the unit's skills (float, creep,
/// carry, zoc immunity) and budget specialise: passable = PASS && (float || LAND); terminal = (ZOC && !no_zoc)
/// || MOUNTAIN || (FOREST && !ROAD && !creep) || (OWN_PORT && !float) || (carry && !WATER).
pub fn movement_flags(s: &State, owner: i32, g: &Geom, flags: &mut [u8]) {
    let n = s.n;
    let (t, u) = (&s.tiles, &s.units);
    let p = s.pclip(owner);
    let tech = &s.players.tech[p];
    let peace = &s.players.peace[p];
    let explored = &t.explored[p];
    for i in 0..n {
        let (terrain, structure) = (t.terrain[i], t.structure[i]);
        let bridge = structure == 48;
        let water = terrain == 1 || terrain == 2;
        let terrain_ok = ((terrain != 1 || tech[10]) && (terrain != 2 || tech[13] || tech[25]) && (terrain != 4 || tech[20])) || bridge;
        let own_port = structure == 8 && t.owner[i] == owner;
        let uo = u.owner[i];
        let hidden_enemy = u.invisible[i] && uo != owner;
        let pass = explored[i] && terrain_ok && (uo == 0 || uo == owner || hidden_enemy);
        let friendly_road = t.owner[i] == 0 || t.owner[i] == owner || peace[s.pclip(t.owner[i])];
        let road = (t.road[i] || structure == 1 || s.cities.owner[i] != 0 || bridge) && friendly_road;
        flags[i] = (pass as u8) * MV_PASS | ((!water || bridge || own_port) as u8) * MV_LAND | (road as u8) * MV_ROAD
            | ((terrain == 4) as u8) * MV_MOUNTAIN | ((terrain == 5) as u8) * MV_FOREST | (own_port as u8) * MV_OWN_PORT
            | (water as u8) * MV_WATER;
    }
    for i in 0..n {
        let uo = u.owner[i];
        // TileData.GetMovementCost reads the raw neighbours: unseen enemies exert zone of control too
        let enemy = uo != 0 && uo != owner && !u.invisible[i] && !peace[s.pclip(uo)];
        if enemy {
            for &j in &g.adj[i] {
                flags[j as usize] |= MV_ZOC;
            }
        }
    }
}

/// PathFinder.IsTileAccessible: a Hide unit may pass a tile held by another player's unit (it never
/// ends there). True when tile `i` stops `owner`'s units only because of such a unit: explored, terrain allowed.
#[inline]
pub fn hide_passable(s: &State, owner: i32, i: usize) -> bool {
    let (t, u) = (&s.tiles, &s.units);
    let tech = &s.players.tech[s.pclip(owner)];
    let terrain = t.terrain[i];
    let terrain_ok = ((terrain != 1 || tech[10]) && (terrain != 2 || tech[13] || tech[25]) && (terrain != 4 || tech[20]))
        || t.structure[i] == 48;
    u.owner[i] != 0 && u.owner[i] != owner && t.explored[s.pclip(owner)][i] && terrain_ok
}

/// PathFinder.IsTileAccessible: any unit may pass a tile held by a peace partner's unit.
#[inline]
pub fn peace_unit(s: &State, owner: i32, i: usize) -> bool {
    let o = s.units.owner[i];
    o != 0 && o != owner && s.players.peace[s.pclip(owner)][s.pclip(o)]
}

/// TileData.GetMovementCost: entering a forest ends the move unless the step runs road to road. `cand` is the step's cost so far.
#[inline]
pub fn forest_step(creep: bool, road_edge: bool, dest_flags: u8, cand: i32, budget: i32) -> i32 {
    if !creep && !road_edge && dest_flags & MV_FOREST != 0 { cand.max(budget) } else { cand }
}

/// PathFinder.IsTileAccessible: without water movement (Fishing), a bridge cannot be entered across its
/// span: not straight from the east or west onto a north-south bridge, nor straight from north or south onto an
/// east-west one (diagonal steps are allowed).
#[inline]
pub fn bridge_blocks(s: &State, owner: i32, from: usize, to: usize) -> bool {
    if s.tiles.structure[to] != 48 || s.players.tech[s.pclip(owner)][10] {
        return false;
    }
    let size = s.size;
    let (dx, dy) = ((to % size) as i32 - (from % size) as i32, (to / size) as i32 - (from / size) as i32);
    match crate::structures::bridge_span(s, to) {
        1 => dy == 0,
        2 => dx == 0,
        _ => false,
    }
}

/// Reusable buffers for movement_targets.
pub struct MoveScratch {
    pub flags: Vec<u8>,
    pub dist: Vec<i32>,
    stamp: Vec<u32>,
    epoch: u32,
    frontier: Vec<(u16, i32)>,
    next: Vec<u16>,
    touched: Vec<u16>,
}

impl MoveScratch {
    pub fn new(n: usize) -> MoveScratch {
        MoveScratch { flags: vec![0; n], dist: vec![INF; n], stamp: vec![0; n], epoch: 0, frontier: Vec::with_capacity(n),
                      next: Vec::with_capacity(n), touched: Vec::with_capacity(n) }
    }
}

/// Destination tiles of the unit on `source` (kernels.movement_targets without its `active` gate), given
/// `ms.flags` = movement_flags of the unit's owner. kernels.movement_targets runs 8 synchronous relaxation sweeps
/// over every tile; this runs the same 8 synchronous rounds (Jacobi: every relaxation in round k reads the
/// distances of round k-1) over the frontier only, i.e. the tiles whose distance changed in the previous round,
/// which is exact: a tile that did not change contributes nothing new. Calls `emit(tile)` per destination.
pub fn reach(s: &State, source: usize, r: &Rules, g: &Geom, ms: &mut MoveScratch, mut emit: impl FnMut(usize)) {
    let u = &s.units;
    let owner = u.owner[source];
    let kind = u.kind[source] as usize;
    let sk = &r.skills[kind];
    let (aquatic, creep, carry) = (sk[FLOAT], sk[CREEP], sk[CARRY]);
    let no_zoc = creep || sk[INFILTRATE] || u.invisible[source];
    let budget = (r.unit_movement[kind] + u.boosted[source] as i32) * 2;
    let hide = sk[HIDE];
    // terminal: any of these bits set (plus FOREST without ROAD for non-creepers, handled below)
    let term_bits = (if no_zoc { 0 } else { MV_ZOC }) | MV_MOUNTAIN | (if aquatic { 0 } else { MV_OWN_PORT });
    let pass_need = if aquatic { MV_PASS } else { MV_PASS | MV_LAND };
    let MoveScratch { flags, dist, stamp, epoch, frontier, next, touched } = ms;
    let terminal = |f: u8| f & term_bits != 0 || (f & (MV_FOREST | MV_ROAD) == MV_FOREST && !creep) || (carry && f & MV_WATER == 0);
    touched.clear();
    frontier.clear();
    dist[source] = 0;
    touched.push(source as u16);
    frontier.push((source as u16, 0));
    for _ in 0..8 {
        if frontier.is_empty() {
            break;
        }
        *epoch = epoch.wrapping_add(1);
        if *epoch == 0 {
            stamp.fill(0);
            *epoch = 1;
        }
        next.clear();
        for &(j, dj) in frontier.iter() {
            let j = j as usize;
            let fj = flags[j];
            if dj >= budget || (j != source && terminal(fj)) {
                continue;
            }
            let road_j = fj & MV_ROAD != 0 && !creep;
            for &i in &g.adj[j] {
                let i = i as usize;
                let fi = flags[i];
                if fi & pass_need != pass_need && i != source
                    && !((hide || peace_unit(s, owner, i)) && (aquatic || fi & MV_LAND != 0) && hide_passable(s, owner, i)) {
                    continue;
                }
                if bridge_blocks(s, owner, j, i) {
                    continue;
                }
                let road_edge = road_j && fi & MV_ROAD != 0;
                let cand = forest_step(creep, road_edge, fi, dj + if road_edge { 1 } else { 2 }, budget);
                if cand < dist[i] {
                    if dist[i] == INF {
                        touched.push(i as u16);
                    }
                    dist[i] = cand;
                    if stamp[i] != *epoch {
                        stamp[i] = *epoch;
                        next.push(i as u16);
                    }
                }
            }
        }
        frontier.clear();
        frontier.extend(next.iter().map(|&i| (i, dist[i as usize])));
    }
    // PathFinder.IsAllowedToFinishOnTile: an Infiltrate unit never ends on an opponent's city, and no
    // unit ends on the city of a player it has peace or a cool-down with
    let po = s.pclip(owner);
    let finish_ok = |i: usize| {
        let co = s.cities.owner[i];
        co == 0 || co == owner
            || (!sk[INFILTRATE] && !s.players.peace[po][s.pclip(co)] && s.players.truce_breaker[po][s.pclip(co)] == 0)
    };
    for &i in touched.iter() {
        let i = i as usize;
        dist[i] = INF;
        let empty = u.owner[i] == 0 || (u.invisible[i] && u.owner[i] != owner);
        if empty && i != source && finish_ok(i) {
            emit(i);
        }
    }
}

/// Destination set of the unit on `source` (kernels.movement_targets).
pub fn movement_targets(s: &State, source: usize, r: &Rules, g: &Geom, ms: &mut MoveScratch, out: &mut [bool]) {
    let n = s.n;
    let u = &s.units;
    let kind = u.kind[source] as usize;
    out[..n].fill(false);
    let active = u.owner[source] == s.current_player && !u.moved[source] && r.unit_supported[kind] && !s.pending_reward()
        && !s.terminated && !s.truncated;
    if !active {
        return;
    }
    movement_flags(s, u.owner[source], g, &mut ms.flags);
    reach(s, source, r, g, ms, |i| out[i] = true);
}

pub fn update_exploration(s: &mut State, player: i32, g: &Geom) {
    let n = s.n;
    let p = (player - 1) as usize;
    let mut explored = vec![false; n];
    for t in 0..n {
        explored[t] = s.tiles.explored[p][t] || !s.fog;
    }
    for c in 0..n {
        if s.cities.owner[c] == player {
            let range = if s.tiles.capital[c] > 0 { 2 } else { s.cities.border[c] };
            g.for_ball(c, range, |t| {
                let corner = g.corners.contains(&t);
                if !corner || s.cities.owner[t] == player {
                    explored[t] = true;
                }
            });
        }
    }
    for i in 0..n {
        if s.units.owner[i] == player {
            let k = s.units.kind[i];
            let range = if s.tiles.terrain[i] == 4 || k == 42 || k == 38 || k == 39 { 2 } else { 1 };
            g.for_ball(i, range, |t| explored[t] = true);
        }
    }
    let mut gain = 0;
    for t in 0..n {
        if explored[t] && !s.tiles.explored[p][t] {
            gain += 5;
        }
        s.tiles.explored[p][t] = explored[t];
    }
    s.players.score[p] = s.players.score[p].wrapping_add(gain);
}

/// Packed public view: kernels.observe(state, player) fields in sorted-key order, int16 (cpu_engine.pack_view).
/// Games with more than two seats add "alive" (first in sorted order) and give the five relation fields one entry
/// per seat (view_fields / view_width).
pub const VIEW_FIELDS: [&str; 40] = [
    "capital", "detect", "embassies_received", "embassies_sent", "explored", "known_players", "own_built",
    "own_city_connected", "own_city_level", "own_city_population", "own_city_production", "own_city_progress",
    "own_city_rewards", "own_city_riot", "own_monuments", "own_unit_attacked", "own_unit_boosted", "own_unit_frozen",
    "own_unit_home", "own_unit_invisible", "own_unit_kills", "own_unit_moved", "own_unit_passenger", "own_unit_poison",
    "owner", "peace", "peace_offers", "player", "resource", "road", "stars", "structure", "structure_level", "tech",
    "terrain", "turn", "unit_health", "unit_owner", "unit_type", "unit_veteran",
];

/// Field names of the packed view of a game with `np` seats, in packing order.
pub fn view_fields(np: usize) -> Vec<&'static str> {
    let mut f = VIEW_FIELDS.to_vec();
    if np > 2 {
        f.insert(0, "alive");
    }
    f
}

pub fn view_width(n: usize) -> usize {
    37 * n + 397
}

/// Packed view width for `np` seats: relations take np entries each, plus "alive" when np > 2.
pub fn view_width_np(n: usize, np: usize) -> usize {
    view_width(n) + if np > 2 { 6 * np - 10 } else { 0 }
}

/// Largest board the observation's stack buffers hold (18x18 is the largest map size).
const MAX_TILES: usize = 512;

#[inline(always)]
fn put_sel(dst: &mut [i16], mask: &[bool], v: &[i32], other: i16) {
    for ((d, &m), &x) in dst.iter_mut().zip(mask).zip(v) {
        *d = if m { x as i16 } else { other };
    }
}

#[inline(always)]
fn put_bool(dst: &mut [i16], v: &[bool]) {
    for (d, &x) in dst.iter_mut().zip(v) {
        *d = x as i16;
    }
}

/// visible ? (unit visible ? v : 0) : -1
#[inline(always)]
fn put_unit(dst: &mut [i16], vis: &[bool], uvis: &[bool], v: &[i32]) {
    for (((d, &a), &b), &x) in dst.iter_mut().zip(vis).zip(uvis).zip(v) {
        *d = if a { if b { x as i16 } else { 0 } } else { -1 };
    }
}

/// kernels.observe packed in VIEW_FIELDS order: one tight loop per field over precomputed visibility and
/// ownership masks. The same code is compiled twice and picked at run time: with AVX2 (the i32 -> i16 selects
/// vectorise) when the CPU has it, else for the baseline target. Integer-only, so both give identical bits.
pub fn observe(s: &State, player: i32, g: &Geom, out: &mut [i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2 (checked just above).
            return unsafe { observe_avx2(s, player, g, out) };
        }
    }
    observe_impl(s, player, g, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn observe_avx2(s: &State, player: i32, g: &Geom, out: &mut [i16]) {
    observe_impl(s, player, g, out)
}

#[inline(always)]
fn observe_impl(s: &State, player: i32, g: &Geom, out: &mut [i16]) {
    let n = s.n;
    let np = s.np;
    assert!(n <= MAX_TILES && out.len() >= view_width_np(n, np));
    let p = (player - 1) as usize;
    let (t, u, c, pl) = (&s.tiles, &s.units, &s.cities, &s.players);
    let tech = &pl.tech[p];
    let (mut vis, mut uvis, mut unit) = ([false; MAX_TILES], [false; MAX_TILES], [false; MAX_TILES]);
    // the seat's cities and units: their fields are zero everywhere else
    let (mut cities, mut units) = ([0u16; MAX_TILES], [0u16; MAX_TILES]);
    let (mut nc, mut nu) = (0usize, 0usize);
    for i in 0..n {
        let v = t.explored[p][i] || !s.fog;
        vis[i] = v;
        unit[i] = u.owner[i] == player;
        uvis[i] = v && (unit[i] || !u.invisible[i]);
        cities[nc] = i as u16;
        nc += (c.owner[i] == player) as usize;
        units[nu] = i as u16;
        nu += unit[i] as usize;
    }
    let (vis, uvis, unit, cities, units) = (&vis[..n], &uvis[..n], &unit[..n], &cities[..nc], &units[..nu]);
    let mut at = 0usize;
    macro_rules! field {
        ($len:expr) => {{
            let f = &mut out[at..at + $len];
            at += $len;
            f
        }};
    }
    macro_rules! own {
        ($list:expr, $v:expr) => {{
            let f = field!(n);
            f.fill(0);
            for &i in $list {
                f[i as usize] = $v[i as usize] as i16;
            }
        }};
    }
    macro_rules! small {
        ($($v:expr),*) => {{ $( out[at] = $v as i16; at += 1; )* }};
    }
    if np > 2 {
        for k in 0..np {
            small!(s.alive(k));
        }
    }
    put_sel(field!(n), vis, &t.capital, -1);
    let detect = field!(n);
    detect.fill(0);
    for i in 0..n {
        if unit[i] {
            detect[i] = g.adj[i].iter().any(|&j| {
                let j = j as usize;
                u.owner[j] != 0 && u.owner[j] != player && u.invisible[j]
            }) as i16;
        }
    }
    for k in 0..np {
        small!(pl.embassy[k][p]);
    }
    for k in 0..np {
        small!(pl.embassy[p][k]);
    }
    put_bool(field!(n), vis);
    for k in 0..np {
        small!(pl.known[p][k]);
    }
    put_bool(field!(128), &pl.built[p]);
    own!(cities, c.connected);
    own!(cities, c.level);
    own!(cities, c.population);
    own!(cities, c.production);
    own!(cities, c.progress);
    let rewards = field!(n * 9);
    rewards.fill(0);
    for &i in cities {
        let i = i as usize;
        for k in 0..9 {
            rewards[i * 9 + k] = c.rewards[i * 9 + k] as i16;
        }
    }
    own!(cities, c.riot);
    put_bool(field!(128), &pl.monuments[p]);
    own!(units, u.attacked);
    own!(units, u.boosted);
    own!(units, u.frozen);
    own!(units, u.home);
    own!(units, u.invisible);
    own!(units, u.kills);
    own!(units, u.moved);
    own!(units, u.passenger);
    own!(units, u.poison);
    put_sel(field!(n), vis, &t.owner, -1);
    for k in 0..np {
        small!(pl.peace[p][k]);
    }
    for k in 0..np {
        small!(pl.peace_offer[p][k]);
    }
    small!(player);
    let hide2 = !(tech[6] || tech[8] || tech[9]);
    let hide5 = !(tech[20] || tech[23] || tech[24]);
    for ((d, &m), &r) in field!(n).iter_mut().zip(vis).zip(&t.resource) {
        let hidden = (r == 2 && hide2) || (r == 5 && hide5);
        *d = if m { if hidden { 0 } else { r as i16 } } else { -1 };
    }
    for ((d, &m), &r) in field!(n).iter_mut().zip(vis).zip(&t.road) {
        *d = if m { r as i16 } else { -1 };
    }
    small!(pl.stars[p]);
    put_sel(field!(n), vis, &t.structure, -1);
    put_sel(field!(n), vis, &t.structure_level, -1);
    put_bool(field!(128), tech);
    put_sel(field!(n), vis, &t.terrain, -1);
    small!(s.turn);
    // unit_health is float32 in the view; the pack casts it to int16 (toward zero).
    for (((d, &a), &b), &h) in field!(n).iter_mut().zip(vis).zip(uvis).zip(&u.health) {
        let v: f32 = if a { if b { h } else { 0.0 } } else { -1.0 };
        *d = v as i16;
    }
    put_unit(field!(n), vis, uvis, &u.owner);
    put_unit(field!(n), vis, uvis, &u.kind);
    for (((d, &a), &b), &x) in field!(n).iter_mut().zip(vis).zip(uvis).zip(&u.veteran) {
        *d = if a { if b { x as i16 } else { 0 } } else { -1 };
    }
    debug_assert_eq!(at, view_width_np(n, np));
}

````


## engine-rs/src/lib.rs

Source SHA-256: 66ba467914cf8136c57154a98a754ab1f569e084f41909ccbe6c5d7c986a92b5

````rust
//! PyO3 bindings: a batch of independent games mirroring rl/cpu_engine.py::CpuGames.

pub mod capture;
pub mod core;
pub mod cov;
pub mod diplomacy;
pub mod discovery;
pub mod economy;
pub mod geom;
pub mod hash;
pub mod infiltration;
pub mod kernels;
pub mod mapgen;
pub mod masks;
pub mod military;
pub mod pathfinding;
pub mod prof;
pub mod rules;
pub mod state;
pub mod structures;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::core::{decode, step, step_with, terminal_rewards, Scratch};
use crate::geom::Geom;
use crate::kernels::{observe, view_fields, view_width_np};
use crate::masks::{legal_masks, mask_width, masks_into, packed_width, Dense, LegalCache, MaskScratch, Packed, Tee};
use crate::rules::Rules;
use crate::state::State;

#[pyclass]
pub struct GameBatch {
    size: usize,
    games: usize,
    /// seats per game (2 or 4); per-seat outputs (rewards, final views, capitals) have this many rows
    players: usize,
    rules: Rules,
    geom: Geom,
    states: Vec<State>,
    max_turns: Option<i32>,
    no_resign: bool,
    max_actions: i32,
    scratch: Scratch,
    mscratch: MaskScratch,
    /// per game: the masks tick() last returned, while they describe the game's state (step validation)
    legal: Vec<LegalCache>,
}

fn err(msg: &str) -> PyErr {
    PyValueError::new_err(msg.to_string())
}

#[pymethods]
impl GameBatch {
    #[new]
    #[pyo3(signature = (size, games, rules_json, engine_revision=None, max_turns=None, no_resign=true, max_actions=10000, players=2))]
    fn new(size: usize, games: usize, rules_json: &str, engine_revision: Option<&str>, max_turns: Option<i32>, no_resign: bool,
           max_actions: i32, players: usize) -> PyResult<Self> {
        if mapgen::ruins(size).is_none() {
            return Err(err("board size must be 11, 14 or 18"));
        }
        if !(2..=crate::state::MAXP).contains(&players) {
            return Err(err("players must be 2, 3 or 4"));
        }
        let rules = Rules::from_json(rules_json).map_err(|e| err(&e))?;
        if let Some(rev) = engine_revision {
            if rules.engine_revision != rev {
                return Err(err("rule table and pinned engine revision differ"));
            }
        }
        let geom = Geom::new(size);
        let states = (0..games).map(|_| State::empty_players(size, players)).collect();
        Ok(GameBatch { size, games, players, rules, geom, states, max_turns, no_resign, max_actions, scratch: Scratch::new(size * size),
                     mscratch: MaskScratch::new(size * size), legal: (0..games).map(|_| LegalCache::new(size * size)).collect() })
    }

    #[getter]
    fn size(&self) -> usize { self.size }
    #[getter]
    fn games(&self) -> usize { self.games }
    #[getter]
    fn players(&self) -> usize { self.players }
    #[getter]
    fn view_width(&self) -> usize { view_width_np(self.size * self.size, self.players) }
    #[getter]
    fn mask_width(&self) -> usize { mask_width(self.size * self.size) }
    fn view_fields(&self) -> Vec<&'static str> { view_fields(self.players) }
    fn family_widths(&self) -> Vec<usize> { core::family_widths(self.size * self.size).to_vec() }

    /// Fresh games from mapgen.reset(seed) for every lane.
    fn reset(&mut self, seeds: PyReadonlyArray1<i32>) -> PyResult<()> {
        let seeds = seeds.as_slice()?;
        if seeds.len() != self.games {
            return Err(err("seeds must have one entry per game"));
        }
        for (i, &seed) in seeds.iter().enumerate() {
            self.states[i] = mapgen::reset_players(seed, &self.geom, self.max_turns, self.players);
        }
        self.forget_masks();
        Ok(())
    }

    /// Fresh games from explicit layouts (maps.state.from_batch_row): terrain/structure/resource [G, n], capitals [G, players].
    fn reset_layouts(&mut self, seeds: PyReadonlyArray1<i32>, terrain: PyReadonlyArray2<i32>, structure: PyReadonlyArray2<i32>,
                     resource: PyReadonlyArray2<i32>, capitals: PyReadonlyArray2<i32>) -> PyResult<()> {
        let n = self.size * self.size;
        let seeds = seeds.as_slice()?;
        let (t, st, rs, caps) = (terrain.as_array(), structure.as_array(), resource.as_array(), capitals.as_array());
        if seeds.len() != self.games || t.shape() != [self.games, n] || st.shape() != [self.games, n] || rs.shape() != [self.games, n]
            || caps.shape() != [self.games, self.players] {
            return Err(err("layout arrays must be [games, n] and capitals [games, players]"));
        }
        for g in 0..self.games {
            let tv: Vec<i32> = t.row(g).to_vec();
            let sv: Vec<i32> = st.row(g).to_vec();
            let rv: Vec<i32> = rs.row(g).to_vec();
            let cap: Vec<usize> = caps.row(g).iter().map(|&c| c as usize).collect();
            self.states[g] = mapgen::initial_state(seeds[g] as u32, &tv, &sv, &rv, &cap, &self.geom,
                                                   self.max_turns.unwrap_or(mapgen::round_limit(self.size)));
        }
        self.forget_masks();
        Ok(())
    }

    /// Only the mapgen layout of one seed (terrain, structure, resource, capitals), for map checks.
    fn layout<'py>(&self, py: Python<'py>, seed: i32) -> PyResult<Bound<'py, PyDict>> {
        let l = mapgen::layout_players(seed, &self.geom, self.players);
        let d = PyDict::new(py);
        d.set_item("terrain", l.terrain.into_pyarray(py))?;
        d.set_item("structure", l.structure.into_pyarray(py))?;
        d.set_item("resource", l.resource.into_pyarray(py))?;
        d.set_item("capitals", l.capitals.iter().map(|&c| c as i32).collect::<Vec<i32>>().into_pyarray(py))?;
        Ok(d)
    }

    /// Apply one decoded choice per game (no auto-reset, no prepare). Returns status [G].
    fn step<'py>(&mut self, py: Python<'py>, family: PyReadonlyArray1<i32>, source: PyReadonlyArray1<i32>,
                 target: PyReadonlyArray1<i32>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let (f, s, t) = (family.as_slice()?, source.as_slice()?, target.as_slice()?);
        let mut status = vec![0i32; self.games];
        for g in 0..self.games {
            let st = &mut self.states[g];
            let a = decode(st, f[g], s[g], t[g]);
            let v = st.state_version;
            status[g] = step(st, a, v, &self.rules, &self.geom, self.max_actions, &mut self.scratch);
        }
        self.forget_masks();
        Ok(status.into_pyarray(py))
    }

    /// One actor tick: step, auto-reset finished games from `seeds`, then masks and packed views.
    /// packed_masks: "masks" holds [G, ceil(mask_width/8)] bits (numpy.packbits order) instead of [G, mask_width] bytes.
    /// Per-seat outputs: rewards [G, players], final_view [G, players, vw]. both_views: "opp_view" is [G, vw] (the seat
    /// not to move) with two seats, else [G, players - 1, vw]: the other seats in turn order after the mover.
    #[pyo3(signature = (family, source, target, seeds, both_views=true, layouts=None, packed_masks=false))]
    fn tick<'py>(&mut self, py: Python<'py>, family: PyReadonlyArray1<i32>, source: PyReadonlyArray1<i32>,
                 target: PyReadonlyArray1<i32>, seeds: PyReadonlyArray1<i32>, both_views: bool,
                 layouts: Option<(PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>)>,
                 packed_masks: bool)
                 -> PyResult<Bound<'py, PyDict>> {
        let n = self.size * self.size;
        let (f, s, t, seeds) = (family.as_slice()?, source.as_slice()?, target.as_slice()?, seeds.as_slice()?);
        if f.len() != self.games || seeds.len() != self.games {
            return Err(err("choice and seed arrays must have one entry per game"));
        }
        let layouts = match &layouts {
            None => None,
            Some((a, b, c, d)) => {
                let (a, b, c, d) = (a.as_array(), b.as_array(), c.as_array(), d.as_array());
                if a.shape() != [self.games, n] || b.shape() != [self.games, n] || c.shape() != [self.games, n] || d.shape() != [self.games, self.players] {
                    return Err(err("layout arrays must be [games, n] and capitals [games, players]"));
                }
                Some((a, b, c, d))
            }
        };
        let max_turns = self.max_turns.unwrap_or(mapgen::round_limit(self.size));
        let np = self.players;
        let vw = view_width_np(n, np);
        let others = np - 1;
        let mw = if packed_masks { packed_width(n) } else { mask_width(n) };
        // The views are written in full, so they start uninitialised (numpy's PyArray::new contract); the masks
        // only write their true entries and the final views only for truncated games, so those start zeroed.
        let (mut masks, view_arr, opp_arr, mut final_view) = prof::time(prof::TICK_ALLOC, || (
            vec![0u8; self.games * mw],
            unsafe { PyArray2::<i16>::new(py, [self.games, vw], false) },
            unsafe { PyArray2::<i16>::new(py, [if both_views { self.games * others } else { 0 }, vw], false) },
            vec![0i16; self.games * np * vw]));
        // SAFETY: fresh contiguous arrays owned here; every element is written in the loop below before they are
        // returned (view: every game; opp: every game when both_views, else empty).
        let (view, opp) = unsafe { (view_arr.as_slice_mut()?, opp_arr.as_slice_mut()?) };
        let mut done = vec![false; self.games];
        let mut terminated = vec![false; self.games];
        let mut truncated = vec![false; self.games];
        let mut rewards = vec![0f32; self.games * np];
        let mut status = vec![0i32; self.games];
        for g in 0..self.games {
            let st = &mut self.states[g];
            let a = decode(st, f[g], s[g], t[g]);
            let v = st.state_version;
            // the masks the previous tick returned describe exactly this state (until it is stepped below)
            let cache = &mut self.legal[g];
            let legal = if cache.valid { Some(&cache.bits[..]) } else { None };
            status[g] = prof::time(prof::TICK_STEP, || step_with(st, a, v, &self.rules, &self.geom, self.max_actions, &mut self.scratch, legal));
            cache.clear();
            terminated[g] = st.terminated;
            truncated[g] = st.truncated;
            done[g] = st.terminated || st.truncated;
            let rw = terminal_rewards(st);
            rewards[g * np..(g + 1) * np].copy_from_slice(&rw[..np]);
            if st.truncated {
                prof::time(prof::TICK_FINAL, || {
                    for seat in 0..np {
                        let off = (g * np + seat) * vw;
                        observe(st, seat as i32 + 1, &self.geom, &mut final_view[off..off + vw]);
                    }
                });
            }
            if done[g] {
                *st = prof::time(prof::TICK_RESET, || match &layouts {
                    None => mapgen::reset_players(seeds[g], &self.geom, self.max_turns, np),
                    Some((a, b, c, d)) => mapgen::initial_state(seeds[g] as u32, &a.row(g).to_vec(), &b.row(g).to_vec(), &c.row(g).to_vec(),
                                                                &d.row(g).iter().map(|&c| c as usize).collect::<Vec<usize>>(), &self.geom, max_turns),
                });
            }
            let st = &self.states[g];
            let row = &mut masks[g * mw..(g + 1) * mw];
            let (rules, geom, no_resign, ms, cache) = (&self.rules, &self.geom, self.no_resign, &mut self.mscratch, &mut self.legal[g]);
            prof::time(prof::TICK_MASKS, || if packed_masks {
                masks_into(st, rules, geom, no_resign, ms, &mut Tee { out: &mut Packed(row), cache });
            } else {
                masks_into(st, rules, geom, no_resign, ms, &mut Tee { out: &mut Dense(row), cache });
            });
            self.legal[g].valid = true;
            prof::time(prof::TICK_OBSERVE, || observe(st, st.current_player, geom, &mut view[g * vw..(g + 1) * vw]));
            if both_views {
                prof::time(prof::TICK_OBSERVE_OPP, || for (k, seat) in other_seats(st).enumerate() {
                    let off = (g * others + k) * vw;
                    observe(st, seat, geom, &mut opp[off..off + vw]);
                });
            }
        }
        prof::time(prof::TICK_OUTPUT, || {
        let d = PyDict::new(py);
        d.set_item("masks", PyArray1::from_vec(py, masks).reshape([self.games, mw])?)?;
        d.set_item("view", view_arr)?;
        if both_views {
            if np > 2 {
                d.set_item("opp_view", opp_arr.reshape([self.games, others, vw])?)?;
            } else {
                d.set_item("opp_view", opp_arr)?;
            }
        }
        d.set_item("final_view", PyArray1::from_vec(py, final_view).reshape([self.games, np, vw])?)?;
        d.set_item("done", done.into_pyarray(py))?;
        d.set_item("terminated", terminated.into_pyarray(py))?;
        d.set_item("truncated", truncated.into_pyarray(py))?;
        d.set_item("rewards", PyArray1::from_vec(py, rewards).reshape([self.games, np])?)?;
        d.set_item("status", status.into_pyarray(py))?;
        Ok(d)
        })
    }

    /// Turn the phase profile on (resetting it) or off; read it with phase_profile().
    fn profile_phases(&self, on: bool) {
        prof::enable(on);
    }

    /// {phase: (calls, total ms)} accumulated since profile_phases(True), for the phases that ran.
    fn phase_profile<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, calls, ms) in prof::snapshot() {
            d.set_item(name, (calls, ms))?;
        }
        Ok(d)
    }

    /// Median milliseconds per batch for each legal-mask family (and the full mask), for profiling. The families
    /// emit into a counting sink (computation only); "full" is masks_into a zeroed dense buffer, "observe" one seat.
    fn profile_masks<'py>(&self, py: Python<'py>, repeats: usize) -> PyResult<Bound<'py, PyDict>> {
        use std::cell::RefCell;
        use std::time::Instant;
        let n = self.size * self.size;
        let d = PyDict::new(py);
        let ms = RefCell::new(MaskScratch::new(n));
        let mut flat = vec![0u8; mask_width(n)];
        let time = |name: &str, f: &mut dyn FnMut(&State)| -> PyResult<()> {
            let mut samples: Vec<f64> = Vec::with_capacity(repeats);
            for _ in 0..repeats.max(1) {
                let t0 = Instant::now();
                for st in &self.states { f(st); }
                samples.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            d.set_item(name, samples[samples.len() / 2])
        };
        let (r, g) = (&self.rules, &self.geom);
        let mut c = masks::Count(0);
        time("reward", &mut |st| { masks::reward_rows(st, &mut c); })?;
        time("move_rows", &mut |st| masks::move_rows(st, r, g, &mut ms.borrow_mut().mv, &mut c))?;
        time("attack_rows", &mut |st| masks::attack_rows(st, r, g, &mut c))?;
        time("summon", &mut |st| masks::summon_rows(st, r, &mut ms.borrow_mut().summon, &mut c))?;
        time("economic_tiles", &mut |st| economy::economic_tiles(st, &mut ms.borrow_mut().build.econ))?;
        time("build", &mut |st| masks::build_rows(st, r, g, &mut ms.borrow_mut().build, &mut c))?;
        time("research", &mut |st| { c.0 += kernels::research_bits(st, r).count_ones() as usize; })?;
        time("harvest_forest_destroy", &mut |st| {
            let b = &mut ms.borrow_mut().build;
            b.prepare(st, r, g);
            masks::economy_rows(st, &b.econ_list, &mut c)
        })?;
        time("units", &mut |st| masks::unit_rows(st, r, g, &mut c))?;
        time("pending_reward", &mut |st| { let _ = st.pending_reward(); })?;
        time("full_count", &mut |st| masks_into(st, r, g, self.no_resign, &mut ms.borrow_mut(), &mut c))?;
        time("full", &mut |st| legal_masks(st, r, g, self.no_resign, &mut ms.borrow_mut(), &mut flat))?;
        let mut v = vec![0i16; view_width_np(n, self.players)];
        time("observe", &mut |st| observe(st, st.current_player, g, &mut v))?;
        d.set_item("entries", c.0)?;
        Ok(d)
    }

    /// Branch-coverage counters accumulated by step() on this thread (diagnostic).
    fn coverage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, v) in cov::NAMES.iter().zip(cov::snapshot().iter()) {
            d.set_item(name, *v)?;
        }
        Ok(d)
    }

    fn reset_coverage(&self) { cov::reset(); }

    /// [G, mask_width] 0/1 bytes, or with packed=true [G, ceil(mask_width/8)] bits (numpy.packbits order).
    #[pyo3(signature = (packed=false))]
    fn legal_masks<'py>(&mut self, py: Python<'py>, packed: bool) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let n = self.size * self.size;
        let mw = if packed { packed_width(n) } else { mask_width(n) };
        let mut masks = vec![0u8; self.games * mw];
        for g in 0..self.games {
            let row = &mut masks[g * mw..(g + 1) * mw];
            if packed {
                masks_into(&self.states[g], &self.rules, &self.geom, self.no_resign, &mut self.mscratch, &mut Packed(row));
            } else {
                masks_into(&self.states[g], &self.rules, &self.geom, self.no_resign, &mut self.mscratch, &mut Dense(row));
            }
        }
        Ok(PyArray1::from_vec(py, masks).reshape([self.games, mw])?)
    }

    /// Packed public view of `player` (None: the player to move) for every game.
    #[pyo3(signature = (player=None))]
    fn observe<'py>(&self, py: Python<'py>, player: Option<i32>) -> PyResult<Bound<'py, PyArray2<i16>>> {
        let vw = view_width_np(self.size * self.size, self.players);
        let mut view = vec![0i16; self.games * vw];
        for g in 0..self.games {
            let st = &self.states[g];
            let p = match player { Some(p) => p, None => st.current_player };
            observe(st, p, &self.geom, &mut view[g * vw..(g + 1) * vw]);
        }
        Ok(PyArray1::from_vec(py, view).reshape([self.games, vw])?)
    }

    /// Packed public views of the seats NOT to move, for every game: [G, vw] with two seats, else
    /// [G * (players - 1), vw] (reshape to [G, players - 1, vw]), the other seats in turn order after the mover.
    fn observe_opponent<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i16>>> {
        let vw = view_width_np(self.size * self.size, self.players);
        let others = self.players - 1;
        let mut view = vec![0i16; self.games * others * vw];
        for g in 0..self.games {
            let st = &self.states[g];
            for (k, seat) in other_seats(st).enumerate() {
                let off = (g * others + k) * vw;
                observe(st, seat, &self.geom, &mut view[off..off + vw]);
            }
        }
        Ok(PyArray1::from_vec(py, view).reshape([self.games * others, vw])?)
    }

    fn terminal_rewards<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let np = self.players;
        let mut out = vec![0f32; self.games * np];
        for g in 0..self.games {
            let r = terminal_rewards(&self.states[g]);
            out[g * np..(g + 1) * np].copy_from_slice(&r[..np]);
        }
        Ok(PyArray1::from_vec(py, out).reshape([self.games, np])?)
    }

    /// Per game: current_player, turn, terminated, truncated (cheap; no full export).
    fn scalars<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("current_player", self.states.iter().map(|s| s.current_player).collect::<Vec<i32>>().into_pyarray(py))?;
        d.set_item("turn", self.states.iter().map(|s| s.turn).collect::<Vec<i32>>().into_pyarray(py))?;
        d.set_item("terminated", self.states.iter().map(|s| s.terminated).collect::<Vec<bool>>().into_pyarray(py))?;
        d.set_item("truncated", self.states.iter().map(|s| s.truncated).collect::<Vec<bool>>().into_pyarray(py))?;
        Ok(d)
    }

    /// Per game: the stars `player` (1-based) would earn at the start of their turn (structures::income).
    fn income<'py>(&self, py: Python<'py>, player: i32) -> PyResult<Bound<'py, PyArray1<i32>>> {
        if player < 1 || player as usize > self.players {
            return Err(err("player out of range"));
        }
        Ok(self.states.iter().map(|s| structures::income(s, player, &self.rules, &self.geom)).collect::<Vec<i32>>().into_pyarray(py))
    }

    /// Every state field as a NumPy array with a leading game axis; keys "tiles.terrain", ..., "seed".
    fn export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let n = self.size * self.size;
        let gm = self.games;
        let d = PyDict::new(py);
        macro_rules! field {
            ($name:expr, $ty:ty, [$($dim:expr),*], $get:expr) => {{
                let dims: Vec<usize> = vec![$($dim),*];
                let per: usize = dims.iter().product();
                let mut buf: Vec<$ty> = Vec::with_capacity(gm * per);
                for g in 0..gm { let s = &self.states[g]; let f: &dyn Fn(&State, &mut Vec<$ty>) = &$get; f(s, &mut buf); }
                let mut shape = vec![gm]; shape.extend(dims);
                d.set_item($name, PyArray1::from_vec(py, buf).reshape(shape)?)?;
            }};
        }
        field!("tiles.terrain", i32, [n], |s, b| b.extend_from_slice(&s.tiles.terrain));
        field!("tiles.owner", i32, [n], |s, b| b.extend_from_slice(&s.tiles.owner));
        field!("tiles.capital", i32, [n], |s, b| b.extend_from_slice(&s.tiles.capital));
        field!("tiles.city", i32, [n], |s, b| b.extend_from_slice(&s.tiles.city));
        field!("tiles.road", bool, [n], |s, b| b.extend_from_slice(&s.tiles.road));
        field!("tiles.structure", i32, [n], |s, b| b.extend_from_slice(&s.tiles.structure));
        field!("tiles.resource", i32, [n], |s, b| b.extend_from_slice(&s.tiles.resource));
        let np = self.players;
        field!("tiles.explored", bool, [np, n], |s, b| for k in 0..s.np { b.extend_from_slice(&s.tiles.explored[k]); });
        field!("tiles.structure_level", i32, [n], |s, b| b.extend_from_slice(&s.tiles.structure_level));
        field!("tiles.structure_turn", i32, [n], |s, b| b.extend_from_slice(&s.tiles.structure_turn));
        field!("units.owner", i32, [n], |s, b| b.extend_from_slice(&s.units.owner));
        field!("units.kind", i32, [n], |s, b| b.extend_from_slice(&s.units.kind));
        field!("units.health", f32, [n], |s, b| b.extend_from_slice(&s.units.health));
        field!("units.veteran", bool, [n], |s, b| b.extend_from_slice(&s.units.veteran));
        field!("units.moved", bool, [n], |s, b| b.extend_from_slice(&s.units.moved));
        field!("units.attacked", bool, [n], |s, b| b.extend_from_slice(&s.units.attacked));
        field!("units.kills", i32, [n], |s, b| b.extend_from_slice(&s.units.kills));
        field!("units.poison", bool, [n], |s, b| b.extend_from_slice(&s.units.poison));
        field!("units.frozen", bool, [n], |s, b| b.extend_from_slice(&s.units.frozen));
        field!("units.boosted", bool, [n], |s, b| b.extend_from_slice(&s.units.boosted));
        field!("units.home", i32, [n], |s, b| b.extend_from_slice(&s.units.home));
        field!("units.created", i32, [n], |s, b| b.extend_from_slice(&s.units.created));
        field!("units.converted", bool, [n], |s, b| b.extend_from_slice(&s.units.converted));
        field!("units.attacks_performed", i32, [n], |s, b| b.extend_from_slice(&s.units.attacks_performed));
        field!("units.prev_x", i32, [n], |s, b| b.extend_from_slice(&s.units.prev_x));
        field!("units.prev_y", i32, [n], |s, b| b.extend_from_slice(&s.units.prev_y));
        field!("units.last_attack", i32, [n], |s, b| b.extend_from_slice(&s.units.last_attack));
        field!("units.passenger", i32, [n], |s, b| b.extend_from_slice(&s.units.passenger));
        field!("units.invisible", bool, [n], |s, b| b.extend_from_slice(&s.units.invisible));
        field!("cities.owner", i32, [n], |s, b| b.extend_from_slice(&s.cities.owner));
        field!("cities.level", i32, [n], |s, b| b.extend_from_slice(&s.cities.level));
        field!("cities.population", i32, [n], |s, b| b.extend_from_slice(&s.cities.population));
        field!("cities.progress", i32, [n], |s, b| b.extend_from_slice(&s.cities.progress));
        field!("cities.production", i32, [n], |s, b| b.extend_from_slice(&s.cities.production));
        field!("cities.border", i32, [n], |s, b| b.extend_from_slice(&s.cities.border));
        field!("cities.rewards", i32, [n, 9], |s, b| b.extend_from_slice(&s.cities.rewards));
        field!("cities.territory", bool, [n, n], |s, b| b.extend_from_slice(&s.cities.territory));
        field!("cities.connected", bool, [n], |s, b| b.extend_from_slice(&s.cities.connected));
        field!("cities.order", i32, [n], |s, b| b.extend_from_slice(&s.cities.order));
        field!("cities.infiltrated", i32, [n], |s, b| b.extend_from_slice(&s.cities.infiltrated));
        field!("cities.riot", bool, [n], |s, b| b.extend_from_slice(&s.cities.riot));
        field!("players.stars", i32, [np], |s, b| b.extend_from_slice(&s.players.stars[..s.np]));
        field!("players.score", i32, [np], |s, b| b.extend_from_slice(&s.players.score[..s.np]));
        field!("players.tech", bool, [np, 128], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.tech[k]); });
        field!("players.tech_turn", i32, [np, 128], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.tech_turn[k]); });
        field!("players.known", bool, [np, np], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.known[k][..s.np]); });
        field!("players.kills", i32, [np], |s, b| b.extend_from_slice(&s.players.kills[..s.np]));
        field!("players.casualties", i32, [np], |s, b| b.extend_from_slice(&s.players.casualties[..s.np]));
        field!("players.pacifist", i32, [np], |s, b| b.extend_from_slice(&s.players.pacifist[..s.np]));
        field!("players.attacked", bool, [np], |s, b| b.extend_from_slice(&s.players.attacked[..s.np]));
        field!("players.built", bool, [np, 128], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.built[k]); });
        field!("players.killed_turn", i32, [np], |s, b| b.extend_from_slice(&s.players.killed_turn[..s.np]));
        field!("players.resigned_turn", i32, [np], |s, b| b.extend_from_slice(&s.players.resigned_turn[..s.np]));
        field!("players.killer", i32, [np], |s, b| b.extend_from_slice(&s.players.killer[..s.np]));
        field!("players.peace", bool, [np, np], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.peace[k][..s.np]); });
        field!("players.embassy", bool, [np, np], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.embassy[k][..s.np]); });
        field!("players.peace_offer", bool, [np, np], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.peace_offer[k][..s.np]); });
        field!("players.truce_breaker", i32, [np, np], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.truce_breaker[k][..s.np]); });
        field!("players.hostile_until", i32, [np], |s, b| b.extend_from_slice(&s.players.hostile_until[..s.np]));
        field!("players.monuments", bool, [np, 128], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.monuments[k]); });
        if np > 2 {
            field!("players.out_order", i32, [np], |s, b| b.extend_from_slice(&s.players.out_order[..s.np]));
        }
        field!("current_player", i32, [], |s, b| b.push(s.current_player));
        field!("turn", i32, [], |s, b| b.push(s.turn));
        field!("version", i32, [], |s, b| b.push(s.version));
        field!("state_version", i32, [], |s, b| b.push(s.state_version));
        field!("actions", i32, [], |s, b| b.push(s.actions));
        field!("terminated", bool, [], |s, b| b.push(s.terminated));
        field!("truncated", bool, [], |s, b| b.push(s.truncated));
        field!("fog", bool, [], |s, b| b.push(s.fog));
        field!("reference_max_turns", i32, [], |s, b| b.push(s.reference_max_turns));
        field!("core_supported", bool, [], |s, b| b.push(s.core_supported));
        field!("seed", u32, [], |s, b| b.push(s.seed));
        field!("initial_seed", u32, [], |s, b| b.push(s.initial_seed));
        Ok(d)
    }

    /// Inverse of export: overwrite every game from arrays with a leading game axis (missing keys are an error).
    fn import_(&mut self, py: Python<'_>, fields: &Bound<'_, PyDict>) -> PyResult<()> {
        let n = self.size * self.size;
        let gm = self.games;
        fn take<'py, T: numpy::Element + Copy>(py: Python<'py>, d: &Bound<'py, PyDict>, key: &str, expect: usize) -> PyResult<Vec<T>> {
            let item = d.get_item(key)?.ok_or_else(|| err(&format!("missing field {key}")))?;
            let arr: PyReadonlyArrayDyn<T> = item.extract()?;
            let v: Vec<T> = arr.as_array().iter().copied().collect();
            if v.len() != expect {
                return Err(err(&format!("field {key}: expected {expect} values, got {}", v.len())));
            }
            let _ = py;
            Ok(v)
        }
        macro_rules! load {
            ($name:expr, $ty:ty, $per:expr, $set:expr) => {{
                let v: Vec<$ty> = take(py, fields, $name, gm * $per)?;
                for g in 0..gm { let s = &mut self.states[g]; let f: &dyn Fn(&mut State, &[$ty]) = &$set; f(s, &v[g * $per..(g + 1) * $per]); }
            }};
        }
        load!("tiles.terrain", i32, n, |s, v| s.tiles.terrain.copy_from_slice(v));
        load!("tiles.owner", i32, n, |s, v| s.tiles.owner.copy_from_slice(v));
        load!("tiles.capital", i32, n, |s, v| s.tiles.capital.copy_from_slice(v));
        load!("tiles.city", i32, n, |s, v| s.tiles.city.copy_from_slice(v));
        load!("tiles.road", bool, n, |s, v| s.tiles.road.copy_from_slice(v));
        load!("tiles.structure", i32, n, |s, v| s.tiles.structure.copy_from_slice(v));
        load!("tiles.resource", i32, n, |s, v| s.tiles.resource.copy_from_slice(v));
        let np = self.players;
        load!("tiles.explored", bool, np * n, |s, v| for k in 0..s.np { s.tiles.explored[k].copy_from_slice(&v[k * n..(k + 1) * n]); });
        load!("tiles.structure_level", i32, n, |s, v| s.tiles.structure_level.copy_from_slice(v));
        load!("tiles.structure_turn", i32, n, |s, v| s.tiles.structure_turn.copy_from_slice(v));
        load!("units.owner", i32, n, |s, v| s.units.owner.copy_from_slice(v));
        load!("units.kind", i32, n, |s, v| s.units.kind.copy_from_slice(v));
        load!("units.health", f32, n, |s, v| s.units.health.copy_from_slice(v));
        load!("units.veteran", bool, n, |s, v| s.units.veteran.copy_from_slice(v));
        load!("units.moved", bool, n, |s, v| s.units.moved.copy_from_slice(v));
        load!("units.attacked", bool, n, |s, v| s.units.attacked.copy_from_slice(v));
        load!("units.kills", i32, n, |s, v| s.units.kills.copy_from_slice(v));
        load!("units.poison", bool, n, |s, v| s.units.poison.copy_from_slice(v));
        load!("units.frozen", bool, n, |s, v| s.units.frozen.copy_from_slice(v));
        load!("units.boosted", bool, n, |s, v| s.units.boosted.copy_from_slice(v));
        load!("units.home", i32, n, |s, v| s.units.home.copy_from_slice(v));
        load!("units.created", i32, n, |s, v| s.units.created.copy_from_slice(v));
        load!("units.converted", bool, n, |s, v| s.units.converted.copy_from_slice(v));
        load!("units.attacks_performed", i32, n, |s, v| s.units.attacks_performed.copy_from_slice(v));
        load!("units.prev_x", i32, n, |s, v| s.units.prev_x.copy_from_slice(v));
        load!("units.prev_y", i32, n, |s, v| s.units.prev_y.copy_from_slice(v));
        load!("units.last_attack", i32, n, |s, v| s.units.last_attack.copy_from_slice(v));
        load!("units.passenger", i32, n, |s, v| s.units.passenger.copy_from_slice(v));
        load!("units.invisible", bool, n, |s, v| s.units.invisible.copy_from_slice(v));
        load!("cities.owner", i32, n, |s, v| s.cities.owner.copy_from_slice(v));
        load!("cities.level", i32, n, |s, v| s.cities.level.copy_from_slice(v));
        load!("cities.population", i32, n, |s, v| s.cities.population.copy_from_slice(v));
        load!("cities.progress", i32, n, |s, v| s.cities.progress.copy_from_slice(v));
        load!("cities.production", i32, n, |s, v| s.cities.production.copy_from_slice(v));
        load!("cities.border", i32, n, |s, v| s.cities.border.copy_from_slice(v));
        load!("cities.rewards", i32, n * 9, |s, v| s.cities.rewards.copy_from_slice(v));
        load!("cities.territory", bool, n * n, |s, v| s.cities.territory.copy_from_slice(v));
        load!("cities.connected", bool, n, |s, v| s.cities.connected.copy_from_slice(v));
        load!("cities.order", i32, n, |s, v| s.cities.order.copy_from_slice(v));
        load!("cities.infiltrated", i32, n, |s, v| s.cities.infiltrated.copy_from_slice(v));
        load!("cities.riot", bool, n, |s, v| s.cities.riot.copy_from_slice(v));
        load!("players.stars", i32, np, |s, v| s.players.stars[..s.np].copy_from_slice(v));
        load!("players.score", i32, np, |s, v| s.players.score[..s.np].copy_from_slice(v));
        load!("players.tech", bool, np * 128, |s, v| for k in 0..s.np { s.players.tech[k].copy_from_slice(&v[k * 128..(k + 1) * 128]); });
        load!("players.tech_turn", i32, np * 128, |s, v| for k in 0..s.np { s.players.tech_turn[k].copy_from_slice(&v[k * 128..(k + 1) * 128]); });
        load!("players.known", bool, np * np, |s, v| for k in 0..s.np { s.players.known[k][..s.np].copy_from_slice(&v[k * s.np..(k + 1) * s.np]); });
        load!("players.kills", i32, np, |s, v| s.players.kills[..s.np].copy_from_slice(v));
        load!("players.casualties", i32, np, |s, v| s.players.casualties[..s.np].copy_from_slice(v));
        load!("players.pacifist", i32, np, |s, v| s.players.pacifist[..s.np].copy_from_slice(v));
        load!("players.attacked", bool, np, |s, v| s.players.attacked[..s.np].copy_from_slice(v));
        load!("players.built", bool, np * 128, |s, v| for k in 0..s.np { s.players.built[k].copy_from_slice(&v[k * 128..(k + 1) * 128]); });
        load!("players.killed_turn", i32, np, |s, v| s.players.killed_turn[..s.np].copy_from_slice(v));
        load!("players.resigned_turn", i32, np, |s, v| s.players.resigned_turn[..s.np].copy_from_slice(v));
        load!("players.killer", i32, np, |s, v| s.players.killer[..s.np].copy_from_slice(v));
        load!("players.peace", bool, np * np, |s, v| for k in 0..s.np { s.players.peace[k][..s.np].copy_from_slice(&v[k * s.np..(k + 1) * s.np]); });
        load!("players.embassy", bool, np * np, |s, v| for k in 0..s.np { s.players.embassy[k][..s.np].copy_from_slice(&v[k * s.np..(k + 1) * s.np]); });
        load!("players.peace_offer", bool, np * np, |s, v| for k in 0..s.np { s.players.peace_offer[k][..s.np].copy_from_slice(&v[k * s.np..(k + 1) * s.np]); });
        load!("players.truce_breaker", i32, np * np, |s, v| for k in 0..s.np { s.players.truce_breaker[k][..s.np].copy_from_slice(&v[k * s.np..(k + 1) * s.np]); });
        load!("players.hostile_until", i32, np, |s, v| s.players.hostile_until[..s.np].copy_from_slice(v));
        load!("players.monuments", bool, np * 128, |s, v| for k in 0..s.np { s.players.monuments[k].copy_from_slice(&v[k * 128..(k + 1) * 128]); });
        if np > 2 {
            load!("players.out_order", i32, np, |s, v| s.players.out_order[..s.np].copy_from_slice(v));
        }
        load!("current_player", i32, 1, |s, v| s.current_player = v[0]);
        load!("turn", i32, 1, |s, v| s.turn = v[0]);
        load!("version", i32, 1, |s, v| s.version = v[0]);
        load!("state_version", i32, 1, |s, v| s.state_version = v[0]);
        load!("actions", i32, 1, |s, v| s.actions = v[0]);
        load!("terminated", bool, 1, |s, v| s.terminated = v[0]);
        load!("truncated", bool, 1, |s, v| s.truncated = v[0]);
        load!("fog", bool, 1, |s, v| s.fog = v[0]);
        load!("reference_max_turns", i32, 1, |s, v| s.reference_max_turns = v[0]);
        load!("core_supported", bool, 1, |s, v| s.core_supported = v[0]);
        load!("seed", u32, 1, |s, v| s.seed = v[0]);
        load!("initial_seed", u32, 1, |s, v| s.initial_seed = v[0]);
        for s in self.states.iter_mut() {
            s.monuments_settled = false;
        }
        self.forget_masks();
        Ok(())
    }

    /// Names of the exported fields, in export order.
    fn field_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let d = self.export(py)?;
        Ok(d.keys())
    }
}

/// Seat ids (1-based) other than the mover, in turn order after it: with two seats the single opponent.
fn other_seats(st: &State) -> impl Iterator<Item = i32> + '_ {
    let np = st.np as i32;
    (1..np).map(move |k| (st.current_player - 1 + k) % np + 1)
}

impl GameBatch {
    /// The games' states changed outside tick(): the cached masks no longer describe them.
    fn forget_masks(&mut self) {
        for c in &mut self.legal {
            c.clear();
        }
    }
}

#[pymodule]
fn polytopia_engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GameBatch>()?;
    m.add("OK", core::OK)?;
    Ok(())
}

#[allow(dead_code)]
fn _unused(_: &PyArray3<i16>) {}

````


## engine-rs/src/mapgen.rs

Source SHA-256: 441e7c3e3da8ec2a2584e11cbaa2400c2fe2931b5d25d360c35c4ac0fe43a2b6

````rust
//! Procedural Bardur Drylands layout and reset (gpu/mapgen.py, maps/state.py::initial_state). Two seats are the
//! reference generator bit for bit; four seats run the same steps with one capital per quadrant.

use crate::geom::Geom;
use crate::hash::DotNetRandom;
use crate::kernels::update_exploration;
use crate::state::State;

pub fn ruins(size: usize) -> Option<i32> {
    match size {
        11 => Some(4),
        14 => Some(5),
        18 => Some(9),
        _ => None,
    }
}

pub fn round_limit(size: usize) -> i32 {
    if size == 18 { 150 } else { 100 }
}

pub struct Layout {
    pub terrain: Vec<i32>,
    pub structure: Vec<i32>,
    pub resource: Vec<i32>,
    /// One capital tile per seat.
    pub capitals: Vec<usize>,
}

/// mapgen._choose: bounded draw over the count of set entries, then the draw-th set index (0 when none).
fn choose(rng: &mut DotNetRandom, mask: &[bool]) -> usize {
    let count = mask.iter().filter(|&&m| m).count() as i32;
    let draw = rng.bounded(count.max(1));
    let mut seen = 0;
    for (i, &m) in mask.iter().enumerate() {
        if m {
            if seen == draw {
                return i;
            }
            seen += 1;
        }
    }
    0
}

pub fn layout(seed: i32, g: &Geom) -> Layout {
    layout_players(seed, g, 2)
}

/// layout for `np` seats (2..=4): capitals in distinct quadrants, each later one drawn among the free quadrants at
/// least half as far from the first capital as the farthest free one.
pub fn layout_players(seed: i32, g: &Geom, np: usize) -> Layout {
    assert!((2..=4).contains(&np));
    let (size, n) = (g.size, g.n);
    let s = size as i32;
    let d = |i: usize, j: usize| g.d(i, j);
    let mut rng = DotNetRandom::new(seed);
    let mut available = [true; 4];
    let mut caps = vec![0usize; np];
    let quadsize = s / 2;
    let centers_x: Vec<i32> = (0..4).map(|c| (c % 2) * quadsize + quadsize / 2).collect();
    let centers_y: Vec<i32> = (0..4).map(|c| (c / 2) * quadsize + quadsize / 2).collect();
    for i in 0..np {
        let (c0x, c0y) = ((caps[0] % size) as i32, (caps[0] / size) as i32);
        let sq: Vec<i32> = (0..4).map(|q| (centers_x[q] - c0x).pow(2) + (centers_y[q] - c0y).pow(2)).collect();
        let maximum = (0..4).filter(|&q| available[q]).map(|q| sq[q]).max().unwrap_or(-1);
        let allowed: Vec<bool> = (0..4).map(|q| available[q] && (i == 0 || sq[q] >= maximum.div_euclid(2))).collect();
        let q = choose(&mut rng, &allowed) as i32;
        let sx = ((q % 2) * quadsize + 2).min(s - 3);
        let ex = (((q % 2) + 1) * quadsize - 2).max(sx + 1).min(s - 2);
        let sy = ((q / 2) * quadsize + 2).min(s - 3);
        let ey = (((q / 2) + 1) * quadsize - 2).max(sy + 1).min(s - 2);
        let cx = rng.bounded(ex - sx);
        let cy = rng.bounded(ey - sy);
        available[q as usize] = false;
        caps[i] = ((cy + sy) * s + cx + sx) as usize;
    }
    let mut land = vec![false; n];
    for &c in &caps {
        land[c] = true;
    }
    let target_land = ((n as f64) * 0.95) as usize;
    let mut attempt = 0i32;
    while land.iter().filter(|&&l| l).count() < target_land && attempt < (n as i32) * 10 {
        let at = rng.bounded(n as i32) as usize;
        land[at] = true;
        attempt += 1;
    }
    let mut villages = vec![false; n];
    for &c in &caps {
        villages[c] = true;
    }
    // Biome traversal: one active array per seat with swap removal, kept only for the RNG stream.
    let mut active = vec![vec![0usize; n]; np];
    for i in 0..np {
        active[i][0] = caps[i];
    }
    let mut counts = vec![1usize; np];
    let mut done = villages.clone();
    let mut changed = true;
    let mut candidates = vec![false; n];
    while changed {
        changed = false;
        for i in 0..np {
            if counts[i] > 0 {
                let index = rng.bounded(counts[i] as i32) as usize;
                let at = active[i][index];
                for t in 0..n {
                    candidates[t] = d(at, t) <= 1 && !done[t];
                }
                let any_preferred = (0..n).any(|t| candidates[t] && land[t]);
                if any_preferred {
                    for t in 0..n {
                        candidates[t] = candidates[t] && land[t];
                    }
                }
                if candidates.iter().any(|&c| c) {
                    let chosen = choose(&mut rng, &candidates);
                    active[i][counts[i]] = chosen;
                    counts[i] += 1;
                    done[chosen] = true;
                    changed = true;
                } else {
                    active[i][index] = active[i][counts[i] - 1];
                    counts[i] -= 1;
                }
            }
        }
    }
    let mut terrain = vec![0i32; n];
    for i in 0..n {
        let coast = !land[i]
            && [1usize, 3, 4, 6].iter().any(|&k| g.nb_valid[i * 8 + k] && land[g.nb[i * 8 + k] as usize]);
        terrain[i] = if land[i] { 3 } else if coast { 1 } else { 2 };
    }
    let forest_cut = 0.14f32 + 0.38f32 * 1.5f32;
    for i in 0..n {
        if land[i] && !villages[i] {
            let value = rng.unit_float();
            terrain[i] = if value < 0.14f32 { 4 } else if value < forest_cut { 5 } else { 3 };
        }
    }
    let edge: Vec<i32> = (0..n)
        .map(|i| {
            let (x, y) = ((i % size) as i32, (i / size) as i32);
            x.min(s - 1 - x).min(y.min(s - 1 - y))
        })
        .collect();
    let village_candidates = |villages: &[bool], terrain: &[i32], out: &mut [bool]| {
        for i in 0..n {
            out[i] = land[i] && !villages[i] && terrain[i] != 4 && edge[i] >= 2 && edge[i] != 3
                && !(0..n).any(|j| d(i, j) < 3 && villages[j]);
        }
    };
    loop {
        village_candidates(&villages, &terrain, &mut candidates);
        if !candidates.iter().any(|&c| c) {
            break;
        }
        let at = choose(&mut rng, &candidates);
        villages[at] = true;
        terrain[at] = 3;
    }
    let mut above: Vec<i32> = villages.iter().map(|&v| if v { 1 } else { 0 }).collect();
    for i in 0..np {
        for _ in 0..2 {
            for t in 0..n {
                candidates[t] = d(caps[i], t) <= 1 && t != caps[i] && above[t] == 0 && terrain[t] != 2;
            }
            if candidates.iter().any(|&c| c) {
                let at = choose(&mut rng, &candidates);
                terrain[at] = 5;
                above[at] = 101;
            }
        }
    }
    for city in 0..n {
        if !villages[city] {
            continue;
        }
        let mut count = (0..n).filter(|&t| d(city, t) <= 1 && above[t] == 101).count() as i32;
        for rad in 1..3i32 {
            for at in 0..n {
                let kind = terrain[at];
                if d(city, at) <= rad && above[at] == 0 && kind != 2 {
                    let value = rng.unit_float();
                    let probability: f32 = if kind == 3 {
                        if rad == 1 { 0.18 } else { 0.06 }
                    } else if kind == 5 {
                        if rad == 1 && count >= 3 { 0.0 } else if rad == 1 { 0.38 } else { 0.12 }
                    } else if kind == 4 {
                        if rad == 1 { 0.11 } else { 0.03 }
                    } else {
                        0.5
                    };
                    let resource = if kind == 3 { 106 } else if kind == 5 { 101 } else if kind == 4 { 105 } else { 103 };
                    let added = value < probability;
                    above[at] = if added { resource } else { 0 };
                    if added && kind == 5 && rad == 1 {
                        count += 1;
                    }
                }
            }
        }
    }
    let resource_value: Vec<i32> = (0..n)
        .map(|t| {
            let a = above[t];
            (if a == 101 || a == 103 || a == 106 { 2 } else if a == 105 { 3 } else { 0 }) + (terrain[t] == 5) as i32
        })
        .collect();
    let scores: Vec<i32> = (0..np).map(|i| (0..n).filter(|&t| d(caps[i], t) <= 2).map(|t| resource_value[t]).sum()).collect();
    let best = *scores.iter().max().unwrap();
    for i in 0..np {
        if scores[i] + 4 < best {
            let mut order: Vec<usize> = (0..n).filter(|&t| d(caps[i], t) <= 2).collect();
            let count = order.len();
            for _ in 0..count {
                let a = rng.bounded(count as i32) as usize;
                let b = rng.bounded(count as i32) as usize;
                order.swap(a, b);
            }
            let (mut at, mut score) = (0usize, scores[i]);
            while at < count && score + 4 < best {
                let idx = order[at];
                if above[idx] == 0 && terrain[idx] == 5 {
                    above[idx] = 101;
                    score += 2;
                }
                at += 1;
            }
        }
    }
    let ruin_count = ruins(size).expect("supported board size");
    let (mut placed, mut attempt) = (0i32, 0i32);
    while placed < ruin_count && attempt < 2000 {
        let at = rng.bounded(n as i32) as usize;
        let valid = above[at] == 0 && terrain[at] > 2 && !(0..n).any(|t| d(at, t) <= 1 && (above[t] == 2 || villages[t]));
        if valid {
            above[at] = 2;
            placed += 1;
        }
        attempt += 1;
    }
    let (mut placed, mut attempt) = (0i32, 0i32);
    while placed < (n / 25) as i32 && attempt < 1000 {
        let at = rng.bounded(n as i32) as usize;
        let valid = terrain[at] <= 2 && above[at] == 0
            && !(0..n).any(|t| d(at, t) <= 1 && (above[t] == 108 || above[t] == 47 || above[t] == 1));
        if valid {
            above[at] = 108;
            placed += 1;
        }
        attempt += 1;
    }
    for &c in &g.corners {
        above[c] = 47;
    }
    Layout {
        terrain,
        structure: above.iter().map(|&a| if a < 100 { a } else { 0 }).collect(),
        resource: above.iter().map(|&a| if a >= 100 { a - 100 } else { 0 }).collect(),
        capitals: caps,
    }
}

/// maps/state.py::initial_state, bitwise mapgen.reset after the layout step.
pub fn initial_state(seed: u32, terrain: &[i32], structure: &[i32], resource: &[i32], caps: &[usize], g: &Geom,
                     max_turns: i32) -> State {
    let (size, n) = (g.size, g.n);
    let np = caps.len();
    let mut s = State::empty_players(size, np);
    s.seed = seed;
    s.initial_seed = seed;
    s.reference_max_turns = max_turns;
    s.tiles.terrain.copy_from_slice(terrain);
    s.tiles.structure.copy_from_slice(structure);
    s.tiles.resource.copy_from_slice(resource);
    for t in 0..n {
        // owner = max over seats of (area ? seat+1 : 0)
        let mut owner = 0;
        for k in 0..np {
            if g.d(caps[k], t) <= 1 {
                owner = owner.max(k as i32 + 1);
            }
        }
        s.tiles.owner[t] = owner;
        s.tiles.city[t] = if owner > 0 { caps[(owner - 1) as usize] as i32 } else { -1 };
    }
    for k in 0..np {
        let c = caps[k];
        let owner = k as i32 + 1;
        s.tiles.capital[c] = owner;
        s.cities.owner[c] = owner;
        s.cities.level[c] = 1;
        s.cities.production[c] = 1;
        s.cities.border[c] = 1;
        for t in 0..n {
            s.cities.territory[c * n + t] = g.d(c, t) <= 1;
        }
        s.cities.order[c] = 1;
        s.units.owner[c] = owner;
        s.units.kind[c] = 2;
        s.units.health[c] = 10.0;
        s.units.home[c] = c as i32 + 1;
        s.units.prev_x[c] = (c % size) as i32;
        s.units.prev_y[c] = (c / size) as i32;
    }
    for k in 0..np {
        s.players.stars[k] = 5;
        s.players.tech[k][0] = true;
        s.players.tech[k][15] = true;
        s.players.score[k] = 390;
    }
    for k in 0..np {
        update_exploration(&mut s, k as i32 + 1, g);
    }
    s
}

pub fn reset(seed: i32, g: &Geom, max_turns: Option<i32>) -> State {
    reset_players(seed, g, max_turns, 2)
}

pub fn reset_players(seed: i32, g: &Geom, max_turns: Option<i32>, np: usize) -> State {
    let board = layout_players(seed, g, np);
    initial_state(seed as u32, &board.terrain, &board.structure, &board.resource, &board.capitals, g,
                  max_turns.unwrap_or(round_limit(g.size)))
}

````


## engine-rs/src/masks.rs

Source SHA-256: 1383702efbaebdf7062700d900bbd5b7eb507cd48d3fb69320d63b3f5c1dae84

````rust
//! gpu/actions.py::legal_masks, written into one flat boolean buffer in catalog order
//! (move, attack, summon, build, research, harvest, forest, reward, capture, recover, promote, disband, heal,
//! destroy, diplomatic, end, resign).
//!
//! The mask is mostly zeros (a few hundred true entries out of 130,868 on 14x14), so every family only emits its
//! true entries into a zeroed sink: dense 0/1 bytes, or bits in numpy.packbits order (the inference server's wire
//! format). Nothing is written for false entries.

use crate::capture;
use crate::core::family_widths;
use crate::diplomacy;
use crate::geom::Geom;
use crate::kernels::{max_health, movement_flags, reach, research_bits, MoveScratch};
use crate::military::{self, SummonCtx};
use crate::rules::Rules;
use crate::state::State;
use crate::structures::{self, BuildCtx};

pub fn mask_width(n: usize) -> usize {
    family_widths(n).iter().sum()
}

/// Bytes per game of the packed masks: ceil(mask_width / 8).
pub fn packed_width(n: usize) -> usize {
    (mask_width(n) + 7) / 8
}

pub fn family_offsets(n: usize) -> [usize; 18] {
    let w = family_widths(n);
    let mut o = [0usize; 18];
    for i in 0..17 {
        o[i + 1] = o[i] + w[i];
    }
    o
}

/// Receives the true entries of one game's flat mask.
pub trait Sink {
    fn set(&mut self, i: usize);
}

/// 0/1 bytes; the buffer must be zeroed.
pub struct Dense<'a>(pub &'a mut [u8]);

impl Sink for Dense<'_> {
    #[inline]
    fn set(&mut self, i: usize) {
        self.0[i] = 1;
    }
}

/// Bits, most significant first (numpy.packbits' default order); the buffer must be zeroed.
pub struct Packed<'a>(pub &'a mut [u8]);

impl Sink for Packed<'_> {
    #[inline]
    fn set(&mut self, i: usize) {
        self.0[i >> 3] |= 0x80 >> (i & 7);
    }
}

/// Buffers reused across legal_masks calls (one per GameBatch).
pub struct MaskScratch {
    pub mv: MoveScratch,
    pub build: BuildCtx,
    pub summon: SummonCtx,
}

impl MaskScratch {
    pub fn new(n: usize) -> MaskScratch {
        MaskScratch { mv: MoveScratch::new(n), build: BuildCtx::new(n), summon: SummonCtx::new(n) }
    }
}

/// Dense masks into `out` (any contents; the first mask_width bytes are overwritten).
pub fn legal_masks(s: &State, r: &Rules, g: &Geom, no_resign: bool, ms: &mut MaskScratch, out: &mut [u8]) {
    out[..mask_width(s.n)].fill(0);
    masks_into(s, r, g, no_resign, ms, &mut Dense(out));
}

/// Forwards to another sink and records every entry in a persistent packed bitset (LegalCache).
pub struct Tee<'a, S: Sink> {
    pub out: &'a mut S,
    pub cache: &'a mut LegalCache,
}

impl<S: Sink> Sink for Tee<'_, S> {
    #[inline]
    fn set(&mut self, i: usize) {
        self.out.set(i);
        self.cache.bits[i >> 3] |= 0x80 >> (i & 7);
        self.cache.list.push(i as u32);
    }
}

/// One game's legal masks from the last tick, kept for the next step's validation (core::step_with): packed bits
/// (masks::Packed layout), the list of set entries (to clear them cheaply) and whether they belong to the game's
/// current state.
pub struct LegalCache {
    pub bits: Vec<u8>,
    list: Vec<u32>,
    pub valid: bool,
}

impl LegalCache {
    pub fn new(n: usize) -> LegalCache {
        LegalCache { bits: vec![0; packed_width(n)], list: Vec::new(), valid: false }
    }
    /// Forget the cached masks (they no longer describe the state).
    pub fn clear(&mut self) {
        for &i in &self.list {
            self.bits[(i >> 3) as usize] = 0;
        }
        self.list.clear();
        self.valid = false;
    }
}

/// Counts the true entries (profiling: the mask computation without its memory traffic).
pub struct Count(pub usize);

impl Sink for Count {
    #[inline]
    fn set(&mut self, _: usize) {
        self.0 += 1;
    }
}

/// Emit every true entry of actions.legal_masks (CpuGames: resign forced False when no_resign).
pub fn masks_into<S: Sink>(s: &State, r: &Rules, g: &Geom, no_resign: bool, ms: &mut MaskScratch, out: &mut S) {
    let active = s.core_supported && !s.terminated && !s.truncated;
    if !active {
        return;
    }
    // reward is gated by `active` only; any reward entry is also State::pending_reward, which gates the rest
    if reward_rows(s, out) {
        return;
    }
    let o = family_offsets(s.n);
    move_rows(s, r, g, &mut ms.mv, out);
    attack_rows(s, r, g, out);
    summon_rows(s, r, &mut ms.summon, out);
    build_rows(s, r, g, &mut ms.build, out);
    let mut bits = research_bits(s, r);
    while bits != 0 {
        out.set(o[4] + bits.trailing_zeros() as usize);
        bits &= bits - 1;
    }
    economy_rows(s, &ms.build.econ_list, out);
    unit_rows(s, r, g, out);
    let other = 3 - s.current_player;
    let capital = diplomacy::opponent_capital(s);
    let abilities = [18, 19, 19, 20, 21, 22];
    let targets = [-1, 0, 1, capital.0 as i32, -1, -1];
    for k in 0..6 {
        if diplomacy::legal_with(s, other, targets[k], abilities[k], capital) {
            out.set(o[14] + k);
        }
    }
    out.set(o[15]);
    // resigning exists only in two-seat games
    if !no_resign && s.np == 2 {
        out.set(o[16]);
    }
}

/// City rewards: the two kinds of the next slot of every own city with one pending. Returns whether any is
/// pending (State::pending_reward).
pub fn reward_rows<S: Sink>(s: &State, out: &mut S) -> bool {
    let o = family_offsets(s.n);
    let (c, cp) = (&s.cities, s.current_player);
    let mut pending = false;
    for i in 0..s.n {
        if c.owner[i] == cp {
            let count = s.reward_count(i);
            if count < c.level[i] - 1 {
                pending = true;
                let (a, b) = match count + 1 { 1 => (3, 4), 2 => (1, 7), 3 => (5, 8), _ => (2, 6) };
                out.set(o[7] + i * 9 + a);
                out.set(o[7] + i * 9 + b);
            }
        }
    }
    pending
}

/// Move rows of the mover's unmoved units (kernels.movement_targets' remaining gate: a supported kind).
pub fn move_rows<S: Sink>(s: &State, r: &Rules, g: &Geom, mv: &mut MoveScratch, out: &mut S) {
    let (n, u, cp) = (s.n, &s.units, s.current_player);
    let o = family_offsets(n);
    let movable = |i: usize| u.owner[i] == cp && !u.moved[i] && r.unit_supported[u.kind[i] as usize];
    let Some(first) = (0..n).find(|&i| movable(i)) else {
        return;
    };
    movement_flags(s, cp, g, &mut mv.flags);
    for i in first..n {
        if movable(i) {
            let base = o[0] + i * n;
            reach(s, i, r, g, mv, |j| out.set(base + j));
        }
    }
}

pub fn attack_rows<S: Sink>(s: &State, r: &Rules, g: &Geom, out: &mut S) {
    let (n, u, cp) = (s.n, &s.units, s.current_player);
    let o = family_offsets(n);
    for i in 0..n {
        if u.owner[i] == cp && !u.attacked[i] {
            let base = o[1] + i * n;
            military::attack_targets(s, i, r, g, |j| out.set(base + j));
        }
    }
}

pub fn summon_rows<S: Sink>(s: &State, r: &Rules, ctx: &mut SummonCtx, out: &mut S) {
    let (n, u, cp) = (s.n, &s.units, s.current_player);
    let o = family_offsets(n);
    ctx.prepare(s, r);
    for i in 0..n {
        if s.cities.owner[i] == cp || u.owner[i] == cp {
            let mut bits = military::summon_row(s, ctx, i) & r.supported_bits;
            while bits != 0 {
                out.set(o[2] + i * 128 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
    }
}

/// Build rows; leaves economic_tiles(s) in `ctx.econ` for the economy rows.
pub fn build_rows<S: Sink>(s: &State, r: &Rules, g: &Geom, ctx: &mut BuildCtx, out: &mut S) {
    let o = family_offsets(s.n);
    ctx.prepare(s, r, g);
    // outside the economy only the neutral road and the neutral bridge can be built
    {
        for i in 0..s.n {
            if !ctx.econ[i] && structures::neutral_road(s, r, ctx, i) {
                out.set(o[3] + i * 128 + 3);
            }
            if !ctx.econ[i] && structures::neutral_bridge(s, r, ctx, i) {
                out.set(o[3] + i * 128 + 48);
            }
        }
    }
    for &i in &ctx.econ_list {
        let i = i as usize;
        let mut bits = structures::build_row(s, r, g, ctx, i) & !(1u128 << 37);
        while bits != 0 {
            out.set(o[3] + i * 128 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}

/// Harvest, forest and destroy (the economy tiles, as listed by BuildCtx::prepare).
pub fn economy_rows<S: Sink>(s: &State, econ: &[u16], out: &mut S) {
    let o = family_offsets(s.n);
    let t = &s.tiles;
    let pi = s.p();
    let tech = &s.players.tech[pi];
    let stars = s.players.stars[pi];
    for &i in econ {
        let i = i as usize;
        let (resource, structure, terrain) = (t.resource[i], t.structure[i], t.terrain[i]);
        let unlocked = (resource == 1 && tech[15]) || (resource == 3 && tech[10]) || (resource == 6 && tech[6]);
        if structure == 0 && unlocked && stars >= 2 {
            out.set(o[5] + i);
        }
        // forest: clear (3 stars), burn, grow
        if structure == 0 || structure == 3 {
            if terrain == 5 && tech[9] && stars >= 3 {
                out.set(o[6] + i * 3);
            }
            if terrain == 5 && tech[16] {
                out.set(o[6] + i * 3 + 1);
            }
            if terrain == 3 && tech[19] && stars >= 5 {
                out.set(o[6] + i * 3 + 2);
            }
        }
        if structures::destructible(structure) && tech[3] {
            out.set(o[13] + i);
        }
    }
}

/// Capture, recover, promote, disband, heal (the mover's units only).
pub fn unit_rows<S: Sink>(s: &State, r: &Rules, g: &Geom, out: &mut S) {
    let o = family_offsets(s.n);
    let (u, cp) = (&s.units, s.current_player);
    let tech2 = s.players.tech[s.p()][2];
    for i in 0..s.n {
        if u.owner[i] != cp {
            continue;
        }
        let idle = !u.moved[i] && !u.attacked[i];
        if idle && capture::capturable(s, i) {
            out.set(o[8] + i);
        }
        if idle && u.health[i] < max_health(s, r, i) {
            out.set(o[9] + i);
        }
        if crate::military::promotable(s, r, i) {
            out.set(o[10] + i);
        }
        if idle && tech2 {
            out.set(o[11] + i);
        }
        if idle && u.kind[i] == 10 && g.adj[i].iter().any(|&j| {
            let j = j as usize;
            u.owner[j] == cp && u.health[j] < max_health(s, r, j)
        }) {
            out.set(o[12] + i);
        }
    }
}

````


## engine-rs/src/military.rs

Source SHA-256: 70c6f892da14427df58688bed35d4c30fb765b79861673ddee1192240f4b564d

````rust
//! gpu/military.py: unit actions (native profile).

use crate::diplomacy::{hostile, protected};
use crate::economy::discover;
use crate::geom::Geom;
use crate::kernels::{combat, max_health};
use crate::rules::{Rules, CARRY, FLOAT, HIDE, INDEPENDENT, SCOUT, SPLASH, STOMP, SURPRISE};
use crate::state::State;
use crate::structures::capital_index;

pub fn exhaust(s: &mut State, source: usize) {
    s.units.moved[source] = true;
    s.units.attacked[source] = true;
}

/// military.remove: killer=None mirrors killer=-1 (index clipped to 0, no credit).
pub fn remove(s: &mut State, target: usize, r: &Rules, killer: Option<usize>, casualty: bool) {
    let u = &s.units;
    let owner = s.pclip(u.owner[target]);
    let k = killer.unwrap_or(0);
    let kp = s.pclip(u.owner[k]);
    let kind = u.kind[target] as usize;
    let passenger = u.passenger[target];
    let value = r.unit_cost[kind] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    let loss = -5 * value * (!u.converted[target]) as i32;
    s.players.score[owner] = s.players.score[owner].wrapping_add(loss);
    s.players.casualties[owner] += casualty as i32;
    s.players.kills[kp] += killer.is_some() as i32;
    if s.units.poison[target] && s.tiles.structure[target] == 0 {
        s.tiles.resource[target] = 7;
    }
    // AttackAction.Execute (game version >= 95): no xp for a killer that is a vehicle (Carry: every
    // boat) or Static; the passenger of a boat keeps the xp it had when it embarked
    let xp = killer.is_some() && !(r.skills[u.kind[k] as usize][CARRY] || r.stat[u.kind[k] as usize]);
    s.units.kills[k] += xp as i32;
    s.units.clear(target);
}

pub fn relocate(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom, involuntary: bool, reveal: Option<&[bool]>) {
    let (n, size) = (s.n, s.size);
    let owner = s.units.owner[source];
    let kind = s.units.kind[source] as usize;
    let passenger_src = s.units.passenger[source];
    let moved_src = s.units.moved[source];
    let attacked_src = s.units.attacked[source];
    let invisible_src = s.units.invisible[source];
    let converted_src = s.units.converted[source];
    let terrain_src = s.tiles.terrain[source];
    s.units.move_unit(source, target);
    s.units.moved[target] = true;
    s.units.attacked[target] = true;
    s.units.prev_x[target] = (source % size) as i32;
    s.units.prev_y[target] = (source / size) as i32;
    let scout = r.skills[kind][SCOUT];
    let own_reveal: Vec<bool>;
    let reveal: &[bool] = match reveal {
        Some(v) => v,
        None => {
            let rs = if terrain_src == 4 || scout { 2 } else { 1 };
            let rt = if s.tiles.terrain[target] == 4 || scout { 2 } else { 1 };
            own_reveal = (0..n).map(|t| g.d(source, t) <= rs || g.d(target, t) <= rt).collect();
            &own_reveal
        }
    };
    discover(s, reveal, owner);
    if r.skills[kind][STOMP] {
        crate::cov::hit(28);
        area_damage(s, target, target, r, g, true);
    }
    let tt = s.tiles.terrain[target];
    let embark = s.tiles.structure[target] == 8 && !r.skills[kind][FLOAT];
    let special_land = (kind == 39 || kind == 40 || kind == 45) && tt > 2;
    let disembark = r.skills[kind][CARRY] && (passenger_src > 0 || special_land) && tt > 2 && !embark;
    let mut new_kind = if embark {
        if kind == 12 { 45 } else if kind == 38 { 39 } else if kind == 37 { 40 } else { 43 }
    } else if disembark {
        passenger_src as usize
    } else {
        kind
    };
    if special_land {
        new_kind = if kind == 39 { 38 } else if kind == 40 { 37 } else { 12 };
    }
    let passenger = if embark && kind != 12 && kind != 38 && kind != 37 { kind as i32 } else if disembark { 0 } else { passenger_src };
    let score_before = r.unit_cost[kind] + if passenger_src > 0 { r.unit_cost[passenger_src as usize] } else { 0 };
    let score_after = r.unit_cost[new_kind] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    if embark || disembark { crate::cov::hit(36); }
    let dash = !involuntary && !moved_src && !attacked_src && r.dash[new_kind];
    s.units.kind[target] = new_kind as i32;
    s.units.passenger[target] = passenger;
    s.units.attacked[target] = if dash { false } else { s.units.attacked[target] || disembark };
    let hidden = r.skills[new_kind][HIDE] && !involuntary;
    s.units.invisible[target] |= hidden;
    s.units.attacked[target] |= hidden && !invisible_src;
    let o = s.pwrap(owner);
    s.players.score[o] = s.players.score[o].wrapping_add(5 * (score_after - score_before) * (!converted_src) as i32);
}

pub fn attack_mask(s: &State, source: usize, r: &Rules, g: &Geom, out: &mut [bool]) {
    out[..s.n].fill(false);
    attack_targets(s, source, r, g, |t| out[t] = true);
}

/// The true entries of attack_mask's row for `source`, visiting only the tiles within the unit's range.
pub fn attack_targets(s: &State, source: usize, r: &Rules, g: &Geom, mut emit: impl FnMut(usize)) {
    let u = &s.units;
    let cp = s.current_player;
    let kind = u.kind[source] as usize;
    let own = u.owner[source] == cp && !u.attacked[source] && u.health[source] > 0.0;
    let mut can = own && (!u.moved[source] || r.dash[kind]) && r.unit_attack[kind] >= 0.0;
    let infil = r.skills[kind][crate::rules::INFILTRATE];
    can = can && (r.unit_attack[kind] > 0.0 || kind == 10 || infil);
    let range = r.unit_range[kind];
    if !can || range < 1 {
        return;
    }
    let stamp = s.turn * s.np as i32 + cp;
    let explored = &s.tiles.explored[s.p()];
    let size = s.size as i32;
    let (sx, sy) = ((source % s.size) as i32, (source / s.size) as i32);
    for y in (sy - range).max(0)..=(sy + range).min(size - 1) {
        for x in (sx - range).max(0)..=(sx + range).min(size - 1) {
            let t = (y * size + x) as usize;
            if t == source || !explored[t] {
                continue;
            }
            let targets = if infil {
                State::foreign(s.cities.owner[t], cp) && !protected(s, cp, s.cities.owner[t])
                    && !(u.owner[t] != 0 && u.owner[t] != s.cities.owner[t]) && s.cities.infiltrated[t] != stamp
            } else {
                State::foreign(u.owner[t], cp) && !u.invisible[t] && !protected(s, cp, u.owner[t])
            };
            if targets {
                debug_assert!(g.d(source, t) > 0 && g.d(source, t) <= range);
                emit(t);
            }
        }
    }
}

#[inline]
fn terrain_bonus(s: &State, r: &Rules, i: usize, tech: &[bool; 128]) -> f32 {
    let (t, u, c) = (&s.tiles, &s.units, &s.cities);
    let terrain = t.terrain[i];
    let city = if c.owner[i] == u.owner[i] && r.fortify[u.kind[i] as usize] {
        if s.reward(i, 1) > 0 { 4.0 } else { 1.5 }
    } else {
        1.0
    };
    let bonus = if terrain == 1 || terrain == 2 {
        if tech[12] { 1.5 } else { 1.0 }
    } else if terrain == 5 {
        if tech[18] { 1.5 } else { 1.0 }
    } else if terrain == 4 {
        if tech[20] { 1.5 } else { 1.0 }
    } else {
        city
    };
    bonus * if u.poison[i] { 0.5 } else { 1.0 }
}

pub fn defense_bonus(s: &State, r: &Rules, i: usize) -> f32 {
    let tech = &s.players.tech[s.pclip(s.units.owner[i])];
    terrain_bonus(s, r, i, tech)
}

pub fn attack(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom) {
    let size = s.size;
    let cp = s.current_player;
    let kind = s.units.kind[source] as usize;
    let defender = s.units.kind[target] as usize;
    let terrain = s.tiles.terrain[target];
    let tech = s.players.tech[s.pwrap(s.units.owner[target])];
    let bonus = terrain_bonus(s, r, target, &tech);
    let hp_s = max_health(s, r, source);
    let hp_t = max_health(s, r, target);
    let dd = r.unit_defense[defender] * if s.units.poison[target] { 0.7 } else { 1.0 };
    let (damage, retaliation, _) = combat(r.unit_attack[kind], s.units.health[source], hp_s, dd, s.units.health[target], hp_t, bonus);
    let dead = s.units.health[target] - damage <= 0.0;
    // BattleHelpers.CanDefenderRetaliate: the attacker must also be one of the defender's attack
    // options, i.e. stand on a tile the defender's owner has explored
    let seen = s.tiles.explored[s.pclip(s.units.owner[target])][source];
    let retaliate = !r.skills[kind][SURPRISE] && !r.stiff[defender] && g.d(source, target) <= r.unit_range[defender] && seen;
    s.players.attacked[s.p()] = true;
    s.units.health[target] += -damage;
    s.units.boosted[target] = false;
    s.units.prev_x[source] = 2 * (source % size) as i32 - (target % size) as i32;
    s.units.prev_y[source] = 2 * (source / size) as i32 - (target / size) as i32;
    s.units.last_attack[source] = target as i32 + 1;
    if r.skills[kind][SPLASH] {
        crate::cov::hit(27);
        area_damage(s, source, target, r, g, false);
    }
    if dead {
        remove(s, target, r, Some(source), true);
    } else {
        s.units.health[source] += -(if retaliate { retaliation } else { 0.0 });
    }
    let attacker_dead = s.units.health[source] <= 0.0;
    if attacker_dead {
        crate::cov::hit(34);
        remove(s, source, r, Some(target), true);
    }
    let st = s.tiles.structure[target];
    let can_enter = terrain == 3 || terrain == 5 || (terrain == 4 && s.players.tech[s.p()][20])
        || (terrain <= 2 && (r.skills[kind][FLOAT] || st == 48 || (st == 8 && s.tiles.owner[target] == cp)));
    let advance = dead && !attacker_dead && r.unit_range[kind] < 2 && can_enter;
    if advance {
        crate::cov::hit(35);
        relocate(s, source, target, r, g, true, None);
    }
    let at = if advance { target } else { source };
    if !attacker_dead {
        let persist = r.persist[kind];
        let escape = r.escape[kind];
        s.units.attacks_performed[at] += 1;
        s.units.attacked[at] = !(dead && persist);
        s.units.moved[at] = !escape;
        if dead && persist {
            // AttackCommand.ExecuteV104: Persist leaves the unit able to attack only when it has
            // another target from where it now stands
            let mut more = false;
            attack_targets(s, at, r, g, |_| more = true);
            s.units.attacked[at] = !more;
        }
    }
}

pub fn convert(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let cp = s.current_player;
    let old = s.pwrap(s.units.owner[target]);
    let (cap, exists) = capital_index(s, cp);
    let mut fallback = 0usize;
    let mut best = i32::MAX;
    for i in 0..n {
        if s.cities.owner[i] == cp && s.cities.order[i] < best {
            best = s.cities.order[i];
            fallback = i;
        }
    }
    let tk = s.units.kind[target] as usize;
    let home = if r.skills[tk][INDEPENDENT] { 0 } else { (if exists { cap } else { fallback }) as i32 + 1 };
    let passenger = s.units.passenger[target];
    let cost = r.unit_cost[tk] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    s.players.score[old] = s.players.score[old].wrapping_sub(5 * cost * (!s.units.converted[target]) as i32);
    s.players.attacked[s.p()] = true;
    s.units.owner[target] = cp;
    s.units.converted[target] = true;
    s.units.home[target] = home;
    s.units.moved[target] = true;
    s.units.attacked[target] = true;
    exhaust(s, source);
    let radius = if s.tiles.terrain[target] == 4 || r.skills[tk][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(target, t) <= radius).collect();
    discover(s, &reveal, cp);
}

/// out[t*128 + kind]
/// The Raft upgrades (Rammership, Scoutship, Bomber): only summonable onto an own Raft.
pub const UPGRADES: u128 = 1 << 41 | 1 << 42 | 1 << 44;

/// Per-call inputs of summon_mask: units per home city and the unlocked, affordable kinds.
pub struct SummonCtx {
    counts: Vec<i32>,
    ok: u128,
}

impl SummonCtx {
    pub fn new(n: usize) -> SummonCtx {
        SummonCtx { counts: vec![0; n], ok: 0 }
    }

    pub fn prepare(&mut self, s: &State, r: &Rules) {
        let n = s.n;
        let u = &s.units;
        let cp = s.current_player;
        let p = s.p();
        self.counts.fill(0);
        for i in 0..n {
            if u.owner[i] == cp && u.home[i] >= 1 && (u.home[i] as usize) <= n {
                self.counts[(u.home[i] - 1) as usize] += 1;
            }
        }
        let mut unlocked = r.unit_static;
        let mut known = crate::rules::bits128(&s.players.tech[p]);
        while known != 0 {
            unlocked |= r.units_by_tech[known.trailing_zeros() as usize];
            known &= known - 1;
        }
        self.ok = unlocked & r.unit_cost_le.le(s.players.stars[p]);
    }
}

/// One row of military.summon_mask as a bitset of unit kinds.
#[inline]
pub fn summon_row(s: &State, ctx: &SummonCtx, t: usize) -> u128 {
    let (u, c) = (&s.units, &s.cities);
    let cp = s.current_player;
    let available = c.owner[t] == cp && ctx.counts[t] <= c.level[t] && u.owner[t] == 0;
    let friendly = s.tiles.owner[t] == cp || (s.tiles.owner[t] > 0 && s.players.peace[s.p()][s.pclip(s.tiles.owner[t])]);
    let upgrade_allowed = u.owner[t] == cp && u.kind[t] == 43 && friendly;
    (if available { ctx.ok & !UPGRADES } else { 0 }) | (if upgrade_allowed { ctx.ok & UPGRADES } else { 0 })
}

pub fn summon_mask(s: &State, r: &Rules, out: &mut [bool]) {
    let mut ctx = SummonCtx::new(s.n);
    ctx.prepare(s, r);
    for t in 0..s.n {
        let bits = summon_row(s, &ctx, t);
        for k in 0..128 {
            out[t * 128 + k] = bits >> k & 1 == 1;
        }
    }
}

pub fn summon(s: &mut State, source: usize, kind: usize, r: &Rules, g: &Geom, costs: bool) {
    let n = s.n;
    let cp = s.current_player;
    s.units.clear(source);
    s.units.owner[source] = cp;
    s.units.kind[source] = kind as i32;
    s.units.health[source] = r.unit_health[kind];
    s.units.home[source] = source as i32 + 1;
    s.units.moved[source] = true;
    s.units.attacked[source] = true;
    s.units.prev_x[source] = -1;
    s.units.prev_y[source] = -1;
    s.units.created[source] = s.turn;
    let p = s.p();
    s.players.stars[p] -= r.unit_cost[kind] * costs as i32;
    s.players.score[p] = s.players.score[p].wrapping_add(5 * r.unit_cost[kind]);
    // TrainAction: ExploreFromTile with UnitDataExtensions.GetSightRange (Scout units see 2 tiles)
    let radius = if s.tiles.terrain[source] == 4 || r.skills[kind][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(source, t) <= radius).collect();
    discover(s, &reveal, cp);
}

/// The unit on `i` may be promoted: 3 kills, not yet a veteran, not Static
/// (UnitDataExtensions.CanBePromoted): a boat is judged by its passenger, so a non-Static passenger is promoted at sea.
#[inline]
pub fn promotable(s: &State, r: &Rules, i: usize) -> bool {
    let u = &s.units;
    let judged = if u.passenger[i] > 0 { u.passenger[i] } else { u.kind[i] } as usize;
    !u.veteran[i] && u.kills[i] >= 3 && !r.stat[judged]
}

pub fn promote(s: &mut State, source: usize, r: &Rules) {
    let hp = max_health(s, r, source) + 5.0;
    s.units.veteran[source] = true;
    s.units.health[source] = hp;
}

pub fn disband(s: &mut State, source: usize, r: &Rules) {
    let p = s.p();
    disband_of(s, source, r, p);
}

/// Disband the unit on `source`, refunding half its cost to seat index `p` (its owner).
pub fn disband_of(s: &mut State, source: usize, r: &Rules, p: usize) {
    let kind = s.units.kind[source];
    let passenger = s.units.passenger[source];
    let underlying = if passenger > 0 {
        passenger
    } else if kind == 39 {
        38
    } else if kind == 40 {
        37
    } else if kind == 45 {
        12
    } else {
        kind
    };
    let refund = r.unit_cost[underlying as usize].div_euclid(2);
    remove(s, source, r, None, false);
    s.players.stars[p] += refund;
}

pub fn heal_others(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let cp = s.current_player;
    for i in 0..s.n {
        if g.d(source, i) == 1 && s.units.owner[i] == cp {
            if !s.units.poison[i] {
                let mh = max_health(s, r, i);
                s.units.health[i] = (s.units.health[i] + 4.0).min(mh);
            }
            s.units.poison[i] = false;
        }
    }
    exhaust(s, source);
}

fn sign(v: i32) -> i32 {
    v.signum()
}

/// military.push: reference alternating push directions with its centre tie breaks.
pub fn push(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let (n, size) = (s.n, s.size);
    let sz = size as i32;
    let u = &s.units;
    let owner = u.owner[source];
    if owner <= 0 {
        return;
    }
    let (x, y) = ((source % size) as i32, (source / size) as i32);
    let center = sz / 2;
    let cx = sign(center - x);
    let mut cy = sign(center - y);
    if cx == 0 && cy == 0 {
        cy = 1;
    }
    let (px, py) = (u.prev_x[source], u.prev_y[source]);
    let prev_valid = px >= 0 && py >= 0 && (px != x || py != y);
    let flip = if owner != s.current_player { -1 } else { 1 };
    let (mx, my) = (sign(x - px) * flip, sign(y - py) * flip);
    let la = u.last_attack[source] - 1;
    let kind = u.kind[source] as usize;
    let ranged = r.unit_range[kind] > 1 && u.attacked[source] && !(r.escape[kind] && u.moved[source]);
    let (ax, ay) = (sign(la.rem_euclid(sz) - x) * flip, sign(la.div_euclid(sz) - y) * flip);
    let use_move = prev_valid && (la < 0 || u.moved[source]);
    let dx = if la >= 0 && ranged { ax } else if use_move { mx } else { cx };
    let dy = if la >= 0 && ranged { ay } else if use_move { my } else { cy };
    let dirs: [[i32; 2]; 8] = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
    let base = dirs.iter().position(|d| d[0] == dx && d[1] == dy).unwrap_or(0) as i32;
    let offsets = [0i32, -1, 1, -2, 2, -3, 3, 4];
    let mut choices: [[i32; 2]; 8] = [[0; 2]; 8];
    for k in 0..8 {
        let d = dirs[(base + offsets[k]).rem_euclid(8) as usize];
        choices[k] = [d[0] + x, d[1] + y];
    }
    for (a, b) in [(1usize, 2usize), (3, 4), (5, 6)] {
        let dista = (choices[a][0] - center).pow(2) + (choices[a][1] - center).pow(2);
        let distb = (choices[b][0] - center).pow(2) + (choices[b][1] - center).pow(2);
        let va = (choices[a][0] - choices[a][1]).abs();
        let vb = (choices[b][0] - choices[b][1]).abs();
        if distb < dista || (distb == dista && vb < va) {
            choices.swap(a, b);
        }
    }
    crate::cov::hit(29);
    let po = s.pclip(owner);
    let mut target = None;
    let mut first_index = 0usize;
    for (k, ch) in choices.iter().enumerate() {
        let idx = (ch[1] * sz + ch[0]).clamp(0, n as i32 - 1) as usize;
        if k == 0 {
            first_index = idx;
        }
        let in_board = ch[0] >= 0 && ch[0] < sz && ch[1] >= 0 && ch[1] < sz;
        let terrain = s.tiles.terrain[idx];
        let st = s.tiles.structure[idx];
        let mut valid = in_board && u.owner[idx] == 0;
        valid = valid && (terrain == 3 || terrain == 5 || (terrain == 4 && s.players.tech[po][20])
            || (terrain <= 2 && (r.skills[kind][FLOAT] || st == 48 || (st == 8 && s.tiles.owner[idx] == owner))));
        if valid {
            target = Some(idx);
            break;
        }
    }
    let _ = first_index;
    match target {
        Some(t) => relocate(s, source, t, r, g, true, None),
        None => remove(s, source, r, None, true),
    }
}

pub fn superunit(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    push(s, source, r, g);
    summon(s, source, 12, r, g, false);
}

pub fn area_damage(s: &mut State, source: usize, center: usize, r: &Rules, g: &Geom, stomp: bool) {
    let n = s.n;
    let src_owner = s.units.owner[source];
    let src_kind = s.units.kind[source] as usize;
    let hp_s = max_health(s, r, source);
    let ah = s.units.health[source];
    let aa = r.unit_attack[src_kind];
    let mut damage = vec![0f32; n];
    let mut dead = vec![false; n];
    let mut any_hit = false;
    for i in 0..n {
        let u = &s.units;
        let targets = g.d(center, i) == 1 && u.owner[i] != 0 && u.owner[i] != src_owner && !protected(s, src_owner, u.owner[i]);
        if targets {
            let dd = r.unit_defense[u.kind[i] as usize] * if u.poison[i] { 0.7 } else { 1.0 };
            let mut dmg = combat(aa, ah, hp_s, dd, u.health[i], max_health(s, r, i), defense_bonus(s, r, i)).2;
            let _ = stomp;
            dmg = dmg.floor();
            damage[i] = dmg;
            dead[i] = u.health[i] - dmg <= 0.0;
            any_hit |= dmg > 0.0;
        }
    }
    for i in 0..n {
        s.units.health[i] -= damage[i];
    }
    for i in 0..n {
        if dead[i] {
            crate::cov::hit(37);
            remove(s, i, r, Some(source), true);
        }
    }
    if any_hit {
        hostile(s);
    }
}

pub fn upgrade(s: &mut State, source: usize, kind: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let before = max_health(s, r, source);
    let old = s.units.kind[source] as usize;
    s.units.kind[source] = kind as i32;
    let hp = (max_health(s, r, source) - (before - s.units.health[source])).max(1.0);
    s.units.health[source] = hp;
    let p = s.p();
    s.players.stars[p] -= r.unit_cost[kind];
    s.players.score[p] = s.players.score[p].wrapping_add(5 * (r.unit_cost[kind] - r.unit_cost[old]));
    let radius = if r.skills[kind][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(source, t) <= radius).collect();
    discover(s, &reveal, s.current_player);
}

````


## engine-rs/src/pathfinding.rs

Source SHA-256: 6488ec8f80522182f3cf980a643665ea8c969791153db83d5d3fd92623b59164

````rust
//! gpu/pathfinding.py: reference-ordered shortest path (Rust BinaryHeap tie behaviour) for fog reveal.

use crate::geom::Geom;
use crate::kernels::{bridge_blocks, forest_step, hide_passable, peace_unit, movement_flags, INF, MV_FOREST, MV_LAND, MV_MOUNTAIN, MV_OWN_PORT, MV_PASS, MV_ROAD, MV_WATER, MV_ZOC};
use crate::rules::{Rules, CARRY, CREEP, FLOAT, HIDE, INFILTRATE, SCOUT};
use crate::state::State;

struct Heap {
    ids: Vec<i32>,
    costs: Vec<i32>,
    count: usize,
}

impl Heap {
    fn push(&mut self, index: i32, cost: i32) {
        let mut at = self.count;
        self.ids[at] = index;
        self.costs[at] = cost;
        while at > 0 && self.costs[at] < self.costs[(at - 1) / 2] {
            let parent = (at - 1) / 2;
            self.ids.swap(at, parent);
            self.costs.swap(at, parent);
            at = parent;
        }
        self.count += 1;
    }

    fn pop(&mut self) -> (i32, i32) {
        let (index, cost) = (self.ids[0], self.costs[0]);
        self.count -= 1;
        let count = self.count;
        let (last_id, last_cost) = (self.ids[count], self.costs[count]);
        let mut at = 0usize;
        while 2 * at + 1 < count {
            let left = 2 * at + 1;
            let right = left + 1;
            let child = if right < count && self.costs[right] <= self.costs[left] { right } else { left };
            self.ids[at] = self.ids[child];
            self.costs[at] = self.costs[child];
            at = child;
        }
        self.ids[at] = last_id;
        self.costs[at] = last_cost;
        while at > 0 && self.costs[at] < self.costs[(at - 1) / 2] {
            let p = (at - 1) / 2;
            self.ids.swap(at, p);
            self.costs.swap(at, p);
            at = p;
        }
        (index, cost)
    }
}

/// Reusable buffers for path search and reveal (one per GameBatch, inside core::Scratch).
pub struct PathScratch {
    heap: Heap,
    flags: Vec<u8>,
    dist: Vec<i32>,
    parent: Vec<i32>,
    /// path tiles from the target back to (excluding) the source: path[k] has rank k
    pub path: Vec<usize>,
    pub reveal: Vec<bool>,
}

impl PathScratch {
    pub fn new(n: usize) -> PathScratch {
        PathScratch { heap: Heap { ids: vec![0; 8 * n + 1], costs: vec![INF; 8 * n + 1], count: 0 }, flags: vec![0; n],
                      dist: vec![INF; n], parent: vec![-1; n], path: Vec::with_capacity(n), reveal: vec![false; n] }
    }
}

/// pathfinding.path_details: the reference-ordered shortest path from `source` to `target` (Dijkstra with the
/// reference heap and its tie behaviour, neighbours in kernels' fixed order, movement data's passable/terminal/
/// road/budget rules). Fills `ps.path` (target first, source excluded) and returns whether the target was found;
/// the reference's mask is {source, target} plus the path and its rank is the index in `ps.path` (source: len).
pub fn search(s: &State, source: usize, target: usize, r: &Rules, g: &Geom, ps: &mut PathScratch) -> bool {
    let n = s.n;
    let u = &s.units;
    let cp = s.current_player;
    let owner = u.owner[source];
    let kind = u.kind[source] as usize;
    let sk = &r.skills[kind];
    let (aquatic, creep, carry) = (sk[FLOAT], sk[CREEP], sk[CARRY]);
    let no_zoc = creep || sk[INFILTRATE] || u.invisible[source];
    let budget = (r.unit_movement[kind] + u.boosted[source] as i32) * 2;
    movement_flags(s, owner, g, &mut ps.flags);
    let flags = &ps.flags;
    let term_bits = (if no_zoc { 0 } else { MV_ZOC }) | MV_MOUNTAIN | (if aquatic { 0 } else { MV_OWN_PORT });
    let terminal = |f: u8| f & term_bits != 0 || (f & (MV_FOREST | MV_ROAD) == MV_FOREST && !creep) || (carry && f & MV_WATER == 0);
    let pass_need = if aquatic { MV_PASS } else { MV_PASS | MV_LAND };
    let hide = sk[HIDE];
    let passable = |i: usize| (flags[i] & pass_need == pass_need && (u.owner[i] == 0 || u.owner[i] == cp || u.invisible[i]))
        || ((hide || peace_unit(s, owner, i)) && (aquatic || flags[i] & MV_LAND != 0) && hide_passable(s, owner, i));
    let (heap, dist, parent) = (&mut ps.heap, &mut ps.dist, &mut ps.parent);
    dist[..n].fill(INF);
    parent[..n].fill(-1);
    heap.ids[0] = source as i32;
    heap.costs[0] = 0;
    heap.count = 1;
    dist[source] = 0;
    let mut found = false;
    while heap.count > 0 && !found {
        let (at, cost) = heap.pop();
        let at = at as usize;
        found = at == target;
        let expand = !found && cost <= dist[at] && (at == source || !terminal(flags[at])) && cost < budget;
        if expand {
            let road_at = flags[at] & MV_ROAD != 0 && !creep;
            for k in 0..8 {
                if !g.nb_valid[at * 8 + k] {
                    continue;
                }
                let dest = g.nb[at * 8 + k] as usize;
                let road_edge = road_at && flags[dest] & MV_ROAD != 0;
                let newcost = forest_step(creep, road_edge, flags[dest], cost + if road_edge { 1 } else { 2 }, budget);
                if passable(dest) && dest != source && newcost < dist[dest] && !bridge_blocks(s, owner, at, dest) {
                    heap.push(dest as i32, newcost);
                    dist[dest] = newcost;
                    parent[dest] = at as i32;
                }
            }
        }
    }
    ps.path.clear();
    if found {
        let mut at = target as i32;
        while at >= 0 && at as usize != source {
            ps.path.push(at as usize);
            at = parent[at as usize];
        }
    }
    found
}

/// pathfinding.reveal_path into `ps.reveal` for the traveled tiles {source} + ps.path[from..] + extra: every tile
/// within 1 (2 from a mountain, or for scouts) of one of them, visiting only the boxes around them.
pub fn reveal_tiles(s: &State, source: usize, from: usize, extra: Option<usize>, r: &Rules, ps: &mut PathScratch) {
    let size = s.size as i32;
    let scout = r.skills[s.units.kind[source] as usize][SCOUT];
    let out = &mut ps.reveal[..s.n];
    out.fill(false);
    for m in std::iter::once(source).chain(ps.path[from..].iter().copied()).chain(extra) {
        let radius = if s.tiles.terrain[m] == 4 || scout { 2 } else { 1 };
        let (mx, my) = ((m % s.size) as i32, (m / s.size) as i32);
        for y in (my - radius).max(0)..=(my + radius).min(size - 1) {
            for x in (mx - radius).max(0)..=(mx + radius).min(size - 1) {
                out[(y * size + x) as usize] = true;
            }
        }
    }
}

````


## engine-rs/src/prof.rs

Source SHA-256: 93f3c9ed5d4198fc0498458954bef258b36d81a6a1048181878295bcbee4c86f

````rust
//! Optional wall-clock profile of tick and step phases (diagnostic, off unless GameBatch.profile_phases(True)).

use std::cell::{Cell, RefCell};
use std::time::Instant;

pub const KINDS: [&str; 18] = [
    "research", "recover", "move", "end_turn", "attack", "summon", "harvest", "forest", "promote", "disband", "reward",
    "build", "destroy", "capture", "heal", "diplomacy", "resign", "unsupported",
];
/// Phase ids: 0..18 step validation per kind, 18..36 step execution per kind, then the fixed phases below.
pub const VALIDATE: usize = 0;
pub const EXECUTE: usize = 18;
pub const STEP_PRE: usize = 36;
pub const STEP_POST: usize = 37;
pub const TICK_STEP: usize = 38;
pub const TICK_RESET: usize = 39;
pub const TICK_MASKS: usize = 40;
pub const TICK_OBSERVE: usize = 41;
pub const TICK_OBSERVE_OPP: usize = 42;
pub const TICK_FINAL: usize = 43;
pub const TICK_ALLOC: usize = 44;
pub const TICK_OUTPUT: usize = 45;
pub const N: usize = 46;

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static TOTAL: RefCell<[(u64, u64); N]> = const { RefCell::new([(0, 0); N]) };
}

pub fn enable(on: bool) {
    ON.with(|c| c.set(on));
    TOTAL.with(|t| *t.borrow_mut() = [(0, 0); N]);
}

/// Run `f`, adding its wall time to phase `id` when profiling is on.
#[inline]
pub fn time<T>(id: usize, f: impl FnOnce() -> T) -> T {
    if !ON.with(|c| c.get()) {
        return f();
    }
    let t0 = Instant::now();
    let out = f();
    let ns = t0.elapsed().as_nanos() as u64;
    TOTAL.with(|t| {
        let mut t = t.borrow_mut();
        t[id].0 += 1;
        t[id].1 += ns;
    });
    out
}

pub fn name(id: usize) -> String {
    match id {
        i if i < EXECUTE => format!("validate.{}", KINDS[i]),
        i if i < STEP_PRE => format!("execute.{}", KINDS[i - EXECUTE]),
        STEP_PRE => "step.pre".into(),
        STEP_POST => "step.post".into(),
        TICK_STEP => "tick.step".into(),
        TICK_RESET => "tick.reset".into(),
        TICK_MASKS => "tick.masks".into(),
        TICK_OBSERVE => "tick.observe".into(),
        TICK_OBSERVE_OPP => "tick.observe_opp".into(),
        TICK_FINAL => "tick.final_view".into(),
        TICK_ALLOC => "tick.alloc".into(),
        _ => "tick.output".into(),
    }
}

/// (name, calls, total ms) for every phase that ran.
pub fn snapshot() -> Vec<(String, u64, f64)> {
    TOTAL.with(|t| {
        t.borrow().iter().enumerate().filter(|(_, v)| v.0 > 0).map(|(i, v)| (name(i), v.0, v.1 as f64 / 1e6)).collect()
    })
}

````


## engine-rs/src/rules.rs

Source SHA-256: 8d9d960d8967bd639d20180c602bdb82b205a7254c86d56481ff74b6cc8a0f36

````rust
//! Rule tables, built from gpu/rules.json exactly as gpu/rules.py::load_rules("native") does.
//! Nothing here is re-typed by hand; the JSON is the single source.

use serde::Deserialize;

pub const FLOAT: usize = 0;
pub const CARRY: usize = 1;
pub const SCOUT: usize = 2;
pub const CREEP: usize = 3;
pub const HIDE: usize = 4;
pub const INDEPENDENT: usize = 5;
pub const SURPRISE: usize = 6;
pub const SPLASH: usize = 7;
pub const STOMP: usize = 8;
pub const INFILTRATE: usize = 9;

#[derive(Deserialize)]
struct UnitRow {
    id: usize,
    name: String,
    health: f32,
    attack: f32,
    defense: f32,
    movement: i32,
    range: i32,
    cost: i32,
    skills: Vec<String>,
}

#[derive(Deserialize)]
struct TechRow {
    id: i64,
    tier: i32,
    requires: Option<i64>,
    next: Vec<i64>,
    structures: Vec<usize>,
}

#[derive(Deserialize)]
struct StructRow {
    id: usize,
    cost: Option<i32>,
    population: Option<i32>,
    score: Option<i32>,
    stars: Option<i32>,
    resource: Option<i32>,
    terrain: Vec<usize>,
    adjacent: Vec<usize>,
    per_city: bool,
    per_tribe: bool,
    tribe: Option<i64>,
}

#[derive(Deserialize)]
struct Table {
    engine_revision: String,
    units: Vec<UnitRow>,
    technologies: Vec<TechRow>,
    structures: Vec<StructRow>,
}

/// For a cost table: the kinds whose cost is at most a given number of stars, as a bitset (`le(stars)`).
pub struct CostIndex {
    min: i32,
    table: Vec<u128>,
}

impl CostIndex {
    pub fn new(cost: &[i32; 128]) -> CostIndex {
        let (min, max) = (*cost.iter().min().unwrap(), *cost.iter().max().unwrap());
        let table = (min..=max).map(|s| (0..128).fold(0u128, |b, k| b | ((cost[k] <= s) as u128) << k)).collect();
        CostIndex { min, table }
    }
    #[inline]
    pub fn le(&self, stars: i32) -> u128 {
        if stars < self.min {
            0
        } else {
            self.table[((stars - self.min) as usize).min(self.table.len() - 1)]
        }
    }
}

pub struct StructureRules {
    pub cost: [i32; 128],
    pub population: [i32; 128],
    pub score: [i32; 128],
    pub stars: [i32; 128],
    pub terrain: [[bool; 16]; 128],
    pub adjacent: [[bool; 128]; 128],
    pub any_adjacent: [bool; 128],
    pub resource: [i32; 128],
    pub per_city: [bool; 128],
    pub per_tribe: [bool; 128],
    pub allowed: [bool; 128],
    pub unlock: [[bool; 128]; 128], // [tech, structure]
    /// Derived bitsets: unlock_by[k] bit i = unlock[i][k]; adjacent_rev[j] bit k = adjacent[k][j];
    /// terrain_bits[t] bit k = terrain[k][t]; no_adjacent_bits / per_city_bits bit k = !any_adjacent[k] / per_city[k].
    pub unlock_by: [u128; 128],
    pub adjacent_rev: [u128; 128],
    pub terrain_bits: [u128; 16],
    pub no_adjacent_bits: u128,
    pub per_city_bits: u128,
    /// tech_unlocks[i]: structures tech i unlocks; static_ok: allowed && cost >= 0 && no resource; per_tribe_bits;
    /// cost_le: kinds affordable with a given number of stars.
    pub tech_unlocks: [u128; 128],
    pub static_ok: u128,
    pub per_tribe_bits: u128,
    pub cost_le: CostIndex,
}

pub struct Rules {
    pub unit_health: [f32; 128],
    pub unit_attack: [f32; 128],
    pub unit_defense: [f32; 128],
    pub unit_movement: [i32; 128],
    pub unit_range: [i32; 128],
    pub dash: [bool; 128],
    pub unit_supported: [bool; 128],
    pub tech_tier: [i32; 128],
    pub tech_edges: [[bool; 128]; 128],
    pub tech_exists: [bool; 128],
    pub unit_cost: [i32; 128],
    pub fortify: [bool; 128],
    pub stiff: [bool; 128],
    pub persist: [bool; 128],
    pub escape: [bool; 128],
    pub stat: [bool; 128],
    pub unit_tech: [i32; 128],
    pub structures: StructureRules,
    pub free_tech: [bool; 128],
    pub free_requires: [i32; 128],
    pub skills: [[bool; 10]; 128],
    pub engine_revision: String,
    /// Derived: tech_pred[k] bit i = tech_edges[i][k] (research reachability).
    pub tech_pred: [u128; 128],
    /// Derived: bits128(unit_supported), bits128(tech_exists).
    pub supported_bits: u128,
    pub tech_exists_bits: u128,
    /// Derived: tech_succ[i] bit k = tech_edges[i][k]; units_by_tech[i]: unit kinds tech i unlocks (cost >= 1,
    /// unit_tech < 127); unit_static: the Warrior (kind 2) when summonable without a tech; unit_cost_le.
    pub tech_succ: [u128; 128],
    pub units_by_tech: [u128; 128],
    pub unit_static: u128,
    pub unit_cost_le: CostIndex,
}

/// Bit k set for every true entry of a 128-wide flag row.
#[inline]
pub fn bits128(row: &[bool; 128]) -> u128 {
    let mut b = 0u128;
    for (k, &v) in row.iter().enumerate() {
        b |= (v as u128) << k;
    }
    b
}

impl Rules {
    pub fn from_json(text: &str) -> Result<Rules, String> {
        let data: Table = serde_json::from_str(text).map_err(|e| format!("rules.json: {e}"))?;
        let mut r = Rules {
            unit_health: [0.0; 128],
            unit_attack: [0.0; 128],
            unit_defense: [0.0; 128],
            unit_movement: [0; 128],
            unit_range: [0; 128],
            dash: [false; 128],
            unit_supported: [false; 128],
            tech_tier: [1; 128],
            tech_edges: [[false; 128]; 128],
            tech_exists: [false; 128],
            unit_cost: [0; 128],
            fortify: [false; 128],
            stiff: [false; 128],
            persist: [false; 128],
            escape: [false; 128],
            stat: [false; 128],
            unit_tech: [127; 128],
            structures: StructureRules {
                cost: [-1; 128],
                population: [0; 128],
                score: [0; 128],
                stars: [0; 128],
                terrain: [[false; 16]; 128],
                adjacent: [[false; 128]; 128],
                any_adjacent: [false; 128],
                resource: [0; 128],
                per_city: [false; 128],
                per_tribe: [false; 128],
                allowed: [false; 128],
                unlock: [[false; 128]; 128],
                unlock_by: [0; 128],
                adjacent_rev: [0; 128],
                terrain_bits: [0; 16],
                no_adjacent_bits: 0,
                per_city_bits: 0,
                tech_unlocks: [0; 128],
                static_ok: 0,
                per_tribe_bits: 0,
                cost_le: CostIndex { min: 0, table: vec![0] },
            },
            free_tech: [false; 128],
            free_requires: [-1; 128],
            skills: [[false; 10]; 128],
            engine_revision: data.engine_revision.clone(),
            tech_pred: [0; 128],
            supported_bits: 0,
            tech_exists_bits: 0,
            tech_succ: [0; 128],
            units_by_tech: [0; 128],
            unit_static: 0,
            unit_cost_le: CostIndex { min: 0, table: vec![0] },
        };
        let core_names = [
            "Raft", "Rammership", "Scoutship", "Bomber", "Juggernaut", "MindBender", "Warrior", "Rider", "Knight",
            "Defender", "Catapult", "Archer", "Swordsman", "Giant", "Cloak", "Dagger", "Dinghy", "Pirate",
        ];
        let skill_names = ["Float", "Carry", "Scout", "Creep", "Hide", "Independent", "Surprise", "Splash", "Stomp", "Infiltrate"];
        for u in &data.units {
            let i = u.id;
            if i >= 128 {
                return Err("unit id out of range".into());
            }
            r.unit_health[i] = u.health;
            r.unit_attack[i] = u.attack;
            r.unit_defense[i] = u.defense;
            r.unit_movement[i] = u.movement;
            r.unit_range[i] = u.range;
            r.dash[i] = u.skills.iter().any(|s| s == "Dash");
            r.unit_supported[i] = core_names.contains(&u.name.as_str());
            for (k, name) in skill_names.iter().enumerate() {
                r.skills[i][k] = u.skills.iter().any(|s| s == name);
            }
            r.unit_cost[i] = u.cost;
            r.fortify[i] = u.skills.iter().any(|s| s == "Fortify");
            r.stiff[i] = u.skills.iter().any(|s| s == "Stiff");
            r.persist[i] = u.skills.iter().any(|s| s == "Persist");
            r.escape[i] = u.skills.iter().any(|s| s == "Escape");
            r.stat[i] = u.skills.iter().any(|s| s == "Static");
        }
        for t in &data.technologies {
            if t.id < 0 {
                continue;
            }
            let i = t.id as usize;
            if i >= 128 {
                return Err("tech id out of range".into());
            }
            r.tech_tier[i] = t.tier;
            r.tech_exists[i] = true;
            let mut others = t.next.clone();
            if let Some(req) = t.requires {
                others.push(req);
            }
            for o in others {
                if o >= 0 && (o as usize) < 128 {
                    r.tech_edges[i][o as usize] = true;
                }
            }
        }
        for i in 25..128 {
            r.tech_exists[i] = false;
        }
        r.tech_exists[38] = true;
        r.tech_exists[44] = true;
        for (unit, tech) in [(2, 0), (3, 1), (4, 3), (5, 7), (8, 17), (9, 18), (10, 22), (11, 24), (38, 38), (41, 14), (42, 13), (44, 44)] {
            r.unit_tech[unit] = tech;
        }
        for st in &data.structures {
            let i = st.id;
            if i >= 128 {
                return Err("structure id out of range".into());
            }
            let s = &mut r.structures;
            s.cost[i] = st.cost.unwrap_or(-1);
            s.population[i] = st.population.unwrap_or(0);
            s.score[i] = st.score.unwrap_or(0);
            s.stars[i] = st.stars.unwrap_or(0);
            s.resource[i] = st.resource.unwrap_or(0);
            for &t in &st.terrain {
                if t < 16 {
                    s.terrain[i][t] = true;
                }
            }
            for &a in &st.adjacent {
                if a < 128 {
                    s.adjacent[i][a] = true;
                    s.any_adjacent[i] = true;
                }
            }
            s.per_city[i] = st.per_city;
            s.per_tribe[i] = st.per_tribe;
            s.allowed[i] = st.tribe.is_none() || st.tribe == Some(4);
        }
        for t in &data.technologies {
            if t.id >= 0 {
                for &st in &t.structures {
                    if st < 128 {
                        r.structures.unlock[t.id as usize][st] = true;
                    }
                }
            }
        }
        for t in &data.technologies {
            let i = t.id;
            if (1..=24).contains(&i) || i == 38 || i == 44 || i == 48 || i == 121 {
                r.free_tech[i as usize] = true;
                r.free_requires[i as usize] = t.requires.map(|v| v as i32).unwrap_or(-1);
            }
        }
        for i in 0..128 {
            r.free_tech[i] &= r.tech_exists[i];
        }
        let rs = &mut r.structures;
        for k in 0..128 {
            for i in 0..128 {
                r.tech_pred[k] |= (r.tech_edges[i][k] as u128) << i;
                rs.unlock_by[k] |= (rs.unlock[i][k] as u128) << i;
                rs.adjacent_rev[i] |= (rs.adjacent[k][i] as u128) << k;
            }
            for t in 0..16 {
                rs.terrain_bits[t] |= (rs.terrain[k][t] as u128) << k;
            }
            rs.no_adjacent_bits |= (!rs.any_adjacent[k] as u128) << k;
            rs.per_city_bits |= (rs.per_city[k] as u128) << k;
        }
        r.supported_bits = bits128(&r.unit_supported);
        r.tech_exists_bits = bits128(&r.tech_exists);
        for i in 0..128 {
            r.tech_succ[i] = bits128(&r.tech_edges[i]);
            r.structures.tech_unlocks[i] = bits128(&r.structures.unlock[i]);
        }
        let rs = &mut r.structures;
        for k in 0..128 {
            rs.static_ok |= ((rs.allowed[k] && rs.cost[k] >= 0 && rs.resource[k] == 0) as u128) << k;
            rs.per_tribe_bits |= (rs.per_tribe[k] as u128) << k;
            let summonable = r.unit_cost[k] >= 1 && r.unit_tech[k] < 127;
            if summonable {
                r.units_by_tech[r.unit_tech[k] as usize] |= 1u128 << k;
                if k == 2 {
                    r.unit_static |= 1u128 << k;
                }
            }
        }
        rs.cost_le = CostIndex::new(&rs.cost);
        r.unit_cost_le = CostIndex::new(&r.unit_cost);
        Ok(r)
    }
}

````


## engine-rs/src/state.rs

Source SHA-256: 123ecb98519ce15002b77fed09b72758ac4531e24830d8f6f92137448902c98b

````rust
//! One game's state: the same fields as gpu/state.py::State, struct-of-arrays, runtime board size and seat count.

/// Most seats a game can have.
pub const MAXP: usize = 4;

#[derive(Clone)]
pub struct Tiles {
    pub terrain: Vec<i32>,
    pub owner: Vec<i32>,
    pub capital: Vec<i32>,
    pub city: Vec<i32>,
    pub road: Vec<bool>,
    pub structure: Vec<i32>,
    pub resource: Vec<i32>,
    pub explored: [Vec<bool>; MAXP],
    pub structure_level: Vec<i32>,
    pub structure_turn: Vec<i32>,
}

#[derive(Clone)]
pub struct Units {
    pub owner: Vec<i32>,
    pub kind: Vec<i32>,
    pub health: Vec<f32>,
    pub veteran: Vec<bool>,
    pub moved: Vec<bool>,
    pub attacked: Vec<bool>,
    pub kills: Vec<i32>,
    pub poison: Vec<bool>,
    pub frozen: Vec<bool>,
    pub boosted: Vec<bool>,
    pub home: Vec<i32>,
    pub created: Vec<i32>,
    pub converted: Vec<bool>,
    pub attacks_performed: Vec<i32>,
    pub prev_x: Vec<i32>,
    pub prev_y: Vec<i32>,
    pub last_attack: Vec<i32>,
    pub passenger: Vec<i32>,
    pub invisible: Vec<bool>,
}

impl Units {
    /// jax.tree.map(lambda x: x.at[i].set(0), units)
    pub fn clear(&mut self, i: usize) {
        self.owner[i] = 0;
        self.kind[i] = 0;
        self.health[i] = 0.0;
        self.veteran[i] = false;
        self.moved[i] = false;
        self.attacked[i] = false;
        self.kills[i] = 0;
        self.poison[i] = false;
        self.frozen[i] = false;
        self.boosted[i] = false;
        self.home[i] = 0;
        self.created[i] = 0;
        self.converted[i] = false;
        self.attacks_performed[i] = 0;
        self.prev_x[i] = 0;
        self.prev_y[i] = 0;
        self.last_attack[i] = 0;
        self.passenger[i] = 0;
        self.invisible[i] = false;
    }
    /// jax.tree.map(lambda x: x.at[t].set(x[s]).at[s].set(0), units)
    pub fn move_unit(&mut self, s: usize, t: usize) {
        macro_rules! mv {
            ($($f:ident),*) => { $( { let v = self.$f[s]; self.$f[t] = v; } )* };
        }
        mv!(owner, kind, health, veteran, moved, attacked, kills, poison, frozen, boosted, home, created, converted,
            attacks_performed, prev_x, prev_y, last_attack, passenger, invisible);
        self.clear(s);
    }
}

#[derive(Clone)]
pub struct Cities {
    pub owner: Vec<i32>,
    pub level: Vec<i32>,
    pub population: Vec<i32>,
    pub progress: Vec<i32>,
    pub production: Vec<i32>,
    pub border: Vec<i32>,
    pub rewards: Vec<i32>,     // [n*9]
    pub territory: Vec<bool>,  // [n*n], row = city tile
    pub connected: Vec<bool>,
    pub order: Vec<i32>,
    pub infiltrated: Vec<i32>,
    pub riot: Vec<bool>,
}

#[derive(Clone)]
pub struct Players {
    pub stars: [i32; MAXP],
    pub score: [i32; MAXP],
    pub tech: [[bool; 128]; MAXP],
    pub tech_turn: [[i32; 128]; MAXP],
    pub known: [[bool; MAXP]; MAXP],
    pub kills: [i32; MAXP],
    pub casualties: [i32; MAXP],
    pub pacifist: [i32; MAXP],
    pub attacked: [bool; MAXP],
    pub built: [[bool; 128]; MAXP],
    pub killed_turn: [i32; MAXP],
    pub resigned_turn: [i32; MAXP],
    pub killer: [i32; MAXP],
    pub peace: [[bool; MAXP]; MAXP],
    pub embassy: [[bool; MAXP]; MAXP],
    pub peace_offer: [[bool; MAXP]; MAXP],
    pub truce_breaker: [[i32; MAXP]; MAXP],
    pub hostile_until: [i32; MAXP],
    pub monuments: [[bool; 128]; MAXP],
    /// Engine-internal (4-player games): 1-based order in which seats left the game (killed or resigned), 0 while
    /// in. Two-player games never set it.
    pub out_order: [i32; MAXP],
}

#[derive(Clone)]
pub struct State {
    pub size: usize,
    pub n: usize,
    /// Seats in this game (2 or 4). Per-player arrays hold MAXP rows; rows >= np stay zero. Two-player games are
    /// bitwise the reference engine's; more seats follow the multi-player rules in docs/four-player-*.md.
    pub np: usize,
    pub tiles: Tiles,
    pub units: Units,
    pub cities: Cities,
    pub players: Players,
    pub current_player: i32,
    pub turn: i32,
    pub version: i32,
    pub state_version: i32,
    pub actions: i32,
    pub terminated: bool,
    pub truncated: bool,
    pub fog: bool,
    pub reference_max_turns: i32,
    pub core_supported: bool,
    pub seed: u32,
    pub initial_seed: u32,
    /// Engine-internal (not a reference State field): structures::remember_monuments has already been applied to
    /// exactly this state, so applying it again is a no-op. Set by core::step; any other way of changing a state
    /// (reset, import) leaves or sets it false.
    pub monuments_settled: bool,
}

impl State {
    pub fn empty(size: usize) -> State {
        State::empty_players(size, 2)
    }

    pub fn empty_players(size: usize, np: usize) -> State {
        assert!((2..=MAXP).contains(&np));
        let n = size * size;
        let z = || vec![0i32; n];
        let b = || vec![false; n];
        State {
            size,
            n,
            np,
            tiles: Tiles {
                terrain: z(), owner: z(), capital: z(), city: vec![-1; n], road: b(), structure: z(), resource: z(),
                explored: [b(), b(), b(), b()], structure_level: z(), structure_turn: z(),
            },
            units: Units {
                owner: z(), kind: z(), health: vec![0.0; n], veteran: b(), moved: b(), attacked: b(), kills: z(),
                poison: b(), frozen: b(), boosted: b(), home: z(), created: z(), converted: b(), attacks_performed: z(),
                prev_x: z(), prev_y: z(), last_attack: z(), passenger: z(), invisible: b(),
            },
            cities: Cities {
                owner: z(), level: z(), population: z(), progress: z(), production: z(), border: z(),
                rewards: vec![0; n * 9], territory: vec![false; n * n], connected: b(), order: z(), infiltrated: z(), riot: b(),
            },
            players: Players {
                stars: [0; MAXP], score: [0; MAXP], tech: [[false; 128]; MAXP], tech_turn: [[0; 128]; MAXP],
                known: [[false; MAXP]; MAXP], kills: [0; MAXP], casualties: [0; MAXP], pacifist: [0; MAXP],
                attacked: [false; MAXP], built: [[false; 128]; MAXP], killed_turn: [0; MAXP], resigned_turn: [0; MAXP],
                killer: [0; MAXP], peace: [[false; MAXP]; MAXP], embassy: [[false; MAXP]; MAXP],
                peace_offer: [[false; MAXP]; MAXP], truce_breaker: [[0; MAXP]; MAXP], hostile_until: [0; MAXP],
                monuments: [[false; 128]; MAXP], out_order: [0; MAXP],
            },
            current_player: 1,
            turn: 0,
            version: 115,
            state_version: 0,
            actions: 0,
            terminated: false,
            truncated: false,
            fog: true,
            reference_max_turns: 45,
            core_supported: true,
            seed: 0,
            initial_seed: 0,
            monuments_settled: false,
        }
    }

    /// Seat index of the player to move (0..np).
    #[inline]
    pub fn p(&self) -> usize {
        (self.current_player - 1) as usize
    }
    /// jnp.clip(owner - 1, 0, np - 1)
    #[inline]
    pub fn pclip(&self, owner: i32) -> usize {
        (owner - 1).clamp(0, self.np as i32 - 1) as usize
    }
    /// JAX gather semantics for `tech[owner - 1]`: a negative index wraps to the last row.
    #[inline]
    pub fn pwrap(&self, owner: i32) -> usize {
        let np = self.np as i32;
        let i = owner - 1;
        (if i < 0 { i + np } else { i }).clamp(0, np - 1) as usize
    }
    /// Seat k (0-based) is still in the game: not killed and not resigned.
    #[inline]
    pub fn alive(&self, k: usize) -> bool {
        self.players.killed_turn[k] == 0 && self.players.resigned_turn[k] == 0
    }
    /// Seats still in the game.
    pub fn alive_count(&self) -> usize {
        (0..self.np).filter(|&k| self.alive(k)).count()
    }
    /// `owner` is another seat's id (not neutral, not the mover's).
    #[inline]
    pub fn foreign(owner: i32, me: i32) -> bool {
        owner > 0 && owner != me
    }
    #[inline]
    pub fn territory(&self, city: usize, tile: usize) -> bool {
        self.cities.territory[city * self.n + tile]
    }
    #[inline]
    pub fn reward(&self, city: usize, k: usize) -> i32 {
        self.cities.rewards[city * 9 + k]
    }
    #[inline]
    pub fn reward_count(&self, city: usize) -> i32 {
        self.cities.rewards[city * 9..city * 9 + 9].iter().sum()
    }
    pub fn pending_reward(&self) -> bool {
        let cp = self.current_player;
        (0..self.n).any(|i| self.cities.owner[i] == cp && self.reward_count(i) < self.cities.level[i] - 1)
    }
}

````


## engine-rs/src/structures.rs

Source SHA-256: d75b0c978e82965d8338c5e2537239ef6ada9e1168ead47a1caf44d8900574b1

````rust
//! gpu/structures.py: buildings, monuments, temples, road networks, income.

use crate::economy::{add_population, add_population_vector, economic_tiles, in_territory};
use crate::geom::Geom;
use crate::rules::{bits128, Rules};
use crate::state::State;

#[inline]
pub fn temple(kind: i32) -> bool {
    (17..=20).contains(&kind)
}

#[inline]
pub fn score_of(s: &State, r: &Rules, t: usize) -> i32 {
    let k = s.tiles.structure[t];
    if temple(k) { 100 * s.tiles.structure_level[t] } else { r.structures.score[k as usize] }
}

/// Native multiplier levels from adjacent owned compatible improvements.
pub fn production_levels(s: &State, r: &Rules, g: &Geom) -> Vec<i32> {
    let n = s.n;
    let t = &s.tiles;
    let mut levels = vec![0i32; n];
    for i in 0..n {
        let k = t.structure[i] as usize;
        let multiplier = r.structures.any_adjacent[k] && t.structure[i] != 50;
        if multiplier {
            let mut count = 0;
            for &j in &g.adj[i] {
                let j = j as usize;
                if t.owner[i] == t.owner[j] && t.owner[i] > 0 && r.structures.adjacent[k][t.structure[j] as usize] {
                    count += 1;
                }
            }
            levels[i] = count;
        } else {
            levels[i] = t.structure_level[i];
        }
    }
    if !t.structure.contains(&50) {
        return levels;
    }
    let mut out = levels.clone();
    for i in 0..n {
        if t.structure[i] == 50 {
            let mut sum = 0;
            for &j in &g.adj[i] {
                let j = j as usize;
                if t.owner[i] == t.owner[j] && t.owner[i] > 0 && r.structures.adjacent[50][t.structure[j] as usize] {
                    sum += levels[j];
                }
            }
            out[i] = sum.min(8);
        }
    }
    out
}

pub fn population_contribution(s: &State, r: &Rules, g: &Geom) -> Vec<i32> {
    contribution_from(s, r, &production_levels(s, r, g))
}

/// population_contribution given production_levels(s).
fn contribution_from(s: &State, r: &Rules, levels: &[i32]) -> Vec<i32> {
    (0..s.n)
        .map(|i| {
            let k = s.tiles.structure[i] as usize;
            r.structures.population[k] * if r.structures.any_adjacent[k] { levels[i] } else { 1 }
        })
        .collect()
}

/// structures.adjust_building_population(before, after): `before_contribution` is population_contribution of the
/// pre-action state.
pub fn adjust_building_population(before_contribution: &[i32], s: &mut State, r: &Rules, g: &Geom) {
    let n = s.n;
    let levels = production_levels(s, r, g);   // s is unchanged until structure_level is set below
    let after = contribution_from(s, r, &levels);
    let mut amount = vec![0i32; n];
    for tile in 0..n {
        let city = s.tiles.city[tile];
        if city >= 0 && (city as usize) < n {
            let city = city as usize;
            if s.cities.owner[city] == s.tiles.owner[tile] {
                amount[city] += after[tile] - before_contribution[tile];
            }
        }
    }
    s.tiles.structure_level = levels;
    if amount.iter().any(|&a| a != 0) {   // all-zero amounts change nothing
        add_population_vector(s, &amount, 1, true);
        add_population_vector(s, &amount, 2, true);
    }
}

pub fn capital_index(s: &State, owner: i32) -> (usize, bool) {
    let mut best = i32::MAX;
    let mut idx = 0usize;
    let mut exists = false;
    for i in 0..s.n {
        if s.cities.owner[i] == owner && s.tiles.capital[i] == owner {
            exists = true;
            if s.cities.order[i] < best {
                best = s.cities.order[i];
                idx = i;
            }
        }
    }
    (idx, exists)
}

pub fn connections(s: &mut State, owner: i32, g: &Geom, native: bool) {
    let n = s.n;
    let (cap, exists) = capital_index(s, owner);
    let t = &s.tiles;
    let po = (owner - 1) as usize;
    let friendly: Vec<bool> = (0..n)
        .map(|i| t.owner[i] == 0 || t.owner[i] == owner || (native && s.players.peace[po][s.pclip(t.owner[i])]))
        .collect();
    let nodes: Vec<bool> = (0..n)
        .map(|i| (t.road[i] || t.structure[i] == 48 || t.structure[i] == 8 || (s.cities.owner[i] > 0 && friendly[i])) && friendly[i])
        .collect();
    let ports: Vec<usize> = (0..n).filter(|&i| t.structure[i] == 8).collect();
    let water: Vec<bool> = (0..n).map(|i| t.terrain[i] == 1 || t.terrain[i] == 2).collect();
    // port_edges[p] = ports reachable from port p through friendly water within the reference's distance rule.
    let mut port_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    for &p in &ports {
        seen.fill(false);
        seen[p] = true;
        stack.clear();
        stack.push(p);
        while let Some(x) = stack.pop() {
            if g.d(p, x) < 4 {
                for &y in &g.adj[x] {
                    let y = y as usize;
                    if !seen[y] && water[y] && friendly[y] {
                        seen[y] = true;
                        stack.push(y);
                    }
                }
            }
        }
        for &q in &ports {
            if seen[q] && g.d(p, q) <= 4 {
                port_edges[p].push(q);
            }
        }
    }
    let mut reached = vec![false; n];
    if exists {
        reached[cap] = true;
        stack.clear();
        stack.push(cap);
        while let Some(x) = stack.pop() {
            for &y in &g.adj[x] {
                let y = y as usize;
                if !reached[y] && nodes[y] {
                    reached[y] = true;
                    stack.push(y);
                }
            }
            for &q in &port_edges[x] {
                if !reached[q] {
                    reached[q] = true;
                    stack.push(q);
                }
            }
        }
    }
    let mut amount = vec![0i32; n];
    let mut total = 0i32;
    for c in 0..n {
        let own = s.cities.owner[c] == owner && c != cap && exists;
        if own {
            let delta = reached[c] as i32 - s.cities.connected[c] as i32;
            amount[c] = delta;
            total += delta;
            s.cities.connected[c] = reached[c];
        }
    }
    amount[cap] += total;
    let pop_owner = if native { owner } else { s.current_player };
    add_population_vector(s, &amount, pop_owner, native);
}

pub fn monument_conditions(s: &State, owner: i32, r: &Rules, g: &Geom, out: &mut [bool; 128]) {
    let c = &s.cities;
    let (mut connected, mut level5) = (0usize, false);
    for i in 0..s.n {
        if c.owner[i] == owner {
            connected += c.connected[i] as usize;
            level5 |= c.level[i] >= 5;
        }
    }
    let bits = monument_bits(s, owner, r, g, connected, level5);
    for k in 0..128 {
        out[k] = bits >> k & 1 == 1;
    }
}

/// monument_conditions as a bitset, given the owner's connected-city count and whether a city of theirs has level 5.
pub fn monument_bits(s: &State, owner: i32, r: &Rules, g: &Geom, connected: usize, level5: bool) -> u128 {
    let p = &s.players;
    let pi = (owner - 1) as usize;
    let tech = &p.tech[pi];
    let all_tech = bits128(tech) | !r.tech_exists_bits == !0;
    let explorer = g.corners.iter().all(|&t| s.tiles.explored[pi][t]) && s.fog;
    let bit = |k: u32, v: bool| (v as u128) << k;
    // the Network task (Grand Bazaar) is enabled by the first city connection, not by the Roads tech
    bit(23, tech[21] && p.pacifist[pi] >= 5) | bit(24, all_tech) | bit(25, connected >= 5)
        | bit(26, tech[5] && p.stars[pi] >= 100) | bit(27, p.kills[pi] >= 10) | bit(28, level5) | bit(29, explorer)
}

pub fn remember_monuments(s: &mut State, r: &Rules, g: &Geom) {
    // one pass over the cities for every seat's connected counts and level-5 flags
    let c = &s.cities;
    let np = s.np as i32;
    let (mut connected, mut level5) = ([0usize; crate::state::MAXP], [false; crate::state::MAXP]);
    for i in 0..s.n {
        let o = c.owner[i];
        if (1..=np).contains(&o) {
            connected[(o - 1) as usize] += c.connected[i] as usize;
            level5[(o - 1) as usize] |= c.level[i] >= 5;
        }
    }
    for owner in 1..=np {
        let pi = (owner - 1) as usize;
        let earned = monument_bits(s, owner, r, g, connected[pi], level5[pi]);
        let m = &mut s.players.monuments[pi];
        for k in 0..128 {
            m[k] |= earned >> k & 1 == 1;
        }
    }
}

/// Per-call inputs of structures.build_mask shared by every tile (BuildCtx::prepare).
pub struct BuildCtx {
    /// economic_tiles(s), filled by prepare(), and the same tiles as a list
    pub econ: Vec<bool>,
    pub econ_list: Vec<u16>,
    own_cities: Vec<usize>,
    /// structures present in the territory of each own city (bit = structure kind)
    present: Vec<u128>,
    unlocked: u128,
    monuments: u128,
}

const ROAD: u128 = 1 << 3;
const BRIDGE: u128 = 1 << 48;

impl BuildCtx {
    pub fn road_unlocked(&self) -> bool {
        self.unlocked & ROAD != 0
    }

    pub fn new(n: usize) -> BuildCtx {
        BuildCtx { econ: vec![false; n], econ_list: Vec::with_capacity(n), own_cities: Vec::with_capacity(n), present: Vec::with_capacity(n), unlocked: 0,
                   monuments: 0 }
    }

    /// Everything of build_mask that does not depend on the tile, including economic_tiles(s) into `econ` (one
    /// pass over each own city's territory row gives both the economy and the structures present per city).
    pub fn prepare(&mut self, s: &State, r: &Rules, g: &Geom) {
        let n = s.n;
        let (t, c, p) = (&s.tiles, &s.cities, &s.players);
        let cp = s.current_player;
        let pi = s.p();
        let rs = &r.structures;
        self.own_cities.clear();
        self.present.clear();
        let econ = &mut self.econ[..n];
        econ.fill(false);
        let mut tribe_has = 0u128;
        let (mut connected, mut level5) = (0usize, false);
        for city in 0..n {
            if c.owner[city] != cp {
                continue;
            }
            connected += c.connected[city] as usize;
            level5 |= c.level[city] >= 5;
            let mut pr = 0u128;
            let row = &c.territory[city * n..city * n + n];
            for tile in 0..n {
                if row[tile] {
                    pr |= 1u128 << t.structure[tile];
                    econ[tile] = true;
                }
            }
            tribe_has |= pr;
            self.own_cities.push(city);
            self.present.push(pr);
        }
        self.econ_list.clear();
        for tile in 0..n {
            // GameLogicData.CanBuild: the tile itself must be the player's, whatever the city rows say
            econ[tile] = econ[tile] && crate::economy::unit_allows_work(s, tile) && t.owner[tile] == cp;
            if econ[tile] {
                self.econ_list.push(tile as u16);
            }
        }
        let mut tech_ok = 0u128;
        let mut known = bits128(&p.tech[pi]);
        while known != 0 {
            tech_ok |= rs.tech_unlocks[known.trailing_zeros() as usize];
            known &= known - 1;
        }
        self.unlocked = tech_ok & rs.static_ok & rs.cost_le.le(p.stars[pi]) & !(rs.per_tribe_bits & tribe_has);
        let earned = monument_bits(s, cp, r, g, connected, level5) | bits128(&p.monuments[pi]);
        // ImprovementAbility Unique (PlayerState.builtUniqueImprovements): a monument is blocked only by the player
        // having built it, not by one standing in a captured city's territory
        self.monuments = earned & !bits128(&p.built[pi]);
    }
}

/// Buildable structure kinds on `tile` as a bitset: one row of structures.build_mask (native profile), with each
/// of its four terms evaluated for all 128 kinds at once as bitset algebra:
///   (a) unlocked & terrain & empty & adjacency & city room   (b) neutral road   (c) the tile's resource building
///   (d) earned monuments on free land; then the bridge span test.
pub fn build_row(s: &State, r: &Rules, g: &Geom, ctx: &BuildCtx, tile: usize) -> u128 {
    let (t, c, p) = (&s.tiles, &s.cities, &s.players);
    let cp = s.current_player;
    let pi = s.p();
    let rs = &r.structures;
    let terrain = rs.terrain_bits[t.terrain[tile].clamp(0, 15) as usize];
    let b = (if neutral_road(s, r, ctx, tile) { ROAD } else { 0 }) | (if neutral_bridge(s, r, ctx, tile) { BRIDGE } else { 0 });
    if !ctx.econ[tile] {
        return b; // every other term carries econ[tile]
    }
    let free = t.structure[tile] == 0;
    let mut adjacency = rs.no_adjacent_bits;
    for &j in &g.adj[tile] {
        let j = j as usize;
        if t.owner[j] == cp {
            adjacency |= rs.adjacent_rev[t.structure[j] as usize];
        }
    }
    // a per-city kind has room iff some own city holding this tile lacks it
    let mut holders = 0usize;
    let mut present_all = !0u128;
    for (k, &city) in ctx.own_cities.iter().enumerate() {
        if c.territory[city * s.n + tile] {
            holders += 1;
            present_all &= ctx.present[k];
        }
    }
    let room = if holders > 0 { !rs.per_city_bits | !present_all } else { 0 };
    // TileData.HasRoad is true on every City tile, villages included, so no road goes there
    let road_ok = !t.road[tile] && c.owner[tile] == 0 && t.structure[tile] != 1;
    let empty = (if free { !ROAD } else { 0 }) | (if road_ok { ROAD } else { 0 });
    let a = ctx.unlocked & terrain & empty & adjacency & room;
    let (res_tech, res_build) = match t.resource[tile] { 2 => (8, 5), 5 => (23, 21), 7 => (0, 37), _ => (0, 0) };
    let resource = free && res_build > 0 && p.tech[pi][res_tech] && p.stars[pi] >= rs.cost[res_build];
    let d = if free { terrain & ctx.monuments } else { 0 };
    let mut v = a | b | ((resource as u128) << res_build) | d;
    if v & BRIDGE != 0 && bridge_span(s, tile) == 0 {
        v &= !BRIDGE;
    }
    v
}

/// TileData.GetPossibleBridgeDirection (from the true terrain): 1 for a north-south span (land north and
/// south), 2 for east-west (land east and west), 0 for none; north-south wins when both hold.
#[inline]
pub fn bridge_span(s: &State, tile: usize) -> u8 {
    let size = s.size;
    let (x, y) = (tile % size, tile / size);
    let land = |i: usize| s.tiles.terrain[i] != 1 && s.tiles.terrain[i] != 2;
    if y > 0 && y + 1 < size && land(tile - size) && land(tile + size) {
        1
    } else if x > 0 && x + 1 < size && land(tile - 1) && land(tile + 1) {
        2
    } else {
        0
    }
}

/// GameLogicData.CanBuild (a Bridge is Freelance like a road): an unlocked bridge on a neutral, explored,
/// structure-less water tile with a span, free of other players' visible units.
#[inline]
pub fn neutral_bridge(s: &State, r: &Rules, ctx: &BuildCtx, tile: usize) -> bool {
    let t = &s.tiles;
    ctx.unlocked & BRIDGE != 0 && t.owner[tile] == 0 && t.explored[s.p()][tile] && t.structure[tile] == 0
        && crate::economy::unit_allows_work(s, tile)
        && r.structures.terrain_bits[t.terrain[tile].clamp(0, 15) as usize] & BRIDGE != 0 && bridge_span(s, tile) != 0
}

/// Term (b) of build_row: an unlocked road on a neutral, explored, road-less tile whose terrain allows roads.
#[inline]
pub fn neutral_road(s: &State, r: &Rules, ctx: &BuildCtx, tile: usize) -> bool {
    let t = &s.tiles;
    ctx.unlocked & ROAD != 0 && t.owner[tile] == 0 && t.explored[s.p()][tile] && !t.road[tile]
        && t.structure[tile] != 1
        && crate::economy::unit_allows_work(s, tile)
        && r.structures.terrain_bits[t.terrain[tile].clamp(0, 15) as usize] & ROAD != 0
}

/// build_mask into out[t*128 + kind] (structures.build_mask, native profile).
pub fn build_mask(s: &State, r: &Rules, g: &Geom, out: &mut [bool]) {
    let mut ctx = BuildCtx::new(s.n);
    ctx.prepare(s, r, g);
    for tile in 0..s.n {
        let bits = build_row(s, r, g, &ctx, tile);
        for kind in 0..128 {
            out[tile * 128 + kind] = bits >> kind & 1 == 1;
        }
    }
}

/// One entry of build_mask (step validation).
pub fn build_legal(s: &State, r: &Rules, g: &Geom, ctx: &mut BuildCtx, tile: usize, kind: usize) -> bool {
    ctx.prepare(s, r, g);
    build_row(s, r, g, ctx, tile) >> kind & 1 == 1
}

pub fn build(s: &mut State, target: usize, kind: usize, r: &Rules, g: &Geom) {
    let before = population_contribution(s, r, g);
    let rs = &r.structures;
    let road = kind == 3;
    let pi = s.p();
    if !road {
        s.tiles.structure[target] = kind as i32;
        s.tiles.structure_level[target] = 1;
        s.tiles.structure_turn[target] = s.turn;
    }
    s.tiles.road[target] |= road;
    s.players.stars[pi] -= rs.cost[kind].max(0);
    if kind == 3 || kind == 8 || kind == 48 {
        connections(s, s.current_player, g, true);
    }
    let score = if temple(kind as i32) { 100 } else { rs.score[kind] };
    let owner = s.tiles.owner[target];
    if owner > 0 {
        let o = s.pclip(owner);
        s.players.score[o] = s.players.score[o].wrapping_add(score);
    }
    s.players.built[pi][kind] |= score > 0;
    adjust_building_population(&before, s, r, g);
}

pub fn destroy_mask(s: &State, out: &mut [bool]) {
    economic_tiles(s, out);
    let econ = out[..s.n].to_vec();
    destroy_mask_with(s, &econ, out);
}

/// Structures Destroy removes (ImprovementDataExtensions.IsDestructable): not cities, ruins or lighthouses.
#[inline]
pub fn destructible(k: i32) -> bool {
    k > 0 && k != 1 && k != 2 && k != 47
}

/// destroy_mask given economic_tiles(s).
pub fn destroy_mask_with(s: &State, econ: &[bool], out: &mut [bool]) {
    let tech3 = s.players.tech[s.p()][3];
    for t in 0..s.n {
        out[t] = econ[t] && destructible(s.tiles.structure[t]) && tech3;
    }
}

pub fn destroy(s: &mut State, target: usize, r: &Rules, g: &Geom) {
    let before = population_contribution(s, r, g);
    let kind = s.tiles.structure[target];
    let score = score_of(s, r, target);
    let owner = s.tiles.owner[target];
    s.tiles.structure[target] = 0;
    s.tiles.structure_level[target] = 0;
    s.tiles.structure_turn[target] = 0;
    s.tiles.road[target] = s.tiles.road[target] && kind != 3 && kind != 48;
    if owner > 0 {
        let o = s.pclip(owner);
        s.players.score[o] = s.players.score[o].wrapping_sub(score);
    }
    adjust_building_population(&before, s, r, g);
    if kind == 3 || kind == 8 || kind == 48 {
        connections(s, s.current_player, g, true);
    }
}

pub fn start_turn(s: &mut State, owner: i32) {
    let n = s.n;
    let mut gain = 0i32;
    for i in 0..n {
        let t = &mut s.tiles;
        if temple(t.structure[i]) && t.owner[i] == owner {
            let expected = (1 + (s.turn - t.structure_turn[i]).div_euclid(2)).clamp(1, 5);
            let new = expected.max(t.structure_level[i]);
            gain += new - t.structure_level[i];
            t.structure_level[i] = new;
        }
    }
    let o = (owner - 1) as usize;
    s.players.score[o] = s.players.score[o].wrapping_add(gain * 100);
}

pub fn income(s: &State, owner: i32, r: &Rules, g: &Geom) -> i32 {
    let n = s.n;
    let (t, c, u) = (&s.tiles, &s.cities, &s.units);
    // the territory bonus is the markets' value; without a market it is zero
    let markets = t.structure.contains(&50);
    let market_value: Vec<i32> = if markets {
        let levels = production_levels(s, r, g);
        (0..n).map(|i| if t.structure[i] == 50 { levels[i] * r.structures.stars[50] } else { 0 }).collect()
    } else {
        Vec::new()
    };
    let mut total = 0i32;
    if markets {
        // ResourceDataUtils.CalculateIncomeFor sums TileDataExtensions.CalculateWork over every tile: each owned
        // market pays on its own, whatever happens to its city (a siege or negative progress stops only the city)
        total += (0..n).filter(|&i| t.owner[i] == owner).map(|i| market_value[i]).sum::<i32>();
    }
    for city in 0..n {
        if c.owner[city] != owner {
            continue;
        }
        // any unit of another player on the city stops its work, a hidden one included
        let siege = (u.owner[city] != 0 && u.owner[city] != c.owner[city]) || c.riot[city];
        if siege {
            continue;
        }
        let mut production = c.production[city] + (t.capital[city] == c.owner[city] && t.capital[city] > 0) as i32;
        if c.progress[city] < 0 {
            production = (production + c.progress[city]).max(0);
        }
        total += production;
    }
    let p = (owner - 1) as usize;
    let np = s.np;
    let embassy_count = (0..np).map(|k| s.players.embassy[p][k] as i32).sum::<i32>()
        + (0..np).map(|k| s.players.embassy[k][p] as i32).sum::<i32>();
    // embassies exist only in two-seat games
    total + embassy_count * if np == 2 && s.players.peace[p][1 - p] { 4 } else { 2 }
}

pub fn _unused(s: &State) {
    let mut v = vec![false; s.n];
    in_territory(s, &mut v);
    let _ = add_population;
}

````


## src/polytopia_bot/maps/variants.py

Source SHA-256: 12d90b1c3e9dee5aaba866151123f75ac851260a50f09076185f847377d11b32

````python
"""Variant Bardur map generator: water, mountains, chokepoints and island maps, for 2 seats (1v1) or 4.

players=4 places one capital per quadrant and generalizes the capital-dependent shapes (continents carve
channels along the boundaries between capitals' regions; shared-landmass presets bridge every capital to the
first). Two-seat maps are bitwise those of bardur-variants-v2; the ridge preset is two-seat only.

Generates the same Layout fields as gpu.mapgen.layout (terrain, structure,
resource, capitals) so a board can initialize the frozen engine unchanged. It
lives outside gpu/ on purpose: the engine digest that checkpoints verify covers
only gpu/. Deterministic per (seed, preset). Host-side NumPy: one map costs well
under a millisecond, so actors generate maps between games.

Terrain ids: 1 shallow water, 2 ocean, 3 field, 4 mountain, 5 forest.
Structures: 1 village, 2 ruin, 47 lighthouse. Resources: 1 animal (forest),
3 fish (shallow water), 5 metal (mountain), 6 fruit (field), 8 starfish.
Distances are Chebyshev, as in the engine.
"""
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np

# v2 adds the 18x18 board (scaling below); 11x11 and 14x14 maps are bitwise those of v1.
GENERATOR_REVISION = 'bardur-variants-v2'
SIZES = (11, 14, 18)
RUINS = {11: 4, 14: 5, 18: 9}          # as gpu.mapgen (reference ruin counts)
MIN_CAPITAL_DISTANCE = {11: 6, 14: 6, 18: 8}
MIN_CAPITAL_DISTANCE_4 = {14: 5, 18: 6}  # four seats: pairwise, one capital per quadrant
# 18x18 scaling (324 tiles, 1.65x the 14x14 area). Noise smoothness is in tiles, so by default
# features keep their size and their count grows with the area (more islands, lakes, clumps).
# Single-landmass shapes scale with the board instead, and lakes grow a little.
LARGE_LAND_SCALE = {'pangea': 18 / 14, 'continents': 18 / 14, 'lakes': (18 / 14) ** .5}
LARGE_RIDGE_PASSES = 1                 # one more pass through the longer wall (3-4 instead of 2-3)
SHALLOW, OCEAN, FIELD, MOUNTAIN, FOREST = 1, 2, 3, 4, 5
VILLAGE, RUIN, LIGHTHOUSE = 1, 2, 47
ANIMAL, FISH, METAL, FRUIT, STARFISH = 1, 3, 5, 6, 8


class Layout(NamedTuple):
    terrain: np.ndarray    # [n] int32
    structure: np.ndarray  # [n] int32
    resource: np.ndarray   # [n] int32
    capitals: np.ndarray   # [players] int32 tile indices


@dataclass(frozen=True)
class Preset:
    name: str
    title: str
    land: tuple = (.9, .95)          # target land fraction range
    mountain: tuple = (.12, .16)     # fraction of free land that is mountain
    forest: tuple = (.3, .4)         # fraction of free non-mountain land that is forest
    land_scale: float = 2.5          # noise smoothness of the coastline (larger = bigger blobs)
    shape: str = 'noise'             # noise | continents | pangea | islands | lakes | ridge
    ridge_passes: tuple = (2, 3)
    village_spacing: int = 3         # minimum Chebyshev distance between villages/capitals
    village_keep: tuple = (.8, 1.)   # fraction of possible villages actually placed
    shared_landmass: bool = False    # force a land route between capitals
    note: str = ''


PRESETS = {p.name: p for p in [
    Preset('drylands', 'Drylands (current-like)', land=(.93, .96), mountain=(.12, .16), forest=(.35, .45),
           land_scale=1.0, note='Almost all land, light mountains. Close to today\'s training maps.'),
    Preset('lakes', 'Lakes', land=(.6, .72), mountain=(.15, .22), forest=(.3, .4), shape='lakes', land_scale=1.6,
           shared_landmass=True, note='One landmass broken up by inland lakes; fishing and short boat hops matter.'),
    Preset('pangea', 'Pangea', land=(.5, .62), mountain=(.2, .28), forest=(.3, .4), shape='pangea', land_scale=2.2,
           shared_landmass=True, note='One central continent ringed by ocean; a land war with long coasts.'),
    Preset('continents', 'Continents', land=(.45, .58), mountain=(.2, .28), forest=(.3, .4), shape='continents',
           land_scale=2.0, note='Each tribe starts on its own continent; reaching the enemy needs ports and boats.'),
    Preset('archipelago', 'Archipelago', land=(.32, .42), mountain=(.15, .22), forest=(.3, .4), shape='islands',
           land_scale=1.1, village_spacing=2, note='Many small islands; expansion is by sea.'),
    Preset('waterworld', 'Water world', land=(.18, .26), mountain=(.1, .16), forest=(.25, .35), shape='islands',
           land_scale=.8, village_spacing=2, village_keep=(.9, 1.), note='Tiny islands in open ocean; navy decides.'),
    Preset('highlands', 'Highlands', land=(.85, .93), mountain=(.32, .42), forest=(.25, .35), land_scale=1.4,
           shared_landmass=True, note='Mountain-heavy land with valleys; slow movement, defensive terrain, metal.'),
    Preset('ridge', 'Ridge with passes', land=(.82, .9), mountain=(.04, .07), forest=(.3, .4), shape='ridge',
           land_scale=1.4, shared_landmass=True, note='A mountain wall between the tribes with 2–3 passes: chokepoints.'),
    Preset('jungle', 'Jungle', land=(.78, .88), mountain=(.1, .15), forest=(.65, .8), land_scale=1.3,
           shared_landmass=True, note='Dense forest: slow movement, ambush cover, lots of animals.'),
]}
# 'mixed' samples a preset uniformly, then its parameters; the proposed training default.
# 'classic' (the frozen engine generator, gpu.mapgen) joins the mix when maps are wired into training.
MIXED = tuple(PRESETS)


def _blur(a, sigma):
    if sigma <= 0:
        return a
    r = max(1, int(3 * sigma))
    k = np.exp(-.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, r, mode='reflect')
    a = np.apply_along_axis(lambda v: np.convolve(v, k, 'valid'), 0, pad)
    return np.apply_along_axis(lambda v: np.convolve(v, k, 'valid'), 1, a)


def _noise(rng, size, scale, octaves=3):
    total, amp = np.zeros((size, size)), 1.
    for o in range(octaves):
        layer = _blur(rng.normal(size=(size, size)), scale / (2 ** o))
        layer = (layer - layer.mean()) / (layer.std() + 1e-9)
        total += amp * layer
        amp *= .5
    return (total - total.mean()) / (total.std() + 1e-9)


def _cheb(size):
    ids = np.arange(size * size)
    x, y = ids % size, ids // size
    return np.maximum(np.abs(x[:, None] - x[None, :]), np.abs(y[:, None] - y[None, :]))


def _components(mask, size):
    """4-connected land components (boats are needed between them)."""
    label = -np.ones(size * size, int)
    k = 0
    for start in np.flatnonzero(mask.ravel()):
        if label[start] >= 0:
            continue
        stack = [start]
        label[start] = k
        while stack:
            i = stack.pop()
            x, y = i % size, i // size
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    j = ny * size + nx
                    if mask.ravel()[j] and label[j] < 0:
                        label[j] = k
                        stack.append(j)
        k += 1
    return label, k


def _line(a, b, size):
    ax, ay, bx, by = a % size, a // size, b % size, b // size
    steps = max(abs(bx - ax), abs(by - ay), 1)
    return [int(round(ay + (by - ay) * t / steps)) * size + int(round(ax + (bx - ax) * t / steps)) for t in range(steps + 1)]


def sample_parameters(rng, preset, size=14):
    pick = lambda r: float(rng.uniform(*r))
    extra = LARGE_RIDGE_PASSES if size == 18 else 0
    return dict(land=pick(preset.land), mountain=pick(preset.mountain), forest=pick(preset.forest),
                village_keep=pick(preset.village_keep),
                passes=int(rng.integers(preset.ridge_passes[0] + extra, preset.ridge_passes[1] + extra + 1)))


def _four_capitals(rng, size, d, edge, x, y):
    """One capital per quadrant, 2+ tiles from the edge and from the quadrant lines, pairwise at least
    MIN_CAPITAL_DISTANCE_4 apart (redrawn until they are; the farthest-spread draw if never)."""
    half = size // 2
    quads = [np.flatnonzero((edge >= 2) & ((x < half) == (q % 2 == 0)) & ((y < half) == (q < 2))
                            & (np.abs(x - (half - .5)) >= 2) & (np.abs(y - (half - .5)) >= 2)) for q in range(4)]
    best, best_gap = None, -1
    for _ in range(200):
        caps = np.array([int(rng.choice(c)) for c in quads])
        gap = min(int(d[a, b]) for i, a in enumerate(caps) for b in caps[i + 1:])
        if gap >= MIN_CAPITAL_DISTANCE_4[size]:
            return caps
        if gap > best_gap:
            best, best_gap = caps, gap
    return best


def generate(seed, preset='mixed', size=14, players=2):
    """Returns (Layout, info). info carries the preset, sampled parameters and map statistics."""
    if size not in SIZES:
        raise ValueError('variant maps support 11, 14 or 18')
    if players not in (2, 4) or (players == 4 and size not in MIN_CAPITAL_DISTANCE_4):
        raise ValueError('variant maps support 2 seats, or 4 seats on 14x14 and 18x18')
    if players == 4 and preset == 'ridge':
        raise ValueError('the ridge preset is two-seat only')
    rng = np.random.default_rng([int(seed), sum(map(ord, preset)), 20261019] + ([players] if players != 2 else []))
    if preset == 'mixed':
        mixed = MIXED if players == 2 else tuple(p for p in MIXED if p != 'ridge')
        preset = mixed[int(rng.integers(len(mixed)))]
    p = PRESETS[preset]
    if size == 18 and p.shape in LARGE_LAND_SCALE:
        p = replace(p, land_scale=p.land_scale * LARGE_LAND_SCALE[p.shape])
    q = sample_parameters(rng, p, size)
    n = size * size
    d = _cheb(size)
    ids = np.arange(n)
    x, y = ids % size, ids // size
    edge = np.minimum(np.minimum(x, size - 1 - x), np.minimum(y, size - 1 - y))

    # --- capitals: opposite sides, at least ~half the board apart, 2 tiles from the edge (four seats: quadrants)
    if players == 4:
        caps = _four_capitals(rng, size, d, edge, x, y)
        c0, c1 = int(caps[0]), int(caps[1])
    else:
        caps = None
    orient = rng.integers(3) if caps is None else 0  # 0 horizontal, 1 vertical, 2 diagonal
    def side(i):
        if orient == 0:
            return (x < size // 2 - 1) if i == 0 else (x > size // 2)
        if orient == 1:
            return (y < size // 2 - 1) if i == 0 else (y > size // 2)
        return (x + y < size - 3) if i == 0 else (x + y > size + 1)
    if caps is None:
        flip = rng.random() < .5
        sides = [side(1 if flip else 0), side(0 if flip else 1)]
        ok0 = np.flatnonzero(sides[0] & (edge >= 2))
        ok1 = np.flatnonzero(sides[1] & (edge >= 2))
        far = d[np.ix_(ok0, ok1)] >= MIN_CAPITAL_DISTANCE[size]
        if far.any():
            a_idx, b_idx = np.nonzero(far)
            k = int(rng.integers(len(a_idx)))
            c0, c1 = int(ok0[a_idx[k]]), int(ok1[b_idx[k]])
        else:  # never expected on the supported boards; take the farthest pair
            a_idx, b_idx = np.unravel_index(np.argmax(d[np.ix_(ok0, ok1)]), (len(ok0), len(ok1)))
            c0, c1 = int(ok0[a_idx]), int(ok1[b_idx])
        caps = np.array([c0, c1])
    near_cap = d[caps].min(0)                      # distance to the nearest capital

    # --- land shape: noise height + preset bias, thresholded to the target land fraction
    h = _noise(rng, size, p.land_scale).ravel()
    centre = ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) ** .5 / (size / 2)
    if p.shape == 'pangea':
        h = h * .6 - 2.2 * centre
    elif p.shape == 'continents':
        near = near_cap / size
        h = h * .7 - 4.0 * near
    elif p.shape == 'islands':
        h = h + _noise(rng, size, .7).ravel() * .8
    elif p.shape == 'lakes':
        h = h * .6 + 1.2 * np.minimum(centre, 1.)  # inland water in pockets, coasts mostly land
        h = -np.abs(h - np.quantile(h, .3))         # lake pockets at a mid height band
    land = h >= np.quantile(h, 1 - q['land'])
    # capitals always start with their 3x3 on land
    for c in caps:
        land[d[c] <= 1] = True
    if p.shape == 'continents':
        # carve a sea channel between the starting regions (no land bridge): where the two nearest capitals are
        # (almost) equally far
        if players == 2:
            between = np.abs(d[c0] - d[c1]) <= 1
        else:
            ranked = np.sort(d[caps], 0)
            between = ranked[1] - ranked[0] <= 1
        land &= ~(between & (near_cap >= 3))
        for c in caps:
            land[d[c] <= 1] = True

    # --- elevation for mountains, separate noise for forests
    elev = _noise(rng, size, 1.2).ravel()
    wood = _noise(rng, size, 1.0).ravel()
    protected = np.zeros(n, bool)
    for c in caps:
        protected |= d[c] <= 1
    terrain = np.full(n, OCEAN, np.int32)
    terrain[land] = FIELD
    free = land & ~protected
    if free.any():
        k = int(round(q['mountain'] * free.sum()))
        if k:
            order = np.flatnonzero(free)[np.argsort(-elev[free])]
            terrain[order[:k]] = MOUNTAIN
    if p.shape == 'ridge':
        # Wall along the perpendicular bisector of the capitals (Euclidean), 1-2 tiles thick.
        ax, ay, bx, by = c0 % size, c0 // size, c1 % size, c1 // size
        e0 = np.hypot(x - ax, y - ay)
        e1 = np.hypot(x - bx, y - by)
        wall = (np.abs(e0 - e1) <= rng.uniform(1.0, 1.6)) & ~protected
        terrain[wall & land] = MOUNTAIN
        terrain[wall & ~land] = MOUNTAIN   # the wall also bridges water so it truly separates
        land |= wall
        # passes: spread along the wall (sorted by position along it), each opened 1 tile wide
        along = np.flatnonzero(wall)
        if len(along):
            tang = (x[along] - (ax + bx) / 2) * -(by - ay) + (y[along] - (ay + by) / 2) * (bx - ax)
            along = along[np.argsort(tang)]
            k = q['passes']
            picks = [along[int((j + .5) * len(along) / k)] for j in range(k)]
            for g in picks:
                gx, gy = g % size, g // size
                # open a one-tile-wide pass straight across the wall's thickness
                ux, uy = (bx - ax), (by - ay)
                norm = max(np.hypot(ux, uy), 1e-9)
                for step in np.arange(-2, 2.5, .5):
                    px, py = int(round(gx + step * ux / norm)), int(round(gy + step * uy / norm))
                    if 0 <= px < size and 0 <= py < size and wall[py * size + px]:
                        terrain[py * size + px] = FIELD
    non_m = land & (terrain != MOUNTAIN) & ~protected
    if non_m.any():
        k = int(round(q['forest'] * non_m.sum()))
        if k:
            order = np.flatnonzero(non_m)[np.argsort(-wood[non_m])]
            terrain[order[:k]] = FOREST
    # shared landmass: if required and capitals are split, cut a land bridge along the direct line
    label, _ = _components(terrain.reshape(size, size) >= FIELD, size)
    for other in caps[1:]:
        if p.shared_landmass and label[c0] != label[other]:
            path = _line(c0, int(other), size)
            for a_, b_ in zip(path, path[1:]):
                # diagonal steps get an orthogonal corner tile so the bridge is 4-connected
                for t in (a_, b_, (a_ // size) * size + b_ % size):
                    if terrain[t] < FIELD:
                        terrain[t] = FIELD
            label, _ = _components(terrain.reshape(size, size) >= FIELD, size)
    # water depth: shallow if orthogonally next to land
    landm = terrain >= FIELD
    grid = landm.reshape(size, size)
    adj = np.zeros_like(grid)
    adj[1:] |= grid[:-1]; adj[:-1] |= grid[1:]; adj[:, 1:] |= grid[:, :-1]; adj[:, :-1] |= grid[:, 1:]
    terrain[~landm & adj.ravel()] = SHALLOW
    terrain[~landm & ~adj.ravel()] = OCEAN

    # --- villages: greedy spaced placement on non-mountain land
    structure = np.zeros(n, np.int32)
    taken = np.zeros(n, bool)
    taken[caps] = True
    structure[caps] = VILLAGE
    for t in rng.permutation(n):
        if terrain[t] in (FIELD, FOREST) and edge[t] >= 1 and not np.any(taken & (d[t] < p.village_spacing)):
            if rng.random() < q['village_keep']:
                taken[t] = True
                structure[t] = VILLAGE
                terrain[t] = FIELD
    villages = structure == VILLAGE

    # --- resources around settlements, by terrain (same families as the frozen generator)
    resource = np.zeros(n, np.int32)
    for c in caps:  # two guaranteed animals next to each capital
        spots = np.flatnonzero((d[c] == 1) & (terrain >= FIELD) & (terrain != MOUNTAIN))
        for t in rng.choice(spots, size=min(2, len(spots)), replace=False):
            terrain[t] = FOREST
            resource[t] = ANIMAL
    near = np.zeros(n, int)
    for v in np.flatnonzero(villages):
        near = np.where(d[v] <= 2, np.minimum(np.where(near == 0, 9, near), np.maximum(d[v], 1)), near)
    for t in np.flatnonzero((near > 0) & (resource == 0) & (structure == 0)):
        r1 = near[t] == 1
        prob, kind = {FIELD: ((.18, .06), FRUIT), FOREST: ((.38, .12), ANIMAL), MOUNTAIN: ((.11, .03), METAL),
                      SHALLOW: ((.5, .5), FISH)}.get(int(terrain[t]), ((0, 0), 0))
        if kind and rng.random() < prob[0 if r1 else 1]:
            resource[t] = kind
    # ruins on land away from villages, starfish at sea, lighthouses on corners
    for _ in range(RUINS[size]):
        spots = np.flatnonzero((terrain >= FIELD) & (structure == 0) & (resource == 0)
                               & ~np.any((d <= 1) & ((structure == RUIN) | villages)[None, :], axis=1))
        if len(spots):
            structure[int(rng.choice(spots))] = RUIN
    sea = np.flatnonzero((terrain <= OCEAN) & (resource == 0) & (structure == 0))
    for t in rng.permutation(sea)[:n // 25]:
        resource[t] = STARFISH
    corners = np.array([0, size - 1, n - size, n - 1])
    structure[corners] = LIGHTHOUSE
    resource[corners] = 0

    label, count = _components(terrain.reshape(size, size) >= FIELD, size)
    info = dict(preset=preset, revision=GENERATOR_REVISION, seed=int(seed), size=size,
                params={k: round(v, 3) if isinstance(v, float) else v for k, v in q.items()},
                stats=dict(land=float((terrain >= FIELD).mean()), water=float((terrain <= OCEAN).mean()),
                           mountain=float((terrain == MOUNTAIN).mean()), forest=float((terrain == FOREST).mean()),
                           villages=int(villages.sum() - len(caps)), landmasses=int(count),
                           capitals_connected=bool(np.all(label[caps] == label[c0])),
                           capital_distance=int(min(d[a, b] for i, a in enumerate(caps) for b in caps[i + 1:]))))
    return Layout(terrain.astype(np.int32), structure, resource, caps.astype(np.int32)), info

````


## src/polytopia_bot/maps/stream.py

Source SHA-256: aebffab112c526fd61de97b5a854709b4d3e2a0d4d23eb7d9195655a4a17409e

````python
"""Map sources for actors and arenas: a weighted mix of map types, deterministic per seed.

A mix is a dict {type: weight}; types are the variant presets plus 'classic'
(the frozen engine generator, gpu.mapgen.layout, i.e. today's training maps).
"""
import numpy as np

from .state import stack
from .variants import PRESETS, generate

TYPES = ('classic',) + tuple(PRESETS)
MIXES = {
    'classic': {'classic': 1.},
    'uniform': {t: 1. for t in TYPES},
    # Proposed training default (see docs/map-variants-2026-10-19.md).
    # Tilted toward water maps, where the current bot most often fails to finish games.
    'proposed-v1': {'classic': .15, 'drylands': .05, 'lakes': .1, 'pangea': .1, 'continents': .12,
                    'archipelago': .12, 'waterworld': .08, 'highlands': .1, 'ridge': .1, 'jungle': .08},
    # Four-seat training default (docs/four-player-2026-10-24.md): land-heavy maps, no water worlds or ridge.
    'four-v1': {'classic': .25, 'drylands': .15, 'lakes': .2, 'pangea': .15, 'continents': .1, 'highlands': .08,
                'jungle': .07},
}


def parse_mix(mix):
    if isinstance(mix, str):
        if mix in MIXES:
            mix = MIXES[mix]
        elif mix in TYPES:
            mix = {mix: 1.}
        else:
            raise ValueError(f'unknown map mix {mix!r}')
    if not mix or any(t not in TYPES for t in mix) or any(w < 0 for w in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError('invalid map mix')
    total = sum(mix.values())
    return {t: w / total for t, w in mix.items() if w > 0}


class MapSource:
    def __init__(self, mix='classic', size=14, players=2):
        """players=4: four-capital maps; 'classic' then comes from engine-rs (mapgen::layout_players), which
        has no JAX counterpart."""
        self.mix, self.size, self.players = parse_mix(mix), size, players
        self.types = list(self.mix)
        self.weights = np.array([self.mix[t] for t in self.types])
        self._classic = None

    def type_for(self, seed):
        """Map type is a pure function of the seed, so arenas can replay exact boards."""
        if len(self.types) == 1:
            return self.types[0]
        u = np.random.default_rng([int(seed), 77]).random()
        return self.types[int(np.searchsorted(np.cumsum(self.weights), u * self.weights.sum(), side='right'))]

    def layout(self, seed, kind=None):
        kind = kind or self.type_for(seed)
        if kind == 'classic':
            if self._classic is None:
                self._classic = self._classic_generator()
            return kind, self._classic(np.int32(seed))
        layout, _ = generate(seed, kind, size=self.size, players=self.players)
        return kind, layout

    def _classic_generator(self):
        if self.players == 2:
            import jax
            from polytopia_bot.gpu import mapgen
            return jax.jit(lambda s: mapgen.layout(s, size=self.size))
        from importlib.resources import files
        from polytopia_engine_rs import GameBatch
        from polytopia_bot.gpu import ENGINE_REVISION
        from .variants import Layout
        rules = files('polytopia_bot.gpu').joinpath('rules.json').read_text()
        batch = GameBatch(self.size, 1, rules, ENGINE_REVISION, None, True, 10_000, self.players)
        def layout(seed):
            d = batch.layout(int(seed))
            return Layout(*(np.asarray(d[k], np.int32) for k in ('terrain', 'structure', 'resource', 'capitals')))
        return layout

    def batch(self, seeds):
        rows, kinds = [], []
        for s in seeds:
            kind, layout = self.layout(s)
            rows.append((int(s), layout))
            kinds.append(kind)
        return stack(rows), kinds

````


## src/polytopia_bot/learning/adapter.py

Source SHA-256: 44e302efa3351669c1a21d3e9bd9af77f0c51ee95d50abb3f32b28acfc57e4fb

````python
def serialize(view, *, size, legal):
    """Accept only an already-redacted device view, never an engine State."""
    v = jax.device_get(view)
    player = int(v['player'])
    tiles, units, cities = [], [], []
    for i, explored in enumerate(v['explored']):
        t = dict(tile=i, known=bool(explored))
        if explored:
            t.update({name: int(v[name][i]) for name in ('terrain', 'owner', 'structure', 'resource', 'road', 'capital', 'structure_level')})
            if v['unit_owner'][i] > 0:
                u = dict(tile=i, owner=int(v['unit_owner'][i]), kind=int(v['unit_type'][i]), health=float(v['unit_health'][i]), veteran=bool(v['unit_veteran'][i]))
                if u['owner'] == player:
                    for name in ('moved', 'attacked', 'poison', 'frozen', 'boosted', 'invisible'):
                        u[name] = bool(v['own_unit_' + name][i])
                    for name in ('kills', 'home', 'passenger'):
                        u[name] = int(v['own_unit_' + name][i])
                    u['detect'] = bool(v['detect'][i])
                units.append(u)
            # Structure 1 denotes villages/cities; 2 is a ruin, not a city.
            if v['structure'][i] == 1:
                c = dict(tile=i, owner=int(v['owner'][i]))
                if c['owner'] == player:
                    for name in ('level', 'progress', 'population', 'production'):
                        c[name] = int(v['own_city_' + name][i])
                    for name in ('connected', 'riot'):
                        c[name] = bool(v['own_city_' + name][i])
                    c['rewards'] = v['own_city_rewards'][i].tolist()
                cities.append(c)
        tiles.append(t)
    visible = dict(board_size=size, turn=int(v['turn']), stars=int(v['stars']),
        technology=np.flatnonzero(v['tech']).tolist(), tiles=tiles, units=units, cities=cities,
        relations={k: np.asarray(v[k]).astype(bool).tolist() for k in ('known_players', 'peace', 'embassies_sent', 'embassies_received', 'peace_offers')},
        achievements={k: np.flatnonzero(v['own_' + k]).tolist() for k in ('monuments', 'built')})
    obs = Observation(player, int(v['state_version']), visible, {}, legal)
    public_state(obs)
    return obs



````


## src/polytopia_bot/learning/interface.py

Source SHA-256: 60c9df50906f99e935dcd2d0599a12918f1250fc9265ca26e7e01c0f84008d11

````python
"""Strict public observation schema and response-local action catalog.

This accepts already-redacted observations, never engine State or snapshots.
The eventual GPU adapter must independently prove that its redaction is correct.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from polytopia_bot.environment import LegalAction, Observation

SCHEMA = "polytopia-public-v2"
SYSTEM = """Play Polytopia using only the supplied public observation and legal catalog.
Unknown tiles have no known terrain or contents. Remembered facts may be stale.
Choose exactly one listed action. Return only {\"action_id\":\"aN\"}, with no prose.
Commands use (kind,source,target,parameter); tiles are row-major indices.
Kinds: 1 move, 2 attack, 3 ability, 4 summon, 5 harvest, 6 build,
7 research, 8 capture, 9 city reward, 10 end turn, 11 resign.
Ability parameters: 1 burn forest, 2 clear forest, 3 grow forest, 4 destroy,
7 recover, 8 disband, 9 heal, 16 promote, 18 break peace, 19 respond to peace,
20 embassy, 21 offer peace, 22 remove embassy. Diplomatic source is a player ID;
peace response target is 0 (reject) or 1 (accept).
Terrain IDs: 1 water, 2 ocean, 3 field, 4 mountain, 5 forest, 6 ice,
7 wetland, 8 mangrove. Resource IDs: 0 none, 1 animal, 2 crop, 3 fish,
5 metal, 6 fruit, 7 spores, 8 starfish, 9 aquatic crop.
City rewards: 1 wall, 2 park, 3 workshop, 4 explorer, 5 border growth,
6 superunit, 7 resources, 8 population. The catalog determines legal choices.
Capture cities and preserve your empire; use income, research and exploration
to support expansion and combat. End turn when no useful action remains."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@lru_cache(maxsize=1)
def rule_table():
    return json.loads((Path(__file__).resolve().parents[1] / "gpu" / "rules.json").read_text())


def rule_glossary(state, actions):
    """Public static definitions for IDs actually present in this observation."""
    v, table = state["visible"], rule_table()
    unit_ids = {u["kind"] for u in v["units"]} | {a.command["parameter"] for a in actions if a.command["kind"] == 4}
    tech_ids = set(v["technology"]) | {a.command["parameter"] for a in actions if a.command["kind"] == 7}
    structure_ids = {t["structure"] for t in v["tiles"] if "structure" in t} | {a.command["parameter"] for a in actions if a.command["kind"] == 6}
    definitions = {}
    for name, ids, fields in (("units", unit_ids, ("id", "name", "cost", "health", "attack", "defense", "movement", "range", "skills")),
                              ("technologies", tech_ids, ("id", "name", "tier", "requires", "structures")),
                              ("structures", structure_ids, ("id", "name", "cost", "population", "stars"))):
        rows = {r["id"]: r for r in table[name]}
        if ids - rows.keys():
            raise ValueError(f"unknown public {name} ID")
        definitions[name] = [{k: rows[i][k] for k in fields} for i in sorted(ids)]
    return definitions


def keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("invalid public schema fields")


def integer(value, lo=0, hi=None):
    if type(value) is not int or value < lo or (hi is not None and value > hi):
        raise ValueError("invalid public integer")
    return value


def command(value):
    keys(value, ("kind", "source", "target", "parameter"))
    integer(value["kind"], 1, 11)
    for k in ("source", "target", "parameter"):
        integer(value[k], -1)
    return dict(value)


def public_state(obs: Observation) -> dict:
    # JSON conversion rejects non-JSON objects, NaNs and nested engine objects.
    v = json.loads(canonical(obs.visible_state))
    # Seats: 2, or one per entry of the relation lists (four-seat games, docs/four-player-2026-10-24.md).
    seats = len(v["relations"]["known_players"]) if isinstance(v.get("relations"), dict) and isinstance(
        v["relations"].get("known_players"), list) else 2
    if seats not in (2, 4):
        raise ValueError("invalid seat count")
    integer(obs.player, 1, seats)
    integer(obs.state_version)
    keys(v, ("board_size", "turn", "stars", "technology", "tiles", "units", "cities"), ("relations", "achievements"))
    n = integer(v["board_size"], 2, 64) ** 2
    integer(v["turn"])
    integer(v["stars"])
    if not isinstance(v["technology"], list):
        raise ValueError("technology must be a list")
    v["technology"] = sorted({integer(t, 0, 127) for t in v["technology"]})
    for name in ("tiles", "units", "cities"):
        if not isinstance(v[name], list):
            raise ValueError("entities must be lists")
        seen = set()
        for row in v[name]:
            if not isinstance(row, dict):
                raise ValueError("entity must be an object")
            tile = integer(row.get("tile"), 0, n - 1)
            if tile in seen:
                raise ValueError("duplicate tile/entity")
            seen.add(tile)
            if name == "tiles":
                keys(row, ("tile", "known"), ("terrain", "owner", "resource", "structure", "road", "capital", "structure_level"))
                if type(row["known"]) is not bool:
                    raise ValueError("known must be boolean")
                if not row["known"] and set(row) != {"tile", "known"}:
                    raise ValueError("unknown tile contains hidden fields")
                for k in set(row) - {"tile", "known"}:
                    integer(row[k])
                if "owner" in row:
                    integer(row["owner"], 0, seats)
            elif name == "units":
                flags = ("veteran", "moved", "attacked", "poison", "frozen", "boosted", "invisible", "detect")
                own_fields = ("kills", "home", "passenger", *flags[1:])
                keys(row, ("tile", "owner", "kind", "health"), (*flags, "kills", "home", "passenger"))
                if row["owner"] != obs.player and set(row) & set(own_fields):
                    raise ValueError("opponent unit internals are not public")
                integer(row["owner"], 1, seats)
                integer(row["kind"], 1, 127)
                if type(row["health"]) not in (int, float) or not math.isfinite(row["health"]) or row["health"] <= 0:
                    raise ValueError("invalid unit health")
                for k in flags:
                    if k in row and type(row[k]) is not bool:
                        raise ValueError("invalid unit flag")
                for k in ("kills", "home", "passenger"):
                    if k in row:
                        integer(row[k])
            else:
                keys(row, ("tile", "owner"), ("level", "population", "production", "progress", "connected", "riot", "rewards"))
                integer(row["owner"], 0, seats)
                if row["owner"] != obs.player and set(row) - {"tile", "owner"}:
                    raise ValueError("opponent city internals are not in this schema")
                for k in set(row) - {"tile", "owner"}:
                    if k in ("connected", "riot"):
                        if type(row[k]) is not bool:
                            raise ValueError("invalid city flag")
                    elif k == "rewards":
                        if not isinstance(row[k], list) or len(row[k]) != 9:
                            raise ValueError("invalid city rewards")
                        for count in row[k]:
                            integer(count)
                    elif k in ("progress", "population"):
                        # Frozen native starvation retains city levels/rewards
                        # while its population progress can become negative.
                        integer(row[k], -(2**31), 2**31 - 1)
                    else:
                        integer(row[k])
        v[name] = sorted(v[name], key=lambda r: r["tile"])
    if len(v["tiles"]) != n:
        raise ValueError("include every tile, with unknown tiles explicitly marked")
    known = {t["tile"] for t in v["tiles"] if t["known"]}
    if any(e["tile"] not in known for name in ("units", "cities") for e in v[name]):
        raise ValueError("entity on unknown tile")
    if "relations" in v:
        keys(v["relations"], ("known_players", "peace", "embassies_sent", "embassies_received", "peace_offers"))
        for value in v["relations"].values():
            if not isinstance(value, list) or len(value) != seats or any(type(x) is not bool for x in value):
                raise ValueError("invalid relations")
    if "achievements" in v:
        keys(v["achievements"], ("monuments", "built"))
        for value in v["achievements"].values():
            if not isinstance(value, list):
                raise ValueError("invalid achievements")
            for item in value:
                integer(item, 0, 127)
    memory = json.loads(canonical(obs.remembered_state))
    keys(memory, (), ("tiles", "schema", "recent_actions", "historical"))
    if set(memory) & {"schema", "recent_actions", "historical"}:
        from .memory import validate_memory
        validate_memory(memory, obs)
    memory.setdefault("tiles", [])
    if not isinstance(memory["tiles"], list):
        raise ValueError("memory tiles must be a list")
    seen = set()
    for row in memory["tiles"]:
        keys(row, ("tile", "last_seen_turn"), ("terrain", "owner", "resource", "structure"))
        tile = integer(row["tile"], 0, n - 1)
        if tile in seen:
            raise ValueError("duplicate memory tile")
        seen.add(tile)
        integer(row["last_seen_turn"], 0, v["turn"])
        for k in set(row) - {"tile", "last_seen_turn"}:
            integer(row[k])
        if "owner" in row:
            integer(row["owner"], 0, 2)
    memory["tiles"].sort(key=lambda r: r["tile"])
    return {"schema": SCHEMA, "player": obs.player, "state_version": obs.state_version,
            "visible": v, "remembered": memory}


@dataclass(frozen=True)
class Prompt:
    messages: list[dict]
    catalog: dict[str, LegalAction]
    state_version: int
    public: dict


def render(obs: Observation) -> Prompt:
    state = public_state(obs)
    actions = []
    ids, commands = set(), set()
    for a in obs.legal_actions:
        if not isinstance(a.action_id, str) or not a.action_id or a.action_id in ids:
            raise ValueError("invalid or duplicate engine action ID")
        c = command(dict(a.command))
        for field in ("source", "target"):
            integer(c[field], -1, state["visible"]["board_size"] ** 2 - 1)
        key = canonical(c)
        if key in commands:
            raise ValueError("duplicate semantic action")
        ids.add(a.action_id)
        commands.add(key)
        # Copy mutable caller data: a later mutation cannot change this prompt.
        actions.append(((c["kind"], c["source"], c["parameter"], c["target"]), LegalAction(a.action_id, c)))
    if not actions:
        raise ValueError("decision requires at least one legal action")
    catalog = {f"a{i}": a for i, (_, a) in enumerate(sorted(actions))}
    state["rules"] = rule_glossary(state, list(catalog.values()))
    state["legal_actions"] = [{"action_id": k, "command": dict(a.command)} for k, a in catalog.items()]
    from .encoding import compact, INSTRUCTIONS
    return Prompt([{"role": "system", "content": SYSTEM + "\n" + INSTRUCTIONS}, {"role": "user", "content": canonical(compact(state))}], catalog, obs.state_version, state)


class ResponseError(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ResponseError("duplicate_key")
        result[k] = v
    return result


def parse_response(text: str, prompt: Prompt, *, current_version: int) -> LegalAction:
    if current_version != prompt.state_version:
        raise ResponseError("stale_version")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError):
        raise ResponseError("invalid_json") from None
    if not isinstance(value, dict) or set(value) != {"action_id"} or not isinstance(value["action_id"], str):
        raise ResponseError("invalid_response_shape")
    if value["action_id"] not in prompt.catalog:
        raise ResponseError("unknown_action_id")
    return prompt.catalog[value["action_id"]]


@dataclass
class Generation:
    text: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]


class PolicyForfeit(RuntimeError):
    def __init__(self, attempts):
        super().__init__("three invalid model responses; no fallback action")
        self.attempts = attempts


def choose_action(obs: Observation, generate: Callable[[list[dict]], Generation]):
    """Return engine action + all attempts; engine.step must recheck the version."""
    prompt = render(obs)
    messages, attempts = list(prompt.messages), []
    for _ in range(3):
        output = generate(messages)
        attempt = {"messages": list(messages), "text": output.text,
                   "prompt_token_ids": output.prompt_token_ids, "response_token_ids": output.response_token_ids}
        attempts.append(attempt)
        try:
            selected = parse_response(output.text, prompt, current_version=obs.state_version)
            attempt["error"] = None
            return selected, attempts
        except ResponseError as e:
            attempt["error"] = str(e)
            messages = messages + [{"role": "assistant", "content": output.text},
                                   {"role": "user", "content": f"Invalid response: {e}. Return one legal action_id JSON object."}]
    raise PolicyForfeit(attempts)

````


## src/polytopia_bot/learning/memory.py

Source SHA-256: 3bcb381f8c64d678dc85752e30896184071ba872349035797a4e8f90c1fad6c4

````python
"""Bounded factual memory updated exclusively from a seat's public observations."""
from copy import deepcopy
from dataclasses import replace
from .interface import canonical, command, integer, keys

SCHEMA = 'polytopia-memory-v1'
ACTION_CAP = 5
HISTORY_CAP = 32


def validate_memory(memory, obs):
    if memory.get('schema') != SCHEMA:
        raise ValueError('unsupported memory schema')
    keys(memory, ('schema', 'recent_actions', 'historical'), ('tiles',))
    if not isinstance(memory['recent_actions'], list) or len(memory['recent_actions']) > ACTION_CAP:
        raise ValueError('invalid recent-action memory')
    previous = -1
    for row in memory['recent_actions']:
        keys(row, ('command', 'turn', 'revision'))
        command(row['command'])
        integer(row['turn'], 0, obs.visible_state['turn'])
        integer(row['revision'], previous + 1, obs.state_version - 1)
        previous = row['revision']
    if not isinstance(memory['historical'], list) or len(memory['historical']) > HISTORY_CAP:
        raise ValueError('invalid historical memory')
    seen = set()
    for row in memory['historical']:
        keys(row, ('entity', 'fact', 'last_seen_turn', 'revision'))
        if row['entity'] not in ('tiles', 'units', 'cities'):
            raise ValueError('invalid historical entity')
        integer(row['last_seen_turn'], 0, obs.visible_state['turn'])
        integer(row['revision'], 0, obs.state_version)
        fact = row['fact']
        tile = integer(fact.get('tile'), 0, obs.visible_state['board_size'] ** 2 - 1)
        key = (row['entity'], tile)
        if key in seen:
            raise ValueError('duplicate historical fact')
        seen.add(key)
        # Reuse the allowlist on historical rows; never accept arbitrary payloads.
        from .interface import public_state
        visible = deepcopy(obs.visible_state)
        visible['tiles'] = [{'tile': i, 'known': i == tile} for i in range(visible['board_size'] ** 2)]
        visible['units'], visible['cities'] = [], []
        if row['entity'] == 'tiles':
            visible['tiles'][tile] = fact
        else:
            visible[row['entity']] = [fact]
        public_state(replace(obs, visible_state=visible, remembered_state={}))


class PublicMemory:
    def __init__(self):
        self.seats = {str(p): dict(revision=-1, fingerprint=None, facts={}, actions=[], memory={}) for p in (1, 2)}

    def observe(self, obs):
        seat = self.seats[str(obs.player)]
        fingerprint = canonical(obs.visible_state)
        if obs.state_version < seat['revision']:
            raise ValueError('observation timestamp regressed')
        if obs.state_version == seat['revision']:
            if fingerprint != seat['fingerprint']:
                raise ValueError('public observation changed without a revision')
            return replace(obs, remembered_state=deepcopy(seat['memory']))
        known = {t['tile'] for t in obs.visible_state['tiles'] if t['known']}
        # Current public information supersedes prior records, including absence.
        facts = {k: v for k, v in seat['facts'].items() if v['fact']['tile'] not in known}
        historical = sorted(facts.values(), key=lambda r: (-r['revision'], r['entity'], r['fact']['tile']))[:HISTORY_CAP]
        facts = {f"{r['entity']}:{r['fact']['tile']}": r for r in historical}
        for entity in ('tiles', 'units', 'cities'):
            for fact in obs.visible_state[entity]:
                if entity == 'tiles' and not fact['known']:
                    continue
                facts[f"{entity}:{fact['tile']}"] = dict(entity=entity, fact=deepcopy(fact),
                    last_seen_turn=obs.visible_state['turn'], revision=obs.state_version)
        memory = dict(schema=SCHEMA, recent_actions=deepcopy(seat['actions']), historical=historical)
        seat.update(revision=obs.state_version, fingerprint=fingerprint, facts=facts, memory=memory)
        return replace(obs, remembered_state=deepcopy(memory))

    def executed(self, obs, selected):
        seat = self.seats[str(obs.player)]
        if seat['actions'] and obs.state_version <= seat['actions'][-1]['revision']:
            raise ValueError('executed action timestamp regressed')
        seat['actions'] = (seat['actions'] + [dict(command=dict(selected.command), turn=obs.visible_state['turn'], revision=obs.state_version)])[-ACTION_CAP:]

    def snapshot(self):
        return dict(schema=SCHEMA, seats=deepcopy(self.seats))

    @classmethod
    def restore(cls, payload):
        if payload.get('schema') != SCHEMA or set(payload.get('seats', {})) != {'1', '2'}:
            raise ValueError('invalid memory snapshot')
        result = cls()
        result.seats = deepcopy(payload['seats'])
        return result

````


## src/polytopia_bot/environment.py

Source SHA-256: 4e61b40fc9511c564a11fcdb5615b0d48832eb00c634e605517a5ca536b204af

````python
"""Engine adapter contract only. This module implements no game mechanics."""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class LegalAction:
    action_id: str
    command: Mapping[str, Any]


@dataclass(frozen=True)
class Observation:
    player: int
    state_version: int
    visible_state: Mapping[str, Any]
    remembered_state: Mapping[str, Any]
    legal_actions: Sequence[LegalAction]


@dataclass(frozen=True)
class Transition:
    actor_observation: Observation
    next_player: int | None
    rewards: Mapping[int, float]
    terminated: bool
    truncated: bool
    reason: str | None


class Environment(Protocol):
    """One logical game; a worker may host multiple instances."""

    def reset(self, *, seed: int, settings: Mapping[str, Any]) -> Observation: ...

    def observe(self, player: int) -> Observation: ...

    def step(self, *, action_id: str, expected_state_version: int) -> Transition: ...

    def snapshot(self) -> bytes:
        """Private trusted-controller state. Never expose to a policy."""
        ...

    def restore(self, snapshot: bytes) -> Observation: ...

    def close(self) -> None: ...

````


## src/polytopia_bot/studio/environment.py

Source SHA-256: 9b845b5544a37e9b9522edd7635793369d5fc5de40ef829b0fab0577d3247dca

````python
"""The public game environment Studio plays through, on engine-rs (the real game's rules, polytopia_bot.ruleset).

Same interface as the retired JAX GPUEnvironment (learning/adapter.py): reset / observe / step / snapshot / restore /
close, legal actions as catalog IDs (the flat index of rl.actions.offsets/shapes) with the decoded command fields,
observations serialized from the redacted per-seat view only. `state` is the single-game NumPy State engine-rs
exports (gpu/state.py layout), used for the HUD and for search.
"""
import base64
import json

import jax
import numpy as np

from polytopia_bot.environment import LegalAction, Transition
from polytopia_bot.gpu import snapshot
from polytopia_bot.learning.adapter import serialize
from polytopia_bot.learning.interface import canonical
from polytopia_bot.learning.memory import PublicMemory
from polytopia_bot.rl.actions import offsets, shapes

KINDS = ('kind', 'source', 'target', 'parameter')


def decode_command(index, state, size):
    """Catalog index -> the engine action's (kind, source, target, parameter), as engine-rs core::decode."""
    n = size * size
    bounds = offsets(size)
    family = int(np.searchsorted(bounds, index, side='right')) - 1
    at = int(index - bounds[family])
    other = 3 - int(state.current_player)
    if family == 14:
        caps = np.flatnonzero((np.asarray(state.tiles.capital) == other) & (np.asarray(state.cities.owner) == other))
        capital = int(caps[0]) if len(caps) else 0
        return dict(kind=3, source=other, target=(-1, 0, 1, capital, -1, -1)[at], parameter=(18, 19, 19, 20, 21, 22)[at])
    if family in (9, 10, 11, 12):
        return dict(kind=3, source=at, target=-1, parameter=(7, 16, 8, 9)[family - 9])
    table = {0: (1, at // n, at % n, 0), 1: (2, at // n, at % n, 0), 2: (4, at // 128, -1, at % 128),
             3: (6, -1, at // 128, at % 128), 4: (7, -1, -1, at), 5: (5, -1, at, 0), 6: (3, -1, at // 3, at % 3 + 1),
             7: (9, -1, at // 9, at % 9), 8: (8, at, -1, 0), 13: (3, -1, at, 4), 15: (10, -1, -1, 0),
             16: (11, -1, -1, 0)}
    return dict(zip(KINDS, table[family]))


def choice_of(index, size):
    """Catalog index -> the (family, source, target) engine-rs steps."""
    bounds = offsets(size)
    family = int(np.searchsorted(bounds, index, side='right')) - 1
    source, target = divmod(int(index - bounds[family]), shapes(size)[family][1])
    return family, source, target


class RustEnvironment:
    def __init__(self, size=14, *, max_rounds=None, max_actions=10_000):
        from polytopia_bot.rl.cpu_engine import unpack_view, view_layout
        from polytopia_bot.rl.rust_engine import RustGames
        self.size = size
        self.games = RustGames(size, 1, no_resign=False,
                               max_turns=None if max_rounds is None else max_rounds - 1,
                               max_actions=max_actions)
        self.layout, self._unpack = view_layout(size), unpack_view
        self.state = None
        self.memory = PublicMemory()
        self._cached = {}

    def _sync(self):
        self.state = jax.tree.map(lambda a: np.asarray(a)[0], self.games.states())

    def view(self, player, state=None):
        """The redacted public view of `player` (a dict like gpu.kernels.observe), optionally of another state."""
        if state is not None:
            scratch = self._scratch()
            scratch.load(jax.tree.map(lambda a: np.asarray(a)[None], state))
            packed = np.asarray(scratch.batch.observe(int(player)))[0]
            version = int(state.state_version)
        else:
            packed = np.asarray(self.games.batch.observe(int(player)))[0]
            version = int(self.state.state_version)
        out = {k: np.asarray(v) for k, v in self._unpack(packed, self.layout).items()}
        out['state_version'] = np.int32(version)
        return out

    def _scratch(self):
        if not hasattr(self, '_render'):
            from polytopia_bot.rl.rust_engine import RustGames
            self._render = RustGames(self.size, 1, no_resign=False)
        return self._render

    def income(self, player):
        return int(np.asarray(self.games.batch.income(int(player)))[0])

    def reset(self, *, seed, settings):
        if set(settings) - {'size'} or settings.get('size', self.size) != self.size:
            raise ValueError('unsupported settings')
        self.games.reset(np.array([seed], np.int32))
        self._sync()
        self.memory, self._cached = PublicMemory(), {}
        self.observe(2)   # both players observe the start, independently, through the same API
        return self.observe(1)

    def observe(self, player):
        if player not in (1, 2) or self.state is None:
            raise ValueError('invalid seat or closed environment')
        if player in self._cached:
            return self._cached[player]
        legal = []
        if player == int(self.state.current_player) and not bool(self.state.terminated | self.state.truncated):
            mask = np.asarray(self.games.batch.legal_masks())[0]
            for index in np.flatnonzero(mask):
                legal.append(LegalAction(str(int(index)), decode_command(int(index), self.state, self.size)))
        obs = self.memory.observe(serialize(self.view(player), size=self.size, legal=legal))
        self._cached[player] = obs
        return obs

    def step(self, *, action_id, expected_state_version):
        if self.state is None or expected_state_version != int(self.state.state_version):
            raise ValueError('stale_version')
        player = int(self.state.current_player)
        before = self.observe(player)
        catalog = {a.action_id: a for a in before.legal_actions}
        if action_id not in catalog:
            raise ValueError('invalid_action')
        selected = catalog[action_id]
        status = int(np.asarray(self.games.batch.step(*(np.array([v], np.int32)
                                                         for v in choice_of(int(action_id), self.size))))[0])
        if status != 0:
            raise RuntimeError(f'catalog/transition discrepancy: status={status}')
        self.memory.executed(before, selected)
        self._sync()
        self._cached = {}
        terminal, truncated = bool(self.state.terminated), bool(self.state.truncated)
        rewards = {p: float(r) for p, r in enumerate(np.asarray(self.games.batch.terminal_rewards())[0], 1)}
        actor = self.observe(player)   # record every new public observation for both seats, even off-turn
        self.observe(3 - player)
        reason = ('resignation' if selected.command['kind'] == 11 else 'elimination') if terminal else (
            'research_horizon' if truncated else None)
        return Transition(actor, None if terminal or truncated else int(self.state.current_player), rewards, terminal,
                          truncated, reason)

    def load(self, state):
        """Continue from a single-game State (e.g. after a search played the turn on its own copy)."""
        self.games.load(jax.tree.map(lambda a: np.asarray(a)[None], state))
        self._sync()
        self._cached = {}

    def snapshot(self):
        return canonical(dict(schema='public-runner-v2', size=self.size,
                              engine=base64.b64encode(snapshot.dumps(self.state, profile='native')).decode(),
                              memory=self.memory.snapshot())).encode()

    def restore(self, payload):
        data = json.loads(payload)
        if data.get('schema') != 'public-runner-v2' or data.get('size') != self.size:
            raise ValueError('incompatible runner snapshot (games saved before 2026-10-27 used the retired rules)')
        self.load(snapshot.loads(base64.b64decode(data['engine'], validate=True), profile='native'))
        self.memory = PublicMemory.restore(data['memory'])
        return self.observe(int(self.state.current_player))

    def close(self):
        self.state, self._cached = None, {}
        self.memory = PublicMemory()

````
