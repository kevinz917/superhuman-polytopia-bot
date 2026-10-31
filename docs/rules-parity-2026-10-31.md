# Rules parity, second pass: the game's map generator, random procedures and remaining rules (2026-10-31)

**What this is.** A rule-by-rule audit of engine-rs against Polytopia itself (Steam 2.17.3, game version 123, game data
v28), after the [first alignment](rules-alignment-2026-10-27.md), and the fixes it led to. The goal is a 1:1 engine:
the same rules, the same map generator and the same random procedures, seed for seed where the game's seeds allow.

**Result** (27,785 recorded game transitions; legal-move sets of recorded and engine-generated positions):

| check | after the first pass (2026-10-27) | now |
|---|---|---|
| recorded transitions identical in every compared field | 19,619 | **19,742** |
| recorded commands our engine rejects | 12 | **0** |
| recorded positions with the game's legal moves, apart from the fog exception | 685 of 686 | **686 of 686** |
| positions our engine generated (1v1 / four seats), same | 130 / 237 | **160 / 240 (all)** |
| differences from ruins rewards / Giant pushes / the Explorer walk | 377 / 339 / 205 | 8 / **0** / 78 |

The remaining ruins and Explorer differences are all a lighthouse's level counter, a representation difference with
no effect on play. Everything else left is a convention with no gameplay effect (when moved/attacked flags reset, a
level counter on structures whose level the engine never reads) or a deliberate difference listed at the end.

## What changed

### Map generation
The default boards now come from a port of the game's own generator for the Drylands preset (Bardur on every seat),
replacing the Polyfish-derived generator. The steps match the game's:
- **Capitals** are drawn per seat from a grid of domains, weighted toward the domain centre.
- **Land:** every tile is land. Drylands has no water, so there are no fish and no starfish.
- **Villages** may go on any land tile at least one tile from the edge and at least three tiles from any other city,
  up to ⌊0.33·width⌋² − players.
- **Terrain:** each seat's continent grows from its capital, and its terrain is drawn from a bag without replacement
  (field 0.5, mountain 0.15, forest 0.4 × 0.8 for Bardur).
- **Lighthouses** go on the four corners, which are set to field.
- **Resources** are rolled per tile around cities, with hash-seeded draws.
- **Starting resources:** two Game near each capital, using existing forests first.
- **Ruins:** tiles/40 + 1, placed over a shuffled tile list, never on a corner and never within 2 tiles of a capital
  or another ruin.
- **Fairness:** up to 50 attempts per board, keeping the first whose resource-and-village fairness is within 1%,
  otherwise the fairest.

Given the attempt seed, boards match the game tile for tile. The game draws its attempt seeds from a cryptographic
RNG; ours come from the reset seed, so a reset seed fixes the board. Compared with the old generator, boards have about
30% more villages, far more metal and fruit, about a third forest instead of more than half, and no water. The home-grown
map variants (water world, islands, …) are unchanged.

### Random procedures
- **Explorer reward** (ruins and city level-ups): 12 steps. Each step scores an 11×11 window around the explorer by
  distance to fog, with a bonus for climbable explored mountains and a penalty on tiles already walked. It moves to
  the best neighbour, breaking ties with a hash of the reward tile and step, and explores radius 1 (2 from a
  mountain). Before, it searched only 3 tiles deep and wandered when no fog was near.
- **Ruins reward:** a hash of the ruin tile shuffles [stars, population, Swordsman, Battleship, Explorer, free tech]
  and the first valid entry wins. The free tech is hash-chosen among the techs the player can unlock (Navigation's
  parent is Sailing).
- **Giant/ruins push:** deterministic, not random. Units now carry a facing, toward the map centre at creation and
  updated by moves and attacks. The occupant is pushed along its facing, then the neighbouring directions in turn,
  onto an explored, enterable tile; a boat falls back to land. A pushed Juggernaut does not stomp.

### Units and combat
- Landing from a boat, or embarking, ends the unit's turn: no attack that turn, Dash or not.
- Damage uses the game's integer arithmetic. It was +1 HP in about 0.2% of stat combinations, mostly veterans.
- Disbanding an upgraded boat refunds half of boat plus passenger.
- Pirates are not Independent.
- A unit's limit counts every unit homed at the city.
- Converted units move their score to the converter.
- A Knight's Persist also counts hidden enemies in range.
- A Giant embarking as a Juggernaut stomps.
- Moving onto a tile with a hidden enemy reveals it; the mover stays and keeps its move. Hidden units elsewhere on
  the path are passed.

### Movement and vision
- A move follows the game's route (A* with its tie order), so multi-tile moves explore the same tiles as in the game.
- Peace partners' ports can be used to embark, and peace partners' hidden units are visible.
- Boats get no road speed.
- Starfish stay hidden until Sailing.

### Diplomacy and meeting
- Seeing another player's city meets them.
- The first embassy costs 4, and each further one 2 more.
- Embassies pay nothing while the two sides are at war, that is, within a round of an attack, a capture or entering
  the other's city. The host's share is part of its capital's income, so a siege or riot stops it.
- A peace request must be answered at the start of the receiver's turn, before anything else.
- An attack no longer cancels a pending request.
- Peace may be offered during the cool-down.
- Meeting through an embassy or a peace request pays the meeting stars.

### Cities and economy
- **Four seats: seats 3 and 4 now get population from their buildings.** Before, they got none.
- City score follows the game: 50 a level and 5 a point of progress, with negative progress costing more.
- A city that gains population off its owner's turn levels up at the start of that owner's next turn, after income.
- Capital connections:
  - A city that changes hands moves or loses its connection bonus.
  - A player who loses their capital loses every connection.
  - Ports link within 5 tiles over water their owner can sail.
- A one-per-city building is judged by the city that rules the tile.
- An infiltrated city's riot lasts to the end of its owner's turn, and the city cannot be infiltrated again until then.
  The stolen income ignores negative progress.
- Dagger spawn priorities match the game's.
- The lighthouse and ruins population rewards go to the current capital, or else the first city.

## Still different
- **Round cap** (100 rounds, 150 on 18×18). The game's Domination mode has none; ours is a training harness.
- **Difficulty handicap.** Self-play seats are both human-equivalent.
- **Roads and bridges on unexplored tiles.** The game accepts them; ours keeps requiring the tile to be explored.
- **No diplomacy in four-seat games.**
- **No in-place disembark on a bridge.** The game has a separate command for it.
- **Resignation.** The game turns a resigning player into an AI; training runs without resignation.
- **Dagger spawn tie-break noise.**
- **Port links over existing route tiles.** The game makes these cheaper.
