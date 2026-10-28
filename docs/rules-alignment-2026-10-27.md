# Rules alignment with the game: one ruleset, `polytopia-2.17.3` (2026-10-27)

**What this is.** The game every bot here trains and plays now follows Polytopia itself (Steam 2.17.3, game
version 123, game data v28): Bardur, 1v1 on 11/14/18 boards and four-seat free-for-all on 18×18 (diplomacy off
with four seats). There is **one ruleset**, `polytopia-2.17.3` (`src/polytopia_bot/ruleset.py`), implemented by
engine-rs. The research rules the earlier runs used (`bardur-1v1-v1/v2`, the four-seat research rules) and the JAX
engine that defined them are retired: nothing that trains, evaluates, searches or serves games can choose them.

The work happened in two steps. First the game's rules were added behind a ruleset switch (`bardur-1v1-v3` /
`bardur-ffa4-v2` next to the research rules) and checked against recorded games. Then the switch was removed: the
game's rules are simply the engine's rules. The tables below compare the research rules ("before") with the final
engine ("after").

**Result** (final engine):

| check | research rules (before) | game rules (after) |
|---|---|---|
| 27,785 recorded game transitions, identical in every compared field | 19,193 | **19,619** |
| recorded commands our engine rejects | 25 | **12** (11 follow a differently-rolled ruins reward, 1 peace offer) |
| recorded positions with the game's legal moves, apart from the fog exception | 380 of 693 | **685 of 686** |
| positions our engine generated (1v1 and four seats), same | 227 of 373 | **367 of 367** |

Every remaining transition difference has a named cause: the AI difficulty handicap (Crazy earns +3/turn; self-play
seats are both human, so it stays), random outcomes (ruins, explorer, Giant push, Dagger spawns), flag/representation
conventions with no gameplay effect, and three small open items (below). The one deliberate exception: the game
accepts roads and bridges on tiles the mover has never explored; ours keeps requiring the tile to be explored.

## How to use it

- Nothing to choose. `RustGames(size, games, players=...)` plays these rules; so do the training actors
  (`rl/distributed.py`, engine-rs only), the arenas (`rl/cpu_arena.py`, `arena_match`, `rl/arena4.py`, the ladders and
  `four_monitor`), the turn search (`rl/search.py`) and Studio (`studio/environment.py`).
- Checkpoints record `ruleset: "polytopia-2.17.3"` and `simulator_sha256 = ruleset.rules_digest()` (the name and the
  rule table). Old checkpoints (the research rules) still load as evaluation references and play under the new
  rules: `ruleset.LEGACY_DIGESTS` lists the two digests they carry (`7b76e65b…` = bardur-1v1-v1, `e4b656d3…` =
  bardur-1v1-v2 and the four-seat research runs), including the champion (scratch-cont-2 final and its fog head).
- `LearnerConfig.engine` must be `rust`; `trusted_step` / `trusted_check_every` / `build_rows` are kept only so older
  run configs parse (they were knobs of the JAX actor engine).

## What was removed, and what remains of the JAX engine

Removed (git history keeps them):
- The ruleset switch: `Rules::real`, `RULESETS`, `for_ruleset`, `RustGames(ruleset=)`, `LearnerConfig.ruleset`, the
  arenas' `ruleset` arguments and `checkpoint_ruleset`.
- The JAX actor engine `cpu_engine.CpuGames` (and the `engine='jax'` actor option) and the JAX turn search
  (`build_planner`, `ParallelPlanner`, `build_replay`, the guarded JAX replay).
- The old single-machine RL stack: `rl/{league, async_league, bootstrap, repair, curriculum, opponents, reward,
  replay_audit, acceptance, arena, diverse_heuristic, public_heuristic, distill_data, distill_train}.py`, the
  `Runtime` / `evaluate` / `run` PPO loop of `rl/runner.py`, `proposals/`, `gpu/{validation, benchmark, rollout}.py`,
  with their experiments, scripts and tests (including the engine-rs vs JAX bitwise test, which cannot hold any more).

Remains, for reasons that are not game rules:
- `gpu/state.py` (the State layout engine-rs exports), `gpu/rules.json` (engine-rs's rule table), `gpu/importer.py`
  (Polyfish worlds), `gpu/mapgen.py` (map layouts for the determinized search and the map variants),
  `gpu/kernels.observe` (read once for the packed-view schema), and `gpu/snapshot.py` (state files for Studio).
- `rl/cpu_engine.py` keeps only the packed-view layout and feature encoding (`view_layout`, `view_features`) that the
  actors and the learner share, and the `Prepared` / `Tick` records.

## How the rules are tested

1. **Transition parity** on 27,785 transitions recorded from the game (1v1 Bardur on Small Dryland maps, the game's
   AI on both seats, Crazy/Normal/Easy): each recorded command is applied to the imported world before it, and 37
   rule-relevant fields are compared with the world after it.
2. **Legal-move parity** on recorded positions: our legal set against the game's.
3. **Legal-move parity on positions our engine generates**, for four seats (no recorded four-seat games exist) and
   for more 1v1 positions.
4. **Unit tests** per rule (`tests/test_real_rules.py`): each rule on one hand-edited position with a control, plus
   random 1v1 and four-seat games that must never reject a legal action. `tests/test_search.py` covers the search.

The recordings and the parity harness depend on a local copy of the game and are not part of this repository; the
unit tests are.

## Every rule changed (research rules → game)

### Combat and units

| # | Rule | Research rules | Game |
|---|---|---|---|
| 1 | Cloak and Dinghy retaliation | retaliate (Cloak dealt 1 HP) | never: both are `Stiff` |
| 2 | Retaliation needs sight | any defender in range retaliates | only if the defender's owner has explored the attacker's tile (the attacker must be one of the defender's attack options) |
| 3 | Kill XP | every killer counts the kill | none for a vehicle (`Carry`: every boat) or a `Static` unit; a passenger keeps the xp it embarked with |
| 4 | Promotion at sea | a boat is Static, so its passenger can never be promoted | judged by the passenger: a Warrior/Archer/… with 3 xp is promoted on a boat |
| 5 | Knight Persist | `attacked` cleared after every kill | cleared only if another attack option remains from the new position |
| 6 | Wiped player | units removed, no refund | every unit disbanded with its refund; explains the earlier "Defenders vanish with +1 star" captures |

Checked and **equal**: one promotion level in both (896,025 unit observations in 14 recorded games never exceed
level 1, xp kept through all 156 recorded promotions); promotion heals to the new maximum (+5) and costs no action;
Heal Others gives a fixed +4 and Recover 4 in friendly / 2 elsewhere; unit stats otherwise equal the game data.

### Movement and sight

| # | Rule | Research rules | Game |
|---|---|---|---|
| 7 | Forest entry | a forest tile with a road can be crossed from anywhere | entering a forest ends the move unless the step runs road to road |
| 8 | Hide units (Cloak, Dinghy) | blocked by other players' units | may pass tiles held by other players' units (never end there) |
| 9 | Peace partners' units | block movement | may be passed (never ended on) |
| 10 | Ending on cities | anywhere free | an Infiltrate unit never ends on an opponent's city; no unit ends on a peace or cool-down partner's city |
| 11 | Bridges, crossing | any direction | without water movement (Fishing) a bridge cannot be entered straight across its span |
| 12 | Training sight | radius 1 (2 on mountains) | 2 for Scout units (Cloak) |
| 13 | Turn-start exploration | re-explores around every own city (radius 2 for capitals) and unit | none |

### Economy and building

| # | Rule | Research rules | Game |
|---|---|---|---|
| 14 | Away stars | meeting pays `1 + 2·min(⌈score/1000⌉, 5)` | +2 more when, with two players, the met player is player 1 (seat 2's first meeting) |
| 15 | Siege and markets | a sieged city pays nothing, markets included | a sieged city's own work stops; every owned market still pays |
| 16 | What is a siege | a visible foreign unit on the city | any other player's unit, hidden ones included |
| 17 | Negative progress | reduces the city's income including its markets | reduces only the city's own work |
| 18 | Roads on villages | allowed on an unowned village | never on a City tile, villages included |
| 19 | Bridges, building | only inside own borders; span from explored tiles | also on neutral water (`Freelance`, like roads); span from the true terrain of the opposite neighbours |
| 20 | Lighthouse | can be destroyed | not destructible |
| 21 | Peace partner on a tile | blocks building, harvesting, forest actions, destroying, neutral roads | does not block |
| 22 | Workable tiles | the tiles of an own city's territory row | additionally the tile itself must be owned by the player |
| 23 | Mountain Temple | unlocked by Philosophy (22) | unlocked by Meditation (21) |
| 24 | Monument terrain | Field, Water, Forest | Field, Water, Ice |
| 25 | Monument uniqueness | blocked by a monument standing in own territory (a captured one too) | blocked only by having built it |
| 26 | Grand Bazaar | Roads tech and 5 connected cities | 5 connected cities |
| 27 | Grow Forest | keeps the field's resource | the resource is removed |
| 28 | Star Fishing | allowed under an improvement (a port) | only on a tile without improvement |
| 29 | Literacy research price | `ceil(price · 0.66)` | `ceil(price · 0.66666f)` (differs from about 50 stars) |

Checked and **equal**: the rest of the tech tree, unit unlocks and the ability techs (Free Spirit disband, Chivalry
destroy, Shields peace, Diplomacy embassy/capital vision); the research price formula; per-city limits (`Limited`);
Pacifist / Wealth / Killer / Metropolis / Explorer monument conditions; temple growth; combat damage (no health
difference in thousands of recorded attacks); flag timing; ruins, explorer, Giant push and Dagger RNG procedures.

## Parity before and after

### Transitions: 27,785 recorded transitions

| ruleset | transitions | identical in every field | rejected by our engine |
|---|---|---|---|
| research rules (before) | 27,785 | 19,193 (69.1%) | 25 |
| game rules (after) | 27,785 | 19,619 (70.6%) | 12 |

The identical share is bounded by conventions, not rules: 4,596 field differences are "ours marks a moved unit that
cannot attack as attacked", 2,827 are the end-of-turn flag reset timing, 1,861 the game's Stay command (a no-op in
ours), 846 a structure-level number our engine never reads. Differences per cause, before → after:

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

The 12 remaining rejections: 11 city-reward choices that follow a ruins reward our RNG resolved differently (the
game's ruins levelled up a city, ours did not, so the reward is not pending in ours) and 1 peace offer (diplomacy is
imported only approximately: cool-down owner).

### Legal moves: recorded positions (every 40th recorded decision)

| ruleset | positions | identical legal sets | identical but for the fog exception | other differing actions |
|---|---|---|---|---|
| research rules | 693 | 214 | 380 | ours-only: build 434, destroy 285, move 57, attack 1; game-only: build 353, move 18, harvest 16, forest 15 |
| game rules (final, sampled per game) | 686 | 349 | **685** | ours-only: attack 1 |

The single remaining case is a hidden Cloak whose `attacked` flag our importer cannot read.

### Legal moves: positions our engine generated

| ruleset | positions | identical but for the fog exception | other differing actions |
|---|---|---|---|
| research rules, 1v1, 4 games 14×14 | 133 | 75 | ours-only: build 220, forest 45, harvest 8, capture 4; game-only: build 281, promote 15 |
| game rules, 1v1, 4 games 14×14 | 130 | **130** | none |
| research rules, four seats, 6 games 18×18 | 240 | 152 | ours-only: build 146, capture 8, move 6; game-only: build 171 |
| game rules, four seats, 6 games 18×18 | 237 | **237** | none |

## Remaining known differences

| Difference | Size | Why it stays |
|---|---|---|
| **Roads and bridges on unexplored tiles.** The game accepts a road (or neutral bridge) command on any tile, explored or not; ours requires the tile to be explored. | 3,273 legal actions in 693 recorded positions; never played by the game's AI | Deliberate. The game's UI cannot target an unexplored tile, and allowing it would leak hidden terrain to the policy through the legal mask. |
| **Difficulty handicap.** The game's capital pays +handicap (Easy 0, Normal/human 1, Hard 2, Crazy 4). | 958 transitions vs AI seats | Self-play seats are both human-equivalent (+1), as before. |
| **Random outcomes.** Ruins rewards, the Explorer reward walk, the tile a city's occupant is pushed to for a Giant, Dagger spawn tiles. | 942 | Same procedures, different dice. Distributions, not rules. |
| **Flag conventions.** End-of-turn vs turn-start flag reset; ours marks a moved non-Dash unit as attacked; Stay command. | 9,284 | No gameplay effect (a moved non-Dash unit cannot attack in either engine). |
| **Stale market level.** The game updates improvement levels at turn start in tile order from neighbours' *stored* levels, so a market can lag one turn behind a sawmill/forge that just changed. | 52 field differences, about 1 star for 1 turn | Tile-order bookkeeping; not ported. |
| **Capital connections after captures.** In 50 field differences (mostly long Easy games with repeated recaptures) the game drops a road link that ours keeps, costing the city its connected flag and 1 population. | rare | Route rule not pinned down. |
| **Resignation in four-seat games.** The game's wipe also disbands a resigning player's units; ours keeps them as obstacles. | none in self-play | Training runs with resignation off (`no_resign`). |
| One exploration and one end-turn star difference. | 2 transitions | Not investigated further. |

## Four seats

Four-seat games follow every rule above that applies to four seats (away stars do not: the game pays them only with
two players). Diplomacy stays off, as in v1.

The four-seat legal-move check covered six 18×18 games on the `four-v1` map mix (5,956 commands, none rejected by our
engine), every 25th decision compared. The engine equals the game's legal set in all 237 positions apart from the fog
exception; the research rules differed in 88 of 240 (Bridges, monuments, Star Fishing on ports, roads on another
player's tile, moves). A capture that takes a player's last city wipes them (units disbanded with refunds), as the
engine now does; zone of control counts every non-hidden enemy, seen or not, which matters more with three opponents.

**Cannot be verified for four seats**, because no recorded four-seat game exists: transitions (income, captures,
meetings, eliminations, turn order after eliminations) are checked only through the shared two-seat code and the
four-seat invariant tests (`tests/test_four_player.py`).

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
3. **Game rules to match:** peace offers are messages answered on the receiver's turn, breaking peace starts a
   cool-down that ends at a later turn start, embassies have a cost and pay income per partner, and the
   movement/build/attack effects already implemented above (#9, #10, #21).
4. **Observations.** The per-seat relation fields exist in the four-seat view (`kernels::VIEW_FIELDS`, one entry per
   seat for the five relation fields), so no new features are needed.
5. **Tests.** Four-seat diplomacy has no recorded data, so it would rest on invariant tests.

Rough size: 2–3 days of engine and network work plus a short grown-network smoke, no training budget beyond that.

## Digests and checkpoint compatibility

- New checkpoints: `ruleset = "polytopia-2.17.3"`, `simulator_sha256 = ruleset.rules_digest()` = `77cc7d6a712953a0b33c3e6c8d30c8129eba84cfa2b9ad691b65d84a692a0101`
  (sha256 of the ruleset name and `gpu/rules.json`).
- Loading (`runner.load_checkpoint` → `ruleset.compatible`): a checkpoint of this ruleset, or one whose digest is in
  `ruleset.LEGACY_DIGESTS` (the research rules; weights load unchanged and play under the game rules).
- Engine code digest (`scripts/engine_rs_digest.py`: sha256 over `engine-rs/src/*.rs`, `Cargo.toml`, `Cargo.lock`,
  `gpu/rules.json`).
