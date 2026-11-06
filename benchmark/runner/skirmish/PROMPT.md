# Polytopia Skirmish

A private extension of this repository. Models write Rust bots, practice against
our frozen 1v1 reference, inspect their own game traces, and revise under a wall
clock budget. Frozen submissions then play a seat-paired round robin and get a
Bradley–Terry Elo ladder anchored to the reference.

The simulator is the existing `engine-rs` through `studio.environment.RustEnvironment`.
No substitute game implementation. Version 1 is Bardur, 14×14 classic maps,
`polytopia-2.17.3`, two seats. All compilation, models, games and artifacts stay
on Modal staging. Default writing budget: set by the controller, including practice and review.

The primary reference is **hero-1v1** (greedy deterministic, update 11650),
SHA-256 `e6f4e1977e7f9cdcfbbe17358d87e49d638d82150496a6d4099e0cd389b0934f`.
It is the strongest released raw 1v1 policy according to its qualification report;
this does not assert it beats every search configuration on every map distribution.
The checkpoint has no trained fog value head, so it is evaluated without search.

## Model-facing tools

- `read_file(path)`: read the Rust starter, game rules or engine reference source.
- `write_source(source)`: replace the bot's `src/main.rs`.
- `compile()`: offline, locked Cargo build to `wasm32-wasip1`.
- `campaign(maps)`: practice against hero, both seats per map; returns W/L/failures and trace IDs.
- `review(game_id, offset, limit)`: page through the bot's own public observations/actions.

Every model gets the same starter and tools, not the reference weights, source,
or test maps. There is no browser, package installer, arbitrary host shell, or
network tool. The authoring container has Modal `block_network=True`; Cargo's
cached dependencies are prepared before timing. Only Rust `std` and `serde_json`
are available. The Rust engine source and rule tables are readable references.
Generated code runs in Wasmtime with WASI stdin/stdout only, no preopened
folders, no environment or network, 256 MiB memory, a fixed instruction-fuel
budget and decision timeout. Process memory persists within a game, then resets.

## Bot wire format

Read one JSON line, choose one legal action, print one JSON line and flush:

```json
{"action_id":"130866"}
```

Input is the existing public `Observation`:
`player`, `state_version`, `visible_state`, `remembered_state`, `legal_actions`.
`visible_state` includes board size, turn, stars, known technology IDs, tiles,
visible units and cities, relations and achievements. Each legal action has a
stable string `action_id` and command `{kind, source, target, parameter}`.
Tile index is `y * 14 + x`; unknown tiles contain `known: false` only.
Own-unit flags and city economy are visible; enemy private tech, bank, RNG seed,
full state and checkpoint never enter the observation. Persistent public memory
is also supplied. The starter only ends turns (or chooses mandatory rewards).

Command kinds: 1 move, 2 attack, 3 ability, 4 recruit, 5 harvest, 6 build,
7 research, 8 capture, 9 city reward, 10 end turn, 11 resign. IDs and numeric
unit/tech/resource definitions are in `rules.json` and the reference Rust code.
Actions must come from the supplied catalog; the referee validates each action.

## Protocol and scoring

Practice uses a dedicated partition. Final games use fresh, hidden map seeds
selected by the trusted controller **after every entrant is frozen**. These
partitions are separate from existing training/acceptance allocations. Version 1
uses classic maps; mixed-map and board-size tracks need separate versioned protocols.
Each map is played from both seats. Run IDs and submissions are immutable;
source, Wasm, prompt, engine, checkpoint, settings and tool/API transcript hashes
are retained on the dedicated `polytopia-skirmish` volume.

Compilation failures are failed submissions. Illegal moves, malformed output,
fuel exhaustion and decision timeouts are forfeits, separately counted from
native defeats. Simulator round/action caps and trusted game-time cutoffs are
**unresolved**, never silently converted into draws or score-based victories.
Elo fits only resolved games; reports include unresolved coverage, per-seat
results and per-pair results. A weak symmetric prior makes all-win/all-loss
fits finite. Ratings are relative to this roster; one short run is a pilot,
not a reliable model ranking. Repeated independent writing runs are needed.


## Explicit runtime and authoring allowances

The controller states the active writing budget and provides a neutral budget
status with tool results. `budget_status()` reads it again. A revision is a
changed `write_source` call, including a version that fails to compile; identical
writes cost no revision. The starter is revision 0. Compilation, reading and
review do not consume revisions, but use wall time and the stated request/game
allowances. After your last permitted edit you may compile/test/review, then
call `finish()`. A failed final edit is a failed submission, not an automatic
rollback to an earlier stronger bot.

For this runtime track, each bot gets 2 seconds per decision, with a one-time
5-second startup allowance, 256 MiB Wasm memory, and 100 billion Wasm fuel units
per game. A game has a 180-second referee cap, the simulator's 100-round cap and
10,000-action cap. Runtime faults are forfeits; referee/simulator caps are unresolved.
These limits apply equally to all generated bots and are independent of prompt framing.
