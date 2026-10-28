# Frozen research rules: `bardur-1v1-v1`

> **Retired 2026-10-27.** These research rules (and the JAX engine that defined them) are no longer used by anything that trains, evaluates or serves games. The game is now Polytopia's own rules (2.17.3), the only ruleset: see [rules-alignment-2026-10-27.md](rules-alignment-2026-10-27.md).

Frozen 2026-10-16 at the user's direction: deliver one working, internally consistent simulator, without chasing the latest commercial version. This decision supersedes the earlier request to track release 2.17.3. `FULL_GAME_READY` means ready to run complete games under **this research contract**, not certified commercial-game equivalence. No further balance changes belong in v1; a rules change requires a new identifier and new validation receipts.

## Supported experiment

Two Bardur players, 11×11 or 14×14 boards, cumulative exploration with fog, all ordinary technologies and units including modern naval units, Cloaks and Daggers. Main experiments use 14×14. Maps use `bardur-drylands-dotnet-v1`, the GPU implementation of corrected reference-derived Drylands generation. It matches the corrected CPU layout oracle, not official game seeds. Special tribes and imported commercial saves are outside this contract.

Players start with a level-one capital, one Warrior, Hunting, and five stars. An atomic action is one move, attack, research, construction, harvest, ability, city reward, capture, end turn or resignation. City rewards must be resolved before ordinary actions. Player one starts; a round advances after player two ends its turn. At the default limit, advancing from round 100 to 101 truncates the game. A separate 10,000-action limit also truncates. These are safety limits, never score victories.

Capturing the opponent's last city or resignation ends the game immediately, with seat-indexed rewards +1 and −1. A truncated state has zero terminal reward and a distinct truncation flag; it has no resolved game outcome. Rejected, stale and post-terminal actions leave the state unchanged.

## Fixed mechanics and compatibility choices

The checked-in rule tables, JAX transition implementation and golden fixtures jointly specify v1. `load_rules()` selects v1; `load_rules("polyfish")` selects the separate differential-validation profile pinned to `a9f90ab54607f5633556d02422d7457f548cc58f`. The original source checkout is unchanged.

| Area | v1 behavior |
|---|---|
| Economy | Ordinary tech graph, costs and structures from pinned tables; dynamic adjacency populations and market income, markets capped at eight; temples gain 100 score per level every two rounds. City starvation preserves levels/rewards and reduces income. |
| Movement | Eight-neighbor paths, terrain access, stopping rules, half-cost roads, friendly-unit transit, naval embark/upgrade/disembark. Hidden contact reveals the first Cloak and stops at the last empty path tile without overwriting occupants. |
| Combat | Health-scaled damage and retaliation, native Knight chains, conversion, recovery, promotion, splash and Stomp. Splash damage is floored. Rafts cannot attack; ships do not inherit passenger Dash/Escape/Persist. |
| Ownership | Capture preserves original capital identity; former owner's home assignments to the captured city are released. Independent captors remain independent. |
| Diplomacy | Directed relation-based embassies cost **five** stars. Each direction gives both players +2 income, doubled at peace. Attack/conversion destroys embassies and applies the implemented embassy cooldown. Offers are explicit actions with explicit accept/reject responses. |
| Breaking peace | Removes the breaker's units inside the other player's territory without a refund, exhausts remaining units, and blocks mutual attacks/captures until the breaker's next turn. It does not automatically end the turn. |
| Cloaks | First movement hides the unit and prevents same-turn infiltration. Detect exposes an adjacent-hidden-unit boolean. Infiltration damages defenders, steals innate city income, blocks one income collection, and spawns at most five independent Daggers/Pirates using fixed seeded priorities. |
| Random effects | Explicit per-game seeds, fixed xxHash-based explorer/ruin/infiltration choices, 12-step reference explorer policy. These are reproducible research semantics; current commercial stochastic equivalence is not claimed. |
| Policy view | Allowlisted visible terrain/resources/units, owned economic and unit state, relations and achievements. Opponent bank/tech and hidden state/RNG are excluded. This is a research observation schema, not a reproduction of every UI statistic or event log. |

Five-star embassy pricing, the break-peace removal policy, reference explorer/ruin selection, conversion scoring and path tie ordering are intentionally frozen as implemented. The source audits in [native-rules-exceptions.md](native-rules-exceptions.md) preserve where commercial fidelity was uncertain. They are not promises of a subsequent version upgrade. Future changes must be explicit experiments, not silent reinterpretations of v1.

## Runtime and evidence

`reset`, legal masks, action decoding, `step`, observations and batched scans execute in JAX without a CPU rules engine. Host code handles orchestration, text/model transport, logging and private snapshot I/O. The factored catalog has no top-K action pruning. One unit occupies a tile; no unit-list capacity silently discards entities.

Snapshots preserve every semantic state field and require an exact schema/profile/implementation digest. The final manifest `configs/ruleset-v1.json` identifies the GPU source and rule-table hashes. Source-specific test and timing receipts live under `docs/benchmarks`; see [gpu-simulation.md](gpu-simulation.md). The map fixture tests, transition tests, controlled capture episode, random long episodes and performance benchmarks establish different facts and must not be conflated.

The random action driver is a test harness. Its high truncation rate says nothing about an expert policy. Readiness does not mean competitive bots, demonstration quality, efficient LM serving, successful SFT or cross-task transfer have been established.
