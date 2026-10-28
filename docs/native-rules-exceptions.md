# Corrected Bardur rules and reference compatibility

> **Retired 2026-10-27.** These research rules (and the JAX engine that defined them) are no longer used by anything that trains, evaluates or serves games. The game is now Polytopia's own rules (2.17.3), the only ruleset: see [rules-alignment-2026-10-27.md](rules-alignment-2026-10-27.md).

User decision, 2026-10-16: target correct native Bardur behavior; document and test deviations from Polyfish. `load_rules()` selects the native profile. `load_rules("polyfish")` is exclusively the pinned differential-testing profile. The subsequent user decision freezes [bardur-1v1-v1](ruleset-v1.md) as one consistent research environment. `FULL_GAME_READY=True` refers to that contract, not exhaustive equivalence to a commercial build. The audit notes below preserve provenance and commercial-fidelity uncertainties; they are not a backlog for unrequested version changes.

Reference: Polyfish `a9f90ab54607f5633556d02422d7457f548cc58f`. The vendor checkout remains unchanged. Golden tests select the reference profile explicitly. Native expectations live in `tests/test_native_rules.py`; they do not accept the defective reference afterstate as their expected result.

| Exception | Reference behavior and evidence | Native behavior | Regression |
|---|---|---|---|
| Embassy overwrites capital | `moves/abilities/diplomacy.rs` calls `create_structure(Embassy)` at a village. Saved executable counterexample: `reference-counterexamples/embassy-overwrites-city.json`. Capture generation requires Village. | Store directed embassies in player relations; preserve the city structure and capture eligibility. | `test_embassy_does_not_overwrite_capital_and_capture_remains_possible` |
| Knight Persist | `actions/units.rs` forced melee advance exhausts the unit; Persist skips setting attacked but does not clear it. | A killing attack leaves another attack available, including after advance. | `test_knight_chain_remains_available_after_melee_advance` |
| Mind Bender conversion | Convert helper exists but the normal attack dispatcher treats Mind Bender as a zero-damage attacker. | Conversion changes allegiance, releases original city capacity, preserves health/veterancy, consumes both units' actions, and does not retaliate or count a kill. | `test_convert_preserves_health_veterancy_and_releases_old_capacity` |
| Reward income counted twice | Workshop/Park modifies stored production and reference income adds reward counts again. | Income uses the production increment once. | `test_workshop_income_is_counted_once` |
| Special-tribe technology from ruins | Free-tech enumeration includes Waterways (48) and Rituals (121) for Bardur. | Ruin tech candidates are restricted to the ordinary tech tree. | `test_native_profile_excludes_special_tribe_ruin_techs` |
| Original capital identity | Reference city capture rewrites `capitalOf` to the captor. | Preserve original capital identity, including embassy eligibility after recapture. | Embassy/capture regression above |
| Disabled peace protocol | Reference offer/response generation is commented out; breaking peace only changes relation flags. | Offer requires response; peace blocks hostile actions. Breaking peace removes the breaker's units in the former ally's territory and imposes a grace period. | `test_peace_requires_response_blocks_attacks_and_has_break_grace_period` |
| Embassy income | Reference structure replacement cannot represent two independent embassies correctly. | Directed embassies contribute independently to both players. | `test_mutual_embassies_at_peace_pay_eight_each` |

Primary rules evidence: [Midjiwan diplomacy overview](https://polytopia.io/diplomacy/) and [Midjiwan's Diplomacy FAQ on Steam](https://store.steampowered.com/news/posts/?appids=874390&enddate=1659376434&feed=steam_community_announcements) (2022-06-02). The FAQ establishes non-hostility at peace, shared roads, a break grace period, removal of aggressor units, embassy income/capture behavior, and original-capital requirements. These historical rules must be checked against later changes where applicable.

Independent implementation cross-check: [GAIGResearch/Tribes](https://github.com/GAIGResearch/Tribes/tree/270e7e564630e5ef3dcc04c96dd66f58ef7273f3), especially `core/actions/unitactions/command/ConvertCommand.java`, Knight action status transitions, and `core/actions/cityactions/command/LevelUpCommand.java`. This older engine corroborates core conversion and Workshop behavior; it is not an authority for modern navy or diplomacy. No Java source is copied into the simulator.

Historical commercial-fidelity audit items included Cloak secrecy, diplomacy timing, network effects and UI equivalence. Procedural reset and long CUDA traces are now implemented and tested. The frozen release contract distinguishes tested research behavior from commercial fidelity; transition tests alone do not certify the latter.

## Additional implemented corrections

| Area | Native candidate behavior | Evidence / regression |
|---|---|---|
| Rafts and passenger skills | Rafts cannot attack. Native naval types do not inherit a passenger's Dash/Escape/Persist; upgrade requires friendly territory. | Official [Path of the Ocean](https://polytopia.io/path-of-the-ocean/); native Raft test. Reference passenger-Dash counterexample saved separately. |
| Naval disband | Refund half the transported land unit's cost; disband does not increment combat casualties. | Historical disband action and independent Tribes disband behavior; `test_native_disband_refunds_passenger_without_recording_a_combat_death`. |
| Modern economy | Markets sum adjacent multiplier levels, capped at eight; temples grow every two turns; Destroy requires Chivalry. | [Midjiwan 2024 balance announcement](https://store.steampowered.com/news/posts/?appids=874390&enddate=1714043324&feed=steam_community_announcements); market/temple/destroy tests. |
| Dynamic production buildings | Adding/removing a farm updates existing neighboring windmill population; ownership changes refresh affected multiplier contributions for both players. | Native adjacency/population regressions; ownership-change coverage remains incomplete. |
| Starvation | Losing population leaves city level and rewards intact; negative progress reduces income until replenished. | Tribes `City.addPopulation` and historical `DecreasePopulationAction`; `test_native_starvation_does_not_undo_city_levels_or_rewards`. |
| Captured home cities | Former owner's units lose the captured home assignment. Independent captors stay independent. | Historical `CaptureCityAction`; `test_capture_frees_enemy_home_capacity_and_keeps_independent_captor_independent`. |
| Monuments | Wealth requires Trade and 100 saved stars. Earned tasks persist after spending/disconnection. Tower requires the full ordinary tree, not a raw count containing special techs. | Tribes wealth/task state; native boundary regressions. |
| Hidden information | Unseen enemies do not project ZOC into the visible action mask. Hidden Cloaks appear empty; neutral road legality does not disclose them. | Paired-state view/catalog regressions and contact tests. Detect intentionally reveals only the nearby-enemy indicator. |
| Cloaks and infiltration | Native Hide/Detect, defended-city damage, one income-collection riot, independent Dagger/Pirate spawns, seeded placement priorities. | [Midjiwan's 2022 balance changes](https://store.steampowered.com/news/posts/?appids=874390&enddate=1669848363&feed=steam_community_announcements), native regressions and 32 isolated historical ordering cases. Current-build replay equivalence remains open. |
| Splash under peace | Juggernaut area damage does not harm an allied player. Actual hostile damage breaks embassy relations. | Peace non-hostility rule plus `test_juggernaut_stomp_respects_peace_and_breaks_embassy_only_on_damage`. |
| Native endings | Last-city capture and resignation terminate immediately, including resignation on turn zero. Finished states reject further actions. | Native capture-episode and resignation regressions. |
| Procedural RNG | Use explicit .NET sampling rather than a 31-bit value masquerading as a uniformly random 32-bit word. | Microsoft runtime vectors; [map-generation audit and exact-layout tests](map-generation.md). |

## Commercial-fidelity boundary (outside the frozen release)

The frozen profile follows the implementation and explicit compatibility choices in `ruleset-v1.md`. No silent changes are allowed. In particular, modern embassy timing/cost restrictions, conversion scoring, hidden contact/exhaustion, city adjacency after complex transfers and the full event/observation boundary still need more independent evidence.

The optional Polyterra probe contains historical decompiled logic with incomplete bundled game data (including missing defence unlocks). Its checked-in fixture establishes only the tested uniform-land spawn ordering and hashes. It is not an independent licensed engine or authoritative current-game oracle. The package stays outside source control and the GPU image; see `tools/native-probe/README.md`.

The final v1 fixes include friendly-unit path reconstruction and safe hidden-contact stopping, floored Bomber splash, and removal of a destroyed bridge's road. Independent regressions cover each. Five-star embassies and the existing explorer/ruin behavior remain fixed by user instruction.
