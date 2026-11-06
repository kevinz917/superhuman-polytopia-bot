# Rules parity, pass 2 (2026-10-31): summary

A second pass bringing engine-rs closer to the game's own rules (Polytopia 2.17.3, game version 123), on top of the
first alignment (docs/rules-alignment-2026-10-27.md). Four pieces of work:

| piece | effect |
|---|---|
| No round limit in the rules | engine unlimited unless a harness passes `max_turns`; N plays exactly N rounds (was N+1); training default 200 rounds (300 on 18x18); Studio human play unlimited |
| Units | Dagger spawns face their city; no resource bonus under an improvement; bump-reveal, splash and convert facing; a hidden unit pushed for a trained one is revealed; city-reward Explorer, infiltration length and ruins list confirmed equal |
| Actions and observation | Disembark in place on a bridge; four-seat diplomacy per pair; unit facing in the observation (obs version 2; old checkpoints keep their input width and still load) |
| Connections | city connections are a flood from the capital (neutral villages link chains; pairwise ownership); shared per-tile route flags, router step cost 5/10; turn-start market level from stored neighbour levels; road terrain field/forest |

The tests for each piece are `tests/test_rules_parity_units.py`, `tests/test_rp2_actions.py`,
`tests/test_rules_parity_connections.py` and `tests/test_engine_parity.py`.

## Parity with recorded games

Checked against games recorded from the real game:

| check | before | after |
|---|---|---|
| 27,785 recorded transitions identical in every field | 19,742 | **19,790** |
| recorded commands our engine rejects | 0 | **0** |
| recorded positions with the game's legal set, apart from the fog-road rule | 686 / 686 | **686 / 686** |
| generated positions, 2 / 4 seats | 160/160, 240/240 | **160/160, 240/240** |

The 11,238 remaining field differences, by cause:

| count | cause | gameplay effect |
|---|---|---|
| 4,596 | ours sets `attacked` on a moved unit without Dash; the game leaves it and blocks the attack by "moved and no Dash" | none: every reader of the flag (attack, recover, heal, break ice) also requires not moved |
| 2,827 | the game clears moved/attacked at end of turn, ours at the owner's next turn start | none |
| 1,861 | the game's "stay" command (marks a unit done) is a no-op in ours | none: a voluntary pass |
| 970 | the built-in AI's difficulty bonus income (Crazy +4) vs ours +1 for both seats | deliberate: human vs human |
| 896 | structure level numbers our engine never reads; lighthouse level counts explorers | none |
| 78 + 8 | explorer walk / ruins: attributed to the lighthouse representation | none known |
| 1 + 1 | an end-turn star case and one Knight Persist row | unexplained |

Still deliberate: roads and bridges cannot target unexplored tiles (the game's UI never allows it); no resignation.
Not observable in recorded games, so resting on unit tests only: Disembark, four-seat diplomacy, the order of
answering simultaneous peace requests. Not modelled because Bardur can't reach them: tribe overrides, Boost, the
10,000-pop flood cap.
