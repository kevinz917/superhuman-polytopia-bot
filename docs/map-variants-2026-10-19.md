# Map variants (2026-10-19)

Today's training maps come from the frozen engine generator (`gpu.mapgen`, Drylands): about 95% land,
~13% mountain, a random map per game from seeds 200000–299999. The new generator
(`src/polytopia_bot/maps/variants.py`, revision `bardur-variants-v1`) adds nine approved map types. It sits
outside `gpu/`, so the engine digest that checkpoints verify is unchanged. A map only changes the game's
starting state (`maps/state.py: initial_state`, bitwise equal to `mapgen.reset` for a classic layout).

| Type | Land | Water | Mountain | Forest | Villages | Landmasses | Capitals on same landmass |
|---|---|---|---|---|---|---|---|
| classic (frozen, today) | 95% | 5% | 13% | 53% | — | 1 | yes |
| drylands | 95% | 5% | 12% | 29% | 10.7 | 1.1 | 100% |
| lakes | 68% | 32% | 11% | 17% | 8.9 | 1.8 | 100% |
| pangea | 59% | 41% | 12% | 14% | 7.5 | 1.2 | 100% |
| continents | 45% | 55% | 9% | 11% | 5.7 | 3.0 | 0% |
| archipelago | 43% | 57% | 6% | 9% | 13.8 | 8.0 | 23% |
| waterworld | 29% | 71% | 3% | 6% | 10.8 | 10.7 | 2% |
| highlands | 90% | 10% | 30% | 16% | 9.2 | 1.2 | 100% |
| ridge (wall + 2–3 passes) | 89% | 11% | 17% | 22% | 10.3 | 1.2 | 100% |
| jungle | 84% | 16% | 9% | 46% | 10.2 | 1.5 | 100% |

(Means over 500 seeds; percentages of all 196 tiles. Capitals are always ≥6 tiles apart, Chebyshev; mean 7.2.)
Previews: `artifacts/map-variants/*.png` (rendered on Modal, original placeholder art).

## Wiring

- `maps/stream.py`: `MapSource(mix)`; a mix is a type name, `classic`, `uniform`, `proposed-v1`, or a weight
  dict. The map type is a pure function of the seed, so arenas replay exact boards.
- `CpuGames(map_batch=True)` starts and restarts games from per-lane layouts; `cpu_arena.match(maps=...)`;
  distributed actors take `LearnerConfig.maps` (default `classic` = unchanged behaviour) and report games and
  truncations per map type to the learner log.
- Tests (`tests/test_maps.py`, run on Modal): classic start bitwise equals `mapgen.reset`; layout invariants
  for 60 seeds of every type; fast and frozen CPU engines identical on variant maps through resets.

## Playability with the current bot (u3142), 24 maps × 2 seats per row, frozen engine

| Map type | Self-play (sampled) W–L–trunc | mean turns | games with a port | vs league96 W–L–trunc |
|---|---|---|---|---|
| classic | 21–22–5 | 43 | 29/48 | 44–4–0 |
| drylands | 20–25–3 | 40 | 20/48 | 36–10–2 |
| lakes | 19–15–14 | 57 | 30/48 | 40–6–2 |
| pangea | 18–15–15 | 52 | 31/48 | 34–11–3 |
| continents | 17–9–22 | 74 | 48/48 | 43–3–2 |
| archipelago | 16–11–21 | 81 | 48/48 | 33–11–4 |
| waterworld | 8–13–27 | 80 | 48/48 | 38–9–1 |
| highlands | 18–15–15 | 57 | 26/48 | 35–7–6 |
| ridge | 17–14–17 | 61 | 31/48 | 35–5–8 |
| jungle | 21–21–6 | 48 | 26/48 | 36–8–4 |

- Every type plays to completion with no illegal actions; the bot already builds ports and crosses water.
- It still beats league96 on every type (69–93%), weakest on archipelago and pangea.
- The gap is finishing games: self-play hits the 100-turn cap in 44–56% of water-map games (10% on classic),
  so on exactly the new maps most strong-vs-strong games give no win/loss signal.
- Seat balance on water maps looks uneven in these small samples (e.g. waterworld seat 1 won 1 of 24
  self-play games); evaluation always uses both seats, and balance is re-measured on larger runs.

## Campaign 3: 4 hours of self-play on the map mix (2026-10-19, 11:23–15:23 PDT)

Pure self-play from u3142 (80% current policy, 20% the run's own earlier snapshots incl. u3142; no outside
bots), map mix `proposed-v1` (15% classic, 85% new types), 4×H100 learner, lr 2e-4, 3 epochs, 262k batch.
Arm A (turn cap bootstraps the critic) and arm B (turn cap = draw) ran 15 actor containers each; B trailed
at both ladder rounds (71% vs 85% decisive on mixed maps) and was stopped at 13:00, its actors moving to A
(24 containers, ~110–125k samples/s). Delivered: `c3-maps-ctl` update 4854 (sha 85049f635cb4…),
`/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz`. ~1.3B learner samples.

Final evaluation vs frozen u3142, deterministic, frozen engine, fresh development maps, both seats:

| Map set | Games | W–L–trunc | Win rate [95% CI] | Of decisive games |
|---|---|---|---|---|
| proposed-v1 mix (318000+128) | 256 | 179–34–43 | 70% [65, 75] | 84% |
| classic only (318500+128) | 256 | 171–59–26 | 67% [62, 72] | 74% |
| all types equally (320000+96) | 192 | 144–15–33 | 75% [69, 80] | 91% |

Wins on every map type (equal-mix set): waterworld 23–3, lakes 22–2, classic 17–4, archipelago 16–1,
continents 16–0, ridge 14–2, highlands 12–0, jungle 9–2, pangea 9–1, drylands 6–0.

Self-play games that hit the 100-turn cap, same 24 maps × 2 seats per type as the u3142 test above:

| Type | u3142 | c3 final |
|---|---|---|
| classic | 5/48 | 9/48 |
| drylands | 3/48 | 8/48 |
| lakes | 14/48 | 8/48 |
| pangea | 15/48 | 7/48 |
| continents | 22/48 | 14/48 |
| archipelago | 21/48 | 4/48 |
| waterworld | 27/48 | 7/48 |
| highlands | 15/48 | 14/48 |
| ridge | 17/48 | 19/48 |
| jungle | 6/48 | 8/48 |
| total | 145/480 (30%) | 98/480 (20%) |

Water maps now finish (waterworld 56% → 15% stalled, archipelago 44% → 8%); mountain chokepoint maps
(ridge, highlands) still stall about a third of the time. Recruit mix broadened: Warrior 44%, Catapult 27%,
Cloak 13%, Defender 6%, plus naval Scoutships and Rammerships (5% and 3%), versus 63% Warrior / 30% Catapult
for u3142 against league96. Training-time policy entropy rose from 1.6 to 2.4 over the run (KL stayed ~0.009).
