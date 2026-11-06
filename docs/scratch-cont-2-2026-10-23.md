# Scratch-cont replays, seat/map effects, and the scratch-cont-2 run (2026-10-22/23)

**Result:** a 7-h continuation of scratch-cont-1 (run `scratch-cont-2`) is the new strongest bot,
**+442 ± 14 Elo** (c3 = 0) vs scratch-cont-1 +383 ± 13, winning **58%** head to head over 512 mixed-map games
(decisive map pairs 54–14). Cost ~$185 of a $200 cap. Final checkpoint: `/results/distributed/scratch-cont-2/final.npz`.

Along the way: the 68% "scratch-cont-1 vs Scratch 7h" number from the old final match was classic maps only;
seats were always paired, but most boards are decided by seat, not by bot.

## 1. Studio replays

`experiments/modal_scratch_replays.py` (Scratch 7h vs OMNI c3) and `experiments/modal_scratch_cont_replays.py`
(Scratch-cont 6h vs Scratch 7h; `--mirror-seed/--mirror-kind` replays one board with seats swapped). Raw
deterministic policies, one 4-core Modal container per game, files written into the local Studio library
(`artifacts/studio-d3142/replays`, not committed).

| Map (seed) | Blue | Red | Winner | Turn |
|---|---|---|---|---|
| classic (321200) | Scratch 7h | OMNI c3 | OMNI c3 | 22 |
| continents (321201) | OMNI c3 | Scratch 7h | Scratch 7h | 42 |
| archipelago (321202) | Scratch 7h | OMNI c3 | Scratch 7h | 36 |
| pangea (321210) | Scratch-cont 6h | Scratch 7h | Scratch-cont | 27 |
| waterworld (321211) | Scratch 7h | Scratch-cont 6h | Scratch 7h | 41 |
| highlands (321212) | Scratch-cont 6h | Scratch 7h | Scratch-cont | 32 |
| waterworld (321211, seats swapped) | Scratch-cont 6h | Scratch 7h | Scratch-cont | 26 |

Scratch-cont wins with archer-heavy armies (39–42 archers, cloaks, catapults on highlands) and forest clearing.

### Waterworld 321211 loss

- **Board:** 9 villages; 6 nearer Blue's capital (5,9), 2 nearer Red's (3,2), 1 contested. Red's corner is walled
  in by ocean. Both seat assignments produced the same village split and Blue won both games, so the seat decided it.
- **Red's play (Scratch-cont):** stuck at 4 cities from turn 7 to 29 while Blue reached 7. Unit cap is
  level + 1 per city (`gpu/military.py: summon_mask`), so 4 cities at levels 1/2/3/3 cap Red at 13 units, reached
  around turn 17. Red then banked 17–30 stars from turn 18 to 31, researched nothing from turn 20 to 31, didn't
  grow its cities, and built only Defenders, which can't answer scoutships bombarding from the water. Scratch 7h in the
  same seat spent its stars every turn (≤ 5 banked) but lost faster.
- **Value head:** the policy's own value estimate stayed flat near −0.05 for both sides all game, so it gives no signal
  about when the game was lost.
- `scripts/replay_summary.py` and `scripts/replay_timeline.py` reproduce these numbers from the replay files.

## 2. Seat and map effects in evaluation

- `cpu_arena.match` always plays each seed in both seats (`items = [(s, seat) for s in seeds for seat in (1, 2)]`),
  so seat advantage cancels in the headline rate.
- `elo_ladder` / `arena_match` default to `map_mix=None` = classic only. The scratch-cont-1 final match (68% vs
  Scratch 7h, 66% vs OMNI c3) never saw a water map. As Blue the candidate won ~79–83%, as Red ~50–57%.
- **Mixed-map check** (`arena scratch-cont-vs-overnight-mixv1-380000`, proposed-v1, 64 maps × 2 seats, 390 s on one
  32-CPU container, ~$0.25): Scratch-cont 6h vs Scratch 7h 81/128 = 63% (ci95 57–70%).

  | Maps | Win rate | As Blue | As Red |
  |---|---|---|---|
  | water (archipelago, waterworld, lakes) | 31/46 (67%) | 16/23 | 15/23 |
  | land (all other types) | 50/82 (61%) | 31/41 | 19/41 |

  Per board: 21 won both games, 4 lost both, 39 split. Of the 39 split boards, Blue won both games on 26 and Red on
  13. So 61% of boards are decided by seat. On the 25 boards the bots decided, Scratch-cont won 21.
  `scripts/arena_by_type_seat.py` produces this breakdown.
- Takeaway: evaluate on `proposed-v1` with paired seats, and read decisive pairs (2-0 vs 0-2) alongside the win rate.
  Symmetric maps would cut noise but not the first-move edge, and would remove the play-from-behind situations where
  the bot is weakest.

## 3. scratch-cont-2 run

**Recipe:** identical to scratch-cont-1 except the parent and seed:
- parent `scratch-cont-1/final.npz` plus `critic_path` to its final critic;
- `proposed-v1` maps, epochs 1, lr 3e-4, batch 262k, turn γ 0.99, omniscient critic, seed 505;
- 1×H100 learner plus 8 × 32-process actor containers, launched from main 0b6e552;
- 25200 s of training, about 11,500 updates at 119k samples/s.

It ran 7 h rather than the planned 8 because Modal bills ~$22/h for this setup (scratch-cont-1: $125 for 6 h), while
`launch()` reserves at $13.82/h. The ledger `runs/scratch-cont-2` (budget_usd 200) therefore holds a $60
billing-margin row.

**Evaluation:**
- **Reference round robin:** the 4 references played each other on mixed maps, 6 pairs × 200 games, seeds 391000+.
- **Hourly ladder:** latest checkpoint vs c3, OMNI c3, Scratch 7h and Scratch-cont 6h, 100 proposed-v1 maps × 2 seats
  per opponent (800 games per checkpoint), fresh seed blocks 390000+.
- **Final:** final model and update 8600 vs the same 4, 256 maps × 2 seats each (512 games per opponent),
  seeds 392000+.
- **Elo:** Bradley-Terry over all games (`scripts/ladder_elo.py: fit`, anchor c3).

| Player | Elo (c3 = 0) | ± |
|---|---|---|
| **scratch-cont-2 final** | **+442** | 14 |
| update 11450 (hour 7) | +436 | 18 |
| update 9700 (hour 6) | +433 | 18 |
| update 8600 (hour 5.5) | +433 | 13 |
| update 1750 (hour 1) | +416 | 18 |
| update 3700 (hour 2) | +413 | 18 |
| update 6600 (hour 3.5) | +403 | 18 |
| scratch-cont-1 (Scratch-cont 6h) | +383 | 13 |
| scratch-overnight-1 (Scratch 7h) | +295 | 13 |
| c5-omni-s1-final (OMNI c3) | +231 | 13 |
| c3 u4854 | 0 | — |

**Shape of the curve:** about +30 Elo in the first hour, flat to hour 3.5, another +25–30 by hour 5.5, then flat.
More hours of this recipe buy little.

**Final model:**
- Win rate: 58% vs Scratch-cont 6h, 69% vs Scratch 7h, 76% vs OMNI c3, 92% vs c3.
- Against Scratch-cont 6h: 60% on land vs 52% on water; 68% as Blue vs 48% as Red. Water maps and the Red seat remain
  the weak spots.

Full per-checkpoint table: appendix.

**Cost:** ~$185 total. Training $162; round robin, hourly ladders and finals ~$20; plus a few dollars of failed
or cancelled evaluation apps.

### Incidents

1. **Ladder hang.** The first hourly `elo_ladder` (call fc-01M5KQ4NXD61ZZAEFD0S4STVJD) silently stopped after round 2.
   The call stayed "running", but it neither copied the next checkpoint nor wrote history for 95 minutes. I cancelled
   it and the scheduled final ladder and started a replacement. Modal later preempted and replayed the replacement,
   which restarted at round 0 and reused seed block 390200. The checkpoint was still fresh, so this only matters for
   map reuse.
2. **Mac sleep.** The local guard (`experiments/scratch_cont2_guard.py`: budget brake, finals launcher, report writer)
   ran under `caffeinate -i`, but the Mac slept anyway from 06:18 to 07:09, probably with the lid closed. Training was
   unaffected. When the Mac fell back asleep during a detached `modal run`'s app creation, the launch failed with
   "app is stopped or disabled"; 4 of the 8 final matches died this way and were relaunched under `caffeinate -s`.
3. **Lessons:** schedule final tournaments on Modal (a ladder round with `first_delay`), not from a local guard. Check
   that ladder history keeps updating. Use `caffeinate -s` on AC power.

### Not addressed

The boxed-in star-hoarding behaviour from the waterworld loss. Candidates: style-biased frozen opponents and mid-game
start states (see the human-game diversity plan).

## Appendix: per-checkpoint results (200 games per opponent; final 512)

Columns after win %: per-map pairs 2-0/split/0-2, land and water win rate, win rate as Blue and as Red, games ending at
the turn cap.

| round | checkpoint | opponent | win % | per-map pairs 2-0/split/0-2 · land · water · seat · capped |
|---|---|---|---|---|
| 1 | update-1750 | c3 | 90% (179/200) | pairs 79/21/0 · land 89% · water 90% · Blue 96% · Red 83% · capped 3 |
| 1 | update-1750 | c5omni | 71% (142/200) | pairs 45/52/3 · land 71% · water 71% · Blue 80% · Red 62% · capped 2 |
| 1 | update-1750 | overnight | 71% (142/200) | pairs 44/54/2 · land 71% · water 71% · Blue 80% · Red 62% · capped 4 |
| 1 | update-1750 | scratch-cont-1 | 52% (105/200) | pairs 16/73/11 · land 53% · water 52% · Blue 62% · Red 43% · capped 4 |
| 2 | update-3700 | c3 | 92% (184/200) | pairs 84/16/0 · land 95% · water 87% · Blue 95% · Red 89% · capped 1 |
| 2 | update-3700 | c5omni | 74% (148/200) | pairs 50/48/2 · land 75% · water 73% · Blue 83% · Red 65% · capped 2 |
| 2 | update-3700 | overnight | 68% (135/200) | pairs 35/65/0 · land 71% · water 62% · Blue 78% · Red 57% · capped 1 |
| 2 | update-3700 | scratch-cont-1 | 52% (103/200) | pairs 13/77/10 · land 51% · water 53% · Blue 65% · Red 38% · capped 0 |
| 3 | update-6600 | c3 | 90% (179/200) | pairs 80/19/1 · land 90% · water 89% · Blue 92% · Red 87% · capped 4 |
| 3 | update-6600 | c5omni | 76% (153/200) | pairs 53/47/0 · land 76% · water 77% · Blue 81% · Red 72% · capped 4 |
| 3 | update-6600 | overnight | 61% (122/200) | pairs 25/72/3 · land 63% · water 56% · Blue 70% · Red 52% · capped 1 |
| 3 | update-6600 | scratch-cont-1 | 52% (104/200) | pairs 13/78/9 · land 53% · water 50% · Blue 57% · Red 47% · capped 2 |
| 4 | update-8600 | c3 | 94% (187/200) | pairs 87/13/0 · land 96% · water 88% · Blue 98% · Red 89% · capped 3 |
| 4 | update-8600 | c5omni | 74% (149/200) | pairs 51/47/2 · land 76% · water 70% · Blue 84% · Red 65% · capped 4 |
| 4 | update-8600 | overnight | 68% (136/200) | pairs 38/60/2 · land 68% · water 68% · Blue 82% · Red 54% · capped 3 |
| 4 | update-8600 | scratch-cont-1 | 56% (113/200) | pairs 23/67/10 · land 58% · water 54% · Blue 70% · Red 43% · capped 6 |
| 5 | update-9700 | c3 | 92% (183/200) | pairs 83/17/0 · land 92% · water 90% · Blue 94% · Red 89% · capped 5 |
| 5 | update-9700 | c5omni | 80% (160/200) | pairs 60/40/0 · land 78% · water 84% · Blue 83% · Red 77% · capped 4 |
| 5 | update-9700 | overnight | 66% (132/200) | pairs 35/62/3 · land 69% · water 60% · Blue 72% · Red 60% · capped 1 |
| 5 | update-9700 | scratch-cont-1 | 53% (106/200) | pairs 15/76/9 · land 54% · water 50% · Blue 61% · Red 45% · capped 6 |
| 6 | update-11450 | c3 | 92% (185/200) | pairs 85/15/0 · land 94% · water 89% · Blue 99% · Red 86% · capped 3 |
| 6 | update-11450 | c5omni | 72% (145/200) | pairs 47/51/2 · land 72% · water 75% · Blue 80% · Red 65% · capped 3 |
| 6 | update-11450 | overnight | 68% (137/200) | pairs 38/61/1 · land 72% · water 61% · Blue 80% · Red 57% · capped 2 |
| 6 | update-11450 | scratch-cont-1 | 60% (120/200) | pairs 27/66/7 · land 59% · water 62% · Blue 69% · Red 51% · capped 1 |
| final | final | c3 | 92% (472/512) | pairs 218/36/2 · land 92% · water 93% · Blue 94% · Red 91% · capped 4 |
| final | final | c5omni | 76% (390/512) | pairs 139/112/5 · land 74% · water 83% · Blue 82% · Red 71% · capped 9 |
| final | final | overnight | 69% (355/512) | pairs 106/143/7 · land 71% · water 66% · Blue 79% · Red 60% · capped 4 |
| final | final | scratch-cont-1 | 58% (296/512) | pairs 54/188/14 · land 60% · water 52% · Blue 68% · Red 48% · capped 9 |
| final | update-8600 | c3 | 91% (468/512) | pairs 212/44/0 · land 92% · water 89% · Blue 96% · Red 87% · capped 4 |
| final | update-8600 | c5omni | 75% (385/512) | pairs 134/117/5 · land 73% · water 81% · Blue 79% · Red 71% · capped 10 |
| final | update-8600 | overnight | 68% (348/512) | pairs 98/152/6 · land 68% · water 69% · Blue 76% · Red 60% · capped 2 |
| final | update-8600 | scratch-cont-1 | 56% (288/512) | pairs 48/192/16 · land 58% · water 52% · Blue 64% · Red 48% · capped 12 |
