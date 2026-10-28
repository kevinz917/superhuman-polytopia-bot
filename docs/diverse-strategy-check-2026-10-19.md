# Diverse-teacher PPO: do the live checkpoints play different strategies? (2026-10-19)

Read-only check of the newest retained snapshots of the two live H100 runs (training untouched):

| label | run | snapshot | sha256 (12) |
|---|---|---|---|
| **anchor112** | `diverse-h100-anchor-v1` (repertoire anchor 0.1, lr 1e-4) | update-112 | 5ddaed5f4c29 |
| **lr3e5u80** | `diverse-h100-lr3e5-v1` (no anchor, lr 3e-5) | update-80 | d55f310b25c3 |

Each snapshot played 3 public-interface qualification jobs on fresh development seeds (16 maps × 2 seats = 32 games per job). Every game was traced and analyzed:
- vs the balanced teacher (seeds 303000–303015)
- vs league96 (303020–303035)
- stochastic vs deterministic self-match (303040–303055)

## Bottom line

1. **Unit play really is no longer Warrior-only.** Old bots built 98–100% Warriors. In nearly every game they could build Catapults, Archers and Riders, and they never did.
   - Both new snapshots field mixed armies: Warriors are only 24–43% of recruits in deterministic play.
   - They build a non-Warrior unit in 89–96% of games (old bots: 0–15%, and those were only Scoutships).
2. **Early openings vary only for anchor112, and much less than the full-game mix suggests.**
   - **anchor112, deterministic:** almost always opens Organization (94/96), then splits into two main lines:
     - Strategy → Defenders within the first 8 turns ("defender turtle", 29/96)
     - Forestry/Riding with Warriors (46/96)
     - It shows 9 opening types and 12 tech/unit signatures, against 2–4 for every old bot. Its first non-Warrior unit comes at median turn 8.
   - **lr3e5u80, deterministic:** a Warrior rush much like the old bots for the first 8 turns. It researches almost nothing in that window (no tech in 78/96 games) and builds 98.5% Warriors. Its diversity is a **mid-game Catapult/Rider switch** (first non-Warrior unit at median turn 13; Catapult in 65/96 games).
   - **Stochastic (sampled) play** is far more varied for both: 24 signatures in 32 games for anchor112 and 16 for lr3e5u80, with archer and rider-rush openings.
3. **Neither snapshot is close to league96.**
   - Both beat the teacher: 32–0 and 31–1.
   - Against league96: **anchor112 2–30**, **lr3e5u80 11–21**.
   - anchor112's slower, tech-first opening loses the expansion race. It has 1.93 cities at turn 8, against 2.73 for league96. No anchor112 opening type beat league96 more than once.
4. **Deterministic play beats stochastic play for both snapshots** (19–13 and 21–11), the same as league96 (18–14). So keep evaluating these checkpoints in deterministic mode.

## Opening types (first 8 turns, per side-game)

How openings are labelled:
- **Named type** (archer, defender-turtle, siege, heavy, rider-rush): the first non-Warrior unit recruited in turns 0–7 sets the type.
- **"warriors + X tech":** no non-Warrior unit, but the bot researched a unit-unlocking tech X.
- **warrior-rush:** 3 or more early Warriors and no unlock tech.
- **economy-first:** everything else.
- **Signature:** the first two techs plus the early non-Warrior units in order.

Seed sets differ between bots: the old-bot traces are the 300000–301200 qualification seeds, and the new ones are 303000+. The league96 row "@303020" is league96 playing on the new seeds as the opponent of the new bots.

| bot (mode) | sides | opening types (count) | distinct types | distinct signatures | first tech (top) | early non-Warrior share | first non-Warrior unit (median turn) |
|---|---|---|---|---|---|---|---|
| **anchor112 det** | 96 | Riding-tech/warriors 34, defender-turtle 29, warrior-rush 12, Math-tech 7, Strategy-tech 5, siege 4, rider-rush 3, archer 1, heavy 1 | **9** | **12** | Organization 94 | 18% | Defender 37, Catapult 31, Rider 11, Swordsman 5 (t8) |
| **anchor112 stoch** | 32 | Riding-tech 9, archer 8, rider-rush 7, defender-turtle 4, warrior-rush 2, economy 1, Strategy-tech 1 | **7** | **24** | Organization 17, Riding 7, Forestry 4 | 29% | Rider 11, Archer 8, Defender 7, Catapult 4 (t6) |
| **lr3e5u80 det** | 96 | warrior-rush 86, economy-first 5, rider-rush 2, Math-tech 2, archer 1 | 5 | 8 | none in 8 turns 78, Forestry 16 | 1.5% | Catapult 65, Rider 19, Archer 4 (t13) |
| **lr3e5u80 stoch** | 32 | warrior-rush 21, archer 6, Riding-tech 3, Math-tech 1, rider-rush 1 | 5 | 16 | Forestry 11, Archery 6, Organization 5 | 9% | Rider 13, Archer 9, Catapult 6 (t11) |
| league96 det | 128 | warrior-rush 125, Math-tech 3 | 2 | 4 | Forestry 75, none 48 | 0% | Scoutship 18, Rammership 1 (t20); 109/128 never |
| league96 det @303020 | 64 | warrior-rush 63, Math-tech 1 | 2 | 4 | none 33, Forestry 28 | 0% | Scoutship 9 (t23); 54/64 never |
| league96 stoch | 32 | warrior-rush 31, Math-tech 1 | 2 | 6 | Forestry 21 | 0% | Scoutship 7, Rammership 2, Archer 1 (t22) |
| low176 det | 128 | warrior-rush 127, economy 1 | 2 | 4 | none 77, Forestry 46 | 0% | Scoutship 18 (t22); 110/128 never |
| low176 stoch | 32 | warrior-rush 32 | 1 | 5 | Forestry 21 | 0% | Scoutship 8, Rammership 2 (t20) |
| native96 det | 64 | warrior-rush 63, Riding-tech 1 | 2 | 3 | Organization 41 | 0% | Scoutship 1 (t32); 63/64 never |
| teacher (balanced) | 128 | Riding-tech 58, warrior-rush 35, economy 35 | 3 | 3 | Organization 122 | 0% | never |

Early economy (turns 0–7, per game):

| bot | Warrior/other recruits | research actions | captures | cities at turn 8 |
|---|---|---|---|---|
| anchor112 det | 4.1 | 3.0 | 1.7 | 1.93 |
| lr3e5u80 det | 5.6 | 0.2 | 2.2 | 2.43 |
| league96 det @303020 | 7.0 | 0.5 | 2.7 | 2.73 |

anchor112 spends early stars on Organization and Strategy and recruits fewer units, so it captures fewer villages. That is the visible cause of the expansion deficit against league96.

## Full-game unit share, unit availability vs use, techs

| bot (mode) | Warrior | Defender | Catapult | Swordsman | Rider | Archer | other | unit types/game | techs at end |
|---|---|---|---|---|---|---|---|---|---|
| anchor112 det | 24.1% | 28.8% | 20.4% | 18.4% | 4.5% | 1.3% | 2.4% | 2.83 | 13.4 |
| anchor112 stoch | 19.8% | 19.2% | 14.1% | 26.5% | 7.1% | 9.8% | 3.6% | 4.41 | 17.4 |
| lr3e5u80 det | 43.1% | 0.5% | 32.3% | 1.0% | 18.7% | 3.6% | 0.9% | 2.69 | 8.9 |
| lr3e5u80 stoch | 28.3% | 2.9% | 24.6% | 0.1% | 26.5% | 14.6% | 3.0% | 3.62 | 10.9 |
| league96 det | 99.0% | 0 | 0 | 0 | 0 | 0 | 1.0% (Scoutship) | 1.15 | 13.4 |
| low176 det | 99.3% | 0 | 0 | 0 | 0 | 0 | 0.7% (Scoutship) | 1.14 | 13.1 |
| teacher | 100% | 0 | 0 | 0 | 0 | 0 | 0 | 1.00 | 9.6 |

Each cell below is the number of games where the unit was used, over the number of games where it was summonable at some point:

| bot | Catapult | Archer | Rider | Defender | Swordsman | Knight |
|---|---|---|---|---|---|---|
| anchor112 det | 44/61 | 13/71 | 15/94 | 48/56 | 35/36 | 5/43 |
| lr3e5u80 det | 74/75 | 14/23 | 42/65 | 11/49 | 7/7 | 1/15 |
| league96 det | 0/120 | 0/110 | 0/103 | 0/44 | 0/16 | 0/7 |
| low176 det | 0/115 | 0/110 | 0/107 | 0/45 | 0/4 | 0/3 |

Tech paths:
- **anchor112** reaches Riding, Organization, Climbing and Mining in about 100% of games, Strategy in 62% and Chivalry in 49%. It builds Swordsmen whenever Smithery is reached (35/36).
- **lr3e5u80** stays narrow, with 8.9 techs at the end: Forestry, then Mathematics (79%) for Catapults and Riding (77%). It rarely gets Climbing (56%), Archery (28%) or Mining (23%).
- **The old bots** research broadly (about 13 techs) but never convert techs into units.

## Results

Wins and losses are from the challenger's point of view. CI is the paired-map bootstrap 95% interval of the win fraction.

| job | anchor112 | lr3e5u80 | league96 (earlier qualification, other seeds) |
|---|---|---|---|
| vs teacher (deterministic) | **32–0** (30.6 turns avg) | **31–1** (22.8 turns) | 32–0, 32–0 (19.8 turns) |
| vs league96 (both deterministic) | **2–30**, CI [0.00, 0.16] | **11–21**, CI [0.22, 0.44] | — |
| stochastic (challenger) vs deterministic | 13–19, CI [0.22, 0.59] | 11–21, CI [0.22, 0.50] | 14–18 |

None of the 192 games was truncated or cut off.

vs league96, win counts by anchor112 opening:
- Riding-tech/warriors 1/12
- defender-turtle 0/10
- Math-tech 1/3
- siege 0/3
- rider-rush 0/2
- archer 0/1
- heavy 0/1

lr3e5u80's wins against league96 all came from warrior-rush openings (10/29) plus a single archer opening (1/1).

Against the teacher, anchor112 games last longer (30.6 turns vs about 20 for league96). Its defensive, tech-heavy play converts slowly, which is consistent with the rising truncation rate in training.

## Training-log trends (`<run>/training.json`, 16-iteration windows)

Columns:
- **trunc** = truncations / (truncations + native completions)
- **L/C** = learner losses per completion
- **anchor KL** = KL to the anchor repertoire (anchor run only)
- **Unit share** = learner summons against all opponents in that window

diverse-h100-anchor-v1 (125 iterations logged):

| iterations | entropy | trunc | L/C | anchor KL | Warrior | Swordsman | Defender | Rider | Catapult | Archer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1–16 | 1.99 | 0.0% | 0.60 | 0.071 | 29% | 19% | 16% | 16% | 9% | 8% |
| 33–48 | 2.72 | 1.7% | 0.53 | 0.135 | 27% | 31% | 11% | 13% | 6% | 5% |
| 65–80 | 2.69 | 8.0% | 0.52 | 0.158 | 22% | 28% | 8% | 16% | 8% | 10% |
| 97–112 | 2.64 | 8.0% | 0.50 | 0.171 | 22% | 28% | 14% | 11% | 9% | 8% |
| 113–125 | 2.57 | 9.2% | 0.58 | 0.166 | 24% | 29% | 18% | 8% | 8% | 6% |

diverse-h100-lr3e5-v1 (91 iterations logged):

| iterations | entropy | trunc | L/C | Warrior | Rider | Archer | Catapult | Swordsman | Defender |
|---|---|---|---|---|---|---|---|---|---|
| 1–16 | 2.03 | 0.0% | 0.46 | 26% | 17% | 25% | 11% | 7% | 12% |
| 33–48 | 2.72 | 3.5% | 0.57 | 24% | 22% | 19% | 4% | 15% | 9% |
| 65–80 | 2.67 | 2.6% | 0.45 | 26% | 29% | 23% | 8% | 0% | 2% |
| 81–91 | 2.65 | 7.9% | 0.44 | 32% | 31% | 10% | 12% | 3% | 3% |

What the logs show:
- **Entropy** jumped from about 2.0 to about 2.7 within the first 32 updates in both runs and has stayed there, falling slightly. The rise came from the first PPO updates, not from late drift.
- **Truncations** climbed to 8–9% in both runs. In the no-anchor run the rise is recent (iterations 81–91). Evaluation games are never truncated, so this reflects slower, longer training games, not stalls.
- **anchor KL** keeps rising (0.07 → 0.17). The anchor slows drift but does not hold the policy near the repertoire.
- **The two runs diverged in unit mix.**
  - The anchor run converged on Swordsman/Defender, with Riders falling from 16% to 8%.
  - The no-anchor run converged on Rider/Catapult, with Defenders and Swordsmen collapsing (12% → 3%, 7% → 3%) and Warriors creeping back up (26% → 32%).
  - The deterministic evaluation mixes match these trends.

## Starting-point baseline (diverse-bc-dagger-v1): not run

The planned baseline was the bootstrap `diverse-bc-dagger-v1` (sha 14d501fe6fb4) against league96 on the same maps (303020–303035), to show whether PPO is adding strength. It never produced games.
- The plan `runs/structured-ppo-v3/diverse-strategy-check-bootstrap-v1-plan.json` passed `validate_plan`: development plans do not require a learned-only protocol, and the bootstrap metadata has its sha256 and `no_resign`.
- It was dispatched at 23:43 PDT. At 23:45 it was stopped along with every other campaign app, because the user asked for the RL campaign to be stopped. The ledger records it as `stopped_user_request`, stopped with 0 tasks, after 72 s of app time. No games completed, and it was not relaunched.

The only strength context so far comes from the first hourly GPU-arena ladder (maps 302000–302031, 64 games each, reported by the campaign session): anchored update 144 went 11–53 and lr 3e-5 update 112 went 19–45 against league96. Both point the same way as this check (anchor112 2–30, lr3e5u80 11–21). Whether that is better or worse than the bootstrap remains unmeasured.

## Caveats

- The old-bot rows come from different seeds than the new rows. The direct, same-map comparison is the "league96 det @303020" row.
- Each job has 16 maps; treat the win rates as screens, not selection evidence. These games are development seeds only; no final maps (400000+) were used.
- Opening labels are rule-based on an 8-turn window. A bot that switches units at turn 9–13 (lr3e5u80) counts as warrior-rush even though its mid-game army is mixed.

## Provenance and cost

Ledger: `runs/structured-ppo-v3/budget.json`; all apps were confirmed `stopped; tasks=0` and terminally accounted.

- **`diverse-strategy-check-v1`** (both checkpoints in one app): preempted after about 6 minutes. Modal re-ran the input, which hit `FileExistsError` on the existing output folder. No job completed. Ledger status `failed_preempted_FileExistsError_no_complete_jobs`; partial logs remain under `/retained/qualifications/diverse-strategy-check-v1/`.
- **Rerun, split per checkpoint on the same seeds:** `diverse-strategy-check-anchor-v2` (1183 s) and `diverse-strategy-check-lr3e5-v2` (815 s). Plans, results and terminal-app receipts are in `runs/structured-ppo-v3/`; traces are in `/retained/qualifications/diverse-strategy-check-{anchor,lr3e5}-v2/`.
- **Trace analysis:** `early-game-old-bots-v1` (176 s) and `early-game-diverse-v1` (164 s), both CPU at $0.07/h. Per-game records are in `runs/structured-ppo-v3/early-game-*-results.json`.
- **Estimated compute:** about $0.29 in total. Conservative ledger bound with the $1-per-app margins: $5.31.
- **New, uncommitted files:**
  - `scripts/analyze_early_game.py`: per-game early record, `classify`, `signature`, `summarize`
  - `experiments/modal_early_game_analysis.py`: Modal wrapper that returns composition and early records
