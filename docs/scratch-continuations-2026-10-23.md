# From-scratch lineage: continuations and the current champion (2026-10-22/23)

The from-scratch policy (`scratch-overnight-1`, 7 h of pure self-play from random weights;
docs/scratch-check-2026-10-22.md) was continued twice on main's faster training stack (Rust engine 10x, actor
levers 1-5: 1×H100 + 8 actor containers × 32 processes). Each continuation resumes the parent's final weights and
its omniscient critic (`critic_path`), with the same recipe (proposed-v1 maps, epochs 1, LR 3e-4, entropy 0.01,
turn clock .99/.95, cap score 1.0, 10% own-snapshot pool). Compiled from the runs' session notes; the raw reports
live in the side-session notes folder of the session that ran them.

**Champion: `scratch-cont-2` final, +442 ± 14 Elo (c3 = 0)**, archived at
`/retained/checkpoints/distributed-final-v1/scratch-cont-2-final/` (with `final-critic.npz` and training logs).

## Lineage

| checkpoint | trained | Elo (c3 = 0, mixed maps) | archived |
|---|---|---|---|
| c3 u4854 | inherited lineage | 0 | `distributed-final-v1/c3-maps-ctl-update-4854` |
| OMNI c3 (c5-omni-s1-final) | c3 + 6 h omni critic | +231 | `distributed-final-v1/c5-omni-s1-final` |
| Scratch 7h (scratch-overnight-1) | 7 h from random weights | +295 | `distributed-final-v1/scratch-7h-final` |
| Scratch-cont 6h (scratch-cont-1) | + ~6 h | +383 ± 13 | `distributed-final-v1/scratch-cont-1-final` |
| **scratch-cont-2** | + 7 h | **+442 ± 14** | `distributed-final-v1/scratch-cont-2-final` |

Elo is a Bradley-Terry fit over all mixed-map games of the scratch-cont-2 evaluation (reference round robin of
6 × 200 games plus the hourly and final matches).

## scratch-cont-1 (Oct 22 12:27-~18:30 PT)

Parent `scratch-overnight-1/final.npz` + its critic, seed 504, launched from main 0b6e552, ~117k samples/s steady
(the learner's ceiling) at ~$22/h billed (~$0.052 per M samples). Extended once by relaunching the same run id
(the learner resumes `latest.npz` with Adam state, log and pool). Ladder at update 4500: 64% vs Scratch 7h, 69% vs
OMNI c3, 80% vs c3. Its final 256-game seat-paired match (`scratch-cont-final2`, 68% vs Scratch 7h) was played on
classic maps only; a mixed-map rerun is below.

## scratch-cont-2 (Oct 22 23:28 - Oct 23 06:40 PT)

Parent `scratch-cont-1/final.npz` + critic, seed 505, 25,200 s, training app `ap-2pDHF9fitS6azUsO345Grb`. Spend
~$185 of a $200 cap (training ~$162, evaluation ~$17-21). Every match below is mixed maps (proposed-v1) and
seat-paired (each map once per seat).

Final tournament (512 games per opponent, seeds 392000+):

| opponent | win | map pairs 2-0 / split / 0-2 | land | water | as Blue | as Red |
|---|---|---|---|---|---|---|
| Scratch-cont 6h | **58%** | 54 / 188 / 14 | 60% | 52% | 68% | 48% |
| Scratch 7h | 69% | 106 / 143 / 7 | 71% | 66% | 79% | 60% |
| OMNI c3 | 76% | 139 / 112 / 5 | 74% | 83% | 82% | 71% |
| c3 | 92% | 218 / 36 / 2 | 92% | 93% | 94% | 91% |

Hourly checkpoints (200 games per opponent; Elo from the same fit):

| checkpoint | ≈ hour | Elo | vs Scratch-cont 6h |
|---|---|---|---|
| update-1750 | 1 | +416 ± 18 | 52% |
| update-3700 | 2 | +413 ± 18 | 52% |
| update-6600 | 3.5 | +403 ± 18 | 52% |
| update-8600 | 5.5 | +433 ± 13 | 56% (512 games: 56%) |
| update-9700 | 6 | +433 ± 18 | 53% |
| update-11450 | 7 | +436 ± 18 | 60% |
| final | 7 | +442 ± 14 | 58% (512 games) |

The curve is a step, not a slope: ~+30 Elo in the first hour, flat for 2.5 h, another ~+25 around hour 5, little
after. Another long run of the identical recipe is expected to add little.

## Seats and maps

Many proposed-v1 boards are lopsided. In a 128-game seat-paired match of Scratch-cont 6h vs Scratch 7h
(seeds 380000+, ~$0.25), 39 of 64 maps went 1-1 (26 to whichever bot was Blue, 13 to Red); of the 25 maps the bots
decided, Scratch-cont won 21. Overall 63% (water 67%, land 61%). Seat-paired evaluation already cancels the seat
effect in the win rate; symmetric maps would only make each game more informative.

Example: waterworld seed 321211 has 6 of 9 villages near Blue. Scratch-cont lost it as Red (stuck at 4 cities,
banked 17-30 stars for 12 turns with no research, turtled on Defenders against scoutship bombardment) and won it as
Blue at turn 26.

## Studio replays

Recorded on Modal with `experiments/modal_scratch_replays.py` (Scratch 7h vs OMNI c3) and
`experiments/modal_scratch_cont_replays.py` (Scratch-cont 6h vs Scratch 7h), saved to the main checkout's
`artifacts/studio-d3142/replays` and served by the :3000 Studio after a restart.

## Incidents and lessons

- The scratch-cont-2 hourly ladder hung silently after round 2 (no checkpoint copy, no history write) and was
  replaced; the replacement was preempted and replayed by Modal, restarting at round 0. Watch ladder history
  freshness.
- The scratch-cont-1 ladder was also preempted and replayed (round labels restart at 0).
- The Mac slept 06:18-07:09 despite `caffeinate -i`; a local guard script then launched the final tournament late,
  and detached `modal run` launches die if the Mac sleeps during app creation. Schedule final tournaments on Modal
  (a detached function with `first_delay`), not from a local watcher.
- Modal bills this fleet at ~$21-22/h, well above the launcher's estimate ($13.82/h for 8×32); budget with the
  billed rate.
