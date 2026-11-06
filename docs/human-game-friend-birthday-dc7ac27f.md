# A friend's birthday game: a friend vs the scratch champion (dc7ac27f)

On a friend's birthday, they played the public site
(https://polytopia-bot-kappa.vercel.app) against the bot and were not able to beat it.

- **Game:** `dc7ac27f319f44e0859410be61752c5c`, map seed 1810988674, 14×14 Bardur 1v1 (ruleset v2).
- **Seats:** Human = player 1 (moves first); bot = player 2.
- **Bot:** scratch champion `scratch-cont-2` + fog-of-war value head
  (checkpoint sha256 `90618f5a…`), leak-free turn search on every move (16 candidate turns × 64 replies).
- **Played:** Oct 23 2026, 23:37 PT → Oct 24 00:00 PT.
- **How it ended:** Modal preempted the play container at 00:00:34 PT mid-game (around turn 10–11), and the
  game was lost with it (games lived only in the container's temp folder). There is no final result. At the last
  snapshot (turn 9, 23:48 PT) the bot led on score, cities, population, territory and income.

All numbers below come from a read-only pull of the live container's session journal at 23:48 PT.

## Scoreboard (end of each turn, Human / bot)

| Turn | Score | Cities | Pop | Tiles | Income | Stars banked | Units |
|---|---|---|---|---|---|---|---|
| 0 | 570 / 530 | 1 / 1 | 2 / 1 | 9 / 9 | 4 / 2 | 1 / 1 | 1W / 2W |
| 1 | 595 / 600 | 1 / 1 | 2 / 2 | 9 / 9 | 4 / 4 | 3 / 1 | 2W / 2W |
| 2 | 640 / 660 | 1 / 1 | 3 / 3 | 9 / 9 | 4 / 4 | 5 / 1 | 3W / 3W |
| 3 | 665 / 690 | 1 / 1 | 3 / 4 | 9 / 9 | 4 / 4 | 9 / 3 | 3W / 3W |
| 4 | 1160 / 1325 | 1 / 2 | 5 / 7 | 9 / 18 | 5 / 8 | 7 / 5 | 3W 1A / 4W |
| 5 | 1185 / 1620 | 1 / 2 | 5 / 7 | 9 / 18 | 5 / 8 | 12 / 4 | 3W 1A / 4W 2A |
| 6 | 1235 / 2125 | 1 / 3 | 5 / 9 | 9 / 27 | 5 / 11 | 22 / 0 | 2W 1A / 4W 3A |
| 7 | 1675 / 2260 | 2 / 3 | 10 / 11 | 18 / 27 | 9 / 11 | 16 / 2 | 2W 2A / 4W 5A |
| 8 | 2330 / 2690 | 3 / 4 | 10 / 15 | 27 / 36 | 10 / 14 | 5 / 2 | 3W 2A 1C / 4W 7A |
| 9 (in progress) | 2465 / 2690 | 3 / 4 | 10 / 15 | 27 / 36 | 10 / 14 | 6 / 2 | 4W 2A 1C / 4W 7A |

W = Warrior, A = Archer, C = Catapult.

## Totals through turn 9

| | Human | Bot |
|---|---|---|
| Tech order | Forestry + Archery (t4), Mathematics (t8), Climbing (t9) | Forestry (t4), Archery (t5), Climbing (t6) |
| Units trained | 7 (warriors, 2 archers, 1 catapult) | 11 (4 warriors, 7 archers) |
| Forests cleared | 0 | 17 |
| Harvests | 4 | 12 |
| Lumber huts | 5 | 3 |
| Captures | 3 | 3 |
| Attacks / kills | 2 / 0 | 2 / 1 |
| Actions | 65 | 98 (all chosen by search) |

## How the game went

- **Turns 0–3: even.** Human banked stars (9 by turn 3); the bot spent as it earned.
- **Turn 4: the bot's big turn (19 actions).** Forestry, a second city, 8 forests cleared for +8 stars,
  2 harvests and a lumber hut; income doubled from 4 to 8.
- **Turn 6: the bot's peak lead (~890 points), 3 cities to 1.** It killed one of the human's warriors.
  Human held 22 unspent stars.
- **Turns 7–8: the human's comeback.** Two cities captured, three more lumber huts, Mathematics and a catapult;
  the gap closed to 225.
- **Turn 9 onward:** the bot still led when the server was preempted.

## Notes

- The bot played its usual archer-first, forest-clearing tempo opening. the human's lumber-hut economy was closing
  the gap, the same pattern as the Oct 21 human game against OMNI c3 (`docs/human-game-a8147e94-2026-10-21.md`).
- The full session journal was not saved before the preemption; this file is the only record of the game.

Happy birthday.
