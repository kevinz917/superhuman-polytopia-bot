# Search on the scratch champion, and a fog-of-war value head for it (2026-10-23)

**Result: the champion (scratch-cont-2 final) + leak-free turn search beats the bot the site serves today
(OMNI c3 + search) 81% of games, about +260 Elo.** Search adds about +50 Elo to the champion itself. A fog-of-war
value head trained for the champion predicts outcomes slightly better than c3's head but makes no measurable
difference to search, so either head can be served.

Champion: `/retained/checkpoints/distributed-final-v1/scratch-cont-2-final/final.npz` (docs/scratch-continuations-2026-10-23.md).
Ledger `runs/scratch-search` ($45 cap, ~$43 committed at 1.33× the nominal rates).

## Why a value head was needed

Turn search scores each candidate turn with a value head that sees only the searching player's view. The scratch
lineage trained with a separate omniscient critic, which cannot be used at play time, and its own `value` head was
never trained (flat at about −0.05: AUC 0.72 on its own games, sign accuracy 37%). Search therefore needs either a
borrowed head (c3's, as the earlier search work did) or a head trained for this policy.

## Fog-of-war head (`scratch-fog-head-1`)

Resume the champion with `omniscient_critic` false and `critic_warmup` 10,000,000 (past the run's end): every gradient
except `critic_hidden` and `value` is zeroed, so the policy stays bitwise identical while the head regresses the
self-play returns (turn clock .99/.95, cap score 1.0) of the champion playing itself. 2,700 s on 1×H100 + 4 actor
containers (64-73k samples/s, ~620 updates), approx KL to the champion 1.6e-7 throughout (numerical noise).
Archived at `/retained/checkpoints/distributed-final-v1/scratch-cont-2-foghead/` (champion policy + trained head).

Check (`experiments/modal_distributed_rl.py::head_check`): parameters that differ from the champion are exactly
`critic_hidden.{w,b}` and `value.{w,b}`. Value-head quality on 24 held-out sampled self-play games of the champion
(proposed-v1 seeds 397000+, 45,615 positions, all games decided), AUC of the acting seat's value for wins vs losses:

| value head | all | turns 0-9 | turns 10-19 | turns 20+ | mean in wins / losses |
|---|---|---|---|---|---|
| **new fog head** | **0.975** | **0.754** | **0.917** | **0.996** | +0.70 / −0.52 |
| c3 u4854 | 0.963 | 0.747 | 0.896 | 0.992 | +0.35 / −0.32 |
| OMNI c3 (dry run, same games) | 0.937 | 0.710 | 0.872 | 0.990 | +0.39 / −0.17 |
| champion's untrained head | 0.719 | 0.519 | 0.569 | 0.781 | −0.05 / −0.07 |

## Search tournament

`experiments/modal_test_time_search.py::sweep`, tags `scratch-search-{a,b,c}`, results under
`/results/test-time-search/<tag>/`. Search = the served configuration (`studio.service.SEARCH`: 16 lanes, depth 1,
rollout temperature 1, 64 opponent replies, determinized), one game per core. Every arm: the same 100 proposed-v1
maps (seeds 396000-396099), each from both seats. Raw opponents play greedily (deterministic). Arm 3 uses the new
`search_arena` option `opponent_search` (the opponent searches with the same settings and its own head).

| arm | matchup | W-L-D | win [95% CI] | Elo | land / water |
|---|---|---|---|---|---|
| 1 | champion + search (c3 head) vs champion raw | 113-86-1 | 56.5% [51, 62] | **+47** [+10, +85] | 58% / 54% |
| 2 | champion + search (own fog head) vs champion raw | 114-84-2 | 57.0% [52, 62] | **+53** [+17, +89] | 60% / 51% |
| 3 | champion + search (c3 head) vs **OMNI c3 + search** | 162-35-3 | **81.0%** [76, 86] | **+260** [+210, +319] | 83% / 77% |
| 4 | champion + search (c3 head) vs raw c3 | 182-14-4 | 91.0% [87, 95] | +424 [+350, +521] | 92% / 89% |

- **Own head vs c3 head (arms 2 vs 1, paired by map and seat):** both win 93, own head only 21, c3 head only 20,
  both lose 66; win-rate difference +0.5 points [−6, +7]. The better-calibrated head does not change search's
  choices enough to matter. Searched turns took 10.9-11.4 s on one core; search overrode the greedy turn ~79% of
  the time in both arms.
- **Search gain on this policy is ~+50 Elo**, below the +80-100 it gave OMNI c3 (docs/search-breadth-2026-10-21.md).
  Plausible reading: the stronger the raw policy, the less a one-turn lookahead finds that the policy did not
  already play.
- **Arm 4 is at the ceiling.** The raw champion already scores 92% vs c3 (512-game final tournament), so 91% with
  search says nothing about search; it is only the Elo anchor.
- **Arm 3 is the shipping comparison**: both sides use the serving configuration, and the champion wins on land
  (83%) and water (77%) maps.

## Cost (nominal, Modal billed ≈ 1.33×)

Head training ~$7.9, head checks ~$0.6, arms 1 + 4 $7.32 (one chunk preempted and replayed), arm 2 $4.13,
arm 3 $12.03 (search on both sides costs ~2.7× a one-sided arm): ~$32 nominal, ~$43 committed in the ledger.

## Code

- `src/polytopia_bot/rl/search_arena.py`: `cfg['opponent_search']` (TurnSearcher settings) and
  `cfg['opponent_value']` make the opponent search too; default unchanged (raw greedy opponent).
- `experiments/modal_distributed_rl.py::head_check`: parameter diff + value-head AUC on fresh self-play games.
