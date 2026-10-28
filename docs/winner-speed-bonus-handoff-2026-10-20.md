# Handoff: winner speed bonus for distributed PPO (2026-10-20)

## Goal

Games run too long and the bot isn't aggressive enough. In c3 u4854 self-play, 20% of games reach the 100-turn
cap with no winner (ridge and highlands about a third; see [map-variants-2026-10-19.md](map-variants-2026-10-19.md)).
Strong-vs-strong ladders had about 25 of 64 games truncated
([distributed-rl-2026-10-19.md](distributed-rl-2026-10-19.md)). We want shorter, more decisive games
**without losing strength**.

## The change: pay the winner more for winning earlier

At a **native win**, the winner's terminal reward becomes

```
R_win  = 1 + b * (1 - T / cap)      T = round the game ended, cap = that game's turn cap
R_loss = -1                          (unchanged)
```

Truncation handling is unchanged (critic bootstrap, or `truncation_reward` if set). Start with `b = 0.3`.
At the current mean game length on classic (~43 rounds) a win pays about 1.17; a round-10 win pays 1.27.

### Why this form

- **Winner only, on purpose.** If both sides' rewards decay with time (a symmetric discount), the losing side
  learns to stall, because a later −1 counts for less. With the loss fixed at −1, the loser gains nothing
  from delay, so the pressure falls only on the side that is winning to finish.
- **Winning stays the top priority.** The slowest win (≥ 1) still beats any loss (−1), so the bot has no reason
  to gamble a win for speed. With b = 0.3 the difference between the fastest and slowest win is 0.3. That is
  under a sixth of the 2-point gap between winning and losing.
- **No dense shaping.** We are not rewarding kills, captures or attacks: those are easy to farm (trading units
  for kill points, capturing and losing the same city). The bonus only depends on how soon the game is won.
- **A cap penalty is not the answer.** Campaign 3 tried scoring the cap as a draw (`truncation_reward=0`). That
  arm trailed control (71% vs 85% of games decided) and was stopped.
- This is the useful part of Duper's `execute_unnecessary_cmd` term
  (`duper/rl_ray_deploy/duper/reward.py`): a bonus only the winner receives, for winning efficiently.

## Where to implement (current `src/polytopia_bot/rl/`)

1. **`LearnerConfig`** ([distributed.py](../src/polytopia_bot/rl/distributed.py) ~line 80): add
   `winner_speed_bonus: float = 0.` (0 = today's behaviour). Actors receive `asdict(config)` through the hello
   message, so `cfg.get('winner_speed_bonus', 0.)` is available on the actor side with no extra plumbing.

2. **Expose the round at game end.** `Tick` ([cpu_engine.py](../src/polytopia_bot/rl/cpu_engine.py) ~line 44)
   does not carry it today. In `one()`, `after.turn` is the round counter (the cap check in
   `gpu/core.py` is `turn > state.reference_max_turns`). Add two fields, e.g. `end_turn = after.turn` and
   `turn_cap = after.reference_max_turns`, return them from `one()`, and update `Tick(*out[:7])` /
   `out[7]` indexing. Read the cap from the state, **not a constant**: classic is 100, the 18×18 ruleset is 150.

3. **Apply the bonus at episode end** (actor loop, ~line 418, where `episode_outcome` is called). Only on
   `terminated` (native), never on `truncated`:
   ```python
   if terminated[i] and bonus:
       frac = 1. - min(end_turn[i], cap[i]) / cap[i]
       rewards[i] = np.where(rewards[i] > 0, rewards[i] + bonus * frac, rewards[i])
   ```
   Do this before `traj.finish`. Keep the `stats['vs']` W/L counting on `r > 0`/`r < 0` as is (it still works).
   Consider logging the mean `end_turn` of native games per map type next to the existing truncation counts.

4. **Things checked that need no change:**
   - The value head is a plain linear output ([network.py](../src/polytopia_bot/rl/network.py) line 66), so
     returns above 1 are representable. No tanh clamp to remove.
   - Actors use `no_resign=True` by default, so the learner cannot farm the bonus by forcing early resignations.
     If any opponent can resign, an early win by resignation is still a legitimate fast win.
   - `SeatTrajectories` computes GAE separately for each seat from that seat's own values. It does not negate
     one seat's value to get the other's, so non-zero-sum terminals are fine here.
   - Advantages are normalized per batch with a 0.05 std floor (line ~784). The bonus slightly increases raw
     advantage spread; there is no need to retune.

5. **Tests** (`tests/test_distributed_rl.py`, next to `test_turn_cap_as_fixed_terminal_reward`), **run on Modal,
   not locally**:
   - bonus 0 reproduces today's rewards bitwise;
   - native win at T=0 → 1 + b, at T=cap → 1, loser always −1;
   - truncation is unaffected, with and without `truncation_reward`;
   - the cap is read from the per-game state (pass a 150-cap case).

## Caveats

- **Values stop being win probabilities.** A critic value near 1.2 means "winning, and soon". Anything that reads
  the value as P(win), or assumes `V(opponent) = −V(me)`, is slightly off. The test-time search code (d=1
  determinized lookahead, on another branch) should be checked for that assumption before it is pointed at a
  bonus-trained checkpoint. Evaluation and Elo stay on plain win/loss, so they are unaffected.
- **Some stalls are a skill problem, not an incentive problem.** Ridge and highlands stall at mountain
  chokepoints. If truncations there don't move, it's a capability problem that reward won't fix.
- **Interaction with gamma.** The per-decision gamma of 0.995 already discounts later wins slightly, but by
  action count rather than rounds. The bonus is the rounds-based pressure. Leave gamma alone in this test so
  the effect is attributable.

## Experiment

Two arms from the same parent with the same league, map mix and budget; only `winner_speed_bonus` differs:

| Arm | `winner_speed_bonus` |
|---|---|
| A (control) | 0 |
| B | 0.3 |

If you can only afford one arm alongside the run already going, use that run as the control, provided it
uses the same parent and settings.

**Report** (deterministic, both seats, fresh development maps ≥ 300000, frozen engine):

1. Win rate of each arm vs the parent (e.g. c3 u4854). **B must be no worse than A**; if it drops, the bonus is
   rejected regardless of anything else.
2. Truncation rate, overall and per map type.
3. Mean round of native endings, per map type.
4. Aggression: attacks per turn, round of first attack, round of first city capture.
5. Head-to-head B vs A.

**Success** = equal-or-better strength, plus a clear drop in game length and truncations. If B is shorter but
weaker, try b = 0.15 once before dropping the idea.

**Watch during training:** the "vs frozen parent" record in `training.json` (`actors.vs`). It predicted every
collapse in the 18×18 runs within about 30 minutes.
