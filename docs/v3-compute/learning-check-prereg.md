# Pre-registration: matched short learning check, V2-6x128 vs V3 (session E, 2026-11-07 PT)

Written before any learning-check run started and before any arena result between trained arms existed. The only
trained checkpoints at writing are the 900 s e2e speed runs; their arena is a smoke test of the evaluator, reported
separately and not used for the decision below.

## Arms (fresh init, one seed each)
Recipe `experiments/claude_v3/hero-1v1-classic.json`: hero-1v1 (batch 262,144, minibatch 16,384, 1 epoch, lr 3e-4 ->
1e-4 over the last 25%, clip .2, segment 32, turn clock .99/.95, cap score 1, 10% historical pool every 250 updates)
with maps `classic`. Seed 801. GPU-resident trainer, engine-cu (main's rules). 1xH100, 4 CPU, 32 GiB. Tokamax
attention + affine LayerNorm, BF16 trunk, HIGHEST precision, micro 1024.
- A: `transformer-v2` 6x128 (V2 with its separate 6x128 omniscient critic; the V2-with-kernel-fixes control)
- B: `transformer-v3-p2` 6x128 (shared encoder, 2x2 patch tokens, 1-block privileged critic)
- C: `transformer-v3-p2` 6x256, if its e2e speed run shows >= 2x A's samples/s (it fits the same dollars with more
  capacity)
Each trains for training_seconds = 3600 of wall time on the same container type, so the arms cost equal dollars.
Snapshots every 10 updates.

## Evaluation (GPU arena, experiments/claude_v3/gpu_arena.py)
Held-out classic seeds 300000-301023 (1024 maps, outside the training table), both seat orders, 2048 games, both
sides deterministic. Score = win 1, draw 1/2. A truncated game is decided by score at the cap; terminal-only results are
reported as well. 95% interval clustered by map (the mean of the two paired games per map).
1. **Equal dollars (primary):** B_final vs A_final; C_final vs A_final.
2. **Equal samples:** B at the snapshot whose update count is nearest at or above A's final update count, vs A_final.
3. Each final vs `random` (the uniform-legal anchor), for an absolute reference.

## Decision rule
- V3 arm **wins at equal dollars** if the lower 95% bound of its score vs A_final is > 0.5. It **loses** if the upper
  bound is < 0.5. Otherwise the result is **inconclusive**.
- Per-sample efficiency is **no worse** if the equal-samples score interval includes or exceeds 0.5, and **worse**
  if the upper bound is < 0.5.
- Strength per dollar is reported as (score at equal dollars, samples/s ratio, $/B). No 2-3x strength claim follows
  from one seed.

## Known limits (stated before results)
One training seed per arm. Early training (tens of millions of samples for A). Current-rules engine-cu with classic
maps; retail 2.18.0 parity is not verified. The arms reach the historical pool at different wall times, since it is
keyed to update counts, which is part of the recipe. Unverified by the other harness until Codex reviews.

## Amendment 1 (2026-11-07 ~03:58 PT, before any learning-check arena; learning arms still training)
The 1024-map deterministic smoke arena of the 900 s speed-run checkpoints timed out at 900 s. A 64-map diagnostic
(arena-diag) showed the cause: with both sides deterministic, 79 of 128 games were still unfinished after 2,000
ticks. Early argmax policies stall (round 30 and rising, few terminations). With T=1 sampling, all 128 games
terminated within 4,063 ticks in 58 s. **Change: both sides sample at T=1 (as the trainer's own games and pool
matches do).** The evaluator now also caps at 6,000 ticks; an unfinished game is decided by current score and is
counted in `unfinished`. Everything else is unchanged. The diagnostic's result (V3p2-15min beat V2-15min 128-0 at
T=1) comes from the speed-run checkpoints, not the learning-check arms, and is reported as a smoke result only.

## Amendment 2 (2026-11-07 ~04:20 PT, before any learning-check arena)
Add **arm D: the hero-recipe CNN** (`normalized-v2` 4x128, the hero1 architecture) with the same recipe, seed, maps,
container and 3600 s. Reason: BORROW.md §0 (session K) says every V3 strength-per-dollar claim must be measured at
equal dollars against the hero CNN on the GPU-resident trainer, not only against V2. D launches after A/B/C, because
the 2-GPU allocation keeps it from running alongside them; its environment is identical. Added matches: D_final vs
A_final, B_final vs D_final (equal dollars), each with the same rule (lower bound > 0.5 = win). The arena runs A/B/C
matches first; D's matches run when D finishes. Every arm keeps `no_peace` off, as A/B/C started (default; Kevin's
28cd1b5 pass-through is now ported at 4902b8d, but not enabled for this study). The capped-game share is reported per
match.
