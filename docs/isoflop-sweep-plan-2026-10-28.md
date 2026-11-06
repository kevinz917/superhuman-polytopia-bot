# Iso-FLOP sweep for the 1v1 net (plan, 2026-10-28, ~$1k on Daytona)

The question: for a fixed training budget, what net size gives the most Elo, and how does that size grow with the
budget? Every big run so far uses 4x128 (4 residual blocks, 128 channels) without having tested alternatives. This
plan follows Chinchilla's approach 2 (IsoFLOP profiles), as in Eric Gu's Melee post, scaled down to a ~$1k budget on
Daytona spot. Nothing here has been launched or built.

## Compute accounting

`rl/network.py` is a residual CNN. Almost all FLOPs are the 3x3 convolutions, applied to all 196 tiles of a 14x14
board. Conv parameters N = blocks x 2 x 9 x width². The stem, heads and the parameter embedding are small or are
lookups and are left out, as Chinchilla leaves out embeddings.

One forward pass ≈ 2 x 196 x N FLOPs. One training sample costs about 8 passes:

| Pass | Count |
|---|---|
| Actor policy forward | 1 |
| Learner critic value pass (`learner_values`) | 1 |
| Policy forward + backward, 1 epoch | 3 |
| Omniscient critic forward + backward (same size as the policy) | 3 |

So **C ≈ 3,136 x N x samples**, this setup's version of Chinchilla's 6·N·D. hero-1v1 through cont1h (4x128,
6.95 B samples) works out to ~2.6e19 FLOPs. Below, budgets are in **B128**: the compute of 1 B samples on 4x128
(3.7e18 FLOPs).

## Arms

| Net | Conv params | vs 4x128 | Samples at C1 / C2 / C3 | Samples per size (trunk + forks) |
|---|---|---|---|---|
| **4x128** (today) | 1.18 M | 1x | 0.25 / 0.75 / 2.25 B | 2.5 B |
| 6x160 | 2.76 M | 2.34x | 0.11 / 0.32 / 0.96 B | 1.07 B |
| 8x192 | 5.31 M | 4.5x | 0.056 / 0.17 / 0.50 B | 0.56 B |

Budgets: **C1 = 0.25, C2 = 0.75, C3 = 2.25 B128**, spaced 3x apart. C3 is ~3x below the hero budget, so the fit
extrapolates by about one more step.

Smaller sizes are cut to fit the budget. 4x96 would need 4.4 B samples, and on Daytona's ~125k/s learner cap that
is ~10 h (~$300) on its own. The main question is whether the next hero run should be bigger than 4x128, and we
already know 4x128 beats 2x64 at large budgets. The cost: if 4x128 comes out best at a budget, we can't tell whether
something smaller would have been better still.

All arms use the hero-1v1 recipe (omniscient critic the same size as the policy, γ .995 / λ .99, turn clock, entropy
.01, LR 3e-4 → 1e-4 over the last 25%), except for **batch 65,536 / minibatch 4,096**. With hero's batch of 262,144,
the 8x192 C1 arm would get only ~210 updates. 65k gives it ~850, and the Oct 23 pilot measured 65k at parity per
sample. Each size starts from its own random init (`random_init_fn --blocks --width`).

### One trunk per size, with cooldown forks

The LR schedule is already warmup-stable-decay shaped: constant 3e-4, then linear decay over the last 25%. Run one
trunk per size at constant LR to 0.75·C3. At 0.75·C_k, fork a cooldown run that decays the LR over 0.25·C_k. Every
budget then gets exactly the schedule a separate run would have had, since the first 75% of a separate run *is* the
trunk. Compute per size is 2.5 B128 instead of 3.25 B128 for separate runs. The C1 and C2 forks are small: they can
run on a 2-GPU learner alongside the trunk, or one after another on the same hosts.

**Seed replicate:** run all three sizes again at C1 on a second seed (0.75 B128). That measures seed-to-seed Elo
noise. If the noise is about as large as the gaps between sizes, the C1 curve can't be read.

## Code (branch `claude/isoflop-sweep`, off `claude/daytona-hero`)

1. **Update budget.** `LearnerConfig.training_updates` stops a run after that many updates and drives the LR decay
   and the entropy-target schedule by updates instead of wall clock (`training_progress`). `training_seconds` stays
   as the backstop. A fork that resumes at 0.75 x U with `training_updates = U` follows exactly the schedule of a
   standalone run of U updates.
2. **Fork points.** `snapshot_at` lists extra updates that save a full snapshot (weights, Adam state, critic).
   `distributed.fork_run(trunk, fork, update)` builds the fork's folder from that snapshot, plus the trunk's earlier
   snapshots (the self-play pool) and its log up to that update. A Learner started there resumes through the normal
   `_resume` path, which restores Adam state, critic, pool and update count.
3. **Snapshot cadence fix (affects every resumed run).** Snapshots used to be written only on updates that were
   multiples of both `checkpoint_every` and `snapshot_every`. A resume rebuilds the pool from snapshot files, so with
   Daytona's `checkpoint_every=20` and `snapshot_every=250`, every other pool entry (the multiples of 250 that are
   not multiples of 500) was missing after a resume. That includes the live hero-1v1 continuation. Snapshots now keep
   their own cadence.
4. **FLOP logging.** Each `training.jsonl` row carries cumulative `flops` = batch x `sample_flops` (8 passes over
   the trunk's conv weights, stem included).
5. **Harness.** `experiments/daytona/sweep.py` defines the 12 arms (3 trunks, 6 forks, 3 seed replicates) as
   `hero.PLANS` entries, so `hero.py launch/keep/keep-remote/status/stop/down` run them. Sweep-only commands:
   `plan`, `stage-anchors` (copies hero-1v1 snapshots from Modal), `init` (random parents),
   `launch` (builds a fork once its trunk has passed the fork point), `matches` (neighbours, replicates, anchor chain)
   and `fit` (Bradley-Terry Elo, N_opt per budget, exponent b). The hub-side helpers run from a scratch copy of the
   code, so the live hero keeper's `/workspace` stays untouched.
6. **Tests.** `tests/test_sweep.py`:
   - a 16-update trunk stops at its budget, its LR follows the update schedule, and its FLOPs add up;
   - a fork at update 9 resumes with bitwise-identical weights, Adam state and critic, and the same pool;
   - the fork then runs the schedule of a 12-update run;
   - every arm config is valid, each fork point lies inside its trunk's constant-LR phase, and FLOPs match within a
     budget.

## Phase 0 (~$40)

- Modal or CPU-sandbox tests for the changes above (distributed, rust engine, resume).
- Learner bench for each size on 4x PRO 6000 and 2x PRO 6000, plus the actor rate for each size. For 4x128 the
  learner caps at ~125k/s and an actor at ~29.5k/s. Bigger nets should become GPU-bound: by FLOPs, roughly ~55k/s
  for 6x160 and ~28k/s for 8x192, so 2 actors and 1 actor. These are projections to be checked.
- A 15-minute smoke of 8x192 at LR 3e-4. Gate: KL and entropy look like 4x128's. If it is unstable, lower the LR
  for *all* sizes rather than tuning each one, and record that as a confound.

## Evaluation (~$20)

Evals run on 4-vCPU CPU sandboxes, as for the hero runs. One round of 320 games costs ~$0.50.

- **Anchors:** c3, the champion, bn4 and the hero-1v1-cont1h final. Add hero-1v1 snapshots near u1700 and u5700 if
  the `hero-runs` volume has them. Low-budget arms will be weaker than the champion (hero-1v1 was 0% against it at
  ~0.45 B samples), so they need anchors near their own strength.
- **Games:** each of the 9 final arms plus the 3 replicates plays 256 seat-paired games on the `proposed-v1` mix
  against its two nearest anchors. Within each budget, neighbouring sizes play 256 games against each other. That
  totals ~7.7k games.
- **One Bradley-Terry Elo fit** over all games, with map-bootstrap CIs.
- **Search latency:** each size's browser WASM time per turn for d=1 search. The forward pass grows with N, so
  6x160 costs ~2.3x 4x128 per search node.

## Fit and decision

1. For each budget, fit Elo against log N with a quadratic and take its argmax N_opt(C). With 3 sizes, the quadratic
   fits exactly, so CIs come from the game bootstrap alone.
2. Fit log N_opt = a + b·log C over the three budgets. Chinchilla's b is 0.5 and Gu's Melee fit is 0.58. With three
   points, b will have a wide CI. Report it as such.
3. Decide from where the optimum falls:
   - **If 6x160 or 8x192 wins at C3 and the trend rises with budget,** the next hero 1v1 run uses the bigger size,
     provided search latency in the browser stays acceptable.
   - **If 4x128 wins at C3,** keep it and spend the budget on samples.
   - **If 8x192 wins at C3,** the curve is not bracketed on the high side. The next step is one 12x256 arm at C3,
     ~$185.

## Cost (projected, Daytona spot)

Rates come from `experiments/daytona/cost_matrix.py` option B (4x PRO 6000 learner + PRO 6000 actors): ~$74 per
B samples for 4x128 at ~112k/s, ~$30/h, with 10% of the clock assumed lost to spot reclaims. Bigger nets are
assumed to cost about the same per FLOP: the learner is just as busy and they need fewer actors.

| Item | Compute | Expected |
|---|---|---|
| 4x128 trunk + forks (2.5 B samples, ~6 h) | 2.5 B128 | $185 |
| 6x160 trunk + forks (1.07 B, ~6 h projected) | 2.5 B128 | $185 |
| 8x192 trunk + forks (0.56 B, ~6 h projected) | 2.5 B128 | $185 |
| C1 seed replicates (3 arms) | 0.75 B128 | $55 |
| Phase 0 (benches, smokes, tests) | | $40 |
| Evaluation (~7.7k games on CPU sandboxes) | | $20 |
| Hub + margin (spot reclaims, bad hosts, reruns; ~20%) | | $135 |
| **Total** | | **~$805** (cap $1,000) |

**Wall time:** ~6 h if all three arms run at once. That needs 3x (4x PRO 6000) learners plus ~8 actor hosts on spot
at the same time, which hasn't been tried. If capacity isn't there, run the arms one after another (~18 h).

**Options within $1k:**
- **Add 4x96 at C1 and C2 only** (+~$100; 1.44 B samples, trunk to 0.56 B). This brackets the small side at the low
  budgets, where a smaller net is most likely to win. Total ~$905.
- **Reuse hero-1v1 as the 4x128 trunk** (−~$185, batch 262,144 for every arm). All three fork points
  (0.19 / 0.56 / 1.69 B) fall inside hero's constant-LR phase. The cost is that the 8x192 C1 arm gets ~210 updates.
  This needs snapshots that include the optimizer state, which hasn't been checked.
