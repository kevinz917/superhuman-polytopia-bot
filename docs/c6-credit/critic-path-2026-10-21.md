# critic_path: continuing from the parent's critic instead of a fresh one (2026-10-21)

Follow-up to the [c6-credit readout](../c6-credit-readout-2026-10-21.md), finding 2: every one-hour continuation of
`c5-omni-s1-final`, including the unchanged-recipe control, lost to its parent, and the suspected cause was the
restart itself: the omniscient critic is re-initialised and warmed up for 60 updates, then the policy trains against
a critic that is still worse than the one it was trained with. This note adds the option that avoids that and
checks it with a short run. Ledger `runs/c6-credit/critic-path/budget.json` (cap $12, raised to $13, see Cost).

## Change

`LearnerConfig.critic_path` (`src/polytopia_bot/rl/distributed.py`): with `omniscient_critic`, the critic starts as
the checkpoint at that path (a parent's `final-critic.npz`, saved by every omni run and archive) instead of the
fresh template; `critic_warmup` can then be 0. `initial_critic()` holds the three cases (none / fresh / loaded) and
logs `critic_loaded`. Validation rejects `critic_path` without `omniscient_critic`. Unit test
`test_critic_path_loads_a_saved_omniscient_critic` saves a perturbed critic, loads it leaf for leaf through the
config, and checks the fresh and absent cases; `tests/test_distributed_rl.py` 13 passed on Modal
(app `ap-NLF675KjNGvRzZxfgRxdlR`). A critic saved for a different board size or width fails the shape check in
`load_checkpoint`, so grown-board continuations still need a critic grown the same way as the policy.

## Verification run `c6-critic-path`

Arm A's recipe exactly (parent `c5-omni-s1-final`, per-decision γ 0.995 / λ 0.99, lr 2e-4, entropy 0.005, 3 epochs,
batch 262,144, minibatch 16,384, segment 32, 5% vs the frozen parent, 15% vs own snapshots, 1×H100 + 12 × 16-vCPU
actors, TF32 convolutions + fp32 heads), with `critic_path` = the parent's `final-critic.npz`, `critic_warmup` 0,
seed 304, 1,500 s of learner clock. Launched 13:35 PDT (app `ap-87lLMLHainXhgAQN5NNsPN`); 127 updates, 33M samples,
24k samples/s; two actor containers were preempted at +4 min (connections 192 → 160) and replaced by +13 min
(192, 14 containers, no duplicate learner); clean stop at 14:00. `protocol.json` carries the `critic_path`.

**The loaded critic is a trained critic from update 1**: value std 0.21 and value loss 0.0044 on the first batch
(a fresh critic starts near 0.07 and needs ~60 updates to reach 0.25), 0.28–0.29 from update 50 on, exactly c5's
own 0.28–0.30. The policy trained from the first update (approx KL 0.006–0.009 per update, entropy 1.7 → 2.3,
clip fraction 0.10), with no warm-up phase.

Training-time sampled record vs the frozen parent (5% of games, both seats), per 50-update window, against arm A's
first windows (fresh critic, policy frozen for updates 1–60):

| window | critic_path: vs parent W–L–cap | of decisive | cap % | value std | raw adv std | control (arm A): vs parent | of decisive |
|---|---|---|---|---|---|---|---|
| 1–50 | 236–212–3 | 52.7% | 1% | 0.333 | 0.047 | 206–192–10 (policy frozen) | 51.8% |
| 51–100 | 482–426–66 | **53.1%** | 16% | 0.292 | 0.039 | 480–516–64 | **48.2%** |
| 101–127 / 101–150 | 312–280–64 | **52.7%** | 20% | 0.289 | 0.039 | 352–380–204 | **48.1%** |

Pooled after the control's warm-up: critic_path 794–706 (52.9% of decisive, updates 51–127) vs control 832–896
(48.1%, updates 51–150), 1,500 and 1,728 decided games, a 4.8-point gap at about 1.3 points of standard error each.
The continuation with its parent's critic never fell below its parent in sampled play; the fresh-critic control was
below from its first policy window.

## Paired ladder: update-100 snapshots vs the parent

Both update-100 snapshots vs `c5-omni-s1-final`, deterministic both sides, frozen engine, the same 128 fresh
`proposed-v1` 14×14 maps × 2 seats (seeds 389600–389727), one 32-core container each, 14:03–14:22 PDT
(app `ap-90oNYZsWLigpja7eA2HSlJ`; board `c6-credit/critic-path/c6-critic-path-ladder.json`; Wilson 95% CIs).
Arm A's update-100 is the snapshot its duplicate learner wrote (resumed from the original at ≈ update 70; see the
[arm A report](ctl-2026-10-21.md)), i.e. still a fresh-critic, per-decision continuation at 100 updates.

| candidate at update 100 | W–L–cap | all games [95% CI] | of decisive [95% CI] | capped | mean turns |
|---|---|---|---|---|---|
| `c6-critic-path` (parent's critic, no warm-up) | 99–102–55 | 38.7% [32.7, 44.6] | 49.3% [42.3, 56.2] | 21% | 50 |
| `c6-credit-ctl` (fresh critic, 60 warm-up) | 92–101–63 | 35.9% [30.1, 41.8] | 47.7% [40.6, 54.7] | 25% | 54 |

Parity with the parent for both, 2.8 points apart on all games and 1.6 on decisive games, well inside the
intervals: 256 games per side cannot separate them at update 100. (Arm A's *final*, update 308, was 43.1% of
decisive; its decline came after update 100, which a 25-minute run cannot reach.)

## Cost

| Item | Modal app | Billed (14:25 PDT) | Ledger |
|---|---|---|---|
| tests (8 vCPU, 19 s) | ap-NLF675KjNGvRzZxfgRxdlR | sub-cent | — |
| `c6-critic-path` (1×H100 + 12 × 16 vCPU, 25 min + ramp) | ap-87lLMLHainXhgAQN5NNsPN | 8.47 (13:00 row; the 14:00 row was not yet published) | 11.06 reserved |
| paired ladder (2 × 32-core containers, 19 min) | ap-90oNYZsWLigpja7eA2HSlJ | not yet published | 1.18 reserved |
| **Total** | | **≈ $10–11 expected** | **12.24 of 13** |

The cap was raised from $12 to $13 at 14:05 so the ladder could start: the ledger counts the training's $11.06
reservation until Modal publishes the closing hour, and the ladder reserved $1.18 on top. Billed cost will settle
around $10–11, under the original $12.

## Reading

1. **The mechanism is confirmed.** With `critic_path` the continuation starts from a trained critic (value std 0.21 on
   update 1 instead of ≈ 0.07), spends no updates in warm-up, and its sampled record against the parent sits at
   52–53% of decisive games in every window, where the fresh-critic control was at 48% from its first policy window.
   The warm-up (13 min of the control's hour) and the early sampled dip are gone.
2. **The deterministic ladder at update 100 does not separate the two** (49.3% vs 47.7% of decisive, both at
   parity with the parent). The control's real damage showed at its final checkpoint (43.1% at update 308), after the
   point this run reached. Whether the loaded critic prevents that later decline, or only the early one, needs the
   one-hour version of this run (≈ $25 with a ladder), ideally alongside the paired γ = 1 confirmation the readout
   recommends, since both should use `critic_path` from now on.
3. **Use it by default for every continuation** of an omni checkpoint on the same board and network
   (`critic_path` = the parent's `final-critic.npz`, `critic_warmup` 0). For a grown board the critic must be grown
   like the policy first. The optimiser state is still reset at a continuation; carrying Adam over from the parent's
   `latest.npz` is the remaining half of the restart cost and a separate change.
