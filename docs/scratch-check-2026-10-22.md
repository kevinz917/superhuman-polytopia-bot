# From-scratch pure self-play check (2026-10-22)

**Verdict: it learns, and fast.** In 20 minutes on 1×H100 + 6×16-vCPU actors (65 M samples, ~$5), a randomly
initialised policy went from coin-flip play to winning 98% of games against its own random init, deciding ~95% of
self-play games by elimination well before the round cap, and beating its own older snapshots in proportion to their
age. It is still far below the inherited lineage (1-54-9 vs c3), which is expected after 20 minutes. Nothing in the
check blocks the 7-h overnight run; the recipe changes below are cadence fixes, not learning fixes.

Run `scratch-check-1`, app `ap-YHPokZFq65eKYu1rv2GvfB`, branch `claude/scratch-selfplay-32b9ec` (A6 + Stage A + Rust,
i.e. main's training code). Ledger `runs/scratch/budget.json` ($10 cap; `runs/` is git-excluded, rows copied below).

## 1. Random-init parent

`/results/a6-scratch/random-init.npz` (+ `.json`, `random-init-report.json`), written by the new
`random_init` entrypoint in `experiments/modal_distributed_rl.py`: `network.init(key(20261022))` with the
architecture `network_spec()` reads from `c5-omni-s1-final` (normalized-v2, 14×14, 2 blocks, width 64, no extra
blocks), saved with `runner.save_checkpoint` and c5's run config so the Learner's `Config(**parent.json['config'])`
works unchanged.

Near-uniform check (Modal, 16 games × 150 sampled steps = 2,400 early-game positions, proposed-v1 maps): entropy of
each head along the sampled path vs log(#legal) at positions with more than one legal choice.

| head | positions | entropy | log(#legal) | ratio |
|---|---|---|---|---|
| family | 2,217 | 1.2184 | 1.2185 | 1.0000 |
| source | 759 | 1.3218 | 1.3218 | 1.0000 |
| target | 1,071 | 1.6459 | 1.7093 | 0.963 |
| full path | 2,400 | 2.278 | 2.306 | 0.988 |

No rescaling was needed: `network.init` already draws the family/source output weights at scale .01/√fan_in
(std ≈ 0.0013, Duper's 0.001). The target head's spatial keys are the layer-normed trunk tiles dotted with a
layer-normed query, so no weight scales them; its logits have std ≈ 0.5, which costs 4% of uniform entropy. The
entrypoint has a `--head-scale` for the non-spatial embeddings if that ever matters.

## 2. Pure recipe and where the brief's assumptions differ from the code

Config (the `a6-h100-smoke` row with these changes):

```json
{"size": 14, "maps": "proposed-v1", "frozen": [], "historical_probability": 0.1, "historical_every": 50,
 "historical_keep": 8, "batch": 262144, "minibatch": 16384, "epochs": 1, "learning_rate": 0.0003,
 "entropy_coef": 0.01, "segment": 32, "max_lag": 10, "publish_every": 2, "checkpoint_every": 30,
 "snapshot_every": 50, "omniscient_critic": true, "critic_warmup": 0, "pbrs_coef": 0.0,
 "cap_score_reward": 1.0, "turn_gamma": 0.99, "turn_lambda": 0.95, "seed": 501, "engine": "rust"}
```

Fresh omniscient critic (no `critic_path`: `initial_critic` returns the `key(1)` template), no frozen opponent, no
anchor. Learner env `POLYTOPIA_MATMUL_PRECISION=DEFAULT`, `POLYTOPIA_HEAD_PRECISION=HIGHEST`.

Checked against the code:

- **Round cap is 100 rounds on 14×14, not Duper's 12** (`gpu.ROUND_LIMITS[14]`; `max_turns=None` uses it). With
  random play nearly every game reached it at first (1.3% decided in updates 1-5); by update 50 most games end by
  elimination long before it, so the cap rule only mattered for the first few minutes.
- **`cap_score_reward` resolves on Polytopia score, not tiles.** At the cap the actor reads
  `states.players.score` (the state before the final action) and gives ±`cap_score_reward` to the leader as a real
  terminal (`ended=True`, no bootstrap); a tie is a draw (0). Score includes territory, cities, techs and units, so
  it is a broader proxy than Duper's tile count.
- **`cap_score_reward > 0` overrides `truncation_reward`**: the cap branch runs first; `episode_outcome` (and
  `truncation_reward`) only handles truncations when `cap_score_reward` is 0. Leaving `truncation_reward` unset is
  right.
- **Self-play share is derived, not configured**: `self_play_probability = 1 - historical - frozen` at publish
  time, so 0.9 here. The pool is empty until update `historical_every`; it holds the last `historical_keep`
  snapshots taken every `historical_every` updates.
- **`turn_gamma`/`turn_lambda` replace `gamma`/`gae_lambda`** everywhere (actor rows and the learner's GAE), with
  γ = λ = 1 inside a turn. Resign is masked out in training (`no_resign=True` in both engines).
- **Snapshot cadence bug in my config**: `snapshots/update-N` is only written when N is a multiple of both
  `checkpoint_every` and `snapshot_every` (the save sits inside the checkpoint branch), so 30/50 produced only
  `update-150`. The pool (`hist-u50/100/150/200`) is in memory and unaffected. Fix for the overnight run: make
  `snapshot_every` a multiple of `checkpoint_every`.
- No adaptive entropy exists; `entropy_coef` 0.01 held entropy at ~2.9 with no sign of collapse.

## 3. Throughput and health

249 updates, 65.3 M samples in 1,203 s: **54.3k samples/s overall, 55-61k steady**. Actor-bound (learner waits on
data), 0 stale drops, mean lag ~1, 96 connections throughout, 0 actor errors, no orchestrator replay (the
`{run}/calls` entry never changed). Per container ~9.5-10k samples/s vs ~7.3k in the A6 smoke from c5: random-ish
early games are cheaper to simulate, and the rate drifted down (61k → 55k) as games got richer.

## 4. Learning curves (25-update windows, from `training.json`)

| updates | entropy | KL | value std | return mean | samples/s | self-play games | decided | pool W/L/D |
|---|---|---|---|---|---|---|---|---|
| 1-25 | 3.07 | .0058 | 0.22 | 0.09 | 61.3k | 5,573 | 2.7% | — |
| 26-50 | 3.75 | .0048 | 0.45 | 0.12 | 59.7k | 3,059 | 34.1% | — |
| 51-75 | 3.22 | .0038 | 0.51 | 0.19 | 57.9k | 3,869 | 93.3% | 71/8/0 |
| 76-100 | 2.89 | .0029 | 0.60 | 0.21 | 58.6k | 3,786 | 98.2% | 349/32/0 |
| 101-125 | 2.80 | .0034 | 0.63 | 0.29 | 57.0k | 3,648 | 97.6% | 334/35/5 |
| 126-150 | 2.90 | .0030 | 0.65 | 0.33 | 56.1k | 3,201 | 97.1% | 428/49/7 |
| 151-175 | 2.99 | .0030 | 0.65 | 0.36 | 56.3k | 2,755 | 94.3% | 350/63/8 |
| 176-200 | 2.98 | .0031 | 0.64 | 0.34 | 56.8k | 3,072 | 93.9% | 244/81/13 |
| 201-225 | 2.90 | .0034 | 0.65 | 0.30 | 56.8k | 3,105 | 96.6% | 315/107/10 |
| 226-249 | 2.90 | .0033 | 0.65 | 0.26 | 55.4k | 2,918 | 95.4% | 291/80/4 |

- **Decided rate** (self-play games ending by elimination rather than the 100-round cap) went 1% → 98% within
  100 updates (8 minutes). Per map type over updates 151-249: classic 100%, pangea 99%, drylands 98%, lakes/jungle
  97%, highlands/waterworld 96%, ridge 95%, archipelago 93%, continents 85% (crossing water is the open problem).
- **Game length**: games got much shorter in rounds but longer in decisions. Random play runs the full 100
  rounds while ending turns early (end-turn is one of a handful of uniformly likely families); the trained policy
  finishes most games in ~20-40 rounds (arena: 19.4 turns vs the random init, 42 vs c3) but spends many more
  decisions per turn moving and fighting with its units, so finished self-play games per update fell from ~220
  (updates 1-25) to ~120-155 (part of that drop is the 10% of lanes that play the pool after update 50).
- **Entropy** rose first (3.07 → 3.75) as games reached positions with more legal moves, then settled at 2.8-3.0;
  the policy is not collapsing. KL 0.003-0.006 and clip fraction 2-5% per update: LR 3e-4 with one epoch is
  conservative.
- **Critic**: value std 0.01 → 0.65 and value loss ~0.003 by update 50: the fresh omniscient critic learned to
  predict outcomes within minutes, with no warm-up.
- **Return mean** is positive (0.1-0.36) because winners make more decisions per game than losers; it is a
  decision-weighted mean of ±1 outcomes, not a sign of reward leakage.
- **Pool record** (whole run, the learner seat vs each frozen snapshot): hist-u50 1719-106-30 (93%), hist-u100
  443-214-9 (67%), hist-u150 220-130-8 (61%), hist-u200 0-5 (5 games). Older snapshots lose more: steady
  improvement, no cycling. One stretch around updates 120-175 lost several single-update windows to hist-u100 and
  then recovered (7-2, 4-1, 5-0 in the last windows).
- **Summon mix** (land units, % of summons):

| updates | warrior | archer | defender | rider | catapult | swordsman | knight | mindbender |
|---|---|---|---|---|---|---|---|---|
| 1-10 (≈ random) | 40.5 | 13.6 | 11.3 | 16.1 | 3.2 | 4.4 | 2.9 | 4.1 |
| 51-100 | 53.9 | 6.7 | 5.0 | 12.7 | 1.1 | 0.8 | 0.3 | 1.9 |
| 101-150 | 61.0 | 6.1 | 5.4 | 9.0 | 0.7 | 0.6 | 0.0 | 1.1 |
| 226-249 | 50.8 | 16.5 | 9.4 | 6.2 | 0.4 | 2.6 | 0.0 | 0.4 |

  The early win is a warrior rush (61% at update 101-150, the same attractor c3 fell into); after that archers,
  defenders and swordsmen came back while warriors fell to 51%. Tech-gated units (catapult, knight) faded and had
  not returned by 20 minutes. The rest of the summons are naval (scout ships ~9-11%, rammers ~4%).

## 5. Arena (frozen engine, proposed-v1 maps, seeds 330000-330031 × 2 seats, unseen by training)

| match | modes | W-L-D | win rate | mean turns |
|---|---|---|---|---|
| final vs random init | both sampled | 63-0-1 | 98.4% [95.3, 100] | 19.4 |
| final vs c3 (u4854) | both deterministic | 1-54-9 | 1.6% [0, 4.7] | 42.1 |
| final vs update-150 | both sampled | stopped by the user before it finished | | |

The final policy's arena mix vs c3 was 55% warrior / 25% archer / 14% scout ship with no catapults; c3 played 48%
warrior / 31% catapult. A first `vs update-100` arena failed because that snapshot was never written (§2).

## 6. Ledger rows (`runs/scratch/budget.json`)

| run | kind | spend | result |
|---|---|---|---|
| random-init-random-init-* | random-init | $0.05 | the table in §1 |
| scratch-check-1 | 1×H100 + 6×16 vCPU, 1,200 s | ~$5.3 est. | §3-4 |
| scratch-final-vs-init | arena | ~$0.15 | 63-0-1 |
| scratch-final-vs-c3 | arena | ~$0.15 | 1-54-9 |
| scratch-final-vs-u100 | arena | ~$0.02 | failed: snapshot missing |
| scratch-final-vs-u150 | arena | ~$0.15 | stopped |

Total ~$5.8 of the $10 cap. Spend is the launcher rate × wall time × 1.33 (the c5 billing ratio); Modal billing
not yet read.

## 7. Overnight run plan (not launched; needs the user's approval)

Same recipe, fresh from the random init (new seed), 7 h, 1×H100 + 12×16-vCPU actors (last night's fleet shape).
Changes from the check:

- `checkpoint_every` 50, `snapshot_every` 250 (a multiple, so snapshots exist: ~35 over the run).
- `historical_every` 250, `historical_keep` 8: at ~100k samples/s an update takes ~2.6 s, so the check's 50-update
  pool would span only ~17 minutes; 250 × 8 spans ~90 minutes, which is what anti-cycling insurance needs.
  `historical_probability` stays 0.1 (0 is the purist option; the check showed no cycling either way).
- Everything else unchanged: LR 3e-4, entropy 0.01, epochs 1, batch 262144/16384, turn clock .99/.95,
  cap score 1.0, fresh critic, no frozen opponents.
- An Elo ladder against c3 and c5-omni-s1-final every 45 minutes (both deterministic, seeds 340000+, 32 maps),
  ~$5 over the night, so progress past c3 is measured during the run instead of after it.

Expected: 12 containers × 7.3-9.5k = 88-114k samples/s (the learner's ceiling is ~117k), ~2.2-2.9 B samples
and ~8-11k updates in 7 h, versus 0.58 B samples in last night's 6-h c5-omni arm, i.e. ~4× the samples for the
same fleet. Cost: the launcher estimates $16.58/h for this fleet and Modal billed the identical shape at ~$22/h last
night, so ~$155 for 7 h plus ~$5 of ladder; a ledger cap of $200 leaves room for the 1.3× billing ratio and one
orchestrator-replay overlap.

**Launched 2026-10-22 00:00 PDT on the user's approval** (12 actors, 7 h, ladder vs c3 and c5-omni, $200 cap):

- `scratch-overnight-1`, app `ap-6mk7N4jvzVTlwndXroG7UL`, parent `/results/a6-scratch/random-init.npz`, the
  config above with `historical_every` 250, `checkpoint_every` 50, `snapshot_every` 250, seed 502,
  `--seconds 25200`. Ledger `runs/scratch-overnight/budget.json` (cap $200; launcher reservation $120 at $16.58/h).
- Elo ladder `scratch-ovn`, app `ap-6NAuSYTh2xz7b7EEaUZ3KC`: every 2,700 s × 9 rounds, `latest.npz` vs c3 u4854 and
  c5-omni-s1-final, both deterministic, 32 proposed-v1 maps × 2 seats, seeds 340000 + 100·round; history in the
  registry `scratch-ovn/history` and `/results/distributed/ladder/scratch-ovn-history.json`.
- First minutes: all 12 actors connected (192 connections), 90.9k samples/s at update 33 (141 s); 80-93k samples/s
  through the first 27 minutes; `snapshots/update-250` written (cadence fix works).
- **04:37:45 preemption wave and orchestrator replay.** Modal preempted the orchestrator and most actor containers;
  the replayed `orchestrate` spawned a duplicate learner (`fc-01M5HPE80W98BCST2R63RTKK95`, resumed from `latest.npz`
  into the same folder) and 12 duplicate actor calls, and overwrote `registry[run]` with the duplicate's tunnel
  (:45075), so restarted original actors joined the duplicate. At 04:50:56 (caught by the status watcher's
  `{run}/calls` check) the 13 duplicate calls were cancelled with `terminate_containers=True`, the original entry
  (`r443.modal.host:42143`, recovered from the launch logs) was written back, and a 90-s guard rewrote the one
  `stopped` mark the duplicate's `finally` left. The original learner never stopped (update 5509 → 5605 by 05:00).
  Preempted actor inputs restarted onto it over the next ten minutes (9 of 12 by 05:00); 3 replacement actors
  (indices 20-22, `more_actors`, app `ap-prhfDmGE0PgqaLCNAgJT9F`) restored the 12-container fleet at 05:01.
  Estimated cost of the incident ~$3 (ledger row `scratch-overnight-1-replay`). The original orchestrator is gone,
  so this run will not write `{run}/result`; the end is the learner's `stopped` mark and `training-complete.json`.
- **Elo ladder so far** (latest checkpoint vs references, both deterministic, 64 games per cell, W-L-D):

| round (≈ time) | checkpoint | vs c3 | vs c5-omni-s1 |
|---|---|---|---|
| 0 (00:46) | update-850 | 1-44-19 (1.6%) | 0-58-6 (0%) |
| 1 (01:31) | update-1800 | 8-32-24 (12.5%) | 3-48-13 (4.7%) |
| 2 (02:16) | update-2800 | 38-20-6 (59.4%) | 21-40-3 (32.8%) |
| 3 (03:01) | update-3850 | 53-8-3 (82.8%) | 24-32-8 (37.5%) |
| 4 (03:46) | update-4850 | 45-15-4 (70.3%) | 31-30-3 (48.4%) |

  From random weights, the run passed c3 (the parent of every earlier lineage) after ~2.3 h and reached parity
  with c5-omni-s1-final, the strongest bot so far, after ~3.8 h.

## 8. Overnight result (`scratch-overnight-1`, 00:00-07:00 PDT)

The learner stopped on schedule: 8,724 updates, 2.287 B samples in 25,201 s (90.7k samples/s overall), 4× the
samples of last night's 6-h c5-omni arm. Pure self-play from the random init, no inherited weights or opponents.

**Ladder** (latest checkpoint every ~50 min, both sides deterministic, 64 games per cell on unseen proposed-v1
seeds 340000 + 100·round, W-L-D, 95% bootstrap CI on the win share):

| round (finished) | checkpoint | vs c3 | vs c5-omni-s1-final |
|---|---|---|---|
| 0 (00:51) | update-850 | 1-44-19 (2%) | 0-58-6 (0%) |
| 1 (01:42) | update-1800 | 8-32-24 (13%) | 3-48-13 (5%) |
| 2 (02:34) | update-2800 | 38-20-6 (59%) [48, 70] | 21-40-3 (33%) [22, 45] |
| 3 (03:23) | update-3850 | 53-8-3 (83%) [75, 91] | 24-32-8 (38%) [27, 50] |
| 4 (04:14) | update-4850 | 45-15-4 (70%) [59, 81] | 31-30-3 (48%) [38, 58] |
| 5 (05:10) | update-5550 | 44-17-3 (69%) [58, 80] | 28-34-2 (44%) [33, 53] |
| 6 (06:04) | update-6950 | 49-11-4 (77%) [67, 84] | 29-29-6 (45%) [38, 53] |
| 7 (06:55) | update-8400 | 51-12-1 (80%) [70, 88] | 37-26-1 (58%) [48, 67] |
| 8 (07:55) | update-8700 | 53-11-0 (83%) [75, 91] | 33-31-0 (52%) [41, 63] |

For scale: c5-omni-s1-final is c3 plus 6 h of omni-critic training, and it scored 77% of all games vs c3 on its
own ladder; this run reached 80-83% vs c3 and edged past c5-omni from random weights in 7 h (rounds 7-8: 70-57-1 combined, 55%). One 64-game cell has a
CI of about ±10 points, so "parity to slightly ahead of c5-omni" is the honest reading of rounds 4-8. The ladder's last cell is
`latest.npz` at update 8700; `final.npz` is update 8724.

**Training curves by hour** (means over the hour; decided = self-play games ending before the round cap; pool =
learner seat vs its own snapshots; summon shares in %):

| hour | updates | samples/s | entropy | KL | value std | decided | pool W/L/D | warrior | archer | defender | swordsman | catapult | knight | cloak | scout ship |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-1 | 1-1169 | 86.9k | 2.93 | .0044 | 0.64 | 90.6% | 10766/3235/409 | 36.1 | 14.0 | 3.4 | 22.7 | 0.8 | 0.2 | 0.3 | 11.7 |
| 1-2 | 1170-2316 | 84.1k | 2.91 | .0048 | 0.68 | 97.9% | 12546/5373/261 | 27.1 | 27.3 | 5.0 | 20.4 | 0.7 | 0.0 | 0.4 | 10.3 |
| 2-3 | 2317-3503 | 87.1k | 2.94 | .0051 | 0.69 | 96.9% | 10190/5060/632 | 25.4 | 34.4 | 9.9 | 9.4 | 0.9 | 0.0 | 2.2 | 10.4 |
| 3-4 | 3504-4715 | 88.6k | 2.90 | .0050 | 0.70 | 97.8% | 10153/7430/567 | 24.7 | 36.2 | 10.4 | 4.8 | 1.5 | 0.0 | 4.5 | 10.9 |
| 4-5 | 4716-5608 | 79.8k | 2.90 | .0047 | 0.70 | 97.5% | 7927/6119/272 | 24.8 | 35.5 | 9.8 | 2.2 | 3.9 | 0.0 | 6.3 | 11.5 |
| 5-6 | 5609-7129 | 115.0k | 2.96 | .0054 | 0.69 | 97.2% | 12403/9596/536 | 26.2 | 32.7 | 10.7 | 1.3 | 3.3 | 1.2 | 7.0 | 11.4 |
| 6-7 | 7130-8723 | 120.3k | 2.96 | .0054 | 0.70 | 97.2% | 12025/9836/582 | 25.5 | 31.9 | 11.6 | 0.7 | 3.0 | 2.7 | 7.1 | 11.1 |

- Entropy held at 2.9-3.0 all night and KL stayed ~0.005: no collapse, no instability, no sign the fixed
  `entropy_coef` 0.01 was wrong.
- The pool win share fell from 75% (hour 0-1) to ~55% (hours 5-7): the snapshots it plays got stronger as fast as
  it did, which is expected once the pool spans ~90 minutes of a still-improving policy; no cycling showed in the
  ladder either.
- The unit mix diversified rather than narrowed: warriors 36% → 25%, archers became the main unit (32-36%),
  defenders 3% → 12%, cloaks 0.3% → 7%, and catapults (3-4%) and knights (2.7%) came back in hours 4-7 after
  fading in the first hours. Swordsmen were an early-hour unit (23%) and faded to under 1%. Compare c3's 99% warrior
  recruiting (docs/unit-diversity-investigation-2026-10-18.md).
- Throughput 80-89k samples/s on 12 actors, then ~124k (the learner's ceiling; inbox full, a few hundred stale rows
  dropped per update) on 15 actors after the replay recovery.

**Cost** (ledger `runs/scratch-overnight`, estimates at the Oct 21 billing ratio of 1.33× the launcher rate; Modal
billing not yet read): run $155, replacement actors $7.6, replay incident ~$3, ladder ~$4: ~$170 of the $200 cap.
The 15-actor stretch happened because original actors 4, 5 and 10 restarted after the 3 replacements were attached.

Final policy: `/results/distributed/scratch-overnight-1/final.npz` (+ `final-critic.npz`), snapshots every 250
updates under `snapshots/`.
