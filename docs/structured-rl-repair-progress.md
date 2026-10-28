# Reinforcement bot repair campaign

Goal remains active: an end-to-end, somewhat decent learned bot, within $500
of Modal spend. Tactical qualification alone does not satisfy it. Full-game
acceptance criteria and experiment hypotheses are in
`docs/structured-rl-repair-plan.md`.

## Implemented and checked

- Joint-entropy gradient corrected with importance sampling. Exhaustive tests
  compare the entropy and every parameter gradient, including after policy drift.
- Versioned normalized CNN with mean/max pooling and separate critic projection;
  legacy checkpoint interpretation retained. New local training CLI defaults to
  the normalized version. The campaign always explicitly uses this version.
- Advantage standardization divisor bounded below by .05. Raw scale, returns,
  values, feature sensitivity, KL and gradient norms are logged. Subsequent runs
  additionally record component gradient norms and action counts.
- Explicit no-resignation policy mask persists through checkpoint reload and
  public API inference. All other legal commands remain available.
- Native capture and attack-then-capture curricula with randomized positions and
  both player identities; no action labels, demonstrations or reward shaping.
- Standalone public-API evaluation against the unchanged balanced teacher or
  family-uniform random baseline, with complete compressed traces and confidence
  intervals. This evaluator is prepared; full-game strength has not yet passed.
- All 22 frozen engine source hashes still match.
- The combined repair, PPO, league and native-rule suite passed 49 tests;
  log: `runs/structured-ppo-v3/combined-tests.log`.

## First hypothesis test: immediate capture

Run `repair-capture0-v1`, Modal app `ap-GXY3QsWuOTBx6NvkVbv5Si`, was intentionally
stopped after qualifying its saved update-16 checkpoint. Modal confirmed
`stopped`, zero tasks. Stopping was an experimental decision, not a training
failure. Later unsaved updates are not the selected model.

The selected model contains 519,267 parameters and learned from 32,768 learner
decisions, using only native terminal rewards. On 32 additional fresh tactical
positions (310000-310031):

| Result | Random initialization | Saved update 16 |
|---|---:|---:|
| Native wins in one deterministic action | 0/32 | 32/32 |
| Mean probability of capture | 0.33169 | 0.74582 |
| Illegal actions | 0 | 0 |

These positions already have the attacker on the enemy's last city. This proves
learning to select capture, not movement, combat, full-game play or strength.

Checkpoint SHA256:
`4fcb3f3408a2d92b23d47fc13098fe54d42c24819feb7bfdadbecd6389ff41f4`.
Both retained checkpoint hashes and all 32 recorded RL/engine source hashes
were verified against an archived source copy. Artifacts:
`runs/structured-ppo-v3/artifacts/repair-capture0-v1/`.
Fresh-position probes: `runs/structured-ppo-v3/capture0-latest-probe.json` and
`runs/structured-ppo-v3/artifacts/repair-capture0-v1/initial-probe.json`.

## Movement experiment: passed

`repair-capture1-v1` uses the exact qualified checkpoint as its parent and
requires moving onto the enemy city before capture. It has passing opponents
only for this tactical stage, and still uses native terminal rewards only.

- Modal app: `ap-zdgUjnRqkfNJ5ZB4PHCBiy`; confirmed stopped, zero tasks.
- Local client session was launched detached; log:
  `runs/structured-ppo-v3/repair-capture1-v1.log`.
- Remote artifacts: volume `polytopia-structured-rl`, staging environment,
  `/repair-capture1-v1/`.
- Training stopped at update 16 after its early-stop criterion. Held-out
  stochastic and deterministic evaluations each won 32/32, with no truncations,
  cutoffs or illegal actions. Deterministic play used exactly move, end, capture
  per learner; the passing opponent supplied the intervening end-turn action.
- Function-reported elapsed time: 353.14 seconds. Observed app wall time:
  375 seconds, approximately $0.269 at the declared maximum allocation rate.
- Full artifacts downloaded to `runs/structured-ppo-v3/downloads/repair-capture1-v1/`.

The first two movement-stage batches recorded 663 native wins and 5 truncations,
with 1,077 movement commands and zero illegal actions. Component-gradient logging is functioning;
the first batch had scaled actor/value/entropy norms 0.4124 / 0.00211 / 0.00703.

The launcher now also supports `--stage -2 --parent RUN_ID` for a development
probe on unchanged native maps with no training. It evaluates 16 paired-seat
maps against random, stochastically and deterministically, to the engine limit.
Use this to check transfer after curriculum qualification; it does not replace
the reserved final tests against random and the balanced teacher.

## Live experiments after movement qualification

**Latest update:** combat qualification passed, the first native transfer probe
completed, and the first full-map training pair was externally cancelled before
its first update. See the continuation record below; the older live-job list
in this section is retained as experiment provenance, not current live state.

Two independent bounded jobs branch from the immutable movement final checkpoint:

- `repair-native-probe1`, app `ap-DVURseH0lMj744R2l8H6e6`: no training, ordinary
  14x14 mapgen games versus random, 16 paired-seat maps, both inference modes.
- `repair-combat1-v1`, app `ap-xHsrcU4a5A8mmj91Ns1v3F`: terminal-reward PPO on
  randomized adjacent attack-then-capture starts with a one-health defender.

Logs are `runs/structured-ppo-v3/RUN_ID.log`, results are on the staging volume
`polytopia-structured-rl` under each run ID. Inspect both jobs and held-out
results before selecting the next training design. Full-map self-play and the
reserved strength tests are still required; this goal is not complete.

Curriculum fixtures are deliberately distinct from full games: fully explored
grassland, two cities, a short 12-round guard, and a passing opponent. The combat
stage adds a one-health defender. These are versioned training-state
distributions using unchanged transition code, not changes to native mapgen or
claims about the primary game's difficulty. Native probes retain the full
100-round / 10,000-action guards and fog-of-war mapgen.

## Budget

The first app's observed wall time was 403 seconds, approximately $0.289 at the
declared maximum allocation rate. This is an estimate, not an invoice. $5 per
run remains reserved conservatively: $20 reserved for four runs out
of $500. Authoritative ledger: `runs/structured-ppo-v3/budget.json`.

After the first two serial runs finished, the launcher was upgraded to atomic
locked reservations and merged status updates. Two concurrency tests passed,
including 32 concurrent attempts against a four-run budget. The two live jobs
therefore cannot overwrite each other's reservation. Never relaunch merely
because observation timed out; check each exact Modal app ID.

Useful inspection:

```bash
modal app list --env staging --json
modal app logs --env staging ap-DVURseH0lMj744R2l8H6e6 --tail 10
modal app logs --env staging ap-xHsrcU4a5A8mmj91Ns1v3F --tail 10
```

For folder downloads, create the local destination directory first. This Modal
CLI nests the remote run folder inside that directory; inspect the result and
verify hashes before using a checkpoint. Do not mistake an NPZ file written to
a nonexistent destination path for a directory download.

## Continuation: native transfer, combat and full-map training

The previous goal turn was progress: it repaired learning code and qualified
two tactical stages. This continuation completed further evidence and changes.

Combat `repair-combat1-v1` passed 32/32 held-out games in both modes. Each
deterministic game required attack, end, capture; no truncations or illegal
actions. Downloaded to `runs/structured-ppo-v3/downloads/repair-combat1-v1/`;
33 archived source hashes, four current inference hashes and three checkpoints
verified, plus 12 legal host-API decisions. Final Adam step 768.

Movement checkpoint transfer to ordinary native maps (`repair-native-probe1`):

| Mode | Wins | Losses | Truncations | Mean maximum cities | Mean final units |
|---|---:|---:|---:|---:|---:|
| Stochastic | 1/32 | 0 | 31 | 2.03125 | 1.21875 |
| Deterministic | 12/32 | 0 | 20 | 5.25 | 10.5 |

No illegal moves or evaluator cutoffs. These remain development results and
fail the final strength gate. Stochastic play recruited 1,283 and disbanded
1,275 units; deterministic play recruited 1,194 and disbanded 848.

Implemented optional `own-potential-v1` shaping, with six regression tests for
discounted-return telescoping, terminal/truncation behavior, cycles and no enemy
private input. The preflight PPO/league/repair/reward/budget suite passed 32 tests
(`full-training-preflight.log`). Frozen engine hashes remain unchanged. Both
planned full-game branches preserve the actor but reset the value output head
and Adam, avoiding the easy-curriculum's near-certain-win value prior.

The first full-map pair and combat-parent native probe were cancelled by remote
signals around 130 seconds into their calls. Modal explicitly reports stopped,
zero tasks for all three. Exact cancellation cause is not established; this was
not a reported PPO numerical failure or timeout at the configured one-hour cap.
No updated/final training checkpoint exists for that pair. Their $5 reservations
remain charged conservatively. IDs:

- `repair-full-native1`: `ap-S7pUjegvZsLLeKI3pJIJUM`.
- `repair-full-potential1`: `ap-GVWe8nyrkcQElFXKSjOaso`.
- `repair-native-combat-probe`: `ap-42RahtZ3JKts12UzR32SpC`.

The launcher now supports `--background`: it uses `Function.spawn()`, stores
the durable function-call ID and exits the local client. Inspect completion with
`scripts/poll_structured_run.py`; timeout means pending, never resubmit. This
follows Modal's durable invocation pattern:
https://modal.com/docs/guide/function-invocation-methods . This is a mitigation
being tested, not proof of the original cancellation cause.

Current retry: `repair-full-native2`, app `ap-zLFlGzhUgEtQDRqYTJLSTe`, call
`fc-01M56WFWZ3DWR0BHV1BYTVCWMS`. Requested 64 full-map PPO updates / 2,400 training
seconds from the qualified combat checkpoint, native rewards, reset critic and
Adam. It was confirmed live with one task after the local client exited.
The CLI's "App completed" means the local entrypoint completed, not that the
detached function is terminal. Monitor the exact app and function-call IDs.

The retry survived the earlier cancellation interval and completed PPO updates.
The potential branch has therefore also resumed through a durable call:
`repair-full-potential2`, app `ap-Z0xSg6kL4UfbatEMm46vJb`, call
`fc-01M56WP2K03YANMBJ0H7QZS85C`. Both are live, with no numerical or legality
failures in their initial updates. $45 is reserved across nine attempts. The
$500 goal remains active and unfulfilled. Do not duplicate live calls; keep at
most two campaign jobs running while diagnosing the earlier cancellations.

The branches' first batches had identical action histograms, 2,048 learner
decisions and 2,059 opponent decisions. Neither saw a native ending. Native-only
advantages were exactly zero, and its actor updates came solely from entropy.
Potential shaping gave raw advantage std 0.06920, reward std 0.01822, and finite
policy/value gradient norms. This verifies the intervention generates training
feedback on identical experience, not that it improves gameplay. Await outcomes.

Potential-run compiler timing: 72.15 seconds; first collection: 46.55 seconds.
A local lowered-IR probe (`collector-graph-probe.json`) found roughly 15 MB of IR
in the full collector, with only about 0.34% growth from legacy to normalized
architecture. IR size is not a GPU throughput benchmark. The first native
run's second update took 63.40 seconds, so report actual update counts if either
run reaches the 2,400-second cap before 64 updates.

The combat checkpoint also completed a small independent host-API development
test against the unchanged balanced teacher on map 300016, paired seats:
0 wins, 2 native losses, no truncations/cutoffs, 17.76 seconds. In seat one it
recruited no units and lost on turn 15. Full public traces and protocol are in
`runs/structured-ppo-v3/combat1-teacher-dev/`. This is a diagnostic, not the
reserved 64-game acceptance benchmark. Map seeds 400000-400031 remain untouched.

For durable polling (no resubmission):

```bash
MODAL_ENVIRONMENT=staging python scripts/poll_structured_run.py repair-full-native2
MODAL_ENVIRONMENT=staging python scripts/poll_structured_run.py repair-full-potential2
```

Local launch logs are no longer live training logs after `--background` returns.
Fetch the exact app's logs or inspect committed volume artifacts instead. The
cancelled combat-parent native probe remains unfinished and should be rerun only
after one of the current jobs finishes, if needed for checkpoint selection.

## October 18 continuation: full-budget objective and new pilots

The user expanded the objective to use the full $500 budget to train the best
bot possible. The minimum strength gate remains necessary, but passing it alone
no longer ends the campaign. The goal remains active; final maps are untouched.

The original native/potential pair remains live. At native update 17 it had
34,816 learner decisions, 2 wins, 2 losses and 48 truncations. Potential update
16 had 32,768 decisions, zero native endings and 46 truncations. Its state
features and values vary, but shaping feedback has not yet demonstrated wins.

Implemented a device public-state opponent with scores checked against the
unchanged balanced host teacher on native and rich tactical positions. Its tie
break is randomized and separately versioned. Collector tests verify that its
actions do not become learner PPO samples. Pure-RL mixed-opponent pilot:

- `repair-mixed-potential1`, app `ap-qAODRNjqUSvdJTfJIvwmyr`, call
  `fc-01M56XFZK0ZDWCX267A0F19HXB`; combat parent, own-potential reward, 25% fixed
  public opponent / otherwise current and historical policies; 32x64 rollouts.

A separate explicitly labeled bootstrap pilot uses teacher trajectories on
native training maps 210000 onward. It is imitation, not evidence of pure-RL
gains. The learned bot uses only the network at inference, and later PPO must
be evaluated separately. It uses 128x16 atomic actions, four BC epochs, and fresh
Adam from the combat actor:

- `repair-bootstrap1`, app `ap-bEz3s2k8LnPBwmxTt4VkoF`, call
  `fc-01M56XCN3ZFS67ZAS7K43Z3M0K`; requested 256 updates / 2,400 training seconds.
- Early collection was approximately 1,380 actions/s; end-to-end post-compile
  updates about 1,020 labeled decisions/s, later about 930/s. These teacher
  atomic actions are not directly comparable to learner-plus-opponent PPO
  transitions. Native endings in its training log are teacher-versus-teacher
  games, not learned-policy wins.
- Downloaded **update 48** (98,304 labels) to
  `runs/structured-ppo-v3/downloads/repair-bootstrap1/`; retain this local copy
  unchanged and use a separate destination for the eventual final download.
  SHA256 `9332e6ba7589d0142b4f971430b914409d5e96280f3ce44cc2697e12099f837c`.
- Verified 36 archived sources, four current inference sources, both checkpoint
  hashes, finite tensors and 12 legal host decisions. Adam step 1,536.
- Four paired-seat development games on 300016-300017 versus the unchanged host
  teacher: **1 win, 3 losses, no truncations/cutoffs**. Wilson interval approximately
  4.6%-69.9%. All four expanded to multiple cities; no disbanding. This is a small
  diagnostic, not the acceptance test. Full public traces are in
  `runs/structured-ppo-v3/bootstrap1-mid-teacher-dev/`.

Implemented an independent-lane learner-only PPO collector. Each lane holds a
pending learner transition while its opponent responds; it flushes on the next
learner turn or ending. Lanes synchronize at rollout completion, not at every
learner action. Public critic inputs, macro-step gamma/lambda, native rewards,
potential transform and pre-reset truncation bootstrap are retained. Fifteen
conformance/native tests passed; performance remains to be measured on GPU.

- `repair-async-potential1`, app `ap-kMtbJO8Hj44Ce1087JhCii`, call
  `fc-01M56XNXR1W06MKPB5G0VV29R3`; combat parent, 25% fixed opponent,
  own-potential reward, 128x64 learner rollouts, 128 requested updates / 2,400 sec.
- Its larger batch and changed RNG schedule mean comparison with the synchronous
  pilot is a throughput/recipe comparison, not an isolated learning ablation.

The new opponent/bootstrap/reward/budget preflight passed 27 tests; independent
collector conformance and native tests passed 15. All 22 frozen hashes still
match. The budget now reserves **$60 across 12 attempts**, including five live
pilots; these reservations are not actual spend.

`experiments/modal_structured_scale.py` is prepared but **not launched**. It
supports 12-hour hard-capped L40S continuations, at most 11 hours of training,
and a $35 atomic reservation against the same ledger. Choose parents and
hyperparameters from the pilots before allocating the bulk of the budget.

The independent collector subsequently compiled in 56.42 seconds. Its first
collection took 19.91 seconds for 8,192 learner transitions. Its second complete
update took 31.15 seconds, about 263 learner decisions/s (29.52 seconds in
collection), versus about 35-40/s in the synchronous 32-lane pilots. No illegal
actions, exhausted loops or probability failures; replay error 1.91e-6. This
early roughly sevenfold recipe improvement includes the larger batch and must
be rechecked as armies grow. A further overlap regression confirms alternating
opponent delays finish in 12 atomic batch steps instead of 16 barriers, retaining
all learner observations in order. Both tests in that file passed.

### Bootstrap completion and first substantial PPO allocation

`repair-bootstrap1` finished all 256 imitation updates, **524,288 labels**, and
saved an immutable final checkpoint before evaluation. Its teacher-driven
training stream recorded 471 native endings and 18 truncations; those are not
the learned bot's outcomes. Final checkpoint:
`92abe382a42855de4be6ac7624ca0e05495e9f7f345203e9dad6884128c7e252`.
Downloaded separately to `runs/structured-ppo-v3/final-downloads/repair-bootstrap1/`.
All 36 archived sources, four current inference files, three checkpoints and
12 host decisions verified; Adam step 8,192. The original update-48 download
remains intact.

Its GPU random-opponent evaluation is still active. A separate **16-game**
deterministic host-teacher development benchmark is running on maps 300000-300007
in `runs/structured-ppo-v3/bootstrap1-final-teacher-dev/` (local exec session
58882; inspect process/log instead of restarting if the session handle expires).
First three completed games: loss, win, win. The latter win took 46 turns,
105 attacks and reached 13 cities. These partial results are not the final
win rate. A subsequent game reached turn 91, so retain engine-limit truncations
in the denominator. No final acceptance maps have been used.

The pure-RL independent-lane pilot reached 40,960 decisions with **65 losses,
1 win and no truncations**, after its first fixed-opponent batch. This supplies
native strategic feedback absent from the weak-only pair; it does not yet show
strong play. Its fifth update took 48.74 seconds, about 168 decisions/s, slower
than the second update as opponent turns grew but still above the old recipe.

Two substantial PPO branches are now submitted from the verified bootstrap
final checkpoint, with no further imitation labels during these runs:

- `scale-bootstrap-native1`: app `ap-d4ipbskvTvgRaqkcplLAD0`, call
  `fc-01M56Y5P7FXHTWVD6J07YGTCRN`, native terminal reward.
- `scale-bootstrap-potential1`: app `ap-TzPYfyZbzKbd9pKyhTifjJ`, call
  `fc-01M56Y6S31T2697XZ1CH6MRSEV`, own-state potential shaping.

Each requests up to **11 training hours**, with a **12-hour hard cap**, one
L40S/8CPU/32GiB, 128x64 learner rollouts, 512 training maps, fresh value output
head/Adam, learning rate 1e-4, gamma .995, lambda .99, entropy .005, and 25%
fixed public opponent / otherwise 50:50 current vs historical snapshots.
The pair uses the same initialization and seed. Both reserve $35, bringing the
ledger to **$130 reserved across 14 attempts**, not $130 spent. Monitor exact
durable handles and checkpoint results; do not blindly resubmit on observation
timeouts. Further budget allocation remains conditional on measured progress.

Bootstrap GPU evaluation subsequently **completed**: stochastic **32 wins / 32**
and deterministic **32 wins / 32** versus non-resigning family-uniform random,
zero losses, truncations, evaluator cutoffs or illegal moves in both modes.
These are development maps, not the untouched final acceptance set. Mean maximum
cities 8.0625 in each mode; final units 16.47 stochastic / 14.72 deterministic.
The result is stored locally in `repair-bootstrap1-summary.json` and in the
final download's `summary.json`. Modal confirms stopped, zero tasks. Function
time 920.42 seconds; app wall 942 seconds, approximately $0.676 at the declared
allocation rate, not an invoice. Its $5 reservation remains retained.

The host-teacher benchmark had reached 15/16 games at the latest inspection:
7 wins, 7 losses, 1 engine truncation. Inspect the completed summary before
reporting a final rate. Both scale apps were confirmed detached with one task
after their local clients exited. The native branch finished collector
compilation; one potential-branch log query hit a server resource limit. That
was an observation error, not evidence that the job failed; its durable call
was pending and its app was live. Do not restart it on that basis.

The host-teacher benchmark has now completed: **7 wins, 8 losses, 1 truncation
in 16 games** (43.75%; Wilson 95% interval 23.10%-66.82%), zero evaluator cutoffs,
293.56 seconds. This remains below the minimum teacher acceptance fraction on
development maps, and does not replace the final test. A matched 16-game
combat-parent benchmark was launched at
`runs/structured-ppo-v3/combat1-teacher-matched-dev/` (local exec session 34804
is not authoritative; inspect the process/log). This isolates the baseline
comparison on the same maps and inference mode without consuming Modal budget.

Both scale branches now produced valid PPO updates. Their first 8,192 decisions
and action histograms matched exactly, against the fixed public opponent; the
native branch had zero raw advantages while shaping supplied std 0.07076. Native
update two saw two losses. Probability replay error stayed below 5.85e-6; no
illegal moves or exhausted loops. The earlier log-query resource error cleared.

## Continued qualification and probability-replay investigation

The previous goal turn was progress: it created/tested faster collection,
qualified the bootstrap, and started substantial PPO work. This continuation
revalidated the live handles before changing anything.

The matched combat-parent benchmark completed **0 wins / 16 native losses** on
the exact 300000-300007 paired maps and deterministic mode where the bootstrap
scored 7 wins, 8 losses and 1 truncation. This is now a matched comparison;
neither result uses final acceptance maps.

The native long run `scale-bootstrap-native1` then failed its probability replay
check at **1.0251998901367188e-5**, just above the **1e-5** threshold. Its durable
function call raised that exception; Modal confirms stopped, zero tasks. No
successful long continuation is claimed. The potential branch remains live and
had reached update 9 (73,728 decisions, 17 native wins / 90 losses). Do not stop
or restart that branch because the independent native branch failed.

Added a diagnostic replay path that recomputes log probabilities in actor-sized
chunks when the full flattened batch fails. The tolerance is unchanged. It
saves the offending actor-sized sample group, parameter/Adam checkpoint and
both replay results before accepting a passing chunked check or raising on a
second failure. Four tests cover exact chunked network evaluation, rejection
of remaining mismatches/nonfinite values, and diagnostic retention.

`repair-replay-diagnostic1`, app `ap-8PQ8nERoaSkRBNyFK1oLMP`, call
`fc-01M56YR26JX5Y8B6W8JDWF1VNG`, is a bounded 12-update rerun with the failed
run's 512-map pool, seed 20261020, native reward, 128x64 rollout, parent and
opponent mix. Both initial checkpoint hashes match exactly:
`7cacea6d0ec423bfc39279e72d8609c93ac449f0996fed23c9193df3721ef934`.
Await the numerical evidence before deciding on a larger native retry. The
failed original did not retain the offending batch; its exact failure cause
is therefore not yet established.

`scripts/watch_structured_checkpoints.py` now retains hash-matched immutable
snapshots, verifies archived source hashes, and queues one local development
evaluation at a time. It performs no Modal mutations or job resubmissions.
Three retention tests pass, including rejecting mismatched checkpoint/source
bytes. A real bootstrap capture also passed. The 13-hour watcher is active in
local exec session **57545**, log `checkpoint-watch.log`, status
`retained-checkpoints/watch-status.json`. It watches the potential long run and
four pure-RL pilots; it intentionally excludes the failed native run. Snapshots
of pure-RL updates 16/32 have already been retained and evaluation started.
Check process state before restarting the watcher; its lock prevents duplicates.

Budget accounting now uses explicit terminal receipts to free unused allocation
after confirmed completion/failure. Nine terminal apps were reconciled at the
maximum allocation rate plus $1 margin each, with original reservations kept
in the ledger. After adding the $5 diagnostic reservation, **$71.54** is committed
including live reservations and completed-job margins; this is not actual
invoiced spend. All still-live jobs retain their full reservation. Accounting
receipts are in `apps-accounting-20261018.json`; the authoritative ledger remains
`budget.json`. This preserves the full $500 optimization objective without
mistaking unused reservations for consumed GPU budget. All 22 frozen hashes match.

The replay diagnostic reproduced the numerical failure at update 5: flattened
8,192-sample replay max error **1.0013580322265625e-5**, with one exceedance;
128-sample chunked replay error **exactly zero over all 8,192 samples**. The
saved offending group is float32 and also matches behavior exactly on reload.
Evidence downloaded and checkpoint hash verified at
`runs/structured-ppo-v3/replay-audits/update-5/` (`report.json`, `samples.npz`,
`policy.npz/json`). This establishes the batch-shape numerical failure class;
the original failed run did not save its precise offending batch.

The diagnostic was deliberately stopped after answering that question. Modal
confirms stopped, zero tasks; its 465-second app lifetime plus $1 margin is
accounted as $1.34. It is not an unexplained failure and should not be resumed.
The native long branch was restarted from the same immutable bootstrap parent
with the unchanged tolerance and chunked fallback:

- `scale-bootstrap-native2`: app `ap-c0KhiK8fgfb3J3hPMRvE2y`, call
  `fc-01M56Z6DYCE2R99MSENSDS44KB`, same 11-hour requested / 12-hour hard cap,
  seed, map pool, actor, critic reset and opponent scheme as the original.
- The existing potential long branch remains unchanged and live.
- New runs also save recovery checkpoints every four early updates, without
  changing the 16-update opponent snapshot cadence.
- The long launcher now accepts `--parent-checkpoint latest` for a verified
  terminal run that lacks a final checkpoint; default remains `final`.

A separate bounded watcher for the new native run is active in local exec
session **57050**, log `checkpoint-watch-native2.log`, status
`retained-checkpoints/watch-native2-status.json`. It shares the immutable
snapshot directory but watches only the new native run. Each watcher evaluates
one checkpoint at a time, at most two local evaluators across the two cohorts.
New watcher versions additionally lock each run to prevent duplicate ownership.
The earlier primary watcher (57545) continues its original five run IDs.

The long potential **update-16** checkpoint scored **5 wins / 3 losses** on the
eight matched teacher development games, versus the bootstrap's **4 wins,
3 losses, 1 truncation** on that exact subset. One former truncation and one
loss became wins, while one former win became a loss. This is encouraging
but a small comparison, not a demonstrated statistically reliable gain or a
final acceptance result. Retained artifact:
`retained-checkpoints/scale-bootstrap-potential1/update-16-02c9fc2afc1f/`.

Pure-RL asynchronous update 16 scored 1 win / 7 losses on those eight games.
The synchronous native-only and potential-only update-32 checkpoints both lost
all eight. The native-only pilot subsequently completed its own random-opponent
evaluation: deterministic **28 wins / 4 truncations** and stochastic **5 wins /
27 truncations**, each over 32 games, no illegal moves or evaluator cutoffs.
Its exact final teacher result is still pending; do not equate update 32 with
the final checkpoint. Current committed budget, after the native retry and
diagnostic accounting, is **$102.88**, including all live reservations and
terminal-job margins, not invoices. Goal remains active and unfulfilled.

The asynchronous pure-RL checkpoint at update 32 now has a complete matched
teacher probe: **4 wins / 4 losses / 0 truncations**, versus 1/7 at update 16.
Mean maximum owned cities rose from 3.75 to 6.75 on those eight games; the new
checkpoint made 542 attacks and 70 capture-family actions. Capture-family counts
include ruins and are not a count of city captures. This is evidence of active
expansion/combat, but eight games do not establish broad strength. An additional
24 development games on seeds 300004–300015, paired seats, are running locally
at `async32-teacher-expanded-dev/` (exec session 97410). Combined with the
original eight this will cover the full 32-game development set. Final seeds
400000–400031 remain untouched.

After verifying 37 archived source hashes, four current inference hashes, both
checkpoint hashes, finite tensors, 2,048 Adam steps and 12 legal host decisions,
the exact update-32 model/optimizer was uploaded as immutable parent
`pure-async32-aa8505267dcd/final.npz`. Remote readback matches SHA-256
`aa8505267dcd20bfe4e263cf217e5dc97b57c148dac6ae104b96fabc8af55375`.
Its lineage and qualification are retained in `qualified-parents/` and uploaded
alongside the checkpoint. This is pure RL without any behavior-cloning labels.

A bounded long continuation has been submitted:

- `scale-pure-potential1`, app `ap-eHw4gVBhXd4TiNkyc2Bzxr`, function call
  `fc-01M56ZJFWRFAEHGDFZSDA3N1CV`.
- 11-hour training request / 12-hour hard limit, one L40S / 8 CPU / 32 GiB,
  $35 reservation, seed 20261021, 512 training maps, 128x64 learner rollout.
- Same own-potential reward, learning rate 1e-4, entropy 0.005, gamma 0.995,
  lambda 0.99 and 25% fixed-public-opponent mix. Actor, critic and Adam preserved.
  New episode states and an opponent bank initialized to the parent make this a
  continuation from weights/optimizer, not a bit-for-bit process resume.
- New replay numerical audit and early recovery checkpoint handling included.
- Dedicated read-only local watcher: exec session 26242, PID 47201 at startup,
  `checkpoint-watch-pure1.log`, `watch-pure1-status.json`, 13-hour bounded lifetime.
  This is a third disjoint watcher cohort; no duplicate ownership of prior runs.

Both earlier synchronous full-map pilots are now confirmed stopped with zero
Modal tasks. Their reservations were reconciled from fresh app receipts:
`repair-full-native2` $3.11 and `repair-full-potential2` $3.05, each including
$1 margin above the maximum-allocation wall-time estimate. The latter's final
random evaluation scored 32/32 deterministic wins and 17 wins / 15 truncations
stochastically. Its final teacher evaluation is still pending, while its earlier
update-32 checkpoint lost all eight teacher games. Do not equate random strength
with teacher strength, or intermediate and final checkpoints.

After adding the pure-RL long allocation, the ledger commits **$134.04** including
all live reservations and completed-job margins, not invoiced spend. Full $500
optimization remains the objective. All 22 frozen engine source hashes match.

The pure continuation's remote initial NPZ is byte-for-byte identical to the
qualified parent (same `aa8505267dcd...` hash), confirming that actor, critic and
optimizer were preserved. Its first 8,192-decision PPO batch completed with
zero illegal actions, zero uncredited endings and replay error 1.43e-6. Feature
standard deviation was 0.0653 and probe value standard deviation 0.0398.

The expanded teacher evaluation materially weakens the quick pure-RL result:
its first 16 additional development games were all losses (4/24 wins when
combined with the original eight). This is a partial result, not the final
32-game score. The long branch remains an optimization experiment, not a
qualified strong bot. No expansion/combat-count claim substitutes for wins.

H7's controlled discount experiment is submitted as
`scale-bootstrap-longhorizon1`, app `ap-Fz6qYSCZzLcnXDR4nuMOcC`, call
`fc-01M56ZRHTQW3R5MV7AKMY4VJTN`. It matches `scale-bootstrap-native2`'s parent,
seed, pool, architecture, rollout, optimizer settings, opponent mixture and
critic/Adam reset, but uses gamma 0.999 instead of 0.995. No shaped reward or
imitation updates occur during either PPO run. Both are explicitly initialized
from the imitation-trained bootstrap. See H7 in the repair plan for the
hypothesis and unchanged safety-limit boundary. The bounded read-only watcher
is local exec session 20230, log `checkpoint-watch-longhorizon1.log`, status
`watch-longhorizon1-status.json`. It watches only this new run.

Native-bootstrap update 4 completed its eight-game teacher probe at **4 wins /
4 losses**, with no truncations or cutoffs. The exact final synchronous native
pilot (update 33) lost all eight teacher games, confirming its earlier result.
Current ledger commitment is **$169.04**, including $140 reserved for four
long jobs. This is a budget bound plus terminal margins, not actual spend.

The pure update-32 expanded evaluation is now complete: the extra 24 games
scored 1 win / 23 losses. Combining disjoint seeds with the original eight
produces **5 wins / 27 losses / 0 truncations / 0 cutoffs across all 32
development games**, native win fraction 15.625%, Wilson 95% interval
[6.86%, 31.75%]. Mean maximum owned cities was 4.6875. Before combination,
checkpoint, opponent, teacher hash, deterministic mode, sampling seed and
evaluator source hashes were matched; all 32 full-trace hashes were verified,
and exactly both seats on each seed 300000–300015 were required. Combined
artifact: `async32-teacher-full-dev/{summary,games}.json`. This supersedes any
impression of 50% broad strength from the eight-game probe.

The pure long continuation remains an experiment with substantial remaining
learning, not an accepted bot. To avoid the same small-probe selection error,
the bootstrapped potential update-16 checkpoint (5/8 on the quick probe) now
has its own 24-game expansion on the same extra seeds at
`bootstrap-potential16-teacher-expanded-dev/`. The initial eight-game probe is
retained separately for a later provenance-checked combination. Final holdout
maps are untouched.

H7 initialization has been independently checked from remote artifacts:
`discount-ablation-initialization.json`. The two native-reward branches differ
only in gamma within Config (0.995 versus 0.999), and their initial model plus
optimizer NPZ hashes match exactly (`7cacea6d0ec4...`). Parent hash, training
map seeds, opponent mixture, collector and critic/Adam reset flags also match.

Latest live-state check confirms all four long apps are detached with one task
each. Native-bootstrap reached update 14 (114,688 decisions); shaped-bootstrap
update 39 (319,488); pure continuation update 5 (40,960 new decisions); long-horizon
native update 1 (8,192). All reported zero illegal actions and zero uncredited
endings. The long-horizon critic is still exactly zero after its first batch,
which had no native rewards, consistent with the explicit zero-head reset;
do not call its value model calibrated yet. The first pure continuation
checkpoint is retained and its quick teacher probe has begun.

The exact final synchronous potential pilot (update 35) now scores **2 wins /
6 losses** on eight teacher games, improving on its update-32 0/8 result.
Its random-opponent 32/32 deterministic score still substantially overstates
teacher strength. Full held-out acceptance and full-budget optimization remain
unfinished. Expanded bootstrapped-potential evaluation is active in local exec
session 34407. No final-test seeds have been used.

## Broader checkpoint qualification and entropy ablation

The bootstrapped shaped-PPO update-16 checkpoint completed the additional
24 teacher development games with 11 wins / 13 losses. Its combined full set
is **16 wins / 16 losses / 0 truncations / 0 cutoffs**, 50% native wins,
Wilson 95% interval [33.63%, 66.37%], over both seats on seeds 300000–300015.
Mean maximum owned cities is 6.46875. The combination checks identical model,
configuration, opponent, teacher, inference mode, sampling seed and evaluator
source hashes, rejects overlap/missing seats, and verifies every full-trace hash.
Artifact: `bootstrap-potential16-teacher-full-dev/`. This is a development
milestone, not a passed final gate or proof that PPO beats its imitation parent.

Independent checkpoint verification passed 37 archived source hashes, four
current inference hashes, both checkpoint hashes, finite tensors and 12 legal
host decisions. The selected checkpoint has 788 Adam steps and policy delta
L2 2.2512 from its reset-critic initial model. Its hash is
`02c9fc2afc1fa6b30d9da6ec0e96ae2b29893bf5afeb168a88a07e80509a2c00`.

The imitation-only parent is being extended to the same 32 teacher games by
adding seeds 300008–300015 at `bootstrap1-teacher-dev-tail/` (exec 75070).
The candidate's deterministic random-opponent benchmark is running at
`bootstrap-potential16-random-full-dev/` (exec 75898), 32 games, development
seeds only. A stochastic teacher benchmark on the same 32-game development set
is also running at `bootstrap-potential16-teacher-stochastic-dev/`. These are
public-API evaluations with full compressed traces, not GPU teacher proxies.

The early-gradient diagnostic found that native-bootstrap update 1 had zero
native rewards, zero raw advantage variance, zero policy/value loss, and only
an entropy gradient; its mean reported approximate KL was 0.01993. The gamma
0.999 branch had the same first-batch behavior. The potential branch had
nonzero advantages despite no native ending. Evidence is saved as
`early-entropy-diagnostic.json`; it motivates H8 but does not prove that entropy
caused later gameplay regressions.

H8 is now live as `scale-bootstrap-noentropy1`, app
`ap-CkkwRJszrI3VzASuhe3aqH`, call `fc-01M5704YV78P4DAH52P3GJDGRF`.
It matches the native-bootstrap control's parent, seed, map pool, gamma,
learning rate, opponent mix, critic/Adam reset and all other Config fields;
only entropy coefficient changes from 0.005 to zero. Remote initial NPZ hashes
match exactly. Evidence: `entropy-ablation-initialization.json`. Its first
8,192-decision batch reports zero total gradient and numerical-zero KL, as
expected without rewards or an entropy bonus. Training actions remain sampled.
The 11-hour requested / 12-hour hard-capped job reserves $35. Its read-only
13-hour watcher is local exec 5545, startup PID 47928,
`checkpoint-watch-noentropy1.log`, `watch-noentropy1-status.json`.

The ledger now commits **$204.04**, including five $35 long-run reservations,
not actual invoiced spend. All five long runs remain experiments under the
full $500 optimization objective. The final-test maps remain untouched.

The asynchronous pure-RL pilot finished training at update 48 (393,216 learner
decisions) and its quick teacher probe scored 2 wins / 5 losses / 1 truncation.
That eight-game probe does not settle whether it is stronger over all maps than
update 32 (5/32 on the complete development set). Its stochastic GPU random
probe reported 30 wins / 2 truncations; the durable function was still live
while the deterministic probe ran. The synchronous mixed-opponent pilot's call
has completed: final update 21, random stochastic 4 wins / 28 truncations,
deterministic 29 wins / 3 truncations, no illegal moves or evaluator cutoffs.
Await fresh stopped/zero-task app receipts before releasing either reservation.

Added `docs/structured-rl-current.md` with current architecture, PPO/reward
semantics, pure-versus-imitation lineage, actual checkpoint paths, public-API
usage, evaluation commands, reserved partitions and budget/provenance workflow.
README now points there; the original v1 page is explicitly historical. The
pure-control usage example was executed successfully, and the stronger candidate
also passed its 12-action host verification. Existing LLM work remains intact.

The matched imitation baseline is now complete over the same 32 teacher games:
**13 wins / 15 losses / 4 engine truncations**, versus shaped PPO update 16's
**16 wins / 16 losses / 0 truncations**. The PPO checkpoint converted five
baseline losses and two baseline truncations into wins, but lost four former
wins; two other former truncations became losses. Net native win fraction is
+9.375 percentage points. A paired bootstrap over maps (keeping both seats
together; 100,000 resamples, fixed seed) gives a broad 95% interval of
[-9.375, +28.125] points. This sample does not establish a reliable gain.
Artifacts: `bootstrap1-teacher-full-dev/` and
`bootstrap-vs-potential16-paired-dev.json`, including exact changed-game traces.

The same PPO update-16 candidate also completed the public-API deterministic
random benchmark at **32 wins / 0 losses / 0 truncations / 0 cutoffs** on the
32 development games. Thus it reaches the predeclared minimum win-rate levels
on development data, but final 64-game-per-opponent acceptance is still
unperformed, stochastic teacher mode is still under comparison, and the full
budget optimization objective is unfinished. Its current inference path is
now the main example in the current guide; the pure-RL control remains clearly
labeled and available separately.

Fresh Modal receipts confirm `repair-mixed-potential1` stopped with zero tasks.
Its reservation was reconciled to $3.15 including margin, bringing commitment
to **$202.19**. The asynchronous pilot was still a live task at that receipt
and retains its full reservation. The older asynchronous pilot's archived
runner sets `summary.complete` before both evaluations finish; it is not a
terminal-job signal. Use the durable function result and app receipt. Current
runner source sets that flag only after both modes have completed.

H9's optional random-opponent curriculum is implemented in the training runner
and both bounded Modal launchers. A separate compiled collector uses the
existing family-uniform random sampler; public and random probabilities are
disjoint and sum to at most one, with the remaining mass split between current
and historical policies. The default zero-random path preserves prior RNG
consumption and selection. Per-rollout opponent flags and exact probabilities
are recorded; no isolated full-game opponent win rate is inferred from them.
No engine or inference-network files changed.

Six new tests exercise real native-map random-opponent collection in both
synchronous and independent-lane modes, including waiting for the opponent,
legal actions, no resignations, learner-only behavior-logp replay, and rejection
of invalid mixtures before creating a run. With reward and numerical replay
regressions, **16 tests pass**. Launcher/runner compilation passes and all
22 frozen simulator hashes match. An initial test assertion incorrectly treated
the synchronous collector's per-step counters as scalars; the test now reduces
counters exactly as the trainer does. No collector defect was found there.

The bounded pilot is launched as `repair-pure-easyopponent1`, app
`ap-Z2qtOkjvBkowS5iD56K7mX`, call `fc-01M570RACRK0PV0PGYYZ71FCRF`.
It preserves the immutable pure update-32 model/optimizer and uses seed
20261021, 512 maps, 128x64 rollout, potential reward, no critic reset,
25% public / 25% random / 25% current / 25% historical rollout allocation.
It requests 128 updates within 2,400 training seconds and has a one-hour hard
limit. Five long runs remain unchanged. A read-only two-hour watcher is live
in local exec 12100, log `checkpoint-watch-easyopponent1.log`, status
`watch-easyopponent1-status.json`.

The asynchronous pure pilot has now returned both random evaluation modes:
30 wins / 2 truncations stochastically, 32/32 wins deterministically, no illegal
actions or evaluator cutoffs. A fresh stopped/zero-task Modal receipt reconciles
it to $3.15 including margin. After the new $5 H9 pilot reservation, commitment
is **$205.34**, including live allocations and terminal-job margins, not an
invoice. The user objective and untouched final acceptance remain open.

H9's remote initialization is verified in
`easy-opponent-ablation-initialization.json`: initial NPZ/Adam bytes match the
pure control exactly (`aa8505267dcd...`), as do training seeds, parent, critic
reset flag and learning hyperparameters. Config differences are the short-run
update/time limits and GPU evaluation action cap; local checkpoint comparisons
use the same public development evaluator. The experimental training change
is the 0.25 random-opponent allocation. A fresh app list shows all five long
runs plus the H9 pilot detached with one live task each.

A useful next resource-efficiency investigation is larger learner batches:
current long jobs use only 128 lanes despite a 519k-parameter model. Benchmark
256/512 lanes under short fixed allocations before reserving long runs, recording
actual learner decisions/second, GPU memory and compilation cost. This is only
a hypothesis: the frozen simulator's kernel/memory costs may prevent scaling,
and changing lane count also changes PPO batch size. No larger-lane run has
been launched or throughput gain claimed yet. Preserve the engine contract
and compare matched horizon/optimizer settings if pursuing this test.

The stochastic teacher evaluation of shaped-PPO update 16 is complete:
**15 wins / 17 losses / 0 truncations / 0 cutoffs** on all 32 development games,
versus deterministic 16/32. This close single-seed comparison does not establish
an inference-mode advantage. Deterministic remains provisional; final mode is
not frozen. Report: `bootstrap-potential16-teacher-stochastic-dev/summary.json`.

H9 has completed both the league and random collector paths on GPU. At update 3,
its totals were 4 native wins / 4 losses, with zero illegal actions,
resignations, uncredited endings or truncations. Updates 2 and 3 selected the
random opponent. This verifies the implemented mixture and provides positive
outcome samples; it is not yet evidence of improved evaluation strength.

For H10, uploaded and readback-verified immutable parent
`bootstrap-potential16-02c9fc2afc1f/final.npz`, SHA
`02c9fc2afc1fa6b30d9da6ec0e96ae2b29893bf5afeb168a88a07e80509a2c00`.
Its lineage includes the full 32-game teacher/random benchmarks and local
source/checkpoint/host-interface verification. Local bundle:
`qualified-parents/bootstrap-potential16-02c9fc2afc1f/`.

The short launcher now allows 256/512 lanes only for full-map PPO; the long
launcher exposes the same lane options, retaining the default 128. Config
validation and compilation checks pass. New training rows record GPU
allocated/peak/limit bytes where available, and include gradient-diagnostic
cost in per-update timing. Total training wall time remains separately recorded,
including checkpoint writes. This telemetry does not alter policy learning.

The 128-lane eight-update control is submitted as `throughput-128-v1`, app
`ap-EikJ4sODXeq2UgijZCShW5`, call `fc-01M5710TG4NZKXAWJ42CGN7YQT`.
The 256-lane comparison is being submitted from the same parent and settings;
its durable handle will be recorded below. Each uses its own bounded single-GPU
worker; existing training jobs remain unchanged.

The gamma-0.999 branch's update-16 quick probe scored 6 wins / 2 losses.
An additional 24-game teacher development evaluation on seeds 300004–300015
is running at `bootstrap-longhorizon16-teacher-expanded-dev/` (local exec 54323).
Do not extrapolate its quick score before that expansion completes.

All H10 throughput trials are durably submitted:

- 128 lanes: `throughput-128-v1`, app `ap-EikJ4sODXeq2UgijZCShW5`, call
  `fc-01M5710TG4NZKXAWJ42CGN7YQT`.
- 256 lanes: `throughput-256-v1`, app `ap-XAiAzoqHFyiL2g6E9WvKbG`, call
  `fc-01M5712N108NR34N2BCZHB1T5A`.
- 512 lanes: `throughput-512-v1`, app `ap-qEDgfw2hzAIvPZouVrulNI`, call
  `fc-01M57130EY2YGDY3KB5HKY0E17`.

The common immutable parent, initial model/Adam bytes, complete source archive,
training map seed pool, opponent probabilities and hyperparameters match; the
only Config difference is lane count. Evidence:
`throughput-initialization.json`. Eight updates collect different total numbers
of learner decisions (65,536 / 131,072 / 262,144), and larger PPO permutations
can change later opponent RNG draws. Stratify by recorded opponent flags and
report actual decisions/optimizer steps; do not call these equal-sample trials.

A dedicated read-only two-hour watcher covers only the three throughput runs:
local exec 57551, log `checkpoint-watch-throughput1.log`, status
`watch-throughput1-status.json`. No long run has been switched to larger lanes.
The 128-lane control has reached update 2, with valid replay and measured peak
GPU allocation 10,956,291,584 bytes against allocator limit 35,773,218,816 bytes.
Its second batch collection took 33.60 seconds, full update 34.76 seconds.
The larger trials are still warming up; these first numbers do not prove a
scaling benefit. Commitment after all three $5 reservations is **$220.34**,
including live reservations and terminal-job margins, not settled spend.

The gamma-0.999 native-reward update-16 candidate completed all 32 teacher
development games at **22 wins / 10 losses / 0 truncations / 0 cutoffs**
(68.75%, Wilson 95% interval [51.43%, 82.05%]). All source-evaluation settings
matched and all 32 full-trace hashes were verified before combination.
Mean maximum owned cities was 7.40625. Combined artifact:
`bootstrap-longhorizon16-teacher-full-dev/`; model hash
`a7a7fa1bfa12dbfc1ca74d8ed1669867d9821fe15b3fca5a98a968c1220bb208`.

Versus its imitation parent, it converted ten losses and two truncations into
wins while losing three former wins: net +9/32, or +28.125 percentage points.
A paired map bootstrap gives [12.5, 40.625] points on this development sample.
Versus shaped-PPO update 16, the point difference is +18.75 points but the
paired interval spans zero [-6.25, 40.625]. These are development comparisons
used in checkpoint selection, not unbiased final-test claims.
Reports: `bootstrap1-teacher-full-dev-vs-longhorizon16-paired-dev.json` and
`bootstrap-potential16-teacher-full-dev-vs-longhorizon16-paired-dev.json`.

The leading candidate also completed the public-API random development benchmark
at **32/32 wins**, without truncations/cutoffs, and passed 38 archived source
hashes, four inference hashes, both checkpoint hashes, finite tensors and 12
legal host decisions. Its matched gamma-0.995 native update-16 control is being
expanded with 24 additional teacher games at
`bootstrap-native16-teacher-expanded-dev/` (exec 22486). That comparison is
needed to isolate the discount change from the broader effect of PPO.
Final acceptance remains untouched and the full-budget objective remains open.

H10's 512-lane trial has passed early GPU replay/legality checks, with peak
allocation 24,827,511,808 bytes versus allocator limit 35,773,218,816 bytes.
The 256-lane trial also passed its first batch; peak allocation was
21,747,591,936 bytes. Do not conclude that throughput scales linearly: early
collection times and game workloads vary materially across shapes and updates.

Actual optimizer counters were extracted from hash-verified NPZ files:
128 lanes completed eight updates with 499 Adam steps beyond the parent
(maximum 512); 512 lanes completed its first four updates with 723 steps
(maximum 1,024). Thus the larger batches do trigger more early stopping, but
are not mostly discarded in this observed window: 1.412 optimizer sample
presentations per collected decision, versus 1.949 for the 128-lane trial.
These are presentations, not counts of distinct samples seen. Evidence:
`throughput-optimizer-progress.json`. Final trial timing/counters are still
pending for the larger configurations.

Prepared a verified immutable continuation parent from the leading native PPO
checkpoint: `bootstrap-longhorizon16-a7a7fa1bfa12/final.npz`, with metadata and
lineage under `qualified-parents/bootstrap-longhorizon16-a7a7fa1bfa12/`.
Remote readback matches its full recorded hash. The lineage includes both
complete 32-game public benchmarks and the checkpoint verification (38 archived
sources, four inference hashes, two checkpoint hashes, 12 legal decisions,
535 Adam steps and policy delta L2 1.7551). This prepares a concrete parent for
scaling after the throughput qualifications; it does not itself start another
long run. All 22 frozen engine hashes still match.

The 128-lane throughput control completed training/evaluation successfully:
65,536 new learner decisions, 499 new Adam steps, 32/32 random wins in both
stochastic and deterministic modes, no illegal moves/truncations/cutoffs.
The 256- and 512-lane trials continue with their original handles.

A full-trace ending audit confirms that all **22 teacher wins** of the leading
native-PPO checkpoint ended in the learned policy's own capture-family action.
None came from opponent resignation. Evidence: `longhorizon16-win-endings-audit.json`.
The current inference guide now points to that verified immutable parent, while
retaining the separately labeled pure-RL control.

Fresh application receipts confirm the 128-lane throughput control stopped with
zero tasks. It was reconciled to $1.53 including margin. Current commitment is
**$216.87**. All five long jobs, the curriculum pilot and the two larger-lane
qualifications were confirmed live with their original IDs. No failed poll or
warming-up job was resubmitted.


H10 completed: 128/256/512 lanes achieved 159.7 / 143.7 / 292.3 learner
decisions per training second including compilation/checkpoint writes. Excluding
the first update, measured per-update rates were 191.5 / 153.8 / 306.4.
The 512-lane trial took 1,549 actual Adam steps (1.513 optimizer sample
presentations per decision); all batches had valid log-probability replay,
zero illegal actions, zero uncredited endings, and zero engine truncations.
All three final checkpoints won 32/32 random games in both inference modes.
These are observed workloads, not equal-sample strength comparisons; the
512-lane opponent draw schedule differed. Qualification report:
`runs/structured-ppo-v3/throughput-qualification.json`.
The 512 checkpoint also passed 38 archived source hashes, four inference hashes,
both checkpoint hashes, finite tensors and 12 legal public-API decisions.

The matched native discount control finished its full teacher development set:
gamma .995 update 16 had 19 wins / 9 losses / 4 truncations, versus gamma .999
at 22 / 10 / 0. Archived training source hashes are identical. The paired
map-bootstrap win-rate difference is +9.375 points with interval
[-9.375, 28.125], so a causal discount advantage is not established.
`native995-vs-native999-update16-paired-dev.json` retains the comparison.
Later native-.995 update 48 scored 8/8 on the small teacher probe; gamma-.999
update 48 scored 7/8. Both are now being expanded over the other 24 development
games (local execs 52661 and 8456). Final maps remain untouched.

Started `scale-native512-v1` from the verified gamma-.999 update-16 parent,
with 512 lanes, native reward, gamma .999, seed 20261024, 512 training maps,
and the existing 25% public / remaining current-historical mixture. Actor,
critic and Adam are preserved. Requested training is 11 hours, hard function
cap 12 hours, $35 reservation. App `ap-HgZB9VP0HnU6AW5uxJgVLK`, durable call
`fc-01M5728WFCYNH498WN144JK8CM`. The CLI's final-log timeout was only an
observation timeout; the job was not resubmitted. Dedicated read-only watcher:
local exec 22136, `checkpoint-watch-native512v1.log`, watch name `native512v1`.

Fresh stopped/zero-task receipts reconciled the 256 and 512 throughput jobs to
$1.89 and $1.87 including margins. Commitment after the new long reservation
is $245.63, not settled spend. All 22 frozen engine hashes still match.


Gamma-.999 update 48 completed the full teacher development benchmark at
**29 wins / 3 losses / 0 truncations / 0 cutoffs** (90.625%, Wilson 95%
[75.78%, 96.76%]), mean maximum cities 8.46875. Its hash is
`c28c58ee53908170d4c597127821ff88c8f33f55b2f87efdd870326e98e72db8`.
All 32 trace hashes were verified; all 29 wins ended in learned-policy capture
commands. Checkpoint validation passed 38 archived sources, four inference
sources, both model hashes, finite tensors and 12 legal host actions; Adam
step 2,583, policy delta L2 3.5795. Relative to update 16, the paired map
bootstrap improvement is +21.875 percentage points with interval [9.375, 37.5].
These remain development-selection measurements, not final acceptance.
Artifacts: `bootstrap-longhorizon48-teacher-full-dev/`,
`longhorizon48-win-endings-audit.json`, `longhorizon16-vs48-paired-dev.json`.
Random development and stochastic teacher evaluation for update 48 are running
(local execs 9231 and 91086); update-16 stochastic comparison also continues
(exec 99168). The 512-lane throughput checkpoint scored 6/8 teacher probe wins;
its remaining 24 teacher games are running (exec 27318).

Added `scripts/combine_structured_evaluations.py` to verify protocol consistency,
unique paired-seat coverage and every trace hash before combining deterministic
development partitions. It excludes final maps and stochastic split runs, whose
sampling streams restart at each partial evaluation. Five tests passed,
covering unresolved outcomes, duplicate/missing games, protocol drift and
changed trace bytes. This utility changes no training or simulator code.


The gamma-.995 update-48 control also completed **29/32 teacher wins**, three
losses, zero truncations/cutoffs, mean maximum cities 8.1875. It passed the same
38-source / four-inference / two-checkpoint / finite / 12-legal-action checks;
Adam step 2,335, policy delta L2 3.2825. Hash:
`eb14a21f16f6e5aea884b74b29619d6ac1051b57e0fa34c8583b6cf2771f180a`.
Combined report: `bootstrap-native48-teacher-full-dev/`. Equal point scores do
not establish equal policies or a discount winner; the losses occur on different
games. Before executing either replication, the plan declared development maps
300100–300115 (both seats) to reduce reliance on the repeatedly inspected original
16 maps. Both update-48 candidates are now evaluated there, deterministic mode:
`bootstrap-longhorizon48-teacher-replication-dev/` (exec 92280) and
`bootstrap-native48-teacher-replication-dev/` (exec 16166). These are additional
development games, not the untouched final acceptance partition.

`scale-native512-v1` has reached its first update with zero illegal moves,
zero uncredited endings, zero truncations and peak GPU allocation 24,827,512,064
bytes. Its initial checkpoint is byte-identical to the qualified gamma-.999
update-16 parent, confirming critic and Adam preservation. Evidence:
`native512-initialization.json`. Its durable call is still pending/running;
no observation timeout led to a restart. Current ledger commitment remains
$245.63; full-budget optimization and final acceptance are still outstanding.


The gamma-.999 update-48 candidate completed its random public-API development
benchmark at **32/32 native wins**, no truncations/cutoffs. Uploaded an immutable
continuation bundle `bootstrap-longhorizon48-c28c58ee5390/final.npz` with metadata
and lineage, verifying remote readback SHA-256. The current guide now uses this
29/32-teacher, 32/32-random candidate by default. Fresh-map and stochastic
comparisons are still running; final acceptance is not claimed.

H11 map-diversity qualification launched two bounded eight-update PPO trials
from that same preserved actor/critic/Adam: 512 lanes, horizon 64, gamma .999,
native reward, seed 20261025, 25% public opponent, no random opponent. Only
training pool size differs: 512 versus 2,048 maps. Both reserve $5, request at
most 2,400 training seconds, and have a one-hour hard cap. Run IDs/handles:
- `diversity-512-v1`: app `ap-cCikHRXWR2L9V1ZaQ3GWsk`,
  call `fc-01M572KWKC50CTVXDX4WHBCE08`.
- `diversity-2048-v1`: app `ap-i3dNyQFn7yj3ArzeBdM9rG`,
  call `fc-01M572KXT1N9WAPX6RGFQ4GSJ5`.

The short launcher now explicitly accepts gamma (default unchanged at .995)
and allows pool 2,048 only for full-map PPO. Compilation and both Config
allocations validated. This changes launcher configuration only, not PPO or
engine code. Dedicated read-only watcher `diversity1` retains checkpoints and
runs the usual eight-game teacher probe. The two final-log observation timeouts
did not cause any resubmission. Ledger commitment is now $255.63, not spend.


Fresh development replication completed on maps 300100–300115, both seats:
- Native gamma .995 update 48: **27 wins / 5 losses / 0 truncations / 0 cutoffs**,
  Wilson 95% [68.25%, 93.14%].
- Native gamma .999 update 48: **26 wins / 6 losses / 0 truncations / 0 cutoffs**,
  Wilson 95% [64.69%, 91.11%].

All 64 trace hashes were verified and every victory ended in a learned-policy
capture, with no opponent resignation. The .999-minus-.995 paired map-bootstrap
difference is -3.125 points, interval [-15.625, 9.375], so the one-game difference
does not establish a discount winner. Report:
`native995-vs-native999-update48-replication-dev.json`. These two checkpoints
were selected before this replication ran; this partition is now development
data and does not replace final acceptance. The gamma-.995 checkpoint's random
benchmark is running separately (exec 44804).

The gamma-.999 update-48 stochastic teacher benchmark completed **25/32 wins**
versus deterministic 29/32 on the original development set, with no unresolved
games. Its update-16 predecessor scored 18/32 stochastic versus 22/32
deterministic. The default example remains deterministic; no claim of a proven
population advantage from these small comparisons is made.

The earlier 512-lane shaped-PPO throughput trial finished its expanded teacher
benchmark at **17/32 wins**, 15 losses, zero unresolved games, versus its parent's
16/32. This qualifies throughput and healthy learning execution, not a clear
strength improvement. Full report: `throughput512-final-teacher-full-dev/`.
The combining utility now explicitly rejects random-opponent partitions as well
as stochastic learned-policy partitions because either would restart a sampling
stream; six tests pass. Existing teacher merges are unaffected.

H11 initialization was read back and verified: both new map-diversity trials
have identical initial actor/critic/Adam hash c28c58ee5390..., identical archived
training sources and Config differences only in pool size. The 512-map list is
an exact prefix of the 2,048-map list. Both completed their first batch; peak GPU
allocations were 24,827,512,320 and 24,961,136,896 bytes respectively. The larger
pool adds approximately 134 MB in this measurement. Evidence:
`diversity-initialization.json`. Dedicated watcher local exec 42027 remains
responsible for both runs. These trials and the six long jobs continue;
full-budget optimization and final acceptance remain outstanding.


H12 lower-learning-rate qualification launched as `conservative-lr-v1`, app
`ap-Gax4ye2Ef2v6rrcBdrwgly`, durable call `fc-01M5736304EMV91EYJH91SVVFD`.
It matches H11's 512-map control except learning rate 3e-5 versus 1e-4: same
qualified gamma-.999 update-48 parent, seed 20261025, 512 lanes, horizon 64,
eight updates, native rewards, gamma .999 and opponent mixture. It preserves
critic/Adam, reserves $5, requests at most 2,400 training seconds, and has the
same one-hour hard cap. The launcher exposes/records the effective learning
rate while preserving previous PPO/BC defaults. Compilation and Config checks
passed. Dedicated watcher `conservative1`, local exec 11002. Commitment is
$260.63, not settled spend. Fresh Modal app-list receipts confirmed all ten
unreconciled owned jobs live with one task each; no jobs were resubmitted.

Native gamma-.995 update 96 completed **32/32 teacher development wins**, no
truncations/cutoffs, Wilson 95% [89.28%, 100%], mean maximum cities 9.1875.
Its checkpoint hash is
`04a811bda8e43511bcfe261a9e7faa911dcc18c76acf3dfd9df9c4d20a285faa`.
All 32 trace hashes and matching source protocols were verified before merging.
Checkpoint checks passed 38 archived sources, four inference sources, both model
hashes, finite tensors and 12 legal host actions; Adam 5,407, policy delta
L2 4.7650. Artifact: `bootstrap-native96-teacher-full-dev/`.

The zero-entropy native update-64 branch completed **30/32 teacher wins**, two
losses, no truncations/cutoffs. It passed the same integrity and public-action
checks; Adam 3,396, policy delta L2 3.8754. Artifact:
`bootstrap-noentropy64-teacher-full-dev/`. These are different training sample
counts, so their comparison does not isolate the entropy coefficient.

Added `scripts/evaluate_structured_match.py` for public-API paired-seat games
between two frozen learned policies, with full compressed traces and checkpoint
hashes. This leaves the existing teacher evaluator unchanged. The real-simulator
self-match validation passed: the gamma-.999 update-48 checkpoint played exactly
the same 987-action/observation sequence in both seats, all policy labels swapped,
all challenger rewards reversed, and the result was one win / one loss, no
truncations. Both trace hashes were verified. Evidence:
`match-self-symmetry-dev/symmetry-verification.json`.

The native update-96 candidate is now being evaluated against gamma-.999
update 48 on separate development maps 300200–300215 (exec 94965), against the
teacher on replication maps 300100–300115 (exec 33572), and against random on
the original 16 maps (exec 44382). Final acceptance seeds remain untouched.
The earlier native update-48 candidate also completed 32/32 random wins without
unresolved outcomes. The fresh-map/teacher/current guide still names the fully
qualified gamma-.999 update-48 candidate while the newer checks finish.


H12 initialization was read back and verified against H11's 512-map control:
identical initial model/Adam hash, identical archived training source hashes,
and the only Config difference is learning rate 3e-5 versus 1e-4. Evidence:
`conservative-lr-initialization.json`. It has passed its first two batches with
valid replay and no illegal moves/uncredited endings. H11's 512-map control
has finished all eight training updates; the 2,048-map branch has reached
update seven. Neither shows GPU-memory or replay problems. Completed-rollout
training outcome counts are not isolated fixed-opponent evaluation win rates:
ongoing games, different lengths and changing opponents can bias that readout.

The native update-96 checkpoint also completed **32/32 public-API random wins**,
no truncations/cutoffs. Its fresh-map teacher and learned-opponent tests continue.
The zero-entropy update-64 checkpoint is undergoing fresh-map teacher and random
evaluation as well (execs 2522 and 7286). To check opponent-specific strengths,
the plan now declares all three pairwise deterministic matches among native
update 96, gamma-.999 update 48 and zero-entropy update 64, on development maps
300200–300215 with both seats. Added no-entropy-vs-native (exec 20256) and
no-entropy-vs-gamma-.999 (exec 98960). Exact traces/checkpoint identities are
retained under `match-noentropy64-vs-native96-dev/` and
`match-noentropy64-vs-longhorizon48-dev/`; no final-test maps are used.


Both H11 trials finished all eight requested training updates, so they collected
matched 262,144 learner decisions. Reported training wall times were 798.18 s
for 512 maps and 880.28 s for 2,048 maps. Their random evaluations are still
finishing. The 512-map trial's stochastic random evaluation is already 32/32;
its final teacher expansion is running locally (exec 57289). Its retained final
checkpoint is `diversity-512-v1/update-8-c85c3b9a5c26`.

The pure easy-opponent pilot finished 46 updates in 2,456.59 training seconds
(the soft cap is checked after an update). It has completed stochastic random
evaluation at 30 wins / 2 engine truncations, no illegal actions or evaluator
cutoffs. Its deterministic evaluation is pending; do not mark the app complete
or release its reservation until the durable call/app receipt is terminal.


The native-.995 update-96 candidate completed **30/32 teacher wins** on the
additional development maps 300100–300115, two losses, zero unresolved games.
Together with the original set this is 62/64 teacher wins; these remain selected
development scores, not final acceptance. All 96 teacher/random trace hashes
were verified, and every one of the 62 teacher victories ended in a learned
capture action. The model also completed its learned-opponent comparison at
**20 wins / 12 losses** versus gamma-.999 update 48, no truncations/cutoffs,
Wilson 95% [45.25%, 77.07%]. All 32 match traces and paired-seat coverage were
verified. This small match is supporting evidence, not universal superiority.

Prepared and uploaded `bootstrap-native96-04a811bda8e4` as an immutable parent
bundle, including model, metadata and lineage with both teacher partitions,
random benchmark, learned-opponent match and verification. Every uploaded file
was read back and hash-verified. The current usage guide now names this parent.

H13 launched matched long continuations from it with seed 20261026, 2,048 native
training maps, gamma .995, native reward, learning rate 1e-4 and the existing
opponent mixture; only lane count differs. Both preserve actor/critic/Adam,
request 11 training hours, reserve $35 and have a 12-hour hard function cap:
- `scale-champion128-v1`: app `ap-0jRkxCKRSt0A6omNSstL9X`,
  call `fc-01M573SKJ2M094ZPFRRPZMNBJG`.
- `scale-champion512-v1`: app `ap-RnKnvLUV0woXfNh02RQB8f`,
  call `fc-01M573SMPDYRFSXD4YZXQRWR8V`.
The final-log observation timeouts caused no retries. Dedicated read-only watcher
`champion1` covers these two runs. This comparison tests strength per wall time;
its decision counts are deliberately not matched.

H11's 2,048-map run passed all eight batches (262,144 decisions): maximum replay
error 4.7684e-6, peak GPU allocation 24,961,136,896 bytes, no illegal decisions,
resignations, uncredited endings or collector exhaustion. Execution evidence:
`diversity2048-execution-qualification.json`. Its teacher expansion is running
(exec 90283), so broader maps are capacity-qualified but not proven stronger.

Fresh stopped/zero-task receipts reconciled the easy-opponent pilot to $3.20
and the 512-map trial to $1.78 including margins. The pure easy-opponent pilot's
complete random evaluation was 30 wins / 2 engine truncations stochastically and
23 wins / 9 truncations deterministically, with no illegal decisions/cutoffs.
It has not demonstrated a teacher-strength improvement and is not being scaled.
Commitment after the two new long allocations is **$325.61**, not settled spend.

The zero-entropy update-64 candidate also completed **29/32 teacher wins** on
the additional development maps and 32/32 random wins, without unresolved games.
Its learned-opponent portfolio matches continue with their original handles.


The three-model learned-opponent portfolio completed all 96 games without
truncations or evaluator cutoffs. Native-.995 update 96 beat each rival 20–12;
gamma-.999 update 48 and zero-entropy update 64 tied 16–16. Aggregate scores:
40/64 wins for native-.995 update 96, 28/64 for each rival. All 96 trace hashes,
checkpoint roles, map/seat coverage and source consistency were verified.
Map-cluster bootstrap intervals (preserving all seats/opponents) are
[51.56%, 73.44%] for the leading model, [34.38%, 54.69%] for gamma-.999 and
[32.81%, 53.13%] for zero-entropy. The leading model's +18.75-point aggregate
score difference versus either rival has interval [0, 37.5], so this remains a
small development portfolio, not a universal strength proof. Full evidence:
`learned-portfolio-v1.json`. The current guide includes the two match results.

H13 initialization verification passed for both long runs: exact restoration
of the champion model and Adam, identical archived learning sources, identical
training seed list, and the only Config difference is environments 128 versus
512. Evidence: `champion-scaling-initialization.json`. Their dedicated watcher
is local exec 82059. Both durable calls were confirmed pending/running; no
observation timeout caused a restart. All 22 frozen engine hashes still match.

H11's 2,048-map trial completed its random evaluation with 32/32 stochastic wins
and 31 wins / one engine truncation deterministically, no illegal actions or
evaluator cutoffs. A fresh stopped/zero-task receipt reconciled its reservation
to $1.90 including margin. Current ledger commitment is **$322.51**, not settled
spend. Both H11 teacher expansions continue. Their final models are also being
matched directly against their common parent on development maps 300200–300215:
`match-diversity512-vs-parent-dev/` (exec 15597) and
`match-diversity2048-vs-parent-dev/` (exec 99766). This distinguishes teacher
specialization from improvement against a learned opponent before selecting
future allocations. No final-test maps have been used.


H11 full teacher results are **26/32** for 512 maps and **22/32** for 2,048 maps,
versus 29/32 for their parent. Both had zero truncations/cutoffs; all 64 trace
hashes and matching protocols were verified before merging. The corresponding
learned-parent matches ended **16–16** and **12–20**, again without unresolved
games. These short trials have not demonstrated strength improvement from
broader training maps. Additional development teacher evaluations are running
for both on maps 300100–300115 (execs 55456 and 95137).

Added H13's matched small-map control: `scale-champion128-smallpool-v1`, app
`ap-yGV36MaVA70QEchHcqyz4G`, durable call `fc-01M5746C8AHSZPFJ1VX8ZXMYZS`.
It starts from the same verified native-.995 update-96 parent with seed 20261026,
128 lanes, native rewards and unchanged optimizer settings, varying only pool
size 512 versus the existing 2,048-map H13 control. It reserves $35, requests
11 training hours and has a 12-hour hard cap. Dedicated watcher name
`champion_small1`; no timeout caused a retry. Commitment is now **$357.51**.
This control preserves a long comparison with the smaller map pool; the wider
pool remains an unproven strength hypothesis despite its successful execution.

H12 lower-rate update eight scored 8/8 on the teacher probe (versus H11 control
7/8). Its final retained checkpoint is `update-8-9c0eb8934d76`. The 24 remaining
teacher games and a full 32-game learned-parent match are now running. The small
probe alone does not justify claiming an improvement or selecting a new default.


The H12 lower-rate pilot completed its cloud evaluation at 32/32 random wins
in both stochastic and deterministic modes, zero illegal actions/truncations/
cutoffs. Its teacher expansion (exec 80584) and learned-parent match (exec 91797)
continue. The H13 small-pool watcher's local exec is 78971.

Hash-verified optimizer counters show that all three H11/H12 eight-update
trials took exactly **2,048 new Adam steps**, or two optimizer sample
presentations per learner decision. Thus KL early stopping did not reduce
sample reuse in this particular comparison; the earlier H10 throughput trial
had different behavior and should not be generalized to these runs. Evidence:
`diversity-conservative-optimizer-progress.json`.

All continuation trials restore model/Adam but start fresh game states and a
fresh recent-opponent bank. A short continuation therefore also has a different
episode-age distribution from its uninterrupted parent run. This is a plausible
contributor to early strength fluctuations, not an established causal diagnosis;
the long continuations and fixed-parent matches are needed before concluding
that a recipe has improved or permanently degraded strength.


H12's completed lower-rate teacher benchmark is **30/32 wins**, two losses,
zero unresolved games (Wilson 95% [79.85%, 98.27%]), versus 26/32 for its
standard-rate matched control. Its learned-parent match is **20–12**, versus
16–16 for that control. These are useful point improvements from one short
seeded experiment, not a proven general learning-rate advantage. The checkpoint
passed 38 archived source hashes, four inference hashes, both checkpoint hashes,
finite tensors and 12 legal host decisions; policy delta L2 0.8205, Adam 4,631.
Fresh-map teacher and current-champion match evaluations are running (execs
93686 and 33959). Reports: `conservative-final-teacher-full-dev/` and
`match-conservative-vs-parent-dev/`.

H11 replication teacher results were **29/32** for the 512-map control and
**22/32** for 2,048 maps. Across original plus additional development maps,
the control has 55/64 wins, the wider pool 44/64; their parent had 55/64.
No unresolved games occurred. Broader maps have not yet shown a strength gain;
the new matched long-duration small-pool control is therefore retained.

Implemented optional frozen learned rivals in `rl/opponents.py` and the PPO
runner, exposed only in the short launcher pending qualification. It validates
disjoint mixture probabilities, bounded/unique cloud pool IDs and nonempty
configured banks, verifies rival checkpoints on load, records hashes and chosen
indices, and uses the existing learned-opponent collector path. Neither engine
nor inference code changed. The zero-probability path preserves the former RNG
stream exactly. **21 tests passed**, including RNG compatibility, disjoint
selection, invalid pools/mixtures, and real native-map learner-only/replay checks.
All 22 frozen engine hashes match. Rival actions are stochastic frozen-network
actions, not teacher labels.

Uploaded and readback-verified the zero-entropy rival bundle
`bootstrap-noentropy64-15b85fdde53e`, including 96 verified teacher/random trace
hashes and checkpoint validation. The other rival is the already verified
`bootstrap-longhorizon48-c28c58ee5390` bundle.

H14 launched two matched short trials from native-.995 update 96, seed 20261027,
128 lanes, horizon 64, 512 maps, native reward, gamma .995, learning rate 1e-4,
entropy .005 and 16 requested updates. Both preserve critic/Adam, reserve $5,
request at most 2,400 training seconds and have a one-hour hard cap:
- `frozen-control-v1`: app `ap-U6TL1s2TrffnhrX5OIs2Vm`,
  call `fc-01M574KTP31E051YH2M61YJSSN`, 25% public / remaining current-history.
- `frozen-rivals-v1`: app `ap-bREDcqYUEbkHTjydZ1Xota`,
  call `fc-01M574KV2JPBCRZWJXXGHN30SV`, 25% public / 25% frozen rivals /
  remaining current-history.
Dedicated read-only watcher `frozen1`, local exec 54191. Neither final-log
observation timeout caused a retry. H14 has no long allocation yet.

A fresh stopped/zero-task receipt reconciled H12 to $1.77 including margin.
Current ledger commitment after H14 is **$364.28**, not settled spend. Final
acceptance maps remain untouched, and the full-budget objective remains open.

H12's fresh teacher evaluation completed at **27 wins / 5 losses**; its direct
match against the current native-.995 update-96 champion ended **17–15**.
Both have all 32 paired-seat games, zero truncations/cutoffs, and all 64 trace
hashes were verified (`conservative-replication-match-verification.json`).
The direct match is too close to establish superiority, and the champion retains
the better aggregate teacher score (62/64 versus H12's 57/64). No default change.

H13's small-pool initialization is verified: identical native96 weights and
archived sources, only `pool_size` differs, and the 512-map seed list is the
exact prefix of the 2,048-map list (`champion-map-pool-initialization.json`).

H14 has now exercised both frozen rivals on GPU: rival indices 1, 0, 1 at
updates 3, 6, 9. Through ten updates, both trials have zero illegal actions,
loop exhaustion, resignations, uncredited outcomes or truncations. Maximum
behavior log-probability replay error is 4.2916e-6, below the unchanged 1e-5
threshold. The control never selected a frozen rival. This qualifies execution,
not improved strength; evidence is `frozen-rivals-execution-qualification.json`
and its hash-bound training snapshots. Existing cloud jobs continue unchanged.

At 09:25 UTC, fresh Modal receipts show all nine long runs and both H14 pilots
live with one task each. `campaign-live-monitor.json` records their current
training snapshots. All have zero invalid actions, resignations, collector
exhaustion and uncredited outcomes, with replay error below 1e-5. Native-.995
has reached update 154 / 1,261,568 decisions; its probe feature standard
deviation is 0.1469. The separate pure run is at update 89 / 729,088 decisions
and remains weak on teacher probes (0/8 at updates 64 and 80). The training
outcome totals are mixed-opponent rollout statistics, not isolated win rates.

Completed paired H11/H12 analysis across both teacher development partitions:
512-map control 55/64, wider map pool 44/64, lower learning rate 57/64. With
100,000 map-cluster bootstrap replicates, the wider-minus-control difference
is -17.19 percentage points (95% [-31.25, -3.125]); lower-rate-minus-control is
+3.125 points (95% [-4.6875, 12.5]). All 192 trace hashes were checked. These
intervals cover map sampling, not training-seed variance; there is only one
training seed per recipe. This does not support spending additional funds on
the wider-pool short recipe or claiming a reliable lower-rate gain. Existing
long pool comparisons continue. Evidence:
`diversity-conservative-paired-teacher-comparison.json`.

Two newer candidates passed 38 archived-source hashes, four current inference
hashes, both checkpoint hashes, finite tensors and 12 legal host decisions:
native-.995 update 144 (`191addd60246`) and champion-512 update 12
(`06db83c193a7`). Both scored 8/8 on the small teacher probe. Broader teacher
replication evaluations now run on maps 300100–300115 (execs 47707 and 46091),
and native144 is matched against the current native96 champion on maps
300200–300215 (exec 42840). The default remains native96 pending complete
results; no new cloud allocation was made and commitment remains $364.28.

Added `scripts/verify_structured_replays.py` to reproduce saved public-API traces
without policy inference. It checks frozen sources, file hashes, every observed
state and legal command, rewards/end flags, result footer and available gameplay
statistics. Reports bind the exact input manifest bytes read at startup, even
if another process later appends games. Seven tests pass, including deliberately
corrupted states/actions/rewards/endings/footer with recomputed hashes. Two full
games reproduced exactly: native96 versus teacher, seed 300100 seat one
(946 actions), and H12 versus champion, seed 300200 seat two (1,299 actions).
This checks simulator reproducibility, not that the network selected each
recorded move. The current guide documents usage; inference code is unchanged.

Champion-512 update 12 completed teacher replication at **32/32 wins**, zero
unresolved games, mean maximum cities 9.03125. All 32 trace hashes were verified
and every winning final command was a learned-policy capture. The original
teacher expansion is running (exec 17124) and a direct match against native96
is running (exec 93878). Native144's completed match against native96 is
**15–17**, with zero unresolved games and all 32 trace hashes verified; it has
not shown head-to-head improvement. Its teacher replication continues.

H14's pilots are confirmed terminal with fresh stopped/zero-task receipts.
Both completed 16 updates / 131,072 learner decisions and achieved 32/32 random
wins in both inference modes, with no invalid actions, engine truncations or
evaluation cutoffs. Both final checkpoints passed 39 archived source hashes,
four inference hashes, checkpoint hashes, finite tensors and 12 legal host
decisions. The control took 1,024 new Adam steps; frozen rivals took 988
(2.0 versus 1.9296875 optimizer presentations per decision). Thus collection
counts match, but KL early stopping reduced optimizer work in the rival branch.
Final execution evidence: `frozen-rivals-final-execution-qualification.json`.

Both H14 final checkpoints are being evaluated on teacher maps 300100–300115
(execs 11888 and 87075). No long frozen-rival allocation has been made. Terminal
accounting is $1.75 for control and $1.76 for rivals, including margins, reducing
total commitment to **$357.79**. This remains conservative task accounting,
not settled spend. The nine existing long jobs continue; final-test maps remain
untouched and native96 remains the default pending completed comparisons.

Native144's teacher replication finished **28/32**, four losses and no unresolved
games, with all trace hashes verified. Together with its 15–17 champion match,
this does not justify replacing native96. Champion-512 update 12 finished the
remaining original-development teacher games **24/24**, joining its earlier
8/8 probe and 32/32 replication: **64/64 teacher development wins** in total.
The original-set merge verifies checkpoint/protocol equality, exact map/seat
coverage and all 32 trace hashes before publishing
`champion512-update12-teacher-full-dev/`. A deterministic 32-game random
benchmark is running (exec 5303); its champion match (exec 93878) also continues.
Perfect teacher development scores do not establish learned-opponent superiority
or final acceptance. Default and final-test partition remain unchanged.

Champion-512 update 12 completed its remaining qualification: **32/32 random
wins** and **20–12 against native96**, without unresolved games. Its 64 teacher
wins all ended in learned captures. All 128 teacher/random/match trace hashes,
map/seat coverage and checkpoint identities were checked. Preserved an immutable
local and Modal bundle `champion51212-06db83c193a7` containing `final.npz`,
`final.json` and evidence lineage; all three uploaded files passed readback hash
verification. Full model SHA:
`06db83c193a7887c2273e7b132b111b62e285ad6921d355f2a768dd96ac15d7e`.
The current guide now selects this development candidate, explicitly labeled
imitation plus PPO and not final-accepted. Native96 remains retained. The 20/32
head-to-head point advantage has a broad Wilson interval [45.25%, 77.07%].

Two additional learned-opponent matches are running on maps 300200–300215:
new candidate versus gamma-.999 update 48 (exec 79297) and versus zero-entropy
update 64 (exec 43016). Alongside the existing three pairings, these will form
a complete four-policy development portfolio rather than relying on saturated
teacher performance alone. H14 teacher replication evaluations continue.

After H14's full GPU execution qualification, exposed its validated immutable
rival-bank options in the long launcher as `--frozen-opponents` and
`--frozen-opponent-probability`. Both local entrypoint and remote function validate
IDs/mixtures; bootstrap rejects the option, and the budget ledger records the
pool and probability. Defaults remain disabled. Compilation and actual Modal CLI
help passed; all 22 frozen engine source hashes match. No long frozen-rival job
has been submitted yet, pending strength comparisons, and commitment remains
$357.79. Existing running jobs retain their mounted source/configuration.

The new candidate's remaining learned matches completed **22–10 against each**
of gamma-.999 update 48 and zero-entropy update 64, without unresolved games.
`learned-portfolio-v2.json` verifies all six pairings, identical maps/evaluator
protocols and 192 trace hashes for a complete four-policy portfolio. Scores:
new candidate 64/96, native96 52/96, the other two 38/96 each. The new candidate's
map-cluster bootstrap win-fraction interval is [60.42%, 72.92%]; its score
difference from native96 is +12.5 points with interval [-1.04, 26.04]. Map
resampling preserves both seats and every pairing; this is not training-seed
uncertainty or universal superiority. The current guide includes these results.

H14 teacher replication finished **25/32 for both control and frozen rivals**,
without unresolved games, versus 30/32 for their native96 parent. All 96 trace
hashes across the two pilots and parent were checked and a paired comparison
saved (`frozen-rivals-paired-teacher-comparison.json`). There is no teacher
point gain from the frozen bank in this trial. A direct rivals-versus-control
match is running (exec 34428) to assess learned-opponent behavior before deciding
on any long allocation for that recipe.

Launched H15's explicitly matched long learning-rate pair from the new immutable
candidate, as specified in the plan. Both use seed 20261029, 512 lanes, 2,048
maps, gamma .995/native reward, entropy .005, restored critic/Adam and disabled
frozen rivals. Each reserves $35 for 11 requested hours / 12-hour hard cap:
- `scale-champion512-ratecontrol-v1`, learning rate 1e-4,
  app `ap-FJ7qbBJTwQIuNW6W1gEdY9`, call `fc-01M5762258D3VC126ASKHSEBV4`.
- `scale-champion512-lowrate-v1`, learning rate 3e-5,
  app `ap-028SEQ2HDzAVwfsGDu4ArB`, call `fc-01M5762R9RZJS491BVQHBV26GT`.

Both durable calls remain pending/running; final-log observation timeouts did not
cause resubmission. `champion-rate-pair-initialization.json` verifies the exact
parent NPZ hash (including optimizer), identical 39 archived sources and training
seed lists, and only `learning_rate` differing in Config. Dedicated read-only
watcher `ratepair1` is local exec 62148. There are now 11 long jobs allocated;
commitment is **$427.79**, not settled spend, leaving $72.21 uncommitted. Final
acceptance remains outstanding and final-test maps remain untouched.

H15's two jobs have published their first updates: both collected identical
first-batch outcome totals, with zero illegal/uncredited actions or outcomes,
replay error 3.3379e-6 and peak GPU memory 24,961,167,616 bytes. Their durable
calls remain active; initialization/compilation time was not treated as failure.

Launched the bounded H16 trajectory-length pilot, `horizon128-v1`: app
`ap-TMiAzVasNDeJAB8YhnGEdy`, call `fc-01M57694SW87TM13KJQHH4HDKC`.
It uses 256 × 128, eight requested updates, 2,400 training seconds / one-hour
hard cap and a $5 reservation, matching H15's standard-rate parent, seed and
training recipe otherwise. The short launcher permits 128 only for native-map
PPO and at most 256 lanes; compilation passed. `horizon128-initialization.json`
verifies exact parent weights/Adam, identical archived sources and map seeds,
and 32,768 decisions per update. Config differences are lanes/horizon plus
bounded update/time/evaluation caps. First training metrics were still pending;
GPU memory and strength are not yet qualified. Watcher `horizon128v1`, exec 25020.

H14's direct learned-policy comparison completed **24–8 for frozen rivals**,
zero unresolved games. All 32 trace hashes and paired-seat coverage passed;
map-cluster bootstrap interval [62.5%, 87.5%], with one training seed per recipe.
This is a meaningful advantage over its matched control despite tied teacher
scores, not proof of improvement against every opponent. Evidence:
`match-frozen-rivals-vs-control-dev/verification.json`.

Launched H17 `scale-champion512-frozen-v1` from the current immutable candidate,
using H15's existing standard-rate job as control. All training Config values
match that control; the treatment adds H14's same two frozen rivals to 25% of
rollouts, keeping public heuristic at 25% and splitting the remaining mass equally
between current/history. It reserves $35 with 11 requested hours / 12-hour hard
cap. App `ap-85XiOma9nsikuFQdkWCraW`, call `fc-01M576DF8H99PFBYBKTS5A1V8H`;
watcher `frozenlong1`, exec 20809. Initial remote-state comparison is pending.
Observation timeouts did not cause retries. Total commitment is **$467.79**,
leaving $32.21 uncommitted; this is not actual or settled spend. Twelve long
jobs and the bounded horizon pilot are allocated. Final acceptance stays open.

H17 initialization now verifies exact champion actor/critic/Adam, identical
Config, archived sources and training seed list against the H15 standard-rate
control. Both rival hashes match their immutable qualified bundles; evidence:
`champion-frozen-initialization.json`. Through update two, replay error is at
most 3.3379e-6, with zero illegal, resigned, exhausted or uncredited transitions.
Peak memory is 24,965,302,272 bytes. Neither of its first two rollouts selected
a frozen rival, so actual 512-lane rival selection still awaits observation.

H16 completed its first 32,768-decision update at peak memory 24,890,187,520
bytes, replay error 3.3379e-6 and zero illegal/uncredited transitions. The first
update took 269.52 seconds including first-use compilation. One update does not
qualify steady-state throughput or final strength; keep observing the same call.

Added `scripts/watch_structured_matches.py`, a serial local learned-opponent
screen that reads existing immutable captures and never controls Modal jobs.
It samples the latest available snapshot per run with round-robin service,
uses one fixed reference, leaves existing/partial outputs alone and confines
seeds to development. It is deliberately not exhaustive snapshot evaluation.
Actual bounded qualification completed eight paired games, verified all trace
hashes and exited normally (`learned-match-watcher-qualification.json`). The
native-.995 update-176 checkpoint scored 0–8 against the fixed champion despite
8/8 on its teacher probe, illustrating the teacher screen's current limitation.
This small learned screen is also a selection signal, not a full strength claim.

The 13-hour serial match watcher is now running as local exec 42123, reference
`champion51212-06db83c193a7`, on twelve bootstrapped long/pilot streams. It began
with gamma-.999 update 160; status is
`retained-checkpoints/match-watch-champion51212-status.json`, log
`match-watch-champion51212.log`. The separate weak pure-RL branch retains its
teacher watcher. Current engine verification still matches all 22 frozen hashes.
No additional cloud allocation was made; commitment remains $467.79 and the
final-test partition remains untouched.

H15 update-eight checkpoints are retained and qualified for a matched comparison:
standard rate `bc71ac6cc8846760cc65d61d55da7bdc3ee239ed02082d83ffbce370ee20eb25`,
lower rate `693cce0d9622a8fc58457f795bb48f2ae3d003d0fd39ab7e308a8a1f57ba64d0`.
Both passed 39 archived-source hashes, four current inference hashes, checkpoint
hashes, finite tensors and 12 legal host decisions. Both took exactly 2,048 new
Adam steps on 262,144 learner decisions, or two presentations per decision.
Evidence: `champion-rate-pair-update8-optimizer-progress.json`. Their full paired
head-to-head is running as exec 32389 in `match-lowrate8-vs-ratecontrol8-dev/`.
The standard-rate snapshot also remains the declared H16 update-eight control.

The automated learned screen completed gamma-.999 update 160 at 3/8, native-.995
update 192 at 0/8 and zero-entropy update 160 at 4/8 against the fixed champion,
without unresolved games. These small screens do not certify overall regression
or superiority. Because native176 and native192 both scored 0/8 despite strong
teacher probes, started a full 32-game native192-versus-champion comparison
(exec 67439, `match-native192-vs-champion51212-dev/`) before any decision to stop
 or redirect that long branch. The current candidate remains unchanged.

H16 reached update three with clean replay/credit checks and stable 24.89-GB peak
memory. Its second/third updates took 173.91 / 143.41 seconds; these early samples
suggest lower throughput than the 512×64 control, but full pilot wall time and
strength are still required. H17 reached update five with clean checks and no
frozen selections yet; keep observing the existing call rather than treating a
short random-selection gap as failure. No allocation or final-test change.

H15's full update-eight learning-rate comparison finished 17–15 for lower rate,
with all 32 trace hashes and paired coverage verified. Map-bootstrap 95% interval
for its win fraction is [0.40625, 0.65625]; both long branches continue. Lower-rate
update16 subsequently scored 6/8 against the fixed champion, and its full 32-game
comparison is running (exec 98742). These remain development selection results.

H17 has now actually executed frozen rival index1 at update11 on 512 lanes,
verified through update13 with replay error below1e-5 and zero invalid, resigned,
exhausted or uncredited transitions. The prior predicted first selection at13
assumed two complete optimization epochs; update12's Adam counter11423 shows
2944 new steps rather than3072, so that assumption did not hold. The actual
selection supersedes the conditional prediction. Evidence is
`champion-frozen-execution-qualification.json`. H16 reached update7, clean.

The full native192-versus-champion comparison finished 4–28 with32 verified
traces, no unresolved games, and map-bootstrap win-fraction interval
[0.03125, 0.25]. A stop request was accepted for owned app
`ap-c0KhiK8fgfb3J3hPMRvE2y` after retaining update208. Its latest receipt still
shows stopping with one task; no reservation has been released yet.

Native2 is now confirmed stopped with zero tasks. App-wall maximum-allocation
accounting plus $1 margin is $7.36, replacing its $35 reservation; total
commitment is $440.15, with $59.85 uncommitted. This is
conservative accounting, not an invoice. Its dedicated read-only local watcher
was stopped after exact command verification; retained checkpoints remain.

H16 completed eight PPO updates /262144 learner decisions and retained
`update-8-21cb68021a75`. It passed39 archived-source hashes,4 current inference
hashes, both checkpoint hashes, finite tensors and12 legal host decisions.
Summed update time was1402.28s versus661.02s for H15's matched standard-rate
control; this excludes volume/checkpoint writes and final evaluation. The pilot
is about2.12x slower for the same decision count. Its full32-game matched
comparison is running as exec71833 in `match-horizon1288-vs-ratecontrol8-dev/`;
optimizer/timing evidence is `horizon128-update8-comparison.json`. The cloud
call was still pending final evaluation on last observation; no budget was
released and no long horizon128 job was started.

Low-rate update16 finished17–15 against the current champion on32 paired games,
with all trace hashes verified and no unresolved results. Map-bootstrap win
fraction95% interval is[0.4375, 0.625]. This does not establish a clear improvement;
retain the current default. H16's durable call completed:32/32 random wins in
each stochastic/deterministic mode, zero invalid/truncated/cutoff games.
Its full learned-opponent comparison is still running.

The pure-RL long branch's update112 scored0/8 teacher wins and update128 scored
0 wins/7 losses/1 engine truncation. A full32-game public teacher evaluation
of update128 is running as exec18109 before deciding whether to stop that
branch for resource reallocation. Earlier pure-RL checkpoints remain retained.

H16 is confirmed stopped/tasks0; conservative app-wall accounting plus margin
is$2.23. Total commitment is$437.38, not actual or settled spend.

H18 submitted the lower-rate/frozen-rival factorial cell fromchampion51212,
seed20261029,512×64,pool2048, gamma.995,lr3e-5,entropy.005. Its rival bank and
25%public/25%frozen/remainingcurrent-history mixture match H17. App
`ap-qKuhaNCASdVYAttlNeBrLe`, call`fc-01M577Y4172NPW6V2G955Z8FNR`; $35 reservation,
11 requested training hours/12-hour hard cap. Dedicated read-only watcher
`lowrate_frozen1` started; initialization comparison pending. Total commitment
is$472.38, leaving$27.62 uncommitted, not actual spend.

H18 initialization is verified: exact champion51212 initial weights/Adam,
identical39 source hashes and2048 training seeds against both H15 lower-rate
and H17 frozen-standard controls. Config matches H15 lower-rate exactly; only
learning_rate differs from H17. Both rival paths/hashes match H17. Evidence:
`champion-lowrate-frozen-initialization.json`. Watcher exec9858. The detached
launcher reported a log-observation timeout; the existing durable call remained
pending and was not restarted. Its protocol and initial checkpoint were then
successfully read from the volume.

H16's full match finished14 wins/18 losses,
0 engine truncations and0 cutoffs against the
matched64-step control. All32 traces/paired-map coverage verified; map-bootstrap
win-fraction95% interval[0.3125, 0.5625]. At2.12x update time for equal learner decisions and
identical2048 new Adam steps, this pilot does not justify a long128-step
allocation. Retain its checkpoint; do not generalize beyond this seed/recipe.

Reconfigured only the local read-only learned-match monitor to include H18
and remove stopped native2/completed H16. Prior monitor PID55643 was checked
by exact command and stopped; its already-started noentropy192 bounded child
PID57084 may finish independently, with output preserved and skipped by the
new monitor. Prior state is archived in
`retained-checkpoints/match-watch-champion51212-before-reconfigure.json`.
New monitor exec37997, status`match-watch-champion51212v2-status.json`, uses
the same fixed champion, dev seeds, locks and existing-output skip behavior.
No training app was stopped or restarted by this reconfiguration.

The gamma999 branch's update192 scored5–3 against the fixed champion on its
small screen, so a full32-game comparison is now running in
`match-longhorizon192-vs-champion51212-dev/`. Native512 update48 scored0–8
and stopped native2's retained update208 also scored0–8. These small screens
are selection evidence, not broad strength conclusions.

H18 has completed4 updates with clean action/credit checks, maximum
replay error3.8147e-06, and peak GPU allocation
24965317376bytes. Its first update took140.88s including compilation.
Frozen selections observed so far:[]; no claims about
frozen-opponent effectiveness are supported yet. The startup allocator reported
nonfatal workspace-allocation fallback messages and then completed PPO normally.

Fresh budget observation at10:16UTC estimates$72.51 of compute from observed
application wall duration at the maximum declared rate, with all receipts
present; this excludes storage/egress and is not an invoice. Commitment remains
$472.38, much of it future training reservation. Evidence is
`spend-observation-current.json`; completing minimum strength gates would not
finish the user's full-budget optimization objective.

Gamma999 update192 finished13–19 against champion51212 on32 paired games,
no unresolved results; all32 trace hashes verified. Map-bootstrap95% interval
[0.25, 0.5625]. Keep current default and continue this alternative discount branch.
Potential-shaped updates224 and256 both scored1–7 in learned screens; a full
32-game test of update256 is running in
`match-potential256-vs-champion51212-dev/` before a resource decision.

Pure-RL update128 completed1 win/30 losses/1 engine truncation over32 teacher
games, zero cutoffs, mean maximum cities4.03125. Its policy took8191 moves,
1480 attacks,1182 summons and155 captures, but zero research actions. Newer
update144 scored0–8. All40 evaluation trace hashes and both paired coverages
were checked, and the latest retained checkpoint hash was verified. This is
a restricted strategy, not simply an evaluation timeout; research omission is
an observation, not a proven sole cause of poor performance.

Requested stop of owned pure continuation app`ap-eHw4gVBhXd4TiNkyc2Bzxr` after
retaining update144 (`a5a4b7cb94e6d8d39e04cdc8b892340057c4f7dfd3ffe494c51b96dfd9528362`).
Wait for stopped/tasks0 before releasing its reservation. Retain pure32 as the
earlier5/32 pure-RL control. The current guide now explicitly says that this
pure branch remains weak and that the strong candidate uses imitation+PPO.

Pure continuation is now confirmed stopped/tasks0. Accounted maximum-allocation
app-wall estimate plus margin is$7.81; commitment is$445.19. This
is not settled billing. Stopped only its verified local read-only watcher.

Potential-shaped update256 finished 6–26 against champion51212 with no unresolved
games; all 32 traces/paired coverage verified and map-bootstrap win-fraction
95% interval [0.0625, 0.3125]. This follows 1–7 screens at updates224 and256.
Requested stop of owned app `ap-TzPYfyZbzKbd9pKyhTifjJ` after retaining latest
update272 (`277c9953fdf5` prefix). Await stopped/tasks0 before accounting.
The stopped branch remains available as checkpoint/opponent diversity; this is
resource allocation evidence, not a universal conclusion about shaping.
Planned H19 reuses freed funds for a second-seed matched lower-rate native pair,
with and without frozen rivals, rather than changing the frozen engine.

Potential continuation is confirmed stopped/tasks0. Conservative app-wall
accounting plus margin is$8.91; commitment is$454.10 including the
new H19 control reservation. Its read-only watcher was stopped after checking
its exact command and empty evaluation queue. No other training app was stopped.

H19 second-seed pair submitted after terminal accounting freed sufficient
budget. Each reserves $35 for 11 requested training hours / 12-hour hard cap.
- `scale-champion512-lowrate-seed2-v1`: app `ap-GxmhzPSd38YiE7mmfs1wA3`, call `fc-01M578PC3692E7R6Y6KEW99YZ0`.
- `scale-champion512-lowrate-frozen-seed2-v1`: app `ap-wic7kR4DtJ7jnkpAeTR5Ra`, call `fc-01M578QGDMD9AH9N5N7ZHZ0XVE`.
Both use champion51212, seed20261030, rate3e-5, native gamma.995 and
512×64/pool2048; only frozen-rival mixture differs. Dedicated checkpoint
watcher `seed2pair1` started. Initial remote comparison remains pending.
Total commitment is $489.10, leaving $10.90 uncommitted; this is not actual spend.

H19 initialization now verifies exact parent weights/Adam for both jobs,
identical39 source hashes and2048-map training pool. Both configs match; versus
H15 lower-rate, only seed changes20261029→20261030. Rival hashes match H18.
Evidence:`champion-seed2-pair-initialization.json`. Checkpoint watcher exec21521.

H18 update8 scored6–2 versus champion in its screen, but no frozen rival had
been selected through the first11 observed updates. Therefore this early result
cannot be attributed to rival training. H15/H18 first rollout aggregate counts
match; both update8 Adam counters are10527, but parameter L2 difference is
0.87393. Same initialization/seed did not yield bitwise-identical training. The
exact cause is not established; do not silently claim identical trajectories
or a frozen-rival benefit. Evidence:`champion-lowrate-frozen-pretreatment-check.json`.

Reconfigured the read-only learned-match monitor to include both H19 jobs and
remove stopped potential1. Prior v2 state is archived with a-before-reconfigure
suffix; any in-flight bounded child keeps its output and is skipped by the new
monitor. New status:`match-watch-champion51212v3-status.json`, same fixed
champion/reference/development seeds, twelve live bootstrapped streams.

H18 actually selected frozen rival0 at update13, with maximum replay error
4.7684e-6 and zero invalid, resigned, exhausted or uncredited transitions through
that update. Updated`champion-lowrate-frozen-execution-qualification.json`.
Its pre-treatment update8 screen remains ineligible for a causal rival-benefit
claim; later checkpoint comparisons will include actual rival exposure.

Both H19 jobs completed their first32768-decision update. First rollout aggregate
stats match, replay error2.861e-6, and action/credit checks are clean. No frozen
rival selected yet in seed2. Evidence:`champion-seed2-pair-execution-qualification.json`.
The first-update times171.40s(control) and134.64s(frozen) include compilation and
are not a steady-state throughput comparison. Both jobs remain live.

H15 update32 screens finished3–5(standard) and4–4(lower-rate) against the fixed
champion, with no unresolved games. No promotion or additional cloud allocation
is justified by those small screens. Total commitment remains$489.10.

Started a matched H17 frozen-standard update16 versus H15 standard-control
update16 comparison on fresh development maps300300–300315, both seats32 games.
Output:`match-frozen16-vs-ratecontrol16-fresh-dev/`. Both have524288 learner
decisions from the same immutable parent/seed/config/source; actual optimizer
steps are recorded separately in`champion-frozen16-matched-optimizer-progress.json`.
This predeclared fresh development partition will also be used for the matched
lower-rate frozen/control comparison once post-exposure update16 is retained.
Do not use final seeds400000–400031 or attribute pre-exposure differences to
frozen opponents. Training-seed replication remains the H19 pair.

Started a full 32-game learned comparison of older native512 update64 against
champion51212 in `match-native51264-vs-champion51212-dev/`. Its previous update48
lost 0–8 in a learned screen, while update64 still wins 8/8 teacher games. This
is another check that a saturated teacher score does not justify continued
allocation by itself. No stop decision until the fuller comparison; retain all
saved checkpoints and distinguish starting-parent strength from recipe effects.

H18 update16 (`073338fde07a` prefix) is retained after actual frozen exposure.
Started its matched comparison against H15 lower-rate update16 on the same
predeclared fresh maps300300–300315, both seats32 games. Output:
`match-lowrate-frozen16-vs-lowrate16-fresh-dev/`. Actual optimizer work is in
`champion-lowrate-frozen16-matched-optimizer-progress.json`; both checkpoints
have524288 learner decisions. Compare this contrast with H17's standard-rate
frozen/control contrast without treating one training seed as conclusive.

H17's matched fresh-map update16 comparison finished10–22 against its
standard-rate control, no unresolved games. All32 trace hashes and paired
coverage verified; map-bootstrap win-fraction95% interval[0.1875, 0.4375]. Its earlier
short-pilot advantage did not transfer to this parent/seed/scale/checkpoint.
Do not promote the rival recipe from the pilot alone. Lower-rate contrast and
second training seed remain pending; this one checkpoint is not a universal
conclusion or grounds for changing the frozen engine.

Both H19 first4-update snapshots are retained, with clean replay/action/credit
checks. The frozen seed2 run selected rival1 at update4. Updated execution
qualification includes actual selection and memory evidence. All twelve live
training jobs retain their existing handles; no further cloud allocation.

Older native512 update64 finished9–23 against champion51212, no unresolved
games,32 trace hashes/paired coverage verified; map-bootstrap95% win-fraction
interval[0.125,0.4375]. Requested stop of its owned app
`ap-HgZB9VP0HnU6AW5uxJgVLK` after retaining latest update80 (`654bb12e6c8a`
prefix). Await stopped/tasks0 before releasing reservation. This reallocates a
weaker-parent branch, not evidence that gamma.999 or512 lanes universally fail.
H20 is predeclared as a stronger-parent entropy ablation matched to H15's
lower-rate control. No new job until terminal accounting permits its$35 bound.

Older native512 is confirmed stopped/tasks0. Conservative app-wall accounting
plus margin is$6.43; commitment is$460.53. Its dedicated local
read-only watcher was stopped after exact command verification. Checkpoints
and evaluations remain retained; this accounting is not settled billing.

H18 lower-rate frozen/control update16 match finished16–16 on fresh maps,
no unresolved games. All32 trace hashes/paired coverage verified; map-bootstrap
win-fraction95% interval[0.375, 0.625]. Both used4096 new Adam steps and524288 learner
decisions. This is no observed advantage, not proof of equivalence. Compared
with standard-rate's10–22 result, the difference in these matchup win fractions
is0.1875 with paired-map interval[0.0, 0.375]; this is not a universal interaction
effect or a training-seed uncertainty interval. Evidence:
`champion-frozen-rate-contrasts-fresh-dev.json`. Continue second-seed and longer
training tests; do not promote from the older pilot result.

H20 submitted as app`ap-zZIBkEXWycJdyfNeUdMk5I`, call`fc-01M579GY280AQHFDBPN36NYQ8N`.
It reserves$35, with11 requested hours and12-hour hard cap. Initialization
verifies exactchampion51212 actor/critic/Adam,39 identical archived sources,
identical training maps/opponent mixture, and only entropy_coef .005→0 differs
from H15 lower-rate control. Evidence:`champion-lowrate-noentropy-initialization.json`.
Dedicated capture watcher exec87374 (`lowrate_noentropy1`). Commitment is
$495.53, leaving$4.47 uncommitted, mostly future training.

Improved the local read-only learned-match monitor with optional`--run-list`
JSON. Valid atomic roster changes preserve service history and the active
evaluator; invalid/unreadable updates retain the last valid roster. Fixed
reference/maps stay outside the roster. Python compilation passed and an actual
initial roster followed by an atomic replacement was observed in the same live
monitor. Evidence:`learned-match-hot-roster-qualification.json`. No new formal
test suite was added for this low-impact monitor change.

The v3 monitor's state was archived before its final migration; its in-flight
champion512 update48 child kept its bounded output. Current monitor is exec91547,
status`match-watch-champion51212v4-status.json`, reading`learned-screen-runs.json`.
Its twelve-run roster includes H20 and excludes stopped native512; future roster
changes do not require restarting it. No cloud job was controlled by the monitor.
Current ruleset verification still passes all22 frozen hashes.

Expanded evaluation of H15 lower-rate update16, the closest full-match challenger
(17–15 versus champion), against the same three immutable references used in the
champion portfolio: native96, gamma99948 and noentropy64. Use32 paired games per
reference on the established300200–300215 development maps, permitting a paired
common-opponent comparison against the champion's existing96 games. The direct
17–15 result remains separate. Started native96(exec10732) and gamma48(exec23988)
comparisons first; zero-entropy reference comparison will follow to bound local
concurrency. Do not promote on one reference or call this related-policy portfolio
universal strength. Complete teacher/random qualification before any promotion.
This adds no cloud allocation; live training continues within$495.53 commitment.

Lower-rate update16 finished21–11 versus native96 and20–12 versus gamma99948,
compared with champion51212's20–12 and22–10. All64 new traces/paired coverages
verified. The common-opponent comparison is incomplete until the noentropy64
match finishes (exec16934). Started the candidate's full teacher development
evaluation on300000–300015, both seats32 games, in`lowrate16-teacher-full-dev/`.
No promotion from these partial portfolio results.

Pinned the exact previously observed training-history bytes for H17, H18,
H19 and H20 qualification reports. Reconstructing the recorded history prefix
with the writer's exact serialization reproduced each original SHA-256 before
saving a separate immutable file and linking it from its report. Later polling
can update-current files without losing the report's precise metric evidence.
H20's first update confirms weighted entropy-gradient norm0, nonzero policy
and value gradients, clean replay/action/credit checks and24.96-GB peak memory.

Lower-rate update16 finished23–9 versus noentropy64. Its three-common-reference
total is64–32, exactly matching champion51212. Rebuilt the complete five-policy
portfolio from all10 pairings/320 games, checking protocol/checkpoint identities,
paired coverage, native endings and every trace hash. Lower-rate16 scores81/128
versus champion's79/128 only because their direct match was17–15; common-opponent
win fractions tie exactly. The paired-map95% interval for common-opponent
difference is[-0.09375,0.09375]; full-portfolio difference0.015625 has interval
[-0.0703125,0.1015625]. These measure map variation, not independent training seeds.

Artifacts:`learned-portfolio-v3.json` and its standalone assertion-checked builder
`build-portfolio-v3.py`. Retain the incumbent pending stronger evidence. The
alternative's second teacher partition is running as exec26627 in
`lowrate16-teacher-replication-dev/`; its first teacher partition is still running.
No random qualification or immutable qualified bundle has been claimed yet.

Lower-rate update16's original teacher partition finished31–1 with zero unresolved
games and mean maximum cities8.71875; all32 trace hashes/paired coverage verified.
The second teacher partition remains running (exec26627). Started its deterministic
32-game non-resigning random qualification as exec61246 in
`lowrate16-random-full-dev/`. Candidate remains an alternative, not a promoted
default: it ties the incumbent's common-opponent total and has not improved the
teacher result. No claim of completed qualification or final acceptance yet.

Lower-rate16 random qualification finished32–0, no unresolved games. Its second
teacher partition is still finishing; no immutable alternative bundle yet.
Started H19 second-seed frozen/control update16 match on the same fresh
300300–300315 maps used for the first-seed contrast, both seats32 games, in
`match-seed2-frozen16-vs-control16-fresh-dev/`. Both have524288 learner decisions;
actual optimizer work is recorded in`champion-seed2-update16-optimizer-progress.json`.
This will test whether seed1's16–16 lower-rate result changes under another
training RNG seed. No new cloud allocation or final-map use.

Lower-rate16 qualification completed:31/32 original teacher,30/32 replication,
32/32 random; all61 teacher wins ended in learned-policy captures. Verified all
224 traces across these three benchmarks and its four learned-policy matches,
checkpoint/protocol identities and paired-map coverage. Refreshed provenance
check passed39 archived sources,4 current inference hashes,2 checkpoint hashes,
finite tensors and12 legal host actions; Adam12575, parent policy L2 delta1.21996.

Created immutable local/Modal bundle`lowrate16-d9be65407e63`, SHA
`d9be65407e63af5a8d55a5123deb075534e8a03299d871449fe8f4457876331b`.
It contains final.npz, adjacent metadata and lineage with exact benchmark hashes.
Upload used force=False and all3 files were read back and hash matched; evidence:
`lowrate16-qualified-bundle-upload.json`. Label remains teacher-imitation+PPO.
It is a qualified alternative and potential opponent, not a promoted default
or final acceptance. Current champion retains64/64 teacher and the tied64/96
common-opponent result. No additional GPU allocation was needed.

H19 second-seed update16 match completed13–19 frozen versus control, zero
truncations/cutoffs. Verified all32 trace hashes, exact checkpoint identities,
paired coverage and summary consistency. Paired-map bootstrap95% interval is
[0.28125,0.53125] (100000 resamples, seed20261030). Both sides used524288
learner decisions and4096 new Adam steps. Together with seed1's16–16, this
provides no replicated frozen-rival advantage at this training budget. Two
training seeds do not support a precise training-seed uncertainty estimate.
Evidence:`match-seed2-frozen16-vs-control16-fresh-dev/verification.json`.

Read-only Modal Environment billing report through2026-10-18T11:00:00Z
reported$94.54891047 across all40 campaign app IDs. The saved report is
`modal-billing-owned-20261018T110144Z.json`. It includes only completed hourly
intervals, may lag, and excludes current-hour and separately attributed
volume/egress costs; it is not a final invoice. No reservations were released.
Commitment remains$495.53, comprising$420 active reservations and$75.53
conservative completed-job accounting. Reservations are not spend.

Started local32-game development comparisons for H20 zero-entropy update16
versus H15 lower-rate update16 on300300–300315, champion512 update64 versus
its retained update12 incumbent, and champion128-smallpool update176 versus
the incumbent (both latter matches on300200–300215). These assess the entropy
ablation, continued training and the persistently weak small-pool branch;
none uses final-test maps or allocates additional cloud compute. All22 frozen
engine hashes still match. Local exec sessions64401,31315,6326 respectively.

At11:06UTC, pinned health snapshots from all12 active branches contain
19,316,736 learner decisions,29,446 credited native completions and20 engine
truncations. Every recorded update passes the1e-5 behavior-logp audit (maximum
9.894371e-6), finite loss, zero invalid actions, zero rollout loop exhaustions,
zero resignations and zero uncredited endings. Collection accounts for85.7–96.3%
of measured update time across branches. These are execution checks and mixed
training outcomes, not held-out strength evidence. Exact immutable snapshots
are linked in`active-training-health-20261018T1106Z.json`.

H20 update16 provenance verification passed39 archived sources,4 current
inference hashes,2 finite/hash-matched checkpoints and12 legal host decisions.
Adam12575 matches H15 lower-rate control at the same checkpoint, giving4096
new optimizer steps on524288 learner decisions. Parent policy L2 change1.21985;
the full matched development evaluation remains running.

H20 update16 matched evaluation completed11–21 against H15 lower-rate control,
zero truncations/cutoffs. All32 trace hashes, checkpoint identities and paired
coverage verified. Paired-map bootstrap95% interval[0.21875,0.4375]; the
ordinary game-level Wilson interval is wider because it does not retain the
same map/seat pairing. Equal524288 learner decisions and4096 new Adam steps.
This favors keeping entropy at this checkpoint; it is one training seed and
does not establish universal inferiority of zero entropy. Both long branches
continue. Artifacts:`match-lowrate-noentropy16-vs-lowrate16-fresh-dev/verification.json`
and`champion-lowrate-noentropy16-optimizer-progress.json`.

Documented a conditional H21 stronger-rival pilot using the incumbent and
qualified lower-rate16 in place of the weaker old frozen bank, with all other
H18 training settings held fixed. It is not submitted or reserved: funding
requires a justified stopped branch or other terminal accounting first.

Champion512 update64 finished17–15 versus incumbent update12, no unresolved
games; paired-map interval[0.375,0.6875]. No promotion from that near tie.
Small-pool champion128 update176 finished12–20, no unresolved games, interval
[0.21875,0.53125]. All64 traces, checkpoint identities and paired coverages
verified. Alongside repeated weak screens, this supports reallocating that
branch's budget; it does not prove small pools are universally inferior.
Retained its latest update192 (`2f4d0854673f`) before requesting stop of owned
app`ap-yGV36MaVA70QEchHcqyz4G`. First receipt is stopping/tasks1: the full$35
reservation stays in place until terminal confirmation, and H21 is not yet
submitted. No other training app was stopped.

Small-pool app reached stopped/tasks0 at04:11:17PDT; terminal receipt saved.
Observed app wall time plus the existing$1 margin accounts$6.38, replacing its
$35 reservation. Removed only this run from the hot learned-match roster and
stopped its exact-command-verified local checkpoint watcherPID52413. The
incumbent and all retained small-pool snapshots remain intact.

Submitted H21 `stronger-rivals-v1`, app`ap-xVh1cOsquwuja9YWsJBU4a`, durable
call`fc-01M57BBWG94TXV59J5DMPDAF69`, with the planned16 updates/2400-second soft,
3600-second hard allocation and$5 reservation. Commitment now$471.91 with
$28.09 uncommitted; reported billing remains the separately scoped$94.55
through11:00UTC. Eleven long jobs continue. Started its read-only two-hour
checkpoint watcher and added it to the existing hot learned-screen roster.
Initialization and actual exposure still require qualification; submission
alone is not a successful experiment. No final maps or engine files changed.

H21 initialization metadata verified: identical champion51212 parent hash,
39 source identities, training-map list and opponent mixture relative to H18.
Only the intended rival bank changes; Config differences are the bounded pilot
update/time limits and its post-training evaluation cap4096 versus10000.
The two rivals match their qualified immutable hashes. Actual training and
first frozen exposure remain pending. Report:`stronger-rivals-initialization.json`.

Started32-game full development match for wide-pool champion128 update176
against the incumbent after its latest8-game screen went0–8. This is an
allocation check, not a stop decision from the small screen. Exec67754,
`match-champion128-wide176-vs-champion51212-dev/`, maps300200–300215.

Submitted the planned H22 gamma.999 pilot from the same strong incumbent,
matching H15 lower-rate settings except gamma and short-run bounds. Run
`champion-gamma999-v1`, app`ap-AUVjIYPzSoxZx647N6T3k5`, durable call
`fc-01M57BF8EPQE3C3DCPNNY4T6AT`. Reserved$5; total commitment$476.91,
uncommitted$23.09. Requested16 updates with2400-second soft/3600-second hard
caps. Added its read-only checkpoint watcher and hot learned-screen entry.
The detached CLI's final-log timeout is not evidence of remote completion;
poll the recorded function call and retain the reservation until terminal
accounting. Initialization qualification remains pending.

H22 initialization now verifies the exact incumbent starting hash,39 identical
source identities, identical training maps/opponent mixture and gamma as the
only training hyperparameter difference versus H15 lower-rate. Update/time and
post-training evaluation caps differ as planned. Report:
`champion-gamma999-initialization.json`. Both pilot apps are confirmed detached
with one live task; no restart followed the detached-client log timeout.

H21's first recorded update passes replay (3.33786e-6), finite loss and all
action/ending-credit checks; peak GPU allocation24.965GB. Pinned the exact
training snapshot in`stronger-rivals-execution-qualification.json`. It has not
yet sampled a frozen rival, so this numerical qualification is not evidence
of a stronger-opponent treatment effect.

The H19 second-seed frozen update16 scored6/8 against the incumbent in its
latest small screen. Started a full32-game learned match on300200–300215 to
check that signal before any promotion, in
`match-seed2-frozen16-vs-champion51212-dev/`. Its separate13–19 result against
its own control on300300–300315 remains reported; different opponent/map
comparisons are not interchangeable and may be nontransitive.

Wide-pool champion128 update176 finished4–28 against the incumbent, zero
unresolved games. All32 trace hashes, checkpoint identities and paired map
coverage verified; map-bootstrap95% interval[0.03125,0.25]. Its latest retained
checkpoint is update192 (`584e4f64b19e`). Stopped only its owned cloud app
`ap-0jRkxCKRSt0A6omNSstL9X`; terminal receipt confirms stopped/tasks0 at04:17:42PDT.
Conservative observed-wall accounting is$6.96, replacing the$35 reservation.
Current commitment$448.87, uncommitted$51.13. This budget choice does not
isolate batch-size causality: the faster512-lane branch has more learner
decisions at similar wall time. Its cloud app remains running.

Removed the stopped128-lane run from the hot learned-screen roster. Replaced
the exact-command-verified shared champion1 checkpoint watcherPID51818 with a
watcher for the surviving512-lane run only, watch name`champion512_only1`,
exec44913. Archived the old watcher status before reconfiguration. Existing
local evaluations are allowed to finish; no training process was restarted.

H22 first update passes replay (3.33786e-6), finite loss and all action/ending
credit checks. Exact snapshot is pinned by
`champion-gamma999-execution-qualification.json`. H21 has now recorded3 clean
updates, but none has selected a frozen rival yet. Ten long jobs and two short
pilots remain active; no additional long allocation precedes pilot evidence.

H19 second-seed frozen update16's expanded learned match finished16–16 against
the incumbent, no unresolved games. All32 trace hashes, checkpoint identities
and paired map coverage verified; map-bootstrap interval[0.34375,0.65625].
Its6–2 initial screen did not establish a gain when expanded. This development
set contains the original four screened maps; it is not an independent final
test. No promotion. Evidence:
`match-seed2-frozen16-vs-champion51212-dev/verification.json`.

Started incumbent stochastic-inference evaluation against the unchanged
balanced teacher on the same300000–300015 development maps, both seats32games,
sampling seed20261019. Its deterministic reference is32/32 there; checkpoint
hash, teacher and map/sampling settings match apart from inference mode.
Exec14743, `champion51212-stochastic-teacher-dev/`. This informs inference-mode
selection on development data; final mode selection still precedes final tests.

Submitted H23 `champion-selfplayonly-v1`, app`ap-EdF9hTOQLhtfqcQMlQyB9p`, call
`fc-01M57BV9JZBT50P9SPF8TW10KM`, using the documented16-update/$5 pilot bounds.
It changes public-opponent probability25%→0% relative to H15 lower-rate,
retaining the same imitation-initialized parent and otherwise matching its
training settings. This is self-play-only continuation, not pure RL from scratch.
Current commitment$453.87, uncommitted$46.13. Added a two-hour read-only
checkpoint watcher and the hot learned-screen entry. Initialization and actual
opponent-selection qualification remain pending; no cloud resubmission follows
the detached CLI's final-log timeout.

H23 metadata initialization verified against H15 lower-rate: exact incumbent
parent hash,39 source identities and identical training-map list. Only the
public-opponent probability changes among training settings; short-run bounds
and post-training evaluation cap differ as planned. Actual selection records
remain pending. Report:`champion-selfplayonly-initialization.json`.

Extended the local learned-match evaluator with per-side stochastic flags and
explicit sampling seeds, preserving both-deterministic defaults. The protocol
records each side's mode and active seed. Compilation passed. A real default
regression reran the prior gamma99948 self-match on300200, both seats: all1974
decision rows (observations, commands, log probabilities/values and ending
flags) match exactly, as do outcomes; only elapsed times differ. Verification:
`match-mode-default-regression-dev/default-regression-verification.json`.
No training, inference implementation or frozen-engine source changed.

Started a32-game development mode comparison with the incumbent checkpoint on
both sides: stochastic challenger seed20261019 versus deterministic reference,
maps300200–300215. Exec35552,
`match-champion51212-stochastic-vs-deterministic-dev/`. This complements the
teacher mode comparison; one sampling seed does not establish a robust ranking.

H21 retained update4 provenance passed39 archived sources,4 current inference
hashes,2 finite/hash-matched checkpoints and12 legal host decisions. Adam9503,
policy deltaL2 .529186. This checkpoint still precedes any observed frozen-rival
selection, so it establishes execution/provenance only.

The continuing512-lane branch's update80 (`18a342345df0`) screened6/8 against
the incumbent. Started an expanded64-game paired-seat comparison on300200–300231,
including16 additional development maps beyond the usual learned-opponent set.
This is development selection, not final acceptance; report the original and
additional map subsets separately. Run:
`match-champion51280-vs-champion51212-expanded-dev/`,2400-second local limit.
No cloud allocation or final-map use. H21 has8 recorded updates but still no
frozen-rival exposure; its small-screen result remains pretreatment evidence.

Incumbent stochastic teacher evaluation finished30–2, zero unresolved games,
versus the deterministic reference's32–0 on identical maps/teacher/checkpoint
and sampling-seed setting. Verified all32 new trace hashes and paired coverage.
Stochastic-minus-deterministic win-fraction difference is−0.0625, paired-map
bootstrap interval[−0.15625,0]. One stochastic seed does not characterize all
sampling variation; no observed teacher benefit or final-mode acceptance.
The direct stochastic-versus-deterministic learned match remains running.
Report:`champion51212-stochastic-teacher-dev/verification.json`.

H23 first2 updates pass replay (maximum3.33786e-6), finite losses and all
action/ending-credit checks. Every recorded opponent selection is self-play,
with public/random flags false and no frozen index. Pinned exact training
snapshot in`champion-selfplayonly-execution-qualification.json`. Its12 early
native completions (1 win/11 losses) are mixed ongoing-game training outcomes,
not a held-out strength result. The bounded pilot continues.

The incumbent's direct mode comparison completed stochastic8–24 deterministic,
no unresolved games. Same exact checkpoint on both sides; stochastic sampling
seed20261019 and deterministic reference are explicitly recorded. All32 trace
hashes and paired coverage verified. Stochastic win fraction.25 has map-bootstrap
95% interval[.125,.40625]. Combined with30/32 versus32/32 teacher wins, this
supports deterministic decisions for the current incumbent. It does not freeze
the final candidate or establish superiority for every future checkpoint.
Evidence:`match-champion51212-stochastic-vs-deterministic-dev/verification.json`.

H21 now has14 recorded updates and first actual frozen exposure at update13,
index0 (the incumbent). Its first12 checkpoints are pretreatment for rival
identity, despite normal training updates. No invalid actions, resignation
endings, loop exhaustion or uncredited endings through the observed history.

Implemented bounded H24 A100-80GB/1024-lane support in the short launcher only.
The installed Modal SDK documents`Function.with_options(gpu=...)`; it preserves
the existing one-hour timeout, one-container limit, CPU/RAM caps and zero retries.
Local guards accept the two intended L40S512/A1001024 combinations and reject
five unsupported GPU/lane/stage/mode combinations. Runtime records actual device
identity and rejects unexpected multi-device or non80GB A100 allocations.

Extended budget accounting to reject rates below a run's recorded rate, and
use decimal ceiling for cents. The new exact-hour test caught binary floating
point rounding$4.14 to$4.15; decimal calculation fixes that while a one-second
excess still rounds up. All8 budget tests pass, including concurrency, duplicate
IDs, wrong/live receipts and GPU-rate protection. All22 frozen engine hashes
still match. Local evidence:`a100-pilot-local-qualification.json`.

Submitted `champion-a1001024-v1`, app`ap-ZpAhf91RVlcbNvs7fnbJZh`, durable call
`fc-01M57CFRDT00VHZ9XZ7Z1MRJSY`. Requested8 updates/524288 learner decisions,
2400-second soft and3600-second hard caps, one A100-80GB,8 CPU/32GiB, no retries.
Reserved$5 at conservative$3.14/hour; total commitment$458.87, uncommitted$41.13.
Read-only checkpoint watcher exec1880 and the existing hot match roster now
include this pilot. Its actual numerical/memory/throughput qualification remains
pending. Compare against H15 lower-rate16 at equal decisions, reporting hardware,
lane count, actual optimizer work and source delta in the budget module.

Rechecked official Modal pricing and GPU configuration on2026-10-18:
https://modal.com/pricing and https://modal.com/docs/guide/gpu. The listed
GPU+8CPU+32GiB rates imply about21.2% extra throughput to offset A10080's extra
cost. No speedup or memory fit has yet been claimed.

H24 confirmed one NVIDIA A100-SXM4-80GB, correct resource caps and the exact
incumbent initial hash. Against H15 lower-rate, all38 other archived sources
match; only`rl/budget.py` differs for the documented accounting protections.
Training maps, opponent mixture and learning hyperparameters match; Config
differences are1024 versus512 lanes and short-run update/time/evaluation bounds.
Report:`champion-a1001024-initialization.json`. Runtime fit/performance pending.

Expanded champion512 update80 comparison completed30–34 overall, with all64
trace hashes, checkpoint identities and paired coverage verified. Original
300200–300215 maps:16–16; additional300216–300231 maps:14–18. Total paired-map
95% interval[.359375,.578125]. No improvement established or promotion.
Saved`match-champion51280-vs-champion51212-expanded-dev/verification.json`.

Pinned H21 history through14 updates and verified first actual frozen exposure
at13/index0, along with numerical/action/ending-credit checks, in
`stronger-rivals-frozen-exposure-qualification.json`. The matching full16-update
strength comparison remains pending.

Started expanded64-game comparison of H19 lower-rate seed2 control update16
(`a05a132eacf8`) against the incumbent on300200–300231. It screened6/8 and
beat its own frozen counterpart19–13 on300300–300315, warranting a fuller
candidate check. Run:`match-seed2-control16-vs-champion51212-expanded-dev/`.
No new cloud allocation; these are development maps, not final acceptance.

H21 and H22 retained update16 checkpoints passed39 archived source hashes,
4 current inference hashes,2 finite/hash-matched checkpoints and12 legal host
decisions each. Both have Adam12575 from8479:4096 new optimizer steps on524288
learner decisions, matching their respective controls. Reports:
`stronger-rivals16-optimizer-progress.json` and
`champion-gamma99916-optimizer-progress.json`.

Started paired32-game matches on300300–300315: H21 stronger-rivals16 versus
H18 old-rivals16 (exec58024,
`match-stronger-rivals16-vs-old-rivals16-fresh-dev/`), and H22 gamma99916 versus
H15 gamma995/lowrate16 (exec77897,
`match-champion-gamma99916-vs-gamma99516-fresh-dev/`). Both pilots' durable cloud
calls remain pending with one live task each while their post-training work
finishes; their reservations have not been released. The A100 pilot has only
initialization/device qualification so far.

H21 and H22 durable calls returned complete; each finished32/32 random-baseline
wins in both stochastic and deterministic device evaluations, with no invalid
actions, engine truncations or evaluation cutoffs. Fresh app receipts confirmed
stopped/tasks0 before terminal accounting: H21$2.14 and H22$2.05, including the
existing$1 per-app margins. Total commitment is now$453.06, headroom$46.94.
These are conservative accounted amounts, not settled invoices. Learned-policy
match evaluations remain separate and are still running.

H24 first2 A1001024 updates passed the1e-5 replay audit (maximum8.70228e-6),
finite loss and all action/ending-credit checks. Peak allocation36,289,602,816
bytes fits the reported63,839,404,032-byte allocator limit. This establishes
initial execution/memory fit only, not final speed, cost efficiency or strength.
Exact snapshot is pinned in`champion-a1001024-execution-qualification.json`.


H21 stronger-rivals16 completed15–17 against matched old-rivals16 on32 fresh
development games, with all32 trace hashes verified. Only2 rollouts used frozen
opponents, both index0; no long allocation is justified by this pilot.
H22 gamma99916 completed21–11 against matched gamma995/lowrate16, with all32
trace hashes verified; its broader64-game incumbent comparison is underway.
Its full teacher development check completed31/32; verification pending.

H19 second-seed lower-rate control16 completed39–25 versus incumbent over64
games. All64 trace hashes, checkpoint hashes, paired coverage and native endings
verified. Original16 maps:23–9; additional16 maps:16–16. Paired-map95% interval
for all32 maps:[.5,.71875]. This warrants broader qualification, not promotion:
the advantage did not replicate on the additional subset. Source/finite/legal
host checks passed39 archives/4 inference/2 checkpoint hashes/12 legal decisions;
Adam12575, policy L2 change1.2401433. Full32-game deterministic teacher and
random checks started on development300000–300015. Final maps remain untouched.


H23 self-play-only continuation finished16 updates,524288 learner decisions and
4096 new Adam steps, matching H15 mixed-control16. All rollouts were self-play;
728 native completions, no engine truncations, invalid actions, resignations,
loop exhaustions or uncredited endings. Max replay discrepancy5.00679e-6.
Checkpoint034499b9c92a passed39 source/4 current inference/2 finite checkpoint
hashes and12 legal host decisions. Cloud random evaluation completed32/32 in
both inference modes. Matched fresh32-game comparison against H15 mixed16 is
running. Exact final training snapshot and work counts are recorded in
`selfplayonly16-optimizer-progress.json`. No new imitation was used in this
continuation; its parent remains imitation-initialized.

Started expanded64-game incumbent comparisons for champion512 update96
(05c414466823) and lowrate-noentropy update32(a46fb08ef327), each after7/8
screening. Outputs:`match-champion51296-vs-champion51212-expanded-dev/` and
`match-lowrate-noentropy32-vs-champion51212-expanded-dev/`. No promotion from
screening results. Started32-game common-reference comparisons for H19control16
against native96, gamma99948 and noentropy64. All are local development runs.
Rechecked frozen ruleset: all22 source hashes match.

H23 durable call complete and fresh app receipt confirms stopped/tasks0.
Conservative terminal accounting$2.42; total commitment$450.48,
headroom$49.52. Reservations and conservative accounting are not invoices.


H22 gamma99916 expanded incumbent match completed29–35, with64 trace hashes,
checkpoint identities, paired coverage and native endings verified. Original
maps15–17; additional maps14–18; aggregate paired-map95% interval[.375,.53125].
Its21–11 matched-control advantage does not establish improvement over the
incumbent. No long allocation or promotion. Full teacher31/32 also verified.

H19control16 full teacher32/32 and random32/32 both verified, all64 traces and
paired coverage; teacher mean maximum cities8.90625, minimum5, median finish
turn21,1348 attacks and231 research actions. Second teacher map set300100–115
and three common learned references are running. `seed2-control16-behavior-dev.json`.

H24 A1001024 finished8 updates/524288 decisions with maximum replay discrepancy
9.53674e-6 and no invalid, resignation, loop-exhaustion or uncredited events.
Peak36.29GB fits. Update time1337.30s versus matched L40S512/control16 at1208.99s:
1.1061x time and1.3440x conservative estimated update-time cost. This compares
operating points, not isolated hardware: lane count, rollout size and experience
schedule change. Startup/evaluation are excluded from these timing costs.
`a1001024-matched-throughput-qualification.json` pins exact histories and counts.
Started32-game fresh development match at equal decisions against H15low16;
cloud post-training evaluation and terminal accounting remain pending. No long
A100 allocation is justified by throughput; strength remains under review.

H24 cloud evaluation finished32/32 random wins in both modes, no invalid
actions, engine truncations or cutoffs. Fresh app receipt confirms stopped
and tasks0; accounted at required$3.14/hour: $2.43 including$1 margin.
Commitment$447.91, headroom$52.09; not invoice spend.


H23 self-play-only16 matched comparison completed18–14 against mixed-control16
on300300–300315. All32 trace hashes, checkpoint identities, paired coverage and
native endings verified. Paired-map95% interval[.4375,.6875]; one seed does not
establish an improvement from removing public opponents. No long allocation.

At2026-10-18T12:00:29Z the public Modal billing report returned$126.65708834 for
all44 ledger-owned applications across completed hourly intervals06:00–12:00UTC.
Report:`modal-billing-owned-20261018T120029Z.json`. This may lag and is not an
invoice; current partial-hour and separately attributed storage/egress excluded.
Commitment remains$447.91, headroom$52.09, with10 long workers still running.
Reservations are not spend, and the full-budget training objective remains active.

Started32-game teacher qualification and source/finite/legal host verification
for champion512 update96 while its expanded64-game incumbent comparison runs.
Started a full32-game check of standard-rate frozen update80 against incumbent
to assess continued allocation after repeated weak8-game screens. Neither is a
promotion or stop decision yet.


H19control16 fully qualified as an alternative, without default promotion:
teacher32+31=63/64 (all63 learned captures), random32/32; learned references
native96:21–11, gamma99948:22–10, noentropy64:24–8, aggregate67/96 versus
incumbent64/96. Paired-map difference95% interval[-.072917,.125]. Incumbent
match remains39–25 over64, with additional-map subset16–16. All256 trace
hashes, protocols, checkpoint identities and paired/native coverage verified.
Longest recorded win1743 actions and loss1693 actions both replayed through the
frozen simulator: all3436 public observations/actions/outcomes matched.
Qualification:`seed2-control16-qualification.json`.
Immutable bundle`seed2-lowrate16-a05a132eacf8` copied locally and uploaded to the
existing Modal volume without overwrite; all3 remote files read back with
matching hashes (`seed2-lowrate16-qualified-bundle-upload.json`).

Champion512 update96 passed38 archived source hashes,4 current inference
hashes,2 finite/checkpoint hashes and12 legal host decisions; Adam29983,
policy L2 from its native96 parent9.5433225. Broader qualification now includes
teacher original+replication, random, incumbent64 and three common references.
No promotion or new cloud allocation yet.

H24 matched strength result16–16 versus L40S512/control16. All32 trace hashes,
checkpoint identities, paired coverage and native endings verified; paired-map
95% interval[0.34375, 0.65625]. No strength gain established; combined with higher measured
update time and estimated cost, no long A100 allocation.


Champion512 update96 expanded match completed46–18 versus incumbent12:
original maps26–6, additional maps20–12. All64 trace hashes, checkpoint identities,
paired/native coverage verified. Paired-map95% intervals: original[.6875,.9375],
additional[.53125,.75], overall[.640625,.796875]. This is the strongest direct
improvement so far. Teacher original31/32 and random32/32 completed; replication
and common-reference qualification continue. Started stochastic-versus-
deterministic32-game mode comparison and stochastic teacher32-game evaluation
on development maps, seed20261019. No final-map use or default promotion yet.
Lowrate-noentropy32 expanded match finished29–35, verified64 traces; its7/8
screen did not establish improvement over the broader map set.

Standard-rate frozen branch update80 finished15–17 versus incumbent, verified
32 traces, paired-map95%[.3125,.625], following matched-control16 at10–22 and
weak later screens. To reallocate budget, confirmed its durable call pending,
verified latest remote96 metadata against locally retained hash882dd68ecff3,
then stopped its exact owned app. Fresh stopped/tasks0 receipt accounted$7.17;
commitment now$420.08, headroom$79.92. Removed it from hot screening roster and
stopped exact owned watcher55178. All snapshots and traces retained. This is an
allocation choice, not evidence that frozen rivals are universally ineffective.

At12:08:46UTC all9 remaining long workers passed fresh history checks:26,624,000
learner decisions,41,842 native completions,10 engine truncations; zero invalid
actions, resignations, loop exhaustions or uncredited outcomes. Finite diagnostic
checks and <=1e-5 replay checks passed. Immutable history snapshots and source
hashes recorded in`active-training-health-20261018T120846Z.json`; fresh owned-app
receipt separately confirmed all9 live with1 task each. Training outcome totals
are not held-out strength or isolated opponent win rates.

Started expanded64-game checks of older long branches longhorizon336 and
noentropy352 after each screened6/8. Outputs:
`match-longhorizon336-vs-champion51212-expanded-dev/` and
`match-noentropy352-vs-champion51212-expanded-dev/`. No allocation decision yet.


Champion512 update96 completed qualification: teacher31+32=63/64 (all63
learned captures), random32/32, and common learned references21–11,23–9,26–6
(total70/96 versus update12's64/96). All256 trace hashes and paired/native
coverage verified. Longest win1413 actions and loss2301 actions replayed every
public observation/action/outcome. Promoted as current development candidate,
not final accepted, based on46–18 direct improvement including20–12 additional
maps. Immutable bundle`champion51296-05c414466823` uploaded without overwrite;
all3 remote files read back and matched. `current-development-candidate.json`
pins the selection and qualification hash. Original and replication teacher
hashes match the earlier incumbent's baselines. Documentation examples now
load update96. No final maps used.

H25 submitted two bounded L40S continuations from that bundle, seed20261031:
`scale-champion96-ratecontrol-v1` app ap-fmVDOnorrLifOB2YhNIy2r,
call fc-01M57F04662QWSS8THE782JHFF; `scale-champion96-lowrate-v1`
app ap-tZitdTqDCE61XdmtkGhR3C, call fc-01M57F18W68W05XNMC6F2FM3A3.
Each reserves$35 with39600-second soft/43200-second hard limit, no retries.
Both initial hashes equal05c414466823; all39 archived source files verified,
2048 training maps match, Config differs only in rate1e-4 versus3e-5. Critic
and Adam preserved. Update32 in each contains1048576 learner decisions and
8192 new Adam steps (29983→38175). Exact initialization, source-change audit,
execution and work reports are saved under h25-* and champion51296-continuation-*.
Separate fixed-parent96 screening preserves the earlier fixed-parent12 monitor.

The update96 stochastic comparison completed12–20 against deterministic on32
games; all32 traces verified, stochastic-win paired-map95%[.25,.5]. Stochastic
teacher32/32 versus deterministic31/32. Retain deterministic default based on
direct play; one sampling seed20261019 is not universal inference-mode proof.
Current candidate metadata records this comparison as completed, without
freezing final-test mode yet.

Older longhorizon336 completed38–26 versus former update12, split19–13 on each
map subset, overall paired-map95%[.5,.6875]. Noentropy352 completed36–28,
split20–12 and16–16, interval[.453125,.671875]. All128 trace hashes and outcomes
verified. Neither establishes superiority to the new update96 candidate.

At14:22:52UTC provider billing reported$182.51750005 for all46 owned apps
through14:00UTC; current partial hour and separately attributed storage/egress
excluded, and the report may lag. Fresh app/call observations confirmed all11
training workers live. All11 new history audits passed numerical/action/credit
checks, pinned in active-training-health-20261018T142255Z.json. H25 had98
updates each when observed; execution qualification is separate from strength.

The laptop rebooted around14:24UTC (uptime2 minutes at14:26; prior process
handles absent). No cloud jobs restarted: all11 exact app/call handles were
rechecked live. Restored one local checkpoint/teacher watcher for all11 runs
(recovered_all_1426) and fixed-reference match watchers champion51212v5 and
champion51296v2. Four interrupted full comparisons had1/1/1/2 completed games;
original folders preserved. Explicit recovery runs use new -recovery1 outputs:
H25lowrate32 versus parent96, original champion128 versus96, noentropy592
versus96 (64 games each), and H25lowrate32 versus matched standard32 on fresh
300300–300315 (32 games). These are process-loss recoveries, not timeout retries.

Added bounded CPU cloud retention because training writes overwrite latest
checkpoints and local retention otherwise depends on laptop uptime. The helper
reuses tested hash/source-verified capture; source volume is mounted read-only,
output volume is separately named polytopia-structured-rl-retained. One CPU,
2GiB,39600-second soft/43200-second hard limits, one container, no retries.
Reserve$2 at conservative$.07/hour (listed CPU+RAM$.063144/hour), including
terminal$1 margin. Budget accounting explicitly recognizes only this declared
CPU profile; legacy/GPU floors and A100 protections remain. All14 budget and
capture tests passed. No PPO/inference/engine change; only budget accounting
changed among archived training files, and already-running jobs are unchanged.
Sources checked2026-10-18: https://modal.com/pricing and
https://modal.com/docs/guide/volumes (read-only mounts, reloads and commits).

Retainer submitted as checkpoint-retainer-v1, app ap-S5rPLebeSK5kSduvydhnSi,
call fc-01M57PQAB6NJ7E6JH667FDR23A, monitoring only the11 ledger-owned training
IDs. Total commitment$492.08, headroom$7.92; reservations are not spend.
Initial remote retention/readback validation is pending.


Cloud retainer first pass captured11 snapshots without errors. Independent
SDK readback verified every model against its stored SHA-256. Subsequent status
showed13 snapshots, confirming ongoing retention of changing checkpoints.
Round-tripped H25lower-rate update96 (74f0b03e81c8) from the separate retained
volume into retainer-roundtrip/:39 archived source files,4 current inference
hashes,2 finite/hash-matched checkpoints and12 legal host decisions passed.
Adam54559, actor L2 change3.4708959. Reports:
checkpoint-retainer-v1-readback-verification.json,
checkpoint-retainer-v1-roundtrip.json and corresponding inference log.
This verifies usable restoration as well as storage; no training process or
training-volume file was changed by the retainer.


Recovered full evaluations exactly reproduced all5 completed pre-reboot games:
6910 decision records identical, including observations, commands, logp/value
and endings; only footer elapsed seconds differ. Protocols also identical.
`reboot-evaluation-recovery-verification.json`. Interrupted originals are not
counted again in full results. H25lowrate32 versus standard32 completed19–13
on fresh32 games, all hashes and coverage verified, paired-map95%[.46875,.71875].
Both used1048576 decisions and8192 new Adam steps; one seed does not establish
rate superiority. Parent96 expanded match remains running. Started deterministic
teacher32 and random32 checks plus full checkpoint/source/finite/legal checks
for H25lowrate32, which showed promise against the stronger parent.

Cloud-retainer qualification is summarized in checkpoint-retainer-v1-qualification.json,
including14 passing tests, exact source hashes/resource limits, remote11-model
readback, growing snapshot count and complete local restoration/inference.


Recovered H25 lower-rate update32 completed **43–21 against parent96** on
300200–300231 (23–9 original maps,20–12 additional). All64 trace hashes and
paired coverage verified; paired-map bootstrap95%[.578125,.765625]. Its teacher
and random checks each completed32/32, all64 traces verified. Source checks
passed39 archived files,4 current inference files,2 finite checkpoints and12
legal decisions. Original champion update128 finished30–34 against96;
old zero-entropy update592 finished37–27, interval[.484375,.671875]. These
remain development comparisons; low32 has not yet replaced the default96.

Terminal overhead consolidation is recorded in shared-overhead-pooling-result.json.
Two stable provider reports and previously captured stopped-app receipts cover
34 terminal applications. Full app-wall compute remains conservatively bounded,
with hourly rates rounded upward and two seconds of timestamp padding. Their
separate margins were consolidated into a reserved$5 campaign overhead buffer;
no live allocation was reduced. Released$33.77 minus new$5 buffer moved total
commitments from$492.08 to$463.31. This is conservative budget accounting,
not provider spend. One terminal frozen-opponent job remains unpooled because
the earlier report did not yet cover its stop time. Actual latest observed
provider billing remains$182.51750005 through14:00UTC, subject to reporting lag.

The user requested cloud-only comparisons in the concurrent comparison chat.
That migration stopped local inference/download watchers and submitted a
$12 bounded Modal CPU screening service, cloud-screens-20261018. This campaign
will not restart local inference or checkpoint-download watchers. Existing local
results remain usable; newly launched evaluations and replay checks execute
on Modal and only small reports are fetched to the Mac.

Audited910 recorded protocols: none used300400–300431. Froze current96,
H25low32 and old noentropy592 before testing three64-game pairings on those
fresh development maps. Added low32 teacher replication300100–115, three32-game
common-reference matches300200–215, and two32-game stochastic-mode checks.
The complete384-game plan is h25-low32-cloud-qualification-v1-plan.json.
One6CPU/16GiB Modal worker,7200-second hard limit,6600-second scheduling horizon,
no retries, reserved$3. App ap-BOiLgdOYR6e6zMcGB7r6HB,
call fc-01M57QQ4BY4KC7BT8ND0BV6WDR. It verifies exact checkpoint hashes before
inference, all output trace hashes/coverage, paired-map intervals, and the
longest native win/loss in the fresh low32/parent96 match. Raw traces stay in
polytopia-structured-rl-retained under qualifications/. No final-test seeds used.
Current commitments$478.31 including the separate cloud screening reservation;
headroom$21.69, not actual unspent invoice balance. No H26 training submitted.

Budget terminal accounting now explicitly recognizes exactly6CPU/16GiB/noGPU
cloud evaluation profiles at a conservative$.42/hour floor. Other/GPU floors
remain protected.23 focused budget/retention tests passed; all22 frozen engine
hashes unchanged. Already-running training code and weights are untouched.


Cloud qualification v1 exposed a runtime-allocation mismatch: initial games
required81.01 then52.74 seconds, so the serial384-game plan was too large for
its1800-second match/7200-second worker limits. Preserved four completed-game
metadata rows and their cloud traces, then explicitly stopped only that CPU
qualification app. Its fresh stopped/tasks0 receipt bounds app-wall compute
plus margin at$1.04 (281 seconds, conservative$.42/hour); not an invoice.
No GPU training or cloud screen worker was stopped. Partial games are excluded
from the replacement comparison, and this is a measured orchestration change,
not a retry after a log-poll timeout.

Replacement h25-low32-cloud-qualification-v2 freezes the same checkpoints,
map sets and sampling modes. Three independent evaluator lanes each use two
CPU affinity slots within one6CPU/16GiB container. Hard limit10800 seconds,
scheduling horizon9900, each matchup at most3600; each run still reserves$3.
All inference and representative replay subprocesses remain on Modal. Completed
job evidence and exceptions are committed separately; incomplete jobs cannot
pass qualification. App ap-m1zhfyrJgMkvLKt8JCN8fs,
call fc-01M57QZM8JFAWEYPZX48WPDRA7. Frozen22-source hash check passed at startup.
Total campaign commitments$479.35 after v1 terminal accounting, headroom$20.65;
no new long training allocation. Actual provider spend is a separate metric.


The v2 affinity wrapper exposed an import-path bug before any learned-match
inference: runpy did not add /workspace/scripts to sys.path, so the sibling
module evaluate_structured_bot could not be imported. Both teacher jobs are
unaffected and continue in v2. Fixed the wrapper to reproduce Python script
import behavior; a subprocess smoke test passed sibling-module import and
arguments/paths containing spaces, with affinity mocked on macOS and no local
checkpoint/simulator inference. Evidence: cloud-qualification-wrapper-validation.json.
Only the seven failed learned-match jobs (320 games) were submitted under
h25-low32-cloud-qualification-v3; the64 teacher games remain assigned to v2.
Inputs and match seeds are unchanged. Worker bounds remain6CPU/16GiB,
10800 seconds,$3. No GPU job or working teacher evaluator was restarted.


Prepared final acceptance evidence validation while cloud comparisons continue.
rl/acceptance.py and scripts/summarize_structured_acceptance.py require exactly
64 paired games per opponent on the declared final32-map partition, one frozen
checkpoint/mode/sampling seed, unchanged evaluator/teacher hashes,14x14/no-resign
configuration, all trace hashes, native-outcome-consistent summaries and complete
longest native win/loss replay receipts. Engine truncations remain unresolved
in the full64-game denominator. The check reports thresholds80%random/50%teacher,
Wilson intervals, action/expansion diagnostics and provenance. It explicitly
certifies only the strength milestone, not full-budget completion or source audit.
15 focused tests passed using synthetic evidence only, including corruption,
duplicate/missing coverage, mode/model/evaluator changes, false terminal rewards,
cutoffs and truncated-game accounting. The broader budget/retention checks also
passed (38 tests before the added truncation case);22 engine hashes unchanged.
No actual final-test evaluation or final mode selection has occurred.


At15:00:44UTC, completed-hour provider billing reported$211.17832353 through
15:00UTC for all52 owned apps. This excludes the current partial hour and
separately attributed storage/egress, may lag, and is not an invoice. Registered
migration verification app ap-izPlDmGfs8kNewT9p83j26 in the campaign ledger
(retrospective discovery from the concurrent migration log); exactly1CPU/2GiB,
1800-second cap. Fresh stopped/tasks0 receipt bounds its133-second app lifetime
plus margin at$1.01. Added a narrowly validated CPU verification profile to
budget accounting. The replaced standalone retainer ap-S5rPLebeSK5kSduvydhnSi
was also confirmed stopped:968 seconds, conservative bound$1.02. No live
reservation was reduced. Total commitments$482.38, headroom$17.62; full spend
and final acceptance remain outstanding.40 budget/retention/acceptance tests
passed, using synthetic acceptance evidence rather than final simulator games.


Added larger development challenges from two later screened checkpoints:
H25 standard-rate112 (7a295abf0a11f6f84199b95b3f3ebe5700c8b54085ddb8dea907639287d0a9da),
which screened6/8 versus champion96, and H15 lower-rate224
(57c694175318b8e1a130b93f5310bc888a52fb6be17f3974eea6a829d328f67d),
which screened7/8 versus champion12. Each now plays64 paired games against
champion96 and64 against H25low32 on300400–300431. This extends the ongoing
development portfolio; it is not a newly untouched final set. Verified all
four current inference-source hashes against both training protocols; no model
was downloaded. Worker also validates finite tensors, exact model/metadata
hashes, parameter counts and Adam steps before playing. App
ap-rJmz2HpeB1hkXs6khiS5cV, call fc-01M57RPGF6SQ7Z5F2JHDQX6QN9,
run later-challengers-cloud-qualification-v1;256 games, three evaluator lanes,
10800-second hard cap,$3 reservation. Full source/optimizer qualification and
teacher checks are still required for promotion of either later checkpoint.

V2 baseline worker ta-01M57QZN98HH9G7R5H42J5CBYR disappeared at15:03:14UTC.
Modal's server log explicitly says the in-progress input was rescheduled;
the new worker then failed the existing-output guard. The durable call is
terminal FileExistsError, not an observation timeout. Underlying disappearance
cause is unconfirmed. Both partial teacher batches contain25 completed games;
raw traces remain in the cloud and their metadata was retained locally.
The old app's fresh stopped/tasks0 receipt bounds697 seconds plus margin at
$1.09; recorded that terminal bound before funding recovery.

Official Modal documentation confirms container crashes can reschedule inputs
independently of application retry settings, and non-preemptible CPU Functions
carry3x CPU/RAM pricing (https://modal.com/docs/guide/retries and
https://modal.com/docs/guide/preemption; checked2026-10-18). A separate bounded
recovery reruns only the two32-game teacher protocols, with unchanged checkpoint,
seeds and modes. Partial originals are excluded from totals. Resource profile:
4CPU/12GiB, nonpreemptible=True,3600-second hard cap,3300-second scheduling
horizon, two evaluator lanes,$2 reservation. Listed combined compute$.853632/hour,
conservative$.86/hour accounting includes the multiplier. This does not prevent
all possible infrastructure failures; the same no-overwrite guard remains.
No learned-match or GPU training worker was restarted. Budget tests verify
that the multiplier cannot be dropped;23 focused budget tests passed.


At15:11:50UTC all11 exact GPU call handles were pending and every training
history advanced since14:22:55. Their current phase histories total71,950,336
learner decisions. Checked every recorded metric for finiteness, exact iteration/
decision counts, cumulative versus per-batch outcomes, native win/loss credit,
zero illegal/resign/loop-exhaust/uncredited counters, and replay logp<=1e-5.
All passed. Recent16-update critic-probe standard deviations remain positive
(roughly.0075–.066 across runs). Memory peaks remain within recorded limits
where telemetry exists; the two older128-lane workers do not record that field.
The first audit attempt incorrectly assumed memory telemetry existed everywhere;
corrected this audit assumption without modifying or restarting training.
Pinned raw histories and source hashes in active-training-health-20261018T151150Z.json;
training-progress-1422-to-1511.json records decision deltas. These are execution
and numerical checks, not held-out policy-strength claims.


Bounded baseline recovery completed successfully. Deterministic low32 won32/32
on replication seeds300100–300115; stochastic low32 also won32/32 on300000–315.
All games terminated natively, with no cutoffs or engine truncations. Remote
verification covers64 trace hashes and exact paired coverage. Independently
fetched only protocols/game manifests/summaries/verification receipts, checked
model/evaluator/teacher hashes and sampling modes, and verified their manifest
hashes against the remote receipts. Revalidated original deterministic32-game
metadata on300000–315: combined deterministic teacher score is64/64 on disjoint
sets, excluding both interrupted25-game batches. Original random score remains
32/32. Report: h25-low32-baseline-recovery-verification.json. Replication averaged
9.3125 maximum cities and recorded1,638 learned attack actions; stochastic check
averaged9.03125 cities with1,533 attacks. These remain development metrics.

Recovery app ap-nH0VsNTIhumZBC5zQwzbsN stopped with zero tasks. Its549-second
full app lifetime at the premium-inclusive$.86/hour plus margin is bounded at
$1.14; total commitments now$484.61, headroom$15.39. Provider spend remains the
separate$211.17832353 completed-hour observation through15:00UTC. Learned-opponent
qualification and checkpoint-specific mode comparison remain in flight; no
promotion or final acceptance was performed.


Fresh300400–300431 portfolio: low32/incumbent96 33–31 (paired-map95%
42.1875–60.9375%), low32/noentropy592 44–20 (57.8125–79.6875%),
noentropy592/incumbent96 28–36 (31.25–56.25%). All192 games ended natively,
zero cutoffs/truncations;192 trace hashes verified remotely. Longest low32/
incumbent win/loss replayed3277 public actions. Earlier43–21 has not
convincingly replicated. Incumbent remains unchanged pending full portfolio.

GPU arena pilot ap-Zz6BW5A8JfvThMHC1v7SCQ / fc-01M57T4Y0C4AFZ8FP5N9Q8X4WS
reserves$5 for1L40S/8CPU/32GiB,3600s hard cap. Cold64-game match reproduced
43–21 in206.26s, zero invalid/resign/truncation/cutoff counters. Remaining
exact parity, repeated-trace, full public-adapter and speed gates pending.
Committed budget$489.61 is not spend; billed observation remains$211.17832353
through15:00. All inference stays on Modal.

User explicitly reaffirmed self-play. Next training allocation is100% learned
opponents:50% current,50% historical, no scripted/random training opponents
and no new imitation labels. Existing checkpoint ancestry still includes BC.
Plan champion96-selfplayonly-v1 requests16 updates,512x64,lr3e-5,seed20261031
from incumbent96, matching H25 lowrate initialization/config except opponent
mixture. Bounded2400s training/3600s hard cap,$5 reservation. Existing running
mixed-opponent experiments remain comparison controls; new allocations use
self-play. No claim of from-scratch pure RL or fresh strength before evaluation.


Self-play pilot is live as ap-ed8hAT06LEQg9azpvEJMd0 /
fc-01M57TENWMM2J0XZB2ZV69K3WP. Verified its remote protocol, all41 archived
source hashes, exact initial checkpoint hash matching incumbent96, native
reward, zero public/random/frozen probabilities, bc_epochs=0 and restored
critic/Adam configuration. First32,768 learner decisions have finite metrics,
zero illegal/resign/loop-exhaust/uncredited counters and replay logp3.10e-6.
No games completed in this first rollout, so it supplies no win-rate evidence.
Fifteen opponent-selection unit tests passed without local model inference.
Commitments$494.61 include the new$5 cap; actual spend remains separately
reported. Full self-play strength and continuation allocation remain pending.

Prepared an11-checkpoint GPU screening roster from immutable cloud snapshots,
verifying each checkpoint metadata hash and four current inference source
files. The fixed roster covers latest retained checkpoints of all11 long
branches, tested against incumbent96 on300500–300531 if the GPU path qualifies.
912 available local protocols and both current cloud qualification plans have
no overlapping declared ranges. This is a draft, not submitted or reserved:
gpu-portfolio-fresh300500-v1-draft-plan.json. New worker
experiments/modal_structured_arena_screen.py requires the exact successful
pilot receipt and unchanged qualified-path source hashes before evaluation.
The current pilot has matched all64 CPU outcomes/action lengths, repeated GPU
traces exactly, and passed its first1663-action public-adapter audit; second
audit and final speed/qualification decision are still pending.


GPU evaluation path qualified: all64 CPU reference outcomes/action lengths
matched exactly, warm GPU traces matched exactly, both longest-win/loss public
adapter re-selection audits passed (1663+1486=3149 actions). Features matched
exactly; maximum logp error2.15e-6 and value error2.39e-7. Projected per64-game
match over11 sequential comparisons is163.35s,4.678x faster than the recorded
764.17s CPU run; the latter includes historical machine load and is not a
controlled hardware microbenchmark. Scope remains deterministic learned-policy
screening, with candidate-specific full public qualification before promotion.

Submitted frozen11-checkpoint screen as ap-w8E4GwL27vd4YsmCcxzg2V /
fc-01M57TSW6C72CF2FZK4DE5VMF5,1L40S8CPU32GiB,$5/3600s hard cap. Exact plan
and successful pilot receipt/source hashes are checked in the cloud before
playing300500–300531 paired-seat games. GPU pilot app independently confirmed
stopped/tasks0 after660s, accounting bound$1.48. Commitments after new
screen and terminal accounting are$496.09, headroom$3.91,
including the existing$5 shared overhead buffer. Reservations are not spend.

H25 low32 completed common references:24–8 native96,22–10 gamma48,27–5 noentropy64, total73/96 versus incumbent70/96. Exact shared reference hashes, inference modes, map/seat coverage and manifests verified. Paired-map bootstrap of the win-fraction improvement (preserving six games per map) is[-0.05208333333333333, 0.12499999999999999]; no established common-reference advantage. First metadata audit used incorrect flat protocol keys, corrected to the existing nested schema without changing game evidence. All24 small JSON artifacts for six completed comparisons retained locally; raw traces stay cloud-side.


15:44 billing refresh confirms$211.17832353 through15:00UTC,52 owned apps
reported out of57 now registered; five newer apps have no rows in that
completed-hour window. Exact provider report saved as
modal-billing-owned-20261018T154423Z.json. Comparison with the14:22 observation
(stable, more than1hour apart, fully covering the stopped app) allowed pooling
only the standard frozen-rival branch's duplicated overhead margin:7.17→6.18
compute bound, stable provider cost6.09633613, two-second timestamp padding
and rounded-up hourly rate. Other recent terminal apps remain ineligible.
Shared overhead buffer remains$5; released$0.99, total commitments$495.10.
No live reservation was reduced and no reservation is counted as spend.

At15:43 self-play pilot reached4 updates/131,072 decisions and32 native
completions (3wins,29losses), zero truncations/illegal/resign/uncredited/
loop-exhaust counters, max logp3.10e-6. This early completion subset is not
an isolated strength evaluation. Later-challenger CPU jobs were still
progressing at54,56,53/64 games respectively; no persisted errors, cutoffs
or truncations. Fourth match remained queued, not failed.


First fresh300500 GPU screen completed: H25 lowrate144
8150ea3b8468771c4ebf552158fbda5cfc38f82db00066e68ae6a758fae126b1
beat incumbent96 44–20/64; paired-map95%[57.8125%,79.6875%]. No invalid,
resign, engine-truncation or runtime-cutoff counters. All model/optimizer tensors
finite;519267 parameters,Adam66847 (=29983+144*256).64,056 atomic actions,
maximum2267/game,264.30s cold match. Promotion remains ineligible from screen
alone. Other10 candidates continue under the existing bounded job.

Submitted candidate-specific public-API qualification as
ap-N10utK3JBkIjc5v4hgzGw2 / fc-01M57V67TZ2SR3HCYFR3VR4Y0C. Three regular
CPU lanes on6CPU16GiB,10800s hard cap,$3 reservation; this is not the premium
baseline-recovery function.288 games:64 exact GPU-reference map repeats for
CPU parity;64 against incumbent on new300600–300631;64 against low32 on
the same new partition;64 teacher across original+replication sets;32 random.
The two incumbent matches include full longest-win/loss replay receipts. CPU
parity repeats are not counted as independent strength replication. Available
protocol map-seed metadata was checked before declaring the fresh partition.
The qualification launcher now honors explicit balanced/random baseline jobs,
rejecting other opponent names; default balanced behavior stays identical.
No model inference or checkpoint download was performed locally. A qualified
checkpoint would be eligible as initialization for100% learned-policy self-play,
not a claim that its mixed-opponent ancestry was itself100% self-play.
Commitments now$498.10; provider spend remains separately reported.

Second fresh GPU screen: H25 standard-rate144 scored33–31/64 against incumbent96, versus lower-rate14444–20 on identical maps/seats. Both collected4,718,592 phase learner decisions. Adam counts low=66847, standard=66557. Paired-map bootstrap of the17.1875percentage-point win-rate difference: [0.0, 0.328125]. Both had zero truncations/cutoffs. This is shared-reference GPU screening, not a direct matchup or completed public qualification.

H25 low32 qualification completed all320 requested learned-policy/mode games, with remote hashes verified and no errors. Final stochastic-versus-deterministic mode check was9–23/32 (one sampling seed); exact checkpoint identity, modes, manifest hashes and summary verified. Thus deterministic stays preferred, but the33–31 fresh incumbent result and common-reference interval still leave overall improvement uncertain; no incumbent promotion.

Concluded original mixed-opponent gamma .999 branch after GPU update656 screen29–35. Preserved its latest committed update688 SHA5c5d7b4432d6a10550db58c65967e531654935f3dfd0c0797868ef6998a4cbec in the cloud retainer before stopping exact app ap-Fz6qYSCZzLcnXDR4nuMOcC. Training/retained metadata matched byte-for-byte and retained NPZ presence was verified. This is an allocation decision favoring upcoming100% self-play, not proof of statistical inferiority.
Terminal accounting from fresh stopped/tasks0 receipts: [{"name": "scale-bootstrap-longhorizon1", "bound": 21.97, "receipt": {"app_id": "ap-Fz6qYSCZzLcnXDR4nuMOcC", "description": "polytopia-structured-scale", "state": "stopped", "tasks": "0", "created_at": "2026-10-18 00:49:00-07:00", "stopped_at": "2026-10-18 08:55:46-07:00"}}, {"name": "h25-low32-cloud-qualification-v3", "bound": 1.42, "receipt": {"app_id": "ap-sn6vPF5YssbDkYX5SwC1pW", "description": "polytopia-development-qualification", "state": "stopped", "tasks": "0", "created_at": "2026-10-18 07:54:04-07:00", "stopped_at": "2026-10-18 08:53:31-07:00"}}]
Commitments$483.49, headroom$16.51; actual spend remains separately observed.


Additional completed results: H15 lowrate224 versus incumbent96 scored41–23
on fresh300400–300431, but31–33 against H25 low32 on those same maps. H25
standard112 also scored41–23 against incumbent; its low32 match remains in
flight. These results reinforce keeping multiple styles/checkpoints and not
assuming a transitive ranking from one incumbent matchup.

Original no-entropy branch's update768 GPU screen finished31–33/64 against
incumbent96 on300500–300531, zero truncations/cutoffs. Preserved latest
update816 SHA26b30d7f5aa7ecec36b30f1e67380ea9af771d6f435c3441984675538aa820ee
in the cloud retainer, matching published metadata byte-for-byte and confirming
NPZ presence. Stop requested for exact app ap-CkkwRJszrI3VzASuhe3aqH to
redirect budget toward100% self-play. This is an allocation decision, not a
claim of statistically demonstrated inferiority. Reservation remains until a
fresh stopped/tasks0 receipt permits terminal accounting.

Self-play pilot reached13updates/425984 decisions,525 native completions
(217wins308losses), no truncations/illegal/resign/uncredited/loop-exhaust
counters, replay max3.82e-6. Training outcomes remain distinct from held-out
strength. No final maps or incumbent promotion performed.

Matched-ablation limitation: H25 lowrate retained update8 and32 onward but no update16 checkpoint in either local metadata or the verified cloud migration manifest. Therefore the selfplay16 pilot can be compared to its unchanged parent and retained low32, but low32 is a different sample count. Do not claim a sample-matched selfplay16/mixed16 experiment. Later selfplay32 can match the existing mixed32 if initialized continuously from the same parent/seed; a restart-at16 would introduce additional episode/RNG/history confounds.
Fresh no-entropy app status: stopped, tasks0; terminal bound=21.76. Total commitments$470.25, headroom$29.75.


At16:01:39 verified all9 remaining long GPU histories and the self-play pilot:
72,613,888 phase learner decisions combined, every metric finite, every
batch/cumulative counter consistent, zero illegal/resign/loop-exhaust/
uncredited counters, all replay logp<=1e-5, memory peaks within recorded
limits. Exact long-run calls pending; pilot reached16 updates/524288 decisions.
Saved raw histories with hashes in active-training-health-20261018T160139Z.json.

Self-play pilot final SHA808f05cc623638ecbd8a4739d75f1f64c281fdd49943ff77ff74eede6cc3bbe5,
519267 finite parameters,Adam34079 (=29983+4096),1393.87s training. Its first
built-in stochastic random evaluation finished32/32 native wins, no illegal
actions/truncations/cutoffs. Full352-game public qualification launched as
ap-ykdt9vyqDbvlwCmBhnuXOc / fc-01M57VWXBX03YE0EX04CYVK90C.
Three standard CPU lanes6CPU16GiB,$3/10800s cap. Fresh300700–300731 parent
and low32 matches, teacher64,random32,common references96,and checkpoint mode32.
The low32 comparison is not sample matched. No final-test maps are used.

Completed-hour provider billing at16:00:58 is$242.96827570 through16:00UTC,
58/59 registered apps have rows; the new self-play qualification starts after
that window. Stable reports observed more than1hour apart permit pooling only
three eligible old terminal overhead margins, releasing$3, preserving the
shared$5 buffer. Before new long/retainer allocations commitments$470.25.

Added validated explicit2/4/8/12-hour hard-cap profiles to the scale launcher,
reserving$7/$12/$23/$35 and leaving at least1hour outside requested training.
Default12-hour behavior remains. Four valid profiles satisfy full hard-wall
compute+>=1margin; four invalid/boundary requests reject. Installed Modal SDK
confirms timeout override support; no model inference in these local checks.

Launched100% self-play long run scale-champion96-selfplay-v1 as
ap-rEmqnLKLw7ykJUBS6W2SKd / fc-01M57W075W7FF5384AR3HFJCSC. Up to25200s
training,28800s hard cap,$23. Same incumbent96 parent,seed20261031,512x64,
2048 maps,lr3e-5,gamma.995,entropy.005,restored critic/Adam. Zero public/random/
frozen opponent probabilities and zero new labels. It deliberately restarts
from the original parent to reproduce the short pilot prefix, then continues
without a mid-run state/RNG/history reset. Must verify update16 prefix and
retain update32 for a truly matched comparison to mixed32. The existing full
public qualification remains pending, so this is an explicitly bounded
self-play training hypothesis, not a claim of demonstrated improvement.

Dedicated CPU retainer selfplay-checkpoint-retainer-v1 launched as
ap-IWTgBSjoKN3MIkAf2S2sn0 / fc-01M57W0QE1ZPP71F86W0F254BA,1CPU2GiB,
30600s requested/43200s hard cap,$2. Watches the short and long self-play runs
every60s, read-only training volume, independently verified copies in retained
volume. First capture already retained the short pilot; no reported errors.
Total commitments$495.25,headroom$4.75; these are not actual spend.


Long self-play remote initialization verified: identical parent/initial NPZ
05c414466823..., all41 source hashes identical to the short pilot and current
checkout, identical training seeds; Config differs only in requested updates,
training wall allocation and end-of-run evaluation action cap. Zero scripted/
random/frozen probabilities and bc_epochs=0, critic/Adam preserved. Exact
update16 trajectory/weight-prefix equality is still pending production of
that checkpoint; do not assume it from config alone.

Short pilot now completed both built-in random modes:32/32 stochastic and
32/32 deterministic, all native, no cutoffs/truncations/illegal actions. These
share map seeds and are separate inference-mode checks, not64 independent
maps or final acceptance. Fresh terminal receipt bounds1658s at$2.20, freeing
$2.80 from its$5 reservation. Full public qualification remains in flight.

GPU portfolio now includes frozen-rival lowrate24044–20 against incumbent96,
but frozen-rival seed2update22428–36; do not infer a replicated recipe gain.
Submitted two-candidate public qualification as
ap-LV6grxqdKBqNrf9lyFAsdb / fc-01M57W4Y76TZZARJC59539SAA5.
portfolio-finalists-cloud-qualification-v1 freezes frozen240 and standard112
(the latter had public41–23 on300400).576 games: for each candidate, teacher64,
random32,parent64 on300600,common references96,and low14432 on300600.
Three regular CPU lanes6CPU16GiB,$3/10800s cap; plan frozen before reading
either candidate's300600 outcomes. These are potential stronger self-play
initializations, not new mixed-opponent training. Deterministic/stochastic mode
qualification is still required if either is selected.

H25 low144 original teacher check finished32/32 native wins (461.78s), no
cutoffs/truncations; replication, random and learned comparisons remain live.
After new finalist evaluation and pilot terminal accounting commitments are
$495.45, headroom$4.55; provider spend remains$242.96827570 through16:00.

### 16:10 UTC: live checks and qualification receipt audit

Exact durable handles for the GPU portfolio, four public qualification jobs,
long self-play run and dedicated retainer are all pending (live). The new long
self-play run has completed three updates / 98,304 learner decisions, with
zero illegal actions, resignations, truncations, loop exhaustion or uncredited
outcomes. The retainer reports no errors. Frozen engine verification again
passes all 22 hashes. Reservations remain $495.45; this is not spending.

Independently fetched and checked the three completed later-challenger match
manifests: 192 games, exact planned checkpoint hashes, deterministic modes,
paired map coverage, evaluator source hash, raw outcome/summary agreement,
cloud trace-hash receipts and representative replay provenance. The two
41–23 parent matches have 3,954 and 3,504 fully replayed public actions;
the low224 versus low32 result is 31–33. The fourth match remains live.
Evidence: later-challengers-cloud-qualification-v1-independent-audit.json.

The long self-play run is NOT an exact reproduction of the short pilot prefix.
Across the first three updates, first-rollout counters, action counts, reward
statistics, advantage standard deviation and replay error agree exactly, but
post-update optimizer metrics already differ at update one; trajectories then
diverge at update two. Identical initial weights, sources, seed and learning
configuration do not establish bitwise reproducibility. The precise numerical
cause is not established by these scalar records; do not attribute it to a
particular GPU kernel without additional evidence. No health check fails and
no restart is warranted. Treat the long run as a separate training replicate;
its checkpoints need their own strength checks. Update32 can still be compared
at matched learner-decision count to mixed32, but cannot inherit the short
pilot's gameplay results. Evidence: selfplay-long-prefix-progress.json.

GPU screening has completed eight of eleven frozen candidates. Newly observed
noentropy176 is 39–25 and seed2-lowrate192 is 38–26 against incumbent96;
their paired-map intervals include 50%. The earlier low144 and frozen240
44–20 results remain the strongest screening leads, with separate public
qualification underway. None is promoted by the screen alone.

Stopped mixed-opponent scale-champion512-lowrate-frozen-seed2-v1 to redirect
capacity to learned-only self-play. Before stopping, verified latest committed
update256 metadata matches its retained copy, with retained NPZ present:
SHA24033019d42c5dd510eb119472fcf3ebab6c8f2b69c2cc23fb3b448bef98940d.
Fresh Modal receipt confirms stopped/tasks0,03:25:43–09:12:06PDT. Terminal
wall-time bound plus margin is$15.92, releasing$19.08 of reservation.
Commitments become$476.37 before new allocations. The decision reflects
allocation priority and the28–36 update224 screen, not proven inferiority.

Prepared the learned-only higher-rate pilot champion96-selfplay-fastlr-v1:
same parent,seed,512x64,2048 maps,gamma/entropy and sixteen requested updates
as the completed self-play pilot, with learning rate1e-4 instead of3e-5.
All training opponents are current/history learned policies; no new labels.
The plan records separate checkpoint qualification and actual optimizer-step
reporting. Launch submitted after terminal accounting, with$5 one-hour cap.

Higher-rate pilot dispatch: ap-sxVc8JVM1xmSh9A7e0ZhON /
fc-01M57WK3GSJHMYCP6G949P2157. Independent retainer dispatch:
selfplay-ratepilot-retainer-v1,ap-PY9mlaaNwx4z8dSCy2UWRl /
fc-01M57WKH5HJX6CVWYSECC4D1YF,1CPU2GiB,$2,4200 requested seconds.
At16:14:34 both exact handles are live and the retainer reports no errors.
Remote pilot initialization confirms the exact incumbent parent/initial hash,
all41 learning/simulator sources match the lower-rate pilot and checkout,
training seeds match, and learning_rate is the only Config difference.
Critic/Adam restored,bc_epochs=0,public/random/frozen probabilities all zero.
Commitments now$483.37,headroom$16.63; provider spending remains the previously
observed$242.96827570 through16:00, not an updated invoice estimate.

Updated the long-run launcher's default public-opponent probability from.25
to0, so default new PPO allocations match the user's self-play direction.
AST inspection verifies public/random/frozen probabilities all default zero;
syntax compilation passes. This changes no running allocation or learning
kernel. Updated the current guide to distinguish old mixed controls from new
self-play and to remove stale local-inference/watcher instructions. All
campaign inference, gameplay comparisons and model retention remain in Modal.

### 16:20–16:24 UTC: completed portfolio and fresh low144 evidence

GPU portfolio completed all11 candidates /704 games on the same32 paired maps,
with no truncations/cutoffs. Independently audited planned checkpoint identity,
all outcomes/coverage and recomputed every paired-map confidence interval.
Low144 and frozen240 each44–20; lowrate-noentropy17639–25; seed2-lowrate192 and
lowrate272 each38–26; original champion28834–30; H25standard14433–31;
standard256 and oldnoentropy768 each31–33; oldgamma65629–35; frozen-seed2-22428–36.
These are selected-checkpoint screens, not11 independent map partitions or
universal recipe rankings. Low144 and frozen240 share36 wins against incumbent96
and each wins8 games the other loses; this is not an ensemble score or direct
match. Terminal app wall2240s bounds$2.61, releasing$2.39 of reservation.

Low144 independent public-API fresh300600 comparison finished46–18/64,
paired-map95 interval[.625,.8125],with3,356 representative win/loss replay actions.
The public300500 parity match finished44–20/64: every outcome and action length
matches its GPU screen, with3,913 representative replay actions. This is not
all64 full policy-action reselection. All160 currently completed qualification
games (including teacher32/32) pass independent manifest/source/hash-receipt
checks. Random, second teacher partition and low32 direct comparison remain
live. The selfplay16 public teacher also finished32/32; its learned comparisons
remain live. Frozen240's first public teacher check is32/32. No qualification
worker reports an error; partial lane counts confirm ongoing game production.

Added a CPU-only cloud bundle materializer, experiments/modal_structured_bundle.py,
with read-only retained source volume, unique hash-suffixed destinations,
source/checkpoint/readback verification and no champion-promotion side effect.
portfolio-bundles-v1 (ap-AF6ccxyycPSuLTZPmKSsUZ /
fc-01M57WVNZSP5T5JRRP5R0061AP) copied low144,frozen240 and standard112 into
training-volume bundles h25-low144-8150ea3b8468,frozen240-6392b2ca69e7,
h25-standard112-7a295abf0a11. All39 archived hashes,4 current inference hashes,
finite519267-parameter models and Adam66847/69919/58512 verified. Readback
metadata/provenance match and model files are present. No model downloaded.
Terminal28s bounds$1.01 of its$2 reservation; syntax and real three-bundle
copy/readback pass. Copying does not certify playing strength.

Started the matched learned-only opponent-diversity pilots from low144:
low144-selfplayonly-v1,ap-Z4m7ffcFiTzFD8SnL84toT /
fc-01M57WYFPRF5C4FPETBJ47XFKS;
low144-selfplay-rivals-v1,ap-ziMpHhIzykhAnib7x0xFHS /
fc-01M57WYVR0ZPJS5E2J8BWKXFAK.
Each16updates,512x64,2048maps,seed20261031,lr3e-5,gamma.995,entropy.005,
2400s requested/3600 hard,$5. The second assigns25% to frozen learned
incumbent96/frozen240; remaining mass is current/history equally. Control has
no fixed rivals. Both remote initializations verify exact low144 weights,
all41 current sources, identical learning Config/map seeds, restored critic/Adam,
zero new labels and zero scripted/random probabilities. Their$2 cloud retainer
low144-selfplay-retainer-v1 is ap-6cdqQFKKmCUCvmFY9TWgEq /
fc-01M57WZ57R65MZ9F3QJ1J3M8XP; exact handle live, no reported errors.

Retired older mixed-opponent scale-champion512-ratecontrol-v1 after retaining
update288 SHA8f59d2eed38190f465c4af29c6345c46c0267d2bd5a3542e0df20884e26ccc40.
Fresh stopped/tasks0 receipt02:39:03–09:19:38PDT bounds$18.26, releasing$16.74.
This reallocates capacity following its update25631–33 screen, not a proof of
statistical inferiority. Also requested stop for original scale-champion512-v1
after verifying retained update320 SHA07297004f18c88f2302d838944857fed3fccfb0376aa0573f5bf51bb1d14dfaf.
Its screened update288 was34–30; incumbent96 and all retained history remain.
Its reservation is not released until a fresh stopped/tasks0 receipt.

Launched low144-mode-common-qualification-v1 as ap-ehzUKq3yzOCbs6Iy1rGQoZ /
fc-01M57X4R2JQMAZJYEQZVS38Q9G,$3 regular6CPU16GiB/10800s hard.
160 development games cover the three earlier common references96, a32-game
stochastic/deterministic comparison and32 stochastic teacher games. Final maps
remain untouched. Commitments after this allocation are$480.25, before any
terminal accounting for the newly stopped original champion branch. Actual
provider spending is still only known through16:00 at$242.96827570.

Original champion branch now confirmed stopped/tasks0,01:59:30–09:23:52PDT;
terminal bound$20.14 releases$14.86. Commitments are$465.39,headroom$34.61.
The new low144 mode/common qualifier's exact call is live; all four loaded
models are finite519267-parameter checkpoints, with expected Adam steps and
three regular two-CPU evaluation lanes. Engine verification again passes22/22.

At16:24 all four learned-only training calls remain live and healthy: long
incumbent self-play14updates/458752 decisions, higher-rate pilot6/196608,
low144 control2/65536 and rival-diversity2/65536. Every recorded metric finite,
zero invalid/resign/loop-exhaust/uncredited outcomes or truncations, maximum
logp mismatch below4.8e-6. Saved selfplay-health-20261018T1624Z.json.
The complete short self-play pilot history is now refreshed locally:16updates,
524288 decisions,681 native completions (301wins/380losses),zero truncations.
Training completion rates are not held-out strength estimates. No checkpoint
promotion, final-mode freeze or final400000-map evaluation has occurred.

### 16:31 UTC: complete low144 qualification and prepared final workflow

Low144's288-game public qualification completed with no errors: teacher63/64
(32/32 original,31/32 replication),random32/32,fresh parent46–18,parity parent
44–20,and low32 comparison39–25. All288 manifests now independently checked
for planned checkpoints/modes/seats/source identities, raw outcome agreement,
cloud trace-hash receipts and paired-map confidence intervals. The low32
comparison interval is[.5,.71875],so do not claim decisive superiority over
that candidate. Existing representative replays total7,269 actions.
Terminal receipt08:48:20–09:29:15PDT bounds$1.29, freeing$1.71. Commitments
are$463.68,headroom$36.32; actual billing remains the16:00 report.

Long self-play update16 has been retained as
SHA542976069a84bf7598edf45117f849521f213752fec9617ca4a74efacdc79347,
confirming it differs from short-pilot808f... at the same decision count.
Higher-rate update8 is521f997c57a6...,low144 control update4 is870f77d7c70c...,
and rival-diversity update4 is2205a1c6ac16.... All four metadata pairs match
their independent retained copies; retainer status files report no errors.
Their own gameplay qualification is still required.

Prepared the final acceptance path without playing any final games. New
scripts/structured_qualification_plan.py validates development partitions and
explicit final plans. Final plans pin the selected checkpoint/training protocol,
mode/sampling seed, runtime versions, teacher and39 relevant source hashes;
they require exactly64 teacher and64 random games with representative replays.
The existing cloud worker now checks learned-only self-play provenance, archived
sources/current inference identity and teacher/runtime hashes before claiming
the final partition. It then uses the existing full trace/replay acceptance
assessor and persists acceptance.json. Development plans cannot consume final
maps. Final32-map blocks are reserved atomically in the budget ledger and
claimed once in cloud storage; a different run cannot silently reuse one.
Reservation phase/partition identities are immutable through status updates.

55 protocol, acceptance and budget tests pass, using synthetic records and
metadata only. All10 existing saved CPU development plans validate, frozen
engine22/22 still passes, and no final ledger reservation exists. The final
workflow has not yet been exercised with real final games. Existing running
workers keep their original code. New source changes are confined to budget/
qualification orchestration, not PPO, observations, policy or simulator math;
future source comparisons must distinguish this budget.py change from learning
code changes rather than claiming every archived file remains identical.

Final preflight additionally rejects an initial/unchanged checkpoint and
requires model parameters plus Adam steps to advance from the hash-verified
initialization. It records policy delta and initial/final optimizer steps before
claiming a partition. The synthetic metadata/partition/source and accounting
suite now passes56 tests; worker/helper syntax compilation passes. This is
preparation, not a claim that real final evaluation has run or passed.

At16:33 the four self-play handles are live: long21updates/688128 decisions,
higher-rate12/393216,low144 control10/327680,and low144 rivals9/294912.
All published metrics are finite, no invalid/resign/loop-exhaust/uncredited
outcomes, no truncations, and maximum replay mismatch below5.8e-6.
Saved selfplay-health-20261018T1634Z.json (internal observed timestamps16:33).
The low144 mode/common evaluator has not yet published a completed job.
No new training/evaluation allocations were made during this final-workflow
preparation turn; commitments remain$463.68,headroom$36.32.

### 16:51 UTC: self-play pilots complete; matched-sample comparison launched

Recovered the previous turn's durable dispatches and results rather than
restarting any job. Historical GPU screen `gpu-historical-dev301000-v1`
(app ap-hkuBvhvfWaSyM58qPBiC0L, call fc-01M57Y26G92E3M3JZDYFJBXCJR)
and learned-only stage-1 screen `gpu-selfplay-stage1-dev300800-v1`
(app ap-aLfjR9zuq9PaBQr1kAAhWT, call fc-01M57YBH04WXVX6M42K0TDEQKX)
are live, each capped at one hour/$5. Their checkpoint rosters were frozen
before outcomes; GPU worker additionally verifies raw checkpoint-metadata
hashes. Historical first results versus incumbent96: low144 38–26, low128
48–16, low160 46–18 on fresh301000. Stage-1 parent control is47–17 on
fresh300800. These are development screens, not promotions or final tests.

The higher-rate learned-only16-update pilot completed524288 decisions,
checkpoint dc39b052def08a5ad3e260526c50f2da48c53d61011815d4fd1a838abb230941.
The original CPU qualifier v1 failed before any games because its plan manually
used a13-character directory prefix rather than the retained12-character prefix.
Its terminal full-app bound is$1.01. Corrected v2 uses sha[:12], a unique run ID,
and the unchanged checkpoint; app ap-qUSeHDqUDIXwWKrH7ioSrC,
call fc-01M57Y76V0BJ9B8ZK4455PKHPH. Both fast/slow checkpoints have519267
finite parameters. Fast Adam33954 versus slow34079: equal decision counts do
not imply equal optimizer work; the PPO loop supports KL-triggered early stopping.
The v2 cloud runtime receipt is Python3.12.10/JAX0.11.2/jaxlib0.11.2/NumPy2.5.4.
No local model loading or gameplay was performed.

Both low144 learned-only pilots completed16updates/524288 decisions. Control
SHA d5e6898806700cabf45b094527e494d06201b8734fc1242bd0f4084893df37c2;
persistent-neural-rivals SHA71fef239712c992471bbb061e1e36344547ad48815a522b6a29c938933c4c33c.
Both scored32/32 against random in each inference mode on the same32 games;
these are not64 independent maps and do not establish learned-opponent strength.
All training counters reconcile, all metrics finite, no invalid actions,
resignations, loop exhaustion, uncredited outcomes or truncations; maximum
behavior-logp replay error5.96e-6. Final checkpoints were retained remotely.
Fresh terminal app receipts allow bounds$2.05/$2.03 for training and$1.03 for
the pair retainer, releasing$6.89 of reservations. Training outcomes683 and699
native completions are health evidence, not held-out strength estimates.

Full live-handle health audit at16:48 covers six older mixed-opponent GPU runs
and the long learned-only run. Their updates are341/307/253/242/209/218 and33,
respectively. All histories finite and counters reconcile; no illegal, resign,
loop-exhausted or uncredited outcomes. Cumulative old mixed branches include
2/3/0/0/0/1 engine truncations respectively; the learned-only run has zero.
Nothing was restarted. Frozen engine verification remains22/22.

Low144's three common-reference jobs are complete:25–7 native96,25–7 gamma48,
30–2 noentropy64, total80/96 versus incumbent70/96. Independent metadata,
coverage, source, outcome and cloud trace-hash receipt audit passed. A paired
bootstrap resampling each map's six games gives a win-rate difference of
+10.42 percentage points with95% interval[0,20.83]; development selection and
small sample still limit the claim. Stochastic-mode jobs remain pending, so
incumbent96 remains selected. Evidence: low144-common-reference-comparison.json.

Long self-play update32 is now retained with SHA
1aec3543b1d18dd18a6fd0c967d42e75eafe328ebc7338be97ab622b3fd0710c.
Launched `selfplay32-matched-qualification-v1` on fresh301100–301131:
64 games versus parent96,64 versus mixed-opponent low32,64 teacher games,
32 random games and32 stochastic-versus-deterministic games. App
ap-7XN0d2x0FeHw1EPuyX7YPN, call fc-01M57YPVRP0JBW2X4F5F096VFV,
regular6CPU/16GiB,3h hard cap,$3 reservation. Self32 and mixed low32 have the
same parent, map pool, seed, learning configuration and1048576 learner decisions;
configured total training duration differs. This is one replicate per mixture,
not proof of general causal superiority. Four inference source hashes match.
All new training remains learned-opponent-only with explicit imitation ancestry.

Earlier later-challenger qualifier completed256 games, with final standard112
versus low32 result34–30; independent audit passed, and terminal bound$1.66.
Fast pilot/retainer bounds$2.19/$1.03 and bundle-materializer bound$1.01 were
also reconciled in the preceding turn. After this turn's terminal releases and
new$3 qualification, commitments are$468.68, leaving$31.32. Actual provider
billing remains$242.96827570 through16:00UTC; reservations are not spend.
No final partition has been reserved or consumed. The campaign remains active.

### 16:54 UTC: stronger screened ancestor enters learned-only training

The historical screen's next result is low17654–10 versus incumbent96 on
fresh301000–301031. Low144 is38–26 on those same maps. Independent JSON
identity/coverage/outcome/bootstrap audit passed for the first four candidates;
low176's paired common-opponent advantage over low144 is+25 points with95%
interval[12.5,37.5]. This is exploratory selection among12 candidates, not a
head-to-head match or final evaluation. The stage-1 self-play screen currently
has low144 parent47–17 and its learned-only16 descendant42–22 versus incumbent
on fresh300800; the paired difference interval[-20.31,+6.25] includes zero.
No claim of a self-play regression or improvement is established by that pair.
Cloud receipts confirm the descendant's Adam70943 versus parent66847.

Prepared low176 as an immutable cloud-only bundle using the existing verified
materializer:519267 finite parameters,Adam75039,39 archived source hashes,
four current inference hashes, and copied checkpoint readback all verified.
Bundle /training/h25-low176-565eb6e315c8/final.npz, SHA
565eb6e315c82b970c0d0694f448aaf5a798de26c41deb7ad9b5436de4489052.
`low176-bundle-v1` appap-aqO7o6TjQbX18horScHxnR,
callfc-01M57YVZ7K787E80FTWDGH9QXN returned complete. Its app-list receipt still
showed detached/tasks1, so its$2 reservation has NOT been released.

Launched public replication `low176-cloud-qualification-v1`,
appap-jsdbxHGeiuhDYb1zHaIHKk,callfc-01M57YW0DM2WNY8N2H4VZ7KZP4,
352 games/$3/regular6CPU16GiB/10800hard: incumbent96 and low144 each64 games
on newly declared301200–301231,teacher64,random32,common96,mode32.
The mixed-trained ancestor itself is not eligible as the final learned-only
self-play descendant. No final maps are involved.

Launched `scale-low176-selfplay-v1`,appap-FVuibq1kFrYF1uNco6L4T6,
callfc-01M57YWR36K51X4N08H028KZRA. This is learned-only current/history
self-play from the exact low176 bundle,seed20261031,512x64,pool2048,lr3e-5,
gamma.995,entropy.005,critic/Adam preserved,bc0/public0/random0/frozen0.
Request10800 training seconds,14400 hard cap,$12 reserved. Independent
retainer `low176-selfplay-retainer-v1`,appap-4cpXQM3mNEaKpO54JHOgt5,
callfc-01M57YX8ENPC40KM5SMN9HJAPZ,reserves$2 for1CPU2GiB. Both exact calls
are pending/live; protocol and initial metadata have not yet been published
at the16:53 poll. CLI log-detach timeouts are NOT job completion and did not
trigger a restart. Plan early16/32-stage checks and preserve all candidates.

Self32 and mixed low32 public preflight now confirms identical Adam38175,
in addition to1048576 phase decisions, matching architecture and initial parent.
That makes this particular pair both decision- and optimizer-count matched;
training opponent scheduling differs and one replicate cannot establish general
causality. Both tensors are finite and contain519267 parameters.

Commitments are now$487.68,headroom$12.32,including every live allocation and
the shared$5 overhead buffer. Actual provider spend remains the16:00 report,
not this commitment total. Terminal applications will be reconciled only after
fresh stopped/tasks0 receipts; another billing report is due after17:00UTC.
The incumbent remains champion512 update96; final acceptance remains unused.

### 16:55 UTC: first self-play descendant beats its parent on fresh public games

Self16 public matches completed:43–21 versus incumbent96 on fresh300700–731,
paired-map95% win interval[57.81%,76.56%];24–40 versus mixed low32 on the same
maps,interval[28.13%,45.31%]. All128 games resolve natively,0truncations/cutoffs.
Independent plan/checkpoint/mode/source/coverage/outcome/bootstrap checks pass;
cloud representative replays verify3,442 and3,392 actions respectively. This
supports improvement over its parent, but not superiority over the broader
portfolio. Low32 has twice as many continuation decisions; self32's new test is
the sample- and optimizer-count-matched comparison. Teacher replication, random,
remaining common references and mode checks remain pending. No promotion yet.

Portfolio public fresh300600 replication also completed frozen24037–27 versus
parent,paired interval[48.44%,67.19%],and standard11241–23,[53.13%,75%]. Both
independent128-game metadata/replay receipt audits pass, with3,221/3,627 public
replay actions. Other qualification jobs remain live. No errors.json has been
published by any of the six active CPU qualifiers at the16:54 observation.
Higher-rate self16's public random baseline is32/32; learned comparisons pending.

At16:55:20 the exact low176 training call remains pending/live and its protocol
and initialization are published. Initial SHA exactly matches low176; no new
labels/public/random/frozen opponents,critic/Adam reset false,seed31 and10800
training seconds verified. Four current inference source hashes match. Protocol
SHA9a945f34da9337dc85bac509fb9b8d465364393f6e98848b85f9582cb1f1258a.
First training history has not yet published; do not invent completed updates.
The independent retainer remains live despite the CLI's log-detach timeout.

Fresh app receipt confirms low176 bundle materializer stopped/tasks0 at09:53:19
PDT after41s full app wall. Bound$1.01 frees$0.99, bringing commitments to
$486.69 and headroom$13.31. Actual billing remains$242.96827570 through16:00UTC;
refresh after17:00 and pool only eligible stable terminal reports. All final
partitions remain untouched and the goal remains active.

### 17:03 UTC: direct diversity test, billing refresh and budget reallocation

Both low144 self-play variants are through the common GPU screen: control42–22,
persistent-neural-rivals44–20 versus incumbent96 on fresh300800–831; parent47–17.
Both descendants have Adam70943 from66847 and524288 phase decisions. Their
paired common-opponent difference is+3.125 points for rivals with95% interval
[-10.94,+15.625],so the screen does not establish a reliable recipe advantage.
Higher-rate incumbent-initialized self16 is34–30 on the same screen. Historical
screen now also has low19250–14,standard12843–21 and standard16029–35; no new
candidate promotion follows from these exploratory comparisons.

Launched the predeclared low144 opponent-diversity hypothesis's direct public
comparison: `low144-selfplay-pair-qualification-v1`,app
ap-nMeaJIm0NQnMhBRXsW2hfH,callfc-01M57Z8P84XM76PWKTGMDP80WK.
448 games:64 rivals-versus-control,64 of each versus low144 parent,64 teacher
and32 random per descendant,32 mode-comparison games per descendant. Direct
matches use newly declared301300–301331 with both seats and representative
replays. Regular6CPU16GiB,10800hard,9900schedule,$3 reservation. The local Modal
client took about3min to deploy; app remained detached/tasks0 and the original
client was kept alive. It then dispatched successfully without duplicate launch.
The pending-dispatch receipt is now resolved; do not double-count its$3.
Public preflight confirms all three exact hashes, finite519267 parameters and
Adam66847/70943/70943. All actual gameplay remains in Modal.

Provider report modal-billing-owned-20261018T170112Z.json covers all79 ledger
apps through17:00UTC and reports$271.65229396. Current partial-hour costs and
storage/egress are outside this completed-hour compute report. Pairing with the
16:00:58 report safely pools$5.90 of duplicate terminal margins: oldgamma,
oldnoentropy,low32 qualifiersv2/v3,baseline recovery and GPU-path pilot. Each has
stable billing, a terminal receipt, sufficient observation spacing and the
separate$5 shared overhead buffer. Commitments after the new qualifier and
pooling are$483.79,headroom$16.21. No live reservation was released.

Requested stop for older mixed-opponent standard-rate continuation
`scale-champion96-ratecontrol-v1`,appap-fmVDOnorrLifOB2YhNIy2r,after verifying
its published update208 is independently retained with SHA
7bb9d7ea700f9b6ec10a613c2a8ac372d453b899b048d87648b42a1c2377786d.
It had219 completed updates at the17:02 decision. Earlier useful112/128 and
other saved checkpoints remain intact. This is a budget priority toward the
more promising lower-rate family and learned-only continuations, not a proof
that every later standard-rate checkpoint is inferior. The first stop CLI call
aborted because noninteractive confirmation was absent; the authorized --yes
retry succeeded. Fresh app receipt is still stopping/tasks1, so its full$35
reservation remains held until stopped/tasks0. No restart is authorized or needed.

At17:02:44 both learned-only long runs are live and healthy: incumbent-origin
self-play43updates/1,409,024 decisions,low176-origin6updates/196,608 decisions.
All logged metrics finite, cumulative batch counters reconcile, and neither
uses public/random training opponents. Both have zero illegal actions,
resignations, loop exhaustion, uncredited outcomes or truncations. Maximum
behavior-logp replay mismatch is4.77e-6 and5.96e-6. Feature/probe-value standard
deviations remain nonzero (.1069/.0198 and.1362/.0187). These are training-health
checks, not strength estimates. Retainer statuses report no errors; low176
update4 is retained as e3140184cfc5...,and incumbent-origin update32 remains
retained. Exact records: selfplay-health-20261018T1703Z.json.

Fresh terminal app receipt now confirms the stopped H25 standard-rate mixed
branch has stopped/tasks0. Full app05:15:17–10:02:54PDT costs at most$13.39
under the declared allocation plus$1 margin, releasing$21.61 from its prior$35
reservation. `scale-champion96-ratecontrol-v1-terminal-app.json` records the
receipt. Commitments are now$462.18,headroom$37.82. Actual completed-hour provider
spend remains$271.65229396 through17:00UTC. Hold capacity for the next evidence-led
learned-only continuation and final qualification; releasing a reservation is not
an expense, refund or instruction to allocate blindly. Five older mixed controls,
two learned-only long runs and ongoing evaluation/retention workers remain live.
No final partition has been reserved or consumed, and incumbent96 is unchanged.

### 17:07 UTC: low144 reference qualification complete; next self-play screen frozen in advance

Low144's mode/common worker completed all160 planned games without errors,
truncations or cutoffs. Deterministic play beat stochastic25–7; stochastic
teacher31/32 is slightly below deterministic32/32 on those original maps.
Independent audit rechecked all160 game manifests, planned checkpoint identities,
policy modes, evaluator/teacher source hashes, paired coverage, outcomes and
cloud trace-hash receipts. Combined with the earlier288-game audit, the new
`h25-low144-qualified-reference.json` records448 audited games, cloud bundle
provenance, preferred deterministic mode and all uncertainty/evidence pointers.
This is a qualified comparison reference and initialization. Its latest training
phase was mixed-opponent, so it is not the requested final learned-only
self-play descendant. Incumbent96 remains unchanged while descendants qualify.

The same audit expanded self16's verified public evidence to256 games:32/32
teacher,32/32 random,23–9 native96 common,43–21 parent96,24–40 low32,and mode
9–23 stochastic versus deterministic. The stochastic win interval is[12.5%,43.75%]
under paired-map bootstrap, supporting deterministic inference for this checkpoint.
Teacher replication and two common-reference jobs remain pending. No final-mode
freeze or final400000 evaluation has occurred.

Fresh terminal receipt for low144 mode/common is09:22:28–10:04:15PDT,
stopped/tasks0. Full-app bound$1.30 releases$1.70. Commitments$460.48,
headroom$39.52; provider-reported actual compute remains$271.65229396 through17:00.
Historical screening has also completed frozen27241–23. Self-play screen now
has short lower-rate16 at35–29 and long lower-rate16 at36–28 against incumbent96
on300800, versus higher-rate16 at34–30. These close screen scores are not proof
of recipe superiority; keep public head-to-head results separate.

Predeclared `selfplay-stage2-screen-intent.json` for fresh301400–301431 before
any outcomes: fixed short lower-rate self16 reference808f...; candidates are
low176 parent, its learned-only8/16 descendants, low144 control/rivals16,
incumbent-origin long32/48, higher-rate16 and long lower-rate16. This tests
learned-only candidates against a different fixed neural reference and includes
the low176 parent to separate inherited strength from continuation. Different
ancestry/phase sample counts are explicit. All required checkpoints and raw
metadata/SHA/source identities must be retained and frozen before a single
bounded576-game GPU dispatch. Planned$5 is NOT yet reserved or spent. No roster
additions after301400 outcomes. Waiting for low17616 and long-incumbent48 is a
checkpoint-availability dependency, not a reason to restart either live run.

At17:09 both exact learned-only producer handles remain pending/live. Low176
update8 is independently retained with full SHA
a0cd39c7182d5536f7891252e3d0a5bd0284e8cfb498121ed8882a2fe0ea2bfb;
raw metadata SHA3d3073b661046f6d977b6e6af62b0059caf3fa5b899f9dd82e69c640ac399c67.
`selfplay-stage2-readiness.json` records the live handles and stages still
awaiting fresh retention verification; do not infer remote absence from an older
manifest, and do not dispatch until the complete roster is verified.

Storage clarification from the existing migration receipt: bulk retained-cache
weights were removed locally, while historical qualified-parent bundles were
explicitly preserved. The incumbent's old qualified-parent file still exists;
it was only stat'ed during this turn, not loaded or copied. All new checkpoint
copies, inference, matches and replays continue to run in Modal. Do not delete
those historical qualified bundles as an inferred cleanup task.

### 17:13 UTC: early self-play checkpoints included before the next fresh screen

Stage-1 eight-update results are now available: low144 control47–17 and neural
rivals46–18 versus incumbent96, compared with42–22 and44–20 at update16.
Both eight-update checkpoints have Adam68895 versus70943 at update16. Independent
identity, paired coverage and native-outcome checks pass. Paired-map early-minus-
later95% intervals are[-3.125,+18.75] points for control and[-9.375,+15.625] for
rivals. These intervals include zero; do not claim established regression.
They still justify retaining early checkpoints in candidate selection.

Before any301400 outcomes or allocation, versioned the stage-2 intent as
`selfplay-stage2-screen-intent-v2.json`, preserving and hashing the original.
Added both low144 eight-update checkpoints based on the separate300800 screen.
The new roster has11 candidates/704 games, remains within the same$5/1h GPU
allocation, and forbids roster changes after301400 outcomes. All candidates
will be checked against the fixed learned-only short self16 reference808f....
This is screening for the best continuation, not a causal comparison of equal
sample counts. No stage-2 reservation or gameplay yet.

The longer incumbent-origin learned-only run's update48 is independently
retained: SHA75c499e33a7b71dbaef15b5366685fab29fe3de4e7cc8899960ecfb09ebb6140,
metadata SHA7ec96a1e542e474a7372ebc1b7cf901030605e08dea2ed08441241e69c3bf2fb.
At17:13 the fresh retainer manifests still lack low17616, the only remaining
stage-2 prerequisite. Both producers remain live. Added a JSON/source-only
preparation helper under runs/structured-ppo-v3/prepare-stage2.py: it refreshes
retainer manifests, refuses existing plans/allocations, and will verify exact
metadata and current inference source identities for all eleven candidates plus
the reference before writing the frozen plan. It never downloads model tensors
or dispatches work. Invoke with the Modal SDK Python and staging environment;
do not rerun once the plan exists. Check the durable call rather than restarting
because the observer times out.

Historical screen now has11/12 candidates, adding oldgamma68827–37. Stage-1 has
9/10, adding high-rate8 at33–31. All currently reported GPU games resolve natively,
without invalid/resign/cutoff/truncation outcomes. Full audits and terminal
reconciliation will follow completion. Self16 teacher replication is31/32,
bringing teacher total63/64; self32 first teacher batch31/32. These are ongoing
public qualifiers, not final tests. Commitments remain$460.48/headroom$39.52.

### 17:17 UTC: stage-2 screen dispatched; both earlier GPU screens fully audited

Low176 completed16updates/524288 decisions by17:14:46, with736 native completions,
323 training wins/413 losses,finite metrics,zero invalid/resign/loop-exhaust/
uncredited/truncation outcomes and maximum logp replay error6.20e-6. Training
outcome counts are not held-out strength. The checkpoint was then independently
retained as SHA0db27757a2620e54362acb42c5f4108c21af624080f80f0569d7ed3534d098a1.
The preparation helper resolved the complete roster, verified metadata for11
candidates plus the reference, and checked seven archived protocols against four
current inference source hashes. All learned-only descendants havebc0/public0/
random0; the sole mixed ancestor is explicitly labeled as an initialization control.

Launched `gpu-selfplay-stage2-dev301400-v1`,appap-kQ8jCp5QYxmhsigm0rfEb0,
callfc-01M5807DE52VQE9ATYNYWM4FFH,$5,L40S8CPU32GiB,3600hard. All11 checkpoint
and raw metadata hashes are frozen before any301400 results. Reference remains
short self16SHA808f...,704 games total on32 fresh maps with paired seats. Worker
source hash and qualified GPU-path receipt are pinned; frozen engine22/22 passed
remotely. This is development screening only; public qualification and final
acceptance are separate. The CLI's log-detach completion message is not remote
function completion. No duplicate launch occurred.

Both preceding screens returned complete and passed independent JSON identity,
reference/metadata, coverage, outcome and bootstrap audits. Historical12-candidate
screen:768 games,767 native,one engine truncation,zero evaluation cutoffs. Its
last old noentropy816 candidate is33 wins/30 losses/1 unresolved on64 games,
NOT a33–30 result over63 games. The unresolved game is seed301017,seat2,
1748 actions,reward0,truncatedtrue,terminatedfalse. Its win interval[42.19%,62.5%]
uses all64 games. Low176 remains the historical screen lead54–10; it and all
other historical candidates have native outcomes for their64 games. Stage-1
10-candidate screen:640 games,all native,zero cutoffs/truncations; final long
lower-rate8 score33–31. GPU completion means requested candidates were evaluated,
not that every historical game resolved natively. Trace hashes remain cloud
receipts; these local audits do not claim full neural action reselection.

Fresh terminal receipts: historical app09:38:32–10:14:36PDT,bound$2.56;
stage-1 app09:43:38–10:14:17PDT,bound$2.33. Both stopped/tasks0. Releasing$5.11
from their two$5 reservations offsets the new$5 stage-2 reservation. Commitments
are$460.37,headroom$39.63. Actual provider compute remains$271.65229396 through
17:00UTC. No final partition has been reserved or consumed; incumbent96 remains
unchanged pending learned-only candidate selection and qualification.

### 17:30 UTC: campaign ownership transferred; four mixed producers stop-requested

The main Codex campaign session was interrupted at 17:20:31 UTC with its goal
paused; this Claude Code session continues the campaign from
`docs/structured-rl-agent-handoff-2026-10-18.md`. Added the read-only
`runs/structured-ppo-v3/observe-campaign.py` (polls calls and copies published JSON;
never dispatches, stops or writes the ledger). Observation
`campaign-observation-20261018T1722Z.json`:

- `selfplay16-cloud-qualification-v1` call returned: all 9 planned jobs/352 games,
  zero truncations/cutoffs. New row: self16 vs no-entropy64 common reference
  26–6. Terminal app receipt and independent audit of that row still pending.
- Stage2 first published candidate: mixed low176 parent 44–20 against the fixed
  short self16 reference. Remaining 10 candidates pending.
- Learned-only producers healthy: incumbent-origin update59/1,933,312 decisions,
  low176-origin update23/753,664; finite, counters reconcile, learned-only
  opponents, zero invalid/resign/loop/truncation, max logp error 4.77e-6/6.20e-6.
  Retainers report no errors.

Stop requested (17:28 UTC) for the four champion512 lower-rate mixed producers
(`scale-champion512-lowrate-v1`, `-frozen-v1`, `-seed2-v1`, `-noentropy-v1`). These
branches train against teacher 0.5/public 0.25 scripted opponents, contradicting the
learned-only training directive, and ranked below the champion96-lowrate line in
the 64-game historical screen. Latest retained checkpoints before stop: update368
e7330a2ad0a2, update336 f39ba5d6fb6e, update272 f6b2bad49064, update256 0364c34a7fd9;
per-run `*-stop-decision.json` records them. `scale-champion96-lowrate-v1` (low176
lineage) continues. Terminal stopped/tasks0 receipts and terminal accounting are
NOT yet confirmed; reservations remain unreleased until they are.

### 17:36 UTC: mixed producers terminal; audits extended; stage3 and bundles dispatched

User confirmed the four champion512 lower-rate stops. Fresh app receipts show all
four stopped/tasks0 (10:25:30–10:26:02 PDT). Terminal bounds at L40S rate plus $1:
lowrate $21.08, frozen $19.70, seed2 $19.13, noentropy $18.51 (each from $35).
`scale-champion96-lowrate-v1` (low176 lineage, mixed) was also stopped after update256
a00cbb1b5e19 was retained and frozen into the stage3 roster; terminal 05:15:54–10:34:35,
bound $14.73. All five rows: `stopped_budget_reallocated_to_learned_only`, with
`*-stop-decision.json` and `*-terminal-app.json`. No mixed producer remains live.

`selfplay16-cloud-qualification-v1` returned complete (352/352 games), app stopped
09:00:42–10:21:47, CPU bound $1.57. New reusable scripts:
`fetch-qualification-evidence.py` (Modal SDK; copies JSON evidence only) and
`audit-evaluations.py` (.venv; reproduces the earlier inline qualification/GPU audits).
Independent audits now pass for selfplay16 (complete, 352 games), portfolio finalists
(12 jobs/448), fast16-v2 (3/96), self32 (1/32), low176 (4/192). New audited rows:
self16 vs no-entropy64 26–6; low176 vs low144 fresh301200 35–29 [45.3%,64.1%];
low176 vs parent96 fresh301200 49–15 [67.2%,85.9%]; frozen240 vs gamma48 26–6;
standard112 vs native96 24–8. All native, zero unresolved.

Predeclared `init-stage3-screen-intent.json` and froze
`gpu-init-stage3-dev301600-v1-plan.json` (via generalized `prepare-screen.py`) before any
301600 outcomes: 10 candidates/640 games vs strongest learned-only checkpoint so far,
low176 self16 0db27757a262. Roster: mixed low176/192/208/224/240/256 (initialization
candidates), low176 self32 197253ace2da and incumbent-origin self64 3a7fab263623
(learned-only), champion512-lowrate368 and frozen336 (rival candidates). Dispatched app
ap-ucxkrro7SucFi2ZzD7ILzF, call fc-01M5816TTV3HHV1PNHBN7BE614, $5/1h.

Bundled candidate parents/rivals into /training (cloud-only copy, provenance and
optimizer verified, no promotion): low176sp32-197253ace2da, h25-low240-628f37fe9338,
h25-low256-a00cbb1b5e19, c512low368-e7330a2ad0a2. Bound $1.01. Commitments $383.10,
headroom $116.90. Stage2 at 5/11 candidates; no new training launched yet.

### 18:08 UTC: stage2/stage3 audited; incumbent-origin run stopped; low256 learned-only pair launched

Stage2 (301400, vs short self16 808f) returned complete; independent audit passes,
704 native games, zero unresolved; app 10:16:20–10:53:41, bound $2.61. Final rows:
low176 self16 47–17, self8 46–18, low144 rivals16 46–18, rivals8 44–20, control8
44–20, control16 40–24, low176 parent 44–20, fastlr16 36–28, incumbent-origin self16
35–29, self32 30–34, self48 29–35.

Stage3 (301600, vs low176 self16 0db2) returned complete; audit passes, 640 native
games, zero unresolved; app 10:33:30–11:04:51, bound $2.36. Mixed low176 31–33,
low192 32–32, low208 33–31, low224 30–34, low240 33–31, low256 37–27 [48.4%,67.2%];
learned-only low176 self32 31–33; incumbent-origin self64 18–46 [17.2%,39.1%];
champion512-lowrate368 24–40; frozen336 34–30. Interpretation: the low176 lineage
(mixed or learned-only) is on a plateau around low176 self16; extra learned-only
updates 16→32 show no measurable gain on these maps. Screen only; no causal claim.

Qualifiers that had returned unnoticed were collected, audited and accounted:
low176 complete (352): teacher 64/64, random 32/32, parent96 49–15, low144 35–29,
common native96 28–4/gamma48 29–3/noentropy64 28–4 (85/96), deterministic beats
stochastic 26–6. self32-matched complete (256): low32 18–46, parent96 26–38.
fast16-v2 complete (192): fast vs slow16 32–32. Portfolio finalists FAILED with
FileExistsError on its qualification directory (likely restart); 15 published jobs
(480 games) audit cleanly, remaining jobs never ran. CPU bounds $1.38/$1.46/$1.49/$1.63.
Low144 pair qualifier published 11 rows: rivals16 vs control16 34–30, each vs parent
144 34–30/32–32 (pending terminal receipt).

Stopped `scale-champion96-selfplay-v1` (incumbent-origin learned-only) at 90 updates;
update80 e97dd710f0fe retained; app 09:02:30–11:04:10, bound $6.25 (from $23). Reason:
stage2/3 show no improvement and self64 18–46 vs low176 self16.

Launched paired learned-only 8h runs from bundle h25-low256-a00cbb1b5e19 (stage3
leader), seed 20261041, lr 3e-5, entropy .005, gamma .995, 512 lanes, pool 2048,
public/random/teacher 0, 25200 s requested/28800 hard, $23 each:
- `low256-league-rivals-v1` (ap-GQsUhtOa6LzgbAKLuPZA1j, fc-01M582YRC34S5PMVYTH9MAV2WX):
  25% immutable learned rivals [low176, low144, incumbent96, frozen240, standard112,
  low176sp32, c512low368], 37.5% current, 37.5% history.
- `low256-selfplay-v1` (ap-EbLjnIyekmZU13y7ihPYuF, fc-01M582Z5624Q419XCCC33ZET9Z):
  50/50 current/history control.
Retainer `low256-pair-retainer-v1` (ap-fqGLKgcfj2ldh2g6DT2woM, $2, 30600 s).
Intents: `low256-*-intent.json`. The first launch attempt failed in the local shell
(option quoting) before any reservation; no duplicate apps. Lineage: imitation +
mixed PPO through low256, then learned-only. Commitments $403.28, headroom $96.72.
Low176 learned-only producer (update ~60) continues to its 3h end (~12:53 PDT).

### 18:30 UTC: budget ceiling raised to $800; per-turn discount option; wave-1 launched

User asked to pour "a few hundred" more Modal dollars in as soon as possible. Ledger ceiling raised
$500 -> $800 (`budget_history` records authorization). Commitments $497.28, headroom $302.72
after the launches below.

Reward design check against Duper (`duper/rl_ray`): its 1v1 reward is terminal-only (+10 to the
survivor; shaping terms were tried in the first 1v1 baseline and later commented out), which
matches our native +/-1 choice; our own potential-shaped branch lost (update256 6-26). The
substantive difference is discounting: Duper's GAE (`flow/utils.py`, `time_decay` flag set on
end_turn in `single_env.py`) applies gamma=lambda=0.95 only at turn ends and 1 within a turn;
ours applied gamma .995/lambda .99 to every learner decision (~600 per game), which shrinks
early-game credit and mildly penalizes taking more actions per turn.

Code: `ppo.turn_discounts()` builds per-transition gamma/lambda (turn factors only on the
learner's end-turn transition, whose transition spans the opponent reply); `repair.run()`
accepts `turn_gamma`/`turn_lambda` (both or neither; native reward only), records
`protocol.discount`, and logs `turn_end_fraction`. `experiments/modal_structured_scale.py`
exposes `--turn-gamma/--turn-lambda` (default off = unchanged behavior and RNG). `runner.py`
(`Config`) was deliberately not modified because the GPU-arena qualification receipt pins it.
New tests in `tests/test_structured_repair.py` (helper, hand-computed per-turn returns, scalar
path unchanged, argument validation); structured repair/rl/league/budget suites: 63 passed.

Wave-1 (all learned-only, 25% frozen-rival league, lr 3e-5, entropy .005, 512 lanes, pool
2048, 25200 s / 28800 hard, $23 each):
- `low256-turn95-league-v1` (ap-h98OHwxsrYWa807tKfCkQc): low256, per-turn .95/.95, seed 20261041.
- `low256-turn98-league-v1` (ap-RlnQkdmVnTuP7FisdUQip3): low256, per-turn .98/.95, seed 20261041.
- `low176sp32-league-v1` (ap-0tiV8HbTpgZVrrzYUnPnMg): learned-only low176 sp32, per-decision.
- `low176sp32-turn95-league-v1` (ap-OfpCzwdp2gn2tdJr5lp2SQ): low176 sp32, per-turn .95/.95.
Retainer `wave1-retainer-v1` (ap-KGIwixOdS44HiWI1oprRW4). Together with the running
low256 league/self-play pair this is a factorial over discount, parent and opponent mix.
Selection will use fresh paired screens of every retained checkpoint against fixed references
(low176 self16, then low256), i.e. a learning curve vs a fixed opponent; loss curves and
training win rates are not strength evidence.

### 20:55 UTC: ladder screens 1-2; losers stopped; winner league96 in qualification; unit-diversity finding

Ladder 1 (`gpu-ladder1-dev301700-v1`, 832 games) and ladder 2 (`gpu-ladder2-dev301800-v1`, 512
games) vs fixed low176 self16 0db2: complete, audited, all native, zero unresolved; bounds
$2.91/$2.20. Learning curves (wins-losses of 64):
- low256 league (per-decision): u32 39-25, u64 37-27 | u80 35-29, u96 37-27, u112 31-33.
- low256 pure self-play: u32 28-36, u64 28-36.
- low256 per-turn .95 league: u16 38-26, u32 32-32 | u48 25-39, u64 31-33.
- low256 per-turn .98 league: u16 29-35, u32 34-30 | u48 29-35, u64 27-37.
- low176sp32 league: u16 23-41, u32 30-34; low176sp32 per-turn .95: 31-33, 30-34.
- Original low176 pure self-play u128: 22-42 vs its own u16 (CI 25-44%).
- Parent low256 control on 301800: 30-34.
Interpretation: long pure self-play degrades (two independent lines). The frozen-rival league
holds at or slightly above parent; per-turn discounting did not help in these runs. Gains are
modest and within overlapping intervals. Screens are selection evidence, not causal claims.

Stopped with retained checkpoints and stop-decision files: low256-selfplay-v1 (u128),
low176sp32-league-v1 and low176sp32-turn95-league-v1 (u64), then low256-turn95/turn98 (u96).
Bounds: $6.81/$4.74/$4.75 (turn pair pending receipts). `scale-low176-selfplay-v1` completed
157 updates/5.14M decisions (bound $8.92). Old CPU helpers stopped (selfplay retainer $1.24,
cloud-screens $2.97); low144 pair qualifier complete (448 games audited; rivals16 vs control16 34-30), $1.47.
`low256-league-rivals-v1` continues as the sole training run.

Winner pick: low256-league-rivals-v1 update96 (f998d15e220c...). Public qualification split across
two CPU apps for speed (`league96-fresh-qualification-v1` matches vs low176 self16 / low256 /
incumbent96 on fresh 301900; `league96-baseline-qualification-v1` teacher/random/common/mode),
416 games, $3 each.

Unit diversity (user research agenda): low176 recruits ~99% Warriors across 256 teacher/learned
games and never Riders/Archers/Defenders/Catapults/Knights/Swordsmen, although these were legal
in most games (Catapult 238, Rider 227, Archer 223 games). Cause: teacher recruit score
(`learning/teacher.py:122`) only penalizes cost, so imitation bootstrap learned cheapest-unit
recruiting and PPO never left it. Full note: `docs/unit-diversity-investigation-2026-10-18.md`;
tooling: `scripts/analyze_unit_composition.py`, `experiments/modal_trace_analysis.py`.

### 21:42 UTC: league96 qualified; final acceptance dispatched (partition 400000)

Development qualification of low256-league-rivals-v1 update96 (f998d15e220c92fe5ad091b73c4fd8952612a69b37fdd607d8b007093b26225d),
split across `league96-fresh-qualification-v1` and `league96-baseline-qualification-v1`: 416 games, both
complete, independent audits pass, zero unresolved; CPU bounds $1.31/$1.37.
- Fresh 301900: vs low176 self16 34-30 [45.3%,62.5%], vs parent low256 36-28 [45.3%,67.2%],
  vs incumbent96 46-18 [64.1%,81.2%] (replay representatives verified).
- Teacher 64/64 (dev + replication), random 32/32.
- Common references: native96 26-6, gamma48 29-3, noentropy64 31-1 (86/96; low176 had 85/96).
- Inference mode: stochastic challenger 14-18 vs deterministic; deterministic selected.

Final plan frozen with `runs/structured-ppo-v3/prepare-final.py` (refuses existing final plans or
reservations): `league96-final-acceptance-v1-plan.json`, deterministic, sampling seed 20261019,
seed base 400000 (32 maps, both seats; teacher-final and random-final, 64 games each, replay
representatives), training protocol 746bec39..., teacher hash e7493739..., runtime Python 3.12.10 /
JAX 0.11.2 / NumPy 2.5.4, 39 frozen sources. The retained snapshot's protocol.json links to the run's
common/protocol.json; the worker follows the link in-container. Dispatched app
ap-x5ibNDDX73N52J2ejiy13i, call fc-01M58FC514K5X9QBH5J1G3799H; the ledger reserves final partition
400000 (the worker claims it in cloud storage before games). Thresholds: teacher >= 32/64,
random >= 52/64 native wins; unresolved games stay in denominators.

### 21:58 UTC: FINAL ACCEPTANCE PASSED — league96

`league96-final-acceptance-v1` returned complete with no errors: on the untouched partition
400000–400031 (32 maps, both seats, deterministic), low256-league-rivals-v1 update96 won
**teacher 64/64** and **random 64/64** native games; zero engine truncations or evaluation cutoffs;
128 verified trace hashes; `final_strength_gate_passed: true`. Training provenance verified in-container
(policy delta L2 3.018, Adam 95,519 -> 120,095; learned-only continuation). The partition claim was
persisted before games. Independent audit (`league96-final-acceptance-v1-independent-audit.json`)
passes, including replay receipts (988 + 412 public actions). `audit-evaluations.py` now expects one
representative replay per outcome that occurred (all-win jobs have only a win representative).
CPU bound $1.11. Commitments $407.36, headroom $392.64 under $800.

Records: `final-accepted-candidate.json` (new); `current-development-candidate.json` now points to
league96 (previous incumbent champion51296 retained as previous_candidate); `docs/structured-rl-current.md`
summary updated. Partition 400000 is consumed; any further iteration must use a new untouched
partition. `low256-league-rivals-v1` continues training (update ~205) under its existing $23 allocation.

### 23:45 UTC: all prior jobs stopped; diverse-strategy restart begins

User asked to stop all work and restart around strategy diversity. Stopped `low256-league-rivals-v1`
at 281 updates (update272 retained; bound $14.74) and its retainer ($1.38); wave-1 retainer $1.32.
No campaign job remained live; commitments $397.80, headroom $402.20 under $800.

Root cause recap: both scripted teachers (host `learning/teacher.py` and device
`rl/public_heuristic.py`, which produced every imitation label) score recruiting as
20*demand + 20*threat - 2*cost and research with a fixed economy priority, so they recruit only
Warriors. New `src/polytopia_bot/rl/diverse_heuristic.py` (VERSION diverse-public-device-v1) keeps
style 0 identical to the balanced heuristic and adds six styles (rider-rush, archer, defender,
siege, heavy knight/swordsman, value-per-cost) via unit score bonuses and research bonuses on the
target techs and all prerequisites. Each seat's style is a pure function of map seed and seat.
`bootstrap.make_collector/run` accept `teacher='diverse'` and `teacher_style` (-1 = per-seat mixture)
and log `teacher_summon_units`; `repair.run` logs `learner_summon_units` per PPO update;
`modal_structured_scale.py` exposes `--teacher/--teacher-style` for bootstrap mode. Tests:
`tests/test_diverse_heuristic.py` (5) plus bootstrap/public-heuristic/structured suites, 73 passed.
Local 16-map x 600-step teacher simulation: balanced 936/936 Warrior summons; diverse Warrior 173,
Defender 151, Rider 136, Archer 115, Catapult 104, Swordsman 60, Knight 6.

Launched (2 h requested / 4 h hard, $12 each), parent `repair-combat1-v1` like the original bootstrap,
128 lanes x 16, 4 BC epochs, lr 1e-4, pool 2048, seed 20261051:
- `diverse-bc-v1` (ap-Tw3q2cX7xF2o7eEzMiLo5r): teacher roll-in 1.0.
- `diverse-bc-dagger-v1` (ap-Q7wDDu9dBiEQ62ZxfESX9M): teacher roll-in 0.5 (online imitation).
Bundle `league96-f998d15e220c` requested (ap-K7LaRsZj7oDGHnGVAI62mr) as a frozen strength rival.
Planned next phase: learned-only league PPO from the better bootstrap, with frozen rivals mixing
the diverse bootstrap (stochastic) and current champions (league96, low176), unit mix tracked live,
selection by fixed-reference ladders against league96 on strength and unit diversity.

### 00:20 UTC (Oct 19): AlphaStar-style diverse league — specialists, identity anchor, main agent

User clarified the goal: expert strategies that flexibly use different troop types, i.e. a diverse
set of strategies. A single deterministic policy imitating a style mixture tends to collapse to the
modal action, and PPO tends to collapse diversity, so the plan is now a population (user: no
preference among options; recommended option taken):
- Six style specialists, each bootstrapped from one fixed teacher style (`--teacher-style i`,
  online imitation 0.5, 1 h, $7 reservations): `spec-{rider-rush,archer,defender,siege,heavy,value}-bc-v1`.
  Early label mix: Rider 96%, Archer 96%, Defender 87%, Catapult 74%, heavy Rider/Swordsman/Knight
  45/35/11%, value Warrior 67%/Defender 23%.
- Identity anchor for specialist PPO: `ppo.loss(..., anchor_coef)` adds mean(exp(d)-1-d) with
  d = logp_anchor - logp on taken actions (sampled KL(pi||anchor)); `repair.run(anchor_path, anchor_coef)`
  computes anchor log-probabilities once per batch and records `protocol.anchor`; launcher
  `--anchor <run> --anchor-coef c`. Three new tests (zero at anchor, positive under drift, gradient moves
  toward anchor) plus validation; structured suites 73 passed.
- Main flexible agent: league PPO from the better generalist bootstrap against all specialists plus
  league96 (bundle `league96-f998d15e220c` complete).
Disclosure: anchored specialists remain tied to teacher-derived style bootstraps; this is not pure RL.

### 00:40 UTC (Oct 19): direction settled — one inclusive network; specialist PPO cancelled

User clarified: the fix is the bootstrap dataset (many strategies using different troop types) in
one network, followed by the best recipe on top; not separate per-strategy networks. The specialist
PPO auto-launcher was stopped before any launch or reservation. The six specialist bootstraps now
serve only as diverse frozen rivals. `runs/structured-ppo-v3/launch-main-ppo.sh` (background) launches
three learned-only league runs once both generalist bootstraps and all six specialist finals exist:
`diverse-dagger-league-lr1e4-v1`, `diverse-dagger-league-lr3e5-v1` (parent diverse-bc-dagger-v1) and
`diverse-bc-league-lr1e4-v1` (parent diverse-bc-v1); 25% frozen rivals (six specialists + league96 +
low176), per-decision gamma .995, entropy .005, 512 lanes, 25200 s / 28800 hard, $23 each, no anchor.
Intent: `diverse-main-ppo-v1-intent.json`.

### 01:30 UTC (Oct 19): diverse main runs launched; GPU and throughput evidence

Specialist bootstraps complete (each 32/32 vs random in both modes; bounds ~$3.80 each). Main runs
launched by `launch-main-ppo.sh` once finals existed: `diverse-dagger-league-lr1e4-v1`
(ap-ryjjIWrNv3CJGYYQxCnNql), `diverse-dagger-league-lr3e5-v1` (ap-A0YhYpOvQu6t1BUTTI63MC),
`diverse-bc-league-lr1e4-v1` (ap-FticBW8Ufzy2SyBE1AsGiu); retainer `diverse-main-retainer-v1`
(ap-3K2dlbO9FSRD1eZX9xqWjn).

Throughput: launcher gained `--gpu L40S|H100|B200` (reservations scaled by GPU_RATES). Same-recipe
benchmarks (stopped after answering): collection step L40S 223 ms, H100 202 ms, B200 164 ms; neither
justifies its price. Diagnosis over 22 runs (`docs/simulator-throughput-diagnosis-2026-10-18.md`,
`throughput-evidence.py/.json`): collection is 91–98% of update time; removing the network from the loop
only takes a step from ~125 to ~110 ms (128 lanes); step cost fits ~90 ms fixed + 0.27 ms/lane; lane
utilization 0.30–0.51. Root cause is the batched simulator step and idle lanes; multi-GPU multiplies but
does not fix them. Multi-GPU session (branch claude/practical-boyd-62b6f5, code committed, paused at user
request) finishing two 4×L40S benchmarks (training phase only) for throughput numbers.

### 02:45 UTC (Oct 19): efficiency first — diverse runs stopped; decision-budget collector adopted

Simulator profile (`claude/dazzling-chatterjee-ccca9e`, report `docs/simulator-profile-2026-10-19.md` on
that branch): network ~2.6% of a step; vmapped `core.step` evaluates all 17 action branches per lane;
lane utilization 0.30 -> 0.99 with a decision-budget collector (3.4× faster real PPO updates, 105.5 -> 31.1 s);
training-only fast-engine proposals are bit-identical and 5.2× faster per step (combined 16–19× learner
decisions/s at 512 lanes). User approved budget collector + fast engine for training only (frozen
engine remains for all evaluation/qualification/final tests) and asked to stop runs and prove efficiency
first. The simulator session already wired `collector_mode='decision-budget'` (launcher default
`--collector decision-budget`) into the main checkout; `tests/test_decision_budget.py` +
repair/league suites pass (36). Diverse runs stopped with retained checkpoints (dagger-lr1e4 u32,
dagger-lr3e5 u16, bc-lr1e4 u16, anchor u12; bounds $3.99/$3.98/$3.97/$2.44; retainers $1.08/$1.04).
Before stopping: unanchored dagger lr1e4 collapsed to Defender 49% + Warrior 40% with 57 truncations;
anchored run stayed diverse (Warr34/Defe21/Swor15/Ride14/Cata8/Arch7). Commitments $478.97, headroom
$321.03. Next: simulator session wires `--engine fast`, proves end-to-end speed on the diverse recipe,
and ranks remaining bottlenecks; restart diverse runs afterwards.

### 2026-10-19 04:03 UTC: campaign budget cap removed; RL jobs paused

User directive: remove the campaign budget altogether and pause RL jobs until a
new GPU configuration is chosen. `rl/budget.py` no longer rejects reservations
over a total; the ledger still records every allocation and run ID
(`budget_usd: null`, `budget_history` records the change). The only live RL app,
`smoke-fast1024-anchor-v1` (`ap-A4IiGRmARo63SVp9qrwkuG`, 900 s L40S/1,024-lane
smoke), was stopped at 21:03:02 PDT after about 6.5 minutes; its status is
`stopped_by_user_pause_2026-10-18`. No campaign training job is live.

### 2026-10-19 04:05–04:55 UTC: GPU cost benchmark (no training)

43 short benchmark jobs (`bench-gpu-*`, $24.46 estimated) compared nine Modal GPU
types at 512/1024/2048 lanes with the decision-budget collector and fast engine.
Full results and recommendation: [gpu-benchmark-2026-10-18.md](gpu-benchmark-2026-10-18.md).
Best value: RTX PRO 6000 at 2048×32 (needs a looser replay-audit tolerance);
without that change, H100 at 2048×32. L40S caps at 1024 lanes. RL training stays paused.


### 05:15 UTC (Oct 19): speedups adopted; H100 2048-lane diverse runs launched

Speedups now in the main checkout (uncommitted): decision-budget collector (default), training-only fast
engine `--engine fast` (default for PPO; frozen engine for all evaluation; bitwise-equivalence tests and
GPU digests), lanes 1024/2048 with `--horizon 16|32|64` (lanes x horizon <= 65,536), passing-audit
diagnostics rate-limited (first 3 + every 64th; failures always save full evidence and abort; 2 new tests),
persistent XLA compile cache on the training volume (first retest inconclusive: 502 s vs 210 s cold while a
41-job GPU sweep ran; needs a clean retest). Ledger cap removed by the user in the GPU-lanes session
("remove the 500 dollar campaign budget altogether"); that session paused the first 1024-lane smoke after
7 updates and benchmarked 9 GPU types (docs/gpu-benchmark-2026-10-18.md; H100 at 2048 lanes recommended).

H100 smoke `smoke-h100-2048-anchor-v1` (2048 lanes, horizon 32, 65,536 decisions/update, anchored diverse
recipe): 25 updates, stored logp_error 0.0, peak 32 GB, 0 invalid, 1 truncation, dropped 1.7–2.8%,
6.5–9 s/update early rising to 16–18 s as games lengthen, unit mix Swordsman 29/Warrior 22/Rider 12/
Defender 12/Archer 9/Catapult 8/Knight 4.

Launched (H100, 2048 lanes, horizon 32, fast engine, decision-budget, 25% frozen rivals, 25200 s / 28800 hard):
`diverse-h100-anchor-v1` (ap-jAEPmo6hjIW7pfmQYxDO5A; lr 1e-4, anchor diverse-bc-dagger-v1 0.1) and
`diverse-h100-lr3e5-v1` (ap-1zRI6uhQxZQiV06lQQmMeX; lr 3e-5, no anchor); retainer `diverse-h100-retainer-v1`
(ap-Ro2fC4jQ1x9ILWnWr1A694). `ladder-loop.sh` dispatches an hourly screen of each run's newest retained
checkpoint vs league96 (frozen engine) on fresh blocks 302000, 302100, ...

### 06:50 UTC (Oct 19): RL campaign stopped at user request

Stopped `diverse-h100-anchor-v1` (264 updates; update256 retained; bound $7.97) and `diverse-h100-lr3e5-v1`
(188 updates; update176 retained; bound $7.97), retainer `diverse-h100-retainer-v1` ($1.11), the hourly ladder
loop, and the strategy-check qualifier `diverse-strategy-check-bootstrap-v1` ($1.01). Ladder 1
(`gpu-h100-ladder1-dev302000-v1`, $1.31; not yet independently audited) vs league96: anchored u144 11–53,
lr3e5 u112 19–45. Last training mixes: anchored Catapult 23/Warrior 22/Swordsman 17/Rider 11/Defender 10/
Archer 10 with truncations ~8% of finished games; lr3e5 narrowing to Warrior 40/Rider 36, truncations ~5%.
Several GPU-lanes-session benchmark rows whose apps had stopped were terminally accounted from their app
receipts (shared app ids, so per-row bounds are conservative). No campaign app remains live
(`polytopia-expert-sft` is unrelated and untouched). Ledger commitments $719.34 (no cap).
