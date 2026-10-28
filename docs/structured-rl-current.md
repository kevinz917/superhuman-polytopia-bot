# Current structured RL bot

**Final acceptance passed (October 18, 14:55 PDT).** The delivered bot is
`low256-league-rivals-v1` update 96, SHA
`f998d15e220c92fe5ad091b73c4fd8952612a69b37fdd607d8b007093b26225d`, deterministic,
at `/retained/checkpoints/low256-league-rivals-v1/update-96-f998d15e220c/latest.npz`
(Modal staging, `polytopia-structured-rl-retained`). On the untouched final partition
400000–400031 (both seats) it won **64/64 against the scripted teacher** (gate ≥ 32) and
**64/64 against random** (gate ≥ 52), with zero unresolved games, 128 verified trace
hashes and replay-verified representatives. Selection: best of six learned-only runs
on fixed-reference ladder screens. Development qualification (416 audited games):
fresh-map 34–30 vs low176 self16, 36–28 vs its parent low256, 46–18 vs the old
incumbent; 86/96 common references. Lineage: teacher imitation and mixed-opponent
PPO through low256, then 96 learned-only updates with a 25% frozen learned-rival
league. **Not from-scratch pure RL.** Known limitations: about 99% Warrior recruiting
([unit diversity note](unit-diversity-investigation-2026-10-18.md)); learned-only gains over
its parent are modest; scripted baselines are at ceiling; research ruleset, no human
evaluation. Record: `runs/structured-ppo-v3/final-accepted-candidate.json`. The sections
below are earlier history; the campaign log has the full chronology.

**October 18 execution update:** checkpoint comparisons and retention now run entirely on Modal. Local download/evaluation watchers were stopped and guarded against accidental restart. See [cloud migration receipt](cloud-comparisons-2026-10-18.md).

The learned bot works through the same public `GPUEnvironment` interface as the
language-model project. Training and strength evaluation are ongoing. No model
has passed the final acceptance gate yet. The [campaign log](structured-rl-repair-progress.md)
records exact runs and results; the [plan](structured-rl-repair-plan.md) defines
the experiments and untouched final-test maps. Its original $500 budget (later
$800) was removed on 2026-10-18; there is no campaign spending cap.

**Self-play direction:** the user requested a self-play setup. New training
allocations use only learned opponents; scripted bots are evaluation benchmarks.
`champion96-selfplayonly-v1` completed sixteen updates from the incumbent with
50% current policy and 50% historical-policy opponent selection, no new imitation
labels, and native win/loss rewards. Both built-in random-play diagnostics won
32/32 games. Fresh public-interface matches finished43–21 against its own
incumbent parent and24–40 against mixed low32, which had twice as many phase
decisions. All128 games resolved natively; representative replays verified6,834
actions. The rest of its qualification remains in progress.
The higher-rate pilot and two learned-only continuations from low144 also
completed sixteen updates; learned-opponent comparisons are ongoing. The long
lower-rate run has produced update32, now being compared with the older mixed
low32 checkpoint at the same1,048,576 phase decisions and8,192 optimizer steps.
This is self-play fine-tuning of an earlier
imitation-bootstrapped model, not from-scratch pure RL. Existing mixed-opponent
runs remain comparison controls. Their results below do not establish the
strength of the new self-play checkpoints. The scale launcher now defaults to
zero scripted-opponent probability. The long run's numerical trajectory differs
from the short pilot despite matching initialization and seed, so each checkpoint
requires its own strength evaluation.

At16:54UTC, the historical GPU screen found low176 at54–10 against incumbent96
on32 fresh paired maps. This is a promising development result, not a promotion.
Its public-interface replication and a bounded three-hour learned-only self-play
continuation are running in Modal, with independent checkpoint retention. The
current incumbent remains update96 while learned-only challengers complete
qualification and selection.

Low144 has now completed448 audited development games and is available as a
qualified comparison reference at `/training/h25-low144-8150ea3b8468/final.npz`.
Its deterministic policy scored63/64 against the teacher,32/32 against random,
80/96 against common learned references, and46–18 against incumbent96 on the
fresh replication. Deterministic beat stochastic25–7 in its mode comparison.
The consolidated receipt is `runs/structured-ppo-v3/h25-low144-qualified-reference.json`.
Its latest training phase still included scripted opponents; the requested final
delivery requires a qualified learned-only continuation. This reference has not
replaced the incumbent or consumed any final maps.

The current learner is synchronous PPO with independently advancing game lanes:
collect a rollout, optimize, then collect using updated weights. Collection and
optimization do not overlap. Historical opponents refresh every16 PPO updates,
retaining four snapshots. This provides evolving self-play, but it is not the
distributed actor/predictor/buffer/trainer architecture in Neuro. In the
reference distributed setup we compared against, the trainer publishes weights
to predictors every 4 trainer iterations. Those publications are periodic latest weights, not evidence-gated stronger
champions. Polytopia held-out checkpoint promotion remains a separate step.

## Policy and learning

- `normalized-v2`: 519,267 parameters, 64 channels, two normalized residual
  blocks, SiLU, mean/max spatial pooling and a separate critic projection.
- Masked autoregressive action family, source and target/parameter heads.
  Voluntary resignation is masked by checkpoint configuration; every other
  legal simulator action is retained. No search, scripted move selection or
  heuristic fallback is used at inference.
- Public observations only: own assets, explored/visible map and public game
  information. The critic has no privileged opponent information.
- PPO collects learner-only transitions spanning the opponent response.
  Independent lanes avoid waiting for the slowest opponent turn at every
  learner decision. Current runs use 128 or 512 lanes, a 64-decision horizon, two
  optimization epochs and minibatches of 256. Learning-rate trials compare
  1e-4 with 3e-5; the usual entropy coefficient is 0.005, with a separate
  zero-entropy continuation. At matched update 16, removing the bonus lost
  11–21 against its control on 32 paired-seat development games, with equal
  learner decisions and optimizer steps. Longer training remains under review.
- The joint entropy gradient is corrected, advantage normalization has a 0.05
  divisor floor, and behavior log probabilities are checked before updating.
  Actor-sized replay resolves measured batch-shape roundoff without relaxing
  the 1e-5 mismatch threshold. Genuine mismatches/nonfinite values fail the run.
- One opponent is selected per rollout. New full-game runs sample current
  or recent historical policy snapshots with equal probability. Earlier
  mixed-opponent controls used a public heuristic in 25% of rollouts.

An earlier pure-RL curriculum pilot used 25% public heuristic, 25% non-resigning
family-uniform random, 25% current and 25% historical opponents. Its separate
`random_opponent_probability` option defaults to zero. Episode state continues
between rollouts in both recipes, so opponent type can change during a game;
rollout tags are not isolated per-opponent game win rates. This pilot tests
whether easier full-game victories improve learning before further scaling.

A separate league experiment mixes two immutable learned rivals into
25% of rollouts, alongside 25% public heuristic and 25% each current/historical
self-play. The rivals use frozen network weights with stochastic actions and
provide no demonstration labels. The default probability is zero, which preserves
the prior opponent-selection RNG stream. The short pilot tied its control on
teacher games but won their learned-policy match 24–8. Long continuations now
compare it against controls at standard and lower learning rates. It is not yet
the default recipe: matched update-16 comparisons on fresh development maps
finished 10–22 at standard rate and 16–16 at lower rate. A second lower-rate
training seed finished 13–19, with all 32 games resolving natively; its paired-map
95% interval is [28.13%, 53.13%]. These two training seeds show no replicated
advantage at update 16. The pilot advantage has not reliably transferred.
A bounded pilot tested 256 lanes × 128 decisions against
512 × 64, holding rollout decision count fixed while changing trajectory length
and lanes. It lost 14–18 against its matched control and took 2.12 times as much
update time, so it was not allocated a long run.

The native objective is win +1 / loss -1. The separately labeled
`own-potential-v1` branch adds `gamma * Phi(next) - Phi(current)`, using only own
assets and exploration. Terminal Phi is zero; truncations retain the final-state
potential and bootstrap. Its critic predicts shaped return, not win probability.
Gamma is recorded in each checkpoint; the current comparison tests 0.995 and
0.999. Discounting is per learner transition, not per game round.

Pure-RL checkpoints use tactical reward curricula followed by full-map PPO,
without imitation labels. Bootstrapped checkpoints additionally use explicit
public-teacher behavior cloning before PPO. Playing against a heuristic in RL
does not itself supply action labels. These branches are reported separately.

## Final acceptance

The same Modal qualification worker supports an explicit `final_acceptance`
plan after development selection. It tests one frozen learned-only self-play
descendant against the teacher and random baseline in64 games each, with
paired seats and representative complete replays. The plan pins the checkpoint,
training protocol, inference mode, sampling seed, Python/JAX/NumPy versions,
and39 evaluator/policy/host/simulator source files. The worker verifies the
archived training sources and excludes newly scripted/imitation training phases.
Earlier imitation ancestry remains part of the selected checkpoint's history.

Development plans cannot use seeds400000 or above. Final partitions are aligned
blocks of32 maps starting at400000. A partition is reserved atomically in the
budget ledger and claimed in the cloud before games; failures or interruptions
do not silently reopen it for model selection. A complete evaluation writes an
integrity-checked `acceptance.json`, including strength thresholds, uncertainty,
unresolved games and expansion/action diagnostics. Passing is a strength
milestone, separate from full-budget optimization and release packaging.
No final partition has yet been run.

## Use an existing checkpoint inside Modal

These examples assume the training volume is mounted at `/training` and the
retained-results volume at `/retained`. They run inside the cloud evaluation
container; the local checkpoint cache was removed after cloud verification.

The current development candidate is **champion-512 update 96**, using teacher
imitation followed by native-outcome PPO with gamma 0.995, 512 lanes and a
2,048-map pool. Its immutable Modal training-volume bundle is
`champion51296-05c414466823`, with SHA-256
`05c41446682389972373e1478ed1ed4e139262a3c69c54f17964ef94b78ce5f3`.

It qualified at **63/64 teacher wins** (31 original, 32 replication), **32/32
random wins**, and **70/96 wins against three common learned references**:
native96 21–11, gamma-.999 update48 23–9, and zero-entropy update64 26–6.
All 63 teacher wins ended in learned-policy captures. It beat the previous
update12 incumbent **46–18 over 64 games**, including 26–6 on the original
maps and 20–12 on the additional maps. The direct match's paired-map 95%
interval is [64.06%, 79.69%]. Its common-reference advantage over update12's
64/96 has an interval including zero; these remain related-policy development
comparisons rather than proof of universal superiority.

All 256 benchmark trace hashes, source/checkpoint provenance, paired coverage
and native outcomes were verified. Its longest recorded win and loss replayed
through 3,714 public observations/actions/outcomes. The candidate selection is
recorded in `runs/structured-ppo-v3/current-development-candidate.json`, with
full evidence in `champion51296-qualification.json`. Deterministic inference is
the current default: it beat stochastic play 20–12 in a checkpoint-specific
comparison. Stochastic play won 32/32 teacher games versus deterministic's
31/32 on the same original maps. Both mode comparisons used one sampling seed;
all 64 traces were verified, and this does not prove universal mode superiority.
This is imitation plus PPO, not the separate pure-RL branch. No final acceptance
maps have been used and the full-budget campaign remains active.

The former incumbent `champion51212-06db83c193a7` remains retained. It qualified
at 64/64 teacher, 32/32 random and 64/96 common-reference wins. The completed
five-policy portfolio at `runs/structured-ppo-v3/learned-portfolio-v3.json`
compares that earlier incumbent with older checkpoints; it does not include the
new update96 candidate. Its 320 traces were verified, and `build-portfolio-v3.py`
reproduces the comparison.

Two qualified alternatives remain available on Modal:
`lowrate16-d9be65407e63` (61/64 teacher, 32/32 random, 64/96 common references)
and `seed2-lowrate16-a05a132eacf8` (63/64 teacher, 32/32 random, 67/96 references).
The second beat update12 39–25 overall but tied 16–16 on the added map subset.
Neither has been shown to outperform the new update96 candidate. A higher-
discount pilot finished 29–35 against update12 and did not earn a long allocation.

```python
from polytopia_bot.learning.adapter import GPUEnvironment
from polytopia_bot.rl.policy import StructuredPolicy

checkpoint = "/training/champion51296-05c414466823/final.npz"
bot = StructuredPolicy(checkpoint, seed=51, deterministic=True)
env = GPUEnvironment(size=14)
observation = env.reset(seed=300000, settings={})
decision = bot.select(observation)
transition = env.step(
    action_id=decision.action.action_id,
    expected_state_version=observation.state_version,
)
env.close()
```

For the separate pure-RL control, use
`/training/pure-async32-aa8505267dcd/final.npz`.
That checkpoint scored **5 wins / 27 losses** on the same 32 teacher development
games. Its SHA-256 is
`aa8505267dcd20bfe4e263cf217e5dc97b57c148dac6ae104b96fabc8af55375`.
Bootstrapped PPO branches continue training; these paths identify immutable
tested snapshots. The later pure-RL continuation was confirmed stopped to
reallocate resources after update 128 scored 1 win / 30 losses / 1 engine truncation
on 32 teacher games, and update 144 scored 0/8. Its 32-game evaluation contained
no research actions and averaged 4.03 maximum cities. All checkpoints and traces
remain available. The pure-RL branch has not demonstrated adequate full-game
strength; the strong candidate above is explicitly imitation plus PPO.

Keep the adjacent `.json` configuration/hash file with the `.npz`. The host
adapter lazily loads/checks the model and selects from the exact public legal
catalog. Training uses a batched device-only path. Deterministic inference has
performed better in development; the final deployment mode has not been frozen.

For the former update12 incumbent specifically, stochastic play scored30/32 against
the teacher versus32/32 deterministically on identical development maps.
With identical weights on both sides, deterministic play won24–8 in a direct
paired-seat match. All64 new game traces were hash verified with no unresolved
outcomes. These one-sampling-seed results support deterministic decisions for
this incumbent, not a universal conclusion for later checkpoints.

Inside the Modal evaluation container, run a paired-seat development benchmark:

```bash
python /workspace/scripts/evaluate_structured_bot.py \
  --checkpoint /training/champion51296-05c414466823/final.npz \
  --output /retained/qualifications/my-new-development-evaluation \
  --opponent balanced --maps 16 --seed-base 300000 \
  --deterministic --seconds-limit 1800
```

Use a new output directory. The evaluator's historical default seed is the final
partition, so always pass `--seed-base 300000` for development. Seeds
400000–400031 are reserved for final acceptance, not checkpoint selection.
`--opponent random` uses a non-resigning family-uniform baseline. The unchanged
host `ScriptedTeacher('balanced')` is the stronger benchmark; the device-side
training heuristic can choose different tied actions and is not substituted
for that final benchmark.

`scripts/evaluate_structured_match.py` runs paired-seat development games
between two checkpoints. Both sides default to deterministic decisions.
Use `--challenger-stochastic` and/or `--reference-stochastic` with an explicit
`--sampling-seed` for an inference-mode comparison; the protocol records each
side's mode and sampling seed. Supplying the same checkpoint on both sides
isolates the choice rule without changing weights. Stochastic outcomes depend
on sampling as well as map seeds, so one seed is not a robust mode ranking.

The native engine's 100-round / 10,000-action limits are unchanged. Engine
truncations and evaluator cutoffs are unresolved outcomes, never draws, wins or
score-adjudicated victories. Confidence intervals are reported alongside wins
over all games. High random-opponent win rates alone have not predicted teacher
strength in these experiments.

## Training and artifacts

`experiments/modal_structured_repair.py` runs bounded pilots;
`experiments/modal_structured_scale.py` runs longer continuations. Both target
Modal staging and the separate `polytopia-structured-rl` volume. Standard runs
use one L40S; a bounded cost-efficiency pilot tests one A100-80GB with 1,024
lanes. Both allocations cap CPU/RAM at eight cores and 32 GiB, with no retries.
Short jobs reserve
$5 with a one-hour hard limit; long jobs reserve $35 with a 12-hour hard limit
and at most 11 hours requested training. The atomic ledger at
`runs/structured-ppo-v3/budget.json` records every allocation and enforces
unique run IDs; it no longer enforces a campaign cap. Reservations are cost
bounds, not provider invoices.

Hardware pilot rates are recorded in the ledger. Terminal accounting rejects a
lower rate than the run recorded; the A100 pilot uses a conservative$3.14/hour
instead of the L40S rate. This pilot does not enable long A100 allocations.
At equal 524,288 learner decisions and 4,096 new optimizer steps, A100/1,024 lanes
took 1,337.30 seconds of updates versus 1,208.99 for L40S/512 lanes: 10.6% more
time and 34.4% higher estimated update-time cost. Both passed numerical checks.
This is a comparison of operating configurations, including different batches
and experience schedules, rather than an isolated hardware benchmark. Startup
and evaluation are excluded from those costs. Its matched strength comparison
finished 16–16 over 32 games, with all traces verified. This operating point has
not earned a long allocation.

Each allocation requires a unique run ID. Durable function-call IDs are stored
in the ledger; `scripts/poll_structured_run.py RUN_ID` observes the existing job
without resubmitting it. A timeout is not evidence that a job stopped.

Runs retain model/Adam checkpoints, configuration, exact source archives/hashes,
training metrics and evaluation reports. Cloud checkpoint watchers preserve
immutable intermediate snapshots and queue public-API development games.
The bounded CPU-only Modal retainer reads the training volume through a read-only mount and
keeps hash-verified snapshots plus archived source in the separate
`polytopia-structured-rl-retained` volume. It cannot submit or stop training.
The one-core/2-GiB job reserves $2, has a 12-hour hard cap and no retries, and
uses the explicitly declared $0.07/hour accounting profile. Existing GPU
accounting floors remain unchanged. Its first 11 remote snapshots passed
independent model-file readback hashes. Local evaluation watchers are stopped;
model files and inference stay in Modal. Small metadata reports can be fetched
locally. Each new training run needs an explicit cloud retention assignment;
the running services do not automatically discover new run IDs.
Continuation restores model and optimizer; episode state, RNG state and the
full historical opponent bank are not process-resumed.

Because the eight-game teacher screen now saturates for many checkpoints,
`experiments/modal_structured_comparisons.py` runs independent fixed-reference
match watchers inside Modal, against champion-512 updates12 and96. Each watcher
uses one bounded CPU evaluator at a time and visits configured runs in
round-robin order. Eight-game screens remain separate from full benchmarks.
The qualified GPU arena screens larger frozen rosters on paired development
maps; promising candidates then receive independent public-API qualification.
Existing partial outputs are preserved, and polling timeouts never trigger
automatic restarts. Small-screen wins alone cannot promote a checkpoint.

Verify frozen engine sources before training with
`.venv/bin/python scripts/verify_ruleset.py`; this hash-only check can run locally.
Inside Modal, verify a complete run or retained snapshot with
`python scripts/verify_structured_checkpoint.py DIRECTORY`
(add `--checkpoint latest` for retained snapshots). That checks provenance,
finite weights, a changed policy and legal host decisions; it does not certify
gameplay strength. All engine changes require a new explicit ruleset contract.

To independently reproduce a saved learned-opponent game inside the Modal
evaluation container, with the retained volume mounted at `/retained`:

```bash
python /workspace/scripts/verify_structured_replays.py \
  /retained/qualifications/later-challengers-cloud-qualification-v1/h25standard112-vs-parent96-dev300400 \
  --seed 300402 --seat 1
```

Omit both selectors to replay every recorded game in the evaluation folder.
The verifier checks frozen-engine hashes, trace hashes, every public observation,
the complete legal action and command, rewards, termination and final gameplay
statistics. It saves a report only after all selected games pass. This verifies
simulator reproducibility; it does not re-run the policy to establish which
checkpoint chose each action. Model provenance is checked separately above.
