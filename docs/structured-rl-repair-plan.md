# RL repair and strength campaign

Authorized objective: iterate on the reinforcement bot in this repository, with
a total $500 Modal budget, until there is an end-to-end somewhat decent bot.
The user subsequently strengthened the objective: use the full available budget
to train the best bot possible. Passing the minimum gates below is now a
milestone, not the stopping criterion. Spend the remaining training allocation
on measured improvements and checkpoint selection, retaining capacity for final
evaluation and respecting the $500 ceiling. Other concurrent LLM
project jobs are outside its budget and will not be touched.

User reaffirmed self-play on October18. All subsequent training allocations
use learned opponents only (current policy and past checkpoints). Scripted
opponents are retained for evaluation. Existing mixed-opponent runs remain
controls and useful checkpoint sources. Final delivery must clearly distinguish
self-play continuation from earlier imitation initialization and include verified
training provenance for the selected self-play descendant. This requirement
does not erase the full-budget optimization or held-out acceptance gates.

## Learned-only self-play learning-rate comparison

After the lower-rate self-play pilot completes sixteen updates with finite
weights, clean collection and both random-play diagnostics at 32/32, test
learning rate 1e-4 against its 3e-5 control. Start from the same immutable
champion96 weights and Adam, seed20261031, 512 lanes, horizon64, 2048-map pool,
gamma.995 and entropy.005. Both use only current/historical learned opponents,
with no new demonstration labels. Request sixteen updates / 524288 learner
decisions, 2400 training seconds and a one-hour hard cap, reserving $5 after
terminal accounting frees capacity. Independently retain checkpoints in Modal.
This is a bounded learning-rate hypothesis, not an established improvement;
report actual optimizer steps and separately qualify each checkpoint. The
short/long lower-rate repetitions already diverge numerically after their first
update, so common initialization and seed are not bitwise reproducibility.

## Learned-opponent diversity from the low144 candidate

Low144's initial GPU screen is44–20 against incumbent96 and its first public
teacher check is32/32. Broader qualification is still pending, so use it as a
bounded experimental initialization, without promoting it. Compare two
sixteen-update,512x64 self-play pilots, both starting from the exact low144
model/critic/Adam,seed20261031,pool2048,lr3e-5,gamma.995,entropy.005.
The control samples current/history equally; the other samples frozen learned
incumbent96 and frozen240 in25% of rollouts, with remaining probability split
equally between current/history. Every training opponent is a neural policy.
Each reserves$5 for2400 training seconds/3600 hard cap; one$2 cloud retainer
captures both. Test whether persistent learned rivals reduce specialization
to recent opponents. Compare actual learner decisions and optimizer steps,
direct paired development games, their own low144 parent and other self-play
descendants. Initialization includes earlier imitation/PPO; no new labels.

## Map diversity qualification (H11)

After the gamma-.999 update-48 checkpoint reached 29/32 teacher and 32/32
random development wins, compare 512 versus 2,048 native training maps.
Both start from the identical retained actor/critic/Adam, use seed 20261025,
512 lanes, 64 learner decisions per rollout, eight PPO updates, gamma .999,
native rewards and the same 25% public / remaining current-historical mixture.
Each has a 2,400-second soft training allocation and one-hour hard cap, $5
reservation. If either soft cap limits updates, report actual decisions and
do not call the completed runs sample matched. Evaluate legality/replay,
memory, throughput and teacher strength before promoting the broader pool.
This changes map diversity only, not the frozen rules or map generator.

## Conservative continuation qualification (H12)

The small teacher probe fluctuated from 8/8 at native-.995 update 48 to 5/8 at
update 80, then back to 8/8 at update 96. This does not establish a broad
regression, but motivates a bounded lower-rate continuation experiment. From
the qualified gamma-.999 update-48 parent, compare learning rate 3e-5 to H11's
512-map control at 1e-4. Preserve actor, critic, Adam, seed 20261025, gamma .999,
512 lanes, 64-step rollouts, eight requested updates and opponent mixture.
Reserve $5 with the same 2,400-second soft / one-hour hard cap. Verify identical
initial weights and archived learning source, then compare actual decisions,
optimizer steps, replay, throughput and teacher results. More optimizer steps
before KL stopping are not automatically better learning.

## H15: long learning-rate comparison from the qualified 512-lane candidate

H12's short lower-rate pilot had 57/64 teacher wins versus its matched control's
55/64, with map-bootstrap intervals including no gain. This remains an open
stability hypothesis. Champion-512 update 12 subsequently achieved 64/64 teacher
wins, 32/32 random wins and 20–12 against the previous champion. Compare long
continuations from its immutable `champion51212-06db83c193a7` bundle at learning
rates 1e-4 and 3e-5. Use seed 20261029, 512 lanes, 2,048 maps, native rewards,
gamma .995, entropy .005, horizon 64, two epochs, minibatches 256, 25% public
opponent and remaining current/history, with critic/Adam preserved. Frozen-rival
probability stays zero. Each reserves $35 for at most 11 requested training
hours / 12-hour hard cap. Verify identical initialization, training seed list
and archived sources; only learning rate may differ in Config. Compare learned
opponents as well as teacher scores, which are now near ceiling. Retain all
qualified earlier checkpoints. This tests a long-run hypothesis; the short
pilot did not establish that the lower rate is better.

## H16: longer trajectories at the same rollout decision count

Test 256 lanes × 128 learner decisions against H15's standard-rate 512 × 64
continuation, using the same immutable champion-512 update-12 parent, seed
20261029, 2,048-map pool, gamma .995, learning rate 1e-4, entropy .005, native
reward and opponent mix. Both collect 32,768 learner decisions per update.
This changes trajectory length and parallel lane count together; it is not a
single-variable estimate of horizon alone. The hypothesis is that longer
within-trajectory credit assignment improves strength despite fewer independent
lanes. Start with eight requested updates, at most 2,400 training seconds and a
one-hour hard cap, reserving $5. Qualify GPU memory, replay, real optimizer steps
and gameplay before any long allocation. Use H15 control's retained update-eight
checkpoint for the sample-count comparison, reporting actual completed work.
No engine, network or PPO algorithm change is required; the pilot launcher
permits 128-step rollouts only for native-map PPO with at most 256 lanes.

## H17: longer frozen-rival continuation from the current candidate

H14's frozen-rival pilot tied its control at 25/32 teacher wins, but won their
direct match 24–8 with no unresolved games. This supports a longer test of
learned-opponent diversity, not a claim of universal improvement. Allocate one
$35 continuation from `champion51212-06db83c193a7`, matching H15's standard-rate
control in seed 20261029, 512 lanes, 2,048 maps, gamma .995, native rewards,
learning rate 1e-4, entropy .005, 64-decision rollouts and restored critic/Adam.
Change only the rollout opponent mixture: 25% public heuristic, 25% uniformly
sampled frozen rivals, and the remaining half split between current/history.
The immutable rival bank is the same two policies used in H14: gamma-.999
update 48 and zero-entropy update 64. Rivals provide stochastic actions, no
labels. Use the existing H15 standard-rate job as control rather than launching
a duplicate. Verify identical source archives, initial weights and map seeds;
compare retained checkpoints at matched decision counts and report optimizer
steps separately. Request at most 11 hours with a 12-hour hard cap. The new
parent and larger batch/pool differ from H14, so this is also a transfer test.

## Acceptance gates established before further training

- Keep `bardur-1v1-v1` mechanics and the 22 frozen simulator files unchanged.
- Public observations only at inference, legal commands, loadable artifact and
  working existing-environment adapter; no scripted fallback inside the bot.
- Correct probability, entropy-gradient, terminal/truncation and checkpoint tests.
- Healthy state-dependent actor features and critic, measured throughout learning.
- Learn capture/combat from reward on held-out tactical positions first. These
  are prerequisite diagnostics, never evidence of full-game strength.
- Final gate on untouched 14x14 native maps: at least 80% native wins versus the
  non-resigning random baseline and at least 50% versus the existing balanced
  scripted teacher across 64 games each, paired seats. Report confidence
  intervals, truncations/cutoffs as unresolved, expansion/combat diagnostics,
  and representative complete replays. Test map seeds 400000-400031 are reserved;
  development evaluation uses 300000-300015. Training starts at 200000.
- Assess stochastic and deterministic inference on development maps; freeze the
  selected mode before final test. Failure on that test requires a new untouched
  test set after further iteration, not repeated selection on the test set.
- After selecting the two update-48 native-reward candidates, use an additional
  development partition, seeds 300100–300115 with both seats, to check whether
  their 29/32 results generalize beyond the repeatedly used selection maps.
  Declare this partition before running either candidate. It is a development
  replication, not the final test; final seeds 400000–400031 remain untouched.
- Continue self-play training with a snapshot pool and fixed-opponent checks.
  Outcome reward is the initial hypothesis. If shaping or demonstration
  bootstrap is necessary, version it explicitly and measure its contribution;
  retain a genuinely learned RL policy without inference-time scripts.

## First hypotheses and experiments

H1: branch-entropy derivatives were missing. Replace the old gradient with
importance-weighted joint entropy; compare every parameter gradient to exhaustive
enumeration both at and away from the behavior policy.

H2: unnormalized residuals and global tanh caused representation saturation.
Use normalized residual activations, normalized global features, mean/max
spatial pooling and a separate critic projection. Preserve legacy checkpoint
interpretation via explicit architecture metadata. Start fresh.

H3: tiny bootstrap residuals should not drive full-strength policy updates.
Floor the advantage standardization divisor at .05. Log raw advantages, target
variance, value variance, feature sensitivity, KL and gradient norms.

H4: terminal-only native captures can be learned from short starts, then expanded
to longer starts and full-map self-play. Randomize tactical locations and both
player identities. Mask voluntary resignation in the policy, preserving all
other legal actions and the engine rules. Passing opponents are only for the
first capture curriculum, not final opponents.

H5 (after native transfer probe): the movement checkpoint's stochastic policy
expanded to 2.03 cities but won only 1/32 games, with 31 engine truncations and
1,283 recruit / 1,275 disband commands. Test outcome-only full-map self-play
against a matched own-state potential-shaping variant, both initialized from
the qualified combat checkpoint. Preserve actor weights, reset only the scalar
critic output head and Adam moments in both branches; the curriculum's near-1
value prior is inappropriate for symmetric self-play.

The planned pair uses identical map pools, seed, 32 lanes, 64-step rollouts,
64 requested updates and a 2,400-second training cap. Report actual updates and
decisions; if the cap stops one branch earlier, do not call compute matched.
Both use the same 50% current / 50% historical snapshot opponent scheme.
Measure native wins, unresolved games, army retention, actions and expansion;
shaped reward is never reported as a native victory.

Potential v1 uses only own city count, own army health, own production and own
explored tiles. Reward is native + gamma*Phi(after) - Phi(before), terminal
Phi=0; truncations retain Phi and bootstrap the shaped critic. Its bounded
potential cannot manufacture return from a recruit/disband cycle, and complete
discounted returns differ from native returns only by the initial potential.
This is still pure RL, with no demonstrated action labels or inference script.
Intermediate shaped rewards are not zero-sum; the native discounted objective
ordering is preserved separately for each seat. Record shaped value semantics
in checkpoint configuration/protocol. Six reward tests cover return telescoping,
terminal/truncation treatment, cycles and exclusion of enemy private state.

H6: weak self-play opponents fail to punish army destruction and inactivity.
Add a separately versioned device public-state heuristic opponent to a fraction
of native-map rollouts. Its scores match the balanced host teacher on tested
positions, but its randomized tie-break differs; it is not an identical policy
or a qualified expert. The unchanged host teacher remains the evaluation
opponent. The learner receives only its own PPO transitions, with no action
labels in this branch.

H7: the previously authorized trajectory bootstrap supplies useful movement,
army and economy behavior before full-game PPO. A separate explicit imitation
branch collects the device teacher's legal labels on native maps 210000 onward.
The first experiment uses teacher-only roll-in, 128 lanes and 16 atomic steps
per collection, four minibatch epochs, and fresh Adam. It starts from the combat
checkpoint, then will be evaluated and continued with PPO if useful. Later
online imitation may label learner-visited states. This branch is not pure RL;
its final bot still uses only its learned network at inference. Keep the pure
RL branch and distinguish imitation gains from subsequent RL gains.

The first full-map native/potential comparison remains unchanged. Its durable
jobs survived the earlier cancellation interval and produced updates, allowing
the independent bootstrap pilot to run concurrently. Log actual throughput
before committing the bulk of the budget to larger allocations.

## Spending

Modal published prices checked October 17, 2026: L40S $0.000542/s,
CPU $0.0000131/core/s and RAM $0.00000222/GiB/s. One L40S with hard limits of
8 CPU and 32 GiB is approximately $2.584224/hour before discounts/credits.
Source: https://modal.com/pricing . No invoices are inferred from estimates.

Short pilots have a one-hour hard timeout, no retries, one container, and at most
40 minutes requested training, leaving time for compilation/evaluation. Reserve
$5 per run including startup margin. Longer qualified continuations have a
12-hour hard timeout, at most 11 training hours and a $35 reservation. Reservations
remain charged until a terminal-accounting pass;
`runs/structured-ppo-v3/budget.json` prevents a launch that
would exceed $500. Record app IDs, live state, actual observed durations and
outcomes. Do not relaunch on a transient observation timeout.

To honor the expanded full-budget objective, an explicit accounting helper can
replace a finished app's unused reservation with its observed full app wall
duration at the maximum declared allocation rate, plus a $1 overhead margin,
rounded upward to cents. It requires a fresh matching Modal app receipt showing
stopped and zero tasks, validates nonnegative duration, and refuses duplicate
accounting or an amount exceeding the reservation. Preserve the original
reservation and receipt. Running/missing/mismatched apps never release funds.
This remains conservative compute accounting, not a settled provider invoice.
Published L40S, CPU and RAM rates were rechecked on October 18 and unchanged:
https://modal.com/pricing . Six budget/concurrency/accounting tests pass.

After the first two serial runs, reservations/status updates use a separate
file lock and atomic replacement so independent experiments can run concurrently
without oversubscription or lost records. Keep only useful hypothesis-driven
jobs running; parallel allocation does not change the $500 total ceiling.

## Initial local evidence

21 targeted tests passed, including entropy gradients, bounded normalization,
representation sensitivity under weight scaling, collector credit assignment,
legacy checkpoint and public-interface tests. All 22 frozen source hashes match.
These prove implementation properties, not gameplay strength.

## Longer credit horizon ablation

H7: discounting by 0.995 per learner decision may overvalue quick outcomes in a
long strategy game. A reward 500 learner decisions away has weight about 0.082;
with gamma 0.999 it has weight about 0.606. This arithmetic motivates a test,
not a claim that discounting caused the original failure or that all games last
500 learner decisions.

Compare a native-reward gamma-0.999 PPO branch with `scale-bootstrap-native2`
(gamma 0.995), using the same verified bootstrap parent, seed 20261020,
512-map pool, 128x64 rollout, learning rate 1e-4, entropy 0.005, lambda 0.99,
25% fixed-public-opponent mixture and initial scalar-critic/Adam reset. Keep the
engine's own 100-round/10,000-action limits unchanged. Both branches use the
same replay audit. The newer launcher's unused parent-checkpoint option does
not alter the learner. This is an imitation-initialized PPO comparison, not
pure RL. Report actual updates/decisions and avoid calling asynchronous
wall-time snapshots compute matched.

Use the existing qualified long-run allocation: 11 training hours requested,
12-hour hard cap, $35 reserved. Retain checkpoints and evaluate only development
maps until final policy selection is frozen. This tests delayed-credit utility;
it does not change the reward to a scripted score or add labels during PPO.

## Bootstrap entropy ablation

H8: an entropy bonus may erase useful imitation priors before sparse outcome
feedback arrives. In `scale-bootstrap-native2`'s first 8,192-decision batch,
there were zero native rewards, zero advantage variance, zero policy/value
loss, but nonzero entropy gradients and mean reported approximate KL about
0.01993. The gamma-0.999 branch shows the same first-batch effect. The shaped
branch instead had nonzero advantages before the first native ending. Exact
logs are archived in `early-entropy-diagnostic.json`. This establishes that the
first native update was entropy-driven; it does not establish that the bonus
caused later gameplay regressions.

Test native PPO with entropy coefficient **zero** against the existing 0.005
branch. Use the same verified bootstrap parent, reset scalar critic and Adam,
seed 20261020, gamma 0.995, lambda 0.99, 512 maps, 128x64 rollout, learning rate
1e-4 and 25% fixed-opponent mix. Sampling remains stochastic during training;
zero bonus does not mean greedy rollouts. Start from fresh Adam to match the
control, and retain the pure-RL and shaped branches unchanged. This comparison
uses imitation initialization but no additional labels during PPO.

The existing long-run allocation is bounded at 11 requested training hours,
12-hour hard timeout and a $35 reservation. Compare development wins and
unresolved outcomes at matched learner-decision counts where checkpoints allow;
retain earlier checkpoints if extra training regresses. Do not infer gameplay
strength from entropy, KL or self-play win fractions alone.

## Easier full-map opponents for pure RL

H9: the pure-RL learner needs positive full-game capture outcomes alongside
harder opponents. In `scale-pure-potential1`, 122,880 new learner decisions
produced 4 native wins and 164 losses, with no truncations. Its first three
quick teacher probes also regressed from the parent's small-sample score.
This motivates an opponent-curriculum pilot, not an inference that random-game
performance alone is sufficient.

Add a separately configured non-resigning, family-uniform random opponent.
The pilot uses disjoint rollout probabilities: 25% public heuristic, 25% random,
25% current policy, 25% historical policy. The existing pure continuation uses
25% public, 37.5% current and 37.5% historical. Episode state persists across
rollouts in both recipes, so a rollout's opponent tag must not be reported as
an isolated per-opponent full-game win rate.

The new collector path uses the existing random-action sampler and unchanged
native simulator. It supplies no imitation labels and changes no rewards.
Retain the same immutable pure update-32 parent, optimizer/critic, seed
20261021, 512-map pool, 128x64 rollout, potential reward and PPO settings as
`scale-pure-potential1`. Run a short 40-minute / one-hour-hard-capped pilot,
reserved at $5, before choosing a longer curriculum allocation. Compare
checkpoints on the same public teacher and random development benchmarks.
Default random-opponent probability is zero; existing jobs and their archived
source remain unchanged. The zero-probability path preserves the prior RNG
consumption and public-opponent selection.

## Rollout throughput qualification

H10: 256 or 512 parallel games may improve useful learner decisions per L40S
second over the present 128 lanes. This is an empirical capacity hypothesis,
not a claim that larger PPO batches are strategically equivalent or better.

Run three bounded eight-update continuations at 128, 256 and 512 lanes from the
same immutable shaped-PPO update-16 checkpoint (32/32 random development wins;
16/32 balanced-teacher development wins). Keep horizon 64, minibatch 256,
learning rate 1e-4, entropy 0.005, gamma 0.995, lambda 0.99, 512 maps, seed
20261023, preserved actor/critic/Adam and the 25% public-opponent mix fixed.
Each reserves $5 and retains the one-hour hard limit / 40-minute soft limit.
These are eight-update qualifications, not 40-minute commitments. Three
independent app handles avoid a shared-worker OOM disrupting another trial.

Measure actual decisions/second after compilation, cumulative training wall
time, replay/legality health and GPU allocated/peak/limit bytes. Larger lanes
also increase PPO samples and potential minibatch updates per rollout, so
report actual decisions and optimizer behavior, not only update counts. All
three trials use the same new telemetry; the legacy runs omitted gradient
probe time from their per-update timing and are not exact timing controls.
Start a larger-lane long run only after its benchmark demonstrates healthy
collection, numerical replay and adequate memory headroom.
Keep the simulator frozen; do not change mechanics to improve throughput.


## Learned-opponent development matches

Teacher/random benchmarks alone can miss specialized opponent weaknesses. Add
paired-seat deterministic matches between frozen learned checkpoints on separate
development maps 300200–300215. Validate seat attribution first by playing a
checkpoint against itself on one map in both seats: the underlying action
sequence must match and terminal rewards must reverse when the challenger seat
reverses. Keep full public-observation/action traces, exact checkpoint hashes,
and unresolved engine truncations/cutoffs. Use this opponent-diversity check
alongside teacher/generalization results; neither substitutes for final acceptance.

The first learned-opponent portfolio contains native-.995 update 96,
native-.999 update 48, and zero-entropy native update 64. Run every pair on the
same 16 maps (300200–300215), both seats, with deterministic inference. Report
pairwise counts and unresolved games before selecting a continuation parent;
do not infer a universal ranking from teacher scores alone.


## H13: Strength per training hour from the qualified champion

After native-.995 update 96 passed 32/32 original teacher, 30/32 replication
teacher and 32/32 random games, and scored 20–12 against gamma-.999 update 48,
start two long continuations from its identical model/critic/Adam. Both use
2,048 native training maps (execution-qualified by H11), seed 20261026, gamma
.995, native reward, learning rate 1e-4, entropy .005, horizon 64, minibatch 256,
two PPO epochs and the existing opponent mixture. Vary only lanes: 128 versus
512. Each reserves $35, requests 11 hours of training, and has a 12-hour hard
cap. This is a wall-time/cost comparison, not a sample-matched comparison.
Record optimizer steps, throughput and teacher/learned-opponent scores; faster
collection alone does not decide the winner. The map-pool strength ablation
continues separately; execution qualification does not prove pool superiority.


H13 adds a 512-map, 128-lane control after H11's eight-update results showed no
teacher improvement from the wider pool (26/32 control versus 22/32 wider pool,
parent 29/32). `scale-champion128-smallpool-v1` starts from the same champion
with seed 20261026 and exactly the same H13 settings except pool size 512
versus 2,048. Reserve $35, request 11 hours, hard cap 12 hours. This keeps a
matched long-duration control for map diversity; it does not assume the short
result establishes permanent inferiority of broader training data.


## H14: Permanent learned rivals alongside current/historical self-play

The native-.995 update-96 champion scores 32/32 original teacher wins but only
20–12 against each of two learned rivals. Test whether retaining those rivals
in training provides useful challenges that the fixed teacher does not.
A bounded pair starts from the identical champion actor/critic/Adam, seed
20261027, 128 lanes, 64-step rollouts, 512 maps, native rewards, gamma .995,
learning rate 1e-4, entropy .005 and 16 requested updates (131,072 decisions).
Both reserve $5, have a 2,400-second soft training allocation and one-hour hard
cap. Control: 25% public teacher, remaining mass split equally current/history.
Variant: 25% public, 25% uniformly sampled frozen learned rivals, 25% current,
25% history. The two frozen rivals are gamma-.999 update 48 and zero-entropy
update 64, using their verified immutable bundles. Their weights are fixed but
actions are sampled stochastically during training. They provide no action
labels, and only learner transitions enter PPO. Selection remains per rollout,
so episode-level opponent win rates cannot be inferred from rollout tags.

The optional selector must preserve the prior RNG stream exactly when disabled,
reject invalid probability sums/empty pools, record rival hashes and chosen
indices, and use the existing collector/replay checks. No engine or inference
changes are involved. Compare teacher, learned-parent/portfolio strength,
throughput and actual optimizer work before granting a long allocation.

## H18: lower-rate continuation with frozen rivals

Complete the learning-rate × frozen-rival comparison from the immutable
champion51212 parent. H15 has standard/lower rate without rivals; H17 has
standard rate with rivals. H18 uses rate3e-5 with the same25% frozen-rival mixture
and two immutable gamma99948/noentropy64 rivals as H17. Keep seed20261029,
512 lanes, horizon64, pool2048, native gamma.995, entropy.005, two PPO epochs,
minibatch256 and restored actor/critic/Adam. Both existing component treatments
are numerically qualified, but neither has demonstrated a long-run advantage.
The hypothesis is an interaction between more conservative policy updates and
stable learned competition. Use H15 lower-rate and H17 standard-rate/frozen
jobs as controls. Opponent-selection RNG can diverge after differing KL stops;
report actual optimizer steps and selection history, not identical experience.
Reserve$35 for11 requested hours with a12-hour hard cap. Verify initial weights,
source archives and seeds before interpreting results. This is another training
candidate, not an assumed improvement, and never changes the final-test maps.

## H19: second training seed for lower-rate native PPO, with and without rivals

Reallocate the weak pure-RL and potential-shaped continuations toward a matched
second-seed pair of the numerically qualified native-reward recipes. Both start
from immutable champion51212 with seed 20261030, learning rate 3e-5, 512 lanes,
horizon 64, pool 2048, gamma .995, entropy .005, two epochs, minibatch 256 and
restored actor/critic/Adam. Control uses 25% public and remaining current/history;
treatment additionally assigns 25% to the same two immutable rivals as H17/H18.
This replicates the H15 lower-rate/H18 comparison under a different training RNG
seed. Training map pool stays identical; only ordering and stochastic trajectories
change. Each gets a $35 reservation, 11 requested hours and 12-hour hard cap,
only after the terminal receipts free sufficient budget. Verify identical parent,
source archives and seed lists. Compare actual optimizer steps and paired-map
learned opponents; do not assume that one good checkpoint establishes a robust
recipe. Multiple training seeds also provide independent candidate trajectories
for the full-budget strength search. Final acceptance maps remain untouched.

## H20: remove entropy bonus from a stronger native-PPO parent

After older native512 update64 lost9–23 against champion51212, reallocate that
branch's unused reservation to an entropy ablation from the stronger parent.
Match H15 lower-rate: champion51212 actor/critic/Adam, seed20261029,512×64,
pool2048, rate3e-5, gamma.995, native reward, two epochs/minibatch256,25% public
and remaining current/history. Change only entropy coefficient .005→0.
The hypothesis is that removing the explicit exploration bonus may better
preserve/refine an already competent policy; the risk is insufficient exploration.
The older zero-entropy branch started from a different parent, so it does not
resolve this contrast. Rollout entropy also depends on encountered states and
legal-action counts; its increase is not proof of harmful policy drift.
Reserve$35 only after terminal accounting leaves sufficient budget; request11
hours with12-hour hard cap. Verify identical initial/source/map state and report
actual optimizer work. Compare matched retained checkpoints on fresh development
maps300300–300315 and use ordinary full learned/teacher qualification before any
promotion. Final maps remain untouched; no source/engine change is needed.

## H21: stronger immutable rivals, bounded pilot

If the full development match confirms the small-pool branch remains weak,
stop only that owned job, verify its terminal receipt and release its unused
reservation before funding this $5 pilot. The old frozen bank contains two
policies the incumbent beats22–10 each; challenging the learner with stronger
fixed rivals may be more useful than simply adding older frozen policies.
Compare with H18 at update16: same champion51212 parent, seed20261029,
512×64, pool2048, native gamma.995, rate3e-5, entropy.005, two epochs and
restored actor/critic/Adam. Keep25% public,25% frozen and remaining mass split
equally current/history. Change only the bank to immutable champion51212
(`06db83c193a7`) and qualified lower-rate16 (`d9be65407e63`). Both have completed
teacher/random and learned-opponent development qualification; neither supplies
demonstration labels to this PPO stage.

Request16 updates with2400-second soft and3600-second hard caps. Verify all
initial/source/map hashes, rival hashes, actual frozen exposure and optimizer
work; a same-seed initialization does not imply bitwise-identical training.
Compare against H18 update16 on paired development maps300300–300315, then
broaden qualification if promising. The incumbent and lower-rate16 are now
explicit training opponents in this variant, so results against them must be
labeled accordingly and cannot alone establish general strength. No long
allocation or promotion follows from the pilot's completion alone. Final maps
remain untouched. This plan does not itself reserve or spend funds.

## H22: longer reward discount horizon from the qualified incumbent

Earlier gamma.999 branches began from different parents, so their later weakness
does not isolate the discount setting. Test gamma.999 against H15 lower-rate's
gamma.995 from the identical champion51212 actor/critic/Adam, seed20261029,
512×64, pool2048, rate3e-5, entropy.005, lambda.99, native reward, two epochs,
minibatch256 and25% public/remaining current-history mixture. No frozen rivals.
This is a longer return-discount horizon, not a longer collected trajectory.
The hypothesis is better credit for delayed wins; possible costs include
greater target variance and adaptation of the restored critic to new returns.
Preserve that critic consistently with the matched control rather than adding
a second treatment. Gamma and actual optimizer work are recorded explicitly.

Use an existing$5 of headroom for16 requested updates,2400-second soft and
3600-second hard caps. Verify starting weights, archived sources, map lists
and the single training-hyperparameter difference. Compare update16 against
H15 lower-rate16 on development maps300300–300315, then use teacher and learned
portfolio checks if promising. A pilot result does not automatically authorize
a long allocation in the experimental plan; use measured results and available
budget. Final acceptance maps and the frozen engine remain unchanged.

## H23: self-play-only continuation from the competent incumbent

The public heuristic was useful for early reward learning, but may be less
useful after the policy reaches64/64 teacher development wins. Compare against
H15 lower-rate using the same champion51212 actor/critic/Adam, seed20261029,
512×64, pool2048, native gamma.995, rate3e-5, entropy.005, lambda.99, two epochs
and minibatch256. Set public-opponent probability25%→0%; no random or frozen
rivals. The remaining mixture becomes50% current/50% recent history. The
hypothesis is more useful experience against stronger learned policies; risks
include cycling, forgotten baseline skills and weak self-play conventions.
Selection is still per rollout, and episode state continues between rollouts.

This is self-play-only continuation of an imitation-initialized policy, not a
new pure-RL-from-scratch result. Use$5 headroom for16 requested updates with
2400-second soft/3600-second hard caps. Verify the exact initial/source/map
state, absence of public/random/frozen selections and actual optimizer work.
Compare against H15 lower-rate16 on300300–300315 and check baseline retention
before any expansion. No automatic long allocation; no final-map use or engine
changes.

## H24: A100-80GB / 1024-lane cost-efficiency pilot

Active runs spend85.7–96.3% of measured update time collecting play, and the
512-lane L40S operating point peaks near25GB of device allocations. Test whether
1024 lanes on an A10080GB improve learner decisions per dollar. This changes
hardware and lane count together; it is an operating-point comparison, not an
isolated hardware or batch-size effect. Memory fit and speedup are hypotheses.

Use the same champion51212 parent, seed20261029, horizon64, pool2048, native
gamma.995, rate3e-5, entropy.005, lambda.99, two epochs/minibatch256,25% public
and remaining current/history. Request8 updates:524288 learner decisions,
matching H15 lower-rate's16 updates at512 lanes. Actual KL-limited optimizer
work and opponent selections may differ and must be reported. Compare model
legality/strength as well as memory, total update time and decisions per dollar.

Reserve$5 with2400-second soft/3600-second hard limits, one A100-80GB,8 CPU,
32GiB RAM, one container, no retries/fallback. Modal's2026-10-18 listed rates
give approximately$3.131424/hour versus$2.584224/hour for the L40S allocation:
about21.2% extra throughput is required to break even. Use a conservative
$3.14/hour accounting rate and never the L40S rate for this job.
Sources:[pricing](https://modal.com/pricing),
[GPU configuration](https://modal.com/docs/guide/gpu).

The short launcher alone permits this hardware/lane combination. Record actual
device identity; no long A100 allocation is implemented or implied. Budget
accounting now rejects a rate below the run's recorded rate, and decimal
rounding preserves exact cents while rounding fractional cents upward. This
changes the archived budget module only; PPO/inference and all frozen engine
sources remain unchanged. Verify that distinction against the control manifest.


## H25: learning-rate comparison from the stronger update96 parent

Champion512 update96 beat incumbent12 by46–18 over64 development games,
including20–12 on the added map subset. After completing teacher/random/common-
reference qualification and publishing its immutable bundle, allocate paired
continuations from exactly`champion51296-05c414466823`: standard rate1e-4 and
lower rate3e-5, both new seed20261031. This tests preserving and improving the
stronger strategy; it does not assume the lower rate is better. The original
champion512 run remains active.

Both use512 lanes,64-decision horizon,2048-map pool,native gamma.995,entropy.005,
lambda.99,two epochs/minibatch256,25% public opponent and remaining current/
historical self-play; no random/frozen rivals or new imitation. Restore critic
and Adam, but episode/RNG/history state is freshly initialized, not resumed.
Each reserves$35 for one L40S+8CPU+32GiB,11 requested hours/12-hour hard cap,
no retries. Current$420.08 commitment permits the$70 pair with$9.92 headroom.
Use ledger state at actual submission, not this planning snapshot.

Parent archives38 files; current code has39, with the previously qualified
optional-opponent helper and rate-aware budget accounting. Audited repair.py
changes only factor opponent selection and add optional-rival metadata; three
zero-frozen tests preserve old selections/RNG. PPO/network/inference/engine
sources are unchanged. New pair must match each other on all39 source hashes,
initial weights and train seeds; Config must differ only in learning rate.
Report actual decisions and Adam work at matched updates, screen against the
new parent separately from the existing fixed-old-reference monitor, and keep
all previously qualified parents. Final maps remain untouched.

## Stronger ancestor for learned-only continuation (October18,16:53UTC)

The predeclared historical screen on fresh301000–301031 found low176 at54–10
against incumbent96, versus low14438–26 on those same64 games. The paired
common-opponent win-rate difference is+25 points with exploratory95% interval
[12.5,37.5]; selecting from12 candidates still requires independent replication.
Public qualification now tests fresh301200 maps against incumbent96 and low144,
plus teacher/random baselines, common references and inference modes.

In parallel, continue exact low176SHA565eb6e315c82b970c0d0694f448aaf5a798de26c41deb7ad9b5436de4489052
with learned-only current/history self-play. Hypothesis: better initialization
improves the resulting self-play policy, even if early updates fluctuate. Keep
seed20261031,512lanes×64,lr3e-5,gamma.995,entropy.005,pool2048 and restore
critic/Adam. No new imitation labels, public opponents, random opponents or
frozen-rival probability. The earlier ancestry remains imitation plus mixed PPO.
Request10800 training seconds with14400 hard cap,$12 reservation, and independent
cloud retention. Evaluate early retained16/32-update stages against their own
parent and other self-play descendants; training wins are not a strength gate.
This is an experimental initialization, not an incumbent promotion. Retain final
acceptance capacity and use later terminal releases for measured follow-up work.
