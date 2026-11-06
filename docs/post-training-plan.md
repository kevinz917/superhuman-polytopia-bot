# Polytopia post-training and transfer experiment

2026-10-16. This is the comprehensive plan after freezing [bardur-1v1-v1](ruleset-v1.md). It supersedes conflicting version-chasing, fidelity gates and mandatory-critic recommendations in older proposals. The existing [SFT checkpoint plan](sft-checkpoint-plan.md) supplies detailed SFT implementation checks; neither document is evidence of a trained model. This task ends with simulator validation and this plan. Teacher construction, corpus generation and training are subsequent work.

## Hypothesis and what would count as success

**SFT and/or RL on Polytopia improves a Qwen 4B model's ability to produce successful, high-quality trajectories on other tasks.** Better Polytopia play is a learning diagnostic, not the primary scientific result. A more complex game might improve planning practice, but could equally make learning slower or less transferable.

Define an expert-quality trajectory operationally: it passes the downstream task's objective verifier, obeys its constraints, and fits a predeclared action/token budget. Also report efficiency among successful trajectories against available reference solutions. Do not label a trajectory expert because its prose sounds thoughtful, because it won a simulator game, or because an LLM judge liked it. Avoid claiming optimality when the reference is only competent.

Predeclare a primary cross-task success metric and untouched evaluation instances before fitting. A useful research success threshold is a five-percentage-point absolute improvement over the unchanged model, with a paired 95% confidence interval above zero and gains in at least two task families. This is a proposed threshold, not a forecast. Report the full interval if the study is underpowered, all failed training seeds, and all task-family regressions.

## What to borrow from the supplied research

[MARSHAL](https://arxiv.org/html/2510.15414v2) trains a shared language policy in competitive and cooperative games, computes future returns before centering them, and separates advantage baselines by player role. Its transfer evidence includes multi-agent reasoning evaluations; that is different from demonstrating better single-agent tool-use trajectories. We should test both settings separately rather than importing its headline as our expected result.

[SPIRAL](https://arxiv.org/html/2506.24119) motivates outcome-driven self-play and role-conditioned baselines. Its simple-game results support the experiment, not an assumption that Polytopia will yield larger gains. Our LoRA implementation, longer horizon and action-only starting interface are deliberate departures, not a reproduction of either paper.

The first RL recipe below is critic-free, with Monte Carlo outcomes and seat-conditioned baselines. PPO with a learned value function is a subsequent variance-reduction comparison if the simple recipe learns too slowly. This replaces the older proposal to require a critic from the outset.

## Freeze the environment and interface

Use 14×14 Bardur versus Bardur with fog and the v1 generator; 11×11 is a curriculum/debug setting. Pin the ruleset manifest, simulator hashes, Python/JAX/CUDA environment and generator revision. Preserve the distinction between elimination/resignation and time/action-limit truncation. Commercial-version replication is not a prerequisite for this experiment; claims must name the research environment.

Build one public observation serializer over `observe` and the complete legal catalog. Include compact rules, seat, turn, own stars/tech, visible board/entities, own city/unit details and relations. Add remembered observations and public events only through a seat-isolated history updated from actual observations. Seeds, snapshots and opponent private fields stay in trusted logs. Pair hidden-world tests at this final serializer boundary, including legal-action ordering, before collecting any teacher or model data.

For SFT-0, construct a fresh context at every decision using current public state plus [compact factual memory](sft-checkpoint-plan.md#decision-context-and-factual-memory-for-sft-0). The runner deterministically maintains separate episode/player histories: initially five recent own actions and up to 32 timestamped historical facts absent from current observations, with caps validated and frozen before prompt optimization. Preserve stale/uncertain status, use only observations available at that time, and give teacher/base/SFT the same memory interface. Carry no full-game conversation or model-written strategic notebook. Implement and test reset, seat isolation, observation-only updates and replay/restore before teacher qualification; the current tile-memory schema alone does not supply this functionality.

One call chooses one atomic command: `{"action_id":"a17"}`. IDs are local to an observation version; records also contain the semantic `(kind, source, target, parameter)`. End-turn and forced rewards are ordinary explicit choices. Never truncate the action catalog to fit a prompt. Begin with a measured 8K context ceiling and 32 output tokens; aim for 2–4K input tokens by compact encoding. If overflow is common, version a lossless compressed representation or larger context before training.

For the primary study, use unconstrained action generation with a strict parser. An invalid response receives a public error message; allow two repairs, then an agent-forfeit outcome distinct from engine elimination. Retain every failed response and its log probabilities. Report first-attempt validity separately from repaired success. A constrained-decoding arm is useful, but must use exactly matching constrained token probabilities in the learner and must be reported separately. No silent scripted fallback.

## Model, controls and splits

Start from [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), using the same model/tokenizer/chat-template revision for every arm. The earlier SFT plan records revision `cdbee75f17c01a7cc42f958dc650907174af0554`; verify it when fetching and record file hashes. It is a non-thinking instruct starting point. Do not compare its tuned version against a different base or reasoning model.

| Arm | Training | Purpose |
|---|---|---|
| B0 | None | Unchanged-model reference |
| B1 | Polytopia SFT | Demonstration transfer |
| B2 | Polytopia RL directly from B0 | Outcome-learning contribution without game SFT |
| B3 | Polytopia SFT then RL | Main practical pipeline |
| C1 | Format/interface-only supervised examples with shuffled legal labels | Separates output-format familiarity from competent strategic labels; never call the labels expert |
| C2 | Non-game supervised or verifiable-reward training | Tests whether equal training compute on another domain gives comparable gains |

Use the same LoRA capacity and comparable optimizer/processed-token budgets in matched comparisons. Report rollout tokens, optimizer tokens, successful games and total GPU-hours separately; equal game counts are not equal compute. Run all arms through a small pilot, then three independent training seeds for preregistered main comparisons. If funding covers fewer seeds, call the result preliminary. Compare B3–B1 for incremental RL, B1–B0 for SFT, B2–B0 for direct RL, and B3–C2 for domain-specific value. Do not select the best arm using final downstream test results.

Reserve at least 500 training maps, 200 validation maps and 500 final game-test maps, each with swapped seats. Check exact and symmetry-equivalent board hashes for overlap. All decisions from a map/episode stay in one split. Freeze a separate downstream development set for harness debugging and sealed final instances for transfer. Teacher tuning, policy checkpoint selection and prompt revisions use development/validation only.

## Teacher and SFT phase

Before the first game-SFT fit, follow the [bounded starting-prompt optimization protocol](sft-checkpoint-plan.md#starting-prompt-optimization-before-sft): six revisions plus the original on 100 development positions, at most 60 selection games and 40 confirmation games, under a measured spend ceiling. Use the unchanged 4B model and disjoint prompt-development maps, then freeze the chosen prompt for demonstration rendering and all base-versus-SFT gameplay comparisons. Preserve the original when improvements are unclear. This is planned work, not an implemented optimizer. APEX setup for the first exploratory transfer check remains after SFT.

The current random sampler is not a teacher. Next build a transparent, observation-limited teacher with explicit economy, expansion, combat, defense and capture priorities, plus two distinct fixed opponents. A bounded search teacher is optional after this works; under fog it must search a declared belief model, not the hidden true state. Keep decision explanations as diagnostics, not fabricated reasoning labels.

Qualify the teacher with 10,000 legal decisions, at least 95% completed games on a 200-game validation set, and a predeclared positive margin against a frozen weak baseline. Measure its strength against both distinct opponents. Failure here means repair the teacher/interface before collecting a large corpus; a rules-complete simulator alone does not make labels useful.

Collect 5–10K unique decisions for a smoke run, then 50K for the main corpus. Preserve complete source episodes, including losses and truncations, and sample labels across phases and action families. Deduplicate public inputs and inspect conflicting labels. Include recovery states; cap repetitive end turns without deleting them from source traces. Optionally replace up to 20% of the main corpus with teacher relabels of smoke-model states from training maps. Restart the main fit from the same base initialization.

Use BF16 frozen base weights with LoRA rank 32, alpha 64, dropout 0 on attention and MLP projections. Initial settings: AdamW, learning rate 1e-4, effective batch 64, 3% warmup, gradient clipping 1, one epoch, no packing. Treat these as starting settings selected on validation. Train only assistant action/EOS tokens. [TRL's SFT trainer](https://huggingface.co/docs/trl/sft_trainer) supports prompt-completion loss masking; explicitly verify the actual chat-template masks and Qwen end-of-message token.

First inspect 64 rendered examples and overfit 128 separate training examples to check gradients, frozen weights and adapter reload. Then run the smoke fit. Save checkpoints at 25%, 50% and 100% of the main epoch and select using validation gameplay, with validity and truncation as separate gates. Require at least 99% first-attempt validity and less than 5% game truncation before the main RL run. These are learning gates, not transfer claims.

## RL architecture and credit assignment

Keep the GPU simulator service separate from the model training runtime. A rollout coordinator batches current-seat observations, requests generations from vLLM, parses outputs, executes commands on JAX state batches and saves trajectories. A learner consumes the exact sampled token sequences and log probabilities. Adapter synchronization occurs only between collection batches initially. [verl's agent loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html) is a framework integration option; the game coordinator and cross-decision accounting still need custom implementation.

Freeze behavior-policy and opponent versions for each collection batch. Start with 64 complete games per update, balanced by learner seat. The provisional opponent mixture is 40% fixed bots, 40% historical model checkpoints, 20% a frozen copy of the current policy. Initially update only the designated learner seat. Both seats can later provide training data when both were actually sampled from the same recorded behavior policy; a historical opponent's trace is not on-policy learner data.

Reward only resolved outcomes: +1 win, −1 loss. V1 has no ordinary drawn terminal. Do not reward stars, territory or plausible explanations in the primary experiment. Agent forfeits are recorded separately and count as losses. A safety-limit truncation is censored, not a zero-reward draw: retain its trace and report it, but omit it from the first critic-free outcome loss. If truncations exceed 5%, stop scaling and improve the teacher/curriculum or use a separately validated bootstrap critic. Exclusion can bias learning, so report its rate and sensitivity explicitly.

For a resolved learner episode e and seat p, compute each decision's undiscounted return G[e,p,t] by summing its remaining rewards. With only a terminal outcome, this is the same outcome for every preceding decision. Set advantage A[e,p,t] = G[e,p,t] − b[p], where b[p] is a leave-one-episode-out mean of resolved outcomes for the same seat and declared opponent stratum. Use balanced batches; merge sparsely sampled strata according to a fixed rule or use a previous-batch baseline, never an outcome-dependent ad hoc choice. This is a proposed estimator inspired by the papers, not an exact reproduction.

Attach that decision advantage to its sampled response tokens, masking prompts and padding. Average tokens within a decision, decisions within an episode, then episodes, so verbosity and game length do not silently change weighting. Optimize the clipped probability-ratio objective with clip 0.2, learning rate 1e-5, one learner epoch and an initial KL coefficient 0.01 to the arm's frozen starting policy. Record actual KL, entropy, clip fraction, gradient norms, baseline counts and advantage distribution. Hyperparameters require pilot validation.

Before any RL update, reproduce stored behavior log probabilities using unchanged weights, tokenizer, temperature and decoding masks. Verify an adapter round-trip through serving, reject stale batches, and test return signs with tiny known-win/loss episodes. These checks prevent false learning from bookkeeping errors. If every result is identical, advantages can vanish; curriculum positions with solvable captures are a better first response than inventing a dense score reward.

Run a 100-game integration stage, then a 1,000-game pilot. Scale to 10,000 games only after improvement over SFT against frozen validation opponents, stable probability accounting and acceptable completion. A 100,000-game run is a later decision based on measured benefit. Critic-based PPO, GAE, asynchronous collection and search-generated rationales are controlled follow-up experiments, not first-run dependencies.

## Transfer evaluation: test trajectories, not only answers

Freeze the downstream scaffold and budgets before comparing weights. Use no Polytopia simulator, teacher or game prompt at downstream inference. Apply the same task prompt, tools, observations, decoding and retry policy to each checkpoint. No downstream fine-tuning in the primary transfer experiment.

| Family | Proposed setup | Main evidence |
|---|---|---|
| Stateful household planning | Text [ALFWorld](https://github.com/alfworld/alfworld), unseen evaluation tasks and held-out variations | Environment success, constraint violations, steps, recovery from failed actions |
| Browser workflows | A pinned local [BrowserGym](https://github.com/ServiceNow/BrowserGym) task suite with reproducible reset and programmatic validators | Correct final application state, tool errors, task completion under an identical budget |
| Code repair, secondary | Frozen small repository issues with hidden tests and isolated containers | Tests passed without forbidden changes, patch validity and tool/action count |
| Interaction transfer, separate | Fixed partner/moderator debate or collaborative task scaffold | Verifier success with the evaluated policy substituted into the same role |

The browser and repair suites need a harness pilot before registration; their availability and success rates are not established here. Begin with 25 development instances per family to ensure B0 is neither uniformly failing nor saturated. Freeze approximately 200 final instances per primary family, using all official unseen tasks when a fixed split is smaller, and pair every arm on identical instances/seeds. Never choose only tasks on which the tuned model looks better. Public benchmarks may occur in pretraining; add fresh held-out variations where possible and disclose provenance.

Primary outcome is an equal-family-weighted verified-success rate. Report paired bootstrap intervals clustered by task/template and training-seed variation. If more than one comparison is confirmatory, predeclare a correction such as Holm; publish unadjusted exploratory results separately. Also report first-attempt validity, success after repair, total tokens, latency, action count, constraint failures and a blinded sample of trajectories. Efficiency among successes is secondary and must include success denominators to avoid survivorship bias.

Keep single-agent and multi-agent scores separate. For the latter, first keep all partners frozen and replace only the tested role; replacing every agent is an additional treatment with a different question. Math/QA accuracy can be a secondary diagnostic, but cannot alone establish better interactive trajectories. Include retention checks on unrelated language/code tasks and report any loss of general competence.

## Cost, scheduling and stop decisions

The measured L4 batch-8 simulator result is about 96 mixed transitions/s, including masks and random sampling, for 32-step traces. Procedural reset took about 1.03s for eight boards after compilation. These small-batch measurements do not predict model throughput or large-batch simulator scaling. The eight-game random validation reached safety limits; it is not a demonstration of fast expert games.

At an illustrative 600 model decisions per game across both seats, 10K games mean 6M model calls. At 2,048 prompt tokens and 32 response tokens, that is about 12.3B prompt tokens and 192M generated tokens before retries and evaluation. At measured aggregate rates of 10, 50 or 200 decisions/s, rollout alone would take about 167, 33 or 8.3 hours. Multiply by the number of allocated GPUs for GPU-hours and add actual learning/synchronization/evaluation time. These are sensitivity calculations, not hardware predictions.

Profile before reserving a large allocation: one 48–80GB GPU for BF16 LoRA SFT is a reasonable starting experiment; QLoRA on 24GB is an alternative arm if needed. For RL first profile separate rollout and learner GPUs, then size replicas/sharding from actual 8K-context memory and throughput. Four billion BF16 weights alone use about 8GB decimal, but activations, caches, optimizer state and references add substantially. Historical availability of eight 4090s is not current capacity evidence.

Use a bounded four-hour model profiling session, then an 8–24-hour pilot ceiling, with dollar cost calculated from the actual provider quote. Measure calls/game, prefill/decode, model idle time, simulator time, optimizer time and adapter synchronization at 32/64/128 concurrent games. If the simulator dominates, optimize its profiled bottleneck without changing v1 rules. If prompts dominate, improve the representation rather than assuming a faster engine solves inference cost.

The execution order after this handoff is: freeze downstream evaluation and public serialization; qualify teachers/opponents; collect and validate SFT data; run unchanged-model baselines and SFT; validate RL accounting and run the pilot; scale the predeclared arms; evaluate sealed transfer tasks once. Stop scaling on simulator regressions, hidden-state leakage, broken log probabilities, persistent truncation, or a measured absence of learning. Preserve every failed run. The final research artifact is a reproducible comparison of trajectories and outcomes, whether or not the thesis succeeds.
