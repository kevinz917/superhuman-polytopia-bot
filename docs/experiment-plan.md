# Earlier experiment proposal — 2026-10-16

**Superseded where conflicting by [post-training-plan.md](post-training-plan.md).** The simulator is now the frozen research ruleset `bardur-1v1-v1`; chasing commercial versions and a mandatory initial critic are no longer requirements. This document is retained as proposal history.

The [first SFT checkpoint plan](sft-checkpoint-plan.md) is the detailed protocol through a validated Qwen3-4B adapter. Its native-rules gate, data split, training recipe, and acceptance criteria govern that initial checkpoint; later RL and transfer sections here remain proposals.

## Decision

Build a GPU-native JAX simulator first, using pinned Polyfish source as the rule oracle. The user selected this implementation order on 2026-10-16; it supersedes the CPU-first proposal. Once the simulator passes its complete-game gates, use a Qwen3-4B-Instruct-2507 language policy, action-only supervised demonstrations, and direct on-policy LM reinforcement learning. The central scientific hypothesis, clarified by the user on 2026-10-16, is that SFT/RL on Polytopia improves a Qwen 4B model’s ability to generate successful expert trajectories on other tasks. Held-out Polytopia strength is an intermediate learning check; cross-task transfer is the primary research outcome. This is a hypothesis to test, not an established result.

The engine choice is provisional until it passes fidelity checks. A complete faithful engine is the eventual target; the first release must faithfully implement a declared game version and every mechanic reachable in the chosen scenario. Restricting maps and tribes reduces testing scope without changing their rules. Mechanics must never silently become no-ops.

The user requested a project in /projects. On this machine the existing projects directory is `/Users/K.Z./Desktop/projects`; this project lives alongside Duper there.

## Research objective update

The comprehensive post-training protocol will be finalized after the complete GPU simulator passes its acceptance gates. It must evaluate held-out non-Polytopia tasks with verifiable trajectory outcomes, comparing the unchanged starting checkpoint, Polytopia SFT, direct Polytopia RL, and SFT followed by RL. Include compute/token-matched non-game training controls and separate single-agent transfer from multi-agent interaction effects. Fix downstream inference budgets and prompts, use multiple training seeds, and report regressions as well as gains. Define “expert trajectory” operationally (task success and quality under the task's verifier), rather than relying on the model's prose or self-rating.

The supplied research discussion points to SPIRAL and MARSHAL as methodological references. Its numerical claims require verification against the original papers before inclusion in the final protocol. A more complex game does not itself establish stronger transfer.

## 1. Environment and game setting

Candidate: [Polyfish](https://github.com/HenBOMB/Polyfish), pinned in `configs/research-sources.json`. Its Rust library supplies state, legal moves, map generation, replay code, and heuristic agents. These are useful starting points, not independent evidence of complete fidelity. See `engine-findings.md` for observed limitations and alternatives.

Main task: **two players, Bardur versus Bardur, 14×14 Drylands, fog of war enabled, native elimination victory for the pinned ruleset**. Confirm its exact victory rule against an authoritative game before accepting results. Use ordinary generated maps and swap policy seats on identical initial states. Symmetric maps are optional debugging cases; do not substitute them for the main map distribution.

Use 11×11 for integration tests and early curriculum games. Train on 14×14 as soon as full games are reliable. Keep regular economy, research, combat, capture, upgrades, city choices, exploration, and all other reachable mechanics. Later evaluate 16×16, additional regular tribes, and Lakes; then expand naval and special-tribe coverage.

Use a research safety limit of 100 full rounds or 10,000 atomic actions, whichever comes first. A full round means both players have taken their turn. Limit hits are **truncations**, never score-based wins. Report them separately; use value bootstrapping for training truncations if the critic is validated. Target less than 5% truncation before scaling. Native game terminals remain terminals.

Implement the game state, transitions, legality, observations, reset and RNG as batched device arrays and compiled JAX operations. Use a separate pinned Rust executable only for differential tests and fixture generation. The GPU runtime must not call a CPU engine or silently fall back for unported rules. A JSON Lines process boundary may later serve debugging and LM integration, but is not the simulation architecture. Initial GPU experiments run in Modal staging. See `gpu-simulation.md` for the current rule coverage and remaining milestones.

**Fidelity gate:** pin Polyfish and its rules version as the implementation source of truth. Compare at least 100 complete seeded source-engine games plus targeted rule fixtures, field by field after every action, before enabling training. The present tests cover a projected state and a small rule slice, not this full gate. Check movement, damage, captures, income, research, population/rewards, visibility, complete legality and native terminal outcomes. Cross-reference other engines or authoritative replays for uncertain source behavior; separately label official-game fidelity until independently validated. Compare imported boards independently of map-generation distribution and random-seed equivalence.

**Performance gate:** run synchronized GPU benchmarks of reset, complete steps including legality, observations, snapshots and full games at multiple batch sizes. Record compiler/warm-up cost separately from steady-state device execution and host transfers. The initial Modal L4 experiment measures only the implemented rule slice. Require no runtime host callback for game logic, deterministic per-game state/RNG, and no silent action/entity-capacity overflow. Choose throughput targets after complete-game measurements; do not infer them from combat or EndTurn microbenchmarks.

## 2. Model and observation interface

Use [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). Its model card identifies a 4B non-thinking model with an Apache-2.0 license and Transformers/vLLM support. This is a reproducible starting choice, not a claim that it is the best small model in 2026. Pin the model and tokenizer revisions during implementation.

Start with a frozen BF16 base and rank-32 LoRA adapters on attention and MLP linear layers. Run action-only first. A later controlled experiment can train a short explicit rationale, capped at 128 tokens, against action-only at both equal token budget and equal game count. Do not invent expert explanations for search-generated moves.

Inputs: rules summary, current player/turn/resources/tech, visible grid, own entities, observed enemy entities, last-seen facts marked with their observation turn, recent public events, and the complete legal-action catalog. Unknown terrain and hidden enemies must remain unknown. Build memory from observation history, never the underlying omniscient state.

Target 2–4K input tokens, allow 8K context initially, and cap action-only output at 32 tokens. These token lengths must be measured on real games. Do not silently drop legal actions or state fields when over budget; first compress the serialization, then explicitly increase the budget or introduce a measured hierarchical interface.

Output one response-local action ID, for example `{"action_id":"a17"}`. The prompt includes each ID's semantic command. Execute one atomic action, regenerate the observation and legal set, and ask again. End-turn and mandatory city-reward choices are explicit actions. Store the semantic command alongside the temporary ID so demonstrations survive indexing changes.

Initially train ordinary sampled completions and validate them, without dynamic legal-token masking. Invalid output is a recorded agent error: allow up to two observations describing the error; after three failures forfeit that episode. Every failed generation and log probability stays in the trajectory. Avoid a silent heuristic fallback. At evaluation, report both the normal policy and an optional separately labeled constrained-output variant. If constrained decoding is introduced into RL, the learner must reproduce the same token constraints and probability normalization.

## 3. Demonstrations and supervised warm-start

Do not train a separate neural self-play teacher first. Build the following demonstration sources:

1. A transparent economy/combat heuristic: prioritize productive city development, legal captures, survival, favorable combat, and sensible turn completion. Establish its actual win rate before calling it competent.
2. A stronger search teacher: start from upstream greedy/heuristic MCTS after auditing its information access and action pruning. Compare 128 and 512 simulations per decision on validation maps; retain the stronger setting per unit of CPU time. Under fog of war, search must use only the player's observation and a reproducible belief model. Hidden state must not enter labels implicitly through a teacher that sees the true board.
3. Human or stronger-bot replay demonstrations if reconstructible and version-compatible. Treat these as optional enrichment; availability and parsers are not established.

First collect 5,000–10,000 examples to verify interface learning. Then target **50,000 deduplicated decisions**, scaling to 200,000 only after a positive pilot. Balance opening economy, exploration, combat, recovery, capture, and endgame; cap repeated end-turn examples. At 2,048 prompt plus 32 completion tokens, 50K examples are about 104M processed tokens per epoch, although only completion tokens receive the SFT loss.

Replay every labeled action for legality. Store teacher identity, search settings, observation history, map ID, game ID, engine/rules version, semantic command, and final outcome. Include good decisions from losing games; avoid treating the winner's every move as optimal. Split by complete game and map family before sampling decisions. Do not randomly split adjacent positions.

Train using TRL/PEFT SFT, response-only loss, LoRA rank 32 / alpha 64 / dropout 0, learning rate 1e-4, effective batch 64, one epoch, max two if validation improves. These are initial tunable settings. Select the checkpoint on validation full-game wins, with action validity as a separate diagnostic. [TRL documents completion/assistant-only loss and PEFT integration](https://huggingface.co/docs/trl/sft_trainer).

Acceptance: at least 99% valid first responses on held-out positions, at least 95% completed games, and competitive performance against the simple heuristic. Demonstrate a meaningful improvement over the unmodified instruction model. If SFT cannot play coherent games, debug representation/teacher quality before adding RL.

## 4. Direct language-model reinforcement learning

Use **verl + vLLM**, with a custom game rollout coordinator and trajectory-to-training adapter. The framework documents [LoRA PPO support](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html) and an extensible [agent loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html). Polytopia integration, independent decision contexts, historical opponents, and cross-decision credit assignment still require implementation; this is not an out-of-the-box configuration. RAGEN and SPIRAL are reference implementations to consult rather than additional framework layers to combine.

Start with PPO and a separately optimized value estimator using the same observation boundary. A practical first critic is another frozen 4B base with its own LoRA adapter and scalar value head; budget this explicitly and place it with the learner. Verify backend support in a tiny integration run. If shared inference/training memory is tight, alternate phases or shard/offload learner components; do not assume all copies fit on one 24GB device.

Begin with 100–500 games against fixed heuristics to verify learning. For the main run sample opponents: 40% fixed bots, 40% historical LM checkpoints, 20% a frozen copy of the current policy. Freeze all opponent identities throughout each collection batch. Sample both seats evenly. Initially update only the designated learner's trace; opponent inference is still part of compute cost. Later training both players' on-policy traces is an efficiency experiment, with seat-specific rewards and separate observations.

Collect 64 complete games per update initially, with a cap on total tokens to prevent OOM. Freeze the learner policy within each rollout batch, record its version and token log probabilities, then update and synchronize adapters before the next collection. Start with LoRA learning rate 1e-5, PPO clip 0.2, one optimization epoch, gamma 1.0 across game decisions, and a small KL penalty to the frozen SFT policy (initial coefficient 0.01; monitor and tune). Fit the critic to outcomes and test value calibration. Initially use Monte Carlo return minus the value baseline, avoiding arbitrarily discounting long games. Introduce GAE only as a measured variance/bias tradeoff.

Reward: +1 native win, -1 native loss, 0 genuine draw; no default rewards for stars, territory, or plausible prose. Use shorter maps and tactical starting-position curricula if terminal rewards are too sparse. Any shaped-reward experiment gets a separate label and is still evaluated on real game outcomes.

Each decision is a separate LM sample with a bounded observation context. Preserve episode ID and decision index, compute returns across the whole episode, then attach the decision advantage to the generated response tokens. Mask observation/prompt tokens out of the policy loss. Do not concatenate a full game into an unbounded context, or treat a rollout chunk as a terminal game. Log per-game and per-decision weighting: start with per-game normalization so long games do not dominate solely through their length.

Pilot budget: 1,000 games. First substantive run: 10,000 games; stretch budget: 100,000 only after held-out improvement. Use two training seeds for the pilot and three for the main comparison if compute permits. Save frozen opponents and evaluation checkpoints at regular game-count intervals. Self-play reward averaging near zero is not a progress metric.

## 5. Evaluation and success criteria

Freeze train/validation/test seed manifests, initial-state hashes, engine revision, and ruleset before model selection. Hold out at least 200 validation maps and 500 final test maps. Run both seat assignments per map. Resample or reserve map families to prevent mirrored or equivalent board leakage.

Compare identical interfaces and inference budgets for: random legal policy, simple heuristic, search teacher at a fixed search budget, original instruction LM, SFT LM, and SFT+RL LM. Include historical-checkpoint cross-play and a withheld bot strategy. Search baselines should report CPU time separately; they do not have the same compute budget as a single LM decision.

Use 100 validation maps × 2 seats for frequent screening. Final comparisons use 500 untouched maps × 2 seats = 1,000 games per opponent pairing, for each training seed where affordable. Report win/draw/loss, truncations, first-response validity, game completion, turns, input/output tokens, and latency. Count timeouts and errors explicitly; never remove unsuccessful games from the denominator.

Primary statistic: paired-map win-score improvement (win=1, draw=0.5, loss=0), with confidence intervals obtained by bootstrapping maps as clusters; include training-seed variability separately. For truncated games report conservative score 0 for the evaluated candidate plus the unresolved fraction, and a sensitivity analysis excluding or completing them. Avoid treating the two swapped-seat games as independent samples. A rough independent-sample 1,000-game interval at 50% is ±3 percentage points; map correlation and multiple comparisons can widen uncertainty.

Success target: at least +10 percentage points over SFT against the preregistered primary fixed opponent, confidence interval excluding zero, improvement in direct RL-versus-SFT play, no major loss across the other fixed opponents, and under 5% truncations. Report all results, including failed seeds. Internal Elo is supplementary, not evidence of top-human strength.

Transfer checks: 16×16 Drylands; held-out regular tribe matchups; subsequently Lakes after naval parity passes. A credible real-Polytopia claim requires legal observations and actions replayed successfully in the authoritative game plus a separate match evaluation there. Simulator performance alone is labeled as such.

## 6. Compute and budget

All figures below are planning assumptions, not measured training requirements or current cluster availability.

| Stage | Initial allocation | Scope |
| --- | --- | --- |
| GPU engine experiments | 1×L4 in Modal staging, 4 CPU cores, 8GB host RAM | Bounded correctness runs and device microbenchmarks; scale after full-rule coverage |
| Reference and corpus generation | 16–32 CPU cores, 64GB RAM, 100GB free SSD | Differential replay checks and heuristic/search labels; eventual GPU-native search |
| SFT smoke test | 1×24GB GPU with QLoRA if necessary, short contexts | Interface learning and memory profiling |
| Main SFT | 1×48–80GB GPU, or 2×24GB with supported sharding | BF16 LoRA, 50K examples, measured token throughput |
| RL pilot, preferred | 4×80GB GPUs, 32 CPU cores, 128GB host RAM | Initially 2 rollout GPUs, 2 learner/critic GPUs; rotate or colocate frozen references as profiled |
| RL using consumer GPUs | 8×24GB GPUs, preferably on one host | Initially 4 rollout replicas and 4 sharded learner/critic GPUs; smaller batches/context and possible offload |
| Storage allowance | 200–500GB | Base model, adapters, optimizer states, compressed observations and traces |

Eight 4090s are an alternative to benchmark if available, not equivalent to one 192GB GPU. PCIe and cross-host communication can dominate learner sharding; keep the learner on one host when possible and place independent inference replicas remotely. Do not assume availability from a historical cluster snapshot.

Four billion BF16 weights occupy approximately 8GB decimal before runtime overhead. Full Adam training can require roughly 64GB for 16 bytes/parameter before activations and additional RL model copies, so full-parameter RL is not the first experiment. LoRA reduces trainable state but does not remove forward/backward activation costs or rollout KV caches. The suggested hardware remains subject to a measured memory test with real prompts.

Example workload: 10,000 games × 600 LM decisions per game across both sides = 6M calls. At 2,048 input tokens and 32 output tokens, this is approximately **12.3B prompt tokens and 192M generated tokens**. Static rules prefixes may cache; changing board states still cost prefill. Teacher search and SFT preprocessing are additional work.

| Measured aggregate decisions/s | Rollout hours for 10K games | Total hours with illustrative 2× overhead | GPU-hours if all 8 GPUs allocated throughout |
| --- | ---: | ---: | ---: |
| 10 | 166.7 | 333.3 | 2,666.7 |
| 50 | 33.3 | 66.7 | 533.3 |
| 200 | 8.3 | 16.7 | 133.3 |

These are sensitivity cases, **not predictions of 4090/H100 throughput**. For 1K games divide by ten; for 100K multiply by ten. Multiple seeds and evaluation games add linearly. Dollar cost is measured GPU-hours multiplied by the actual chosen provider rate; no rental rate is assumed. Owned hardware still consumes allocation time and power.

The first LM-training GPU allocation should be a 4-hour profiling session, followed by an 8–24-hour pilot ceiling. Measure completed decisions/s, not decode tokens/s alone, using actual 14×14 observations at 32/64/128 concurrent games. Record prefill/decode time, engine idle time, PPO update time, sync time, memory peaks, and opponent inference. Replace the calculator's example values with those measurements before choosing the main run's budget.

## 7. Ordered milestones

1. **GPU simulator.** Port the complete reachable rules into JAX, validate against the pinned source, implement GPU reset/RNG and run full-game GPU benchmarks. The initial core exists, but combat execution, economy/capture and other rules remain. This is the first deliverable; the old 3–5 day CPU-adapter estimate does not apply.
2. **Corpus and evaluation harness, approximately 3–5 days.** Establish bot opponents, split maps, generate 5–10K examples, run original LM, and measure real prompt sizes. Upgrade to 50K demonstrations after teacher validation.
3. **SFT and integration, approximately 2–4 days plus measured GPU time.** Train adapters, verify held-out gameplay, measure memory, and round-trip learner adapters into the rollout server.
4. **Direct RL pilot, approximately 3–7 days of engineering plus pilot compute.** Verify trajectory/log-prob accounting, terminal credit assignment, opponent freezing, and checkpoint reproducibility. Compare 1K-game runs to SFT before scaling.
5. **Main experiment.** Execute the measured 10K-game budget and preregistered evaluation; then decide whether more games, better demonstrations, or wider task coverage is justified.

The earlier 2–4 week end-to-end estimate assumed a CPU adapter. It does not cover the newly selected GPU engine port. Re-estimate implementation and training budgets after complete-game GPU parity and throughput measurements.
