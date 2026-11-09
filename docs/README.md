# Docs and research notebook

Dated files are lab-notebook entries, written when each experiment ran. They keep their original wording, numbers and
internal run names (`c3`, `OMNI`, `u3142`, …), and may cite scripts, Modal volume paths or run ledgers that are not
part of this repository. Start with the champion's three reports, then read backwards as far as you like.

## The champion

- [scratch-check-2026-10-22.md](scratch-check-2026-10-22.md): pure self-play from random weights. A 20-minute check, then the 7-hour overnight run that passed every earlier bot.
- [scratch-continuations-2026-10-23.md](scratch-continuations-2026-10-23.md): two continuations, the champion (+442 Elo over c3), final tournament, seat and map effects.
- [scratch-search-2026-10-23.md](scratch-search-2026-10-23.md): turn search on the champion (+50 Elo, 81% vs the previous best with search) and the fog-of-war value head.
- [pilots-2026-10-23.md](pilots-2026-10-23.md): follow-ups: low-learning-rate polish, larger batches, a 4×128 network.
- [champion-vs-real-crazy-2026-10-24.md](champion-vs-real-crazy-2026-10-24.md): the champion against the game's own Crazy AI, 48–0 with search and 24–0 raw.
- [scratch-cont-2-2026-10-23.md](scratch-cont-2-2026-10-23.md): replays, seat and map effects, and the run that became the champion.

## After the champion: the game's rules, hero runs and four players

- [rules-alignment-2026-10-27.md](rules-alignment-2026-10-27.md): one ruleset, the game's own (2.17.3); every rule that changed and the parity before and after.
- [hero-runs-plan-2026-10-26.md](hero-runs-plan-2026-10-26.md), [hero-throughput-2026-10-28.md](hero-throughput-2026-10-28.md): the from-scratch 4×128 runs on the game's rules (hero-1v1 is in `weights/hero-1v1/`) and the throughput fixes behind them.
- [four-player-2026-10-24.md](four-player-2026-10-24.md): four-seat 18×18 free-for-all: the N-seat engine, rank rewards, the trainer and the big runs.
- [rules-parity-2-summary.md](rules-parity-2-summary.md): the second rules pass: unit facing, disembark, four-seat diplomacy, city connections.
- [hero-ffa4-levers-2026-10-28.md](hero-ffa4-levers-2026-10-28.md): where the four-player run's samples per second go.
- [hero-daytona-2026-10-28.md](hero-daytona-2026-10-28.md): moving the hero runs to Daytona.
- [verda-big-run-2026-10-23.md](verda-big-run-2026-10-23.md), [verda-big-run-2026-10-24-bn-main1.md](verda-big-run-2026-10-24-bn-main1.md), [verda-long-run-plan-2026-10-24.md](verda-long-run-plan-2026-10-24.md): the 4×128 bn-main runs on Verda.

## From scratch on the game's rules: trunk and reward experiments

Checkpoints from these runs are in [`weights/`](../weights).

- [curio-hero-2026-10-31.md](curio-hero-2026-10-31.md), [platforms.md](platforms.md): bighero, 26 hours from random weights on a self-hosted GPU cluster, and how launches differ by platform.
- [obs-v3-history-2026-11-01.md](obs-v3-history-2026-11-01.md): observation version 3, with leak-free opponent history.
- [lookout-2026-11-01.md](lookout-2026-11-01.md): a hybrid attention trunk; it trailed the pooled conv trunk at equal samples.
- [capfix-arm-b-brief-2026-11-02.md](capfix-arm-b-brief-2026-11-02.md), [farsight-v2-arm-b-analysis-2026-11-03.md](farsight-v2-arm-b-analysis-2026-11-03.md) and the [run archive](run-archive/farsight-v2-capfix-2026-11-03): farsight-v2, score-at-cap and frozen developer opponents, and why the pooled trunk did not beat hero-1v1.
- [isoflop-sweep-plan-2026-10-28.md](isoflop-sweep-plan-2026-10-28.md), [chinchilla-sweep-plan-2026-10-28.md](chinchilla-sweep-plan-2026-10-28.md): network-size sweeps, planned but not run.

## Plato: a transformer from scratch, then peace

The weights are in [`weights/plato/`](../weights/plato).

- [plato-2026-11-09.md](plato-2026-11-09.md): the 3.2M-parameter hybrid transformer trained from scratch in four stages ($1,498, 25.6 B samples), the stage-4 peace curriculum with collusion monitors and the cap = loss rule, every preregistered check, the learning curve and the negatives. Data: [benchmarks/2026-11-09-plato.json](benchmarks/2026-11-09-plato.json).
- [plato-strategy-2026-11-09.md](plato-strategy-2026-11-09.md): how Plato plays compared with its $300 checkpoint on the same maps: tech order, the Knight answer to hero-1v1's rush, income by round, Explorer vs Workshop, Lumber Huts, diplomacy and how games end. Data: [benchmarks/2026-11-09-plato-strategy.json](benchmarks/2026-11-09-plato-strategy.json).

## Humans against the bot

- [human-game-a8147e94-2026-10-21.md](human-game-a8147e94-2026-10-21.md), [human-game-friend-birthday-dc7ac27f.md](human-game-friend-birthday-dc7ac27f.md): two games on the public site, move by move.
- [site-games-analysis-2026-10-31.md](site-games-analysis-2026-10-31.md): 65 site games; humans win with Riding and knights against the bot's fixed archer opening.

## Rules, engine and Studio

- [ruleset-v1.md](ruleset-v1.md), [ruleset-v2.md](ruleset-v2.md): the retired research rules (v2 adds the 18×18 board) the champion was trained on.
- [native-rules-exceptions.md](native-rules-exceptions.md), [engine-findings.md](engine-findings.md): corrected rules and fidelity questions against the reference.
- [gpu-simulation.md](gpu-simulation.md), [map-generation.md](map-generation.md), [map-variants-2026-10-19.md](map-variants-2026-10-19.md): the JAX engine, its validation, map generators.
- [environment-contract.md](environment-contract.md): the observation and action boundary policies use.
- [rust-engine-2026-10-21.md](rust-engine-2026-10-21.md), [rust-engine-10x-2026-10-22.md](rust-engine-10x-2026-10-22.md): the Rust engine, first as a bit-exact port of the JAX engine.
- [replay-studio.md](replay-studio.md): the Studio, its agent API, and the README clip recorder.

## Earlier lineage: imitation bootstrap to OMNI c3

- [structured-rl.md](structured-rl.md), [structured-rl-current.md](structured-rl-current.md), [structured-rl-smoke-2026-10-17.md](structured-rl-smoke-2026-10-17.md), [structured-rl-league.md](structured-rl-league.md), [structured-rl-league-result-2026-10-17.md](structured-rl-league-result-2026-10-17.md), [structured-rl-postmortem-2026-10-17.md](structured-rl-postmortem-2026-10-17.md), [structured-rl-repair-plan.md](structured-rl-repair-plan.md), [structured-rl-repair-progress.md](structured-rl-repair-progress.md): the first structured-state PPO bot, bootstrapped from a scripted teacher.
- [structured-rl-agent-handoff-2026-10-18.md](structured-rl-agent-handoff-2026-10-18.md): the state of that campaign on Oct 18.
- [unit-diversity-investigation-2026-10-18.md](unit-diversity-investigation-2026-10-18.md), [cloud-comparisons-2026-10-18.md](cloud-comparisons-2026-10-18.md), [gpu-benchmark-2026-10-18.md](gpu-benchmark-2026-10-18.md), [simulator-throughput-diagnosis-2026-10-18.md](simulator-throughput-diagnosis-2026-10-18.md)
- [distributed-rl-2026-10-19.md](distributed-rl-2026-10-19.md): the move to CPU actors and a GPU learner.
- [campaign-log-2026-10-19-20.md](campaign-log-2026-10-19-20.md), [diverse-strategy-check-2026-10-19.md](diverse-strategy-check-2026-10-19.md), [policy-probes-2026-10-19.md](policy-probes-2026-10-19.md), [simulator-profile-2026-10-19.md](simulator-profile-2026-10-19.md), [next-run-and-strength-2026-10-20.md](next-run-and-strength-2026-10-20.md), [winner-speed-bonus-handoff-2026-10-20.md](winner-speed-bonus-handoff-2026-10-20.md)
- [c5-omni-long-2026-10-21.md](c5-omni-long-2026-10-21.md): the omniscient critic, the biggest single lever (OMNI c3).
- [c5-upgo-2026-10-21.md](c5-upgo-2026-10-21.md), [c6-credit-readout-2026-10-21.md](c6-credit-readout-2026-10-21.md), [credit-assignment-review-2026-10-21.md](credit-assignment-review-2026-10-21.md), [long-horizon-credit-2026-10-20.md](long-horizon-credit-2026-10-20.md): UPGO and credit assignment, including the turn clock.
- [rl-18x18-2026-10-19.md](rl-18x18-2026-10-19.md), [rl-18x18-bottlenecks-2026-10-20.md](rl-18x18-bottlenecks-2026-10-20.md), [c5-18x18-omni-2026-10-21.md](c5-18x18-omni-2026-10-21.md): 18×18 boards.

## Test-time search

- [test-time-search-2026-10-19.md](test-time-search-2026-10-19.md), [test-time-search-c3-2026-10-19.md](test-time-search-c3-2026-10-19.md), [search-speed-2026-10-19.md](search-speed-2026-10-19.md): one-turn lookahead, the information leak, and its leak-free determinized form.
- [omni-search-eval-2026-10-20.md](omni-search-eval-2026-10-20.md), [search-depth-2026-10-21.md](search-depth-2026-10-21.md), [search-breadth-2026-10-21.md](search-breadth-2026-10-21.md), [search-sweep-findings-2026-10-21.md](search-sweep-findings-2026-10-21.md): depth and breadth saturate at one turn.
- [search-distillation-2026-10-19.md](search-distillation-2026-10-19.md), [search-distillation-refit-2026-10-21.md](search-distillation-refit-2026-10-21.md): distilling search into the policy did not help.

## Training throughput

- [throughput-research-2026-10-21.md](throughput-research-2026-10-21.md), [10x-training-plan-2026-10-21.md](10x-training-plan-2026-10-21.md): where samples per second go.
- [stage-a-2026-10-21.md](stage-a-2026-10-21.md), [e2e-smoke-2026-10-21.md](e2e-smoke-2026-10-21.md), [a6-batch-assembly-2026-10-22.md](a6-batch-assembly-2026-10-22.md), [levers-egress-2026-10-22.md](levers-egress-2026-10-22.md), [overnight-bottlenecks-2026-10-22.md](overnight-bottlenecks-2026-10-22.md): the fixes, measured.
- [training-speed-proposal-2026-10-17.md](training-speed-proposal-2026-10-17.md), [training-speed-result-2026-10-17.md](training-speed-result-2026-10-17.md), [speed-runtime-v2.md](speed-runtime-v2.md), [speed-comparison-cpu-vs-gpu-2026-10-19.md](speed-comparison-cpu-vs-gpu-2026-10-19.md): earlier speed work, from one GPU to CPU actors.
- [rl-sampling-bottlenecks-2026-10-21.md](rl-sampling-bottlenecks-2026-10-21.md): what bounds the sampling rate of an actor-bound run.
- [pufferlib-evaluation-2026-10-20.md](pufferlib-evaluation-2026-10-20.md): why we did not switch frameworks.
- [alphastar-review-2026-10-21.md](alphastar-review-2026-10-21.md): AlphaStar's ablations read against this program.
- [katago-review-2026-10-23.md](katago-review-2026-10-23.md): KataGo read against this program.

## The language-model line

Before the self-play bots, the project trained 4B language models to play from text. These notes cover it.

- [experiment-plan.md](experiment-plan.md), [llm-game-transfer-research-2026-10-18.md](llm-game-transfer-research-2026-10-18.md), [post-training-plan.md](post-training-plan.md), [transfer-experiment-design-2026-10-18.md](transfer-experiment-design-2026-10-18.md): the plan and the literature behind it.
- [sft-pipeline.md](sft-pipeline.md), [sft-checkpoint-plan.md](sft-checkpoint-plan.md), [sft-checkpoint-result.md](sft-checkpoint-result.md), [sft-execution.md](sft-execution.md), [sft-efficiency-audit-2026-10-16.md](sft-efficiency-audit-2026-10-16.md), [sft-efficiency-implementation.md](sft-efficiency-implementation.md), [sft-pipeline-config-2026-10-19.md](sft-pipeline-config-2026-10-19.md), [sft-eval-speedup-2026-10-19.md](sft-eval-speedup-2026-10-19.md), [sft-trainer-benchmark-2026-10-19.md](sft-trainer-benchmark-2026-10-19.md): the supervised fine-tuning stack.
- [movement-diagnostic-2026-10-17.md](movement-diagnostic-2026-10-17.md), [movement-epoch2-result-2026-10-17.md](movement-epoch2-result-2026-10-17.md), [movement-mixed-result-2026-10-17.md](movement-mixed-result-2026-10-17.md), [movement-overfit-result-2026-10-17.md](movement-overfit-result-2026-10-17.md), [movement-scale40k-10k-investigation-2026-10-17.md](movement-scale40k-10k-investigation-2026-10-17.md), [movement-scale40k-result-2026-10-17.md](movement-scale40k-result-2026-10-17.md): why the models moved units badly, and what helped.
- [qwen35-thinking-gameplay-result-2026-10-17.md](qwen35-thinking-gameplay-result-2026-10-17.md), [qwen35-unrestricted-gameplay-2026-10-17.md](qwen35-unrestricted-gameplay-2026-10-17.md), [qwen35-unrestricted-gameplay-progress-2026-10-17.md](qwen35-unrestricted-gameplay-progress-2026-10-17.md), [qwen35-thinking-debug-2026-10-18.md](qwen35-thinking-debug-2026-10-18.md), [qwen35-independent-diagnosis-2026-10-18.md](qwen35-independent-diagnosis-2026-10-18.md), [qwen35-thinking-comparison-final-2026-10-18.md](qwen35-thinking-comparison-final-2026-10-18.md), [qwen35-thinking-handoff-2026-10-18.md](qwen35-thinking-handoff-2026-10-18.md), [qwen35-current-benchmark.md](qwen35-current-benchmark.md), [qwen3-instruct-baseline-2026-10-18.md](qwen3-instruct-baseline-2026-10-18.md): Qwen3.5 thinking against non-thinking play.
- [compact-prompt-v2-2026-10-19.md](compact-prompt-v2-2026-10-19.md): a compact full-state prompt.
- [expert-sft-research-report-2026-10-19.md](expert-sft-research-report-2026-10-19.md), [expert-sft-result-2026-10-19.md](expert-sft-result-2026-10-19.md), [expert-sft-scaling-curve-2026-10-19.md](expert-sft-scaling-curve-2026-10-19.md), [handoff-sft-2026-10-19.md](handoff-sft-2026-10-19.md): imitating the RL bot makes a 4B model win.

## Writing

- [blog-post-polytopia.md](blog-post-polytopia.md): the blog post draft.
- [polytopia-bench-post.md](polytopia-bench-post.md): the Polytopia Bench announcement draft.
- [branch-fold-2026-11-04.md](branch-fold-2026-11-04.md): how the experiment branches were folded together before this release.
