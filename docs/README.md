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

## After the champion: the game's rules, hero runs and four players

- [rules-alignment-2026-10-27.md](rules-alignment-2026-10-27.md): one ruleset, the game's own (2.17.3); every rule that changed and the parity before and after.
- [hero-runs-plan-2026-10-26.md](hero-runs-plan-2026-10-26.md), [hero-throughput-2026-10-28.md](hero-throughput-2026-10-28.md): the from-scratch 4×128 runs on the game's rules (hero-1v1 is in `weights/hero-1v1/`) and the throughput fixes behind them.
- [four-player-2026-10-24.md](four-player-2026-10-24.md): four-seat 18×18 free-for-all: the N-seat engine, rank rewards, the trainer and the big runs.

## Rules, engine and Studio

- [ruleset-v1.md](ruleset-v1.md), [ruleset-v2.md](ruleset-v2.md): the retired research rules (v2 adds the 18×18 board) the champion was trained on.
- [native-rules-exceptions.md](native-rules-exceptions.md), [engine-findings.md](engine-findings.md): corrected rules and fidelity questions against the reference.
- [gpu-simulation.md](gpu-simulation.md), [map-generation.md](map-generation.md), [map-variants-2026-10-19.md](map-variants-2026-10-19.md): the JAX engine, its validation, map generators.
- [environment-contract.md](environment-contract.md): the observation and action boundary policies use.
- [rust-engine-2026-10-21.md](rust-engine-2026-10-21.md), [rust-engine-10x-2026-10-22.md](rust-engine-10x-2026-10-22.md): the Rust engine, first as a bit-exact port of the JAX engine.
- [replay-studio.md](replay-studio.md): the Studio, its agent API, and the README clip recorder.

## Earlier lineage: imitation bootstrap to OMNI c3

- [structured-rl.md](structured-rl.md), [structured-rl-current.md](structured-rl-current.md), [structured-rl-smoke-2026-10-17.md](structured-rl-smoke-2026-10-17.md), [structured-rl-league.md](structured-rl-league.md), [structured-rl-league-result-2026-10-17.md](structured-rl-league-result-2026-10-17.md), [structured-rl-postmortem-2026-10-17.md](structured-rl-postmortem-2026-10-17.md), [structured-rl-repair-plan.md](structured-rl-repair-plan.md), [structured-rl-repair-progress.md](structured-rl-repair-progress.md): the first structured-state PPO bot, bootstrapped from a scripted teacher.
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
- [rl-sampling-bottlenecks-2026-10-21.md](rl-sampling-bottlenecks-2026-10-21.md): what bounds the sampling rate of an actor-bound run.
- [pufferlib-evaluation-2026-10-20.md](pufferlib-evaluation-2026-10-20.md): why we did not switch frameworks.
- [alphastar-review-2026-10-21.md](alphastar-review-2026-10-21.md): AlphaStar's ablations read against this program.
