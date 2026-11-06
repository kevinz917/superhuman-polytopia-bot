# The main benchmark: continuous bot building

The showcase is a single benchmark: given 40 minutes, the public SDK and practice facilities, how strong a Polytopia bot can a coding agent produce?

The agent receives one uninterrupted session. It chooses its own experiment schedule and may compile, run practice matches on nine public maps against public-greedy-v1, inspect public traces, write tests and revise. All those operations and inference latency count toward the same active budget. Initial setup, final grading and explicit CPU queue waits do not. There are no forced feedback stages, resets, automatic repairs or separate one-shot/control leaderboards.

At submission or cutoff, the trusted controller stops coder processes and atomically captures current source. A separate offline verifier rebuilds it using the original locked Cargo project, scores every declared held-out matchup and replay-audits played games. Broken current source is retained as scheduled compilation losses; infrastructure failures remain unresolved errors. The final score is mean fixed-panel points times 100. Champion performance is a separate breakdown. One attempt is an entire authored bot, not one model response or one evaluation game.

Standard held-out evaluation uses six fixed references, ten maps per family across drylands/lakes/continents, and both seats: 360 matches. The integration-demo profile uses one map/family (36 matches) with the same 40-minute authoring allowance. One independently authored bot per model is an integration result, not a dependable ranking. Full experiments repeat independently authored attempts and retain all outcomes.

The existing 200-round, 40,000-action, four-trillion-fuel, 256-MiB Wasm, two-second decision contract is retained. Public practice uses the same runtime, and additionally respects the coding deadline. Practice opponents and final references are separate. No checkpoint, held-out seeds or final grading feedback enter the authoring container.

The benchmark is packaged in native Harbor task format and runs on the local Colima/Docker VM through a host external-agent adapter using the existing CLI proxy. Read [the Harbor operator guide](../../harbor/README.md) and [the full protocol](../../skirmish/continuous/PROTOCOL.md). The v6 staged conditions remain archived research experiments; their completed results are not results of this new protocol.
