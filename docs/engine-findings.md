# Engine inspection

Updated 2026-10-16. The pinned library and validation oracle now build locally. An initial JAX rule slice has projected parity tests against this source. Full-game and official-game fidelity remain unverified; see `gpu-simulation.md`.

## Primary candidate: Polyfish Rust

Pinned revision: `a9f90ab54607f5633556d02422d7457f548cc58f`.

- [`game.rs`](https://github.com/HenBOMB/Polyfish/blob/a9f90ab54607f5633556d02422d7457f548cc58f/polyfish-rs/src/game.rs) exposes JSON import/export, direct cloning, legal moves, move application, current player, and game termination. Its search clone obscures fog; a comment describes additional handling needed for opponent vision. Do not assume this is a safe public observation serializer.
- [`mapgen.rs`](https://github.com/HenBOMB/Polyfish/blob/a9f90ab54607f5633556d02422d7457f548cc58f/polyfish-rs/src/mapgen.rs) accepts size, map type, tribe list, seed, version, and symmetry. Its inspected default version is 115; this is not established as the current official rules version. It uses its local `DotNetRandom` implementation; identical official game seeds are not presumed to produce identical maps.
- [`types.rs`](https://github.com/HenBOMB/Polyfish/blob/a9f90ab54607f5633556d02422d7457f548cc58f/polyfish-rs/src/types.rs) maps Tiny to 11×11 and Small to 14×14, and includes Drylands.
- [`heuristic_mcts.rs`](https://github.com/HenBOMB/Polyfish/blob/a9f90ab54607f5633556d02422d7457f548cc58f/polyfish-rs/src/ai/heuristic_mcts.rs) exposes random, greedy, and search agents. Its root search removes EndTurn when other moves exist. Audit this before using it as a teacher: a legal action is not necessarily worth taking. The caller also needs a checked observation/belief boundary before search.
- [`Cargo.toml`](https://github.com/HenBOMB/Polyfish/blob/a9f90ab54607f5633556d02422d7457f548cc58f/polyfish-rs/Cargo.toml) declares a Rust 2024 crate and includes web and neural-network dependencies. The engine is not currently a minimal standalone package; first reuse its library, then split optional dependencies only if build or deployment cost warrants it.
- The [module license declaration](https://github.com/HenBOMB/Polyfish/blob/a9f90ab54607f5633556d02422d7457f548cc58f/LICENSE) assigns Apache-2.0 to the Rust engine, AGPL-3.0-only to the UI, and MIT to the mod/scraper. This project needs the Rust engine; retain applicable notices when importing code.

Unresolved: authoritative replay availability; exact supported rules version; full reachable mechanic coverage; legal-action completeness; deterministic RNG restoration; semantic state parity; hidden-information leaks through actions, identifiers, and error messages; Linux build; throughput. No trained upstream agent is presumed strong.

## Alternatives

[Polyterra 0.1.1](https://pypi.org/project/polyterra-env/) advertises a bundled C# backend and PettingZoo API. Its page currently documents macOS ARM support only, a 512-slot action list, decompiled game logic, and a placeholder source repository URL. It may be useful for a local comparison, but Linux availability, action overflow behavior, source access, and fidelity are unresolved. Do not assume it can be deployed on GPU cluster CPU workers as published.

[PolyVision](https://github.com/NeoLorenzo/PolyVision) documents a Java/Py4J environment whose initial task is ten-turn Bardur Drylands economy optimization. It is relevant infrastructure research, but that stated task does not satisfy our full combat 1v1 evaluation requirement.

## Selection rule

The user selected a GPU-native simulator as the first deliverable, with pinned Polyfish as the implementation reference. Port rules into JAX and retain the Rust engine only as a differential-test oracle. Record source ambiguities and compare other engines where useful. Full-game Polyfish parity and official-game fidelity are separate claims; neither is established by the initial kernel tests.
