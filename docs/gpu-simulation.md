# GPU simulator implementation and validation

Updated 2026-10-16. **`FULL_GAME_READY=True` for frozen `bardur-1v1-v1`.** The user selected one consistent research ruleset instead of tracking commercial releases; see [the frozen contract](ruleset-v1.md). Strategic bots, demonstration collection, SFT and RL execution are outside the current implementation phase. The random-action drivers are correctness tests, not teachers or competitive opponents.

## Device boundary

The simulator uses fixed-shape JAX PyTrees with tile-indexed units and cities, two players, explicit game RNG, and an atomic `(kind, source, target, parameter)` action ABI. Normal transitions call no CPU game engine. `jit`, `vmap` and `lax.scan` support batching and device-resident sequences. Reference engines are validation tools only.

`load_rules()` selects the frozen **native research profile**. `load_rules("polyfish")` explicitly selects compatibility with Polyfish `a9f90ab54607f5633556d02422d7457f548cc58f`. Passing reference tests does not certify native rules. See [native-rule exceptions](native-rules-exceptions.md) for the evidence and remaining uncertainty.

A rejection leaves every state field unchanged. Statuses are success (0), invalid action (1), unsupported state/rule (2), stale version (3), and finished episode (4). Native elimination and resignation yield seat-indexed win/loss rewards. Safety turn/action limits are truncations, not score wins or draws.

## Implemented behavior

| Area | Implemented | Validation boundary |
|---|---|---|
| Movement | Multiple hops, roads, terrain access/stopping, ZOC, friendly transit, strict reference path order, exploration, ports, bridges, embark/disembark, naval movement, hidden contact | Reference ordering matches covered cases; native friendly-transit and hidden-contact regressions pass |
| Combat | Damage/retaliation, range, advance, death, promotion, recovery, healing others, Persist/Escape, splash/Stomp, Mind Bender conversion, native Raft restrictions | Ordinary and naval goldens; native exceptions have separate regressions |
| Economy | Research, harvest, forest abilities, buildings/destruction, population/levels/rewards, dynamic multiplier and market levels, temple growth, roads/port connections, monuments | Native task persistence, wealth savings and market/temple rules covered; commercial ownership-transfer equivalence is not certified |
| Capture/exploration | Villages/cities, original capital identity, elimination, ruins, water rewards, explorers, starfish, lighthouses | Reference seeded rewards covered; commercial generator/RNG equivalence unproven |
| Diplomacy | Directed embassies, income, peace offers/responses, break grace, friendly roads/healing, capital capture cleanup | Five-star pricing and explicit v1 timing are frozen; commercial equivalence is not claimed |
| Cloaks | Hide, Detect indicator, contact reveal, infiltration damage/riot, Dagger/Pirate spawning | Uniform-land seeded placement matches 32 historical probe cases; commercial replay equivalence is not claimed |
| Legal actions | Full factored catalog without arbitrary action-count limits, device decoder | 580 supported reference action-set cases; native decode/execution and information-boundary regressions |
| Policy view | Masked tiles/resources/enemy units; owned unit/city state; diplomacy and achievements | Paired-hidden-world tests pass for covered cases; the policy schema is intentionally narrower than the commercial UI |
| Reset/snapshots | Procedural GPU maps, device gather from initial pools, selective lane reset, private lossless snapshots and deterministic restore/replay | Forty exact corrected-reference layouts plus initial-state tests; snapshots require exact schema/profile/implementation hash |
| Rollout testing | Device-side family-uniform legal sampling, decode and mixed transitions; finished lanes freeze | Test driver only; not an expert policy and not a trained model |

Special-tribe mechanics are not the target. Reachable ordinary Bardur mechanics may not be omitted merely because they are rare. The importer refuses unsupported source fields rather than silently dropping their effects; it is a reference-state projection, not a native commercial-save importer.

## Evidence

The final workspace CPU suite passed **1,977 tests in 77.07 seconds**, including separately added learning-interface tests. The preceding simulator-only snapshot contained 1,951 tests. Fixtures include 28 core boards, 4,096 arithmetic combat cases, 580 action-space states and 1,148 transition comparisons. These counts describe tests/cases, not independent full games.

Reference transition comparisons check all represented source fields, including unit flags, passengers, home cities, economy, exploration and RNG. Native-only fields are independently tested; they are not assumed equal to defective Polyfish behavior.

The ten-seed reference replay receipt records 5,441 matching transitions. It contains one native elimination, eight reference turn-cap truncations, and one trace stopped at the runner's 600-action limit. It is **not ten completed native games**. The 100-seed replay completed with 50,392 matching transitions: 13 native eliminations and 87 reference turn-cap truncations. All traces reached an engine terminal/truncation; this is not 100 native victories. Its receipt is `benchmarks/2026-10-16-reference-replays-100.jsonl`. The run predates later native-only corrections and did not capture per-file hashes.

Historical L4 receipts under `docs/benchmarks/` cover earlier snapshots: 135, 337 and 859 passing tests. The subsequent expanded run `ap-bMnQ1nsLUS8I8gkH4EYGlD` exceeded its 600-second test budget; it is not a pass. The replacement run streams progress and has a 1,200-second test budget within a 1,800-second function limit. Final tested-file SHA-256 hashes identify the exact mounted source.

The optional historical probe checks isolated hash and infiltration-order outputs. Its underlying package contains decompiled logic and incomplete bundled tables; it is not an authoritative current-game oracle or a runtime dependency. No reference engine or proprietary commercial-game package is included in the CUDA image.

## Modal staging

```sh
modal run --env staging experiments/modal_gpu.py
modal run --env staging experiments/modal_rollout.py
```

Both entrypoints reject a non-staging Modal environment. Each app uses one L4, at most one container, no retries and no persistent service. The main validation run has a 30-minute maximum; the separate mixed-action benchmark has a 15-minute maximum.

The original microbenchmarks measure repeated EndTurn and one-unit movement/research/observation, with explicit compilation, warm-up, synchronization and sample timings. Their throughput must not be described as full-game throughput.

The mixed benchmark additionally measures the complete action masks, family-uniform random sampling, decoding and mixed transitions on 14×14 boards. It reports actual action counts and active transitions. Fixed-length traces are still not completed-game or LM throughput. The mixed benchmark now times procedural GPU reset separately. Neither benchmark includes model inference or training.

## Frozen release acceptance

The earlier open-ended commercial-fidelity gates are superseded by the user's explicit instruction to freeze one working, consistent ruleset. Version-specific uncertainties are documented compatibility choices in [ruleset-v1.md](ruleset-v1.md), not reasons to silently keep changing the environment. The research profile includes all implemented ordinary action families, procedural resets, legal catalogs, terminal/truncation handling and deterministic private snapshots.

The 1,893-test expanded snapshot passed on L4 in 742.61 seconds; its complete receipt is `benchmarks/2026-10-16-l4-expanded-1893.json`. A later 72-test CUDA suite covers RNG, procedural maps and additional native cases. The mixed receipt `benchmarks/2026-10-16-l4-procedural-mixed.json` records 20.73 transitions/s at batch 1 and 96.09 at batch 8 over 32-step traces; batch-8 reset median is approximately 1.03 seconds. Compilation is excluded from those rates.

The final frozen-source eight procedural 14×14 games produced 5,710 legal transitions on CPU and L4, with identical reported action counts, per-lane action totals and final rounds. All eight reached the round-100 safety limit; none was an elimination. This is a stability check, not a policy-strength result or a full-state CPU/GPU trajectory equality proof. A separate deterministic capture episode exercises elimination and absorbing terminal behavior.

The final frozen-source suite passed **1,977 tests on CPU in 77.07 seconds and 1,977 tests on L4 in 752.13 seconds**. Matching CPU/L4 long-episode summaries and source-specific receipts are linked from `configs/ruleset-v1.json`. No teacher qualification, demonstration collection or training result is claimed by these simulator checks. See the comprehensive post-training plan for the next phase.

Legacy microbenchmark receipts contain `full_game_ready: false` and `Partial rule slice` in their metadata. Those labels describe the original benchmark coverage, not the frozen release flag; the benchmark itself still measures only EndTurn and one-unit queries. The ruleset manifest and release acceptance receipt are the authoritative release status.
