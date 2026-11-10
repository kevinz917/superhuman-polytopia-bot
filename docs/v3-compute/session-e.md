# V3 compute bets — Claude session E (2026-11-07 PT)

Brief: `/Users/Shared/AgentBrain/briefs/20261107-v3-run-the-big-bets.md`. Branch `perf/claude-v3-compute` (from
`perf/claude-v2-learner-kernels` 2e64be6), worktree `/Users/Shared/PolytopiaV3/wt-v3-compute`. Ledger
`/Users/Shared/PolytopiaClaude/v3-budget.json` (cap $100, container-seconds x Modal list price). Modal app
`claude-v3-compute`, volume `claude-v3-results`. Headless: every open choice below has a recorded default.

## Decisions taken without a human (each with the reason)

1. **Categorical embedding stem = per-tile linear map of the existing feature planes.** On one-hot planes a linear
   map is exactly a sum of learned categorical embeddings, and unknown/absent/zero stay distinct because the planes
   keep them distinct. This keeps the observation ABI, so arenas, CPU actors and Codex's Hero1 distillation adapter
   read V3 unchanged. The compute change is V2's 3x3 conv (9 x 138 x W MACs/tile) -> 1x1 (138 x W). A gather-based
   implementation would only skip materializing the [196, 138] one-hot (~108 KB/row, microseconds of HBM time).
2. **Pointer heads: keep V2's family/source/target factorization.** V2's target head is already a query-key pointer
   over tile keys or parameter embeddings, and its source head is a per-tile score with one learned query per
   family (a pointer with 17 queries). V3 only replaces CNN-style mean/max pooling for `g` with the global token.
   Keeping the factorization lets Hero1's conditional teacher targets attach unchanged (DESIGN.md section 2). The
   head share of model time is measured below as the upper bound on any head saving.
3. **Fewer tokens: 2x2 tile patches (50 tokens at 14x14), not fog/empty dropping.** Patching gives a fixed 4x token
   cut that batches without padding or packing. Tiles stay addressable: decoding uses a learned per-sub-position
   projection plus a skip from the tile embedding. Fog dropping is variable-length, needs packing and can remove
   movement targets (Codex review, DESIGN.md section 4).
4. **Encode once per turn: measured as a bound, not implemented.** The board changes after every atomic action, so a
   cached encoding is stale (Codex review). That would be a different policy class needing its own learning check.
   Its payoff is bounded by the rollout share of the end-to-end time, measured below.
5. **Critic: a critic-exclusive branch reading the public tokens before the final norm, plus a 1x1 embedding of the
   other seat's view, then k transformer blocks (k = 1 default, 0 = pooled MLP), pooled -> MLP -> value.** The public
   encoder never reads privileged input. Value gradients do train the shared encoder, as DESIGN.md section 1 allows.
6. **Environment for my runs: engine-cu (main's current rules) + `classic` maps** (Jack's rule: latest official
   rules/maps; `classic` is the engine-rs generator port, seed-exact). The map table was built on my own volume. The
   V2 control is re-run by me on the same maps, so V2-vs-V3 comparisons are matched. Retail 2.18.0 parity remains
   unverified (Steam check, Kevin/Jack action).
7. **Evaluation: a new GPU arena (engine-cu, host-pointer mode).** Paired seats on held-out classic seeds 300000+,
   T=1 sampled play (deterministic argmax stalled early policies: 79/128 games unfinished at 2,000 ticks; amendment
   1), results both terminal-only and score-at-cap. The trainer's frozen-opponent path cannot pit two
   architectures against each other, and the CPU arena cost session B $55.

## Results (all 1xH100 Modal list price $4.39/h incl. 4 CPU / 32 GiB; engine-cu current rules; classic maps)

### Speed (bench-01/02 model-only; e2e = 900-3600 s GPU-resident runs from fresh init)
See `design-4a.md` table. Headline: V3p2-6x128 **60.5k samples/s, $20.2/B** vs V2-6x128 10.9k/s, $112/B (5.5x),
hero CNN 4x128 48.3k/s, $25.3/B; V3 unpatched 6x128 17.2k/s $71/B; V3p2-6x256 35.0k/s $34.9/B.

### Strength (GPU arena, held-out classic seeds 300000+, paired seats, T=1, score-at-cap, map-clustered 95% CI)
First learning check (`learning-check-prereg.md`; no_peace OFF; 3600 s each, seed 801):
| Match | games | score [95% CI] | conquest W-L |
|---|---|---|---|
| B V3p2-6x128 vs A V2-6x128 (equal $) | 2048 | **0.392** [0.371, 0.413] | 138-1076 |
| C V3p2-6x256 vs A (equal $) | 2048 | 0.190 [0.173, 0.207] | 0-1583 |
| B vs C | 2048 | 0.952 [0.942, 0.961] | 1443-0 |
| **D CNN-4x128 vs A (equal $)** | 2048 | **0.966** [0.958, 0.974] | 1946-44 |
| **B vs D (equal $)** | 2048 | **0.003** [0.001, 0.005] | 2-2020 |
| B@u150 vs A final (equal samples, 39.3M) | 2048 | 0.002 [0.000, 0.004] | 0-2015 |
| C@u150 vs A final (equal samples) | 2048 | 0.007 | 1-1980 |
| B curve vs A final: u50 / u100 / u190 / u250 / u350 / u450 / u550 / u650 / final u753 | 1024 each | 0.000 / 0.000 / 0.032 / 0.066 / 0.254 / 0.224 / 0.295 / 0.299 / 0.392 | |
| 15-min smoke: V3p2 vs V2 (equal $, 900 s each) | 1024 | 0.999 | |
| 15-min smoke: V3p2-6x256 vs V3p2-6x128 | 1024 | 0.046 | |

Reading: V3p2 makes samples 5.5x cheaper than V2, but per sample it learns far less. At equal samples it lost
4-2044, and it needs more than 5x V2's samples to match V2 (curve). So per dollar it is behind V2 at 1 h, and far
behind the hero CNN, which beat both. Only in the first 15 minutes, when V2 has 9M samples, does V3p2's volume win.
Peace drift (capped share B 17%, C 97% in the second half with no_peace off) is real, but it is not the main cause
for B: B's curve rose monotonically. The smaller V3 beat the larger one at every budget tried.
