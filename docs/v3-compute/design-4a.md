## 4a. V3 compute graph and measured cost (Claude session E, 2026-11-07 PT)

Source: private branch `perf/claude-v3-compute` (worktree `/Users/Shared/PolytopiaV3/wt-v3-compute`), module
`src/polytopia_bot/rl/transformer_v3.py`, trainer integration in `rl/gpu_resident.py`. Decisions log and raw
results: `docs/v3-compute/session-e.md`, `experiments/claude_v3/results-*.json`. Session A reviewed 6ce53d3
(information flow OK). Codex review of the graph is still pending.

**Graph (`transformer-v3[-p2][-c<k>]`, width W, depth D, 14x14 board, 2 seats, obs v3: 138 board channels, 535 global)**
- Stem: per-tile linear map 138 -> W, i.e. summed categorical embeddings plus a scalar projection on the existing planes
  (observation ABI unchanged). Session A's and Codex's note stands: this reads the dense planes, so it saves stem
  MACs (V2: 3x3 conv, 9 x 138 x W per tile), not input bandwidth.
- Tokens: `-p2` concatenates the 2x2 tile embeddings (4W) and maps them linearly to W, giving 49 patch tokens + 1
  global token = **50 tokens** (V2 and unpatched V3: 197). The global token is a linear map of the 535 global
  features.
- D pre-LN blocks as V2 (4 heads, 4x SiLU FFN, learned relative bias [4, 2g-1, 2g-1] with g = 7 for p2, 14 for p1,
  Tokamax Triton attention, Tokamax affine LayerNorm), then a final affine norm.
- Tiles for the heads: `-p2` maps the patch output W -> 4W, unfolds it to 196 tiles and adds the tile embedding (skip),
  so every tile keeps its own key and source score. `g` = the final global token (no mean/max pooling).
- Heads: V2's family (g -> 17) / source (tile -> 17, per family) / target (query-key pointer over 196 tiles or 128
  parameter embeddings), with unchanged action IDs, so Hero1 conditional targets attach (section 2).
- Critic (separate tree, critic-only): public tokens before the final norm + per-token embedding of the other seat's
  view (138 -> W, patched as the stem) + linear(global of the other seat), then k blocks (default 1), a norm, pooled
  [mean tiles, max tiles, global token] -> W -> 1. Public tokens never read privileged ones. There is no attention mask,
  because the critic stream is a separate computation after the encoder. Value gradients train the shared encoder.
- One public encoder pass per row: rollout `transformer_v3.sample_and_value` and learner `transformer_v3.evaluate`
  return policy and value from the same tokens.
- Leakage tests (`tests/test_transformer_v3.py`, all four architecture variants), all passing:
  - policy logp and entropy are bit-identical under a changed privileged view;
  - d(policy)/d(privileged) = 0 exactly;
  - the critic responds to the privileged view;
  - the public-only actor path equals the learner policy;
  - one-encoder sample/value equals the separate calls.

**Parameters (policy + critic):** V2 6x128: 2,114,939 + 2,490,491. V3p2 6x128: 1,878,650 + 400,805. V3p2 6x256:
6,440,186 + 1,423,525.

**Measured cost, 1xH100.** All rows use the same kernels, BF16 trunk, HIGHEST precision and micro 1024.

| Arm | model-only us/row (learner + rollout) | vs V2 | e2e samples/s (GPU-resident, classic maps) | $/B |
|---|---|---|---|---|
| V2 6x128 (2 trunks) | 83.0 | 1.00 | 10,931 | 111.7 |
| V3 6x128 (shared, 197 tok) | 49.0 | 1.69 | — | — |
| V3 6x128 c0 | 42.7 | 1.94 | — | — |
| V3 4x128 / 2x128 / 2x64 | 36.2 / 23.5 / 15.3 | 2.30 / 3.54 / 5.42 | — | — |
| **V3p2 6x128 (50 tok)** | **12.7** | **6.54** | **60,547** | **20.2** |
| V3p2 4x128 / 2x128 / 12x128 | 10.3 / 7.7 / 20.5 | 8.1 / 10.8 / 4.1 | — | — |
| V3p2 6x256 | 23.0 | 3.61 | 34,949 | 34.9 |

Model-only rows come from random-feature PPO minibatches and rollout calls (`experiments/claude_v3/bench_v3.py`). The
e2e rows are 900 s runs from fresh init and include the engine. The Kevin CNN reference is $29.7/B (40.5k samples/s,
proposed-v1 maps, his numbers).

**What each bet bought (at 6x128):**
- Shared encoder + 1-block critic: 1.69x.
- Removing the critic block (c0): a further 1.15x.
- 2x2 patches: a further 3.9x, i.e. 6.5x model-only in total; **5.5x end to end** (the engine and host are now ~45% of
  wall).
- Pointer heads: measured as **not a lever**. My per-block regression put the non-trunk intercept at ~5 us/row
  for V3p2 (~40%). Session G (perf/claude-v3-arch-g 8b69de4) decomposed it: critic branch 2.1 us/row (17%), heads +
  unpatch 1.25 (~10%), stem 0.6, floor 0.9. A factored patch-level decoder saved nothing. So the critic branch and the
  trunk glue (transposes, converts, reductions: ~half of the 147 ms per minibatch in G's trace) are the next model-side
  targets, not the heads.
- Encode once per turn: session A's §4b bound is +15-23% on V2. On V3p2 the rollout forward is ~1.0 s of a 4.3 s update,
  so the ceiling is below ~20%. Not implemented, because the board changes after each atomic action.
- Width/depth: see the equal-dollar ladder below. At 15 min (~$1.1), V3p2-6x128 beat V3p2-6x256 977-47.

**Strength per dollar (the measure that decides).** GPU arena, 2048 paired games, T=1, held-out classic seeds,
score-at-cap, 95% CI clustered by map. Equal 1 h of 1xH100 per arm, seed 801, no_peace off:
- hero CNN 4x128 vs V2: 0.966 [0.958, 0.974];
- V3p2-6x128 vs V2: 0.392 [0.371, 0.413];
- V3p2 vs CNN: 0.003;
- at equal samples (39.3M), V3p2 vs V2: 0.002.

**The compute bets made V3 samples 5.5x cheaper, but no V3 variant tested so far learns enough per sample to beat V2
per dollar at 1 h, let alone the CNN.** Recipe fixes (no_peace, LR warmup) and a 3x3 conv stem are being tested
against the CNN under `rematch-nopeace-prereg.md`. Until a V3 arm clears the CNN at equal dollars, **the GPU-resident
hero CNN is the strength-per-dollar choice**, and V3's compute graph is an option, not the selected replacement.
