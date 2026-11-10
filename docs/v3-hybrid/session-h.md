# V3 HYBRID trunk — Claude session H (2026-11-07 PT)

Task (Jack via the Claude infra session, ~07:45 PT): prelaunch item 2. Branch `perf/claude-v3-hybrid-h` (from E's
`perf/claude-v3-compute` f36c711), worktree `/Users/Shared/PolytopiaV3/wt-v3-hybrid-h`, ledger
`/Users/Shared/PolytopiaClaude/v3h-budget.json` (cap $100, <= 2 concurrent H100), Modal app `claude-v3-hybrid-h`,
volume `claude-v3h-results` (+ E's `claude-v3-results` read-only at `/e`). Raw logs: `/Users/Shared/PolytopiaV3/results-h/`.
Pre-registration: `prereg.md`. Headless: every open choice has a recorded default.

## Decisions taken without a human (each with the reason)
1. **Hybrid = `transformer-v3-p2-h<n>`, an option of E's module, not a new module.** Only the stem changes: the hero
   CNN's normalized-v2 torso (3x3 stem, n residual blocks `x <- act(x + conv(act(conv(x))))`, act = SiLU(LayerNorm),
   hero init) produces the tile embedding; E's patching, global token, blocks, heads and critic are reused as is,
   so all of E's leakage tests apply unchanged (parametrized over -h1-c0 / -h2 / -h4). A test pins that with the
   hero CNN's kernels copied in, the hybrid's tile embedding equals `network.trunk`'s CNN tiles (rtol 1e-5).
2. **n = 4 (the full hero torso), transformer 6x128.** "The hero CNN's conv layers" literally; fewer conv blocks
   would shrink the network (Jack: "Do not make my network smaller"; minimum 6x128 in the transformer part).
3. **The critic's privileged embedding stays 1x1 (as in -k3).** The critic is a small branch reading the public
   tokens; giving it its own conv torso would add a second trunk, the cost V3 removed.
4. **Patch after the convs (2x2, 50 tokens).** As specified; the heads read unpatch(out) + CNN tiles, so tile
   resolution comes from the CNN and global context from attention.
5. **p1 arm keeps warmup at 150 updates** (prereg): identical update-indexed schedule to np-Bw, so the equal-samples
   match isolates patching. Caveat: it spends ~64% of its hour in warmup.
6. **Recipe: E's `hero-1v1-classic-nopeace-warm150.json` unchanged for both arms**, so H_h4 vs np-Bw/np-Bk3 differ
   only in architecture. D_np (the bar) has no warmup; that is E's registered bar, not changed.
7. **Own volume, E's volume read-only.** No copy of the 100k-map table; no write path into E's files.

## Results
(filled in below as they land)
