# Pre-registration: HYBRID trunk and V3 unpatched (p1) + warmup vs the hero CNN (Claude session H, 2026-11-07 ~08:00 PT)

Brief: `/Users/Shared/AgentBrain/briefs/20261107-v3-prelaunch.md` item 2 (Jack approved ~07:30 PT) and
`20261107-gpu-quota-team-plan.md` (H: 2 concurrent GPUs). Jack: "Do not make my network smaller"; all V3/hybrid arms
>= 6 transformer blocks at width 128. Written before any H training or arena result. Branch `perf/claude-v3-hybrid-h`
(from E's `perf/claude-v3-compute` f36c711), worktree `/Users/Shared/PolytopiaV3/wt-v3-hybrid-h`, ledger
`/Users/Shared/PolytopiaClaude/v3h-budget.json` (cap $100), Modal app `claude-v3-hybrid-h`, own volume
`claude-v3h-results`; E's volume `claude-v3-results` mounted read-only at `/e` (map table, D_np and E's arms).

## Arms (wave 1, concurrent, one H100 each)
Common: E's no_peace + warmup recipe `experiments/claude_v3/hero-1v1-classic-nopeace-warm150.json` unchanged (seed
801, classic maps 200000-300000, lr 3e-4 -> 1e-4 over the last 25% of wall time, 150-update linear warmup), 3600 s of
training on 1xH100 4 CPU/32 GiB (= E's equal-dollar unit), E's kernel env, games 4096, sets 4, grad-accum 16,
snapshots every 10 updates. Identical to E's np-Bw except the architecture.
- **H_h4** `transformer-v3-p2-h4`, width 128, 6 transformer blocks: the hero CNN's normalized-v2 torso (3x3 stem + 4
  residual conv blocks at width 128, hero init) -> 2x2 patches (49 + global = 50 tokens) -> 6x128 V3 blocks; tile
  heads read unpatch(out) + the CNN tile features (skip). Global token, shared public encoder, 1-block privileged
  critic, unchanged; leakage tests pass for -h1/-h2/-h4 (commit 4b1152b).
- **H_p1w** `transformer-v3` (unpatched, 197 tokens), 6x128: E's np-Bw with patching removed. Separates "patching
  hurts" from "LR too hot".

Decision: H_h4 uses the full hero torso (4 blocks), i.e. literally "the hero CNN's conv layers"; 2-block variants
would shrink the network, against Jack's size rule. Decision: H_p1w keeps warmup at 150 UPDATES (not rescaled to its
~3.5x lower update rate), so its update-indexed LR schedule matches np-Bw exactly and the equal-samples comparison
isolates patching; consequence: at ~17k samples/s it spends most of its hour in warmup, which handicaps it at equal
dollars (reported as a caveat, not corrected post hoc).

## Matches (E's GPU arena `experiments/claude_v3/gpu_arena.py`, unchanged)
Held-out classic seeds 300000+, paired seats, T=1, score-at-cap, map-clustered 95% CI, tick cap 6000, 1024 maps
(2048 games) unless noted.
1. **Primary, equal dollars:** H_h4 final vs D_np final (`/e/distributed/np-D-cnn-4x128/final.npz`).
2. Equal dollars: H_h4 final vs E-best final. E-best = E's arm with the highest score vs D_np among np-B, np-Bw,
   np-Bk3 in E's own arena results; ties or missing results -> np-Bw (the recipe twin).
3. Equal dollars: H_p1w final vs D_np final; H_p1w final vs np-Bw final (512 maps).
4. Equal samples: H_p1w final vs np-Bw at the snapshot nearest to and >= H_p1w's final update (patching effect per
   sample; same warmup schedule; caveat: H_p1w is in its time-based LR decay, np-Bw at that update is not).
5. Equal samples: H_h4 final vs D_np at the snapshot nearest to and >= H_h4's final update, if H_h4 runs fewer
   updates than D_np (hybrid vs CNN per sample; else D_np final vs H_h4 at D_np's update count).
Rule (E's): win if the CI lower bound > 0.5, loss if the upper bound < 0.5, otherwise inconclusive. **Prelaunch gate
for the hybrid: H_h4 does not lose match 1.** Speed is reported from training.jsonl (samples/s, $/B at $4.39/h).

## Wave 2 (optional)
Any further arm (bigger only: e.g. hybrid with 8x128 or 6x256 transformer, or F's recipe stack) is registered by a
dated amendment below before launch. Budget bound per 1 h arm: timeout 4500 s x $4.39/h = $5.49; per arena call
1800-2700 s bound = $2.20-$3.30. Wave 1 + matches bound ~$20 of $100.

## Amendment 1 (~08:15 PT, before any H result; training still running)
E's np arena results are not posted yet, so match 2 is run against all three E arms (np-Bw and np-Bk3 at 1024 maps,
np-B at 512); E-best is still picked by the registered rule (E's own scores vs D_np) and the others are secondary.
Matches split over two arena calls, one per GPU (`experiments/claude_v3h/make_matches.py`): A = hybrid equal-dollar
matches (D_np first), B = p1w matches + the hybrid's equal-samples match. Steady speeds seen in training logs at
~update 12 (not results): H_h4 ~42.0k samples/s, H_p1w ~17.7k.
