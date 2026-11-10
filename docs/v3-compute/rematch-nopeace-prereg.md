# Pre-registration: no_peace rematch, V2-6x128 vs V3p2-6x128 (session E, 2026-11-07 ~05:05 PT)

Written after the first learning check (learning-check-prereg.md). There, B lost to A at equal dollars (0.392) with
signs of the peace-until-cap equilibrium in B and C, and before any no_peace run started.

Hypothesis: B's 1-hour loss comes from the recipe (peace allowed, `no_peace` off, so faster arms reach the degenerate
equilibrium sooner), not from the V3 architecture.

Arms: exactly arms A and B of the first check (same seed 801, maps, container, kernels, 3600 s, snapshots every 10)
with recipe `experiments/claude_v3/hero-1v1-classic-nopeace.json` (adds `no_peace: true`, passed to engine-cu by
4902b8d). A_np = `transformer-v2` 6x128, B_np = `transformer-v3-p2` 6x128. They run one after the other within the
2-GPU allocation.

Matches (GPU arena, as Amendment 1: held-out classic seeds 300000-301023, 2048 games, T=1, score-at-cap, map-clustered
95% CI, tick cap 6000):
1. Primary, equal dollars: B_np final vs A_np final.
2. Equal samples: B_np at the snapshot nearest to and at least A_np's final update count, vs A_np final.
3. Recipe effect: B_np final vs B final; A_np final vs A final.
Decision rule as before: a win if the lower bound is > 0.5, a loss if the upper bound is < 0.5, otherwise
inconclusive. Each arm reports its second-half capped-game share from training.jsonl. The hypothesis is supported if
B_np's capped share is < 5% AND B_np does not lose to A_np.

## Amendment 1 (~05:55 PT, before any no_peace result)
Add arm **B_np_w**: B_np plus `lr_warmup_updates: 150` (recipe `hero-1v1-classic-nopeace-warm150.json`; linear
warmup over the first 150 updates, ~12 min at V3p2 speed). Reason, from the training logs only (no arena result
involved): over updates 20-50, B had clip fraction 14-15% and approx_kl .009-.012, against A's 4.9% / .005, and B's
return std reached A's update-50 level only at update ~100. Same matches and rule: B_np_w final vs A_np final
(equal dollars), and B_np_w final vs B_np final (warmup effect).

## Amendment 2 (~06:00 PT, before any no_peace result; the first learning check's D-vs-A and B-vs-D are known)
Add arm **D_np**: the hero-recipe CNN (`normalized-v2` 4x128) with the no_peace recipe. Reason: in the first check,
the CNN at equal dollars beat both V2 (0.966) and V3p2 (0.997), so it is the bar V3 must clear (BORROW.md §0). Added
matches: B_np final vs D_np final, B_np_w final vs D_np final, and A_np final vs D_np final. Same rule. Launch order,
2 GPUs: B_np (running), D_np, B_np_w, A_np.

## Amendment 3 (~06:02 PT, before any no_peace result)
**Drop A_np. Add B_np_k3**: `transformer-v3-p2-k3` 6x128 (a 3x3 conv stem before patching, 35ea74e) with the no_peace
+ warmup-150 recipe. Reasons: (1) at equal dollars V2 lost to the CNN 0.966, so V2 is no longer the bar and A_np would
spend $5.5 measuring a dominated arm; (2) the open question is whether a V3 fix closes the per-sample gap to the CNN,
and the stem's missing locality is the leading architectural hypothesis. **Primary comparisons are now vs D_np
(equal dollars): B_np, B_np_w, B_np_k3 final vs D_np final.** Secondary: B_np_w vs B_np (warmup effect), B_np_k3
vs B_np_w (stem effect), and equal-samples versions at each V3 arm's snapshot nearest D_np's update count, where
available. The B_np-vs-A_np matches in the original registration are withdrawn, together with A_np.

## Amendment 4 (~06:25 PT): equal-samples ladder from the first check's snapshots (no new training)
At update 110 (28.8M samples) each: E (V3 6x128, unpatched, 30-min run) vs A, B vs A, D (CNN) vs A, E vs B; 512 maps
each. This separates the cost of the patches (E vs B) from the cost of the shared encoder, stem and critic (E vs A),
and measures the CNN's per-sample efficiency against V2 (D vs A). Note: E's run was 1800 s, so its LR schedule decayed
earlier. At update 110, E is in its decay phase while A, B and D are still at constant LR. Reported as a caveat.
