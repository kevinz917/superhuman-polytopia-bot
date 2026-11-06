# Polytopia bot: RL/search experiments of 2026-10-19 and 2026-10-20 (internal document research)

Scope: every experiment recorded in the project's own committed reports dated 2026-10-19/20, plus the immediately
preceding context (distributed campaigns 1–3, unit-diversity investigation, value-head probes). All numbers are
quoted as the source documents state them. Read-only; no training or Modal commands were run.

**Source key (paths are repo-relative; branch given where the doc lives only there).** Main checkout:
`/Users/K.Z./Desktop/projects/polytopia-bot`. Worktree `.claude/worktrees/sharp-goldstine-e442a1` = branch
`claude/rl-18x18-a4c945`.

| Label | Path | Where |
|---|---|---|
| TTS-u3142 | `docs/test-time-search-2026-10-19.md` | worktree / all branches |
| TTS-c3 | `docs/test-time-search-c3-2026-10-19.md` | worktree |
| SPEED | `docs/search-speed-2026-10-19.md` | worktree (also `claude/fast-search`) |
| DISTILL | `docs/search-distillation-2026-10-19.md` | branch `claude/search-distill-1ae5b0` only |
| L18 | `docs/rl-18x18-2026-10-19.md` | worktree |
| BOTTLE | `docs/rl-18x18-bottlenecks-2026-10-20.md` | worktree |
| HORIZON | `docs/long-horizon-credit-2026-10-20.md` | worktree |
| NEXT | `docs/next-run-and-strength-2026-10-20.md` | worktree |
| ARMS | `docs/ablation-v1/arms-2026-10-20.md` | worktree |
| READOUT | `docs/ablation-v1/readout-2026-10-20.md` | worktree |
| DIAG | `docs/ablation-v1/diagnosis-18x18-2026-10-20.md` | branch `claude/diag-18x18-roundrobin-bb5639` only |
| GATE | `docs/ablation-v1/precision-gate-2026-10-20.md` | branch `claude/precision-gate-d7f0e8` only |
| RULES2 | `docs/ruleset-v2.md` | worktree |
| DRL | `docs/distributed-rl-2026-10-19.md` | main (identical copy in worktree) |
| MAPS | `docs/map-variants-2026-10-19.md` | main (identical copy in worktree) |
| UNITS | `docs/unit-diversity-investigation-2026-10-18.md` | main |
| PROBES | `docs/policy-probes-2026-10-19.md` | branch `claude/ecstatic-kilby-e38baf` only |
| DIVERSE | `docs/diverse-strategy-check-2026-10-19.md` | worktree / main |

Naming: **u3142** = `dist-c460div-v1` update 3142 (campaign-1 bot); **E3** = `dist-e340ent-v1` final (sibling
lineage); **c3** = `c3-maps-ctl` update 4854 (campaign-3 bot, map mix); **grown c3** = c3 with two extra residual
blocks for 18×18; **league96** = older imitation+RL bot. Elo CIs are paired-map bootstrap 95% intervals unless
noted. Score = (W + ½·cap)/N; "of decisive" excludes capped games.

---

## Key question 1: Test-time search on u3142 and c3 (leaky vs leak-free, d=1 vs robust, classic vs map mix)

### Takeaway
One-turn lookahead (d=1, K=16 whole-turn candidates scored by the critic after the opponent's reply) is the only
inference-time lever that moved strength: leaky it adds +125–141 Elo on u3142 and +108–162 on c3; removing the
fog-of-war leak (determinization) keeps roughly half on u3142 and classic maps (+56 to +82) but most of it on the
map mix c3 was trained on (+104 of +162). Robust search (64 averaged replies, sequential halving) scales with
compute when leaky (+172 on u3142 vs E3; +200 on c3 classic) but adds only +13 to +29 (n.s.) once leak-free, at
~2.5× the compute. Critic-only scoring (d=0) at T=1 and a Gumbel root search failed.

### Cited Findings

**Method / protocol (u3142 study)**
- Search unit is a whole turn: K lanes copy the engine state; lane 0 = raw greedy policy (so K=1 is exactly the raw bot); lanes 1..K−1 sample the whole turn at temperature T; scored from the searcher's own observation by the critic at END (d=0), after END, or after the opponent's reply played by the searcher's own network (d=1); win/loss lanes scored ±1, lanes hitting the 100-turn cap scored 0; ties keep the greedy lane — [TTS-u3142 §Method](docs/test-time-search-2026-10-19.md)
- Evaluation: development seeds only (360000–360099 vs u3142, 361000–361099 vs league96, 362000–362099 vs E3); each map in both seats, 200 games per arm; opponents play raw greedy; Elo vs the K=1 arm on the same maps with paired-map bootstrap 95% CI; compute = wall-clock seconds of one Modal CPU core per searched turn, "Modal cores are ~3× slower than an M-series core" — [TTS-u3142 §Method/Evaluation](docs/test-time-search-2026-10-19.md)
- u3142 plays "about 17–19 sub-actions per turn against strong opponents and about 14 against league96" — [TTS-u3142 §Method](docs/test-time-search-2026-10-19.md)

**u3142 mirror (K=1 = 80–80–40, score 0.500), all leaky**
- k4 d=0 T=1: 95–82–23, +23 [−12, +56], 0.34 s/turn; k16 d=0 T=1: 99–83–18, +28 [−9, +65], 1.00 s; k64 d=0 T=1: 90–82–28, +14 [−19, +47], 3.79 s — [TTS-u3142 §Results/vs raw u3142](docs/test-time-search-2026-10-19.md)
- k16 scored after END: 95–86–19, +16 [−21, +53], 1.30 s — [TTS-u3142 §Results/vs raw u3142](docs/test-time-search-2026-10-19.md)
- k16 d=0 T=0.5: 99–64–37, +61 [+31, +94], 1.33 s; k64 d=0 T=0.5: 108–54–38, +96 [+63, +131], 4.01 s — [TTS-u3142 §Results/vs raw u3142](docs/test-time-search-2026-10-19.md)
- k4 d=1 greedy rollout: 119–46–35, +133 [+100, +171]*, 0.64 s; k16 d=1 greedy rollout: 162–24–14, +295 [+244, +354]*, 1.80 s (*"rollout policy is exactly the opponent (deterministic u3142)... upper bounds, not strength") — [TTS-u3142 §Results/vs raw u3142](docs/test-time-search-2026-10-19.md)
- k16 d=1 rollout T=1: 123–54–23, score 0.672 [0.62, 0.72], **+125 [+87, +168]**, 1.73 s/turn, 584 lane steps/turn, override 0.89 — [TTS-u3142 §Results/vs raw u3142](docs/test-time-search-2026-10-19.md)
- k16 root=first (Gumbel-style distinct first sub-actions): 13–179–8, score 0.085, **−413 [−512, −338]** — [TTS-u3142 §Results/vs raw u3142](docs/test-time-search-2026-10-19.md)

**vs E3 (raw u3142 already 57.5%: 94–64–42)**
- k16 d=0 T=0.5: 123–47–30, 0.690, +86 [+41, +134], 1.16 s; k16 d=1 greedy rollout: 140–39–21, 0.752 [0.70, 0.80], **+141 [+87, +196]**, 2.03 s — [TTS-u3142 §Results/vs E3](docs/test-time-search-2026-10-19.md)

**vs league96 (raw u3142 93%: 186–13–1)**
- all search arms 0.922–0.968, Elo CIs ±100–150 and all spanning 0; "league96 cannot separate the arms" — [TTS-u3142 §Results/vs league96](docs/test-time-search-2026-10-19.md)

**Compute (u3142 study)**
- ~3.7 ms per lane step on one Modal core; d=1 ≈ 2× d=0; after-END scoring +~30%; one process compiles in ~170 s; a d=1 K=16 game (~45 searched turns) ≈ 80 s — [TTS-u3142 §Compute](docs/test-time-search-2026-10-19.md)

**Robust d=1 (averaged replies + sequential halving), vs E3, leaky**
- d=1 one random reply K16: 132–49–19, +101 [+51, +151], 1.95 s; robust K8/32 replies: 138–40–22, +134 [+84, +187], 2.44 s; robust K16/64: 147–33–20, 0.785 [0.74, 0.83], **+172 [+118, +233]**, 4.51 s/turn, 1498 lane steps/turn — [TTS-u3142 §Robust d=1](docs/test-time-search-2026-10-19.md)
- Paired: robust K16/64 vs one-random-reply K16 **+72 [+22, +125]**; robust K8/32 vs one-random K16 +33 [−16, +81] (same compute); K16/64 vs K8/32 +39 [−17, +96]; one random vs one greedy reply −40 [−91, +7] — [TTS-u3142 §Robust d=1](docs/test-time-search-2026-10-19.md)
- Predicted gain per turn roughly halves with averaging (0.041 → 0.019–0.024), "so it overrates its picks less"; "first setting where more compute keeps paying off" — [TTS-u3142 §Robust d=1/Reading](docs/test-time-search-2026-10-19.md)

**Leak-free (determinized) addendum, u3142**
- vs E3: leak-free d=1 K=16 greedy rollout 122–48–30, 0.685 [0.63, 0.73], **+82 [+30, +139]** at 4.73 s/turn, 2.21 plans/turn (leaky +141 [+87, +196] at 2.03 s) — [TTS-u3142 §Addendum/Results](docs/test-time-search-2026-10-19.md)
- vs u3142: leak-free d=1 K=16 T=1 rollout 103–71–26, 0.580 [0.53, 0.63], **+56 [+21, +92]** at 4.64 s/turn, 2.30 plans/turn (leaky +125 [+87, +168] at 1.73 s) — [TTS-u3142 §Addendum/Results](docs/test-time-search-2026-10-19.md)
- Leak-free minus leaky, paired: −58 [−114, −6] vs E3; −69 [−122, −19] vs u3142; "hidden information was roughly 40–55% of the measured gain"; cost "~4.7 s of one Modal core per turn, ~2.4× the leaky search and ~45× the raw policy" — [TTS-u3142 §Addendum/Answer](docs/test-time-search-2026-10-19.md)
- Determinization: one sampled world per plan consistent with the searcher's observation; plan cut at the first surprise and re-planned (2.2–2.3 plans/turn); unexplored tiles, hidden crop/metal, fogged/invisible enemy units (except detected ones and ones on own city tile), unseen enemy cities, enemy stars/tech (symmetric priors), RNG seeds all redrawn; residual leaks: enemy city count/internals, seen-unit internals, score/monuments/diplomacy — [TTS-u3142 §Addendum/How it works](docs/test-time-search-2026-10-19.md)
- "The prior is crude. No enemy units are imagined in the fog... the gap between the two arms is 'leak + belief error'"; fallback never triggered in 400 evaluation games — [TTS-u3142 §Addendum](docs/test-time-search-2026-10-19.md)
- Truncations vs E3 rise 21 → 30 leak-free (raw 42); sub-actions/turn ~17.7 and override 0.87–0.89 unchanged — [TTS-u3142 §Addendum/Results](docs/test-time-search-2026-10-19.md)
- "Not yet measured: leak-free robust d=1" (at the time of this doc; measured later on c3) — [TTS-u3142 §Addendum/Answer](docs/test-time-search-2026-10-19.md)

**Failure modes (u3142)**
- Optimizer's curse at d=0: predicted gain of chosen over greedy turn grows +0.013 (K=4), +0.024 (K=16), +0.032 (K=64) per turn while realized strength stays +14 to +28; overrides 65–95% of turns — [TTS-u3142 §Failure modes 1](docs/test-time-search-2026-10-19.md)
- Scoring-time hypothesis refuted: after-END scoring gave +16 [−21, +53], turns just as short (16.2 sub-actions vs 19.0 greedy) — [TTS-u3142 §Failure modes 2](docs/test-time-search-2026-10-19.md)
- root=first: 93% overrides, turns shrank to 9 sub-actions, recruiting 0.9 units/turn, lost 179/200 in ~20 turns; "critic is confidently wrong on states the policy never reaches" (stated as plausible, not tested) — [TTS-u3142 §Failure modes 3](docs/test-time-search-2026-10-19.md)
- Value-head calibration: in the mirror the greedy-turn critic mean is within ~0.03 of outcome per turn bucket; vs E3 it underestimates u3142's edge by 0.08–0.13 through turn 59; correlation with outcome 0.45–0.52 in turns 0–9 rising to 0.6–0.87; vs league96 mean V ≈ 0.03 in turns 0–9 while outcome +0.87; after turn 80 games drift to the cap with V ≈ 0; "The critic was trained with bootstrap at the cap, so it has no explicit notion that the cap is a draw. Search never exploited this." — [TTS-u3142 §Failure modes 4](docs/test-time-search-2026-10-19.md)
- No turn-cap stalling: truncations 40 → 14–28 (T=1) vs u3142, 42 → 21–30 vs E3; T=0.5 keeps 37–38 — [TTS-u3142 §Failure modes 5](docs/test-time-search-2026-10-19.md)

**c3 (u4854) study — setup**
- 100 dev maps × 2 seats = 200 games per arm, frozen engine, 100-turn cap; seeds 364000–364099 (c3 mirror classic), 365000–365099 (vs u3142 classic), 366000–366099 (c3 mirror on `proposed-v1` mix; 14–30 games per map type); all arms K=16, d=1, T=1 candidates and replies; robust = `replies=64`; leak-free = `determinize=True` (≈1.9–2.2 plans/turn); CIs from 4000-resample paired bootstrap; K=1 reproduces `cpu_arena.match` exactly — [TTS-c3 §Setup](docs/test-time-search-c3-2026-10-19.md)

**c3 + search vs raw c3, `proposed-v1` mix**
- leaky d=1: 126–39–35, 0.718, +162 [+127, +200], 1.70 s/turn; leak-free d=1: 110–52–38, 0.645 [0.59, 0.69], **+104 [+69, +141]**, 3.72 s; leak-free robust: 114–49–37, 0.662 [0.62, 0.70], **+117 [+87, +149]**, 10.77 s/turn — [TTS-c3 §Results/mix](docs/test-time-search-c3-2026-10-19.md)
- Leak-free robust minus leak-free d=1: +13 [−26, +54]; per-type leak-free robust scores: lakes 0.75 (14), waterworld 0.75 (16), continents 0.71 (26), archipelago 0.70 (22), classic 0.68 (14), ridge 0.68 (30), pangea 0.66 (16), jungle 0.65 (24), highlands 0.58 (18), drylands 0.47 (20); "drylands is the only type below 0.5 for both leak-free arms" (per-type ±0.2) — [TTS-c3 §Results/mix](docs/test-time-search-c3-2026-10-19.md)

**c3 + search vs raw c3, classic**
- leaky d=1: 116–56–28, +108 [+72, +145], 2.05 s; leak-free d=1: 95–80–25, 0.537, +26 [−9, +61], 4.38 s; leaky robust: 139–35–26, 0.760, **+200 [+162, +244]**, 4.43 s; leak-free robust: 100–70–30, 0.575, **+53 [+17, +89]**, 10.47 s — [TTS-c3 §Results/classic](docs/test-time-search-c3-2026-10-19.md)
- Removing the leak costs −81 [−132, −34] (d=1) and −148 [−197, −99] (robust) on classic; −58 [−101, −16] (d=1) on the mix; leaky robust adds +93 [+51, +138] over leaky d=1: "averaging replies mostly helps when the replies are real" — [TTS-c3 §Answer](docs/test-time-search-c3-2026-10-19.md)

**c3 + search vs raw u3142, classic**
- raw c3: 132–41–27, 0.728 [0.67, 0.78]; leaky d=1: 162–19–19, 0.858, +141 [+85, +204]; leak-free d=1: 140–29–31, 0.777, +47 [−10, +106], 5.28 s; leak-free robust: 148–26–26, 0.805, **+76 [+14, +141]**, 10.77 s — [TTS-c3 §Results/vs u3142](docs/test-time-search-c3-2026-10-19.md)
- u3142 vs c3 comparison (classic, vs raw self): leaky d=1 +125 vs +108; leak-free d=1 +56 vs +26; "The stronger policy leaves about as much for leaky search but less for leak-free plain d=1 on classic maps" — [TTS-c3 §Compared with u3142](docs/test-time-search-c3-2026-10-19.md)
- Caveats: determinized fog is filled from the classic generator even on variant maps ("mix-map leak-free numbers are, if anything, low"); opponents never search; robust leak-free games ≈10 min of one Modal core each; one 100-turn mix game took ≈65 min of search time — [TTS-c3 §Caveats](docs/test-time-search-c3-2026-10-19.md)

**Cost**
- u3142 study: main ≈ $24.6 (v1 14.90, v2 7.83, v3b 1.11, v3 failed 0.37, smokes 0.35, orchestrators 0.05); robust follow-up 5.01 + 0.06; leak-free addendum ≈ 6.45; **grand total ≈ $36.2**, estimated from container wall time, "not an invoice" — [TTS-u3142 §Cost](docs/test-time-search-2026-10-19.md)
- c3 study: c3-v2 sweep 29.25; c3-v1 failed sweep ≈ 35 ("workers crashed on an unresolved `c3` path; 44 containers idle ~25 min"); smoke/tests ≈ 0.5; **total ≈ $65** — [TTS-c3 §Cost](docs/test-time-search-c3-2026-10-19.md)

### Inferences
- Fraction of the gain that is hidden information, by the authors' paired comparisons: u3142 ≈ 40–55% (−58 of +141 vs E3, −69 of +125 mirror); c3 classic ≈ 75% (−81 of +108 d=1; −148 of +200 robust); c3 map mix ≈ 36% (−58 of +162). The leak share is largest on classic maps for the stronger policy.
- Leak-free robust vs leak-free d=1 is never significant (+13 [−26, +54] mix; classic +53 vs +26, overlapping CIs), so the reply-averaging lever is mostly a leaky artefact; a better belief prior (enemy units imagined in fog) is the untested lever that could recover part of the lost Elo (authors explicitly say +56–82 "is not a ceiling").
- Search reduces turn-cap draws in every leaky arm and still beats raw in leak-free arms, so search does not create the stalling problem; it partially fixes it.

### Gaps
- Leak-free robust d=1 was never run on u3142 (only on c3).
- No sweep of `margin`, `prior_coef`, larger K, d=1 at T=0.5, or d>1 in either study.
- Per-type c3 results are 14–30 games each (±0.2), so map-type ordering is unreliable.
- No searching opponents and no human/independent-bot evaluation anywhere.

---

## Key question 2: Fast parallel searcher (speed, equivalence, hardware)

### Takeaway
`TurnSearcher(parallel=True)` plays bit-identical turns to the lockstep leak-free robust searcher in 0.95–1.2 s
mean (0.6–0.8 s median) on a 16-core Modal CPU container versus 9.4–10.7 s on one core (8–10×); more than 16
cores cannot help, host variance is now larger than the remaining gap to 1 s, and an L4 GPU is far slower
(14.7 s/turn).

### Cited Findings
- Baseline cost of leak-free robust search on c3: ≈10.5 s per searched turn on one Modal core; parallel: "the same turns, bit for bit, in 0.95–1.2 s per turn on a 16-core Modal CPU container (median 0.6–0.8 s), 8–10× faster"; "one 32-core host ran the same code at 1.8 s. A GPU does not help (L4: 14.7 s/turn)" — [SPEED intro](docs/search-speed-2026-10-19.md)
- Profile (one pinned core, 8 positions): `plan` 98.7% (2.5 plans/turn after reveals, ≈3,500 lane-steps/turn, ≈3.0 ms/lane-step); guarded replay 0.8% (0.08 s/turn); belief sample 0.1% (0.01 s/turn) — [SPEED §Where the time went](docs/search-speed-2026-10-19.md)
- Per lane-step: network forward ≈1.3–1.7 ms (5 3×3 convs width 64 on 14×14, ≈50 GFLOP/s on one core; batching 16/64 lanes saves only 13–20%), legal masks ≈1.9 ms, observe+encode ≈0.3 ms, engine step ≈5 ms standalone (3.0 ms inside the loop); "no single piece dominates... What is slow is the structure" (`lax.map` sums all ≈1,400 lane-steps on one core) — [SPEED §Where the time went](docs/search-speed-2026-10-19.md)
- Design: each lane its own single-lane while-loop executable dispatched from a thread pool (XLA releases the GIL); pinned worker threads (16 unpinned threads 6.8× over one lane, pinned 13–14×; 32 threads 10× vs 22–24×); round-1 reply lanes start as soon as a candidate finishes (1.02 → 0.95 s/turn); critical path ≈280 lane-steps per turn instead of ≈3,700 total; "More than 16 cores cannot help" — [SPEED §What changed](docs/search-speed-2026-10-19.md)
- Equivalence: 32 single plans (4 runs) bitwise identical in actions, chosen lane, per-lane steps and scores; 84 whole turns with re-planning identical; digests identical across 16/8/4 workers, 16 vs 32 cores, Eigen threading on/off, before/after chaining; `tests/test_parallel_search.py` + search/belief tests 11 passed on Modal — [SPEED §Equivalence](docs/search-speed-2026-10-19.md)
- Benchmark (s/turn, 24 positions): lockstep 1 core 9.39–10.72 mean, p50 6.5–7.0, p90 17.5–20.0, max 38.6–43.9; parallel 4 workers 2.72; 8 workers 1.69; 16 workers on 16-core (2 hosts) **0.95 / 1.17** mean, p50 0.60/0.80, p90 1.84/2.06, max 2.69/3.57; 32-core slow host 1.77; L4 vmap-batched (3 early turns) 14.72 — [SPEED §Benchmark](docs/search-speed-2026-10-19.md)
- GPU: ≈8.6 ms per lane-step vs ≈3 ms on one CPU core, 7.5 min compile; "under vmap every `lax.cond`/`switch` of the engine runs all branches" with ≈90 ms fixed latency per step; nothing to amortize with 16–64 lanes — [SPEED §Benchmark](docs/search-speed-2026-10-19.md)
- Cost: 16-core container ≈ $0.9/h; ≈ $0.00025 per searched turn; compile ≈80–125 s at start — [SPEED §Benchmark](docs/search-speed-2026-10-19.md)
- Serving: `lanes=16, depth=1, rollout_temperature=1., replies=64, determinize=True, parallel=True` in a process owning ≥16 cores; keep `XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'` (Eigen threading 1.12–1.32 s vs 0.95–1.02 s) — [SPEED §Serving notes](docs/search-speed-2026-10-19.md)

### Inferences
- The searcher is now cheap enough to serve (~1 s/turn) but the GPU path is a dead end for this engine shape; speedups beyond this need algorithmic changes (one-phase replies, skipping heads on value-only steps), one of which would require its own Elo check.

### Gaps
- No total dollar figure for the speed study is given in SPEED (only per-hour and per-turn rates).
- Host variance (0.95–1.77 s for the same code) is unresolved; no instance pinning was tested.

---

## Key question 3: Search distillation (Run 2): why it failed, label-noise statistics, effect on search-on-top

### Takeaway
One round of AlphaZero-style distillation of leak-free d=1 search into c3 (23.4M labelled decisions, ≈ $110)
produced seven students all within ±3 Elo of c3 (CIs ≈ ±13) on 1,000 games each, because 87% of search
overrides are within the searcher's own lane-score noise and no student predicts override labels better than c3
itself; worse, distilled networks lose more than half of their search gain (+40–45 vs +104), most likely because
the refit critic was trained toward an optimistic max-over-lanes value.

### Cited Findings
- Target and verdict: "Can the +104 Elo that leak-free d=1 turn search adds to c3... be baked into the raw network"; "**No.**... All seven fine-tuned checkpoints... score 0.495–0.503 against raw c3 over the same 500 fresh `proposed-v1` dev maps × both seats (1,000 games each): every one is within ±3 Elo of c3, with 95% CIs about ±13 Elo. So any raw gain is below ~+15 Elo, far short of the +40 target." — [DISTILL §Answer](docs/search-distillation-2026-10-19.md) (branch `claude/search-distill-1ae5b0`)
- Teacher: c3 with `TurnSearcher(lanes=16, depth=1, temperature=1, rollout_temperature=1, determinize=True)`, leak-free only; data from self-play on `proposed-v1` with the teacher in both seats, 24 × 32-core containers for 105 min; stored per executed sub-action: observation, legal masks, action label, all 16 lane scores and chosen lane, outcome; recorder reproduces searcher state bit for bit; training seeds 100000–147999; seed % 20 == 0 held out — [DISTILL §Method](docs/search-distillation-2026-10-19.md)
- Loss: CE(search action) + β·KL(student‖c3) + 0.5·(v − search value)²; value target = chosen lane's score; Adam, batch 4096, lr 1e-4 (200-step warm-up, cosine to 1e-5), one H100 at 32k samples/s per variant — [DISTILL §Method/Training](docs/search-distillation-2026-10-19.md)
- Label-noise statistics: "Search overrides greedy in 87% of plans, and the median override gain (chosen minus greedy lane score) is 0.023, about the median spread of the 16 lane scores (0.018)" — [DISTILL §Method/Training](docs/search-distillation-2026-10-19.md)
- Data: 16,652 games / 23.37M labelled decisions (15,812 train, 840 held out); 1,427 decisions per complete game, 51.7 turns; seat-1 score 0.595, 14.5% of games hit the cap; 87.3% of plans overrode greedy; held-out labels equal c3's greedy action 50.3% (100% in greedy plans, 44.5% in overriding plans); throughput 4.6 decisions per core-second (≈16.5k per core-hour, ≈530k per 32-core container-hour); map mix of decisions: archipelago 14%, continents 13%, classic 13%, ridge 12%, lakes 10%, highlands 10%, jungle 8%, pangea 8%, waterworld 7%, drylands 5% — [DISTILL §Results/Data](docs/search-distillation-2026-10-19.md)
- Raw students vs raw c3 (500 maps × 2 seats, seeds 380000–380499): kl0.1 uniform 397–405–198, 0.496, −3 [−16, +10] (KL to c3 0.012, entropy 1.97); kl0.1-g0.05 397–407–196, −3 [−16, +9]; kl0.1-g0.1 407–414–179, −2 [−16, +10]; kl0.03-g0.05 403–396–201, 0.503, +2 [−10, +15] (KL 0.109, entropy 1.76); conf0.1 412–405–183, +2 [−10, +15]; conf0.2 404–403–193, +0 [−11, +12]; pilot 405–402–193, +1 [−11, +13]; c3 entropy 2.02 — [DISTILL §Results/Raw](docs/search-distillation-2026-10-19.md)
- Held-out accuracy on override plans 0.444–0.447 vs c3's 0.445; held-out label CE barely moves (c3 1.952, best student 1.936); value loss against search value drops 0.0084 → 0.0028 for every student; per-type scores 0.43–0.55; "The pilot's +40 [+12, +69] on the 100 screening maps (380500–380599) did not hold on the 500 evaluation maps (+1 [−11, +13]). It was the best of four noisy screens." — [DISTILL §Results/Raw](docs/search-distillation-2026-10-19.md)
- Search on top (leak-free d=1 K=16, 100 maps × 2 seats, seeds 380000–380099, vs own raw policy): c3 111–53–36, 0.645, **+104 [+69, +141]**; kl0.03-g0.05 student 94–71–35, 0.557, +40 [+7, +74]; kl0.1 student 98–72–30, 0.565, +45 [+10, +80]; c3 minus student on same maps +64 [+18, +111] and +58 [+20, +100] — [DISTILL §Results/Search on top](docs/search-distillation-2026-10-19.md)
- Why: (1) "improvement operator is weak per decision and noisy... the max over 15 noisy scores beats greedy 87% of the time"; students disagree with c3's greedy on 8–18% of decisions where search kept it yet predict override labels no better; (2) "Turn-level selection is not decomposable"; (3) "The refit critic costs search... a max over 16 noisy one-sample estimates, so it is optimistic"; uniform student kept entropy and KL 0.012 and still lost most of the search gain; "A distillation meant to keep search working should leave the critic alone (value_coef = 0, or a value anchor to c3); this was not tested." — [DISTILL §Why it did not work](docs/search-distillation-2026-10-19.md)
- PPO continuation (step 4) not run, conditional on a positive raw result — [DISTILL §Answer](docs/search-distillation-2026-10-19.md)
- Cost: label generation ≈ 81; packing 0.2; training 4.1; pilots 2.5; raw evaluation 13.1; search arms 8.0; smokes/tests 1.2; **total ≈ $110** of the $200 budget; rates $0.0472/core-h + $0.008/GiB-h, H100 $3.95/h; "The Modal bill was not queried" — [DISTILL §Cost](docs/search-distillation-2026-10-19.md)
- Cross-reference: searched decisions run at 4.6 per core-second vs ~156 raw decisions per core-second in actors, "~34× fewer samples per dollar on 14×14... expect ~40×" at 18×18; "Campaign 3's 1.3B samples would become ~30M searched samples for the same money" — [HORIZON §Search in training](docs/long-horizon-credit-2026-10-20.md)

### Inferences
- The failure is consistent with the u3142 study's optimizer's-curse finding: the d=1 search gain is an aggregate of many near-noise choices, so its per-sub-action marginal is nearly c3's own T=1 distribution.
- The critic-refit explanation is the authors' stated "likely cause" and was not isolated by an ablation (value_coef = 0), so it is a hypothesis, not a confirmed mechanism.

### Gaps
- No arm with the critic frozen; no multi-round expert iteration; no soft targets over all lanes' first actions (the generator does not record them).
- Per-game statistics come from only 13.5k of 16.65k games whose container logs survived restarts.

---

## Key question 4: 18×18 campaign (Run 1): engineering, three runs, final eval, tech-baseline correction, turn-clock pilot, bottlenecks

### Takeaway
The 18×18 engineering (ruleset `bardur-1v1-v2`, grown c3, size-as-config) is complete and validated, but three
PPO self-play runs from grown c3 ($235 of ≈$270 spent, cap $300) and a turn-clock + KL-anchor pilot (~$40) never
produced a checkpoint beating its parent deterministically (best: 36% of all games, 47% of decisive); the
diagnosed defect (per-decision discount drowning the ±1 outcome over ~2,400 decisions) was fixed in the pilot
(critic value std 0.25–0.28, no drift) yet the policy stayed at parity, and the "c3 is tier-3 starved" premise
was retracted.

### Cited Findings

**Result and final evaluation (run 3 update 300, frozen engine, both seats)**
- "**Result: no stronger 18×18 bot.**... Three self-play runs from grown c3 ($235 of ~$270 spent; cap $300) did not produce a checkpoint that beats its parent when both play deterministically. The best (run 3, update 300) wins 47% of decisive deterministic games against it and 60% when both sides sample" — [L18 intro](docs/rl-18x18-2026-10-19.md)
- vs grown c3, 18×18 `proposed-v1`, deterministic (371000+128): 93–105–58, 36% [32, 41], 47% of decisive; all 10 types × 24 maps deterministic (373000+100k): 172–197–111, 36%, 47%; both sampled (371200+64): 51–34–43, 40% [33, 47], 60% of decisive; vs c3 on 14×14 `proposed-v1` deterministic (371500+128): 85–115–56, 33% [29, 38], 43% — [L18 intro table](docs/rl-18x18-2026-10-19.md); numbers confirmed in `docs/rl-18x18/l18-final-v3u300.json` (mix18 93/105/58, CI [0.316, 0.410]; mix14 85/115/56; mix18_sampled 51/34/43)
- Targets "≥65% vs grown c3 on 18×18 and a turn-cap rate ≤ ~20%. Neither was met"; self-play cap rate 31% — [L18 intro](docs/rl-18x18-2026-10-19.md)

**Tech-usage baseline correction**
- "c3 is not tier-3 starved. On 14×14 it researches ~13 of 26 technologies and 3.4 tier-3 techs per game; what it rarely reaches are the tier-3 economy techs (Construction 18% of games, Trade 16%, Philosophy 12%, Chivalry 11%)" — [L18 intro](docs/rl-18x18-2026-10-19.md)
- Step 0 baseline, c3 sampled self-play 14×14, 480 games (seeds 374000+100k): 115 cap (24%), mean 56 turns, 12.8 techs, 3.4 tier-3; cap rate by type from 8% (drylands, waterworld) to 40% (lakes) — [L18 §Step 0](docs/rl-18x18-2026-10-19.md)

**Engineering delivered**
- Ruleset v2 = v1 + 18×18 (324 tiles), round limit 150 on 18 (100 on 11/14), 9 ruins; v1 checkpoints still load (`COMPATIBLE_DIGESTS`); 20 Polyfish-oracle Large layouts match bitwise; 11/14 layouts unchanged; variant maps at 18 (capitals ≥ 8 apart, ~1.65× islands/villages) — [RULES2](docs/ruleset-v2.md), [L18 §What was built](docs/rl-18x18-2026-10-19.md)
- `network.grow`: grown c3 gives "identical choices, log-probs and values to c3 on 640 states"; two new pre-activation residual blocks with zero second kernel; `target_logits` reads only first 128 rows so one network runs on any board; actor vs learner log-probs at 18×18 agree to 4e-6 — [L18 §What was built](docs/rl-18x18-2026-10-19.md)
- Learner options found necessary: `advantage_floor`, `critic_warmup`; tests on Modal: test_rl_18x18 10, test_maps 7, test_mapgen 63, test_gpu 135, etc. all pass — [L18 §What was built](docs/rl-18x18-2026-10-19.md)
- Grown c3 smoke on 18×18 (480 games, seeds 372000+100k): no illegal action; 147 cap (31%), mean 88 turns (57% longer than 56), 15.2 techs, 4.1 tier-3 — [L18 §c3 on 18×18 before training](docs/rl-18x18-2026-10-19.md)
- Throughput: actors 2.0–3.4k samples/s per 32-core container (14×14 ~5k); learner 39k sample-epochs/s per H100 (16,384 minibatch OOMs at 45 GiB; 4,096 fits); 4×H100 ≈ 52k samples/s at 3 epochs ≈ 19 actor containers' worth — [L18 §c3 on 18×18 before training](docs/rl-18x18-2026-10-19.md)

**The three runs (parent grown c3, `proposed-v1` at 18×18, 4×H100, 16–22 actors, 262,144 batches, 3 epochs, critic bootstrap at cap, no outside bots)**
- Training-time win rate vs grown c3 (sampled, decisive share) per 100-update window, run 1 / run 2 / run 3 / (c3 on 14×14 vs its parent for reference): 0–99: 50/43/49 (57); 100–199: 46/49/54 (65); 200–299: 31/52/58 (68); 300–399: 16/53/60 (69); 400–499: 17/47/59 (73); 500–599: 13/41/56 (70); 600–699: —/37/53 (72); "Equal policies read 42–50% in the first windows, so ±5 points is noise" — [L18 §The three runs](docs/rl-18x18-2026-10-19.md)
- Deterministic ladder vs grown c3 (32 maps × 2): run 1 u360 10–43–11; run 2 u220 27–23–14; run 2 u720 7–41–16; run 3 u340 20–27–17; selection run 3 u300/u400/u500/u600 (370700+64): 52–48–28 / 46–58–24 / 45–53–30 / 45–57–26 — [L18 §The three runs](docs/rl-18x18-2026-10-19.md)
- **Run 1** (c3's settings: lr 2e-4, entropy 0.005, γ 0.995, 5% frozen parent): entropy 1.1 → 3.8, collapse within 300 updates (16%), stopped after 1 h; cause: raw advantage std 0.024–0.04 at 18×18 (14×14: 0.047–0.065) below the normalizer floor 0.05, halving the policy gradient; "loss was −0.018 ≈ −0.005 × entropy, so training was almost pure entropy maximization" — [L18 §The three runs](docs/rl-18x18-2026-10-19.md)
- **Run 2** (advantage floor 0.01, entropy 0.002): 54% of decisive at u220 then slide to 37%, stopped after 1.5 h; "~1,500 decisions per seat against ~785 on 14×14", 0.995^1500 ≈ 5e-4 (14×14: 0.02); critic flat (value std 0.15 vs 0.28) — [L18 §The three runs](docs/rl-18x18-2026-10-19.md)
- **Run 3** (γ 0.9974, 40-update critic warm-up, lr 1e-4, entropy 0.002, floor 0.01, 15% frozen parent, 10% own snapshots): critic recovered (value std 0.27); sampled play 49% → 60% by u300–400 then back to 53%; deterministic never past parity; stopped at u646 (1.0 h) — [L18 §The three runs](docs/rl-18x18-2026-10-19.md)
- "larger boards are not a drop-in extension for this PPO setup... what helped... bought a +10-point sampled gain, not the 65% target" — [L18 §The three runs](docs/rl-18x18-2026-10-19.md)

**Before/after (sampled self-play, same maps): c3 14×14 / grown c3 18×18 / run 3 u300 18×18**
- cap rate 24% / 31% / 31%; mean length 56 / 88 / 90; techs 12.8 / 15.2 / 15.0; tier-3 3.4 / 4.1 / 4.0; Trade 16% / 13% / 6%; Chivalry 11% / 13% / 7%; recruit mix Warrior 36%, Catapult 24%, Cloak 16% / Catapult 26%, Cloak 26%, Warrior 24% / Catapult 29%, Cloak 28%, Warrior 22%; "training did not move any of these"; ridge 62% and drylands 44% capped in run-3 self-play — [L18 §Turn cap, techs and units](docs/rl-18x18-2026-10-19.md)

**Budget (Modal billing, staging)**
- Step 0 1.94; engineering checks 0.96 (ledger 1.66); smokes 8.30; run 1 (1.05 h) 65.61; run 2 (1.5 h) 99.57; run 3 (1.0 h) 70.24; ladders/selection 7.36 (14.87); final eval 5.60 (16.93); **total billed 259.58, ledger 279.12**, "Projected total about $270 of the $300 cap"; runs billed ~28% above estimate because 18×18 actor processes hold ~2.2 GiB each (~70 GiB per container vs 32 GiB request) — [L18 §Budget](docs/rl-18x18-2026-10-19.md)

**Long-horizon diagnosis and turn-clock pilot (`l18-turnclock-v1`)**
- Measured: decisions per player per game 890 / ~550 (14×14) vs 2,400 / ~1,850 (18×18) mean/median; decisions per player-turn 13.7 (p90 21) vs 23.7 (p90 37); legal actions per decision median 50 / 80; 0.995^2400 ≈ 6e-6 vs 0.995^890 ≈ 0.01; per-turn horizon ~9 turns in an 88-turn game vs ~15 in a 56-turn game — [HORIZON §Our numbers](docs/long-horizon-credit-2026-10-20.md)
- "We are the outlier: a per-atomic-action discount in a game whose atomic-action count grows with army size" (AlphaZero/MuZero no discount; OpenAI Five horizon in game seconds; AlphaStar/DeepNash undiscounted; Turn-PPO per-turn advantages) — [HORIZON §What strong game systems do](docs/long-horizon-credit-2026-10-20.md)
- Pilot setup: run-3 settings plus turn clock γ_turn 0.97, λ_turn 0.92 applied only at own end-turn; KL anchor to grown c3 `anchor_coef` 0.05; 80 critic-only warm-up updates; 40 min, ~310 updates, ~80M samples, 4×H100, 16 actors (2 lost), learner-bound ~39k samples/s — [HORIZON §Pilot result](docs/long-horizon-credit-2026-10-20.md)
- Pilot training record vs frozen grown c3 (sampled, decisive) / value std / entropy: u0–49 48% / 0.22 / 1.8; u50–99 54% / 0.28 / 2.4; u100–149 50% / 0.26 / 2.6; u150–199 56% / 0.25 / 2.8; u200–249 55% / 0.25 / 2.8; u250–310 53% / 0.26 / 2.8; deterministic vs grown c3 (370800+48 × 2): u200 37–39–20 (49% of decisive), final 37–34–25 (52%) — [HORIZON §Pilot result](docs/long-horizon-credit-2026-10-20.md)
- Reading: critic fixed (value std 0.25–0.28 vs 0.15–0.27 earlier, c3's 0.28 on 14×14); anchor held drift (KL 0.008 → 0.019); "But the policy did not improve"; "discounting by decisions was a real defect... not the only one"; full $220 run "Not recommended" — [HORIZON §Pilot result](docs/long-horizon-credit-2026-10-20.md)
- Pilot cost ~$40 + ~$3 tests, ledger `runs/rl-18x18-v2` ($55 cap) — [HORIZON §Pilot result](docs/long-horizon-credit-2026-10-20.md)

**Bottlenecks study (measurements on the pilot; ≈ $2)**
- Wall time per update 7.51 s: GPU training 6.67 s (89%), bookkeeping 0.28 s, every-10th-update save 5.9 s (7.9 s with snapshot), amortised overhead 0.83 s (11%); `ready_wait_seconds` 0.00; batch age 10–11 s; inbox 128/128 full; policy lag mean 5.5–6.1 updates (max 10), ~900 stale samples dropped per update (0.35%); consumed ≈35k samples/s vs produced ≈36k/s; "The learner never waits... backpressure throttles the actors by roughly 4–8%" — [BOTTLE §1](docs/rl-18x18-bottlenecks-2026-10-20.md)
- First update 172 s (42 s learner compile + 130 s actor compile, 80–136 s per container); 2 of 16 actor containers placed 28 min late — [BOTTLE §1](docs/rl-18x18-bottlenecks-2026-10-20.md)
- Learner (one H100, 4,096 minibatch, 65,536 batch): anchor on 2.107 s/epoch (31.1k sample-epochs/s); anchor off 1.605 s (40.8k); of 6.67 s: PPO fwd+bwd 4.8 s (72%), anchor forward recomputed each epoch 1.5 s (22%), host→device copy ~0.3 s (5%); grown net 243 MFLOP/forward at 18×18 (147 without extra blocks; 89 at 14×14); ≈30 TFLOP/s per H100 ≈ 45% of fp32 CUDA-core peak, "tensor cores unused" — [BOTTLE §Learner](docs/rl-18x18-bottlenecks-2026-10-20.md)
- Actors (32 pinned procs × 24 games): 18×18 grown 2,561–2,669 steps/s per container, tick 288–300 ms (forward 105–129 ms = 36–43%, engine 169–179 ms = 57–60%); 14×14 c3 8,550 steps/s, 90 ms tick; structures `build_mask` 46.6 of 93.7 ms tick at 18×18 (~45%; 31% at 14×14) — [BOTTLE §Actors](docs/rl-18x18-bottlenecks-2026-10-20.md)
- Container scaling at 18×18: 32 vCPU/32 procs 2,669 steps/s = 83 per paid vCPU; 16/16 1,724 = 108; 8/8 1,110 = 139 ("1.66× the samples per actor dollar") — [BOTTLE §Actors scale badly](docs/rl-18x18-bottlenecks-2026-10-20.md)
- Cost structure: learner $17.1/h (33%), 16 actor containers $34.4/h (67%), total $51.5/h ≈ $0.40 per million samples (actors 0.27, learner 0.13); time budget of the 40-min run: first batch 7%, 80 warm-up updates 10 min (25%), ~11% bookkeeping — [BOTTLE §Cost structure](docs/rl-18x18-bottlenecks-2026-10-20.md)
- Levers (ranked, not applied): TF32/bf16 on learner (expected 1.5–2.5×), anchor log-probs once per batch (−1.0 s of 6.67), checkpoint/status off training thread (11%), cheaper critic warm-up, 8/16-vCPU containers (1.3–1.66×), build-mask proposal (−20–25% actor time), dropping the two grown blocks (−15–20% actor, −30% learner); combined ≈ $0.20 per million samples — [BOTTLE §2 Levers](docs/rl-18x18-bottlenecks-2026-10-20.md)
- Network size context: "4 residual blocks × 64 channels, 0.5M parameters, about the budget Jones (2021) found necessary for 9×9 Hex; KataGo's strength tracked network size" — [NEXT §What the three threads agree on](docs/next-run-and-strength-2026-10-20.md)

### Inferences
- Four 18×18 runs (three plus the pilot) with three different credit-assignment fixes all ended within ~25 Elo of the start (per DIAG), so credit assignment was necessary but not sufficient; the remaining candidates the authors name are shaping, curriculum, advantage filtering and network capacity.
- The 18×18 learner is learner-bound at fp32 and the fleet is memory-overbilled; the Stage-1 gate later measured 3.28× learner speedup at 18×18, larger than BOTTLE's 1.5–2.5× estimate.
- The selected run-3 u300's 60%-of-decisive sampled figure was selection-inflated (DIAG re-measured 53% on fresh seeds).

### Gaps
- No 18×18 run longer than 1.5 h; no curriculum (14→18) and no capacity change were tried.
- The per-seat decision count is reported inconsistently across docs (see Inconsistencies).

---

## Key question 5: Ablation program (Stage 0 diagnosis, Stage 1 precision gate, Stage 2 five arms)

### Takeaway
Stage 0 showed the 18×18 children are transitive and nearly flat (+24–28 ±19 Elo over grown c3, χ² 2.1/6 df, no
cycles: no learning signal, not cycling); Stage 1 passed TF32 convolutions + fp32 heads (mean k3 1.8e-7, max
|log r| 6e-3; learner epoch 1.92× faster at 14×14, 3.28× at 18×18); Stage 2 (one hour each from c3 on 14×14,
one seed per arm) found the omniscient critic the only arm that beats c3 (138–75–43, 54% [50.0, 58.2], 65% of
decisive, +14 over control, wins on all 10 map types, on 80M vs 113M samples) — borderline against the
pre-registered 58% bar — while continuing c3 unchanged, cap-score and filter+EMA were neutral and PBRS hurt.

### Cited Findings

**Pre-registration**
- Parent c3 u4854, 14×14 `proposed-v1`, 1×H100 + 12 × 16-vCPU actors (24 games/process), 3,600 s each, launched together; ledger cap $200; control = c3's recipe (lr 2e-4, entropy 0.005, 3 epochs, batch 262,144, minibatch 16,384, segment 32, γ 0.995, λ 0.99, per-decision clock, critic bootstrap at cap, 5% vs frozen c3, 15% vs own snapshots); arms: `abl-control` (seed 101), `abl-capscore` `cap_score_reward 0.5` (102), `abl-omni` separate critic on both seats' views with 60 critic-only warm-up updates (103), `abl-pbrs` zero-sum potential coef 0.3 over stars/units/territory/cities annealed to 0 at update 500 (104), `abl-filter` 1 epoch, top-25% |advantage|, EMA 0.999 deployed (105) — [ARMS](docs/ablation-v1/arms-2026-10-20.md)
- Pass rule: "≥ 58% of games vs c3 with the 95% CI above 50%, and ≥ 5 points above the control on the same maps. Differences under 5 points are noise (one seed per arm)"; final ladder = deployed `final.npz` vs c3 deterministic, 128 fresh `proposed-v1` maps × 2 seats (seeds 371300–371427) in one `select` call — [ARMS](docs/ablation-v1/arms-2026-10-20.md)
- Launch 13:28 PDT with `POLYTOPIA_MATMUL_PRECISION=DEFAULT POLYTOPIA_HEAD_PRECISION=HIGHEST`; Modal preempted learners of arms 2 and 3 at ~updates 13 and 24; relaunched unchanged as `abl-omni-2`, `abl-pbrs-2` at 13:50 PDT — [ARMS §Launch log](docs/ablation-v1/arms-2026-10-20.md)

**Stage 0 (DIAG, branch `claude/diag-18x18-roundrobin-bb5639`)**
- Round-robin of grown c3, r2u220, r3u300, pu200 (pilot u200), pfinal; 10 pairs × 128 games (seeds 371800+64, both seats), sampled play, frozen engine, cap 150; 1,280 games — [DIAG §Setup](docs/ablation-v1/diagnosis-18x18-2026-10-20.md)
- "every pair lands within 44–56%, a Bradley-Terry fit explains all ten pairs with χ² = 2.1 on 6 degrees of freedom, and there is no 3-cycle"; BT Elo (grown = 0): r3u300 +27.7 ± 19.5 (202-180-130 over 512), pu200 +24.5 ± 19.4, pfinal +23.9 ± 19.4, r2u220 +2.7 ± 19.4, grown 181-210-121; largest pair residual 2.3 pts, |z| ≤ 0.84 — [DIAG §1–2](docs/ablation-v1/diagnosis-18x18-2026-10-20.md)
- Pooled, three later children vs grown: 53.8% ± 1.5% (z = 2.6; 54.9% of decisive), ≈ +26 Elo; "+25 Elo is a quarter of the +100 Elo (65%) target"; "Run 3 u300's earlier 60% of decisive sampled games... reads here as 53% of decisive on fresh seeds... inflated by selection" — [DIAG §2](docs/ablation-v1/diagnosis-18x18-2026-10-20.md)
- Verdict: not cycling, not hidden progress, "**Flat everywhere (no learning signal)? Yes.**"; recommendation: skip PFSP pool, go to Stage 2 — [DIAG §3](docs/ablation-v1/diagnosis-18x18-2026-10-20.md)
- Side finding: "the first mover wins 69% of decisive games on 18×18 (664/964...)"; seat 1 66%, seat 2 28% of its decisive games — [DIAG §3](docs/ablation-v1/diagnosis-18x18-2026-10-20.md)
- 25% of games capped, mean 81 turns; cost billed $14.39 as of 14:15 PDT ("expect ≈ $15–16"), reserved $13; six of ten containers preempted and replayed (preempted pairs $1.34–1.89 vs $0.98–1.18) — [DIAG §1, §4](docs/ablation-v1/diagnosis-18x18-2026-10-20.md)

**Stage 1 (GATE, branch `claude/precision-gate-d7f0e8`)**
- Gate on 20,480 real actor rows per board (f16-rounded weights the actor ran); pass = mean k3 < 1e-4 and max |log r| < 0.02: fp32/fp32 k3 1.8e-12 / 2.1e-12, max 1.7e-5 / 1.9e-5; TF32/TF32 k3 4.8e-7 / 4.4e-7, max 9.0e-3 / 1.09e-2; **TF32/fp32 k3 1.8e-7 / 1.9e-7, max 6.1e-3 / 6.5e-3** (14 / 18); no row outside the PPO clip — [GATE §Gate](docs/ablation-v1/precision-gate-2026-10-20.md)
- Context: the f16 weight transport already gives k3 3.0e-7 / max 6.8e-3 in production fp32; TF32/fp32 against fp32 learner weights 2.9e-7 / 7.2e-3 — [GATE §Gate](docs/ablation-v1/precision-gate-2026-10-20.md)
- Learner throughput one H100 (sample-epochs/s): fp32 161,946 (14×14) / 40,427 (18×18); TF32/TF32 313,640 (1.94×) / 133,645 (3.31×); **TF32/fp32 311,179 (1.92×) / 132,526 (3.28×)**; heads ≈ 1% of epoch — [GATE §Learner throughput](docs/ablation-v1/precision-gate-2026-10-20.md)
- Actor fleet: 14×14 8-vCPU 234–241 steps/s per paid vCPU, 16-vCPU 236–243, 32-vCPU 216 (267 earlier) — "container size barely matters"; 18×18 identical configs differ up to 1.6× run to run, "cannot rank container sizes"; 16-vCPU chosen for 14×14 arms (~3.8k steps/s each) — [GATE §Actor fleet](docs/ablation-v1/precision-gate-2026-10-20.md)
- Caveats: rows are early/mid game (70 of 256 lanes finished a game at 14×14, 10 at 18×18); one checkpoint per board at epoch 0; cost estimate ≈ $2 ($12 reserved) — [GATE §Caveats, §Cost](docs/ablation-v1/precision-gate-2026-10-20.md)

**Stage 2 training-time record (sampled vs frozen c3, 5% of games; W–L of decisive)**
- control: 431 updates, 113M samples, first 100 updates 617–691 (47%), last 150 423–464 (48%), entropy 2.15, value std 0.279, self-play cap 23%; cap-score: 434 / 114M, 661–690 (49%), 543–598 (48%), 2.06, 0.278, 25%; **omniscient critic: 307 / 80M, 641–595 (52%), 758–457 (62%)**, 2.29, 0.278, 22%; PBRS: 469 / 123M, 568–625 (48%), 477–636 (43%), 2.18, 0.280, 20%; filter+EMA: 493 / 129M, 576–572 (50%), 481–447 (52%), 1.26 (metric over top-|A| quarter), 0.362, 29% — [READOUT §Training-time record](docs/ablation-v1/readout-2026-10-20.md)
- Omni climbed per 50-update window after warm-up: 52% → 55% → 58% → 63% → 66%; PBRS fell to 38–40% while the potential was strong and recovered toward 45% as it annealed — [READOUT §Training-time record](docs/ablation-v1/readout-2026-10-20.md)

**Stage 2 final deterministic ladder vs c3 (128 maps × 2 seats, seeds 371300+128)**
- control 102–116–38, 40% [36, 44], 47% of decisive, 52 turns; cap-score 103–114–39, 40% [36, 45], 47%, 51; filter+EMA 105–101–50, 41% [37, 45], 51%, 53 ("+4 over control, within noise"); **omniscient critic 138–75–43, 54% [50.0, 58.2], 65%, 50 — "borderline: +14 over control, 65% of decisive, wins on all 10 map types; 54% of all games is under the 58% bar and the CI touches 50%"**; PBRS 95–116–45, 37% [32, 41], 45%, 52 ("worse than control") — [READOUT §Final ladder](docs/ablation-v1/readout-2026-10-20.md); JSONs `docs/ablation-v1/abl-ladder-a.json` (control 102/116/38, CI [0.355, 0.441]) and `abl-ladder-b.json` (omni 138/75/43, CI [0.4999, 0.5820]) confirm
- Omni per type: ridge 12–1, continents 16–5, archipelago 14–6, highlands 20–11, waterworld 14–9, classic 25–20; recruit mix Warrior 34% vs c3's 43%, Defender 14%; 80M samples vs control's 113M because actors ship the opponent's view and run the critic forward (22k vs 35k samples/s) — [READOUT §Final ladder](docs/ablation-v1/readout-2026-10-20.md)
- Reading: (1) omniscient critic "is the one change that helps... The mechanism is the one the search study pointed at: fog is where c3's value estimates go wrong"; (2) "Continuing c3 with its own recipe does nothing (control 47% of decisive, deterministic). c3's plateau at u4854 was real at this compute"; (3) cap-score "changes nothing in one hour... only ~8% of training games reach it" at 14×14; (4) "Potential shaping hurt (43% sampled, 37% deterministic)... Devlin–Kudenko preserves equilibria in the limit, not the path"; (5) filter+EMA neutral, "makes the learner 8× cheaper per update. Worth keeping as a cost measure" — [READOUT §Reading](docs/ablation-v1/readout-2026-10-20.md)
- Recommendation: omniscient critic, longer (3 h), with the filter recipe, first on 14×14 from c3 then 18×18 with the turn clock; checkpoint archived at `/retained/checkpoints/distributed-final-v1/abl-omni-2-final/` (`final.npz`, `final-critic.npz`) — [READOUT §Reading](docs/ablation-v1/readout-2026-10-20.md)
- Cost (billed at 15:15 PDT / ledger bound): diag-roundrobin-18 16.97 / 16.97; precision-gate 1.35 / 12.00; abl-tests-1 0.00 / 0.60; abl-smoke-all 1.01 / 2.02; abl-control 19.22 / 19.47; abl-capscore 18.54 / 19.45; abl-omni 2.67 / 7.90; abl-pbrs 2.97 / 7.81; abl-filter 19.05 / 19.60; abl-omni-2 20.67 / 20.67; abl-pbrs-2 19.78 / 19.78; abl-ladder-a 1.51 / 2.79; abl-ladder-b 0.60 / 2.18; **total 124.34 / 151.24**, "final figure should land near $155", cap $200 — [READOUT §Cost](docs/ablation-v1/readout-2026-10-20.md)

### Inferences
- The omniscient-critic result is the strongest positive training-side finding of the two days, but it rests on one seed, one hour, 256 ladder games, and a CI whose lower bound is 49.99% (JSON), so it is suggestive rather than confirmed; its sample disadvantage (80M vs 113M) argues the effect is understated at equal samples.
- The control's 40% (47% of decisive) deterministic result against its own parent after one more hour means "more of the same" is slightly negative, which lowers the baseline all arms are compared against.
- The two fog-related findings (search leak share; omniscient critic) point at the same mechanism — value estimation under fog — as the dominant source of noise in the learning signal.

### Gaps
- One seed per arm; no replication of the omni result; no equal-sample comparison for arm 4 reported in the readout despite being pre-registered.
- PBRS was tested only with one coefficient (0.3) and one anneal schedule; the filter arm was tested only in combination (1 epoch + filter + EMA), so the three components are not separated.
- Stage 2's 18×18 follow-up (turn clock + omni) is recommended but not run.

---

## Key question 6: Earlier context (distributed campaigns 1–3, unit diversity, value-head probes, diverse-strategy check)

### Takeaway
Campaign 1 (2026-10-19, ~$330–360) produced u3142 (91.4% vs league96 on untouched maps) via a CPU-actor /
multi-GPU PPO league; campaign 2 (from u3142, control vs cap-as-draw) was paused after ~$40 with no committed
report; campaign 3 (4 h on the map mix) produced c3, which beats u3142 70% (84% of decisive) on the mix and halves
water-map stalls (30% → 20% capped); the value-head probe report shows u3142's critic is a good ranker from turn
10 (AUC 0.92–0.995) but a badly scaled probability, near-blind to the opponent, and ≈ a material count.

### Cited Findings

**Campaign 1 (DRL)**
- Delivered u3142: vs league96 untouched final maps (400032–400095, 64 × 2) **117–10–1, 91.4% [0.87, 0.96]**; development 122–6–0 (95.3%); vs previous diverse bot 62–2–0; vs E3 final 56–36–36 (61% of decisive); lineage league96 → A (300) → C (460) → D (3142), ~1.0 billion learner samples — [DRL intro](docs/distributed-rl-2026-10-19.md)
- Architecture: 32-core CPU actor containers (pinned processes, `lax.map` single-game engine, 24 games each, 32-decision GAE segments) → H100:2 learner, 262,144 batch, samples older than `max_lag` dropped; opponent mix 50% self-play, 15% snapshots, 35% frozen rivals; throughput ~35–50k learner samples/s vs ~1–6k before — [DRL §Architecture, §Campaign](docs/distributed-rl-2026-10-19.md)
- Ladder progression vs league96: A u300 75%; C u460 82%; D u380 91%; D u880 93%; D u1380–u1880 91%; vs D u880 (32 × 2): D u1880 28–11–25, E3 u280 36–15–13 ("Strong-vs-strong games often hit the engine turn cap") — [DRL §Campaign, §Caveats](docs/distributed-rl-2026-10-19.md)
- "What mattered: (1) throughput... (2) a larger step size with 3 epochs (D's jump from 82% to 91% within 40 minutes); (3) training against frozen league96 and the earlier diverse bots inside a historical-snapshot league"; unit KL / entropy bonus "did not, by themselves, increase deterministic unit variety against league96" — [DRL §Campaign](docs/distributed-rl-2026-10-19.md)
- Cost "About $330–360 in total" (estimate, not invoice) — [DRL §Cost](docs/distributed-rl-2026-10-19.md)

**Campaign 2 (no committed doc)**
- Only record found: the user's memory note: runs `c2-ctl-v2` (critic bootstrap at cap) and `c2-draw-v2` (`truncation_reward=0`), parent u3142, H100:4 + 15 actors, ~$44/h each, launched 09:57–10:00 PDT, paused 10:20 PDT by the user; ctl latest update-320, draw update-340; "~$40 spent; no arm laddered yet" — [memory note distributed-rl-campaign-2.md](/Users/K.Z./.claude/projects/-Users-K-Z--Desktop-projects-polytopia-bot/memory/distributed-rl-campaign-2.md) (not a committed report; no `docs/` file references "campaign 2")

**Campaign 3 (MAPS)**
- Nine new map types (`bardur-variants-v1`); u3142 playability: self-play hits the cap in 44–56% of water-map games (10% on classic); beats league96 on every type (69–93%) — [MAPS §Playability](docs/map-variants-2026-10-19.md)
- 4 h self-play from u3142 on `proposed-v1` (15% classic, 85% new), 4×H100, lr 2e-4, 3 epochs, 262k batch; arm A (cap bootstraps critic) vs arm B (cap = draw): B trailed (71% vs 85% decisive) and was stopped at 13:00; delivered c3 u4854, ~1.3B samples — [MAPS §Campaign 3](docs/map-variants-2026-10-19.md)
- c3 vs u3142 deterministic: mix (318000+128) 179–34–43, 70% [65, 75], 84% of decisive; classic 171–59–26, 67% [62, 72], 74%; all types equal 144–15–33, 75% [69, 80], 91%; self-play cap total 145/480 (30%) → 98/480 (20%); waterworld 27 → 7, archipelago 21 → 4, ridge 17 → 19; recruit mix Warrior 44%, Catapult 27%, Cloak 13%, Defender 6%, Scoutship 5%, Rammership 3%; entropy 1.6 → 2.4 — [MAPS §Campaign 3](docs/map-variants-2026-10-19.md)
- No dollar figure for campaign 3 appears in MAPS.

**Unit-diversity investigation (UNITS, 2026-10-18)**
- Low176 recruits Warriors in ~99% of summons; never a Rider/Archer/Defender/Catapult/Knight/Swordsman/Mind Bender in 256 audited games although Catapult was summonable in 238, Rider 227, Archer 223; cause: teacher score `20*demand + 20*threat − 2*unit_cost` always picks the cheapest unit; all strong checkpoints descend from imitation of it — [UNITS §Findings, §Why](docs/unit-diversity-investigation-2026-10-18.md)

**Diverse-strategy check (DIVERSE, 2026-10-19)**
- anchor112 and lr3e5u80 field mixed armies (Warriors 24–43% of recruits; non-Warrior unit in 89–96% of games) but lose to league96 (2–30 and 11–21); anchor112 has 1.93 cities at turn 8 vs 2.73 for league96; deterministic beats stochastic for both (19–13, 21–11); cost ≈ $0.29 — [DIVERSE §Bottom line, §Provenance](docs/diverse-strategy-check-2026-10-19.md)

**Value-head probes on u3142 (PROBES, branch `claude/ecstatic-kilby-e38baf`)**
- Setup: 512 games on dev seeds 331000–331127, frozen engine, 14×14 Drylands; 661k decisions logged; self-play 134–87 seat 1 with 35 capped games; vs league96 244–7–5 — [PROBES §Setup](docs/policy-probes-2026-10-19.md)
- Self-play AUC by turn bucket: 0–9 0.74 [0.69, 0.78]; 10–19 0.92; 20–34 0.97; 35–49 0.96; 50–74 0.995; 75–100 1.0; privileged material heuristic (true state incl. hidden units) 0.76 / 0.90 / 0.96 / 0.89 / 0.95 / 1.0; recalibration slope 8.9–24.6 ("logit divided by 9–25"); Brier raw → recalibrated 0.147 → 0.104 (turns 10–19), 0.104 → 0.059 (20–34) — [PROBES §1](docs/policy-probes-2026-10-19.md)
- "It does not know the opponent": vs league96 raw Brier 0.22 / 0.14 / 0.07 vs constant 0.025 / 0.029 / 0.073 for turns 0–9 / 10–19 / 20–34 — [PROBES §Verdict, §1](docs/policy-probes-2026-10-19.md)
- Linear fit on true-state material explains R² 0.79–0.86 of the value in turns 10–49; largest coefficients city and unit difference; tech difference ≈ nothing — [PROBES §1](docs/policy-probes-2026-10-19.md)
- Turn cap: in the 35 capped games "the value stays near 0 from start to end (turns ≥ 90: mean |v| 0.02)"; "v ≈ 0 therefore means 'even *or* headed for the cap'" — [PROBES §Verdict, §1](docs/policy-probes-2026-10-19.md)
- Per-move ΔV: bot's move rated above a uniform random legal move 63% (d = 0.23 [0.21, 0.26]), same-family 59%; Monte-Carlo (132 states, 12 rollouts/arm): bot − value-worst rollout gap 0.087 [0.048, 0.139], Pearson 0.62, Spearman 0.31; real only in turns 3–14 (0.23 [0.12, 0.35]); 60% of states identical in all 36 rollouts — [PROBES §2](docs/policy-probes-2026-10-19.md)
- Probes: every concept linearly decodable but "none of it more than from the raw inputs"; enemy star bank not inferred (R² ≤ 0.08); "From league96 to u3142 almost nothing changes (most differences ≤0.02)" — [PROBES §3](docs/policy-probes-2026-10-19.md)
- Occlusion: own units 4–8% of tiles carry 37–51% of |ΔV|; single tiles move value by ≤ 0.05–0.14; policy far more local (1–5 nats) — [PROBES §4](docs/policy-probes-2026-10-19.md)
- Cost ≈ $12.5 (CPU only, estimate) — [PROBES §Cost](docs/policy-probes-2026-10-19.md)

**Strength context (NEXT, Part 1)**
- Chain: league96 vs scripted teacher 64–0 (> +500); u3142 vs league96 117–10–1 (+410); c3 vs u3142 mix 179–34–43 (+150–290); c3 + leak-free robust search +117 [+87, +149]; leaky upper bound +200; "c3 sits roughly 1,000+ Elo above a hand-written heuristic. That number has no external meaning"; "We have never played a human, an independent bot, or the commercial game" — [NEXT §Part 1](docs/next-run-and-strength-2026-10-20.md)
- "c3 is the strongest agent in our environment by a wide margin... and is demonstrably sub-optimal. Nothing supports 'superhuman'"; exploitability untested — [NEXT §Part 1](docs/next-run-and-strength-2026-10-20.md)

### Inferences
- The probe findings (critic ≈ material count; blind to opponent and fog; v ≈ 0 ambiguous at the cap) are the mechanistic backdrop for every later failure: d=0 search's optimizer's curse, the cap-score arm's null, and the omniscient critic's gain.

### Gaps
- Campaign 2 has no committed report; whether its arms were ever laddered is unknown from the repo.
- Campaign 3's cost is not recorded in MAPS.

---

## Key question 7: Total money spent per line of work

### Takeaway
Across the two days the committed reports account for roughly $700–760 of Modal spend, dominated by the 18×18
campaign (≈ $260–280), the ablation program (≈ $124 billed, ≈ $151–155 expected), the search-distillation run
(≈ $110), and the two test-time-search studies (≈ $36 + ≈ $65, of which ≈ $35 was a failed sweep); most figures
are launcher estimates or partially settled billing, not invoices.

### Cited Findings
| Line of work | Spend | Basis | Source |
|---|---|---|---|
| Test-time search on u3142 (main + robust + leak-free) | ≈ $36.2 (24.6 + 5.07 + 6.45) | container wall-time estimate | [TTS-u3142 §Cost](docs/test-time-search-2026-10-19.md) |
| Test-time search on c3 | ≈ $65 (29.25 sweep + ≈ 35 failed sweep + ≈ 0.5) | estimate | [TTS-c3 §Cost](docs/test-time-search-c3-2026-10-19.md) |
| Fast parallel searcher | not stated (≈ $0.9/h per 16-core container; ≈ $0.00025/turn) | — | [SPEED §Benchmark](docs/search-speed-2026-10-19.md) |
| Search distillation | ≈ $110 of $200 budget | measured container seconds × list rates; "Modal bill was not queried" | [DISTILL §Cost](docs/search-distillation-2026-10-19.md) |
| 18×18 campaign (engineering, 3 runs, evals) | billed 259.58 / ledger 279.12; projected ≈ $270 of $300 cap; runs alone $235 | Modal billing per app, partly unsettled | [L18 §Budget](docs/rl-18x18-2026-10-19.md) |
| Turn-clock + KL-anchor pilot | ~$40 + ~$3 (ledger `runs/rl-18x18-v2`, $55 cap) | wall-time bound | [HORIZON §Pilot result](docs/long-horizon-credit-2026-10-20.md) |
| Bottlenecks diagnosis | ≈ $2 (not in a ledger) | estimate | [BOTTLE intro](docs/rl-18x18-bottlenecks-2026-10-20.md) |
| Ablation program (Stage 0–2) | billed 124.34 / ledger 151.24; "should land near $155"; cap $200 | Modal billing at 15:15 PDT | [READOUT §Cost](docs/ablation-v1/readout-2026-10-20.md) |
| · of which Stage 0 round-robin | $14.39 billed at 14:15 PDT (expect ≈ $15–16); table 16.97 | billing | [DIAG §4](docs/ablation-v1/diagnosis-18x18-2026-10-20.md), [READOUT §Cost](docs/ablation-v1/readout-2026-10-20.md) |
| · of which precision gate | ≈ $2 estimate; table 1.35 billed / 12.00 reserved | estimate | [GATE §Cost](docs/ablation-v1/precision-gate-2026-10-20.md) |
| Value-head probes (u3142) | ≈ $12.5 | estimate | [PROBES §Cost](docs/policy-probes-2026-10-19.md) |
| Diverse-strategy check | ≈ $0.29 (ledger bound $5.31) | estimate | [DIVERSE §Provenance](docs/diverse-strategy-check-2026-10-19.md) |
| Campaign 1 (distributed PPO → u3142) | ≈ $330–360 | app lifetimes × rates | [DRL §Cost](docs/distributed-rl-2026-10-19.md) |
| Campaign 2 (paused) | ~$40 | memory note only | [memory](/Users/K.Z./.claude/projects/-Users-K-Z--Desktop-projects-polytopia-bot/memory/distributed-rl-campaign-2.md) |
| Campaign 3 (map mix → c3) | not stated | — | [MAPS](docs/map-variants-2026-10-19.md) |

### Inferences
- Sum of the Oct 19–20 search/RL lines with stated figures (TTS-u3142 36.2 + TTS-c3 65 + DISTILL 110 + L18 ~270 + pilot ~43 + BOTTLE 2 + ablation ~155 + PROBES 12.5 + DIVERSE 0.3) ≈ $694 projected, ≈ $660 billed-so-far equivalents; adding campaign 1 (~$345) and campaign 2 (~$40) gives ≈ $1,080 for the whole Oct 18–20 RL effort excluding campaign 3 and the speed study.

### Gaps
- No invoice-level reconciliation exists for the search studies, distillation, probes, or campaign 1; campaign 3 and the speed study have no stated total.

---

## Key question 8: Cross-cutting analysis (levers, falsified premises, failure modes, confidence, open questions, inconsistencies)

### Takeaway
Only two levers moved strength with decent confidence: test-time d=1 search (+56 to +117 leak-free, CIs clear of
zero, 200–1,000 games) and the omniscient critic in training (+14 points over control, one seed, borderline CI).
Everything else tested on Oct 19–20 was null or negative; the recurring causes are a critic that is noisy under
fog and ambiguous at the turn cap, a per-decision discount that collapses on long games, and selection/noise
artefacts in small screens.

### Cited Findings

**Levers that moved strength (with magnitude and confidence)**
- Leak-free d=1 search: +56 [+21, +92] (u3142 mirror), +82 [+30, +139] (vs E3), +104 [+69, +141] (c3 mix), +26 [−9, +61] (c3 classic); leak-free robust +117 [+87, +149] (mix), +53 [+17, +89] (classic), +76 [+14, +141] (vs u3142); 200 games per arm — [TTS-u3142 §Addendum](docs/test-time-search-2026-10-19.md), [TTS-c3 §Results](docs/test-time-search-c3-2026-10-19.md)
- Omniscient critic (training): 54% [50.0, 58.2] vs control 40% [36, 44] on the same 256 games, 65% vs 47% of decisive; one seed, one hour — [READOUT §Final ladder](docs/ablation-v1/readout-2026-10-20.md)
- Earlier (Oct 19): campaign 3's map-mix training: c3 vs u3142 70% [65, 75] on 256 games — [MAPS §Campaign 3](docs/map-variants-2026-10-19.md)
- Engineering levers with measured effect but no strength claim: parallel search 8–10× (bitwise equal); TF32 convs 1.92× / 3.28× learner epoch (gate passed); 8-vCPU containers 1.66× samples per actor dollar at 18×18 (single benches; not reproducible per GATE) — [SPEED](docs/search-speed-2026-10-19.md), [GATE](docs/ablation-v1/precision-gate-2026-10-20.md), [BOTTLE §Actors scale badly](docs/rl-18x18-bottlenecks-2026-10-20.md)

**Levers that did not move strength**
- d=0 critic-only search at T=1: +14 to +28, all n.s.; after-END scoring +16 [−21, +53] — [TTS-u3142 §Results](docs/test-time-search-2026-10-19.md)
- Gumbel root search over the first sub-action: −413 — [TTS-u3142 §Results](docs/test-time-search-2026-10-19.md)
- Leak-free robust over leak-free d=1: +13 [−26, +54] (mix) — [TTS-c3 §Results/mix](docs/test-time-search-c3-2026-10-19.md)
- Search distillation: all students within ±3 Elo of c3 — [DISTILL §Answer](docs/search-distillation-2026-10-19.md)
- 18×18 self-play (4 configurations): +2.7 to +27.7 ± 19.4 Elo over grown c3 — [DIAG §2](docs/ablation-v1/diagnosis-18x18-2026-10-20.md)
- Continuing c3 with its own recipe: 40% vs c3 (47% of decisive) — [READOUT §Final ladder](docs/ablation-v1/readout-2026-10-20.md)
- Cap-score reward ±0.5: 40% (47%); filter + 1 epoch + EMA: 41% (51%, "+4 over control, within noise"); PBRS: 37% (45%), negative — [READOUT §Final ladder](docs/ablation-v1/readout-2026-10-20.md)
- Cap-as-draw (campaign 3 arm B): trailed 71% vs 85% decisive, stopped — [MAPS §Campaign 3](docs/map-variants-2026-10-19.md)

**Premises falsified**
- "Scoring time explains d=0's failure" — refuted by after-END scoring (+16 [−21, +53]) — [TTS-u3142 §Failure modes 2](docs/test-time-search-2026-10-19.md)
- "d=1 search gain is mostly real skill" — 40–55% (u3142) and up to ~75% (c3 classic) of the gain was hidden information — [TTS-u3142 §Addendum](docs/test-time-search-2026-10-19.md), [TTS-c3 §Answer](docs/test-time-search-c3-2026-10-19.md)
- "Search gain can be distilled in one round (+40 target)" — no raw gain; search gain on students drops to +40–45 — [DISTILL §Answer](docs/search-distillation-2026-10-19.md)
- "c3 is tier-3 starved / tech-limited" (18×18 premise) — c3 researches ~13 of 26 techs, 3.4 tier-3 per game — [L18 intro](docs/rl-18x18-2026-10-19.md)
- "18×18 is a drop-in extension of the 14×14 PPO recipe" — run 1 collapsed (advantage std below the 0.05 floor) — [L18 §The three runs](docs/rl-18x18-2026-10-19.md)
- "Fixing credit assignment (turn clock) will restore learning at 18×18" — critic fixed, policy at parity — [HORIZON §Pilot result](docs/long-horizon-credit-2026-10-20.md)
- "18×18 children are cycling / evaluation hides progress (PFSP needed)" — transitive, flat — [DIAG §3](docs/ablation-v1/diagnosis-18x18-2026-10-20.md)
- "Potential-based shaping is safe because it preserves equilibria" — hurt on the path (37%) — [READOUT §Reading 4](docs/ablation-v1/readout-2026-10-20.md)
- "TF32 will cost learner parity" — passes the gate with margin — [GATE verdict](docs/ablation-v1/precision-gate-2026-10-20.md)
- "A GPU will speed up search" — L4 14.7 s/turn vs ~1 s on 16 CPU cores — [SPEED intro](docs/search-speed-2026-10-19.md)
- "The pilot distillation screen (+40 on 100 maps) was real" — +1 on 500 maps, "best of four noisy screens" — [DISTILL §Results](docs/search-distillation-2026-10-19.md)
- "Run 3 u300 is 60% of decisive sampled" — 53% on fresh seeds; selection inflation — [DIAG §2](docs/ablation-v1/diagnosis-18x18-2026-10-20.md)

**Recurring failure modes**
- *Stalled/capped games*: 24% of c3 14×14 self-play, 31% at 18×18, ridge 62% / drylands 44% in run-3 self-play; 25% in the Stage-0 round-robin; ladders vs D u880 had 25/64 truncations; cap-score and cap-as-draw both failed to help — [L18 §Turn cap](docs/rl-18x18-2026-10-19.md), [DIAG §1](docs/ablation-v1/diagnosis-18x18-2026-10-20.md), [DRL §Campaign](docs/distributed-rl-2026-10-19.md), [READOUT](docs/ablation-v1/readout-2026-10-20.md), [MAPS](docs/map-variants-2026-10-19.md)
- *Critic calibration under fog / cap*: value ≈ material count, blind to hidden enemy stars and to the opponent's identity, v ≈ 0 at the cap; flat critic (value std 0.15) in 18×18 run 2; optimizer's curse at d=0; refit critic kills search gain after distillation; omniscient critic is the one arm that helps — [PROBES](docs/policy-probes-2026-10-19.md), [L18](docs/rl-18x18-2026-10-19.md), [TTS-u3142 §Failure modes](docs/test-time-search-2026-10-19.md), [DISTILL §Why](docs/search-distillation-2026-10-19.md), [READOUT §Reading 1](docs/ablation-v1/readout-2026-10-20.md)
- *Noisy learning signal*: advantage std 0.024–0.04 at 18×18 below the 0.05 floor; 0.995^2400 ≈ 6e-6; distillation labels within lane-score noise (median gain 0.023 vs spread 0.018); samples 5–6 policy versions stale, reused 3× — [L18 §Run 1](docs/rl-18x18-2026-10-19.md), [HORIZON §Our numbers](docs/long-horizon-credit-2026-10-20.md), [DISTILL §Method](docs/search-distillation-2026-10-19.md), [NEXT §What the three threads agree on](docs/next-run-and-strength-2026-10-20.md)
- *Hidden-information leak*: every pre-addendum search arm ran on the true state; leak-free costs −58 to −148 Elo depending on policy and map — [TTS-u3142 §Failure modes 6, §Addendum](docs/test-time-search-2026-10-19.md), [TTS-c3 §Answer](docs/test-time-search-c3-2026-10-19.md)
- *Infrastructure*: Modal preemptions (ablation learners, 6/10 diag containers, probe containers), a $35 failed c3-v1 sweep, 28% memory overbilling at 18×18, host variance up to 1.6–1.9× — [ARMS §Launch log](docs/ablation-v1/arms-2026-10-20.md), [DIAG §4](docs/ablation-v1/diagnosis-18x18-2026-10-20.md), [TTS-c3 §Cost](docs/test-time-search-c3-2026-10-19.md), [L18 §Budget](docs/rl-18x18-2026-10-19.md), [GATE §Actor fleet](docs/ablation-v1/precision-gate-2026-10-20.md)
- *Seat imbalance at 18×18*: first mover wins 69% of decisive games — [DIAG §3](docs/ablation-v1/diagnosis-18x18-2026-10-20.md)

**Confidence per finding (sample size / seeds)**
- High: search d=1 leaky and leak-free gains (200 games per arm, paired CIs clear of 0, reproduced across u3142/c3 and across sessions: c3 leak-free d=1 +104 reproduced on a fresh block in DISTILL); parallel-search equivalence (bitwise, 84 turns); TF32 gate (20,480 rows, two boards); distillation null (1,000 games per student, 7 students); 18×18 plateau (1,280-game round-robin + four runs).
- Medium: hidden-information share (single paired comparison per setting; belief prior crude); omniscient critic (+14, one seed, CI lower bound 49.99%); PBRS harm (one coefficient/schedule, one seed); c3's cap-rate reduction (24-map blocks per type).
- Low: per-map-type orderings (14–30 games); 18×18 fleet-shape cost advantage (single benches, 1.6× run-to-run variance); any absolute Elo/strength claim ("no external meaning", per NEXT).

**Open questions the data leaves**
- Does the omniscient critic clear 58% with 3 h, a second seed, and equal samples? Does it transfer to 18×18 with the turn clock? — [READOUT §Reading](docs/ablation-v1/readout-2026-10-20.md)
- Would a better belief prior (imagined enemy units in fog, variant-aware fog fill) recover part of the −58 to −148 leak-free penalty? — [TTS-u3142 §Addendum](docs/test-time-search-2026-10-19.md), [TTS-c3 §Caveats](docs/test-time-search-c3-2026-10-19.md)
- Does distillation work with the critic frozen, soft lane targets, or several rounds? — [DISTILL §Why](docs/search-distillation-2026-10-19.md)
- Is network capacity (0.5M params, 4 × 64) the 18×18 ceiling? Would a 14→18 curriculum carry skill? — [NEXT §Stage 2](docs/next-run-and-strength-2026-10-20.md), [HORIZON §Candidates](docs/long-horizon-credit-2026-10-20.md)
- Can an exploiter beat frozen c3 (> 60% would end "superhuman" talk)? No human or independent-bot games exist — [NEXT §Part 1](docs/next-run-and-strength-2026-10-20.md)
- Is the winner-speed bonus (uncommitted handoff `docs/winner-speed-bonus-handoff-2026-10-20.md`, `R_win = 1 + b(1 − T/cap)`, b = 0.3) the cap fix that cap-score and cap-as-draw were not? Proposed, not run.

**Inconsistencies between docs (flagged)**
- Stage 0 cost: DIAG "Billed: $14.39... expect ≈ $15–16"; READOUT text "Billed $14.4"; READOUT cost table "diag-roundrobin-18 16.97 / 16.97"; NEXT planned "≈ $8" — [DIAG §4](docs/ablation-v1/diagnosis-18x18-2026-10-20.md), [READOUT §Stage 0, §Cost](docs/ablation-v1/readout-2026-10-20.md), [NEXT §Stage 0](docs/next-run-and-strength-2026-10-20.md)
- Precision-gate cost: READOUT text "Billed ≈ $2" vs READOUT table billed 1.35 / ledger bound 12.00; GATE "≈ $2" estimate with $12 reserved — [READOUT](docs/ablation-v1/readout-2026-10-20.md), [GATE §Cost](docs/ablation-v1/precision-gate-2026-10-20.md)
- GATE 18×18 speedup: verdict says "3.27× faster at 18×18", its table and READOUT say 3.28× — [GATE verdict vs §Learner throughput](docs/ablation-v1/precision-gate-2026-10-20.md)
- Decisions per seat/game: L18 "~1,500 decisions per seat against ~785 on 14×14"; HORIZON measured 2,400 / ~1,850 (18×18) and 890 / ~550 (14×14) mean/median; PROBES "~650 decisions per seat" for u3142 games; DISTILL 1,427 decisions per complete game (c3 searched self-play) — different bots, map sets and statistics, but the L18 figure is not reconciled with HORIZON's — [L18 §Run 2](docs/rl-18x18-2026-10-19.md), [HORIZON §Our numbers](docs/long-horizon-credit-2026-10-20.md), [PROBES §Verdict](docs/policy-probes-2026-10-19.md), [DISTILL §Data](docs/search-distillation-2026-10-19.md)
- Pilot learner rate: HORIZON "~39k samples/s with the anchor's extra forward pass" vs BOTTLE "262,144 / 7.51 s ≈ 35k/s" consumed for the same run — [HORIZON §Pilot result](docs/long-horizon-credit-2026-10-20.md), [BOTTLE §1](docs/rl-18x18-bottlenecks-2026-10-20.md)
- 14×14 32-vCPU actor throughput: BOTTLE 8,550 steps/s vs GATE 6,915 "today" (GATE attributes it to host variance) — [BOTTLE §Actors](docs/rl-18x18-bottlenecks-2026-10-20.md), [GATE §Actor fleet](docs/ablation-v1/precision-gate-2026-10-20.md)
- Same arm, different lane-step counts: k16 d=1 greedy rollout vs E3 listed as 570 lane steps/turn in the vs-u3142 table context and 618 in the robust table ("v2") — [TTS-u3142 §Results](docs/test-time-search-2026-10-19.md)
- Plans per leak-free turn: 2.2–2.3 (u3142 addendum), ≈1.9–2.2 (c3), 2.5 and 2.8 (SPEED, different position sets) — [TTS-u3142](docs/test-time-search-2026-10-19.md), [TTS-c3 §Setup](docs/test-time-search-c3-2026-10-19.md), [SPEED](docs/search-speed-2026-10-19.md)
- Omni CI: READOUT prints "[50.0, 58.2]" while `abl-ladder-b.json` has ci95 [0.49990234375, 0.58203125] — the lower bound is marginally below 50%, consistent with "the CI touches 50%" — [READOUT §Final ladder](docs/ablation-v1/readout-2026-10-20.md)
- 18×18 spend: L18 intro "$235 of ~$270 spent; cap $300" refers to the three runs (65.61 + 99.57 + 70.24 = 235.42), while the table totals 259.58 billed / 279.12 ledger — consistent once read that way, but the intro wording could be read as the total — [L18 intro, §Budget](docs/rl-18x18-2026-10-19.md)

### Inferences
- The one consistent through-line is value estimation under partial observability: the two positive levers both bypass it (search looks ahead and, when leaky, sees through fog; the omniscient critic sees both views at training time), while the null/negative levers all operate through the same noisy critic.
- The ablation's control result (c3 continued is slightly behind c3) implies that further self-play compute with the existing recipe has zero or negative marginal value on 14×14, which strengthens the case for recipe changes over longer runs.

### Gaps
- No experiment isolates the belief-prior quality from the leak itself; no exploiter, human, or external-bot games; no multi-seed replication of any training-side result; campaign 2 and the speed study have no committed cost; campaign 3 has no cost at all.
