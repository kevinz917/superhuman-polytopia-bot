# Campaign log, 2026-10-19 → 2026-10-20

One page per line of work from the two days, so the sessions that produced them can be closed. Every number
is from the committed report named in each section; branches are listed so the code can be found. Nothing is
on `main`; `claude/rl-c5-base` (this branch) is the union of the 18×18/ablation branch and the omni-search eval,
and is the base for the next campaign.

## Scoreboard

| Bot | vs raw c3 (deterministic, both seats, 14×14 proposed-v1 mix) | Where |
|---|---|---|
| c3 u4854 (campaign 3) | reference | `/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz` |
| c3 + leak-free d=1 search | +104 [+69, +141] (seeds 366000+), +70 [+31, +111] (386000+) | search in `rl/search.py`, `determinize=True` |
| **OMNI** = c3 + 1 h omniscient-critic training | **+125 [+92, +160]**, 112–43–45 | `/retained/checkpoints/distributed-final-v1/abl-omni-2-final/final.npz` (+ `final-critic.npz`) |
| **OMNI + leak-free d=1 search** | **+210 [+166, +260]**, 140–32–28; +139 [+102, +179] vs raw OMNI | best bot today, ~1.2 s/turn on an M-series core |

Spend across the two days ≈ $700: search studies ≈ $100, fast searcher small, distillation ≈ $110, 18×18 ≈ $260,
turn-clock pilot ≈ $43, ablation ≈ $155, omni-search eval $12.7.

## 1. Test-time turn search (branches `claude/objective-tesla-f8f1f3`, `claude/sharp-goldstine-e442a1`, `claude/search-c3-u4854`)

Reports: `docs/test-time-search-2026-10-19.md` (u3142), `docs/test-time-search-c3-2026-10-19.md` (c3).

- d=1 turn search (K=16 sampled candidate turns, opponent reply played out, critic at the next own turn): +125
  [+87, +168] on the u3142 mirror, +108 (classic) / +162 (mix) on c3, leaky. d=0 critic-only is flat (optimizer's
  curse); Gumbel root search over first sub-actions collapsed (−413).
- Leak-free (determinized fog, `rl/belief.py`): +56 (u3142 mirror), +82 (vs E3), +26 n.s. (c3 classic),
  **+104 [+69, +141] (c3 mix)**. Hidden information was 40–75% of the leaky gain.
- Robust (64 averaged replies, sequential halving): +172/+200 leaky, but only +13 [−26, +54] over plain leak-free
  d=1 on the mix at 2.5× compute. Averaging replies mostly helps when the replies are real.
- Compute: ~2 s/turn leaky d=1, ~4 s leak-free, ~10.5 s leak-free robust on one Modal core.
- Value-head probes (`claude/ecstatic-kilby-e38baf`, `docs/policy-probes-2026-10-19.md`): u3142's critic ≈ material
  count, good ranker from turn 10 (AUC 0.92–0.995), blind to the enemy star bank and to which opponent it faces,
  v ≈ 0 in capped games.

## 2. Fast parallel searcher (branch `claude/fast-search`, pushed)

Report `docs/search-speed-2026-10-19.md`. `TurnSearcher(parallel=True)`: identical plans bit-for-bit (32 plans, 84
turns), 0.95–1.2 s/turn mean on 16 Modal cores vs 9.4–10.7 s lockstep; >16 cores no help (halving rounds are
sequential); L4 GPU 14.7 s/turn (worse than one CPU core). Cuts latency, not total compute: for evals and data
generation run one game per pinned core instead.

## 3. Search distillation (branch `claude/search-distill-1ae5b0`)

Report `docs/search-distillation-2026-10-19.md`. 16,652 self-play games with both seats searching (leak-free d=1),
23.4M labelled decisions (training seeds 100000–147999), seven students (CE on searched turn + KL to c3 + value
loss). **All within ±3 Elo of c3** over 1,000 games each on 500 maps (seeds 380000–380499). Search overrode c3's
own turn in 87% of plans by a median 0.023, about the lane-score spread (0.018): labels ≈ random samples of c3. No
student predicted the labels better than c3 (44.5%). Search on top of students only +40–45 vs +104 on c3 (authors
blame the refit critic; untested → arm E of the next campaign). Two checkpoints under
`/retained/checkpoints/search-distill-v1/`, neither stronger than c3. ≈ $110.

## 4. 18×18 campaign (branch `claude/rl-18x18-a4c945`)

Reports `docs/rl-18x18-2026-10-19.md`, `docs/rl-18x18-bottlenecks-2026-10-20.md`, `docs/long-horizon-credit-2026-10-20.md`,
`docs/ruleset-v2.md`.

- Engineering delivered and validated: ruleset `bardur-1v1-v2` (18×18 Large, round limit 150; 11/14 bitwise
  unchanged, 20 oracle layouts match), variant maps scaled to 18×18, `network.grow` (c3 + 2 zero-init pre-activation
  residual blocks, identical outputs on 640 states; `parameter_embedding` needed no remap, it indexes parameter ids
  not tiles), size as config end to end, actor/learner log-prob check 4e-6, arena tech tracking, budget cap in
  `rl/budget.py` + `scripts/ledger_spend.py`.
- Three self-play runs from grown c3: run 1 collapsed (advantage std below the 0.05 floor → entropy maximisation),
  run 2 flat critic (0.995^1500), run 3 (γ 0.9974 + critic warm-up) peaked ~60% sampled then drifted. Final: run-3
  u300 vs grown c3 **93–105–58 (36%, 47% of decisive)**, 33% on 14×14. Turn-clock + KL-anchor pilot ($55): critic
  fixed (value std 0.25–0.28), policy at parity (37–34–25). Grown c3 remains the 18×18 policy
  (`/retained/checkpoints/distributed-final-v1/c3-u4854-grown18x2/final.npz`).
- Premise corrected: c3 is not tier-3 starved (~13 of 26 techs, 3.4 tier-3 per game); it rarely reaches the tier-3
  economy techs (Construction 18%, Trade 16%, Philosophy 12%, Chivalry 11%).
- Bottlenecks: learner 89% of a 7.5 s update in fp32 (tensor cores idle); actors 67% of $/h; 8-vCPU containers
  1.66× samples per actor-$ at 18×18; first mover wins 69% of decisive 18×18 games (ladders must be seat-balanced).
- ≈ $260 of $300.

## 5. Ablation program (branches `claude/rl-18x18-a4c945`, `claude/diag-18x18-roundrobin-bb5639`, `claude/precision-gate-d7f0e8`)

Plan `docs/next-run-and-strength-2026-10-20.md`; pre-registration `docs/ablation-v1/arms-2026-10-20.md`; readout
`docs/ablation-v1/readout-2026-10-20.md`; side reports `docs/ablation-v1/diagnosis-18x18-2026-10-20.md`,
`docs/ablation-v1/precision-gate-2026-10-20.md`.

- Stage 0: round-robin of grown c3 and four plateaued 18×18 children, 10 pairs × 128 games: every pair 44–56%, one
  transitive ordering (χ² 2.1 on 6 df), no cycles; later children +24–28 ± 19 Elo over grown c3. **No learning
  signal, not cycling** → leagues/PFSP are not the remedy for that plateau.
- Stage 1: TF32 convolutions + fp32 heads pass the importance-ratio gate (max |log r| 6e-3); learner epoch 1.92×
  faster at 14×14, 3.28× at 18×18. Setting: `POLYTOPIA_MATMUL_PRECISION=DEFAULT POLYTOPIA_HEAD_PRECISION=HIGHEST`.
- Stage 2, five 1-h arms from c3 on 14×14 (1×H100 + 12 × 16-vCPU actors ≈ $16.6/h each), final ladder deterministic
  vs c3 on 128 fresh maps × 2 seats (seeds 371300+):

  | arm | W–L–cap | all games | decisive |
  |---|---|---|---|
  | control (c3 recipe continued) | 102–116–38 | 40% | 47% |
  | score at the cap | 103–114–39 | 40% | 47% |
  | 1 epoch + advantage filter + EMA | 105–101–50 | 41% | 51% |
  | **omniscient critic** (`omniscient_critic`, 60 critic-only warm-up updates) | **138–75–43** | **54% [50.0, 58.2]** | **65%** |
  | PBRS shaping, annealed | 95–116–45 | 37% | 45% |

  The omni arm won on all 10 map types, on 80M samples vs the control's 113M (actors ship the opponent's view and
  run the critic forward: 22k vs 35k samples/s), with its curve still rising (52 → 55 → 58 → 63 → 66% per window).
  Checkpoint `/retained/checkpoints/distributed-final-v1/abl-omni-2-final/`. ≈ $155 of $200.

## 6. Omni + search eval (branch `claude/omni-search-eval-ba2c50`)

Report `docs/omni-search-eval-2026-10-20.md`; 6 arms × 200 games on seeds 386000–386099, $12.7. Raw OMNI +125 over
c3; OMNI + leak-free d=1 +210 vs c3 and +139 in the mirror (search gain undiminished); c3 + search loses to raw
OMNI (−23 [−56, +10]). OMNI's stale value head scores lanes as well as c3's (+205 vs +210), so no scorer swap is
needed. New `TurnSearcher(value_params=)` option (policy and scorer can differ; plain path byte-identical).

## 7. Literature review and proposals

`reports/Polytopia RL next experiments.md` (main checkout, with notes under `research_notes/`). Grounding:
AlphaStar's value function used the opponent's observations during training only (best-response 22% → 82%, its
largest RL ablation; also V-trace 49% → +TD(λ) 73% → +UPGO 82%); unbiased asymmetric critic = hidden state +
own history (Baisero & Amato 2022), which is the form implemented; hard "search-chosen action" labels are the
known-bad distillation target (ExIt +50 ± 13 Elo for soft targets; Gumbel MuZero completed-Q targets); the
closest genre result (Generals.io 2026) used sparse win/loss, γ = 1, cap scored 0, EMA policy, no shaping;
KataGo: global pooling 1.6×, aux heads 1.65×; a 0.5M-parameter CNN is below compute-optimal.

## Falsified this week (do not re-run)

| Premise | Result |
|---|---|
| Search's gain is mostly real skill | 40–75% was hidden information |
| One round of hard-label distillation bakes in search | +0 Elo, seven students |
| c3 is tech-starved; bigger boards unlock tech | c3 researches 13/26; 18×18 gave no stronger bot |
| Credit assignment alone fixes 18×18 | turn clock fixed the critic, policy at parity |
| 18×18 children cycle (need a league) | transitive, flat |
| Reward tweaks finish games / add strength | cap-as-draw lost, cap-score neutral, PBRS hurt |
| Continuing c3 with its recipe helps | control 40% after 1 h |
| Search inside training is affordable | 34–40× fewer samples per dollar |

## Next campaign (2026-10-20 overnight, $500 cap, approved)

A/B omni critic 6 h × 2 seeds (14×14), C omni + UPGO 3 h, D 18×18 omni + turn clock 3 h, E frozen-critic
distillation re-fit, F shared ladder + search on the winner. One session per line, all from `claude/rl-c5-base`.
