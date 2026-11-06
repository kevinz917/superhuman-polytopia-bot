# Why the pooled/history trunk did not beat hero-1v1 (farsight-v2 + Arm B, Nov 1-3 2026)

Runs covered: `farsight-v2-1v1` (obs v3 + trunk pooling layout 2, loss-at-cap, Daytona, Nov 1-2),
`farsight-v2-capfix` = Arm B (same network, score-at-cap, two 12-h segments, Nov 2-3), compared with
`hero-1v1` + `hero-1v1-cont1h` (4x128 plain trunk, obs v2, score-at-cap, Modal, Oct 26-27).
Ledgers, keeper logs, eval histories and the hand-off narrative are archived in
`docs/run-archive/farsight-v2-capfix-2026-11-03/`; the shareable write-up is `writeup.html` in that folder.

## Headline results

Final 128-game ladders (win share of all games; caps count as non-wins):

| Reference | Arm B u28957 (7.6 B samples) | farsight-v2 u9779 (2.6 B) |
|---|---|---|
| champion (2x64, obs v1) | 56.2% (72-47-9) | 21% |
| hero-1v1 final u14871 (3.9 B) | 47.7% (61-56-11) | 18% |
| hero-1v1-cont1h u11500 (~6.9 B) | 39.1% (50-73-5) | - |
| hero-1v1-cont1h final | 35.9% | - |
| bighero u10000 (loss-at-cap control) | 85.2% | 80% |
| farsight-v2 final | 45.3% (58-39-31 capped) | - |
| own u13014 (segment-1 end) | 68.8% | - |
| c3 | 84.4% | 56% |

Spend: farsight-v2 ~$483, Arm B ~$780 (two segments), series ~$1,260.

## 1. Per update, Arm B and hero-1v1 were the same bot until update ~13,000

Both runs use the same batch (~262k samples/update), so update counts are comparable.

| Updates | hero-1v1 vs champion | Arm B vs champion |
|---|---|---|
| ~5,700-6,200 | 25% | 23% |
| ~12,650-13,000 | 47% | 46% |
| ~14,850 (hero-1v1 final) | 52% | - |
| ~26,500 (hero-1v1 continuation end) | 68% | - |
| ~29,000 (Arm B end) | - | 56% |

The pooling blocks and history planes neither helped nor hurt learning per sample. The gap opened in
the second segment.

## 2. The second segment differed in the optimizer, not the architecture

hero-1v1's continuation started with a fresh critic and a learning rate decaying 3e-4 -> 1e-4 over the
whole segment: 52% -> 68% in 11,650 updates. Arm B's segment 2 warm-restarted at a flat 2e-4 and sat
for ~6,000 updates with no gain against its own pool (51-52.7% vs 53-54.5% in segment 1) and 42% vs
the champion at two ladder rounds. Only after the switch to a 1.5e-4 -> 4e-5 anneal did it climb
(58% at u24460). About 40% of segment 2 learned nothing; a straight-line extrapolation with the anneal
from u14000 lands near hero-1v1's 68%. This is the largest single explanation for the final gap.

## 3. The additions cost 2-3x throughput, which is their real price

Obs v3 encoding ran ~10k samples/s per PRO 6000 actor box vs ~29.5k for obs v2 on the same hosts
(Oct 28). The bigger views also halved the learner's Python intake wall (~120k/s -> ~47k/s; a fresh
learner process does 90-105k/s for a while). hero-1v1 reached 3.9 B samples in 8 h; farsight-v2
reached 2.6 B in 12 h. At equal samples the architectures tie, so at equal dollars the plain trunk wins.

## 4. Why the additions added nothing per sample

- Global pooling (KataGo) pays off when the policy needs whole-board facts the convolution cannot
  reach. On 14x14 maps four 3x3 blocks already see most of the board; the scalar gates settled near
  +-0.2 and the pool-probe showed bias/conv ratios of ~0.4, i.e. a small perturbation, not a new
  capability (farsight's layout 1, with ratios of 4-10, was actively harmful and collapsed).
- History planes give memory of enemy positions under fog. The skill axis that separates these bots is
  macro: city growth, tech order, when to field knights. hero-1v1 out-grows Arm B at turn 16 (mean
  city level 2.68 vs 2.49) with no history input. Perception was not the bottleneck.
- Parameter counts are near identical (2.04 M vs 1.97 M): no capacity to harvest.

## 5. The design looked strong because its baseline was broken

farsight-v2's 80-86% vs bighero at equal updates was against a loss-at-cap run that never learned
mid-game growth (docs: bighero deficiency, Nov 2). Any bot that grows beats bighero. Against the
champion farsight-v2 was at 21%, against hero-1v1 at 18%. The Arm B gate confirmed the rest: changing
the cap rule on this architecture lost the head-to-head to its own loss-at-cap sibling at u6200
(22% vs v2_u5140) and reached parity only at u13000. Score-at-cap bought generality and fewer capped
games, not a faster learner.

## 6. Why hero-1v1 is strong

- Score-at-cap from update zero made growth a path to +1 from the first long games; it built the
  economy loop early and never lost it.
- It found a strategic niche the champion lineage lacks: Riding -> Chivalry knights (10% of its army vs
  the champion's 3%; 87% win rate in knight games).
- Cheap observations: more games per hour, and this recipe converts games into strength.
- Its continuation was run correctly: fresh critic, annealed LR, no flat warm restart.
- It is probably under-measured: trained on the pre-Oct-31 engine revision, so slightly
  off-distribution on today's ruleset, and its last checkpoint (cont1h u17840, on the Daytona volume at
  /results/distributed/hero-1v1-cont1h/latest.npz) has never been laddered.

## Confounds

One seed per arm. 128-game ladders carry about +-9 points. hero-1v1 trained on Modal H100s and an older
engine revision, so this is a recipe comparison, not a controlled ablation.

## The clean test (not run)

Widen hero-1v1's u17840 weights to obs v3 + pooling with the existing `widen_obs.py --block-pool`
(verified output-identical on bighero's checkpoints) and continue both the widened and the plain
versions under one identical anneal for equal samples. That isolates the architecture from the
optimizer, throughput and engine differences above. Cheaper first step: ladder u17840 alone (~$25, 1 h)
to learn whether hero-1v1 was still climbing past 68%.

## Operational lessons recorded in this series

- Never warm-restart a decayed segment at full LR; resume with an anneal.
- `stream_views` must be off on 4-GPU PRO 6000 learners (page-streaming deadlock after update 1).
- Daytona: org cap 250 vCPU; GPU quota drops to 0 when GPU credits run out (spot "evictions" that are
  really refused creates); hub sandbox TTL expires; keeper hardened accordingly (retries, round
  deferral, fleet teardown before the final ladder, per-iteration try/except, state-file reset on launch).
- `runs/` is git-ignored; copy ledgers into `docs/run-archive/` (a folder literally named runs/ is git-ignored) when a series ends or they are lost with the
  worktree (this happened once on Nov 2).
