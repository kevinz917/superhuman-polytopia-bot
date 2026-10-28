# Distilling leak-free turn search into c3's raw policy (2026-10-19)

Can the +104 Elo that leak-free d=1 turn search adds to c3 (docs/test-time-search-c3-2026-10-19.md) be baked
into the raw network, AlphaZero-style, so the bot is stronger with no search at play time?

## Answer

- **No.** One round of distillation on 23.4M search-labelled decisions leaves the raw network exactly as strong
  as c3. All seven fine-tuned checkpoints (six variants plus the pilot) score 0.495–0.503 against raw c3 over the same 500 fresh `proposed-v1`
  dev maps × both seats (1,000 games each): every one is within ±3 Elo of c3, with 95% CIs about ±13 Elo. So
  any raw gain is below ~+15 Elo, far short of the +40 target.
- **The labels are mostly noise the network cannot predict.** Search overrides the greedy turn in 87% of plans,
  usually by less than the spread of its own lane scores, and the chosen lane is a T=1 sample of c3. On
  held-out games no student predicts the overriding labels better than c3 itself (44.4–44.7% vs 44.5% joint
  accuracy). Students that moved further from c3 only lost agreement with c3's argmax, which did not cost
  strength either. Down-weighting low-gain overrides, or keeping only overrides that beat greedy by ≥0.1 or ≥0.2,
  changed nothing.
- **Distillation made the network a worse base for search.** Leak-free d=1 search adds +104 [+69, +141] on top
  of c3 but only +40 [+7, +74] on top of the gain-weighted student (same 100 maps; difference +64 [+18, +111]).
  The uniform KL-0.1 student, whose entropy stayed near c3's (1.97 vs 2.02), gets only +45 [+10, +80]
  (difference +58 [+20, +100]). So the loss of search gain is not about sharper policies. The likely cause is
  the critic, which every student refit to search values (a max over noisy lanes).
- PPO continuation (step 4) was not run: it was conditional on a positive raw result. Total spend ≈ $110 of
  the $200 budget.
- What might work instead: lower-noise targets (many more lanes or reply samples per decision, a margin before
  overriding greedy, or soft targets from all lanes' first actions, which the generator does not record today),
  and several expert-iteration rounds rather than one, with the critic held fixed so search keeps working. Each
  round costs about as much data as this run. The cheap win is still search at play time.

## Method

**Teacher.** c3 = `c3-maps-ctl` update 4854 playing every turn through
`search.TurnSearcher(lanes=16, depth=1, temperature=1, rollout_temperature=1, determinize=True)`: 16 candidate
turns (lane 0 greedy, 15 sampled at T=1), each scored by the critic at the searcher's next turn after one
sampled opponent reply. All lanes play on a world sampled from the searcher's own view (belief.py), and the
plan is committed only up to the first revealed difference. Leak-free only: the student cannot see hidden
information.

**Data** (`src/polytopia_bot/rl/distill_data.py`, `experiments/modal_search_distill.py::generate`). Self-play on
the `proposed-v1` map mix with the teacher in both seats, one pinned game process per core, 24 × 32-core Modal
containers for 105 minutes. For every sub-action actually executed on the true game we store:

- the mover's observation (packed int16 view, exactly the encoder input);
- the frozen-engine legal path masks;
- the executed action (the label);
- the plan it came from (all 16 lane scores and the chosen lane);
- the game outcome, when the game finished.

A recorder replays each committed plan on the true state and must reproduce the searcher's state bit for bit,
and every label must be legal under its mask (tests/test_search_distill.py, passed on Modal). Training seeds are
100000–147999; dev seeds 300000–399999 and final seeds 400000+ were not used for data. Games with
seed % 20 == 0 are held out. Games are repacked into zstd-framed shards for the trainer
(`distill_train.pack`), because per-game `.npz` decoding was GIL-bound and left the GPU idle.

**Training** (`src/polytopia_bot/rl/distill_train.py`). Per labelled sub-action the loss is
CE(search action) + β · KL(student ‖ c3) + 0.5 · (v − search value)². The KL is taken along the labelled path
(family; source under the labelled family; target under the labelled family and source). The value target is
the chosen lane's score.

Search overrides greedy in 87% of plans, and the median override gain (chosen minus greedy lane score) is
0.023, about the median spread of the 16 lane scores (0.018). So variants re-weight override labels:

- *gain*: weight min(gain / g, 1);
- *conf*: keep only overrides with gain ≥ g and drop greedy-plan labels.

Training uses Adam, batch 4096, lr 1e-4 (200-step warm-up, cosine to 1e-5), all variants on the same batches,
on one H100 at 32k samples/s per variant.

**Evaluation.** Raw greedy student vs raw greedy c3 (`cpu_arena.match`, frozen engine, both seats,
`proposed-v1` maps, map type per seed) on dev seeds 380000–380499. A game scores 1 / ½ / 0 for
win / 100-turn truncation / loss. Elo CIs are a paired-map bootstrap (4000 resamples). Raw c3 vs itself
scores exactly 0.5 (same deterministic policy in both seats). Pilot screening used 380500–380599 only. The
search arms reuse `search_arena` (100 maps, 380000–380099; candidate searches, opponent is its own raw policy).

## Results

### Data

| | |
|---|---|
| games / labelled decisions | 16,652 / 23.37M (15,812 train games, 840 held out) |
| decisions per game (complete games) | 1,427 mean, 51.7 turns |
| self-play outcome | seat-1 score 0.595, 14.5% of games hit the 100-turn cap |
| plans that overrode the greedy turn | 87.3% |
| labels equal to c3's own greedy action (held out) | 50.3% (100% in greedy plans, 44.5% in overriding plans) |
| throughput | 4.6 decisions per core-second (≈16.5k per core-hour, ≈ 530k per 32-core container-hour) |
| map mix (decisions) | archipelago 14%, continents 13%, classic 13%, ridge 12%, lakes 10%, highlands 10%, jungle 8%, pangea 8%, waterworld 7%, drylands 5% |

Per-game statistics come from the 13.5k games whose container logs survived; several containers were
restarted by Modal near the end and overwrote their logs, but their game files are intact.

### Raw distilled policy vs raw c3 (500 maps × 2 seats, seeds 380000–380499)

| checkpoint | β (KL) | label weighting | steps × batch | held-out acc (override / greedy plans) | KL to c3 | entropy | W–L–trunc | score [95% CI] | Elo vs c3 [95% CI] |
|---|---|---|---|---|---:|---:|---|---|---|
| c3 (teacher) | — | — | — | 0.445 / 1.000 | 0 | 2.02 | — | 0.500 | 0 |
| kl0.1 | 0.1 | uniform | 10.7k × 4096 (2 ep) | 0.447 / 0.880 | 0.012 | 1.97 | 397–405–198 | 0.496 [0.48, 0.51] | −3 [−16, +10] |
| kl0.1-g0.05 | 0.1 | gain, g = 0.05 | 2 ep | 0.445 / 0.832 | 0.084 | 1.80 | 397–407–196 | 0.495 [0.48, 0.51] | −3 [−16, +9] |
| kl0.1-g0.1 | 0.1 | gain, g = 0.1 | 2 ep | 0.444 / 0.822 | 0.127 | 1.71 | 407–414–179 | 0.496 [0.48, 0.51] | −2 [−16, +10] |
| kl0.03-g0.05 | 0.03 | gain, g = 0.05 | 2 ep | 0.445 / 0.824 | 0.109 | 1.76 | 403–396–201 | 0.503 [0.49, 0.52] | +2 [−10, +15] |
| conf0.1 | 0.1 | overrides with gain ≥ 0.1 only | 5.4k × 4096 (1 ep) | 0.446 / 0.920 | 0.007 | 2.03 | 412–405–183 | 0.503 [0.48, 0.52] | +2 [−10, +15] |
| conf0.2 | 0.1 | overrides with gain ≥ 0.2 only | 1 ep | 0.444 / 0.893 | 0.010 | 2.02 | 404–403–193 | 0.500 [0.48, 0.52] | +0 [−11, +12] |
| pilot (gain, 800 steps on 7M) | 0.1 | gain, g = 0.05 | 800 × 4096 | — | — | — | 405–402–193 | 0.501 [0.48, 0.52] | +1 [−11, +13] |

Held-out metrics are on 197k decisions from the 840 held-out games. "acc" means the student's greedy action
equals the label at all three levels. Held-out label CE barely moves (c3 1.952, best student 1.936), and the
value loss against the search value drops from 0.0084 to 0.0028 for every student. No map type stands out:
per-type scores are 0.43–0.55 on 38–136 games each. The pilot's +40 [+12, +69] on the 100 screening maps
(380500–380599) did not hold on the 500 evaluation maps (+1 [−11, +13]). It was the best of four noisy
screens.

### Search on top (leak-free d=1, K=16, 100 maps × 2 seats, seeds 380000–380099)

| searcher (vs its own raw policy) | W–L–trunc | score [95% CI] | Elo [95% CI] | $ |
|---|---|---|---|---:|
| c3 | 111–53–36 | 0.645 [0.60, 0.69] | **+104 [+69, +141]** | 2.67 |
| kl0.03-g0.05 student | 94–71–35 | 0.557 [0.51, 0.60] | +40 [+7, +74] | 2.48 |
| kl0.1 student | 98–72–30 | 0.565 [0.52, 0.61] | +45 [+10, +80] | 2.82 |

c3's search gain minus each student's on the same maps: +64 [+18, +111] (kl0.03-g0.05) and +58 [+20, +100]
(kl0.1). The c3 arm reproduces
the earlier study's +104 on a fresh map block.

## Why it did not work

1. **The improvement operator is weak per decision and noisy.** The d=1 searcher picks the best of 15 T=1
   turns by one-sample critic estimates. Its +104 Elo comes from many small, mostly noisy choices, and the
   max over 15 noisy scores beats greedy 87% of the time. The marginal action distribution it induces at each
   sub-action is c3's own T=1 distribution, slightly re-weighted. Fitting that distribution moves the argmax
   in essentially random directions: students disagree with c3's greedy action on 8–18% of decisions where
   search kept it, yet predict the overriding labels no better than c3.
2. **Turn-level selection is not decomposable.** Search chooses a whole turn after seeing its outcome under a
   sampled reply. The per-sub-action marginal of those choices loses the coherence that made the turn good.
3. **The refit critic costs search.** The value loss trained every student's critic toward the chosen
   lane's score. That is a max over 16 noisy one-sample estimates, so it is optimistic and blurs the small
   differences the searcher ranks lanes by. The uniform student kept c3's entropy and nearly its policy
   (KL 0.012) and still lost most of the search gain, which points at the critic rather than at sharper
   sampling. A distillation meant to keep search working should leave the critic alone (value_coef = 0, or a
   value anchor to c3); this was not tested.

## Cost

| item | $ |
|---|---:|
| label generation: 24 × 32-core containers × 105 min (+ restarted containers) | ≈ 81 |
| packing (2 runs) | 0.2 |
| training: main 4 variants × 2 epochs (H100, 23 min) + conf 2 variants × 1 epoch | 4.1 |
| pilots: 2 short trainings + 4-arm screen on 100 maps | 2.5 |
| raw evaluation: 7 checkpoints × 1,000 games | 13.1 |
| search arms: 3 × 200 games | 8.0 |
| smokes and tests (generator, trainer × 3, eval, unit tests) | 1.2 |
| **total** | **≈ 110** |

Container costs are from measured container seconds at $0.0472/core-hour + $0.008/GiB-hour; GPU jobs are
from wall time at $3.95/h for the H100 plus CPU and memory. The Modal bill was not queried. Ledger:
`runs/search-distill/budget.json`.

## Artifacts

- Checkpoints: `/retained/checkpoints/search-distill-v1/kl0.03-g0.05/final.npz` (the searched student) and
  `/retained/checkpoints/search-distill-v1/kl0.1/final.npz` (closest to c3), both with runner `.json` metadata
  (parent config = c3's). Neither is stronger than c3; c3 stays the reference bot. All other checkpoints are in
  `/results/search-distill/train/{t1,t2,pilot2}/` on the `polytopia-structured-rl` volume.
- Data: `/results/search-distill/d1/games/*.npz.zst` (per game) and `/results/search-distill/d1/packed/`
  (shards), on the same volume.
- Local results: `runs/search-distill/` (chunk logs, training histories, per-game eval rows).

## Reproduce

```bash
modal run --env staging experiments/modal_search_distill.py::tests
modal run --env staging experiments/modal_search_distill.py::smoke_gen --tag smoke1 --minutes 14
modal run --detach --env staging experiments/modal_search_distill.py::generate --tag d1 --containers 24 --minutes 105
modal run --env staging experiments/modal_search_distill.py::pack_data --data d1
modal run --detach --env staging experiments/modal_search_distill.py::train --tag t1 --data d1/packed \
  --variants-json '[{"name":"kl0.1-g0.05","kl_coef":0.1,"gain_scale":0.05},{"name":"kl0.1-g0.1","kl_coef":0.1,"gain_scale":0.1},{"name":"kl0.03-g0.05","kl_coef":0.03,"gain_scale":0.05},{"name":"kl0.1","kl_coef":0.1}]' \
  --options-json '{"epochs":2,"batch":4096,"eval_every":1000}'
modal run --detach --env staging experiments/modal_search_distill.py::train --tag t2 --data d1/packed \
  --variants-json '[{"name":"conf0.1","kl_coef":0.1,"gain_min":0.1,"greedy_weight":0.0},{"name":"conf0.2","kl_coef":0.1,"gain_min":0.2,"greedy_weight":0.0}]' \
  --options-json '{"epochs":1,"batch":4096,"eval_every":1000}'
modal run --detach --env staging experiments/modal_search_distill.py::evaluate --tag e1 --spec-json '[...]'   # raw arms in 100-map blocks
modal run --env staging experiments/modal_search_distill.py::eval_status --tag e1
.venv/bin/python scripts/analyze_search_distill.py data runs/search-distill/d1/chunks.json
.venv/bin/python scripts/analyze_search_distill.py train runs/search-distill/train/t1/train.json
.venv/bin/python scripts/analyze_search_distill.py eval runs/search-distill/eval/e1
```
