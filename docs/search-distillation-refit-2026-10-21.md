# Search distillation with the critic frozen (2026-10-21)

Follow-up to docs/search-distillation-2026-10-19.md. That study distilled leak-free d=1 turn search into c3's raw
policy and found no raw gain, and a side effect: leak-free search on top of the students added only +40–45 Elo
instead of the +104 it adds to c3. Its stated likely cause was the critic, which every student refit toward the
chosen lane's score (a max over 16 noisy one-sample estimates). This note tests that one hypothesis for about $10:
the same distillation with the critic left exactly as c3's.

## Answer

- **Mostly not supported.** Distilling with the critic parameters bitwise frozen at c3's gives students that are
  raw-equal to c3 (83–83–34 vs raw c3, −0 Elo [−31, +31]) and still lose most of the search gain. Leak-free d=1
  search on top of the primary frozen-critic student (kl0.1 recipe) adds **+54 [+21, +89]** in the mirror and
  +60 [+23, +98] against raw c3; on the second frozen-critic student (kl0.03-g0.05 recipe) +67 [+31, +104]. c3
  gets +104 [+69, +141] on the same maps. Paired by map, c3's search gain exceeds the two frozen students' pooled
  gain (+61 [+31, +90]) by **+43 [+5, +84]**; the frozen students exceed their refit twins (pooled +43 [+17,
  +69]) by only +18 [−12, +47] (per student: +9 [−34, +52] and +27 [−9, +63]). So freezing the critic recovers at
  most a third of the lost gain, and that part is not statistically distinguishable from zero; the larger part
  of the loss survives with c3's critic in place. The pre-registered pass (≥ +80 with the paired CI above +45)
  fails for both students.
- **The refit critics were not measurably worse at ranking either.** On the realized path the greedy lane's score
  correlates 0.62–0.64 with the outcome for every critic (c3's own, the two refit ones, the frozen one), CIs
  overlapping; the refit critics' value MSE on held-out search targets had dropped from 0.0084 to 0.0028, and
  anchoring the value output to c3's instead (vloss_teacher 1e-5) changes nothing about the policy.
- So most of the search gain is lost on the **policy side**, not in the critic: the students' argmax play is as strong as c3's
  (raw ties on 100 and, earlier, 500 maps), but the T=1 candidate turns they sample for the searcher are worse
  raw material than c3's. The kl0.1 recipe moves the policy only slightly (KL 0.012 to c3, entropy 1.97 vs
  2.02), yet that is where the +50 Elo goes. The distillation loss fits the marginal of search's past choices,
  which reshapes exactly the sampling distribution the searcher draws its 15 candidates from, toward turns that
  were chosen by max-over-noise. A direct split would be a searcher with c3's candidates and the student's critic,
  and the reverse (the `value_params` searcher option on branch claude/omni-search-eval-ba2c50 enables the latter);
  not run here, ≈ $6 for two 200-game arms.
- For the distillation programme this closes the cheap fix: a student that keeps search working must keep c3's
  *sampling* behaviour, not just its critic. Keeping the critic fixed is still free and harmless (the frozen
  student is not worse than the refit ones), so future rounds should do it, but it does not by itself make
  expert iteration viable. The cheap win remains search at play time on c3 itself.

## Method

Everything is the earlier study's recipe except the value loss; the labelled dataset is reused unchanged
(`/results/search-distill/d1/packed`: 22.1M training decisions from 15,812 self-play games of c3 + leak-free d=1
K=16 search, seeds 100000–147999; 840 held-out games).

**Students** (`distill_train.py`, one H100 job, all variants on the same batches: Adam, batch 4096, lr 1e-4 with
200-step warm-up and cosine to 1e-5, 2 epochs = 10.8k steps):

| student | policy loss | value loss | critic parameters |
|---|---|---|---|
| kl0.1-vfrozen (primary) | CE(search action) + 0.1 · KL(student ‖ c3), uniform label weights — the earlier kl0.1 student | none (`value_coef = 0`) | bitwise c3's |
| kl0.1-vanchor | same | 0.5 · (v − v_c3)², an anchor to c3's own value on the same observation | trained, toward c3's output |
| kl0.03-g0.05-vfrozen | CE + 0.03 · KL, override labels weighted min(gain / 0.05, 1) — the earlier kl0.03-g0.05 student | none | bitwise c3's |

With `value_coef = 0` the gradient of every critic-only parameter (`critic_hidden`, `value`) is identically zero,
so Adam's update is exactly zero and those tensors stay bitwise equal to c3's; the trainer checks this when it
saves (`critic_unchanged` in protocol.json) and tests/test_search_distill.py covers it. The value *output* can
still drift because the trunk is shared with the policy, which is why the anchored variant exists: it trains
the critic toward c3's value instead of the search value. The primary student is the kl0.1 recipe because that
student's policy stayed closest to c3's (KL 0.012, entropy 1.97 vs 2.02) and still lost most of the search
gain (+45 vs +104), which was the evidence that pointed at the critic in the first place.

**Evaluation** on the same boards as the earlier search arms: `proposed-v1` dev maps, seeds 380000–380099, both
seats (200 games per arm), map type per seed. Raw arms: deterministic student vs deterministic c3
(`cpu_arena.match`). Search arms: the student playing every turn through
`TurnSearcher(lanes=16, depth=1, temperature=1, rollout_temperature=1, determinize=True)` (leak-free) against
the raw greedy student (the mirror, the same arm the earlier students ran) and against raw greedy c3. A game
scores 1 / ½ / 0 for win / 100-turn truncation / loss; Elo CIs are a paired-map bootstrap (4000 resamples over
map seeds); differences between arms are paired by map. The c3 and earlier-student search arms are the
2026-10-19 rows (same seeds, same settings), not re-run.

**Critic ranking quality.** `scripts/analyze_search_distill.py critic`: from each search arm's per-turn trace
(greedy-lane score, chosen-lane score), the Pearson correlation of the greedy lane's score with the game
outcome over all search turns (bootstrap CI over games), and restricted to turns 10–40.

**Pre-registered pass:** search on top of the frozen-critic student ≥ +80 Elo (mirror), with the paired-map CI
of its difference from the refit students' +40–45 above zero.

## Results

### Training (held-out metrics on 197k decisions from 840 held-out games, step 10,727)

| student | critic parameters bitwise c3's | held-out acc (override / greedy plans) | CE | KL to c3 | entropy | value MSE vs search target | value MSE vs c3's value |
|---|---|---|---:|---:|---:|---:|---:|
| c3 (teacher) | — | 0.445 / 1.000 | 1.952 | 0 | 2.022 | 0.0084 | 0 |
| kl0.1-vfrozen | **yes** | 0.447 / 0.880 | 1.936 | 0.0121 | 1.967 | 0.0088 | 0.00013 |
| kl0.1-vanchor | no (trained toward c3's output) | 0.447 / 0.880 | 1.936 | 0.0121 | 1.967 | 0.0085 | 0.00001 |
| kl0.03-g0.05-vfrozen | **yes** | 0.445 / 0.824 | 2.020 | 0.1091 | 1.758 | 0.0102 | 0.00038 |
| earlier kl0.1 (refit critic) | no | 0.447 / 0.880 | 1.936 | 0.012 | 1.97 | 0.0028 | — |

The three students' policy metrics reproduce the 2026-10-19 run step for step (same data order, same batches;
the value loss hardly touches the shared trunk). The frozen critic's output drifts by 0.00013 MSE through the
trunk, 60× less than the refit critics moved; the anchor removes even that. Training took 1,121 s on one H100
for all three variants.

### Raw student vs raw c3 (100 maps × 2 seats, seeds 380000–380099, deterministic both seats)

| student | W–L–trunc | score [95% CI] | Elo vs c3 [95% CI] |
|---|---|---|---|
| kl0.1-vfrozen | 83–83–34 | 0.500 [0.46, 0.55] | −0 [−31, +31] |
| kl0.1-vanchor | 81–80–39 | 0.502 [0.46, 0.55] | +2 [−28, +31] |
| kl0.03-g0.05-vfrozen | 81–83–36 | 0.495 [0.46, 0.53] | −3 [−30, +23] |

As expected (and as pre-registered, not a failure): the earlier students were also within ±3 Elo of c3 on 500
maps. No map type stands out (0.34–0.57 per type on 10–32 games).

### Search on top (leak-free d=1, K=16, T=1/1, determinized; 100 maps × 2 seats, seeds 380000–380099)

| searcher | opponent (raw greedy) | W–L–trunc | score [95% CI] | Elo [95% CI] | $ |
|---|---|---|---|---|---:|
| c3 (2026-10-19) | c3 | 111–53–36 | 0.645 [0.60, 0.69] | **+104 [+69, +141]** | 2.67 |
| kl0.1 refit student (2026-10-19) | itself | 98–72–30 | 0.565 [0.52, 0.61] | +45 [+10, +80] | 2.82 |
| kl0.03-g0.05 refit student (2026-10-19) | itself | 94–71–35 | 0.557 [0.51, 0.60] | +40 [+7, +74] | 2.48 |
| **kl0.1-vfrozen** | itself (mirror) | 95–64–41 | 0.578 [0.53, 0.62] | **+54 [+21, +89]** | 2.90 |
| **kl0.1-vfrozen** | c3 | 95–61–44 | 0.585 [0.53, 0.64] | +60 [+23, +98] | 2.63 |
| **kl0.03-g0.05-vfrozen** (replication) | itself (mirror) | 98–60–42 | 0.595 [0.55, 0.65] | **+67 [+31, +104]** | 3.12 |

Paired-map differences (Elo of the first arm minus the second, 100 shared maps, bootstrap over maps):

| comparison | Δ Elo [95% CI] |
|---|---|
| c3 + search (vs c3) − kl0.1-vfrozen + search (mirror) | **+49 [+9, +91]** |
| c3 + search (vs c3) − kl0.1-vfrozen + search (vs c3) | +44 [+0, +92] |
| kl0.1-vfrozen + search (mirror) − kl0.1 refit + search (mirror) | +9 [−34, +52] |
| kl0.1-vfrozen + search (mirror) − kl0.03-g0.05 refit + search (mirror) | +14 [−27, +56] |
| kl0.1-vfrozen + search vs c3 − kl0.1-vfrozen + search mirror | +5 [−34, +44] |
| c3 + search (vs c3) − kl0.03-g0.05-vfrozen + search (mirror) | +37 [−9, +84] |
| kl0.03-g0.05-vfrozen + search (mirror) − kl0.03-g0.05 refit + search (mirror) | +27 [−9, +63] |
| kl0.03-g0.05-vfrozen + search − kl0.1-vfrozen + search (both mirror) | +13 [−25, +50] |
| **pooled**: c3 − mean of the two frozen-critic students | **+43 [+5, +84]** |
| **pooled**: mean of the two frozen-critic students − mean of the two refit students | **+18 [−12, +47]** |
| pooled: c3 − mean of the two refit students | +61 [+24, +100] |

Pooled rows average the two arms' per-map scores before converting to Elo (same 100 maps, bootstrap over maps).

The two opponents for the frozen student give the same answer (+54 / +60), as they should, since its raw
policy ties c3. One caveat on the c3 reference: c3's search gain has been measured three times, +104 (test-time
search study, other maps), +104 (these maps) and +70 [+31, +111] (omniscient-critic study, maps 386000–386099),
so the true gap between c3 and the students may be somewhat under the +49 measured here; the paired CI
excludes zero but is wide.

### Critic ranking quality (from the search arms' per-turn traces)

| searcher's critic | search turns | corr(greedy-lane score, outcome) [95% CI] | corr(chosen lane) | corr, turns 10–40 | MSE greedy |
|---|---:|---|---|---|---:|
| c3 (its own arm) | 11,066 | 0.620 [0.585, 0.652] | 0.620 | 0.658 | 0.428 |
| kl0.1 refit critic | 10,853 | 0.622 [0.596, 0.647] | 0.629 | 0.651 | 0.464 |
| kl0.03-g0.05 refit critic | 10,676 | 0.628 [0.591, 0.658] | 0.631 | 0.679 | 0.417 |
| c3's critic in the frozen student (mirror arm) | 11,405 | 0.640 [0.609, 0.668] | 0.638 | 0.680 | 0.400 |
| c3's critic in the frozen student (vs c3) | 11,277 | 0.635 [0.603, 0.663] | 0.637 | 0.685 | 0.385 |
| c3's critic in the kl0.03-g0.05 frozen student (mirror) | 11,333 | 0.651 [0.622, 0.679] | 0.649 | 0.690 | 0.389 |

The refit critics correlate with the outcome exactly as well as c3's (the frozen student's rows use c3's critic
on different games, which shows the sampling spread of this statistic). This metric sees the critic on the
realized path, not the within-plan ranking among 16 lanes; but a critic blurred by fitting a max over noisy
lanes should have shown up as a weaker correlation, and did not.

## Cost

| item | $ |
|---|---:|
| unit tests on Modal (2 tests, 4 cpus, 151 s) | 0.02 |
| training: 3 variants × 2 epochs on one H100 (1,121 s + startup) | 2.3 |
| raw evaluation: 3 arms × 200 games | 1.1 |
| search arms: kl0.1-vfrozen mirror and vs c3 (2 × 200 games) | 5.5 |
| replication search arm: kl0.03-g0.05-vfrozen mirror (200 games) | 3.1 |
| orchestrators | 0.2 |
| **total** | **≈ 12.2** of the $15 cap |

Container costs from measured container seconds at $0.0472/core-hour + $0.008/GiB-hour (32-core containers
expose 48 CPUs; the search arena pins 32 workers, measured at ~32 busy cores per container); the GPU job from
wall time at $3.95/h for the H100 plus CPU and memory. Modal bill not queried. Ledger rows tagged
`study: refit` in `runs/search-distill/budget.json` (launches are refused past the per-study cap).

## Artifacts

- Checkpoints: `/results/search-distill/train/refit1/{kl0.1-vfrozen,kl0.1-vanchor,kl0.03-g0.05-vfrozen}.npz`
  (+ `.json`, runner format, parent config c3's) on the `polytopia-structured-rl` volume; `protocol.json` there
  records `critic_unchanged`. None is stronger than c3; nothing archived to `/retained`.
- Evaluation rows: `/results/search-distill/eval/r1/` and `r2/` on the volume; locally
  `runs/search-distill/eval/{r1,r2}/` (per-game rows with search traces), `runs/search-distill/eval/refit-combined/`
  (links to the arms the pairs use), `runs/search-distill/train/refit1/train.json`, specs in
  `runs/search-distill/refit/`.
- Code: `distill_train.py` (per-variant `value_coef`, `value_anchor`, `critic_unchanged`),
  `experiments/modal_search_distill.py` (per-study ledger cap `DISTILL_STUDY`, measured-rate worst cases),
  `scripts/analyze_search_distill.py critic`, `tests/test_search_distill.py` (frozen-critic test).

## Reproduce

```bash
modal run --env staging experiments/modal_search_distill.py::tests
DISTILL_STUDY=refit modal run --detach --env staging experiments/modal_search_distill.py::train --tag refit1 --data d1/packed \
  --variants-json "$(cat runs/search-distill/refit/variants.json)" --options-json '{"epochs":2,"batch":4096,"eval_every":1000}' --hours 1
modal run --env staging experiments/modal_search_distill.py::train_status --tag refit1
DISTILL_STUDY=refit modal run --detach --env staging experiments/modal_search_distill.py::evaluate --tag r1 --spec-json runs/search-distill/refit/eval-r1.json
modal run --env staging experiments/modal_search_distill.py::eval_status --tag r1
.venv/bin/python scripts/analyze_search_distill.py eval runs/search-distill/eval/r1 --pair ...
.venv/bin/python scripts/analyze_search_distill.py critic runs/search-distill/eval/e2/c3-d1det-vs-raw-c3.json runs/search-distill/eval/r1/kl0.1-vfrozen-d1det-vs-raw-self.json
```
