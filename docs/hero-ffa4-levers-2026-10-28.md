# 4p hero: where the samples/s go, and the levers (2026-10-28)

Live run `hero-ffa4-cont1` on Modal: learner 4x H200 + 7 A10G actors, $38.5/h. It consumes **93k samples/s**,
which is about **$115 per billion samples**. All numbers are medians over updates 50–150 of
`/results/distributed/hero-ffa4-cont1/training.jsonl` unless marked as a bench.

## Where the time goes (per 131,072-sample update, 1.41 s)

| stage | s | note |
|---|---|---|
| GPU train step | 1.11 | train-only bench on the same GPUs: 0.78 s; the rest is contention with assembly on the same 4 GPUs |
| batch assembly, overlapped with training | **1.38** | **the binding stage** |
| · coded-view rebuild (on the GPUs) | 0.61 | 0.29 s in the Oct 28 replay bench (`a6/v7-h200-*`) |
| · omniscient-critic value pass (on the GPUs) | 0.72 | |
| learner waiting for assembly | 0.27 | |

- **Actors are oversupplied:** they deliver 110k samples/s, so ~20% is dropped as stale (mean lag 7). That's
  ~$4/h of actor time producing nothing.
- **Policy update is healthy:** KL 0.008, clip 0.10, entropy 3.2.

## Measured today

**Train-only bench** (`bench_learner`, the run's own config, 4 GPUs, synthetic 131k batch):

| learner | train step | train samples/s | GPU $/h (Modal) |
|---|---|---|---|
| 4x H200 (current) | 0.78 s | 169k | 18.16 |
| **4x H100** | **0.67 s** | **195k** | **15.80** |
| 4x B200 | 0.46 s | 287k | 25.00 |

On this model, H100 trains faster than H200 and is cheaper. B200 is 1.7x H200, but only pays off once assembly is
under ~0.6 s.

## The levers, ranked by $/sample impact

### 1. Stop scoring padding in the value pass (code, small, exact) — est. +15–20% samples/s at the same cost
- **Main batch:** padded to a multiple of `VALUE_CHUNK × devices` = 8,192 × 4. The assembler overshoots 131,072 by
  part of one message, so every batch pads to **163,840 rows (+25%)**.
- **Segment-boundary rows** (bootstrap values, ~5,000 per update) are padded the same way, to **32,768 rows (6.5x)**.
- **Net:** the critic scores ~196k rows to get ~136k real ones. About 30% of the 0.72 s is padding, and the rows past
  131,072 are scored and then thrown away.
- **Fix:**
  - pad to a multiple of `devices × value_sub` (1–4k rows) instead of `devices × 8,192`;
  - stop dequeuing at exactly `batch` rows, carrying the remainder of the last message into the next batch.

  Values for real rows are unchanged. The existing bitwise gate (`assembly_bench` gate cases) checks this.
- **Expected effect:**
  - value pass 0.72 → ~0.5 s, assembly 1.38 → ~1.15 s;
  - ~112k samples/s on the same fleet;
  - **~$96/B** instead of $115/B.

### 2. Switch the 4p learner to 4x H100 (config) — est. −$2.4/h now, faster train step
- 0.67 s vs 0.78 s per train step (bench), at $15.80 vs $18.16/h of GPU.
- **Caveat:** the comments say an 80 GB H100 needs 4-seat value sub-chunks (`value_sub` ≤ 2k), which makes the value
  pass slower. So H100 is the right call only together with lever 1 (or 5).
- Verify with one `assembly_bench` on captured 4p messages.

### 3. Then 4x B200 + more actors — est. ~1.5–1.7x samples/s, ~10% cheaper per sample
- With lever 1 in, assembly (~1.15 s) still binds over the B200's 0.46 s train step. The B200 also speeds up the
  GPU parts of assembly (2–3x in the Oct 27 B200 replays: value 0.13–0.14 s, views 0.4–0.5 s).
- **Expected:** ~0.6–0.7 s per update, ~190–220k samples/s with 13–15 actors.
  - Cost: learner ~$27/h + 14 × $2.62 actors ≈ **$64/h**.
  - Per sample: **~$85–95/B**, and 2x the samples in the same 8 h.
- **Caveat:** B200 capacity on Modal was unavailable on Oct 27 (the 4x B200 attempt waited 20 min); keep H100 as the
  fallback.

### 4. A cheaper critic (model change, needs an A/B) — est. up to ~2x on top
- **Why it matters:** the omniscient critic is a full 4x128 trunk on a 4-view, ~620-channel stem, about 1.4x the
  policy's FLOPs. It is ~58% of the train step's compute and all of the value pass.
- **Candidate:** a 2x64 critic (~1/8 the compute) would cut the train step ~50% and the value pass ~85%.
- **Risk:** the omniscient critic is what made 4p learn ("the one lever", Oct 20 ablation), so its size needs a test.
  The critic is not used in play, so the policy architecture and the checkpoint stay compatible; only the critic
  restarts. Continuations already re-learn the critic for ~1 h.
- **Test:** two 2-h arms from u11550, control vs small critic, judged on the four-seat tables vs PREV/START/CHAMP.
  About $80 per arm on Modal.

### 5. Train on the high-|advantage| half (config, needs an A/B) — est. up to ~1.4x when train-bound
- `advantage_keep 0.5`, the Generals.io "Average Joe" recipe trick that already exists here. The critic still scores
  every row, but the train step runs on half of them.
- Only helps once training is the binding stage (after levers 1/3). Same A/B harness as lever 4, one more arm.

### 6. Right-size the actors (no code) — −$2.6/h now
- 20% of samples are stale-dropped, so 6 A10G containers would feed the current learner. Re-size whenever the learner
  changes: each container delivers ~14.7k/s, and you need ~1.2x the learner rate to cover lag.

### Not worth it now
- **Moving the critic onto the actors' idle GPUs:** served inference rejects actor-side critics, and 4 seats require
  learner-side values. Multi-day change, and levers 1/3/4 get most of the gain.
- **More PPO epochs:** the learner is the bottleneck, so reusing samples doesn't raise learning per learner-second.
- **Daytona for 4p:** trains with KL ~1 there, cause unknown.

## Suggested sequence

| step | what | cost | expected |
|---|---|---|---|
| 1 | Implement lever 1 + bitwise gate test on Modal | ~$5 | — |
| 2 | `assembly_bench` with captured 4p messages: base vs lever 1, on 4x H200 / H100 / B200 | ~$20 | confirms the H100/B200 choice |
| 3 | A/B arms from u11550 (control, small critic, `advantage_keep` .5), 2 h each, 4-seat tables | ~$240 | decides levers 4/5 |
| 4 | Next 4p segment on the winners | — | **~2–3x samples/s at ~25–40% lower $/sample** vs today's $115/B |

The live run is left as is. Changing it mid-segment would cost a restart and the critic warm-up, for at most ~$20 of
savings.
