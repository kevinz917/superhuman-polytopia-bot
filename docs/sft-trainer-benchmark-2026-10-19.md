# Expert-SFT trainer benchmark (2026-10-19)

**Question.** Which training setup is cheapest and fastest for the next expert-SFT run? The run is about 160k examples: about 1.49B tokens at today's prompt length, or about 0.50B with compact prompt v2. The training objective must not change.

**Short answer.**
- The trainer reaches about 16% MFU on B200. Attention (44% of GPU time) and unfused elementwise kernels (37%) dominate; GEMMs are only 18%.
- Communication costs nothing: the all-reduce takes 5 ms of an 11 s step. Data loading and rank imbalance together take under 3%.
- The two cheap knobs give little: a larger token budget gives +0.5% and Liger SwiGLU gives +1.5%, though SwiGLU also cuts peak memory by 17 GB.
- **cuDNN attention gives +41% (52.6k → 74.1k tok/s on 4 × B200).** It lowers the cost from $0.139 to $0.098 per million tokens. It needs two things:
  1. 256-token length buckets and a one-time plan prewarm (2 min); without them it is 2–4× *slower*.
  2. A decision on the parity gate. cuDNN fails the current bf16-path gate, but so does every independent bf16 kernel. Against an fp32 ground truth it is as accurate as the qualified path or better. Its 32-step loss trajectory stays within the noise floor of variants that pass the gate. See §3.
- H100 is never cheaper than B200 at today's length (5–13% more per token). At compact-prompt length the two cost the same.

All numbers come from Modal (env `staging`). Every arm uses the same rows, the same epoch-0 batch order (seed 17), the same initial adapter and the same response-only loss. Costs use `runs/sft0/cost-ledger.json` rates (B200 $0.00182392/GPU-s, H100 $0.00118492/GPU-s, each with 4 CPU / 16 GiB). Those are estimates, not billing.

## 1. Recommendation

| Scenario | Configuration | Est. wall (train) | Est. cost |
|---|---|---|---|
| Today's prompt, 1.49B tokens, cuDNN accepted | **8 × B200**, budget 16384, Liger SwiGLU, cuDNN SDPA with 256-token buckets + prewarm | **~2.9 h** | **~$152** |
| Today's prompt, cuDNN not accepted | 8 × B200, budget 16384 + Liger SwiGLU (bit-path parity holds) | ~4.1 h | ~$210 |
| B200:8 unavailable | 4 × B200, same as above. **Raise the `train` timeout** (6 h today) | 5.6 h (cuDNN) / 7.8 h | $147 / $204 |
| Compact prompt v2, 0.50B tokens | Whatever is available. 8 × B200 with cuDNN is fastest (~0.8 h, ~$44); 8 × H100 with no checkpointing is ~1.5 h, ~$52 | 0.8–1.5 h | $44–52 |

Practical notes:
- **The 6 h timeout in `experiments/modal_expert_sft.py::train` is too short** for 1.49B tokens on 4 × B200 with the qualified runtime (7.8 h + 3 min setup). Use 8 GPUs, cuDNN, or a 12 h timeout.
- **B200 multi-GPU capacity was scarce today.** A `B200:4` and a `B200:8` request sat at 0 containers for 35 min while other sessions held B200s; the `B200:4` resubmission then started. `H100:8` and `H100:4` started within a minute. The 8 × B200 rows are therefore estimates (§2), not measurements.
- 8 GPUs cost the same per token as 4 (communication is 5 ms/step), so 8 is strictly better whenever capacity exists.

## 2. Throughput, memory and cost (today's prompt length)

Corpus: `expert-sft-v1/prepared` has 39,626 rows; mean 9,329 tokens, median 8,407, p90 15,928, max 24,574; 8.6 supervised tokens per row. The token-weighted mean length is 11.6k. *m* means measured: 30 steps after 2 warm-up steps on 4 × B200, 20 on 8 × H100, 10 on 4 × H100. Each step is a real 64-example effective batch. *e* means estimated.

| Setup | | tok/s | tok/s/GPU | Peak GB (stress 24.6k) | $/M tok | 1.49B: wall | 1.49B: $ | 0.75B: wall | 0.75B: $ |
|---|---|---|---|---|---|---|---|---|---|
| 4×B200 qualified (`castfree_segmented_8192_checkpoint_False`) | m | 52.6k | 13.15k | 150 | $0.139 | 7.9 h | $207 | 3.9 h | $104 |
| 4×B200 budget 16384 | m | 52.9k | 13.21k | 150 | $0.138 | 7.8 h | $206 | 3.9 h | $103 |
| 4×B200 budget 16384 + Liger SwiGLU | m | 53.4k | 13.35k | 133 | $0.137 | 7.8 h | $204 | 3.9 h | $102 |
| 4×B200 budget 32768 | m(1 GPU) | OOM without checkpointing; alternating checkpointing 11.7k/GPU (−12%) | | 105 | | | | | |
| **4×B200 cuDNN b256 + SwiGLU + 16384** | m | **74.1k** | **18.54k** | 133 | **$0.098** | **5.6 h** | **$147** | 2.8 h | $73 |
| 8×B200 qualified | e | 102k | 12.75k | 150 | $0.143 | 4.1 h | $213 | 2.0 h | $107 |
| 8×B200 cuDNN b256 + SwiGLU + 16384 | e | 144k | 17.98k | 133 | $0.101 | 2.9 h | $152 | 1.4 h | $76 |
| 8×H100 adaptive checkpointing (>10,240-token chunks) | m | 60.6k | 7.58k | 69 | $0.156 | 6.8 h | $233 | 3.4 h | $117 |
| 8×H100 SwiGLU + adaptive (>12,288) | m | 65.0k | 8.13k | 72 | $0.146 | 6.4 h | $218 | 3.2 h | $109 |
| 4×H100 SwiGLU + adaptive (>12,288) | m | 33.6k | 8.41k | 72 | $0.141 | 12.3 h | $210 | 6.2 h | $105 |

- The 4×B200 qualified run reproduces production. The expert-sft-v1 run logged 52.8k tok/s and 11.08 s/step; this run measured 52.6k and 11.15 s.
- The 8×B200 rows are 2 × the measured 4×B200 rate × 0.97. The 0.97 covers the larger rank-wait share seen on 8×H100: 4% of the step, against 1.5% on 4×H100 and 1.1% on 4×B200.
- Wall times are training only. A run also has fixed costs: model load and stress, about 3 min; cuDNN plan prewarm, 2 min; checkpoint saves, about 2 min per run at today's cadence.
- **H100 needs checkpointing.** Without it, a 24.6k-token example needs about 150 GB. Full checkpointing runs at 7.26k tok/s/GPU. Alternating checkpointing (8.37k) runs out of memory on the 24.6k stress example. Adaptive checkpointing passes: chunks above 10–12k tokens are fully checkpointed, shorter ones not, and 35% of rows exceed 10,240 tokens. It needs a small trainer change (§5).

### Compact prompt v2 (mean 3,150 tokens, p90 4,598)

This is a proxy: one effective batch of real corpus rows of 2,000–4,600 tokens (mean 3,459), timed on one GPU. The table multiplies per-GPU throughput by N × 0.98 to account for the measured data/wait overhead. 160k × 3,150 ≈ 0.50B tokens.

| Setup | tok/s/GPU (1 GPU, m) | N-GPU tok/s | $/M tok | 0.50B: wall | 0.50B: $ |
|---|---|---|---|---|---|
| B200 qualified 8192 | 18.06k | 4×: 70.8k / 8×: 142k | $0.103 | 2.0 h / 1.0 h | $52 |
| B200 budget 16384 (+SwiGLU) | 18.09k (18.19k) | — | $0.102 | — | $52 |
| B200 cuDNN b256 + SwiGLU + 16384 | 21.16k | 4×: 82.9k / 8×: 166k | $0.088 | 1.7 h / 0.8 h | $44 |
| H100 qualified 8192 (no checkpointing; peak 56 GB) | 11.83k | 4×: 46.4k / 8×: 92.8k | $0.102 | 3.0 h / 1.5 h | $52 |

At compact length, attention drops to 16–23% of kernel time and elementwise kernels rise to about 50%. The cuDNN gain falls to +17%, and B200 and H100 cost the same per token. H100 cannot use budget 16384 at this length: it runs out of memory without checkpointing.

## 3. Numerical parity

**Strict gate**, as in the original qualification: same probe batch, |Δloss| < 0.01, relative gradient error < 0.03, cosine > 0.999, measured against the qualified bf16 path. Probe: 6 rows at length quantiles 5–97% of a real batch, 3.0k–20k tokens, at the initial adapter.

| Variant (B200) | Δloss | rel. grad err | cosine | Gate |
|---|---|---|---|---|
| budget 16384 | 0 | 0.015 | 0.9999 | pass |
| budget 32768 + alternating checkpointing | 4e-8 | 0.015 | 0.9999 | pass |
| Liger SwiGLU | 0 | 0.016 | 0.9999 | pass |
| H100 checkpointing / alternating / adaptive (vs H100 checkpointed path) | 0 | 0.015 | 0.9999 | pass |
| cuDNN SDPA (GQA, repeat_kv, contiguous, 256-bucket: all identical) | 0.018 | 0.88 | 0.59 | **fail** |
| Liger RoPE | 0.019 | 0.94 | 0.57 | **fail** |
| Liger RMSNorm | 0.049 | 0.79 | 0.68 | **fail** |

**Ground truth.** Several independent kernels each "fail" by similar amounts, so I re-ran the probe on 4 short rows (3.1k–9.2k tokens) against an fp32 reference: fp32 weights, no autocast, math SDPA. A second fp32 path (default SDPA) agrees with it exactly.

| Variant | Δloss vs fp32 | rel. grad err vs fp32 | cosine vs fp32 |
|---|---|---|---|
| **qualified bf16 path** | 0.029 | 1.43 | 0.36 |
| bf16 math SDPA | 0.063 | 1.02 | 0.48 |
| **cuDNN SDPA** | **0.0005** | 1.45 | 0.32 |
| Liger SwiGLU | 0.029 | 1.43 | 0.36 |
| Liger RoPE | 0.121 | 0.98 | 0.49 |
| Liger RMSNorm | 0.095 | 0.92 | 0.51 |

What this shows:
- **At the initial adapter, bf16 gradient direction is mostly rounding noise.** Every bf16 path, the qualified one included, has cosine 0.3–0.5 against fp32. Per-row losses are tiny (0.03–0.4 summed over the row's ~9 supervised tokens), so a few near-certain tokens dominate the gradient.
- **The strict gate therefore checks that two variants share the same bf16 rounding path, not that they are correct.** Variants built on the same flash kernel and elementwise ops pass at 1.5% error. Any change of attention kernel or fused norm fails, whether it is more or less accurate.
- On loss, cuDNN is the most accurate bf16 path measured (Δ 0.0005 vs 0.029).
- Liger RoPE and Liger RMSNorm move *further* from fp32 than the qualified path (Δ 0.12 and 0.095). Reject them.

**Training-trajectory check.** On 4 × B200, all arms ran 32 identical steps (2 warm-up + 30) from the same initial adapter. Loss deviation from the qualified arm, per step:

| Arm | max \|Δ\| | mean \|Δ\| (steps 3–32) | mean loss, last 10 steps (qualified: 0.3164) |
|---|---|---|---|
| budget 16384 (passes strict gate) | 0.013 | 0.0036 | 0.3212 |
| Liger SwiGLU + 16384 (passes) | 0.013 | 0.0032 | 0.3182 |
| **cuDNN b256 + SwiGLU + 16384** | 0.030 (step 2, loss 0.89) | **0.0025** | **0.3139** |
| 8×H100 adaptive (other world size; 22 steps) | 0.009 | 0.0037 | — |

cuDNN's trajectory stays within the run-to-run noise floor that gate-passing variants produce. **Recommendation:** accept cuDNN under a replacement gate with two parts: (a) the fp32-anchored loss check and (b) a trajectory check against the noise floor. Optionally confirm with one held-out-games evaluation of the resulting adapter. That is the user's decision; the current gate would reject it.

**Side finding: the cosine metric underflows.** `F.cosine_similarity` in float32 on these 66M-element gradients multiplies two tiny squared norms and underflows: it reports 1.0085 for a gradient against itself. The bench computes cosine in float64. `scripts/speed_training_benchmark.py` and `scripts/speed_distributed_benchmark.py` still use the float32 call, so their cosine values are unreliable when gradients are small. Relative gradient error is unaffected.

## 4. Where time goes

**Per optimizer step, 4 × B200**, mean over ranks and 30 steps:

| Phase | qualified 8192 | cuDNN b256 + SwiGLU + 16384 |
|---|---|---|
| data (read + parse 64 JSON rows per rank, partition) | 0.18 s (1.6%) | 0.17 s (2.1%) |
| forward | 3.63 s (32.5%) | 2.81 s (35.6%) |
| backward | 7.09 s (63.6%) | 4.72 s (59.7%) |
| waiting for slowest rank | 0.13 s (1.1%) | 0.10 s (1.2%) |
| all-reduce (66M fp32 LoRA grads) | 0.005 s | 0.005 s |
| clip + AdamW | 0.007 s | 0.007 s |
| **step** | **11.15 s** | **7.91 s** |

**CUDA kernel time by category**, one fwd+bwd of a ~9.2k-token row on B200:

| | qualified | Liger SwiGLU | cuDNN | cuDNN + all Liger |
|---|---|---|---|---|
| attention | 44% | 44% | 21% | 28% |
| elementwise / reductions (RMSNorm, RoPE, SiLU·mul, LoRA scale + add, casts) | 37% | 34% | 52% | 32% |
| GEMM | 18% | 19% | 25% | 33% |
| total ms | 614 | 608 | 446 | 350 |

- **Attention on B200** runs PyTorch's built-in FlashAttention-2 kernel (`pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel`), which is not tuned for Blackwell. cuDNN's Blackwell SDPA halves attention time.
- **cuDNN builds an execution plan for every new shape**, at about 1.3 s each. A full run sees thousands of distinct lengths, so unbucketed cuDNN measured 2–4× slower than baseline on the first pass. Fix: zero-pad each segment's q/k/v to a multiple of 256 inside attention, then prebuild the 96 plans (≤ 24,576 tokens) once, which took 121 s. Padding is exact under causal attention: real queries never attend to the padded tail, and padded outputs are dropped.
- **Elementwise work** is now the largest remaining cost: about 37%, and about 50% at compact length. Liger fuses RMSNorm and RoPE, but those kernels fail parity against fp32 (§3). The objective-preserving next lever is `torch.compile` on the decoder layer, or folding the LoRA scale into `lora_B`. Neither was tested here.
- **MFU.** At the token-weighted mean length of 11.6k, a token costs about 26.9 GFLOP: 14.9 linear (fwd + input-grad bwd + LoRA) plus 3.4 attention forward × 3.5. That gives 357 TFLOP/s/GPU (~16% of B200 dense bf16) for the qualified path, about 510 TFLOP/s (~23%) with cuDNN, and 218 TFLOP/s (~22%) on H100.

### Cheap changes evaluated

| Change | Effect | Verdict |
|---|---|---|
| Skip the loss on non-supervised positions | **Already done.** `packed_response_loss` applies `lm_head` only to the ~8.6 supervised positions per row. The only remaining objective-preserving skip is computing the last decoder layer's queries and MLP only at supervised positions: at most ~1/36 of layer compute (~2.5%). | Not worth it |
| Token budget 8192 → 16384 (fewer accumulation chunks: 15.1 → 11.4 per rank per step) | +0.5% on 4 × B200; same peak (the 24.6k-token example sets it); bit-path parity holds | Free, take it |
| Token budget 32768 | OOM without checkpointing on B200; −12% with alternating checkpointing | No |
| Liger SwiGLU | +1.5%; peak 150 → 133 GB; parity holds | Take it (needs liger-kernel in the image) |
| Liger RoPE / RMSNorm | +8% / −5% alone; worse than the qualified path against fp32 | Reject |
| cuDNN SDPA, bucketed and prewarmed | +41% at today's length, +17% at compact length | Take it if the parity gate is revised (§3) |
| Data prefetch thread / parse only own partition | ≤ 1.6% | Optional |
| Partition ranks by attention cost (n²) instead of tokens | ≤ 1.1% on 4 × B200 (≤ 4% on 8 GPUs) | Optional |
| Shared prompt prefix | All rows share a 1,018-token system prefix (11% of tokens). Exploiting it in training needs a prefix-tree attention kernel. Compact prompt v2 makes the system prompt 1,056 of 3,150 tokens. | Not cheap |

## 5. Trainer changes needed (implemented in §7)

All of these sit behind the benchmark receipt's `selected` variant, so the trainer keeps qualifying runtimes by receipt.

1. **Budget 16384 + Liger SwiGLU.** Add `liger-kernel==0.6.3` to `training/pyproject.toml` and lock it. In `train_expert`, after `get_peft_model` and before `configure_checkpointing`, add `if variant.get('liger_swiglu'): apply_liger_kernel_to_qwen3(rope=False, rms_norm=False, swiglu=True, cross_entropy=False, fused_linear_cross_entropy=False, model=model.get_base_model())`. Then issue a new receipt whose `selected` variant has `token_budget: 16384, liger_swiglu: true`.
2. **cuDNN.** Move `bench_segmented_sdpa` from `scripts/sft_train_bench.py` (GQA, `bucket=256`, `contiguous=True`) into `fast_training.py` as a new backend, e.g. `segmented_cudnn`, and add it to the allow-list in `packed_response_loss`. Wrap each chunk's forward *and* backward in `torch.nn.attention.sdpa_kernel([SDPBackend.CUDNN_ATTENTION])`. Prebuild the plans for buckets ≤ `max_length` before the corpus stress test (`Runtime.prewarm`). Qualification needs the revised gate from §3.
3. **Adaptive checkpointing (H100 only).** Per chunk, call `configure_checkpointing(model, True if chunk_tokens > threshold else False)`. Use a threshold of 10,240 tokens (12,288 with SwiGLU) on 80 GB GPUs. With compact prompts on H100, plain no-checkpointing at budget 8192 fits (56 GB).
4. **8 GPUs.** `experiments/modal_expert_sft.py::train` hard-codes `gpu='B200:4'` and `--nproc_per_node=4`. Effective batch 64 splits to 8 rows per rank unchanged; `partition_effective_batch` and the SUM all-reduce already handle any world size.
5. **Timeout.** Raise `train`'s `timeout=6*3600` for any single-container run longer than ~5.5 h.

## 6. Reproduce

New files only:
- `scripts/sft_train_bench.py`, the benchmark itself:
  - `explore`: one GPU; strict and fp32 parity, per-chunk forward/backward timing on one effective batch, stress test, kernel mix.
  - `dist`: the expert trainer's own step loop under torchrun, with per-phase timing and loss trajectories.
- `experiments/modal_sft_train_bench.py`, the Modal app `polytopia-sft-train-bench` with presets.
- Results: `runs/sft-train-bench-2026-10-19/*.json` locally, and `sft-train-bench/<tag>/` on volume `polytopia-sft0-training`.

```bash
/Users/K.Z./.local/bin/modal run --env staging -m experiments.modal_sft_train_bench --action dist --gpu B200:4 --preset dist_b200 --steps 30
```

Other presets:
- `explore_b200` and `explore_b200_r2`: variant sweeps on one B200.
- `explore_h100`: variant sweep on one H100.
- `truth_b200`: the fp32 ground-truth check; pass `--parity-only --tag truth-...`.
- `short_b200` and `short_h100`: the compact-length proxy; pass `--tag short-...`.
- `dist_h100` with `--gpu H100:8`, and `dist_h100x4` with `--gpu H100:4`.
- `dist_b200x8`: defined but not run, because no B200:8 capacity was available.

Benchmark spend was **4.56 GPU-h (2.25 B200 + 2.31 H100), ≈ $25** at ledger rates. That is slightly over the 4 GPU-h target: the fp32 ground-truth run and the compact-length proxy were added after the first round of parity failures.

## 7. Implementation and qualification receipts

Section 5 is now implemented. The default runtime did not change: existing protocols keep their old receipt and the strict gate.

- **Trainer** (`expert_trainer.py`, `fast_training.py`). The receipt's `selected` variant drives everything, through `build_model` and `ChunkStep`. The trainer and the qualifier share these two functions.
  - `liger_swiglu: true` fuses only the MLP SiLU·mul, on the model instance only.
  - `token_budget` sets the packing budget.
  - `backend: segmented_cudnn` pads each segment to a multiple of 256 tokens. cuDNN SDPA is forced around each chunk's forward and backward pass, and the plans for all buckets up to `max_length` are built before the corpus stress test.
  - `checkpointing: 'adaptive'` with `adaptive_tokens` gives adaptive checkpointing for H100.
- **Gate.** `train_expert(..., parity_gate='strict' | 'revised')`.
  - `strict` is the original gate. Every receipt carries it.
  - `revised` is §3's replacement: (a) fp32-anchored loss, where the candidate's |Δloss| vs fp32 is at most the qualified path's + 0.01; and (b) trajectory, where the mean |Δloss| vs the qualified arm over steps 3–32 is at most 1.5 × the noise floor. The noise floor is the largest such deviation among the other arms that pass the strict gate.
- **Launcher** (`experiments/modal_expert_sft.py`). `--gpus 4|8` (B200), `--benchmark-receipt <path under runs/>` and `--parity-gate`, all recorded in the protocol. The `train` timeout is now 12 h.
- **Qualifier** (`scripts/qualify_expert_trainer.py`, `experiments/modal_qualify_expert_trainer.py`). One 4 × B200 run measures both gates for each candidate and writes `raw.json`, `qualification.json` and one `receipt-<candidate>.json` per candidate.

**Receipts.** Tag `qualify-b200x4-1792452187`, on volume `polytopia-sft0-training` under `trainer-qualification/<tag>/` and locally in `runs/trainer-qualification/<tag>/`. Reference: `castfree_segmented_8192_checkpoint_False` from `movement-scale40k-v1`. Same epoch-0 order (trainer sampler, seed 17) and same initial adapter for every arm. 32 steps (2 warm-up) at constant lr 1e-4.

| Arm | strict (Δloss / rel err / cos) | Δloss vs fp32 | tok/s (4×B200) | peak GB | traj. mean \|Δ\| (max) | strict | revised |
|---|---|---|---|---|---|---|---|
| qualified 8192 (reference) | — | 0.0291 | 53.4k | 150.5 | — | — | — |
| budget 16384 (noise arm) | 0 / 0.015 / 0.9999 | 0.0291 | 53.5k | 150.5 | 0.0020 (0.007) | pass | — |
| **budget 16384 + Liger SwiGLU** | 0 / 0.016 / 0.9999 | 0.0291 | 54.0k (+1.2%) | 133.3 | 0.0028 (0.009); floor 0.0020 | **pass** | pass |
| **cuDNN b256 + SwiGLU + 16384** | 0.018 / 0.88 / 0.59 | **0.0005** | **75.2k (+41%)** | 133.3 | **0.0021** (0.007); floor 0.0028 | fail | **pass** |

- `receipt-b16384_swiglu.json` is accepted under either gate. `receipt-cudnn_b256_swiglu_16384.json` is accepted only with `--parity-gate revised`. Accepting cuDNN remains the user's decision.
- **The trajectory floor is thin.** It comes from one or two arms of one 32-step run. SwiGLU's deviation is 1.41 × its single-arm floor, so it passes the 1.5 × tolerance narrowly. SwiGLU passes the strict gate anyway. cuDNN deviates *less* from the qualified arm than SwiGLU does.
- Cold cuDNN prewarm took 107 s (96 buckets, ≤ 24,576 tokens).
- `speedup` in the receipts is chained to the base receipt's baseline: 1.553 for SwiGLU and 2.163 for cuDNN. `speedup_vs_reference` is 1.012 and 1.410.
- Spend: two 4 × B200 runs, 3,565 container-seconds in total, about 4.0 GPU-h, about $26 at ledger rates. The first run (`qualify-b200x4-1792450265`) finished all GPU work, then crashed while assembling the receipt. Rank 0 now writes `raw.json` first, and receipts can be re-derived on CPU with `--assemble`.

```bash
/Users/K.Z./.local/bin/modal run --env staging -m experiments.modal_qualify_expert_trainer --action qualify
/Users/K.Z./.local/bin/modal run --detach --env staging -m experiments.modal_expert_sft --action launch --run-id <id> --gpus 8 --benchmark-receipt runs/trainer-qualification/qualify-b200x4-1792452187/receipt-b16384_swiglu.json
```
