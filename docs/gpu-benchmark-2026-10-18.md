# GPU cost benchmark for structured-RL PPO (2026-10-18)

Question: with the decision-budget collector and the fast training engine, which
Modal GPU gives the most PPO training per dollar, and is a faster GPU worth it?

**Short answer:** GPU class still matters far less than lane count. Collection
(game simulation) is over 90% of every update on every GPU; gradient steps take
under half a second. Best value per decision: **RTX PRO 6000 at 2048 lanes ×
32 horizon** ($0.092–0.114 per million learner decisions, 8.9–11.1k decisions/s).
Next is **H100 at 2048×32** (6% more per decision, 17% faster). The **L40S tops
out at 1024 lanes** (it runs out of memory at 2048), where it ties the RTX PRO 6000
on cost but runs 1.3–1.4× slower. B200/B300 are the fastest single GPUs, but cost
25–190% more per decision. A10, L4 and A100 are not worth it.

## Method

- `experiments/modal_gpu_benchmark.py` runs short real PPO (`repair.run`,
  decision-budget collector, fast engine, parent `diverse-bc-dagger-v1`, seed
  20261019, native 14×14 maps) on one GPU per job, one job per (GPU, lanes) pair,
  with 420 s of training each. `scripts/profile_collector.py train` does the work;
  `--horizon` sets decisions per lane per update.
- Rates are Modal list prices (modal.com/pricing, read 2026-10-18) plus 8 CPU
  cores and 32 GiB ($0.633/h).
- **Matched updates:** per-update time grows as games progress (L40S 512: 5.7 s at
  update 2, 9.4 s at update 16). Runs that compile slowly fit fewer updates and look
  faster. All runs share a seed and the deterministic engine, so the same
  (lanes, horizon) config plays nearly the same games on every GPU. The tables
  compare the same update indices (`scripts/compare_gpu_benchmark.py`). Update 1
  includes compilation and is excluded.
- 43 jobs, $24.46 estimated total. Ledger rows `bench-gpu-*`; raw rows are in
  `runs/structured-ppo-v3/gpu-benchmark-rows/`.

## Results (updates 2–11; updates 2–5 in `gpu-benchmark-matched.json`)

$/M = dollars per million learner decisions. L40S rate $2.58/h.

| Config | GPU | $/h | s/update | learner dec/s | $/M |
|---|---|---:|---:|---:|---:|
| 512 × 64 | L40S | 2.58 | 7.39 | 4,433 | **0.162** |
| | RTX PRO 6000 | 3.66 | 6.53 | 5,021 | 0.203 |
| | A100-80GB (PCIe) | 3.13 | 9.19 | 3,564 | 0.244 |
| | H100 | 4.58 | 8.16 | 4,014 | 0.317 |
| | H200 | 5.17 | 7.28 | 4,502 | 0.319 |
| | B200 | 6.88 | 7.75 | 4,231 | 0.452 |
| | B300 | 7.73 | 7.18 | 4,566 | 0.470 |
| 1024 × 64 | L40S | 2.58 | 12.49 | 5,245 | **0.137** |
| | RTX PRO 6000 | 3.66 | 9.55 | 6,862 | 0.148 |
| | H100 | 4.58 | 11.63 | 5,635 | 0.226 |
| | H200 | 5.17 | 10.49 | 6,248 | 0.230 |
| | B200 | 6.88 | 10.79 | 6,076 | 0.315 |
| | B300 | 7.73 | 11.05 | 5,929 | 0.362 |
| 2048 × 32 | RTX PRO 6000 | 3.66 | 7.37 | 8,891 | **0.114** |
| | H100 | 4.58 | 6.35 | 10,321 | 0.123 |
| | H200 | 5.17 | 5.90 | 11,103 | 0.129 |
| | A100-80GB (SXM4) | 3.13 | 9.92 | 6,609 | 0.132 |
| | B200 | 6.88 | 5.27 | 12,430 | 0.154 |
| 2048 × 64 | RTX PRO 6000 | 3.66 | 17.64 | 7,429 | **0.137** |
| | H200 | 5.17 | 16.48 | 7,952 | 0.181 |

Rows in different configs are not directly comparable. At 2048×32, each lane
advances half as far per update as at 1024×64, so its games are earlier and its
steps cheaper. At roughly matched per-lane progress (2048×32 updates 2–11 vs
1024×64 updates 2–5), 2048 lanes still gives about 1.2× the throughput on H100.

## Memory limits (hard failures, not tuning)

| GPU (memory) | Largest config that ran |
|---|---|
| L4, A10/A10G (24 GB) | none reliably: L4 512 ran 4 updates then OOM'd in a longer run; A10 OOM'd at 512 |
| L40S (48 GB) | 1024 × 64 (2048 × 32 OOM) |
| A100-80GB, H100 (80 GB) | 2048 × 32 (2048 × 64 OOM) |
| RTX PRO 6000 (96 GB), H200 (141 GB) | 2048 × 64 |

Memory grows with both lanes and decisions per update. It also grows during a
run as games get longer.

## Replay-audit tolerance (decision needed before leaving L40S/H100)

The training replay audit requires |behavior logp − replay logp| ≤ 1e-5.
Passing L40S/H100/H200 runs already peaked at 9.8e-6. Every B200, B300 and RTX PRO
6000 run, and one A100-SXM4 run, failed at 1.1–3e-5 and stopped. The v2/v3 runs used
a benchmark-only tolerance of 1e-4 (training default unchanged); their worst
errors were 1.9e-5–6.2e-5. That is float32 kernel-order drift, about a 0.006% PPO
ratio change. B200/B300 also print a cuBLAS < 13.2 warning about concurrent-kernel
TMEM corruption. Training on RTX PRO 6000, B200 or B300 needs either a raised audit
tolerance (e.g. 5e-5 or 1e-4) or a newer CUDA/cuBLAS. Keep it at 1e-5 to stay on L40S/H100/H200.

## Caveats

- One run per (GPU, config); 7 minutes covers early game phases only. Rankings
  held across the 2–5 and 2–11 windows. Longer runs will be slower per update.
- Modal's A100-80GB gave PCIe for some jobs and SXM4 for others.
- One-time fast-engine compile took 3–10 minutes. It is CPU-bound and varies by host.
- Multi-GPU (earlier: 4 × L40S at 512 lanes each = 3.69× throughput) was not re-measured.

## Recommendation

1. **Default single-GPU config: RTX PRO 6000, 2048 lanes × 32 horizon.** Cheapest
   per decision and about 1.4× L40S-1024 throughput at matched game progress. Needs
   the audit tolerance raised.
2. **Without changing the audit: H100, 2048 × 32.** About 10% more per decision than
   L40S-1024 and about 1.6× its throughput at matched game progress.
3. **Fastest per run regardless of cost: B200, 2048 × 32** (+25–35% $/decision over H100).
4. For more throughput beyond that, scale out with more GPUs rather than buying a
   bigger one. Avoid A10/L4, A100, and B200/B300 at ≤1024 lanes.
