# bn-main1: 10-h 4×128 segment on Verda and the matched-sample match (2026-10-24)

**Run.** 4×128 from `random-init-b4w128`, seed 702, LR 3e-4 flat, batch 262,144, minibatch 4,096, 1 epoch.
2× RTX PRO 6000 learner + 20 on-demand Turin actor boxes (18 × 16 vCPU, 2 × 32 vCPU) in FIN-03; 2 of 22 slots
never got capacity (503 / storage quota). 2026-10-23 23:10 → 10-24 09:10 PT, steady ~40k samples/s, learner ~85%
busy, zero learner restarts. 5,431 updates, final.npz = update 5,400 (1.42B samples). $96.32. The Modal controller
synced checkpoints every 15 min and tore down on completion (no VMs or volumes left).

**Match.** final.npz vs scratch-overnight-1 update-5250 (1.38B samples, 2×64), 128 proposed-v1 maps × 2 seats,
deterministic both, seeds 396000+: **84–138–34 (32.8% [28.1, 37.5]); decided 38%, ≈ −85 Elo.** 32 min, ~$2.

| | approx KL | clip fraction | entropy | value loss |
|---|---|---|---|---|
| bn-main1 u200–5300 | 0.0057 → 0.0073 | 0.062 → 0.085 | 2.74–2.81 | 0.0009–0.0012 |
| Scratch 7h u200–5300 | 0.0042 → 0.0049 | 0.043 → 0.053 | 2.88–2.92 | 0.0035–0.0036 |

**Confound.** Scratch 7h trained at minibatch 16,384; bn-main1 at 4,096 (A1's H100 OOM workaround carried over),
i.e. 4× the Adam steps per batch at the same LR with no epoch-level KL stop (epochs = 1). Its policy moves ~1.5×
further per update. So the deficit is the network, the minibatch, an untuned LR for the larger net, or "capacity
shows later" — not separable from this run. Next: matched snapshots at u1000–4000 for the trend; continue only at
minibatch 16,384 (same speed on 2× PRO 6000).
