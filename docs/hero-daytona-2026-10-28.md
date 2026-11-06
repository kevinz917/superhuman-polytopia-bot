# Continuing the hero runs on Daytona: 8-hour plan (2026-10-28, not started)

Branch `claude/daytona-hero` = main (f14a8c0, with the Oct 28 throughput fixes) + the Daytona harness.

- Controller: `experiments/daytona/hero.py`
- In-sandbox code: `experiments/daytona/node.py hero_*`
- Host check: `experiments/daytona/gpu_check.py`
- Cost model: `experiments/daytona/cost_matrix.py`

Nothing below has been launched.

## What gets continued

| run | stopped on Modal | resume point (with Adam state) | samples so far | elapsed clock |
|---|---|---|---|---|
| `hero-1v1-cont1h` | 06:02 CST Oct 28 | u11650 | 6.95 B total | 26,854 s |
| `hero-ffa4` | 06:04 CST Oct 28 | u11550 | 1.51 B | 26,052 s |

Both are true resumes (`Learner._resume`). Weights, Adam moments, omniscient critic, update count, training log, the
historical pool (8 snapshots) and the clock all come back. Three settings change:
- `training_seconds` becomes elapsed + the new hours;
- LR returns to 3e-4 and decays to 1e-4 over the last 25% of the segment, as each Modal segment did;
- `checkpoint_every` drops to 20, so a reclaimed learner loses at most ~1 min of updates.

The 4p smoke's table showed the usual dip after an LR reset: CUR was −22 Elo against PREV after 250 updates, which is
within noise.

## What is on Daytona now

- **Volume `hero-runs`**: both run folders and every eval reference, at their Modal paths, sha256-verified. It also
  holds the code and the engine wheel at `/vol/code/<sha>`, and the XLA compile cache at `/vol/jax-cache/<gpu>`.
- **`hero-hub`**: a 4 vCPU CPU sandbox ($0.33/h) for volume bookkeeping and the 4p table summaries.
- **No GPU sandboxes running.**

## Measurements (Daytona spot, 2026-10-28)

**Learner, train step only** (`Learner.benchmark`, hero configs). Live numbers are from the smokes.

| learner | 1v1 /s | 4p /s | GPU $/h | placeable as spot |
|---|---|---|---|---|
| 1x H100 | 91.7k | 49.5k | 2.27 | yes |
| 2x H100 | 182.6k (live: 1.48 s/262k batch) | 98.7k | 4.54 | yes, but 2 of 3 H100 learners were reclaimed |
| 4x H100 / 4x H200 / 2x H200 | — | — | — | **refused** ("no spot capacity") |
| 1x RTX 5090 | OOM (32 GB) | OOM | 0.74 | yes |
| 4x RTX 5090 | 164.8k | 79.7k | 2.96 | yes |
| 1x RTX PRO 6000 | 53.1k | 31.2k | 1.74 | yes |
| 2x RTX PRO 6000 | 105.2k | 62.7k | 3.48 | yes |
| **4x RTX PRO 6000** | 205.7k (live 1.86 s/batch, **~125k/s with assembly**) | 116.8k (live 1.91 s/batch, **~65k/s**) | 6.96 | yes, plenty of stock, none reclaimed |

**Actor sandboxes** (1 GPU + 16 vCPU, 48 processes, bf16 server, `server_cores` 2, pinned to 16 CPUs). The actors are
CPU-bound, and the CPU model depends on the GPU's host, so the GPU type is really choosing a host CPU:

| actor host | 1v1 /s | 4p /s |
|---|---|---|
| RTX PRO 6000 (EPYC 9455, Zen 5) | **~29.5k live** (4 actors fed 118k/s) | 28.1k bench (OOM-restarted at 128 GiB, needs 160); ~20k assumed live |
| RTX 4090 (EPYC 7713, Zen 3) | ~23k bench | **6.3k live** |

**Evals** run on 4-vCPU CPU sandboxes. One 1v1 round (5 references × 32 maps × 2 seats) takes ~5 min and ~$0.5. The
final 64-map round on the smoke checkpoint gave 63% vs champion, 55% vs bn4 and 95% vs c3. The 4p table
(800 games, Bradley-Terry ratings) also ran end to end.

## Cost matrix, 8 h

From `python experiments/daytona/cost_matrix.py --hours 8`. Daytona rows assume 10% of the clock lost to spot
reclaims; Modal rows include egress.

| opt | run | where | learner | actors | samples/s | $/h | samples in 8 h | $ in 8 h | $/B |
|---|---|---|---|---|---|---|---|---|---|
| A | 1v1 | Modal | 4x H100 | 17x A10G | 208k | 66.6 | 6.0 B | 533 | 89 |
| **B** | 1v1 | Daytona | 4x PRO 6000 | 5x PRO 6000 | 112k | 30.2 | 3.2 B | **241** | 74 |
| C | 1v1 | Daytona | 2x H100 | 6x PRO 6000 | 144k | 29.7 | 4.2 B | 238 | 57 |
| D | 1v1 | Daytona | 4x RTX 5090 | 5x PRO 6000 | 108k | 26.2 | 3.1 B | 209 | 67 |
| **E** | 4p | Modal | 4x H200 | 7x A10G | 92k | 41.2 | 2.6 B | **330** | 124 |
| **F** | 4p | Daytona | 4x PRO 6000 | 5x PRO 6000 (160 GiB) | 58k | 38.0 | 1.7 B | **304** | 180 |
| G | 4p | Daytona | 4x PRO 6000 | 11x RTX 4090 | 58k | 47.4 | 1.7 B | 379 | 225 |

Evals add ~$1–2 per round (2-h cadence plus a final round: ~$15 for both runs over 8 h). The hub adds ~$3.

Reading the matrix:
- **1v1:**
  - Daytona B costs less than half of Modal A, but delivers ~55% of the samples, because the 4x PRO 6000 learner tops
    out at ~125k/s.
  - C would be faster and cheaper per sample, but its live H100 learner rate came from a broken smoke, and H100 spot
    learners were reclaimed twice.
  - Per sample, Daytona is 15–35% cheaper.
- **4p (the long run):**
  - Modal E is cheaper per sample and ~1.6x faster. On Daytona the 4p learner is the limit (~65k/s live vs 92k/s on
    Modal's 4x H200), and 4x H200 isn't available as spot.
  - F carries a 5th actor because its live actor rate is unverified (the 4p PRO 6000 smoke landed on a bad host).
    If 4 actors turn out to keep the learner busy, F drops to ~$263.
  - Daytona F only wins if Daytona credits pay for it.

## Recommendation (8 h)

- **If the ~$10k Daytona credits cover spot GPUs:** run both on Daytona (B + F). That's ~$560 in credits, giving
  3.2 B (1v1) and 1.7 B (4p) samples.
- **If paying cash:** run 1v1 on Daytona (B, $241) and 4p on Modal at main's defaults (E, $330), ~$590 together. This
  gives 1.6x more 4p samples than F for ~$25 more.
- **For more 1v1 samples at ~$530:** use Modal A.

## Risks found and handled

1. **Spot reclaims (no warning).** The keeper detects them within a minute, resumes the learner from the volume
   (losing at most 20 updates) and replaces actors. Each sandbox has a TTL, so billing stops even if the keeper
   stops. The keeper runs on the Mac: keep it awake with `caffeinate` and plugged in, lid open. Alternatively, give the
   hub a `DAYTONA_API_KEY` so the keeper can run there.
2. **Bad hosts.** One 4x PRO 6000 host moved 0.2–1.6 GB/s to 3 of its GPUs (healthy hosts: 13–16 GB/s). Its learner
   trained with KL ~1 from the first update and 2–30x slower steps. Now every learner and actor host must pass
   `gpu_check.py` before use (up to 3 hosts tried), and the keeper recycles a learner whose KL stays above 0.3.
3. **Duplicate node starts from dropped exec replies.** Fixed: nodes are launched as async toolbox-session commands
   with a pidfile guard.
4. **No on-demand GPUs.** The org has no "GPU credits" (on-demand is refused), so learners can only be spot. On-demand
   learners would remove most of risk 1; that's a Wallet step on the Daytona dashboard.

## Runbook

```bash
cd .claude/worktrees/daytona-hero
python experiments/daytona/hero.py publish                      # if HEAD changed
python experiments/daytona/hero.py launch hero-1v1-cont1h --hours 8
python experiments/daytona/hero.py launch hero-ffa4 --hours 8 --actors 5
caffeinate -i python experiments/daytona/hero.py keep hero-1v1-cont1h --max-usd 300 > runs/hero-daytona/keep-1v1.log 2>&1 &
caffeinate -i python experiments/daytona/hero.py keep hero-ffa4 --max-usd 350 > runs/hero-daytona/keep-ffa4.log 2>&1 &
python experiments/daytona/hero.py status hero-ffa4               # sandboxes, $/h, learner log tail
python experiments/daytona/hero.py stop hero-ffa4                 # early stop: final.npz, then the keeper tears down
python experiments/daytona/hero.py down hero-ffa4                 # delete everything of the run now
```

Results go to `/vol/results/distributed/<run>/`. Eval history goes to `/vol/runs/<run>/eval-history.json` and
`runs/hero-daytona/<run>-eval-history.json`.
