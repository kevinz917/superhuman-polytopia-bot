# Verda big run: launcher fixes and runbook (2026-10-23)

Goal: run the 4×128 network on Verda for 16–21 h (~8B samples) without the Mac in the loop. Plan and costs:
4× RTX PRO 6000 learner on-demand ($9.97/h) + ~8 × CPU-TURIN.120V spot actors ($0.576/h each), about
$30–36 per billion samples (extrapolated from the Oct 23 Verda pilot and A1's Modal logs; unmeasured on Verda).

## What changed

| Fix | Where | How it works |
|---|---|---|
| Learning-rate decay | `LearnerConfig.learning_rate_end`, `lr_decay_fraction`; `scheduled_learning_rate` | Linear from `learning_rate` to `learning_rate_end` over the last `lr_decay_fraction` of `training_seconds` (default off: constant rate). The rate is an input to the jitted step, like the entropy coefficient; logged per row as `learning_rate`. |
| Controller off the Mac | `experiments/verda/modal_controller.py` (deployed Modal app `verda-controller`, every 3 min) + `control.py` (pure decisions, `test_control.py`) | After `bigrun.py arm`, the Mac is out of the loop: the tick probes the learner over SSH, lists the run's VMs and the account balance, and acts. |
| Deadline and spend cap | `control.decide` | Teardown at the deadline (learner start + hours + slack), at the cap (max of the balance drop and a per-VM hourly estimate), when the learner VM is gone or offline, its service is down or unreachable for 3 ticks, it writes no row within the 45-min startup grace, or no row for 20 min, or training completes. |
| Checkpoint sync | `modal_controller.sync` | Every 15 min: changed `latest*`, `training.jsonl`, `protocol.json`, `snapshots/*` are tarred over SSH into `/results/distributed/<run>/` (Modal's layout, so `elo_ladder` / `arena` read them unchanged); at the end `latest*` is also published as `final*`. |
| Spot actor replacement | `control.decide` + `setup_actor` | The controller creates all actors (spot, OS disk deleted on discontinue) and replaces any that is discontinued, offline, vanished, or not provisioned in time, with a new generation hostname (`<run>-actor-<slot>-g<k>`), and opens/closes its learner firewall rule. |
| Learner restart | `setup_node.sh` systemd units | `polytopia-learner` (Restart=on-failure) resumes `latest.npz` with Adam state; `polytopia-actor` (Restart=always) reconnects. |
| Fast actor provisioning | `bigrun.py setup` | The engine wheel is built once on the learner and shipped in the bundle, so replacement actors install it (no Rust compile). |
| Orphaned disks | `sweep_volumes`, `bigrun.py down` | Detached volumes named `<run>-*` are deleted after every delete and teardown (the CLI's `--with-volumes` left them in the pilot). |

Tests on Modal: `modal_selftest.py` (control.py, 9 passed); `test_distributed_rl.py` + `test_rl_18x18.py`
(the 18×18 tests had called `train_epoch` without the entropy argument since 7637dc1; the wrapper's defaults fix that).

## One-time setup (the user)

`python experiments/verda/bigrun.py keys` prints the steps: generate the controller SSH key, register it on Verda,
create the Modal secrets `verda-ssh` (the private key) and `verda-api` (VERDA_CLIENT_ID / VERDA_CLIENT_SECRET),
deploy the controller. Secrets are never written by the code.

## Runbook

```
bigrun.py init --run bn1 --config experiments/verda/configs/bignet-4x128.json \
    --parent /results/pilots/random-init-b4w128.npz --actors 8 --hours 1 --cap 40
bigrun.py preflight --run bn1
bigrun.py up --run bn1 && bigrun.py setup --run bn1
bigrun.py bench --run bn1                     # 2 and 4 GPUs x minibatch 4096 / 8192 / 16384
bigrun.py start --run bn1 && bigrun.py arm --run bn1
bigrun.py status --run bn1                    # any time; the Mac may sleep
bigrun.py down --run bn1                      # early stop (otherwise the controller tears down itself)
```

## Quotes (not run)

| Step | What it measures | Time | Cost (cap) |
|---|---|---|---|
| Learner bench | 4× PRO 6000 on-demand: boot + setup ~20 min, 6 arms (2 / 4 GPUs × 3 minibatch sizes) ~15–20 min | ~40 min | ~$7 ($12) |
| 1-h smoke (same learner) | 8 × CPU-TURIN.120V spot actors, 1 h of 4×128 training from the random init: samples/s per box and in total, learner time per batch, one forced actor deletion (replacement), checkpoint sync, the deadline teardown | ~1.6 h after the bench | ~$22 ($30) |
| Both, one learner | | ~2.3 h | ~$29 ($40) |

Unverified until the smoke: spot flags (`--contract spot --is-spot`), spot availability for CPU-TURIN.120V in FIN-03,
per-box throughput on 120-vCPU boxes, Verda egress pricing for the sync, and that `vm list` reports discontinued spot
VMs with one of the statuses control.DEAD lists.

## Results: bench and smoke `bn-smoke1` (Oct 23, 21:40–23:05 PT, $10.50 total)

**Bench** (FIN-02, 4×128 net, `Learner.benchmark`, 4 epochs): 1× RTX PRO 6000 25.8k samples/s (minibatch 4096 and
8192; 16384 out of memory, 33 GB); 2× 50.3k / 51.0k / 51.8k at 4096 / 8192 / 16384. Linear in GPUs; minibatch size
does not matter. In the live run, 5.47 s per 262k batch (H100 on Modal: 4.88 s). $1.68.

**Smoke** (2× PRO 6000 learner on-demand in FIN-02; 1 h of training from the 4×128 random init; $8.15, cap $25):
222 updates, 58.2M samples, `final.npz` + critic synced to `/results/distributed/bn-smoke1/`, then the controller tore
everything down itself ("training complete", no VMs or volumes left). Its spend estimate matched the balance drop.

| Actor box | CPU | Samples/s per process (4×128 net) | $ per billion samples |
|---|---|---|---|
| CPU-TURIN.64V.256G on-demand (FIN-03) | EPYC 9655 (Zen 5) | **136** | **$24.5** |
| CPU.180V.720G spot (FIN-02) | EPYC 9654 (Zen 4) | 61 | $22 (evicted after ~20 min) |
| CPU.180V.720G on-demand (FIN-02) | EPYC 9654 (Zen 4) | 61 | $55 |

With all three boxes (Turin 64 + 32, Genoa 180), 26–27k samples/s, the sum of the boxes; the learner waited ~4.2 s of
every 9.7 s update. Actors in FIN-03 against a FIN-02 learner worked with no visible penalty.

What happened and what changed during the run:
- Capacity: 4 of 6 actor creates were refused (503 "Not enough resources"), including types `verda availability`
  reported as available. Added an ordered fallback list (`actor_options`: type, spot, rate, optional location; set
  with `bigrun.py scale --actor-options`), a retry on the next tick after a failed create, and every option's error
  in the log.
- The one spot box was reclaimed after ~20 min; the controller saw it missing and replaced it with a new generation.
- `training.jsonl` is written only at checkpoints (every 50 updates), so the stall probe now reads `learner.log`,
  which gets a row every update.

Plan for the main run, given this: learner and actors **on-demand** (no evictions), actors on **Turin** 32/64-vCPU
boxes with a fallback list, 2× PRO 6000 learner in FIN-03 (~48k samples/s, about 11 × 32 vCPUs of Turin actors):
~$9.20/h, ~$53 per billion, ~45–50 h for 7–8B samples within ~$450.
