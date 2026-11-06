# Launch platforms: Modal and Curio

Both platforms run the same learner, actors, engine and rules from `main`. They differ only in the launcher and in a
handful of `LearnerConfig` knobs, kept in each platform's own config files. `LearnerConfig`'s defaults are the Modal
behaviour, so nothing written for Modal changes when Curio-only features are added; `tests/test_platform_profiles.py`
pins both profiles.

| | Modal (H100 / B200 / A10G, staging env) | Curio (8x RTX 4090 head + 2 EPYC servers) |
|---|---|---|
| Launcher | `experiments/modal_distributed_rl.py` (`launch`, `start_elo_ladder`, `start_fleet_keeper`, ...); `experiments/hero/launch.sh` | `experiments/curio/curio.py` (`check`, `publish`, `launch`, `status`, `stop`, `fetch`), keeper `experiments/curio/keeper.py` on the head |
| Configs | `experiments/hero/*.json` | `experiments/curio/configs/*.json` + `configs/cluster.json` (GPU/core layout) |
| Learner GPUs | all GPUs of the learner container | GPUs 4-7 train; GPUs 0-3 run the value pass + inference servers |
| Actors | separate A10G containers (GPU inference) or CPU containers | 4 GPU-inference actor nodes on the head's own CPUs |
| Ladders / search | Modal CPU functions | CPU servers, cores 256-511 |
| Checkpoints | Modal volumes `/results`, `/retained` | `/data/polytopia/{ckpt,runs,ladder}` on the head |
| Measured throughput (1v1, 4x128, omniscient critic) | ~208k samples/s live (Oct 28, H100s) | ~77k samples/s live (Oct 31) |

## Knobs that differ

| knob | Modal (default) | Curio | why |
|---|---|---|---|
| `send_thread` | `true` | `false` | +21% assembly on H100; hung multi-GPU learners on workstation GPUs (Daytona PRO 6000) |
| `stream_views` | `true` | `false` | DeviceRows page streaming deadlocked the 4-GPU allreduce on the 4090 head |
| `coded_views` | config's choice (`false` default) | `true` | actors send sparse row deltas; 66k vs 35k samples/s on Curio |
| `value_devices` | `0` | `4` | value pass + mirrored coded-view decode on the GPUs shared with inference (77k vs 66k) |
| `serialize_gpu` | `false` | `true` | one lock around GPU launches; how Curio's measured runs ran (no measurable cost) |
| `codec_shape_caps` | `false` | `true` | never-shrinking decode shapes; how Curio's measured runs ran (same views) |
| `unroll_loops` | `false` | `false` | tried on Curio, no gain |

Curio also sets process environment in `experiments/curio/common.py` (`LEARNER_ENV`):
`XLA_FLAGS=--xla_gpu_enable_command_buffer=` (CUDA graphs crash on driver 550 with JAX's CUDA 12.9 libraries), bf16
trunk, default NCCL transport. Modal's images set their own environment in `modal_distributed_rl.py`.

## What is shared on purpose

The game itself is the same everywhere: the 1:1 engine (`engine-rs`), the training round cap
(`ruleset.TRAINING_ROUND_LIMITS`, 200 rounds; the rules have none), and observation version 2 (unit facing) for new
networks. Old checkpoints keep their observation version and load on both platforms.

## Launching

Modal (example, as the Oct 26 hero runs): `experiments/hero/launch.sh run 1v1` after setting the parent, or
`modal run --detach --env staging experiments/modal_distributed_rl.py::launch --config-json "$(cat experiments/hero/hero-1v1.json)" ...`.

Curio: see `docs/curio-hero-2026-10-31.md`. In short:

```bash
python experiments/curio/curio.py check
python experiments/curio/curio.py publish
python experiments/curio/curio.py launch --run RUN --config hero-1v1-scratch.json --parent /data/polytopia/ckpt/... --no-critic --no-start-ref --hours 23
```
