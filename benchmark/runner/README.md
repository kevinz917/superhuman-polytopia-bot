# polytopia-bot

[Polytopia Skirmish](harbor/README.md) is a Harbor-format benchmark: one continuous
40-minute coding session to build a Rust bot, with on-demand public practice and
trace inspection, followed by independent held-out evaluation. It runs in the local
VM with host CLI-proxy inference, or Daytona with host OpenRouter inference. The [current protocol](docs/skirmish/CONTINUOUS.md)
replaces staged authoring tracks for the main showcase; earlier pilots remain archived.

**Status (2026-10-21).** Strongest bots, all from pure self-play PPO on the frozen `bardur-1v1` simulator
(details and every number in the [campaign log](docs/campaign-log-2026-10-19-20.md)):

| Board | Bot | vs previous best (deterministic, both seats) | Checkpoint |
|---|---|---|---|
| 14×14 | `c5-omni-s1` — c3 + 6 h with an [omniscient critic](docs/c5-omni-long-2026-10-21.md) | 77% of all games vs c3 (+255 Elo); +338–354 with [leak-free turn search](docs/omni-search-eval-2026-10-20.md) | `/retained/checkpoints/distributed-final-v1/c5-omni-s1-final/` |
| 18×18 | `c5-l18-omni` — grown c3 + [omniscient critic + turn clock](docs/c5-18x18-omni-2026-10-21.md) | 62% of all games / 76% of decisive vs grown c3 | `/retained/checkpoints/distributed-final-v1/c5-l18-omni-final/` |
| 14×14 (previous) | `c3-maps-ctl` u4854 — [map-variant self-play](docs/map-variants-2026-10-19.md) | 67–75% vs u3142 | `/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz` |

Inference-time [turn search](docs/test-time-search-2026-10-19.md) (`rl/search.py`, leak-free with
`determinize=True`, ~1 s/turn with `parallel=True` on 16 cores) adds +80–140 Elo to any of them. The
[research review](reports/Polytopia%20RL%20next%20experiments.md) relates the experiments to the literature and
lists the next ones. The older SFT/LLM line is unchanged below.

**Execution update (2026-10-16):** SFT-0 work has started. See the [execution log](docs/sft-execution.md) for the live adapter, public memory, reserved maps, qualification attempts and model profiling. No trained game checkpoint is claimed.

GPU-native Polytopia simulation, followed by language-model post-training.

An independent **structured-state PPO bot** now lives alongside the language-model
pipeline. It shares the frozen 1v1 simulator, uses masked action-family/source/target
heads, and supports an optional warm start from the existing demonstrations.
Start with the [current RL bot and inference guide](docs/structured-rl-current.md).
The normalized network, corrected PPO learner, public-API adapter and bounded
Modal self-play runs are working; gameplay strength is still being developed.
The [repair campaign](docs/structured-rl-repair-progress.md) records current
checkpoints, evaluations and the $500 budget. The earlier
[smoke](docs/structured-rl-smoke-2026-10-17.md) and
[64-update continuation](docs/structured-rl-league-result-2026-10-17.md) are
historical experiments whose failures are analyzed in the
[postmortem](docs/structured-rl-postmortem-2026-10-17.md).

Project initialized on 2026-10-16. Objective: supervised warm-start of a small pretrained language model, followed by direct reinforcement learning from Polytopia games.

**First deliverable: the GPU simulator.** Polyfish is the pinned rules reference and validation oracle. The simulator uses JAX device arrays and compiled, batched operations; the oracle is outside its runtime loop.

**The game is Polytopia's own rules (Steam 2.17.3, game version 123), the only ruleset (`polytopia-2.17.3`).** The Rust
engine `engine-rs` implements it and everything plays on it: training actors, arenas and ladders, the turn search,
Studio and the real-game bridge. It is checked against the real game on 27,785 recorded transitions and their legal
moves; see [rules-alignment-2026-10-27.md](docs/rules-alignment-2026-10-27.md). The earlier JAX research simulator
(`bardur-1v1-v1/v2`, [ruleset-v1.md](docs/ruleset-v1.md)) is retired; `gpu/` keeps only its state layout, importer,
map generator and rule table, and only the frozen expert-SFT line (`polytopia_bot.learning`) still runs it.

Run the local tests (JAX selects the CPU on this Mac):

```sh
uv venv
uv pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

Run the same tests and synchronized microbenchmarks on one NVIDIA L4 in Modal staging:

```sh
modal run --env staging experiments/modal_gpu.py
```

The final workspace suite passed **1,977 tests on CPU and 1,977 tests on NVIDIA L4**. Eight long procedural episodes also passed, totaling 5,710 legal transitions; all eight ended at the safety round limit. CPU and CUDA checks have source-specific receipts in [docs/benchmarks](docs/benchmarks/). Modal entrypoints reject non-staging environments and create bounded single-container jobs. The full test job has a 30-minute limit; the mixed benchmark and long-episode validation have separate bounded entrypoints. Measurements distinguish compilation, reset, microbenchmarks and mixed traces from LM throughput.

Rebuild the validation oracle and regenerate fixtures:

```sh
python3 scripts/bootstrap_oracle.py
PYTHONPATH=src .venv/bin/python scripts/generate_gpu_fixtures.py
```

Golden fixtures are checked in, so normal CPU/GPU tests do not require Rust or a CPU simulator. The oracle source and dependencies are pinned; third-party notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- [First SFT checkpoint plan](docs/sft-checkpoint-plan.md): full validation protocol from scripted demonstrations to a tested Qwen3-4B adapter.
- [Experiment plan](docs/experiment-plan.md): broader engine, SFT, RL, transfer, and compute proposals.
- [Environment contract](docs/environment-contract.md): GPU boundary, future policy transport and trajectory requirements.
- [Engine findings](docs/engine-findings.md): source inspection, pinned candidate revision, and unresolved fidelity questions.
- [GPU simulator](docs/gpu-simulation.md): implemented rules, design, parity scope and remaining milestones.
- [Pilot configuration](configs/pilot.json): proposed values, not an executable trainer configuration.
- [Compute calculator](scripts/estimate_compute.py): transparent workload estimates; no assumed GPU benchmark.
- [Python interface](src/polytopia_bot/environment.py): typed adapter boundary, without a substitute game implementation.

Run the dependency-free budget calculator:

```sh
python3 scripts/estimate_compute.py --games 10000 --decisions-per-game 600 --decisions-per-second 50 --gpus 8 --overhead 2
```

Here, decisions per game includes **both** LM-controlled players. Throughput is aggregate across rollout workers and includes prompt processing, decoding, and environment interaction. The overhead multiplier adds learning, synchronization, and evaluation; it must eventually be measured.

Model demonstrations, strategic bots, SFT and RL execution are downstream work. This phase stops after the frozen simulator, its validation receipts and the comprehensive post-training plan.

Minimal device-side use:

```python
import jax
import jax.numpy as jnp
from polytopia_bot.gpu.mapgen import reset
from polytopia_bot.gpu.rules import load_rules
from polytopia_bot.gpu.core import action, step
from polytopia_bot.gpu.kernels import observe
from polytopia_bot.gpu.actions import legal_masks

rules = load_rules()
state = jax.jit(lambda seed: reset(seed, size=14))(jnp.int32(7))
view = observe(state, state.current_player)
catalog = jax.jit(lambda s: legal_masks(s, rules, size=14))(state)
advance = jax.jit(lambda s, a: step(s, a, s.state_version, rules, size=14))
result = advance(state, action(10))  # End turn; choose a catalog action in a policy.
assert int(result.status) == 0
```

Use `jax.vmap` over reset/step for independent games. Keep full states and snapshots private; policies consume only the public view and legal catalog.

## Rust bot-writing benchmark

[Polytopia Skirmish](harbor/README.md) gives a coding agent 40 active minutes to build an offline Rust bot, with on-demand practice and trace inspection. One frozen final submission receives a seat-balanced score against a fixed opponent panel including hero. Peer Elo ladders are separate postprocessing.
The [pressure-study framework](docs/skirmish/pressure-study.md) adds controlled prompt framing, time and revision budgets, randomized independent trials and run-level comparisons.
