# PufferLib re-evaluation (2026-10-20)

**Verdict: no.** PufferLib 5.0 is a C/CUDA-only stack whose trainer requires the environment to be rewritten in C, exposes flat observations to a MinGRU/MLP network, and has no GPU-environment self-play. Adopting it means discarding the frozen JAX engine (1,977 tests, bitwise CPU/GPU parity), the spatial CNN with masked autoregressive heads, and the Modal actor/learner setup, for a speedup we can get more cheaply by porting only the engine step. It has never trained a 4X or turn-based strategy game comparable to Polytopia; its published strength results are arcade "solves", a 9×9 Go bot against a greedy scripted opponent, and a chess self-play setup evaluated against Maia. A few of its ideas are worth borrowing (listed at the end).

Sources: puffer.ai homepage and docs (fetched 2026-10-20), the `5.0` branch of PufferAI/PufferLib (`src/pufferl.cu`, `config/default.ini`, `ocean/*`), the 3.0/4.0 blog posts, the RLC 2025 paper, and a third-party migration audit (m2kulkarni/PufferWelfare PR 7).

## What PufferLib is today

| Item | State on 2026-10-20 |
|---|---|
| Version | `5.0` branch is the default (4,297 commits); no GitHub releases, PyPI stable is still 3.0. 4.0 shipped April 2026. 5.0 is described as "a native C/CUDA redesign, not a package update". |
| Trainer | ~10k lines of CUDA C (PuffeRL). PPO variant with V-trace option, sync or 1-epoch async (CleanBa-style), bf16 by default with fp32 master weights, CUDA graphs, static allocator, bitwise deterministic when seeded. Torch backend kept only as fallback. |
| Network | PufferNet = MinGRU recurrence + highway residuals over a **flat** observation vector. No conv encoder in the trainer. A custom encoder means writing CUDA (`breakout.cu` is the example). |
| Actions | Multi-discrete heads (`ACT_SIZES {a, b, ...}`) with a per-head legal mask written by the env each step. Heads are sampled **independently**; there is no conditional/autoregressive masking. Chess works around this with pick-then-place phases (two env steps per move). |
| Environments | Must be C (`ocean/<env>/<env>.h` with `puf_init/reset/step`). Optional CUDA envs (`--cu`), but "GPU env backend does not support selfplay or multi-policy". All Python/Gymnasium/third-party wrappers were removed ("100x+ slower"). |
| Self-play | `[selfplay]` section: up to 8 frozen historical policies drawn from a pool of ≤16 of the current run's checkpoints, `hist_policy_percent` of env slots play them, resampled every `opp_timeout_steps`. Separate `league.py` with anchored Elo. Checkpoints store weights only; historical arch is fixed to MinGRU at the audited pin. |
| Sweeps | Protein (GP + genetic over wall-clock/score Pareto front), multi-GPU sweeps, also tunes reward coefficients. |
| Hardware | NVIDIA CUDA required to train; CPU is eval-only. ROCm hipify and Metal backends are in-progress PRs. |
| Throughput claims | 3.0: 3–5M sps; 4.0: ~15M sps standard model, >20M small model, on Breakout on one RTX 5090; docs headline "up to 60M sps". These are tiny-observation arcade envs with tiny nets. |
| License / company | MIT. PufferAI sells custom env building and support ("contracts from $10k/month"). Discord-first support, PRs reviewed on stream. |

## What it has trained on

The `ocean/` folder on `5.0` holds 85 environments. By category:

- **Arcade / control** (the headline results): breakout, pong, enduro, freeway, space_invaders, tetris, snake, 2048, pacman, flappy, asteroids, cartpole, double_pendulum, drone, docking, robot_arm, mujoco bindings.
- **Two-player board games with self-play**: go (9×9, scripted `enemy_greedy_hard` by default, self-play optional), chess (3.5k lines of C, self-play with multi-bank frozen opponents, Maia-based eval), checkers, guerrillacheckers, backgammon, hex, connect4, tron, slimevolley, chain_reaction, soccer, robocode, admiral, tripletriad.
- **Multi-agent sims**: nmmo3 (Neural MMO 3, trained on "a petabyte of compressed data" on one server), moba (5v5 on 128×128; **scripted Dire team, no self-play, shaped XP/tower/death rewards, no win reward**), overcooked, rware, boids, ants, trash_pickup, battle, tactical, tower_defence, terraform, drive (PufferDrive), osrs variants.
- **Puzzle / research probes**: boxoban, maze, memory, tmaze, chain_mdp, lightsout, laser_puzzle, nethack, craftax (hybrid: reset in C, step proxied to JAX).

Nothing in the suite is a 4X: no economy plus exploration plus fog of war plus city building plus unit production in one turn-based game. The closest analogues are chess and admiral, which are perfect-information board games with small flat observations. Published results for the board games are internal (chess logs PGNs and scores per colour; go logs win rate against the greedy bot). The "tiny super-human models in seconds" claim refers to arcade environments.

## Fit against our stack

| Dimension | Ours | PufferLib 5.0 | Consequence |
|---|---|---|---|
| Engine | JAX, rules-table driven (`gpu/`, ~2.6k lines + 30 KB `rules.json`), 1,977 tests, frozen `bardur-1v1-v1`, bitwise CPU/GPU parity via `cpu_engine.py` | Hand-written C per env | Full rewrite of the frozen ruleset in C. Chess took 3.5k lines; Polytopia (movement/ZOC/naval/combat/economy/tech/diplomacy/cloaks/ruins/map gen) is several times larger. Every parity receipt would need rebuilding. |
| Observation | 14×14 or 18×18 board, ~60 channels, plus global vector | Flat vector into MinGRU/MLP | Our CNN inductive bias is lost unless we write a CUDA conv encoder ourselves. |
| Action | Masked autoregressive (family → source → target), masks conditional on earlier heads | Independent multi-discrete heads, one static mask per head | Needs three env phases per decision (as chess does with pick/place), tripling steps and changing the training distribution; or a flat joint space (17 × 324 × 324 ≈ 1.8M) which is infeasible. |
| Self-play | Historical pool + frozen external rivals (league96, diverse bots), per-seat GAE, evidence-gated promotion, Elo ladder on Modal | Pool of own-run checkpoints only (≤16), opponent arch fixed, swap by step count, no external frozen rivals, anchored Elo module | Loses external-rival training and our promotion gating; would need C-side changes to the trainer. |
| Distributed | CPU actor containers (32 cores) streaming to a 2×H100 learner over Modal tunnels; actors 67% of cost | Single box: OpenMP env threads + GPU on one machine, NCCL multi-GPU | Modal's largest CPU box is 64 cores, so a single-node design caps env throughput at one container's worth unless we re-add our own actor fabric. |
| Numerics | fp32 HIGHEST with a parity gate for behaviour logp | bf16 default, `--float` for testing | Our 1e-5 mismatch gate would need re-establishing. |
| Precision | Deterministic replays and trace hashes are part of acceptance | Deterministic when seeded | Comparable. |

## Where it would actually help

Our 18×18 diagnosis found the actor tick is ~60% engine, ~40% policy forward, with `build_mask` at ~45% of the engine (n² scaling), and 32-vCPU Modal containers at ~140 steps/s per vCPU. A C step for Polytopia would plausibly be 20–100 µs instead of 2–3 ms on Modal CPUs, which is the one place PufferLib's philosophy pays. That gain comes from writing the engine in C, not from the trainer or the library. If we want it, the cheaper path is a C or Numba step behind the existing `GPUEnvironment` contract, verified against the frozen JAX engine with the same bitwise tests used for `cpu_engine.py`, feeding our own learner. Even that is a multi-week port of a ruleset we deliberately froze, and the learner is currently the bound on 18×18.

## Ideas worth borrowing without adopting the library

- **MinGRU recurrence** for fog-of-war memory: parallel over time, cheaper than an LSTM, and the authors report it matches or beats LSTM on every env tested. We currently have no recurrence.
- **Protein-style reward and hyperparameter sweeps on the time/score Pareto front**, including tuning reward coefficients (our unit-diversity KL and entropy terms were hand-set).
- **Async learner/actor pipelining with a one-epoch-stale actor** (CleanBa); our learner already never waits, so the benefit is smaller for us.
- **Anchored Elo league with per-run observation refresh**, close to what `rl/elo.py` and the ladder already do.
- **Per-core pinned env processes and pinned host memory**, which we already use on actors.

## When to revisit

Revisit only if one of these changes: we start a new game from scratch where a C engine is the natural first implementation; PufferLib's GPU-env path gains self-play and a conv encoder; or we decide to rewrite the engine in C anyway for the browser/Studio and the trainer comes for free. None applies now.
