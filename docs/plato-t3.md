# Plato T3 (update 96,070)

T3 is the latest variant of Plato, trained by Jack Yuan: the same 6×128 hybrid transformer
(`transformer-v3-p2-h4`, 3,243,386 parameters), continued from Plato update 88,890 onto board sizes 11 to 20, six
map types and opponents of every tribe, with the bot always playing Bardur. It is the bot the website serves as
"Plato". This page covers what is published, how to run it, how it was trained, how strong it measured, and what is
missing.

**Status of this release.** The files were exported and checked statically (hashes, a comment scrub that leaves
code unchanged, syntax). **No game was played from the exported files for this release**, the engine was not
rebuilt from the published source, and the adapted runner below has not been executed in its released form. The
strength numbers further down come from an earlier run on the same weights and engine binary.

## What is published

| Path | What |
|---|---|
| [`weights/plato-t3/`](../weights/plato-t3) | `plato.bin` (browser weight format) and its manifest. T3 only |
| [`plato-t3/runtime/`](../plato-t3/runtime) | `engine_wasm.wasm`, the prebuilt engine T3 runs on, and the map boards `bank-{11,14,16,18,20}.json` |
| [`plato-t3/engine/`](../plato-t3/engine) | Source of that engine: `engine-rs` (rules), `engine-wasm` (browser bindings and the network), `rules.json` |
| [`plato-t3/arena/`](../plato-t3/arena) | Headless runner: `play.py`, `arena.mjs`, `summarize.py`, and an optional Modal launcher |
| [`plato-t3/tools/`](../plato-t3/tools) | `export_browser_weights.py` (checkpoint → `.bin`) and `bin_to_npz.py` (the reverse) |
| [`plato-t3/training/`](../plato-t3/training) | Source snapshot of T3's training line: model, observation, trainer, simulator, recipe |
| [`plato-t3-source-manifest.json`](plato-t3-source-manifest.json) | Every exported file with source ref, source commit, hashes, and whether it was edited for release |

The T3 engine lives in its own directory instead of replacing the repository's [`engine-rs/`](../engine-rs). The
two have diverged: T3's engine adds tribes, board sizes, observation versions up to 6 and a public-view module, and
moved the Python bindings out of `lib.rs`. Replacing the top-level engine would change what the published Plato
(update 85,270) runtime, Studio and tests build against, so both are kept.

## Setup

Node 20 or newer runs the games. Python 3 with NumPy is needed only to convert an opponent checkpoint to the
browser format. Nothing has to be compiled: the prebuilt engine is used by default.

```bash
python3 -m pip install numpy
```

## Play a game

From the repository root. Each board is played twice with the seats swapped; both sides play Bardur.

```bash
python3 plato-t3/arena/play.py --opponent weights/hero-1v1/final.npz --boards 2 --out runs/plato-t3/hero.json
```

That converts hero-1v1 (already public, a 4×128 CNN) to the browser format, plays four games on the engine's
classic 14×14 dryland board, prints a win/draw/loss table and saves the rows. Other opponents and boards:

```bash
# the original Plato (update 85,270)
python3 plato-t3/arena/play.py --opponent weights/plato/final.npz --boards 2 --out runs/plato-t3/plato.json
# real-map-generator boards: 14x14, one board per map type (6 boards, 12 games)
python3 plato-t3/arena/play.py --opponent weights/hero-1v1/final.npz --panel real --size 14 --boards 1 --out runs/plato-t3/real14.json
# T3 against itself on 20x20 (the second side samples its moves)
python3 plato-t3/arena/play.py --opponent t3 --panel real --size 20 --boards 1 --out runs/plato-t3/mirror20.json
# print the table again from a saved file
python3 plato-t3/arena/summarize.py runs/plato-t3/hero.json
```

`--opponent` takes a checkpoint `.npz` (with its `.json` beside it), a browser `.bin`, or `t3`. A CNN opponent
plays 14×14 only. The exporter accepts `normalized-v2` CNNs at 14×14 (width 64 or 128, observation version 1) and
`transformer-v3-p2-h4` 6×128 networks. Use opponents trained on the `polytopia-2.17.3` ruleset; older checkpoints
such as `weights/champion` were trained on different rules.

How a game runs: one engine instance holds the true state. Before each turn the mover's weights are loaded; the
bot proposes a move from **its public view**, and the runner checks the proposal against the true legal list, as
the website does. A proposal that is not legal ends the turn and is counted under "refused proposals". T3 is
served with the `joint` decode (the single most likely whole move); CNN bots with `greedy`. A game that reaches
round 200 is a draw.

To run on Modal instead of your machine (optional, same options):

```bash
modal run plato-t3/arena/modal_arena.py --opponent weights/hero-1v1/final.npz --boards 2 --out runs/plato-t3/hero.json
```

## The engine

`plato-t3/runtime/engine_wasm.wasm` (973,799 bytes, SHA-256
`d5525b0cd2f774065cffd927c7ef5ce577813fe2e71378800887aa9737e7ffde`) is the binary the website ships. It was built
from the engine source at the private commit recorded in the source manifest. The source published in
`plato-t3/engine/` is that source with comments rewritten for release; every file has the same line count and the
same code outside comments (checked mechanically).

To build it yourself you need Rust with the `wasm32-unknown-unknown` target:

```bash
(cd plato-t3/engine/engine-wasm && cargo fetch --locked --target wasm32-unknown-unknown)
ALLOW_OTHER_RUSTC=1 plato-t3/engine/engine-wasm/scripts/build-site-wasm.sh /tmp/engine_wasm.wasm
python3 plato-t3/arena/play.py --opponent weights/hero-1v1/final.npz --wasm /tmp/engine_wasm.wasm --out runs/plato-t3/rebuilt.json
```

The script builds a clean copy at a fixed path with remapped paths, as the original did. Without
`ALLOW_OTHER_RUSTC` it insists on rustc 1.99.1, the toolchain of the shipped binary. A byte-identical rebuild from
the published source has **not** been attempted; treat the hash above as identifying the shipped binary, not as a
reproducibility claim for this tree.

## How it was trained

T3 continued Plato's training from update 88,890 to 96,070 with PPO self-play on the project's C/CUDA simulator.
The recipe is [`stage5-t3-g99.json`](../plato-t3/training/experiments/claude_v3/stage5-t3-g99.json):

| Setting | Value |
|---|---|
| Batch / minibatch / epochs | 524,288 rows / 32,768 / 1 |
| Learning rate | 4.243e-4 decaying to 1.414e-4 over the last quarter, 75 warm-up updates |
| Discount, GAE λ | 0.995, 0.99 |
| Turn clock | `turn_gamma` 0.99, `turn_lambda` 0.95 |
| Entropy coefficient, clip, target KL | 0.01, 0.2, 0.03 |
| Critic | privileged (reads both seats), critic-only updates until update 89,090 |
| Round cap | 200; a capped game is a loss for both sides (`truncation_reward` −1, no score reward) |
| Self-play / historical opponents | 0.5 / 0.1, a snapshot every 125 updates, 8 kept |

The recipe file alone is not the whole launch. Board sizes, per-size game counts and the league settings came from
the launch manifest and environment variables;
[`t3-harness.example.json`](../plato-t3/training/t3-harness.example.json) shows their shape, taken from a 16-GPU
qualification run with T3's inputs (sizes 11 to 20 grouped per process, learner on the Bardur seat, peace legal in
every game set, observation version 6, a learning-rate restart at update 88,890). The production launch manifest
and the opponent pool it used are not in the source repository.

The snapshot in `plato-t3/training/` mirrors the source repository's paths:

- `src/polytopia_bot/rl/transformer_v3.py`, `observation.py`, `actions.py`: the network and observation version 6
- `src/polytopia_bot/rl/gpu_resident.py`, `gpu_engine.py`, `distributed.py`: the GPU-resident PPO trainer, with
  mixed board sizes in one run
- `engine-cu/`: the C/CUDA simulator the trainer steps
- `experiments/claude_v3l/flagship_harness.py`, `experiments/claude_v3/gpu_arena.py`: league hooks and the paired
  arena

It contains only the files that changed between the Plato source already in this repository and T3's commit. It is
published for reading and as a starting point; it is **not** runnable end to end from this repository, for the
reasons in the next section.

## What is not included

- **The training checkpoint (`.npz`), the privileged critic and optimizer state.** Not available where this release
  was prepared. `plato.bin` holds the full policy in float32; `bin_to_npz.py` restores the policy arrays.
- **The map tables T3 trained on and the generator that made them.** They came from the project's full-game engine
  port, which is not published. The boards in `plato-t3/runtime/bank-*.json` are outputs of that generator.
- **The opponent pool checkpoints and the production launch manifest.**
- **The launch, supervision and budget layer** for T3 (`launcher.py`, `supervisor.py`, `modal_flagship.py`,
  `group.py`, `policy.py`, `make_manifest.py`, `stage_check.py`, the qualification reports). The stage-4 versions
  already in [`experiments/claude_v3l/`](../experiments/claude_v3l) show how it works.
- **The panel against the game's built-in AI** and the tests that depend on the full-game engine.
- **The website's interface code.** `arena.mjs` drives the engine the same way.
- **Later variants** (T4, T5) and the previous variant's (update 88,890) weights.

## Measured strength

One run, 12 November 2026, on this engine binary and these weights. Bardur against Bardur, every board played with
the seats swapped. Score is wins plus half of draws.

| Boards | T3 against | Score | Games |
|---|---|---|---|
| Classic 14×14 dryland | Ligma (the website's 4×128 CNN, update 5,933) | **.810** | 800 |
| Classic 14×14 dryland | Plato update 88,890 as served, sampled moves | .657 | 800 |
| Classic 14×14 dryland | Plato update 88,890 with the same most-likely-move decoding | .505 | 800 |
| Real-map-generator boards, 14×14 | Plato update 88,890 | .769 | 160 |
| Real-map-generator boards, 14×14 | Ligma | .750 | 44 |
| The project's older mixed-map boards, 14×14 | Ligma | **.366** (T3 loses) | 796 |

On classic boards T3 is level with the previous variant once both use the same decoding; its gain there comes
from how moves are picked. It is clearly ahead on the real-map-generator boards it was trained for, and it loses
to the older CNN on the project's earlier mixed-map boards. No games against humans were measured. Ligma and Plato
update 88,890 are not part of this release, so these rows cannot be regenerated from the published files; the
opponents you can run are the public checkpoints under `weights/`.

## Hidden-information caveat

An audit of the training line found that T3 was trained with legal-move masks computed from the **true** state:
enemy units the mover could not see could change which moves were offered. It also found two observation leaks,
since fixed in the engine. So T3 learned in a setting that gave it slightly more than a player sees. The published
engine and the website do not do this: the bot's candidates are computed from its public view
(`engine-rs/src/public.rs`, `engine-wasm/src/public.rs`), and a referee resolves each proposal against the real
state. The numbers above were measured that way, from the public view.

## Provenance

- Weights and prebuilt engine: the website's staged T3 release.
- Engine source: the engine branch that binary was built from.
- Training snapshot: the T3 training branch at the recipe's commit.

Refs and commits are in the [source manifest](plato-t3-source-manifest.json). Files were copied out individually
onto this repository's history; comments were rewritten to plain descriptions and private paths, coordination
notes and infrastructure names removed. Code outside comments is unchanged except two error-message strings
(listed in the manifest).
