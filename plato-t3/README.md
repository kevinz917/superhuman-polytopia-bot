# Plato T3

Everything specific to Plato T3 (update 96,070), the latest Plato variant and the bot the website serves. The guide
is [`docs/plato-t3.md`](../docs/plato-t3.md); the weights are in [`weights/plato-t3/`](../weights/plato-t3).

| Directory | Contents |
|---|---|
| `runtime/` | `engine_wasm.wasm` (the prebuilt engine T3 runs on) and the map boards `bank-<size>.json` |
| `engine/` | Source of that engine: `engine-rs`, `engine-wasm`, and the rules table it embeds. Separate from the repository's top-level `engine-rs/`, which the earlier Plato runtime builds against |
| `arena/` | `play.py` plays T3 against another bot from the command line; `arena.mjs` is the game loop; `summarize.py` prints the table; `modal_arena.py` is an optional Modal launcher |
| `tools/` | `export_browser_weights.py` (checkpoint `.npz` → browser `.bin`), `bin_to_npz.py` (the reverse) |
| `training/` | Source snapshot of T3's training line (model, observation version 6, trainer, C/CUDA simulator, recipe). For reading; not runnable on its own |

```bash
python3 plato-t3/arena/play.py --opponent weights/hero-1v1/final.npz --boards 2 --out runs/plato-t3/hero.json
```

Needs Node 20+ and NumPy. No game was played from these files for this release; see the guide for what was and was
not checked.
