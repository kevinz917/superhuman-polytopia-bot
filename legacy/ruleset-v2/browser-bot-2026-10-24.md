# The champion in the browser (2026-10-24)

The game engine, the champion policy and its turn search now run entirely in the browser: no Studio server,
no Modal, no inference cost. The page loads a 620 KB WebAssembly module and the 2 MB champion weights once,
then every move, bot decision and search is computed on the player's machine.

## Run it

```bash
cd web
npm run browser:wasm                                            # engine-wasm -> public/bot/engine_wasm.wasm
npm run browser:model -- path/to/final.npz                      # champion -> public/bot/champion.bin (+ .json)
npm run browser                                                 # http://localhost:5174
```

`npm run browser:build` writes a static site to `web/dist-browser/` (3.6 MB) that any static host can serve.
`browser:model` needs a Python with NumPy (the repo's `.venv`). The weights come from the open-source branch (`weights/champion/final.npz`, scratch-cont-2 + fog head,
SHA-256 `90618f5a…`). Sprites come from `artifacts/mobile-assets` when installed (dev server only); without
them the board draws its procedural shapes.

## How it is built

| Piece | Where | What |
|---|---|---|
| Engine | `engine-rs` (unchanged rules) | PyO3 bindings moved to `src/python.rs` behind the default `python` feature, so the crate also builds for `wasm32-unknown-unknown` |
| WASM module | `engine-wasm/` | C ABI over engine-rs: games, legal actions, views, steps; the policy network (`net.rs`, a CPU port of `rl/network.py` normalized-v2); features from the packed view exactly as training (`view.rs`); turn search (`search.rs`) |
| Weights | `scripts/export_browser_weights.py` | `final.npz` -> flat float32 file in checkpoint leaf order |
| Game worker | `web/src/local/worker.ts` | Answers the Studio API (`/api/...`, same JSON as `studio/service.py`) in a Web Worker, so the React UI is unchanged; games persist in IndexedDB as seed + actions and every frame is rebuilt on demand |
| Search workers | `web/src/local/search.ts` | Lanes of each plan round spread over `min(cores - 1, 8)` workers |
| Switch | `web/src/api/client.ts` | `vite --mode browser` (`VITE_BACKEND=local`) routes `api()` to the worker; the server build is untouched |

Search is the leak-free depth-1 searcher (`TurnSearcher(lanes=16, depth=1, determinize=True)`): lanes play
candidate turns on a world sampled from the bot's own view (`rl/belief.py` ported), one sampled opponent reply
each, scored by the fog-of-war value head; the best turn is committed up to the first divergence from the
sample. The public site serves the 64-reply robust variant; one reply was within noise of it on OMNI
(docs/search-breadth-2026-10-21.md) at a third of the cost.

## Checks

- **Policy parity with JAX** (`experiments/modal_browser_parity.py` + `web/tests/local_parity.ts`): 8 full games
  of the JAX champion on engine-rs, replayed in the WASM build: **9,695 / 9,695 decisions identical**, max
  |value| difference 1.5e-6, max |log p| difference 4.9e-6, same final rewards.
- **Parallel search = single-thread search**: 12 / 12 positions gave identical plans.
- **Speed** (M-series Mac, Chrome-based pane): raw policy 1.4 ms per decision (a 1,072-decision self-play game in
  1.5 s); search 0.44 s per searched turn on average (max 1.3 s) over a full game on 8 workers; frame scrubbing
  2-3 ms.
- **Strength** (`experiments/modal_browser_arena.py`, 200 seeds x both seats, seeds 396500+, WASM under Node on
  Modal): champion + browser search vs raw champion **215-185 (53.8%, +26 Elo, 95% CI -8 to +61)**. Seat 1 won
  67% of games either way. Positive but not significant at 400 games; the server's 64-reply search measured
  +47/+53 on the champion (docs/scratch-search-2026-10-23.md), inside this interval. The arena ran 10.2 s per
  searched turn on one Modal vCPU, about 10x slower than one Mac core.
- Web tests (12), `tsc -b`, oxlint and the static build pass.

## Limits

- 14x14, normalized-v2 2x64 checkpoints only (the export checks).
- Named rival opponents and trace reconstruction need the Studio server; importing a `bardur-replay-v1` file works.
- The whole game state lives in the page, so fog of war is only as private as the player's devtools.
