# Legacy gameplay environment: ruleset v2 engine + browser bot

This folder preserves the **old gameplay environment** that the public site and the champion still use, next to the
current engine. It is a verbatim copy (`git read-tree`) of branch `claude/browser-bot` at commit `a6ab2ad`
(2026-10-24), also tagged `legacy/ruleset-v2-browser-bot`.

## Why it exists

- Since 2026-10-27 (commit 3ac7ce8) the live `engine-rs/` at the repository root implements **only** Polytopia
  2.17.3, the real game's rules; the research rulesets and every ruleset switch were removed. That engine is what
  every current training run, arena and ladder uses.
- The **champion** (scratch-cont-2 + fog head, observation v1) was trained and evaluated on **ruleset v2**
  (`docs/ruleset-v2.md`, the "Bardur 1v1 native" research rules). It only plays correctly on that engine.
- The public site (branch `external-website`, deployed on Vercel) runs the champion entirely in the browser:
  `web/public/bot/engine_wasm.wasm` (620 KB) is `engine-wasm/` here compiled against `engine-rs/` here, with the
  champion weights exported by `scripts/export_browser_weights.py`. Rebuilding that wasm needs this engine, not the
  root one.

## Layout

| Path | What |
|---|---|
| `engine-rs/` | The ruleset-v2 Rust engine (PyO3 bindings behind the `python` feature, off for the browser build). Independent crate; the root `engine-rs/` is the 2.17.3 engine. |
| `engine-wasm/` | `cdylib` crate: engine + policy network + leak-free turn search compiled to WebAssembly (`../engine-rs` path dependency, so keep the two folders side by side). |
| `web-local/` | The browser runtime that drives the wasm (`src/`: worker, engine bridge, search, store), its parity/arena/smoke tests, and the `browser:*` npm scripts of the time (`package-scripts.txt`). The deployable copy lives in `external-website:web/src/local`. |
| `scripts/export_browser_weights.py` | Champion `.npz` -> `champion.bin` + `champion.json` for the page. |
| `experiments/modal_browser_*.py` | Parity (9,695/9,695 positions) and arena checks of the wasm policy against the JAX policy, on Modal. |
| `browser-bot-2026-10-24.md` | The original write-up: build steps, sizes, parity and search results. |

## Build

```bash
cd legacy/ruleset-v2/engine-wasm
cargo build --release --target wasm32-unknown-unknown   # -> target/wasm32-unknown-unknown/release/engine_wasm.wasm
```

Nothing at the repository root depends on this folder, and nothing here depends on the root engine. Python
checkpoints of the research-rule bots still load as evaluation references on the current engine (their checkpoint
JSON carries the legacy ruleset digest), but to *play* them faithfully use this engine.
