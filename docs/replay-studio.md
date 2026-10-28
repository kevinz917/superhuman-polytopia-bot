# Bardur replay studio

A local mobile-style game interface for the frozen `bardur-1v1-v2` simulator. Humans, screenshot-driven models, structured policies, and replay playback share the same board renderer and versioned action boundary. The local opponent is the champion checkpoint in `weights/champion/` (self-play PPO from random weights, 519,267 parameters), optionally with leak-free turn search.

## Run it

```sh
(cd web && npm install && npm run build)
./scripts/studio.sh --search
```

Then open http://127.0.0.1:8765. The interface is a React + TypeScript + PixiJS (WebGL) app in `web/` (see `web/README.md`). Without a build, the server falls back to the original canvas page, which also stays available at `/legacy/`. Menu → Play the bot starts a fresh match; choose either seat. Menu → Watch self-play uses the checkpoint for both players. Menu → Replay library opens saved games and imports. Inference runs in the local Python process on CPU; nothing is sent to any training infrastructure.

First engine startup compiles CPU kernels and may take a minute or two (search compiles separately). The interface can load recorded games during startup. The service binds loopback only.

Options: `--checkpoint` (any `final.npz` with its adjacent `final.json`, whose SHA-256 is verified), `--storage` (default `artifacts/studio`), `--port`, `--search` and `--search-workers`, `--opponent NAME=PATH` (named rival checkpoints for watch games), and `--featured REPLAY_ID`.

## Graphics

The board draws placeholder shapes (`web/src/board/art/mock.ts`, `web/src/board/procedural.ts`); no game graphics ship with this repository.

## Recording clips

`tools/studio/record-clip.cjs` plays a library replay headlessly with Playwright and captures timestamped frames; `tools/studio/encode-clip.sh` turns them into the animated WebP and MP4 used in the README:

```sh
PLAYWRIGHT_MODULE=/path/to/node_modules/playwright STUDIO_URL=http://127.0.0.1:8765 \
  node tools/studio/record-clip.cjs <replay_id> /tmp/frames 900 600 8 900
zsh tools/studio/encode-clip.sh /tmp/frames assets/game-new 2 600
```

## Play and replay controls

- Tap a unit to select it; move (blue) and attack (red) markers draw above terrain. Tap the same tile again to select the city or tile beneath it, and once more to clear. A selected city outlines its territory and dims the rest of the map.
- While something is selected in live play, the bottom bar shows its stats and round order buttons with star costs (training, harvesting, building, abilities, capture), replacing Menu / Game Stats / Tech Tree / Next Turn as on mobile. Units your technology unlocks but you cannot train yet are greyed out.
- City level-ups open an overlaid reward choice (Workshop / Explorer, City wall / Resources, Population growth / Border growth, Park / Super unit). Tapping a technology opens its detail card with a Research button.
- Tech Tree shows owned and purchasable technologies. Next Turn lets the checkpoint play its complete turn, one visible transition at a time.
- Menu → All legal actions exposes every simulator action, including diplomacy and resignation. Pending city upgrades select their city automatically.
- Movement hops tile by tile along a path; training, construction and captures pop in with a puff; harvests lift the resource and fly population into the city bar, which fills; level-ups and captures burst with stars; fog lifts tile by tile, and the Explorer reward walks an explorer through the revealed land. The star counter ticks with a floating +/−, and the camera follows actions that happen off-screen. Melee attacks lunge, ranged units fire projectiles, and hits show flashes and floating damage. Retaliation follows the first strike, and defeated units fade into particles. These same transitions run in live play, self-play, and forward replay steps.
- Next bot actions wait for the current animation. Playback speed scales the animation duration. Seeking/backward steps cancel immediately. Menu → Animate moves and combat disables transitions; the OS reduced-motion preference is respected.
- Drag to pan, use a mouse wheel or pinch to zoom. Arrow keys step replays; Space toggles playback. Replay controls provide seek, speed, and perspective selection.
- Save/export creates `bardur-replay-v1` JSON. Live sessions persist private snapshots and frame journals and resume after server restart.

The bundled verified match contains 1,353 original logged actions and 1,354 frames, including its final board. Reconstruction checks every original public observation, legal-action record, transition termination/truncation, reward, and result count against the frozen engine. Policy selections are **not** rerun. The source is `match-champion51296-vs-longhorizon48-dev/300200-seat2.jsonl.gz` in the main research checkout; only a read is performed. Provenance records its compressed trace hash and original header.

Legacy JSONL traces can also be imported without reconstruction; they show only the recorded acting-player observations. Such traces lack a final post-action frame and are labeled accordingly. To reconstruct a complete trace, POST `/api/import` with base64 `data`, `name`, and `reconstruct: true`.

## Model integration

### Structured observations

`tools/studio/client.py` has no third-party dependencies. Supply a policy callback returning one of the legal action IDs:

```python
import sys
sys.path.insert(0, 'tools/studio')
from client import Client

client = Client()
client.start(seat=1)
# Replace select_action with your model inference.
client.run(select_action, max_decisions=1000)
```

`select_action(observation)` receives only its seat's public observation, memory, state version, legal actions, and game status. It does not receive hidden state, engine snapshots, opponent observations, or the random seed. Attach to an existing interface with `Client(game_id='...')` and refresh the browser to follow external actions.

### Actual UI / screenshot models

`tools/studio/ui-agent.cjs` launches the real page and repeatedly calls a model-driver module with a PNG screenshot, visible control rectangles, and instructions. The driver returns a click, key, wait, or stop. It operates the same DOM controls a person uses; it does not translate decisions into simulator actions behind the UI.

```sh
npm install --prefix artifacts/browser playwright
node tools/studio/ui-agent.cjs /absolute/path/to/model-driver.cjs
```

Driver contract:

```js
module.exports = async ({image, controls, instruction}) => {
  // image: PNG Buffer; call your chosen vision model here.
  // Return: {type:'click', x, y}, {type:'key', key:'Escape'},
  //         {type:'wait'}, or {type:'stop'}.
};
```

`CHROME_PATH`, `PLAYWRIGHT_MODULE`, `STUDIO_URL`, and `MAX_UI_STEPS` configure the runner. Model-provider credentials and inference calls belong to the supplied driver. No external vision model has been evaluated by this implementation. The checkpoint opponent is tested end to end.

The browser also exposes `window.bardur.observe()`, `.act(action_id, state_version)`, `.controls()`, `.refresh()`, `.selectTile(tile)`, `.game`, `.animating`, `.busy`, and `.whenIdle()`. The screenshot adapter waits until `.busy` is false before requesting a model decision. `.act()` advances the built-in opponent after your turn. Screenshots and the returned control rectangles reflect the current board camera.

### HTTP contract

All mutations require `X-Studio-Client: 1` and JSON. Engine work returns `{job_id}`; poll `/api/jobs/{id}` until `done` or `error`. Always preserve `expected_state_version`; stale, illegal, and out-of-turn actions are rejected without changing the game.

| Request | Purpose |
| --- | --- |
| `POST /api/games` `{mode,seat,seed?}` | Create `play`, `watch`, or `agent` game |
| `GET /api/games/{id}/observation` | Seat-scoped observation and legal catalog |
| `POST /api/games/{id}/action` `{action_id,expected_state_version}` | Apply one human/model action |
| `POST /api/games/{id}/advance` `{count:1..8}` | Run built-in opponent/self-play |
| `GET /api/games/{id}?frame=0&view=1` | Render a frame |
| `POST /api/games/{id}/save` or `/close` | Save replay or save and close |
| `GET /api/games/{id}/export` | Download replay JSON |
| `GET /api/library` | Saved matches and resumable sessions |
| `GET /api/replays/{id}?frame=0&view=all` | Seek a saved match |
| `POST /api/import` `{name,data,reconstruct?}` | Import base64 JSON or gzip trace |

A cold persisted session returns a restoration job with `restored: true`; repeat the original request after it completes. The Python client and browser handle this automatically.

## Visibility boundaries

Active human/agent games enforce their assigned seat even if a caller requests full-board or opponent views. Hidden views remain server-side. Opponent action parameters and policy metrics are redacted in both event locations. A separate `event.animation` cue may contain only move/attack kind and source/target tiles when both endpoints and the acting unit are already visible to the seat. Hidden attackers receive no such cue. Merely disappearing into fog is never interpreted as a death. Live exports contain only the assigned seat and omit the map seed. Completed games and explicit watch games can expose spectator views. HUD score/income are render-only additions; the trained policy receives its original unchanged observation schema.

This is a trusted single-user local tool. It is not an authenticated multi-user competition server.

## Mobile visual references and assets

Visual targets: the black star field, white Score / Stars (+income) / Turn HUD, snowy cyan Bardur terrain, faceted fog, health labels, city bars, and four circular Menu / Game Stats / Tech Tree / Next Turn controls.

- [Official App Store mobile screenshots](https://apps.apple.com/us/app/the-battle-of-polytopia/id1006393168)
- [Publisher public game graphics](https://polytopia.io/game-graphics/)
- [Publisher brand and fan-creation guidelines](https://polytopia.io/brand-assets-guide-lines/)
- [Portrait mobile reference](https://imag.malavida.com/mvimgbig/download-fs/the-battle-of-polytopia-23501-1.jpg)
- [Additional portrait reference](https://minireview.io/common/uploads/review/e47efed06444c7151d41ac189348b786.jpg)

The legacy page (`/legacy/`) can use 46 publisher sprite images: `tools/studio/mobile-assets.json` records their public source URLs and SHA-256 hashes, and `python tools/studio/setup.py` installs them into the ignored `artifacts/mobile-assets`; they are never committed or redistributed. The interface contains the unofficial-product notice. Midjiwan owns its graphics.

Four-seat replays (`four_replay_fn` in `experiments/modal_distributed_rl.py`) record every seat's own fog-of-war view per frame, so the replay dock offers Player 1-4 and God view (number keys 1-4, 5 for God view; 1, 2, 3 in two-seat games) and the insights panel covers all four seats.

This closely follows the mobile visual language, but is not a pixel-identical commercial client: fog, city assembly, typography, action panels, and exact animation timing differ. City names are cosmetic. Gameplay is the game's 2.17.3 rules for Bardur ([rules-alignment-2026-10-27.md](rules-alignment-2026-10-27.md)); it does not add all commercial tribes, map types, game modes, or effects. No audio is included.

## Validation

```sh
# Browser suites accept STUDIO_URL to target another server (default http://127.0.0.1:8765).
.venv/bin/python -m pytest tests/studio/test_visibility.py -q
.venv/bin/python tests/studio/api_checks.py
node tests/studio/browser.cjs
node --test tests/studio/animation.test.mjs
node tests/studio/animation-browser.cjs
node tests/studio/mobile-flow.cjs
npm --prefix web test
.venv/bin/python scripts/verify_ruleset.py
```

The API integration test saves `artifacts/restart-check.json`; after restarting, compare the restored observation and frame count against that fixture. Browser checks cover real tile moves, a checkpoint turn, technology controls, hidden-information restrictions, export, phone layout, and checkpoint self-play. Screenshot artifacts are stored under `artifacts/` for visual review. Tests use their own interactive-map seed range and never write to the research runs.

Animation validation covers real replay actions 4 (move), 171 (melee and counter), 172 (defeat), and 729 (ranged), plus seek cancellation, 8× playback, reduced-motion, and the animation setting. Public-observation tests cover hidden attackers and visible-only opponent cues. Browser tests also assert that a real human move produces a movement animation.
