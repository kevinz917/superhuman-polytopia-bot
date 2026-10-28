# Bardur studio web app

The game interface for the local studio: React + TypeScript for the UI, PixiJS (WebGL) for the board, built with Vite.

```sh
npm install
npm run build   # writes web/dist, which the studio server serves at /
npm run dev     # http://127.0.0.1:5173 with hot reload; proxies /api and /assets to the studio on :8765
npm test        # animation planner unit tests (Node's built-in TypeScript support)
```

The original single-file interface is still served at `/legacy/`.

## Layout

- `src/board/BoardScene.ts`: the Pixi scene. The board is built in world units (one tile is 64 × 38.4), and the camera is a single transform on a render group, so panning and zooming never rebuild geometry. Content rebuilds only when the observed state changes. It renders on demand, not every frame.
- `src/game/controller.ts`: game flow (loading frames, selection, actions, bot turns, replay playback, and the `window.bardur` agent API). React subscribes to it, and it drives the board imperatively, so per-frame work never goes through React.
- `src/game/animation.ts`: the transition planner, built only from the two public observations.
- `src/ui/`: HUD, selection bar, reward popup, replay dock, dialogs and tech tree. Element IDs match the original page so the Playwright suites in `tests/studio/` run against either interface (`STUDIO_URL=...`).
