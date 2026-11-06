// Smoke test of the in-browser engine + bot in Node: one deterministic self-play game of the champion.
//   node web/tests/local_engine_smoke.ts engine-wasm/target/wasm32-unknown-unknown/release/engine_wasm.wasm web/public/bot/champion.bin
import { readFileSync } from "node:fs";
import { Engine } from "../src/local/engine.ts";

const [wasm, weights, seedArg] = process.argv.slice(2);
const engine = await Engine.load(readFileSync(wasm));
const bytes = readFileSync(weights);
engine.loadModel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const id = engine.newGame(Number(seedArg ?? 1810000001));
let status = engine.status(id);
const view = engine.view(id, 1) as { visible_state: { tiles: unknown[]; units: unknown[] } };
console.log("start", status, "tiles", view.visible_state.tiles.length, "units", view.visible_state.units.length,
  "legal", engine.legal(id).length);
let decisions = 0;
const started = performance.now();
let actTime = 0;
while (!status.terminated && !status.truncated && decisions < 6000) {
  const t0 = performance.now();
  const d = engine.act(id, true, decisions);
  actTime += performance.now() - t0;
  status = engine.step(id, d.action_id);
  decisions++;
}
console.log("end", status, "decisions", decisions, `act ${(actTime / decisions).toFixed(2)} ms/decision`,
  `total ${((performance.now() - started) / 1000).toFixed(1)} s`);
