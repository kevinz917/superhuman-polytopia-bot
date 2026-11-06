// Arena of the in-browser bot: champion + turn search (engine-wasm search.rs, single-thread plan_turn: the same
// plans as the worker pool) against the raw deterministic champion, one game per call.
//   node web/tests/local_arena.ts <engine_wasm.wasm> <champion.bin> <seed> <search seat 1|2>
// Prints one JSON line: {seed, search_seat, rewards, turns, decisions, search_seconds, searched_turns}.
import { readFileSync } from "node:fs";
import { Engine } from "../src/local/engine.ts";

const [wasm, weights, seedArg, seatArg] = process.argv.slice(2);
const engine = await Engine.load(readFileSync(wasm));
const bytes = readFileSync(weights);
engine.loadModel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const seed = Number(seedArg), searchSeat = Number(seatArg);
const id = engine.newGame(seed);
let s = engine.status(id);
let queue: string[] = [], queueVersion = -1, decisions = 0, searchSeconds = 0, searchedTurns = 0;
while (!s.terminated && !s.truncated) {
  let action: string;
  if (s.current_player === searchSeat) {
    // As local/worker.ts: plan once per turn (re-plan if the queue no longer matches), replay one per step.
    if (!queue.length || queueVersion !== s.state_version) {
      const t = performance.now();
      queue = engine.plan(id, 16, (seed ^ s.state_version) >>> 0).plan;
      searchSeconds += (performance.now() - t) / 1000;
      searchedTurns++;
    }
    action = queue.shift()!;
  } else action = engine.act(id, true, decisions).action_id;
  s = engine.step(id, action);
  queueVersion = s.state_version;
  decisions++;
}
console.log(JSON.stringify({ seed, search_seat: searchSeat, rewards: s.rewards, turns: s.turn, decisions,
  search_seconds: Math.round(searchSeconds * 10) / 10, searched_turns: searchedTurns }));
