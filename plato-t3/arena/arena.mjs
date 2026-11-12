// Headless bot-vs-bot matches on the browser engine (engine_wasm.wasm) and browser weights (.bin).
//
//   node arena.mjs <assets dir> <specs.json> <out.json>
//
// <assets dir> holds engine_wasm.wasm, one <bot>.bin per bot named in the specs, and bank-<size>.json for the board
// sizes the specs use. play.py builds that directory and the specs; see docs/plato-t3.md.
//
// The mover's weights are loaded into the one engine instance before each of its turns. It proposes with the decode
// named for that bot from its public view, and the proposal is refereed against the true legal list, as the site's
// worker does (an impossible proposal ends the turn).
import fs from "node:fs";
import path from "node:path";

const [assets, specsPath, outPath] = process.argv.slice(2);
if (!assets || !specsPath || !outPath) {
  console.error("usage: node arena.mjs <assets dir> <specs.json> <out.json>");
  process.exit(2);
}
const wasmBytes = fs.readFileSync(path.join(assets, "engine_wasm.wasm"));
const decoder = new TextDecoder(), encoder = new TextEncoder();
const MODES = { sampled: 0, greedy: 1, joint: 2 };
const MAX_STEPS = 60000;

class Engine {
  constructor(x) { this.x = x; }
  static async load() {
    const { instance } = await WebAssembly.instantiate(wasmBytes, {});
    return new Engine(instance.exports);
  }
  out(code) {
    const bytes = new Uint8Array(this.x.memory.buffer, this.x.output_ptr(), this.x.output_len());
    const value = JSON.parse(decoder.decode(bytes));
    if (code < 0) throw Error(value.error);
    return value;
  }
  write(json) {
    const bytes = encoder.encode(JSON.stringify(json));
    const ptr = this.x.input(bytes.length);
    new Uint8Array(this.x.memory.buffer, ptr, bytes.length).set(bytes);
  }
  loadModel(weights) {
    const ptr = this.x.input(weights.byteLength);
    new Uint8Array(this.x.memory.buffer, ptr, weights.byteLength).set(weights);
    return this.out(this.x.load_model()) ?? {};
  }
  newGame(seed, layout) {
    if (!layout) return this.x.game_new(seed);
    this.write(layout);
    return this.out(this.x.game_new_layout(seed)).id;
  }
  free(id) { this.x.game_free(id); }
  status(id) { return this.out(this.x.game_status(id)); }
  legal(id) { return this.out(this.x.game_legal(id)); }
  step(id, actionId) { return this.out(this.x.game_step(id, Number(actionId))); }
  act(id, mode, seed) { return this.out(this.x.bot_act(id, MODES[mode], seed >>> 0)); }
  get publicProjection() { return typeof this.x.bot_public_candidates === "function" ? this.x.bot_public_candidates() : 0; }
}

const cache = new Map();
const json = (name) => { if (!cache.has(name)) cache.set(name, JSON.parse(fs.readFileSync(path.join(assets, name)))); return cache.get(name); };

// Board sources: "dryland" is the engine's own classic 14x14 dryland board for a seed (two Bardur seats); "bank" is
// a board of the published map bank (the real game's map generator, sizes 11-20, six map types), Bardur vs Bardur.
function board(spec) {
  if (spec.source === "dryland") return { seed: spec.seed, layout: undefined };
  if (spec.source === "bank") {
    const b = json(`bank-${spec.size}.json`).filter((x) => x.tribes[0] === 4 && x.tribes[1] === 4 && x.map === spec.map)[spec.index];
    if (!b) throw Error(`no bank board ${spec.size}/${spec.map}/${spec.index}`);
    return { seed: b.seed, layout: { size: b.size, tribes: b.tribes, terrain: b.terrain, structure: b.structure, resource: b.resource, capitals: b.capitals } };
  }
  throw Error(`unknown board source ${spec.source}`);
}

// One engine instance holds the game; the mover's weights are loaded before its turn (load_model also sets the
// mover's diplomacy rule: the Plato line was trained with peace treaties legal, the older CNN bots without, so
// those never propose or accept one). One state, so the two sides cannot disagree about it.
const ROUND_CAP = 200; // layout games have no round limit in this engine; the training harness capped at 200
const weights = new Map();
const bytes = (bot) => { if (!weights.has(bot)) weights.set(bot, fs.readFileSync(path.join(assets, `${bot}.bin`))); return weights.get(bot); };
let engine = null, loaded = null, info = {}, loads = 0, loadMs = 0;
function use(bot) {
  if (loaded === bot) return;
  const t0 = performance.now();
  info[bot] = engine.loadModel(bytes(bot));
  loaded = bot;
  loads++;
  loadMs += performance.now() - t0;
}

async function play(spec) {
  if (!engine) engine = await Engine.load();
  const { seed, layout } = board(spec);
  const g = engine.newGame(seed, layout);
  const refused = [0, 0], decisions = [0, 0], ms = [0, 0];
  let s = engine.status(g), steps = 0, error = null;
  const loads0 = loads, loadMs0 = loadMs;
  try {
    while (!s.terminated && !s.truncated && s.turn <= ROUND_CAP && steps < MAX_STEPS) {
      const k = s.current_player - 1;
      use(spec.bots[k]);
      if (spec.decodes[k] === "joint" && info[spec.bots[k]].joint !== true) throw Error(`${spec.bots[k]} has no joint decode on this engine`);
      const sampled = spec.decodes[k] === "sampled";
      const t0 = performance.now();
      const d = engine.act(g, spec.decodes[k], sampled ? (Math.imul(spec.sample_seed, 0x9e3779b1) ^ s.state_version) >>> 0 : s.state_version);
      ms[k] += performance.now() - t0;
      decisions[k]++;
      const legal = engine.legal(g);
      let id = String(d.action_id);
      if (!legal.some((a) => String(a.action_id) === id)) {
        const end = legal.find((a) => a.command.kind === 10);
        if (!end) throw Error("refused proposal and no end-turn");
        id = String(end.action_id);
        refused[k]++;
      }
      s = engine.step(g, id);
      steps++;
    }
  } catch (err) {
    error = String(err.message ?? err);
  }
  engine.free(g);
  const r = s.rewards ?? [];
  const outcome = error ? "error" : steps >= MAX_STEPS ? "step_cap" : s.terminated ? "terminated" : "truncated";
  return {
    id: spec.id, outcome, error, rewards: [r[0], r[1]], turn: s.turn, steps, refused, decisions,
    ms_per_decision: ms.map((m, k) => (decisions[k] ? Math.round((m / decisions[k]) * 100) / 100 : null)),
    model_loads: loads - loads0, ms_per_load: loads > loads0 ? Math.round(((loadMs - loadMs0) / (loads - loads0)) * 100) / 100 : null,
    info: spec.bots.map((b) => ({ obs_version: info[b]?.obs_version ?? null })), projection: engine.publicProjection,
    panel: spec.panel, board: spec.board, bots: spec.bots, decodes: spec.decodes,
  };
}

const specs = JSON.parse(fs.readFileSync(specsPath));
const results = [];
for (const spec of specs) {
  const t0 = performance.now();
  let row;
  try {
    row = await play(spec);
  } catch (err) {
    row = { id: spec.id, outcome: "error", error: String(err.message ?? err), panel: spec.panel, board: spec.board, bots: spec.bots, decodes: spec.decodes };
  }
  row.seconds = Math.round((performance.now() - t0) / 100) / 10;
  results.push(row);
  fs.writeFileSync(outPath, JSON.stringify(results));
  console.error(`${results.length}/${specs.length} ${row.id} ${row.outcome} rewards ${JSON.stringify(row.rewards ?? null)} turn ${row.turn ?? "-"} ${row.seconds}s`);
}
