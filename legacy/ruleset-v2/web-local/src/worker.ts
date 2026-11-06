// The Studio API (src/polytopia_bot/studio/service.py) answered inside the browser: engine-wasm runs the game and
// the champion policy, so a game needs no server. The interface talks to this worker through local/backend.ts
// with the same paths and JSON as the HTTP server. Games are rebuilt from seed + actions (store.ts).
import type { Command, GameEvent, Observation, Rules, SeriesPoint } from "../api/types.ts";
import { Engine, type Status } from "./engine.ts";
import { ABILITY, actionLabel, RESOURCE, REWARD, TERRAIN } from "./labels.ts";
import { SearchPool } from "./search.ts";
import { store, type GameRecord } from "./store.ts";

// A dedicated worker scope (the app compiles against the DOM library, not WebWorker).
const scope = self as unknown as { postMessage(message: unknown): void; onmessage: ((e: MessageEvent) => void) | null };

const SCHEMA = "bardur-replay-v1";
const RULESET = "bardur-1v1-v2";
const SHORT = "Champion";
// Leak-free d=1 turn search, one sampled opponent reply per candidate turn (TurnSearcher(lanes=16, depth=1,
// determinize=True)): averaging 64 replies was not measurably stronger at ~3x the cost
// (docs/search-breadth-2026-10-21.md).
const SEARCH = { lanes: 16, depth: 1, replies: 0, determinize: true };
const CHECKPOINT_EVERY = 64;

class Conflict extends Error {}

type Views = Record<string, Observation>;

interface Live {
  rec: GameRecord;
  game: number; // engine id of the state after the last frame
  checkpoints: Map<number, number>; // frame index -> engine id of a copy of that state
  cache: Map<number, Views>;
  queue: string[];
  queueVersion: number;
  error: string | null;
}

let engine: Engine;
let rules: Rules;
let pool: SearchPool;
let bot: Record<string, unknown>;
const live = new Map<string, Live>(); // open games and opened replays, by "<kind>:<id>"
const records = new Map<string, GameRecord>();
const key = (kind: GameRecord["kind"], id: string) => `${kind}:${id}`;

const ready = (async () => {
  const [wasm, weights, meta] = await Promise.all([
    fetch("/bot/engine_wasm.wasm").then((r) => r.arrayBuffer()),
    fetch("/bot/champion.bin").then((r) => {
      if (!r.ok) throw Error("Model weights missing: run npm run browser:model");
      return r.arrayBuffer();
    }),
    fetch("/bot/champion.json").then((r) => r.json()),
  ]);
  engine = await Engine.load(wasm);
  engine.loadModel(weights);
  rules = engine.rules();
  pool = new SearchPool(wasm, weights);
  bot = {
    name: `Bardur · ${SHORT}`,
    training: "Self-play PPO from random weights (scratch-cont-2), running in this browser",
    checkpoint_sha256: meta.source_sha256,
    mode: "deterministic",
    parameters: meta.parameter_elements,
    ruleset: RULESET,
    search: { ...SEARCH, workers: pool.size },
  };
  for (const r of await store.all()) records.set(key(r.kind, r.id), r);
})();

const uid = () => crypto.randomUUID().replace(/-/g, "");

// ---------- frames ----------

/** Engine state of a record, rebuilt from its seed and actions, with a copy every CHECKPOINT_EVERY frames. */
function materialize(rec: GameRecord): Live {
  const k = key(rec.kind, rec.id);
  const existing = live.get(k);
  if (existing) return existing;
  const game = engine.newGame(rec.seed);
  const checkpoints = new Map<number, number>([[0, engine.clone(game)]]);
  rec.actions.forEach((a, i) => {
    engine.step(game, a);
    if ((i + 1) % CHECKPOINT_EVERY === 0) checkpoints.set(i + 1, engine.clone(game));
  });
  const l: Live = { rec, game, checkpoints, cache: new Map(), queue: [], queueVersion: -1, error: null };
  live.set(k, l);
  return l;
}

function release(l: Live) {
  engine.free(l.game);
  for (const id of l.checkpoints.values()) engine.free(id);
  live.delete(key(l.rec.kind, l.rec.id));
}

/** Both seats' views and the spectator view of frame `index` (service.frame). */
function views(l: Live, index: number): Views {
  const cached = l.cache.get(index);
  if (cached) {
    l.cache.delete(index);
    l.cache.set(index, cached);
    return cached;
  }
  const last = l.rec.frames.length - 1;
  let game = l.game;
  let temporary = false;
  if (index !== last) {
    const base = Math.floor(index / CHECKPOINT_EVERY) * CHECKPOINT_EVERY;
    game = engine.clone(l.checkpoints.get(base)!);
    temporary = true;
    for (let i = base; i < index; i++) engine.step(game, l.rec.actions[i]);
  }
  const v: Views = { "1": engine.view(game, 1), "2": engine.view(game, 2), all: engine.view(game, 0) };
  if (temporary) engine.free(game);
  l.cache.set(index, v);
  if (l.cache.size > 48) l.cache.delete(l.cache.keys().next().value!);
  return v;
}

function over(l: Live) {
  const f = l.rec.frames[l.rec.frames.length - 1];
  return f.terminated || f.truncated;
}

function status(l: Live): Status {
  return engine.status(l.game);
}

// ---------- persistence ----------

const pending = new Map<string, ReturnType<typeof setTimeout>>();
function persist(rec: GameRecord, now = false) {
  const k = key(rec.kind, rec.id);
  clearTimeout(pending.get(k));
  pending.delete(k);
  records.set(k, rec);
  if (now) return store.put(rec);
  pending.set(k, setTimeout(() => (pending.delete(k), store.put(rec)), 1500));
}

// ---------- the Studio service ----------

function describe(l: Live) {
  const r = l.rec;
  const f = r.frames[r.frames.length - 1];
  return {
    id: r.id, title: r.title, mode: r.mode, seat: r.seat, frames: r.frames.length, current_player: f.current_player,
    terminated: f.terminated, truncated: f.truncated, bot, busy: false, error: l.error, search: r.search,
    opponent: null, opponent_seat: 2,
  };
}

function create(data: Record<string, unknown>) {
  const mode = String(data.mode ?? "play");
  if (!["play", "watch", "agent"].includes(mode)) throw Error("mode must be play, watch or agent");
  if (data.opponent) throw Error("Named opponents need the Studio server");
  const seat = Number(data.seat ?? 1);
  if (seat !== 1 && seat !== 2) throw Error("seat must be 1 or 2");
  const seed = data.seed === undefined ? 1810000000 + Math.floor(Math.random() * 1000000) : Number(data.seed);
  if (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 31) throw Error("seed outside int32 range");
  const search = Boolean(data.search ?? true) && mode !== "agent";
  const id = uid();
  const rec: GameRecord = {
    id, kind: "game", title: mode === "watch" ? `${SHORT} self-play` : `You vs ${SHORT}`,
    mode: mode as GameRecord["mode"], seat, seed, search, created: Date.now() / 1000, actions: [], events: [null], frames: [],
  };
  const game = engine.newGame(seed);
  const s = engine.status(game);
  engine.free(game);
  rec.frames.push({ current_player: s.current_player, turn: s.turn, terminated: s.terminated, truncated: s.truncated });
  materialize(rec);
  persist(rec, true);
  return { game_id: id, mode, seat, search, opponent: null, opponent_seat: 2 };
}

function apply(l: Live, actionId: string, source: string, metrics: Record<string, unknown> | null) {
  const before = status(l);
  let command: Command;
  let after: Status;
  try {
    command = engine.decode(l.game, actionId);
    after = engine.step(l.game, actionId);
  } catch (e) {
    throw new Conflict((e as Error).message);
  }
  const reason = after.terminated ? (command.kind === 11 ? "resignation" : "elimination") : after.truncated ? "research_horizon" : null;
  const event = {
    player: before.current_player, turn: before.turn,
    action: { action_id: actionId, command, label: actionLabel(command, rules) },
    source, metrics, rewards: { 1: after.rewards[0], 2: after.rewards[1] }, reason,
  } as unknown as GameEvent;
  const r = l.rec;
  r.actions.push(actionId);
  r.events.push(event);
  r.frames.push({ current_player: after.current_player, turn: after.turn, terminated: after.terminated, truncated: after.truncated });
  const index = r.frames.length - 1;
  if (index % CHECKPOINT_EVERY === 0) l.checkpoints.set(index, engine.clone(l.game));
  persist(r);
  if (after.terminated || after.truncated) save(l);
  return after;
}

function action(l: Live, data: Record<string, unknown>) {
  if (l.rec.mode === "watch") throw new Conflict("Spectator games are controlled by the bot");
  const s = status(l);
  if (s.current_player !== l.rec.seat) throw new Conflict("It is the bot’s turn");
  if (Number(data.expected_state_version) !== s.state_version) throw new Conflict("State changed; refresh before acting");
  apply(l, String(data.action_id), l.rec.mode, null);
  return describe(l);
}

async function plan(l: Live, s: Status) {
  const started = performance.now();
  const p = await pool.plan(engine, l.game, l.rec.seed, l.rec.actions, SEARCH.lanes, (l.rec.seed ^ s.state_version) >>> 0);
  l.queue = [...p.plan];
  l.queueVersion = s.state_version;
  const { plan: steps, value, ...search } = p;
  const seconds = (performance.now() - started) / 1000;
  console.info(`[in-browser bot] turn ${s.turn}: searched ${SEARCH.lanes} candidate turns x ${p.plans} round(s) on ` +
    `${pool.size} worker thread(s) in ${seconds.toFixed(2)} s -> ${steps.length} moves`);
  return { search: { ...search, sub_actions: steps.length, workers: pool.size }, value, seconds };
}

async function advance(l: Live, data: Record<string, unknown>) {
  l.error = null;
  const count = Math.max(1, Math.min(Number(data.count ?? 1), 8));
  for (let i = 0; i < count; i++) {
    const s = status(l);
    if (s.terminated || s.truncated) break;
    if (l.rec.mode !== "watch" && s.current_player === l.rec.seat) break;
    if (l.rec.search) {
      let metrics: Record<string, unknown> | null = null;
      if (!l.queue.length || l.queueVersion !== s.state_version) metrics = await plan(l, s);
      if (l.queue.length) {
        try {
          const after = apply(l, l.queue.shift()!, "search", metrics);
          l.queueVersion = after.state_version;
          continue;
        } catch (e) {
          if (!(e instanceof Conflict)) throw e;
          console.warn("search action rejected; the policy plays this step", e.message);
          l.queue = [];
        }
      }
    }
    const started = performance.now();
    const d = engine.act(l.game, true, s.state_version);
    apply(l, d.action_id, "checkpoint", {
      value: d.value, log_probability: d.log_probability, seconds: (performance.now() - started) / 1000,
    });
  }
  return describe(l);
}

function setSearch(l: Live, data: Record<string, unknown>) {
  if (typeof data.search !== "boolean") throw Error("search must be true or false");
  l.rec.search = data.search;
  l.queue = [];
  persist(l.rec);
  return describe(l);
}

/** service.visible_animation: motion/combat cue whose endpoints and actor are public to the seat. */
function visibleAnimation(previous: Observation, current: Observation, event: GameEvent) {
  const c = event.action?.command ?? {};
  const { kind, source, target } = c;
  if (!(kind === 1 || kind === 2) || typeof source !== "number" || typeof target !== "number") return null;
  const before = previous.visible_state, after = current.visible_state;
  for (const i of [source, target])
    for (const v of [before, after]) if (i < 0 || i >= v.tiles.length || !v.tiles[i].known) return null;
  const old = new Map(before.units.map((u) => [u.tile, u]));
  const now = new Map(after.units.map((u) => [u.tile, u]));
  const actor = old.get(source);
  if (!actor || actor.owner !== event.player) return null;
  if (kind === 1 && (now.has(source) || now.get(target)?.owner !== actor.owner)) return null;
  if (kind === 2 && (!old.has(target) || old.get(target)!.owner === actor.owner)) return null;
  return { kind, source, target };
}

/** The frame a seat may see: until the game ends, the opponent's actions are redacted to public cues. */
function seatFrame(l: Live, index: number, requested: string) {
  const r = l.rec;
  const completed = over(l);
  const f = r.frames[index];
  const keyView = requested === "actor" ? String(f.current_player) : requested;
  const v = views(l, index);
  let event = r.events[index];
  if (r.mode !== "watch" && !completed && event && event.player !== r.seat) {
    const cue = index > 0 ? visibleAnimation(views(l, index - 1)[String(r.seat)], v[String(r.seat)], event) : null;
    event = {
      player: event.player, turn: event.turn, source: event.source ?? "checkpoint",
      action: { label: "Opponent acted", command: {} }, reason: event.reason,
      ...(cue ? { animation: cue } : {}),
    };
  }
  return {
    frame: { index, current_player: f.current_player, terminated: f.terminated, truncated: f.truncated, event },
    observation: v[keyView], event, frame_index: index,
  };
}

function publicFrame(l: Live, index: string | null, view: string | null) {
  const r = l.rec;
  const i = Math.max(0, Math.min(index === null ? r.frames.length - 1 : Number(index), r.frames.length - 1));
  const completed = over(l);
  let requested = String(view || (r.mode !== "watch" ? r.seat : "all"));
  if (r.mode !== "watch" && !completed) requested = String(r.seat);
  if (!["1", "2", "all", "actor"].includes(requested)) throw Error("Unavailable perspective");
  return {
    ...describe(l), ...seatFrame(l, i, requested), perspective: requested,
    available_views: r.mode === "watch" || completed ? ["1", "2", "all", "actor"] : [String(r.seat)],
  };
}

function observation(l: Live) {
  const s = status(l);
  const seat = l.rec.seat;
  const obs = views(l, l.rec.frames.length - 1)[String(seat)];
  const yourTurn = s.current_player === seat;
  return {
    schema: "bardur-agent-v1", game_id: l.rec.id, ...obs,
    legal_actions: yourTurn && !s.terminated && !s.truncated ? engine.legal(l.game) : [],
    your_turn: yourTurn, terminated: s.terminated, truncated: s.truncated,
  };
}

function save(l: Live) {
  const replay: GameRecord = { ...l.rec, kind: "replay", actions: [...l.rec.actions], events: [...l.rec.events], frames: [...l.rec.frames] };
  // An open copy of the older replay is stale now.
  const stale = live.get(key("replay", replay.id));
  if (stale) release(stale);
  persist(replay, true);
  return { replay_id: replay.id };
}

async function close(l: Live) {
  const result = save(l);
  release(l);
  records.delete(key("game", l.rec.id));
  clearTimeout(pending.get(key("game", l.rec.id)));
  await store.remove("game", l.rec.id);
  return result;
}

// ---------- replays ----------

function replayViews(r: GameRecord) {
  const completed = r.frames[r.frames.length - 1];
  return completed.terminated || completed.truncated || r.mode === "watch" ? ["1", "2", "all", "actor"] : [String(r.seat)];
}

function replayFrame(r: GameRecord, index: string | null, view: string | null) {
  if (r.imported) return importedFrame(r, index, view);
  const l = materialize(r);
  const available = replayViews(r);
  const i = Math.max(0, Math.min(Number(index ?? 0), r.frames.length - 1));
  let requested = view || "actor";
  if (!available.includes(requested)) requested = available[0];
  const f = r.frames[i];
  return {
    id: r.id, title: r.title, mode: "replay", frames: r.frames.length, ...seatFrame(l, i, requested),
    perspective: requested, available_views: available, bot, result: null,
    provenance: { source: "browser-local", seed: r.seed, search: r.search, checkpoint_sha256: bot.checkpoint_sha256 },
    terminated: f.terminated, truncated: f.truncated,
  };
}

interface ImportedFrame {
  current_player: number;
  terminated: boolean;
  truncated: boolean;
  views: Record<string, Observation>;
  event: GameEvent | null;
}

function importedFrame(r: GameRecord, index: string | null, view: string | null) {
  const replay = r.imported as { frames: ImportedFrame[]; available_views: string[]; bot?: unknown; result?: unknown; note?: string; provenance?: unknown };
  const i = Math.max(0, Math.min(Number(index ?? 0), replay.frames.length - 1));
  const f = replay.frames[i];
  let requested = view || "actor";
  if (!replay.available_views.includes(requested)) requested = replay.available_views[0];
  const k = requested === "actor" && !("actor" in f.views) ? String(f.current_player) : requested;
  const { views: _, ...frame } = f;
  return {
    id: r.id, title: r.title, mode: "replay", frames: replay.frames.length, frame_index: i, frame: { ...frame, index: i },
    observation: f.views[k], perspective: requested, available_views: replay.available_views, event: f.event ?? null,
    bot: replay.bot, result: replay.result, note: replay.note, provenance: replay.provenance,
    terminated: f.terminated, truncated: f.truncated,
  };
}

/** service.replay_series: per-turn economy and army of both players at the first frame of each turn. */
function series(r: GameRecord) {
  if (r.imported || !["1", "2", "all"].every((v) => replayViews(r).includes(v))) return { available: false };
  const l = materialize(r);
  const units = new Map(rules.units.map((u) => [u.id, u]));
  const army = (view: Observation["visible_state"], p: number) =>
    view.units.reduce((sum, u) => {
      const t = units.get(u.kind);
      if (u.owner !== p || !t || t.cost <= 0) return sum;
      return sum + (t.cost * u.health) / (t.health + 5 * Number(u.veteran));
    }, 0);
  const sample = (i: number, p: number): SeriesPoint => {
    const v = views(l, i);
    const mine = v[String(p)], world = v.all.visible_state;
    const cities = world.cities.filter((c) => c.owner === p);
    return {
      frame: i, turn: world.turn, stars: mine.visible_state.stars, income: mine.hud!.income, score: mine.hud!.score,
      cities: cities.length, population: cities.reduce((s, c) => s + (c.population ?? 0), 0),
      units: world.units.filter((u) => u.owner === p).length, army: Math.round(army(world, p) * 10) / 10,
    };
  };
  const players: Record<string, SeriesPoint[]> = { 1: [], 2: [] };
  const seen = new Set<string>();
  r.frames.forEach((f, i) => {
    const p = f.current_player;
    if (String(p) in players && !seen.has(`${p}:${f.turn}`)) {
      seen.add(`${p}:${f.turn}`);
      players[p].push(sample(i, p));
    }
  });
  const last = r.frames.length - 1;
  for (const p of ["1", "2"]) {
    const s = players[p];
    if (s.length && s[s.length - 1].frame !== last) s.push({ ...sample(last, Number(p)), final: true });
  }
  return { available: true, players };
}

/** A bardur-replay-v1 document of a record: every frame with the views its viewer may see. */
function exportReplay(r: GameRecord) {
  if (r.imported) return r.imported;
  const l = materialize(r);
  const available = replayViews(r);
  const frames = r.frames.map((f, i) => {
    const v = views(l, i);
    const seatView = seatFrame(l, i, available[0] === "1" && available.length > 1 ? "1" : available[0]);
    return {
      index: i, current_player: f.current_player, terminated: f.terminated, truncated: f.truncated,
      event: available.length > 1 ? r.events[i] : seatView.event,
      views: available.length > 1 ? { 1: v["1"], 2: v["2"], all: v.all } : { [available[0]]: v[available[0]] },
    };
  });
  const last = r.frames[r.frames.length - 1];
  return {
    schema: SCHEMA, title: r.title, created: r.created, bot, ruleset: RULESET,
    available_views: available.length > 1 ? ["1", "2", "all", "actor"] : available, frames,
    complete: last.terminated || last.truncated,
    provenance: { source: "browser-local", seed: r.seed, checkpoint_sha256: bot.checkpoint_sha256, ...(r.search ? { search: true } : {}) },
  };
}

async function importReplay(data: Record<string, unknown>) {
  let bytes = Uint8Array.from(atob(String(data.data)), (c) => c.charCodeAt(0));
  if (bytes[0] === 0x1f && bytes[1] === 0x8b)
    bytes = new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
  let replay: { schema?: string; title?: string; frames?: ImportedFrame[]; created?: number };
  try {
    replay = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw Error("The browser build imports Bardur replays (bardur-replay-v1 JSON) only");
  }
  if (replay.schema !== SCHEMA || !replay.frames?.length) throw Error("Not a Bardur replay (bardur-replay-v1)");
  const r: GameRecord = {
    id: uid(), kind: "replay", title: replay.title || String(data.name ?? "Imported replay").slice(0, 120), mode: "watch",
    seat: 1, seed: 0, search: false, created: replay.created ?? Date.now() / 1000, actions: [], events: [],
    frames: replay.frames.map((f) => ({ current_player: f.current_player, turn: f.views.all?.visible_state.turn ?? 0, terminated: f.terminated, truncated: f.truncated })),
    imported: replay as Record<string, unknown>,
  };
  persist(r, true);
  return { replay_id: r.id, frames: r.frames.length };
}

function library() {
  const all = [...records.values()];
  return {
    replays: all.filter((r) => r.kind === "replay").map((r) => ({
      id: r.id, title: r.title, frames: r.frames.length,
      complete: r.frames[r.frames.length - 1].terminated || r.frames[r.frames.length - 1].truncated,
      created: r.created, source: r.imported ? "imported" : "browser-local", available_views: replayViews(r), featured: false,
    })),
    games: all.filter((r) => r.kind === "game").map((r) => describe(materialize(r))),
  };
}

// ---------- routing ----------

async function route(path: string, body: Record<string, unknown> | undefined) {
  const url = new URL(path, "http://local");
  const p = url.pathname, q = url.searchParams;
  if (p === "/api/meta")
    return { bot, ready: true, warmup_job: null, search: { available: true, ready: true }, rules, terrain: TERRAIN, resources: RESOURCE, abilities: ABILITY, rewards: REWARD };
  if (p === "/api/library") return library();
  if (p === "/api/games" && body) return create(body);
  if (p === "/api/import" && body) return importReplay(body);
  const m = p.match(/^\/api\/(games|replays)\/([a-f0-9]{32})(?:\/(action|advance|observation|save|export|close|settings|series))?$/);
  if (!m) throw Error("Unknown endpoint");
  const [, kind, id, op] = m;
  if (kind === "replays") {
    const r = records.get(key("replay", id));
    if (!r) throw Error("Replay not found");
    if (op === "export") return exportReplay(r);
    if (op === "series") return series(r);
    return replayFrame(r, q.get("frame"), q.get("view"));
  }
  const r = records.get(key("game", id));
  if (!r) throw Error("Game not found");
  const l = materialize(r);
  if (["action", "advance", "save", "close", "settings"].includes(op) && !body) throw Error("POST required");
  if (op === "settings") return setSearch(l, body!);
  if (op === "action") return action(l, body!);
  if (op === "advance") {
    try {
      return await advance(l, body!);
    } catch (e) {
      l.error = (e as Error).message;
      throw e;
    }
  }
  if (op === "save") return save(l);
  if (op === "close") return close(l);
  if (op === "observation") return observation(l);
  if (op === "export") return exportReplay(r);
  return publicFrame(l, q.get("frame"), q.get("view"));
}

// One request at a time: every route reads or steps the shared engine.
let chain: Promise<unknown> = Promise.resolve();
scope.onmessage = (e: MessageEvent<{ id: number; path: string; body?: Record<string, unknown> }>) => {
  const { id, path, body } = e.data;
  const run = chain.then(async () => {
    await ready;
    return route(path, body);
  });
  chain = run.catch(() => undefined);
  run.then(
    (value) => scope.postMessage({ id, value }),
    (error) => scope.postMessage({ id, error: (error as Error).message || String(error) }),
  );
};
