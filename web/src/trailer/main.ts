// Trailer scene: the last turns of fresh engine-rs games (scripts/trailer/gen_games.py, build_timeline.py) played
// at blitz speed on the board. Each half-turn one 3D hand carries the key piece while the rest of that
// side's army moves with it; captures flash red (the piece lost) and green (the piece that won); when a side is
// eliminated the winner's hand lifts the board by its edge and flips it in 3D onto the next game. An eval bar above
// the board (evalbar.ts) tugs between the two sides, swinging on the in-game score changes.
// Everything is a function of time for scripts/trailer/render.py, which steps frames in order.
import { Application, BitmapText, Container, Graphics, Matrix } from "pixi.js";
import * as THREE from "three";
import { MockArt } from "../board/art/mock.ts";
import type { BoardArt } from "../board/art/types.ts";
import type { GroundInfo } from "../board/art/types.ts";
import { DIAMOND, EDGES, HH, HW, STEPS, TEAMS, TH, type Team } from "../board/geometry.ts";
import type { Tile } from "../api/types.ts";
import { cityName } from "../game/vocab.ts";
import { Bench, BENCH_TIMES } from "./bench.ts";
import { EvalBar } from "./evalbar.ts";
import { COS, Stage, type HandPose } from "./stage3d.ts";

const W = 1920,
  H = 1080;
export const DURATION = 20;
// After the last game the board, hands and eval bar fade out over OUTRO and the bench leaderboard rises in.
const OUTRO = 0.45;
const FLIP = 0.46,
  FADE_IN = 0.2,
  FADE_OUT = 0.3;
// Which timeline game plays when, how long its shown turns take (seconds) and which models' logos stand for its players
// on the eval bar. A mirrored game is flipped left to right, so the winners alternate sides from game to game.
const PLAN = [
  { game: 0, length: 2.55, hold: 0.32, logos: { 1: "openai", 2: "claude" } },
  { game: 1, length: 2.35, hold: 0.3, mirror: true, logos: { 1: "kimi", 2: "gemini" } },
  { game: 2, length: 1.85, hold: 0.28, logos: { 1: "qwen", 2: "deepseek" } },
  { game: 3, length: 1.7, hold: 0.3, logos: { 1: "zai", 2: "minimax" } },
];
const LEAD_IN = 0.12;
const TEAM: Record<number, Team> = { 1: TEAMS.blue, 2: TEAMS.gold };
const RIM: Record<number, number> = { 1: 0x5fb4ff, 2: 0xffc36b };
const GREEN = 0x2bd96b,
  RED = 0xff2d2d;

// ---------- data from build_timeline.py ----------
type TileRec = [number, number, number, number, number, number]; // terrain, owner, structure, resource, road, capital
interface Act {
  kind: number;
  unit?: number;
  from?: number;
  to?: number;
  at?: number;
  moved?: boolean;
  hit?: number;
  died?: number[];
  born?: { id: number; tile: number; owner: number; kind: number }[];
  changed?: [number, number][];
  tiles?: Record<string, TileRec>;
  cities?: Record<string, [number, number]>;
  scores: [number, number]; // both players' in-game scores after the action
}
interface Half {
  player: number;
  turn: number;
  actions: Act[];
}
interface RawGame {
  size: number;
  winner: number;
  scores: [number, number]; // at the window start
  tiles: TileRec[];
  cities: Record<string, [number, number]>;
  units: { id: number; tile: number; owner: number; kind: number }[];
  halves: Half[];
}

// ---------- schedule ----------
interface Motion {
  t0: number;
  t1: number;
  kind: "move" | "lunge";
  from: number;
  to: number;
  featured: boolean;
}
interface UnitTrack {
  id: number;
  owner: number;
  tile: number; // where it is when first shown
  kind: number;
  born: number; // -inf when on the board at the start
  died: number; // +inf if it survives
  push: [number, number];
  motions: Motion[];
  kinds: [number, number][]; // [time, kind]
  hits: number[];
  wins: number[];
}
interface TileEvent {
  t: number;
  tiles?: Record<string, TileRec>;
  cities?: Record<string, [number, number]>;
  capture?: { tile: number; owner: number };
}
interface CityState {
  owner: number;
  level: number;
  pop: number; // dots filled on the growth bar, 0..level
}
interface CityEvent extends CityState {
  t: number;
  tile: number;
  burst?: "capture" | "level";
}
interface Grip {
  side: number;
  game: number;
  unit: number;
  t0: number;
  t1: number;
  kind: "move" | "lunge" | "flip";
}
interface Game {
  raw: RawGame;
  slot: Record<number, number>; // player -> hand (1 right, 2 left), by where the player's cities sit
  start: number;
  mate: number;
  end: number; // flip starts
  winner: number;
  units: Map<number, UnitTrack>;
  tileEvents: TileEvent[];
  grips: Grip[];
  lastAt: number; // tile of the deciding action
  scores: [number, number];
  steps: { t: number; scores: [number, number] }[]; // score changes, for the eval bar
  logos: Record<number, string>;
  cityStart: Map<number, CityState>;
  cityEvents: CityEvent[];
}

const N = 14;
const col = (t: number) => t % N,
  row = (t: number) => Math.floor(t / N);
const clamp = (x: number, a = 0, b = 1) => Math.max(a, Math.min(b, x)),
  smooth = (a: number, b: number, x: number) => {
    const u = clamp((x - a) / (b - a));
    return u * u * (3 - 2 * u);
  },
  lerp = (a: number, b: number, u: number) => a + (b - a) * u,
  easeInOut = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2);
const hash = (a: number, b = 0) => {
  const s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453;
  return s - Math.floor(s);
};
const tilePos = (t: number) => ({ x: (col(t) - row(t)) * HW, y: (col(t) + row(t)) * HH });

// ---------- spin ----------
// While the games play the board turns slowly about its centre, like a turntable, and the hands turn with it. The
// ground turns as one flat layer (a turn of the ground plane is an affine map of the isometric drawing); trees,
// houses, pieces and labels stay upright and move to where their tile has turned. The turn holds during each flip.
const SPIN_RATE = THREE.MathUtils.degToRad(7); // per second of play
// The hands' shoulders follow half the turn, so the hands orbit with the board but always reach in from the sides.
const SHOULDER_SPIN = 0.5;
const CC = (N - 1) / 2;
let SPIN = 0;
function spinAt(t: number) {
  let paused = 0;
  games.forEach((g, gi) => {
    const resume = gi + 1 < games.length ? games[gi + 1].start : Infinity;
    paused += Math.max(0, Math.min(t, resume) - g.end);
  });
  return SPIN_RATE * Math.max(0, t - LEAD_IN - paused);
}
// Pixi world point -> where it is once the board has turned by SPIN.
function rot(x: number, y: number) {
  const dc = (x / HW + y / HH) / 2 - CC,
    dr = (y / HH - x / HW) / 2 - CC,
    c = Math.cos(SPIN),
    s = Math.sin(SPIN),
    c2 = c * dc - s * dr,
    r2 = s * dc + c * dr;
  return { x: (c2 - r2) * HW, y: (c2 + r2 + 2 * CC) * HH };
}
function spinMatrix() {
  const o = rot(0, 0),
    ex = rot(1, 0),
    ey = rot(0, 1);
  return new Matrix(ex.x - o.x, ex.y - o.y, ey.x - o.x, ey.y - o.y, o.x, o.y);
}

// The game seen in a mirror: swapping each tile's column and row flips the board left to right on screen.
function mirror(raw: RawGame): RawGame {
  const m = (t: number) => col(t) * N + row(t);
  const keys = <T>(o: Record<string, T>) =>
    Object.fromEntries(Object.entries(o).map(([k, v]) => [String(m(Number(k))), v]));
  const tiles: TileRec[] = [];
  raw.tiles.forEach((t, k) => (tiles[m(k)] = t));
  return {
    ...raw,
    tiles,
    cities: keys(raw.cities),
    units: raw.units.map((u) => ({ ...u, tile: m(u.tile) })),
    halves: raw.halves.map((h) => ({
      ...h,
      actions: h.actions.map((a) => ({
        ...a,
        from: a.from === undefined ? undefined : m(a.from),
        to: a.to === undefined ? undefined : m(a.to),
        at: a.at === undefined ? undefined : m(a.at),
        born: a.born?.map((b) => ({ ...b, tile: m(b.tile) })),
        tiles: a.tiles && keys(a.tiles),
        cities: a.cities && keys(a.cities),
      })),
    })),
  };
}

function schedule(raw: RawGame, start: number, length: number, hold: number, gi: number, logos: Record<number, string>): Game {
  // The players sit at the left and right of the board; whoever's cities lie further right plays the right hand.
  const across = (p: number) => {
    const own = Object.entries(raw.cities).filter(([, [o]]) => o === p).map(([k]) => col(+k) - row(+k));
    return own.length ? own.reduce((a, b) => a + b, 0) / own.length : p === 1 ? 1 : -1;
  };
  const slot: Record<number, number> = across(1) >= across(2) ? { 1: 1, 2: 2 } : { 1: 2, 2: 1 };
  const units = new Map<number, UnitTrack>();
  const track = (id: number, owner: number, tile: number, kind: number, born: number) =>
    units.set(id, { id, owner, tile, kind, born, died: Infinity, push: [0, 0], motions: [], kinds: [], hits: [], wins: [] });
  for (const u of raw.units) track(u.id, u.owner, u.tile, u.kind, -Infinity);
  const tileEvents: TileEvent[] = [];
  const grips: Grip[] = [];
  const steps: Game["steps"] = [];
  let cities = raw.cities,
    last = start,
    lastAt = -1;
  // Actions stream one after another in their real order, at random intervals: a unit action takes about one
  // beat, a build or a training a third of one, and each new turn a short breath. Each move animates over its
  // own window, so a few pieces are in flight at once but never a whole army.
  const weight = (a: Act) => (a.unit !== undefined ? 1 : 0.3);
  const beats = raw.halves.reduce((n, h) => n + 1.5 + h.actions.reduce((m, a) => m + weight(a), 0), 0);
  const beat = length / beats;
  const busy = new Map<number, number>(); // unit id -> when its current motion ends
  const handFree: Record<number, number> = { 1: -Infinity, 2: -Infinity };
  let cursor = start,
    k = 0;
  raw.halves.forEach((half) => {
    cursor += 1.5 * beat;
    const side = slot[half.player];
    half.actions.forEach((a) => {
      const r = hash(++k, gi * 7 + 3);
      let s = cursor;
      cursor += beat * weight(a) * (0.3 + 1.4 * r);
      const unitAct = a.unit !== undefined && a.from !== undefined && a.to !== undefined && units.has(a.unit);
      if (unitAct) s = Math.max(s, (busy.get(a.unit!) ?? -Infinity) + 0.02);
      const move = !!(a.moved || a.kind === 1);
      // The hand takes the piece whenever it is free.
      const feat = unitAct && (move || a.kind === 2) && s >= handFree[side] + 0.04;
      const e = s + (feat ? (move ? 0.26 : 0.24) : move ? 0.2 : a.kind === 2 ? 0.18 : 0.12);
      if (feat) handFree[side] = e;
      if (unitAct) {
        const u = units.get(a.unit!)!;
        if (move || a.kind === 2) u.motions.push({ t0: s, t1: e, kind: move ? "move" : "lunge", from: a.from!, to: a.to!, featured: feat });
        busy.set(a.unit!, e);
        if (feat) grips.push({ side, game: gi, unit: a.unit!, t0: s, t1: e, kind: move ? "move" : "lunge" });
        if (a.died?.length) u.wins.push(e);
        lastAt = a.to!;
      }
      if (a.hit !== undefined) units.get(a.hit)?.hits.push(e);
      for (const id of a.died ?? []) {
        const u = units.get(id);
        if (!u) continue;
        u.died = e;
        if (a.from !== undefined && a.to !== undefined) {
          const p = tilePos(a.from),
            q = tilePos(a.to),
            l = Math.hypot(q.x - p.x, q.y - p.y) || 1;
          u.push = [(q.x - p.x) / l, (q.y - p.y) / l];
        } else u.push = [hash(id) - 0.5, -1];
      }
      for (const b of a.born ?? []) track(b.id, b.owner, b.tile, b.kind, e);
      for (const [id, kind] of a.changed ?? []) units.get(id)?.kinds.push([e, kind]);
      if (a.tiles || a.cities) {
        const ev: TileEvent = { t: e, tiles: a.tiles, cities: a.cities };
        if (a.cities)
          for (const [k, [owner]] of Object.entries(a.cities))
            if (owner && cities[k] && cities[k][0] !== owner) ev.capture = { tile: Number(k), owner };
        if (a.cities) cities = a.cities;
        tileEvents.push(ev);
        if (ev.capture) lastAt = ev.capture.tile;
      }
      // Score changes keep their real order: one shows when its action lands, never before the one it follows.
      const prev = steps.length ? steps[steps.length - 1] : { t: -Infinity, scores: raw.scores };
      if (a.scores[0] !== prev.scores[0] || a.scores[1] !== prev.scores[1]) steps.push({ t: Math.max(e, prev.t), scores: a.scores });
      last = Math.max(last, e);
    });
  });
  for (const u of units.values()) u.motions.sort((a, b) => a.t0 - b.t0);
  tileEvents.sort((a, b) => a.t - b.t);
  const g: Game = { raw, slot, start, mate: last, end: last + hold, winner: raw.winner, units, tileEvents, grips, lastAt, scores: raw.scores, steps, logos, cityStart: new Map(), cityEvents: [] };
  grow(g, gi);
  addFillers(g, gi);
  return g;
}

// Cities grow through each game as they do in play: each fills its growth bar a dot at a time and levels up when it is
// full, its houses growing with it; a captured city keeps its level and starts its bar again.
function grow(g: Game, gi: number) {
  const rate = (tile: number, owner: number) => 0.8 + 1.6 * hash(tile, gi * 5 + owner); // dots per second
  const state = new Map<number, CityState & { next: number }>();
  const add = (tile: number, owner: number, level: number, pop: number, t: number) =>
    state.set(tile, { owner, level, pop, next: owner ? t + (0.2 + 0.8 * hash(tile, gi + 17)) / rate(tile, owner) : Infinity });
  for (const [k, [owner, level]] of Object.entries(g.raw.cities)) {
    const lv = Math.max(1, level);
    add(Number(k), owner, lv, owner ? Math.floor(hash(Number(k), gi + 3) * (lv + 1)) : 0, g.start);
  }
  g.cityStart = new Map([...state].map(([k, s]) => [k, { owner: s.owner, level: s.level, pop: s.pop }]));
  const reals = g.tileEvents.filter((e) => e.cities);
  let ri = 0;
  for (;;) {
    let tile = -1,
      tn = Infinity;
    for (const [k, c] of state) if (c.next < tn) [tile, tn] = [k, c.next];
    const tr = ri < reals.length ? reals[ri].t : Infinity;
    if (Math.min(tn, tr) >= g.mate + 0.4) break;
    if (tr <= tn) {
      const ev = reals[ri++];
      for (const [k, [owner, level]] of Object.entries(ev.cities!)) {
        const tile = Number(k),
          c = state.get(tile);
        if (c && c.owner === owner) continue;
        const lv = Math.max(1, level, c?.level ?? 0);
        add(tile, owner, lv, 0, ev.t);
        g.cityEvents.push({ t: ev.t, tile, owner, level: lv, pop: 0, burst: c && owner ? "capture" : undefined });
      }
    } else {
      const c = state.get(tile)!;
      c.next = tn >= g.mate ? Infinity : tn + 1 / rate(tile, c.owner);
      if (++c.pop > c.level) {
        c.level++;
        c.pop = 0;
      }
      g.cityEvents.push({ t: tn, tile, owner: c.owner, level: c.level, pop: c.pop, burst: c.pop === 0 ? "level" : undefined });
    }
  }
}

// In these endgames the losing side has few pieces and little left to do, and either hand would sit out the other's
// turn. The loser gets a small reserve of pieces on its own land, knocked off when a winner's piece moves onto them
// or at the mate; then every gap in a hand's work is filled with a quick shuffle of one of its idle
// pieces, a tile over and back, so the real positions are untouched and both hands keep moving pieces until the mate.
const OUT = 0.19,
  PAUSE = 0.03,
  RESERVE = 5,
  RESERVE_KINDS = [9, 4, 9, 2, 4, 8];
function addFillers(g: Game, gi: number) {
  const water = (t: number) => g.raw.tiles[t][0] === 1 || g.raw.tiles[t][0] === 2;
  const loser = 3 - g.winner,
    survivors = [...g.units.values()].filter((u) => u.owner === loser && u.died > g.mate).length;
  if (survivors < RESERVE) {
    // Each reserve piece stands on an empty tile of the loser's land until the first winner piece moves onto it,
    // which knocks it off, or the mate.
    const until = g.raw.tiles.map((t, k) =>
      t[1] === loser
        ? g.mate + 0.03 + (g.lastAt >= 0 ? Math.hypot(col(k) - col(g.lastAt), row(k) - row(g.lastAt)) * 0.022 : 0)
        : -Infinity,
    );
    const taker: (Motion & { u: UnitTrack })[] = [];
    const blocked = new Set<number>(Object.keys(g.raw.cities).map(Number));
    for (const u of g.units.values()) {
      blocked.add(u.tile);
      for (const m of u.motions)
        if (u.owner === loser) blocked.add(m.from).add(m.to);
        else if (m.kind === "move" && m.t1 < until[m.to]) {
          until[m.to] = m.t1;
          taker[m.to] = { ...m, u };
        }
    }
    const free = [...Array(N * N).keys()]
      .filter((t) => until[t] > g.start + 0.8 && !water(t) && !blocked.has(t))
      .sort((a, b) => until[b] - until[a] + 0.3 * (hash(a, gi) - hash(b, gi)));
    const placed: number[] = [];
    for (const t of free) {
      if (placed.length >= RESERVE - survivors) break;
      if (placed.some((q) => Math.max(Math.abs(col(q) - col(t)), Math.abs(row(q) - row(t))) < 2)) continue;
      placed.push(t);
    }
    placed.forEach((t, i) => {
      const id = 100000 + i,
        by = taker[t];
      let push: [number, number] = [hash(id, gi) - 0.5, -1];
      if (by && by.t1 === until[t]) {
        by.u.wins.push(by.t1);
        const p = tilePos(by.from),
          q = tilePos(t),
          l = Math.hypot(q.x - p.x, q.y - p.y) || 1;
        push = [(q.x - p.x) / l, (q.y - p.y) / l];
      }
      g.units.set(id, {
        id,
        owner: loser,
        tile: t,
        kind: RESERVE_KINDS[i % RESERVE_KINDS.length],
        born: -Infinity,
        died: until[t],
        push,
        motions: [],
        kinds: [],
        hits: [],
        wins: [],
      });
    });
  }
  const tileAt = (u: UnitTrack, t: number) => {
    let tile = u.tile;
    for (const m of u.motions) if (m.kind === "move" && m.t1 <= t) tile = m.to;
    return tile;
  };
  // Tiles any piece stands on or passes through during [a, b].
  const occupied = (a: number, b: number) => {
    const taken = new Set<number>();
    for (const u of g.units.values()) {
      if (u.born > b || u.died < a) continue;
      taken.add(tileAt(u, a));
      for (const m of u.motions) if (m.t0 < b && m.t1 > a) taken.add(m.from).add(m.to);
    }
    return taken;
  };
  const idle = (u: UnitTrack, a: number, b: number) =>
    u.born < a - 0.1 &&
    u.died > b + 0.3 &&
    !u.motions.some((m) => m.t1 > a - 0.05 && m.t0 < b + 0.05) &&
    !u.hits.some((h) => h > a - 0.05 && h < b + 0.3);
  let k = 0;
  for (const player of [1, 2]) {
    const side = g.slot[player];
    let c = g.start + 0.05,
      lastUnit = -1;
    while (c < g.mate - 0.3) {
      const end = c + 2 * OUT + PAUSE,
        clash = g.grips.find((gr) => gr.side === side && gr.t1 > c - 0.04 && gr.t0 < end + 0.04);
      if (clash) {
        c = clash.t1 + 0.04;
        continue;
      }
      const taken = occupied(c, end),
        options: { u: UnitTrack; from: number; to: number }[] = [];
      for (const u of g.units.values()) {
        if (u.owner !== player || !idle(u, c, end)) continue;
        const from = tileAt(u, c);
        for (const [dc, dr] of STEPS) {
          const cc = col(from) + dc,
            rr = row(from) + dr,
            to = rr * N + cc;
          if (cc >= 0 && rr >= 0 && cc < N && rr < N && water(to) === water(from) && !taken.has(to))
            options.push({ u, from, to });
        }
      }
      if (!options.length) {
        c += 0.08;
        continue;
      }
      // Another piece than the last one, if there is one.
      const fresh = options.filter((o) => o.u.id !== lastUnit),
        pool = fresh.length ? fresh : options;
      const { u, from, to } = pool[Math.floor(hash(++k, gi * 13 + player) * pool.length)];
      u.motions.push(
        { t0: c, t1: c + OUT, kind: "move", from, to, featured: true },
        {
          t0: c + OUT + PAUSE,
          t1: end,
          kind: "move",
          from: to,
          to: from,
          featured: true,
        },
      );
      u.motions.sort((a, b) => a.t0 - b.t0);
      g.grips.push(
        { side, game: gi, unit: u.id, t0: c, t1: c + OUT, kind: "move" },
        {
          side,
          game: gi,
          unit: u.id,
          t0: c + OUT + PAUSE,
          t1: end,
          kind: "move",
        },
      );
      lastUnit = u.id;
      c = end + 0.05;
    }
  }
}


// ---------- camera ----------
const K = (0.8 * H) / (N * TH),
  OX = W / 2,
  OY = H / 2 - (N - 1) * HH * K + 46;

// ---------- board view ----------
interface UnitView {
  node: Container;
  flash: Container;
  kind: number;
  grasp: number; // scene units above the tile where the pinch closes
}
class Board {
  root = new Container();
  private ground = new Container();
  private surface = new Container();
  fxGround = new Graphics();
  objects = new Container();
  fxTop = new Graphics();
  private labelLayer = new Container();
  // The ground turns as a layer, clipped to the board's top face; the earth sides are drawn upright under it.
  private floor = new Container();
  private floorMask = new Graphics();
  private sides = new Graphics();
  private top = new Container();
  private labels = new Map<number, { node: Container; bar: Graphics; width: number; state: CityState }>();
  private shown = new Map<number, CityState>();
  private grown = 0;
  private burstLayer = new Container();
  private bursts: { node: Container; text: BitmapText; stars: Container[]; ev: CityEvent; size: number }[] = [];
  private tileViews: { base: Container; surf: Container; holder: Container }[] = [];
  tiles: TileRec[] = [];
  cities: Record<string, [number, number]> = {};
  private applied = 0;
  private units = new Map<number, UnitView>();
  private game: Game;
  private art: BoardArt;

  constructor(game: Game, art: BoardArt) {
    this.game = game;
    this.art = art;
    this.objects.sortableChildren = true;
    this.floor.addChild(this.ground, this.surface, this.fxGround);
    this.floor.mask = this.floorMask;
    this.top.addChild(this.fxTop);
    this.root.addChild(this.sides, this.floor, this.floorMask, this.objects, this.labelLayer, this.top, this.burstLayer);
    const depth = (t: number) => col(t) + row(t);
    for (const t of [...Array(N * N).keys()].sort((a, b) => depth(a) - depth(b))) {
      const p = tilePos(t);
      const base = new Container(),
        surf = new Container(),
        holder = new Container();
      base.position.set(p.x, p.y);
      surf.position.set(p.x, p.y);
      holder.position.set(p.x, p.y);
      holder.zIndex = depth(t) * 100 + 10;
      this.ground.addChild(base);
      this.surface.addChild(surf);
      this.objects.addChild(holder);
      this.tileViews[t] = { base, surf, holder };
    }
    this.reset();
    // "Captured!" and "Level N!" over the city, in a ring of stars, as the game shows them.
    for (const ev of game.cityEvents) {
      if (!ev.burst) continue;
      const capture = ev.burst === "capture",
        node = new Container(),
        stars: Container[] = [];
      for (let i = 0; i < (capture ? 12 : 7); i++) {
        const s = new Container();
        s.addChild(art.star(0, 0, capture ? 11 : 8));
        stars.push(node.addChild(s));
      }
      const text = new BitmapText({
        text: capture ? "Captured!" : `Level ${ev.level}!`,
        style: {
          fontFamily: "Josefin Sans, Avenir Next, sans-serif",
          fontSize: 48,
          fill: "#ffffff",
          dropShadow: { color: "#000000", blur: 6, distance: 0, alpha: 0.9 },
        },
      });
      text.anchor.set(0.5, 0.5);
      node.addChild(text);
      node.visible = false;
      this.burstLayer.addChild(node);
      this.bursts.push({ node, text, stars, ev, size: capture ? 21 : 15 });
    }
  }

  // Turn the board to SPIN: the ground layers by the spin matrix, everything upright to its turned tile.
  spin() {
    const m = spinMatrix();
    this.floor.setFromMatrix(m);
    this.top.setFromMatrix(m);
    const corners = [
      [-0.5, -0.5],
      [N - 0.5, -0.5],
      [N - 0.5, N - 0.5],
      [-0.5, N - 0.5],
    ].map(([c, r]) => rot((c - r) * HW, (c + r) * HH));
    this.floorMask.clear().poly(corners.flatMap((p) => [p.x, p.y])).fill(0xffffff);
    // The earth sides that face the camera: each drops straight down the screen from its top edge.
    const g = this.sides.clear(),
      cx = corners.reduce((n, p) => n + p.x, 0) / 4,
      cy = corners.reduce((n, p) => n + p.y, 0) / 4,
      drop = 15;
    corners.forEach((a, i) => {
      const b = corners[(i + 1) % 4];
      let nx = b.y - a.y,
        ny = -(b.x - a.x);
      if (nx * ((a.x + b.x) / 2 - cx) + ny * ((a.y + b.y) / 2 - cy) < 0) [nx, ny] = [-nx, -ny];
      if (ny <= 0) return;
      const lit = nx < 0;
      g.poly([a.x, a.y, b.x, b.y, b.x, b.y + drop, a.x, a.y + drop]).fill(lit ? 0x8f5c34 : 0x6a4124);
      g.poly([a.x, a.y, b.x, b.y, b.x, b.y + 2.2, a.x, a.y + 2.2]).fill(lit ? 0xa77a4a : 0x7f5734);
    });
    for (let t = 0; t < N * N; t++) {
      const p = tilePos(t),
        q = rot(p.x, p.y),
        h = this.tileViews[t].holder;
      h.position.set(q.x, q.y);
      h.zIndex = (q.y / HH) * 100 + 10;
    }
    for (const [tile, l] of this.labels) {
      const p = tilePos(tile),
        q = rot(p.x, p.y);
      l.node.position.set(q.x, q.y);
    }
  }

  drawBursts(t: number) {
    for (const b of this.bursts) {
      const age = t - b.ev.t;
      b.node.visible = age >= 0 && age < 0.8;
      if (!b.node.visible) continue;
      const p0 = tilePos(b.ev.tile),
        p = rot(p0.x, p0.y),
        pop = age < 0.12 ? lerp(0.3, 1.15, age / 0.12) : lerp(1.15, 1, clamp((age - 0.12) / 0.12));
      b.node.position.set(p.x, p.y - 26 - 22 * smooth(0, 0.8, age));
      b.node.alpha = 1 - smooth(0.55, 0.8, age);
      b.text.scale.set((b.size / 48) * pop);
      b.stars.forEach((s, i) => {
        const a = (i / b.stars.length) * Math.PI * 2 + age * 2.2,
          r = 10 + (b.size * 3.2) * smooth(0, 0.5, age);
        s.position.set(Math.cos(a) * r, Math.sin(a) * r * 0.6);
        s.alpha = 1 - smooth(0.3, 0.7, age);
      });
    }
  }

  private reset() {
    this.tiles = this.game.raw.tiles.map((t) => [...t] as TileRec);
    this.cities = { ...this.game.raw.cities };
    this.applied = 0;
    this.shown = new Map([...this.game.cityStart].map(([k, c]) => [k, { ...c }]));
    this.grown = 0;
    for (let t = 0; t < N * N; t++) this.drawTile(t);
    for (const k of [...this.labels.keys()]) this.dropLabel(k);
    for (const k of this.shown.keys()) this.label(k);
  }

  // Bring tiles and cities to time t, rebuilding only what changed and its neighbours.
  sync(t: number) {
    const evs = this.game.tileEvents,
      grows = this.game.cityEvents;
    if ((this.applied > 0 && evs[this.applied - 1].t > t) || (this.grown > 0 && grows[this.grown - 1].t > t)) this.reset();
    const dirty = new Set<number>();
    while (this.applied < evs.length && evs[this.applied].t <= t) {
      const ev = evs[this.applied++];
      for (const [k, v] of Object.entries(ev.tiles ?? {})) {
        this.tiles[Number(k)] = v;
        dirty.add(Number(k));
      }
      if (ev.cities) {
        for (const k of new Set([...Object.keys(ev.cities), ...Object.keys(this.cities)]))
          if (JSON.stringify(ev.cities[k]) !== JSON.stringify(this.cities[k])) dirty.add(Number(k));
        this.cities = ev.cities;
      }
    }
    const relabel = new Set<number>();
    while (this.grown < grows.length && grows[this.grown].t <= t) {
      const ev = grows[this.grown++],
        was = this.shown.get(ev.tile);
      if (!was || was.owner !== ev.owner || was.level !== ev.level) dirty.add(ev.tile);
      this.shown.set(ev.tile, { owner: ev.owner, level: ev.level, pop: ev.pop });
      relabel.add(ev.tile);
    }
    for (const k of relabel) this.label(k);
    const around = new Set<number>();
    for (const t of dirty)
      for (let dc = -1; dc <= 1; dc++)
        for (let dr = -1; dr <= 1; dr++) {
          const c = col(t) + dc,
            r = row(t) + dr;
          if (c >= 0 && r >= 0 && c < N && r < N) around.add(r * N + c);
        }
    for (const t of around) this.drawTile(t);
  }

  private tile(t: number): Tile {
    const [terrain, owner, structure, resource, road, capital] = this.tiles[t];
    return { tile: t, known: true, terrain, owner, structure, resource, road, capital };
  }

  private info(t: Tile): GroundInfo {
    const nb = (k: number) => {
      const c = col(t.tile) + STEPS[k][0],
        r = row(t.tile) + STEPS[k][1];
      return c >= 0 && r >= 0 && c < N && r < N ? this.tile(r * N + c) : null;
    };
    const water = (x: Tile | null) => !!x && (x.terrain === 1 || x.terrain === 2);
    let border = 0,
      shores = 0,
      roads = 0,
      walls = 0;
    EDGES.forEach((k, j) => {
      const o = nb(k);
      if (t.owner && !(o && o.owner === t.owner)) border |= 1 << j;
      if (water(t) && o && !water(o)) shores |= 1 << j;
    });
    const joins = (x: Tile | null) => !!x && (!!x.road || x.structure === 1 || x.structure === 8 || x.structure === 48);
    if (t.road || (joins(t) && STEPS.some((_, k) => nb(k)?.road)))
      for (let k = 0; k < 8; k++) if (joins(nb(k)) && (t.road || nb(k)!.road)) roads |= 1 << k;
    [1, 7].forEach((k, b) => {
      const o = nb(k);
      if (t.terrain === 1 ? o?.terrain === 2 : t.terrain === 2 && !o) walls |= 1 << b;
    });
    return { border, roads, shores, walls, team: t.owner ? TEAM[t.owner] ?? TEAMS.neutral : null };
  }

  private drawTile(t: number) {
    const v = this.tileViews[t],
      tile = this.tile(t);
    for (const c of [...v.base.removeChildren(), ...v.surf.removeChildren(), ...v.holder.removeChildren()]) c.destroy({ children: true });
    this.art.ground(tile, this.info(tile), v.base, v.surf);
    v.holder.addChild(this.art.terrain(tile, []));
    const city = this.cities[String(t)];
    if (city) {
      const [owner, level] = city,
        shown = this.shown.get(t)?.level ?? level;
      v.holder.addChild(this.art.city(owner ? TEAM[owner] ?? TEAMS.neutral : TEAMS.neutral, owner, shown || 1, !!tile.capital, []));
    } else {
      const st = this.art.structure(tile, []);
      if (st) v.holder.addChild(st);
    }
  }

  // The game's city label: a band in the owner's colour with the name, the level and a star (and the crown on a
  // capital), over the growth bar. Only the bar is repainted while the city fills it.
  private label(tile: number) {
    const st = this.shown.get(tile),
      old = this.labels.get(tile);
    if (!st || !st.owner) return this.dropLabel(tile);
    if (old && old.state.owner === st.owner && old.state.level === st.level) {
      old.state = { ...st };
      old.bar.clear();
      this.art.paintBar(old.bar, old.width, st.level + 1, st.pop / (st.level + 1));
      return;
    }
    this.dropLabel(tile);
    const p = tilePos(tile),
      capital = !!this.tiles[tile][5],
      name = cityName(tile),
      width = Math.max(62, name.length * 6 + 26) + (capital ? 12 : 0),
      band = this.art.labelBand(TEAM[st.owner] ?? TEAMS.neutral),
      node = new Container();
    node.position.set(p.x, p.y);
    node.addChild(new Graphics().roundRect(-width / 2, 15, width, 16, 2).fill(band));
    const text = (value: string | number, x: number) => {
      const b = new BitmapText({ text: String(value), style: { fontFamily: "Josefin Sans, Avenir Next, sans-serif", fontSize: 48, fill: "#ffffff" } });
      b.scale.set(12 / 48);
      b.anchor.set(0.5, 0.5);
      b.position.set(x, 23);
      node.addChild(b);
    };
    text(name, capital ? -3 : -9);
    text(st.level, width / 2 - 6);
    node.addChild(this.art.star(width / 2 - 16, 23, 11));
    if (capital) {
      const cap = this.art.capitalIcon(-width / 2 + 5, 23, 11);
      if (cap) node.addChild(cap);
    }
    const bar = new Graphics();
    node.addChild(bar);
    this.art.paintBar(bar, width, st.level + 1, st.pop / (st.level + 1));
    this.labelLayer.addChild(node);
    this.labels.set(tile, { node, bar, width, state: { ...st } });
  }

  private dropLabel(tile: number) {
    this.labels.get(tile)?.node.destroy({ children: true });
    this.labels.delete(tile);
  }

  unit(id: number, owner: number, kind: number): UnitView {
    let v = this.units.get(id);
    if (v && v.kind !== kind) {
      v.node.destroy({ children: true });
      v = undefined;
    }
    if (!v) {
      const a = this.art.unit({ tile: 0, owner, kind, health: 10, veteran: false }, TEAM[owner] ?? TEAMS.neutral);
      const node = new Container();
      node.addChild(a.body);
      const flash = a.flash ?? new Container();
      flash.blendMode = "add";
      node.addChild(flash);
      this.objects.addChild(node);
      const top = a.body.getLocalBounds().minY;
      v = { node, flash, kind, grasp: (-top * 0.62 * K) / COS };
      this.units.set(id, v);
    }
    return v;
  }

  hideUnits() {
    for (const v of this.units.values()) v.node.visible = false;
  }
}

// ---------- unit state ----------
interface UnitState {
  x: number;
  y: number;
  height: number; // scene units
  lift: number;
  carried: boolean;
  kind: number;
}
function unitAt(u: UnitTrack, t: number): UnitState {
  let tile = u.tile,
    kind = u.kind;
  for (const [tk, k] of u.kinds) if (t >= tk) kind = k;
  for (const m of u.motions) {
    if (t < m.t0) break;
    if (t < m.t1) {
      const a = tilePos(m.from),
        b = tilePos(m.to),
        v = (t - m.t0) / (m.t1 - m.t0);
      if (m.kind === "move") {
        const s = m.featured ? smooth(0.12, 0.9, v) : easeInOut(v),
          lift = m.featured ? smooth(0, 0.25, v) * (1 - smooth(0.82, 1, v)) : Math.sin(Math.PI * v) * 0.35;
        return { x: lerp(a.x, b.x, s), y: lerp(a.y, b.y, s), height: lift * (m.featured ? 70 : 34), lift, carried: m.featured, kind };
      }
      // A strike: lunge a third of the way to the target and back.
      const s = Math.sin(Math.PI * clamp(v * 1.15)) * 0.38,
        lift = m.featured ? Math.sin(Math.PI * v) : 0.3 * Math.sin(Math.PI * v);
      return { x: lerp(a.x, b.x, s), y: lerp(a.y, b.y, s), height: lift * (m.featured ? 48 : 18), lift, carried: m.featured, kind };
    }
    if (m.kind === "move") tile = m.to;
  }
  const p = tilePos(tile);
  return { x: p.x, y: p.y, height: 0, lift: 0, carried: false, kind };
}

// ---------- globals ----------
let app: Application;
let stage: Stage;
let games: Game[] = [];
let boards: Board[] = [];
let pixiCanvas: HTMLCanvasElement;
let threeCanvas: HTMLCanvasElement;
let fade: HTMLDivElement;
let bar: EvalBar;
let bench: Bench;
let shown = -1;

function diamond(g: Graphics, x: number, y: number, s: number, color: number, alpha: number) {
  if (alpha <= 0.003) return;
  g.poly(DIAMOND.map((v, i) => (i % 2 ? y + v * s : x + v * s))).fill({ color, alpha });
}
function ring(g: Graphics, x: number, y: number, r: number, width: number, color: number, alpha: number) {
  if (alpha <= 0.003) return;
  g.ellipse(x, y, r, r * 0.6).stroke({ width, color, alpha });
}

function drawGame(gi: number, t: number) {
  const g = games[gi],
    b = boards[gi];
  if (shown !== gi) {
    if (shown >= 0) boards[shown].root.visible = false;
    b.root.visible = true;
    shown = gi;
  }
  SPIN = spinAt(t);
  b.sync(t);
  b.spin();
  b.drawBursts(t);
  b.hideUnits();
  const gf = b.fxGround.clear(),
    tf = b.fxTop.clear();
  for (const u of g.units.values()) {
    if (t < u.born) continue;
    const st = unitAt(u, Math.min(t, u.died));
    const v = b.unit(u.id, u.owner, st.kind),
      node = v.node;
    node.visible = true;
    node.rotation = 0;
    node.alpha = 1;
    node.scale.set(1);
    node.tint = 0xffffff;
    v.flash.alpha = 0;
    v.flash.tint = 0xffffff;
    // Trails behind every piece in motion.
    for (const m of u.motions) {
      if (t < m.t0 || t > m.t1 + 0.3 || m.kind !== "move") continue;
      const a = tilePos(m.from),
        cur = unitAt(u, Math.min(t, m.t1 - 1e-4)),
        fadeOut = 1 - clamp((t - m.t1) / 0.3);
      for (let i = 0; i < 10; i++) {
        const u0 = i / 10,
          u1 = (i + 1) / 10;
        gf.moveTo(lerp(a.x, cur.x, u0), lerp(a.y, cur.y, u0))
          .lineTo(lerp(a.x, cur.x, u1), lerp(a.y, cur.y, u1))
          .stroke({ width: 1.5 + 4.5 * u1, color: TEAM[u.owner].main, alpha: 0.8 * u1 * fadeOut, cap: "round" });
      }
    }
    if (t >= u.died) {
      // Loss: knocked off its tile, flashing red, tumbling away.
      const age = t - u.died;
      if (age > 0.45) {
        node.visible = false;
        continue;
      }
      const w = age / 0.45,
        side = hash(u.id, gi) - 0.5,
        dx = u.push[0] * 46 * w + side * 34 * w,
        dy = u.push[1] * 26 * w - Math.sin(Math.PI * Math.min(1, w * 1.3)) * 26;
      const q = rot(st.x, st.y);
      node.position.set(q.x + dx, q.y + dy);
      node.rotation = side * 3 * w;
      node.scale.set(1 + 0.25 * smooth(0, 0.25, w));
      node.alpha = 1 - smooth(0.4, 1, w);
      node.tint = 0xff8a80;
      node.zIndex = 1e6;
      v.flash.tint = RED;
      v.flash.alpha = 1 - smooth(0, 0.8, w);
      ring(tf, st.x + dx, st.y + dy - 8, 6 + 26 * smooth(0, 0.35, age), 2.5, RED, 0.9 * (1 - clamp(age / 0.35)));
      for (let k = 0; k < 7; k++) {
        const a = hash(u.id * 13 + k, gi) * Math.PI * 2,
          r = (10 + 30 * hash(k, u.id)) * smooth(0, 0.45, age);
        tf.circle(st.x + dx + Math.cos(a) * r, st.y + dy - 8 + Math.sin(a) * r * 0.6, 2 * (1 - w)).fill({ color: RED, alpha: 1 - w });
      }
      continue;
    }
    const born = t - u.born;
    const q = rot(st.x, st.y);
    node.position.set(q.x, q.y - (st.height * COS) / K);
    node.scale.set((1 + 0.12 * st.lift) * (born < 0.15 ? Math.max(0.001, smooth(0, 0.15, born)) : 1));
    node.zIndex = st.carried ? 1e6 : (q.y / HH) * 100 + 50 + st.lift * 200;
    if (born < 0.3) {
      v.flash.tint = 0xffffff;
      v.flash.alpha = 0.8 * (1 - born / 0.3);
    }
    for (const h of u.hits)
      if (t >= h && t < h + 0.25) {
        v.flash.tint = RED;
        v.flash.alpha = Math.max(v.flash.alpha, 0.9 * (1 - (t - h) / 0.25));
      }
    for (const w of u.wins)
      if (t >= w && t < w + 0.35) {
        v.flash.tint = GREEN;
        v.flash.alpha = Math.max(v.flash.alpha, 0.95 * (1 - (t - w) / 0.35));
        diamond(gf, st.x, st.y, 0.9, GREEN, 0.6 * (1 - (t - w) / 0.35));
        ring(tf, st.x, st.y - 4, 10 + 36 * smooth(0, 0.3, t - w), 3, GREEN, 0.9 * (1 - (t - w) / 0.35));
      }
  }
  // City captures: a green burst on the city and its land.
  for (const ev of g.tileEvents) {
    if (!ev.capture || t < ev.t || t > ev.t + 0.5) continue;
    const age = t - ev.t,
      p = tilePos(ev.capture.tile);
    for (let dc = -1; dc <= 1; dc++)
      for (let dr = -1; dr <= 1; dr++) {
        const c = col(ev.capture.tile) + dc,
          r = row(ev.capture.tile) + dr;
        if (c < 0 || r < 0 || c >= N || r >= N) continue;
        const q = tilePos(r * N + c);
        diamond(gf, q.x, q.y, 0.96, GREEN, 0.55 * Math.sin(Math.PI * clamp(age / 0.45)));
      }
    ring(tf, p.x, p.y - 6, 20 + 90 * smooth(0, 0.45, age), 4, GREEN, 0.95 * (1 - clamp(age / 0.5)));
  }
  // The end: green sweeps the winner's land from the deciding tile.
  const age = t - g.mate;
  if (age >= 0 && g.lastAt >= 0) {
    const cp = tilePos(g.lastAt);
    for (let tile = 0; tile < N * N; tile++) {
      const owner = b.tiles[tile][1];
      if (!owner) continue;
      const d = Math.hypot(col(tile) - col(g.lastAt), row(tile) - row(g.lastAt)),
        u = (age - d * 0.022) / 0.38;
      if (u < 0 || u > 1) continue;
      const p = tilePos(tile);
      diamond(gf, p.x, p.y, 0.96, owner === g.winner ? GREEN : RED, 0.6 * Math.sin(Math.PI * u));
    }
    ring(tf, cp.x, cp.y, 30 + 420 * smooth(0, 0.5, age), 7, GREEN, 0.9 * (1 - clamp(age / 0.45)));
  }
}

// ---------- hands ----------
const grips: Record<number, Grip[]> = { 1: [], 2: [] };
let REST: Record<number, THREE.Vector3>;
let SHOULDER: Record<number, THREE.Vector3>;
let HAND_SCALE = 1;

// Screen pixel (sx, sy) at scene height Y -> scene point.
function fromScreen(sx: number, sy: number, Y: number) {
  const a = (sx - OX) / (HW * K),
    b = (sy - OY + Y * COS) / (HH * K);
  return stage.ground(a * HW, b * HH, Y);
}

function unitGrip(gi: number, id: number, t: number) {
  const u = games[gi].units.get(id)!,
    st = unitAt(u, Math.min(t, u.died)),
    v = boards[gi].unit(u.id, u.owner, st.kind);
  const q = rot(st.x, st.y);
  return stage.ground(q.x, q.y, st.height + v.grasp * (1 + 0.1 * st.lift));
}

// ---------- flip ----------
const centre = () => stage.ground(0, (N - 1) * HH, 0);
// A scene point turned with the board (or by a share of its turn).
function sceneRot(v: THREE.Vector3, share = 1) {
  const a = (v.x - v.z) / stage.T,
    b = (v.x + v.z) / stage.T,
    full = SPIN;
  SPIN *= share;
  const q = rot(a * HW, b * HH);
  SPIN = full;
  return stage.ground(q.x, q.y, v.y);
}
// The board's front-to-back diagonal, turned with it.
const axis = () => sceneRot(centre().add(new THREE.Vector3(100, 0, 100))).sub(centre()).normalize();
function flipAt(t: number) {
  for (let gi = 0; gi + 1 < games.length; gi++) {
    const g = games[gi],
      v = (t - g.end) / FLIP;
    if (v < 0 || v >= 1) continue;
    const sign = g.slot[g.winner] === 1 ? 1 : -1; // the winner's edge goes up
    // A dip as the hand slams the edge, a fast turn in the air, then a hard landing.
    const turn = easeInOut(smooth(0.1, 0.72, v)),
      rise = v < 0.1 ? -6 * Math.sin((Math.PI * v) / 0.1) : 200 * Math.sin(Math.PI * clamp((v - 0.1) / 0.64)) ** 0.8;
    return { gi, v, angle: sign * Math.PI * turn, rise: v > 0.74 ? 0 : rise, landed: v >= 0.74 };
  }
  return null;
}

// The board's right or left vertex as the slab turns, for the winner's flip grip.
function edgePoint(gi: number, t: number) {
  const right = games[gi].slot[games[gi].winner] === 1,
    corner = right ? tilePos(N - 1) : tilePos(N * (N - 1)),
    q = rot(corner.x + (right ? HW : -HW), corner.y),
    p = stage.ground(q.x, q.y, 0);
  const f = flipAt(t);
  if (!f || f.gi !== gi) return p;
  return p.applyMatrix4(stage.slabMatrix(f.angle, axis(), centre(), f.rise));
}

function handPose(side: number, t: number): HandPose {
  const list = grips[side],
    shoulder = sceneRot(SHOULDER[side], SHOULDER_SPIN);
  const make = (grip: THREE.Vector3, pinch: number, hover: number): HandPose => ({
    grip: grip.clone().add(new THREE.Vector3(0, hover, 0)),
    toward: grip.clone().sub(shoulder).setY(0),
    pitch: THREE.MathUtils.degToRad(lerp(22, 9, clamp(hover / 160))),
    roll: THREE.MathUtils.degToRad(-14),
    pinch,
    curl: 0.4,
    scale: HAND_SCALE * (1 + 0.0008 * hover),
    alpha: 1,
  });
  const i = list.findIndex((gr) => t >= gr.t0 && t <= gr.t1);
  if (i >= 0) {
    const gr = list[i];
    if (gr.kind === "flip") return make(edgePoint(gr.game, t), 0.85, 0);
    return make(unitGrip(gr.game, gr.unit, t), 1, 0);
  }
  const next = list.find((gr) => gr.t0 > t),
    prev = [...list].reverse().find((gr) => gr.t1 < t);
  const point = (gr: Grip, at: number) => (gr.kind === "flip" ? edgePoint(gr.game, at) : unitGrip(gr.game, gr.unit, at));
  const rest = sceneRot(REST[side], SHOULDER_SPIN);
  const A = prev ? { t: prev.t1, p: point(prev, prev.t1) } : { t: -1, p: rest },
    B = next ? { t: next.t0, p: point(next, next.t0) } : { t: DURATION + 1, p: rest };
  // Release, travel bowed up through the air, close again on arrival.
  const travel = (from: THREE.Vector3, to: THREE.Vector3, t0: number, t1: number, h0: number, h1: number, p0: number, p1: number) => {
    const v = clamp((t - t0) / (t1 - t0)),
      e = easeInOut(v),
      hover = lerp(h0, h1, e) + Math.sin(Math.PI * v) ** 0.8 * 110 * (1 - Math.max(h0, h1) / 200),
      pinch = v < 0.5 ? lerp(p0, 0.05, smooth(0, 0.25, v)) : lerp(0.05, p1, smooth(0.65, 1, v));
    return make(from.clone().lerp(to, e), pinch, hover);
  };
  const long = B.t - A.t > 0.7 || !prev || !next;
  if (!long) return travel(A.p, B.p, A.t, B.t, 0, 0, 1, 1);
  const leave = Math.min(A.t + 0.3, (A.t + B.t) / 2),
    arrive = Math.max(B.t - 0.28, (A.t + B.t) / 2);
  if (t < leave && prev) return travel(A.p, rest, A.t, leave, 0, 170, 1, 0.15);
  if (t > arrive && next) return travel(rest, B.p, arrive, B.t, 170, 0, 0.15, 1);
  const drift = (t - leave) * 1.1 + side;
  return make(rest.clone().add(new THREE.Vector3(Math.sin(drift * 2) * 8, 0, Math.cos(drift * 1.6) * 8)), 0.15, 170);
}

// ---------- flip slab ----------
let slabFor = -1;
function grabCanvas() {
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  c.getContext("2d")!.drawImage(pixiCanvas, 0, 0);
  return c;
}
function prepareSlab(gi: number) {
  if (slabFor === gi) return;
  drawGame(gi, games[gi].end);
  app.render();
  const front = grabCanvas();
  drawGame(gi + 1, games[gi + 1].start);
  app.render();
  const back = grabCanvas();
  const d = axis();
  stage.setSlab(front, back, N, d, centre(), Math.atan2(d.x - d.z, d.x + d.z));
  slabFor = gi;
}

// ---------- sound cues ----------
// Everything scripts/trailer/sound.py scores the video to, in seconds: per game its timeline and what happens on the
// board (hands taking pieces, pieces lost, cities captured or levelled), the eval bar's big swings, and the closer.
function soundEvents() {
  const side = (s: number) => (s === 1 ? 1 : -1); // pan: right hand right
  return {
    duration: DURATION,
    games: games.map((g, gi) => ({
      start: g.start,
      mate: g.mate,
      end: g.end,
      landed: gi + 1 < games.length ? g.end + FLIP * 0.74 : null,
      winnerPan: side(g.slot[g.winner]),
      grips: g.grips.map((gr) => ({ t: gr.t0, pan: side(gr.side), kind: gr.kind })),
      deaths: [...g.units.values()].filter((u) => u.died <= g.end).map((u) => ({ t: u.died, pan: side(g.slot[u.owner]) })),
      captures: g.cityEvents.filter((e) => e.burst === "capture").map((e) => e.t),
      levels: g.cityEvents.filter((e) => e.burst === "level").map((e) => e.t),
    })),
    bench: { at: games[games.length - 1].end + OUTRO * 0.5, rows: 12, ...BENCH_TIMES },
  };
}

// ---------- frame ----------
function frame(t: number) {
  const f = flipAt(t);
  SPIN = spinAt(t);
  let gi = 0;
  while (gi + 1 < games.length && t >= games[gi].end + FLIP * 0.74) gi++;
  if (f && !f.landed) {
    prepareSlab(f.gi);
    SPIN = spinAt(t);
    stage.setSlabPose(f.angle, axis(), centre(), f.rise);
    pixiCanvas.style.opacity = "0";
  } else {
    stage.hideSlab();
    pixiCanvas.style.opacity = "1";
    drawGame(gi, t);
  }
  // The landing shakes the frame for a moment.
  let shake: [number, number] = [0, 0];
  for (const g of games) {
    const age = t - (g.end + FLIP * 0.74);
    if (age >= 0 && age < 0.22) {
      const a = 12 * (1 - age / 0.22);
      shake = [a * Math.sin(age * 140), a * 0.7 * Math.cos(age * 115)];
    }
  }
  const tr = shake[0] || shake[1] ? `translate(${shake[0].toFixed(1)}px, ${shake[1].toFixed(1)}px)` : "";
  pixiCanvas.style.transform = threeCanvas.style.transform = tr;
  app.render();
  const playing = games[Math.min(gi, games.length - 1)];
  stage.hands.forEach((h, k) => {
    const player = Number(Object.keys(playing.slot).find((p) => playing.slot[+p] === k + 1));
    h.setRim(RIM[player]);
    h.pose(handPose(k + 1, t));
  });
  stage.render();
  const last = games[games.length - 1].end,
    gone = smooth(last, last + OUTRO, t);
  if (!(f && !f.landed)) pixiCanvas.style.opacity = (1 - gone).toFixed(3);
  threeCanvas.style.opacity = bar.canvas.style.opacity = (1 - gone).toFixed(3);
  bench.draw(t - last - OUTRO * 0.5);
  // The eval bar turns over with the board, showing the next game once the board is past edge-on.
  if (f) bar.draw(games[Math.abs(f.angle) > Math.PI / 2 ? f.gi + 1 : f.gi], t, Math.abs(Math.cos(f.angle)));
  else bar.draw(playing, t);
  fade.style.opacity = Math.max(1 - smooth(0, FADE_IN, t), smooth(DURATION - FADE_OUT, DURATION, t)).toFixed(3);
}

function stars() {
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  c.className = "layer";
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < 220; i++) {
    const big = i % 7 === 0;
    ctx.fillStyle = `rgba(255,255,255,${big ? 0.8 : 0.35})`;
    ctx.fillRect(hash(i, 1) * W, hash(i, 2) * H, big ? 1.6 : 1, big ? 1.6 : 1);
  }
  const vignette = ctx.createRadialGradient(W / 2, H / 2, H * 0.3, W / 2, H / 2, H * 0.95);
  vignette.addColorStop(0, "rgba(10,22,40,0.35)");
  vignette.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, W, H);
  return c;
}

async function main() {
  document.body.append(stars());
  pixiCanvas = document.createElement("canvas");
  pixiCanvas.className = "layer";
  document.body.append(pixiCanvas);
  threeCanvas = document.createElement("canvas");
  threeCanvas.className = "layer";
  document.body.append(threeCanvas);
  // The eval bar is the point of the benchmark, so it stays in front of the hands.
  bar = new EvalBar(W, H, TEAM);
  document.body.append(bar.canvas);
  bench = new Bench(W, H);
  document.body.append(bench.canvas);
  fade = document.createElement("div");
  fade.className = "layer fade";
  document.body.append(fade);

  app = new Application();
  await app.init({
    canvas: pixiCanvas,
    width: W,
    height: H,
    backgroundAlpha: 0,
    antialias: true,
    resolution: 1,
    autoStart: false,
    sharedTicker: false,
    preference: "webgl",
    preserveDrawingBuffer: true,
  });
  app.ticker.stop();
  const world = new Container();
  world.position.set(OX, OY);
  world.scale.set(K);
  app.stage.addChild(world);

  stage = new Stage(threeCanvas, { W, H, K, OX, OY, HW, HH }, N);
  HAND_SCALE = (5.2 * stage.T) / 0.18;
  await Promise.all([
    stage.loadHands("/trailer/right-hand.glb", [RIM[1], RIM[2]]),
    document.fonts.load('600 60px "Josefin Sans"').catch(() => null),
    document.fonts.load('400 48px "Josefin Sans"').catch(() => null),
    bar.load(PLAN.flatMap((p) => Object.values(p.logos))),
    bench.load(),
    document.fonts.load('600 46px "Instrument Sans"').catch(() => null),
    document.fonts.load('400 24px "Instrument Sans"').catch(() => null),
  ]);
  const art = new MockArt();

  const raw: RawGame[] = await (await fetch("/trailer/games.json")).json();
  let at = LEAD_IN;
  games = PLAN.map((p, gi) => {
    const g = schedule(p.mirror ? mirror(raw[p.game]) : raw[p.game], at, p.length, p.hold, gi, p.logos);
    at = g.end + FLIP;
    return g;
  });
  for (const g of games) for (const gr of g.grips) grips[gr.side].push(gr);
  games.forEach((g, gi) => {
    if (gi + 1 < games.length) grips[g.slot[g.winner]].push({ side: g.slot[g.winner], game: gi, unit: -1, t0: g.end, t1: g.end + FLIP * 0.42, kind: "flip" });
  });
  for (const s of [1, 2]) grips[s].sort((a, b) => a.t0 - b.t0);
  boards = games.map((g) => {
    const b = new Board(g, art);
    b.root.visible = false;
    world.addChild(b.root);
    return b;
  });
  REST = { 1: fromScreen(W * 0.9, H * 0.5, 170), 2: fromScreen(W * 0.1, H * 0.5, 170) };
  SHOULDER = { 1: fromScreen(W * 1.6, H * 0.42, 0), 2: fromScreen(-W * 0.6, H * 0.42, 0) };

  const query = new URLSearchParams(location.search);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (window as any).trailer = { frame, stage, handPose, soundEvents, duration: DURATION, games: games.map((g) => ({ start: g.start, mate: g.mate, end: g.end, winner: g.winner, slot: g.slot, steps: g.steps })), ready: true };
  if (query.has("t")) frame(Number(query.get("t")));
  else if (query.has("play")) {
    const speed = Number(query.get("speed") ?? 1),
      t0 = performance.now();
    const loop = () => {
      frame((((performance.now() - t0) / 1000) * speed) % DURATION);
      requestAnimationFrame(loop);
    };
    loop();
  } else frame(1);
}

main();
