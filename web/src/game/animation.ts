// Presentation only: plans contain exclusively the two supplied public views.
// Never infer a death from a unit merely disappearing into fog.
import type { Command, GameEvent, Observation, Unit, VisibleState } from "../api/types.ts";

export const clamp = (v: number) => Math.max(0, Math.min(1, v));
export const ease = (v: number) => {
  v = clamp(v);
  return v * v * (3 - 2 * v);
};
// Overshooting ease used for the mobile "pop" on spawn, build and level up.
export const pop = (v: number) => {
  v = clamp(v);
  if (v === 0) return 0;
  const c = 1.9;
  return 1 + (c + 1) * (v - 1) ** 3 + c * (v - 1) ** 2;
};
export const ranged = (kind: number) => [8, 9, 41, 42, 43].includes(kind);
export const naval = (kind: number) => kind >= 39 && kind <= 45;
const same = (a?: Unit, b?: Unit) =>
  !!a && !!b && a.owner === b.owner && a.kind === b.kind;
const STEP_MS = 260;

export interface Actor {
  unit: Unit;
  end: Unit | null | undefined;
  from: number;
  to: number;
  type: "move" | "attack" | "hurt";
  start?: number;
  finish?: number;
  path?: number[];
  aim?: number;
  impact?: number;
  counter?: boolean;
  hitAt?: number;
  deathAt?: number | null;
  advance?: boolean;
}
export type Effect =
  | { type: "damage"; tile: number; amount: number; at: number }
  | { type: "impact" | "death" | "convert" | "puff"; tile: number; at: number }
  | { type: "harvest"; tile: number; resource: number; at: number }
  | { type: "claim" | "capture"; tile: number; owner: number; at: number }
  | { type: "levelup"; tile: number; level: number; at: number };
export interface Projectile {
  from: number;
  to: number;
  start: number;
  end: number;
  kind: number;
}
export interface Pop {
  tile: number;
  layer: "unit" | "structure" | "city";
  at: number;
  span: number;
}
export interface Bar {
  tile: number;
  from: number;
  fromLevel?: number;
  to: number;
  toLevel?: number;
}
export interface Plan {
  duration: number;
  actors: Actor[];
  effects: Effect[];
  projectiles: Projectile[];
  hidden: Set<number>;
  pops: Pop[];
  reveals: { tile: number; at: number }[];
  bars: Bar[];
  flights: { from: number; to: number; at: number }[];
  explorer?: { path: number[]; owner: number; start: number; end: number };
}

// Shortest eight-way path across tiles the viewer can see in the new frame.
// Purely cosmetic: the engine already validated the move.
export function tilePath(view: VisibleState, from: number, to: number, kind: number) {
  const n = view.board_size;
  if (from === to) return [from];
  const passable = (i: number) => {
    const t = view.tiles[i];
    if (!t?.known) return false;
    const water = t.terrain === 1 || t.terrain === 2;
    if (t.terrain === undefined) return true;
    return naval(kind) ? water || t.structure === 8 : !water || t.structure === 8;
  };
  const prev = new Map<number, number>([[from, -1]]);
  let frontier = [from];
  for (let depth = 0; depth < 8 && frontier.length && !prev.has(to); depth++) {
    const next: number[] = [];
    for (const i of frontier) {
      const x = i % n,
        y = Math.floor(i / n);
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx,
            yy = y + dy,
            j = yy * n + xx;
          if ((!dx && !dy) || xx < 0 || yy < 0 || xx >= n || yy >= n) continue;
          if (prev.has(j) || (j !== to && !passable(j))) continue;
          prev.set(j, i);
          next.push(j);
        }
    }
    frontier = next;
  }
  if (!prev.has(to)) return [from, to];
  const path = [to];
  while (path[0] !== from) path.unshift(prev.get(path[0])!);
  return path;
}

export const chebyshev = (n: number, a: number, b: number) =>
  Math.max(
    Math.abs((a % n) - (b % n)),
    Math.abs(Math.floor(a / n) - Math.floor(b / n)),
  );

// Greedy walk through newly revealed tiles, as the mobile Explorer wanders.
function explorerWalk(n: number, start: number, tiles: number[]) {
  const left = new Set(tiles),
    walk = [start];
  let at = start;
  while (left.size) {
    let best = -1,
      bestD = Infinity;
    for (const t of left) {
      const d = chebyshev(n, at, t);
      if (d < bestD) {
        bestD = d;
        best = t;
      }
    }
    left.delete(best);
    walk.push(best);
    at = best;
  }
  return walk;
}

export function planTransition(
  before: Observation | null,
  after: Observation | null,
  event: GameEvent | null | undefined,
): Plan | null {
  if (
    !before ||
    !after ||
    after.state_version !== before.state_version + 1 ||
    before.player !== after.player ||
    before.visible_state.board_size !== after.visible_state.board_size
  )
    return null;
  const n = after.visible_state.board_size;
  const old = new Map(before.visible_state.units.map((u) => [u.tile, u]));
  const next = new Map(after.visible_state.units.map((u) => [u.tile, u]));
  const visible = (tile: number) =>
    !!before.visible_state.tiles[tile]?.known &&
    !!after.visible_state.tiles[tile]?.known;
  const command: Partial<Command> = event?.action?.command?.kind
    ? event.action.command
    : event?.animation || {};
  const own = event?.player !== undefined && event.player === after.player;
  const plan: Plan = {
    duration: 480,
    actors: [],
    effects: [],
    projectiles: [],
    hidden: new Set(),
    pops: [],
    reveals: [],
    bars: [],
    flights: [],
  };
  const controlled = new Set<number>();
  const actor = (unit: Unit, end: Unit | null | undefined, motion: Partial<Actor> & { type: Actor["type"] }) => {
    if (controlled.has(unit.tile)) return;
    controlled.add(unit.tile);
    if (end) plan.hidden.add(end.tile);
    plan.actors.push({ unit, end, from: unit.tile, to: end?.tile ?? unit.tile, ...motion });
  };
  const mover = (unit: Unit, end: Unit) => {
    const path = tilePath(after.visible_state, unit.tile, end.tile, unit.kind);
    plan.duration = Math.max(plan.duration, (path.length - 1) * STEP_MS + 120);
    actor(unit, end, { type: "move", start: 0, finish: 0.86, path });
  };
  const source = old.get(command.source!),
    target = old.get(command.target!);
  if (
    command.kind === 1 &&
    source &&
    visible(command.source!) &&
    after.visible_state.tiles[command.target!]?.known &&
    (before.visible_state.tiles[command.target!]?.known || source.owner === before.player)
  ) {
    const end = next.get(command.target!);
    if (end?.owner === source.owner) mover(source, end);
  } else if (command.kind === 2 && source && target && visible(source.tile) && visible(target.tile)) {
    plan.duration = 1100;
    const impact = 0.34;
    const endSource =
      next.get(source.tile)?.owner === source.owner
        ? next.get(source.tile)
        : next.get(target.tile)?.owner === source.owner
          ? next.get(target.tile)
          : null;
    const endTarget = same(target, next.get(target.tile)) ? next.get(target.tile) : null;
    const converted =
      next.get(target.tile)?.owner === source.owner &&
      next.get(target.tile)?.kind === target.kind &&
      source.kind === 10;
    actor(source, endSource, {
      type: "attack",
      aim: target.tile,
      impact,
      counter: false,
      hitAt: !endSource || endSource.health < source.health ? 0.64 : undefined,
      deathAt: endSource ? null : 0.64,
      advance: !!endSource && endSource.tile !== source.tile,
    });
    actor(target, converted ? next.get(target.tile) : endTarget, {
      type: "attack",
      aim: source.tile,
      counter: !!endTarget && ((!!endSource && endSource.health < source.health) || !endSource),
      impact: 0.64,
      hitAt: !converted && (!endTarget || endTarget.health < target.health) ? impact : undefined,
      deathAt: !endTarget && !converted ? impact : null,
    });
    const shot = (unit: Unit, to: number, start: number, end: number) => {
      if (ranged(unit.kind)) plan.projectiles.push({ from: unit.tile, to, start, end, kind: unit.kind });
    };
    if (!converted) shot(source, target.tile, 0.08, impact);
    if (plan.actors[1].counter) shot(target, source.tile, 0.43, 0.64);
    plan.effects.push({ type: converted ? "convert" : "impact", tile: target.tile, at: impact });
    if (plan.actors[1].counter) plan.effects.push({ type: "impact", tile: source.tile, at: 0.64 });
    for (const a of plan.actors)
      if (a.deathAt !== null && a.deathAt !== undefined)
        plan.effects.push({ type: "death", tile: a.from, at: a.deathAt });
  } else if (!command.kind) {
    // Redacted opponent moves can only be interpolated when the public delta is
    // unambiguous and both endpoints were and remain visible.
    const gone = [...old.values()].filter((u) => !same(u, next.get(u.tile)));
    const arrived = [...next.values()].filter((u) => !same(u, old.get(u.tile)));
    if (
      gone.length === 1 &&
      arrived.length === 1 &&
      same(gone[0], arrived[0]) &&
      gone[0].health === arrived[0].health &&
      visible(gone[0].tile) &&
      visible(arrived[0].tile)
    )
      mover(gone[0], arrived[0]);
  }
  // Health changes are already public. They also cover splash and visible
  // damage from attacks whose source was correctly redacted by the server.
  for (const u of old.values()) {
    const a = plan.actors.find((a) => a.from === u.tile);
    const end = a ? a.end : same(u, next.get(u.tile)) ? next.get(u.tile) : null;
    if (!end && a?.deathAt == null) continue;
    const damage = Math.ceil(u.health) - Math.ceil(end?.health || 0);
    if (damage <= 0) continue;
    const at = a?.hitAt ?? 0.16;
    if (!a && visible(u.tile)) actor(u, end, { type: "hurt", hitAt: at });
    if (a || visible(u.tile)) plan.effects.push({ type: "damage", tile: u.tile, amount: damage, at });
  }
  townAndTerrain(plan, before, after, command, visible, own, n);
  if (!plan.actors.length && !plan.effects.length && !plan.pops.length && !plan.reveals.length && !plan.bars.length)
    return null;
  return plan;
}

// Public map changes the mobile client celebrates: spawns, construction,
// harvests, city growth, captures, border growth and fog lifting.
function townAndTerrain(
  plan: Plan,
  before: Observation,
  after: Observation,
  command: Partial<Command>,
  visible: (t: number) => boolean,
  own: boolean,
  n: number,
) {
  const b = before.visible_state,
    a = after.visible_state;
  const oldUnits = new Map(b.units.map((u) => [u.tile, u]));
  const oldCities = new Map(b.cities?.map((c) => [c.tile, c]) || []);
  const newCities = new Map(a.cities?.map((c) => [c.tile, c]) || []);
  const moved = new Set(plan.actors.map((x) => x.to));
  for (const u of a.units) {
    if (oldUnits.has(u.tile) || moved.has(u.tile) || !visible(u.tile)) continue;
    const trained = command.kind === 4 && command.target === u.tile;
    const inCity = newCities.get(u.tile)?.owner === u.owner;
    if (trained || inCity) {
      plan.pops.push({ tile: u.tile, layer: "unit", at: 0.05, span: 0.5 });
      plan.effects.push({ type: "puff", tile: u.tile, at: 0.05 });
    }
  }
  for (const t of a.tiles) {
    const o = b.tiles[t.tile];
    if (!t.known || !o?.known) continue;
    if (t.structure && t.structure !== 1 && t.structure !== o.structure) {
      plan.pops.push({ tile: t.tile, layer: "structure", at: 0.04, span: 0.55 });
      plan.effects.push({ type: "puff", tile: t.tile, at: 0.04 });
    }
    if (o.resource && !t.resource && !t.structure) {
      plan.effects.push({ type: "harvest", tile: t.tile, resource: o.resource, at: 0 });
      plan.duration = Math.max(plan.duration, 900);
    }
    if (t.owner && t.owner !== o.owner) plan.effects.push({ type: "claim", tile: t.tile, owner: t.owner, at: 0.3 });
  }
  for (const c of a.cities || []) {
    const o = oldCities.get(c.tile);
    if (!visible(c.tile) || !o) continue;
    if (c.owner && c.owner !== o.owner) {
      plan.pops.push({ tile: c.tile, layer: "city", at: 0.2, span: 0.6 });
      plan.effects.push({ type: "capture", tile: c.tile, owner: c.owner, at: 0.2 });
      plan.duration = Math.max(plan.duration, 1300);
    }
    if (c.level === undefined || o.level === undefined || c.owner !== o.owner) continue;
    if (c.level > o.level) {
      plan.pops.push({ tile: c.tile, layer: "city", at: 0.45, span: 0.5 });
      plan.effects.push({ type: "levelup", tile: c.tile, level: c.level, at: 0.45 });
      plan.duration = Math.max(plan.duration, 1500);
    }
    if (c.level !== o.level || c.progress !== o.progress) {
      plan.bars.push({ tile: c.tile, from: o.progress || 0, fromLevel: o.level, to: c.progress || 0, toLevel: c.level });
      plan.duration = Math.max(plan.duration, 900);
      // Population pips fly from the worked tile into the city's bar.
      const worked = [command.target, command.source].find(
        (t) => t !== undefined && t >= 0 && t !== c.tile && b.tiles[t]?.known,
      );
      if (worked !== undefined && (c.progress || 0) + c.level * 10 > (o.progress || 0) + o.level * 10)
        plan.flights.push({ from: worked, to: c.tile, at: 0.15 });
    }
  }
  // Fog lifts only for the viewer's own actions; enemy paths stay private.
  if (!own) return;
  const revealed = a.tiles.filter((t) => t.known && !b.tiles[t.tile]?.known).map((t) => t.tile);
  if (!revealed.length) return;
  if (command.kind === 9 && command.parameter === 4) {
    const walk = explorerWalk(n, command.target!, revealed);
    plan.duration = Math.max(plan.duration, Math.min(3600, Math.max(900, walk.length * 150)));
    plan.explorer = { path: walk, owner: after.player, start: 0.04, end: 0.92 };
    walk.slice(1).forEach((tile, i) =>
      plan.reveals.push({ tile, at: 0.04 + (0.88 * (i + 1)) / Math.max(1, walk.length - 1) - 0.05 }),
    );
    return;
  }
  const origin = plan.actors[0]?.to ?? command.target ?? revealed[0];
  const far = Math.max(...revealed.map((t) => chebyshev(n, origin, t)), 1);
  for (const tile of revealed) plan.reveals.push({ tile, at: 0.35 + (0.35 * chebyshev(n, origin, tile)) / far });
  plan.duration = Math.max(plan.duration, 900);
}

export interface Pose {
  travel: number;
  segment: number;
  local: number;
  hop: number;
  lunge: number;
  flash: number;
  alpha: number;
  scale: number;
  shake: number;
  health: number;
}

export function sampleActor(a: Actor, progress: number): Pose {
  let travel = 0,
    hop = 0,
    lunge = 0,
    segment = 0,
    local = 0;
  if (a.type === "move") {
    const phase = clamp((progress - a.start!) / (a.finish! - a.start!));
    const steps = Math.max(1, (a.path?.length || 2) - 1);
    const scaled = phase * steps;
    segment = Math.min(steps - 1, Math.floor(scaled));
    local = steps === 1 ? ease(phase) : ease(scaled - segment);
    travel = phase >= 1 ? 1 : (segment + local) / steps;
    hop = Math.sin((steps === 1 ? phase : scaled - segment) * Math.PI) * 7;
  } else if (a.type === "attack") {
    const attacks = (!a.counter && a.impact === 0.34) || a.counter;
    if (attacks && !ranged(a.unit.kind)) {
      const start = a.counter ? 0.43 : 0.04;
      const phase = clamp((progress - start) / ((a.counter ? 0.82 : 0.52) - start));
      lunge = Math.sin(phase * Math.PI) * 0.36;
    }
    if (a.advance) {
      const phase = clamp((progress - 0.72) / 0.24);
      travel = ease(phase);
      local = travel;
      hop = Math.sin(phase * Math.PI) * 5;
    }
  }
  const damage = a.hitAt === undefined ? 0 : clamp((progress - a.hitAt) / 0.23);
  const dying = a.deathAt === null || a.deathAt === undefined ? 0 : clamp((progress - a.deathAt) / 0.32);
  const flash = damage > 0 && damage < 1 ? Math.sin(damage * Math.PI * 3) ** 2 * (1 - damage) : 0;
  return {
    travel,
    segment,
    local,
    hop,
    lunge,
    flash,
    alpha: 1 - dying,
    scale: 1 - dying * 0.32,
    shake: damage > 0 && damage < 1 ? Math.sin(damage * 24) * (1 - damage) * 3 : 0,
    health: progress < (a.hitAt ?? 0) ? a.unit.health : (a.end?.health ?? 0),
  };
}
