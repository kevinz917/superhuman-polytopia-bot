// WebGL board: Pixi scene graph in world units (one tile = 64 x 38.4).
// The camera is a single transform, so panning and zooming never rebuild
// geometry; a new frame rebuilds only the tiles and labels whose content
// changed, so scrubbing a replay touches a handful of nodes per step.
import { Application, BitmapText, Container, Graphics } from "pixi.js";
import type { GameEvent, Layer, LegalAction, Observation, Tile, Unit, VisibleState } from "../api/types.ts";
import { clamp, ease, planTransition, pop, sampleActor, type Plan } from "../game/animation.ts";
import { cityName } from "../game/vocab.ts";
import { createArt, type BoardArt, type GroundInfo } from "./art/index.ts";
import { setIconSource } from "./assets.ts";
import { DIAMOND, EDGES, flat, HH, HW, shape, STEPS, TEAMS, TH, TW, type Point, type Team } from "./geometry.ts";

const DIM_TINT = 0x6f7b85;

// destroy({ children: true }) leaves a Graphics' own context (and its GPU
// geometry) alive, which leaked every rebuilt frame. A bare destroy() frees
// owned contexts and keeps the shared ones above.
const dispose = (node: Container) => {
  for (const child of node.removeChildren()) dispose(child);
  node.destroy();
};
const lerpTint = (k: number) => {
  const mix = (c: number, d: number) => Math.round(c + (d - c) * k);
  const r = mix(255, (DIM_TINT >> 16) & 255),
    g = mix(255, (DIM_TINT >> 8) & 255),
    b = mix(255, DIM_TINT & 255);
  return (r << 16) | (g << 8) | b;
};

export interface UpdateOptions {
  scene: string;
  animate?: boolean;
  speed?: number;
  layer?: Layer | null;
  territory?: number[];
  home?: number; // the seat drawn in blue ("you"); the other seat is red
  ready?: number[]; // tiles of the player's units that can still move or attack
}

interface TileView {
  base: Container; // ground diamond and sides
  surf: Container; // road, shorelines and border
  holder: Container; // depth-sorted terrain, buildings and unit
  dim: Graphics;
  sel: Container;
  groundKey: string;
  objectKey: string;
  dimmables: Container[];
  terrain?: Container;
  city?: Container;
  structure?: Container;
  unit?: UnitView;
}
interface LabelView {
  node: Container;
  key: string;
  bar: Graphics;
  width: number;
  city: { level: number; progress: number };
}
interface UnitView extends Container {
  hp?: BitmapText;
  flash?: Container;
  ready?: Container;
  shownHealth?: number;
}
// One unit's health before and after a previewed attack (0: it dies).
export interface HealthPreview {
  tile: number;
  health: number;
  after: number;
  max: number;
}
interface Running {
  plan: Plan;
  duration: number;
  started: number;
  actors: UnitView[];
  fogs: Map<number, Container>;
  fx: Graphics;
  texts: { node: BitmapText; tile: number; at: number; dy: number }[];
  harvest: { node: Container; tile: number; at: number }[];
  bursts: { node: Container; tile: number; at: number }[];
  explorer?: Container;
}

export class BoardScene {
  readonly app = new Application();
  // A render group: panning/zooming changes one GPU transform instead of
  // re-walking every tile, unit and label.
  private world = new Container({ isRenderGroup: true });
  private stars = new Graphics();
  private ground = new Container();
  private surface = new Container(); // roads, shorelines and borders: above every cube, as in game
  private outline = new Graphics();
  private outlineTop = new Graphics(); // crisp border above trees and houses
  private objects = new Container();
  private labelLayer = new Container();
  // Move and attack targets, depth-sorted among the units in `objects`: above
  // their own tile's unit, below the units in front, whose health badges they used to cover.
  private markers: Container[] = [];
  private fxLayer = new Container();
  private previewLayer = new Container(); // attack preview: health bars and skulls above everything
  private previewed: UnitView[] = [];
  private tiles: TileView[] = [];
  private labels = new Map<number, LabelView>();
  private builtSize = 0;
  private art!: BoardArt; // real or mock, chosen once the graphics have loaded
  private ready = false;
  private destroyed = false;

  zoom = 1;
  pan = { x: 0, y: 0 };
  private anchor: number | null = null;
  private size = 14;
  private k = 1; // screen pixels per world unit
  private ox = 0;
  private oy = 0;
  private w = 0;
  private h = 0;

  private observation: Observation | null = null;
  private scene: string | null = null;
  private selected: number | null = null;
  private layer: Layer | null = null;
  private territory = new Set<number>();
  private movable = new Set<number>();
  private home = 1;
  // Seats in the game: two-seat games colour by viewer (you blue, the other red); more seats get fixed colours.
  private seats = 2;
  private targets: LegalAction[] = [];
  private selectedAt = 0;

  private running: Running | null = null;
  private finished: Promise<void> = Promise.resolve();
  private resolveFinished: (() => void) | null = null;
  private finishTimer = 0;
  private frameHandle = 0;
  private liveUntil = 0;
  private camera: { from: { x: number; y: number }; to: { x: number; y: number }; t0: number } | null = null;

  private pointers = new Map<number, { x: number; y: number }>();
  private drag: { x: number; y: number; px: number; py: number } | null = null;
  private pinch: { distance: number; zoom: number } | null = null;
  moved = false;
  // The tile under the mouse (not touch), for the attack preview; null when it leaves the board.
  onHover: ((tile: number | null) => void) | null = null;
  private hovered: number | null = null;
  private reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

  private canvas: HTMLCanvasElement;
  private overlay: HTMLElement;
  private select: (tile: number) => void;
  constructor(canvas: HTMLCanvasElement, overlay: HTMLElement, select: (tile: number) => void) {
    this.canvas = canvas;
    this.overlay = overlay;
    this.select = select;
  }

  async init() {
    const wrap = this.canvas.parentElement!;
    const r = wrap.getBoundingClientRect();
    await this.app.init({
      canvas: this.canvas,
      width: Math.max(1, r.width),
      height: Math.max(1, r.height),
      backgroundAlpha: 0,
      antialias: true,
      autoDensity: true,
      resolution: Math.min(2, window.devicePixelRatio || 1),
      autoStart: false,
      sharedTicker: false,
      preference: "webgl",
    });
    if (this.destroyed) return;
    this.app.ticker.stop();
    this.objects.sortableChildren = true;
    this.world.addChild(this.ground, this.surface, this.outline, this.objects, this.outlineTop, this.labelLayer, this.fxLayer);
    this.app.stage.addChild(this.stars, this.world);
    this.resize();
    this.world.addChild(this.previewLayer); // above every other layer
    new ResizeObserver(() => this.resize()).observe(wrap);
    this.bindInput(wrap);
    this.reducedMotion.addEventListener("change", () => this.reducedMotion.matches && this.finishAnimation());
    document.addEventListener("visibilitychange", () => document.hidden && this.finishAnimation());
    await document.fonts?.load('400 16px "Josefin Sans"').catch(() => null);
    if (this.destroyed) return;
    this.art = createArt();
    setIconSource(this.app.renderer, (name) => this.art.icon(name));
    this.canvas.dataset.art = this.art.mode;
    this.ready = true;
    this.rebuild();
    this.invalidate();
  }

  destroy() {
    this.destroyed = true;
    cancelAnimationFrame(this.frameHandle);
    try {
      this.app.destroy(false, { children: true });
    } catch {
      // Not initialised yet.
    }
  }

  // ---------- frame loop ----------
  // Renders on demand: input, state changes and running animations only.
  invalidate() {
    if (this.frameHandle || !this.ready) return;
    this.frameHandle = requestAnimationFrame(() => {
      this.frameHandle = 0;
      this.frame();
    });
  }
  private frame() {
    const now = performance.now();
    let again = false;
    if (this.camera) {
      const t = ease((now - this.camera.t0) / 380);
      this.pan = {
        x: this.camera.from.x + (this.camera.to.x - this.camera.from.x) * t,
        y: this.camera.from.y + (this.camera.to.y - this.camera.from.y) * t,
      };
      if (t >= 1) this.camera = null;
      else again = true;
    }
    if (this.running) {
      const p = clamp((now - this.running.started) / this.running.duration);
      if (p >= 1) this.finishAnimation();
      else {
        this.stepAnimation(p);
        again = true;
      }
    }
    if (now < this.liveUntil) {
      this.stepSelection();
      again = true;
    } else if (this.liveUntil) {
      // Settle on the final pose even when frames arrive late (throttled or
      // slow devices); otherwise the borders stay at their first, invisible frame.
      this.stepSelection();
      this.liveUntil = 0;
    }
    this.applyCamera();
    this.app.render();
    if (again) this.invalidate();
  }

  // ---------- camera ----------
  private resize() {
    const r = this.canvas.parentElement!.getBoundingClientRect();
    this.w = r.width;
    this.h = r.height;
    if (!this.w || !this.h) return;
    this.app.renderer.resize(this.w, this.h);
    this.stars.clear();
    for (let i = 0; i < 110; i++) {
      const x = (((i * 7919 + 31) % 10000) / 10000) * this.w,
        y = (((i * 3167 + 101) % 10000) / 10000) * this.h;
      this.stars.rect(x, y, i % 7 ? 1 : 1.5, i % 7 ? 1 : 1.5).fill({ color: 0xffffff, alpha: i % 7 ? 0.46 : 1 });
    }
    this.invalidate();
  }
  private layout() {
    const n = this.size,
      mobile = this.w < 760;
    const tw = (mobile ? 92 : Math.min(110, (this.w - 100) / n, (this.h - 190) / (n * 0.6))) * this.zoom;
    this.k = tw / TW;
    const th = tw * 0.6;
    this.ox = this.w / 2 + this.pan.x;
    this.oy = (this.h - 50) / 2 - ((n - 1) * th) / 2 + this.pan.y;
    const v = this.observation?.visible_state;
    if (mobile && v) {
      if (this.anchor === null) {
        const own = v.cities.filter((a) => a.owner === this.observation!.player);
        this.anchor = (
          own.find((a) => v.tiles[a.tile].capital === this.observation!.player) ||
          own[0] ||
          v.tiles.find((t) => t.known) || { tile: Math.floor((n * n) / 2) }
        ).tile;
      }
      const ax = this.anchor % n,
        ay = Math.floor(this.anchor / n);
      this.ox = this.w / 2 - ((ax - ay) * tw) / 2 + this.pan.x;
      this.oy = this.h * 0.48 - ((ax + ay) * th) / 2 + this.pan.y;
    }
  }
  private applyCamera() {
    this.layout();
    this.world.scale.set(this.k);
    this.world.position.set(this.ox, this.oy);
    this.overlay.style.transform = `matrix(${this.k},0,0,${this.k},${this.ox},${this.oy})`;
  }
  zoomBy(z: number) {
    this.zoom = Math.max(0.45, Math.min(2.7, this.zoom * z));
    this.invalidate();
  }
  center() {
    this.pan = { x: 0, y: 0 };
    this.zoom = 1;
    this.anchor = null;
    this.camera = null;
    this.invalidate();
  }
  // Mobile camera follow: glide only when the action happens off-screen.
  focus(tile: number) {
    if (this.drag || this.pinch || !this.w || !this.observation) return;
    this.layout();
    const { x, y } = this.screen(tile),
      margin = Math.min(60, this.w * 0.1),
      top = 100,
      bottom = this.h - 170;
    if (x > margin && x < this.w - margin && y > top && y < bottom) return;
    this.camera = {
      from: { ...this.pan },
      to: { x: this.pan.x + this.w / 2 - x, y: this.pan.y + (top + bottom) / 2 - y },
      t0: performance.now(),
    };
    this.invalidate();
  }
  private local(tile: number) {
    const n = this.size,
      col = tile % n,
      row = Math.floor(tile / n);
    return { x: (col - row) * HW, y: (col + row) * HH };
  }
  private screen(tile: number) {
    const p = this.local(tile);
    return { x: this.ox + p.x * this.k, y: this.oy + p.y * this.k };
  }
  hit(x: number, y: number) {
    this.layout();
    const a = (x - this.ox) / (HW * this.k),
      b = (y - this.oy) / (HH * this.k);
    const col = Math.round((a + b) / 2),
      row = Math.round((b - a) / 2);
    return col >= 0 && row >= 0 && col < this.size && row < this.size ? row * this.size + col : null;
  }

  private bindInput(wrap: HTMLElement) {
    wrap.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      this.camera = null;
      if (this.pointers.size === 2) {
        const [a, b] = [...this.pointers.values()];
        this.pinch = { distance: Math.hypot(a.x - b.x, a.y - b.y), zoom: this.zoom };
        this.drag = null;
        this.moved = true;
      } else {
        this.drag = { x: e.clientX, y: e.clientY, px: this.pan.x, py: this.pan.y };
        this.moved = false;
      }
    });
    const hover = (tile: number | null) => {
      if (tile === this.hovered) return;
      this.hovered = tile;
      this.onHover?.(tile);
    };
    wrap.addEventListener("pointermove", (e) => {
      if (e.pointerType !== "mouse" || this.drag) return;
      const r = this.canvas.getBoundingClientRect();
      hover(this.hit(e.clientX - r.left, e.clientY - r.top));
    });
    wrap.addEventListener("pointerleave", () => hover(null));
    window.addEventListener("pointermove", (e) => {
      if (!this.pointers.has(e.pointerId)) return;
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.pinch && this.pointers.size === 2) {
        const [a, b] = [...this.pointers.values()];
        this.zoom = Math.max(
          0.45,
          Math.min(2.7, (this.pinch.zoom * Math.hypot(a.x - b.x, a.y - b.y)) / Math.max(1, this.pinch.distance)),
        );
        this.invalidate();
        return;
      }
      if (!this.drag) return;
      const dx = e.clientX - this.drag.x,
        dy = e.clientY - this.drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 6) {
        this.moved = true;
        this.pan = { x: this.drag.px + dx, y: this.drag.py + dy };
        this.invalidate();
      }
    });
    const release = (e: PointerEvent) => {
      if (this.drag && !this.moved && e.type === "pointerup" && e.target === this.canvas) {
        const r = this.canvas.getBoundingClientRect();
        const tile = this.hit(e.clientX - r.left, e.clientY - r.top);
        if (tile !== null) this.select(tile);
      }
      this.pointers.delete(e.pointerId);
      this.drag = null;
      this.pinch = null;
    };
    window.addEventListener("pointerup", release);
    window.addEventListener("pointercancel", release);
    wrap.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.zoomBy(e.deltaY < 0 ? 1.1 : 1 / 1.1);
      },
      { passive: false },
    );
  }

  // ---------- state ----------
  get animating() {
    return !!this.running;
  }
  whenIdle() {
    return this.finished;
  }

  update(
    obs: Observation,
    selected: number | null,
    targets: LegalAction[],
    event: GameEvent | null,
    options: UpdateOptions,
  ) {
    const home = options.home ?? 1;
    const known = (obs.visible_state.relations as { known_players?: unknown[] } | undefined)?.known_players;
    const seats = Array.isArray(known) ? known.length : 2;
    if (home !== this.home || seats !== this.seats) {
      // Every team colour flips: rebuild all tiles and labels.
      this.home = home;
      this.seats = seats;
      this.builtSize = 0;
    }
    const changed =
      options.scene !== this.scene || obs.state_version !== this.observation?.state_version || !this.builtSize;
    if (changed) {
      const plan =
        options.scene === this.scene && options.animate !== false && !this.reducedMotion.matches && !document.hidden
          ? planTransition(this.observation, obs, event)
          : null;
      this.finishAnimation();
      this.showPreview(null);
      if (options.scene !== this.scene) this.anchor = null;
      this.observation = obs;
      this.scene = options.scene;
      this.size = obs.visible_state.board_size;
      this.selected = selected;
      this.rebuild();
      if (plan) this.startAnimation(plan, options.speed || 1);
    } else if (options.animate === false) this.finishAnimation();
    const layer = options.layer ?? null;
    if (selected !== this.selected || layer !== this.layer) this.selectedAt = performance.now();
    this.selected = selected;
    this.layer = layer;
    this.territory = new Set(options.territory || []);
    this.movable = new Set(options.ready || []);
    this.targets = targets;
    this.applySelection();
    this.invalidate();
  }

  // ---------- building the scene ----------
  private team(owner: number | undefined): Team {
    if (!owner) return TEAMS.neutral;
    if (this.seats > 2) return [TEAMS.blue, TEAMS.red, TEAMS.green, TEAMS.gold][(owner - 1) % 4];
    return owner === this.home ? TEAMS.blue : TEAMS.red;
  }
  private text(value: string | number, size: number, color = "#ffffff", shadow = false) {
    // Large glyphs scaled down stay crisp at every zoom level.
    const t = new BitmapText({
      text: String(value),
      style: {
        fontFamily: "Josefin Sans, Avenir Next, sans-serif",
        fontSize: 48,
        fill: color,
        ...(shadow ? { dropShadow: { color: "#000000", blur: 6, distance: 0, alpha: 0.9 } } : {}),
      },
    });
    t.scale.set(size / 48);
    t.anchor.set(0.5, 0.5);
    return t;
  }

  // One persistent node set per tile, in depth order.
  private reset(v: VisibleState) {
    this.clearMarkers();
    for (const layer of [this.ground, this.surface, this.objects, this.labelLayer])
      for (const child of layer.removeChildren()) dispose(child);
    this.tiles = [];
    this.labels.clear();
    const n = v.board_size,
      depth = (tile: number) => (tile % n) + Math.floor(tile / n);
    for (const tile of v.tiles.map((t) => t.tile).sort((a, b) => depth(a) - depth(b))) {
      const p = this.local(tile);
      const view: TileView = {
        base: new Container(),
        surf: new Container(),
        holder: new Container(),
        dim: shape("dim", (g) => g.poly(DIAMOND).fill({ color: 0x081828, alpha: 0.55 })),
        sel: this.art.selector(),
        groundKey: "",
        objectKey: "",
        dimmables: [],
      };
      view.dim.visible = view.sel.visible = false;
      const ground = new Container();
      ground.position.set(p.x, p.y);
      ground.addChild(view.base);
      this.ground.addChild(ground);
      const surface = new Container();
      surface.position.set(p.x, p.y);
      surface.addChild(view.surf, view.dim, view.sel);
      this.surface.addChild(surface);
      view.holder.position.set(p.x, p.y);
      view.holder.zIndex = depth(tile) * 100 + 10;
      this.objects.addChild(view.holder);
      this.tiles[tile] = view;
    }
    this.builtSize = n;
  }

  private rebuild() {
    if (!this.ready || !this.observation) return;
    const v = this.observation.visible_state;
    if (this.builtSize !== v.board_size || this.tiles.length !== v.tiles.length) this.reset(v);
    const units = new Map(v.units.map((u) => [u.tile, u])),
      cities = new Map(v.cities.map((c) => [c.tile, c]));
    for (const t of v.tiles) {
      const view = this.tiles[t.tile];
      const info = t.known ? this.groundInfo(t, v) : null;
      const groundKey = info
        ? `${t.terrain}|${t.road ? 1 : 0}|${info.team?.name}|${info.border}|${info.roads}|${info.shores}|${info.walls}`
        : "fog";
      if (groundKey !== view.groundKey) {
        view.groundKey = groundKey;
        for (const child of [...view.base.removeChildren(), ...view.surf.removeChildren()]) dispose(child);
        if (info) this.art.ground(t, info, view.base, view.surf);
      }
      const city = cities.get(t.tile),
        u = units.get(t.tile);
      const objectKey = t.known
        ? [
            t.terrain,
            t.resource,
            t.structure,
            t.structure_level ?? 0,
            city ? `${this.team(city.owner).name}:${city.level || 1}:${t.capital ? 1 : 0}` : "",
            u ? `${u.kind}:${this.team(u.owner).name}:${Math.ceil(u.health)}:${u.veteran ? 1 : 0}:${u.kills ?? 0}` : "",
          ].join("|")
        : "fog";
      if (objectKey === view.objectKey) continue;
      view.objectKey = objectKey;
      const holder = view.holder;
      for (const child of holder.removeChildren()) dispose(child);
      view.dimmables = [];
      view.terrain = view.city = view.structure = view.unit = undefined;
      if (!t.known) {
        holder.addChild(this.art.fog(t.tile));
        continue;
      }
      view.terrain = holder.addChild(this.art.terrain(t, view.dimmables));
      if (city)
        view.city = holder.addChild(
          this.art.city(this.team(city.owner), city.owner, city.level || 1, !!t.capital, view.dimmables),
        );
      else {
        const st = this.art.structure(t, view.dimmables);
        if (st) view.structure = holder.addChild(st);
      }
      if (u) view.unit = holder.addChild(this.unitView(u));
    }
    const owned = new Set<number>();
    for (const c of v.cities) {
      if (!c.owner) continue;
      owned.add(c.tile);
      const key = `${this.team(c.owner).name}|${c.level || 1}|${c.progress || 0}|${v.tiles[c.tile].capital ? 1 : 0}`,
        old = this.labels.get(c.tile);
      if (old?.key === key) continue;
      if (old) dispose(old.node);
      this.labelLayer.addChild(this.label(v, c.tile, c.owner, c.level || 1, c.progress || 0, this.local(c.tile), key));
    }
    for (const [tile, label] of this.labels)
      if (!owned.has(tile)) {
        dispose(label.node);
        this.labels.delete(tile);
      }
    this.placeTargets(v);
  }
  private neighbour(tile: number, k: number) {
    const n = this.size,
      col = (tile % n) + STEPS[k][0],
      row = Math.floor(tile / n) + STEPS[k][1];
    return col >= 0 && row >= 0 && col < n && row < n ? row * n + col : -1;
  }
  // Everything a tile's ground depends on from its neighbours (see GroundInfo).
  private groundInfo(t: Tile, v: VisibleState): GroundInfo {
    const at = (k: number) => {
      const id = this.neighbour(t.tile, k);
      return id >= 0 ? v.tiles[id] : null;
    };
    const water = (x: Tile | null) => !!x && x.known && (x.terrain === 1 || x.terrain === 2);
    let border = 0,
      shores = 0,
      roads = 0,
      walls = 0;
    EDGES.forEach((k, j) => {
      const o = at(k);
      if (t.owner && !(o && o.known && o.owner === t.owner)) border |= 1 << j;
      if (water(t) && o && o.known && !water(o)) shores |= 1 << j;
    });
    // Roads join road tiles, cities and villages, ports and bridges.
    const joins = (x: Tile | null) => !!x && x.known && (!!x.road || x.structure === 1 || x.structure === 8 || x.structure === 48);
    if (t.road || (joins(t) && STEPS.some((_, k) => at(k)?.road)))
      for (let k = 0; k < 8; k++) if (joins(at(k)) && (t.road || at(k)!.road)) roads |= 1 << k;
    // A side wall shows where the tile in front is lower: deep ocean before shallow water, the edge before ocean.
    [1, 7].forEach((k, b) => {
      const o = at(k);
      if (t.terrain === 1 ? o?.known && o.terrain === 2 : t.terrain === 2 && !o) walls |= 1 << b;
    });
    return { border, roads, shores, walls, team: t.owner ? this.team(t.owner) : null };
  }
  private unitView(u: Unit): UnitView {
    const c: UnitView = new Container();
    const art = this.art.unit(u, this.team(u.owner));
    c.addChild(art.body);
    if (art.flash) {
      art.flash.blendMode = "add";
      art.flash.alpha = 0;
      c.flash = c.addChild(art.flash);
    }
    c.hp = this.text(Math.ceil(u.health), art.hp.size, "#ffffff", art.hp.shadow);
    c.hp.position.set(art.hp.x, art.hp.y);
    c.shownHealth = Math.ceil(u.health);
    c.addChild(c.hp);
    if (u.kills && !u.veteran) c.addChild(this.art.xp(Math.min(3, u.kills)));
    if (u.veteran) c.addChild(art.veteran);
    art.ready.visible = false;
    c.ready = c.addChild(art.ready);
    return c;
  }
  // An effect star `size` across, wrapped so the effect can scale and move it freely.
  private starNode(size: number) {
    const c = new Container();
    c.addChild(this.art.star(0, 0, size));
    return c;
  }
  // Capitals reserve room on the left for their icon.
  private labelWidth(tile: number, capital: boolean) {
    return Math.max(62, cityName(tile).length * 6 + 26) + (capital ? 12 : 0);
  }
  private label(
    v: VisibleState,
    tile: number,
    owner: number,
    level: number,
    progress: number,
    p: { x: number; y: number },
    key: string,
  ) {
    const c = new Container();
    c.position.set(p.x, p.y);
    const capital = !!v.tiles[tile].capital,
      width = this.labelWidth(tile, capital),
      team = this.team(owner);
    const band = this.art.labelBand(team);
    c.addChild(shape(`label:${width}:${band.color}:${band.alpha}`, (g) => g.roundRect(-width / 2, 15, width, 16, 2).fill(band)));
    const name = this.text(cityName(tile), 12);
    name.position.set(capital ? -3 : -9, 23);
    const lv = this.text(level, 12);
    lv.position.set(width / 2 - 6, 23);
    c.addChild(name, lv);
    c.addChild(this.art.star(width / 2 - 16, 23, 11));
    if (capital) {
      const cap = this.art.capitalIcon(-width / 2 + 5, 23, 11);
      if (cap) c.addChild(cap);
    }
    const bar = new Graphics();
    c.addChild(bar);
    const view = { node: c, key, bar, width, city: { level, progress } };
    this.labels.set(tile, view);
    this.paintBar(view, level + 1, clamp(progress / (level + 1)));
    return c;
  }
  private paintBar(view: LabelView, segments: number, amount: number) {
    view.bar.clear();
    this.art.paintBar(view.bar, view.width, segments, amount);
  }

  // ---------- selection ----------
  private applySelection() {
    if (!this.ready || !this.observation) return;
    const territory = this.territory.size > 0;
    for (const [tile, view] of this.tiles.entries()) {
      if (!view) continue;
      view.sel.visible = tile === this.selected && !territory;
      view.dim.visible = territory && !this.territory.has(tile);
      if (view.unit) view.unit.position.y = 0;
      if (view.unit?.ready) view.unit.ready.visible = !this.running && this.movable.has(tile);
    }
    this.outline.clear();
    this.outlineTop.clear();
    if (territory) {
      const n = this.size,
        edges: [Point, Point][] = [];
      for (const tile of this.territory) {
        const p = this.local(tile),
          col = tile % n,
          row = Math.floor(tile / n);
        const corners: Point[] = [[p.x, p.y - HH], [p.x + HW, p.y], [p.x, p.y + HH], [p.x - HW, p.y]];
        this.outline.poly(corners.flat()).fill({ color: 0xffffff, alpha: 0.1 });
        [row > 0 ? tile - n : -1, col < n - 1 ? tile + 1 : -1, row < n - 1 ? tile + n : -1, col > 0 ? tile - 1 : -1].forEach(
          (other, j) => {
            if (!this.territory.has(other)) edges.push([corners[j], corners[(j + 1) % 4]]);
          },
        );
      }
      // Wide cyan glow on the ground, thick white border drawn over the trees
      // and houses so it reads like the mobile city selection; stacked
      // strokes instead of blur filters.
      for (const [g, width, color, alpha] of [
        [this.outline, 18, 0x00c8ff, 0.2],
        [this.outline, 10, 0x3fd6ff, 0.45],
        [this.outlineTop, 7, 0x00c8ff, 0.35],
        [this.outlineTop, 4, 0xffffff, 0.95],
      ] as const) {
        for (const [a, b] of edges) g.moveTo(a[0], a[1]).lineTo(b[0], b[1]);
        g.stroke({ width, color, alpha, cap: "round" });
      }
    }
    this.clearMarkers();
    if (!this.running)
      for (const target of this.targets) {
        const p = this.local(target.command.target);
        const m = new Container();
        m.position.set(p.x, p.y);
        m.zIndex = (p.y / HH) * 100 + 60;
        m.addChild(this.art.marker(target.command.kind === 2));
        this.objects.addChild(m);
        this.markers.push(m);
      }
    // The pop settles within 420 ms of the selection changing; later updates
    // (e.g. scrubbing with a selection held) need just one render.
    this.liveUntil = this.selectedAt + 420;
    this.stepSelection();
  }
  private clearMarkers() {
    for (const m of this.markers) dispose(m);
    this.markers = [];
  }
  // Selection pop: tile/marker grow, unit hop, territory dim fade-in.
  private stepSelection() {
    const age = (performance.now() - this.selectedAt) / 1000;
    const fade = ease(age / 0.18);
    const territory = this.territory.size > 0;
    for (const [tile, view] of this.tiles.entries()) {
      if (!view) continue;
      const dimmed = territory && !this.territory.has(tile);
      const tint = dimmed ? lerpTint(fade) : 0xffffff;
      for (const s of view.dimmables) s.tint = tint;
      view.dim.alpha = fade;
    }
    this.outline.alpha = this.outlineTop.alpha = fade;
    const grow = pop(age / 0.28);
    for (const m of this.markers) m.scale.set(grow);
    if (this.selected !== null) {
      const sel = this.tiles[this.selected]?.sel;
      if (sel) sel.scale.set(0.82 + 0.18 * pop(age / 0.3));
      const unit = this.tiles[this.selected]?.unit;
      if (unit && this.layer === "unit" && unit.visible) unit.position.y = -Math.sin(clamp(age / 0.32) * Math.PI) * 9;
    }
  }

  // ---------- attack preview ----------
  // A health bar over every unit the attack changes: what it keeps, and in red
  // what it loses. A unit that would die shows a skull instead (and, with `fade`, fades).
  showPreview(units: HealthPreview[] | null, fade = false) {
    for (const view of this.previewed) view.alpha = 1;
    this.previewed = [];
    for (const child of this.previewLayer.removeChildren()) dispose(child);
    if (!this.ready) return;
    for (const u of units ?? []) {
      // Drawn over the unit's own health badge (-15, -34), the bar reaching out
      // to its left, so it never covers the unit standing behind.
      const p = this.local(u.tile),
        c = new Container(),
        dead = u.after <= 0,
        w = 22,
        x0 = -44,
        keep = clamp(u.after / u.max),
        had = clamp(u.health / u.max);
      c.position.set(p.x, p.y);
      const g = new Graphics();
      g.roundRect(x0 - 4, -41.5, w + 30, 15, 7.5).fill({ color: 0x0b1620, alpha: 0.9 }).stroke({ width: 1.2, color: 0xffffff, alpha: 0.9 });
      g.roundRect(x0, -36.5, w, 5, 2.5).fill(0x35444f);
      g.roundRect(x0, -36.5, Math.max(2.5, w * had), 5, 2.5).fill(0xff4b3a);
      if (keep > 0) g.roundRect(x0, -36.5, Math.max(2.5, w * keep), 5, 2.5).fill(0x6fdc5a);
      g.circle(-15, -34, 8.5).fill(dead ? 0x0b1620 : 0xd8392b).stroke({ width: 1.5, color: 0xffffff });
      c.addChild(g);
      if (dead) c.addChild(this.art.skull(-15, -34, 13));
      else {
        const t = this.text(Math.ceil(u.after), 11, "#ffffff");
        t.position.set(-15, -33.5);
        c.addChild(t);
      }
      this.previewLayer.addChild(c);
      const view = this.tiles[u.tile]?.unit;
      if (dead && view && fade) {
        view.alpha = 0.45;
        this.previewed.push(view);
      }
    }
    this.invalidate();
  }

  // ---------- animation ----------
  private startAnimation(plan: Plan, speed: number) {
    const duration = Math.max(90, plan.duration / Math.max(1, speed));
    const fx = new Graphics();
    this.fxLayer.addChild(fx);
    const run: Running = {
      plan,
      duration,
      started: performance.now(),
      actors: [],
      fogs: new Map(),
      fx,
      texts: [],
      harvest: [],
      bursts: [],
    };
    this.running = run;
    this.finished = new Promise((resolve) => (this.resolveFinished = resolve));
    for (const tile of plan.hidden) if (this.tiles[tile]?.unit) this.tiles[tile].unit!.visible = false;
    for (const a of plan.actors) {
      const view = this.unitView(a.unit);
      this.objects.addChild(view);
      run.actors.push(view);
    }
    for (const r of plan.reveals) {
      const g = this.art.fog(r.tile),
        p = this.local(r.tile);
      g.position.set(p.x, p.y);
      g.zIndex = (p.y / HH) * 100 + 60;
      this.objects.addChild(g);
      run.fogs.set(r.tile, g);
    }
    for (const e of plan.effects) {
      if (e.type === "damage") {
        const t = this.text("−" + e.amount, 20, "#ff6b54", true);
        t.visible = false;
        this.fxLayer.addChild(t);
        run.texts.push({ node: t, tile: e.tile, at: e.at, dy: 0 });
      } else if (e.type === "harvest") {
        // The resource rises first (an empty node when the art has none), then six stars circle out.
        const c = new Container();
        c.addChild(this.art.resourceIcon(e.resource) ?? new Container());
        for (let i = 0; i < 6; i++) c.addChild(this.starNode(9));
        this.fxLayer.addChild(c);
        run.harvest.push({ node: c, tile: e.tile, at: e.at });
      } else if (e.type === "capture" || e.type === "levelup") {
        const c = new Container();
        for (let i = 0; i < 12; i++) c.addChild(this.starNode(11));
        const label = this.text(e.type === "levelup" ? `Level ${e.level}!` : "Captured!", 19, "#ffffff", true);
        c.addChild(label);
        c.visible = false;
        this.fxLayer.addChild(c);
        run.bursts.push({ node: c, tile: e.tile, at: e.at });
      }
    }
    if (plan.explorer) {
      run.explorer = this.art.explorer(this.team(plan.explorer.owner));
      this.objects.addChild(run.explorer);
      this.focus(plan.explorer.path[Math.floor(plan.explorer.path.length / 2)]);
    }
    const kind = plan.projectiles.length
      ? "projectile"
      : plan.actors.some((a) => a.type === "attack")
        ? "attack"
        : plan.actors.some((a) => a.type === "move")
          ? "move"
          : "damage";
    this.canvas.dataset.animation = kind;
    this.overlay.classList.add("animating");
    this.clearMarkers();
    clearTimeout(this.finishTimer);
    this.finishTimer = window.setTimeout(() => this.finishAnimation(), duration + 80);
    this.stepAnimation(0);
    this.canvas.dispatchEvent(
      new CustomEvent("bardur:animation-start", {
        detail: {
          kind,
          version: this.observation!.state_version,
          duration,
          actors: plan.actors.length,
          projectiles: plan.projectiles.length,
          effects: plan.effects.map((e) => e.type),
        },
      }),
    );
  }

  finishAnimation() {
    const run = this.running;
    if (!run) return;
    clearTimeout(this.finishTimer);
    this.running = null;
    for (const a of run.actors) dispose(a);
    for (const g of run.fogs.values()) dispose(g);
    if (run.explorer) dispose(run.explorer);
    for (const child of this.fxLayer.removeChildren()) dispose(child);
    for (const tile of run.plan.hidden) if (this.tiles[tile]?.unit) this.tiles[tile].unit!.visible = true;
    for (const p of run.plan.pops) {
      const view = this.tiles[p.tile];
      const node = p.layer === "unit" ? view?.unit : p.layer === "city" ? view?.city : view?.structure;
      node?.scale.set(1);
    }
    for (const b of run.plan.bars) {
      const label = this.labels.get(b.tile);
      if (label) this.paintBar(label, label.city.level + 1, clamp(label.city.progress / (label.city.level + 1)));
    }
    this.canvas.dataset.animation = "idle";
    this.overlay.classList.remove("animating");
    this.applySelection();
    this.invalidate();
    this.resolveFinished?.();
    this.resolveFinished = null;
    this.canvas.dispatchEvent(new CustomEvent("bardur:animation-end"));
  }

  private stepAnimation(p: number) {
    const run = this.running!,
      plan = run.plan;
    // Actors move through the depth-sorted layer, so trees in front occlude them.
    plan.actors.forEach((a, i) => {
      const view = run.actors[i],
        pose = sampleActor(a, p),
        multi = !!a.path && a.path.length > 2;
      const from = this.local(multi ? a.path![pose.segment] : a.from),
        to = this.local(multi ? a.path![pose.segment + 1] : a.to),
        t = multi ? pose.local : pose.travel;
      let x = from.x + (to.x - from.x) * t,
        y = from.y + (to.y - from.y) * t;
      if (a.aim !== undefined) {
        const aim = this.local(a.aim),
          origin = this.local(a.from);
        x += (aim.x - origin.x) * pose.lunge;
        y += (aim.y - origin.y) * pose.lunge;
      }
      view.position.set(x + pose.shake, y - pose.hop);
      view.zIndex = (y / HH) * 100 + 50;
      view.alpha = pose.alpha;
      view.scale.set(pose.scale);
      view.visible = pose.alpha > 0;
      if (view.flash) view.flash.alpha = Math.min(1, pose.flash * 1.6);
      const hp = Math.ceil(pose.health);
      if (view.hp && hp !== view.shownHealth) {
        view.hp.text = String(hp);
        view.shownHealth = hp;
      }
    });
    for (const pp of plan.pops) {
      const view = this.tiles[pp.tile];
      const node = pp.layer === "unit" ? view?.unit : pp.layer === "city" ? view?.city : view?.structure;
      node?.scale.set(Math.max(0.0001, pop((p - pp.at) / pp.span)));
    }
    for (const r of plan.reveals) {
      const g = run.fogs.get(r.tile)!,
        k = clamp((p - r.at) / 0.22);
      g.alpha = 1 - k;
      g.scale.set(1 + k * 0.35);
    }
    for (const b of plan.bars) {
      const label = this.labels.get(b.tile);
      if (!label || b.fromLevel === undefined) continue;
      const k = ease((p - 0.25) / 0.5);
      let segments = label.city.level + 1,
        amount: number;
      if (b.toLevel! > b.fromLevel && k < 0.6) {
        segments = b.fromLevel + 1;
        const start = b.from / segments;
        amount = start + (1 - start) * (k / 0.6);
      } else if (b.toLevel! > b.fromLevel) amount = (b.to / segments) * ((k - 0.6) / 0.4);
      else amount = (b.from + (b.to - b.from) * k) / segments;
      this.paintBar(label, segments, clamp(amount));
    }
    if (run.explorer && plan.explorer) {
      const e = plan.explorer,
        k = clamp((p - e.start) / (e.end - e.start)),
        steps = Math.max(1, e.path.length - 1),
        scaled = k * steps,
        seg = Math.min(steps - 1, Math.floor(scaled)),
        local = scaled - seg;
      const a = this.local(e.path[seg]),
        b = this.local(e.path[Math.min(seg + 1, e.path.length - 1)]);
      const y = a.y + (b.y - a.y) * ease(local);
      run.explorer.position.set(a.x + (b.x - a.x) * ease(local), y - Math.sin(local * Math.PI) * 8);
      run.explorer.zIndex = (y / HH) * 100 + 55;
      run.explorer.alpha = k >= 1 ? 0 : k > 0.94 ? (1 - k) / 0.06 : 1;
    }
    this.drawEffects(run, p);
  }

  private drawEffects(run: Running, p: number) {
    const g = run.fx,
      plan = run.plan;
    g.clear();
    for (const shot of plan.projectiles) {
      if (p < shot.start || p > shot.end) continue;
      const t = clamp((p - shot.start) / (shot.end - shot.start)),
        a = this.local(shot.from),
        b = this.local(shot.to);
      const heavy = [8, 41].includes(shot.kind),
        arc = heavy ? 50 : 15;
      const x = a.x + (b.x - a.x) * t,
        y = a.y + (b.y - a.y) * t - 22 - Math.sin(t * Math.PI) * arc;
      if (heavy) {
        g.circle(x, y, 7).fill({ color: 0xffb541, alpha: 0.35 });
        g.circle(x, y, 4.5).fill(shot.kind === 8 ? 0x675e4e : 0xffcf71);
        for (let j = 1; j < 5; j++)
          g.circle(x - (b.x - a.x) * 0.015 * j, y - (b.y - a.y) * 0.015 * j, Math.max(0.5, 3 - j * 0.6)).fill({
            color: 0xefc78c,
            alpha: 0.47,
          });
      } else {
        const angle = Math.atan2(b.y - a.y - Math.cos(t * Math.PI) * arc * Math.PI, b.x - a.x),
          cos = Math.cos(angle),
          sin = Math.sin(angle);
        const at = (dx: number, dy: number): Point => [x + dx * cos - dy * sin, y + dx * sin + dy * cos];
        g.moveTo(...at(-10, 0)).lineTo(...at(7, 0)).stroke({ width: 2, color: 0x513b28 });
        g.poly(flat([at(10, 0), at(3, -3), at(3, 3)])).fill(0xf8fcff);
        g.moveTo(...at(-8, 0)).lineTo(...at(-12, -3)).stroke({ width: 1.5, color: 0xffffff });
      }
    }
    for (const effect of plan.effects) {
      const age = p - effect.at;
      if (age < 0) continue;
      const at = this.local(effect.tile),
        x = at.x,
        y = at.y - 20;
      if (effect.type === "impact" && age < 0.22) {
        const t = age / 0.22;
        g.moveTo(x - 11 * (1 + t), y + 9).lineTo(x + 11 * (1 + t), y - 9).stroke({ width: 4, color: 0xfff5c4, alpha: 1 - t });
        g.moveTo(x - 5, y - 12).lineTo(x + 5, y + 12).stroke({ width: 2, color: 0xffffff, alpha: 1 - t });
        for (let i = 0; i < 7; i++) {
          const a = (i * Math.PI * 2) / 7,
            r = 8 + 19 * t;
          g.circle(x + Math.cos(a) * r, y + Math.sin(a) * r * 0.7, 2).fill({ color: 0xfff2c0, alpha: 1 - t });
        }
      } else if (effect.type === "death" && age < 0.34) {
        const t = age / 0.34;
        for (let i = 0; i < 9; i++) {
          const a = i * 2.39996,
            r = 6 + 25 * t;
          g.ellipse(x + Math.cos(a) * r, y + Math.sin(a) * r * 0.55 - 15 * t, 3 + 5 * t, 2 + 3 * t).fill({
            color: i % 2 ? 0xe8ebee : 0xaebbc9,
            alpha: (1 - t) * 0.85,
          });
        }
      } else if (effect.type === "convert" && age < 0.5) {
        const t = age / 0.5;
        g.ellipse(x, y, 8 + 30 * t, 12 + 35 * t).stroke({ width: 3, color: 0x72ecff, alpha: 1 - t });
      } else if (effect.type === "puff" && age < 0.4) {
        const t = age / 0.4;
        for (let i = 0; i < 8; i++) {
          const a = (i * Math.PI * 2) / 8,
            r = 10 + 26 * ease(t);
          g.ellipse(x + Math.cos(a) * r, y + 20 + Math.sin(a) * r * 0.5, 5 + 4 * t, 4 + 3 * t).fill({
            color: 0xf4f8fb,
            alpha: (1 - t) * 0.9,
          });
        }
      } else if (effect.type === "claim" && age < 0.5) {
        const t = age / 0.5;
        g.poly(flat([[x, at.y - HH], [x + HW, at.y], [x, at.y + HH], [x - HW, at.y]])).fill({
          color: effect.owner === 1 ? 0xd8f4ff : 0xffd6cc,
          alpha: Math.sin(t * Math.PI) * 0.6,
        });
      }
    }
    for (const f of plan.flights) {
      const age = p - f.at;
      if (age < 0 || age > 0.45) continue;
      const a = this.local(f.from),
        b = this.local(f.to);
      for (let i = 0; i < 3; i++) {
        const t = ease(clamp((age - i * 0.05) / 0.32));
        if (t <= 0 || t >= 1) continue;
        g.circle(a.x + (b.x - a.x) * t, a.y + (b.y + 36 - a.y) * t - Math.sin(t * Math.PI) * 40, 5).fill(0x00a7f5);
      }
    }
    for (const d of run.texts) {
      const age = p - d.at,
        at = this.local(d.tile);
      d.node.visible = age >= 0 && age < 0.34;
      if (!d.node.visible) continue;
      const t = clamp(age / 0.34);
      d.node.alpha = 1 - t * t;
      d.node.position.set(at.x + 14, at.y - 20 - (12 + 26 * ease(t)));
    }
    for (const h of run.harvest) {
      const age = p - h.at,
        at = this.local(h.tile);
      h.node.visible = age >= 0 && age < 0.75;
      if (!h.node.visible) continue;
      const t = age / 0.75;
      h.node.alpha = 1 - ease(t);
      h.node.position.set(at.x, at.y - 20);
      const [resource, ...stars] = h.node.children;
      if (resource) resource.position.y = 20 - 34 * ease(t);
      stars.forEach((s, i) => {
        const a = (i * Math.PI) / 3 + t * 2;
        s.position.set(Math.cos(a) * (12 + 24 * t), Math.sin(a) * (8 + 14 * t) - 10 * t + 4.5);
      });
    }
    for (const b of run.bursts) {
      const age = p - b.at,
        at = this.local(b.tile);
      b.node.visible = age >= 0 && age < 0.55;
      if (!b.node.visible) continue;
      const t = age / 0.55;
      b.node.alpha = 1 - t * t;
      b.node.position.set(at.x, at.y - 20);
      const items = b.node.children,
        label = items[items.length - 1];
      items.slice(0, -1).forEach((s, i) => {
        const a = (i * Math.PI * 2) / 12,
          r = 14 + 46 * ease(t);
        s.position.set(Math.cos(a) * r, -8 + Math.sin(a) * r * 0.6 + 5);
        s.scale.set((11 - 5 * t) / 11);
      });
      label.position.set(0, -(28 + 30 * ease(t)) + 20);
    }
  }

  // ---------- accessible tile buttons ----------
  // Board-space buttons that move with the camera via one CSS transform.
  private placeTargets(v: VisibleState) {
    const n = v.board_size;
    if (this.overlay.children.length !== v.tiles.length) {
      this.overlay.replaceChildren();
      v.tiles.forEach((t) => {
        const b = document.createElement("button");
        b.className = "tile-hit";
        b.dataset.tile = String(t.tile);
        const col = t.tile % n,
          row = Math.floor(t.tile / n);
        b.style.cssText = `left:${(col - row) * HW - HW}px;top:${(col + row) * HH - HH}px;width:${TW}px;height:${TH}px`;
        b.onclick = () => {
          if (!this.moved) this.select(t.tile);
        };
        b.addEventListener("keydown", (e) => {
          const delta = ({ ArrowRight: 1, ArrowLeft: -1, ArrowDown: n, ArrowUp: -n } as Record<string, number>)[e.key];
          if (delta) {
            e.preventDefault();
            (this.overlay.children[Math.max(0, Math.min(v.tiles.length - 1, t.tile + delta))] as HTMLElement).focus();
          }
        });
        this.overlay.append(b);
      });
    }
    const units = new Set(v.units.map((u) => u.tile)),
      cities = new Set(v.cities.map((c) => c.tile));
    const terrain: Record<number, string> = { 1: "water", 2: "ocean", 3: "field", 4: "mountain", 5: "forest" };
    for (const t of v.tiles) {
      const b = this.overlay.children[t.tile] as HTMLButtonElement;
      const label = `Tile ${t.tile}, column ${(t.tile % n) + 1}, row ${Math.floor(t.tile / n) + 1}, ${t.known ? terrain[t.terrain ?? 0] || "land" : "unexplored"}${units.has(t.tile) ? ", unit" : ""}${cities.has(t.tile) ? ", city" : ""}`;
      if (b.getAttribute("aria-label") !== label) b.setAttribute("aria-label", label);
      const tab = t.tile === this.selected || (this.selected === null && t.tile === 0) ? 0 : -1;
      if (b.tabIndex !== tab) b.tabIndex = tab;
    }
  }
}
