import { sprites, loadAssets, darkImage } from "./assets.js";
import {
  planTransition,
  sampleActor,
  clamp,
  ease,
  pop,
} from "./animation.js";
const CITY_NAMES = ["Guur", "Barduria", "Uld", "Linlak", "Vern", "Norg", "Skal", "Holm"];
export const cityName = (tile) => CITY_NAMES[(tile * 7) % CITY_NAMES.length];
const RESOURCE_SPRITES = { 1: "animal", 2: "crop", 5: "metal", 6: "fruit" };
export const cityTerritory = (v, cityTile) => {
  // Fallback for recordings without render territory: nearest same-owner city.
  const n = v.board_size,
    city = v.cities.find((c) => c.tile === cityTile);
  if (!city?.owner) return [];
  const d = (a, b) =>
    Math.max(
      Math.abs((a % n) - (b % n)),
      Math.abs(Math.floor(a / n) - Math.floor(b / n)),
    );
  const rivals = v.cities.filter((c) => c.owner === city.owner);
  return v.tiles
    .filter(
      (t) =>
        t.known &&
        t.owner === city.owner &&
        rivals.every((c) => d(t.tile, cityTile) <= d(t.tile, c.tile)),
    )
    .map((t) => t.tile);
};
// Public Midjiwan sprites with procedural terrain/fog and shared hit projection.
export class BoardRenderer {
  constructor(canvas, overlay, select) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.overlay = overlay;
    this.select = select;
    this.zoom = 1;
    this.pan = { x: 0, y: 0 };
    this.selected = null;
    this.targets = [];
    this.observation = null;
    this.drag = null;
    this.moved = false;
    this.anchor = null;
    this.animation = null;
    this.scene = null;
    this.layer = null;
    this.territory = new Set();
    this.selectedAt = 0;
    this.liveUntil = 0;
    this.finished = Promise.resolve();
    this.reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
    this.reducedMotion.addEventListener("change", () => {
      if (this.reducedMotion.matches) this.finishAnimation();
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) this.finishAnimation();
    });
    loadAssets(() => this.draw());
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    const wrap = canvas.parentElement;
    this.pointers = new Map();
    this.pinch = null;
    wrap.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.pointers.size === 2) {
        const [a, b] = [...this.pointers.values()];
        this.pinch = {
          distance: Math.hypot(a.x - b.x, a.y - b.y),
          zoom: this.zoom,
        };
        this.drag = null;
        this.moved = true;
      } else {
        this.drag = {
          x: e.clientX,
          y: e.clientY,
          px: this.pan.x,
          py: this.pan.y,
        };
        this.moved = false;
      }
    });
    window.addEventListener("pointermove", (e) => {
      if (!this.pointers.has(e.pointerId)) return;
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.pinch && this.pointers.size === 2) {
        const [a, b] = [...this.pointers.values()];
        this.zoom = Math.max(
          0.45,
          Math.min(
            2.7,
            (this.pinch.zoom * Math.hypot(a.x - b.x, a.y - b.y)) /
              Math.max(1, this.pinch.distance),
          ),
        );
        this.settleZoom();
        return;
      }
      if (!this.drag) return;
      const dx = e.clientX - this.drag.x,
        dy = e.clientY - this.drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 6) {
        this.moved = true;
        this.pan = { x: this.drag.px + dx, y: this.drag.py + dy };
        this.requestDraw();
      }
    });
    const release = (e) => {
      if (
        this.drag &&
        !this.moved &&
        e.type === "pointerup" &&
        e.target === canvas
      ) {
        const r = canvas.getBoundingClientRect();
        const i = this.hit(e.clientX - r.left, e.clientY - r.top);
        if (i !== null) this.select(i);
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
  resize() {
    const r = this.canvas.getBoundingClientRect();
    this.w = r.width;
    this.h = r.height;
    const d = window.devicePixelRatio || 1;
    this.canvas.width = r.width * d;
    this.canvas.height = r.height * d;
    this.ctx.setTransform(d, 0, 0, d, 0, 0);
    this.draw();
  }
  zoomBy(z) {
    this.zoom = Math.max(0.45, Math.min(2.7, this.zoom * z));
    this.settleZoom();
  }
  // While zooming, the cached board is scaled; it re-renders sharply once the gesture rests.
  settleZoom() {
    this.zoomingUntil = performance.now() + 160;
    this.requestDraw();
    clearTimeout(this.settleTimer);
    this.settleTimer = setTimeout(() => this.requestDraw(), 180);
  }
  center() {
    this.pan = { x: 0, y: 0 };
    this.zoom = 1;
    this.anchor = null;
    this.draw();
  }
  // Mobile camera follow: glide only when the action happens off-screen.
  focus(tile) {
    if (this.drag || this.pinch || !this.w || !this.observation) return;
    const { x, y } = this.project(tile),
      margin = Math.min(60, this.w * 0.1),
      top = 100,
      bottom = this.h - 170;
    if (x > margin && x < this.w - margin && y > top && y < bottom) return;
    const dx = this.w / 2 - x,
      dy = (top + bottom) / 2 - y,
      start = { ...this.pan },
      t0 = performance.now();
    cancelAnimationFrame(this.panFrame);
    const step = () => {
      const k = ease((performance.now() - t0) / 380);
      this.pan = { x: start.x + dx * k, y: start.y + dy * k };
      if (!this.animation) this.requestDraw();
      if (k < 1) this.panFrame = requestAnimationFrame(step);
    };
    this.panFrame = requestAnimationFrame(step);
  }
  get animating() {
    return !!this.animation;
  }
  whenIdle() {
    return this.finished;
  }
  finishAnimation() {
    if (!this.animation) return;
    cancelAnimationFrame(this.animationFrame);
    clearTimeout(this.animationTimeout);
    // Animated objects stay drawn live until an idle moment re-renders the
    // board image, so the end of a move never costs a frame.
    const rest = { ...this.animation, version: this.observation?.state_version };
    this.rest = rest;
    (window.requestIdleCallback || ((fn) => setTimeout(fn, 120)))(() => {
      if (this.rest !== rest) return;
      this.rest = null;
      this.requestDraw();
    });
    this.animation = null;
    this.canvas.dataset.animation = "idle";
    this.overlay.classList.remove("animating");
    this.draw();
    this.resolveAnimation?.();
    this.resolveAnimation = null;
    this.canvas.dispatchEvent(new CustomEvent("bardur:animation-end"));
  }
  update(obs, selected, targets, event, options = {}) {
    const scene = options.scene ?? "board";
    const changed =
      scene !== this.scene ||
      obs.state_version !== this.observation?.state_version;
    if (changed) {
      const plan =
        scene === this.scene &&
        options.animate !== false &&
        !this.reducedMotion.matches &&
        !document.hidden
          ? planTransition(this.observation, obs, event)
          : null;
      this.finishAnimation();
      this.observation = obs;
      this.scene = scene;
      if (plan) {
        const duration = Math.max(
          90,
          plan.duration / Math.max(1, options.speed || 1),
        );
        this.animation = { ...plan, duration, started: performance.now() };
        this.finished = new Promise((resolve) => {
          this.resolveAnimation = resolve;
        });
        this.canvas.dataset.animation = plan.projectiles.length
          ? "projectile"
          : plan.actors.some((a) => a.type === "attack")
            ? "attack"
            : plan.actors.some((a) => a.type === "move")
              ? "move"
              : "damage";
        this.overlay.classList.add("animating");
        const frame = () => {
          if (!this.animation) return;
          if (performance.now() - this.animation.started >= duration) {
            this.finishAnimation();
            return;
          }
          this.draw();
          this.animationFrame = requestAnimationFrame(frame);
        };
        this.animationFrame = requestAnimationFrame(frame);
        if (plan.explorer)
          this.focus(plan.explorer.path[Math.floor(plan.explorer.path.length / 2)]);
        this.animationTimeout = setTimeout(
          () => this.finishAnimation(),
          duration + 80,
        );
        this.canvas.dispatchEvent(
          new CustomEvent("bardur:animation-start", {
            detail: {
              kind: this.canvas.dataset.animation,
              version: obs.state_version,
              duration,
              actors: plan.actors.length,
              projectiles: plan.projectiles.length,
              effects: plan.effects.map((e) => e.type),
            },
          }),
        );
      }
    } else if (options.animate === false) this.finishAnimation();
    const layer = options.layer ?? null;
    if (selected !== this.selected || layer !== this.layer) {
      this.selectedAt = selected === null ? 0 : performance.now();
      this.kick(520);
    }
    this.observation = obs;
    this.selected = selected;
    this.layer = layer;
    this.territory = new Set(options.territory || []);
    this.targets = targets;
    this.event = event;
    this.draw();
  }
  // Short redraw loop for selection pops while no transition is running.
  kick(ms) {
    this.liveUntil = performance.now() + ms;
    if (this.liveFrame) return;
    const frame = () => {
      this.liveFrame = 0;
      if (!this.animation) this.draw();
      if (performance.now() < this.liveUntil)
        this.liveFrame = requestAnimationFrame(frame);
    };
    this.liveFrame = requestAnimationFrame(frame);
  }
  selectionAge() {
    return this.selectedAt ? (performance.now() - this.selectedAt) / 1000 : 9;
  }
  progress() {
    const a = this.animation;
    return a ? clamp((performance.now() - a.started) / a.duration) : 1;
  }
  popScale(tile, layer) {
    const a = this.animation;
    if (!a?.pops) return 1;
    const entry = a.pops.find((p) => p.tile === tile && p.layer === layer);
    if (!entry) return 1;
    return pop((this.progress() - entry.at) / entry.span);
  }
  withScale(scale, x, y, draw) {
    if (scale === 1) return draw();
    if (scale <= 0.01) return;
    const c = this.ctx;
    c.save();
    c.translate(x, y);
    c.scale(scale, scale);
    c.translate(-x, -y);
    draw();
    c.restore();
  }
  project(i) {
    const n = this.size;
    return {
      x: this.ox + (((i % n) - Math.floor(i / n)) * this.tw) / 2,
      y: this.oy + (((i % n) + Math.floor(i / n)) * this.th) / 2,
    };
  }
  hit(x, y) {
    const a = (x - this.ox) / (this.tw / 2),
      b = (y - this.oy) / (this.th / 2);
    const col = Math.round((a + b) / 2),
      row = Math.round((b - a) / 2);
    return col >= 0 && row >= 0 && col < this.size && row < this.size
      ? row * this.size + col
      : null;
  }
  poly(points, fill, stroke, width = 1) {
    const c = this.ctx;
    c.beginPath();
    points.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y)));
    c.closePath();
    if (fill) {
      c.fillStyle = fill;
      c.fill();
    }
    if (stroke) {
      c.strokeStyle = stroke;
      c.lineWidth = width;
      c.stroke();
    }
  }
  line(points, color, width = 1) {
    const c = this.ctx;
    c.beginPath();
    points.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y)));
    c.strokeStyle = color;
    c.lineWidth = width;
    c.lineCap = "round";
    c.lineJoin = "round";
    c.stroke();
  }
  ellipse(x, y, rx, ry, fill) {
    const c = this.ctx;
    c.beginPath();
    c.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    c.fillStyle = fill;
    c.fill();
  }
  text(s, x, y, size, color = "#304752", align = "center", weight = 600) {
    const c = this.ctx;
    c.font = `${weight} ${size}px Josefin Sans, Avenir Next, sans-serif`;
    c.textAlign = align;
    c.textBaseline = "middle";
    c.fillStyle = color;
    c.fillText(s, x, y);
  }
  cube(x, y, w, d, h, top = "#fafcfb", left = "#c4d1d1", right = "#9fbbc3") {
    this.poly(
      [
        [x, y - h - d / 2],
        [x + w / 2, y - h],
        [x, y - h + d / 2],
        [x - w / 2, y - h],
      ],
      top,
    );
    this.poly(
      [
        [x - w / 2, y - h],
        [x, y - h + d / 2],
        [x, y + d / 2],
        [x - w / 2, y],
      ],
      left,
    );
    this.poly(
      [
        [x, y - h + d / 2],
        [x + w / 2, y - h],
        [x + w / 2, y],
        [x, y + d / 2],
      ],
      right,
    );
  }
  tree(x, y, s, variant = 0) {
    this.ellipse(x + 1 * s, y + 2 * s, 10 * s, 4 * s, "#54727020");
    this.poly(
      [
        [x - 2 * s, y],
        [x + 2 * s, y + 1 * s],
        [x + 2 * s, y - 13 * s],
        [x - 2 * s, y - 13 * s],
      ],
      "#6b6658",
    );
    for (let j = 0; j < 3; j++) {
      const yy = y - j * 7 * s;
      const width = (13 - j * 3) * s;
      const top = yy - (18 - j * 2) * s;
      this.poly(
        [
          [x - width, yy - 4 * s],
          [x, top],
          [x, yy + 1 * s],
        ],
        variant ? "#365e55" : "#355b50",
      );
      this.poly(
        [
          [x, top],
          [x + width, yy - 4 * s],
          [x, yy + 1 * s],
        ],
        variant ? "#477564" : "#477763",
      );
      this.poly(
        [
          [x, top],
          [x - width * 0.52, top + 8 * s],
          [x, top + 5 * s],
          [x + width * 0.47, top + 8 * s],
        ],
        "#e6efdf",
      );
    }
  }
  mountain(x, y, s) {
    this.ellipse(x, y, 23 * s, 8 * s, "#607b7d20");
    this.poly(
      [
        [x - 25 * s, y - 1 * s],
        [x - 5 * s, y - 37 * s],
        [x + 3 * s, y + 9 * s],
      ],
      "#93a6a6",
    );
    this.poly(
      [
        [x - 5 * s, y - 37 * s],
        [x + 23 * s, y],
        [x + 3 * s, y + 9 * s],
      ],
      "#b9c7c5",
    );
    this.poly(
      [
        [x - 5 * s, y - 37 * s],
        [x - 15 * s, y - 19 * s],
        [x - 8 * s, y - 22 * s],
        [x - 4 * s, y - 16 * s],
        [x + 1 * s, y - 23 * s],
        [x + 9 * s, y - 18 * s],
      ],
      "#f9fbf5",
    );
    this.poly(
      [
        [x + 11 * s, y - 27 * s],
        [x + 30 * s, y + 1 * s],
        [x + 17 * s, y + 7 * s],
        [x + 1 * s, y - 2 * s],
      ],
      "#a6b8b7",
    );
    this.poly(
      [
        [x + 11 * s, y - 27 * s],
        [x + 3 * s, y - 12 * s],
        [x + 10 * s, y - 15 * s],
        [x + 14 * s, y - 10 * s],
        [x + 20 * s, y - 13 * s],
      ],
      "#f9fbf7",
    );
  }
  house(x, y, s, owner, small = false) {
    const h = (small ? 10 : 15) * s,
      w = (small ? 17 : 22) * s,
      d = w * 0.65;
    this.cube(x, y, w, d, h, "#d3ddd7", "#a9bdb7", "#e2e6da");
    const roof = owner === 2 ? "#b36054" : owner === 1 ? "#3f7584" : "#887b6a";
    this.poly(
      [
        [x - w * 0.66, y - h],
        [x - w * 0.05, y - h - 12 * s],
        [x + w * 0.66, y - h],
        [x, y - h + d * 0.65],
      ],
      roof,
    );
    this.poly(
      [
        [x - w * 0.66, y - h],
        [x - w * 0.05, y - h - 12 * s],
        [x + w * 0.1, y - h - 8 * s],
        [x - w * 0.42, y - h + 2 * s],
      ],
      "#eef3e7",
    );
    this.poly(
      [
        [x + 3 * s, y - 2 * s],
        [x + 8 * s, y - 4 * s],
        [x + 8 * s, y - 11 * s],
        [x + 3 * s, y - 9 * s],
      ],
      "#536762",
    );
  }
  city(x, y, s, city, tile) {
    const owner = city.owner,
      level = city.level || 1;
    this.cube(
      x,
      y + 2 * s,
      53 * s,
      27 * s,
      2 * s,
      "#c9d9ce",
      "#b6c4ba",
      "#a0b9b2",
    );
    this.house(x - 12 * s, y - 6 * s, s * 0.72, owner, true);
    this.house(x + 12 * s, y - 5 * s, s * 0.8, owner, true);
    this.house(x, y + 4 * s, s, owner);
    if (level > 2) this.house(x + 19 * s, y + 5 * s, s * 0.65, owner, true);
    if (owner) {
      this.line(
        [
          [x - 18 * s, y],
          [x - 18 * s, y - 38 * s],
        ],
        "#5e746f",
        1.6 * s,
      );
      this.poly(
        [
          [x - 18 * s, y - 38 * s],
          [x - 5 * s, y - 34 * s],
          [x - 18 * s, y - 29 * s],
        ],
        owner === 1 ? "#3289b6" : "#d16c57",
      );
      if (tile.capital)
        this.text(
          "★",
          x - 18 * s,
          y - 44 * s,
          10 * s,
          owner === 1 ? "#347fa1" : "#c16550",
        );
    }
  }
  deer(x, y, s) {
    this.ellipse(x, y, 8 * s, 3 * s, "#30464e15");
    this.poly(
      [
        [x - 6 * s, y - 7 * s],
        [x + 4 * s, y - 7 * s],
        [x + 7 * s, y - 13 * s],
        [x + 10 * s, y - 12 * s],
        [x + 7 * s, y - 3 * s],
        [x - 5 * s, y - 3 * s],
      ],
      "#997b57",
    );
    this.line(
      [
        [x - 4 * s, y - 4 * s],
        [x - 4 * s, y],
        [x - 2 * s, y],
      ],
      "#775b43",
      1.6 * s,
    );
    this.line(
      [
        [x + 4 * s, y - 4 * s],
        [x + 5 * s, y],
      ],
      "#775b43",
      1.6 * s,
    );
    this.line(
      [
        [x + 8 * s, y - 12 * s],
        [x + 7 * s, y - 18 * s],
        [x + 4 * s, y - 20 * s],
      ],
      "#73583f",
      1.2 * s,
    );
    this.line(
      [
        [x + 7 * s, y - 17 * s],
        [x + 11 * s, y - 20 * s],
      ],
      "#73583f",
      1.2 * s,
    );
  }
  unit(x, y, s, u) {
    const color = u.owner === 1 ? "#287ba7" : "#c85a4c",
      light = u.owner === 1 ? "#5da4c5" : "#df8a71";
    s *= u.kind === 12 ? 1.25 : 1;
    this.ellipse(x, y + 3 * s, 13 * s, 5 * s, "#29495635");
    if (u.kind >= 39 && u.kind <= 45) {
      this.poly(
        [
          [x - 20 * s, y - 2 * s],
          [x + 11 * s, y + 9 * s],
          [x + 24 * s, y - 2 * s],
          [x - 10 * s, y - 11 * s],
        ],
        "#766451",
      );
      this.poly(
        [
          [x - 20 * s, y - 2 * s],
          [x + 11 * s, y + 9 * s],
          [x + 8 * s, y + 13 * s],
          [x - 15 * s, y + 4 * s],
        ],
        "#4d554d",
      );
      this.line(
        [
          [x, y],
          [x, y - 34 * s],
        ],
        "#6c6a52",
        2 * s,
      );
      this.poly(
        [
          [x + 2 * s, y - 32 * s],
          [x + 18 * s, y - 12 * s],
          [x + 2 * s, y - 10 * s],
        ],
        color,
      );
    } else {
      if (u.kind === 3 || u.kind === 4) {
        this.ellipse(x, y - 1 * s, 16 * s, 7 * s, "#7c6c55");
        this.poly(
          [
            [x + 9 * s, y - 2 * s],
            [x + 11 * s, y - 18 * s],
            [x + 18 * s, y - 17 * s],
            [x + 17 * s, y - 3 * s],
          ],
          "#7c6c55",
        );
        y -= 6 * s;
      }
      if (u.kind === 8) {
        this.cube(x, y, 25 * s, 15 * s, 9 * s, "#9e8d6e", "#776c54", "#b49b6f");
        this.line(
          [
            [x - 7 * s, y],
            [x + 9 * s, y - 26 * s],
          ],
          "#77654c",
          4 * s,
        );
        this.ellipse(x + 9 * s, y - 26 * s, 5 * s, 3 * s, "#596b6d");
      } else {
        this.poly(
          [
            [x - 6 * s, y - 4 * s],
            [x - 7 * s, y + 3 * s],
            [x - 1 * s, y + 3 * s],
            [x, y - 3 * s],
          ],
          "#42504b",
        );
        this.poly(
          [
            [x + 1 * s, y - 3 * s],
            [x + 2 * s, y + 3 * s],
            [x + 8 * s, y + 3 * s],
            [x + 6 * s, y - 5 * s],
          ],
          "#42504b",
        );
        this.poly(
          [
            [x - 7 * s, y - 20 * s],
            [x + 6 * s, y - 20 * s],
            [x + 9 * s, y - 5 * s],
            [x - 9 * s, y - 5 * s],
          ],
          color,
        );
        this.poly(
          [
            [x, y - 20 * s],
            [x + 6 * s, y - 20 * s],
            [x + 9 * s, y - 5 * s],
            [x, y - 6 * s],
          ],
          light,
        );
        this.ellipse(x, y - 25 * s, 7 * s, 8 * s, "#e7d4ac");
        this.poly(
          [
            [x - 8 * s, y - 27 * s],
            [x - 6 * s, y - 34 * s],
            [x + 5 * s, y - 35 * s],
            [x + 8 * s, y - 27 * s],
          ],
          "#534c43",
        );
        this.poly(
          [
            [x - 8 * s, y - 29 * s],
            [x - 13 * s, y - 35 * s],
            [x - 12 * s, y - 28 * s],
            [x - 7 * s, y - 24 * s],
          ],
          "#eee9cf",
        );
        this.poly(
          [
            [x + 7 * s, y - 29 * s],
            [x + 12 * s, y - 35 * s],
            [x + 12 * s, y - 28 * s],
            [x + 7 * s, y - 24 * s],
          ],
          "#eee9cf",
        );
        this.line(
          [
            [x - 3 * s, y - 25 * s],
            [x - 2 * s, y - 25 * s],
          ],
          "#504b43",
          1.5 * s,
        );
        this.line(
          [
            [x + 3 * s, y - 25 * s],
            [x + 4 * s, y - 25 * s],
          ],
          "#504b43",
          1.5 * s,
        );
        if (u.kind === 9) {
          const c = this.ctx;
          c.beginPath();
          c.arc(x + 12 * s, y - 13 * s, 9 * s, -1.4, 1.4);
          c.strokeStyle = "#7c6851";
          c.lineWidth = 2 * s;
          c.stroke();
          this.line(
            [
              [x + 13 * s, y - 22 * s],
              [x + 13 * s, y - 4 * s],
            ],
            "#b4bdac",
            s,
          );
        } else if (u.kind === 5) {
          this.poly(
            [
              [x + 6 * s, y - 20 * s],
              [x + 17 * s, y - 16 * s],
              [x + 16 * s, y - 5 * s],
              [x + 10 * s, y],
            ],
            "#e2d5ae",
            "#7e8a7c",
            s,
          );
        } else if (u.kind === 10) {
          this.line(
            [
              [x + 13 * s, y],
              [x + 13 * s, y - 32 * s],
            ],
            "#8b7758",
            2 * s,
          );
          this.ellipse(x + 13 * s, y - 34 * s, 4 * s, 4 * s, "#b6cae5");
        } else {
          this.line(
            [
              [x + 11 * s, y - 7 * s],
              [x + 15 * s, y - 23 * s],
            ],
            u.kind === 11 ? "#b5c9cf" : "#7f6b4e",
            3 * s,
          );
          this.line(
            [
              [x + 9 * s, y - 11 * s],
              [x + 16 * s, y - 9 * s],
            ],
            "#aaa583",
            2 * s,
          );
        }
      }
    }
    const hp = Math.ceil(u.health);
    this.ellipse(x + 15 * s, y + 4 * s, 8 * s, 6 * s, "#fffffff0");
    this.text(hp, x + 15 * s, y + 4 * s, 9 * s, color);
    if (u.veteran) this.text("★", x - 11 * s, y - 38 * s, 10 * s, "#c4a03a");
    if (u.moved && u.attacked)
      this.ellipse(x - 13 * s, y + 3 * s, 3 * s, 3 * s, "#a4b3b4");
  }
  building(x, y, s, tile) {
    const st = tile.structure;
    if (st === 5) {
      this.poly(
        [
          [x - 24 * s, y],
          [x, y - 12 * s],
          [x + 24 * s, y],
          [x, y + 12 * s],
        ],
        "#c6ac69",
      );
      for (let j = -2; j <= 2; j++)
        this.line(
          [
            [x + j * 5 * s - 9 * s, y - 5 * s],
            [x + j * 5 * s + 9 * s, y + 5 * s],
          ],
          "#e4d89c",
          2 * s,
        );
    } else if (st === 2) {
      this.cube(
        x - 8 * s,
        y,
        9 * s,
        7 * s,
        18 * s,
        "#d7dbcd",
        "#9baba4",
        "#c4cec5",
      );
      this.cube(
        x + 9 * s,
        y,
        9 * s,
        7 * s,
        14 * s,
        "#d7dbcd",
        "#9baba4",
        "#c4cec5",
      );
      this.poly(
        [
          [x - 12 * s, y - 17 * s],
          [x + 13 * s, y - 20 * s],
          [x + 13 * s, y - 15 * s],
          [x - 12 * s, y - 12 * s],
        ],
        "#b9c3b8",
      );
    } else if (st === 12) {
      this.house(x, y, s * 0.65, tile.owner, true);
      this.line(
        [
          [x + 7 * s, y + 5 * s],
          [x + 18 * s, y + 1 * s],
        ],
        "#886d4d",
        4 * s,
      );
    } else if (st > 2 && st !== 3) {
      this.house(x, y, s * 0.8, tile.owner);
      if ([6, 13, 22, 50].includes(st)) {
        this.line(
          [
            [x, y - 22 * s],
            [x, y - 42 * s],
          ],
          "#f4f5e8",
          2 * s,
        );
        this.line(
          [
            [x - 10 * s, y - 37 * s],
            [x + 10 * s, y - 27 * s],
          ],
          "#dddcc2",
          3 * s,
        );
        this.line(
          [
            [x - 10 * s, y - 27 * s],
            [x + 10 * s, y - 37 * s],
          ],
          "#dddcc2",
          3 * s,
        );
      }
    }
  }
  sprite(name, x, y, width) {
    const a = sprites[name];
    if (!a) return false;
    const h = (width * a.h) / a.w;
    this.ctx.drawImage(
      this.dimming ? darkImage(a) : a.image,
      a.x,
      a.y,
      a.w,
      a.h,
      x - width / 2,
      y - h,
      width,
      h,
    );
    return true;
  }
  fog(x, y, s, i) {
    const w = this.tw / 2,
      h = this.th / 2;
    const top = [x, y - h],
      right = [x + w, y],
      bottom = [x, y + h],
      left = [x - w, y];
    const peak = [x + (((i * 17) % 13) - 6) * s, y - 10 * s];
    this.poly([top, right, peak], "#ffffff");
    this.poly([right, bottom, peak], i % 2 ? "#b1c2f5" : "#c2d2f9");
    this.poly([bottom, left, peak], "#faffff");
    this.poly([left, top, peak], i % 3 ? "#8c9dd5" : "#c4d0f1");
    this.poly([peak, [x + w * 0.35, y + h * 0.3], bottom], "#e8efff");
  }
  // One draw per display frame, however many pointer events arrive.
  requestDraw() {
    if (this.pendingFrame) return;
    this.pendingFrame = requestAnimationFrame(() => {
      this.pendingFrame = 0;
      this.draw();
    });
  }
  layout(v) {
    this.size = v.board_size;
    const mobile = this.w < 760;
    this.tw =
      (mobile
        ? 92
        : Math.min(
            110,
            (this.w - 100) / this.size,
            (this.h - 190) / (this.size * 0.6),
          )) * this.zoom;
    this.th = this.tw * 0.6;
    this.ox = this.w / 2 + this.pan.x;
    this.oy = (this.h - 50) / 2 - ((this.size - 1) * this.th) / 2 + this.pan.y;
    if (mobile) {
      if (this.anchor === null) {
        const own = v.cities.filter((a) => a.owner === this.observation.player);
        this.anchor = (
          own.find(
            (a) => v.tiles[a.tile].capital === this.observation.player,
          ) ||
          own[0] ||
          v.tiles.find((t) => t.known) || {
            tile: Math.floor((this.size * this.size) / 2),
          }
        ).tile;
      }
      const ax = this.anchor % this.size,
        ay = Math.floor(this.anchor / this.size);
      this.ox = this.w / 2 - ((ax - ay) * this.tw) / 2 + this.pan.x;
      this.oy = this.h * 0.48 - ((ax + ay) * this.th) / 2 + this.pan.y;
    }
  }
  starLayer() {
    const d = window.devicePixelRatio || 1,
      key = `${this.w}x${this.h}@${d}`;
    if (this.starKey !== key) {
      const canvas = (this.starCanvas ||= document.createElement("canvas"));
      canvas.width = Math.ceil(this.w * d);
      canvas.height = Math.ceil(this.h * d);
      const c = canvas.getContext("2d");
      c.setTransform(d, 0, 0, d, 0, 0);
      for (let i = 0; i < 110; i++) {
        const x = (((i * 7919 + 31) % 10000) / 10000) * this.w,
          y = (((i * 3167 + 101) % 10000) / 10000) * this.h;
        c.fillStyle = i % 7 ? "#ffffff75" : "#fff";
        c.fillRect(x, y, i % 7 ? 1 : 1.5, i % 7 ? 1 : 1.5);
      }
      this.starKey = key;
    }
    return this.starCanvas;
  }
  // Objects drawn live each frame instead of baked into the board image.
  restingAnimation() {
    const r = this.rest;
    return r && r.version === this.observation?.state_version ? r : null;
  }
  liveTiles() {
    const a = this.animation || this.restingAnimation(),
      units = new Set(a?.hidden || []),
      cities = new Set(),
      structures = new Set();
    for (const p of a?.pops || [])
      (p.layer === "unit" ? units : p.layer === "city" ? cities : structures).add(p.tile);
    if (this.layer === "unit" && this.selected !== null) units.add(this.selected);
    return { units, cities, structures };
  }
  // The static board is rendered once into an offscreen image and reused
  // while panning, zooming or animating; it rebuilds only when its content changes.
  sceneLayer(v) {
    const live = this.liveTiles(),
      liveKey = ["units", "cities", "structures"]
        .map((k) => [...live[k]].sort((a, b) => a - b).join(","))
        .join(";");
    const base = [
      this.scene,
      this.observation.state_version,
      this.size,
      this.selected,
      this.layer,
      this.territory.size,
      liveKey,
      Object.keys(sprites).length,
      window.devicePixelRatio || 1,
    ].join("/");
    const key = base + "/" + this.tw.toFixed(3);
    const cache = this.cache;
    if (cache?.key === key) return cache;
    // Mid-gesture zoom scales the existing image; it re-renders once settled.
    if (cache?.base === base && performance.now() < this.zoomingUntil) return cache;
    const n = this.size,
      tw = this.tw,
      th = this.th,
      s = tw / 64,
      padX = tw * 0.9,
      padTop = tw * 1.2,
      W = n * tw + 2 * padX,
      H = padTop + n * th + 50 * s;
    // 2x density is visually indistinguishable for this art and keeps rebuilds cheap.
    const d = Math.min(2, window.devicePixelRatio || 1),
      scale = Math.max(
        0.25,
        Math.min(d, 8192 / W, 8192 / H, Math.sqrt(16e6 / (W * H))),
      );
    this.pool ||= [document.createElement("canvas"), document.createElement("canvas")];
    const canvas = this.pool[0] === cache?.canvas ? this.pool[1] : this.pool[0];
    // Keep the previous image for the selection dim cross-fade.
    if (
      cache &&
      cache.territory !== this.territory.size > 0 &&
      cache.tw === tw &&
      cache.version === this.observation.state_version
    ) {
      this.fadeFrom = cache;
      this.fadeStart = performance.now();
    } else this.fadeFrom = null;
    canvas.width = Math.ceil(W * scale);
    canvas.height = Math.ceil(H * scale);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    const screen = { ctx: this.ctx, w: this.w, h: this.h, ox: this.ox, oy: this.oy };
    Object.assign(this, { ctx, w: W, h: H, ox: n * tw / 2 + padX, oy: padTop + th / 2 });
    try {
      this.paintScene(v, s);
    } finally {
      Object.assign(this, screen);
    }
    this.cache = {
      canvas,
      key,
      base,
      tw,
      cox: n * tw / 2 + padX,
      coy: padTop + th / 2,
      w: W,
      h: H,
      territory: this.territory.size > 0,
      version: this.observation.state_version,
    };
    return this.cache;
  }
  blit(layer, alpha = 1) {
    const k = this.tw / layer.tw,
      c = this.ctx;
    c.globalAlpha = alpha;
    c.drawImage(
      layer.canvas,
      this.ox - layer.cox * k,
      this.oy - layer.coy * k,
      layer.w * k,
      layer.h * k,
    );
    c.globalAlpha = 1;
  }
  draw() {
    const c = this.ctx;
    if (!c || !this.w) return;
    c.clearRect(0, 0, this.w, this.h);
    c.drawImage(this.starLayer(), 0, 0, this.w, this.h);
    if (!this.observation) {
      this.text(
        "Preparing the square…",
        this.w / 2,
        this.h / 2,
        16,
        "#fff",
        "center",
        400,
      );
      return;
    }
    const v = this.observation.visible_state;
    this.layout(v);
    const s = this.tw / 64;
    const scene = this.sceneLayer(v);
    const fade = this.fadeFrom
      ? ease((performance.now() - this.fadeStart) / 180)
      : 1;
    if (fade < 1) {
      this.blit(this.fadeFrom);
      this.blit(scene, fade);
    } else {
      this.fadeFrom = null;
      this.blit(scene);
    }
    this.drawLive(v, s);
    this.placeTargets(v);
  }
  drawLive(v, s) {
    const c = this.ctx,
      a = this.animation;
    if (this.territory.size) this.territoryOutline(s);
    const units = new Map(v.units.map((u) => [u.tile, u]));
    if (this.layer === "unit" && !a?.hidden.has(this.selected)) {
      const u = units.get(this.selected);
      if (u) {
        const { x, y } = this.project(this.selected),
          hop = Math.sin(clamp(this.selectionAge() / 0.32) * Math.PI) * 9 * s;
        this.drawUnit(u, x, y - hop, s);
      }
    }
    const rest = !a && this.restingAnimation();
    if (rest) {
      const cities = new Map(v.cities.map((x) => [x.tile, x]));
      const tiles = new Set([...rest.hidden, ...(rest.pops || []).map((p) => p.tile)]);
      for (const tile of [...tiles].sort((p, q) => this.project(p).y - this.project(q).y)) {
        const { x, y } = this.project(tile);
        const kinds = new Set((rest.pops || []).filter((p) => p.tile === tile).map((p) => p.layer));
        if (kinds.has("city")) this.paintCity(cities.get(tile), x, y, s);
        if (kinds.has("structure")) this.paintStructure(v.tiles[tile], x, y, s);
        const u = units.get(tile);
        if (u && (rest.hidden.has(tile) || kinds.has("unit")) && !(this.layer === "unit" && tile === this.selected))
          this.drawUnit(u, x, y, s);
      }
    }
    if (a) {
      const cities = new Map(v.cities.map((x) => [x.tile, x]));
      for (const p of a.pops || []) {
        const { x, y } = this.project(p.tile),
          scale = this.popScale(p.tile, p.layer);
        if (p.layer === "unit") {
          const u = units.get(p.tile);
          if (u && !a.hidden.has(p.tile)) this.drawUnit(u, x, y, s, { scale });
        } else
          this.withScale(scale, x, y + 9 * s, () =>
            p.layer === "city"
              ? this.paintCity(cities.get(p.tile), x, y, s)
              : this.paintStructure(v.tiles[p.tile], x, y, s),
          );
      }
      for (const r of a.reveals || []) {
        const k = clamp((this.progress() - r.at) / 0.22);
        if (k >= 1) continue;
        const { x, y } = this.project(r.tile);
        c.save();
        c.globalAlpha = 1 - k;
        this.withScale(1 + k * 0.35, x, y, () => this.fog(x, y, s, r.tile));
        c.restore();
      }
    }
    this.drawAnimation(s);
    this.drawMarkers(s);
    for (const bar of a?.bars || []) {
      const city = v.cities.find((x) => x.tile === bar.tile);
      if (!city?.owner || bar.fromLevel === undefined) continue;
      const k = ease((this.progress() - 0.25) / 0.5);
      let segments = (city.level || 1) + 1,
        amount;
      if (bar.toLevel > bar.fromLevel && k < 0.6) {
        segments = bar.fromLevel + 1;
        const start = bar.from / segments;
        amount = start + (1 - start) * (k / 0.6);
      } else if (bar.toLevel > bar.fromLevel)
        amount = (bar.to / segments) * ((k - 0.6) / 0.4);
      else amount = (bar.from + (bar.to - bar.from) * k) / segments;
      const { x, y } = this.project(city.tile);
      this.paintBar(city, x, y, s, segments, clamp(amount));
    }
    this.drawAnimation(s, true);
  }
  // Static board content, painted into the cached scene image.
  paintScene(v, s) {
    const c = this.ctx,
      live = this.liveTiles();
    const units = new Map(v.units.map((u) => [u.tile, u])),
      cities = new Map(v.cities.map((u) => [u.tile, u]));
    const order = [...v.tiles].sort(
      (a, b) =>
        (a.tile % this.size) +
        Math.floor(a.tile / this.size) -
        ((b.tile % this.size) + Math.floor(b.tile / this.size)),
    );
    for (const t of order) {
      const { x, y } = this.project(t.tile),
        w = this.tw / 2,
        h = this.th / 2;
      if (
        x + w < -30 ||
        x - w > this.w + 30 ||
        y + h < -70 ||
        y - h > this.h + 70
      )
        continue;
      const water = t.terrain === 1 || t.terrain === 2,
        depth = 7 * s;
      this.poly(
        [
          [x - w, y],
          [x, y + h],
          [x, y + h + depth],
          [x - w, y + depth],
        ],
        t.known ? (water ? "#208dd1" : "#58221f") : "#7787ba",
      );
      this.poly(
        [
          [x, y + h],
          [x + w, y],
          [x + w, y + depth],
          [x, y + h + depth],
        ],
        t.known ? (water ? "#25b5ed" : "#824b16") : "#afbae1",
      );
      if (!t.known) {
        this.poly(
          [
            [x, y - h],
            [x + w, y],
            [x, y + h],
            [x - w, y],
          ],
          "#d6e1fa",
        );
        continue;
      }
      if (!water && sprites.ground) {
        const a = sprites.ground;
        c.drawImage(
          a.image,
          0,
          0,
          a.image.width,
          a.image.width * 0.612,
          x - w,
          y - h,
          this.tw,
          this.th,
        );
      } else {
        const a = [x, y - h],
          b = [x + w, y],
          d = [x, y + h],
          e = [x - w, y],
          mid = [x + 9 * s, y - 2 * s];
        this.poly([a, b, d, e], water ? "#4abcf0" : "#ddffff");
        this.poly([a, b, mid], water ? "#64c8f1" : "#d2fcff");
        this.poly([b, d, mid], water ? "#4db5e9" : "#f3ffff");
        this.poly([d, e, mid], water ? "#6bd2f4" : "#c8f8fc");
      }
      if (t.road)
        this.line(
          [
            [x - w, y],
            [x, y + h * 0.4],
            [x + w, y],
          ],
          "#9b7e59",
          2 * s,
        );
      if (t.owner) {
        const col = t.tile % this.size,
          row = Math.floor(t.tile / this.size),
          color = t.owner === 1 ? "#10171c" : "#cb3026";
        const corners = [
            [x, y - h],
            [x + w, y],
            [x, y + h],
            [x - w, y],
          ],
          adjacent = [
            row > 0 ? t.tile - this.size : -1,
            col < this.size - 1 ? t.tile + 1 : -1,
            row < this.size - 1 ? t.tile + this.size : -1,
            col > 0 ? t.tile - 1 : -1,
          ];
        adjacent.forEach((id, j) => {
          if (id >= 0 && v.tiles[id].known && v.tiles[id].owner === t.owner)
            return;
          const a = corners[j],
            b = corners[(j + 1) % 4];
          for (let k = 0; k < 5; k++) {
            const lo = k / 5 + 0.03,
              hi = (k + 1) / 5 - 0.03;
            const xx = a[0] + (b[0] - a[0]) * lo,
              yy = a[1] + (b[1] - a[1]) * lo,
              x2 = a[0] + (b[0] - a[0]) * hi,
              y2 = a[1] + (b[1] - a[1]) * hi;
            this.poly(
              [
                [xx, yy],
                [x2, y2],
                [x2, y2 - 7 * s],
                [xx, yy - 7 * s],
              ],
              color,
            );
          }
        });
      }
      const diamond = [
        [x, y - h],
        [x + w, y],
        [x, y + h],
        [x - w, y],
      ];
      // A selected city keeps its own land bright and dims everything else.
      if (this.territory.size && !this.territory.has(t.tile))
        this.poly(diamond, "rgba(8,24,40,0.42)");
      if (t.tile === this.selected && !this.territory.size)
        this.poly(diamond, "#ffffff38", "#ffffff", 2.5 * s);
    }
    for (const t of order) {
      const { x, y } = this.project(t.tile);
      if (!t.known) {
        this.fog(x, y, s, t.tile);
        continue;
      }
      // Pre-darkened sprites dim land outside a selected city (no canvas filters).
      this.dimming = this.territory.size > 0 && !this.territory.has(t.tile);
      if (t.terrain === 4) {
        if (!this.sprite("mountain", x, y + this.th * 0.48, this.tw))
          this.mountain(x, y, s);
      }
      if (t.terrain === 5 && t.structure !== 1) {
        if (!this.sprite("forest", x, y + this.th * 0.45, this.tw * 0.96))
          this.tree(x, y, s);
      }
      if (t.resource === 1)
        this.sprite("animal", x - 6 * s, y + 13 * s, this.tw * 0.43);
      if (t.resource === 6)
        this.sprite("fruit", x, y + this.th * 0.4, this.tw * 0.75);
      if (t.resource === 2)
        this.sprite("crop", x, y + this.th * 0.42, this.tw * 0.75);
      if (t.resource === 5)
        this.sprite("metal", x + 10 * s, y + this.th * 0.38, this.tw * 0.33);
      if (t.resource === 3) {
        for (let j = 0; j < 3; j++) {
          const xx = x + (j - 1) * 10 * s,
            yy = y + j * 4 * s;
          this.poly(
            [
              [xx - 5 * s, yy - 3 * s],
              [xx + 5 * s, yy],
              [xx - 5 * s, yy + 3 * s],
              [xx - 1 * s, yy],
            ],
            "#249bd6",
          );
        }
      }
      const city = cities.get(t.tile);
      if (city) {
        if (!live.cities.has(t.tile)) this.paintCity(city, x, y, s);
      } else if (!live.structures.has(t.tile)) this.paintStructure(t, x, y, s);
      this.dimming = false;
      const u = units.get(t.tile);
      if (u && !live.units.has(t.tile)) this.drawUnit(u, x, y, s);
    }
    for (const city of v.cities) {
      if (!city.owner) continue;
      const { x, y } = this.project(city.tile);
      this.paintLabel(v, city, x, y, s);
    }
  }
  paintCity(city, x, y, s) {
    if (!city) return;
    if (city.owner) {
      const lev = city.level || 1;
      this.sprite(
        "house-" + (lev > 5 ? 7 : Math.max(1, lev)),
        x,
        y + 9 * s,
        this.tw * (lev > 3 ? 0.7 : 0.53),
      );
      if (lev >= 3)
        this.sprite("house-1", x - 17 * s, y + 13 * s, this.tw * 0.3);
      if (lev >= 4)
        this.sprite("house-2", x + 19 * s, y + 11 * s, this.tw * 0.32);
    } else {
      this.sprite("house-1", x - 8 * s, y + 6 * s, this.tw * 0.31);
      this.sprite("house-1", x + 9 * s, y + 5 * s, this.tw * 0.26);
    }
  }
  paintStructure(t, x, y, s) {
    const key = {
      2: "ruin",
      5: "farm",
      6: "windmill",
      8: "port",
      12: "lumber",
      13: "sawmill",
      17: "temple",
      18: "temple",
      19: "temple",
      20: "temple",
      21: "mine",
      22: "forge",
      50: "market",
    }[t.structure];
    if (key) this.sprite(key, x, y + this.th * 0.37, this.tw * 0.74);
    else if (t.structure > 2 && t.structure !== 3) this.building(x, y, s, t);
  }
  labelWidth(city, s) {
    return Math.max(62, cityName(city.tile).length * 6 + 26) * s;
  }
  paintLabel(v, city, x, y, s) {
    const c = this.ctx,
      name = cityName(city.tile),
      level = city.level || 1,
      width = this.labelWidth(city, s);
    c.fillStyle = city.owner === 1 ? "#0008" : "#851b2b99";
    c.fillRect(x - width / 2, y + 15 * s, width, 16 * s);
    this.text(name, x - 9 * s, y + 23 * s, 12 * s, "#fff", "center", 400);
    this.sprite("star", x + width / 2 - 16 * s, y + 29 * s, 12 * s);
    this.text(level, x + width / 2 - 6 * s, y + 23 * s, 12 * s, "#fff", "center", 400);
    if (v.tiles[city.tile].capital)
      this.sprite("capital", x - width / 2 + 5 * s, y + 29 * s, 11 * s);
    this.paintBar(city, x, y, s, level + 1, clamp((city.progress || 0) / (level + 1)));
  }
  paintBar(city, x, y, s, segments, amount) {
    const c = this.ctx,
      width = this.labelWidth(city, s);
    c.fillStyle = "#c6d4dc";
    c.beginPath();
    c.roundRect(x - width / 2, y + 33 * s, width, 7 * s, 3 * s);
    c.fill();
    if (amount > 0) {
      c.fillStyle = "#00a7f5";
      c.beginPath();
      c.roundRect(x - width / 2, y + 33 * s, width * amount, 7 * s, 3 * s);
      c.fill();
    }
    for (let j = 1; j < Math.min(segments, 9); j++)
      this.line(
        [
          [x - width / 2 + (width * j) / segments, y + 33 * s],
          [x - width / 2 + (width * j) / segments, y + 40 * s],
        ],
        "#84949d",
        0.5 * s,
      );
  }
  drawUnit(u, x, y, s, pose = {}) {
    const c = this.ctx;
    c.save();
    c.globalAlpha = pose.alpha ?? 1;
    if (pose.flash)
      c.filter = `brightness(${1 + pose.flash * 3}) saturate(${1 - pose.flash})`;
    if (pose.scale !== undefined && pose.scale !== 1) {
      c.translate(x, y);
      c.scale(pose.scale, pose.scale);
      c.translate(-x, -y);
    }
    const names = {
      2: "warrior",
      3: "rider",
      4: "knight",
      5: "defender",
      8: "catapult",
      9: "archer",
      10: "mindbender",
      11: "swordsman",
      12: "giant",
      37: "dagger",
      38: "cloak",
      39: "boat",
      40: "pirate",
      41: "bombership",
      42: "scoutship",
      43: "boat",
      44: "rammership",
      45: "juggernaut",
    };
    const key = "unit-" + names[u.kind],
      a = sprites[key];
    this.ellipse(
      x,
      y + 5 * s,
      11 * s,
      6 * s,
      u.owner === 1 ? "#45c4f055" : "#ff464c99",
    );
    if (a) {
      const height = this.tw * (u.kind === 12 ? 0.95 : 0.72),
        width = (height * a.w) / a.h;
      this.sprite(key, x, y + 9 * s, width);
    } else this.unit(x, y - 2 * s, s, u);
    c.save();
    c.shadowColor = "#000";
    c.shadowBlur = 3 * s;
    this.text(
      Math.ceil(u.health),
      x - 15 * s,
      y - 34 * s,
      13 * s,
      "#fff",
      "center",
      400,
    );
    c.restore();
    if (u.veteran) this.sprite("star", x + 14 * s, y - 30 * s, 9 * s);

    c.restore();
  }
  drawAnimation(s, effectsOnly = false) {
    const animation = this.animation;
    if (!animation) return;
    const c = this.ctx,
      p = clamp((performance.now() - animation.started) / animation.duration);
    if (!effectsOnly) {
      const poses = animation.actors
        .map((a) => {
          const pose = sampleActor(a, p),
            multi = a.path && a.path.length > 2,
            from = this.project(multi ? a.path[pose.segment] : a.from),
            to = this.project(multi ? a.path[pose.segment + 1] : a.to),
            t = multi ? pose.local : pose.travel;
          let x = from.x + (to.x - from.x) * t,
            y = from.y + (to.y - from.y) * t - pose.hop * s;
          if (a.aim !== undefined) {
            const aim = this.project(a.aim);
            x += (aim.x - from.x) * pose.lunge;
            y += (aim.y - from.y) * pose.lunge;
          }
          return { a, pose, x: x + pose.shake * s, y };
        })
        .sort((a, b) => a.y - b.y);
      for (const { a, pose, x, y } of poses) {
        if (pose.alpha <= 0) continue;
        this.drawUnit({ ...a.unit, health: pose.health }, x, y, s, pose);
      }
      if (animation.explorer) this.drawExplorer(animation.explorer, p, s);
      return;
    }
    for (const shot of animation.projectiles) {
      if (p < shot.start || p > shot.end) continue;
      const t = clamp((p - shot.start) / (shot.end - shot.start)),
        a = this.project(shot.from),
        b = this.project(shot.to);
      const heavy = [8, 41].includes(shot.kind),
        arc = (heavy ? 50 : 15) * s;
      const x = a.x + (b.x - a.x) * t,
        y = a.y + (b.y - a.y) * t - 22 * s - Math.sin(t * Math.PI) * arc;
      if (heavy) {
        c.save();
        c.shadowColor = "#ffb541";
        c.shadowBlur = 10 * s;
        this.ellipse(
          x,
          y,
          4.5 * s,
          4.5 * s,
          shot.kind === 8 ? "#675e4e" : "#ffcf71",
        );
        c.restore();
        for (let j = 1; j < 5; j++)
          this.ellipse(
            x - (b.x - a.x) * 0.015 * j,
            y - (b.y - a.y) * 0.015 * j,
            Math.max(0.5, 3 - j * 0.6) * s,
            Math.max(0.5, 3 - j * 0.6) * s,
            "#efc78c77",
          );
      } else {
        const angle = Math.atan2(
          b.y - a.y - Math.cos(t * Math.PI) * arc * Math.PI,
          b.x - a.x,
        );
        c.save();
        c.translate(x, y);
        c.rotate(angle);
        this.line(
          [
            [-10 * s, 0],
            [7 * s, 0],
          ],
          "#513b28",
          2 * s,
        );
        this.poly(
          [
            [10 * s, 0],
            [3 * s, -3 * s],
            [3 * s, 3 * s],
          ],
          "#f8fcff",
        );
        this.line(
          [
            [-8 * s, 0],
            [-12 * s, -3 * s],
          ],
          "#fff",
          1.5 * s,
        );
        c.restore();
      }
    }
    for (const effect of animation.effects) {
      const age = p - effect.at;
      if (age < 0) continue;
      const at = this.project(effect.tile),
        x = at.x,
        y = at.y - 20 * s;
      c.save();
      if (effect.type === "damage") {
        const t = clamp(age / 0.34);
        c.globalAlpha = 1 - t * t;
        c.shadowColor = "#000";
        c.shadowBlur = 4 * s;
        this.text(
          "−" + effect.amount,
          x + 14 * s,
          y - (12 + 26 * ease(t)) * s,
          20 * s,
          "#ff6b54",
          "center",
          600,
        );
      } else if (effect.type === "impact" && age < 0.22) {
        const t = age / 0.22;
        c.globalAlpha = 1 - t;
        this.line(
          [
            [x - 11 * s * (1 + t), y + 9 * s],
            [x + 11 * s * (1 + t), y - 9 * s],
          ],
          "#fff5c4",
          4 * s,
        );
        this.line(
          [
            [x - 5 * s, y - 12 * s],
            [x + 5 * s, y + 12 * s],
          ],
          "#fff",
          2 * s,
        );
        for (let i = 0; i < 7; i++) {
          const a = (i * Math.PI * 2) / 7,
            r = (8 + 19 * t) * s;
          this.ellipse(
            x + Math.cos(a) * r,
            y + Math.sin(a) * r * 0.7,
            2 * s,
            2 * s,
            "#fff2c0",
          );
        }
      } else if (effect.type === "death" && age < 0.34) {
        const t = age / 0.34;
        c.globalAlpha = (1 - t) * 0.85;
        for (let i = 0; i < 9; i++) {
          const a = i * 2.39996,
            r = (6 + 25 * t) * s;
          this.ellipse(
            x + Math.cos(a) * r,
            y + Math.sin(a) * r * 0.55 - 15 * t * s,
            (3 + 5 * t) * s,
            (2 + 3 * t) * s,
            i % 2 ? "#e8ebee" : "#aebbc9",
          );
        }
      } else if (effect.type === "puff" && age < 0.4) {
        const t = age / 0.4;
        c.globalAlpha = (1 - t) * 0.9;
        for (let i = 0; i < 8; i++) {
          const a = (i * Math.PI * 2) / 8,
            r = (10 + 26 * ease(t)) * s;
          this.ellipse(
            x + Math.cos(a) * r,
            y + 20 * s + Math.sin(a) * r * 0.5,
            (5 + 4 * t) * s,
            (4 + 3 * t) * s,
            "#f4f8fb",
          );
        }
      } else if (effect.type === "harvest" && age < 0.75) {
        const t = age / 0.75;
        c.globalAlpha = 1 - ease(t);
        const sprite = RESOURCE_SPRITES[effect.resource];
        this.withScale(1 + 0.25 * t, x, y + 20 * s, () => {
          if (!sprite || !this.sprite(sprite, x, y + 20 * s - 34 * ease(t) * s, this.tw * 0.55))
            this.ellipse(x, y - 10 * s - 30 * t * s, 8 * s, 5 * s, "#5bc3ff");
        });
        for (let i = 0; i < 6; i++) {
          const a = (i * Math.PI) / 3 + t * 2;
          this.sprite(
            "star",
            x + Math.cos(a) * (12 + 24 * t) * s,
            y + Math.sin(a) * (8 + 14 * t) * s - 10 * t * s,
            9 * s,
          );
        }
      } else if (effect.type === "claim" && age < 0.5) {
        const t = age / 0.5,
          w = this.tw / 2,
          h = this.th / 2;
        c.globalAlpha = Math.sin(t * Math.PI) * 0.6;
        this.poly(
          [
            [x, at.y - h],
            [x + w, at.y],
            [x, at.y + h],
            [x - w, at.y],
          ],
          effect.owner === 1 ? "#d8f4ff" : "#ffd6cc",
        );
      } else if (
        (effect.type === "capture" || effect.type === "levelup") &&
        age < 0.55
      ) {
        const t = age / 0.55;
        c.globalAlpha = 1 - t * t;
        for (let i = 0; i < 12; i++) {
          const a = (i * Math.PI * 2) / 12,
            r = (14 + 46 * ease(t)) * s;
          this.sprite(
            "star",
            x + Math.cos(a) * r,
            y - 8 * s + Math.sin(a) * r * 0.6,
            (11 - 5 * t) * s,
          );
        }
        c.shadowColor = "#000";
        c.shadowBlur = 5 * s;
        this.text(
          effect.type === "capture" ? "Captured!" : `Level ${effect.level}!`,
          x,
          y - (28 + 30 * ease(t)) * s,
          19 * s,
          "#fff",
          "center",
          600,
        );
      } else if (effect.type === "convert" && age < 0.5) {
        const t = age / 0.5;
        c.globalAlpha = 1 - t;
        c.strokeStyle = "#72ecff";
        c.lineWidth = 3 * s;
        c.beginPath();
        c.ellipse(x, y, (8 + 30 * t) * s, (12 + 35 * t) * s, 0, 0, Math.PI * 2);
        c.stroke();
      }
      c.restore();
    }
    for (const flight of animation.flights || []) {
      const age = p - flight.at;
      if (age < 0 || age > 0.45) continue;
      const a = this.project(flight.from),
        b = this.project(flight.to);
      for (let i = 0; i < 3; i++) {
        const t = ease(clamp((age - i * 0.05) / 0.32));
        if (t <= 0 || t >= 1) continue;
        this.ellipse(
          a.x + (b.x - a.x) * t,
          a.y + (b.y + 36 * s - a.y) * t - Math.sin(t * Math.PI) * 40 * s,
          5 * s,
          5 * s,
          "#00a7f5",
        );
      }
    }
  }
  drawExplorer(walk, p, s) {
    const k = clamp((p - walk.start) / (walk.end - walk.start));
    if (k >= 1) return;
    const steps = Math.max(1, walk.path.length - 1),
      scaled = k * steps,
      seg = Math.min(steps - 1, Math.floor(scaled)),
      local = scaled - seg;
    const a = this.project(walk.path[seg]),
      b = this.project(walk.path[Math.min(seg + 1, walk.path.length - 1)]);
    const x = a.x + (b.x - a.x) * ease(local),
      y = a.y + (b.y - a.y) * ease(local) - Math.sin(local * Math.PI) * 8 * s;
    const c = this.ctx;
    c.save();
    c.globalAlpha = k > 0.94 ? (1 - k) / 0.06 : 1;
    // A hooded explorer in tribe colour, walking with a staff.
    this.ellipse(x, y + 4 * s, 10 * s, 4 * s, "#29495640");
    this.poly(
      [
        [x - 8 * s, y],
        [x - 5 * s, y - 20 * s],
        [x + 5 * s, y - 20 * s],
        [x + 8 * s, y],
      ],
      "#8a7a63",
    );
    this.poly(
      [
        [x - 6 * s, y - 15 * s],
        [x + 6 * s, y - 15 * s],
        [x + 5 * s, y - 11 * s],
        [x - 5 * s, y - 11 * s],
      ],
      walk.owner === 1 ? "#287ba7" : "#c85a4c",
    );
    this.ellipse(x, y - 25 * s, 6 * s, 6 * s, "#e7d4ac");
    this.poly(
      [
        [x - 7 * s, y - 24 * s],
        [x, y - 36 * s],
        [x + 7 * s, y - 24 * s],
        [x + 3 * s, y - 28 * s],
        [x - 3 * s, y - 28 * s],
      ],
      "#6c5e4a",
    );
    this.line(
      [
        [x + 10 * s, y + 2 * s],
        [x + 12 * s, y - 30 * s],
      ],
      "#6b5236",
      2 * s,
    );
    c.restore();
  }
  drawMarkers(s) {
    if (!this.targets.length || this.animation) return;
    const grow = pop(this.selectionAge() / 0.28);
    for (const target of this.targets) {
      const { x, y } = this.project(target.command.target);
      this.withScale(grow, x, y, () => {
        if (target.command.kind === 2) {
          this.ellipse(x, y, 17 * s, 10 * s, "#e8402fd0");
          this.ellipse(x, y, 11 * s, 6.5 * s, "#fff");
          this.ellipse(x, y, 6 * s, 3.5 * s, "#e8402f");
          this.line([[x - 21 * s, y], [x - 13 * s, y]], "#fff", 2 * s);
          this.line([[x + 13 * s, y], [x + 21 * s, y]], "#fff", 2 * s);
        } else if (!this.sprite("move-target", x, y + 9 * s, 30 * s)) {
          this.ellipse(x, y, 14 * s, 8.5 * s, "#9ccbffe0");
          this.ellipse(x, y, 9 * s, 5.5 * s, "#fff");
          this.ellipse(x, y, 4.5 * s, 2.8 * s, "#7fcaff");
        }
      });
    }
  }
  territoryOutline(s) {
    const n = this.size,
      fade = ease(this.selectionAge() / 0.18),
      edges = [];
    for (const tile of this.territory) {
      const { x, y } = this.project(tile),
        w = this.tw / 2,
        h = this.th / 2,
        col = tile % n,
        row = Math.floor(tile / n);
      const corners = [
        [x, y - h],
        [x + w, y],
        [x, y + h],
        [x - w, y],
      ];
      [
        row > 0 ? tile - n : -1,
        col < n - 1 ? tile + 1 : -1,
        row < n - 1 ? tile + n : -1,
        col > 0 ? tile - 1 : -1,
      ].forEach((other, j) => {
        if (!this.territory.has(other))
          edges.push([corners[j], corners[(j + 1) % 4]]);
      });
    }
    const c = this.ctx;
    c.save();
    c.globalAlpha = fade;
    c.shadowColor = "#00c8ff";
    c.shadowBlur = 10 * s;
    for (const [a, b] of edges) this.line([a, b], "#ffffff", 3.4 * s);
    c.restore();
  }
  // Hit areas live in board space; panning moves them with a single transform.
  placeTargets(v) {
    const n = v.board_size;
    if (this.overlay.children.length !== v.tiles.length) {
      this.overlay.replaceChildren();
      this.layoutKey = this.labelKey = null;
      v.tiles.forEach((t) => {
        const b = document.createElement("button");
        b.className = "tile-hit";
        b.dataset.tile = t.tile;
        b.onclick = () => {
          if (!this.moved) this.select(t.tile);
        };
        b.addEventListener("keydown", (e) => {
          const delta = {
            ArrowRight: 1,
            ArrowLeft: -1,
            ArrowDown: this.size,
            ArrowUp: -this.size,
          }[e.key];
          if (delta) {
            e.preventDefault();
            this.overlay.children[
              Math.max(0, Math.min(v.tiles.length - 1, t.tile + delta))
            ].focus();
          }
        });
        this.overlay.append(b);
      });
    }
    const layoutKey = `${n}/${this.tw}`;
    if (layoutKey !== this.layoutKey) {
      this.layoutKey = layoutKey;
      for (const t of v.tiles) {
        const col = t.tile % n,
          row = Math.floor(t.tile / n);
        this.overlay.children[t.tile].style.cssText =
          `left:${((col - row) * this.tw) / 2 - this.tw / 2}px;top:${((col + row) * this.th) / 2 - this.th / 2}px;width:${this.tw}px;height:${this.th}px`;
      }
    }
    const labelKey = `${this.scene}/${this.observation.state_version}/${this.selected}`;
    if (labelKey !== this.labelKey) {
      this.labelKey = labelKey;
      const units = new Set(v.units.map((u) => u.tile)),
        cities = new Set(v.cities.map((c) => c.tile));
      for (const t of v.tiles) {
        const b = this.overlay.children[t.tile];
        b.setAttribute(
          "aria-label",
          `Tile ${t.tile}, column ${(t.tile % n) + 1}, row ${Math.floor(t.tile / n) + 1}, ${t.known ? { 1: "water", 2: "ocean", 3: "field", 4: "mountain", 5: "forest" }[t.terrain] || "land" : "unexplored"}${units.has(t.tile) ? ", unit" : ""}${cities.has(t.tile) ? ", city" : ""}`,
        );
        b.tabIndex =
          t.tile === this.selected || (this.selected === null && t.tile === 0)
            ? 0
            : -1;
      }
    }
    this.overlay.style.transform = `translate(${this.ox}px, ${this.oy}px)`;
  }
}
