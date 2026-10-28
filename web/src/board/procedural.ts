// Placeholder shapes for the mock art mode (art/mock.ts), in the Polytopia
// style: faces lit from the left, team-coloured cloth, roofs and sails so a
// player's own pieces and the enemy's read at a glance. Tile-local units,
// origin at the diamond centre; things stand on y ≈ 6.
import type { GraphicsContext } from "pixi.js";
import { TEAMS, type Team } from "./geometry.ts";


const SKIN = 0xf3c99b,
  WOOD = 0x8a5a33,
  WOOD_DARK = 0x5e3b20,
  STEEL = 0xc9d3dc,
  IRON = 0x8e9aa5,
  BOOT = 0x3d2f25,
  WALL = 0xf1e3c4,
  WALL_SHADE = 0xcfb98f,
  WHITE = 0xffffff;

type G = GraphicsContext;

// ---------- units ----------
const HORSE = new Set([3, 4]);
const naval = (kind: number) => kind >= 39 && kind <= 45;

function person(g: G, t: Team, kind: number, ox = 0, oy = 0, s = 1, legs = true) {
  const X = (x: number) => ox + x * s,
    Y = (y: number) => oy + y * s;
  if (legs) {
    g.rect(X(-5), Y(-5), 4 * s, 10 * s).fill(BOOT);
    g.rect(X(1), Y(-5), 4 * s, 10 * s).fill(BOOT);
  }
  const robe = kind === 10 || kind === 38;
  const hem = robe ? 4 : -4;
  g.poly([X(-7), Y(hem), X(0), Y(hem), X(0), Y(-20), X(-6), Y(-20)]).fill(t.main);
  g.poly([X(0), Y(hem), X(7), Y(hem), X(6), Y(-20), X(0), Y(-20)]).fill(t.dark);
  if (!robe) g.rect(X(-7), Y(-9), 14 * s, 2.2 * s).fill(WOOD_DARK);
  g.circle(X(0), Y(-25), 5.5 * s).fill(SKIN);
  if (kind === 37 || kind === 38) {
    // Hooded: cloak and dagger.
    g.poly([X(-7), Y(-20), X(7), Y(-20), X(6), Y(-28), X(0), Y(-33), X(-6), Y(-28)]).fill(0x352e40);
    g.circle(X(0), Y(-24), 3.6 * s).fill(0xd9ae80);
  } else if (kind === 10) {
    g.poly([X(-7), Y(-27), X(7), Y(-27), X(0), Y(-41)]).fill(t.light);
  } else if (kind === 9) {
    g.poly([X(-6), Y(-26), X(6), Y(-26), X(4), Y(-32), X(-4), Y(-32)]).fill(t.dark);
    g.poly([X(4), Y(-31), X(11), Y(-35), X(5), Y(-29)]).fill(WHITE);
  } else {
    // Bardur horned helmet.
    g.poly([X(-6), Y(-25), X(6), Y(-25), X(4.5), Y(-31.5), X(-4.5), Y(-31.5)]).fill(IRON);
    g.poly([X(-5.5), Y(-28), X(-11), Y(-35), X(-4.5), Y(-30.5)]).fill(0xf4efe2);
    g.poly([X(5.5), Y(-28), X(11), Y(-35), X(4.5), Y(-30.5)]).fill(0xf4efe2);
  }
  const line = (x1: number, y1: number, x2: number, y2: number, width: number, color: number) =>
    g.moveTo(X(x1), Y(y1)).lineTo(X(x2), Y(y2)).stroke({ width: width * s, color, cap: "round" });
  if (kind === 2 || kind === 12) {
    line(7, -10, 11, -26, 3, WOOD);
    g.circle(X(11.5), Y(-27), 3.2 * s).fill(WOOD_DARK);
  } else if (kind === 11) {
    line(8, -9, 13, -31, 2.6, STEEL);
    line(5, -13, 11, -14.5, 2, WOOD_DARK);
    g.roundRect(X(-12), Y(-19), 8 * s, 12 * s, 2.5 * s).fill(t.light).stroke({ width: 1.3 * s, color: WHITE });
  } else if (kind === 5) {
    g.roundRect(X(-13), Y(-22), 12 * s, 18 * s, 3.5 * s).fill(t.main).stroke({ width: 1.8 * s, color: WHITE });
    g.circle(X(-7), Y(-13), 2.6 * s).fill(WHITE);
  } else if (kind === 9) {
    g.moveTo(X(10), Y(-29)).quadraticCurveTo(X(19), Y(-16), X(10), Y(-3)).stroke({ width: 2.2 * s, color: WOOD });
    line(10, -29, 10, -3, 0.8, WHITE);
  } else if (kind === 37) {
    line(7, -10, 11, -18, 2.2, STEEL);
  } else if (kind === 10) {
    line(9, 2, 9, -30, 2, WOOD);
    g.circle(X(9), Y(-32), 3.4 * s).fill(0x9fe8ff);
  } else if (kind === 4) {
    line(5, -12, 17, -44, 2.2, STEEL);
    g.poly([X(15.5), Y(-40), X(15.5), Y(-34), X(23), Y(-38)]).fill(t.main);
  } else if (kind === 3) {
    line(6, -10, 12, -30, 2, WOOD);
    g.poly([X(11), Y(-31), X(14), Y(-37), X(14.5), Y(-30)]).fill(STEEL);
  }
}

function horse(g: G, t: Team) {
  for (const x of [-9, -4, 5, 10]) g.rect(x - 1.5, -4, 3, 10).fill(0x6e4526);
  g.ellipse(0, -7, 13, 6.5).fill(0x9c6b3e);
  g.poly([8, -11, 15, -21, 19, -17, 12, -6]).fill(0x8a5c33);
  g.poly([10, -14, 15, -22, 13, -23, 8, -13]).fill(0x3d2a1a);
  g.poly([-6, -12, 6, -12, 7, -3, -7, -3]).fill(t.main);
}

function catapult(g: G, t: Team) {
  g.poly([-15, 0, 13, -6, 15, -2, -13, 4]).fill(WOOD);
  g.circle(-9, 3, 4).fill(0x4a3524);
  g.circle(9, -1, 4).fill(0x4a3524);
  g.poly([-5, -2, -1, -18, 2, -18, 1, -3]).fill(WOOD_DARK);
  g.moveTo(-1, -16).lineTo(-15, -30).stroke({ width: 2.5, color: WOOD, cap: "round" });
  g.ellipse(-16, -31, 4, 2.4).fill(WOOD_DARK);
  g.circle(-16, -33, 2.2).fill(0x77808a);
  g.moveTo(8, -3).lineTo(8, -26).stroke({ width: 1.4, color: WOOD_DARK });
  g.poly([8.5, -26, 8.5, -19, 17, -22.5]).fill(t.main);
}

function ship(g: G, t: Team, kind: number) {
  const big = kind === 45,
    k = big ? 1.25 : kind === 42 ? 0.85 : 1;
  g.poly([-16 * k, -4, 0, -4, 0, 6, -11 * k, 6]).fill(WOOD);
  g.poly([0, -4, 16 * k, -4, 11 * k, 6, 0, 6]).fill(WOOD_DARK);
  if (kind === 44) g.poly([15 * k, -3, 23 * k, 0, 14 * k, 3]).fill(STEEL);
  if (kind === 41 || big) g.rect(-6, -9, 12, 5).fill(0x3a3f45);
  const masts = big ? [-6, 6] : [0];
  for (const m of masts) {
    g.moveTo(m, -4).lineTo(m, -36 * k).stroke({ width: 1.8, color: WOOD_DARK });
    g.poly([m + 1, -33 * k, m + 1, -9, m + 13 * k, -11]).fill(kind === 40 ? 0x222222 : t.main);
    g.poly([m - 1, -30 * k, m - 1, -11, m - 9 * k, -12]).fill(kind === 40 ? 0x3a3a3a : t.dark);
  }
  if (kind === 40) g.rect(1, -22, 12, 3).fill(t.main);
}

export function drawUnit(g: G, kind: number, t: Team) {
  if (naval(kind)) return ship(g, t, kind);
  if (kind === 8) return catapult(g, t);
  if (HORSE.has(kind)) {
    horse(g, t);
    return person(g, t, kind, -1, -8, 0.82, false);
  }
  person(g, t, kind, 0, 5, kind === 12 ? 1.35 : 1);
}

// ---------- cities ----------
function house(g: G, x: number, y: number, s: number, roof: Team) {
  const a = 9 * s,
    b = 5.4 * s,
    h = 8 * s,
    e = 1.4 * s;
  g.poly([x - a, y, x, y + b, x, y + b - h, x - a, y - h]).fill(WALL);
  g.poly([x, y + b, x + a, y, x + a, y - h, x, y + b - h]).fill(WALL_SHADE);
  g.rect(x + a * 0.35, y - h * 0.15, 2.4 * s, 4 * s).fill(0x6b4a2e);
  const L = [x - a - e, y - h],
    F = [x, y + b - h + e * 0.6],
    R = [x + a + e, y - h],
    B = [x, y - b - h - e * 0.6],
    A = [x, y - h - 8 * s];
  g.poly([...L, ...B, ...A]).fill(roof.light);
  g.poly([...B, ...R, ...A]).fill(roof.main);
  g.poly([...L, ...F, ...A]).fill(roof.main);
  g.poly([...F, ...R, ...A]).fill(roof.dark);
}

function tower(g: G, x: number, y: number, t: Team) {
  const a = 5,
    b = 3,
    h = 24;
  g.poly([x - a, y, x, y + b, x, y + b - h, x - a, y - h]).fill(0xe2d6bd);
  g.poly([x, y + b, x + a, y, x + a, y - h, x, y + b - h]).fill(0xb9a684);
  g.poly([x - a - 1, y - h, x, y + b - h + 1, x, y - h - 9]).fill(t.main);
  g.poly([x, y + b - h + 1, x + a + 1, y - h, x, y - h - 9]).fill(t.dark);
  g.moveTo(x, y - h - 9).lineTo(x, y - h - 21).stroke({ width: 1.3, color: WOOD_DARK });
  g.poly([x + 0.6, y - h - 21, x + 0.6, y - h - 14, x + 10, y - h - 17.5]).fill(t.main);
}

// Houses grow with the city level; the capital adds a flagged tower.
export function drawCity(g: G, level: number, t: Team, capital: boolean) {
  if (t.name === "neutral") {
    house(g, 9, 3, 0.62, t);
    house(g, -7, 7, 0.72, t);
    return;
  }
  const parts: [number, number, number][] = [[0, 5, level >= 4 ? 1.15 : 1]];
  if (level >= 2) parts.push([-14, 9, 0.72]);
  if (level >= 3) parts.push([14, 9, 0.72]);
  if (level >= 5) parts.push([0, 14, 0.78]);
  if (capital) tower(g, -11, -1, t);
  for (const [x, y, s] of parts.sort((p, q) => p[1] - q[1])) house(g, x, y, s, t);
}

// Small team flag beside publisher city sprites, which carry no team colour.
export function drawPennant(g: G, t: Team) {
  g.moveTo(15, 4).lineTo(15, -24).stroke({ width: 1.6, color: WOOD_DARK });
  g.poly([15.7, -24, 15.7, -15, 27, -19.5]).fill(t.main).stroke({ width: 1, color: WHITE });
}

// ---------- structures ----------
export function drawStructure(g: G, id: number) {
  const n = TEAMS.neutral;
  if (id === 2) {
    // Ruin: tumbled stones.
    for (const [x, y, r, c] of [
      [-6, 6, 5, 0x9aa3ab],
      [5, 4, 6, 0x7f8992],
      [0, 9, 4, 0xb4bcc3],
    ])
      g.poly([x - r, y, x - r * 0.4, y - r * 1.1, x + r * 0.6, y - r * 0.9, x + r, y, x, y + r * 0.5]).fill(c);
    return;
  }
  if (id === 5) {
    for (let k = -2; k <= 2; k++)
      g.moveTo(-14 + k * 5, 2 + k * 3)
        .lineTo(k * 5, 10 + k * 3)
        .stroke({ width: 2.4, color: k % 2 ? 0xd8b54a : 0xb8963a, cap: "round" });
    return;
  }
  if (id === 47) {
    // Lighthouse: a tapered tower in red and white bands with a lantern on top.
    const x = 0,
      y = 7,
      h = 30,
      base = 7,
      top = 4.2;
    const w = (k: number) => base + (top - base) * k; // half-width at height fraction k
    for (let k = 0; k < 4; k++) {
      const y0 = y - (h * k) / 4,
        y1 = y - (h * (k + 1)) / 4,
        w0 = w(k / 4),
        w1 = w((k + 1) / 4),
        band = k % 2 ? 0xffffff : 0xd9453b;
      g.poly([x - w0, y0, x, y0 + w0 * 0.6, x, y1 + w1 * 0.6, x - w1, y1]).fill(band);
      g.poly([x, y0 + w0 * 0.6, x + w0, y0, x + w1, y1, x, y1 + w1 * 0.6]).fill(band === 0xffffff ? 0xd8dde3 : 0xa8302a);
    }
    // Gallery, lantern room and the cap.
    g.poly([x - top - 2, y - h, x, y - h + 3.5, x + top + 2, y - h, x, y - h - 3.5]).fill(0x4a4f57);
    g.poly([x - 3.4, y - h - 1, x, y - h + 1, x, y - h - 8, x - 3.4, y - h - 10]).fill(0xfff2a8);
    g.poly([x, y - h + 1, x + 3.4, y - h - 1, x + 3.4, y - h - 10, x, y - h - 8]).fill(0xf5c846);
    g.poly([x - 4.4, y - h - 10, x, y - h - 7.5, x + 4.4, y - h - 10, x, y - h - 16]).fill(0x5a6069);
    return;
  }
  if (id === 48) {
    // Bridge: a plank deck on posts, laid along the NW–SE axis of the diamond.
    const L = 24,
      w = 5,
      ux = 0.86,
      uy = 0.5, // along the deck
      px = 0.86,
      py = -0.5; // across the deck
    const corner = (a: number, b: number) => [a * ux + b * px, 6 + a * uy + b * py];
    for (const a of [-L, -L / 2, 0, L / 2, L])
      for (const b of [-w, w]) {
        const [x, y] = corner(a, b);
        g.rect(x - 1.2, y - 7, 2.4, 9).fill(WOOD_DARK);
      }
    g.poly([...corner(-L, -w), ...corner(L, -w), ...corner(L, w), ...corner(-L, w)]).fill(WOOD);
    g.poly([...corner(-L, w), ...corner(L, w), ...corner(L, w + 1.6), ...corner(-L, w + 1.6)]).fill(WOOD_DARK);
    for (let k = -4; k <= 4; k++) {
      const [x0, y0] = corner(k * 5.2, -w),
        [x1, y1] = corner(k * 5.2, w);
      g.moveTo(x0, y0).lineTo(x1, y1).stroke({ width: 0.8, color: WOOD_DARK, alpha: 0.6 });
    }
    for (const b of [-w, w]) {
      const [x0, y0] = corner(-L, b),
        [x1, y1] = corner(L, b);
      g.moveTo(x0, y0 - 5).lineTo(x1, y1 - 5).stroke({ width: 1.2, color: WOOD_DARK });
    }
    return;
  }
  if (id >= 23 && id <= 29) {
    // Monument: a stone plinth under a tall column with a gilded top. Each one
    // gets its own accent so they stay distinguishable at a glance.
    const accent = { 23: 0xf6f0e0, 24: 0x7ac6ff, 25: 0xf0a43a, 26: 0xd9b24a, 27: 0xd9453b, 28: 0x4fb55a, 29: 0xffd34d }[id] || 0xf2c94c;
    const y = 8;
    g.poly([-16, y - 2, 0, y + 7, 16, y - 2, 0, y - 11]).fill(0xd9dde2);
    g.poly([-16, y - 2, 0, y + 7, 0, y + 11, -16, y + 2]).fill(0xaab2bb);
    g.poly([0, y + 7, 16, y - 2, 16, y + 2, 0, y + 11]).fill(0x8f98a2);
    g.poly([-6, y - 4, 0, y - 1, 0, y - 32, -6, y - 35]).fill(0xe9ecf0);
    g.poly([0, y - 1, 6, y - 4, 6, y - 35, 0, y - 32]).fill(0xb9c1c9);
    g.poly([-9, y - 35, 0, y - 30, 9, y - 35, 0, y - 40]).fill(accent);
    g.poly([-9, y - 35, 0, y - 30, 0, y - 27, -9, y - 32]).fill(accent - 0x303030 > 0 ? accent - 0x303030 : accent);
    g.poly([0, y - 30, 9, y - 35, 9, y - 32, 0, y - 27]).fill(accent - 0x505050 > 0 ? accent - 0x505050 : accent);
    g.circle(0, y - 44, 3.2).fill(0xffe36b);
    return;
  }
  const roof = { 6: 0xe9d9a6, 17: 0xf2c94c, 18: 0xf2c94c, 19: 0xf2c94c, 20: 0xf2c94c, 21: 0x6f7880, 22: 0x5a5f66, 50: 0xf0a43a }[
    id
  ];
  house(g, 0, 6, 0.85, roof ? { name: "", main: roof, dark: roof - 0x202020, light: roof } : n);
  if (id === 6)
    for (const [dx, dy] of [
      [12, -8],
      [-12, 8],
      [8, 12],
      [-8, -12],
    ])
      g.moveTo(0, -8).lineTo(dx, -8 + dy).stroke({ width: 2, color: 0xf6f0e0 });
  if (id === 12 || id === 13) for (let k = 0; k < 3; k++) g.roundRect(-16 + k * 3, 8 - k * 3, 10, 3.4, 1.7).fill(WOOD);
}

// ---------- terrain ----------
export function drawMountain(g: G) {
  g.poly([-22, 6, -11, -14, -10, 6]).fill(0xa7b0ba);
  g.poly([-10, 6, -11, -14, 0, 6]).fill(0x7d8791);
  g.poly([-16, 9, 2, -26, 4, 9]).fill(0xb3bcc5);
  g.poly([4, 9, 2, -26, 19, 9]).fill(0x7f8a95);
  g.poly([2, -26, -4.3, -13.8, -0.5, -11.5, 2.5, -15]).fill(0xffffff);
  g.poly([2, -26, 2.5, -15, 5, -12, 7.9, -13.8]).fill(0xdfe8ef);
}

function pine(g: G, x: number, y: number, s: number) {
  g.rect(x - 1.5 * s, y - 4 * s, 3 * s, 5 * s).fill(WOOD_DARK);
  for (let k = 0; k < 3; k++) {
    const base = y - 3 * s - k * 6 * s,
      apex = base - 11 * s,
      w = (9 - 2.2 * k) * s;
    g.poly([x - w, base, x, base + 1.5 * s, x, apex]).fill(0x2e8a4f);
    g.poly([x, base + 1.5 * s, x + w, base, x, apex]).fill(0x1e6538);
  }
}
export function drawForest(g: G) {
  pine(g, 10, 1, 0.85);
  pine(g, -12, 3, 0.8);
  pine(g, -1, 10, 1);
}

export function drawResource(g: G, id: number) {
  if (id === 1) {
    // Animal: a deer.
    for (const x of [-10, -7, -3, 0]) g.moveTo(x, 6).lineTo(x, 12).stroke({ width: 1.4, color: 0x6e4526 });
    g.ellipse(-5, 5, 7, 3.6).fill(0xa8713f);
    g.poly([0, 4, 4, -4, 7, -3, 3, 6]).fill(0x96633a);
    g.ellipse(6, -4, 3.2, 2).fill(0xa8713f);
    g.moveTo(5, -6).lineTo(3, -11).lineTo(1, -12).stroke({ width: 1.1, color: 0xe9dcc3 });
    g.moveTo(6, -6).lineTo(8, -11).lineTo(10, -12).stroke({ width: 1.1, color: 0xe9dcc3 });
  } else if (id === 6) {
    g.circle(-4, 2, 6).fill(0x3f9a4a);
    g.circle(4, 3, 6.5).fill(0x34843f);
    g.circle(0, -2, 6).fill(0x4fb55a);
    for (const [x, y] of [
      [-5, 0],
      [2, -3],
      [5, 4],
      [-1, 5],
      [-6, 5],
    ])
      g.circle(x, y, 1.8).fill(x > 0 ? 0xe0405a : 0x9d3fd0);
  } else if (id === 2) {
    for (let k = 0; k < 6; k++) {
      const x = -10 + k * 4,
        y = 8 - (k % 2) * 2;
      g.moveTo(x, y).lineTo(x + 1, y - 12).stroke({ width: 1.3, color: 0xb68d2e });
      g.ellipse(x + 1, y - 13, 1.6, 3).fill(0xe8bf45);
    }
  } else if (id === 3) {
    // Fish: a small shoal just under the surface.
    for (let j = 0; j < 3; j++) {
      const xx = (j - 1) * 10,
        yy = j * 4;
      g.poly([xx - 5, yy - 3, xx + 5, yy, xx - 5, yy + 3, xx - 1, yy]).fill(0x249bd6);
    }
  } else if (id === 8) {
    // Starfish: five arms on a pale shoal.
    g.ellipse(0, 4, 12, 6).fill({ color: 0xbfe9ff, alpha: 0.8 });
    const arm: number[] = [];
    for (let k = 0; k < 10; k++) {
      const a = -Math.PI / 2 + (k * Math.PI) / 5,
        r = k % 2 ? 3.2 : 9;
      arm.push(Math.cos(a) * r, 2 + Math.sin(a) * r * 0.75);
    }
    g.poly(arm).fill(0xf07a3c);
    g.circle(0, 2, 2).fill(0xffb27a);
  } else if (id === 5) {
    g.poly([-8, 8, -6, -2, 3, -5, 8, 3, 4, 9]).fill(0x7d8791);
    g.poly([-6, -2, 3, -5, 1, 1]).fill(0xb7c0c8);
    for (const [x, y] of [
      [-3, 2],
      [3, 0],
      [1, 5],
    ])
      g.circle(x, y, 1.3).fill(0xd8e2ea);
  }
}

// Five-point star (veteran badge, city level).
export function drawStar(g: G, x: number, y: number, r: number) {
  const points: number[] = [];
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5,
      k = i % 2 ? r * 0.45 : r;
    points.push(x + Math.cos(a) * k, y + Math.sin(a) * k);
  }
  g.poly(points).fill(0xffd23f).stroke({ width: r * 0.18, color: 0xb8860b });
}
