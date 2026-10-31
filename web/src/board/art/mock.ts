// Mock mode: every piece is a low-poly placeholder shape from procedural.ts; no game or publisher graphics.
import { Container, type Graphics } from "pixi.js";
import type { Tile, Unit } from "../../api/types.ts";
import { RESOURCE_SPRITES, STRUCTURE_SPRITES, UNIT_SPRITES } from "../../game/vocab.ts";
import { DEPTH, DIAMOND, flat, HH, HW, pivoted, shape, STEPS, TEAMS, type Point, type Team } from "../geometry.ts";
import { drawCity, drawForest, drawMountain, drawResource, drawStar, drawStructure, drawUnit } from "../procedural.ts";
import type { BoardArt, GroundInfo, UnitArt } from "./types.ts";

const ROAD = { width: 2.2, color: 0x9b7e59, cap: "round" as const, join: "round" as const };

export class MockArt implements BoardArt {
  readonly mode = "mock" as const;

  ground(t: Tile, info: GroundInfo, base: Container, surface: Container) {
    const water = t.terrain === 1 || t.terrain === 2;
    base.addChild(
      shape("ground:" + (water ? "water" : "land"), (g) => {
        g.poly(flat([[-HW, 0], [0, HH], [0, HH + DEPTH], [-HW, DEPTH]])).fill(water ? 0x208dd1 : 0x58221f);
        g.poly(flat([[0, HH], [HW, 0], [HW, DEPTH], [0, HH + DEPTH]])).fill(water ? 0x25b5ed : 0x824b16);
        const a: Point = [0, -HH],
          b: Point = [HW, 0],
          d: Point = [0, HH],
          e: Point = [-HW, 0],
          mid: Point = [9, -2];
        g.poly(flat([a, b, d, e])).fill(water ? 0x4abcf0 : 0xddffff);
        g.poly(flat([a, b, mid])).fill(water ? 0x64c8f1 : 0xd2fcff);
        g.poly(flat([b, d, mid])).fill(water ? 0x4db5e9 : 0xf3ffff);
        g.poly(flat([d, e, mid])).fill(water ? 0x6bd2f4 : 0xc8f8fc);
      }),
    );
    if (t.road)
      // One leg from the centre toward each connected neighbour; a lone road is a short stub.
      surface.addChild(
        shape("road:" + info.roads, (o) => {
          if (!info.roads) o.moveTo(-HW * 0.3, HH * 0.12).lineTo(HW * 0.3, HH * 0.12).stroke(ROAD);
          STEPS.forEach(([dc, dr], k) => {
            if (info.roads & (1 << k)) o.moveTo(0, 0).lineTo(((dc - dr) * HW) / 2, ((dc + dr) * HH) / 2).stroke(ROAD);
          });
          o.circle(0, 0, 1.3).fill(ROAD.color);
        }),
      );
    if (!info.team || !info.border) return;
    const color = info.team.main;
    surface.addChild(
      shape(`border:${info.team.name}:${info.border}`, (o) => {
        // Edge j (NE, SE, SW, NW) runs between corners j and j + 1, clockwise from the top.
        const corners: Point[] = [[0, -HH], [HW, 0], [0, HH], [-HW, 0]];
        for (let j = 0; j < 4; j++) {
          if (!(info.border & (1 << j))) continue;
          const p = corners[j],
            q = corners[(j + 1) % 4];
          for (let s = 0; s < 5; s++) {
            const lo = s / 5 + 0.03,
              hi = (s + 1) / 5 - 0.03;
            const x1 = p[0] + (q[0] - p[0]) * lo,
              y1 = p[1] + (q[1] - p[1]) * lo,
              x2 = p[0] + (q[0] - p[0]) * hi,
              y2 = p[1] + (q[1] - p[1]) * hi;
            o.poly([x1, y1, x2, y2, x2, y2 - 7, x1, y1 - 7]).fill(color);
          }
        }
      }),
    );
  }

  // The fog pattern repeats every 78 tiles (17 mod 13, odd/even, mod 3).
  fog(i: number) {
    return shape("fog:" + (i % 78), (g) => {
      const top: Point = [0, -HH],
        right: Point = [HW, 0],
        bottom: Point = [0, HH],
        left: Point = [-HW, 0];
      const peak: Point = [((i * 17) % 13) - 6, -10];
      // Side faces show where the board ends.
      g.poly(flat([[-HW, 0], [0, HH], [0, HH + DEPTH], [-HW, DEPTH]])).fill(0x7787ba);
      g.poly(flat([[0, HH], [HW, 0], [HW, DEPTH], [0, HH + DEPTH]])).fill(0xafbae1);
      g.poly(flat([top, right, peak])).fill(0xffffff);
      g.poly(flat([right, bottom, peak])).fill(i % 2 ? 0xb1c2f5 : 0xc2d2f9);
      g.poly(flat([bottom, left, peak])).fill(0xfaffff);
      g.poly(flat([left, top, peak])).fill(i % 3 ? 0x8c9dd5 : 0xc4d0f1);
      g.poly(flat([peak, [HW * 0.35, HH * 0.3], bottom])).fill(0xe8efff);
    });
  }

  terrain(t: Tile, dim: Container[]) {
    const c = new Container(),
      add = (g: Graphics) => dim.push(c.addChild(g));
    if (t.terrain === 4) add(shape("mountain", drawMountain));
    if (t.terrain === 5 && t.structure !== 1) add(shape("forest", drawForest));
    if (t.resource) add(shape("resource:" + t.resource, (g) => drawResource(g, t.resource!)));
    return c;
  }

  city(team: Team, owner: number, level: number, capital: boolean, dim: Container[]) {
    const c = pivoted(),
      shown = owner ? Math.min(level, 5) : 0;
    dim.push(c.addChild(shape(`city:${shown}:${team.name}:${capital}`, (g) => drawCity(g, shown, team, capital))));
    return c;
  }

  structure(t: Tile, dim: Container[]) {
    const id = t.structure ?? 0;
    if (id <= 1 || id === 3) return null;
    const c = pivoted();
    // Temples grow a little with their level.
    const grow = id >= 17 && id <= 20 ? 1 + 0.08 * (Math.min(5, Math.max(1, t.structure_level ?? 1)) - 1) : 1;
    const g = shape("structure:" + id, (ctx) => drawStructure(ctx, id));
    g.scale.set(grow);
    dim.push(c.addChild(g));
    return c;
  }

  unit(u: Unit, team: Team): UnitArt {
    const body = new Container();
    // A team ring under every unit and a health badge above it.
    body.addChild(
      shape("ring:" + team.name, (g) =>
        g.ellipse(0, 5, 13, 7).fill({ color: team.main, alpha: 0.35 }).stroke({ width: 2, color: team.main }),
      ),
    );
    body.addChild(shape(`unit:${u.kind}:${team.name}`, (g) => drawUnit(g, u.kind, team)));
    body.addChild(shape("hp:" + team.name, (g) => g.circle(-15, -34, 8).fill(team.main).stroke({ width: 1.5, color: 0xffffff })));
    const flash = shape(`unit:${u.kind}:${team.name}`, (g) => drawUnit(g, u.kind, team));
    const glow = shape(`unit:${u.kind}:${team.name}`, (g) => drawUnit(g, u.kind, team));
    return { body, flash, hp: { x: -15, y: -33.5, size: 11, shadow: false }, veteran: this.star(14, -34, 10), glow };
  }

  explorer(team: Team) {
    return shape("explorer:" + team.name, (g) =>
      g
        .ellipse(0, 4, 10, 4)
        .fill({ color: 0x294956, alpha: 0.25 })
        .poly([-8, 0, -5, -20, 5, -20, 8, 0])
        .fill(0x8a7a63)
        .poly([-6, -15, 6, -15, 5, -11, -5, -11])
        .fill(team.main)
        .circle(0, -25, 6)
        .fill(0xe7d4ac)
        .poly([-7, -24, 0, -36, 7, -24, 3, -28, -3, -28])
        .fill(0x6c5e4a)
        .moveTo(10, 2)
        .lineTo(12, -30)
        .stroke({ width: 2, color: 0x6b5236 }),
    );
  }

  selector() {
    return shape("sel", (g) => g.poly(DIAMOND).fill({ color: 0xffffff, alpha: 0.22 }).stroke({ width: 2.5, color: 0xffffff }));
  }

  marker(attack: boolean) {
    return attack
      ? shape("marker:attack", (g) =>
          g
            .ellipse(0, 0, 17, 10)
            .fill({ color: 0xe8402f, alpha: 0.82 })
            .ellipse(0, 0, 11, 6.5)
            .fill(0xffffff)
            .ellipse(0, 0, 6, 3.5)
            .fill(0xe8402f)
            .moveTo(-21, 0)
            .lineTo(-13, 0)
            .moveTo(13, 0)
            .lineTo(21, 0)
            .stroke({ width: 2, color: 0xffffff }),
        )
      : shape("marker:move", (g) =>
          g
            .ellipse(0, 0, 14, 8.5)
            .fill({ color: 0x9ccbff, alpha: 0.88 })
            .ellipse(0, 0, 9, 5.5)
            .fill(0xffffff)
            .ellipse(0, 0, 4.5, 2.8)
            .fill(0x7fcaff),
        );
  }

  star(x: number, y: number, size: number) {
    return shape(`star:${x}:${y}:${size}`, (g) => drawStar(g, x, y, size / 2));
  }

  // Three pips under the health badge fill with kills; all three turn gold once the unit can be promoted.
  xp(kills: number) {
    return shape(`xp:${kills}`, (g) => {
      g.roundRect(-23, -26.5, 16, 6, 3).fill({ color: 0x10202c, alpha: 0.8 });
      for (let i = 0; i < 3; i++)
        g.circle(-20 + i * 5, -23.5, 1.8).fill(i < kills ? (kills >= 3 ? 0xffd34d : 0x8ee6ff) : 0x51616c);
    });
  }

  capitalIcon() {
    return null;
  }

  skull(x: number, y: number, size: number) {
    const k = size / 12;
    return shape(`skull:${x}:${y}:${size}`, (g) =>
      g
        .circle(x, y - 1 * k, 5.2 * k)
        .fill(0xf4f1e8)
        .roundRect(x - 3 * k, y + 2 * k, 6 * k, 3.6 * k, 1 * k)
        .fill(0xf4f1e8)
        .circle(x - 2 * k, y - 1 * k, 1.5 * k)
        .fill(0x1a1a1a)
        .circle(x + 2 * k, y - 1 * k, 1.5 * k)
        .fill(0x1a1a1a)
        .poly([x, y + 0.6 * k, x - 0.8 * k, y + 1.9 * k, x + 0.8 * k, y + 1.9 * k])
        .fill(0x1a1a1a)
        .rect(x - 1.5 * k, y + 3.6 * k, 0.6 * k, 2 * k)
        .rect(x + 0.9 * k, y + 3.6 * k, 0.6 * k, 2 * k)
        .fill(0x1a1a1a),
    );
  }

  labelBand(team: Team) {
    return { color: team.dark, alpha: 0.88 };
  }

  paintBar(g: Graphics, width: number, segments: number, amount: number) {
    g.roundRect(-width / 2, 33, width, 7, 3).fill(0xc6d4dc);
    if (amount > 0) g.roundRect(-width / 2, 33, Math.max(6, width * amount), 7, 3).fill(0x00a7f5);
    for (let j = 1; j < Math.min(segments, 9); j++) {
      const x = -width / 2 + (width * j) / segments;
      g.moveTo(x, 33).lineTo(x, 40);
    }
    g.stroke({ width: 0.5, color: 0x84949d });
  }

  resourceIcon(resource: number) {
    const g = shape("harvest:" + resource, (ctx) => drawResource(ctx, resource));
    g.position.set(0, 8);
    const c = new Container();
    c.addChild(g);
    c.scale.set(1.3);
    return c;
  }

  icon(name: string) {
    const unit = Object.entries(UNIT_SPRITES).find(([, n]) => "unit-" + n === name);
    if (unit) return shape("icon:" + name, (g) => drawUnit(g, +unit[0], TEAMS.blue));
    const structure = Object.entries(STRUCTURE_SPRITES).find(([, n]) => n === name);
    if (structure) return shape("icon:" + name, (g) => drawStructure(g, +structure[0]));
    const resource = Object.entries({ ...RESOURCE_SPRITES, 3: "fish" }).find(([, n]) => n === name);
    if (resource) return shape("icon:" + name, (g) => drawResource(g, +resource[0]));
    const house = /^house-(\d)$/.exec(name);
    if (house) return shape("icon:" + name, (g) => drawCity(g, Math.min(5, +house[1]), TEAMS.blue, false));
    if (name === "capital") return shape("icon:capital", (g) => drawCity(g, 2, TEAMS.blue, true));
    if (name === "forest") return shape("icon:forest", drawForest);
    if (name === "mountain") return shape("icon:mountain", drawMountain);
    if (name === "water" || name === "ocean")
      return shape("icon:" + name, (g) => g.poly(DIAMOND).fill(name === "water" ? 0x4abcf0 : 0x2a7fc4));
    return shape("icon:star", (g) => drawStar(g, 0, 0, 14));
  }
}
