// Board geometry, team colours and shape caching shared by the board and both art modes.
import { Container, Graphics, GraphicsContext } from "pixi.js";

// World units: one tile is 64 x 38.4, origin at the centre of its top face.
export const TW = 64,
  TH = 38.4,
  HW = TW / 2,
  HH = TH / 2,
  DEPTH = 7;

export type Point = [number, number];
export const flat = (points: Point[]) => points.flat();
export const DIAMOND = flat([[0, -HH], [HW, 0], [0, HH], [-HW, 0]]);

// Neighbours as [column step, row step], clockwise from straight down the screen: S, SW, W, NW, N, NE, E, SE.
export const STEPS: Point[] = [[1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]];
// The four tile edges NE, SE, SW, NW as indices into STEPS (edge masks use this bit order).
export const EDGES = [5, 7, 1, 3];

export interface Team {
  name: string;
  main: number;
  dark: number;
  light: number;
}
export const TEAMS = {
  blue: { name: "blue", main: 0x2f95e0, dark: 0x1b5f9e, light: 0x8ccfff },
  red: { name: "red", main: 0xe2483d, dark: 0x9c2a23, light: 0xff9f92 },
  green: { name: "green", main: 0x3fae5a, dark: 0x25703a, light: 0x9fe3ae },
  gold: { name: "gold", main: 0xe0a92f, dark: 0x9a7019, light: 0xffdc8c },
  neutral: { name: "neutral", main: 0xb3a58c, dark: 0x857861, light: 0xdcd2bf },
} satisfies Record<string, Team>;

// Identical shapes share one GraphicsContext, so its tessellation and GPU
// buffers are built once instead of once per tile per frame.
const shapes = new Map<string, GraphicsContext>();
export const shape = (key: string, draw: (c: GraphicsContext) => void) => {
  let context = shapes.get(key);
  if (!context) {
    context = new GraphicsContext();
    draw(context);
    shapes.set(key, context);
  }
  return new Graphics(context);
};

// Scales around the base of a building for the mobile "pop".
export function pivoted() {
  const c = new Container();
  c.pivot.set(0, 9);
  c.position.set(0, 9);
  return c;
}
