// The board's art behind one interface: "mock" draws placeholder shapes. The board asks the art for every
// piece and handles layout, selection and animation.
import type { Container, Graphics } from "pixi.js";
import type { Tile, Unit } from "../../api/types.ts";
import type { Team } from "../geometry.ts";

export type ArtMode = "mock";

// What a tile's ground depends on besides the tile itself, worked out by the board from its neighbours.
export interface GroundInfo {
  border: number; // edge bit j (NE, SE, SW, NW): the neighbour there has another owner
  roads: number; // bit k: a road runs toward neighbour STEPS[k]
  shores: number; // edge bit j: a water tile's neighbour there is land
  walls: number; // bit 0 / 1: the front-left / front-right neighbour is lower (deep ocean, or off the board)
  team: Team | null; // owner of the tile
}

export interface UnitArt {
  body: Container;
  flash: Container | null; // a copy blended additively for hit flashes
  hp: { x: number; y: number; size: number; shadow: boolean };
  veteran: Container; // badge for veteran units
  glow: Container | null; // the figure alone, which the board fills and blurs into the halo of a unit that can still act
}

export interface BoardArt {
  readonly mode: ArtMode;
  // Terrain cube into `base`; roads, shorelines and borders into `surface` (drawn above every cube).
  ground(t: Tile, info: GroundInfo, base: Container, surface: Container): void;
  fog(tile: number): Container;
  // Mountain, forest and resource of a known tile; pieces that dim with the territory go into `dim`.
  terrain(t: Tile, dim: Container[]): Container;
  city(team: Team, owner: number, level: number, capital: boolean, dim: Container[]): Container;
  structure(t: Tile, dim: Container[]): Container | null;
  unit(u: Unit, team: Team): UnitArt;
  explorer(team: Team): Container;
  selector(): Container;
  marker(attack: boolean): Container;
  // A star centred at (x, y), `size` across: city level, veteran badge, effects.
  star(x: number, y: number, size: number): Container;
  // Experience toward promotion (1-3 kills) for a unit that is not a veteran yet.
  xp(kills: number): Container;
  capitalIcon(x: number, y: number, size: number): Container | null;
  // Attack preview mark for a unit that would die, centred at (x, y), `size` across.
  skull(x: number, y: number, size: number): Container;
  labelBand(team: Team): { color: number; alpha: number };
  // City growth bar: `segments` cells, `amount` of it filled, centred, `width` wide, top at y = 33.
  paintBar(g: Graphics, width: number, segments: number, amount: number): void;
  // Harvest effect: the resource rising off the tile, base at y = 20.
  resourceIcon(resource: number): Container | null;
  // Interface icon (selection bar, tech tree) by sprite name; null leaves it to the sprite file.
  icon(name: string): Container | null;
}
