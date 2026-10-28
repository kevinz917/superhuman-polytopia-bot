// Names, costs and icons shared by the board and the interface.
import type { LegalAction, Meta, Observation, VisibleState } from "../api/types.ts";

// Bardur-style names from two syllables. 73 is coprime with 196, so every tile
// of the 14 × 14 board gets a different name.
const HEADS = ["Gu", "Ba", "U", "Li", "Ve", "No", "Ska", "Ho", "Tro", "Bjo", "Kra", "Ru", "Sva", "Fe"];
const TAILS = ["ur", "rdur", "ld", "nlak", "rn", "rg", "l", "lm", "nd", "rk", "gg", "st", "ga", "nheim"];
export const cityName = (tile: number) => {
  const i = (tile * 73) % 196;
  return HEADS[i % 14] + TAILS[Math.floor(i / 14)];
};

export const UNIT_SPRITES: Record<number, string> = {
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
export const unitSprite = (kind: number) => "unit-" + (UNIT_SPRITES[kind] || "warrior");
export const RESOURCE_SPRITES: Record<number, string> = { 1: "animal", 2: "crop", 5: "metal", 6: "fruit" };
export const STRUCTURE_SPRITES: Record<number, string> = {
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
};
export const HARVEST: Record<number, string> = {
  1: "Hunting",
  3: "Fishing",
  4: "Whaling",
  6: "Harvest fruit",
  7: "Harvest fungi",
};
export const TERRAIN_TEXT: Record<number, string> = {
  1: "Shallow water. Boats can sail here.",
  2: "Deep ocean, only for ships.",
  3: "Open field.",
  4: "Mountains give a defence bonus and hold metal.",
  5: "Forests slow movement and give a defence bonus.",
};

const svg = (body: string) => `<svg viewBox="0 0 40 40" aria-hidden="true">${body}</svg>`;
export const ICONS = {
  heart: svg('<path fill="#ff6f8e" d="M20 33 6 19a8 8 0 0 1 14-9 8 8 0 0 1 14 9z"/>'),
  cross: svg('<path fill="#fff" d="M16 6h8v10h10v8H24v10h-8V24H6v-8h10z"/>'),
  disband: svg('<path stroke="#fff" stroke-width="6" stroke-linecap="round" d="m10 10 20 20m0-20L10 30"/>'),
  flame: svg(
    '<path fill="#ff9b2f" d="M20 35c-7 0-11-5-11-11 0-7 7-10 6-18 6 3 9 8 9 13 2-1 3-3 3-6 3 3 5 7 5 11 0 6-5 11-12 11z"/><path fill="#ffe36b" d="M20 35c-3 0-5-2-5-5 0-4 4-5 4-9 3 2 6 5 6 9 0 3-2 5-5 5z"/>',
  ),
  axe: svg('<path stroke="#a57a4b" stroke-width="4" stroke-linecap="round" d="M14 34 27 9"/><path fill="#dfe8ee" d="M22 6c6-2 11 1 12 6l-9 5-6-4z"/>'),
  hammer: svg('<path stroke="#a57a4b" stroke-width="4" stroke-linecap="round" d="M13 34 24 15"/><path fill="#c5cdd3" d="m17 8 10-4 8 14-10 4z"/>'),
  wall: svg(
    '<path fill="#c9d3d7" d="M5 16h30v18H5z"/><path fill="#9aa9b0" d="M5 10h6v6H5zm9 0h6v6h-6zm9 0h6v6h-6zm9 0h3v6h-3zM5 22h12v1H5zm18 0h12v1H23zM11 28h18v1H11z"/>',
  ),
  border: svg(
    '<path fill="none" stroke="#fff" stroke-width="3" stroke-dasharray="4 3" d="M20 3 37 20 20 37 3 20z"/><path fill="#79d0ff" d="m20 12 8 8-8 8-8-8z"/>',
  ),
  explorer: svg(
    '<path fill="#8a7a63" d="M13 36 15 18h10l2 18z"/><path fill="#287ba7" d="M14 22h12v4H14z"/><circle cx="20" cy="13" r="5" fill="#e7d4ac"/><path fill="#6c5e4a" d="M13 14 20 3l7 11-3-3h-8z"/><path stroke="#6b5236" stroke-width="2" d="M30 37 32 10"/>',
  ),
};

// An interface icon: `icon` (drawn SVG) or `sprite` (a board piece's placeholder shape).
export interface Art {
  sprite?: string;
  icon?: string;
}
export const REWARDS: Record<number, Art & { text: string }> = {
  1: { text: "Units in this city get a big defence bonus.", icon: ICONS.wall },
  2: { text: "+1 star per turn and +250 score.", sprite: "forest" },
  3: { text: "+1 star per turn.", sprite: "sawmill" },
  4: { text: "An explorer reveals the land around the city.", icon: ICONS.explorer },
  5: { text: "Expand the city's borders.", icon: ICONS.border },
  6: { text: "A Giant appears in the city.", sprite: "unit-giant" },
  7: { text: "+5 stars right now.", sprite: "star" },
  8: { text: "+3 population.", sprite: "house-3" },
};
const ABILITY_ART: Record<number, Art> = {
  1: { icon: ICONS.flame },
  2: { icon: ICONS.axe },
  3: { sprite: "forest" },
  4: { icon: ICONS.hammer },
  7: { icon: ICONS.heart },
  8: { icon: ICONS.disband },
  9: { icon: ICONS.cross },
  16: { sprite: "star" },
  18: { sprite: "star" },
  19: { sprite: "star" },
  20: { sprite: "star" },
  21: { sprite: "star" },
  22: { sprite: "star" },
};

export const titleCase = (s: string) => s.replace(/([a-z])([A-Z])/g, "$1 $2");
export const unitName = (meta: Meta, kind: number) =>
  meta.rules.units.find((u) => u.id === kind)?.name || "Unit";

export function actLabel(meta: Meta, a: LegalAction) {
  const c = a.command,
    p = c.parameter,
    k = c.kind;
  return titleCase(
    k === 3
      ? p === 19
        ? c.target === 1
          ? "Accept peace"
          : "Reject peace"
        : meta.abilities[p] || "Ability"
      : k === 4
        ? "Train " + unitName(meta, p)
        : k === 6
          ? "Build " + (meta.rules.structures.find((s) => s.id === p)?.name || p)
          : k === 7
            ? "Research " + (meta.rules.technologies.find((t) => t.id === p)?.name || p)
            : k === 9
              ? meta.rewards[p] || "Reward"
              : ({ 1: "Move", 2: "Attack", 5: "Harvest", 8: "Capture", 10: "End turn", 11: "Resign" } as Record<number, string>)[k] ||
                "Action",
  );
}

export function cost(meta: Meta, obs: Observation, a: LegalAction): number | null {
  const c = a.command;
  let n: number | null | undefined;
  if (c.kind === 4) n = meta.rules.units.find((u) => u.id === c.parameter)?.cost;
  if (c.kind === 6) n = meta.rules.structures.find((s) => s.id === c.parameter)?.cost;
  if (c.kind === 5) n = 2;
  if (c.kind === 7) n = techCost(meta, obs, c.parameter);
  return n !== undefined && n !== null && n >= 0 ? n : null;
}

// Tier × cities + 4, a third off with Philosophy. Shown for locked techs too.
export function techCost(meta: Meta, obs: Observation, id: number) {
  const v = obs.visible_state,
    t = meta.rules.technologies.find((t) => t.id === id)!;
  const n = 4 + v.cities.filter((x) => x.owner === obs.player).length * t.tier;
  return v.technology.includes(22) ? Math.ceil(n * 0.66) : n;
}

export function structureSprite(meta: Meta, id: number) {
  const name = (meta.rules.structures.find((s) => s.id === id)?.name || "").toLowerCase();
  for (const key of ["farm", "windmill", "port", "sawmill", "mine", "forge", "market", "temple"])
    if (name.includes(key)) return key;
  if (name.includes("lumber")) return "lumber";
  if (name.includes("road") || name.includes("bridge")) return "ground";
  return "star";
}

export function actionArt(meta: Meta, obs: Observation, a: LegalAction): Art {
  const c = a.command;
  if (c.kind === 4) return { sprite: unitSprite(c.parameter) };
  if (c.kind === 6) return { sprite: structureSprite(meta, c.parameter) };
  if (c.kind === 5) return { sprite: RESOURCE_SPRITES[obs.visible_state.tiles[c.target]?.resource ?? 0] || "water" };
  if (c.kind === 8) return { sprite: "capital" };
  if (c.kind === 9) return REWARDS[c.parameter] || { sprite: "star" };
  if (c.kind === 3) return ABILITY_ART[c.parameter] || { sprite: "star" };
  return { sprite: "star" };
}

// Fallback for recordings without render territory: nearest same-owner city.
export function cityTerritory(v: VisibleState, cityTile: number) {
  const n = v.board_size,
    city = v.cities.find((c) => c.tile === cityTile);
  if (!city?.owner) return [];
  const d = (a: number, b: number) =>
    Math.max(Math.abs((a % n) - (b % n)), Math.abs(Math.floor(a / n) - Math.floor(b / n)));
  const rivals = v.cities.filter((c) => c.owner === city.owner);
  return v.tiles
    .filter((t) => t.known && t.owner === city.owner && rivals.every((c) => d(t.tile, cityTile) <= d(t.tile, c.tile)))
    .map((t) => t.tile);
}
