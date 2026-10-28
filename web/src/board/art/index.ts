import { MockArt } from "./mock.ts";
import type { BoardArt } from "./types.ts";

export { artMode } from "./mode.ts";
export type { ArtMode, BoardArt, GroundInfo, UnitArt } from "./types.ts";

export const createArt = (): BoardArt => new MockArt();
