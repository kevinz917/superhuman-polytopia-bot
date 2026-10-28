// The board is drawn with placeholder shapes only; no game graphics ship with this repository.
import type { ArtMode } from "./types.ts";

export const artMode: ArtMode = "mock";
// Stylesheets pick per-mode images from <html data-art>.
document.documentElement.dataset.art = artMode;
