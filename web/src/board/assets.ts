// Interface icons rendered from the art's placeholder shapes with the board's renderer.
import type { Container, Renderer } from "pixi.js";

let iconSource: { renderer: Renderer; icon: (name: string) => Container | null } | null = null;
const mockIcons: Record<string, string> = {};
export function setIconSource(renderer: Renderer, icon: (name: string) => Container | null) {
  iconSource = { renderer, icon };
}
const BLANK = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";

// Image URL for an interface icon: a rendered placeholder shape.
export function spriteURL(name: string) {
  if (!(name in mockIcons) && iconSource) {
    const node = iconSource.icon(name);
    mockIcons[name] = node ? iconSource.renderer.extract.canvas({ target: node, resolution: 3 }).toDataURL!() : BLANK;
  }
  return mockIcons[name] ?? BLANK;
}
