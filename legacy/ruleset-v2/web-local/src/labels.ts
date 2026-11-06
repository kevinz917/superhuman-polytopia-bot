// Names the interface shows for engine ids and actions (studio/service.py: TERRAIN, RESOURCE, ABILITY, REWARD,
// action_label).
import type { Command, Rules } from "../api/types.ts";

export const TERRAIN: Record<string, string> = { 0: "Unknown", 1: "Shallow water", 2: "Ocean", 3: "Field", 4: "Mountain", 5: "Forest", 6: "Ice", 7: "Wetland", 8: "Flooded field" };
export const RESOURCE: Record<string, string> = { 0: "None", 1: "Animal", 2: "Crop", 3: "Fish", 4: "Whale", 5: "Metal", 6: "Fruit", 7: "Fungi", 8: "Starfish" };
export const ABILITY: Record<string, string> = { 1: "Burn forest", 2: "Clear forest", 3: "Grow forest", 4: "Destroy", 7: "Recover", 8: "Disband", 9: "Heal others", 16: "Promote", 18: "Break peace", 19: "Respond to peace", 20: "Establish embassy", 21: "Offer peace", 22: "Remove embassy" };
export const REWARD: Record<string, string> = { 1: "City wall", 2: "Park", 3: "Workshop", 4: "Explorer", 5: "Border growth", 6: "Super unit", 7: "Resources", 8: "Population growth" };
const KIND: Record<number, string> = { 1: "Move", 2: "Attack", 5: "Harvest", 8: "Capture", 10: "End turn", 11: "Resign" };

export function actionLabel(c: Command, rules: Rules): string {
  const name = (group: keyof Rules, id: number) => rules[group].find((x) => x.id === id)?.name ?? String(id);
  const p = c.parameter;
  if (c.kind === 3) {
    if (p === 19) return c.target === 1 ? "Accept peace" : "Reject peace";
    return ABILITY[p] ?? `Ability ${p}`;
  }
  if (c.kind === 4) return "Train " + name("units", p);
  if (c.kind === 6) return "Build " + name("structures", p);
  if (c.kind === 7) return "Research " + name("technologies", p);
  if (c.kind === 9) return REWARD[p] ?? `City reward ${p}`;
  return KIND[c.kind] ?? `Action ${c.kind}`;
}
