import { useSyncExternalStore } from "react";
import { controller } from "../game/controller.ts";

// Re-render whenever the game controller changes.
export function useGame() {
  useSyncExternalStore(controller.subscribe, controller.getVersion);
  return controller;
}
