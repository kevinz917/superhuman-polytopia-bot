import { useEffect } from "react";
import { controller } from "./game/controller.ts";
import { Board, BoardTools } from "./ui/Board.tsx";
import { GameNav, OpponentMode, ReplayDock, Toast, viewModes } from "./ui/Controls.tsx";
import { DialogHost } from "./ui/Dialogs.tsx";
import { GameInfo, Hud, Status } from "./ui/Hud.tsx";
import { ReplayInsights } from "./ui/Insights.tsx";
import { RewardOverlay, SelectionBar } from "./ui/SelectionBar.tsx";

export default function App() {
  useEffect(() => {
    controller.exposeAgentApi();
    void controller.init();
    const keys = (e: KeyboardEvent) => {
      const focus = document.activeElement?.tagName ?? "";
      if (controller.dialog || ["INPUT", "SELECT", "BUTTON"].includes(focus)) return;
      const d = controller.data;
      if (e.code === "Space") {
        e.preventDefault();
        controller.togglePlay();
      }
      if (d && e.code === "ArrowRight") controller.seek(d.frame_index + 1, true);
      if (d && e.code === "ArrowLeft") controller.seek(d.frame_index - 1);
      if (e.code === "Escape") controller.clearSelection();
      const mode = viewModes(d?.available_views).find((m) => e.key === m.key);
      if (d && mode && d.perspective !== mode.view) controller.setPerspective(mode.view);
    };
    window.addEventListener("keydown", keys);
    return () => window.removeEventListener("keydown", keys);
  }, []);
  return (
    <main id="app">
      <Hud />
      <Board />
      <BoardTools />
      <GameInfo />
      <Status />
      <OpponentMode />
      <SelectionBar />
      <RewardOverlay />
      <ReplayDock />
      <ReplayInsights />
      <GameNav />
      <footer>Unofficial Polytopia research interface · Bardur 1v1</footer>
      <Toast />
      <DialogHost />
      <input
        id="import-file"
        type="file"
        accept=".json,.jsonl,.gz"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void controller.importFile(file);
          e.target.value = "";
        }}
      />
    </main>
  );
}
