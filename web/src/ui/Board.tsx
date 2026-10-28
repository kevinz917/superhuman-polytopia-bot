import { useEffect, useRef } from "react";
import { BoardScene } from "../board/BoardScene.ts";
import { controller } from "../game/controller.ts";

// Mounts the WebGL board once; the controller drives it imperatively.
export function Board() {
  const canvas = useRef<HTMLCanvasElement>(null),
    overlay = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const scene = new BoardScene(canvas.current!, overlay.current!, controller.selectTile);
    let alive = true;
    void scene.init().then(() => {
      if (alive) controller.attachBoard(scene, canvas.current!);
    });
    return () => {
      alive = false;
      scene.destroy();
    };
  }, []);
  return (
    <div id="board-wrap">
      <canvas id="board" ref={canvas} aria-label="Isometric Bardur battlefield" />
      <div id="tile-controls" ref={overlay} role="group" aria-label="Game board tiles" />
    </div>
  );
}

export function BoardTools() {
  return (
    <div className="board-tools">
      <button id="zoom-in" aria-label="Zoom in" onClick={() => controller.board?.zoomBy(1.2)}>
        ＋
      </button>
      <button id="zoom-out" aria-label="Zoom out" onClick={() => controller.board?.zoomBy(1 / 1.2)}>
        −
      </button>
      <button id="recenter" aria-label="Center board" onClick={() => controller.board?.center()}>
        ⌖
      </button>
    </div>
  );
}
