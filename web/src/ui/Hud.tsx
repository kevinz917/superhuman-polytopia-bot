import { useEffect, useRef, useState } from "react";
import { titleCase } from "../game/vocab.ts";
import { useGame } from "./useGame.ts";

// Star counter ticks like the mobile HUD; a floating +/- shows the change.
function useStarTicker(value: number, key: string, enabled: boolean) {
  const [shown, setShown] = useState(value);
  const [floats, setFloats] = useState<{ id: number; delta: number }[]>([]);
  const [bump, setBump] = useState(0);
  const last = useRef({ value, key });
  useEffect(() => {
    const prev = last.current;
    last.current = { value, key };
    if (prev.key !== key || !enabled || prev.value === value) {
      setShown(value);
      return;
    }
    const from = prev.value,
      delta = value - from,
      t0 = performance.now(),
      id = t0;
    setFloats((f) => [...f, { id, delta }]);
    setBump((b) => b + 1);
    const timer = setTimeout(() => setFloats((f) => f.filter((x) => x.id !== id)), 1100);
    let frame = 0;
    const step = () => {
      const k = Math.min(1, (performance.now() - t0) / 550);
      setShown(Math.round(from + delta * k));
      if (k < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(frame);
      clearTimeout(timer);
      setShown(value);
    };
  }, [value, key, enabled]);
  return { shown, floats, bump };
}

export function Hud() {
  const game = useGame();
  const d = game.data,
    v = d?.observation.visible_state;
  const ticker = useStarTicker(
    v?.stars ?? 0,
    `${game.current?.kind}/${game.current?.id}/${d?.perspective}`,
    game.animationEnabled && game.scrubTarget === null,
  );
  return (
    <header className="topbar">
      <div className="scoreboard">
        <button className="hud-stat hud-score" title="Leaderboard" onClick={() => d && game.openDialog({ type: "details" })}>
          <span>Score</span>
          <strong id="score">{d?.observation.hud?.score?.toLocaleString() ?? "—"}</strong>
        </button>
        <div className={"hud-stat hud-stars" + (ticker.bump % 2 ? " star-bump" : ticker.bump ? " star-bump-alt" : "")}>
          <span id="income">
            {typeof d?.observation.hud?.income === "number" ? `Stars (+${d.observation.hud.income})` : "Stars"}
          </span>
          <div>
            <img alt="" src={"/icons/star.svg"} />
            <strong id="stars">{v ? ticker.shown : "—"}</strong>
            {ticker.floats.map((f) => (
              <span key={f.id} className={"star-float" + (f.delta < 0 ? " spent" : "")}>
                {(f.delta > 0 ? "+" : "−") + Math.abs(f.delta)}
              </span>
            ))}
          </div>
        </div>
        <div className="hud-stat">
          <span>Turn</span>
          <strong id="turn">{v?.turn ?? "—"}</strong>
        </div>
      </div>
    </header>
  );
}

export function GameInfo() {
  const game = useGame();
  const d = game.data;
  const kind =
    d?.mode === "replay"
      ? "RECORDED MATCH"
      : d?.mode === "watch"
        ? "LIVE SELF-PLAY"
        : d?.mode === "agent"
          ? "EXTERNAL AGENT"
          : d
            ? "HUMAN VS LEARNED BOT"
            : "RECORDED MATCH";
  return (
    <aside className="game-info">
      <div className="eyebrow" id="game-kind">
        {kind}
      </div>
      <h1 id="game-title">{d?.title ?? "The square awaits."}</h1>
      <div id="game-subtitle">
        {!d
          ? "Loading your replay library…"
          : d.mode === "play" || d.mode === "agent"
            ? `You are Player ${d.seat} (blue) · 14 × 14 · Drylands`
            : d.note
              ? "Original public-observation trace"
              : `${game.meta?.bot.name} · 14 × 14 · Drylands`}
      </div>
    </aside>
  );
}

export function Status() {
  const game = useGame();
  const d = game.data;
  let message = "Connecting to the game engine",
    busy = false;
  if (game.statusOverride) {
    message = game.statusOverride;
    busy = game.busy;
  } else if (d) {
    if (d.terminated || d.truncated) {
      const rewards = d.event?.rewards;
      const winner = rewards ? Object.keys(rewards).find((k) => rewards[k] > 0) : null;
      message = d.truncated
        ? "Game reached the research horizon"
        : winner
          ? `Player ${winner} wins · replay saved`
          : "Game finished · replay saved";
    } else if (game.busy && (d.mode === "watch" || d.current_player !== d.seat)) {
      const rival = d.mode === "watch" && d.opponent && d.current_player === d.opponent_seat;
      message = rival ? `${d.opponent} is thinking…` : `${game.botName()} is ${d.search ? "searching" : "thinking"}…`;
      busy = true;
    } else if (game.busy) message = "Your turn";
    else if (d.mode === "replay")
      message =
        d.frame_index === d.frames - 1 && d.result
          ? `Recorded result · ${d.result.reward > 0 ? "challenger wins" : d.result.reward < 0 ? "challenger loses" : "no native winner"}`
          : "Replay · Space to play · ← → to step · 1 2 3 to switch view";
    else if (d.mode === "watch")
      message = game.playing
        ? d.opponent
          ? `Live · ${d.title}${d.search ? ` · ${game.botName()} uses turn search` : ""}`
          : `Live self-play · both seats use ${game.botName()}${d.search ? " with turn search" : ""}`
        : "Self-play paused · press play to watch";
    else if (d.frame_index !== d.frames - 1) message = "Reviewing your game · jump to latest to play";
    else if (d.current_player === d.seat) message = "Your turn · select a unit or city";
    else {
      message = `${game.botName()} is ${d.search ? "searching" : "thinking"}…`;
      busy = true;
    }
  }
  return (
    <div
      id="status"
      role="status"
      aria-live="polite"
      className={(busy ? "busy" : "") + (d?.terminated || d?.truncated ? " terminal" : "")}
    >
      <span className="status-dot" />
      <span id="status-text">{message}</span>
    </div>
  );
}

// In a player's fog view the opponent's actions stay hidden unless they are
// public (ending the turn) or every tile they touch is visible to that player.
export function EventLabel() {
  const game = useGame();
  const d = game.data,
    ev = d?.event;
  if (!ev) return <span id="event-label">Opening position</span>;
  const viewer = Number(d.perspective),
    c = ev.action.command,
    tiles = d.observation.visible_state.tiles;
  const seen =
    !viewer ||
    ev.player === viewer ||
    c.kind === 10 ||
    c.kind === 11 ||
    (!!c.kind && c.kind !== 7 && [c.source, c.target].every((t) => t === undefined || t < 0 || tiles[t]?.known));
  return <span id="event-label">{`P${ev.player} · ${seen ? titleCase(ev.action.label) : "Out of sight"}`}</span>;
}
