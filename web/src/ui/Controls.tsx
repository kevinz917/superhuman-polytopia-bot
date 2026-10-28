import { useEffect, useState } from "react";
import { EventLabel } from "./Hud.tsx";
import { useGame } from "./useGame.ts";

export function GameNav() {
  const game = useGame();
  const canEnd = game.canAct() && game.legal.some((a) => a.command.kind === 10) && !game.animating;
  return (
    <nav className="game-nav">
      <button id="menu-button" onClick={() => game.openDialog({ type: "menu" })}>
        <span aria-hidden="true" className="round-icon menu-icon">
          <svg viewBox="0 0 40 40">
            <path d="M10 10h20M10 20h20M10 30h20" />
          </svg>
        </span>
        <span>Menu</span>
      </button>
      <button id="history-button" onClick={() => game.data && game.openDialog({ type: "details" })}>
        <span aria-hidden="true" className="round-icon stats-icon">
          <img src="/icons/head.svg" alt="" />
        </span>
        <span>Game Stats</span>
      </button>
      <button id="tech-button" onClick={() => game.data && game.openDialog({ type: "tech" })}>
        <span aria-hidden="true" className="round-icon tech-nav-icon">
          <svg viewBox="0 0 48 58">
            <path fill="#00cfee" stroke="none" d="m9 39 15-9 15 9v10L24 58 9 49z" />
            <path fill="#007abc" stroke="none" d="m9 39 15 9v10L9 49z" />
            <path fill="#d5faff" stroke="none" d="m20 3 7-3 6 4v28l-8 8-8-7z" />
            <path fill="#80d2eb" stroke="none" d="m27 0 6 4v28l-8 8V5z" />
            <path fill="none" stroke="#4b97c9" strokeWidth="2" d="m12 34 12 7m-14-2 12 7m-9-16 12 7" />
          </svg>
        </span>
        <span>Tech Tree</span>
      </button>
      <button
        id="end-turn"
        disabled={!canEnd}
        onClick={() => {
          const a = game.legal.find((x) => x.command.kind === 10);
          if (a) void game.takeAction(a);
        }}
      >
        <span aria-hidden="true" className="round-icon end-icon">
          <svg viewBox="0 0 50 50">
            <path d="m11 27 11 11 20-27" />
          </svg>
        </span>
        <span id="end-label">Next Turn</span>
      </button>
    </nav>
  );
}

// Opponent mode for live games: the raw policy, or leak-free turn search on top of it.
export function OpponentMode() {
  const game = useGame();
  const d = game.data;
  if (!game.meta?.search?.available || game.current?.kind !== "games" || !d || d.mode === "agent") return null;
  const on = !!d.search;
  const modes = [
    { search: false, label: "Policy", hint: "Raw network, one forward pass per move (instant)" },
    {
      search: true,
      label: "Search",
      hint: "Leak-free turn search: 16 candidate turns × 64 sampled replies, from the bot's own view (a few seconds per turn)",
    },
  ];
  return (
    <div className="opponent-mode" role="radiogroup" aria-label="Opponent mode">
      <span className="opponent-mode-label">Opponent</span>
      {modes.map((m) => (
        <button
          key={m.label}
          role="radio"
          aria-checked={on === m.search}
          className={on === m.search ? "active" : ""}
          title={m.hint}
          onClick={() => on !== m.search && void game.setSearch(m.search)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}

// Replay viewing modes: each player's own fog of war, or the whole board. Number keys pick a player; the god
// view takes the next number (3 in a two-seat game, 5 with four seats).
export function viewModes(available: string[] = []) {
  const seats = available.filter((v) => /^\d+$/.test(v)).sort((a, b) => Number(a) - Number(b));
  const modes = seats.map((s) => ({ view: s, label: `Player ${s}`, hint: `Only what Player ${s} can see`, key: s }));
  if (available.includes("all"))
    modes.push({ view: "all", label: "God view", hint: "The whole board, no fog", key: String(seats.length + 1) });
  return modes;
}

export function ReplayDock() {
  const game = useGame();
  const d = game.data;
  const hidden =
    !d || (!["replay", "watch"].includes(d.mode) && d.frame_index === d.frames - 1 && !d.terminated && !d.truncated);
  const modes = viewModes(d?.available_views);
  // The thumb follows the pointer; the board catches up as frames load.
  const position = game.scrubTarget ?? d?.frame_index ?? 0;
  return (
    <div id="replay-dock" className={"replay-dock" + (hidden ? " hidden" : "")}>
      {modes.length > 1 && (
        <div id="view-modes" className="view-modes" role="radiogroup" aria-label="Viewing mode">
          {modes.map((m) => {
            const active = d?.perspective === m.view;
            return (
              <button
                key={m.view}
                role="radio"
                aria-checked={active}
                className={`view-mode view-${m.view}` + (active ? " active" : "")}
                title={`${m.hint} (${m.key})`}
                onClick={() => !active && game.setPerspective(m.view)}
              >
                <span className="view-dot" aria-hidden="true" />
                {m.label}
              </button>
            );
          })}
        </div>
      )}
      <div className="transport">
        <button id="first" aria-label="First action" onClick={() => game.seek(0)}>
          |◀
        </button>
        <button id="prev" aria-label="Previous action" onClick={() => d && game.seek(d.frame_index - 1)}>
          ‹
        </button>
        <button
          id="play"
          className="play-button"
          aria-label={game.playing ? "Pause replay" : "Play replay"}
          onClick={game.togglePlay}
        >
          {game.playing ? "Ⅱ" : "▶"}
        </button>
        <button id="next" aria-label="Next action" onClick={() => d && game.seek(d.frame_index + 1, true)}>
          ›
        </button>
        <button id="last" aria-label="Latest action" onClick={() => d && game.seek(d.frames - 1)}>
          ▶|
        </button>
      </div>
      <div className="timeline">
        <div className="timeline-label">
          <EventLabel />
          <span id="frame-count">{d ? `${position} / ${d.frames - 1}` : "0 / 0"}</span>
        </div>
        <input
          id="scrubber"
          aria-label="Replay position"
          type="range"
          min={0}
          max={d ? d.frames - 1 : 0}
          value={position}
          onChange={(e) => game.seek(Number(e.target.value))}
        />
      </div>
      <select id="speed" aria-label="Playback speed" value={game.speed} onChange={(e) => game.setSpeed(Number(e.target.value))}>
        {[1, 2, 4, 8].map((s) => (
          <option key={s} value={s}>
            {s}×
          </option>
        ))}
      </select>
    </div>
  );
}

export function Toast() {
  const game = useGame();
  const [dismissed, setDismissed] = useState(0);
  const message = game.toastMessage;
  useEffect(() => {
    if (!message) return;
    const t = setTimeout(() => setDismissed(message.id), 5500);
    return () => clearTimeout(t);
  }, [message]);
  const visible = !!message && message.id !== dismissed;
  return (
    <div id="toast" role="alert" className={visible ? "" : "hidden"}>
      {message?.text}
    </div>
  );
}
