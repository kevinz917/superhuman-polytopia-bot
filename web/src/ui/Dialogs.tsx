import { useEffect, useRef, useState } from "react";
import { api } from "../api/client.ts";
import type { FrameData, LegalAction, Library as LibraryData } from "../api/types.ts";
import { spriteURL } from "../board/assets.ts";
import { actLabel, cost, structureSprite, techCost, titleCase, unitSprite } from "../game/vocab.ts";
import { ActionButton, Heading } from "./common.tsx";
import { useGame } from "./useGame.ts";

export function DialogHost() {
  const game = useGame();
  const ref = useRef<HTMLDialogElement>(null);
  const dialog = game.dialog;
  useEffect(() => {
    const el = ref.current!;
    if (dialog && !el.open) el.showModal();
    if (!dialog && el.open) el.close();
  }, [dialog]);
  return (
    <dialog
      id="dialog"
      ref={ref}
      className={dialog?.type === "tech" ? "technology-dialog" : ""}
      onClose={() => game.dialog && game.openDialog(null)}
      onClick={(e) => {
        if (e.target !== ref.current) return;
        const r = ref.current.getBoundingClientRect();
        if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) game.openDialog(null);
      }}
    >
      <button id="dialog-close" className="close" aria-label="Close dialog" onClick={() => game.openDialog(null)}>
        ×
      </button>
      <div id="dialog-content">
        {dialog?.type === "menu" && <Menu />}
        {dialog?.type === "library" && <Library />}
        {dialog?.type === "details" && <Details />}
        {dialog?.type === "chronicle" && <Chronicle />}
        {dialog?.type === "relations" && <Relations />}
        {dialog?.type === "actions" && <AllActions />}
        {dialog?.type === "tech" && <TechTree />}
        {dialog?.type === "resign" && <Resign action={dialog.action} />}
      </div>
    </dialog>
  );
}

function Menu() {
  const game = useGame();
  const [seed, setSeed] = useState("");
  const [seat, setSeat] = useState(1);
  // Watch games: "" = the bot against itself; "name|seat" = a named rival (raw policy) in that seat.
  const [rival, setRival] = useState("");
  const rivals = game.meta?.bot.opponents ?? [];
  const opponent = rival ? { name: rival.slice(0, rival.lastIndexOf("|")), seat: Number(rival.slice(rival.lastIndexOf("|") + 1)) } : undefined;
  const start = (mode: "play" | "watch" | "agent") => game.start(mode, seed ? Number(seed) : undefined, seat, opponent);
  const cards: [string, string, () => void, boolean?][] = [
    ["Play the bot", `Take on ${game.botName()}. Switch it between raw policy and turn search any time.`, () => start("play"), true],
    ["Replay library", "Review saved matches and import recorded game traces.", () => game.openDialog({ type: "library" })],
  ];
  return (
    <>
      <Heading
        eyebrow="THE SQUARE IS YOURS"
        title="Bardur"
        text="Watch a learned policy explore, build, and battle. Or take a seat across the board."
      />
      <div className="menu-row">
        <input
          id="new-seed"
          type="number"
          placeholder="Random map"
          min={0}
          max={2147483647}
          aria-label="Map seed"
          value={seed}
          onChange={(e) => setSeed(e.target.value)}
        />
        <select id="new-seat" aria-label="Your seat" value={seat} onChange={(e) => setSeat(Number(e.target.value))}>
          {[1, 2].map((n) => (
            <option key={n} value={n}>
              Play as Player {n}
            </option>
          ))}
        </select>
        {rivals.length > 0 && (
          <select id="new-rival" aria-label="Self-play opponent" value={rival} onChange={(e) => setRival(e.target.value)}>
            <option value="">Self-play: {game.botName()} vs itself</option>
            {rivals.flatMap((name) => [
              <option key={name + "|2"} value={name + "|2"}>
                {game.botName()} (blue) vs {name} (red)
              </option>,
              <option key={name + "|1"} value={name + "|1"}>
                {name} (blue) vs {game.botName()} (red)
              </option>,
            ])}
          </select>
        )}
      </div>
      <div className="menu-cards">
        {cards.map(([title, text, fn, primary]) => (
          <button key={title} className={"menu-card" + (primary ? " primary" : "")} onClick={fn}>
            <strong>{title}</strong>
            <small>{text}</small>
          </button>
        ))}
      </div>
      <label className="animation-setting">
        <input
          type="checkbox"
          aria-label="Animate moves and combat"
          checked={game.animationEnabled}
          onChange={(e) => game.setAnimationEnabled(e.target.checked)}
        />
        <span>Animate moves and combat</span>
      </label>
      <p>Polytopia 2.17.3 rules. Independent local research interface.</p>
      {game.current && (
        <div className="menu-row">
          <button className="pill" onClick={() => game.openDialog(null)}>
            Resume
          </button>
          <button className="pill" onClick={() => game.openDialog({ type: "chronicle" })}>
            Chronicle
          </button>
          <button className="pill" onClick={() => game.exportReplay()}>
            Export replay
          </button>
          <button className="pill" onClick={() => game.saveReplay()}>
            Save to library
          </button>
          {game.current.kind === "games" && (
            <button className="pill" onClick={() => game.closeGame().catch((e) => game.toast(e.message))}>
              Save &amp; close game
            </button>
          )}
          {game.canAct() && (
            <>
              <button className="pill" onClick={() => game.openDialog({ type: "relations" })}>
                Tribe relations
              </button>
              <button className="pill" onClick={() => game.openDialog({ type: "actions" })}>
                All legal actions
              </button>
            </>
          )}
        </div>
      )}
      <p style={{ fontSize: 9 }}>NOT AN OFFICIAL Polytopia PRODUCT. NOT APPROVED BY OR ASSOCIATED WITH MIDJIWAN.</p>
    </>
  );
}

function Library() {
  const game = useGame();
  const [lib, setLib] = useState<LibraryData | null>(null);
  useEffect(() => {
    api<LibraryData>("/api/library").then(setLib, (e) => game.toast(e.message));
  }, [game]);
  return (
    <>
      <Heading eyebrow="THE CHRONICLE" title="Replay library" text="Every saved game can be stepped through one action at a time." />
      <div className="menu-row">
        <button className="pill" onClick={() => document.getElementById("import-file")?.click()}>
          ＋ Import replay
        </button>
        <button className="pill" onClick={() => game.openDialog({ type: "menu" })}>
          New game
        </button>
      </div>
      {[...(lib?.replays ?? [])]
        // The featured game first, then games recorded in the real game, then OMNI games (the strongest bot),
        // then newest first.
        .sort(
          (a, b) =>
            Number(!!b.featured) - Number(!!a.featured) ||
            Number(/^real-game/.test(b.source ?? "")) - Number(/^real-game/.test(a.source ?? "")) ||
            Number(/OMNI/.test(b.title)) - Number(/OMNI/.test(a.title)) ||
            b.created - a.created,
        )
        .map((r) => (
          <button key={r.id} className="library-item" onClick={() => game.openReplay(r.id)}>
            <span>▷</span>
            <div>
              <strong>{r.title}</strong>
              <small>
                {r.frames.toLocaleString()} frames · {r.complete ? "Recorded outcome" : "In progress"} ·{" "}
                {r.available_views.includes("all") ? "All perspectives" : "Player observations"}
              </small>
            </div>
            <span>→</span>
          </button>
        ))}
      {/* Unfinished live games follow the recorded replays; the Resume pill in the menu reaches the current one. */}
      {lib?.games.map((g) => (
        <button key={g.id} className="library-item" onClick={() => game.openGame(g.id)}>
          <span>◉</span>
          <div>
            <strong>{g.title}</strong>
            <small>
              Live game · {g.frames - 1} actions · Player {g.current_player} to move
            </small>
          </div>
        </button>
      ))}
      {lib && !lib.replays.length && !lib.games.length && <p>Start a game or import a .jsonl.gz trace to begin.</p>}
    </>
  );
}

function Details() {
  const game = useGame();
  const d = game.data!;
  const pairs: [string, string | number][] = [
    ["Rules", "bardur-1v1-v1"],
    ["Mode", d.mode],
    ["Checkpoint", d.bot?.name || "See imported trace provenance"],
    ["Training", d.bot?.training || "Recorded experiment"],
    ["Checkpoint hash", d.bot?.checkpoint_sha256 || "Stored in original experiment protocol"],
    ["Actions", d.frames - 1],
    ["Perspective", d.perspective],
    ["Controls", "Click a unit, then a highlighted tile. Drag to pan; scroll to zoom."],
    [
      "Agent endpoint",
      game.current?.kind === "games" ? `GET /api/games/${game.current.id}/observation` : "Start “Connect a model” to get an agent seat.",
    ],
  ];
  return (
    <>
      <Heading
        eyebrow="MATCH DETAILS"
        title={d.title}
        text="The same public observation and legal-action interface powers the human and model players."
      />
      <Leaderboard />
      <table className="detail-table">
        <tbody>
          {pairs.map(([k, v]) => (
            <tr key={k}>
              <td>{k}</td>
              <td>{String(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {d.note && <p>{d.note}</p>}
      <button className="pill" onClick={() => game.exportReplay()}>
        Export replay
      </button>
    </>
  );
}

// Mobile "Game Stats" ranking: every tribe's score and city count, best first.
function Leaderboard() {
  const game = useGame();
  const d = game.data!,
    obs = d.observation,
    v = obs.visible_state;
  const home = ["play", "agent"].includes(d.mode) ? (d.seat ?? 1) : 1;
  // Replays recorded before the HUD carried every player fall back to what is visible.
  const rows =
    obs.hud?.players ??
    [1, 2].map((player) => ({
      player,
      score: player === obs.player ? (obs.hud?.score ?? null) : null,
      cities: v.cities.filter((c) => c.owner === player).length,
    }));
  const ranked = [...rows].sort((a, b) => (b.score ?? -1) - (a.score ?? -1) || b.cities - a.cities);
  return (
    <ol className="leaderboard" aria-label="Leaderboard">
      {ranked.map((r, i) => (
        <li key={r.player} className={r.player === home ? "home" : "away"}>
          <span className="rank">{i + 1}</span>
          <span className="tribe-badge" aria-hidden="true">
            <img src={"/icons/head.svg"} alt="" />
          </span>
          <span className="who">
            <strong>
              Player {r.player}
              {d.mode === "play" || d.mode === "agent" ? (r.player === d.seat ? " (you)" : d.mode === "play" ? " (bot)" : "") : ""}
            </strong>
            <small>
              {r.cities} {r.cities === 1 ? "city" : "cities"}
            </small>
          </span>
          <span className="points">
            {r.score === null ? "—" : r.score.toLocaleString()}
            <small>points</small>
          </span>
        </li>
      ))}
    </ol>
  );
}

function Chronicle() {
  const game = useGame();
  const [rows, setRows] = useState<{ i: number; label: string }[]>([]);
  useEffect(() => {
    const d = game.data,
      current = game.current;
    if (!d || !current) return;
    let alive = true;
    const stop = d.frame_index,
      start = Math.max(0, stop - 19);
    void (async () => {
      const out: { i: number; label: string }[] = [];
      for (let i = stop; i >= start; i--) {
        const f = await api<FrameData>(`/api/${current.kind}/${current.id}?frame=${i}&view=${game.perspective}`);
        out.push({ i, label: f.event ? `Player ${f.event.player} · ${titleCase(f.event.action.label)}` : "Opening position" });
        if (alive) setRows([...out]);
      }
    })().catch((e) => game.toast(e.message));
    return () => {
      alive = false;
    };
  }, [game]);
  return (
    <>
      <Heading eyebrow="THE CHRONICLE" title="Recent moves" text="Select a move to return to that position." />
      {rows.map((r) => (
        <button
          key={r.i}
          className="history-row"
          onClick={() => {
            game.openDialog(null);
            game.seek(r.i);
          }}
        >
          <small>{String(r.i).padStart(4, "0")}</small>
          <span>{r.label}</span>
        </button>
      ))}
    </>
  );
}

function Relations() {
  const game = useGame();
  const actions = game.legal.filter((a) => a.command.kind === 3 && a.command.parameter >= 18);
  return (
    <>
      <Heading
        eyebrow="TRIBE RELATIONS"
        title="The other Bardur"
        text="Diplomatic actions are available when the other tribe is known and the required technology is researched."
      />
      {actions.length && game.canAct() ? (
        <div className="actions">
          {actions.map((a) => (
            <ActionButton key={a.action_id} action={a} />
          ))}
        </div>
      ) : (
        <p>No diplomatic actions available.</p>
      )}
    </>
  );
}

function AllActions() {
  const game = useGame();
  const [filter, setFilter] = useState("");
  return (
    <>
      <Heading
        eyebrow="AVAILABLE ORDERS"
        title="All legal actions"
        text="Every action offered by the frozen simulator, including city rewards and resignation."
      />
      <input placeholder="Filter actions" aria-label="Filter legal actions" value={filter} onChange={(e) => setFilter(e.target.value)} />
      <div className="actions" style={{ maxHeight: "48vh", justifyContent: "start" }}>
        {game.legal
          .filter((a) => actLabel(game.meta!, a).toLowerCase().includes(filter.toLowerCase()))
          .map((a) => (
            <ActionButton
              key={a.action_id}
              action={a}
              extra={[
                ...(a.command.source >= 0 ? [`From ${a.command.source}`] : []),
                ...(a.command.target >= 0 ? [`To ${a.command.target}`] : []),
              ]}
            />
          ))}
      </div>
    </>
  );
}

function Resign({ action }: { action: LegalAction }) {
  const game = useGame();
  return (
    <>
      <Heading eyebrow="LEAVE THE SQUARE" title="Resign this game?" text="This ends the game and saves the replay." />
      <button className="pill" onClick={() => game.perform(action)}>
        Resign
      </button>
      <button className="pill" onClick={() => game.openDialog(null)}>
        Keep playing
      </button>
    </>
  );
}

const BRANCHES = [
  [1, 2, 3, 4, 5],
  [6, 7, 38, 8, 9],
  [10, 13, 14, 44, 12],
  [15, 16, 17, 18, 19],
  [20, 21, 22, 23, 24],
];
const TECH_ICONS: Record<number, string> = {
  1: "unit-rider",
  2: "temple",
  3: "unit-knight",
  4: "ground",
  5: "market",
  6: "fruit",
  7: "unit-defender",
  38: "unit-cloak",
  8: "farm",
  9: "windmill",
  10: "water",
  13: "unit-scoutship",
  14: "unit-bombership",
  44: "unit-rammership",
  12: "temple",
  15: "animal",
  16: "lumber",
  17: "unit-catapult",
  18: "unit-archer",
  19: "forest",
  20: "mountain",
  21: "temple",
  22: "unit-mindbender",
  23: "mine",
  24: "unit-swordsman",
};

// What each technology does beyond the units and buildings in the rules,
// worded like the mobile tech popup and checked against the engine's gates.
const TECH_INFO: Record<number, { text: string; abilities?: string[] }> = {
  1: { text: "Riders are fast units for scouting and raids." },
  2: { text: "Temples add score and population to your cities.", abilities: ["Disband units"] },
  3: { text: "Knights chain attacks across the battlefield.", abilities: ["Destroy buildings"] },
  4: { text: "Roads speed up movement and connect cities to your capital." },
  5: { text: "Markets earn stars next to windmills, sawmills and forges." },
  6: { text: "Gather fruit to grow your cities.", abilities: ["Harvest fruit", "Reveals crops"] },
  7: { text: "Defenders hold the line with high defence.", abilities: ["Offer peace"] },
  38: { text: "Cloaks are hidden scouts that slip past enemy lines.", abilities: ["Establish embassy", "Reveals enemy capital"] },
  8: { text: "Farms on crops grow your cities." },
  9: { text: "Windmills grow cities next to farms.", abilities: ["Burn forest"] },
  10: { text: "Ports let land units sail as rafts.", abilities: ["Fishing", "Move into shallow water"] },
  13: { text: "Scouts explore the seas quickly.", abilities: ["Move into ocean"] },
  14: { text: "Bombers strike ships and coasts from range." },
  44: { text: "Rammers are sturdy warships." },
  12: { text: "Water temples add score from the sea.", abilities: ["Defence bonus in water"] },
  15: { text: "Hunt wild animals to grow your cities.", abilities: ["Hunting"] },
  16: { text: "Lumber huts turn forests into population.", abilities: ["Clear forest"] },
  17: { text: "Catapults hit from long range; sawmills boost lumber huts." },
  18: { text: "Archers attack from two tiles away.", abilities: ["Defence bonus in forest"] },
  19: { text: "Forest temples add score from the woods.", abilities: ["Grow forest"] },
  20: { text: "Mountains become passable high ground.", abilities: ["Move onto mountains", "Defence bonus on mountains", "Reveals metal"] },
  21: { text: "Rewards turns spent without attacking.", abilities: ["Pacifist"] },
  22: { text: "Mind benders convert enemy units; mountain temples add score.", abilities: ["Research costs −33%"] },
  23: { text: "Mines on metal grow your cities." },
  24: { text: "Swordsmen are strong attackers; forges boost mines." },
};

function TechTree() {
  const game = useGame();
  const [picked, setPicked] = useState<number | null>(null);
  const meta = game.meta!,
    d = game.data!,
    obs = d.observation,
    v = obs.visible_state;
  const pos = (angle: number, r: number) => ({
    x: 300 + Math.cos((angle * Math.PI) / 180) * r,
    y: 300 + Math.sin((angle * Math.PI) / 180) * r,
  });
  // Edges follow the rules' `next` lists: root → two tier-2 techs → one tier-3 each.
  const nodes: { id: number; parent: number; p: { x: number; y: number }; from: { x: number; y: number } }[] = [];
  BRANCHES.forEach((ids, i) => {
    const angle = -90 + i * 72,
      p1 = pos(angle, 85);
    nodes.push({ id: ids[0], parent: 0, p: p1, from: { x: 300, y: 300 } });
    for (let side = 0; side < 2; side++) {
      const a = angle + (side ? 17 : -17),
        p2 = pos(a, 165);
      nodes.push({ id: ids[1 + side * 2], parent: ids[0], p: p2, from: p1 });
      nodes.push({ id: ids[2 + side * 2], parent: ids[1 + side * 2], p: pos(a, 250), from: p2 });
    }
  });
  const owned = (id: number) => id === 0 || v.technology.includes(id);
  const research = (id: number) => game.legal.find((a) => a.command.kind === 7 && a.command.parameter === id);
  const tech = picked !== null ? meta.rules.technologies.find((t) => t.id === picked) : null;
  const pickedNode = nodes.find((n) => n.id === picked);
  const pickedAction = picked !== null ? research(picked) : undefined;
  const unlocks = tech
    ? [
        ...(tech.units || []).map((id) => ({
          key: "u" + id,
          name: meta.rules.units.find((x) => x.id === id)?.name,
          sprite: unitSprite(id),
        })),
        ...(tech.structures || [])
          .map((id) => meta.rules.structures.find((x) => x.id === id))
          .filter((s) => s && s.tribe == null)
          .map((s) => ({ key: "s" + s!.id, name: s!.name, sprite: structureSprite(meta, s!.id) })),
      ].filter((x) => x.name)
    : [];
  return (
    <>
      <Heading
        eyebrow="KNOWLEDGE IS POWER"
        title="Technology"
        text="Learn a technology to unlock its neighboring branches. Available research is blue."
      />
      <div className="tech-tree">
        <svg viewBox="0 0 600 600" aria-hidden="true">
          {nodes.map(({ id, p, from }) => (
            <line
              key={id}
              x1={from.x}
              y1={from.y}
              x2={p.x}
              y2={p.y}
              stroke={owned(id) ? "#9dbeb3" : "#d8e2df"}
              strokeWidth={2}
            />
          ))}
        </svg>
        <div className="tech-center">
          <img src={"/icons/head.svg"} alt="Bardur" />
        </div>
        {nodes.map(({ id, parent, p }) => {
          const t = meta.rules.technologies.find((x) => x.id === id)!,
            learned = owned(id),
            reachable = !learned && owned(parent),
            price = techCost(meta, obs, id);
          const state = learned
            ? " researched"
            : reachable
              ? " available" + (research(id) ? " ready" : v.stars < price ? " short" : "")
              : " locked";
          return (
            <button
              key={id}
              className={"tech" + state + (id === picked ? " picked" : "")}
              style={{ left: p.x / 6 + "%", top: p.y / 6 + "%" }}
              aria-pressed={id === picked}
              onClick={() => setPicked(id === picked ? null : id)}
            >
              <span className="tech-icon" aria-hidden="true">
                <img src={spriteURL(TECH_ICONS[id] || "star")} alt="" />
              </span>
              <span>{titleCase(t.name)}</span>
              <small>{learned ? "Learned" : `${price} ★`}</small>
            </button>
          );
        })}
      </div>
      {tech && pickedNode && (
        <div id="tech-detail" className="tech-detail">
          <button className="tech-detail-close" aria-label="Close technology details" onClick={() => setPicked(null)}>
            ×
          </button>
          <div className="tech-detail-head">
            <span className={"tech-detail-icon" + (owned(tech.id) ? " researched" : owned(pickedNode.parent) ? " available" : "")}>
              <img src={spriteURL(TECH_ICONS[tech.id] || "star")} alt="" />
            </span>
            <div>
              <h3>{titleCase(tech.name)}</h3>
              <div className="tech-detail-meta">
                Tier {tech.tier} ·{" "}
                {owned(tech.id) ? "Learned" : (
                  <>
                    {techCost(meta, obs, tech.id)} <img src={"/icons/star.svg"} alt="stars" />
                  </>
                )}
              </div>
            </div>
          </div>
          <p>{TECH_INFO[tech.id]?.text ?? "Opens the next branch of the tree."}</p>
          {(unlocks.length > 0 || (TECH_INFO[tech.id]?.abilities?.length ?? 0) > 0) && (
            <div className="tech-unlocks" aria-label="Unlocks">
              {unlocks.map((u) => (
                <span key={u.key} className="tech-unlock">
                  <img src={spriteURL(u.sprite)} alt="" />
                  {titleCase(u.name!)}
                </span>
              ))}
              {TECH_INFO[tech.id]?.abilities?.map((a) => (
                <span key={a} className="tech-unlock ability">
                  {a}
                </span>
              ))}
            </div>
          )}
          {owned(tech.id) ? (
            <p className="tech-state">You have learned this technology.</p>
          ) : pickedAction && game.canAct() ? (
            <button className="research-button" data-action-id={pickedAction.action_id} onClick={() => game.takeAction(pickedAction)}>
              Research · {cost(meta, obs, pickedAction)} ★
            </button>
          ) : (
            <p className="tech-state">
              {!owned(pickedNode.parent)
                ? `Requires ${titleCase(meta.rules.technologies.find((t) => t.id === pickedNode.parent)!.name)}.`
                : v.stars < techCost(meta, obs, tech.id)
                  ? `You need ${techCost(meta, obs, tech.id) - v.stars} more ★.`
                  : game.canAct()
                    ? "Not available right now."
                    : d.mode === "replay" || d.mode === "watch"
                      ? "Research is available when you play."
                      : "Research is available on your turn."}
            </p>
          )}
        </div>
      )}
    </>
  );
}
