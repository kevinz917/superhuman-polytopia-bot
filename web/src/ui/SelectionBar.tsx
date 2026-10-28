import { useEffect, type ReactNode } from "react";
import type { LegalAction } from "../api/types.ts";
import {
  actLabel,
  cityName,
  HARVEST,
  RESOURCE_SPRITES,
  REWARDS,
  TERRAIN_TEXT,
  titleCase,
  unitName,
  unitSprite,
  UNIT_SPRITES,
} from "../game/vocab.ts";
import { Art, RoundAction } from "./common.tsx";
import { useGame } from "./useGame.ts";

function StatLine({ parts }: { parts: [string, string | number][] }) {
  return (
    <div className="stat-line">
      {parts.map(([label, value]) => (
        <span key={label}>
          <small>{label}</small>
          <b>{value}</b>
        </span>
      ))}
    </div>
  );
}

// Mobile selection bar: what is selected, its stats, and round order buttons.
export function SelectionBar() {
  const game = useGame();
  const d = game.data,
    meta = game.meta,
    selected = game.selected;
  // Live play swaps the round controls for this bar; replays keep their transport.
  const live = game.current?.kind === "games" && d?.mode !== "watch";
  useEffect(() => {
    document.body.classList.toggle("selecting", selected !== null && live);
  }, [selected, live]);
  if (selected === null || !d || !meta) return <div id="selection" className="selection hidden" />;
  const v = d.observation.visible_state,
    p = d.observation.player,
    t = v.tiles[selected],
    u = v.units.find((x) => x.tile === selected),
    city = v.cities.find((c) => c.tile === selected),
    canAct = game.canAct(),
    legal = canAct ? game.legal : [];
  if (!t) return null;
  let title = "",
    portrait: Parameters<typeof Art>[0]["art"] = { sprite: "ground" };
  const detail: ReactNode[] = [];
  const orders: ReactNode[] = [];
  if (!t.known) {
    title = "Unexplored";
    detail.push(<p key="fog">Explore this tile to reveal what lies beneath the clouds.</p>);
  } else if (game.layer === "unit" && u) {
    const rules = meta.rules.units.find((r) => r.id === u.kind);
    title = unitName(meta, u.kind);
    portrait = { sprite: unitSprite(u.kind) };
    detail.push(
      <StatLine
        key="stats"
        parts={[
          ["Health", `${Math.ceil(u.health)}/${(rules?.health || 10) + (u.veteran ? 5 : 0)}`],
          ["Attack", rules?.attack ?? "—"],
          ["Defence", rules?.defense ?? "—"],
          ["Move", rules?.movement ?? "—"],
          ["Range", rules?.range ?? "—"],
        ]}
      />,
    );
    const notes: string[] = [];
    if (u.owner !== p) notes.push(`Player ${u.owner}`);
    if (u.veteran) notes.push("Veteran");
    if (u.owner === p && u.moved && u.attacked) notes.push("No moves left");
    if (rules?.skills?.length) notes.push(rules.skills.map(titleCase).join(" · "));
    if (notes.length) detail.push(<p key="notes">{notes.join(" · ")}</p>);
    for (const a of legal.filter(
      (a) =>
        (a.command.kind === 8 && a.command.source === selected) ||
        (a.command.kind === 3 && a.command.source === selected && a.command.parameter < 18),
    ))
      orders.push(<RoundAction key={a.action_id} action={a} />);
  } else if (game.layer === "city" && city) {
    const level = city.level || 1;
    title = city.owner ? cityName(selected) : "Village";
    portrait = { sprite: city.owner && level > 2 ? "house-3" : "house-1" };
    if (!city.owner) detail.push(<p key="v">Move a unit here and capture it to found a new city.</p>);
    else if (city.level !== undefined) {
      detail.push(
        <StatLine
          key="stats"
          parts={[
            ["Level", level],
            ["Population", `${city.progress}/${level + 1}`],
            ["Income", `+${city.production} ★`],
          ]}
        />,
      );
      if (v.tiles[selected].capital) detail.push(<p key="cap">Capital</p>);
    } else detail.push(<p key="o">Player {city.owner} city</p>);
    if (city.owner === p && canAct) {
      // Every unit your technology unlocks, greyed out when you cannot train it.
      const known = new Set(
        meta.rules.technologies.filter((x) => x.id === 0 || v.technology.includes(x.id)).flatMap((x) => x.units),
      );
      for (const kind of [...known].filter((k) => UNIT_SPRITES[k] && k < 39)) {
        const a = legal.find(
          (a) => a.command.kind === 4 && a.command.source === selected && a.command.parameter === kind,
        );
        const shown: LegalAction = a || { action_id: `train-${kind}`, command: { kind: 4, parameter: kind, source: selected, target: -1 } };
        orders.push(<RoundAction key={kind} action={shown} label={unitName(meta, kind)} enabled={!!a} />);
      }
      for (const a of legal.filter(
        (a) => a.command.kind === 4 && a.command.source === selected && !known.has(a.command.parameter),
      ))
        orders.push(<RoundAction key={a.action_id} action={a} />);
    }
  } else {
    const resource = t.resource ? meta.resources[t.resource] : null;
    title = resource || meta.terrain[t.terrain ?? 0] || "Land";
    portrait = {
      sprite:
        RESOURCE_SPRITES[t.resource ?? 0] ||
        ({ 4: "mountain", 5: "forest", 1: "water", 2: "ocean" } as Record<number, string>)[t.terrain ?? 0] ||
        "ground",
    };
    const lines: string[] = [];
    if (resource) lines.push(meta.terrain[t.terrain ?? 0]);
    if (TERRAIN_TEXT[t.terrain ?? 0]) lines.push(TERRAIN_TEXT[t.terrain ?? 0]);
    if ((t.structure ?? 0) > 2) lines.push(meta.rules.structures.find((x) => x.id === t.structure)?.name || "");
    lines.push(t.owner ? (t.owner === p ? "Your territory" : `Player ${t.owner} territory`) : "Unclaimed land");
    detail.push(<p key="t">{lines.filter(Boolean).join(" · ")}</p>);
    for (const a of legal.filter(
      (a) =>
        a.command.target === selected &&
        ([5, 6].includes(a.command.kind) || (a.command.kind === 3 && a.command.source < 0 && a.command.parameter < 18)),
    ))
      orders.push(
        <RoundAction
          key={a.action_id}
          action={a}
          label={a.command.kind === 5 ? HARVEST[t.resource ?? 0] || "Harvest" : actLabel(meta, a)}
        />,
      );
  }
  return (
    <div id="selection" className={"selection" + (live ? "" : " over-dock")} data-layer={game.layer ?? undefined}>
      <button id="clear-selection" className="close" aria-label="Deselect tile" onClick={() => game.clearSelection()}>
        ×
      </button>
      <div className="selection-head">
        <span id="selection-portrait" className="selection-portrait">
          <Art art={portrait} />
        </span>
        <div>
          <div className="eyebrow" id="tile-coord">
            TILE {selected} · {(selected % v.board_size) + 1}, {Math.floor(selected / v.board_size) + 1}
          </div>
          <h2 id="tile-title">{title}</h2>
          <div id="tile-detail">{detail}</div>
        </div>
      </div>
      <div id="actions" className="actions orders">
        {orders}
      </div>
    </div>
  );
}

// City level-up choice, overlaid on the board like the mobile popup.
export function RewardOverlay() {
  const game = useGame();
  const pending = game.pendingReward();
  if (!pending || game.busy || game.animating) return <div id="reward" className="reward hidden" />;
  const v = game.data!.observation.visible_state,
    city = v.cities.find((c) => c.tile === pending.command.target)!,
    choices = game.legal.filter((a) => a.command.kind === 9 && a.command.target === pending.command.target);
  return (
    <div id="reward" className="reward" role="dialog" aria-modal="true" aria-label="City reward">
      <div id="reward-card" className="reward-card">
        <div className="eyebrow">
          {cityName(city.tile)} · Level {city.level}
        </div>
        <h2>
          {cityName(city.tile)} has grown to level {city.level}!
        </h2>
        <p>Choose your reward</p>
        <div className="reward-choices">
          {choices.map((a) => (
            <RewardChoice key={a.action_id} action={a} />
          ))}
        </div>
      </div>
    </div>
  );
}

function RewardChoice({ action }: { action: LegalAction }) {
  const game = useGame();
  const info = REWARDS[action.command.parameter] || { text: "", sprite: "star" };
  return (
    <button className="reward-choice" data-action-id={action.action_id} onClick={() => game.takeAction(action)}>
      <span className="round-icon reward-icon">
        <Art art={info} />
      </span>
      <strong>{actLabel(game.meta!, action)}</strong>
      <small>{info.text}</small>
    </button>
  );
}
