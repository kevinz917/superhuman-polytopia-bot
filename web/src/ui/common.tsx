// Shared hooks and small building blocks for the interface.
import type { LegalAction } from "../api/types.ts";
import { spriteURL } from "../board/assets.ts";
import { actionArt, actLabel, cost, type Art as ArtSpec } from "../game/vocab.ts";
import { useGame } from "./useGame.ts";

// The drawn icon, else the piece's placeholder shape.
export function Art({ art, className }: { art: ArtSpec; className?: string }) {
  if (art.icon) return <span className={"art " + (className || "")} dangerouslySetInnerHTML={{ __html: art.icon }} />;
  return <img className={className} src={spriteURL(art.sprite || "star")} alt="" />;
}

export function Heading({ eyebrow, title, text }: { eyebrow: string; title: string; text?: string }) {
  return (
    <>
      <div className="eyebrow">{eyebrow}</div>
      <h2>{title}</h2>
      {text && <p>{text}</p>}
    </>
  );
}

// Mobile-style circular order button: icon, star cost badge, label.
export function RoundAction({ action, label, enabled = true }: { action: LegalAction; label?: string; enabled?: boolean }) {
  const game = useGame();
  const meta = game.meta!,
    obs = game.data!.observation;
  const price = cost(meta, obs, action);
  return (
    <button
      className="round-action"
      disabled={!enabled}
      data-action-id={enabled ? action.action_id : undefined}
      onClick={() => enabled && game.takeAction(action)}
    >
      <span className="round-icon order-icon">
        <Art art={actionArt(meta, obs, action)} />
        {price !== null && <small className="order-cost">{price}</small>}
      </span>
      <span>{label ?? actLabel(meta, action)}</span>
    </button>
  );
}

// Plain list button used in menus that list every legal action.
export function ActionButton({ action, extra }: { action: LegalAction; extra?: string[] }) {
  const game = useGame();
  const price = cost(game.meta!, game.data!.observation, action);
  return (
    <button
      className={action.command.kind === 11 ? "action-button danger" : "action-button"}
      data-action-id={action.action_id}
      onClick={() => game.takeAction(action)}
    >
      {actLabel(game.meta!, action)}
      {price !== null && <small>{price} ★</small>}
      {extra?.map((e) => <small key={e}>{e}</small>)}
    </button>
  );
}
