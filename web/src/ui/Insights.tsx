// Replay side panel: economy over time (small multiples, one per measure) and a military power map of the
// current frame. Seats are coloured as on the replay board: blue, red, green, gold.
import { useMemo, useState } from "react";
import type { Meta, Observation, SeriesPoint } from "../api/types.ts";
import { useGame } from "./useGame.ts";

const TEAM: Record<number, string> = { 1: "#2f95e0", 2: "#e2483d", 3: "#3fae5a", 4: "#e0a92f" };
const NAME: Record<number, string> = { 1: "Blue", 2: "Red", 3: "Green", 4: "Gold" };
const METRICS: { key: keyof SeriesPoint; label: string; unit: string }[] = [
  { key: "stars", label: "Stars banked", unit: "★" },
  { key: "income", label: "Production", unit: "★/turn" },
  { key: "army", label: "Army value", unit: "★" },
  { key: "cities", label: "Cities", unit: "" },
];
const W = 272,
  H = 62,
  PAD = { l: 28, r: 6, t: 4, b: 4 };

// The sample in effect at a turn: the latest turn start at or before it.
function at(points: SeriesPoint[], turn: number) {
  let found: SeriesPoint | undefined;
  for (const p of points) if (p.turn <= turn && !p.final) found = p;
  return found ?? points[0];
}

// The closing frame plots half a turn after the last turn start.
const xTurn = (p: SeriesPoint) => p.turn + (p.final ? 0.5 : 0);

const fmt = (v: number) => (v >= 100 ? Math.round(v).toString() : (Math.round(v * 10) / 10).toString());

function nice(max: number) {
  if (max <= 0) return 1;
  const step = 10 ** Math.floor(Math.log10(max));
  return Math.ceil(max / step) * step;
}

function Chart({ metric, seats, players, maxTurn, turn, hover, setHover, seek }: {
  metric: (typeof METRICS)[number];
  seats: number[];
  players: Record<string, SeriesPoint[]>;
  maxTurn: number;
  turn: number;
  hover: number | null;
  setHover: (t: number | null) => void;
  seek: (t: number) => void;
}) {
  const top = nice(Math.max(1, ...seats.flatMap((p) => (players[p] ?? []).map((s) => Number(s[metric.key]) || 0))));
  const x = (t: number) => PAD.l + (t / Math.max(1, maxTurn)) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - v / top) * (H - PAD.t - PAD.b);
  const shown = hover ?? turn;
  const turnAt = (clientX: number, el: SVGSVGElement) => {
    const r = el.getBoundingClientRect();
    const t = Math.round((((clientX - r.left) * (W / r.width) - PAD.l) / (W - PAD.l - PAD.r)) * maxTurn);
    return Math.max(0, Math.min(Math.floor(maxTurn), t));
  };
  return (
    <figure className="insight-chart">
      <figcaption>
        <span>
          {metric.label}
          {metric.unit && <small> {metric.unit}</small>}
        </span>
        <span className="insight-values">
          {seats.map((p) => {
            const s = players[p] && at(players[p], shown);
            return (
              <span key={p}>
                <i style={{ background: TEAM[p] }} />
                {s ? fmt(Number(s[metric.key]) || 0) : "–"}
              </span>
            );
          })}
        </span>
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${metric.label} per turn, ${seats.map((p) => `player ${p} (${NAME[p].toLowerCase()})`).join(", ")}`}
        onPointerMove={(e) => setHover(turnAt(e.clientX, e.currentTarget))}
        onPointerLeave={() => setHover(null)}
        onClick={(e) => seek(turnAt(e.clientX, e.currentTarget))}
      >
        {[0, top].map((v) => (
          <g key={v}>
            <line className="insight-grid" x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} />
            <text className="insight-tick" x={PAD.l - 4} y={y(v) + 3}>
              {fmt(v)}
            </text>
          </g>
        ))}
        <line className="insight-now" x1={x(turn)} x2={x(turn)} y1={PAD.t} y2={H - PAD.b} />
        {seats.map((p) => {
          const pts = (players[p] ?? []).map((s) => `${x(xTurn(s))},${y(Number(s[metric.key]) || 0)}`).join(" ");
          return <polyline key={p} points={pts} fill="none" stroke={TEAM[p]} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />;
        })}
        {hover !== null && (
          <>
            <line className="insight-hover" x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={H - PAD.b} />
            {seats.map((p) => {
              const s = players[p] && at(players[p], hover);
              return s ? <circle key={p} cx={x(hover)} cy={y(Number(s[metric.key]) || 0)} r={4} fill={TEAM[p]} className="insight-dot" /> : null;
            })}
          </>
        )}
      </svg>
    </figure>
  );
}

// Net threat on every tile: each unit projects its army value (star cost x health share) over the tiles it
// could reach and hit next turn (movement + range, Chebyshev distance), fading with distance.
function influence(meta: Meta, obs: Observation, seats: number[]) {
  const v = obs.visible_state,
    n = v.board_size;
  const rules = new Map(meta.rules.units.map((u) => [u.id, u]));
  const power: Record<number, Float32Array> = {},
    army: Record<number, number> = {};
  for (const p of seats) {
    power[p] = new Float32Array(n * n);
    army[p] = 0;
  }
  for (const u of v.units) {
    const r = rules.get(u.kind);
    if (!r || r.cost <= 0 || !(u.owner in power)) continue;
    const value = (r.cost * u.health) / (r.health + (u.veteran ? 5 : 0));
    army[u.owner] += value;
    const reach = r.movement + r.range,
      ux = u.tile % n,
      uy = Math.floor(u.tile / n);
    for (let dy = -reach; dy <= reach; dy++)
      for (let dx = -reach; dx <= reach; dx++) {
        const tx = ux + dx,
          ty = uy + dy;
        if (tx < 0 || ty < 0 || tx >= n || ty >= n) continue;
        power[u.owner][ty * n + tx] += value * (1 - Math.max(Math.abs(dx), Math.abs(dy)) / (reach + 1));
      }
  }
  return { power, army, n };
}

function mix(a: [number, number, number], b: [number, number, number], t: number) {
  return `rgb(${a.map((c, i) => Math.round(c + (b[i] - c) * t)).join(",")})`;
}
const RGB = { 1: [47, 149, 224], 2: [226, 72, 61], 3: [63, 174, 90], 4: [224, 169, 47] } as Record<
  number,
  [number, number, number]
>;
const LAND: [number, number, number] = [58, 63, 68],
  WATER: [number, number, number] = [30, 34, 38];

// Each tile takes the colour of the strongest side there, by its margin over the runner-up.
function PowerMap({ meta, obs, perspective, seats }: { meta: Meta; obs: Observation; perspective: string; seats: number[] }) {
  const { power, army, n } = useMemo(() => influence(meta, obs, seats), [meta, obs, seats]);
  const v = obs.visible_state;
  const hw = W / (2 * n),
    hh = hw / 2,
    h = n * 2 * hh;
  const lead = (i: number) => {
    const ranked = seats.map((p) => [p, power[p][i]]).sort((a, b) => b[1] - a[1]);
    return { side: ranked[0][0], margin: ranked[0][1] - (ranked[1]?.[1] ?? 0) };
  };
  let scale = 8;
  for (let i = 0; i < n * n; i++) scale = Math.max(scale, lead(i).margin);
  const tiles = v.tiles.map((t) => {
    const col = t.tile % n,
      row = Math.floor(t.tile / n);
    const cx = W / 2 + (col - row) * hw,
      cy = (col + row) * hh + hh;
    const { side, margin } = lead(t.tile);
    const base = t.terrain === 1 || t.terrain === 2 ? WATER : LAND;
    const fill = !t.known
      ? "#101214"
      : margin === 0
        ? mix(base, base, 0)
        : mix(base, RGB[side], Math.min(1, Math.sqrt(margin / scale)));
    return (
      <polygon key={t.tile} points={`${cx},${cy - hh} ${cx + hw},${cy} ${cx},${cy + hh} ${cx - hw},${cy}`} fill={fill}>
        <title>
          {t.known
            ? seats.map((p) => `${NAME[p]} threat ${fmt(power[p][t.tile])}`).join(" · ")
            : "Unexplored from this view"}
        </title>
      </polygon>
    );
  });
  const pos = (tile: number) => {
    const col = tile % n,
      row = Math.floor(tile / n);
    return { x: W / 2 + (col - row) * hw, y: (col + row) * hh + hh };
  };
  return (
    <figure className="insight-map">
      <figcaption>
        <span>Military power</span>
        <span className="insight-values">
          {seats.map((p) => (
            <span key={p}>
              <i style={{ background: TEAM[p] }} />
              {fmt(army[p])} ★
            </span>
          ))}
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${h}`} role="img" aria-label="Military power map: tiles tinted by which side can strike them next turn">
        {tiles}
        {v.cities
          .filter((c) => c.owner)
          .map((c) => {
            const p = pos(c.tile);
            return <rect key={"c" + c.tile} x={p.x - 3} y={p.y - 3} width={6} height={6} className="insight-city" stroke={TEAM[c.owner]} />;
          })}
        {v.units
          .filter((u) => seats.includes(u.owner))
          .map((u) => {
            const p = pos(u.tile);
            return <circle key={"u" + u.tile} cx={p.x} cy={p.y} r={1.8} className="insight-unit" fill={TEAM[u.owner]} />;
          })}
      </svg>
      {seats.length === 2 ? (
        <div className="insight-scale">
          <span>Blue stronger</span>
          <i />
          <span>Red stronger</span>
        </div>
      ) : null}
      <small className="insight-note">
        Tiles each army can reach and strike next turn, weighted by unit value and health.
        {perspective !== "all" && " Fog applies: only units this player can see."}
      </small>
    </figure>
  );
}

export function ReplayInsights() {
  const game = useGame();
  const [hover, setHover] = useState<number | null>(null);
  const d = game.data,
    s = game.series;
  if (!d || d.mode !== "replay" || !s?.available || !s.players || !game.meta) return null;
  if (!game.insightsOpen)
    return (
      <button className="insights-tab" onClick={() => game.setInsightsOpen(true)}>
        Insights
      </button>
    );
  const players = s.players;
  const seats = Object.keys(players).map(Number).sort((a, b) => a - b);
  // Four-seat recordings carry no production figure; skip measures with no data.
  const metrics = METRICS.filter((m) => seats.some((p) => players[p]?.some((pt) => pt[m.key] != null)));
  const maxTurn = Math.max(1, ...seats.flatMap((p) => (players[p] ?? []).map(xTurn)));
  const turn = d.observation.visible_state.turn;
  const seek = (t: number) => {
    const target = at(players[seats[0]], t);
    if (target) game.seek(target.frame);
  };
  return (
    <aside className="insights" aria-label="Match insights">
      <header>
        <div>
          <div className="eyebrow">Match insights</div>
          <div className="insight-legend">
            {seats.map((p) => (
              <span key={p}>
                <i style={{ background: TEAM[p] }} />
                Player {p}
              </span>
            ))}
          </div>
        </div>
        <button className="insights-close" aria-label="Hide insights" onClick={() => game.setInsightsOpen(false)}>
          ×
        </button>
      </header>
      <div className="insight-sub">
        {hover !== null ? `Turn ${hover}` : `Turn ${turn}`} · start of each player's turn · click a chart to jump
      </div>
      {metrics.map((m) => (
        <Chart key={m.key} metric={m} seats={seats} players={players} maxTurn={maxTurn} turn={turn} hover={hover} setHover={setHover} seek={seek} />
      ))}
      <PowerMap meta={game.meta} obs={d.observation} perspective={d.perspective} seats={seats} />
    </aside>
  );
}
