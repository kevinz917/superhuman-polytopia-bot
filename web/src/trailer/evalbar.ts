// Eval bar: a tug-of-war above the board between the two sides, in the team colours, the right player's colour on the
// right, with a win-% and the logo of the model playing that side at each end. It is a visualisation, not a measurement: its swings land on the real in-game score
// changes but their sizes tell a story (an even start that follows the real swings, the loser pulling ahead mid-game,
// the winner's comeback), each easing in fast with an overshoot. It snaps to the winner at the mate and turns over
// with the board onto the next game. Like the rest of the scene it is a pure function of t.
import type { Team } from "../board/geometry.ts";

export interface BarGame {
  start: number;
  mate: number;
  winner: number;
  slot: Record<number, number>; // player -> hand (1 right, 2 left)
  scores: [number, number]; // both players' scores at the window start
  steps: { t: number; scores: [number, number] }[]; // every change, in time order
  logos: Record<number, string>; // player -> logo name in /trailer/logos
}

const EASE = 0.15,
  MATE = 0.24,
  PULSE = 0.3,
  GROUP = 0.09; // score changes closer than this make one swing
const BW = 1000,
  BH = 28,
  Y = 78,
  GAP = 34,
  FONT = 60,
  LOGO = 100,
  LOGO_AT = 250; // logo centre, out from the bar's end

const clamp = (x: number, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const hash = (a: number, b = 0) => {
  const s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453;
  return s - Math.floor(s);
};
// Back-out: reaches the target in u = 1 after overshooting it by about 10%.
const back = (u: number) => {
  if (u <= 0) return 0;
  if (u >= 1) return 1;
  const c = 1.7,
    v = u - 1;
  return 1 + (c + 1) * v * v * v + c * v * v;
};
const css = (c: number) => `#${c.toString(16).padStart(6, "0")}`;
const rightPlayer = (g: BarGame) => (g.slot[1] === 1 ? 1 : 2);

// The right player's share, as a starting value and the swings that move it.
interface Story {
  from: number;
  swings: { t: number; to: number; dur: number }[];
}
const stories = new Map<BarGame, Story>();
function story(g: BarGame): Story {
  let st = stories.get(g);
  if (st) return st;
  const w = g.winner - 1,
    winnerShare = (s: [number, number]) => s[w] / (s[0] + s[1] || 1),
    right = (x: number) => (g.winner === rightPlayer(g) ? x : 1 - x),
    bound = (x: number) => 0.5 + 0.4 * Math.tanh((x - 0.5) / 0.4);
  const groups: { t: number; d: number }[] = [];
  let prev = winnerShare(g.scores);
  for (const s of g.steps) {
    const v = winnerShare(s.scores),
      last = groups[groups.length - 1];
    if (last && s.t - last.t < GROUP) last.d += v - prev;
    else groups.push({ t: s.t, d: v - prev });
    prev = v;
  }
  // The winner's side, unbounded; bound() keeps the bar off its ends until the mate.
  let x = 0.5 + (hash(g.start, 1) - 0.5) * 0.14;
  st = { from: right(bound(x)), swings: [] };
  groups.forEach((gr, i) => {
    const p = (gr.t - g.start) / (g.mate - g.start),
      r = hash(i, g.start);
    if (p < 0.38) x += (Math.sign(gr.d) || (r < 0.5 ? -1 : 1)) * clamp(Math.abs(gr.d) * 5, 0.05, 0.16) * (0.8 + 0.5 * r);
    else if (p < 0.75) x = r < 0.2 ? x + 0.06 : Math.min(x, 0.5) - 0.07 - 0.09 * r; // the loser pulls ahead
    else x += 0.14 + 0.1 * r; // the comeback
    st!.swings.push({ t: gr.t, to: right(bound(x)), dur: EASE });
  });
  st.swings.push({ t: g.mate, to: right(1), dur: MATE });
  stories.set(g, st);
  return st;
}

// Every swing adds its own eased step, so swings that land while the bar is still moving chain smoothly.
function share(g: BarGame, t: number) {
  const { from, swings } = story(g);
  let prev = from,
    x = from;
  for (const s of swings) {
    if (s.t > t) break;
    x += (s.to - prev) * back((t - s.t) / s.dur);
    prev = s.to;
  }
  // A faint live wobble between swings, gone at the mate.
  x += 0.012 * (Math.sin(t * 9.3 + g.start * 3) + 0.6 * Math.sin(t * 23.1 + g.start)) * clamp((g.mate - t) / 0.1);
  return clamp(x);
}

// A glow on the split as it moves, stronger for bigger swings.
function pulse(g: BarGame, t: number) {
  const { from, swings } = story(g);
  let p = 0,
    prev = from;
  for (const s of swings) {
    if (s.t > t) break;
    const age = t - s.t,
      life = PULSE * (s.dur / EASE);
    if (age < life) p = Math.max(p, (0.35 + 0.65 * clamp(Math.abs(s.to - prev) / 0.12)) * (1 - age / life));
    prev = s.to;
  }
  return p;
}

export class EvalBar {
  canvas = document.createElement("canvas");
  private ctx: CanvasRenderingContext2D;
  private W: number;
  private team: Record<number, Team>;
  private logos = new Map<string, HTMLImageElement>();

  constructor(W: number, H: number, team: Record<number, Team>) {
    this.W = W;
    this.team = team;
    this.canvas.width = W;
    this.canvas.height = H;
    this.canvas.className = "layer";
    this.ctx = this.canvas.getContext("2d")!;
  }

  async load(names: string[]) {
    await Promise.all(
      [...new Set(names)].map(async (name) => {
        const img = new Image();
        img.src = `/trailer/logos/${name}.svg`;
        await img.decode();
        this.logos.set(name, img);
      }),
    );
  }

  // squash: the bar's width as it turns over with the board (1 flat, 0 edge-on).
  draw(g: BarGame, t: number, squash = 1) {
    const ctx = this.ctx,
      cx = this.W / 2,
      x0 = cx - BW / 2,
      R = rightPlayer(g),
      L = 3 - R,
      x = share(g, t),
      split = x0 + BW * (1 - x),
      glow = pulse(g, t);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (squash <= 0.01) return;
    ctx.save();
    ctx.translate(cx, 0);
    ctx.scale(squash, 1);
    ctx.translate(-cx, 0);

    ctx.save();
    ctx.beginPath();
    ctx.roundRect(x0, Y - BH / 2, BW, BH, BH / 2);
    ctx.clip();
    const side = (a: number, b: number, team: Team) => {
      const grad = ctx.createLinearGradient(0, Y - BH / 2, 0, Y + BH / 2);
      grad.addColorStop(0, css(team.light));
      grad.addColorStop(0.45, css(team.main));
      grad.addColorStop(1, css(team.dark));
      ctx.fillStyle = grad;
      ctx.fillRect(a, Y - BH / 2, b - a, BH);
    };
    side(x0, split, this.team[L]);
    side(split, x0 + BW, this.team[R]);
    ctx.restore();

    // The halfway mark.
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.fillRect(cx - 1.5, Y - BH / 2 - 15, 3, 9);
    ctx.fillRect(cx - 1.5, Y + BH / 2 + 6, 3, 9);

    // The split, gone once one side has the whole bar.
    ctx.save();
    ctx.globalAlpha = clamp(Math.min(x, 1 - x) / 0.03);
    ctx.shadowColor = "rgba(255,255,255,0.9)";
    ctx.shadowBlur = 8 + 30 * glow;
    ctx.fillStyle = "#fff";
    const hh = BH / 2 + 12 + 6 * glow;
    ctx.fillRect(split - 2.5, Y - hh, 5, 2 * hh);
    ctx.restore();

    ctx.font = `600 ${FONT}px "Josefin Sans", "Avenir Next", sans-serif`;
    ctx.textBaseline = "middle";
    ctx.fillStyle = css(this.team[L].light);
    ctx.textAlign = "right";
    ctx.fillText(`${Math.round((1 - x) * 100)}%`, x0 - GAP, Y + 5);
    ctx.fillStyle = css(this.team[R].light);
    ctx.textAlign = "left";
    ctx.fillText(`${Math.round(x * 100)}%`, x0 + BW + GAP, Y + 5);
    const logo = (player: number, cx: number) => {
      const img = this.logos.get(g.logos[player]);
      if (img) ctx.drawImage(img, cx - LOGO / 2, Y - LOGO / 2, LOGO, LOGO);
    };
    logo(L, x0 - LOGO_AT);
    logo(R, x0 + BW + LOGO_AT);
    ctx.restore();
  }
}
