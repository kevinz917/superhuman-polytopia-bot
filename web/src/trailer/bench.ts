// Closing card: the Polytopia Bench leaderboard as the website draws it (kevin-website, posts/polytopia-bench):
// a white card with one row per model, a gridded 0-100 track, a green bar for the mean of its five bots growing in
// row after row, five dots for the five bots fading in after, and mean ± SD on the right. Once it is built the
// selection steps down the top rows as if someone clicked through them, while the card slowly pushes in. Real numbers
// from the site's data (bench.json). Like the rest of the scene it is a pure function of t.

interface BenchRow {
  label: string;
  logo: string;
  mean: number;
  sd: number;
  bots: number[];
}

const INK = "#000000",
  SECONDARY = "#6b6b6b",
  RULE = "#eaeaea",
  GRID = "#e9e9e4",
  ACCENT_DEEP = "#178a39",
  ACCENT = "#22a447",
  ACCENT_SOFT = "#a6e2b4",
  ACCENT_TINT = "#edf8ef",
  FONT = '"Instrument Sans", "SF Pro Text", "Helvetica Neue", sans-serif';
// The site's layout at about 2.2x: name column, track, score column.
const CW = 1360,
  CH = 900,
  PAD = 44,
  NAME = 330,
  SCORE = 190,
  GAP = 26,
  ROW = 54,
  TRACK = 40,
  BAR = 24,
  DOT = 15,
  DOT_TOP = [2, 9, 5, 11, 7]; // the site's dot offsets in its 20 px track
// Bars grow over 1 s, 0.1 s apart; each row's dots fade in 0.35 s from 0.8 s after its bar starts. Then the selection
// moves to each row in TOUR, and the card scales up by PUSH over HOLD seconds.
const RISE = 0.45,
  BARS_AT = 0.35,
  GROW = 1.0,
  STAGGER = 0.1,
  TOUR = [
    [0, 0],
    [4.0, 1],
    [5.0, 2],
    [6.0, 3],
  ],
  PUSH = 0.04,
  HOLD = 7.7;
// For the sound track: when, after the card starts to rise, each bar starts and the selection moves.
export const BENCH_TIMES = { rise: RISE, bars: BARS_AT, stagger: STAGGER, grow: GROW, tour: TOUR.slice(1).map(([t]) => t) };

const clamp = (x: number, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const easeOut = (u: number) => 1 - (1 - clamp(u)) ** 3.5; // close to the site's cubic-bezier(.2, .7, .2, 1)
const mix = (a: string, b: string, u: number) => {
  const c = (h: string, k: number) => parseInt(h.slice(1 + 2 * k, 3 + 2 * k), 16);
  return `rgb(${[0, 1, 2].map((k) => Math.round(c(a, k) + (c(b, k) - c(a, k)) * u)).join(",")})`;
};
// Which row is selected at `age`, as a float that glides between rows.
const selected = (age: number) => {
  let at = 0;
  for (const [t, row] of TOUR) at += (row - at) * easeOut((age - t) / 0.35);
  return at;
};

export class Bench {
  canvas = document.createElement("canvas");
  private ctx: CanvasRenderingContext2D;
  private W: number;
  private H: number;
  private rows: BenchRow[] = [];
  private logos = new Map<string, HTMLImageElement>();

  constructor(W: number, H: number) {
    this.W = W;
    this.H = H;
    this.canvas.width = W;
    this.canvas.height = H;
    this.canvas.className = "layer";
    this.ctx = this.canvas.getContext("2d")!;
  }

  async load() {
    this.rows = (await (await fetch("/trailer/bench.json")).json()).models;
    await Promise.all(
      [...new Set(this.rows.map((r) => r.logo))].map(async (name) => {
        // The logos are white for the black scene; on the white card the white parts turn black.
        const svg = (await (await fetch(`/trailer/logos/${name}.svg`)).text()).replaceAll("#ffffff", INK),
          img = new Image();
        img.src = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
        await img.decode();
        this.logos.set(name, img);
      }),
    );
  }

  // age: seconds since the card started to rise (negative: not shown).
  draw(age: number) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.W, this.H);
    if (age <= 0) return;
    const rise = easeOut(age / RISE),
      x0 = (this.W - CW) / 2,
      y0 = (this.H - CH) / 2 + 40 * (1 - rise);
    ctx.save();
    ctx.globalAlpha = clamp(age / (RISE * 0.7));
    const push = 1 + PUSH * clamp(age / HOLD);
    ctx.translate(this.W / 2, this.H / 2);
    ctx.scale(push, push);
    ctx.translate(-this.W / 2, -this.H / 2);
    ctx.translate(x0, y0);

    ctx.shadowColor = "rgba(0,0,0,0.5)";
    ctx.shadowBlur = 60;
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.roundRect(0, 0, CW, CH, 18);
    ctx.fill();
    ctx.shadowColor = "transparent";
    ctx.strokeStyle = RULE;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Header.
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = INK;
    ctx.font = `600 46px ${FONT}`;
    ctx.textAlign = "left";
    ctx.fillText("Polytopia Bench", PAD, PAD + 44);
    ctx.fillStyle = SECONDARY;
    ctx.font = `400 24px ${FONT}`;
    ctx.textAlign = "right";
    ctx.fillText("12 models · 40 minutes each · 21,600 games", CW - PAD, PAD + 40);

    const trackX = PAD + NAME + GAP,
      trackW = CW - PAD - SCORE - GAP - trackX,
      head = PAD + 104,
      sel = selected(age);
    ctx.font = `400 21px ${FONT}`;
    ctx.textAlign = "left";
    ctx.fillText("Panel points /100", PAD, head);
    [0, 25, 50, 75, 100].forEach((n, i) => {
      ctx.textAlign = i === 0 ? "left" : i === 4 ? "right" : "center";
      ctx.fillText(String(n), trackX + (trackW * n) / 100, head);
    });
    ctx.textAlign = "right";
    ctx.fillText("Mean ± SD", CW - PAD, head);

    this.rows.forEach((r, i) => {
      const top = head + 16 + i * ROW,
        mid = top + ROW / 2,
        t0 = BARS_AT + i * STAGGER,
        grow = easeOut((age - t0) / GROW),
        dots = clamp((age - t0 - 0.8) / 0.35),
        on = clamp(1 - Math.abs(sel - i)),
        first = on > 0.5;
      // The selected row (the site opens on the top model): tinted, full-green bar, bold name. The tint glides.
      if (i === 0) {
        ctx.fillStyle = ACCENT_TINT;
        ctx.beginPath();
        ctx.roundRect(PAD - 12, top + 2 + sel * ROW, CW - 2 * PAD + 24, ROW - 4, 10);
        ctx.fill();
      }
      const img = this.logos.get(r.logo);
      if (img) ctx.drawImage(img, PAD, mid - 17, 34, 34);
      ctx.fillStyle = mix(INK, ACCENT_DEEP, on);
      ctx.font = `${first ? 600 : 400} 27px ${FONT}`;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(r.label, PAD + 48, mid + 1);

      const ty = mid - TRACK / 2;
      ctx.fillStyle = GRID;
      for (let q = 0; q <= 4; q++) ctx.fillRect(trackX + (trackW * q) / 4 - (q === 4 ? 2 : 0), ty, 2, TRACK);
      const w = (trackW * r.mean * grow) / 100;
      if (w > 0.5) {
        ctx.fillStyle = mix(ACCENT_SOFT, ACCENT, on);
        ctx.beginPath();
        ctx.roundRect(trackX, mid - BAR / 2, w, BAR, [0, BAR / 2, BAR / 2, 0]);
        ctx.fill();
      }
      if (dots > 0) {
        ctx.globalAlpha *= dots;
        r.bots.forEach((score, k) => {
          const cx = trackX + (trackW * score) / 100,
            cy = ty + DOT_TOP[k] * 2 + DOT / 2;
          ctx.beginPath();
          ctx.arc(cx, cy, DOT / 2 + 2, 0, Math.PI * 2);
          ctx.fillStyle = "#fff";
          ctx.fill();
          ctx.beginPath();
          ctx.arc(cx, cy, DOT / 2 - 1.5, 0, Math.PI * 2);
          ctx.lineWidth = 3;
          ctx.strokeStyle = first ? "#0f6b2c" : ACCENT_DEEP;
          ctx.stroke();
        });
        ctx.globalAlpha /= dots;
      }

      // The score counts up with its bar.
      ctx.textAlign = "right";
      ctx.fillStyle = SECONDARY;
      ctx.font = `400 20px ${FONT}`;
      const sd = `± ${r.sd.toFixed(1)}`;
      ctx.fillText(sd, CW - PAD, mid + 1);
      const sdW = ctx.measureText(sd).width;
      ctx.fillStyle = INK;
      ctx.font = `600 29px ${FONT}`;
      ctx.fillText((r.mean * grow).toFixed(1), CW - PAD - sdW - 10, mid + 1);
    });

    // Key.
    const key = head + 16 + this.rows.length * ROW + 40;
    ctx.font = `400 20px ${FONT}`;
    ctx.fillStyle = SECONDARY;
    ctx.textBaseline = "middle";
    ctx.beginPath();
    ctx.arc(PAD + 8, key, 7, 0, Math.PI * 2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = ACCENT_DEEP;
    ctx.stroke();
    ctx.textAlign = "left";
    ctx.fillText("Five dots = five independently authored bots", PAD + 26, key + 1);
    ctx.textAlign = "right";
    ctx.fillText("Win = 1 · draw = ½ · loss = 0", CW - PAD, key + 1);
    ctx.restore();
  }
}
