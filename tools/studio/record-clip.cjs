// Record a Studio replay as timestamped screenshots.
// node tools/studio/record-clip.cjs <replay_id> <out_dir> [width] [height] [speed] [maxSeconds]
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE);

const [id, out, W = "960", H = "960", SPEED = "8", MAX = "600"] = process.argv.slice(2);
fs.mkdirSync(out, { recursive: true });

(async () => {
  const browser = await chromium.launch({ headless: process.env.HEADED !== "1", args: ["--use-angle=metal", "--enable-gpu"] });
  const page = await browser.newPage({ viewport: { width: +W, height: +H }, deviceScaleFactor: 2 });
  await page.addInitScript(([rid, anim]) => {
    localStorage.setItem("bardur-current", JSON.stringify({ kind: "replays", id: rid }));
    localStorage.setItem("bardur-animations", anim);
  }, [id, process.env.ANIM || "off"]);
  await page.goto(process.env.STUDIO_URL || "http://127.0.0.1:8765");
  await page.waitForFunction(() => window.bardur && document.querySelector('input[type="range"]'));
  await page.waitForTimeout(2500);
  await page.selectOption("#speed", SPEED);
  await page.click(".play-button");
  await page.evaluate(() => document.activeElement && document.activeElement.blur());
  await page.addStyleTag({
    content: `.replay-dock, #status, nav.game-nav, header.topbar, footer, #app > footer, .board-tools, .status, .game-info, .toast { opacity: 0 !important; pointer-events: none !important; }`,
  });
  const pos = () =>
    page.evaluate(() => {
      const r = document.querySelector('input[type="range"]');
      return [Number(r.value), Number(r.max)];
    });
  const t0 = Date.now();
  const log = [];
  let n = 0;
  for (;;) {
    const t = (Date.now() - t0) / 1000;
    const file = path.join(out, `f${String(n).padStart(5, "0")}.png`);
    await page.screenshot({ path: file });
    const [v, max] = await pos();
    log.push({ file: path.basename(file), t, frame: v });
    n++;
    if (v >= max || t > +MAX) break;
  }
  // Hold the final position for a moment.
  await page.waitForTimeout(800);
  const t = (Date.now() - t0) / 1000;
  await page.screenshot({ path: path.join(out, `f${String(n).padStart(5, "0")}.png`) });
  log.push({ file: `f${String(n).padStart(5, "0")}.png`, t, frame: -1 });
  fs.writeFileSync(path.join(out, "log.json"), JSON.stringify(log));
  console.log("frames", n + 1, "seconds", t.toFixed(1));
  await browser.close();
})();
