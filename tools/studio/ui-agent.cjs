/** Screenshot/click adapter: models operate the actual public-facing UI.
 * Usage: node tools/studio/ui-agent.cjs /absolute/path/to/model-driver.cjs
 * Driver exports async function({ image, controls, instruction }) ->
 *   {type:'click',x,y} | {type:'key',key:'Escape'} | {type:'wait'} | {type:'stop'}.
 * `image` is a PNG Buffer. The driver handles its own provider and credentials.
 * No network model calls are made by this adapter itself.
 */
const path = require("node:path");
const { chromium } = require(
  process.env.PLAYWRIGHT_MODULE ||
    "../../artifacts/browser/node_modules/playwright",
);
(async () => {
  if (!process.argv[2])
    throw Error(
      "Supply a model-driver module path. See docs/replay-studio.md.",
    );
  const decide = require(path.resolve(process.argv[2]));
  const browser = await chromium.launch({
    headless: process.env.HEADLESS === "1",
    executablePath:
      process.env.CHROME_PATH ||
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  await page.goto(process.env.STUDIO_URL || "http://127.0.0.1:8765");
  await page.waitForFunction(() => window.bardur);
  await page.getByRole("button", { name: "Menu", exact: true }).click();
  await page.getByRole("button", { name: /Connect a model/ }).click();
  await page.waitForFunction(() => window.bardur.game?.kind === "games");
  for (let step = 0; step < Number(process.env.MAX_UI_STEPS || 1000); step++) {
    await page.waitForFunction(() => !window.bardur.busy);
    const state = await page.evaluate(() => window.bardur.observe());
    if (state.terminated || state.truncated) break;
    if (!state.your_turn) {
      await page.waitForTimeout(300);
      continue;
    }
    const controls = await page.evaluate(() => window.bardur.controls());
    const next = await decide({
      image: await page.screenshot(),
      controls,
      instruction:
        "Play Bardur. Select a unit then a highlighted destination, use city actions and technology, and end your turn. Return a click, key, wait, or stop.",
    });
    if (next.type === "stop") break;
    if (next.type === "click")
      await page.mouse.click(Number(next.x), Number(next.y));
    else if (next.type === "key") await page.keyboard.press(next.key);
    else if (next.type !== "wait") throw Error("Unsupported UI action");
    await page.waitForTimeout(350);
  }
  // Keep the visible browser available for inspection until the user closes it.
  console.log(
    "UI driver finished.",
    await page.evaluate(() => window.bardur.game),
  );
  if (process.env.HEADLESS === "1") await browser.close();
  else console.log("Close the browser to end this process.");
})();
