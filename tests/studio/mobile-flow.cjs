// Phone-sized walkthrough of the mobile interaction flow on a fresh game.
// Saves mid-animation captures under artifacts/mobile-flow/ for visual review.
const { chromium } = require("../../artifacts/browser/node_modules/playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const URL = process.env.STUDIO_URL || "http://127.0.0.1:8765";
const OUT = process.env.FLOW_OUT || "artifacts/mobile-flow";
(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({
    executablePath:
      process.env.CHROME_PATH ||
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: true,
  });
  const page = await browser.newPage({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
    reducedMotion: "no-preference",
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const shot = async (name) => {
    const dims = await page.evaluate(() => ({
      wrap: document.getElementById("board-wrap").getBoundingClientRect().width,
      app: document.getElementById("app").getBoundingClientRect().width,
      scroll: document.documentElement.scrollWidth,
    }));
    assert.equal(dims.wrap, 390, `${name} board width ${JSON.stringify(dims)}`);
    assert.equal(dims.scroll, 390, `${name} page overflow ${JSON.stringify(dims)}`);
    await page.screenshot({ path: `${OUT}/${name}.png` });
  };
  const idle = () =>
    page.waitForFunction(() => !window.bardur.busy, null, { timeout: 120000 });
  const obs = () => page.evaluate(() => window.bardur.observe());
  const select = (tile) => page.evaluate((t) => window.bardur.selectTile(t), tile);
  await page.goto(URL);
  await page.waitForFunction(() => window.bardur?.game !== undefined);
  await page.getByRole("button", { name: "Menu", exact: true }).click();
  await page.getByRole("button", { name: /Play the bot/ }).click();
  await page.waitForFunction(
    () =>
      window.bardur?.game?.kind === "games" &&
      document.getElementById("status-text").textContent.includes("Your turn"),
    null,
    { timeout: 120000 },
  );
  await page.evaluate(() => {
    window.flowAnimations = [];
    document
      .getElementById("board")
      .addEventListener("bardur:animation-start", (e) =>
        window.flowAnimations.push(e.detail),
      );
  });
  let o = await obs();
  const v = o.visible_state;
  const capital = v.cities.find((c) => c.owner === o.player).tile;
  const warrior = v.units.find((u) => u.owner === o.player).tile;
  await shot("01-start");

  // Tap 1: the unit. Move markers render above terrain.
  await select(warrior);
  await page.waitForTimeout(450);
  assert.equal(await page.locator("#selection").getAttribute("data-layer"), "unit");
  await page.waitForFunction(() => getComputedStyle(document.querySelector(".game-nav")).opacity === "0", null, { timeout: 3000 });
  await shot("02-unit-selected");

  // Tap 2: the capital beneath. Territory comes from the server.
  await select(warrior);
  await page.waitForTimeout(450);
  assert.equal(await page.locator("#selection").getAttribute("data-layer"), "city");
  const frame = await page.evaluate((g) =>
    fetch(`/api/games/${g.id}`).then((r) => r.json()), await page.evaluate(() => window.bardur.game));
  const territory = frame.observation.hud.territory[String(capital)];
  assert(territory.includes(capital) && territory.length >= 9, "capital territory");
  assert(await page.locator(".round-action").count(), "city training buttons");
  await shot("03-city-territory");

  // Harvest until the capital levels up; capture the harvest and level-up beats.
  for (let i = 0; i < 4 && !(await page.locator("#reward:not(.hidden)").count()); i++) {
    o = await obs();
    const harvest = o.legal_actions.find((a) => a.command.kind === 5);
    if (!harvest) break;
    await select(harvest.command.target);
    await page.waitForTimeout(250);
    await page.locator(`#actions [data-action-id="${harvest.action_id}"]`).click();
    await page.waitForFunction(() => window.bardur.animating, null, { timeout: 30000 });
    await page.waitForTimeout(230);
    await shot(`04-harvest-${i}`);
    await page.waitForTimeout(500);
    await shot(`05-harvest-${i}-late`);
    await idle();
    await page.waitForTimeout(300);
  }
  const effects = (await page.evaluate(() => window.flowAnimations)).flatMap((a) => a.effects);
  assert(effects.includes("harvest"), "harvest animates");
  if (await page.locator("#reward:not(.hidden)").count()) {
    assert(effects.includes("levelup"), "level up animates");
    await shot("06-reward-overlay");
    await page.locator(".reward-choice", { hasText: "Explorer" }).click();
    await page.waitForFunction(() => window.bardur.animating, null, { timeout: 30000 });
    await page.waitForTimeout(900);
    await shot("07-explorer-walk");
    await page.waitForTimeout(1200);
    await shot("08-explorer-late");
    await idle();
  } else console.log("no level up reachable from this seed");

  // Move: hop along a path, then the tech tree detail flow.
  o = await obs();
  const move = o.legal_actions.find((a) => a.command.kind === 1);
  if (move) {
    await select(move.command.source);
    await page.waitForTimeout(350);
    await page.locator(`[data-tile="${move.command.target}"]`).click();
    await page.waitForFunction(() => window.bardur.animating, null, { timeout: 30000 });
    await page.waitForTimeout(120);
    await shot("09-move");
    await idle();
  }
  await page.getByRole("button", { name: "Tech Tree", exact: true }).click();
  const affordable = await page.locator(".tech.available").count();
  await page.locator(affordable ? ".tech.available" : ".tech:not(.researched)").first().click();
  await page.waitForTimeout(250);
  assert(await page.locator("#tech-detail:not(.hidden) h3").count(), "tech detail");
  if (affordable)
    assert(await page.locator("#tech-detail .research-button").count(), "research button");
  await shot("10-tech-detail");
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();

  // End the turn; the checkpoint plays, then train a unit to see the spawn pop.
  await page.getByRole("button", { name: "Next Turn", exact: true }).click();
  await page.waitForTimeout(1500);
  await shot("11-bot-turn");
  await page.waitForFunction(
    () => document.getElementById("status-text").textContent.includes("Your turn") && !window.bardur.busy,
    null,
    { timeout: 180000 },
  );
  o = await obs();
  const train = o.legal_actions.find((a) => a.command.kind === 4);
  if (train) {
    await select(train.command.source);
    await page.waitForTimeout(300);
    if ((await page.locator("#selection").getAttribute("data-layer")) !== "city")
      await select(train.command.source);
    await page.locator(`#actions [data-action-id="${train.action_id}"]`).click();
    await page.waitForFunction(() => window.bardur.animating, null, { timeout: 30000 });
    await page.waitForTimeout(140);
    await shot("12-train-pop");
    await idle();
    const last = await page.evaluate(() => window.flowAnimations.at(-1));
    assert(last.effects.includes("puff"), "training pops the new unit");
  }
  await shot("13-end");
  const game = await page.evaluate(() => window.bardur.game);
  await page.request.post(`${URL}/api/games/${game.id}/close`, {
    headers: { "X-Studio-Client": "1" },
    data: {},
  });
  assert.deepEqual(errors, []);
  console.log("PASS mobile flow", JSON.stringify(await page.evaluate(() => window.flowAnimations.map((a) => a.effects.join("+") || a.kind))));
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
