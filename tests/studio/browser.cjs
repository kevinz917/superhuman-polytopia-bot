const { chromium } = require("../../artifacts/browser/node_modules/playwright");
const assert = require("node:assert/strict");
const URL = process.env.STUDIO_URL || "http://127.0.0.1:8765";
(async () => {
  const browser = await chromium.launch({
    executablePath:
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: true,
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  const closeGame = async (id) => {
    await page.waitForFunction(() => !window.bardur.busy);
    const response = await page.request.post(
      `${URL}/api/games/${id}/close`,
      { headers: { "X-Studio-Client": "1" }, data: {} },
    );
    const { job_id } = await response.json();
    await until(async () => {
      const result = await (
        await page.request.get(`${URL}/api/jobs/${job_id}`)
      ).json();
      if (result.status === "error") throw Error(result.error);
      return result.status === "done";
    });
  };
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(URL);
  await page.waitForFunction(() => window.bardur?.game);
  await page.getByRole("button", { name: "Menu", exact: true }).click();
  await page.getByRole("button", { name: /Play the bot/ }).click();
  await page.waitForFunction(
    () =>
      window.bardur?.game?.kind === "games" &&
      document.getElementById("status-text").textContent.includes("Your turn"),
    { timeout: 120000 },
  );
  const until = async (fn) => {
    for (let i = 0; i < 300; i++) {
      if (await fn()) return;
      await page.waitForTimeout(200);
    }
    throw Error("condition timed out");
  };
  await page.evaluate(() => {
    window.liveAnimations = [];
    document
      .getElementById("board")
      .addEventListener("bardur:animation-start", (e) =>
        window.liveAnimations.push(e.detail),
      );
  });
  let obs = await page.evaluate(() => window.bardur.observe());
  let move = obs.legal_actions.find((a) => a.command.kind === 1);
  assert(move, "legal move");
  await page.locator(`[data-tile="${move.command.source}"]`).click();
  await page.locator(`[data-tile="${move.command.target}"]`).click();
  await until(async () => {
    const o = await page.evaluate(() => window.bardur.observe());
    return (
      o.state_version > obs.state_version &&
      (await page.locator("#end-turn").isEnabled())
    );
  });
  assert(
    (await page.evaluate(() => window.liveAnimations)).some(
      (e) => e.kind === "move",
    ),
    "Human move animates",
  );
  console.log("human move accepted and animated");
  await page.getByRole("button", { name: "Next Turn", exact: true }).click();
  await until(async () => {
    const o = await page.evaluate(() => window.bardur.observe());
    return (
      o.visible_state.turn > obs.visible_state.turn &&
      o.your_turn &&
      o.state_version > obs.state_version + 2
    );
  });
  console.log("bot played full turn");
  obs = await page.evaluate(() => window.bardur.observe());
  const game = await page.evaluate(() => window.bardur.game);
  const publicResult = await page.request.get(
    `${URL}/api/games/${game.id}?view=all`,
  );
  const publicData = await publicResult.json();
  assert.equal(publicData.perspective, "1");
  assert(!publicData.frame.views, "no hidden views");
  if (publicData.event?.player === 2) {
    assert.deepEqual(publicData.event.action.command, {});
    assert.deepEqual(publicData.frame.event.action.command, {});
  }
  console.log("live fog guard passed");
  await page.getByRole("button", { name: "Tech Tree", exact: true }).click();
  assert((await page.locator(".tech").count()) === 25);
  await page.screenshot({ path: "artifacts/tech-desktop.png" });
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await page.screenshot({ path: "artifacts/play-desktop.png" });
  await page.getByRole("button", { name: "Menu", exact: true }).click();
  await page
    .getByRole("button", { name: "Save to library", exact: true })
    .click();
  await page.waitForTimeout(600);
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  const response = await page.request.get(
    `${URL}/api/games/${game.id}/export`,
  );
  const replay = await response.json();
  assert.equal(replay.schema, "bardur-replay-v1");
  assert.deepEqual(replay.available_views, ["1"]);
  assert(!("seed" in replay.provenance));
  assert(replay.frames.every((f) => Object.keys(f.views).join() === "1"));
  console.log("redacted export passed", replay.frames.length);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "artifacts/play-mobile.png" });
  assert.equal(
    await page.evaluate(() => document.documentElement.scrollWidth),
    390,
  );
  console.log("mobile layout passed");
  await closeGame(game.id);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByRole("button", { name: "Menu", exact: true }).click();
  await page.getByRole("button", { name: /Watch self-play/ }).click();
  await page.waitForFunction(
    () => document.getElementById("game-kind").textContent === "LIVE SELF-PLAY",
  );
  await page.waitForTimeout(1500);
  await page.getByRole("button", { name: "Pause replay", exact: true }).click();
  await page.waitForTimeout(1000);
  const watch = await page.evaluate(() => window.bardur.game);
  const wr = await page.request.get(
    `${URL}/api/games/${watch.id}?view=all`,
  );
  assert.equal((await wr.json()).perspective, "all");
  await page.screenshot({ path: "artifacts/watch-desktop.png" });
  console.log("self-play working");
  await closeGame(watch.id);
  assert.deepEqual(errors, []);
  await browser.close();
})();
