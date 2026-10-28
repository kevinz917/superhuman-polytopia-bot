const { chromium } = require("../../artifacts/browser/node_modules/playwright");
const assert = require("node:assert/strict");
const URL = process.env.STUDIO_URL || "http://127.0.0.1:8765";
const fs = require("node:fs");
(async () => {
  const browser = await chromium.launch({
    executablePath:
      process.env.CHROME_PATH ||
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: true,
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
    reducedMotion: "no-preference",
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(URL);
  await page.waitForFunction(() => window.bardur?.game);
  const lib = await (
    await page.request.get(`${URL}/api/library`)
  ).json();
  const replay = lib.replays
    .filter((r) => r.source === "verified-engine-replay")
    .sort((a, b) => b.created - a.created)[0];
  assert(replay);
  await page.getByRole("button", { name: "Menu", exact: true }).click();
  await page.getByRole("button", { name: /Replay library/ }).click();
  await page
    .getByRole("button", {
      name: new RegExp(replay.title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")),
    })
    .first()
    .click();
  await page.waitForTimeout(500);
  await page.evaluate(() => {
    window.animationEvents = [];
    const canvas = document.getElementById("board");
    canvas.addEventListener("bardur:animation-start", (e) =>
      window.animationEvents.push(e.detail),
    );
  });
  const cases = { move: 4, melee: 171, death: 172, ranged: 729 };
  for (const [kind, index] of Object.entries(cases)) {
    await page.getByLabel("Replay position").fill(String(index - 1));
    await page.waitForFunction(
      (i) =>
        document.getElementById("scrubber").value === String(i) &&
        !window.bardur.animating,
      index - 1,
    );
    const count = await page.evaluate(() => window.animationEvents.length);
    await page
      .getByRole("button", { name: "Next action", exact: true })
      .click();
    await page.waitForFunction(
      (n) => window.animationEvents.length > n && window.bardur.animating,
      count,
    );
    const ev = await page.evaluate(() => window.animationEvents.at(-1));
    assert.equal(
      ev.kind,
      kind === "ranged" ? "projectile" : kind === "move" ? "move" : "attack",
    );
    if (kind === "death") assert(ev.effects.includes("death"));
    if (kind === "melee")
      assert.equal(ev.effects.filter((e) => e === "damage").length, 2);
    if (kind === "ranged") assert(ev.projectiles > 0);
    await page.waitForTimeout(
      kind === "move" ? 150 : kind === "ranged" ? 160 : 400,
    );
    await page.screenshot({ path: `artifacts/animation-${kind}.png` });
    await page.waitForFunction(() => !window.bardur.animating);
    assert.equal(
      await page.getByLabel("Replay position").inputValue(),
      String(index),
    );
  }
  // A seek during a strike cancels immediately; no leftover actors or delayed playback.
  await page.getByLabel("Replay position").fill("170");
  await page.waitForTimeout(80);
  await page.getByRole("button", { name: "Next action", exact: true }).click();
  await page.waitForFunction(() => window.bardur.animating);
  await page.getByLabel("Replay position").fill("900");
  await page.waitForTimeout(150);
  assert.equal(await page.evaluate(() => window.bardur.animating), false);
  assert.equal(await page.getByLabel("Replay position").inputValue(), "900");
  // Reduced-motion disables transitions without changing the underlying replay.
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.getByLabel("Replay position").fill("170");
  await page.waitForTimeout(80);
  const reducedCount = await page.evaluate(() => window.animationEvents.length);
  await page.getByRole("button", { name: "Next action", exact: true }).click();
  await page.waitForTimeout(180);
  assert.equal(
    await page.evaluate(() => window.animationEvents.length),
    reducedCount,
  );
  assert.equal(await page.getByLabel("Replay position").inputValue(), "171");
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.getByLabel("Playback speed").selectOption("8");
  await page.getByLabel("Replay position").fill("728");
  await page.waitForTimeout(80);
  await page.getByRole("button", { name: "Next action", exact: true }).click();
  await page.waitForTimeout(240);
  assert(
    (await page.evaluate(() => window.animationEvents.at(-1).duration)) <= 150,
  );
  assert.equal(await page.evaluate(() => window.bardur.animating), false);
  // Disable setting persists and bypasses movement animations.
  await page.getByRole("button", { name: "Menu", exact: true }).click();
  await page.getByLabel("Animate moves and combat").uncheck();
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await page.getByLabel("Replay position").fill("3");
  await page.waitForTimeout(80);
  const disabledCount = await page.evaluate(
    () => window.animationEvents.length,
  );
  await page.getByRole("button", { name: "Next action", exact: true }).click();
  await page.waitForTimeout(160);
  assert.equal(
    await page.evaluate(() => window.animationEvents.length),
    disabledCount,
  );
  assert.deepEqual(errors, []);
  console.log(
    "PASS: real replay movement, melee/counterattack, defeat, ranged projectile, seek cancellation, reduced motion, 8x speed, animation setting; no browser errors",
  );
  fs.writeFileSync(
    "artifacts/animation-validation.json",
    JSON.stringify(
      {
        replay: replay.id,
        cases,
        events: await page.evaluate(() => window.animationEvents),
        errors,
      },
      null,
      2,
    ),
  );
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
