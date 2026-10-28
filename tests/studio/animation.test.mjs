import test from "node:test";
import assert from "node:assert/strict";
import {
  planTransition,
  sampleActor,
} from "../../src/polytopia_bot/studio/static/animation.js";
const u = (tile, owner = 1, kind = 2, health = 10) => ({
  tile,
  owner,
  kind,
  health,
});
const obs = (units, version = 0, hidden = []) => ({
  player: 1,
  state_version: version,
  visible_state: {
    board_size: 14,
    units,
    tiles: Array.from({ length: 196 }, (_, tile) => ({
      tile,
      known: !hidden.includes(tile),
    })),
  },
});
const event = (kind, source = 10, target = 11) => ({
  action: { command: { kind, source, target } },
});
test("movement starts at source, hops between tiles, finishes exactly at destination", () => {
  const a = obs([u(10)]),
    b = obs([u(11)], 1),
    copy = JSON.stringify([a, b]);
  const p = planTransition(a, b, event(1));
  assert.equal(p.actors.length, 1);
  assert(p.hidden.has(11));
  assert.equal(sampleActor(p.actors[0], 0).travel, 0);
  assert.equal(sampleActor(p.actors[0], 1).travel, 1);
  assert(sampleActor(p.actors[0], 0.4).hop > 0);
  assert.equal(JSON.stringify([a, b]), copy);
});
test("melee hit and retaliation are sequenced and use actual damage", () => {
  const p = planTransition(
    obs([u(10), u(11, 2)]),
    obs([u(10, 1, 2, 7), u(11, 2, 2, 5)], 1),
    event(2),
  );
  assert.equal(p.projectiles.length, 0);
  assert(p.actors[1].counter);
  assert.deepEqual(
    p.effects
      .filter((e) => e.type === "damage")
      .map((e) => [e.tile, e.amount, e.at]),
    [
      [10, 3, 0.64],
      [11, 5, 0.34],
    ],
  );
  assert.equal(sampleActor(p.actors[0], 0.1).health, 10);
  assert.equal(sampleActor(p.actors[0], 0.8).health, 7);
});
test("ranged attack uses a projectile without a melee lunge or fictitious counter hit", () => {
  const p = planTransition(
    obs([u(10, 1, 9), u(11, 2)]),
    obs([u(10, 1, 9), u(11, 2, 2, 5)], 1),
    event(2),
  );
  assert.equal(p.projectiles[0].kind, 9);
  assert.equal(sampleActor(p.actors[0], 0.2).lunge, 0);
  assert.equal(sampleActor(p.actors[0], 0.7).flash, 0);
  assert(!p.actors[1].counter);
});
test("defender fades out, and melee attacker advances after the killing hit", () => {
  const p = planTransition(
    obs([u(10), u(11, 2, 2, 2)]),
    obs([u(11)], 1),
    event(2),
  );
  assert(p.actors[0].advance);
  assert.equal(sampleActor(p.actors[0], 0.5).travel, 0);
  assert.equal(sampleActor(p.actors[0], 1).travel, 1);
  assert.equal(sampleActor(p.actors[1], 1).alpha, 0);
  assert.equal(p.effects.filter((e) => e.type === "death").length, 1);
});
test("attacker killed by retaliation fades only after counterattack", () => {
  const p = planTransition(
    obs([u(10, 1, 2, 2), u(11, 2)]),
    obs([u(11, 2, 2, 8)], 1),
    event(2),
  );
  assert(p.actors[1].counter);
  assert.equal(sampleActor(p.actors[0], 0.5).alpha, 1);
  assert(sampleActor(p.actors[0], 0.9).alpha < 1);
});
test("conversion changes ownership without a death or damage effect", () => {
  const p = planTransition(
    obs([u(10, 1, 10), u(11, 2)]),
    obs([u(10, 1, 10), u(11)], 1),
    event(2),
  );
  assert(p.effects.some((e) => e.type === "convert"));
  assert(!p.effects.some((e) => ["damage", "death"].includes(e.type)));
});
test("redacted combat only animates public damage and invents no attacker", () => {
  const p = planTransition(
    obs([u(11)]),
    obs([u(11, 1, 2, 6)], 1),
    event(undefined),
  );
  assert.equal(p.projectiles.length, 0);
  assert.equal(p.actors.length, 1);
  assert.equal(p.actors[0].type, "hurt");
  assert.deepEqual(
    p.effects.map((e) => e.type),
    ["damage"],
  );
});
test("fog disappearance, ambiguous moves, seeks and perspective switches never fabricate combat", () => {
  assert.equal(
    planTransition(obs([u(10, 2)]), obs([], 1, [10]), event(undefined)),
    null,
  );
  assert.equal(
    planTransition(obs([u(10, 2)]), obs([u(11, 2)], 1, [10]), event(undefined)),
    null,
  );
  assert.equal(
    planTransition(
      obs([u(10), u(12)]),
      obs([u(11), u(13)], 1),
      event(undefined),
    ),
    null,
  );
  assert.equal(planTransition(obs([u(10)]), obs([u(11)], 2), event(1)), null);
  assert.equal(
    planTransition(obs([u(10)]), { ...obs([u(11)], 1), player: 2 }, event(1)),
    null,
  );
});
test("unambiguous visible opponent motion can animate without private action coordinates", () => {
  const p = planTransition(
    obs([u(10, 2)]),
    obs([u(11, 2)], 1),
    event(undefined),
  );
  assert.equal(p.actors[0].type, "move");
  assert.equal(p.actors[0].from, 10);
  assert.equal(p.actors[0].to, 11);
});

test("own exploration animates into newly revealed tiles without revealing an enemy path", () => {
  const before = obs([u(10)], 0, [11]),
    after = obs([u(11)], 1);
  assert.equal(planTransition(before, after, event(1)).actors[0].type, "move");
  assert.equal(
    planTransition(obs([u(10, 2)], 0, [11]), obs([u(11, 2)], 1), event(1)),
    null,
  );
});

test("safe public opponent cue animates visible combat despite redacted action payload", () => {
  const p = planTransition(
    obs([u(10, 2), u(11)]),
    obs([u(10, 2, 2, 7), u(11, 1, 2, 5)], 1),
    { action: { command: {} }, animation: { kind: 2, source: 10, target: 11 } },
  );
  assert.equal(p.actors[0].type, "attack");
  assert(p.actors[1].counter);
});

test("attack advance into a port changes unit kind without inventing attacker death", () => {
  const p = planTransition(
    obs([u(10), u(11, 2, 2, 1)]),
    obs([u(11, 1, 39)], 1),
    event(2),
  );
  assert(p.actors[0].advance);
  assert.equal(p.actors[0].deathAt, null);
  assert.equal(p.effects.filter((e) => e.type === "death").length, 1);
});
