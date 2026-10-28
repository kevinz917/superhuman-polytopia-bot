import { BoardRenderer, cityName, cityTerritory } from "./renderer.js";
import { spriteURL } from "./assets.js";
const $ = (id) => document.getElementById(id);
let meta,
  current = null,
  data = null,
  legal = [],
  selected = null,
  layer = null,
  shownStars = null,
  starsKey = null,
  playing = false,
  busy = false,
  timer = null,
  frameRequest = 0,
  ticking = false;
let animationEnabled = localStorage.getItem("bardur-animations") !== "off";
const board = new BoardRenderer($("board"), $("tile-controls"), selectTile);
const unitName = (k) =>
  meta?.rules.units.find((u) => u.id === k)?.name || "Unit";
const titleCase = (s) => s.replace(/([a-z])([A-Z])/g, "$1 $2");
const text = (tag, value, className) => {
  const e = document.createElement(tag);
  e.textContent = value;
  if (className) e.className = className;
  return e;
};
const button = (label, fn, className = "pill") => {
  const e = text("button", label, className);
  e.onclick = fn;
  return e;
};
const UNIT_SPRITES = {
  2: "warrior",
  3: "rider",
  4: "knight",
  5: "defender",
  8: "catapult",
  9: "archer",
  10: "mindbender",
  11: "swordsman",
  12: "giant",
  37: "dagger",
  38: "cloak",
  39: "boat",
  40: "pirate",
  41: "bombership",
  42: "scoutship",
  43: "boat",
  44: "rammership",
  45: "juggernaut",
};
const RESOURCE_SPRITES = { 1: "animal", 2: "crop", 5: "metal", 6: "fruit" };
const HARVEST = {
  1: "Hunting",
  3: "Fishing",
  4: "Whaling",
  6: "Harvest fruit",
  7: "Harvest fungi",
};
const TERRAIN_TEXT = {
  1: "Shallow water. Boats can sail here.",
  2: "Deep ocean, only for ships.",
  3: "Open field.",
  4: "Mountains give a defence bonus and hold metal.",
  5: "Forests slow movement and give a defence bonus.",
};
const svgIcon = (body, view = "0 0 40 40") =>
  `<svg viewBox="${view}" aria-hidden="true">${body}</svg>`;
const ICONS = {
  heart: svgIcon(
    '<path fill="#ff6f8e" d="M20 33 6 19a8 8 0 0 1 14-9 8 8 0 0 1 14 9z"/>',
  ),
  cross: svgIcon(
    '<path fill="#fff" d="M16 6h8v10h10v8H24v10h-8V24H6v-8h10z"/>',
  ),
  disband: svgIcon(
    '<path stroke="#fff" stroke-width="6" stroke-linecap="round" d="m10 10 20 20m0-20L10 30"/>',
  ),
  flame: svgIcon(
    '<path fill="#ff9b2f" d="M20 35c-7 0-11-5-11-11 0-7 7-10 6-18 6 3 9 8 9 13 2-1 3-3 3-6 3 3 5 7 5 11 0 6-5 11-12 11z"/><path fill="#ffe36b" d="M20 35c-3 0-5-2-5-5 0-4 4-5 4-9 3 2 6 5 6 9 0 3-2 5-5 5z"/>',
  ),
  axe: svgIcon(
    '<path stroke="#a57a4b" stroke-width="4" stroke-linecap="round" d="M14 34 27 9"/><path fill="#dfe8ee" d="M22 6c6-2 11 1 12 6l-9 5-6-4z"/>',
  ),
  hammer: svgIcon(
    '<path stroke="#a57a4b" stroke-width="4" stroke-linecap="round" d="M13 34 24 15"/><path fill="#c5cdd3" d="m17 8 10-4 8 14-10 4z"/>',
  ),
  wall: svgIcon(
    '<path fill="#c9d3d7" d="M5 16h30v18H5z"/><path fill="#9aa9b0" d="M5 10h6v6H5zm9 0h6v6h-6zm9 0h6v6h-6zm9 0h3v6h-3zM5 22h12v1H5zm18 0h12v1H23zM11 28h18v1H11z"/>',
  ),
  border: svgIcon(
    '<path fill="none" stroke="#fff" stroke-width="3" stroke-dasharray="4 3" d="M20 3 37 20 20 37 3 20z"/><path fill="#79d0ff" d="m20 12 8 8-8 8-8-8z"/>',
  ),
  explorer: svgIcon(
    '<path fill="#8a7a63" d="M13 36 15 18h10l2 18z"/><path fill="#287ba7" d="M14 22h12v4H14z"/><circle cx="20" cy="13" r="5" fill="#e7d4ac"/><path fill="#6c5e4a" d="M13 14 20 3l7 11-3-3h-8z"/><path stroke="#6b5236" stroke-width="2" d="M30 37 32 10"/>',
  ),
};
const REWARDS = {
  1: { text: "Units in this city get a big defence bonus.", icon: ICONS.wall },
  2: { text: "+1 star per turn and +250 score.", sprite: "forest" },
  3: { text: "+1 star per turn.", sprite: "sawmill" },
  4: { text: "An explorer reveals the land around the city.", icon: ICONS.explorer },
  5: { text: "Expand the city's borders.", icon: ICONS.border },
  6: { text: "A Giant appears in the city.", sprite: "unit-giant" },
  7: { text: "+5 stars right now.", sprite: "star" },
  8: { text: "+3 population.", sprite: "house-3" },
};
const ABILITY_ICONS = {
  1: { icon: ICONS.flame },
  2: { icon: ICONS.axe },
  3: { sprite: "forest" },
  4: { icon: ICONS.hammer },
  7: { icon: ICONS.heart },
  8: { icon: ICONS.disband },
  9: { icon: ICONS.cross },
  16: { sprite: "star" },
};
function structureSprite(id) {
  const name = (
    meta.rules.structures.find((s) => s.id === id)?.name || ""
  ).toLowerCase();
  for (const key of [
    "farm",
    "windmill",
    "port",
    "sawmill",
    "mine",
    "forge",
    "market",
    "temple",
  ])
    if (name.includes(key)) return key;
  if (name.includes("lumber")) return "lumber";
  if (name.includes("road") || name.includes("bridge")) return "ground";
  return "star";
}
function actionArt(a) {
  const c = a.command;
  if (c.kind === 4) return { sprite: "unit-" + (UNIT_SPRITES[c.parameter] || "warrior") };
  if (c.kind === 6) return { sprite: structureSprite(c.parameter) };
  if (c.kind === 5) {
    const r = data.observation.visible_state.tiles[c.target]?.resource;
    return RESOURCE_SPRITES[r] ? { sprite: RESOURCE_SPRITES[r] } : { sprite: "water" };
  }
  if (c.kind === 8) return { sprite: "capital" };
  if (c.kind === 9) return REWARDS[c.parameter] || { sprite: "star" };
  if (c.kind === 3) return ABILITY_ICONS[c.parameter] || { sprite: "star" };
  return { sprite: "star" };
}
function art(target, { sprite, icon }) {
  if (icon) target.innerHTML = icon;
  else {
    const img = document.createElement("img");
    img.src = spriteURL(sprite);
    img.alt = "";
    target.append(img);
  }
  return target;
}
// Mobile-style circular order button: icon, star cost badge, label.
function roundAction(a, label = actLabel(a), enabled = true) {
  const b = document.createElement("button");
  b.className = "round-action";
  b.disabled = !enabled;
  const ring = art(text("span", "", "round-icon order-icon"), actionArt(a));
  const price = cost(a);
  if (price) ring.append(text("small", price.replace(" ★", ""), "order-cost"));
  b.append(ring, text("span", label));
  if (enabled && a.action_id !== undefined) {
    b.dataset.actionId = a.action_id;
    b.onclick = () => takeAction(a);
  }
  return b;
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").classList.remove("hidden");
  setTimeout(() => $("toast").classList.add("hidden"), 5500);
}
async function api(path, body) {
  const r = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Studio-Client": "1",
          },
          body: JSON.stringify(body),
        },
  );
  const result = await r.json();
  if (!r.ok) throw Error(result.error || r.statusText);
  return result;
}
async function waitJob(j) {
  if (!j.job_id) return j;
  const id = j.job_id;
  for (;;) {
    await new Promise((r) => setTimeout(r, 200));
    j = await api("/api/jobs/" + id);
    if (j.status === "done") return j.result;
    if (j.status === "error") throw Error(j.error);
  }
}
async function job(path, body) {
  return waitJob(await api(path, body));
}
function status(message, isBusy = false) {
  $("status-text").textContent = message;
  $("status").classList.toggle("busy", isBusy);
  $("status").classList.toggle(
    "terminal",
    !!(data?.terminated || data?.truncated),
  );
}
function openDialog() {
  $("dialog").className = "";
  playing = false;
  updatePlay();
  $("dialog-content").replaceChildren();
  if (!$("dialog").open) $("dialog").showModal();
  return $("dialog-content");
}
function heading(node, eyebrow, title, description) {
  node.append(text("div", eyebrow, "eyebrow"), text("h2", title));
  if (description) node.append(text("p", description));
}
function updatePlay() {
  $("play").textContent = playing ? "Ⅱ" : "▶";
  $("play").setAttribute(
    "aria-label",
    playing ? "Pause replay" : "Play replay",
  );
}
function actLabel(a) {
  const c = a.command,
    p = c.parameter,
    k = c.kind;
  return titleCase(
    k === 3
      ? p === 19
        ? c.target === 1
          ? "Accept peace"
          : "Reject peace"
        : meta.abilities[p] || "Ability"
      : k === 4
        ? "Train " + unitName(p)
        : k === 6
          ? "Build " +
            (meta.rules.structures.find((s) => s.id === p)?.name || p)
          : k === 7
            ? "Research " +
              (meta.rules.technologies.find((t) => t.id === p)?.name || p)
            : k === 9
              ? meta.rewards[p] || "Reward"
              : {
                  1: "Move",
                  2: "Attack",
                  5: "Harvest",
                  8: "Capture",
                  10: "End turn",
                  11: "Resign",
                }[k] || "Action",
  );
}
function cost(a) {
  const c = a.command,
    k = c.kind,
    p = c.parameter,
    v = data.observation.visible_state;
  let n;
  if (k === 4) n = meta.rules.units.find((u) => u.id === p)?.cost;
  if (k === 6) n = meta.rules.structures.find((s) => s.id === p)?.cost;
  if (k === 5) n = 2;
  if (k === 7) {
    const t = meta.rules.technologies.find((t) => t.id === p);
    n =
      4 +
      v.cities.filter((c) => c.owner === data.observation.player).length *
        t.tier;
    if (v.technology.includes(22)) n = Math.ceil(n * 0.66);
  }
  return n !== undefined && n !== null && n >= 0 ? `${n} ★` : "";
}
function actionButton(a) {
  const b = button(
    actLabel(a),
    () => takeAction(a),
    a.command.kind === 11 ? "action-button danger" : "action-button",
  );
  b.dataset.actionId = a.action_id;
  const price = cost(a);
  if (price) b.append(text("small", price));
  return b;
}
function canAct() {
  return (
    current?.kind === "games" &&
    data?.mode !== "watch" &&
    !busy &&
    !board.animating &&
    data?.current_player === data?.seat &&
    !data?.terminated &&
    !data?.truncated &&
    data?.frame_index === data?.frames - 1
  );
}
async function load(index = null, view = null, animate = true) {
  if (!current) return;
  const request = ++frameRequest;
  if (!animate) board.finishAnimation();
  let url = `/api/${current.kind}/${current.id}?view=${view || $("perspective").value || "actor"}`;
  if (index !== null) url += `&frame=${index}`;
  const next = await api(url);
  if (request !== frameRequest) return;
  if (next.job_id) {
    await waitJob(next);
    return load(index, view, animate);
  }
  data = next;
  legal = [];
  if (
    current.kind === "games" &&
    next.mode !== "watch" &&
    !next.terminated &&
    !next.truncated &&
    next.frame_index === next.frames - 1
  ) {
    const obs = await api(`/api/games/${current.id}/observation`);
    if (request !== frameRequest) return;
    legal = obs.legal_actions;
  }
  render({ animate });
  await board.whenIdle();
  if (request === frameRequest) render();
}
function render(animationOptions = {}) {
  if (!data) return;
  const v = data.observation.visible_state,
    p = data.observation.player;
  $("tribe-name").replaceChildren(
    text("span", "Bardur"),
    text(
      "small",
      data.perspective === "all" ? "Spectator view" : `Player ${p}`,
    ),
  );
  document.querySelector(".tribe-dot").style.background =
    p === 1 ? "#297fa5" : "#c76856";
  animateStars(v.stars, `${current?.kind}/${current?.id}/${data.perspective}`);
  $("income").textContent = data.observation.hud
    ? `Stars (+${data.observation.hud.income})`
    : "Stars";
  $("score").textContent = data.observation.hud?.score?.toLocaleString() ?? "—";
  $("turn").textContent = v.turn;
  $("game-title").textContent = data.title;
  $("game-subtitle").textContent =
    data.mode === "play" || data.mode === "agent"
      ? `You are Player ${data.seat} · 14 × 14 · Drylands`
      : data.note
        ? "Original public-observation trace"
        : `${meta.bot.name} · 14 × 14 · Drylands`;
  $("game-kind").textContent =
    data.mode === "replay"
      ? "RECORDED MATCH"
      : data.mode === "watch"
        ? "LIVE SELF-PLAY"
        : data.mode === "agent"
          ? "EXTERNAL AGENT"
          : "HUMAN VS LEARNED BOT";
  $("mode-label").textContent =
    data.mode === "replay"
      ? "REPLAY"
      : data.mode === "watch"
        ? "LIVE MATCH"
        : "PLAYING";
  $("mode-dot").style.background =
    data.mode === "replay" ? "#d29a47" : "#6a9d88";
  const choice = $("perspective").value;
  $("perspective").replaceChildren(
    ...data.available_views.map((value) => {
      const o = document.createElement("option");
      o.value = value;
      o.textContent =
        {
          all: "Full board",
          actor: "Acting player",
          1: "Player 1 fog",
          2: "Player 2 fog",
        }[value] || value;
      return o;
    }),
  );
  $("perspective").value = data.perspective;
  $("scrubber").max = data.frames - 1;
  $("scrubber").value = data.frame_index;
  $("frame-count").textContent = `${data.frame_index} / ${data.frames - 1}`;
  $("event-label").textContent = data.event
    ? `P${data.event.player} · ${titleCase(data.event.action.label)}`
    : "Opening position";
  $("end-turn").disabled =
    !canAct() || !legal.some((a) => a.command.kind === 10);
  $("end-label").textContent = "Next Turn";
  $("replay-dock").classList.toggle(
    "hidden",
    !["replay", "watch"].includes(data.mode) &&
      data.frame_index === data.frames - 1 &&
      !data.terminated &&
      !data.truncated,
  );
  const reward = canAct() && legal.find((a) => a.command.kind === 9);
  if (reward) {
    selected = reward.command.target;
    layer = "city";
  }
  if (selected !== null && !layersAt(selected).includes(layer))
    layer = layersAt(selected)[0];
  const targets =
    canAct() && selected !== null && layer === "unit"
      ? legal.filter(
          (a) =>
            [1, 2].includes(a.command.kind) && a.command.source === selected,
        )
      : [];
  const ownerCity =
    layer === "city" && v.cities.find((c) => c.tile === selected && c.owner);
  board.update(data.observation, selected, targets, data.event, {
    layer: selected === null ? null : layer,
    territory: ownerCity
      ? data.observation.hud?.territory?.[selected] ||
        cityTerritory(v, selected)
      : [],
    scene: `${current.kind}/${current.id}/${data.perspective}/${data.perspective === "all" ? "" : p}`,
    animate: animationEnabled && animationOptions.animate !== false,
    speed: ["replay", "watch"].includes(data.mode)
      ? Number($("speed").value)
      : 1,
  });
  if (board.animating) {
    $("end-turn").disabled = true;
    // Follow the action like the mobile camera when it happens off-screen.
    const cue = data.event?.animation || data.event?.action?.command || {};
    const spot = [cue.target, cue.source].find((t) => t >= 0);
    if (spot !== undefined) board.focus(spot);
  }
  renderSelection();
  renderReward(reward);
  if (data.terminated || data.truncated) {
    const rewards = data.event?.rewards;
    const winner = rewards
      ? Object.keys(rewards).find((k) => rewards[k] > 0)
      : null;
    status(
      data.truncated
        ? "Game reached the research horizon"
        : winner
          ? `Player ${winner} wins · replay saved`
          : "Game finished · replay saved",
    );
  } else if (busy && (data.mode === "watch" || data.current_player !== data.seat))
    status("The Champion is thinking…", true);
  else if (busy) status("Your turn", false);
  else if (data.mode === "replay")
    status(
      data.frame_index === data.frames - 1 && data.result
        ? `Recorded result · ${data.result.reward > 0 ? "challenger wins" : data.result.reward < 0 ? "challenger loses" : "no native winner"}`
        : "Replay · Space to play · ← → to step",
    );
  else if (data.mode === "watch")
    status(
      playing
        ? "Live self-play · both seats use the Champion"
        : "Self-play paused · press play to watch",
    );
  else
    status(
      data.frame_index !== data.frames - 1
        ? "Reviewing your game · jump to latest to play"
        : data.current_player === data.seat
          ? "Your turn · select a unit or city"
          : "The Champion is thinking…",
      data.current_player !== data.seat,
    );
  updatePlay();
}
function layersAt(tile) {
  const v = data?.observation.visible_state,
    t = v?.tiles[tile];
  if (!t?.known) return ["tile"];
  const out = [];
  if (v.units.some((u) => u.tile === tile)) out.push("unit");
  out.push(v.cities.some((c) => c.tile === tile) ? "city" : "tile");
  return out;
}
function statLine(parts) {
  const row = text("div", "", "stat-line");
  for (const [label, value] of parts) {
    const item = text("span", "");
    item.append(text("small", label), text("b", String(value)));
    row.append(item);
  }
  return row;
}
function renderSelection() {
  // Live play swaps the round controls for the selection bar; replays keep
  // their transport and float the selection above it.
  const live = current?.kind === "games" && data?.mode !== "watch";
  document.body.classList.toggle("selecting", selected !== null && live);
  $("selection").classList.toggle("over-dock", !live);
  if (selected === null || !data) {
    $("selection").classList.add("hidden");
    return;
  }
  const v = data.observation.visible_state,
    p = data.observation.player,
    t = v.tiles[selected],
    u = v.units.find((u) => u.tile === selected),
    city = v.cities.find((c) => c.tile === selected);
  if (!t) return;
  $("selection").classList.remove("hidden");
  $("selection").dataset.layer = layer;
  $("tile-coord").textContent =
    `TILE ${selected} · ${(selected % v.board_size) + 1}, ${Math.floor(selected / v.board_size) + 1}`;
  const portrait = $("selection-portrait");
  portrait.replaceChildren();
  const detail = $("tile-detail");
  detail.replaceChildren();
  const orders = $("actions");
  orders.replaceChildren();
  const mine = (a) => canAct() && a;
  if (!t.known) {
    $("tile-title").textContent = "Unexplored";
    detail.append(text("p", "Explore this tile to reveal what lies beneath the clouds."));
    art(portrait, { sprite: "ground" });
    return;
  }
  if (layer === "unit" && u) {
    const rules = meta.rules.units.find((r) => r.id === u.kind) || {};
    $("tile-title").textContent = unitName(u.kind);
    art(portrait, { sprite: "unit-" + (UNIT_SPRITES[u.kind] || "warrior") });
    detail.append(
      statLine([
        ["Health", `${Math.ceil(u.health)}/${(rules.health || 10) + (u.veteran ? 5 : 0)}`],
        ["Attack", rules.attack ?? "—"],
        ["Defence", rules.defense ?? "—"],
        ["Move", rules.movement ?? "—"],
        ["Range", rules.range ?? "—"],
      ]),
    );
    const notes = [];
    if (u.owner !== p) notes.push(`Player ${u.owner}`);
    if (u.veteran) notes.push("Veteran");
    if (u.owner === p && u.moved && u.attacked) notes.push("No moves left");
    if (rules.skills?.length) notes.push(rules.skills.map(titleCase).join(" · "));
    if (notes.length) detail.append(text("p", notes.join(" · ")));
    for (const a of legal.filter(
      (a) =>
        mine(a) &&
        ((a.command.kind === 8 && a.command.source === selected) ||
          (a.command.kind === 3 &&
            a.command.source === selected &&
            a.command.parameter < 18)),
    ))
      orders.append(roundAction(a));
    return;
  }
  if (layer === "city" && city) {
    const level = city.level || 1;
    $("tile-title").textContent = city.owner ? cityName(selected) : "Village";
    art(portrait, { sprite: city.owner && level > 2 ? "house-3" : "house-1" });
    if (!city.owner)
      detail.append(text("p", "Move a unit here and capture it to found a new city."));
    else if (city.level !== undefined) {
      detail.append(
        statLine([
          ["Level", level],
          ["Population", `${city.progress}/${level + 1}`],
          ["Income", `+${city.production} ★`],
        ]),
      );
      if (v.tiles[selected].capital) detail.append(text("p", "Capital"));
    } else detail.append(text("p", `Player ${city.owner} city`));
    if (city.owner === p && canAct()) {
      // Every unit your technology unlocks, greyed out when you cannot train it.
      const known = new Set(
        meta.rules.technologies
          .filter((t) => t.id === 0 || v.technology.includes(t.id))
          .flatMap((t) => t.units),
      );
      for (const kind of [...known].filter((k) => UNIT_SPRITES[k] && k < 39)) {
        const a = legal.find(
          (a) =>
            a.command.kind === 4 &&
            a.command.source === selected &&
            a.command.parameter === kind,
        );
        orders.append(
          roundAction(
            a || { command: { kind: 4, parameter: kind, source: selected } },
            unitName(kind),
            !!a,
          ),
        );
      }
      for (const a of legal.filter(
        (a) =>
          a.command.kind === 4 &&
          a.command.source === selected &&
          !known.has(a.command.parameter),
      ))
        orders.append(roundAction(a));
    }
    return;
  }
  const resource = t.resource ? meta.resources[t.resource] : null;
  $("tile-title").textContent = resource || meta.terrain[t.terrain] || "Land";
  art(portrait, {
    sprite:
      RESOURCE_SPRITES[t.resource] ||
      { 4: "mountain", 5: "forest", 1: "water", 2: "ocean" }[t.terrain] ||
      "ground",
  });
  const lines = [];
  if (resource) lines.push(meta.terrain[t.terrain]);
  if (TERRAIN_TEXT[t.terrain]) lines.push(TERRAIN_TEXT[t.terrain]);
  if (t.structure > 2)
    lines.push(
      meta.rules.structures.find((x) => x.id === t.structure)?.name || "",
    );
  lines.push(t.owner ? (t.owner === p ? "Your territory" : `Player ${t.owner} territory`) : "Unclaimed land");
  detail.append(text("p", lines.filter(Boolean).join(" · ")));
  for (const a of legal.filter(
    (a) =>
      mine(a) &&
      a.command.target === selected &&
      ([5, 6].includes(a.command.kind) ||
        (a.command.kind === 3 && a.command.source < 0 && a.command.parameter < 18)),
  ))
    orders.append(
      roundAction(a, a.command.kind === 5 ? HARVEST[t.resource] || "Harvest" : actLabel(a)),
    );
}
function selectTile(tile) {
  if (rewardOpen()) return;
  if (canAct() && selected !== null && layer === "unit") {
    const actions = legal.filter(
      (a) =>
        [1, 2].includes(a.command.kind) &&
        a.command.source === selected &&
        a.command.target === tile,
    );
    if (actions.length === 1) {
      takeAction(actions[0]);
      return;
    }
  }
  // Mobile tap cycle: unit, then the city or tile beneath it, then clear.
  const layers = data ? layersAt(tile) : ["tile"];
  if (tile === selected) {
    const next = layers.indexOf(layer) + 1;
    if (next >= layers.length) {
      selected = null;
      layer = null;
    } else layer = layers[next];
  } else {
    selected = tile;
    layer = layers[0];
  }
  render();
}
function rewardOpen() {
  return !$("reward").classList.contains("hidden");
}
// City level-up choice, overlaid on the board like the mobile popup.
function renderReward(pending) {
  const box = $("reward");
  if (!pending || busy || board.animating) {
    box.classList.add("hidden");
    return;
  }
  const v = data.observation.visible_state,
    city = v.cities.find((c) => c.tile === pending.command.target),
    choices = legal.filter(
      (a) => a.command.kind === 9 && a.command.target === pending.command.target,
    );
  const key = choices.map((a) => a.action_id).join();
  if (box.dataset.key === key && !box.classList.contains("hidden")) return;
  box.dataset.key = key;
  const card = $("reward-card");
  card.replaceChildren(
    text("div", `${cityName(city.tile)} · Level ${city.level}`, "eyebrow"),
    text("h2", `${cityName(city.tile)} has grown to level ${city.level}!`),
    text("p", "Choose your reward"),
  );
  const row = text("div", "", "reward-choices");
  for (const a of choices) {
    const b = document.createElement("button");
    b.className = "reward-choice";
    b.dataset.actionId = a.action_id;
    b.onclick = () => {
      box.classList.add("hidden");
      takeAction(a);
    };
    b.append(
      art(text("span", "", "round-icon reward-icon"), REWARDS[a.command.parameter] || {}),
      text("strong", actLabel(a)),
      text("small", REWARDS[a.command.parameter]?.text || ""),
    );
    row.append(b);
  }
  card.append(row);
  box.classList.remove("hidden");
}
// Star counter ticks like the mobile HUD; floating +/- shows the change.
function animateStars(value, key) {
  const el = $("stars");
  if (key !== starsKey || shownStars === null || !animationEnabled) {
    starsKey = key;
    shownStars = value;
    el.textContent = value;
    return;
  }
  if (value === shownStars) return;
  const from = shownStars,
    delta = value - from,
    t0 = performance.now();
  shownStars = value;
  const float = text("span", (delta > 0 ? "+" : "−") + Math.abs(delta), "star-float");
  float.classList.toggle("spent", delta < 0);
  el.parentElement.append(float);
  setTimeout(() => float.remove(), 1100);
  el.parentElement.classList.remove("star-bump");
  void el.offsetWidth;
  el.parentElement.classList.add("star-bump");
  const step = () => {
    const k = Math.min(1, (performance.now() - t0) / 550);
    if (shownStars !== value) return;
    el.textContent = Math.round(from + delta * k);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
async function takeAction(a) {
  if (!canAct()) return;
  const id = current.id;
  if ([11].includes(a.command.kind)) {
    const c = openDialog();
    heading(
      c,
      "LEAVE THE SQUARE",
      "Resign this game?",
      "This ends the game and saves the replay.",
    );
    c.append(
      button("Resign", async () => {
        $("dialog").close();
        await perform(a, id);
      }),
      button("Keep playing", () => $("dialog").close()),
    );
    return;
  }
  await perform(a, id);
}
async function perform(a, id) {
  busy = true;
  render();
  status("Applying move…", true);
  $("dialog").close();
  try {
    await job(`/api/games/${id}/action`, {
      action_id: a.action_id,
      expected_state_version: data.observation.state_version,
    });
    selected = null;
    await load();
  } catch (e) {
    toast(e.message);
    await load();
  } finally {
    busy = false;
    render();
  }
  await botTurn();
}
async function botTurn() {
  if (
    !current ||
    current.kind !== "games" ||
    busy ||
    data.terminated ||
    data.truncated ||
    data.mode === "watch" ||
    data.current_player === data.seat
  )
    return;
  busy = true;
  render();
  const id = current.id;
  try {
    while (
      current?.id === id &&
      !data.terminated &&
      !data.truncated &&
      data.current_player !== data.seat
    ) {
      await job(`/api/games/${id}/advance`, { count: 1 });
      await load();
      await new Promise((r) => setTimeout(r, 130));
    }
  } catch (e) {
    toast(e.message);
    status("Bot paused after an error · reopen the game to retry");
  } finally {
    busy = false;
    render();
  }
}
async function start(mode) {
  if (busy) return toast("Wait for the current move to finish.");
  const seed = $("new-seed")?.value,
    seat = $("new-seat")?.value || 1;
  const params = { mode, seat: Number(seat) };
  if (seed) params.seed = Number(seed);
  $("dialog").close();
  playing = false;
  busy = true;
  status(
    meta.ready
      ? "Setting up the square…"
      : "Preparing engine and checkpoint · first start can take a minute",
    true,
  );
  try {
    const result = await job("/api/games", params);
    current = { kind: "games", id: result.game_id };
    localStorage.setItem("bardur-current", JSON.stringify(current));
    selected = null;
    $("perspective").value = mode === "watch" ? "all" : seat;
    await load(null, mode === "watch" ? "all" : String(seat));
    board.center();
    busy = false;
    render();
    if (mode === "watch") {
      playing = true;
      updatePlay();
      tick();
    } else await botTurn();
  } catch (e) {
    toast(e.message);
    busy = false;
    status("Could not start game · " + e.message);
  }
}
async function openReplay(id) {
  playing = false;
  current = { kind: "replays", id };
  selected = null;
  legal = [];
  localStorage.setItem("bardur-current", JSON.stringify(current));
  $("dialog").close();
  await load(0, "all");
  board.center();
}
async function tick() {
  clearTimeout(timer);
  if (!playing || !current || ticking) return;
  ticking = true;
  try {
    if (data.frame_index < data.frames - 1) {
      await load(data.frame_index + 1);
    } else if (
      current.kind === "games" &&
      data.mode === "watch" &&
      !data.terminated &&
      !data.truncated
    ) {
      if (!busy) {
        busy = true;
        await job(`/api/games/${current.id}/advance`, { count: 1 });
        busy = false;
        await load();
      }
    } else {
      playing = false;
      updatePlay();
      return;
    }
  } catch (e) {
    playing = false;
    busy = false;
    toast(e.message);
    updatePlay();
    return;
  } finally {
    ticking = false;
  }
  if (playing)
    timer = setTimeout(tick, Math.max(40, 180 / Number($("speed").value)));
}
function togglePlay() {
  if (!current) return;
  if (
    data.mode !== "watch" &&
    current.kind === "games" &&
    data.frame_index === data.frames - 1
  )
    return toast("Your game is live. Select a unit to play.");
  playing = !playing;
  updatePlay();
  if (playing) {
    if (data.mode === "replay" && data.frame_index === data.frames - 1) {
      load(0).then(tick);
    } else tick();
  }
}
function menu() {
  const c = openDialog();
  heading(
    c,
    "THE SQUARE IS YOURS",
    "Bardur",
    "Watch a learned policy explore, build, and battle. Or take a seat across the board.",
  );
  const row = text("div", "", "menu-row");
  const seed = document.createElement("input");
  seed.id = "new-seed";
  seed.placeholder = "Random map";
  seed.type = "number";
  seed.min = 0;
  seed.max = 2147483647;
  seed.setAttribute("aria-label", "Map seed");
  const seat = document.createElement("select");
  seat.id = "new-seat";
  seat.setAttribute("aria-label", "Your seat");
  for (const n of [1, 2]) {
    const o = text("option", `Play as Player ${n}`);
    o.value = n;
    seat.append(o);
  }
  row.append(seed, seat);
  c.append(row);
  const cards = text("div", "", "menu-cards");
  const card = (title, description, fn, primary) => {
    const b = button("", fn, "menu-card" + (primary ? " primary" : ""));
    b.append(text("strong", title), text("small", description));
    cards.append(b);
  };
  card(
    "Play the bot",
    "Take on the Champion. Every move uses the trained checkpoint.",
    () => start("play"),
    true,
  );
  card(
    "Watch self-play",
    "Two learned players. Watch live, pause, and rewind.",
    () => start("watch"),
  );
  card(
    "Replay library",
    "Review saved matches and import recorded game traces.",
    library,
  );
  card(
    "Connect a model",
    "Control one seat through the public API or visible board.",
    () => start("agent"),
  );
  c.append(cards);
  const animationRow = text("label", "", "animation-setting");
  const animationToggle = document.createElement("input");
  animationToggle.type = "checkbox";
  animationToggle.checked = animationEnabled;
  animationToggle.setAttribute("aria-label", "Animate moves and combat");
  animationToggle.onchange = () => {
    animationEnabled = animationToggle.checked;
    localStorage.setItem("bardur-animations", animationEnabled ? "on" : "off");
    if (!animationEnabled) board.finishAnimation();
  };
  animationRow.append(
    animationToggle,
    text("span", "Animate moves and combat"),
  );
  c.append(animationRow);
  const p = text(
    "p",
    "Frozen Bardur 1v1 research rules. Independent local research interface using Midjiwan’s public graphics.",
  );
  c.append(p);
  if (current) {
    const actions = text("div", "", "menu-row");
    actions.append(
      button("Resume", () => $("dialog").close()),
      button("Chronicle", () => history().catch((e) => toast(e.message))),
      button("Export replay", exportReplay),
      button("Save to library", saveReplay),
    );
    if (current.kind === "games")
      actions.append(
        button("Save & close game", async () => {
          await job(`/api/games/${current.id}/close`, {});
          current = null;
          localStorage.removeItem("bardur-current");
          library();
        }),
      );
    if (canAct())
      actions.append(
        button("Tribe relations", relations),
        button("All legal actions", allActions),
      );
    c.append(actions);
  }
  const notice = text(
    "p",
    "NOT AN OFFICIAL Polytopia PRODUCT. NOT APPROVED BY OR ASSOCIATED WITH MIDJIWAN.",
  );
  notice.style.fontSize = "9px";
  c.append(notice);
}
async function library() {
  const c = openDialog();
  heading(
    c,
    "THE CHRONICLE",
    "Replay library",
    "Every saved game can be stepped through one action at a time.",
  );
  const actions = text("div", "", "menu-row");
  actions.append(
    button("＋ Import replay", () => $("import-file").click()),
    button("New game", menu),
  );
  c.append(actions);
  try {
    const lib = await api("/api/library");
    for (const g of lib.games) {
      const b = button(
        "",
        async () => {
          current = { kind: "games", id: g.id };
          localStorage.setItem("bardur-current", JSON.stringify(current));
          $("dialog").close();
          await load();
          botTurn();
        },
        "library-item",
      );
      b.append(text("span", "◉"));
      const title = text("div", "");
      title.append(
        text("strong", g.title),
        text(
          "small",
          `Live game · ${g.frames - 1} actions · Player ${g.current_player} to move`,
        ),
      );
      b.append(title);
      c.append(b);
    }
    for (const r of lib.replays.sort((a, b) => b.created - a.created)) {
      const b = button("", () => openReplay(r.id), "library-item");
      b.append(text("span", "▷"));
      const title = text("div", "");
      title.append(
        text("strong", r.title),
        text(
          "small",
          `${r.frames.toLocaleString()} frames · ${r.complete ? "Recorded outcome" : "In progress"} · ${r.available_views.includes("all") ? "All perspectives" : "Player observations"}`,
        ),
      );
      b.append(title, text("span", "→"));
      c.append(b);
    }
    if (!lib.replays.length && !lib.games.length)
      c.append(text("p", "Start a game or import a .jsonl.gz trace to begin."));
  } catch (e) {
    toast(e.message);
  }
}
function relations() {
  const c = openDialog();
  heading(
    c,
    "TRIBE RELATIONS",
    "The other Bardur",
    "Diplomatic actions are available when the other tribe is known and the required technology is researched.",
  );
  const actions = legal.filter(
    (a) => a.command.kind === 3 && a.command.parameter >= 18,
  );
  if (actions.length && canAct()) {
    const row = text("div", "", "actions");
    row.append(...actions.map(actionButton));
    c.append(row);
  } else c.append(text("p", "No diplomatic actions available."));
}
function allActions() {
  const c = openDialog();
  heading(
    c,
    "AVAILABLE ORDERS",
    "All legal actions",
    "Every action offered by the frozen simulator, including city rewards and resignation.",
  );
  const filter = document.createElement("input");
  filter.placeholder = "Filter actions";
  filter.setAttribute("aria-label", "Filter legal actions");
  c.append(filter);
  const list = text("div", "", "actions");
  list.style.maxHeight = "48vh";
  list.style.justifyContent = "start";
  const show = () => {
    list.replaceChildren();
    for (const a of legal.filter((a) =>
      actLabel(a).toLowerCase().includes(filter.value.toLowerCase()),
    )) {
      const b = actionButton(a);
      if (a.command.source >= 0)
        b.append(text("small", `From ${a.command.source}`));
      if (a.command.target >= 0)
        b.append(text("small", `To ${a.command.target}`));
      list.append(b);
    }
  };
  filter.oninput = show;
  c.append(list);
  show();
}
function techTree() {
  if (!data) return;
  const c = openDialog();
  $("dialog").className = "technology-dialog";
  heading(
    c,
    "KNOWLEDGE IS POWER",
    "Technology",
    "Learn a technology to unlock its neighboring branches. Available research is blue.",
  );
  const v = data.observation.visible_state;
  const tree = text("div", "", "tech-tree");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 600 600");
  svg.setAttribute("aria-hidden", "true");
  tree.append(svg);
  const center = text("div", "", "tech-center");
  const head = document.createElement("img");
  head.src = "/assets/head.png";
  head.alt = "Bardur";
  center.append(head);
  tree.append(center);
  const branches = [
    [1, 2, 3, 4, 5],
    [6, 7, 38, 8, 9],
    [10, 13, 14, 44, 12],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24],
  ];
  const pos = (angle, r) => ({
    x: 300 + Math.cos((angle * Math.PI) / 180) * r,
    y: 300 + Math.sin((angle * Math.PI) / 180) * r,
  });
  const connect = (a, b, owned) => {
    const line = document.createElementNS(svg.namespaceURI, "line");
    for (const [k, val] of Object.entries({
      x1: a.x,
      y1: a.y,
      x2: b.x,
      y2: b.y,
      stroke: owned ? "#9dbeb3" : "#d8e2df",
      "stroke-width": 2,
    }))
      line.setAttribute(k, val);
    svg.append(line);
  };
  const node = (id, p, parent) => {
    const t = meta.rules.technologies.find((t) => t.id === id),
      owned = v.technology.includes(id),
      a = legal.find((a) => a.command.kind === 7 && a.command.parameter === id);
    connect(parent, p, owned);
    const b = button(
      "",
      () => techDetail(t, owned, a),
      "tech" + (owned ? " researched" : a ? " available" : ""),
    );
    b.style.left = p.x / 6 + "%";
    b.style.top = p.y / 6 + "%";
    const icon = text("span", "", "tech-icon");
    const img = document.createElement("img");
    const images = {
      1: "unit-rider",
      2: "temple",
      3: "unit-knight",
      4: "ground",
      5: "market",
      6: "fruit",
      7: "unit-defender",
      38: "unit-cloak",
      8: "farm",
      9: "windmill",
      10: "water",
      13: "unit-scoutship",
      14: "unit-bombership",
      44: "unit-rammership",
      12: "temple",
      15: "animal",
      16: "lumber",
      17: "unit-catapult",
      18: "unit-archer",
      19: "forest",
      20: "mountain",
      21: "temple",
      22: "unit-mindbender",
      23: "mine",
      24: "unit-swordsman",
    };
    img.src =
      "/assets/" +
      (images[id] === "road" ? "ground" : images[id] || "star") +
      ".png";
    img.alt = "";
    icon.append(img);
    icon.setAttribute("aria-hidden", "true");
    b.append(
      icon,
      text("span", titleCase(t.name)),
      text("small", owned ? "Learned" : a ? cost(a) : `Tier ${t.tier}`),
    );
    tree.append(b);
  };
  branches.forEach((ids, i) => {
    const angle = -90 + i * 72,
      p1 = pos(angle, 85);
    node(ids[0], p1, { x: 300, y: 300 });
    for (let side = 0; side < 2; side++) {
      const a = angle + (side ? 17 : -17),
        p2 = pos(a, 165);
      node(ids[1 + side * 2], p2, p1);
      node(ids[2 + side * 2], pos(a, 250), p2);
    }
  });
  c.append(tree, text("div", "", "tech-detail hidden"));
  c.querySelector(".tech-detail").id = "tech-detail";
}
// Mobile flow: tapping a technology explains it before you pay for it.
function techDetail(t, owned, a) {
  const box = $("tech-detail");
  box.classList.remove("hidden");
  const names = (group, ids) =>
    ids
      .map((id) => meta.rules[group].find((x) => x.id === id)?.name)
      .filter(Boolean)
      .map(titleCase);
  const unlocks = [
    ...names("units", t.units || []),
    ...names("structures", t.structures || []),
  ];
  box.replaceChildren(
    text("h3", titleCase(t.name)),
    text(
      "p",
      unlocks.length
        ? "Unlocks " + unlocks.join(", ") + "."
        : "Opens the next branch of the tree.",
    ),
  );
  if (owned) box.append(text("p", "You have learned this technology.", "tech-state"));
  else if (a && canAct()) {
    const go = button(`Research · ${cost(a)}`, () => takeAction(a), "research-button");
    go.dataset.actionId = a.action_id;
    box.append(go);
  } else
    box.append(
      text(
        "p",
        canAct()
          ? "Learn a neighbouring technology or save up more stars first."
          : "Research is available on your turn.",
        "tech-state",
      ),
    );
}

function details() {
  if (!data) return;
  const c = openDialog();
  heading(
    c,
    "MATCH DETAILS",
    data.title,
    "The same public observation and legal-action interface powers the human and model players.",
  );
  const t = text("table", "", "detail-table");
  const pairs = [
    ["Rules", "bardur-1v1-v1"],
    ["Mode", data.mode],
    ["Checkpoint", data.bot?.name || "See imported trace provenance"],
    ["Training", data.bot?.training || "Recorded experiment"],
    [
      "Checkpoint hash",
      data.bot?.checkpoint_sha256 || "Stored in original experiment protocol",
    ],
    ["Actions", data.frames - 1],
    ["Perspective", data.perspective],
    [
      "Controls",
      "Click a unit, then a highlighted tile. Drag to pan; scroll to zoom.",
    ],
    [
      "Agent endpoint",
      current?.kind === "games"
        ? `GET /api/games/${current.id}/observation`
        : "Start “Connect a model” to get an agent seat.",
    ],
  ];
  for (const [k, v] of pairs) {
    const tr = document.createElement("tr");
    tr.append(text("td", k), text("td", String(v)));
    t.append(tr);
  }
  c.append(t);
  if (data.note) c.append(text("p", data.note));
  c.append(button("Export replay", exportReplay));
}
async function history() {
  if (!data) return;
  const c = openDialog();
  heading(
    c,
    "THE CHRONICLE",
    "Recent moves",
    "Select a move to return to that position.",
  );
  const start = Math.max(0, data.frame_index - 19),
    stop = data.frame_index;
  for (let i = stop; i >= start; i--) {
    const f = await api(
      `/api/${current.kind}/${current.id}?frame=${i}&view=${$("perspective").value}`,
    );
    const ev = f.event;
    const row = button(
      "",
      () => {
        $("dialog").close();
        load(i);
      },
      "history-row",
    );
    row.append(
      text("small", String(i).padStart(4, "0")),
      text(
        "span",
        ev
          ? `Player ${ev.player} · ${titleCase(ev.action.label)}`
          : "Opening position",
      ),
    );
    c.append(row);
  }
}
function exportReplay() {
  if (!current) return;
  const a = document.createElement("a");
  a.href = `/api/${current.kind}/${current.id}/export`;
  a.download = "bardur-replay.json";
  a.click();
}
async function saveReplay() {
  if (current?.kind !== "games")
    return toast("This match is already in the library.");
  try {
    await job(`/api/games/${current.id}/save`, {});
    toast("Replay saved to the library.");
  } catch (e) {
    toast(e.message);
  }
}
$("home").onclick = $("menu-button").onclick = menu;
$("library").onclick = library;
$("dialog-close").onclick = () => $("dialog").close();
$("dialog").addEventListener("click", (e) => {
  if (e.target === $("dialog")) {
    const r = $("dialog").getBoundingClientRect();
    if (
      e.clientX < r.left ||
      e.clientX > r.right ||
      e.clientY < r.top ||
      e.clientY > r.bottom
    )
      $("dialog").close();
  }
});
$("clear-selection").onclick = () => {
  selected = null;
  layer = null;
  render();
};
$("zoom-in").onclick = () => board.zoomBy(1.2);
$("zoom-out").onclick = () => board.zoomBy(1 / 1.2);
$("recenter").onclick = () => board.center();
$("tech-button").onclick = techTree;
$("history-button").onclick = details;
$("info-toggle").onclick = details;
$("end-turn").onclick = () => {
  const a = legal.find((a) => a.command.kind === 10);
  if (a) takeAction(a);
};
$("play").onclick = togglePlay;
const seek = (i, animate = false) => {
  playing = false;
  updatePlay();
  load(Math.max(0, Math.min(i, data.frames - 1)), null, animate).catch((e) =>
    toast(e.message),
  );
};
$("first").onclick = () => seek(0);
$("prev").onclick = () => seek(data.frame_index - 1);
$("next").onclick = () => seek(data.frame_index + 1, true);
$("last").onclick = () => seek(data.frames - 1);
$("scrubber").oninput = (e) => seek(Number(e.target.value));
$("perspective").onchange = () => load(data.frame_index);
$("speed").onchange = () => {
  clearTimeout(timer);
  if (playing && !ticking) tick();
};
window.addEventListener("keydown", (e) => {
  if (
    $("dialog").open ||
    ["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)
  )
    return;
  if (e.code === "Space") {
    e.preventDefault();
    togglePlay();
  }
  if (e.code === "ArrowRight") seek(data.frame_index + 1, true);
  if (e.code === "ArrowLeft") seek(data.frame_index - 1);
  if (e.code === "Escape" && !rewardOpen()) {
    selected = null;
    layer = null;
    render();
  }
});
$("import-file").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    if (file.size > 32 * 1024 * 1024)
      throw Error("Choose a replay smaller than 32 MiB.");
    status("Importing recorded game…", true);
    const bytes = new Uint8Array(await file.arrayBuffer());
    let encoded = "";
    for (let i = 0; i < bytes.length; i += 0x8000)
      encoded += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    const r = await job("/api/import", {
      name: file.name,
      data: btoa(encoded),
    });
    await openReplay(r.replay_id);
    toast(`Imported ${r.frames.toLocaleString()} frames.`);
  } catch (e) {
    toast(e.message);
  }
  e.target.value = "";
};
// Model integrations can use actual DOM controls or the same versioned action API.
window.bardur = {
  get game() {
    return current ? { ...current } : null;
  },
  get animating() {
    return board.animating;
  },
  get busy() {
    return busy || board.animating;
  },
  whenIdle: () => board.whenIdle(),
  observe: () =>
    current?.kind === "games"
      ? api(`/api/games/${current.id}/observation`)
      : Promise.reject(Error("Start an agent game first")),
  act: async (action_id, expected_state_version) => {
    if (current?.kind !== "games") throw Error("No live game");
    if (busy || board.animating)
      throw Error("Wait for the current move to finish");
    busy = true;
    render();
    try {
      await job(`/api/games/${current.id}/action`, {
        action_id: String(action_id),
        expected_state_version,
      });
      await load();
    } finally {
      busy = false;
      render();
    }
    await botTurn();
    return window.bardur.observe();
  },
  controls: () => ({
    buttons: [...document.querySelectorAll("button,select,input")]
      .filter(
        (e) => e.offsetWidth && e.offsetHeight && !e.closest("#tile-controls"),
      )
      .map((e) => ({
        label: e.getAttribute("aria-label") || e.textContent,
        disabled: e.disabled,
        rect: e.getBoundingClientRect().toJSON(),
      })),
    tiles: [...$("tile-controls").children].map((e) => ({
      tile: Number(e.dataset.tile),
      label: e.getAttribute("aria-label"),
      rect: e.getBoundingClientRect().toJSON(),
    })),
    actions: [...document.querySelectorAll("[data-action-id]")].map((e) => ({
      action_id: e.dataset.actionId,
      label: e.textContent,
      rect: e.getBoundingClientRect().toJSON(),
    })),
  }),
  refresh: () => load(),
  selectTile,
};
async function init() {
  try {
    meta = await api("/api/meta");
    const saved = JSON.parse(localStorage.getItem("bardur-current") || "null");
    const lib = await api("/api/library");
    if (saved) {
      current = saved;
      try {
        await load(saved.kind === "replays" ? 0 : null);
      } catch {
        current = null;
      }
    }
    if (!current && lib.replays.length)
      await openReplay(
        (
          lib.replays
            .sort((a, b) => b.created - a.created)
            .find((r) => r.source === "verified-engine-replay") ||
          lib.replays[0]
        ).id,
      );
    if (!current) menu();
    if (current?.kind === "games") botTurn();
    if (!meta.ready) {
      const warm = async () => {
        const j = await api("/api/jobs/" + meta.warmup_job);
        if (j.status === "done") {
          meta.ready = true;
          if (!current) status("Engine ready · choose a game");
        } else if (j.status === "error") {
          toast("Engine startup failed: " + j.error);
        } else setTimeout(warm, 1500);
      };
      warm();
    }
  } catch (e) {
    status("Connection failed · " + e.message);
    toast(e.message);
  }
}
init();
