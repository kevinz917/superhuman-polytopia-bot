// Game flow, independent of React: loading frames, selection, actions, the
// checkpoint opponent, replay playback. The UI subscribes; the board is driven
// imperatively so frame-rate work never passes through React.
import { api, job, waitJob } from "../api/client.ts";
import type { Current, FrameData, Layer, LegalAction, Library, Meta, Series } from "../api/types.ts";
import type { BoardScene } from "../board/BoardScene.ts";
import { cityTerritory } from "./vocab.ts";

export type DialogState =
  | { type: "menu" }
  | { type: "library" }
  | { type: "details" }
  | { type: "chronicle" }
  | { type: "relations" }
  | { type: "actions" }
  | { type: "tech" }
  | { type: "resign"; action: LegalAction }
  | null;

const storage = {
  get(key: string) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, value: string | null) {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch {
      // Private mode: preferences simply do not persist.
    }
  },
};

export class GameController {
  meta: Meta | null = null;
  current: Current | null = null;
  data: FrameData | null = null;
  legal: LegalAction[] = [];
  selected: number | null = null;
  layer: Layer | null = null;
  playing = false;
  busy = false;
  animating = false;
  speed = 1;
  perspective = "actor";
  animationEnabled = storage.get("bardur-animations") !== "off";
  // Opponent mode for new games; the live game's own setting is data.search.
  searchPreferred = storage.get("bardur-search") !== "off";
  dialog: DialogState = null;
  toastMessage: { text: string; id: number } | null = null;
  statusOverride: string | null = null;
  board: BoardScene | null = null;
  // Where the scrubber thumb is while frames for it are still loading.
  scrubTarget: number | null = null;
  // Replay insights panel: per-turn economy of the open replay, and whether the panel is shown.
  series: Series | null = null;
  insightsOpen = storage.get("bardur-insights") !== "off";
  private seriesFor: string | null = null;

  private listeners = new Set<() => void>();
  private version = 0;
  private frameRequest = 0;
  private timer = 0;
  private ticking = false;
  private seeking = false;
  // Replay frames never change, so revisiting one while scrubbing is free.
  private frames = new Map<string, FrameData>();

  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };
  getVersion = () => this.version;
  emit() {
    this.version++;
    for (const fn of this.listeners) fn();
  }

  attachBoard(board: BoardScene, canvas: HTMLCanvasElement) {
    this.board = board;
    canvas.addEventListener("bardur:animation-start", () => {
      this.animating = true;
      this.emit();
    });
    canvas.addEventListener("bardur:animation-end", () => {
      this.animating = false;
      this.emit();
    });
    if (this.data) this.refresh({ animate: false });
  }

  toast(text: string) {
    this.toastMessage = { text, id: Date.now() };
    this.emit();
  }
  openDialog(dialog: DialogState) {
    this.dialog = dialog;
    if (dialog) this.playing = false;
    this.emit();
  }

  canAct() {
    const d = this.data;
    return (
      this.current?.kind === "games" &&
      d?.mode !== "watch" &&
      !this.busy &&
      !this.board?.animating &&
      d?.current_player === d?.seat &&
      !d?.terminated &&
      !d?.truncated &&
      d?.frame_index === (d?.frames ?? 0) - 1
    );
  }

  layersAt(tile: number): Layer[] {
    const v = this.data?.observation.visible_state,
      t = v?.tiles[tile];
    if (!v || !t?.known) return ["tile"];
    const out: Layer[] = [];
    if (v.units.some((u) => u.tile === tile)) out.push("unit");
    out.push(v.cities.some((c) => c.tile === tile) ? "city" : "tile");
    return out;
  }

  // Units with something left to do this turn: a move, an attack, a capture,
  // or recover / heal others / promote. Disband is always legal, so it does not count.
  readyUnits() {
    if (!this.canAct()) return [];
    const tiles = new Set<number>();
    for (const { command: c } of this.legal)
      if ([1, 2, 8].includes(c.kind) || (c.kind === 3 && c.source >= 0 && [7, 9, 16].includes(c.parameter)))
        tiles.add(c.source);
    return [...tiles];
  }

  pendingReward() {
    return this.canAct() ? this.legal.find((a) => a.command.kind === 9) : undefined;
  }

  // Push the current selection and frame to the board.
  refresh(options: { animate?: boolean } = {}) {
    const data = this.data;
    if (!data) return this.emit();
    const v = data.observation.visible_state;
    const reward = this.pendingReward();
    if (reward) {
      this.selected = reward.command.target;
      this.layer = "city";
    }
    if (this.selected !== null && !this.layersAt(this.selected).includes(this.layer!))
      this.layer = this.layersAt(this.selected)[0];
    const targets =
      this.canAct() && this.selected !== null && this.layer === "unit"
        ? this.legal.filter((a) => [1, 2].includes(a.command.kind) && a.command.source === this.selected)
        : [];
    const ownerCity = this.layer === "city" && v.cities.find((c) => c.tile === this.selected && c.owner);
    const board = this.board;
    if (board) {
      board.update(data.observation, this.selected, targets, data.event, {
        scene: `${this.current?.kind}/${this.current?.id}/${data.perspective}/${data.perspective === "all" ? "" : data.observation.player}`,
        animate: this.animationEnabled && options.animate !== false,
        speed: ["replay", "watch"].includes(data.mode) ? this.speed : 1,
        layer: this.selected === null ? null : this.layer,
        home: ["play", "agent"].includes(data.mode) ? (data.seat ?? 1) : 1,
        territory: ownerCity
          ? data.observation.hud?.territory?.[String(this.selected)] || cityTerritory(v, this.selected!)
          : [],
        ready: this.readyUnits(),
      });
      if (board.animating) {
        // Follow the action like the mobile camera when it happens off-screen.
        const cue = data.event?.animation || data.event?.action?.command || {};
        const spot = [cue.target, cue.source].find((t) => t !== undefined && t >= 0);
        if (spot !== undefined) board.focus(spot);
      }
    }
    this.emit();
  }

  private async fetchFrame(url: string) {
    const cached = this.frames.get(url);
    if (cached) {
      this.frames.delete(url);
      this.frames.set(url, cached);
      return cached;
    }
    const next = await api<FrameData>(url);
    if (this.current?.kind === "replays" && !next.job_id) {
      this.frames.set(url, next);
      if (this.frames.size > 240) this.frames.delete(this.frames.keys().next().value!);
    }
    return next;
  }

  // Resolves true once this frame is on the board, false if a newer load won.
  async load(index: number | null = null, view: string | null = null, animate = true): Promise<boolean> {
    if (!this.current) return false;
    const request = ++this.frameRequest;
    if (!animate) this.board?.finishAnimation();
    let url = `/api/${this.current.kind}/${this.current.id}?view=${view || this.perspective || "actor"}`;
    if (index !== null) url += `&frame=${index}`;
    const next = await this.fetchFrame(url);
    if (request !== this.frameRequest) return false;
    if (next.job_id) {
      await waitJob(next);
      return this.load(index, view, animate);
    }
    this.data = next;
    this.perspective = next.perspective;
    this.legal = [];
    if (
      this.current.kind === "games" &&
      next.mode !== "watch" &&
      !next.terminated &&
      !next.truncated &&
      next.frame_index === next.frames - 1
    ) {
      const obs = await api<{ legal_actions: LegalAction[] }>(`/api/games/${this.current.id}/observation`);
      if (request !== this.frameRequest) return false;
      this.legal = obs.legal_actions;
    }
    if (this.current.kind === "replays" && this.seriesFor !== this.current.id) void this.loadSeries(this.current.id);
    this.refresh({ animate });
    if (this.board?.animating) {
      await this.board.whenIdle();
      if (request === this.frameRequest) this.refresh();
    }
    return true;
  }

  private async loadSeries(id: string) {
    this.seriesFor = id;
    this.series = null;
    try {
      const series = await api<Series>(`/api/replays/${id}/series`);
      if (this.seriesFor === id) this.series = series;
    } catch {
      // Older servers have no series endpoint: the panel stays hidden.
      if (this.seriesFor === id) this.series = { available: false };
    }
    this.emit();
  }

  setInsightsOpen(open: boolean) {
    this.insightsOpen = open;
    storage.set("bardur-insights", open ? null : "off");
    this.emit();
  }

  selectTile = (tile: number) => {
    if (this.pendingReward() && !this.busy && !this.board?.animating) return;
    if (this.canAct() && this.selected !== null && this.layer === "unit") {
      const actions = this.legal.filter(
        (a) => [1, 2].includes(a.command.kind) && a.command.source === this.selected && a.command.target === tile,
      );
      if (actions.length === 1) {
        void this.takeAction(actions[0]);
        return;
      }
    }
    // Mobile tap cycle: unit, then the city or tile beneath it, then clear.
    const layers = this.data ? this.layersAt(tile) : (["tile"] as Layer[]);
    if (tile === this.selected) {
      const next = layers.indexOf(this.layer!) + 1;
      if (next >= layers.length) {
        this.selected = null;
        this.layer = null;
      } else this.layer = layers[next];
    } else {
      this.selected = tile;
      this.layer = layers[0];
      // Keep the selection (and its move targets) on screen, e.g. for agents.
      this.board?.focus(tile);
    }
    this.refresh();
  };

  clearSelection() {
    if (this.pendingReward()) return;
    this.selected = null;
    this.layer = null;
    this.refresh();
  }

  async takeAction(a: LegalAction) {
    if (!this.canAct()) return;
    if (a.command.kind === 11) return this.openDialog({ type: "resign", action: a });
    await this.perform(a);
  }

  async perform(a: LegalAction) {
    const id = this.current!.id;
    this.busy = true;
    this.dialog = null;
    this.statusOverride = "Applying move…";
    this.emit();
    try {
      await job(`/api/games/${id}/action`, {
        action_id: a.action_id,
        expected_state_version: this.data!.observation.state_version,
      });
      this.selected = null;
      this.statusOverride = null;
      await this.load();
    } catch (e) {
      this.statusOverride = null;
      this.toast((e as Error).message);
      await this.load();
    } finally {
      this.busy = false;
      this.refresh();
    }
    await this.botTurn();
  }

  async botTurn() {
    const d = this.data;
    if (
      !this.current ||
      this.current.kind !== "games" ||
      this.busy ||
      !d ||
      d.terminated ||
      d.truncated ||
      d.mode === "watch" ||
      d.current_player === d.seat
    )
      return;
    this.busy = true;
    this.emit();
    const id = this.current.id;
    try {
      while (
        this.current?.id === id &&
        !this.data!.terminated &&
        !this.data!.truncated &&
        this.data!.current_player !== this.data!.seat
      ) {
        await job(`/api/games/${id}/advance`, { count: 1 });
        await this.load();
        await new Promise((r) => setTimeout(r, 130));
      }
    } catch (e) {
      this.toast((e as Error).message);
      this.statusOverride = "Bot paused after an error · reopen the game to retry";
    } finally {
      this.busy = false;
      this.refresh();
    }
  }

  async start(mode: "play" | "watch" | "agent", seed?: number, seat = 1, opponent?: { name: string; seat: number }) {
    if (this.busy) return this.toast("Wait for the current move to finish.");
    const params: Record<string, number | string | boolean> = { mode, seat };
    if (this.meta?.search?.available && mode !== "agent") params.search = this.searchPreferred;
    if (mode === "watch" && opponent) {
      params.opponent = opponent.name;
      params.opponent_seat = opponent.seat;
    }
    if (seed !== undefined && !Number.isNaN(seed)) params.seed = seed;
    this.dialog = null;
    this.playing = false;
    this.busy = true;
    this.statusOverride = this.meta?.ready
      ? "Setting up the square…"
      : "Preparing engine and checkpoint · first start can take a minute";
    this.emit();
    try {
      const result = await job<{ game_id: string }>("/api/games", params);
      this.current = { kind: "games", id: result.game_id };
      storage.set("bardur-current", JSON.stringify(this.current));
      this.selected = null;
      this.perspective = mode === "watch" ? "all" : String(seat);
      await this.load(null, this.perspective);
      this.board?.center();
      this.busy = false;
      this.statusOverride = null;
      this.refresh();
      if (mode === "watch") {
        this.playing = true;
        this.emit();
        void this.tick();
      } else await this.botTurn();
    } catch (e) {
      this.toast((e as Error).message);
      this.busy = false;
      this.statusOverride = "Could not start game · " + (e as Error).message;
      this.emit();
    }
  }

  async openReplay(id: string) {
    this.playing = false;
    this.current = { kind: "replays", id };
    this.selected = null;
    this.legal = [];
    storage.set("bardur-current", JSON.stringify(this.current));
    this.dialog = null;
    await this.load(0, "all");
    this.board?.center();
  }

  async openGame(id: string) {
    this.current = { kind: "games", id };
    storage.set("bardur-current", JSON.stringify(this.current));
    this.dialog = null;
    await this.load();
    void this.botTurn();
  }

  async tick(): Promise<void> {
    clearTimeout(this.timer);
    if (!this.playing || !this.current || this.ticking || !this.data) return;
    this.ticking = true;
    try {
      if (this.data.frame_index < this.data.frames - 1) await this.load(this.data.frame_index + 1);
      else if (this.current.kind === "games" && this.data.mode === "watch" && !this.data.terminated && !this.data.truncated) {
        if (!this.busy) {
          this.busy = true;
          await job(`/api/games/${this.current.id}/advance`, { count: 1 });
          this.busy = false;
          await this.load();
        }
      } else {
        this.playing = false;
        this.emit();
        return;
      }
    } catch (e) {
      this.playing = false;
      this.busy = false;
      this.toast((e as Error).message);
      return;
    } finally {
      this.ticking = false;
    }
    if (this.playing) this.timer = window.setTimeout(() => this.tick(), Math.max(40, 180 / this.speed));
  }

  togglePlay = () => {
    const d = this.data;
    if (!this.current || !d) return;
    if (d.mode !== "watch" && this.current.kind === "games" && d.frame_index === d.frames - 1)
      return this.toast("Your game is live. Select a unit to play.");
    this.playing = !this.playing;
    this.emit();
    if (this.playing) {
      if (d.mode === "replay" && d.frame_index === d.frames - 1) void this.load(0).then(() => this.tick());
      else void this.tick();
    }
  };

  // Scrubbing fires far faster than frames arrive: keep one load in flight
  // and always continue with the newest position instead of queueing them.
  seek = (i: number, animate = false) => {
    if (!this.data) return;
    this.playing = false;
    const target = Math.max(0, Math.min(i, this.data.frames - 1));
    if (animate) {
      this.emit();
      this.load(target, null, true).catch((e) => this.toast(e.message));
      return;
    }
    this.scrubTarget = target;
    this.emit();
    if (this.seeking) return;
    this.seeking = true;
    void (async () => {
      try {
        // Stop once the newest target was loaded, whatever frame_index the server
        // echoes; comparing against the echo spun forever (on cached frames
        // without yielding) when the two disagreed.
        let loaded: number | null = null;
        while (this.scrubTarget !== null && this.scrubTarget !== loaded) {
          loaded = this.scrubTarget;
          if (!(await this.load(loaded, null, false))) break;
        }
      } catch (e) {
        this.toast((e as Error).message);
      } finally {
        this.seeking = false;
        this.scrubTarget = null;
        this.emit();
      }
    })();
  };

  setSpeed(speed: number) {
    this.speed = speed;
    clearTimeout(this.timer);
    if (this.playing && !this.ticking) void this.tick();
    this.emit();
  }
  setPerspective(view: string) {
    this.perspective = view;
    void this.load(this.data?.frame_index ?? null, view);
  }
  botName() {
    return this.meta?.bot.name.split(" · ").pop() ?? "The bot";
  }

  // Switches the opponent between the raw policy and turn search. Takes effect
  // from the bot's next sub-action, so it works mid-game and mid-bot-turn.
  async setSearch(on: boolean) {
    this.searchPreferred = on;
    storage.set("bardur-search", on ? "on" : "off");
    const d = this.data;
    if (this.current?.kind === "games" && d && d.mode !== "agent") {
      try {
        await api(`/api/games/${this.current.id}/settings`, { search: on });
        if (this.data === d) this.data = { ...d, search: on };
        this.toast(on ? "Opponent now searches each turn." : "Opponent now plays its raw policy.");
      } catch (e) {
        this.toast((e as Error).message);
      }
    }
    this.emit();
  }

  setAnimationEnabled(on: boolean) {
    this.animationEnabled = on;
    storage.set("bardur-animations", on ? "on" : "off");
    if (!on) this.board?.finishAnimation();
    this.emit();
  }

  exportReplay() {
    if (!this.current) return;
    const a = document.createElement("a");
    a.href = `/api/${this.current.kind}/${this.current.id}/export`;
    a.download = "bardur-replay.json";
    a.click();
  }
  async saveReplay() {
    if (this.current?.kind !== "games") return this.toast("This match is already in the library.");
    try {
      await job(`/api/games/${this.current.id}/save`, {});
      this.toast("Replay saved to the library.");
    } catch (e) {
      this.toast((e as Error).message);
    }
  }
  async closeGame() {
    if (this.current?.kind !== "games") return;
    await job(`/api/games/${this.current.id}/close`, {});
    this.current = null;
    this.data = null;
    storage.set("bardur-current", null);
    this.openDialog({ type: "library" });
  }
  async importFile(file: File) {
    try {
      if (file.size > 32 * 1024 * 1024) throw Error("Choose a replay smaller than 32 MiB.");
      this.statusOverride = "Importing recorded game…";
      this.emit();
      const bytes = new Uint8Array(await file.arrayBuffer());
      let encoded = "";
      for (let i = 0; i < bytes.length; i += 0x8000) encoded += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      const r = await job<{ replay_id: string; frames: number }>("/api/import", { name: file.name, data: btoa(encoded) });
      this.statusOverride = null;
      await this.openReplay(r.replay_id);
      this.toast(`Imported ${r.frames.toLocaleString()} frames.`);
    } catch (e) {
      this.statusOverride = null;
      this.toast((e as Error).message);
    }
  }

  async init() {
    try {
      this.meta = await api<Meta>("/api/meta");
      this.emit();
      const saved = JSON.parse(storage.get("bardur-current") || "null") as Current | null;
      const lib = await api<Library>("/api/library");
      if (saved) {
        this.current = saved;
        try {
          await this.load(saved.kind === "replays" ? 0 : null, saved.kind === "replays" ? "all" : null);
        } catch {
          this.current = null;
        }
      }
      if (!this.current && lib.replays.length) {
        const pick =
          lib.replays.find((r) => r.featured) ||
          [...lib.replays].sort((a, b) => b.created - a.created).find((r) => r.source === "verified-engine-replay") ||
          lib.replays[0];
        await this.openReplay(pick.id);
      }
      if (!this.current) this.openDialog({ type: "menu" });
      if (this.current?.kind === "games") void this.botTurn();
      if (!this.meta.ready && this.meta.warmup_job) {
        const warm = async () => {
          const j = await api<{ status: string; error?: string }>("/api/jobs/" + this.meta!.warmup_job);
          if (j.status === "done") {
            this.meta!.ready = true;
            if (!this.current) this.statusOverride = "Engine ready · choose a game";
            this.emit();
          } else if (j.status === "error") this.toast("Engine startup failed: " + j.error);
          else setTimeout(warm, 1500);
        };
        void warm();
      }
    } catch (e) {
      this.statusOverride = "Connection failed · " + (e as Error).message;
      this.toast((e as Error).message);
    }
  }

  // Model integrations can use the real DOM controls or this versioned API.
  exposeAgentApi() {
    const bardur = {
      whenIdle: () => this.board?.whenIdle(),
      observe: () =>
        this.current?.kind === "games"
          ? api(`/api/games/${this.current.id}/observation`)
          : Promise.reject(Error("Start an agent game first")),
      act: async (action_id: string | number, expected_state_version: number) => {
        if (this.current?.kind !== "games") throw Error("No live game");
        if (this.busy || this.board?.animating) throw Error("Wait for the current move to finish");
        this.busy = true;
        this.emit();
        try {
          await job(`/api/games/${this.current.id}/action`, { action_id: String(action_id), expected_state_version });
          await this.load();
        } finally {
          this.busy = false;
          this.refresh();
        }
        await this.botTurn();
        return api(`/api/games/${this.current!.id}/observation`);
      },
      controls: () => ({
        buttons: [...document.querySelectorAll<HTMLElement>("button,select,input")]
          .filter((e) => e.offsetWidth && e.offsetHeight && !e.closest("#tile-controls"))
          .map((e) => ({
            label: e.getAttribute("aria-label") || e.textContent,
            disabled: (e as HTMLButtonElement).disabled,
            rect: e.getBoundingClientRect().toJSON(),
          })),
        tiles: [...(document.getElementById("tile-controls")?.children ?? [])].map((e) => ({
          tile: Number((e as HTMLElement).dataset.tile),
          label: e.getAttribute("aria-label"),
          rect: e.getBoundingClientRect().toJSON(),
        })),
        actions: [...document.querySelectorAll<HTMLElement>("[data-action-id]")].map((e) => ({
          action_id: e.dataset.actionId,
          label: e.textContent,
          rect: e.getBoundingClientRect().toJSON(),
        })),
      }),
      refresh: () => this.load(),
      selectTile: (tile: number) => this.selectTile(tile),
    };
    Object.defineProperties(bardur, {
      game: { get: () => (this.current ? { ...this.current } : null) },
      animating: { get: () => !!this.board?.animating },
      busy: { get: () => this.busy || !!this.board?.animating },
    });
    (window as unknown as { bardur: typeof bardur }).bardur = bardur;
  }
}

export const controller = new GameController();
