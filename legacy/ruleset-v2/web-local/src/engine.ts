// Typed wrapper over engine-wasm's C ABI (engine-wasm/src/lib.rs): inputs go through `input`, JSON results come
// back through output_ptr/output_len, and a negative return code means the output holds {"error": ...}.
import type { Command, LegalAction, Observation, Rules } from "../api/types.ts";

interface Exports {
  memory: WebAssembly.Memory;
  input(len: number): number;
  output_ptr(): number;
  output_len(): number;
  load_model(): number;
  rules_json(): number;
  game_clone(id: number): number;
  game_decode(id: number, index: number): number;
  game_new(seed: number): number;
  game_free(id: number): void;
  game_status(id: number): number;
  game_view(id: number, player: number): number;
  game_legal(id: number): number;
  game_step(id: number, index: number): number;
  bot_act(id: number, deterministic: number, seed: number): number;
  bot_plan(id: number, lanes: number, seed: number): number;
  game_from_actions(seed: number): number;
  search_lanes(id: number, round: number, sample: number, seed: number, from: number, to: number): number;
  search_commit(id: number, round: number, sample: number, seed: number): number;
}

export interface Status {
  current_player: number;
  turn: number;
  state_version: number;
  terminated: boolean;
  truncated: boolean;
  rewards: [number, number];
}

export interface BotDecision {
  action_id: string;
  value: number;
  log_probability: number;
}

export interface BotPlan {
  plan: string[];
  plans: number;
  overrides: number;
  fallbacks: number;
  value: number | null;
}

export interface LaneResult {
  lane: number;
  actions: string[];
  score: number;
  steps: number;
}

const decoder = new TextDecoder();

export class Engine {
  private x: Exports;
  private constructor(x: Exports) {
    this.x = x;
  }

  static async load(wasm: BufferSource | Response | Promise<Response>): Promise<Engine> {
    const source = wasm instanceof Response || wasm instanceof Promise ? await (await wasm).arrayBuffer() : wasm;
    const { instance } = await WebAssembly.instantiate(source, {});
    return new Engine(instance.exports as unknown as Exports);
  }

  private out<T>(code: number): T {
    const bytes = new Uint8Array(this.x.memory.buffer, this.x.output_ptr(), this.x.output_len());
    const value = JSON.parse(decoder.decode(bytes));
    if (code < 0) throw Error(value.error);
    return value as T;
  }

  private write(json: unknown) {
    const bytes = new TextEncoder().encode(JSON.stringify(json));
    const ptr = this.x.input(bytes.length); // may grow memory: take the buffer after it
    new Uint8Array(this.x.memory.buffer, ptr, bytes.length).set(bytes);
  }

  loadModel(weights: ArrayBuffer) {
    const ptr = this.x.input(weights.byteLength);
    new Uint8Array(this.x.memory.buffer, ptr, weights.byteLength).set(new Uint8Array(weights));
    this.out(this.x.load_model());
  }

  rules(): Rules {
    return this.out(this.x.rules_json());
  }

  newGame(seed: number): number {
    return this.x.game_new(seed);
  }
  clone(id: number): number {
    const copy = this.x.game_clone(id);
    if (copy < 0) throw Error("unknown game");
    return copy;
  }
  decode(id: number, actionId: string): Command {
    return this.out(this.x.game_decode(id, Number(actionId)));
  }
  free(id: number) {
    this.x.game_free(id);
  }
  status(id: number): Status {
    return this.out(this.x.game_status(id));
  }
  /** Seat 1 or 2: that player's fogged observation; 0: the omniscient spectator view. */
  view(id: number, player: number): Observation {
    return this.out(this.x.game_view(id, player));
  }
  legal(id: number): LegalAction[] {
    return this.out(this.x.game_legal(id));
  }
  step(id: number, actionId: string): Status {
    return this.out(this.x.game_step(id, Number(actionId)));
  }
  act(id: number, deterministic: boolean, seed: number): BotDecision {
    return this.out(this.x.bot_act(id, deterministic ? 1 : 0, seed >>> 0));
  }
  /** Turn search with every lane on this thread. */
  plan(id: number, lanes: number, seed: number): BotPlan {
    return this.out(this.x.bot_plan(id, lanes, seed >>> 0));
  }
  /** A game rebuilt from its seed and catalog entries. */
  fromActions(seed: number, actions: string[]): number {
    this.write(actions);
    return this.out<{ id: number }>(this.x.game_from_actions(seed)).id;
  }
  /** Search lanes [from, to) of a plan round (engine-wasm search_lanes). */
  lanes(id: number, round: number, sample: boolean, seed: number, from: number, to: number): LaneResult[] {
    return this.out(this.x.search_lanes(id, round, sample ? 1 : 0, seed >>> 0, from, to));
  }
  /** Commits a round's chosen turn to game `id`; returns the entries applied (none: the sample was rejected). */
  commit(id: number, round: number, sample: boolean, seed: number, chosen: string[]): string[] {
    this.write(chosen);
    return this.out<{ applied: string[] }>(this.x.search_commit(id, round, sample ? 1 : 0, seed >>> 0)).applied;
  }
}
