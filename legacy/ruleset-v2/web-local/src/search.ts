// Turn search with its lanes spread over Web Workers: the rounds of engine-wasm search.rs::plan_turn, where each
// round's lanes run on the pool (search_worker.ts) and the chosen turn is committed here. A round's sampled
// world depends only on (state, seed, round), so every worker searches the same one, and the plan equals the
// single-thread plan_turn's.
import type { BotPlan, Engine, LaneResult } from "./engine.ts";

const MAX_REJECTED = 8;

interface Job {
  seed: number;
  history: string[];
  round: number;
  sample: boolean;
  planSeed: number;
  from: number;
  to: number;
}

export class SearchPool {
  readonly size: number;
  private workers: Worker[] = [];
  private calls = new Map<number, { resolve: (v: LaneResult[]) => void; reject: (e: Error) => void }>();
  private next = 0;

  constructor(wasm: ArrayBuffer, weights: ArrayBuffer) {
    const cores = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
    this.size = typeof Worker === "undefined" ? 0 : Math.max(1, Math.min(cores - 1, 8));
    for (let i = 0; i < this.size; i++) {
      const w = new Worker(new URL("./search_worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent<{ id: number; value?: LaneResult[]; error?: string }>) => {
        const call = this.calls.get(e.data.id);
        this.calls.delete(e.data.id);
        if (e.data.error !== undefined) call?.reject(Error(e.data.error));
        else call?.resolve(e.data.value!);
      };
      w.postMessage({ init: { wasm: wasm.slice(0), weights: weights.slice(0) } });
      this.workers.push(w);
    }
  }

  private run(worker: number, job: Job): Promise<LaneResult[]> {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.calls.set(id, { resolve, reject });
      this.workers[worker].postMessage({ id, job });
    });
  }

  /** Plans the rest of the mover's turn in engine game `game` (seed + `history` rebuild it on the workers). */
  async plan(engine: Engine, game: number, seed: number, history: string[], lanes: number, planSeed: number): Promise<BotPlan> {
    if (!this.size) return engine.plan(game, lanes, planSeed);
    const truth = engine.clone(game);
    try {
      const mover = engine.status(truth).current_player;
      const out: BotPlan = { plan: [], plans: 0, overrides: 0, fallbacks: 0, value: null };
      let rejected = 0;
      for (let round = 1; ; round++) {
        const s = engine.status(truth);
        if (s.current_player !== mover || s.terminated || s.truncated) break;
        const sample = rejected <= MAX_REJECTED;
        const per = Math.ceil(lanes / this.size);
        const jobs = [];
        for (let w = 0; w * per < lanes; w++) {
          const job = { seed, history: [...history, ...out.plan], round, sample, planSeed, from: w * per, to: Math.min(lanes, (w + 1) * per) };
          jobs.push(this.run(w, job));
        }
        const results = (await Promise.all(jobs)).flat().sort((a, b) => a.lane - b.lane);
        const best = results.reduce((b, r) => (r.score > b.score ? r : b), results[0]);
        const applied = engine.commit(truth, round, sample, planSeed, best.actions);
        if (!applied.length) {
          rejected++;
          continue;
        }
        if (!sample) out.fallbacks++;
        rejected = 0;
        out.plans++;
        if (best.lane !== 0) out.overrides++;
        out.value ??= best.score;
        out.plan.push(...applied);
      }
      return out;
    } finally {
      engine.free(truth);
    }
  }
}
