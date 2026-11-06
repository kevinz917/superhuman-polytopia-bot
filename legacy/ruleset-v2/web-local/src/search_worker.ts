// One search worker of local/search.ts: runs a share of a plan round's lanes. It keeps the last game it rebuilt
// and extends it when the next history continues it (each round adds the turn committed so far).
import { Engine } from "./engine.ts";

// A dedicated worker scope (the app compiles against the DOM library, not WebWorker).
const scope = self as unknown as { postMessage(message: unknown): void; onmessage: ((e: MessageEvent) => void) | null };

let engine: Engine | null = null;
let cached: { seed: number; history: string[]; game: number } | null = null;

function gameFor(e: Engine, seed: number, history: string[]) {
  const c = cached;
  if (c && c.seed === seed && c.history.length <= history.length && c.history.every((a, i) => a === history[i])) {
    for (const a of history.slice(c.history.length)) e.step(c.game, a);
    c.history = history;
    return c.game;
  }
  if (c) e.free(c.game);
  cached = { seed, history, game: e.fromActions(seed, history) };
  return cached.game;
}

let ready: Promise<Engine> | null = null;
scope.onmessage = async (e: MessageEvent) => {
  if (e.data.init) {
    const { wasm, weights } = e.data.init;
    ready = Engine.load(wasm).then((x) => {
      x.loadModel(weights);
      engine = x;
      return x;
    });
    return;
  }
  const { id, job } = e.data;
  try {
    const x = engine ?? (await ready!);
    const game = gameFor(x, job.seed, job.history);
    scope.postMessage({ id, value: x.lanes(game, job.round, job.sample, job.planSeed, job.from, job.to) });
  } catch (err) {
    cached = null;
    scope.postMessage({ id, error: (err as Error).message || String(err) });
  }
};
