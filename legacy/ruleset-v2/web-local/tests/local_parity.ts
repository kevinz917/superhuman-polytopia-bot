// Compares the in-browser bot with the JAX policy on the reference trajectories of
// experiments/modal_browser_parity.py: at every recorded position, the wasm bot's deterministic decision, value and
// log-probability against JAX's. The game always follows JAX's recorded action, so positions stay identical.
//   node web/tests/local_parity.ts <engine_wasm.wasm> <champion.bin> <jax.jsonl>
import { readFileSync } from "node:fs";
import { Engine } from "../src/local/engine.ts";

const [wasm, weights, trajectories] = process.argv.slice(2);
const engine = await Engine.load(readFileSync(wasm));
const bytes = readFileSync(weights);
engine.loadModel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));

let decisions = 0;
let agree = 0;
let valueErr = 0;
let logpErr = 0;
const misses: string[] = [];
for (const line of readFileSync(trajectories, "utf8").split("\n").filter(Boolean)) {
  const r = JSON.parse(line) as { seed: number; index: number[]; value: number[]; logp: number[]; rewards: number[] };
  const id = engine.newGame(r.seed);
  r.index.forEach((index, i) => {
    const d = engine.act(id, true, i);
    decisions++;
    if (Number(d.action_id) === index) {
      agree++;
      logpErr = Math.max(logpErr, Math.abs(d.log_probability - r.logp[i]));
    } else if (misses.length < 10) misses.push(`seed ${r.seed} step ${i}: wasm ${d.action_id} jax ${index} (jax logp ${r.logp[i].toFixed(4)})`);
    valueErr = Math.max(valueErr, Math.abs(d.value - r.value[i]));
    engine.step(id, String(index));
  });
  const status = engine.status(id);
  if (!status.terminated && !status.truncated) throw Error(`seed ${r.seed}: game did not end where JAX's did`);
  if (status.rewards.join() !== r.rewards.join()) throw Error(`seed ${r.seed}: rewards ${status.rewards} vs ${r.rewards}`);
  engine.free(id);
}
console.log(`${agree}/${decisions} decisions identical (${((100 * agree) / decisions).toFixed(3)}%), ` +
  `max |value diff| ${valueErr.toExponential(2)}, max |logp diff| on agreeing ${logpErr.toExponential(2)}`);
for (const m of misses) console.log("  " + m);
