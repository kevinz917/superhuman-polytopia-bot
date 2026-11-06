"""Reference trajectories for the in-browser bot (engine-wasm): the JAX policy's decisions on the Rust engine.

    modal run --env staging experiments/modal_browser_parity.py --games 8 --out artifacts/browser-parity/jax.jsonl
    node web/tests/local_parity.ts engine-wasm/target/wasm32-unknown-unknown/release/engine_wasm.wasm \
        web/public/bot/champion.bin artifacts/browser-parity/jax.jsonl

Each game is played from mapgen.reset(seed) by the deterministic champion (rl/network.py::sample, resignation
masked as in serving) on engine-rs, with features from the packed int16 view exactly as in training. A line
records the seed and, per decision, the catalog index, value and log-probability, so the browser build can
replay the same positions and compare its own decision at each one. Staging only.
"""
from pathlib import Path
import json
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT.parent / 'open-source/weights/champion'
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('curl', 'build-essential', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'maturin>=1.7,<2', 'zstandard==0.23.0')
    .add_local_dir(ROOT / 'engine-rs', '/workspace/engine-rs', copy=True, ignore=['**/target/**'])
    .run_commands('cd /workspace/engine-rs && PATH=/root/.cargo/bin:$PATH maturin build --release -o /workspace/wheels '
                  '&& pip install /workspace/wheels/*.whl')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(CHECKPOINT, '/workspace/checkpoint'))
app = modal.App('polytopia-browser-parity')


@app.function(image=image, cpu=4, memory=8192, timeout=3600, retries=0)
def trajectories(games: int, seed0: int):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from polytopia_engine_rs import GameBatch
    from polytopia_bot.gpu import ENGINE_REVISION
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import flatten_choice
    from polytopia_bot.rl.cpu_engine import view_features, view_layout
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.rl.rust_engine import split_masks

    size = 14
    rules_json = (Path('/workspace/src/polytopia_bot/gpu/rules.json')).read_text()
    batch = GameBatch(size, games, rules_json, ENGINE_REVISION, None, True, 10000)
    seeds = np.arange(seed0, seed0 + games, dtype=np.int32)
    batch.reset(seeds)
    layout = view_layout(size)
    x0 = view_features(jnp.zeros(batch.view_width, jnp.int16), layout, size)
    template = network.init(jax.random.key(0), x0.board.shape[-1], x0.global_state.size, size=size, width=64, blocks=2,
                            architecture='normalized-v2')
    params = load_checkpoint('/workspace/checkpoint/final.npz', template)
    features = jax.jit(jax.vmap(lambda v: view_features(v, layout, size)))
    act = jax.jit(jax.vmap(lambda x, m, k: network.sample(params, x, m, k, size=size, deterministic=True)))
    flat = jax.jit(jax.vmap(lambda c: flatten_choice(c, size)))
    records = [{'seed': int(s), 'index': [], 'value': [], 'logp': []} for s in seeds]
    done = np.zeros(games, bool)
    key = jax.random.key(0)
    while not done.all():
        x = features(jnp.asarray(batch.observe(None)))
        masks = split_masks(batch.legal_masks(False), size)
        key, draw = jax.random.split(key)
        choice, _, logp, value = act(x, jax.tree.map(jnp.asarray, masks), jax.random.split(draw, games))
        index = np.asarray(flat(choice))
        for g in np.flatnonzero(~done):
            records[g]['index'].append(int(index[g]))
            records[g]['value'].append(float(value[g]))
            records[g]['logp'].append(float(logp[g]))
        status = batch.step(*(np.asarray(a, np.int32) for a in choice))
        if (status[~done] != 0).any():
            raise RuntimeError(f'engine rejected a policy action: {status}')
        fields = batch.export()
        done = np.asarray(fields['terminated'], bool) | np.asarray(fields['truncated'], bool)
    for r, rewards in zip(records, batch.terminal_rewards()):
        r['rewards'] = [float(v) for v in rewards]
    return records


@app.local_entrypoint()
def main(games: int = 8, seed0: int = 1810000001, out: str = 'artifacts/browser-parity/jax.jsonl'):
    records = trajectories.remote(games, seed0)
    path = ROOT / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    print(f'{path}: {len(records)} games, {sum(len(r["index"]) for r in records)} decisions')
