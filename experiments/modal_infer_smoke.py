"""Smoke for GPU inference in actor containers: network forward cost per sample on one CPU core vs a cheap GPU.

    modal run --env staging experiments/modal_infer_smoke.py

Synthetic features (forward cost does not depend on values). Measures the trunk plus the family/source heads,
which is nearly all of a policy forward. Nets: 2x64 (champion) and 4x128 (big net); boards: 14x14 two seats and
18x18 four seats. CPU = one process, XLA single-threaded (as the actors run); GPU = batched, fp32 / TF32 / bf16.
"""
from pathlib import Path
import json
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax==0.11.2', 'numpy>=2.1')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu',
          'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**']))
gpu_image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda12]==0.11.2', 'numpy>=2.1')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cuda', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**']))
actor_image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('curl', 'build-essential', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0', 'maturin>=1.7,<2')
    .add_local_dir(ROOT / 'engine-rs', '/workspace/engine-rs', copy=True, ignore=['**/target/**'])
    .run_commands('cd /workspace/engine-rs && PATH=/root/.cargo/bin:$PATH maturin build --release -o /workspace/wheels '
                  '&& pip install /workspace/wheels/*.whl')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'scripts', '/workspace/scripts', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-infer-smoke')

NETS = {'2x64': dict(blocks=2, width=64), '4x128': dict(blocks=4, width=128)}
BOARDS = {'1v1-14': dict(size=14, players=2), 'ffa4-18': dict(size=18, players=4)}


def _bench(batches, dtypes, seconds=3.0):
    import time
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import feature_shapes
    from polytopia_bot.rl.observation import Features
    rows = []
    for board_name, b in BOARDS.items():
        ch, gl = feature_shapes(b['players'])
        size = b['size']
        for net_name, n in NETS.items():
            params = network.init(jax.random.key(0), ch, gl, size=size, width=n['width'], blocks=n['blocks'],
                                  architecture='normalized-v2')
            for dt in dtypes:
                cast = jnp.bfloat16 if dt == 'bf16' else jnp.float32
                p = jax.tree.map(lambda a: a.astype(cast) if a.dtype == jnp.float32 else a, params)

                def forward(p, board, glob):
                    def one(x, g):
                        tiles, gg, value = network.trunk(p, Features(x, g))
                        return network.dense(gg, p['family']).sum() + network.dense(tiles, p['source']).sum() + value
                    return jax.vmap(one)(board, glob)
                f = jax.jit(forward)
                for bs in batches:
                    board = jnp.zeros((bs, size, size, ch), cast)
                    glob = jnp.zeros((bs, gl), cast)
                    jax.block_until_ready(f(p, board, glob))
                    n_calls, t0 = 0, time.perf_counter()
                    while time.perf_counter() - t0 < seconds:
                        jax.block_until_ready(f(p, board, glob))
                        n_calls += 1
                    dt_s = (time.perf_counter() - t0) / n_calls
                    rows.append(dict(board=board_name, net=net_name, dtype=dt, batch=bs,
                                     ms_per_call=round(dt_s * 1e3, 3), samples_per_s=round(bs / dt_s)))
                    print(json.dumps(rows[-1]), flush=True)
    return rows


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=1800, retries=0)
def cpu_bench():
    import subprocess
    model = subprocess.run(['bash', '-c', "grep -m1 'model name' /proc/cpuinfo"], capture_output=True, text=True).stdout
    return dict(host=model.strip(), rows=_bench([8, 24, 64], ['f32'], seconds=2.0))


@app.function(image=gpu_image, gpu='L4', cpu=4, memory=16384, timeout=1800, retries=0)
def gpu_bench(precision: str):
    os.environ['POLYTOPIA_MATMUL_PRECISION'] = precision
    import subprocess
    name = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True,
                          text=True).stdout.strip()
    return dict(gpu=name, precision=precision, rows=_bench([512, 2048, 8192], ['f32', 'bf16'], seconds=2.0))


@app.local_entrypoint()
def main(gpus: str = 'L4,A10G,L40S'):
    calls = [('cpu', cpu_bench.spawn())]
    for g in gpus.split(','):
        calls.append((g, gpu_bench.with_options(gpu=g).spawn('DEFAULT')))
    out = {}
    for name, c in calls:
        try:
            out[name] = c.get()
        except Exception as e:   # one GPU type unavailable should not lose the others
            out[name] = dict(error=repr(e))
    path = ROOT / 'runs' / 'infer-smoke' / 'result.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print('wrote', path)


# ---- actor tick split on the Rust engine (scripts/bench_actor_split.py): what stays on the CPU with GPU inference

@app.function(image=actor_image, cpu=16, memory=49152, timeout=1800, retries=0)
def actor_split(size: int, players: int, games: int, seconds: int, blocks: int, width: int, maps: str, processes: int):
    """`processes` pinned copies at once (production packs 32 per cpu=16 container), so cores contend as in a run."""
    import subprocess
    cores = sorted(os.sched_getaffinity(0))
    model = subprocess.run(['bash', '-c', "grep -m1 'model name' /proc/cpuinfo"], capture_output=True, text=True).stdout
    procs = [subprocess.Popen(['taskset', '-c', str(cores[i % len(cores)]), 'python', '/workspace/scripts/bench_actor_split.py',
                               str(size), str(players), str(games), str(seconds), str(blocks), str(width), maps],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd='/workspace',
                              env=dict(os.environ, SEED=str(i))) for i in range(processes)]
    rows, errors = [], []
    for p in procs:
        out, err = p.communicate()
        if p.returncode == 0:
            rows.append(json.loads(out.strip().splitlines()[-1]))
        else:
            errors.append(err[-2000:])
    keys = [k for k in rows[0] if k.endswith('_ms')] if rows else []
    mean = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
    return dict(host=model.strip(), cpus=len(cores), processes=processes, ok=len(rows), errors=errors[:2],
                samples_per_s_per_process=round(sum(r['samples_per_s'] for r in rows) / max(1, len(rows)), 1),
                container_samples_per_s=round(sum(r['samples_per_s'] for r in rows)), **mean,
                config=dict(size=size, players=players, games=games, net=f'{blocks}x{width}', maps=maps))


@app.local_entrypoint()
def split(nets: str = '1x8,2x64,4x128', seconds: int = 90, games: int = 24, processes: int = 32):
    calls = []
    for size, players, maps, mem in ((14, 2, 'proposed-v1', 49152), (18, 4, 'four-v1', 98304)):
        for net in nets.split(','):
            b, w = map(int, net.split('x'))
            calls.append(actor_split.with_options(memory=mem).spawn(size, players, games, seconds, b, w, maps, processes))
    out = []
    for c in calls:
        try:
            out.append(c.get())
        except Exception as e:
            out.append(dict(error=repr(e)))
        print(json.dumps(out[-1]), flush=True)
    path = ROOT / 'runs' / 'infer-smoke' / 'actor-split.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print('wrote', path)
