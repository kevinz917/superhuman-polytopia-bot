"""In-node GPU inference server for the actor fleet: equivalence tests and fleet throughput / logp smoke.

    modal run --env staging experiments/modal_inference_server.py::tests
    modal run --env staging experiments/modal_inference_server.py::fleet --arms-json '[...]' --gpu L4 --cpu 32

Every arm of one call runs in ONE container, one after another, so arms share the host (CPU model, GPU). An arm is
a JSON object for scripts/bench_actor_fleet.py: processes, games, mode (local|cuda), size, players, maps, blocks,
width, seconds, measure, serve_options ({trunk_dtype, precision, deadline_ms, inflight}), logp_check.
"""
from pathlib import Path
import json
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
# The actors' engine is the Rust one: the wheel is built from engine-rs with maturin, as in modal_distributed_rl.py.
image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('curl', 'build-essential', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
    .pip_install('jax[cuda12]==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0', 'pytest>=9,<10', 'maturin>=1.7,<2')
    .add_local_dir(ROOT / 'engine-rs', '/workspace/engine-rs', copy=True, ignore=['**/target/**'])
    .run_commands('cd /workspace/engine-rs && PATH=/root/.cargo/bin:$PATH maturin build --release -o /workspace/wheels '
                  '&& pip install /workspace/wheels/*.whl')
    .env({'PYTHONPATH': '/workspace/src', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'scripts', '/workspace/scripts', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'tests', '/workspace/tests', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-inference-server')
results = modal.Volume.from_name('polytopia-structured-rl')   # checkpoints for arms with "weights"


@app.function(image=image, cpu=8, memory=32768, timeout=2400, retries=0)
def run_tests(paths: str):
    import subprocess
    out = subprocess.run(['python', '-m', 'pytest', '-x', '-q', '--durations=5', *paths.split(',')], cwd='/workspace',
                         capture_output=True, text=True, env=dict(os.environ, JAX_PLATFORMS='cpu'))
    return out.returncode, out.stdout[-6000:] + out.stderr[-3000:]


@app.local_entrypoint()
def tests(paths: str = 'tests/test_inference_server.py'):
    code, text = run_tests.remote(paths)
    print(text)
    print('exit', code)


@app.function(image=image, gpu='L4', cpu=32, memory=131072, timeout=5400, retries=0, volumes={'/results': results})
def fleet_bench(arms: list):
    import subprocess
    import threading
    host = subprocess.run(['bash', '-c', "grep -m1 'model name' /proc/cpuinfo; nproc; nvidia-smi --query-gpu=name --format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().replace('\n', ' / ')
    rows = [dict(host=host)]
    for arm in arms:
        util, stop = [], threading.Event()

        def sample_gpu():
            while not stop.wait(5):
                q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True).stdout.strip()
                if q:
                    util.append([float(x) for x in q.split(',')])
        t = threading.Thread(target=sample_gpu, daemon=True)
        t.start()
        out = subprocess.run(['python', '/workspace/scripts/bench_actor_fleet.py', json.dumps(arm)], capture_output=True,
                             text=True, cwd='/workspace', env=dict(os.environ, JAX_PLATFORMS='cpu'))
        stop.set()
        if out.returncode == 0:
            rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        else:
            rows.append(dict(arm=arm, error=(out.stdout + out.stderr)[-4000:]))
        tail = util[len(util) // 2:]   # second half of the arm: past compile
        rows[-1]['gpu_util_mean'] = round(sum(u[0] for u in tail) / len(tail), 1) if tail else None
        rows[-1]['gpu_mem_mib_max'] = max((u[1] for u in util), default=None)
        rows[-1]['server_log'] = [l[:1500] for l in out.stdout.splitlines() if '"stage": "server' in l][-2:]
    return rows


@app.local_entrypoint()
def fleet(arms_json: str, gpu: str = 'L4', cpu: int = 32, memory_gib: int = 128, out: str = '', parallel: bool = False):
    """--parallel: one container per arm (different hosts; faster wall clock)."""
    arms = json.loads(arms_json)
    fn = fleet_bench.with_options(gpu=gpu.split(',') if ',' in gpu else gpu, cpu=cpu, memory=memory_gib * 1024)   # a list = fallbacks
    if parallel:
        calls = [fn.spawn([arm]) for arm in arms]
        rows = []
        for c in calls:
            try:
                rows.extend(c.get())
            except Exception as e:
                rows.append(dict(error=repr(e)))
    else:
        rows = fn.remote(arms)
    for row in rows:
        print(json.dumps(row))
    if out:
        path = ROOT / out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1))
        print('wrote', path)
