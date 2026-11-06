"""Per-phase actor tick profile (scripts/bench_tick_phases.py) on one Modal CPU container.

    modal run --env staging experiments/modal_tick_phases.py --arms "1:,16:,1:--xla_cpu_enable_fast_math=true"

Each arm = processes:extra_xla_flags, run sequentially in ONE container so arms share the host.
"""
from pathlib import Path
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'scripts', '/workspace/scripts', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-tick-phases')


@app.function(image=cpu_image, cpu=16, memory=49152, timeout=3600, retries=0)
def phases(arms: str, games: int, warm: int, repeats: int, slow: int):
    import subprocess
    cpu = subprocess.run(['bash', '-c', "grep -m1 'model name' /proc/cpuinfo; nproc; grep -m1 -o 'avx512f\\|avx2' /proc/cpuinfo | head -1"],
                         capture_output=True, text=True).stdout.strip().replace('\n', ' / ')
    rows = [cpu]
    cores = sorted(os.sched_getaffinity(0))
    for arm in arms.split(','):
        p, flags = arm.split(':', 1)
        env = dict(os.environ, EXTRA_XLA_FLAGS=flags, SLOW=str(slow))
        procs = []
        for i in range(int(p)):
            e = dict(env, PIN_CORE=str(cores[i % len(cores)]))
            procs.append(subprocess.Popen(['python', '/workspace/scripts/bench_tick_phases.py', str(games), str(warm),
                                           str(repeats)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                          cwd='/workspace', env=e))
        for i, pr in enumerate(procs):
            out, err = pr.communicate()
            rows.append(f'arm={arm} proc={i} ' + (out.strip().splitlines()[-1] if pr.returncode == 0 else err[-3000:]))
    return rows


@app.local_entrypoint()
def main(arms: str = '1:,16:,1:--xla_cpu_enable_fast_math=true', games: int = 24, warm: int = 120, repeats: int = 15,
         slow: int = 1):
    for row in phases.remote(arms, games, warm, repeats, slow):
        print(row)


@app.function(image=cpu_image, cpu=4, memory=49152, timeout=1800, retries=0)
def ingest(rows: int, batch: int):
    import subprocess
    out = subprocess.run(['python', '/workspace/scripts/bench_ingest.py', str(rows), str(batch)], capture_output=True,
                         text=True, cwd='/workspace')
    return out.stdout.strip().splitlines()[-1] if out.returncode == 0 else out.stderr[-3000:]


@app.local_entrypoint()
def ingest_main(rows: int = 512, batch: int = 262144):
    print(ingest.remote(rows, batch))


gpu_image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda12]==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cuda', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'scripts', '/workspace/scripts', ignore=['**/__pycache__/**']))


@app.function(image=gpu_image, gpu='L4', cpu=4, memory=16384, timeout=1200, retries=0)
def gpu_inference(batches: str, precision: str = 'HIGHEST'):
    import subprocess
    out = subprocess.run(['python', '/workspace/scripts/bench_gpu_inference.py', batches], capture_output=True,
                         text=True, cwd='/workspace', env=dict(os.environ, POLYTOPIA_MATMUL_PRECISION=precision))
    return out.stdout.strip().splitlines()[-1] if out.returncode == 0 else out.stderr[-3000:]


@app.local_entrypoint()
def gpu_main(batches: str = '256,1024,4096,16384', gpus: str = 'L4,H100', precision: str = 'HIGHEST'):
    calls = {g: gpu_inference.with_options(gpu=g).spawn(batches, precision) for g in gpus.split(',')}
    for g, c in calls.items():
        print(g, c.get())
