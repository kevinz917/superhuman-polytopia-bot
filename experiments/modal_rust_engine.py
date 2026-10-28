"""Rust actor engine (engine-rs/): equality harness and per-phase benchmark on Modal CPU containers.

    modal run --env staging experiments/modal_rust_engine.py::equality --size 14 --games 24 --ticks 2000 --procs 8
    modal run --env staging experiments/modal_rust_engine.py::bench --sizes 14,18

The image installs a Rust toolchain, builds the crate with maturin (release, abi3) and installs
the wheel next to the JAX CPU engine. Staging only.
"""
from pathlib import Path
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('curl', 'build-essential', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'pytest>=9,<10', 'maturin>=1.7,<2', 'zstandard==0.23.0')
    .add_local_dir(ROOT / 'engine-rs', '/workspace/engine-rs', copy=True, ignore=['**/target/**'])
    .run_commands('cd /workspace/engine-rs && PATH=/root/.cargo/bin:$PATH maturin build --release -o /workspace/wheels '
                  '&& pip install /workspace/wheels/*.whl && ls -la /workspace/wheels')
    .env({'PYTHONPATH': '/workspace/src:/workspace:/workspace/scripts', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'tests', '/workspace/tests', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'scripts', '/workspace/scripts', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'experiments', '/workspace/experiments', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'configs', '/workspace/configs'))
app = modal.App('polytopia-rust-engine')


def _cpu():
    import subprocess
    return subprocess.run(['bash', '-c', "grep -m1 'model name' /proc/cpuinfo; nproc"], capture_output=True, text=True).stdout.strip().replace('\n', ' / ')


@app.function(image=image, cpu=16, memory=32768, timeout=6 * 3600, retries=0)
def run_equality(size: int, games: int, ticks: int, procs: int, seed0: int, maps: str = '', policy: str = 'random',
                 end_boost: float = 0.12, frozen: int = 0, resign_every: int = 97):
    import subprocess
    cores = sorted(os.sched_getaffinity(0))
    procs_ = []
    for i in range(procs):
        env = dict(os.environ, PIN_CORE=str(cores[i % len(cores)]))
        procs_.append(subprocess.Popen(['python', '/workspace/tests/test_rust_engine.py', '--size', str(size), '--games', str(games),
                                        '--ticks', str(ticks), '--seed', str(seed0 + i), '--policy', policy, '--end-boost', str(end_boost), '--resign-every', str(resign_every)]
                                       + (['--maps', maps] if maps else []) + (['--frozen'] if frozen else []), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, cwd='/workspace', env=env))
    rows = [_cpu()]
    for i, p in enumerate(procs_):
        out, err = p.communicate()
        result = [l for l in out.splitlines() if l.startswith('RESULT ')]
        rows.append(result[-1] if result else f'proc {i} rc={p.returncode} ' + err[-4000:] + out[-2000:])
    return rows


@app.local_entrypoint()
def equality(size: int = 14, games: int = 24, ticks: int = 300, procs: int = 1, seed0: int = 0, maps: str = '',
             policy: str = 'random', end_boost: float = 0.12, frozen: int = 0, resign_every: int = 97):
    for row in run_equality.remote(size, games, ticks, procs, seed0, maps, policy, end_boost, frozen, resign_every):
        print(row)


@app.function(image=image, cpu=16, memory=32768, timeout=3600, retries=0)
def run_bench(sizes: str, games: int, warm: int, repeats: int, with_jax: int):
    import subprocess
    cores = sorted(os.sched_getaffinity(0))
    rows = [_cpu()]
    for size in sizes.split(','):
        env = dict(os.environ, PIN_CORE=str(cores[0]), JAX=str(with_jax))
        p = subprocess.run(['python', '/workspace/scripts/bench_rust_engine.py', str(games), str(warm), str(repeats), size],
                           capture_output=True, text=True, cwd='/workspace', env=env)
        rows.append(p.stdout.strip().splitlines()[-1] if p.returncode == 0 else p.stderr[-4000:])
    wheel = subprocess.run(['bash', '-c', 'ls -la /workspace/wheels; python -c "import polytopia_engine_rs,os;print(os.path.getsize(polytopia_engine_rs.__file__))"'],
                           capture_output=True, text=True).stdout
    rows.append(wheel)
    return rows


@app.local_entrypoint()
def bench(sizes: str = '14,18', games: int = 24, warm: int = 150, repeats: int = 25, with_jax: int = 1):
    for row in run_bench.remote(sizes, games, warm, repeats, with_jax):
        print(row)


@app.function(image=image, cpu=16, memory=32768, timeout=3 * 3600, retries=0)
def run_tests(paths: str, seconds: int = 0):
    """The repository's pytest suite inside the image that has the Rust wheel installed.
    paths: comma-separated pytest arguments; seconds>0 kills the run after that long and returns its partial output."""
    import subprocess
    try:
        out = subprocess.run(['python', '-m', 'pytest', '-q', '-rs', '-p', 'no:cacheprovider', *paths.split(',')],
                             cwd='/workspace', capture_output=True, text=True, timeout=seconds or None)
    except subprocess.TimeoutExpired as e:
        text = lambda b: (b.decode(errors='replace') if isinstance(b, bytes) else (b or ''))
        return -1, f'TIMEOUT after {seconds}s\n' + text(e.stdout)[-12000:] + text(e.stderr)[-3000:]
    return out.returncode, out.stdout[-12000:] + out.stderr[-3000:]


@app.local_entrypoint()
def tests(paths: str = 'tests', seconds: int = 0):
    rc, text = run_tests.remote(paths, seconds)
    print(text)
    print('pytest exit code', rc)


@app.function(image=image, cpu=16, memory=65536, timeout=3 * 3600, retries=0)
def run_tests_sharded(workers: int):
    """One pytest process per test file, `workers` at a time; returns per-file status lines."""
    import glob
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    files = sorted(glob.glob('/workspace/tests/test_*.py'))

    def one(path):
        p = subprocess.run(['python', '-m', 'pytest', '-q', '-rs', '-p', 'no:cacheprovider', path], cwd='/workspace',
                           capture_output=True, text=True, env=dict(os.environ, XLA_FLAGS='--xla_cpu_multi_thread_eigen=false',
                                                                   OMP_NUM_THREADS='2'))
        lines = [l for l in p.stdout.splitlines() if l.strip()]
        summary = lines[-1] if lines else p.stderr.strip().splitlines()[-1:] or ['(no output)']
        tail = '' if p.returncode in (0, 5) else '\n' + p.stdout[-2500:] + p.stderr[-1000:]
        return f'rc={p.returncode} {path.split("/")[-1]}: {summary}{tail}'
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, files))


@app.local_entrypoint()
def tests_sharded(workers: int = 16):
    for row in run_tests_sharded.remote(workers):
        print(row)
