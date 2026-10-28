"""Rust engine A/B on Modal: several engine-rs revisions built side by side in one image.

Every revision in RUST_REVS (name=git-rev, comma separated; default base=d5a5550, the merged engine that the
JAX lockstep harness certified) plus `work` (this working tree's engine-rs, or the rev in RUST_WORK) is built with maturin into its own
directory, so one process can load all of them (scripts/rust_ab_common.py) and

  * diff:  play random games through the baseline and `work` in lockstep and compare every exported State
           field, the dense masks, the packed masks (when `work` has them), both views, status and rewards
           after every tick (Rust vs Rust: no JAX in the loop, so millions of steps per minute);
  * bench: time every revision on the same warmed states, interleaved, on one pinned core.

    RUST_REVS=base=d5a5550 modal run --env staging experiments/modal_rust_ab.py::diff --size 14 --ticks 4000 --procs 16
    RUST_REVS=base=d5a5550,l1=<sha> modal run --env staging experiments/modal_rust_ab.py::bench --sizes 14,18

The authoritative gate stays tests/test_rust_engine.py against the JAX engine (experiments/modal_rust_engine.py).
"""
from pathlib import Path
import os
import subprocess
import tempfile
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
REVS = [tuple(x.split('=', 1)) for x in os.environ.get('RUST_REVS', 'base=d5a5550').split(',') if x]


def _archive(rev):
    """engine-rs of a git revision, extracted into a temp dir (local side only)."""
    sha = subprocess.run(['git', 'rev-parse', '--short=12', rev], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    out = Path(tempfile.gettempdir()) / 'polytopia-rust-revs' / sha
    if not (out / 'engine-rs' / 'Cargo.toml').exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=out.parent))   # extract privately, then rename: parallel runs may race here
        archive = subprocess.run(['git', 'archive', sha, 'engine-rs'], cwd=ROOT, capture_output=True, check=True).stdout
        subprocess.run(['tar', '-x', '-C', str(tmp)], input=archive, check=True)
        try:
            tmp.rename(out)
        except OSError:
            pass   # another run got there first
    return out / 'engine-rs'


BUILD = ('cd /workspace/revs/{name}/engine-rs && PATH=/root/.cargo/bin:$PATH CARGO_TARGET_DIR=/workspace/target-{name} '
         'maturin build --release -o /workspace/revs/{name}/wheel && pip install --no-deps --target /workspace/revs/{name}/site '
         '/workspace/revs/{name}/wheel/*.whl')
image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('curl', 'build-essential', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'pytest>=9,<10', 'maturin>=1.7,<2', 'zstandard==0.23.0'))
for name, rev in REVS:
    if modal.is_local():
        image = image.add_local_dir(_archive(rev), f'/workspace/revs/{name}/engine-rs', copy=True, ignore=['**/target/**'])
    else:
        image = image.add_local_dir(ROOT / 'engine-rs', f'/workspace/revs/{name}/engine-rs', copy=True, ignore=['**/target/**'])
    image = image.run_commands(BUILD.format(name=name))
WORK = os.environ.get('RUST_WORK', '')   # a git rev to build as `work` instead of the working tree
image = (image
    .add_local_dir(_archive(WORK) if WORK and modal.is_local() else ROOT / 'engine-rs', '/workspace/revs/work/engine-rs',
                   copy=True, ignore=['**/target/**'])
    .run_commands(BUILD.format(name='work'))
    .env({'PYTHONPATH': '/workspace/src:/workspace:/workspace/scripts', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'tests', '/workspace/tests', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'scripts', '/workspace/scripts', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-rust-ab')
dev_image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('curl', 'build-essential', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
    .pip_install('numpy>=2.1')
    .add_local_dir(ROOT / 'engine-rs', '/workspace/engine-rs', ignore=['**/target/**']))
cargo_cache = modal.Volume.from_name('polytopia-rust-cargo-target', create_if_missing=True)


@app.function(image=dev_image, cpu=8, memory=8192, timeout=1800, retries=0, volumes={'/cache': cargo_cache})
def run_cargo(cmd: str):
    """`cargo <cmd>` on this working tree's engine-rs (target dir cached in a volume): compile checks and Rust unit tests."""
    p = subprocess.run(['bash', '-c', f'cd /workspace/engine-rs && PATH=/root/.cargo/bin:$PATH CARGO_TARGET_DIR=/cache/target CARGO_HOME=/cache/cargo-home '
                                      f'PYO3_PYTHON=python3 cargo {cmd} 2>&1'], capture_output=True, text=True)
    cargo_cache.commit()
    return p.returncode, p.stdout[-20000:]


@app.local_entrypoint()
def cargo(cmd: str = 'check --release'):
    rc, out = run_cargo.remote(cmd)
    print(out)
    print('cargo exit code', rc)


def _cpu():
    return subprocess.run(['bash', '-c', "grep -m1 'model name' /proc/cpuinfo; nproc"], capture_output=True, text=True).stdout.strip().replace('\n', ' / ')


@app.function(image=image, cpu=4, memory=8192, timeout=3600, retries=0)
def run_bench(revs: str, sizes: str, games: int, warm: int, repeats: int):
    cores = sorted(os.sched_getaffinity(0))
    rows = [_cpu()]
    for size in sizes.split(','):
        env = dict(os.environ, PIN_CORE=str(cores[0]))
        p = subprocess.run(['python', '/workspace/scripts/bench_rust_ab.py', '--revs', revs, '--size', size, '--games', str(games),
                            '--warm', str(warm), '--repeats', str(repeats)], capture_output=True, text=True, cwd='/workspace', env=env)
        rows.append(p.stdout.strip().splitlines()[-1] if p.returncode == 0 else p.stdout[-3000:] + p.stderr[-4000:])
    return rows


@app.local_entrypoint()
def bench(revs: str = '', sizes: str = '14,18', games: int = 24, warm: int = 400, repeats: int = 25):
    revs = revs or ','.join([n for n, _ in REVS] + ['work'])
    for row in run_bench.remote(revs, sizes, games, warm, repeats):
        print(row)
