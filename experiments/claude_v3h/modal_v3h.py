"""Claude session H (hybrid trunk) Modal entry points: session E's modal_v3.py with H's own app (claude-v3-hybrid-h) and
volume (claude-v3h-results); session E's volume claude-v3-results is mounted READ-ONLY at /e for its map table and
checkpoints (D_np and E's arms), so H never writes E's files.

    # model-only cost of several arms in one H100 container (bench_v3.py per arm, own subprocess each)
    env -u HTTP_PROXY -u HTTPS_PROXY modal run --env staging experiments/claude_v3h/modal_v3h.py::bench \
        --arms-json "$(cat arms.json)" --out results/bench-01.json
    # end-to-end GPU-resident PPO (engine-cu + JAX on the same card), fresh network of any architecture
    env -u HTTP_PROXY -u HTTPS_PROXY modal run --env staging experiments/claude_v3h/modal_v3h.py::train \
        --run-id v3-e2e-01 --architecture transformer-v3 --width 128 --blocks 6 --seconds 900

Image: the challenger's Tokamax image recipe (experiments/claude_challenger/modal_gpu_resident.py: engine-cu build,
torch 2.6 for tensor moves, jax 0.11.2, cuDNN 9.28, Triton 3.8, Tokamax 1dc0149, ptxas fix) without the V2 checkpoint.
Kernel defaults are the challenger's best V2 GPU-resident config: Tokamax attention, Tokamax affine LayerNorm, no
remat, BF16 trunk, HIGHEST matmul/head precision. Every call runs with retries=0 and an explicit timeout, so its
container-hour bound is known before launch (ledger experiments/claude_v3/ledger.py).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
sys.path.insert(0, str(ROOT))
from experiments.modal_gpu_engine import TRAIN_SCRIPT, torch_base   # noqa: E402

TOKAMAX = ('tokamax @ https://github.com/openxla/tokamax/archive/1dc0149a34b09d7c559c04533029eb550e2c7027.tar.gz'
           '#sha256=6b35ec6eb51f1e33191b5824034874ee5714c710468e5f69c5614d634f5dfd1c')
image = (torch_base
    .pip_install('jax[cuda12]==0.11.2', 'jax-triton==0.5.0', 'triton==3.8.0', 'nvidia-cudnn-cu12==9.28.0.42', TOKAMAX)
    .run_commands("P=$(python -c 'import nvidia; print(list(nvidia.__path__)[0])')/cuda_nvcc/bin/ptxas && test -x $P "
                  "&& mv /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas-12.4 && ln -s $P /usr/local/cuda/bin/ptxas "
                  "&& ptxas --version | tail -1")
    .env({'PYTHONPATH': '/workspace/src:/workspace:/workspace/scripts', 'JAX_PLATFORMS': 'cuda',
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'experiments', '/workspace/experiments', ignore=['**/__pycache__/**']))
results = modal.Volume.from_name('claude-v3h-results', create_if_missing=True)
e_results = modal.Volume.from_name('claude-v3-results').read_only()
app = modal.App('claude-v3-hybrid-h')
VOLUMES = {'/results': results, '/e': e_results}
KERNEL_ENV = {'POLYTOPIA_TRANSFORMER_ATTENTION': 'tokamax-triton-bfloat16', 'POLYTOPIA_LAYER_NORM': 'tokamax-affine',
              'POLYTOPIA_TRANSFORMER_REMAT': '0', 'PC_CUDA_WARP_TICK': '1'}
BENCH_ENV = dict(KERNEL_ENV, POLYTOPIA_TRUNK_DTYPE='bfloat16', POLYTOPIA_MATMUL_PRECISION='HIGHEST',
                 POLYTOPIA_HEAD_PRECISION='HIGHEST', XLA_PYTHON_CLIENT_MEM_FRACTION='.9')
CACHE = {'JAX_COMPILATION_CACHE_DIR': '/results/.jax-compilation-cache'}


def _gpu_name() -> str:
    return subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip()


@app.function(image=image, gpu='H100!', cpu=4, memory=32768, timeout=2400, retries=0, volumes=VOLUMES)
def bench_arms(arms: list) -> dict:
    started = time.time()
    rows = []
    for arm in arms:
        t = time.time()
        env = dict(os.environ, **BENCH_ENV, **CACHE, **{k: str(v) for k, v in arm.get('env', {}).items()})
        try:
            p = subprocess.run(['python', '/workspace/experiments/claude_v3/bench_v3.py', json.dumps(arm)],
                               capture_output=True, text=True, timeout=arm.get('timeout', 600), env=env)
            found = [l[7:] for l in p.stdout.splitlines() if l.startswith('RESULT ')]
            row = json.loads(found[-1]) if found else dict(name=arm['name'], error=(p.stderr or p.stdout)[-3000:])
        except subprocess.TimeoutExpired:
            row = dict(name=arm['name'], error=f"timeout after {arm.get('timeout', 600)} s")
        row['arm_wall_seconds'] = round(time.time() - t, 1)
        print(json.dumps({k: v for k, v in row.items() if k not in ('learner_times', 'rollout_times')}), flush=True)
        rows.append(row)
    return dict(rows=rows, container_seconds=round(time.time() - started, 1), gpu=_gpu_name())


@app.function(image=image, gpu='H100!', cpu=4, memory=32768, timeout=3600, retries=0, volumes=VOLUMES)
def train_run(args: list, env: dict) -> dict:
    e = dict(os.environ, **CACHE, **{k: str(v) for k, v in env.items()})
    cmd = ['python', TRAIN_SCRIPT, *args]
    gpu = _gpu_name()
    print(gpu, flush=True)
    print(' '.join(cmd), json.dumps(env), flush=True)
    t = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd='/workspace', env=e)
    lines, last = [], time.time()
    for line in p.stdout:
        print(line, end='', flush=True)
        lines.append(line)
        if time.time() - last > 300:
            results.commit()
            last = time.time()
    p.wait()
    results.commit()
    out = ''.join(lines)
    found = [l for l in out.splitlines() if l.startswith('RESULT ')]
    updates = [l for l in out.splitlines() if '"stage": "update"' in l]
    return dict(gpu=gpu, rc=p.returncode, wall_seconds=round(time.time() - t, 1), cmd=cmd, env=env,
                result=json.loads(found[-1][7:]) if found else None, updates=updates, tail=out[-6000:])


@app.function(image=image, gpu='H100!', cpu=4, memory=32768, timeout=1800, retries=0, volumes=VOLUMES)
def arena_runs(matches: list) -> dict:
    """gpu_arena.py per match (own subprocess each), results appended to /results/arena/<name>.json."""
    results.reload()
    e_results.reload()
    started, rows = time.time(), []
    env = dict(os.environ, **BENCH_ENV, **CACHE)
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.6'
    for m in matches:
        t = time.time()
        try:
            p = subprocess.run(['python', '/workspace/experiments/claude_v3/gpu_arena.py', json.dumps(m)],
                               capture_output=True, text=True, timeout=m.get('timeout', 900), env=env)
            found = [l[7:] for l in p.stdout.splitlines() if l.startswith('RESULT ')]
            row = json.loads(found[-1]) if found else dict(error=(p.stderr or p.stdout)[-3000:])
            row['progress'] = [l for l in p.stderr.splitlines() if l.startswith('{"ticks"')][-40:]
        except subprocess.TimeoutExpired as error:
            tail = (error.stderr or b'')
            tail = tail.decode() if isinstance(tail, bytes) else tail
            row = dict(error=f"timeout after {m.get('timeout', 900)} s", progress=tail.splitlines()[-20:])
        row.update(name=m.get('name'), wall_seconds=round(time.time() - t, 1))
        out = Path('/results/arena') / f"{m.get('name', 'match')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=1))
        print(json.dumps(row), flush=True)
        rows.append(row)
    results.commit()
    return dict(rows=rows, container_seconds=round(time.time() - started, 1), gpu=_gpu_name())


@app.local_entrypoint()
def arena(matches_json: str, out: str = '', timeout: int = 1800):
    res = arena_runs.with_options(timeout=timeout).remote(json.loads(matches_json))
    text = json.dumps(res, indent=1)
    print(text[-6000:])
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)


@app.local_entrypoint()
def bench(arms_json: str, out: str = '', gpu: str = 'H100!', timeout: int = 2400):
    arms = json.loads(arms_json)
    t = time.time()
    res = bench_arms.with_options(gpu=gpu, timeout=timeout).remote(arms)
    res['client_wall_seconds'] = round(time.time() - t, 1)
    res['gpu_request'] = gpu
    text = json.dumps(res, indent=1)
    print(text[-4000:])
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)


@app.local_entrypoint()
def train(run_id: str, architecture: str = 'transformer-v3', width: int = 128, blocks: int = 6, seconds: float = 900.,
          config: str = 'experiments/hero/hero-1v1.json', games: int = 4096, sets: int = 4, grad_accum: int = 16,
          updates: int = 0, gpu: str = 'H100!', devices: int = 1, cpu: int = 4, memory_gib: int = 32,
          timeout: int = 3600, map_table: str = '/e/maps/table-classic-14-2p-200000-300000.npz', extra: str = '', env_json: str = '', out: str = ''):
    env = dict(KERNEL_ENV, **json.loads(env_json or '{}'))
    args = ['--config', config, '--parent', '', '--architecture', architecture, '--width', str(width),
            '--blocks', str(blocks), '--run-id', run_id, '--run-root', '/results/distributed', '--seconds', str(seconds),
            '--games', str(games), '--sets', str(sets), '--grad-accum', str(grad_accum), '--matmul-precision', 'HIGHEST',
            '--checkpoint-every', '1000000', '--snapshot-every', '1000000', '--jax-mem-fraction', '.45']
    if updates:
        args += ['--updates', str(updates)]
    if devices != 1:
        args += ['--devices', str(devices)]
    if map_table:
        args += ['--map-table', map_table]
    args += extra.split()
    t = time.time()
    res = train_run.with_options(gpu=gpu, cpu=cpu, memory=memory_gib * 1024, timeout=timeout).remote(args, env)
    res['client_wall_seconds'] = round(time.time() - t, 1)
    res['cpu'], res['memory_gib'], res['gpu_request'] = cpu, memory_gib, gpu
    text = json.dumps(res, indent=1)
    print(text[-8000:])
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
