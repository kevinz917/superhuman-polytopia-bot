"""C/CUDA engine (engine-cu/): build, CPU lockstep gate vs engine-rs, GPU gate, bench, on Modal (staging only).

    modal run --env staging experiments/modal_gpu_engine.py::build
    modal run --env staging experiments/modal_gpu_engine.py::equality --size 14 --games 24 --ticks 2000 --procs 8
    modal run --env staging experiments/modal_gpu_engine.py::equality --size 14 --games 24 --ticks 2000 --procs 8 \\
        --policy /results/distributed/hero-1v1-cont1h/latest.npz      # policy play: the positions a trained bot reaches
    modal run --env staging experiments/modal_gpu_engine.py::gpu_equality --ticks 500
    modal run --env staging experiments/modal_gpu_engine.py::gpu_equality --ticks 500 --warp 1      # the warp-per-game tick
    modal run --env staging experiments/modal_gpu_engine.py::gpu_equality --size 14 --ticks 400 --profile --modes rust \
        [--policy /results/distributed/hero-1v1-cont1h/latest.npz]   # phase profile (lib*_prof.so) of the C engines
    modal run --env staging experiments/modal_gpu_engine.py::tick_bench --gpu H100 --games 4096   # engine alone, thread vs warp
    modal run --env staging experiments/modal_gpu_engine.py::bench --sizes 14,18
    modal run --env staging experiments/modal_gpu_engine.py::jax_train --policy-only --gpu H100 --bf16-trunk ...   # the 97.5k bench
    modal run --detach --env staging experiments/modal_gpu_engine.py::launch --run-id gpu-hero-smoke --gpu H100 \\
        --parent /results/distributed/hero-1v1-cont1h/latest.npz \\
        --critic-path /results/distributed/hero-1v1-cont1h/latest-critic.npz --seconds 600 --grad-accum 4   # the recipe trainer
    modal run --env staging experiments/modal_gpu_engine.py::jax_train --gpu H100:4 --devices 4 --run-id gpu-hero-dp4 \\
        --parent ... --critic-path ... --seconds 600     # data parallel on 4 GPUs of one container (--games per device)
    (recipe defaults: --games 4096, 4 sets, ring 160, in-rollout critic values; `--extra "--value-check"` passes further
    gpu_resident_train.py flags such as --value-at-assembly / --stale-values / --jax-mem-fraction .5)

The image is the CUDA 12.4 devel image (nvcc compiles without a GPU) with a Rust toolchain: engine-rs is built with
maturin (the reference and the layout generator), engine-cu with `make cpu` and `make cuda` (fatbin for T4 / A10G /
L4: sm_75, sm_86, sm_89). The CUDA build failing does not fail the image, so the CPU gate stays runnable; the GPU
functions then fail loudly on the missing libpc_cuda.so.
"""
from pathlib import Path
import json
import os
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
ARCHFLAGS = ('-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 '
             '-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90')
# every engine-cu variant (engine-cu/Makefile: full and SIZE=14, each also PROFILE=1); rl/gpu_engine.py picks
# lib*_14.so for boards <= 14 (POLYTOPIA_PC_VARIANT=full: the full build) and lib*_prof.so with POLYTOPIA_PC_PROFILE=1
MAKE_CUDA = f'make -C /workspace/engine-cu -j4 cuda-variants ARCHFLAGS="{ARCHFLAGS}"'
MAKE_CPU = 'make -C /workspace/engine-cu -j4 cpu-variants'
# Layers split so a failure localizes and the heavy torch wheels live only in the PPO image.
base_image = (modal.Image.from_registry('nvidia/cuda:12.4.1-devel-ubuntu22.04', add_python='3.12')
    .apt_install('build-essential', 'curl', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
    .pip_install('numpy>=2.1', 'pytest>=9,<10', 'maturin>=1.7,<2', 'jax==0.11.2'))
rust_image = (base_image
    .add_local_dir(ROOT / 'engine-rs', '/workspace/engine-rs', copy=True, ignore=['**/target/**'])
    .run_commands('cd /workspace/engine-rs && PATH=/root/.cargo/bin:$PATH maturin build --release -o /workspace/wheels '
                  '&& pip install /workspace/wheels/*.whl && ls -la /workspace/wheels'))
image = (rust_image
    .add_local_dir(ROOT / 'engine-cu', '/workspace/engine-cu', copy=True, ignore=['**/build/**'])
    .run_commands(MAKE_CPU,
                  f'({MAKE_CUDA}) || echo "CUDA BUILD FAILED (libpc_cuda.so missing; CPU gate still runs)"',
                  'ls -la /workspace/engine-cu/build')
    .env({'PYTHONPATH': '/workspace/src:/workspace:/workspace/scripts', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'tests', '/workspace/tests', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'scripts', '/workspace/scripts', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'experiments', '/workspace/experiments', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'configs', '/workspace/configs'))
# torch (CUDA 12.4 wheels) only for the GPU-resident PPO smoke. The build layers are shared with jax_image below (Modal
# allows no build step after a non-copy add_local_dir, so the local-source mounts come last on each branch).
torch_base = (rust_image
    .pip_install('torch==2.6.0', extra_index_url='https://download.pytorch.org/whl/cu124')
    .add_local_dir(ROOT / 'engine-cu', '/workspace/engine-cu', copy=True, ignore=['**/build/**'])
    .run_commands(MAKE_CPU, MAKE_CUDA, 'ls -la /workspace/engine-cu/build')
    .env({'PYTHONPATH': '/workspace/src:/workspace:/workspace/scripts', 'JAX_PLATFORMS': 'cpu'}))
torch_image = (torch_base
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'experiments', '/workspace/experiments', ignore=['**/__pycache__/**']))
# torch + CUDA jax (the production network and PPO through dlpack) for gpu_resident_train.py. The same pip spec as
# experiments/modal_infer_smoke.py; pip upgrades the nvidia-* wheels torch pinned (cudnn 9.1 -> what jax 0.11.2 needs)
# with a warning, torch loads the newer libcudnn.so.9 / libcublas.so.12 by soname.
jax_image = (torch_base
    .pip_install('jax[cuda12]==0.11.2')
    .env({'PYTHONPATH': '/workspace/src:/workspace:/workspace/scripts', 'JAX_PLATFORMS': 'cuda',
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'experiments', '/workspace/experiments', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-gpu-engine')
HARNESS = '/workspace/tests/test_gpu_engine.py'
# The run checkpoints (experiments/modal_distributed_rl.py VOLUMES, same mount paths: /results/distributed/<run>/latest.npz,
# /retained/checkpoints/...), mounted read-only: the gate only reads a policy from them.
results = modal.Volume.from_name('polytopia-structured-rl', create_if_missing=False)
retained = modal.Volume.from_name('polytopia-structured-rl-retained', create_if_missing=False)
VOLUMES = {'/results': results.read_only(), '/retained': retained.read_only()}
# The recipe trainer (jax_train / launch) writes its run folder /results/distributed/<run-id> (checkpoints, training.jsonl)
# and reads /results/maps: the same volume object modal_distributed_rl.py mounts, writable; /retained stays read-only.
TRAIN_VOLUMES = {'/results': results, '/retained': retained.read_only()}
TRAIN_SCRIPT = '/workspace/experiments/gpu_resident_train.py'


def _policy_row(policy):
    """One line with the checkpoint's config (observation version, width x blocks, architecture, seats, stage)."""
    import json
    from pathlib import Path
    path = Path(policy)
    if path.is_dir():
        path = path / 'latest.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    c = meta['config']
    return (f'policy {path}: obs_version={c.get("obs_version", 1)} width={c["width"]} blocks={c["blocks"]} '
            f'extra_blocks={c.get("extra_blocks", 0)} block_pool={c.get("block_pool", 0)} attention={c.get("attention", 0)} '
            f'architecture={c["architecture"]} players={c.get("players", 2)} size={c["size"]} no_resign={c.get("no_resign")} '
            f'stage={meta.get("stage")} parameter_elements={meta.get("parameter_elements")} ruleset={meta.get("ruleset")}')


def _cpu():
    import subprocess
    return subprocess.run(['bash', '-c', "grep -m1 'model name' /proc/cpuinfo; nproc"], capture_output=True, text=True).stdout.strip().replace('\n', ' / ')


def _train_fn(gpu, devices):
    """run_jax_train on `gpu` (a Modal GPU string such as 'H100' or 'H100:4'): one data-parallel container gets 64 GiB of
    host memory and 8 CPUs (per-device pinned staging, N devices' XLA compiles)."""
    multi = devices != 1 or ':' in gpu
    if gpu == 'L4' and not multi:
        return run_jax_train
    return run_jax_train.with_options(gpu=gpu, **(dict(memory=65536, cpu=8) if multi else {}))


def _gpu():
    import subprocess
    return subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], capture_output=True,
                          text=True).stdout.strip()


def _harness_args(size, games, ticks, seed, players, end_boost, resign_every, export_every, extra=()):
    return ['python', HARNESS, '--size', str(size), '--games', str(games), '--ticks', str(ticks), '--seed', str(seed),
            '--players', str(players), '--end-boost', str(end_boost), '--resign-every', str(resign_every),
            '--export-every', str(export_every), *extra]


def _result(p, i, out, err):
    result = [l for l in out.splitlines() if l.startswith('RESULT ')]
    return result[-1] if result else f'proc {i} rc={p.returncode} ' + err[-4000:] + out[-2000:]


@app.function(image=image, cpu=16, memory=32768, timeout=6 * 3600, retries=0, volumes=VOLUMES)
def run_equality(size: int, games: int, ticks: int, procs: int, seed0: int, players: int = 2, end_boost: float = 0.12,
                 resign_every: int = 97, export_every: int = 50, no_resign: int = 1, policy: str = '',
                 policy_temp: float = 1.0, epsilon: float = 0.05, variant: str = 'auto'):
    """`procs` pinned harness processes (seeds seed0..), C CPU vs engine-rs; one RESULT row each. policy: a checkpoint
    path inside the mounted volumes (/results/distributed/<run>/latest.npz): the harness plays its actions (--policy,
    test_gpu_engine.py docstring) instead of uniform-random ones; the first row then prints the checkpoint's config.
    variant: POLYTOPIA_PC_VARIANT (auto: libpc_cpu_14.so for boards <= 14; full: libpc_cpu.so)."""
    import subprocess
    cores = sorted(os.sched_getaffinity(0))
    rows = [_cpu()]
    extra = ['--no-resign', str(no_resign), '--quiet']
    if policy:
        rows.append(_policy_row(policy))
        extra += ['--policy', policy, '--policy-temp', str(policy_temp), '--epsilon', str(epsilon)]
    procs_ = []
    for i in range(procs):
        env = dict(os.environ, PIN_CORE=str(cores[i % len(cores)]), POLYTOPIA_PC_VARIANT=variant)
        procs_.append(subprocess.Popen(_harness_args(size, games, ticks, seed0 + i, players, end_boost, resign_every, export_every,
                                                     extra),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd='/workspace', env=env))
    for i, p in enumerate(procs_):
        out, err = p.communicate()
        rows.append(_result(p, i, out, err))
    return rows


@app.local_entrypoint()
def equality(size: int = 14, games: int = 24, ticks: int = 300, procs: int = 1, seed0: int = 0, players: int = 2,
             end_boost: float = 0.12, resign_every: int = 97, export_every: int = 50, no_resign: int = 1, policy: str = '',
             policy_temp: float = 1.0, epsilon: float = 0.05, variant: str = 'auto'):
    for row in run_equality.remote(size, games, ticks, procs, seed0, players, end_boost, resign_every, export_every, no_resign,
                                   policy, policy_temp, epsilon, variant):
        print(row)


@app.function(image=image, gpu='T4', cpu=4, memory=16384, timeout=2 * 3600, retries=0, volumes=VOLUMES)
def run_gpu_equality(size: int, games: int, ticks: int, seed0: int, players: int = 2, end_boost: float = 0.12,
                     resign_every: int = 97, export_every: int = 50, modes: str = 'rust,cpu', warp: int = 0,
                     profile: bool = False, variant: str = 'auto', policy: str = '', policy_temp: float = 1.0,
                     epsilon: float = 0.05):
    """The harness on a GPU container: 'rust' = engine-rs vs libpc_cuda.so (--device 0), 'cpu' = libpc_cpu.so vs
    libpc_cuda.so (--cpu-vs-cuda --device 0). One RESULT row per mode. warp 1: the warp-per-game tick kernel
    (PC_CUDA_WARP_TICK=1). variant: POLYTOPIA_PC_VARIANT (auto: lib*_14.so for boards <= 14; full). profile: the
    lib*_prof.so builds and the harness's PROFILE rows (the tick's phases per game; the CUDA engine's in clock64 cycles,
    the CPU driver's in ns). policy: a checkpoint in the mounted volumes plays the actions (--policy, as ::equality)."""
    import subprocess
    rows = [_cpu(), _gpu(), subprocess.run(['ls', '-la', '/workspace/engine-cu/build'], capture_output=True, text=True).stdout]
    env = dict(os.environ, PC_CUDA_WARP_TICK=str(int(warp)), POLYTOPIA_PC_VARIANT=variant,
               POLYTOPIA_PC_PROFILE='1' if profile else '0')
    if policy:
        rows.append(_policy_row(policy))
    for i, mode in enumerate(modes.split(',')):
        extra = ['--device', '0', '--quiet'] + (['--cpu-vs-cuda'] if mode == 'cpu' else []) + (['--profile'] if profile else [])
        if policy:
            extra += ['--policy', policy, '--policy-temp', str(policy_temp), '--epsilon', str(epsilon)]
        p = subprocess.run(_harness_args(size, games, ticks, seed0, players, end_boost, resign_every, export_every, extra),
                           capture_output=True, text=True, cwd='/workspace', env=env)
        rows.append(f'{mode}: ' + _result(p, i, p.stdout, p.stderr))
        rows += [f'{mode}: {l}' for l in p.stdout.splitlines() if l.startswith('PROFILE ')]
    return rows


@app.local_entrypoint()
def gpu_equality(size: int = 14, games: int = 24, ticks: int = 300, seed0: int = 0, players: int = 2, end_boost: float = 0.12,
                 resign_every: int = 97, export_every: int = 50, modes: str = 'rust,cpu', gpu: str = 'T4', warp: int = 0,
                 profile: bool = False, variant: str = 'auto', policy: str = '', policy_temp: float = 1.0,
                 epsilon: float = 0.05):
    fn = run_gpu_equality if gpu == 'T4' else run_gpu_equality.with_options(gpu=gpu)
    for row in fn.remote(size, games, ticks, seed0, players, end_boost, resign_every, export_every, modes, warp, profile,
                         variant, policy, policy_temp, epsilon):
        print(row)


@app.function(image=torch_image, gpu='L4', cpu=8, memory=32768, timeout=3600, retries=0)
def run_tick_bench(size: int = 14, games: int = 4096, players: int = 2, ticks: int = 300, warmup: int = 30, seed: int = 0,
                   kernels: str = 'thread,warp', profile: bool = False, variant: str = 'auto', check: bool = True):
    """engine-cu/tools/tick_bench.py: the engine alone at trainer scale (device pointers, random legal play sampled on the
    GPU), the thread- and warp-per-game tick kernels on the same games, outputs compared by per-tick checksums. profile:
    the lib*_prof.so builds (phase split per kernel; their timings carry the profile's clock reads and syncs)."""
    import subprocess
    env = dict(os.environ, POLYTOPIA_PC_VARIANT=variant, POLYTOPIA_PC_PROFILE='1' if profile else '0')
    cmd = ['python', '/workspace/engine-cu/tools/tick_bench.py', '--size', str(size), '--games', str(games), '--players',
           str(players), '--ticks', str(ticks), '--warmup', str(warmup), '--seed', str(seed), '--kernels', kernels]
    cmd += [] if check else ['--no-check']
    p = subprocess.run(cmd, capture_output=True, text=True, cwd='/workspace', env=env)
    rows = [l for l in p.stdout.splitlines() if l.startswith('TICKBENCH')]
    return [_gpu(), f'rc={p.returncode}', *rows] + ([] if rows and p.returncode == 0 else [p.stdout[-3000:], p.stderr[-4000:]])


@app.local_entrypoint()
def tick_bench(size: int = 14, games: int = 4096, players: int = 2, ticks: int = 300, warmup: int = 30, seed: int = 0,
               kernels: str = 'thread,warp', profile: bool = False, variant: str = 'auto', check: bool = True,
               gpu: str = 'H100'):
    fn = run_tick_bench if gpu == 'L4' else run_tick_bench.with_options(gpu=gpu)
    for row in fn.remote(size, games, players, ticks, warmup, seed, kernels, profile, variant, check):
        print(row)


@app.function(image=image, cpu=16, memory=32768, timeout=3600, retries=0)
def run_bench(sizes: str, games: int, ticks: int, players: int):
    """Harness --bench on one pinned core per board size: C CPU tick vs engine-rs tick (ms per tick of `games` games)."""
    import subprocess
    cores = sorted(os.sched_getaffinity(0))
    rows = [_cpu()]
    for i, size in enumerate(sizes.split(',')):
        env = dict(os.environ, PIN_CORE=str(cores[0]))
        p = subprocess.run(_harness_args(int(size), games, ticks, i, players, 0.12, 0, 0, ['--bench', '--quiet']),
                           capture_output=True, text=True, cwd='/workspace', env=env)
        rows.append(_result(p, i, p.stdout, p.stderr))
    return rows


@app.local_entrypoint()
def bench(sizes: str = '14,18', games: int = 24, ticks: int = 2000, players: int = 2):
    for row in run_bench.remote(sizes, games, ticks, players):
        print(row)


@app.function(image=torch_image, gpu='L4', cpu=4, memory=16384, timeout=1800, retries=0)
def run_ppo_smoke(games: int, size: int, steps: int, seconds: float, width: int, blocks: int, bf16: bool, iters: int = 0,
                  lr: float = 3e-4, report_every: int = 1):
    """experiments/gpu_ppo_smoke.py on one GPU: the CUDA engine in device-pointer mode under a torch PPO loop. Returns
    the GPU name, the stdout tail and the RESULT line."""
    import subprocess
    cmd = ['python', '/workspace/experiments/gpu_ppo_smoke.py', '--games', str(games), '--size', str(size), '--steps', str(steps),
           '--seconds', str(seconds), '--iters', str(iters), '--width', str(width), '--blocks', str(blocks), '--lr', str(lr),
           '--report-every', str(report_every)] + (['--bf16'] if bf16 else [])
    p = subprocess.run(cmd, capture_output=True, text=True, cwd='/workspace')
    result = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
    return [_gpu(), f'rc={p.returncode}', p.stdout[-6000:], p.stderr[-4000:] if p.returncode else p.stderr[-1500:],
            result[-1] if result else 'RESULT missing']


@app.local_entrypoint()
def ppo_smoke(games: int = 4096, size: int = 14, steps: int = 32, seconds: float = 300.0, width: int = 64, blocks: int = 2,
              bf16: bool = False, iters: int = 0, lr: float = 3e-4, report_every: int = 1, gpu: str = 'L4'):
    fn = run_ppo_smoke if gpu == 'L4' else run_ppo_smoke.with_options(gpu=gpu)
    for row in fn.remote(games, size, steps, seconds, width, blocks, bf16, iters, lr, report_every):
        print(row)


def _train_cmd(policy_only, games, size, steps, seconds, width, blocks, architecture, lr, minibatch, epochs, check_dlpack,
               iters, bf16_trunk, matmul_precision, gae, report_every, pool, sets, head_precision, config, parent, critic_path,
               run_id, ring, batch, grad_accum, checkpoint_every, snapshot_every, map_table, updates, seed, fp32_trunk, extra='',
               devices=1):
    """The gpu_resident_train.py command line: the recipe trainer (default) or --policy-only (the old benchmark); `extra`:
    further flags, space-separated (recipe mode)."""
    cmd = ['python', TRAIN_SCRIPT, '--games', str(games), '--size', str(size), '--seconds', str(seconds), '--width', str(width),
           '--blocks', str(blocks), '--architecture', architecture, '--matmul-precision', matmul_precision,
           '--head-precision', head_precision, '--report-every', str(report_every)]
    if sets:
        cmd += ['--sets', str(sets)]
    if minibatch:
        cmd += ['--minibatch', str(minibatch)]
    if seed >= 0:
        cmd += ['--seed', str(seed)]
    if policy_only:
        cmd += ['--policy-only', '--steps', str(steps), '--iters', str(iters), '--lr', str(lr), '--epochs', str(epochs),
                '--pool', str(pool), '--gae', gae]
        cmd += ['--bf16-trunk'] if bf16_trunk else []
        cmd += ['--check-dlpack'] if check_dlpack else []
        return cmd
    cmd += ['--config', config, '--parent', parent, '--ring', str(ring), '--updates', str(updates)]
    if devices != 1:
        cmd += ['--devices', str(devices)]
    for flag, value in (('--critic-path', critic_path), ('--run-id', run_id), ('--map-table', map_table)):
        if value:
            cmd += [flag, value]
    for flag, value in (('--batch', batch), ('--grad-accum', grad_accum), ('--checkpoint-every', checkpoint_every),
                        ('--snapshot-every', snapshot_every)):
        if value:
            cmd += [flag, str(value)]
    cmd += ['--fp32-trunk'] if fp32_trunk else []
    cmd += extra.split()
    return cmd


@app.function(image=jax_image, gpu='L4', cpu=4, memory=32768, timeout=12 * 3600, retries=0, volumes=TRAIN_VOLUMES)
def run_jax_train(games: int = 4096, size: int = 14, steps: int = 32, seconds: float = 120.0, width: int = 128, blocks: int = 4,
                  architecture: str = 'normalized-v2', lr: float = 3e-4, minibatch: int = 0, epochs: int = 1,
                  check_dlpack: bool = False, iters: int = 0, bf16_trunk: bool = False, matmul_precision: str = 'DEFAULT',
                  gae: str = 'seat', report_every: int = 1, pool: int = 512, sets: int = 0, head_precision: str = 'HIGHEST',
                  policy_only: bool = False, config: str = 'experiments/hero/hero-1v1.json', parent: str = '',
                  critic_path: str = '', run_id: str = '', ring: int = 160, batch: int = 0, grad_accum: int = 0,
                  checkpoint_every: int = 0, snapshot_every: int = 0, map_table: str = '', updates: int = 0, seed: int = -1,
                  fp32_trunk: bool = False, jax_mem_fraction: str = '', extra: str = '', commit_every: float = 300.,
                  devices: int = 1, env_json: str = ''):
    """experiments/gpu_resident_train.py on one GPU: the CUDA engine in device-pointer mode, the production JAX network
    and PPO through dlpack. Default: the hero recipe trainer writing /results/distributed/<run_id> (the volume is
    committed every commit_every seconds while it runs, so ladders and a resume see latest.npz); policy_only: the
    self-play benchmark. devices: data-parallel GPUs of the container (gpu='H100:4' with devices 4; 0 = all visible).
    Streams the trainer's lines to the function log; returns [gpu, rc, stdout tail, RESULT line]."""
    import subprocess
    cmd = _train_cmd(policy_only, games, size, steps, seconds, width, blocks, architecture, lr, minibatch, epochs, check_dlpack,
                     iters, bf16_trunk, matmul_precision, gae, report_every, pool, sets, head_precision, config, parent,
                     critic_path, run_id, ring, batch, grad_accum, checkpoint_every, snapshot_every, map_table, updates, seed,
                     fp32_trunk, extra, devices=devices)
    env = dict(os.environ, JAX_COMPILATION_CACHE_DIR='/results/.jax-compilation-cache')
    if jax_mem_fraction:
        env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = jax_mem_fraction
    env.update({k: str(v) for k, v in json.loads(env_json or '{}').items()})   # e.g. PC_CUDA_WARP_TICK, POLYTOPIA_PC_VARIANT
    print(_gpu(), flush=True)
    print(' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd='/workspace', env=env)
    lines, last = [], time.time()
    for line in p.stdout:
        print(line, end='', flush=True)
        lines.append(line)
        if not policy_only and time.time() - last > commit_every:
            results.commit()
            last = time.time()
    p.wait()
    if not policy_only:
        results.commit()
    out = ''.join(lines)
    result = [l for l in out.splitlines() if l.startswith('RESULT ')]
    return [_gpu(), f'rc={p.returncode}', out[-8000:], result[-1] if result else 'RESULT missing']


@app.local_entrypoint()
def jax_train(games: int = 4096, size: int = 14, steps: int = 32, seconds: float = 120.0, width: int = 128, blocks: int = 4,
              architecture: str = 'normalized-v2', lr: float = 3e-4, minibatch: int = 0, epochs: int = 1,
              check_dlpack: bool = False, iters: int = 0, bf16_trunk: bool = False, matmul_precision: str = 'DEFAULT',
              gae: str = 'seat', report_every: int = 1, pool: int = 512, sets: int = 0, head_precision: str = 'HIGHEST',
              policy_only: bool = False, config: str = 'experiments/hero/hero-1v1.json', parent: str = '',
              critic_path: str = '', run_id: str = '', ring: int = 160, batch: int = 0, grad_accum: int = 0,
              checkpoint_every: int = 0, snapshot_every: int = 0, map_table: str = '', updates: int = 0, seed: int = -1,
              fp32_trunk: bool = False, jax_mem_fraction: str = '', extra: str = '', gpu: str = 'L4', devices: int = 1,
              env_json: str = ''):
    """Blocking run (the recipe trainer by default; --policy-only for the benchmark). --gpu H100 for the real thing;
    --gpu H100:4 --devices 4 for the data-parallel trainer (--games per device)."""
    fn = _train_fn(gpu, devices)
    for row in fn.remote(games, size, steps, seconds, width, blocks, architecture, lr, minibatch, epochs, check_dlpack, iters,
                         bf16_trunk, matmul_precision, gae, report_every, pool, sets, head_precision, policy_only, config, parent,
                         critic_path, run_id, ring, batch, grad_accum, checkpoint_every, snapshot_every, map_table, updates, seed,
                         fp32_trunk, jax_mem_fraction, extra, devices=devices, env_json=env_json):
        print(row)


@app.local_entrypoint()
def launch(run_id: str, config: str = 'experiments/hero/hero-1v1.json', parent: str = '', critic_path: str = '',
           seconds: float = 600.0, games: int = 4096, sets: int = 0, ring: int = 160, minibatch: int = 0, batch: int = 0,
           grad_accum: int = 0, checkpoint_every: int = 0, snapshot_every: int = 0, map_table: str = '', updates: int = 0,
           seed: int = -1, width: int = 128, blocks: int = 4, architecture: str = 'normalized-v2', fp32_trunk: bool = False,
           matmul_precision: str = 'DEFAULT', head_precision: str = 'HIGHEST', jax_mem_fraction: str = '', extra: str = '',
           gpu: str = 'H100', devices: int = 1, env_json: str = ''):
    """Detached-friendly recipe run (`modal run --detach ... ::launch --run-id X --seconds 36000`): spawns run_jax_train
    on `gpu` and prints the run folder and the call id. The trainer runs for `seconds` (the LR schedule's length) and
    the function stops within its 12-h limit; follow it with `modal app logs` or the run folder's training.jsonl."""
    fn = _train_fn(gpu, devices)
    call = fn.spawn(games, 14, 32, seconds, width, blocks, architecture, 3e-4, minibatch, 1, False, 0, False, matmul_precision,
                    'seat', 1, 512, sets, head_precision, False, config, parent, critic_path, run_id, ring, batch, grad_accum,
                    checkpoint_every, snapshot_every, map_table, updates, seed, fp32_trunk, jax_mem_fraction, extra, devices=devices,
                    env_json=env_json)
    print(json.dumps(dict(run_id=run_id, run_dir=f'/results/distributed/{run_id}', call_id=call.object_id, app_id=app.app_id,
                          gpu=gpu, devices=devices, seconds=seconds)), flush=True)


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0)
def run_build():
    """Rebuild both drivers inside the image (the image build's own make output is not kept) and return the logs."""
    import subprocess
    out = []
    for cmd in ('nvcc --version | tail -2', 'cc --version | head -1', 'make -C /workspace/engine-cu clean',
                MAKE_CPU,
                # ptxas per kernel (k_tick, k_tick_warp, ...): registers, stack frame, spills, for the full and the 14 build
                f'make -C /workspace/engine-cu cuda ARCHFLAGS="-arch=sm_90" NVEXTRA="-Xptxas -v" 2>&1 | grep -E "Compiling entry|stack frame|Used [0-9]+ registers" | head -40',
                f'make -C /workspace/engine-cu cuda SIZE=14 ARCHFLAGS="-arch=sm_90" NVEXTRA="-Xptxas -v" 2>&1 | grep -E "Compiling entry|stack frame|Used [0-9]+ registers" | head -40',
                'rm -f /workspace/engine-cu/build/libpc_cuda*.so && ' + MAKE_CUDA + ' 2>&1 | tail -5', 'ls -la /workspace/engine-cu/build',
                'python -c "from polytopia_bot.rl.gpu_engine import GpuGames; g = GpuGames(11, 2); print(\'cpu driver ok\', g.mask_bytes, g.view_width)"'):
        p = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, cwd='/workspace')
        out.append(f'$ {cmd}\n{p.stdout}{p.stderr}rc={p.returncode}')
    return out


@app.local_entrypoint()
def build():
    for row in run_build.remote():
        print(row)


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0)
def run_pytest(paths: str):
    """pytest on the CPU image (the engine-cu unit tests never run on the Mac)."""
    import subprocess
    p = subprocess.run(['python', '-m', 'pytest', '-x', '-q', *paths.split(',')], capture_output=True, text=True, cwd='/workspace')
    return p.returncode, (p.stdout + p.stderr)[-6000:]


@app.local_entrypoint()
def pytest_(paths: str = 'tests/test_gpu_resident.py,tests/test_gpu_engine.py'):
    rc, out = run_pytest.remote(paths)
    print(out)
    print('rc', rc)
