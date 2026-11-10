"""Session L Modal app for V3 flagship stages and their qualification tests. DEPLOYED (modal deploy), started with
Function.spawn by launcher.py, so a stage outlives the local client. Own names: app claude-v3l-flagship, volumes
claude-v3l-flagship (flagship stages only) and claude-v3l-qual (qualification), control Dict claude-v3l-control.

    env -u HTTP_PROXY -u HTTPS_PROXY modal deploy --env staging experiments/claude_v3l/modal_flagship.py

Image: session E's V3 recipe (experiments/claude_v3/modal_v3.py: engine-cu, torch 2.6, jax 0.11.2, cuDNN 9.28,
Triton 3.8, Tokamax 1dc0149, ptxas fix), plus V3L_SOURCE_SHA = the git HEAD the image was built from (the supervisor
pauses on a manifest/image mismatch). Workers: Retries(3) = Jack's "auto-resume, max 3 per stage" for lost
containers; the supervisor's own attempt ledger on the volume enforces the per-stage count across launches.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal
import modal.experimental  # noqa: F401  (clustered: get_cluster_info)

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
sys.path.insert(0, str(ROOT))
try:
    from experiments.modal_gpu_engine import torch_base   # noqa: E402
except ModuleNotFoundError:     # the CPU helper containers carry no sources; their image is cpu_image, not this one
    torch_base = modal.Image.debian_slim()


def _source_sha():
    if not modal.is_local():
        return os.environ.get('V3L_SOURCE_SHA', '')
    sha = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', 'src', 'experiments', 'engine-cu'],
                           capture_output=True, text=True).stdout.strip()
    return sha + ('-dirty' if dirty else '')


TOKAMAX = ('tokamax @ https://github.com/openxla/tokamax/archive/1dc0149a34b09d7c559c04533029eb550e2c7027.tar.gz'
           '#sha256=6b35ec6eb51f1e33191b5824034874ee5714c710468e5f69c5614d634f5dfd1c')
SOURCE_SHA = _source_sha()
image = (torch_base
    .pip_install('jax[cuda12]==0.11.2', 'jax-triton==0.5.0', 'triton==3.8.0', 'nvidia-cudnn-cu12==9.28.0.42', TOKAMAX)
    .run_commands("P=$(python -c 'import nvidia; print(list(nvidia.__path__)[0])')/cuda_nvcc/bin/ptxas && test -x $P "
                  "&& mv /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas-12.4 && ln -s $P /usr/local/cuda/bin/ptxas "
                  "&& ptxas --version | tail -1")
    .env({'PYTHONPATH': '/workspace/src:/workspace:/workspace/scripts:/workspace/experiments', 'JAX_PLATFORMS': 'cuda',
          'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'V3L_SOURCE_SHA': SOURCE_SHA,
          'V3L_APP': os.environ.get('V3L_APP', 'claude-v3l-flagship')})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'experiments', '/workspace/experiments', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'benchmark' / 'references', '/workspace/benchmark/references', ignore=['*.wasm']))
cpu_image = modal.Image.debian_slim(python_version='3.12')

flag_vol = modal.Volume.from_name('claude-v3l-flagship', create_if_missing=True)
qual_vol = modal.Volume.from_name('claude-v3l-qual', create_if_missing=True)
e_vol = modal.Volume.from_name('claude-v3-results')           # session E's, mounted READ-ONLY (map table, E arms)
g_vol = modal.Volume.from_name('claude-v3g-results')          # session G's, READ-ONLY (G arms)
h_vol = modal.Volume.from_name('claude-v3h-results')          # session H's, READ-ONLY (H arms)
r_vol = modal.Volume.from_name('polytopia-transformer-rl-retained')   # READ-ONLY: checkpoints/friend-champion
control = modal.Dict.from_name(os.environ.get('V3L_CONTROL_DICT', 'claude-v3l-control'), create_if_missing=True)
APP_NAME = os.environ.get('V3L_APP', 'claude-v3l-flagship')   # 'claude-v3l-arena': evaluation-only deployment
app = modal.App(APP_NAME)
RETRIES = modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0, max_delay=30.0)
MAP_TABLE = 'maps/table-classic-14-2p-200000-300000.npz'


def _work(manifest: dict, launch_id: str, vol) -> dict:
    sys.path.insert(0, '/workspace/experiments/claude_v3l')
    from supervisor import Supervisor
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', '/flagship/.jax-compilation-cache')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    print(f'[v3l] worker start {time.strftime("%Y-%m-%d %H:%M:%S %Z")} source {SOURCE_SHA} gpus:\n{gpu}', flush=True)
    vol.reload()
    run_dir = Path('/flagship/runs') / manifest['run_id']
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest, cwd='/workspace')
    out = Supervisor(manifest, launch_id, run_dir, control, commit=vol.commit).run()
    out['gpu'] = gpu
    vol.commit()
    return out


@app.function(image=image, gpu='H100!:8', cpu=32, memory=131072, timeout=86400, retries=RETRIES, scaledown_window=2,
              volumes={'/flagship': flag_vol})
def stage_worker(manifest: dict, launch_id: str) -> dict:
    """A FLAGSHIP stage (8xH100, 32 CPU / 128 GiB). Only launcher.py's flagship path spawns it (go file required)."""
    if manifest.get('mode') not in ('flagship', 'control') or manifest.get('faults'):
        raise SystemExit('stage_worker: flagship / CNN-control manifests only, without faults')
    return _work(manifest, launch_id, flag_vol)


@app.function(image=image, gpu='H100!:8', cpu=32, memory=131072, timeout=7200, retries=RETRIES, scaledown_window=2,
              volumes={'/flagship': qual_vol})
def qual_worker_8(manifest: dict, launch_id: str) -> dict:
    """Qualification on 8xH100 (Q1-Q3), own qualification volume."""
    if manifest.get('mode') != 'qualification':
        raise SystemExit('qual_worker_8: qualification manifests only')
    return _work(manifest, launch_id, qual_vol)


def _work_group(manifest: dict, launch_id: str, vol) -> dict:
    """One node of a clustered 2 x 8xH100 group: node 0 runs the supervisor (tiered policy over the whole group),
    the others run group.run_agent (their 8 ranks per generation). See group.py."""
    sys.path.insert(0, '/workspace/experiments/claude_v3l')
    from flagship_harness import repair_run_dir
    from group import run_agent, sha256
    from supervisor import Supervisor
    info = modal.experimental.get_cluster_info()
    node, ips = info.rank, info.container_ips
    ip0 = ips[0]
    coord = f'[{ip0}]' if ':' in ip0 else ip0
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', '/flagship/.jax-compilation-cache')
    print(f'[v3l] node {node}/{len(ips)} start {time.strftime("%Y-%m-%d %H:%M:%S %Z")} source {SOURCE_SHA} ips {ips}',
          flush=True)
    vol.reload()
    run_dir = Path('/flagship/runs') / manifest['run_id']
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest, cwd='/workspace')
    if node != 0:
        out = run_agent(control, manifest['run_id'], node, reload=vol.reload, run_dir=str(run_dir), cwd='/workspace')
        return dict(node=node, agent=out)

    def prepare_gen():
        repair_run_dir(run_dir, log=lambda m: print(m, flush=True))
        vol.commit()
        latest = run_dir / 'latest.npz'
        return sha256(latest) if latest.exists() else None
    out = Supervisor(manifest, launch_id, run_dir, control, commit=vol.commit, coord_host=coord,
                     prepare_gen=prepare_gen).run()
    vol.commit()
    return dict(out, node=0)


@app.function(image=image, gpu='H100!:8', cpu=32, memory=131072, timeout=86400, retries=RETRIES, scaledown_window=2,
              volumes={'/flagship': flag_vol})
@modal.clustered(size=2, rdma=True)
def stage_worker_16(manifest: dict, launch_id: str) -> dict:
    """A FLAGSHIP stage on 16 H100 (2 clustered nodes, RDMA, one trainer process per GPU; session A's P16 layout)."""
    if manifest.get('mode') != 'flagship' or manifest.get('faults'):
        raise SystemExit('stage_worker_16: flagship manifests only, without faults')
    return _work_group(manifest, launch_id, flag_vol)


@app.function(image=image, gpu='H100!:8', cpu=32, memory=131072, timeout=5400, retries=RETRIES, scaledown_window=2,
              volumes={'/flagship': qual_vol})
@modal.clustered(size=2, rdma=True)
def qual_worker_16(manifest: dict, launch_id: str) -> dict:
    """Qualification on 16 H100 (2 clustered nodes), own qualification volume."""
    if manifest.get('mode') != 'qualification':
        raise SystemExit('qual_worker_16: qualification manifests only')
    return _work_group(manifest, launch_id, qual_vol)


@app.function(image=image, gpu='H100!', cpu=4, memory=32768, timeout=2700, retries=RETRIES, scaledown_window=2,
              volumes={'/flagship': qual_vol})
def qual_worker_1(manifest: dict, launch_id: str) -> dict:
    """Qualification on 1xH100 (Q0 harness tests), own qualification volume."""
    if manifest.get('mode') != 'qualification':
        raise SystemExit('qual_worker_1: qualification manifests only')
    return _work(manifest, launch_id, qual_vol)


@app.function(image=cpu_image, cpu=1, memory=4096, timeout=900, retries=0,
              volumes={'/e': e_vol.read_only(), '/fq': qual_vol, '/ff': flag_vol})
def seed_map_table() -> dict:
    """Copy E's classic 14x14 map table into both L volumes (sha256 recorded)."""
    import hashlib
    import shutil
    src = Path('/e') / MAP_TABLE
    h = hashlib.sha256(src.read_bytes()).hexdigest()
    out = dict(src=str(src), sha256=h, bytes=src.stat().st_size)
    for root, vol in (('/fq', qual_vol), ('/ff', flag_vol)):
        dst = Path(root) / MAP_TABLE
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == h
        vol.commit()
        out[root] = str(dst)
    return out


@app.function(image=cpu_image, cpu=1, memory=2048, timeout=300, retries=0, volumes={'/fq': qual_vol, '/ff': flag_vol})
def read_state(volume: str, run_id: str) -> dict:
    """The monitor's volume view of a run: status, heartbeat, attempts, resumes, nan and the last 300 events."""
    vol, root = (flag_vol, '/ff') if volume == 'flagship' else (qual_vol, '/fq')
    vol.reload()
    d = Path(root) / 'runs' / run_id / 'v3l'
    out = {}
    for name in ('status.json', 'heartbeat.json'):
        p = d / name
        out[name] = json.loads(p.read_text()) if p.exists() else None
    for name in ('attempts.jsonl', 'resumes.jsonl', 'nan.jsonl', 'events.jsonl'):
        p = d / name
        rows = []
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        out[name] = rows[-300:]
    good = Path(root) / 'runs' / run_id / 'GOOD.json'
    out['GOOD.json'] = json.loads(good.read_text()) if good.exists() else None
    return out


ARENA_ENV = {'POLYTOPIA_TRANSFORMER_ATTENTION': 'tokamax-triton-bfloat16', 'POLYTOPIA_LAYER_NORM': 'tokamax-affine',
             'POLYTOPIA_TRANSFORMER_REMAT': '0', 'PC_CUDA_WARP_TICK': '1', 'POLYTOPIA_TRUNK_DTYPE': 'bfloat16',
             'POLYTOPIA_MATMUL_PRECISION': 'HIGHEST', 'POLYTOPIA_HEAD_PRECISION': 'HIGHEST',
             'XLA_PYTHON_CLIENT_MEM_FRACTION': '.6'}


@app.function(image=image, gpu='H100!', cpu=4, memory=32768, timeout=3600, retries=0, scaledown_window=2,
              volumes={'/flagship': qual_vol, '/ff': flag_vol, '/e': e_vol.read_only(), '/g': g_vol.read_only(),
                       '/h': h_vol.read_only(), '/r': r_vol.read_only()})
def arena_worker(matches: list) -> dict:
    """Session E's gpu_arena.py per match (paired maps, both seats), results to /flagship/arena/<name>.json.
    Checkpoints: /flagship/runs/... (L qualification), /ff/runs/... (flagship + CNN control), /e, /g, /h (other
    sessions, read-only), /r/checkpoints/friend-champion (champion), /workspace/benchmark/references/hero-1v1.npz."""
    qual_vol.reload()
    flag_vol.reload()
    started, rows = time.time(), []
    env = dict(os.environ, **ARENA_ENV, JAX_COMPILATION_CACHE_DIR='/flagship/.jax-compilation-cache')
    for m in matches:
        t = time.time()
        try:
            p = subprocess.run(['python', '/workspace/experiments/claude_v3/gpu_arena.py', json.dumps(m)],
                               capture_output=True, text=True, timeout=m.get('timeout', 1500), env=env)
            found = [l[7:] for l in p.stdout.splitlines() if l.startswith('RESULT ')]
            row = json.loads(found[-1]) if found else dict(error=(p.stderr or p.stdout)[-3000:])
        except subprocess.TimeoutExpired:
            row = dict(error=f"timeout after {m.get('timeout', 1500)} s")
        row.update(name=m.get('name'), match=m, wall_seconds=round(time.time() - t, 1))
        out = Path('/flagship/arena') / f"{m.get('name', 'match')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=1))
        qual_vol.commit()
        print(json.dumps({k: v for k, v in row.items() if k != 'progress'})[:3000], flush=True)
        rows.append(row)
    return dict(rows=rows, container_seconds=round(time.time() - started, 1))
