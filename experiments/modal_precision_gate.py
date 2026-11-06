"""TF32 precision gate for the distributed PPO learner, plus learner throughput per precision mode.

    modal run --env staging experiments/modal_precision_gate.py::main             # rows -> gate + bench
    modal run --env staging experiments/modal_precision_gate.py::main --skip-rows # reuse saved rows

Actors run the policy in fp32 on CPU and record log-probs; the learner recomputes them on the GPU. With
TF32 on the learner the importance ratio at epoch 0 is no longer exactly 1. make_rows generates real actor
rows (sampled self-play, the actor's transport format); gate recomputes them on one H100 per precision
mode; bench times the PPO epoch per mode. Precision is read by rl.network at import, so each mode runs in
a fresh subprocess (experiments/precision_gate_worker.py) with its environment set.
"""
from pathlib import Path
import json
import os
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
WORKER = '/workspace/experiments/precision_gate_worker.py'
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_file(ROOT / 'experiments/precision_gate_worker.py', WORKER))
gpu_image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda12]==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cuda', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_file(ROOT / 'experiments/precision_gate_worker.py', WORKER))
app = modal.App('polytopia-precision-gate')
results = modal.Volume.from_name('polytopia-structured-rl', create_if_missing=False)
retained = modal.Volume.from_name('polytopia-structured-rl-retained', create_if_missing=False)
VOLUMES = {'/results': results, '/retained': retained}
OUT = '/results/precision-gate'
CHECKPOINTS = {14: '/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz',
               18: '/retained/checkpoints/distributed-final-v1/c3-u4854-grown18x2/final.npz'}
# (POLYTOPIA_MATMUL_PRECISION, POLYTOPIA_HEAD_PRECISION): convs, heads
MODES = {'fp32': ('HIGHEST', 'HIGHEST'), 'tf32': ('DEFAULT', 'DEFAULT'), 'tf32-torso': ('DEFAULT', 'HIGHEST')}
LEDGER = '/Users/K.Z./Desktop/projects/polytopia-bot/runs/ablation-v1'  # shared across worktrees
RUN_ID = 'precision-gate'


def _worker(*args, env=None):
    import subprocess
    out = subprocess.run(['python', WORKER, *map(str, args)], capture_output=True, text=True,
                         env=dict(os.environ, **(env or {})))
    if out.returncode:
        raise RuntimeError(out.stderr[-4000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def _mode_env(mode):
    convs, heads = MODES[mode]
    return {'POLYTOPIA_MATMUL_PRECISION': convs, 'POLYTOPIA_HEAD_PRECISION': heads}


@app.function(image=cpu_image, cpu=16, memory=65536, timeout=3600, volumes=VOLUMES, retries=0)
def make_rows(size: int, processes: int = 16, games: int = 16, ticks: int = 640, stride: int = 8):
    """~processes*games*ticks/stride actor rows (default 20,480) from one checkpoint."""
    retained.reload()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    summary = _worker('rows', CHECKPOINTS[size], size, f'{OUT}/rows-{size}.npz', processes, games, ticks, stride, size)
    results.commit()
    return summary


@app.function(image=gpu_image, gpu='H100', cpu=8, memory=65536, timeout=3600, volumes=VOLUMES, retries=0)
def gate(sizes: list):
    retained.reload()
    results.reload()
    out = []
    for mode in MODES:
        for size in sizes:
            path = f'{OUT}/gate-{mode}-{size}.json'
            row = _worker('gate', CHECKPOINTS[size], size, f'{OUT}/rows-{size}.npz', path, env=_mode_env(mode))
            out.append(dict(mode=mode, **row))
            print(json.dumps(out[-1]), flush=True)
    results.commit()
    return out


@app.function(image=gpu_image, gpu='H100', cpu=16, memory=65536, timeout=3600, volumes=VOLUMES, retries=0)
def bench(size: int, epochs: int = 4):
    retained.reload()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    config = {'size': size, 'batch': 65536, 'minibatch': 4096}
    out = []
    for mode in MODES:
        row = _worker('bench', CHECKPOINTS[size], json.dumps(config), epochs, f'{OUT}/bench-{mode}-{size}.json',
                      env=_mode_env(mode))
        out.append(dict(mode=mode, **row))
        print(json.dumps(out[-1]), flush=True)
    results.commit()
    return out


@app.local_entrypoint()
def main(skip_rows: bool = False, skip_bench: bool = False, sizes: str = '14,18'):
    import sys
    sys.path.insert(0, str(ROOT / 'src'))
    from polytopia_bot.rl.budget import record
    sizes = [int(s) for s in sizes.split(',')]
    ledger = json.loads((Path(LEDGER) / 'budget.json').read_text())
    row = next(r for r in ledger['runs'] if r['run_id'] == RUN_ID)
    if row.get('modal_app_id'):  # a rerun: keep every app on the row for ledger_spend
        record(LEDGER, RUN_ID, extra_app_ids=[*row.get('extra_app_ids', []), app.app_id], status='running')
    else:
        record(LEDGER, RUN_ID, modal_app_id=app.app_id, status='running')
    benches = [] if skip_bench else [bench.spawn(s) for s in sizes]
    if not skip_rows:
        for s, r in zip(sizes, [make_rows.spawn(s) for s in sizes]):
            print('rows', s, json.dumps(r.get()))
    report = dict(gate=gate.remote(sizes), bench=[b for c in benches for b in c.get()])
    folder = ROOT / 'docs/ablation-v1'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    (folder / f'precision-gate-{stamp}.json').write_text(json.dumps(report, indent=1) + '\n')
    for g in report['gate']:
        print('gate', g['mode'], g['size'], {k: g[k] for k in ('mean_ratio', 'mean_k3', 'max_abs_log_r', 'p999_abs_log_r',
                                                               'passes')})
    for b in report['bench']:
        print('bench', b['mode'], b['config']['size'], round(b['sample_epochs_per_s']))
    record(LEDGER, RUN_ID, status='done')
