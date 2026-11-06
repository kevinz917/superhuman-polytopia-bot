"""GPU-type cost benchmark: short real PPO updates (decision-budget collector, fast
training engine) on one GPU per job, one job per (GPU, lanes) pair. No training
output is kept beyond /results/<run_id>/; every job is recorded in the campaign
ledger at its GPU's Modal rate.

  modal run --env staging --detach experiments/modal_gpu_benchmark.py \
      --tag v1 --gpus L40S,H100,B200 --lanes 512,1024,2048

Run ids are bench-gpu-<tag>-<gpu>-<lanes>. Per-GPU rates are Modal list prices
(https://modal.com/pricing, read 2026-10-18) plus 8 CPU cores and 32 GiB.
"""
from pathlib import Path
import json
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda12]==0.11.2', 'numpy>=2.1')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cuda', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_file(ROOT / 'scripts/profile_collector.py', '/workspace/scripts/profile_collector.py'))
app = modal.App('polytopia-gpu-benchmark', image=image)
volume = modal.Volume.from_name('polytopia-structured-rl', create_if_missing=False)

HOST_USD_PER_HOUR = (8 * 0.0000131 + 32 * 0.00000222) * 3600
GPU_USD_PER_HOUR = {'L4': 0.7992, 'A10': 1.1016, 'L40S': 1.9512, 'A100-80GB': 2.4984,
                    'RTX-PRO-6000': 3.0312, 'H100!': 3.9492, 'H200': 4.5396, 'B200': 6.2496, 'B300': 7.0992}
HARD_TIMEOUT = 3600


def rate(gpu):
    return round(GPU_USD_PER_HOUR[gpu] + HOST_USD_PER_HOUR, 6)


def run_id_for(tag, gpu, lanes):
    return f"bench-gpu-{tag}-{gpu.rstrip('!').lower()}-{lanes}"


@app.function(gpu='L40S', cpu=(8, 8), memory=(32768, 32768), timeout=HARD_TIMEOUT,
              retries=0, scaledown_window=2, volumes={'/results': volume})
def bench(run_id: str, gpu: str, parent: str, lanes: int, horizon: int, train_seconds: int, audit_tolerance: float):
    import subprocess
    import sys
    import time
    started = time.monotonic()
    root = Path('/results') / run_id
    root.mkdir(parents=True, exist_ok=False)
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    args = [sys.executable, '/workspace/scripts/profile_collector.py', 'train', '--modes', 'budget',
            '--engines', 'fast', '--lanes', str(lanes), '--horizon', str(horizon),
            '--train-seconds', str(train_seconds), '--checkpoint', f'/results/{parent}/final.npz',
            '--out', str(root / 'train.json'), '--audit-tolerance', str(audit_tolerance)]
    try:
        with open(root / 'log.txt', 'w') as log:
            log.write(json.dumps(dict(requested_gpu=gpu, nvidia_smi=smi)) + '\n')
            print(smi, flush=True)
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            proc.wait()
        return proc.returncode
    finally:
        elapsed = time.monotonic() - started
        (root / 'usage.json').write_text(json.dumps(dict(requested_gpu=gpu, nvidia_smi=smi,
            function_seconds=elapsed, rate_usd_per_hour=rate(gpu), estimate_usd=elapsed * rate(gpu) / 3600,
            scope='function-duration estimate, not invoice')))
        volume.commit()


@app.local_entrypoint()
def main(tag: str, gpus: str = 'L40S', lanes: str = '512,1024,2048', horizon: int = 64,
         train_seconds: int = 420, parent: str = 'diverse-bc-dagger-v1', audit_tolerance: float = 0.):
    import sys
    import time
    from modal.config import config
    sys.path.insert(0, str(ROOT / 'src'))
    from polytopia_bot.rl.budget import reserve, record
    if config.get('environment') != 'staging':
        raise RuntimeError('Pass --env staging')
    gpu_list, lane_list = gpus.split(','), [int(x) for x in lanes.split(',')]
    if not tag.isalnum() or any(g not in GPU_USD_PER_HOUR for g in gpu_list) or not 60 <= train_seconds <= 1500:
        raise ValueError('alphanumeric tag, known GPU names and 60-1500 s of training required')
    folder = ROOT / 'runs/structured-ppo-v3'
    for gpu in gpu_list:
        for n in lane_list:
            run_id = run_id_for(tag, gpu, n)
            reserve(folder, dict(run_id=run_id, parent=parent, mode='gpu-benchmark-train', seconds=train_seconds,
                purpose='GPU cost benchmark: short decision-budget PPO with the fast engine; no checkpoints kept',
                gpu_type=gpu.rstrip('!'), cpu_cores=8, memory_gib=32, environments=n, horizon=horizon,
                audit_tolerance=audit_tolerance or 1e-5,
                rate_usd_per_hour=rate(gpu), reserved_usd=round(rate(gpu) * HARD_TIMEOUT / 3600 + 1, 2),
                hard_timeout_seconds=HARD_TIMEOUT, status='reserved', modal_app_id=app.app_id,
                start_unix=time.time()))
            try:
                call = bench.with_options(gpu=gpu).spawn(run_id, gpu, parent, n, horizon, train_seconds, audit_tolerance)
                record(folder, run_id, status='submitted', function_call_id=call.object_id)
                print(json.dumps(dict(run_id=run_id, gpu=gpu, lanes=n, function_call_id=call.object_id)), flush=True)
            except BaseException:
                record(folder, run_id, status='client_error_check_remote_state')
                raise
