"""Unit/technology composition of retained public traces, computed entirely in Modal.

Reads *.jsonl.gz traces under /retained/qualifications/<run>/<job>/ through a
read-only mount and returns only the JSON summary (no inference, no gameplay).
Usage: modal run --env staging --detach experiments/modal_trace_analysis.py \
         --run-id <analysis-id> --jobs <qualification_run>/<job>,...
"""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.12')
    .add_local_file(ROOT/'scripts/analyze_unit_composition.py', '/workspace/scripts/analyze_unit_composition.py')
    .add_local_file(ROOT/'src/polytopia_bot/gpu/rules.json', '/workspace/src/polytopia_bot/gpu/rules.json'))
app = modal.App('polytopia-trace-analysis', image=image)
retained = modal.Volume.from_name('polytopia-structured-rl-retained', environment_name='staging')


@app.function(cpu=(1, 1), memory=(2048, 2048), timeout=3600, max_containers=1, retries=0,
              scaledown_window=2, volumes={'/retained': retained.with_mount_options(read_only=True)})
def analyze(jobs):
    import gzip
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from analyze_unit_composition import game, aggregate
    out = {}
    for job in jobs:
        folder = Path('/retained/qualifications') / job
        traces = sorted(folder.glob('*.jsonl.gz'))
        games = [game(gzip.open(p, 'rt').read().splitlines()) for p in traces]
        out[job] = dict(aggregate=aggregate(games), games=games)
    return out


@app.local_entrypoint()
def main(run_id: str, jobs: str):
    import sys
    import time
    sys.path.insert(0, str(ROOT/'src'))
    from polytopia_bot.rl.budget import reserve, record
    folder = ROOT/'runs/structured-ppo-v3'
    names = [j for j in jobs.split(',') if j]
    reserve(folder, dict(run_id=run_id, resource_type='cpu-checkpoint-verifier', gpu_type=None, cpu_cores=1,
        memory_gib=2, rate_usd_per_hour=.07, reserved_usd=2, hard_timeout_seconds=3600, status='reserved',
        modal_app_id=app.app_id, start_unix=time.time(), jobs=names))
    call = analyze.spawn(names)
    record(folder, run_id, status='submitted', function_call_id=call.object_id)
    print(json.dumps(dict(run_id=run_id, modal_app_id=app.app_id, function_call_id=call.object_id)), flush=True)
