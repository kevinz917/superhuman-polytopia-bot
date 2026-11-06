"""Retain verified checkpoints independently of the laptop; never control training."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.12')
    .env({'PYTHONPATH': '/workspace'})
    .add_local_file(ROOT/'scripts/watch_structured_checkpoints.py',
                    '/workspace/watch_structured_checkpoints.py'))
app = modal.App('polytopia-checkpoint-retainer', image=image)
training = modal.Volume.from_name('polytopia-structured-rl', create_if_missing=False)
retained = modal.Volume.from_name('polytopia-structured-rl-retained', create_if_missing=True)


@app.function(cpu=(1, 1), memory=(2048, 2048), timeout=43200,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={'/training': training.with_mount_options(read_only=True),
                       '/retained': retained})
def retain(run_id: str, run_ids: list[str], seconds: int, interval: int):
    import time
    from watch_structured_checkpoints import capture, safe_name, write_json
    for name in [run_id, *run_ids]:
        safe_name(name)
    if (not 1 <= len(run_ids) <= 32 or len(set(run_ids)) != len(run_ids)
            or not 1 <= seconds <= 39600 or not 60 <= interval <= 600):
        raise ValueError('invalid bounded retention allocation')
    status_dir = Path('/retained')/run_id
    status_dir.mkdir(exist_ok=False)
    destination = Path('/retained/checkpoints')
    destination.mkdir(exist_ok=True)
    started = time.monotonic()
    captured, finished = set(), set()
    read = lambda name: (Path('/training')/name).read_bytes()
    try:
        while time.monotonic()-started < seconds:
            training.reload()
            errors = []
            for name in run_ids:
                if name in finished:
                    continue
                try:
                    snapshot = capture(read, name, destination, checkpoint='final')
                    if snapshot is not None:
                        finished.add(name)
                    else:
                        snapshot = capture(read, name, destination)
                    if snapshot is not None:
                        captured.add(str(snapshot))
                except Exception as exc:
                    # A torn or unpublished checkpoint is observed again later.
                    # No restart, stop, deletion or mutation of training is possible.
                    errors.append(dict(run_id=name, error=type(exc).__name__, detail=str(exc)[:500]))
            report = dict(updated_unix=time.time(), monitored_run_ids=run_ids,
                captured=sorted(captured), finished=sorted(finished), errors=errors,
                seconds=time.monotonic()-started, source_mount_read_only=True,
                source_volume='polytopia-structured-rl',
                destination_volume='polytopia-structured-rl-retained')
            write_json(status_dir/'status.json', report)
            retained.commit()
            print(json.dumps(dict(captured=len(captured), finished=len(finished), errors=errors)), flush=True)
            if len(finished) == len(run_ids):
                break
            time.sleep(min(interval, max(0, seconds-(time.monotonic()-started))))
        return report
    finally:
        retained.commit()


@app.local_entrypoint()
def main(run_id: str, run_ids: str, seconds: int=39600, interval: int=60):
    import sys
    import time
    from modal.config import config
    sys.path.insert(0, str(ROOT/'src'))
    from polytopia_bot.rl.budget import reserve, record
    if config.get('environment') != 'staging':
        raise RuntimeError('Pass --env staging')
    names = run_ids.split(',')
    folder = ROOT/'runs/structured-ppo-v3'
    ledger = json.loads((folder/'budget.json').read_text())
    owned = {r['run_id'] for r in ledger['runs'] if r.get('function_call_id')}
    if (not 1 <= len(names) <= 32 or len(set(names)) != len(names)
            or not set(names) <= owned or not 1 <= seconds <= 39600
            or not 60 <= interval <= 600):
        raise ValueError('retention requires bounded, ledger-owned run IDs')
    reserve(folder, dict(run_id=run_id, monitored_run_ids=names, seconds=seconds,
        interval=interval, resource_type='cpu-checkpoint-retainer', gpu_type=None,
        cpu_cores=1, memory_gib=2, rate_usd_per_hour=.07, hard_timeout_seconds=43200,
        reserved_usd=2, status='reserved', modal_app_id=app.app_id, start_unix=time.time(),
        price_source='https://modal.com/pricing'))
    try:
        call = retain.spawn(run_id, names, seconds, interval)
        record(folder, run_id, status='submitted', function_call_id=call.object_id)
        print(json.dumps(dict(run_id=run_id, modal_app_id=app.app_id,
                             function_call_id=call.object_id)), flush=True)
    except BaseException:
        record(folder, run_id, status='client_error_check_remote_state')
        raise
