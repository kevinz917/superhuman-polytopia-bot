"""Unadapted Qwen3-4B-Instruct-2507 on the qwen35-thinking-30paired-v1 games; all execution on Modal.

Launch (module form):
  modal run --detach --env staging -m experiments.modal_qwen3_instruct_baseline --run-id qwen3-instruct-2507-30paired-v1
"""
import hashlib
import json
from pathlib import Path
import modal
from experiments.modal_thinking_repaired import image, volume, ROOT

app = modal.App('polytopia-qwen3-instruct-baseline')
QWEN35 = 'qwen35-thinking-30paired-v1'


@app.function(image=image, gpu='H100', cpu=8, memory=49152, timeout=14400,
              max_containers=1, retries=0, scaledown_window=2, volumes={'/workspace/runs': volume})
def worker(run_id, sha):
    import subprocess, time
    volume.reload(); root = Path('/workspace/runs')/run_id
    if hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest() != sha:
        raise ValueError('protocol changed')
    if (root/'instruct').exists():
        raise ValueError('instruct output already exists')
    started = time.monotonic()
    proc = subprocess.Popen(['python', '/workspace/scripts/benchmark_qwen3_instruct.py', str(root)], cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit(); time.sleep(10)
        result = dict(exit_code=proc.returncode, worker_seconds=time.monotonic()-started, finished_unix=time.time())
        (root/'instruct-execution.json').write_text(json.dumps(result, indent=2)+'\n'); volume.commit()
        if proc.returncode:
            raise RuntimeError(result)
        return result
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=15)
        volume.commit()


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes={'/workspace/runs': volume})
def analyze(run_id):
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from summarize_qwen3_instruct_baseline import summarize
    volume.reload()
    small = summarize(Path('/workspace/runs')/run_id, Path('/workspace/runs')/QWEN35)
    volume.commit()
    return small


@app.function(image=image, cpu=.25, memory=2048, timeout=15000,
              max_containers=1, retries=0, scaledown_window=2, volumes={'/workspace/runs': volume})
def coordinate(run_id, sha):
    import time, traceback
    volume.reload(); root = Path('/workspace/runs')/run_id
    status = dict(status='dispatching', started_unix=time.time()); call = None
    try:
        if (root/'calls.json').exists():
            raise ValueError('already dispatched')
        call = worker.spawn(run_id, sha)
        (root/'calls.json').write_text(json.dumps(dict(worker=call.object_id), indent=2)+'\n')
        status.update(status='waiting', calls=dict(worker=call.object_id))
        (root/'collection.json').write_text(json.dumps(status, indent=2)+'\n'); volume.commit()
        status['worker'] = call.get()
        volume.reload()
        small = analyze.remote(run_id)
        volume.reload()
        resolved = all(n not in small['arms'] or small['arms'][n]['operational_failures'] == 0 for n in ('instruct',))
        status.update(status='complete' if resolved else 'completed_with_unresolved_games', finished_unix=time.time())
    except Exception:
        status.update(status='failed', error=traceback.format_exc(), finished_unix=time.time())
        if call is not None:
            try: call.cancel(terminate_containers=True)
            except Exception: pass
        raise
    finally:
        (root/'collection.json').write_text(json.dumps(status, indent=2)+'\n'); volume.commit()
    return status


@app.local_entrypoint()
def main(run_id: str, analyze_only: bool = False):
    from modal.config import config
    if config.get('environment') != 'staging':
        raise ValueError('staging required')
    if run_id == QWEN35:
        raise ValueError('never write into the Qwen3.5 run')
    root = ROOT/'runs'/run_id
    if analyze_only:
        print(json.dumps(analyze.remote(run_id), indent=2)); return
    if (root/'dispatch.json').exists():
        raise ValueError('already dispatched')
    with volume.batch_upload() as upload:
        upload.put_file(root/'protocol.json', run_id+'/protocol.json')
    sha = hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
    call = coordinate.spawn(run_id, sha)
    receipt = dict(app_id=app.app_id, call_id=call.object_id, protocol_sha256=sha,
                   launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'dispatch.json').write_text(json.dumps(receipt, indent=2)+'\n'); print(json.dumps(receipt))
