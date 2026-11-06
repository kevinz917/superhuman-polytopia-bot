"""Continue thinking games interrupted by the 4-hour Modal worker limit; all on Modal.

  modal run --env staging -m experiments.modal_thinking_continuation --action test
  modal run --detach --env staging -m experiments.modal_thinking_continuation \
      --action continue --run-id qwen35-thinking-30paired-v1

The coordinator refuses to start while any original thinking worker is still running.
"""
import json
from pathlib import Path
import time
import modal
from experiments.modal_thinking_repaired import image, volume, ROOT

app = modal.App('polytopia-thinking-continuation')
WORKER_TIMEOUT = 14400


@app.function(image=image.add_local_dir(ROOT/'tests', '/workspace/tests', ignore=['**/__pycache__/**']),
              cpu=4, memory=16384, timeout=3600)
def test():
    import subprocess, os
    result = subprocess.run(['python', '/workspace/tests/test_async_resume.py'], cwd='/workspace',
        env=dict(os.environ, PYTHONPATH='/workspace/src:/workspace/tests'), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-4000:]+result.stderr[-8000:])
    return result.stdout[-4000:]


@app.function(image=image.add_local_dir(ROOT/'tests', '/workspace/tests', ignore=['**/__pycache__/**']),
              cpu=4, memory=16384, timeout=3600)
def test_chain():
    import subprocess, os
    result = subprocess.run(['python', '/workspace/tests/test_continuation_chain.py'], cwd='/workspace',
        env=dict(os.environ, PYTHONPATH='/workspace/src:/workspace/tests:/workspace/scripts'), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-4000:]+result.stderr[-8000:])
    return result.stdout[-4000:]


LONG_TIMEOUT = 86400  # Modal's maximum; later segments should not need another restart.


def status_name(segment):
    return 'continuation-collection.json' if segment == 1 else f'continuation-{segment}-collection.json'


def assumed_interrupted_seconds(root):
    """Interrupted workers leave no exit receipt; charge them their full platform timeout."""
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from continue_thinking_games import segment_on, segment_execution, segments_present, shard_ids
    out = {}
    for k in range(0, segments_present(root)+1):
        for s in shard_ids(root):
            on = segment_on(root, k, s)
            if (on/'games').exists() and not ((on/'report.json').exists() and segment_execution(root, k, s).exists()):
                out[f'{k}:{s}'] = float(WORKER_TIMEOUT if k <= 1 else LONG_TIMEOUT)
    return out


@app.function(image=image, gpu='H100', cpu=8, memory=49152, timeout=LONG_TIMEOUT,
              max_containers=4, retries=0, scaledown_window=2, volumes={'/workspace/runs': volume})
def worker_long(run_id, shard, segment):
    import subprocess
    sys_path = '/workspace/scripts'
    volume.reload(); root = Path('/workspace/runs')/run_id
    folder = root/('continuation' if segment == 1 else f'continuation-{segment}')/'shards'/str(shard)
    started = time.monotonic()
    proc = subprocess.Popen(['python', sys_path+'/continue_thinking_games.py', 'worker', str(root), str(shard), str(segment)],
                            cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit(); time.sleep(10)
        result = dict(mode='on', shard=shard, segment=segment, exit_code=proc.returncode,
                      worker_seconds=time.monotonic()-started, finished_unix=time.time())
        (folder/'on-execution.json').write_text(json.dumps(result)+'\n'); volume.commit()
        if proc.returncode:
            raise RuntimeError(result)
        return result
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=15)
        volume.commit()


@app.function(image=image, cpu=2, memory=16384, timeout=LONG_TIMEOUT, max_containers=1, retries=0,
              volumes={'/workspace/runs': volume})
def coordinate_segment(run_id, segment):
    import sys, traceback
    sys.path.insert(0, '/workspace/scripts')
    from continue_thinking_games import prepare, finalize
    if segment < 2:
        raise ValueError('segment 1 uses coordinate()')
    volume.reload(); root = Path('/workspace/runs')/run_id
    previous = json.loads((root/status_name(segment-1)).read_text())
    terminations = {}
    for shard, call_id in previous.get('calls', {}).items():
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
            terminations[shard] = dict(kind='returned', result=result)
        except TimeoutError:
            raise RuntimeError(f'segment {segment-1} worker {shard} is still running; refusing to continue')
        except Exception as exc:
            terminations[shard] = dict(kind=type(exc).__name__, message=str(exc)[:500])
    status = dict(status='preparing', segment=segment, started_unix=time.time(), terminations=terminations)
    status_path = root/status_name(segment)
    def publish(**kw):
        status.update(kw, observed_unix=time.time())
        status_path.write_text(json.dumps(status, indent=2)+'\n'); volume.commit()
    try:
        publish()
        receipt = prepare(root, terminations, segment=segment); volume.commit()
        spawned = {s: worker_long.spawn(run_id, int(s), segment).object_id
                   for s, info in receipt['shards'].items() if info['continued']}
        publish(status='waiting', calls=spawned)
        results = {}
        while len(results) < len(spawned):
            for s, call_id in spawned.items():
                if s in results:
                    continue
                try:
                    results[s] = modal.FunctionCall.from_id(call_id).get(timeout=0)
                except TimeoutError:
                    pass
            if len(results) < len(spawned):
                publish(status='waiting', calls=spawned, completed=results); time.sleep(30)
        volume.reload()
        assumed = assumed_interrupted_seconds(root)
        segments = finalize(root, assumed); volume.commit()
        summary = json.loads((root/'final'/'summary.json').read_text())
        resolved = all(g['status'] in ('native', 'forfeit') for s in summary['modes'].values() for g in s['games'])
        publish(status='complete' if resolved else 'completed_with_unresolved_games', calls=spawned,
                completed=results, segments=segments, interrupted_worker_seconds_assumed=assumed)
    except Exception:
        # Never cancel continuation workers from here: their checkpoints remain resumable.
        publish(status='failed', error=traceback.format_exc())
        raise
    return status


@app.function(image=image, gpu='H100', cpu=8, memory=49152, timeout=WORKER_TIMEOUT,
              max_containers=2, retries=0, scaledown_window=2, volumes={'/workspace/runs': volume})
def worker(run_id, shard):
    import subprocess
    volume.reload(); root = Path('/workspace/runs')/run_id
    folder = root/'continuation'/'shards'/str(shard)
    started = time.monotonic()
    proc = subprocess.Popen(['python', '/workspace/scripts/continue_thinking_games.py', 'worker', str(root), str(shard)],
                            cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit(); time.sleep(10)
        result = dict(mode='on', shard=shard, exit_code=proc.returncode,
                      worker_seconds=time.monotonic()-started, finished_unix=time.time())
        (folder/'on-execution.json').write_text(json.dumps(result)+'\n'); volume.commit()
        if proc.returncode:
            raise RuntimeError(result)
        return result
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=15)
        volume.commit()


@app.function(image=image, cpu=2, memory=16384, timeout=15000, max_containers=1, retries=0,
              volumes={'/workspace/runs': volume})
def coordinate(run_id):
    import sys, traceback
    sys.path.insert(0, '/workspace/scripts')
    from continue_thinking_games import prepare, finalize
    volume.reload(); root = Path('/workspace/runs')/run_id
    calls = json.loads((root/'calls.json').read_text())
    terminations = {}
    for name in ['on-0', 'on-1']:
        try:
            result = modal.FunctionCall.from_id(calls[name]).get(timeout=0)
            terminations[name] = dict(kind='returned', result=result)
        except TimeoutError:
            raise RuntimeError(f'{name} is still running; refusing to continue its games')
        except Exception as exc:
            terminations[name] = dict(kind=type(exc).__name__, message=str(exc)[:500])
    status = dict(status='preparing', started_unix=time.time(), terminations=terminations)
    status_path = root/'continuation-collection.json'
    def publish(**kw):
        status.update(kw, observed_unix=time.time())
        status_path.write_text(json.dumps(status, indent=2)+'\n'); volume.commit()
    try:
        publish()
        receipt = prepare(root, terminations); volume.commit()
        spawned = {}
        for shard, info in receipt['shards'].items():
            if info['continued']:
                spawned[shard] = worker.spawn(run_id, int(shard)).object_id
        publish(status='waiting', calls=spawned)
        results = {}
        while len(results) < len(spawned):
            for shard, call_id in spawned.items():
                if shard in results:
                    continue
                try:
                    results[shard] = modal.FunctionCall.from_id(call_id).get(timeout=0)
                except TimeoutError:
                    pass
            if len(results) < len(spawned):
                publish(status='waiting', calls=spawned, completed=results); time.sleep(30)
        volume.reload()
        # The interrupted workers ran until the platform limit; no exit receipt exists for them.
        original_seconds = assumed_interrupted_seconds(root)
        segments = finalize(root, original_seconds); volume.commit()
        summary = json.loads((root/'final'/'summary.json').read_text())
        resolved = all(g['status'] in ('native', 'forfeit') for s in summary['modes'].values() for g in s['games'])
        publish(status='complete' if resolved else 'completed_with_unresolved_games', calls=spawned,
                completed=results, segments=segments, original_worker_seconds_assumed=original_seconds)
    except Exception:
        # Never cancel continuation workers from here: their checkpoints remain resumable.
        publish(status='failed', error=traceback.format_exc())
        raise
    return status


@app.function(image=image, cpu=2, memory=16384, timeout=1800, volumes={'/workspace/runs': volume.read_only()})
def rehearse(run_id):
    """Reconcile the live run's committed checkpoints without writing to the volume."""
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from continue_thinking_games import prepare
    receipt = prepare(Path('/workspace/runs')/run_id, dict(note='rehearsal while workers run'),
                      out='/tmp/rehearsal/continuation')
    return {s: dict(continued=len(i['continued']), finished_originally=len(i.get('finished_originally', [])),
                    kept=i.get('timing_rows_kept'), abandoned=i.get('timing_rows_abandoned'),
                    sample={k: v for k, v in list(i['continued'].items())[:2]} if i['continued'] else None)
            for s, i in receipt['shards'].items()}


@app.function(image=image, cpu=2, memory=16384, timeout=3600, volumes={'/workspace/runs': volume})
def finalize_only(run_id):
    """Re-run only the stitch/summary step after continuation workers finished."""
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from continue_thinking_games import finalize
    volume.reload(); root = Path('/workspace/runs')/run_id
    segments = finalize(root, assumed_interrupted_seconds(root)); volume.commit()
    return segments


@app.local_entrypoint()
def main(action: str = 'test', run_id: str = '', segment: int = 2):
    if action == 'test':
        print(test.remote()); return
    if action == 'test-chain':
        print(test_chain.remote()); return
    if action == 'continue-segment':
        local = ROOT/'runs'/run_id/f'continuation-{segment}-dispatch.json'
        if local.exists():
            raise ValueError(f'segment {segment} already dispatched')
        call = coordinate_segment.spawn(run_id, segment)
        receipt = dict(app_id=app.app_id, call_id=call.object_id, run_id=run_id, segment=segment,
                       launcher_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest())
        local.write_text(json.dumps(receipt, indent=2)+'\n'); print(json.dumps(receipt)); return
    if action == 'rehearse':
        print(json.dumps(rehearse.remote(run_id), indent=1)); return
    if action == 'finalize':
        print(json.dumps(finalize_only.remote(run_id), indent=1)); return
    if action != 'continue' or not run_id:
        raise ValueError('use --action test, or --action continue --run-id RUN')
    local = ROOT/'runs'/run_id/'continuation-dispatch.json'
    if local.exists():
        raise ValueError('continuation already dispatched')
    call = coordinate.spawn(run_id)
    receipt = dict(app_id=app.app_id, call_id=call.object_id, run_id=run_id,
                   launcher_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest())
    local.write_text(json.dumps(receipt, indent=2)+'\n'); print(json.dumps(receipt))
