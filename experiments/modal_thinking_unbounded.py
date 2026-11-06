"""80-game inference-only comparison without a benchmark token/time budget."""
import hashlib
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
    .run_commands('uv pip install --system vllm==0.19.0 transformers==4.57.6 peft==0.17.1 accelerate==1.10.1 jax==0.11.2')
    .env({'PYTHONPATH':'/workspace/src','HF_HOME':'/workspace/runs/hf-cache',
          'TOKENIZERS_PARALLELISM':'false','JAX_PLATFORMS':'cpu'})
    .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'scripts','/workspace/scripts',ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'configs','/workspace/configs'))
volume = modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
app = modal.App('polytopia-qwen35-unrestricted-games')


@app.function(image=image,gpu='H100',cpu=8,memory=49152,timeout=86400,
              max_containers=2,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def evaluate(run_id,mode,protocol_sha):
    import subprocess,time,tarfile
    volume.reload()
    root = Path('/workspace/runs')/run_id
    if hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()!=protocol_sha:
        raise ValueError('protocol changed')
    out = root/mode
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    proc = subprocess.Popen(['python','/workspace/scripts/benchmark_thinking_unbounded.py',str(root),mode],cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit()
            time.sleep(15)
        report = json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else None
        result = dict(mode=mode,exit_code=proc.returncode,occupied_seconds=time.monotonic()-started,
                      complete=report is not None and report['complete'])
        (out/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
        with tarfile.open(root/(mode+'.tar'),'w') as archive:
            archive.add(out,arcname=mode)
        volume.commit()
        return result
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        volume.commit()


@app.function(image=image,cpu=.125,memory=1024,timeout=86400,
              max_containers=1,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def coordinate(run_id,protocol_sha):
    """Own both GPU calls and final aggregation remotely, independent of desktop."""
    import subprocess,time,traceback
    volume.reload()
    root=Path('/workspace/runs')/run_id
    if hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()!=protocol_sha:
        raise ValueError('protocol changed')
    if (root/'calls.json').exists():raise ValueError('run already dispatched')
    status=dict(status='dispatching',started_unix=time.time(),protocol_sha256=protocol_sha)
    calls={}
    try:
        for mode in ['off','on']:
            calls[mode]=evaluate.spawn(run_id,mode,protocol_sha).object_id
        (root/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
        status.update(status='waiting',calls=calls)
        (root/'collection.json').write_text(json.dumps(status,indent=2)+'\n');volume.commit()
        results={}
        for mode,call_id in calls.items():
            results[mode]=modal.FunctionCall.from_id(call_id).get()
            print(json.dumps(dict(event='MODE_FINISHED',**results[mode])),flush=True)
        volume.reload()
        subprocess.run(['python','/workspace/scripts/summarize_unrestricted_games.py',str(root)],check=True)
        summary=json.loads((root/'summary.json').read_text())
        a,b=summary['modes']['off'],summary['modes']['on']
        lines=['# Qwen3.5-4B restarted gameplay comparison','',
            f'Run: `{run_id}`. Forty matched games per mode; current harness and unadapted weights. '
            'No separate output or thinking token budget.','',
            '| Metric | Thinking off | Thinking on |','|---|---:|---:|']
        for title,key in [('Native wins','native_wins'),('Native losses','native_losses'),
            ('Invalid-action forfeits','forfeits'),('Valid actions','valid_actions'),
            ('Completed model calls','calls'),('Output tokens','output_tokens')]:
            lines.append(f'| {title} | {a[key]} | {b[key]} |')
        lines+=['',f"Off statuses: `{a['status_counts']}`.",f"On statuses: `{b['status_counts']}`.",'',
            'Context exhaustion, truncations and infrastructure failures are not native losses or draws.',
            f"Paired comparison: `{summary['paired_win_difference']}`.",'',
            'See protocol.json and summary.json for all per-game outcomes, runtime and limitations.','']
        (root/'result.md').write_text('\n'.join(lines))
        status.update(status='complete',results=results,finished_unix=time.time())
    except Exception:
        status.update(status='failed',calls=calls,error=traceback.format_exc(),finished_unix=time.time())
        raise
    finally:
        (root/'collection.json').write_text(json.dumps(status,indent=2)+'\n');volume.commit()
    return status


@app.local_entrypoint()
def main(run_id: str='qwen35-thinking-games-unrestricted-v2'):
    from modal.config import config
    if config.get('environment')!='staging':raise ValueError('staging required')
    root = ROOT/'runs'/run_id
    if (root/'calls.json').exists() or (root/'coordinator-call.json').exists():
        raise ValueError('already dispatched; do not duplicate')
    with volume.batch_upload() as upload:
        upload.put_file(root/'protocol.json',run_id+'/protocol.json')
    sha = hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
    coordinator=coordinate.spawn(run_id,sha)
    (root/'coordinator-call.json').write_text(json.dumps(dict(call_id=coordinator.object_id,
        protocol_sha256=sha),indent=2)+'\n')
    import time
    for _ in range(60):
        try:data=b''.join(volume.read_file(run_id+'/calls.json'))
        except FileNotFoundError:time.sleep(1);continue
        (root/'calls.json').write_bytes(data)
        break
    print(json.dumps(coordinator.get()),flush=True)
    for name in ['collection.json','summary.json','result.md']:
        (root/name).write_bytes(b''.join(volume.read_file(run_id+'/'+name)))
    for mode in ['off','on']:
        target = root/(mode+'.tar')
        with target.open('wb') as f:
            for chunk in volume.read_file(run_id+'/'+mode+'.tar'):f.write(chunk)
        import tarfile
        with tarfile.open(target) as archive:archive.extractall(root,filter='data')
