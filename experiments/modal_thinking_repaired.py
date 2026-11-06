"""Targeted diagnostic and fresh paired gameplay; no training jobs."""
import hashlib
import json
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
    .run_commands('uv pip install --system vllm==0.19.0 transformers==4.57.6 peft==0.17.1 accelerate==1.10.1 jax==0.11.2')
    .env({'PYTHONPATH':'/workspace/src','HF_HOME':'/workspace/runs/hf-cache',
          'TOKENIZERS_PARALLELISM':'false','JAX_PLATFORMS':'cpu'})
    .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'scripts','/workspace/scripts',ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'configs','/workspace/configs'))
volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
app=modal.App('polytopia-qwen35-repaired-comparison')


@app.function(image=image,gpu='H100',cpu=8,memory=49152,timeout=14400,
              max_containers=2,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def run(run_id,mode,protocol_sha):
    import subprocess,time
    volume.reload();root=Path('/workspace/runs')/run_id
    if hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()!=protocol_sha:
        raise ValueError('protocol changed')
    proc=subprocess.Popen(['python','/workspace/scripts/benchmark_thinking_repaired.py',str(root),mode],cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit();time.sleep(10)
        result=dict(mode=mode,exit_code=proc.returncode,finished_unix=time.time())
        (root/(mode+'-execution.json')).write_text(json.dumps(result,indent=2)+'\n')
        volume.commit()
        if proc.returncode:raise RuntimeError(result)
        return result
    finally:
        if proc.poll() is None:proc.terminate();proc.wait(timeout=15)
        volume.commit()


@app.function(image=image,cpu=.25,memory=2048,timeout=15000,
              max_containers=1,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def coordinate(run_id,protocol_sha):
    import subprocess,time,traceback
    volume.reload();root=Path('/workspace/runs')/run_id
    calls={};status=dict(status='dispatching',started_unix=time.time())
    try:
        if (root/'calls.json').exists():raise ValueError('already dispatched')
        calls={m:run.spawn(run_id,m,protocol_sha).object_id for m in ['off','on']}
        (root/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
        status.update(status='waiting',calls=calls)
        (root/'collection.json').write_text(json.dumps(status,indent=2)+'\n');volume.commit()
        results={m:modal.FunctionCall.from_id(c).get() for m,c in calls.items()}
        volume.reload()
        subprocess.run(['python','/workspace/scripts/summarize_unrestricted_games.py',str(root)],check=True)
        summary=json.loads((root/'summary.json').read_text())
        lines=['# Repaired Qwen3.5-4B thinking comparison','',
               'Fresh games in both modes. Same weights, sampling, maps, legal actions, and repaired prompt. '
               'No output/thinking token cap. Exact repetition aborts are reported as generation failures, not native losses.','',
               '| Metric | Thinking off | Thinking on |','|---|---:|---:|']
        a,b=(summary['modes'][m] for m in ['off','on'])
        for label,key in [('Native wins','native_wins'),('Native losses','native_losses'),('Invalid-action forfeits','forfeits'),
                          ('Valid actions','valid_actions'),('Completed calls','calls'),('Output tokens','output_tokens'),
                          ('Gameplay seconds','elapsed_seconds')]:
            lines.append(f'| {label} | {a[key]} | {b[key]} |')
        lines.extend(['',f"Off statuses: {a['status_counts']}",f"On statuses: {b['status_counts']}",
                      f"Paired win difference: {summary['paired_win_difference']}",
                      'Development benchmark with one sampling sequence, not a general model ranking.'])
        (root/'result.md').write_text('\n'.join(lines)+'\n')
        resolved=all(g['status'] in ('native','forfeit') for s in summary['modes'].values() for g in s['games'])
        status.update(status='complete' if resolved else 'completed_with_unresolved_games',
                      results=results,finished_unix=time.time())
    except Exception:
        status.update(status='failed',error=traceback.format_exc(),calls=calls,finished_unix=time.time())
        # A failed worker cannot produce a paired comparison. Stop its siblings
        # instead of leaving an orphaned GPU running after coordinator failure.
        for call_id in calls.values():
            try:modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
            except Exception:pass
        raise
    finally:
        (root/'collection.json').write_text(json.dumps(status,indent=2)+'\n');volume.commit()
    return status


@app.local_entrypoint()
def main(run_id: str, mode: str='probe'):
    from modal.config import config
    if config.get('environment')!='staging':raise ValueError('staging required')
    root=ROOT/'runs'/run_id
    receipt=root/(mode+'-dispatch.json')
    if receipt.exists():raise ValueError('already dispatched')
    with volume.batch_upload() as upload:
        upload.put_file(root/'protocol.json',run_id+'/protocol.json')
        if mode=='probe':upload.put_file(root/'probe-inputs.json',run_id+'/probe-inputs.json')
    sha=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
    calls=({'coordinator':coordinate.spawn(run_id,sha)} if mode=='games'
           else {'probe':run.spawn(run_id,'probe',sha)})
    receipt.write_text(json.dumps(dict(protocol_sha256=sha,calls={m:c.object_id for m,c in calls.items()}),indent=2)+'\n')
    print(receipt.read_text(),flush=True)
    if mode=='games':
        # The detached cloud coordinator owns progress and aggregation. Do not
        # retain a local log-streaming/waiting process for hours.
        return
    for m,c in calls.items():print(json.dumps(c.get()),flush=True)
