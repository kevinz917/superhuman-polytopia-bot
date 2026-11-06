"""Bounded, inference-only train/held-out movement diagnosis; no training or RL."""
import hashlib,json
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
 .run_commands('uv pip install --system vllm==0.10.2 transformers==4.57.1 peft==0.17.1 accelerate==1.10.1 jax==0.11.2 requests==2.34.2')
 .env({'PYTHONPATH':'/workspace/src','TOKENIZERS_PARALLELISM':'false','HF_HOME':'/workspace/runs/hf-cache','JAX_PLATFORMS':'cpu','VLLM_USE_V1':'1','VLLM_CACHE_ROOT':'/workspace/runs/vllm-cache','TORCHINDUCTOR_CACHE_DIR':'/workspace/runs/inductor-cache'})
 .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
 .add_local_dir(ROOT/'scripts','/workspace/scripts',ignore=['**/__pycache__/**']))
volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
app=modal.App('polytopia-movement-diagnostic')


@app.function(image=image,gpu='A100-80GB',cpu=8,memory=32768,timeout=1250,
              max_containers=1,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def evaluate_training(run_id,protocol_sha):
    import subprocess,time
    volume.reload();root=Path('/workspace/runs')/run_id
    if hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()!=protocol_sha:raise ValueError('protocol changed')
    if (root/'execution.json').exists() or (root/'training-results').exists():raise ValueError('run already exists')
    started=time.monotonic()
    try:
        proc=subprocess.run(['python','/workspace/scripts/evaluate_movement_training.py',str(root)],cwd='/workspace',timeout=1150)
        result=dict(exit_code=proc.returncode)
    except subprocess.TimeoutExpired:
        result=dict(exit_code=124)
    result['occupied_seconds']=time.monotonic()-started
    if (root/'inference-summary.json').exists():result['summary']=json.loads((root/'inference-summary.json').read_text())
    (root/'execution.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit();return result


@app.local_entrypoint()
def main(run_id: str='movement-diagnostic-v1'):
    from modal.config import config
    if config.get('environment')!='staging':raise ValueError('staging required')
    root=ROOT/'runs'/run_id
    if (root/'call.json').exists():raise ValueError('already dispatched; reconnect to existing call')
    protocol=json.loads((root/'protocol.json').read_text())
    if protocol['run_id']!=run_id:raise ValueError('run ID mismatch')
    with volume.batch_upload() as upload:
        for name in ['protocol.json','training-positions.jsonl.gz']:
            upload.put_file(root/name,run_id+'/'+name)
    (root/'source-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['src','scripts','experiments'] for p in (ROOT/directory).rglob('*.py')},indent=2))
    call=evaluate_training.spawn(run_id,hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest())
    (root/'call.json').write_text(json.dumps(dict(call_id=call.object_id)))
    result=call.get();(root/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['exit_code']==0:
        (root/'training-report.json').write_bytes(b''.join(volume.read_file(run_id+'/training-results/report.json')))
    print(json.dumps(result,indent=2))
