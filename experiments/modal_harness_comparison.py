"""Bounded inference-only comparison on common A100 hardware and runtime."""
import hashlib
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
    .run_commands('uv pip install --system vllm==0.19.0 transformers==4.57.6 peft==0.17.1 accelerate==1.10.1 jax==0.11.2')
    .env({'PYTHONPATH':'/workspace/src', 'HF_HOME':'/workspace/runs/hf-cache',
          'TOKENIZERS_PARALLELISM':'false', 'JAX_PLATFORMS':'cpu', 'VLLM_USE_V1':'1'})
    .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'scripts','/workspace/scripts',ignore=['**/__pycache__/**']))
volume = modal.Volume.from_name('polytopia-sft0-training', environment_name='staging')
app = modal.App('polytopia-harness-comparison')


@app.function(image=image, gpu='A100-80GB', cpu=8, memory=49152, timeout=2800,
              max_containers=3, retries=0, scaledown_window=2, volumes={'/workspace/runs':volume})
def evaluate_model(run_id, key, protocol_sha):
    import subprocess
    import time
    volume.reload()
    root = Path('/workspace/runs')/run_id
    if hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest() != protocol_sha:
        raise ValueError('protocol changed')
    out = root/key
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        proc = subprocess.run(['python','/workspace/scripts/evaluate_decision_harness.py',str(root),key],
                              cwd='/workspace', timeout=2680)
        result = dict(exit_code=proc.returncode)
    except subprocess.TimeoutExpired:
        result = dict(exit_code=124)
    result.update(model=key, occupied_seconds=time.monotonic()-started)
    (out/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    volume.commit()
    return result


@app.local_entrypoint()
def main(run_id: str='decision-harness-v1'):
    from modal.config import config
    if config.get('environment') != 'staging':
        raise ValueError('staging required')
    root = ROOT/'runs'/run_id
    if (root/'calls.json').exists():
        raise ValueError('already dispatched; reconnect instead')
    p = json.loads((root/'protocol.json').read_text())
    with volume.batch_upload() as upload:
        for name in ['protocol.json','positions.jsonl.gz','source-hashes.json']:
            upload.put_file(root/name,run_id+'/'+name)
    sha = hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
    calls = {}
    for key in p['models']:
        call = evaluate_model.spawn(run_id,key,sha)
        calls[key] = call.object_id
        (root/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
    results = {}
    for key, call_id in calls.items():
        results[key] = modal.FunctionCall.from_id(call_id).get()
        print(json.dumps(results[key]),flush=True)
        for name in ['execution.json','runtime.json','results/report.json','results/predictions.jsonl.gz']:
            path = root/key/name
            path.parent.mkdir(parents=True,exist_ok=True)
            try:
                path.write_bytes(b''.join(volume.read_file(run_id+'/'+key+'/'+name)))
            except FileNotFoundError:
                pass
    (root/'execution.json').write_text(json.dumps(results,indent=2)+'\n')
