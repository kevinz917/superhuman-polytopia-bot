"""Two bounded inference-only gameplay arms; never loads a Polytopia adapter."""
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
app = modal.App('polytopia-qwen35-thinking-games')


@app.function(image=image,gpu='H100',cpu=8,memory=49152,timeout=2400,
              max_containers=2,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def evaluate(run_id,mode,protocol_sha):
    import subprocess,time,tarfile
    volume.reload()
    root = Path('/workspace/runs')/run_id
    if hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()!=protocol_sha:
        raise ValueError('protocol changed')
    p = json.loads((root/'protocol.json').read_text())
    out = root/mode
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    proc = subprocess.Popen(['python','/workspace/scripts/benchmark_thinking_games.py',str(root),mode],cwd='/workspace')
    try:
        while proc.poll() is None:
            if time.monotonic()-started > p['occupied_seconds_limit']:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                break
            volume.commit()
            time.sleep(10)
        result = dict(mode=mode,exit_code=proc.returncode,occupied_seconds=time.monotonic()-started)
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


@app.local_entrypoint()
def main(run_id: str='qwen35-thinking-games-v1'):
    from modal.config import config
    if config.get('environment')!='staging':raise ValueError('staging required')
    root = ROOT/'runs'/run_id
    if (root/'calls.json').exists():raise ValueError('already dispatched; do not duplicate')
    with volume.batch_upload() as upload:
        upload.put_file(root/'protocol.json',run_id+'/protocol.json')
    sha = hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
    calls = {}
    for mode in ['off','on']:
        calls[mode] = evaluate.spawn(run_id,mode,sha).object_id
        (root/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
    for mode,call_id in calls.items():
        result = modal.FunctionCall.from_id(call_id).get()
        print(json.dumps(result),flush=True)
        target = root/(mode+'.tar')
        with target.open('wb') as f:
            for chunk in volume.read_file(run_id+'/'+mode+'.tar'):f.write(chunk)
        import tarfile
        with tarfile.open(target) as archive:archive.extractall(root,filter='data')
