"""One bounded cloud GPU for diagnosing Qwen serving and prompt efficiency."""
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
app=modal.App('polytopia-thinking-serving-diagnostic')


@app.function(image=image,gpu='H100',cpu=8,memory=49152,timeout=3600,
              retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def run(run_id):
    import subprocess,time
    root=Path('/workspace/runs')/run_id
    volume.reload()
    protocol=json.loads((root/'protocol.json').read_text())
    for phase in protocol.get('phases',['speed-eager','speed-graphs','prompts-graphs']):
        with (root/(phase+'.log')).open('w') as log:
            proc=subprocess.Popen(['python','/workspace/scripts/diagnose_thinking_serving.py',str(root),phase],
                cwd='/workspace',stdout=log,stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:volume.commit();time.sleep(10)
                if proc.returncode:raise RuntimeError(f'{phase}: {proc.returncode}')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait(timeout=15)
                volume.commit()
    return dict(status='complete')


@app.local_entrypoint()
def main(run_id: str):
    root=ROOT/'runs'/run_id
    if (root/'dispatch.json').exists():raise ValueError('already dispatched')
    with volume.batch_upload() as upload:
        for name in ['protocol.json','inputs.json']:upload.put_file(root/name,run_id+'/'+name)
        if (root/'off-inputs.json').exists():upload.put_file(root/'off-inputs.json',run_id+'/off-inputs.json')
    call=run.spawn(run_id)
    (root/'dispatch.json').write_text(json.dumps(dict(app_id=app.app_id,call_id=call.object_id),indent=2)+'\n')
    print(call.get())
