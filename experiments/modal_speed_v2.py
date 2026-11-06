"""Bounded speed-v2 benchmark launcher; immutable source image per invocation."""
import hashlib,json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
wheel='https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
image=(modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
 .add_local_file(ROOT/'training/pyproject.toml','/workspace/training/pyproject.toml',copy=True)
 .add_local_file(ROOT/'training/uv.lock','/workspace/training/uv.lock',copy=True)
 .run_commands('UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple uv sync --project /workspace/training --locked --no-dev --extra simulation')
 .run_commands(f'uv pip install --python /workspace/training/.venv/bin/python --no-deps {wheel}',
               'uv pip install --python /workspace/training/.venv/bin/python einops==0.8.1')
 .env({'PYTHONPATH':'/workspace/src','TOKENIZERS_PARALLELISM':'false','HF_HOME':'/workspace/runs/hf-cache','JAX_PLATFORMS':'cpu'})
 .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
 .add_local_dir(ROOT/'scripts','/workspace/scripts',ignore=['**/__pycache__/**'])
 .add_local_dir(ROOT/'configs','/workspace/configs'))
volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
app=modal.App('polytopia-speed-v2',image=image)

@app.function(gpu='B200',cpu=8,memory=32768,timeout=2100,max_containers=1,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def benchmark(protocol):
 import os,subprocess,time
 root=Path('/workspace/runs')/protocol['run_id'];root.mkdir(parents=True,exist_ok=False)
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2));started=time.monotonic()
 try:
  proc=subprocess.run(['/workspace/training/.venv/bin/python','/workspace/scripts/speed_training_benchmark.py',str(root/'protocol.json'),str(root)],timeout=1950,cwd='/workspace')
  result={'exit_code':proc.returncode}
 except subprocess.TimeoutExpired:result={'exit_code':124,'error':'bounded benchmark timeout'}
 finally:volume.commit()
 result['occupied_seconds']=time.monotonic()-started
 result['benchmark']=json.loads((root/'benchmark.json').read_text()) if (root/'benchmark.json').exists() else None
 (root/'execution.json').write_text(json.dumps(result,indent=2));volume.commit();return result

@app.local_entrypoint()
def main(run_id: str='speed-v2/training-benchmark-v1'):
 from modal.config import config
 if config.get('environment')!='staging':raise ValueError('staging required')
 m=json.loads((ROOT/'runs/sft0/prepared-efficient-smoke-v1/manifest.json').read_text())
 protocol=dict(run_id=run_id,config=m['config'],prepared='/workspace/runs/sft0/prepared-efficient-smoke-v1',tokens_sha256=m['tokens_sha256'],
  sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/speed_training_benchmark.py',ROOT/'src/polytopia_bot/learning/fast_training.py']})
 out=ROOT/'runs'/run_id;out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 call=benchmark.spawn(protocol);(out/'call.json').write_text(json.dumps({'call_id':call.object_id}));result=call.get()
 (out/'execution.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
