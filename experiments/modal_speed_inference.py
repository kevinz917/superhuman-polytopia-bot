"""Bounded vLLM inference benchmark on the same A100 hardware as prior games."""
import hashlib,json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
 .run_commands('uv pip install --system vllm==0.10.2 transformers==4.57.1 peft==0.17.1 accelerate==1.10.1 jax==0.11.2 requests==2.34.2')
 .env({'PYTHONPATH':'/workspace/src','TOKENIZERS_PARALLELISM':'false','HF_HOME':'/workspace/runs/hf-cache','JAX_PLATFORMS':'cpu','VLLM_USE_V1':'1','VLLM_CACHE_ROOT':'/workspace/runs/vllm-cache','TORCHINDUCTOR_CACHE_DIR':'/workspace/runs/inductor-cache'})
 .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
 .add_local_dir(ROOT/'scripts','/workspace/scripts',ignore=['**/__pycache__/**'])
 .add_local_dir(ROOT/'configs','/workspace/configs'))
volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
app=modal.App('polytopia-speed-inference',image=image)

@app.function(gpu='A100-80GB',cpu=8,memory=32768,timeout=700,max_containers=1,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def startup(protocol):
 import os,subprocess,time
 volume.reload();root=Path('/workspace/runs')/protocol['run_id'];root.mkdir(parents=True,exist_ok=False)
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.monotonic()
 env=dict(os.environ)
 if protocol.get('local_cache'):env['HF_HOME']='/tmp/polytopia-hf-cache'
 proc=subprocess.run(['python','/workspace/scripts/speed_inference_benchmark.py',str(root/'protocol.json'),str(root/'vllm.json'),'vllm'],timeout=600,cwd='/workspace',env=env)
 result=dict(exit_code=proc.returncode,occupied_seconds=time.monotonic()-start)
 if proc.returncode==0:
  r=json.loads((root/'vllm.json').read_text());old=json.loads(Path('/workspace/runs/speed-v2/inference-benchmark-v1/vllm.json').read_text())
  result.update({k:v for k,v in r.items() if k!='generations'})
  result['identical_prompts']=sum(a['prompt_token_ids']==b['prompt_token_ids'] for a,b in zip(old['generations'],r['generations']))
  result['identical_responses']=sum(a['response_token_ids']==b['response_token_ids'] for a,b in zip(old['generations'],r['generations']))
  result['passed']=result['identical_prompts']==result['identical_responses']==128
 (root/'execution.json').write_text(json.dumps(result,indent=2));volume.commit();return result

@app.function(gpu='A100-80GB',cpu=8,memory=32768,timeout=2100,max_containers=1,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def benchmark(protocol):
 import subprocess,time
 root=Path('/workspace/runs')/protocol['run_id'];root.mkdir(parents=True,exist_ok=False)
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.monotonic();result={'variants':[]}
 for mode in ['hf','hf_merged','vllm']:
  proc=subprocess.run(['python','/workspace/scripts/speed_inference_benchmark.py',str(root/'protocol.json'),str(root/(mode+'.json')),mode],timeout=600,cwd='/workspace')
  row={'mode':mode,'exit_code':proc.returncode}
  if (root/(mode+'.json')).exists():row.update(json.loads((root/(mode+'.json')).read_text()))
  result['variants'].append(row);volume.commit()
 result['occupied_seconds']=time.monotonic()-start
 if all(r['exit_code']==0 for r in result['variants']):
  ref=result['variants'][0]
  for r in result['variants']:
   r['speedup']=ref['seconds']/r['seconds']
   r['exact_output_matches']=sum(a['response_token_ids']==b['response_token_ids'] for a,b in zip(ref['generations'],r['generations']))
 (root/'execution.json').write_text(json.dumps(result,indent=2));volume.commit()
 return {**result,'variants':[{k:v for k,v in r.items() if k!='generations'} for r in result['variants']]}

@app.local_entrypoint()
def main(run_id: str='speed-v2/inference-benchmark-v1',startup_only: bool=False,local_cache: bool=False,load_strategy: str=''):
 from modal.config import config
 if config.get('environment')!='staging':raise ValueError('staging required')
 m=json.loads((ROOT/'runs/sft0/prepared-efficient-smoke-v1/manifest.json').read_text())
 p=dict(run_id=run_id,config=m['config'],prepared='/workspace/runs/sft0/prepared-efficient-smoke-v1',
   examples_sha256=hashlib.sha256((ROOT/'runs/sft0/prepared-efficient-smoke-v1/examples.jsonl').read_bytes()).hexdigest(),
   adapter='/workspace/runs/sft0/efficient-smoke-v1/model/adapter-final')
 if startup_only:p['load_strategy']='eager'
 if local_cache:
  if not startup_only:raise ValueError('local cache test requires startup-only')
  p.update(load_strategy='lazy',local_cache=True)
 if load_strategy:
  if not startup_only or load_strategy not in ('lazy','eager'):raise ValueError('startup load strategy must be lazy/eager')
  p['load_strategy']=load_strategy
 out=ROOT/'runs'/run_id;out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(json.dumps(p,indent=2))
 call=(startup if startup_only else benchmark).spawn(p);(out/'call.json').write_text(json.dumps({'call_id':call.object_id}));r=call.get()
 (out/'execution.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
