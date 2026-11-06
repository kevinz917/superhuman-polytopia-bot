"""Compare fixed real training-game prefixes with serial and pipelined CPU work."""
import hashlib,json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
 .add_local_file(ROOT/'training/pyproject.toml','/workspace/training/pyproject.toml',copy=True)
 .add_local_file(ROOT/'training/uv.lock','/workspace/training/uv.lock',copy=True)
 .run_commands('UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple uv sync --project /workspace/training --locked --no-dev --extra simulation')
 .env({'PYTHONPATH':'/workspace/src','TOKENIZERS_PARALLELISM':'false','JAX_PLATFORMS':'cpu'})
 .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**']))
volume=modal.Volume.from_name('polytopia-sft0-gameplay',environment_name='staging')
training=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
app=modal.App('polytopia-rollout-efficiency',image=image)

@app.function(gpu='A100-80GB',cpu=4,memory=24576,timeout=1500,max_containers=1,retries=0,scaledown_window=2,volumes={'/results':volume,'/workspace/runs':training})
def benchmark(protocol):
 import subprocess,time,os
 started=time.monotonic();root=Path('/results')/protocol['run_id'];root.mkdir(exist_ok=False)
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
 code=r'''
import hashlib,json,os,time
from pathlib import Path
from polytopia_bot.learning.serving import BatchGenerator
from polytopia_bot.learning.rollout_profile import profile_prefixes
root=Path(os.environ['ROLLOUT_PROFILE_ROOT']);p=json.loads((root/'protocol.json').read_text())
for name,sha in p['adapter_files'].items():
 if hashlib.sha256((Path(p['adapter'])/name).read_bytes()).hexdigest()!=sha:raise ValueError('adapter changed')
g=BatchGenerator(p['config'],adapter=p['adapter']);results={}
for mode in ['serial','pipeline']:
 results[mode]=profile_prefixes(root/mode,p['lane_jobs'],g,mode=mode,batches=16,seconds_limit=600)
 print(json.dumps(dict(mode=mode,seconds=results[mode]['elapsed_seconds'],requests=results[mode]['requests'],steps=results[mode]['steps'])),flush=True)
a={(r['lane'],r['request']):(r['messages_hash'],r['generation_hash']) for r in results['serial']['batches']};b={(r['lane'],r['request']):(r['messages_hash'],r['generation_hash']) for r in results['pipeline']['batches']}
r=dict(schema='rollout-efficiency-comparison-v1',exact_inputs_and_generations_match=a==b,steps_match=results['serial']['steps']==results['pipeline']['steps'],speedup=results['serial']['elapsed_seconds']/results['pipeline']['elapsed_seconds'],results=results)
(root/'comparison.json').write_text(json.dumps(r,indent=2))
if not r['exact_inputs_and_generations_match'] or not r['steps_match']:raise ValueError('profile workload changed')
'''
 try:
  p=subprocess.run(['/workspace/training/.venv/bin/python','-c',code],env={**os.environ,'ROLLOUT_PROFILE_ROOT':str(root)},timeout=1380,check=False);result={'exit_code':p.returncode}
 except subprocess.TimeoutExpired:result={'error':'bounded profile timeout'}
 result['elapsed_seconds']=time.monotonic()-started;result['comparison']=json.loads((root/'comparison.json').read_text()) if (root/'comparison.json').exists() else None
 (root/'execution.json').write_text(json.dumps(result,indent=2));volume.commit();result['occupied_seconds_with_commit']=time.monotonic()-started;return result

@app.local_entrypoint()
def main():
 from modal.config import config as modal_config
 if modal_config.get('environment')!='staging':raise ValueError('staging required')
 old=json.loads((ROOT/'runs/sft0/relabel-games-v1/protocol.json').read_text());new=json.loads((ROOT/'runs/sft0/repaired-games-allocation-v2.json').read_text())
 # Exactly sixteen registered training-map jobs, fixed before either timing.
 jobs=old['jobs'][:16]
 train_maps={r['board_hash'] for r in json.loads((ROOT/'runs/sft0/splits-v1/maps.json').read_text())['maps']['train']}
 if any(j['board_hash'] not in train_maps for j in jobs):raise ValueError('training-only workload required')
 p=dict(run_id='sft0-rollout-efficiency-v1',lane_jobs=[jobs[::2],jobs[1::2]],config=new['config'],adapter=new['adapter'],adapter_files=new['adapter_files'],training_only=True,seconds_limit=1500,compute_planning_allowance_usd=1.20,sources={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),ROOT/'src/polytopia_bot/learning/efficient_gameplay.py',ROOT/'src/polytopia_bot/learning/rollout_profile.py']})
 out=ROOT/'runs/sft0/efficiency-rollout-benchmark-v1';out.mkdir(exist_ok=False);(out/'protocol.json').write_text(json.dumps(p,indent=2));call=benchmark.spawn(p);(out/'call.json').write_text(json.dumps({'call_id':call.object_id}));r=call.get();(out/'execution.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='comparison'},indent=2))
