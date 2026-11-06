"""Cloud-owned speed validation: fit, smoke, early positions, final evaluation."""
import hashlib,json,sys
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
wheel='https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
training_image=(modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
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

inference_image=(modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
 .run_commands('uv pip install --system vllm==0.10.2 transformers==4.57.1 peft==0.17.1 accelerate==1.10.1 jax==0.11.2 requests==2.34.2')
 .env({'PYTHONPATH':'/workspace/src','TOKENIZERS_PARALLELISM':'false','HF_HOME':'/workspace/runs/hf-cache','JAX_PLATFORMS':'cpu','VLLM_USE_V1':'1','VLLM_CACHE_ROOT':'/workspace/runs/vllm-cache','TORCHINDUCTOR_CACHE_DIR':'/workspace/runs/inductor-cache'})
 .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
 .add_local_dir(ROOT/'scripts','/workspace/scripts',ignore=['**/__pycache__/**'])
 .add_local_dir(ROOT/'configs','/workspace/configs'))

evaluation_volume=modal.Volume.from_name('polytopia-sft0-position-evaluation',environment_name='staging')
app=modal.App('polytopia-speed-workflow')

@app.function(image=training_image,gpu='B200:4',cpu=16,memory=65536,timeout=1000,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def distributed_benchmark(payload):
 import subprocess,time
 root=Path('/workspace/runs')/payload['run_id'];root.mkdir(parents=True,exist_ok=False)
 (root/'input.json').write_text(json.dumps(payload));start=time.monotonic()
 proc=subprocess.run(['/workspace/training/.venv/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=4','/workspace/scripts/speed_distributed_benchmark.py',str(root/'input.json'),str(root/'benchmark.json')],cwd='/workspace',timeout=900)
 result=dict(exit_code=proc.returncode,occupied_seconds=time.monotonic()-start)
 if (root/'benchmark.json').exists():result['benchmark']=json.loads((root/'benchmark.json').read_text())
 (root/'execution.json').write_text(json.dumps(result,indent=2));volume.commit();return result

@app.function(image=training_image,cpu=4,memory=16384,timeout=900,volumes={'/workspace/runs':volume})
def prepare_cache(config):
 import subprocess
 code='from huggingface_hub import snapshot_download; snapshot_download('+repr(config['model_id'])+', revision='+repr(config['revision'])+')'
 subprocess.run(['/workspace/training/.venv/bin/python','-c',code],check=True,timeout=800);volume.commit()
 return {'cached':True}

@app.function(image=inference_image,gpu='A100-80GB',cpu=8,memory=32768,timeout=2100,max_containers=8,retries=0,scaledown_window=2,
 volumes={'/workspace/runs':volume,'/evaluation':evaluation_volume})
def positions(payload):
 import subprocess,time
 volume.reload();evaluation_volume.reload()
 root=Path('/workspace/runs')/payload['run_id'];root.mkdir(parents=True,exist_ok=False)
 payload={**payload,'output':str(root/'results')};(root/'input.json').write_text(json.dumps(payload))
 start=time.monotonic()
 try:
  proc=subprocess.run(['python','/workspace/scripts/speed_evaluate.py',str(root/'input.json')],cwd='/workspace',timeout=min(1950,payload.get('occupied_seconds_limit',1950)))
  result={'exit_code':proc.returncode}
 except subprocess.TimeoutExpired:result={'exit_code':124}
 result.update(occupied_seconds=time.monotonic()-start,run_id=payload['run_id'])
 if (root/'summary.json').exists():result['summary']=json.loads((root/'summary.json').read_text())
 (root/'execution.json').write_text(json.dumps(result,indent=2));volume.commit();return result

@app.function(image=training_image,cpu=2,memory=4096,timeout=1200,retries=0,volumes={'/workspace/runs':volume})
def positions_sharded(payload):
 import time
 root=Path('/workspace/runs')/payload['run_id'];root.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();calls=[]
 for i,source in enumerate(payload['shard_files']):
  calls.append(positions.spawn({**payload,'run_id':payload['run_id']+f'/shard-{i}','shard_files':[source],
                               'seconds_limit':600,'occupied_seconds_limit':900}))
 (root/'calls.json').write_text(json.dumps([call.object_id for call in calls]));volume.commit()
 executions=[call.get() for call in calls];volume.reload()
 if any(e['exit_code']!=0 or not e.get('summary',{}).get('complete') for e in executions):raise ValueError('position shard incomplete')
 reports=[json.loads((root/f'shard-{i}/results/report.json').read_text()) for i in range(len(calls))]
 rows=[row for report in reports for row in report['results']]
 if len(rows)!=5000 or len({r['position_id'] for r in rows})!=5000:raise ValueError('position shard coverage differs')
 report=dict(schema='saved-position-evaluation-v1',allocated=5000,decisions=5000,complete=True,incomplete=[],results=rows,
  first_attempt_valid=sum(r['first_attempt_valid'] for r in rows),eventually_valid=sum(r['valid'] for r in rows),
  teacher_agreement=sum(r['teacher_agreement'] for r in rows),attempts=sum(r['attempts'] for r in rows),elapsed_seconds=time.monotonic()-started)
 (root/'results').mkdir();(root/'results/report.json').write_text(json.dumps(report,indent=2))
 summary={k:v for k,v in report.items() if k!='results'}
 summary.update(agreement=report['teacher_agreement']/5000,validity=report['first_attempt_valid']/5000)
 summary['by_action_family']={}
 for execution in executions:
  for family,counts in execution['summary'].get('by_action_family',{}).items():
   target=summary['by_action_family'].setdefault(family,{key:0 for key in counts})
   for key,value in counts.items():target[key]+=value
 result=dict(exit_code=0,run_id=payload['run_id'],occupied_seconds=time.monotonic()-started,
             gpu_seconds=sum(e['occupied_seconds'] for e in executions),summary=summary,shards=executions)
 (root/'execution.json').write_text(json.dumps(result,indent=2));volume.commit();return result

@app.function(image=training_image,cpu=1,memory=2048,timeout=1300,retries=0,volumes={'/workspace/runs':volume})
def paired_position_benchmark(protocol):
 import time
 root=Path('/workspace/runs')/protocol['run_id'];root.mkdir(parents=True,exist_ok=False)
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2));started=time.monotonic()
 calls={kind:positions_sharded.spawn({**payload,'run_id':protocol['run_id']+'/'+kind}) for kind,payload in protocol['models'].items()}
 (root/'calls.json').write_text(json.dumps({k:c.object_id for k,c in calls.items()}));volume.commit()
 result=dict(models={kind:call.get() for kind,call in calls.items()},occupied_seconds=time.monotonic()-started)
 volume.reload()
 references=[Path('/workspace/runs')/protocol['reference_run']/(kind+'-positions')/'execution.json' for kind in result['models']]
 while not all(path.exists() for path in references):
  if time.monotonic()-started>1100:raise TimeoutError('reference position evaluation not ready')
  time.sleep(10);volume.reload()
 from polytopia_bot.learning.smoke_gate import position_comparison
 comparisons={}
 for kind,execution in result['models'].items():
  reference=Path('/workspace/runs')/protocol['reference_run']/(kind+'-positions')
  old=json.loads((reference/'results/report.json').read_text());new=json.loads((root/kind/'results/report.json').read_text())
  checked=position_comparison(old,new)
  before=old['first_attempt_valid']/5000;after=new['first_attempt_valid']/5000
  old_time=json.loads((reference/'execution.json').read_text())['occupied_seconds']
  comparisons[kind]=dict(agreement_before=checked['base_agreement'],agreement_after=checked['smoke_agreement'],
    agreement_delta=checked['agreement_gain'],validity_before=before,validity_after=after,
    wall_speedup=old_time/execution['occupied_seconds'],same_inputs=True)
 result['comparisons']=comparisons
 limits=protocol['acceptance']
 result['passed']=all(c['agreement_delta']>=-limits['max_agreement_drop'] and
   c['validity_after']>=c['validity_before']-limits['max_validity_drop'] and
   c['wall_speedup']>=limits['min_wall_speedup'] for c in comparisons.values()) and comparisons['trained']['validity_after']>=limits['min_trained_first_attempt_valid']
 (root/'execution.json').write_text(json.dumps(result,indent=2));volume.commit();return result

@app.function(image=training_image,gpu='B200',cpu=8,memory=32768,timeout=5100,max_containers=1,retries=0,scaledown_window=2,
 volumes={'/workspace/runs':volume})
def train(payload,early=None):
 import subprocess,time
 volume.reload()
 root=Path('/workspace/runs')/payload['run_id'];root.mkdir(parents=True,exist_ok=False)
 payload={**payload,'output':str(root/'model')};(root/'input.json').write_text(json.dumps(payload,indent=2))
 start=time.monotonic();calls=[];published=set()
 command=['/workspace/training/.venv/bin/python']
 if payload.get('world_size',1)>1:command+=['-m','torch.distributed.run','--standalone','--nproc_per_node='+str(payload['world_size']),'/workspace/scripts/speed_train_distributed.py']
 else:command+=['/workspace/scripts/speed_train.py']
 proc=subprocess.Popen(command+[str(root/'input.json')],cwd='/workspace')
 try:
  while proc.poll() is None:
   if time.monotonic()-start>payload['seconds_limit']:
    proc.terminate();proc.wait(timeout=30);raise TimeoutError('bounded training time')
   if early:
    for ready in sorted((root/'model').glob('checkpoint-*.ready.json')):
     cp=json.loads(ready.read_text())
     if cp['step']!=16 or cp['step'] in published:continue
     volume.commit()
     call=positions.spawn({**early,'run_id':payload['run_id']+'-early16','count':512,'adapter':cp['path'],'adapter_files':cp['adapter_files']})
     calls.append(dict(call_id=call.object_id,step=cp['step'],launched_seconds=time.monotonic()-start));published.add(cp['step'])
     game_call=early_games.spawn({**early,'run_id':payload['run_id']+'-early16-games','adapter':cp['path'],'adapter_files':cp['adapter_files']})
     calls.append(dict(call_id=game_call.object_id,step=cp['step'],kind='paired-gameplay',launched_seconds=time.monotonic()-start))
     (root/'early-calls.json').write_text(json.dumps(calls,indent=2));volume.commit()
   for early_call in calls:
    if 'result' in early_call:continue
    try:early_result=modal.FunctionCall.from_id(early_call['call_id']).get(timeout=0)
    except TimeoutError:continue
    early_call.update(result=early_result,completed_seconds=time.monotonic()-start)
    (root/'early-results.json').write_text(json.dumps(calls,indent=2));volume.commit()
    print('EARLY_DIAGNOSTIC '+json.dumps(early_call),flush=True)
   time.sleep(1)
  result=dict(exit_code=proc.returncode,occupied_seconds=time.monotonic()-start,early_calls=calls)
  path=root/'model/run.json'
  if path.exists():result['run']=json.loads(path.read_text())
  (root/'execution.json').write_text(json.dumps(result,indent=2));volume.commit();return result
 finally:
  if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
  volume.commit()

@app.function(image=inference_image,gpu='A100-80GB',cpu=16,memory=49152,timeout=5100,max_containers=4,retries=0,scaledown_window=2,
 volumes={'/workspace/runs':volume})
def games(payload,job_queue,index):
 import queue,time
 volume.reload()
 from polytopia_bot.learning.fast_serving import ContinuousGenerator
 from polytopia_bot.learning.fast_gameplay import play_continuous
 from polytopia_bot.learning.interface import digest
 root=Path('/workspace/runs')/payload['run_id']/f'worker-{index}';root.mkdir(parents=True,exist_ok=False)
 rules=json.loads(Path('/workspace/configs/ruleset-v1.json').read_text())
 for name,sha in rules['source_sha256'].items():
  if hashlib.sha256((Path('/workspace')/name).read_bytes()).hexdigest()!=sha:raise ValueError('frozen simulator changed')
 start=time.monotonic();generator=ContinuousGenerator(payload['config'],adapter=payload.get('adapter'))
 for key,val in [('tokenizer_hash',digest(generator.tokenizer.backend_tokenizer.to_str())),('chat_template_hash',digest(generator.tokenizer.chat_template))]:
  if val!=payload['prompt_manifest'][key]:raise ValueError('game tokenizer mismatch')
 if payload.get('adapter_files'):
  for name,sha in payload['adapter_files'].items():
   if hashlib.sha256((Path(payload['adapter'])/name).read_bytes()).hexdigest()!=sha:raise ValueError('game adapter mismatch')
 load=time.monotonic()-start
 def take_job():
  try:return job_queue.get(block=False)
  except queue.Empty:return None
 last_commit=time.monotonic()
 def checkpoint(row):
  nonlocal last_commit
  if time.monotonic()-last_commit>30:volume.commit();last_commit=time.monotonic()
 report=play_continuous(root/'games',take_job,generator,lanes=8,seconds_limit=payload.get('game_seconds_limit',4500),action_limit=payload.get('action_limit',10000),on_complete=checkpoint,heartbeat=volume.commit)
 result=dict(load_seconds=load,occupied_seconds=time.monotonic()-start,report=report)
 (root/'execution.json').write_text(json.dumps(result,indent=2));volume.commit();return result

@app.function(image=training_image,cpu=1,memory=2048,timeout=1800,retries=0,volumes={'/workspace/runs':volume})
def early_games(payload):
 import gzip,time
 from polytopia_bot.learning.progress_diagnostics import public_progress
 root=Path('/workspace/runs')/payload['run_id'];root.mkdir(parents=True,exist_ok=False)
 with modal.Queue.ephemeral() as base_jobs,modal.Queue.ephemeral() as trained_jobs:
  base_jobs.put_many(payload['diagnostic_jobs']);trained_jobs.put_many(payload['diagnostic_jobs'])
  common={**payload,'action_limit':200,'game_seconds_limit':900}
  calls={kind:games.spawn({**common,'run_id':payload['run_id']+'/'+kind,'adapter':adapter,'adapter_files':files},jobs,0)
    for kind,adapter,files,jobs in [('base',None,None,base_jobs),('trained',payload['adapter'],payload['adapter_files'],trained_jobs)]}
  result=dict(schema='early-paired-gameplay-v2',action_limit=200,started_at=time.time())
  for kind,call in calls.items():
   result[kind]=call.get();(root/'summary.json').write_text(json.dumps(result,indent=2));volume.commit()
  volume.reload();result['public_progress']={}
  for kind in calls:
   result['public_progress'][kind]=[]
   for job in payload['diagnostic_jobs']:
    path=root/kind/'worker-0/games'/(job['id']+'.jsonl.gz')
    with gzip.open(path,'rt') as stream:metrics=public_progress((json.loads(line) for line in stream),job['seat'])
    result['public_progress'][kind].append(dict(id=job['id'],**metrics))
  result['finished_at']=time.time();(root/'summary.json').write_text(json.dumps(result,indent=2));volume.commit();return result

@app.function(image=training_image,cpu=16,memory=32768,timeout=1800,retries=0,volumes={'/workspace/runs':volume})
def audit(run_id):
 import subprocess
 volume.reload()
 root=Path('/workspace/runs')/run_id
 p=subprocess.run(['/workspace/training/.venv/bin/python','/workspace/scripts/speed_audit_remote.py',str(root)],cwd='/workspace',timeout=1700)
 if p.returncode!=0:raise ValueError('remote replay or integrity audit failed')
 volume.commit();return json.loads((root/'audit.json').read_text())

@app.function(image=training_image,cpu=2,memory=4096,timeout=18000,retries=0,volumes={'/workspace/runs':volume})
def coordinate(protocol):
 import time
 root=Path('/workspace/runs')/protocol['run_id'];root.mkdir(parents=True,exist_ok=False)
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
 state=dict(status='preparing',stages=[],started_at=time.time())
 def phase(name):
  state['status']=name;(root/'status.json').write_text(json.dumps(state,indent=2));volume.commit()
 def record(name,result):
  state['stages'].append(dict(name=name,result=result,finished_at=time.time()));(root/'status.json').write_text(json.dumps(state,indent=2));volume.commit()
 try:
  record('cache',prepare_cache.remote(protocol['smoke']['config']))
  runner=train.with_options(gpu='B200:4' if protocol['world_size']==4 else 'B200',cpu=16 if protocol['world_size']==4 else 8,memory=65536 if protocol['world_size']==4 else 32768)
  phase('fit');fit=runner.remote(protocol['fit']);record('fit',fit)
  if fit['exit_code']!=0 or fit.get('run',{}).get('status')!='completed':raise ValueError('fresh fit gate failed')
  phase('training');smoke=runner.remote(protocol['smoke'],protocol['positions']);record('smoke',smoke)
  if smoke['exit_code']!=0 or smoke.get('run',{}).get('status')!='completed':raise ValueError('smoke training failed')
  adapter='/workspace/runs/'+protocol['smoke']['run_id']+'/model/adapter-final';hashes=smoke['run']['adapter_files']
  phase('evaluating')
  calls={}
  for kind,path,files in [('base',None,None),('trained',adapter,hashes)]:
   calls[kind]=positions_sharded.spawn({**protocol['positions'],'run_id':protocol['run_id']+'/'+kind+'-positions','adapter':path,'adapter_files':files})
  calls['previous']=positions_sharded.spawn({**protocol['positions'],'run_id':protocol['run_id']+'/previous-positions'})
  state['position_calls']={key:call.object_id for key,call in calls.items()};phase('evaluating')
  # Position reporting is independent of game completion. Poll call.get(timeout=0)
  # from this cloud coordinator while game workers execute.
  game_results={}
  for kind,path,files in [('base',None,None),('trained',adapter,hashes)]:
   with modal.Queue.ephemeral() as jobs:
    jobs.put_many(protocol['jobs'])
    payload={**protocol['positions'],'run_id':protocol['run_id']+'/'+kind+'-games','adapter':path,'adapter_files':files}
    game_calls=[games.spawn(payload,jobs,i) for i in range(4)]
    state.setdefault('game_calls',{})[kind]=[call.object_id for call in game_calls];phase('evaluating')
    pending=list(game_calls);results=[]
    while pending:
     for key,call in list(calls.items()):
      try:r=call.get(timeout=0)
      except TimeoutError:continue
      record(key+'-positions',r);del calls[key]
     for call in list(pending):
      try:r=call.get(timeout=0)
      except TimeoutError:continue
      results.append(r);pending.remove(call)
     if pending:time.sleep(5)
    game_results[kind]=results;record(kind+'-games',results)
  audit_call=audit.spawn(protocol['run_id'])
  state['audit_call']=audit_call.object_id;phase('evaluating')
  for key,call in calls.items():record(key+'-positions',call.get())
  for early in smoke.get('early_calls',[]):record('early-'+early.get('kind','position'),modal.FunctionCall.from_id(early['call_id']).get())
  for stage in state['stages']:
   if stage['name'].endswith('-positions'):
    r=stage['result']
    if r['exit_code']!=0 or not r.get('summary',{}).get('complete') or r['summary']['decisions']!=5000:
     raise ValueError('full position evaluation incomplete: '+stage['name'])
  phase('auditing');record('audit',audit_call.get())
  state['status']='completed'
 except Exception as exc:
  state.update(status='failed',error=repr(exc));raise
 finally:
  state['finished_at']=time.time();(root/'status.json').write_text(json.dumps(state,indent=2));volume.commit()
 return state

@app.local_entrypoint()
def main(run_id: str='speed-v2/end-to-end-v1',cache_only: bool=False,benchmark_distributed: bool=False,benchmark_positions: bool=False):
 from modal.config import config
 if config.get('environment')!='staging':raise ValueError('staging required')
 if cache_only:
  print(prepare_cache.remote(json.loads((ROOT/'configs/sft.json').read_text())));return
 if benchmark_positions:
  previous=ROOT/'runs/speed-v2/end-to-end-v1'
  full=json.loads((previous/'protocol.json').read_text());smoke=json.loads((previous/'smoke-execution.json').read_text())
  adapter='/workspace/runs/'+full['smoke']['run_id']+'/model/adapter-final'
  models={kind:{**full['positions'],'adapter':path,'adapter_files':files} for kind,path,files in
    [('base',None,None),('trained',adapter,smoke['run']['adapter_files'])]}
  protocol=dict(run_id=run_id,models=models,reference_run=full['run_id'],
                acceptance=dict(max_agreement_drop=.01,max_validity_drop=.005,min_wall_speedup=2.,min_trained_first_attempt_valid=.95,complete_positions_per_model=5000))
  out=ROOT/'runs'/run_id;out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
  call=paired_position_benchmark.spawn(protocol);(out/'call.json').write_text(json.dumps({'call_id':call.object_id}));result=call.get()
  (out/'execution.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
 b=json.loads((ROOT/'runs/speed-v2/training-benchmark-v4/execution.json').read_text())
 if benchmark_distributed:
  m=json.loads((ROOT/'runs/sft0/prepared-efficient-smoke-v1/manifest.json').read_text())
  payload=dict(run_id='speed-v2/distributed-benchmark-v1',config=m['config'],prepared='/workspace/runs/sft0/prepared-efficient-smoke-v1',tokens_sha256=m['tokens_sha256'],benchmark=b['benchmark'])
  out=ROOT/'runs'/payload['run_id'];out.mkdir(parents=True,exist_ok=False)
  call=distributed_benchmark.spawn(payload);(out/'call.json').write_text(json.dumps({'call_id':call.object_id}));r=call.get();(out/'execution.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));return
 if b['exit_code']!=0 or not b['benchmark']['parity']['passed'] or b['benchmark']['speedup']<=1:raise ValueError('passing faster benchmark required')
 infer=json.loads((ROOT/'runs/speed-v2/inference-benchmark-v1/execution.json').read_text())
 if any(r['exit_code']!=0 for r in infer['variants']):raise ValueError('inference benchmark incomplete')
 if next(r for r in infer['variants'] if r['mode']=='vllm')['speedup']<=1:raise ValueError('vLLM not faster')
 scaling=json.loads((ROOT/'runs/speed-v2/distributed-benchmark-v1/execution.json').read_text())
 if scaling['exit_code']!=0 or not scaling['benchmark']['parity']['passed']:raise ValueError('distributed benchmark must pass')
 sharding=json.loads((ROOT/'runs/speed-v2/sharded-positions-v1/execution.json').read_text())
 if not sharding['passed']:raise ValueError('position sharding benchmark must pass')
 world=4 if scaling['benchmark']['wall_speedup_vs_current']>b['benchmark']['speedup'] else 1
 allocation=json.loads((ROOT/'runs/sft0/efficient-smoke-allocation-v1.json').read_text())
 p=dict(run_id=run_id,benchmark=b,scaling=scaling,position_sharding=sharding,world_size=world,positions=json.loads((ROOT/'runs/sft0/efficient-positions-allocation-v1.json').read_text()))
 p['positions'].update(seconds_limit=1800)
 for stage,prepared in [('fit','prepared-efficient-fit-v1'),('smoke','prepared-efficient-smoke-v1')]:
  m=json.loads((ROOT/'runs/sft0'/prepared/'manifest.json').read_text())
  p[stage]=dict(run_id=run_id+'/'+stage,prepared='/workspace/runs/sft0/'+prepared,config=m['config'],stage=stage,
    readiness=allocation['readiness'],benchmark=b['benchmark'],world_size=world,seconds_limit=(900 if stage=='fit' else 1800) if world==4 else (1800 if stage=='fit' else 4800))
 p['jobs']=json.loads((ROOT/'runs/sft0/efficient-games-allocation-v1.json').read_text())['jobs']
 groups={}
 for job in p['jobs']:groups.setdefault((job['opponent'],job['seat']),[]).append(job)
 p['positions']['diagnostic_jobs']=[job for _,group in sorted(groups.items()) for job in sorted(group,key=lambda j:j['id'])[:2]]
 p['sources']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['src','scripts','experiments','configs'] for f in (ROOT/folder).rglob('*') if f.is_file() and f.suffix in ('.py','.json')}
 out=ROOT/'runs'/run_id;out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(json.dumps(p,indent=2))
 call=coordinate.spawn(p);(out/'call.json').write_text(json.dumps({'call_id':call.object_id}));r=call.get()
 (out/'status.json').write_text(json.dumps(r,indent=2));print(json.dumps({'status':r['status']}))
