"""Cloud-owned extra epoch with asynchronous fixed-checkpoint diagnostics."""
import hashlib,json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
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
inference_image=(modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
 .run_commands('uv pip install --system vllm==0.10.2 transformers==4.57.1 peft==0.17.1 accelerate==1.10.1 jax==0.11.2 requests==2.34.2')
 .env({'PYTHONPATH':'/workspace/src','TOKENIZERS_PARALLELISM':'false','HF_HOME':'/workspace/runs/hf-cache','JAX_PLATFORMS':'cpu','VLLM_USE_V1':'1','VLLM_CACHE_ROOT':'/workspace/runs/vllm-cache','TORCHINDUCTOR_CACHE_DIR':'/workspace/runs/inductor-cache'})
 .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
 .add_local_dir(ROOT/'scripts','/workspace/scripts',ignore=['**/__pycache__/**']))
volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
app=modal.App('polytopia-movement-continuation')


@app.function(image=inference_image,gpu='A100-80GB',cpu=8,memory=32768,timeout=1700,max_containers=4,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def evaluate_checkpoint(run_id,checkpoint):
    import subprocess,time
    volume.reload();root=Path('/workspace/runs')/run_id
    out=root/f"evaluation-{checkpoint['step']}";out.mkdir(parents=True,exist_ok=False)
    payload=dict(root=str(root),adapter=checkpoint['path'],adapter_files=checkpoint['adapter_files'],output=str(out/'results'))
    (out/'input.json').write_text(json.dumps(payload));started=time.monotonic()
    try:
        proc=subprocess.run(['python','/workspace/scripts/evaluate_movement_checkpoint.py',str(out/'input.json')],cwd='/workspace',timeout=1550)
        result=dict(exit_code=proc.returncode)
    except subprocess.TimeoutExpired:result=dict(exit_code=124)
    result.update(step=checkpoint['step'],occupied_seconds=time.monotonic()-started)
    if (out/'summary.json').exists():result['summary']=json.loads((out/'summary.json').read_text())
    (out/'execution.json').write_text(json.dumps(result,indent=2));volume.commit();return result


@app.function(image=training_image,gpu='B200:4',cpu=16,memory=65536,timeout=5700,max_containers=1,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def train_epoch(protocol):
    import subprocess,time
    volume.reload();payload=protocol['training'];root=Path('/workspace/runs')/payload['run_id'];root.mkdir(parents=True,exist_ok=False)
    payload={**payload,'output':str(root/'model')};(root/'input.json').write_text(json.dumps(payload,indent=2));volume.commit()
    started=time.monotonic();calls={}
    proc=subprocess.Popen(['/workspace/training/.venv/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=4','/workspace/scripts/speed_train_distributed.py',str(root/'input.json')],cwd='/workspace')
    def publish():
        for path in (root/'model').glob('checkpoint-*.ready.json'):
            cp=json.loads(path.read_text());step=cp['step']
            if step in calls:continue
            if step not in protocol['checkpoint_steps']:raise ValueError('unexpected checkpoint')
            volume.commit();call=evaluate_checkpoint.spawn(protocol['run_id'],cp)
            calls[step]=dict(call_id=call.object_id,checkpoint=cp,launched_seconds=time.monotonic()-started)
            (root/'evaluation-calls.json').write_text(json.dumps(calls,indent=2));volume.commit()
            print('CHECKPOINT_EVALUATION_LAUNCHED '+str(step),flush=True)
    try:
        while proc.poll() is None:
            if time.monotonic()-started>payload['seconds_limit']:
                proc.terminate();proc.wait(timeout=30);raise TimeoutError('bounded continuation elapsed')
            publish();time.sleep(1)
        publish()
        result=dict(exit_code=proc.returncode,occupied_seconds=time.monotonic()-started,evaluation_calls=calls)
        if (root/'model/run.json').exists():result['run']=json.loads((root/'model/run.json').read_text())
        (root/'execution.json').write_text(json.dumps(result,indent=2));volume.commit();return result
    finally:
        if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
        volume.commit()


@app.function(image=training_image,cpu=1,memory=4096,timeout=7200,retries=0,volumes={'/workspace/runs':volume})
def coordinate(run_id,protocol_sha):
    import time
    volume.reload();root=Path('/workspace/runs')/run_id
    if hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()!=protocol_sha:raise ValueError('protocol changed')
    p=json.loads((root/'protocol.json').read_text())
    for packed,expanded in p.get('compressed_inputs',{}).items():
        target=root/expanded
        if not target.exists():
            import gzip,shutil
            temporary=target.with_suffix('.unpacking')
            with gzip.open(root/packed,'rb') as source,temporary.open('wb') as dest:
                shutil.copyfileobj(source,dest)
            temporary.replace(target)
            volume.commit()
    if (root/'status.json').exists():
        state=json.loads((root/'status.json').read_text())
        if state['status']=='completed':return state
        if state['status'] not in ('running','evaluating') or not all(k in state for k in ('baseline_call','training_call')):
            raise ValueError('saved coordinator cannot safely resume')
        baseline_call=modal.FunctionCall.from_id(state['baseline_call'])
        training_call=modal.FunctionCall.from_id(state['training_call'])
        state.setdefault('coordinator_resumed_at',[]).append(time.time())
    else:
        state=dict(status='running',started_at=time.time(),evaluations={})
        initial=p['training'].get('scale') or p['training'].get('mixed') or p['training'].get('probe') or p['training']['continuation']
        baseline=dict(step=0,path=initial['adapter'],adapter_files=initial['adapter_files'])
        baseline_call=evaluate_checkpoint.spawn(run_id,baseline);training_call=train_epoch.spawn(p)
        state.update(baseline_call=baseline_call.object_id,training_call=training_call.object_id)
    def save():
        (root/'status.json').write_text(json.dumps(state,indent=2));volume.commit()
    save()
    try:
        while True:
            if '0' not in state['evaluations']:
                try:state['evaluations']['0']=baseline_call.get(timeout=0);save()
                except TimeoutError:pass
            try:state['training']=training_call.get(timeout=0);save();break
            except TimeoutError:time.sleep(5)
        result=state['training']
        if result['exit_code'] or result.get('run',{}).get('status')!='completed':raise ValueError('continuation failed')
        calls={str(k):modal.FunctionCall.from_id(v['call_id']) for k,v in result['evaluation_calls'].items()}
        if set(calls)!=set(str(s) for s in p['checkpoint_steps'] if s):raise ValueError('intermediate checkpoint missing')
        calls={key:call for key,call in calls.items() if key not in state['evaluations']}
        if '0' not in state['evaluations']:calls['0']=baseline_call
        state['status']='evaluating';save()
        while calls:
            for key,call in list(calls.items()):
                try:r=call.get(timeout=0)
                except TimeoutError:continue
                state['evaluations'][key]=r;del calls[key];save()
            if calls:time.sleep(5)
        if any(r['exit_code'] or not r.get('summary',{}).get('complete') or r['summary']['decisions']!=p['evaluation_count'] for r in state['evaluations'].values()):
            raise ValueError('checkpoint evaluation incomplete')
        state['status']='completed'
    except Exception as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:state['finished_at']=time.time();save()
    return state


@app.local_entrypoint()
def main(run_id: str='movement-epoch2-v1',resume: bool=False):
    from modal.config import config
    if config.get('environment')!='staging':raise ValueError('staging required')
    root=ROOT/'runs'/run_id
    if (root/'call.json').exists() and not resume:raise ValueError('already dispatched; use --resume to reconnect')
    if resume and not (root/'call.json').exists():raise ValueError('no saved run to resume')
    protocol=json.loads((root/'protocol.json').read_text())
    if not resume:
        with volume.batch_upload() as upload:
            for name in protocol.get('upload_files',['protocol.json','positions.jsonl.gz','scoring.jsonl.gz']):
                if name!='protocol.json' and name in protocol.get('uploads',{}) and hashlib.sha256((root/name).read_bytes()).hexdigest()!=protocol['uploads'][name]:
                    raise ValueError('upload changed after allocation')
                upload.put_file(root/name,run_id+'/'+name)
    source_file='recovery-source-hashes.json' if resume else 'source-hashes.json'
    (root/source_file).write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','scripts','experiments'] for p in (ROOT/folder).rglob('*.py')},indent=2))
    call=coordinate.spawn(run_id,hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest())
    (root/('recovery-call.json' if resume else 'call.json')).write_text(json.dumps(dict(call_id=call.object_id)))
    state=call.get();(root/'status.json').write_text(json.dumps(state,indent=2))
    for step in protocol['checkpoint_steps']:
        (root/f'report-{step}.json').write_bytes(b''.join(volume.read_file(run_id+f'/evaluation-{step}/results/report.json')))
    print(json.dumps(dict(status=state['status'],workflow_seconds=state['finished_at']-state['started_at'])))
