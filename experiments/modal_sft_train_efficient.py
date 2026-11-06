"""Versioned efficient LoRA runner; frozen SFT engine and old runner preserved."""
import hashlib,json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('uv==0.9.25')
    .add_local_file(ROOT/'training/pyproject.toml','/workspace/training/pyproject.toml',copy=True)
    .add_local_file(ROOT/'training/uv.lock','/workspace/training/uv.lock',copy=True)
    .run_commands('UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple uv sync --project /workspace/training --locked --no-dev')
    .env({'PYTHONPATH':'/workspace/src','TOKENIZERS_PARALLELISM':'false'})
    .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
    .add_local_file(ROOT/'configs/ruleset-v1.json','/workspace/configs/ruleset-v1.json'))
volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging',create_if_missing=True)
app=modal.App('polytopia-sft-efficient-training',image=image)


def _run_training(allocation):
    import subprocess,tarfile,time
    root=Path('/workspace/runs')/allocation['run_id'];root.mkdir(parents=True,exist_ok=False)
    payload={**allocation,'output':str(root/'model')}
    (root/'allocation.json').write_text(json.dumps(payload,indent=2))
    code=r'''
import json,os,sys
from pathlib import Path
from polytopia_bot.learning.efficient_trainer import train_efficient
os.chdir('/workspace')
p=json.loads(Path(sys.argv[1]).read_text())
benchmark=json.loads(Path(p['benchmark']).read_text())
train_efficient(p['prepared'],p['output'],p['config'],seed=p['seed'],stage=p['stage'],readiness=p['readiness'],benchmark=benchmark)
'''
    started=time.monotonic();result={}
    try:
        process=subprocess.run(['/workspace/training/.venv/bin/python','-c',code,str(root/'allocation.json')],
            timeout=allocation['seconds_limit']-(900 if allocation['stage']=='fit' else 300),check=False)
        result['exit_code']=process.returncode
        if process.returncode==0 and allocation['stage']=='fit':
            reload_code=r'''
import json,os,sys
from pathlib import Path
from polytopia_bot.learning.inference import LocalGenerator
from polytopia_bot.learning.training import training_response_accuracy
from polytopia_bot.learning.data import JsonlDataset
os.chdir('/workspace');p=json.loads(Path(sys.argv[1]).read_text())
root=Path(p['output']);receipt=json.loads((root/'run.json').read_text())
generator=LocalGenerator(p['config'],adapter=str(root/'adapter-final'))
result=training_response_accuracy(generator.model,JsonlDataset(Path(p['prepared'])/'tokens.jsonl'),generator.tokenizer,max_new_tokens=p['config']['max_new_tokens'])
before=receipt['fit_after']['results'];after=result['results']
result['responses_identical_after_reload']=len(before)==len(after) and all(a['response_token_ids']==b['response_token_ids'] for a,b in zip(before,after))
(root/'fit-reload.json').write_text(json.dumps(result,indent=2))
if not result['responses_identical_after_reload']:raise ValueError('fit reload changed greedy responses; investigate before smoke')
'''
            reload_process=subprocess.run(['/workspace/training/.venv/bin/python','-c',reload_code,str(root/'allocation.json')],timeout=600,check=False)
            result['reload_exit_code']=reload_process.returncode
    except subprocess.TimeoutExpired:
        result['error']='training occupied-time ceiling reached'
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        receipt=root/'model/run.json'
        result['run']=json.loads(receipt.read_text()) if receipt.exists() else None
        (root/'execution.json').write_text(json.dumps(result,indent=2))
        archive=Path('/workspace/runs')/(allocation['run_id'].replace('/','-')+'.tar')
        # Optimizer/resume states remain durable on the volume; download model
        # adapters, manifests and diagnostics without duplicating optimizer GBs.
        def included(info):
            return None if Path(info.name).name in ('optimizer.pt','scheduler.pt','rng_state.pth') else info
        with tarfile.open(archive,'w') as tar:tar.add(root,arcname='training-run',filter=included)
        volume.commit()
    result['archive']=archive.name
    with archive.open('rb') as stream:result['archive_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    result['occupied_seconds_with_archive']=time.monotonic()-started
    return result


@app.function(gpu='A100-80GB',cpu=4,memory=16384,timeout=29100,max_containers=1,retries=0,
              scaledown_window=2,volumes={'/workspace/runs':volume})
def run_training(allocation):
    return _run_training(allocation)


@app.function(gpu='H100',cpu=4,memory=16384,timeout=29100,max_containers=1,retries=0,
              scaledown_window=2,volumes={'/workspace/runs':volume})
def run_training_h100(allocation):
    return _run_training(allocation)


@app.function(gpu='B200',cpu=4,memory=16384,timeout=29100,max_containers=1,retries=0,
              scaledown_window=2,volumes={'/workspace/runs':volume})
def run_training_b200(allocation):
    return _run_training(allocation)


@app.local_entrypoint()
def main(allocation: str, output: str):
    from modal.config import config as modal_config
    if modal_config.get('environment')!='staging':raise ValueError('Pass --env staging')
    protocol=json.loads((ROOT/allocation).read_text())
    if protocol.get('schema')!='sft-training-allocation-v1' or protocol.get('runtime_version')!='response-only-length-grouped-v1' or protocol['stage'] not in ('fit','smoke','main'):
        raise ValueError('frozen training allocation required')
    if not 600<=protocol['seconds_limit']<=28800:raise ValueError('invalid bounded training allocation')
    if protocol.get('gpu','A100-80GB') not in ('A100-80GB','H100','B200'):raise ValueError('unsupported training GPU')
    for name,sha in protocol.get('gate_receipts',{}).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('training prerequisite changed: '+name)
    for name,sha in protocol['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('source changed after allocation: '+name)
    files=[]
    for name,sha in protocol['uploads'].items():
        path=Path(name)
        if path.is_absolute() or path.parts[0]!='runs' or '..' in path.parts:
            raise ValueError('only relative run artifacts may be uploaded')
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:raise ValueError('upload changed after allocation')
        files.append(path)
    # Reuse immutable evidence already uploaded by another stage; reject drift.
    existing={entry.path.lstrip('/') for entry in volume.iterdir('/',recursive=True)}
    new=[]
    for path in files:
        remote=str(Path(*path.parts[1:]))
        if remote in existing:
            h=hashlib.sha256()
            for chunk in volume.read_file(remote):h.update(chunk)
            if h.hexdigest()!=protocol['uploads'][str(path)]:raise ValueError('existing remote artifact differs: '+remote)
        else:new.append((path,remote))
    if new:
        with volume.batch_upload() as batch:
            for path,remote in new:batch.put_file(ROOT/path,'/'+remote)
    out=ROOT/output;out.mkdir(parents=True,exist_ok=False)
    (out/'allocation.json').write_text(json.dumps(protocol,indent=2))
    runner={'A100-80GB':run_training,'H100':run_training_h100,'B200':run_training_b200}[protocol.get('gpu','A100-80GB')]
    result=runner.with_options(timeout=protocol['seconds_limit']).remote(protocol)
    (out/'execution.json').write_text(json.dumps(result,indent=2))
    archive=out/result['archive']
    with archive.open('xb') as stream:
        for chunk in volume.read_file(result['archive']):stream.write(chunk)
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=result['archive_sha256']:raise ValueError('downloaded archive mismatch')
    import tarfile
    with tarfile.open(archive) as tar:tar.extractall(out,filter='data')
    print(json.dumps({k:v for k,v in result.items() if k!='run'},indent=2))
