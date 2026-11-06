"""Four bounded, independent shards of the frozen saved-position evaluation."""
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
    .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**']))
volume=modal.Volume.from_name('polytopia-sft0-position-evaluation',environment_name='staging',create_if_missing=True)
training_volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
app=modal.App('polytopia-sft0-coordinated-positions',image=image)


@app.function(gpu='A100-80GB',cpu=4,memory=16384,timeout=3900,max_containers=4,retries=0,
              scaledown_window=2,volumes={'/evaluation':volume,'/workspace/runs':training_volume})
def run_shard(allocation,shard):
    import subprocess,tarfile,time
    root=Path('/evaluation')/allocation['run_id']/f'shard-{shard}';root.mkdir(parents=True,exist_ok=False)
    payload={**allocation,'shard':shard,'output':str(root/'results')}
    (root/'allocation.json').write_text(json.dumps(payload))
    code=r'''
import gzip,hashlib,importlib.metadata,json,os,sys,time
from pathlib import Path
import torch
from polytopia_bot.learning.freeze import verify_prompt
from polytopia_bot.learning.interface import digest
from polytopia_bot.learning.offline import evaluate
from polytopia_bot.learning.serving import BatchGenerator
os.chdir('/workspace');p=json.loads(Path(sys.argv[1]).read_text());started=time.monotonic()
strategy=verify_prompt(p['prompt_manifest'],p['config'])
source=p['shard_files'][p['shard']];path=Path('/evaluation')/source
if hashlib.sha256(path.read_bytes()).hexdigest()!=p['uploads'][source]:raise ValueError('position shard changed')
positions=[json.loads(line) for line in gzip.open(path,'rt')]
if p.get('adapter'):
 for name,sha in p['adapter_files'].items():
  if hashlib.sha256((Path(p['adapter'])/name).read_bytes()).hexdigest()!=sha:raise ValueError('adapter file changed')
generator=BatchGenerator(p['config'],adapter=p.get('adapter'))
model=dict(config=p['config'],adapter=p.get('adapter'),device=torch.cuda.get_device_name(),load_seconds=time.monotonic()-started,tokenizer_hash=digest(generator.tokenizer.backend_tokenizer.to_str()),chat_template_hash=digest(generator.tokenizer.chat_template),packages={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate']})
Path(sys.argv[1]).parent.joinpath('model.json').write_text(json.dumps(model,indent=2))
for key in ('tokenizer_hash','chat_template_hash'):
 if model[key]!=p['prompt_manifest'][key]:raise ValueError('frozen interface mismatch: '+key)
evaluate(p['output'],positions,generator,strategy=strategy,batch_size=8,seconds_limit=max(1,p['seconds_per_shard']-(time.monotonic()-started)))
'''
    started=time.monotonic();result={}
    try:
        p=subprocess.run(['/workspace/training/.venv/bin/python','-c',code,str(root/'allocation.json')],
                         timeout=allocation['seconds_per_shard']+30,check=False)
        result['exit_code']=p.returncode
    except subprocess.TimeoutExpired:result['error']='saved-position shard time ceiling reached'
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        path=root/'results/report.json';result['report']=json.loads(path.read_text()) if path.exists() else None
        (root/'execution.json').write_text(json.dumps(result,indent=2))
        archive=Path('/evaluation')/f"{allocation['run_id']}-shard-{shard}.tar"
        with tarfile.open(archive,'w') as tar:tar.add(root,arcname=f'shard-{shard}')
        volume.commit()
    result['archive']=archive.name
    with archive.open('rb') as stream:result['archive_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    result['occupied_seconds_with_archive']=time.monotonic()-started
    return result


@app.function(cpu=1,memory=1024,timeout=4000,retries=0,volumes={'/evaluation':volume})
def coordinate(allocation):
    import time
    started=time.monotonic()
    bounded=run_shard.with_options(timeout=allocation['seconds_per_shard']+120)
    calls=[bounded.spawn(allocation,i) for i in range(4)]
    path=Path('/evaluation')/(allocation['run_id']+'-coordinator.json')
    record=dict(call_ids=[c.object_id for c in calls],results=[])
    path.write_text(json.dumps(record,indent=2));volume.commit()
    for call in calls:
        record['results'].append(call.get());path.write_text(json.dumps(record,indent=2));volume.commit()
    record['elapsed_seconds']=time.monotonic()-started
    path.write_text(json.dumps(record,indent=2));volume.commit()
    return record


@app.local_entrypoint()
def main(allocation: str,output: str):
    from modal.config import config as modal_config
    if modal_config.get('environment')!='staging':raise ValueError('Pass --env staging')
    p=json.loads((ROOT/allocation).read_text())
    if p.get('schema')!='position-evaluation-allocation-v1' or not 300<=p['seconds_per_shard']<=3600:
        raise ValueError('valid bounded position allocation required')
    for name,sha in p['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('evaluation source changed')
    existing={entry.path.lstrip('/') for entry in volume.iterdir('/',recursive=True)}
    new=[]
    for name,sha in p['uploads'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('position file changed')
        if name in existing:
            h=hashlib.sha256()
            for chunk in volume.read_file(name):h.update(chunk)
            if h.hexdigest()!=sha:raise ValueError('existing remote position file differs')
        else:new.append(name)
    if new:
        with volume.batch_upload() as batch:
            for name in new:batch.put_file(ROOT/name,'/'+name)
    out=ROOT/output;out.mkdir(parents=True,exist_ok=False)
    (out/'allocation.json').write_text(json.dumps(p,indent=2))
    call=coordinate.with_options(timeout=p['seconds_per_shard']+240).spawn(p)
    (out/'coordinator-call.json').write_text(json.dumps({'call_id':call.object_id}))
    coordination=call.get()
    (out/'coordinator.json').write_text(json.dumps(coordination,indent=2));results=[]
    import tarfile
    for result in coordination['results']:
        results.append(result)
        (out/'execution.json').write_text(json.dumps(results,indent=2))
        archive=out/result['archive']
        with archive.open('xb') as stream:
            for chunk in volume.read_file(result['archive']):stream.write(chunk)
        with archive.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=result['archive_sha256']:raise ValueError('downloaded shard archive mismatch')
        with tarfile.open(archive) as tar:tar.extractall(out,filter='data')
    reports=[r['report'] for r in results if r.get('report')];rows=[row for r in reports for row in r['results']]
    if len({r['position_id'] for r in rows})!=len(rows):raise ValueError('overlapping evaluation shards')
    report=dict(schema='saved-position-evaluation-v1',allocated=5000,decisions=len(rows),
        complete=len(reports)==4 and all(r['complete'] for r in reports) and len(rows)==5000,
        results=rows,first_attempt_valid=sum(r['first_attempt_valid'] for r in rows),
        eventually_valid=sum(r['valid'] for r in rows),teacher_agreement=sum(r['teacher_agreement'] for r in rows),
        attempts=sum(r['attempts'] for r in rows),occupied_function_seconds=sum(r['elapsed_seconds'] for r in results),
        occupied_seconds_with_archive=sum(r['occupied_seconds_with_archive'] for r in results))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
