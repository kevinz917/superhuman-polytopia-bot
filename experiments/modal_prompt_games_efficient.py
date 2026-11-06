"""Versioned pipelined gameplay; original frozen runner remains available."""
import hashlib
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('uv==0.9.25')
    .add_local_file(ROOT/'training/pyproject.toml','/workspace/training/pyproject.toml',copy=True)
    .add_local_file(ROOT/'training/uv.lock','/workspace/training/uv.lock',copy=True)
    .run_commands('UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple uv sync --project /workspace/training --locked --no-dev --extra simulation')
    .env({'PYTHONPATH':'/workspace/src','TOKENIZERS_PARALLELISM':'false','JAX_PLATFORMS':'cpu'})
    .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
    .add_local_file(ROOT/'configs/ruleset-v1.json','/workspace/ruleset.json'))
volume = modal.Volume.from_name('polytopia-sft0-gameplay',environment_name='staging',create_if_missing=True)
training_volume = modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
app = modal.App('polytopia-sft0-pipelined-games',image=image)


@app.function(gpu='A100-80GB',cpu=4,memory=24576,timeout=29100,max_containers=4,retries=0,
              scaledown_window=2,volumes={'/results':volume,'/workspace/runs':training_volume})
def evaluate(run_id, jobs, config, seconds_limit,adapter=None,adapter_files=None,model_interface=None):
    import subprocess,tarfile,time
    if not 60<=seconds_limit<=28800 or Path(run_id).name!=run_id:
        raise ValueError('invalid bounded allocation')
    started=time.monotonic()
    root=Path('/results')/run_id
    root.mkdir(exist_ok=False)
    payload=dict(jobs=jobs,config=config,seconds_limit=seconds_limit,output=str(root/'games'),adapter=adapter,adapter_files=adapter_files,model_interface=model_interface)
    (root/'input.json').write_text(json.dumps(payload))
    code=r'''
import hashlib,importlib.metadata,json,sys,time
from pathlib import Path
import torch
from polytopia_bot.learning.serving import BatchGenerator
from polytopia_bot.learning.efficient_gameplay import play_pipelined
from polytopia_bot.learning.interface import digest
from polytopia_bot.learning.splits import board_hashes
from polytopia_bot.gpu.mapgen import layout
import jax
import jax.numpy as jnp
payload=json.loads(Path(sys.argv[1]).read_text())
rules=json.loads(Path('/workspace/ruleset.json').read_text())
for name,sha in rules['source_sha256'].items():
 if hashlib.sha256((Path('/workspace')/name).read_bytes()).hexdigest()!=sha:raise ValueError('frozen simulator mismatch: '+name)
generate=jax.jit(lambda seed: layout(seed,size=14))
for job in payload['jobs']:
 if board_hashes(jax.device_get(generate(jnp.int32(job['seed']))),14)[0]!=job['board_hash']:raise ValueError('reserved board mismatch')
if payload['adapter']:
 for name,sha in payload['adapter_files'].items():
  if hashlib.sha256((Path(payload['adapter'])/name).read_bytes()).hexdigest()!=sha:raise ValueError('adapter file changed')
start=time.monotonic();generator=BatchGenerator(payload['config'],adapter=payload['adapter'])
receipt=dict(config=payload['config'],adapter=payload['adapter'],adapter_files=payload['adapter_files'],load_seconds=time.monotonic()-start,device=torch.cuda.get_device_name(),tokenizer_hash=digest(generator.tokenizer.backend_tokenizer.to_str()),chat_template_hash=digest(generator.tokenizer.chat_template),packages={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','accelerate','jax','jaxlib']})
Path(sys.argv[1]).parent.joinpath('model.json').write_text(json.dumps(receipt,indent=2))
for key,expected in (payload['model_interface'] or {}).items():
 if receipt[key]!=expected:raise ValueError('frozen model interface mismatch: '+key)
result=play_pipelined(payload['output'],[payload['jobs'][::2],payload['jobs'][1::2]],generator,seconds_limit=max(1,payload['seconds_limit']-(time.monotonic()-start)),batch_size=8)
Path(payload['output']).joinpath('report.json').write_text(json.dumps(result,indent=2))
'''
    result={}
    try:
        process=subprocess.run(['/workspace/training/.venv/bin/python','-c',code,str(root/'input.json')],
                               timeout=seconds_limit+30,check=False)
        result['exit_code']=process.returncode
    except subprocess.TimeoutExpired:
        result['error']='occupied-time ceiling interrupted subprocess'
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        report=root/'games/report.json'
        result['report']=json.loads(report.read_text()) if report.exists() else None
        (root/'execution.json').write_text(json.dumps(result,indent=2))
        archive=Path('/results')/f'{run_id}.tar'
        with tarfile.open(archive,'w') as tar:
            tar.add(root,arcname=run_id,filter=lambda info: None if '/games/shard-' in info.name else info)
            for lane in sorted((root/'games').glob('shard-*')):
                tar.add(lane,arcname=run_id+'-lane'+lane.name.split('-')[-1])
        volume.commit()
    result['archive']=archive.name
    result['archive_sha256']=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    result['occupied_seconds_with_archive']=time.monotonic()-started
    return result


@app.function(cpu=1,memory=1024,timeout=4000,retries=0,volumes={'/results':volume})
def coordinate(allocation):
    import time
    started=time.monotonic();shards=allocation['shards']
    bounded=evaluate.with_options(timeout=allocation['seconds_limit']+120)
    calls=[bounded.spawn(f"{allocation['run_id']}-shard{i}",allocation['jobs'][i::shards],
        allocation['config'],allocation['seconds_limit'],allocation.get('adapter'),allocation.get('adapter_files'),
        {k:allocation[k] for k in ('tokenizer_hash','chat_template_hash')}) for i in range(shards)]
    path=Path('/results')/(allocation['run_id']+'-coordinator.json')
    record=dict(call_ids=[c.object_id for c in calls],results=[])
    path.write_text(json.dumps(record,indent=2));volume.commit()
    for call in calls:
        record['results'].append(call.get());path.write_text(json.dumps(record,indent=2));volume.commit()
    record['elapsed_seconds']=time.monotonic()-started;path.write_text(json.dumps(record,indent=2));volume.commit()
    return record


@app.local_entrypoint()
def main(protocol: str, output: str):
    from modal.config import config as modal_config
    if modal_config.get('environment')!='staging':raise ValueError('Pass --env staging')
    allocation=json.loads((ROOT/protocol).read_text())
    if allocation.get('schema')!='sft-gameplay-allocation-v1' or allocation.get('runtime_version')!='pipelined-cpu-lanes-v1':raise ValueError('versioned pipeline allocation required')
    for receipt in allocation['prerequisites']:
        path=ROOT/receipt['path']
        if path.parent.joinpath('invalidation.json').exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['sha256']:
            raise ValueError('changed or invalidated prerequisite')
    for relative,sha in allocation['sources'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=sha:raise ValueError('source changed after allocation')
    out=ROOT/output
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(allocation,indent=2))
    shards=allocation.get('shards',1)
    if not 1<=shards<=4:raise ValueError('at most four independent inference containers')
    call=coordinate.with_options(timeout=allocation['seconds_limit']+240).spawn(allocation)
    (out/'coordinator-call.json').write_text(json.dumps({'call_id':call.object_id}))
    coordination=call.get()
    (out/'coordinator.json').write_text(json.dumps(coordination,indent=2))
    results=[]
    import tarfile
    for result in coordination['results']:
        results.append(result)
        (out/'execution.json').write_text(json.dumps(results,indent=2))
        archive=out/result['archive']
        with archive.open('xb') as stream:
            for chunk in volume.read_file(result['archive']):stream.write(chunk)
        if hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()!=result['archive_sha256']:
            raise ValueError('downloaded trace archive hash mismatch')
        with tarfile.open(archive) as tar:tar.extractall(out,filter='data')
        print(json.dumps({k:v for k,v in result.items() if k!='report'},indent=2))
    combined=out/'combined';(combined/'games').mkdir(parents=True)
    reports=[r['report'] for r in results if r.get('report')]
    merged=dict(schema='gameplay-v1',allocated_games=len(allocation['jobs']),
        finished_games=sum(r['finished_games'] for r in reports),
        games=[g for r in reports for g in r['games']],
        unstarted_jobs=[j for r in reports for j in r['unstarted_jobs']],
        complete=len(reports)==shards and all(r['complete'] for r in reports),
        elapsed_seconds=sum(r['elapsed_seconds'] for r in results),
        occupied_seconds_with_archive=sum(r['occupied_seconds_with_archive'] for r in results),
        shard_archives=[r['archive'] for r in results])
    (combined/'games/report.json').write_text(json.dumps(merged,indent=2))
    models=[json.loads((out/f"{allocation['run_id']}-shard{i}"/'model.json').read_text()) for i in range(shards)
            if (out/f"{allocation['run_id']}-shard{i}"/'model.json').exists()]
    if len(models)==shards:
        for m in models:
            if any(m[k]!=models[0][k] for k in ('config','adapter','adapter_files','tokenizer_hash','chat_template_hash')):
                raise ValueError('model interface differs between shards')
        for key in ('tokenizer_hash','chat_template_hash'):
            if key in allocation and models[0][key]!=allocation[key]:
                raise ValueError('frozen model interface mismatch: '+key)
        (combined/'model.json').write_text(json.dumps({**models[0],'shard_models':models},indent=2))
