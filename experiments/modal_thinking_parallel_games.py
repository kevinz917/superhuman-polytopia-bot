"""Paired games across independent vLLM replicas; all execution on Modal."""
import hashlib
import json
from pathlib import Path
import modal
from experiments.modal_thinking_repaired import image, volume, ROOT

app=modal.App('polytopia-thinking-parallel-comparison')


@app.function(image=image.add_local_dir(ROOT/'tests','/workspace/tests',ignore=['**/__pycache__/**']),
              cpu=1,memory=4096,timeout=300,volumes={'/workspace/runs':volume})
def verify():
    import subprocess,os,sys,runpy
    result=subprocess.run(['python','/workspace/tests/test_thinking_shards.py'],
        env=dict(os.environ,PYTHONPATH='/workspace/scripts:/workspace/src'),
        text=True,capture_output=True)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    sys.path[:0]=['/workspace/src','/workspace/tests']
    tests=runpy.run_path('/workspace/tests/test_thinking_debug.py')
    count=0
    for name,fn in tests.items():
        if name.startswith('test_'):
            fn();count+=1
    from tokenizers import Tokenizer
    from polytopia_bot.learning.thinking_debug import repeated_suffix
    volume.reload()
    cache=Path('/workspace/runs/hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer.json')
    tokenizer=Tokenizer.from_file(str(cache))
    source=Path('/workspace/runs/qwen35-thinking-independent-v1')
    evidence=[]
    for i in [0,1,3]:
        row=json.loads((source/f'batch-40/precise_recipe-{i}.json').read_text())
        tokens=tokenizer.encode(row['text'],add_special_tokens=False).ids
        old=repeated_suffix(tokens,max_period=16);new=repeated_suffix(tokens)
        evidence.append(dict(request=i,retokenized_tokens=len(tokens),old_period=old,new_period=new))
        if old or not new:raise ValueError(f'prose-loop reproduction failed: {evidence}')
    controls=0
    for profile in ['baseline','focused']:
        for i in range(4):
            row=json.loads((source/f'batch-40/{profile}-{i}.json').read_text())
            tokens=tokenizer.encode(row['text'],add_special_tokens=False).ids
            if any(repeated_suffix(tokens[:end]) for end in range(1024,len(tokens)+1,64)):
                raise ValueError(f'completed control would be aborted: {profile}-{i}')
            controls+=1
    (source/'loop-reproduction.json').write_text(json.dumps(evidence,indent=2)+'\n');volume.commit()
    return result.stdout+result.stderr+f'\n{count} prompt/repetition checks and {controls} completed-stream controls passed on Modal.\n'+json.dumps(evidence)


@app.function(image=image,gpu='H100',cpu=8,memory=49152,timeout=14400,
              max_containers=4,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def worker(run_id,mode,shard,sha):
    import subprocess,time
    volume.reload();root=Path('/workspace/runs')/run_id/'shards'/str(shard)
    if hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()!=sha:
        raise ValueError('shard protocol changed')
    started=time.monotonic()
    proc=subprocess.Popen(['python','/workspace/scripts/benchmark_thinking_repaired.py',str(root),mode],cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit();time.sleep(10)
        result=dict(mode=mode,shard=shard,exit_code=proc.returncode,
                    worker_seconds=time.monotonic()-started,finished_unix=time.time())
        (root/(mode+'-execution.json')).write_text(json.dumps(result)+'\n');volume.commit()
        if proc.returncode:
            raise RuntimeError(result)
        return result
    finally:
        if proc.poll() is None:
            proc.terminate();proc.wait(timeout=15)
        volume.commit()


@app.function(image=image,cpu=.5,memory=4096,timeout=15000,
              max_containers=1,retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def coordinate(run_id,sha):
    import sys,time,subprocess,traceback
    sys.path.insert(0,'/workspace/scripts')
    from merge_thinking_shards import merge
    volume.reload();root=Path('/workspace/runs')/run_id
    if hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()!=sha:
        raise ValueError('parent protocol changed')
    p=json.loads((root/'protocol.json').read_text())
    maps=sorted({j['seed'] for j in p['jobs']})
    pairing=p['pairing']
    if (p['replicas_per_mode']!=2 or len(maps)<2
        or len(p['jobs'])!=pairing['games_per_mode'] or len(maps)!=pairing['maps']
        or len(p['jobs'])!=2*len(maps)
        or len({j['id'] for j in p['jobs']})!=len(p['jobs'])
        or any(sorted(j['seat'] for j in p['jobs'] if j['seed']==seed)!=[1,2] for seed in maps)):
        raise ValueError('expected two replicas and exact paired map/seat coverage matching the protocol')
    calls={};results={};status=dict(status='dispatching',started_unix=time.time())
    if (root/'collection.json').exists():
        status=json.loads((root/'collection.json').read_text())
        status['collector_resumed_unix']=time.time()
    try:
        if (root/'calls.json').exists():
            restored=json.loads((root/'calls.json').read_text())
            expected={f'{mode}-{shard}' for mode in ['off','on'] for shard in range(p['replicas_per_mode'])}
            if restored.keys()!=expected or len(set(restored.values()))!=len(expected):
                raise ValueError('invalid persisted worker calls')
            for shard in range(p['replicas_per_mode']):
                child=json.loads((root/'shards'/str(shard)/'protocol.json').read_text())
                selected=set(maps[shard::p['replicas_per_mode']])
                if child['parent_protocol_sha256']!=sha or child['jobs']!=[j for j in p['jobs'] if j['seed'] in selected]:
                    raise ValueError('persisted shard protocol differs')
            calls=restored
        else:
            shas={}
            for shard in range(p['replicas_per_mode']):
                selected=set(maps[shard::p['replicas_per_mode']])
                child=dict(p,jobs=[j for j in p['jobs'] if j['seed'] in selected],shard=shard,parent_protocol_sha256=sha)
                folder=root/'shards'/str(shard);folder.mkdir(parents=True,exist_ok=False)
                data=(json.dumps(child,indent=2)+'\n').encode();(folder/'protocol.json').write_bytes(data)
                shas[shard]=hashlib.sha256(data).hexdigest()
            volume.commit()
            for mode in ['off','on']:
                for shard in range(p['replicas_per_mode']):
                    calls[f'{mode}-{shard}']=worker.spawn(run_id,mode,shard,shas[shard]).object_id
            (root/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
            volume.commit()
        while len(results)<len(calls):
            for name,call_id in calls.items():
                if name in results:
                    continue
                try:
                    results[name]=modal.FunctionCall.from_id(call_id).get(timeout=0)
                except TimeoutError:
                    pass
            volume.reload();merge(root)
            status.update(status='waiting',calls=calls,completed_workers=results,observed_unix=time.time())
            (root/'collection.json').write_text(json.dumps(status,indent=2)+'\n');volume.commit()
            if len(results)<len(calls):time.sleep(30)
        volume.reload();merge(root,final=True)
        subprocess.run(['python','/workspace/scripts/summarize_unrestricted_games.py',str(root)],check=True)
        summary=json.loads((root/'summary.json').read_text())
        resolved=all(g['status'] in ('native','forfeit') for s in summary['modes'].values() for g in s['games'])
        status.update(status='complete' if resolved else 'completed_with_unresolved_games',finished_unix=time.time())
        lines=['# Paired Qwen3.5-4B comparison','',
            f"{len(p['jobs'])} games per mode, same {len(maps)} validation maps and both seats. Two independent H100/vLLM replicas per mode.",'',
            '| Metric | Thinking off | Thinking on |','|---|---:|---:|']
        for label,key in [('Native wins','native_wins'),('Native losses','native_losses'),('Invalid-action forfeits','forfeits'),('Output tokens','output_tokens'),('Median request seconds','median_request_seconds')]:
            lines.append(f"| {label} | {summary['modes']['off'][key]} | {summary['modes']['on'][key]} |")
        lines.extend(['',f"Paired map-level uncertainty: {summary['paired_win_difference']}",
            'Unfinished/context/infrastructure failures are not native losses or draws. This is a development comparison, not untouched-test generalization.'])
        (root/'result.md').write_text('\n'.join(lines)+'\n')
    except Exception:
        status.update(status='failed',error=traceback.format_exc(),calls=calls,finished_unix=time.time())
        for call in calls.values():
            try:modal.FunctionCall.from_id(call).cancel(terminate_containers=True)
            except Exception:pass
        raise
    finally:
        (root/'collection.json').write_text(json.dumps(status,indent=2)+'\n');volume.commit()
    return status


@app.local_entrypoint()
def main(run_id: str='', verify_only: bool=False, resume_only: bool=False):
    if verify_only:
        print(verify.remote());return
    if not run_id:raise ValueError('run_id required')
    root=ROOT/'runs'/run_id
    if resume_only:
        # Recovery reuses persisted call IDs and never uploads or replaces protocols.
        existing=json.loads(b''.join(volume.read_file(run_id+'/calls.json')))
        if not existing:raise ValueError('no existing calls to resume')
        data=b''.join(volume.read_file(run_id+'/protocol.json'))
        sha=hashlib.sha256(data).hexdigest()
        if sha!=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest():
            raise ValueError('remote protocol differs')
        call=coordinate.spawn(run_id,sha)
        receipt=dict(app_id=app.app_id,call_id=call.object_id,protocol_sha256=sha,
            existing_workers=existing,recovery_launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (root/f'recovery-{call.object_id}.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt));return
    if (root/'dispatch.json').exists():raise ValueError('already dispatched')
    with volume.batch_upload() as upload:
        upload.put_file(root/'protocol.json',run_id+'/protocol.json')
    sha=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
    call=coordinate.spawn(run_id,sha)
    receipt=dict(app_id=app.app_id,call_id=call.object_id,protocol_sha256=sha)
    (root/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
