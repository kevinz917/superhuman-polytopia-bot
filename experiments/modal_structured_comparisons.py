"""Cloud-only checkpoint retention and bounded development comparison workers."""
import json
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax==0.11.2','numpy>=2.1')
    .env({'PYTHONPATH':'/workspace/src','JAX_PLATFORMS':'cpu','MODAL_ENVIRONMENT':'staging',
          'POLYTOPIA_CLOUD_EVALUATION':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    .run_commands('mkdir -p /workspace/.venv/bin && ln -s /usr/local/bin/python /workspace/.venv/bin/python')
    .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'scripts','/workspace/scripts',ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'configs','/workspace/configs'))
app=modal.App('polytopia-cloud-comparisons',image=image)
retained=modal.Volume.from_name('polytopia-structured-rl-retained',environment_name='staging')
training=modal.Volume.from_name('polytopia-structured-rl',environment_name='staging')


@app.function(cpu=(6,6),memory=(16384,16384),timeout=43200,max_containers=1,
              retries=0,scaledown_window=2,volumes={'/workspace/runs/structured-ppo-v3':retained,
                  '/training':training.with_mount_options(read_only=True)})
def screens(run_id,run_ids,hours):
    import subprocess,time,hashlib,os
    from polytopia_bot.learning.async_gameplay import atomic_json
    root=Path('/workspace/runs/structured-ppo-v3')
    os.environ['POLYTOPIA_TRAINING_MOUNT']='/training'
    retained.reload()
    (root/'checkpoints').mkdir(exist_ok=True)
    link=root/'retained-checkpoints'
    if not link.exists():link.symlink_to('checkpoints',target_is_directory=True)
    out=root/'cloud-evaluations'/run_id;out.mkdir(parents=True,exist_ok=False)
    commands={
        'retain-and-teacher':['python','scripts/watch_structured_checkpoints.py',*run_ids,'--hours',str(hours),'--watch-name','cloud'],
        'champion12':['python','scripts/watch_structured_matches.py','--run-list',str(root/'learned-screen-runs.json'),
            '--reference',str(root/'qualified-parents/champion51212-06db83c193a7/final.npz'),'--hours',str(hours),'--watch-name','cloud-champion12'],
        'champion96':['python','scripts/watch_structured_matches.py','--run-list',str(root/'learned-screen-champion96-runs.json'),
            '--reference',str(root/'qualified-parents/champion51296-05c414466823/final.npz'),'--hours',str(hours),'--watch-name','cloud-champion96']}
    for ref in (root/'qualified-parents').glob('*/final.npz'):
        meta=json.loads(ref.with_suffix('.json').read_text())
        if hashlib.sha256(ref.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('reference hash mismatch')
    processes={};streams={};started=time.time()
    try:
        for name,command in commands.items():
            streams[name]=(out/(name+'.log')).open('w')
            processes[name]=subprocess.Popen(command,cwd='/workspace',stdout=streams[name],stderr=subprocess.STDOUT,start_new_session=True)
        while any(p.poll() is None for p in processes.values()):
            training.reload()
            report=dict(run_id=run_id,started_unix=started,updated_unix=time.time(),
                execution='Modal CPU only; no local checkpoint downloads or inference',
                processes={n:dict(pid=p.pid,returncode=p.poll()) for n,p in processes.items()},run_ids=run_ids)
            atomic_json(out/'status.json',report);retained.commit();time.sleep(30)
        return report
    finally:
        import signal
        for p in processes.values():
            # Stop bounded child evaluators along with their watcher on expiry.
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for stream in streams.values():stream.close()
        retained.commit()


@app.function(cpu=(1,1),memory=(2048,2048),timeout=1800,retries=0,
              volumes={'/retained':retained})
def verify_import(manifest_path):
    import hashlib,time
    retained.reload();root=Path('/retained');manifest=json.loads((root/manifest_path).read_text())
    results=[]
    for row in manifest['files']:
        path=root/row['remote_path']
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=row['sha256']:raise ValueError('cloud import mismatch: '+row['relative_path'])
        results.append(dict(relative_path=row['relative_path'],sha256=actual,bytes=path.stat().st_size,remote_path=row['remote_path']))
    receipt=dict(verified_unix=time.time(),files=results,volume='polytopia-structured-rl-retained')
    target=root/manifest_path.replace('manifest.json','verified.json')
    target.write_text(json.dumps(receipt,indent=2)+'\n');retained.commit()
    return dict(files=len(results),bytes=sum(r['bytes'] for r in results),receipt=str(target.relative_to(root)))


@app.local_entrypoint()
def main(run_id: str='cloud-screens-20261018',hours: float=11.,action: str='screens'):
    import sys,time
    if action=='verify':
        print(json.dumps(verify_import.remote('migration-20261018/manifest.json')),flush=True);return
    if not 0<hours<=11:raise ValueError('bounded allocation required')
    folder=ROOT/'runs/structured-ppo-v3'
    run_ids=sorted(set(json.loads((folder/'learned-screen-runs.json').read_text())+
                       json.loads((folder/'learned-screen-champion96-runs.json').read_text())))
    with retained.batch_upload() as upload:
        for name in ['budget.json','learned-screen-runs.json','learned-screen-champion96-runs.json']:
            upload.put_file(folder/name,name)
        for name in ['champion51212-06db83c193a7','champion51296-05c414466823']:
            for suffix in ['json','npz']:
                path=Path('qualified-parents')/name/('final.'+suffix);upload.put_file(folder/path,str(path))
    sys.path.insert(0,str(ROOT/'src'))
    from polytopia_bot.rl.budget import reserve,record
    reserve(folder,dict(run_id=run_id,resource_type='cpu-cloud-comparisons',gpu_type=None,
        cpu_cores=6,memory_gib=16,reserved_usd=12,hard_timeout_seconds=43200,
        status='reserved',modal_app_id=app.app_id,start_unix=time.time(),run_ids=run_ids))
    call=screens.spawn(run_id,run_ids,hours)
    record(folder,run_id,status='submitted',function_call_id=call.object_id)
    receipt=dict(run_id=run_id,app_id=app.app_id,function_call_id=call.object_id,run_ids=run_ids)
    out=ROOT/'runs/cloud-comparison-migration-20261018';out.mkdir(exist_ok=True)
    (out/'cloud-dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
