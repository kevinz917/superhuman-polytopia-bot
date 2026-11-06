"""Single-entrypoint Mini-SWE authoring, source verification and held-out tournament."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),'/workspace']
import modal
from experiments import modal_skirmish_v5 as base
from scripts import finalize_skirmish_cohort as grading
from scripts import rebuild_skirmish_sources as rebuilding
from scripts import retry_skirmish_authoring as recovery

app=base.app
status=base.status
results=base.results
MOUNTS=base.MOUNTS
operator_image=grading.operator_image


def require_unfinished_cohort(root,state):
    """A completed report cannot be made resumable by replacing its status file."""
    if state.get('status') in ('completed','provisional') or (root/'report.json').exists():
        raise ValueError('Cohort is already evaluated; allocate a new prefix instead of resuming it')


@app.local_entrypoint()
def inspect_results(prefix: str,output: str='',watch: bool=False):
    from collections import Counter
    from scripts.render_skirmish_v5 import compact_snapshot,render
    while True:
        data=compact_snapshot(base.inspect_cohort.remote(prefix))
        if output:Path(output).write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps({**{key:data.get(key) for key in ('status','authoring_finished','total','games_finished','evaluation_games','errors')},
            'stage_counts':dict(Counter(s['status'] for s in data.get('current_stages',{}).values()))}),flush=True)
        if data['status'] in ('completed','provisional') and output:
            Path(output).with_suffix('.md').write_text(render(data))
        if not watch or data['status'] in ('completed','provisional','needs_attention'):return
        time.sleep(30)


@app.function(nonpreemptible=True,image=operator_image,cpu=2,memory=6144,timeout=14400,retries=0,max_containers=20,volumes=MOUNTS)
def pipeline_author(run_id: str,config: dict):
    from polytopia_bot.skirmish.mini import run_mini_trial
    results.reload()  # A replay must see an earlier committed attempt, never overwrite it.
    sandbox_app=modal.App.lookup('polytopia-skirmish-sandboxes',environment_name='staging',create_if_missing=True)
    return run_mini_trial(run_id,config,modal.Image.from_id(config['_coding_image_id']),sandbox_app,results)


@app.function(image=operator_image,cpu=1,memory=1024,timeout=240,retries=0,max_containers=6,volumes=MOUNTS)
def pipeline_rebuild(prefix: str,run_id: str,stage: int):
    return rebuilding.rebuild.local(prefix,run_id,stage)


def recover_completion(root,cell):
    """Recover a lost controller return only after every non-error stage is frozen."""
    from polytopia_bot.skirmish.common import write_json
    path=root/'stages.json'
    if (root/'report.json').exists() or not path.exists():return False
    stages=json.loads(path.read_text());expected=4 if cell['track']=='feedback-3r-40m' else 1
    if len(stages)!=expected or any(s['end_reason'] not in ('deadline','agent_exit') for s in stages):return False
    protocol=json.loads((root/'protocol.json').read_text())
    final=root/f'stage-{expected-1}'
    report=dict(status='completed' if stages[-1]['returncode']==0 else 'failed_submission',
        run_id=root.name,model=cell['model'],track=cell['track'],stages=stages,
        submission_quality=stages[-1]['submission_quality'],artifact_root=str(root),
        final_source_sha256=stages[-1]['source_sha256'],final_wasm_sha256=stages[-1].get('wasm_sha256'),
        protocol_sha256=grading.file_sha(root/'protocol.json'),
        trajectory_sha256={p.name:grading.file_sha(p) for p in root.glob('stage-*.trajectory.json')},
        controller_return_recovered=True,recovery='complete frozen stage receipts, validated before evaluation; no new model calls')
    (root/'main.rs').write_bytes((final/'main.rs').read_bytes())
    if stages[-1]['returncode']==0:(root/'bot.wasm').write_bytes((final/'bot.wasm').read_bytes())
    write_json(root/'report.json',report);return True


@app.function(nonpreemptible=True,image=operator_image,cpu=1,memory=4096,timeout=86400,retries=0,max_containers=1,volumes=MOUNTS)
def pipeline(prefix: str,manifest: dict,image_id: str,smoke: bool=False,resume: bool=False,max_infrastructure_retries: int=1):
    from polytopia_bot.skirmish.common import canonical,sha,write_json,identifier
    from polytopia_bot.skirmish.agent import source_receipt
    from polytopia_bot.skirmish.cohort import authoring_cells
    if not 0<=max_infrastructure_retries<=3:raise ValueError('Infrastructure retry count must be explicit and bounded')
    identifier(prefix);results.reload();root=Path('/results/mini-v5')/prefix;root.mkdir(parents=True,exist_ok=True)
    sources=source_receipt();host_sources={name:sha(Path(module.__file__).read_bytes()) for name,module in [('evaluator',grading),('rebuilder',rebuilding),('recovery',recovery)]}
    host_sources['pipeline']=sha(Path(__file__).read_bytes())
    cells=authoring_cells(prefix,manifest,image_id,smoke=smoke)
    allocation=dict(manifest=manifest,image_id=image_id,smoke=smoke,execution_source=sources,host_sources=host_sources,max_infrastructure_retries=max_infrastructure_retries)
    digest=sha(canonical(allocation).encode());path=root/'authoring-plan.json'
    if path.exists():
        if not resume:raise ValueError('Cohort already allocated; inspect and explicitly resume it')
        old_state=json.loads((root/'status.json').read_text())
        require_unfinished_cohort(root,old_state)
        plan=json.loads(path.read_text());assert plan['sha256']==digest,'immutable pipeline/manifest/image changed'
        old_id=old_state.get('coordinator_call_id')
        if old_id and old_id!=modal.current_function_call_id():
            old=modal.FunctionCall.from_id(old_id)
            if any(s=='PENDING' for s in recovery.controller_states(old,old_id)):raise ValueError('Original coordinator still running')
    else:
        plan=dict(**allocation,sha256=digest,cells=cells,references=base.reference_receipts(manifest['references']),created_unix=time.time(),pipeline_schema='polytopia-mini-pipeline-v1')
        write_json(path,plan);(root/'frozen-pipeline.py').write_bytes(Path(__file__).read_bytes());results.commit()
    state=dict(prefix=prefix,status='probing',smoke=smoke,total=len(cells),authoring_finished=0,runs={},errors=[],coordinator_call_id=modal.current_function_call_id())
    def save():write_json(root/'status.json',state);results.commit()
    save();secret=modal.Secret.from_dict({'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})
    checks=list(base.probe.with_options(secrets=[secret]).starmap(((prefix,m,image_id) for m in manifest['models'])))
    write_json(root/'probes.json',checks);results.commit()
    if any(not p['passed'] for p in checks):state.update(status='needs_attention',errors=[p for p in checks if not p['passed']]);save();return state
    bound=pipeline_author.with_options(secrets=[secret]);current={};exhausted={}
    def allocate(cell,index):
        run_id=cell['run_id']+(f'-infra{index}' if index else '')
        path=root/'calls'/f'{run_id}.json'
        if path.exists():call=modal.FunctionCall.from_id(json.loads(path.read_text())['call_id'])
        else:
            call=bound.spawn(run_id,cell['config']);write_json(path,dict(call_id=call.object_id,run_id=run_id,attempt=index));results.commit()
        current[cell['run_id']]=dict(call=call,run_id=run_id,index=index,observed_terminal=False)
    for cell in cells:
        allocated=[0]+[int(p.stem.rsplit('infra',1)[1]) for p in (root/'calls').glob(cell['run_id']+'-infra*.json')]
        allocate(cell,max(allocated))
    state['status']='authoring';save()
    while True:
        results.reload()
        for cell in cells:
            item=current[cell['run_id']]
            if recover_completion(Path('/results/mini-runs')/item['run_id'],cell):results.commit()
        readiness=grading.finalize.local(prefix,True)
        if readiness['status'] in ('ready','needs_rebuild'):break
        if readiness['status']=='needs_attention':state.update(status='needs_attention',errors=[readiness]);save();return state
        missing=set(readiness['missing']);state['authoring_finished']=readiness['eligible']
        for cell in cells:
            name=cell['run_id'];item=current[name]
            if name not in missing:
                exhausted.pop(name,None);state['runs'][name]=dict(status='artifact_ready',effective_run_id=item['run_id']);continue
            if name in exhausted:continue
            failure=None
            try:
                report=item['call'].get(timeout=0)
                if report['status'] in ('completed','failed_submission'):failure='returned artifact failed validation'
                else:failure=report.get('reason',report.get('error',report['status']))
            except TimeoutError:
                states=recovery.controller_states(item['call'],item['call'].object_id)
                if states and all(s!='PENDING' for s in states):failure='terminal controller returned no output'
            except Exception as error:failure=type(error).__name__
            if failure is None:continue
            record=dict(run_id=item['run_id'],call_id=item['call'].object_id,reason=failure,observed_unix=time.time())
            write_json(root/'infrastructure-events'/f'{item["run_id"]}.json',record);results.commit()
            if item['index']>=max_infrastructure_retries:
                exhausted[name]=record;state['runs'][name]=dict(status='infrastructure_exhausted',error=failure)
            else:allocate(cell,item['index']+1)
        state['errors']=list(exhausted.values());save()
        if missing and missing.issubset(exhausted):state['status']='needs_attention';save();return state
        time.sleep(30)
    # All slots are frozen. No additional authoring or hidden feedback is supplied.
    state.update(status='source_verification',authoring_finished=len(cells));save()
    jobs=rebuilding.pending.local(prefix,True)
    for proof in pipeline_rebuild.starmap(jobs):
        if not proof['identical']:state.update(status='needs_attention',errors=[dict(kind='source_rebuild_mismatch',proof=proof)]);save();return state
    ready=grading.finalize.local(prefix,True)
    if ready['status']!='ready':state.update(status='needs_attention',errors=[ready]);save();return state
    def close_orphans(nodes):
        for node in nodes:
            if node.function_name=='pipeline_author' and node.status.name=='PENDING':modal.FunctionCall.from_id(node.function_call_id).cancel(terminate_containers=True)
            close_orphans(node.children)
    close_orphans(modal.FunctionCall.from_id(modal.current_function_call_id()).get_call_graph())
    return grading.finalize.local(prefix,False)


@app.local_entrypoint()
def launch(prefix: str,manifest: str='configs/skirmish/mini-v5-top11.json',smoke: bool=False,resume: bool=False,max_infrastructure_retries: int=1):
    from modal.config import config
    if config.get('environment')!='staging':raise ValueError('Pass --env staging')
    if not os.environ.get('OPENROUTER_API_KEY'):raise ValueError('Authorized OpenRouter key required')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from polytopia_bot.skirmish.cohort import validate_manifest
    data=validate_manifest(json.loads(Path(manifest).read_text()));image_id=base.mini_coding_image.build(app).object_id
    call=pipeline.with_options(secrets=[modal.Secret.from_dict({'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})]).spawn(prefix,data,image_id,smoke,resume,max_infrastructure_retries)
    print(json.dumps(dict(prefix=prefix,coordinator_call_id=call.object_id,image_id=image_id,models=len(data['models']),authoring_cells=len(data['models'])*data['replicates']*len(data['tracks']),smoke=smoke)))
