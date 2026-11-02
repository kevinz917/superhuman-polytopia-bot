"""Targeted infrastructure recovery; never retry a valid bot or a compilation zero."""
import io
import json
import os
import sys
import hashlib
import time
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),'/workspace']
import modal
from experiments.modal_skirmish_v5 import app,results,MOUNTS


def controller_states(call,call_id):
    pending=list(call.get_call_graph());states=[]
    while pending:
        node=pending.pop();pending.extend(node.children)
        if node.function_call_id==call_id:states.append(node.status.name)
    return states


def inspect_slot(prefix,cell_id,attempt,current_call_id=None):
    from polytopia_bot.skirmish.agent import source_receipt
    from polytopia_bot.skirmish.common import identifier
    results.reload();identifier(prefix);identifier(cell_id)
    if not 1<=attempt<=9:raise ValueError('Explicit infrastructure attempt index required')
    base=Path('/results/mini-v5')/prefix
    plan=json.loads((base/'authoring-plan.json').read_text())
    assert source_receipt()==plan['execution_source'],'Restore the frozen authoring source before retrying'
    cell=next(cell for cell in plan['cells'] if cell['run_id']==cell_id)
    run_id=cell_id+f'-infra{attempt}';identifier(run_id)
    if (Path('/results/mini-runs')/run_id).exists():raise ValueError('Attempt already started; reconcile it instead')
    call_path=base/'calls'/f'{run_id}.json'
    if call_path.exists():
        previous=json.loads(call_path.read_text())
        if previous['call_id']!=current_call_id:
            if current_call_id is not None:raise ValueError('Another call owns this allocation')
            allocated=modal.FunctionCall.from_id(previous['call_id'])
            try:allocated.get(timeout=0)
            except TimeoutError:
                states=controller_states(allocated,previous['call_id'])
                if not states or any(state=='PENDING' for state in states):raise ValueError('Allocated call is still pending; do not duplicate it') from None
                failure_type='terminal_without_output'
            except Exception as error:failure_type=type(error).__name__
            else:raise ValueError('Allocated call returned without an authoring directory; inspect it manually')
            # A failed/cancelled launcher with no authoring directory made no model calls.
            archive=base/'operator-recoveries'/'failed-unstarted-allocations';archive.mkdir(parents=True,exist_ok=True)
            (archive/(run_id+'-'+previous['call_id']+'.json')).write_text(json.dumps(dict(previous=previous,error_type=failure_type,source_directory_created=False,reclaimed_unix=time.time()),indent=2))
            call_path.unlink();results.commit()
    prior=[]
    for folder in [Path('/results/mini-runs')/cell_id]+sorted(Path('/results/mini-runs').glob(cell_id+'-infra*')):
        if not folder.exists():continue
        status=json.loads((folder/'status.json').read_text())
        report=json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else status
        if report['status'] in ('completed','failed_submission'):raise ValueError('A valid frozen attempt exists, including compilation failure; do not retry')
        previous_call=json.loads((base/'calls'/f'{folder.name}.json').read_text())['call_id']
        call=modal.FunctionCall.from_id(previous_call)
        terminal_states=controller_states(call,previous_call)
        if not terminal_states or any(state=='PENDING' for state in terminal_states):raise ValueError('Prior controller may still be running: '+folder.name)
        try:returned=call.get(timeout=0)
        except Exception:pass
        else:
            if returned.get('status') in ('completed','failed_submission'):raise ValueError('Recover the valid returned artifact instead of retrying: '+folder.name)
        sandbox_returncode=None
        if status.get('sandbox_id'):
            sandbox=modal.Sandbox.from_id(status['sandbox_id']);sandbox_returncode=sandbox.poll()
            if sandbox_returncode is None:
                # The controller is terminal, so this is an abandoned workspace, not a healthy run.
                archive=folder/'infrastructure-interruption';archive.mkdir(exist_ok=True)
                try:
                    source=sandbox.filesystem.read_bytes('/bot/src/main.rs');(archive/'main.rs').write_bytes(source)
                    snapshot=dict(source_sha256=hashlib.sha256(source).hexdigest(),source_bytes=len(source),qualified=False,captured_unix=time.time(),reason='abandoned workspace after terminal controller; not a deadline submission')
                except Exception as error:snapshot=dict(qualified=False,source_unavailable=True,error_type=type(error).__name__)
                (archive/'receipt.json').write_text(json.dumps(snapshot,indent=2));results.commit();sandbox.terminate();sandbox_returncode=sandbox.poll()
        prior.append(dict(run_id=folder.name,status=report['status'],reason=report.get('reason'),error=report.get('error'),sandbox_returncode=sandbox_returncode,terminal_controller_states=terminal_states))
    if not prior:raise ValueError('No interrupted infrastructure attempt to recover')
    return dict(prefix=prefix,cell_id=cell_id,run_id=run_id,config=cell['config'],prior=prior,attempt=attempt)


@app.function(cpu=1,memory=1024,timeout=180,volumes=MOUNTS)
def inspect_recovery(prefix: str,cell_id: str,attempt: int):
    return inspect_slot(prefix,cell_id,attempt)


@app.function(nonpreemptible=True,cpu=2,memory=6144,timeout=14400,retries=0,max_containers=2,volumes=MOUNTS)
def recover_author(prefix: str,cell_id: str,attempt: int,reason: str):
    from polytopia_bot.skirmish.common import write_json
    from polytopia_bot.skirmish.mini import run_mini_trial
    # Refresh before the original directory guard, including transparent cloud replays.
    info=inspect_slot(prefix,cell_id,attempt,modal.current_function_call_id())
    receipt={k:v for k,v in info.items() if k!='config'}
    receipt.update(reason=reason,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   call_id=modal.current_function_call_id(),started_unix=time.time(),
                   policy='fresh original-budget infrastructure retry; no access to earlier bot attempts or hidden games')
    root=Path('/results/mini-v5')/prefix/'operator-recoveries'
    write_json(root/(info['run_id']+'.json'),receipt);results.commit()
    sandbox_app=modal.App.lookup('polytopia-skirmish-sandboxes',create_if_missing=True,environment_name='staging')
    report=run_mini_trial(info['run_id'],info['config'],modal.Image.from_id(info['config']['_coding_image_id']),sandbox_app,results)
    receipt.update(finished_unix=time.time(),status=report['status']);write_json(root/(info['run_id']+'.json'),receipt);results.commit()
    return report


@app.local_entrypoint()
def retry(prefix: str,cell_id: str,attempt: int,reason: str):
    if not os.environ.get('OPENROUTER_API_KEY'):raise ValueError('Authorized OpenRouter key required')
    info=inspect_recovery.remote(prefix,cell_id,attempt)
    call=recover_author.with_options(secrets=[modal.Secret.from_dict({'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})]).spawn(prefix,cell_id,attempt,reason)
    receipt=dict(call_id=call.object_id,run_id=info['run_id'],cell_id=cell_id,attempt=attempt,reason=reason,created_unix=time.time())
    # Metadata upload is not another function invocation; the author remains the detached root call.
    with results.batch_upload() as batch:
        batch.put_file(io.BytesIO((json.dumps(receipt,indent=2)+'\n').encode()),f'/mini-v5/{prefix}/calls/{info["run_id"]}.json')
    print(json.dumps(receipt,indent=2))


@app.local_entrypoint()
def retry_batch(prefix: str,cell_ids: str,attempts: str,reason: str):
    if not os.environ.get('OPENROUTER_API_KEY'):raise ValueError('Authorized OpenRouter key required')
    cells=[value.strip() for value in cell_ids.split(',') if value.strip()];indices=[int(value) for value in attempts.split(',')]
    if len(cells)!=len(indices) or len(set(cells))!=len(cells):raise ValueError('Supply one explicit attempt index per distinct cell')
    planned=[]
    for cell,index in zip(cells,indices):
        try:info=inspect_recovery.remote(prefix,cell,index)
        except ValueError as error:
            print(json.dumps(dict(cell_id=cell,attempt=index,skipped=str(error))),flush=True)
        else:planned.append((info,index))
    bound=recover_author.with_options(secrets=[modal.Secret.from_dict({'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})])
    for info,index in planned:
        call=bound.spawn(prefix,info['cell_id'],index,reason)
        receipt=dict(call_id=call.object_id,run_id=info['run_id'],cell_id=info['cell_id'],attempt=index,reason=reason,created_unix=time.time())
        with results.batch_upload() as batch:batch.put_file(io.BytesIO((json.dumps(receipt,indent=2)+'\n').encode()),f'/mini-v5/{prefix}/calls/{info["run_id"]}.json')
        print(json.dumps(receipt),flush=True)
