"""Complete repeated v5 benchmark. All bots, sources and games remain on Modal staging."""
import json
import os
from pathlib import Path
import time
from functools import lru_cache
import modal

from experiments.modal_skirmish import runtime_image, mini_coding_image, results, MOUNTS, ROOT

runtime_image=runtime_image.add_local_dir(ROOT/'experiments','/workspace/experiments',ignore=['**/__pycache__/**'])
app=modal.App('polytopia-skirmish-v5',image=runtime_image)


@lru_cache(maxsize=1)
def evaluation_reference():
    """Reuse compiled deterministic inference between sequential games in one worker."""
    from polytopia_bot.skirmish.game import warm_reference
    from polytopia_bot.skirmish.common import REFERENCE_SHA
    return warm_reference('/results/references/hero-1v1/final.npz',REFERENCE_SHA)


@app.function(cpu=2,memory=6144,timeout=14400,retries=0,max_containers=20,volumes=MOUNTS)
def author(run_id: str, config: dict):
    from polytopia_bot.skirmish.mini import run_mini_trial
    sandbox_app=modal.App.lookup('polytopia-skirmish-sandboxes',create_if_missing=True,environment_name='staging')
    return run_mini_trial(run_id,config,modal.Image.from_id(config['_coding_image_id']),sandbox_app,results)


@app.function(cpu=1,memory=2048,timeout=300,retries=0,max_containers=11,volumes=MOUNTS)
def probe(prefix: str, entry: dict, image_id: str):
    """Two real tool-use stages exercise provider transport and conversation continuation."""
    import subprocess,sys
    from polytopia_bot.skirmish.common import write_json
    from polytopia_bot.skirmish.mini import stop_coder
    folder=Path('/results/mini-v5')/prefix/'probes'/entry['label']
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'report.json').exists():return json.loads((folder/'report.json').read_text())
    sandbox=modal.Sandbox.create('sleep','infinity',image=modal.Image.from_id(image_id),
        app=modal.App.lookup('polytopia-skirmish-sandboxes',create_if_missing=True,environment_name='staging'),
        cpu=2,memory=2048,block_network=True,timeout=260,workdir='/bot')
    history=None;stages=[]
    try:
        for stage in range(2):
            trajectory=folder/f'stage-{stage}.json'
            config=dict(model=entry['model'],parameters={**entry['parameters'],'max_tokens':2048},
                sandbox_id=sandbox.object_id,system='Follow the tool transport test exactly.',
                task='Transport test only. First use bash: printf SMOKE_OK. Then, in a separate tool call, use bash: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT. Do not read files or write a bot.',
                seconds=90,deadline_unix=time.time()+90,trajectory=str(trajectory),history=history)
            with (folder/f'stage-{stage}.log').open('w') as log:
                proc=subprocess.Popen([sys.executable,'-m','polytopia_bot.skirmish.mini_worker'],
                    stdin=subprocess.PIPE,stdout=log,stderr=log,text=True)
                try:proc.communicate(json.dumps(config),timeout=90)
                except subprocess.TimeoutExpired:proc.kill();proc.communicate();raise RuntimeError('provider probe timed out')
            stop_coder(sandbox)
            data=json.loads(trajectory.read_text())
            if proc.returncode or data['info'].get('exit_status')!='Submitted':raise RuntimeError('provider tool/resume probe failed')
            if not any(m.get('role')=='tool' and 'SMOKE_OK' in str(m.get('content','')) for m in data['messages']):raise RuntimeError('tool was not executed')
            stages.append(data['info']['model_stats']);history=str(trajectory)
        report=dict(passed=True,label=entry['label'],model=entry['model'],stages=stages)
    except Exception as error:report=dict(passed=False,label=entry['label'],error=f'{type(error).__name__}: {error}',stages=stages)
    finally:sandbox.terminate()
    write_json(folder/'report.json',report);results.commit();return report


def load_bot(spec, budget):
    from polytopia_bot.skirmish.common import sha,REFERENCE_SHA
    from polytopia_bot.skirmish.game import WasiBot,warm_reference
    if spec['kind']=='expert':return evaluation_reference()
    folder=Path('/results/mini-runs')/spec['run_id']
    if spec['kind']=='submission':folder=folder/f'stage-{spec["stage"]}'
    if sha((folder/'main.rs').read_bytes())!=spec['source_sha256'] or sha((folder/'bot.wasm').read_bytes())!=spec['wasm_sha256']:
        raise ValueError('immutable evaluated artifact digest mismatch')
    return WasiBot(folder/'bot.wasm',budget.fuel,budget.move_seconds)


@app.function(cpu=2,memory=6144,timeout=420,retries=1,max_containers=32,volumes=MOUNTS)
def game(prefix: str, job: dict):
    import uuid
    from polytopia_bot.skirmish.common import Budget,write_json,sha
    from polytopia_bot.skirmish.game import play_game
    from polytopia_bot.skirmish.mini import audit_trace
    results.reload()
    root=Path('/results/mini-v5')/prefix/'games'/f'{job["index"]:06d}'
    summary=root/'summary.json'
    if summary.exists():
        row=json.loads(summary.read_text())
        if row['job_sha256']!=job['job_sha256']:raise ValueError('different plan already occupies game index')
        return row
    candidate=job['candidate'];opponent=job['opponent']
    metadata=dict(job_index=job['index'],job_sha256=job['job_sha256'],model_label=candidate['model_label'],
        track=candidate['track'],replicate=candidate['replicate'],stage=candidate['stage'],
        submission_quality=candidate['submission_quality'],opponent_label=opponent['label'])
    if job['invalid_submission']:
        row=dict(a=candidate['label'],b=opponent['label'],seat=job['seat'],seed=job['seed'],map_type=job['map_type'],
            phase=job['phase'],status='invalid_submission',reason='compile_failure',result='loss',winner=3-job['seat'],
            actions=0,audit=dict(passed=True,virtual=True,actions=0))
    else:
        attempt=root/uuid.uuid4().hex;attempt.mkdir(parents=True)
        trace=attempt/'game.jsonl';budget=Budget()
        a=load_bot(candidate,budget)
        try:b=load_bot(opponent,budget)
        except BaseException:a.close();raise
        row=play_game(a,b,job['seed'],job['seat'],budget,trace,phase=job['phase'],
                      labels=[candidate['label'],opponent['label']],map_type=job['map_type'],cap_policy='engine_draw')
        actions=audit_trace(row,trace)
        row.update(trace_relative=str(trace.relative_to(Path('/results/mini-v5')/prefix)),
                   audit=dict(passed=True,virtual=False,actions=actions))
    row.update(metadata)
    write_json(summary,row);results.commit();return row


def reference_receipts(references):
    from polytopia_bot.skirmish.mini import load_submission
    from polytopia_bot.skirmish.common import REFERENCE_SHA
    output=[]
    for ref in references:
        ref=dict(ref)
        if ref['kind']=='expert':ref['checkpoint_sha256']=REFERENCE_SHA
        else:
            _,report,_=load_submission(ref['run_id'])
            ref.update(source_sha256=report['final_source_sha256'],wasm_sha256=report['final_wasm_sha256'],
                       origin_model=report['model'],origin_track=report['track'])
        output.append(ref)
    return output


@app.function(cpu=1,memory=2048,timeout=90,retries=0,volumes=MOUNTS)
def archive_interrupted(run_id: str):
    """Preserve an interrupted workspace without treating it as a valid submission."""
    from polytopia_bot.skirmish.common import identifier,write_json,sha
    from polytopia_bot.skirmish.mini import stop_coder
    root=Path('/results/mini-runs')/identifier(run_id)
    if not (root/'status.json').exists():return dict(status='no_workspace_created')
    status=json.loads((root/'status.json').read_text())
    if not status.get('sandbox_id'):return dict(status='no_sandbox_created')
    try:
        sandbox=modal.Sandbox.from_id(status['sandbox_id']);stop_coder(sandbox)
        source=sandbox.filesystem.read_bytes('/bot/src/main.rs')
        folder=root/'infrastructure-interruption';folder.mkdir(exist_ok=True)
        (folder/'main.rs').write_bytes(source)
        receipt=dict(status='archived_invalid_attempt',source_sha256=sha(source),source_bytes=len(source),
                     captured_unix=time.time(),reason='controller/transport failure; not a qualified deadline submission')
        write_json(folder/'receipt.json',receipt);results.commit();sandbox.terminate();return receipt
    except Exception as error:return dict(status='sandbox_unavailable',error=type(error).__name__)


@app.function(cpu=1,memory=4096,timeout=86400,retries=0,max_containers=1,volumes=MOUNTS)
def coordinate(prefix: str, manifest: dict, image_id: str, smoke: bool=False, resume: bool=False):
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from polytopia_bot.skirmish.common import write_json,canonical,sha
    from polytopia_bot.skirmish.cohort import authoring_cells,make_evaluation_plan,summarize_evaluation,validate_run_protocol
    from polytopia_bot.skirmish.protocol import make_map_suite
    from polytopia_bot.skirmish.agent import source_receipt
    results.reload()
    root=Path('/results/mini-v5')/prefix;root.mkdir(parents=True,exist_ok=True)
    cells=authoring_cells(prefix,manifest,image_id,smoke=smoke)
    execution_source=source_receipt()
    digest=sha(canonical(dict(manifest=manifest,image_id=image_id,smoke=smoke,execution_source=execution_source)).encode())
    plan_path=root/'authoring-plan.json'
    if plan_path.exists():
        if not resume:raise ValueError('cohort already allocated; use explicit resume after inspecting the existing call')
        plan=json.loads(plan_path.read_text())
        if plan['sha256']!=digest:raise ValueError('cohort already has a different immutable manifest/image')
    else:
        plan=dict(sha256=digest,manifest=manifest,image_id=image_id,cells=cells,smoke=smoke,
                  execution_source=execution_source,
                  references=reference_receipts(manifest['references']),created_unix=time.time(),
                  retry_policy='one fresh full-budget attempt after infrastructure failure; never retry compile failure or starter')
        write_json(plan_path,plan);results.commit()
    state=dict(status='probing',prefix=prefix,smoke=smoke,total=len(cells),runs={},errors=[],coordinator_call_id=modal.current_function_call_id())
    def save():write_json(root/'status.json',state);results.commit()
    save()
    secret=modal.Secret.from_dict({'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})
    pending=[probe.with_options(secrets=[secret]).spawn(prefix,e,image_id) for e in manifest['models']]
    checks=[call.get() for call in pending]
    write_json(root/'probes.json',checks)
    if not all(p['passed'] for p in checks):
        state.update(status='needs_attention',errors=[p for p in checks if not p['passed']]);save();return state
    state['status']='authoring';save()
    bound=author.with_options(secrets=[secret]);calls=[]
    for cell in cells:
        call_path=root/'calls'/f'{cell["run_id"]}.json'
        if call_path.exists():
            call=modal.FunctionCall.from_id(json.loads(call_path.read_text())['call_id'])
        else:
            call=bound.spawn(cell['run_id'],cell['config'])
            write_json(call_path,dict(call_id=call.object_id));results.commit()
        calls.append((cell,call))
    def finish(item):
        cell,call=item
        try:report=call.get()
        except Exception as error:report=dict(status='infrastructure_error',error=f'{type(error).__name__}: {error}')
        if report['status'] not in ('completed','failed_submission'):
            replacement=cell['run_id']+'-infra1'
            path=root/'calls'/f'{replacement}.json'
            original=report
            original['source_archive']=archive_interrupted.remote(cell['run_id'])
            if path.exists():retry=modal.FunctionCall.from_id(json.loads(path.read_text())['call_id'])
            else:
                retry=bound.spawn(replacement,cell['config']);write_json(path,dict(call_id=retry.object_id,original_status=original['status']));results.commit()
            try:report=retry.get()
            except Exception as error:report=dict(status='infrastructure_error',error=f'{type(error).__name__}: {error}')
            report={**report,'effective_run_id':replacement,'excluded_infrastructure_attempt':original}
        return cell['run_id'],report
    with ThreadPoolExecutor(max_workers=20) as pool:
        for future in as_completed([pool.submit(finish,item) for item in calls]):
            run_id,report=future.result()
            write_json(root/'runs'/f'{run_id}.json',report)
            state['runs'][run_id]={k:report.get(k) for k in ('status','model','track','effective_run_id','submission_quality','error')}
            state['authoring_finished']=len(state['runs']);save()
    reports={c['run_id']:json.loads((root/'runs'/f'{c["run_id"]}.json').read_text()) for c in cells}
    if any(r['status'] not in ('completed','failed_submission') for r in reports.values()):
        state['status']='needs_attention';save();return state
    try:
        results.reload()
        for cell in cells:
            run_id=reports[cell['run_id']].get('effective_run_id',cell['run_id'])
            protocol=json.loads((Path('/results/mini-runs')/run_id/'protocol.json').read_text())
            validate_run_protocol(cell,protocol,execution_source,image_id)
    except Exception as error:
        state.update(status='needs_attention',errors=[dict(kind='protocol_mismatch',error=str(error))]);save();return state
    path=root/'evaluation-plan.json'
    if path.exists():evaluation=json.loads(path.read_text())
    else:
        evaluation=make_evaluation_plan(cells,reports,make_map_suite(1 if smoke else 5),plan['references'],smoke=smoke)
        write_json(path,evaluation);results.commit()
    state.update(status='evaluating',evaluation_games=len(evaluation['jobs']),games_finished=0);save()
    rows={};errors=[]
    def execute(jobs):
        for result in game.starmap(((prefix,j) for j in jobs),order_outputs=False,return_exceptions=True):
            if isinstance(result,BaseException):errors.append(str(result)[:500])
            else:
                rows[result['job_index']]=result
                if len(rows)%25==0:state['games_finished']=len(rows);save()
    execute(evaluation['jobs'])
    missing=[j for j in evaluation['jobs'] if j['index'] not in rows]
    if missing:execute(missing)
    if len(rows)!=len(evaluation['jobs']):
        state.update(status='needs_attention',games_finished=len(rows),errors=errors);save();return state
    ordered=[rows[i] for i in range(len(rows))]
    write_json(root/'games.json',ordered)
    report=summarize_evaluation(evaluation,cells,ordered)
    report.update(prefix=prefix,manifest_sha256=sha(canonical(manifest).encode()),allocation_sha256=digest,
                  execution_source_sha256=sha(canonical(execution_source).encode()),references=plan['references'],
                  audit=dict(passed=all(r['audit']['passed'] for r in ordered),
                             actual_games=sum(not r['audit']['virtual'] for r in ordered),
                             virtual_compile_failure_losses=sum(r['audit']['virtual'] for r in ordered),
                             actions=sum(r['audit']['actions'] for r in ordered)))
    # Elo is secondary and roster-dependent. Pool every independent build, never
    # choose a representative/best bot. Reference games anchor the fit.
    from polytopia_bot.skirmish.tournament import summarize
    report['peer_ladders']={}
    for track in manifest['tracks']:
        part=[]
        for row in ordered:
            if row['track']!=track or row['stage']!=(3 if track=='feedback-3r-40m' else 0) or row['status']=='invalid_submission':continue
            job=evaluation['jobs'][row['job_index']]
            part.append({**row,'a':row['model_label'],'b':job['opponent'].get('model_label',job['opponent']['label'])})
        labels=sorted({r[k] for r in part for k in ('a','b')})
        report['peer_ladders'][track]=summarize(part,labels,bootstrap=100)
    write_json(root/'report.json',report)
    state.update(status=report['status'],games_finished=len(rows),audit=report['audit'],errors=[]);save()
    return dict(prefix=prefix,status=report['status'],games=len(rows),audit=report['audit'])


@app.function(cpu=1,memory=2048,timeout=60,volumes=MOUNTS)
def inspect_cohort(prefix: str):
    from polytopia_bot.skirmish.common import identifier
    results.reload()
    root=Path('/results/mini-v5')/identifier(prefix)
    state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else dict(status='not_found')
    if (root/'authoring-plan.json').exists():
        plan=json.loads((root/'authoring-plan.json').read_text())
        state['manifest']=plan['manifest']
        state['image_id']=plan['image_id']
        state['current_stages']={}
        for cell in plan['cells']:
            folder=Path('/results/mini-runs')/cell['run_id']
            if (folder/'status.json').exists():
                active=json.loads((folder/'status.json').read_text())
                item={k:active.get(k) for k in ('status','stage','model','track')}
                start=folder/f'stage-{active.get("stage",0)}.start.json'
                if start.exists():item['clock']=json.loads(start.read_text())
                state['current_stages'][cell['run_id']]=item
    if (root/'report.json').exists():
        state['report']=json.loads((root/'report.json').read_text())
        state['runs_detail']={p.stem:{k:v for k,v in json.loads(p.read_text()).items() if k not in ('excluded_infrastructure_attempt','stages','trajectory_sha256')}
                              for p in (root/'runs').glob('*.json')}
        state['stage_receipts']={p.stem:[{k:v for k,v in stage.items() if k!='compile_output'} for stage in json.loads(p.read_text()).get('stages',[])]
                                 for p in (root/'runs').glob('*.json')}
        state['infrastructure_exclusions']={p.stem:r['excluded_infrastructure_attempt']
            for p in (root/'runs').glob('*.json') if (r:=json.loads(p.read_text())).get('excluded_infrastructure_attempt')}
        scored=sum(s.get('instance_cost') or 0 for stages in state['stage_receipts'].values() for s in stages)
        probes=json.loads((root/'probes.json').read_text())
        probe_cost=sum(s.get('instance_cost') or 0 for p in probes for s in p.get('stages',[]))
        excluded=0.
        for run in state['infrastructure_exclusions']:
            folder=Path('/results/mini-runs')/run
            for p in folder.glob('stage-*.trajectory.json'):
                excluded+=json.loads(p.read_text()).get('info',{}).get('model_stats',{}).get('instance_cost',0) or 0
        state['api_cost_report']=dict(scored_completed_calls=scored,probe_completed_calls=probe_cost,
            excluded_completed_calls=excluded,recorded_total=scored+probe_cost+excluded,
            coverage='partial completed-call receipts only; interrupted calls and Modal compute are additional')
    return state


@app.local_entrypoint()
def run(prefix: str, manifest: str='configs/skirmish/mini-v5-top11.json', smoke: bool=False):
    from modal.config import config
    if config.get('environment')!='staging':raise RuntimeError('pass --env staging')
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from polytopia_bot.skirmish.cohort import validate_manifest
    data=validate_manifest(json.loads(Path(manifest).read_text()))
    if not os.environ.get('OPENROUTER_API_KEY'):raise RuntimeError('trusted OpenRouter key required')
    image_id=mini_coding_image.build(app).object_id
    call=coordinate.with_options(secrets=[modal.Secret.from_dict({'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})]).spawn(prefix,data,image_id,smoke)
    print(json.dumps(dict(prefix=prefix,call_id=call.object_id,image_id=image_id,authoring_cells=len(data['models'])*data['replicates']*2,smoke=smoke)),flush=True)


@app.local_entrypoint()
def status(prefix: str, output: str='', watch: bool=False):
    from collections import Counter
    while True:
        data=inspect_cohort.remote(prefix)
        if output:Path(output).write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(dict(status=data['status'],authoring_finished=data.get('authoring_finished'),
            total=data.get('total'),games_finished=data.get('games_finished'),evaluation_games=data.get('evaluation_games'),
            stage_counts=dict(Counter(s['status'] for s in data.get('current_stages',{}).values())),errors=data.get('errors'))),flush=True)
        if data['status'] in ('completed','provisional') and output:
            from scripts.render_skirmish_v5 import render
            Path(output).with_suffix('.md').write_text(render(data))
        if not watch or data['status'] in ('completed','provisional','needs_attention'):return
        time.sleep(30)


@app.local_entrypoint()
def resume(prefix: str):
    """Explicitly reuse a frozen cohort after its original coordinator has stopped."""
    data=inspect_cohort.remote(prefix)
    if data['status'] in ('completed','provisional'):raise ValueError('terminal cohort already has a report; do not rerun it')
    if not data.get('manifest') or not data.get('image_id'):raise ValueError('no existing immutable plan')
    if data.get('coordinator_call_id'):
        try:modal.FunctionCall.from_id(data['coordinator_call_id']).get(timeout=0)
        except TimeoutError:raise RuntimeError('original coordinator is still running; do not start a duplicate')
        except Exception:pass  # A finished failed call may be resumed from its frozen receipts.
    if not os.environ.get('OPENROUTER_API_KEY'):raise RuntimeError('trusted OpenRouter key required')
    call=coordinate.with_options(secrets=[modal.Secret.from_dict({'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})]).spawn(
        prefix,data['manifest'],data['image_id'],data['smoke'],True)
    print(json.dumps(dict(prefix=prefix,resume_call_id=call.object_id)),flush=True)
