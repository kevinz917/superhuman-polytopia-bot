"""Offline recovery using frozen author artifacts and the original Rust game engine.
Uniform 900-second referee guard and streaming replay verification.
No model requests or selection by game score. Refuses incomplete authoring.
"""
import sys,json,time,hashlib,math,os,re
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),'/workspace']
import modal
from experiments.modal_skirmish_v5 import app,results,MOUNTS,runtime_image
operator_image=runtime_image.add_local_python_source('scripts')
EVALUATION_REVISION='referee-900-stream-v1'
REFEREE_SECONDS=900

def saved_model_stats(path):
    """Read optional agent telemetry without loading its full conversation."""
    try:
        with Path(path).open('rb') as stream:
            head=stream.read(65536);stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-65536));tail=stream.read()
        for block in (head,tail):
            text=block.decode('utf-8',errors='replace')
            for match in re.finditer(r'"model_stats"\s*:\s*',text):
                try:stats,_=json.JSONDecoder().raw_decode(text[match.end():])
                except ValueError:continue
                if isinstance(stats,dict) and ('api_calls' in stats or 'instance_cost' in stats):return stats
    except OSError:pass
    return {}

def file_sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
 return digest.hexdigest()

def play_game_streamed(a, b, seed, seat, budget, trace_path, *, phase, labels, authoring_deadline=None,
              map_type='classic', cap_policy='unresolved'):
    """Run the original move protocol with bounded memory trace hashing."""
    if not os.environ.get('MODAL_TASK_ID'):
        raise RuntimeError('games must run on Modal')
    from polytopia_bot.studio.environment import RustEnvironment
    from polytopia_bot.skirmish.game import ReferenceBot, WasiBot, BotFault, reset_map
    from polytopia_bot.skirmish.common import canonical, sha
    from dataclasses import asdict
    env = RustEnvironment(size=14)
    bots = {seat: a, 3 - seat: b}
    started = time.monotonic()
    row = dict(a=labels[0], b=labels[1], seed=seed, seat=seat, phase=phase, map_type=map_type,
               status='unresolved', reason='game_timeout', winner=None, actions=0)
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    private = path.with_suffix('.private.jsonl')
    chain = '0' * 64
    try:
        obs = reset_map(env, seed, map_type)
        # Warm reference inference outside the game-clock; compile time isn't a bot's move time.
        for bot in bots.values():
            if isinstance(bot, ReferenceBot):
                bot.select(obs)
        started = time.monotonic()
        game_deadline = started+budget.game_seconds
        if authoring_deadline is not None:
            game_deadline = min(game_deadline,authoring_deadline)
        for bot in bots.values():
            if isinstance(bot,WasiBot): bot.absolute_deadline = game_deadline
        with path.open('x') as public_log, private.open('x') as private_log:
            for _ in range(10_000):
                if time.monotonic() >= game_deadline:
                    break
                player = int(env.state.current_player)
                obs = env.observe(player)
                try:
                    action_id = bots[player].select(obs)
                except BotFault as error:
                    if time.monotonic() >= game_deadline:
                        row.update(reason='authoring_deadline' if authoring_deadline is not None else 'game_timeout')
                    else:
                        row.update(status='forfeit', reason=str(error), winner=3 - player, fault_player=player)
                    break
                entry = dict(player=player, version=obs.state_version, action_id=action_id,
                             turn=obs.visible_state['turn'])
                chain = sha((chain + canonical(entry)).encode())
                private_log.write(canonical(dict(**entry, chain=chain)) + '\n')
                if player == seat:
                    public_log.write(canonical(dict(observation=asdict(obs), action_id=action_id)) + '\n')
                transition = env.step(action_id=action_id, expected_state_version=obs.state_version)
                row['actions'] += 1
                if transition.terminated:
                    winner = max(transition.rewards, key=transition.rewards.get)
                    row.update(status='native', reason=transition.reason, winner=winner)
                    break
                if transition.truncated:
                    row.update(reason='engine_truncation')
                    break
            else:
                row.update(reason='action_limit')
        row.update(turns=int(env.state.turn), seconds=time.monotonic() - started,
                   trace_sha256=file_sha(path), private_trace_sha256=file_sha(private),
                   action_chain_sha256=chain)
    finally:
        env.close()
        a.close()
        b.close()
    row['result'] = 'unresolved' if row['winner'] is None else 'win' if row['winner'] == seat else 'loss'
    from polytopia_bot.skirmish.protocol import apply_cap_policy
    return apply_cap_policy(row,cap_policy)


def audit_trace_streamed(row, path):
    from dataclasses import asdict
    from polytopia_bot.skirmish.common import canonical, sha
    from polytopia_bot.skirmish.game import reset_map
    from polytopia_bot.studio.environment import RustEnvironment
    private=path.with_suffix('.private.jsonl')
    assert file_sha(path)==row['trace_sha256']
    assert file_sha(private)==row['private_trace_sha256']
    env=RustEnvironment(size=14);count=0;chain='0'*64
    try:
        with path.open() as public_file,private.open() as private_file:
            public=iter(json.loads(line) for line in public_file)
            reset_map(env,row['seed'],row['map_type'])
            for line in private_file:
                entry=json.loads(line);expected=entry.pop('chain')
                chain=sha((chain+canonical(entry)).encode());assert chain==expected
                obs=env.observe(int(env.state.current_player))
                assert obs.player==entry['player'] and obs.state_version==entry['version']
                assert obs.visible_state['turn']==entry['turn']
                if obs.player==row['seat']:assert next(public)==dict(observation=asdict(obs),action_id=entry['action_id'])
                env.step(action_id=entry['action_id'],expected_state_version=obs.state_version);count+=1
            assert next(public,None) is None
            assert count==row['actions'] and chain==row['action_chain_sha256']
            assert int(env.state.turn)==row['turns']
            if row['status']=='native':
                assert bool(env.state.terminated)
                assert list(env.games.batch.terminal_rewards()[0])[row['winner']-1]>0
            elif row['status']=='draw':
                assert row['winner'] is None and row['reason'] in ('engine_truncation','action_limit')
                if row['reason']=='engine_truncation':assert bool(env.state.truncated)
            elif row['status']=='forfeit':assert not bool(env.state.terminated) and row['winner']==3-row['fault_player']
    finally:env.close()
    return count


@app.function(nonpreemptible=True,image=operator_image,cpu=2,memory=6144,timeout=2100,retries=1,max_containers=64,volumes=MOUNTS)
def recovery_game(prefix: str, job: dict):
 import uuid
 from experiments.modal_skirmish_v5 import load_bot
 from polytopia_bot.skirmish.common import Budget,write_json
 results.reload();root=Path('/results/mini-v5')/prefix/'games'/f'{job["index"]:06d}';summary=root/'summary.json'
 assert job['evaluation_revision']==EVALUATION_REVISION and job['referee_seconds']==REFEREE_SECONDS
 assert job['evaluation_source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'evaluation source changed'
 if summary.exists():
  row=json.loads(summary.read_text());assert row['job_sha256']==job['job_sha256'];return row
 candidate=job['candidate'];opponent=job['opponent']
 metadata=dict(job_index=job['index'],job_sha256=job['job_sha256'],model_label=candidate['model_label'],track=candidate['track'],replicate=candidate['replicate'],stage=candidate['stage'],submission_quality=candidate['submission_quality'],opponent_label=opponent['label'],evaluation_revision=EVALUATION_REVISION,referee_seconds=REFEREE_SECONDS)
 if job['invalid_submission']:
  row=dict(a=candidate['label'],b=opponent['label'],seat=job['seat'],seed=job['seed'],map_type=job['map_type'],phase=job['phase'],status='invalid_submission',reason='compile_failure',result='loss',winner=3-job['seat'],actions=0,audit=dict(passed=True,virtual=True,actions=0))
 else:
  attempt=root/uuid.uuid4().hex;attempt.mkdir(parents=True);trace=attempt/'game.jsonl';budget=Budget(game_seconds=REFEREE_SECONDS);a=load_bot(candidate,budget)
  try:b=load_bot(opponent,budget)
  except BaseException:a.close();raise
  row=play_game_streamed(a,b,job['seed'],job['seat'],budget,trace,phase=job['phase'],labels=[candidate['label'],opponent['label']],map_type=job['map_type'],cap_policy='engine_draw')
  actions=audit_trace_streamed(row,trace);row.update(trace_relative=str(trace.relative_to(Path('/results/mini-v5')/prefix)),audit=dict(passed=True,virtual=False,actions=actions))
 row.update(metadata);write_json(summary,row);results.commit();return row

@app.function(nonpreemptible=True,image=operator_image,cpu=1,memory=4096,timeout=86400,retries=0,max_containers=1,volumes=MOUNTS)
def finalize(prefix: str, dry_run: bool=False):
 from polytopia_bot.skirmish.common import write_json,canonical,sha
 from polytopia_bot.skirmish.cohort import validate_run_protocol,make_evaluation_plan,summarize_evaluation
 from polytopia_bot.skirmish.agent import source_receipt
 from polytopia_bot.skirmish.protocol import make_map_suite
 from polytopia_bot.skirmish.tournament import summarize
 results.reload();root=Path('/results/mini-v5')/prefix;plan=json.loads((root/'authoring-plan.json').read_text());assert plan['execution_source']==source_receipt(),'execution source drift'
 if (root/'report.json').exists():
  if dry_run:return dict(status='already_evaluated',prefix=prefix,report_exists=True)
  raise ValueError('Cohort already has a report; do not overwrite it')
 cells=plan['cells'];reports={};selection=[];missing=[];all_attempts=[];costs={};validation_failures=[];rebuild_pending=[];rebuild_failed=[]
 for cell in cells:
  candidates=[Path('/results/mini-runs')/cell['run_id']]+sorted(Path('/results/mini-runs').glob(cell['run_id']+'-infra*'),key=lambda p:int(p.name.rsplit('infra',1)[1]))
  eligible=[]
  for folder in candidates:
   cost=0.
   for trajectory in folder.glob('stage-*.trajectory.json'):
    try:
     value=float(saved_model_stats(trajectory).get('instance_cost',0) or 0)
     if math.isfinite(value) and value>=0:cost+=value
    except (ValueError,TypeError):pass
   costs[folder.name]=dict(model_label=cell['model_label'],recorded_api_usd=cost)
   p=folder/'report.json'
   if not p.exists():
    status_path=folder/'status.json'
    if status_path.exists():
     observed=json.loads(status_path.read_text());all_attempts.append(dict(run_id=folder.name,status=observed.get('status'),reason=observed.get('reason'),report_available=False))
    continue
   r=json.loads(p.read_text());all_attempts.append(dict(run_id=folder.name,status=r['status'],reason=r.get('reason'),report_available=True))
   if r['status'] not in ('completed','failed_submission'):continue
   try:
    validate_run_protocol(cell,json.loads((folder/'protocol.json').read_text()),plan['execution_source'],plan['image_id'])
    assert len(r['stages'])==(4 if cell['track']=='feedback-3r-40m' else 1)
    for s in r['stages']:
     stage=folder/f'stage-{s["stage"]}';assert sha((stage/'main.rs').read_bytes())==s['source_sha256']
     clock=json.loads((folder/f'stage-{s["stage"]}.start.json').read_text())
     if s.get('cutoff'):assert s['cutoff']['captured_unix']>=clock['started_unix'],'cutoff predates current stage clock; interrupted invocation metadata mixed'
     if s.get('wasm_sha256'):assert sha((stage/'bot.wasm').read_bytes())==s['wasm_sha256']
   except Exception as error:
    validation_failures.append(dict(run_id=folder.name,error_type=type(error).__name__,error=str(error)));continue
   eligible.append((folder,r))
  if not eligible:missing.append(cell['run_id']);continue
  # Earliest valid attempt, before any games. Never choose by performance.
  folder,r=eligible[0];r=dict(r)
  if folder.name!=cell['run_id']:r['effective_run_id']=folder.name
  base_report=Path('/results/mini-runs')/cell['run_id']/'report.json'
  if folder.name!=cell['run_id']:
   fallback=base_report if base_report.exists() else base_report.with_name('status.json')
   if fallback.exists():r['excluded_infrastructure_attempt']={**json.loads(fallback.read_text()),'exclusion_reason':'no complete validated frozen artifact selected from this interrupted attempt; see operator recovery ledger'}
  if not plan['smoke']:
   for stage in r['stages']:
    proof_path=root/'stage-rebuild-checks'/f'{folder.name}-stage-{stage["stage"]}.json'
    try:proof=json.loads(proof_path.read_text())
    except (FileNotFoundError,ValueError):proof={}
    if proof.get('source_sha256')!=stage['source_sha256'] or proof.get('expected_wasm_sha256')!=stage.get('wasm_sha256'):rebuild_pending.append(dict(run_id=folder.name,stage=stage['stage']))
    elif not proof.get('identical'):rebuild_failed.append(dict(run_id=folder.name,stage=stage['stage']))
  reports[cell['run_id']]=r;selection.append(dict(cell_id=cell['run_id'],selected=folder.name,later_valid_duplicates_excluded=[p.name for p,_ in eligible[1:]],rule='earliest valid attempt by fixed attempt index, without using scores'))
 if missing:return dict(status='needs_authoring',eligible=len(reports),total=len(cells),missing=missing,artifact_validation_failures=validation_failures)
 if rebuild_failed:return dict(status='needs_attention',reason='fresh source rebuild differs',failures=rebuild_failed)
 if rebuild_pending:return dict(status='needs_rebuild',eligible=len(reports),total=len(cells),pending=rebuild_pending)
 previous=json.loads((root/'status.json').read_text())
 if dry_run:return dict(status='ready' if len(reports)==len(cells) else 'needs_authoring',eligible=len(reports),total=len(cells),selection=selection,existing_evaluation_plan=(root/'evaluation-plan.json').exists())
 if previous['status'] in ('completed','provisional'):raise ValueError('cohort already evaluated; do not overwrite it')
 old=previous.get('coordinator_call_id');this=modal.current_function_call_id()
 if old and old!=this:modal.FunctionCall.from_id(old).cancel(terminate_containers=True)
 # Every slot is frozen. Cancel queued/stranded author calls before sampling maps.
 cancelled=[]
 coordinator_ids={old}
 for p in (root/'operator-recoveries').glob('coordinator*.json'):
  d=json.loads(p.read_text());coordinator_ids.update(d.get(k) for k in ('coordinator_call_id','previous_coordinator','resume_call_id'))
 for coordinator_id in coordinator_ids:
  if not coordinator_id or coordinator_id==this:continue
  def walk(nodes):
   for node in nodes:
    if node.function_name=='author' and node.status.name=='PENDING':
     modal.FunctionCall.from_id(node.function_call_id).cancel(terminate_containers=True);cancelled.append(node.function_call_id)
    walk(node.children)
  walk(modal.FunctionCall.from_id(coordinator_id).get_call_graph())
 for path in (root/'calls').glob('*.json'):
  call_id=json.loads(path.read_text())['call_id'];call=modal.FunctionCall.from_id(call_id)
  try:call.get(timeout=0)
  except TimeoutError:call.cancel(terminate_containers=True);cancelled.append(call_id)
  except Exception:pass
 cleanup_failures=[]
 # Terminate any unused coder sandbox; source/trajectories already on the volume are retained.
 for cell in cells:
  for folder in [Path('/results/mini-runs')/cell['run_id']]+list(Path('/results/mini-runs').glob(cell['run_id']+'-infra*')):
   path=folder/'status.json'
   if not path.exists():continue
   s=json.loads(path.read_text())
   if s.get('sandbox_id'):
    try:sb=modal.Sandbox.from_id(s['sandbox_id']);running=sb.poll() is None
    except Exception as error:
     cleanup_failures.append(dict(run_id=folder.name,sandbox_id=s['sandbox_id'],error_type=type(error).__name__));continue
    if running:
     archive=folder/'infrastructure-interruption';archive.mkdir(exist_ok=True)
     try:
      source=sb.filesystem.read_bytes('/bot/src/main.rs');(archive/'main.rs').write_bytes(source);write_json(archive/'receipt.json',dict(source_sha256=sha(source),source_bytes=len(source),captured_unix=time.time(),qualified=False,reason='unused interrupted attempt at final authoring closure; selected attempt already frozen'))
     except Exception as error:write_json(archive/'receipt.json',dict(qualified=False,source_unavailable=True,error=type(error).__name__))
     sb.terminate()
 selected_ids={item['selected'] for item in selection}
 for attempt in all_attempts:attempt['selected']=attempt['run_id'] in selected_ids
 snapshot=root/'operator-recoveries'/'frozen-evaluator.py';snapshot.parent.mkdir(parents=True,exist_ok=True)
 if snapshot.exists():assert file_sha(snapshot)==sha(Path(__file__).read_bytes()),'frozen evaluator snapshot differs'
 else:snapshot.write_bytes(Path(__file__).read_bytes())
 write_json(root/'operator-recoveries'/'finalizer-authoring-selection.json',dict(selection=selection,artifact_validation_failures=validation_failures,unused_sandbox_cleanup_failures=cleanup_failures,cancelled_stranded_calls=cancelled,previous_coordinator=old,finalizer_call_id=this,all_attempts=all_attempts,source_sha256=sha(Path(__file__).read_bytes()),created_unix=time.time()))
 for run,r in reports.items():write_json(root/'runs'/f'{run}.json',r)
 path=root/'evaluation-plan.json'
 evaluation=json.loads(path.read_text()) if path.exists() else make_evaluation_plan(cells,reports,make_map_suite(1 if plan['smoke'] else 5),plan['references'],smoke=plan['smoke'])
 if path.exists():
  assert evaluation.get('evaluation_revision')==EVALUATION_REVISION,'existing plan has a different evaluation revision; never silently rescore it'
  assert evaluation['evaluation_source_sha256']==sha(Path(__file__).read_bytes()),'evaluation source changed; restore the frozen evaluator before resuming'
 else:
  evaluation.update(evaluation_revision=EVALUATION_REVISION,referee_seconds=REFEREE_SECONDS,guest_limits=dict(decision_seconds=2,memory_mib=256,fuel_per_game=1000000000000),evaluation_source_sha256=sha(Path(__file__).read_bytes()))
  for job in evaluation['jobs']:
   job.pop('job_sha256',None);job.update(evaluation_revision=EVALUATION_REVISION,referee_seconds=REFEREE_SECONDS,evaluation_source_sha256=sha(Path(__file__).read_bytes()));job['job_sha256']=sha(canonical(job).encode())
  write_json(path,evaluation)
 state=dict(status='evaluating',prefix=prefix,smoke=plan['smoke'],total=len(cells),authoring_finished=len(cells),runs={k:{field:r.get(field) for field in ('status','model','track','effective_run_id','submission_quality')} for k,r in reports.items()},errors=[],coordinator_call_id=this,evaluation_games=len(evaluation['jobs']),games_finished=0)
 def save():write_json(root/'status.json',state);results.commit()
 save();rows={};errors=[]
 def execute(jobs):
  for result in recovery_game.starmap(((prefix,j) for j in jobs),order_outputs=False,return_exceptions=True):
   if isinstance(result,BaseException):errors.append(str(result)[:500])
   else:
    rows[result['job_index']]=result
    if len(rows)%25==0:state['games_finished']=len(rows);save()
 execute(evaluation['jobs']);missing_jobs=[j for j in evaluation['jobs'] if j['index'] not in rows]
 if missing_jobs:execute(missing_jobs)
 if len(rows)!=len(evaluation['jobs']):state.update(status='needs_attention',games_finished=len(rows),errors=errors);save();return state
 ordered=[rows[i] for i in range(len(rows))];write_json(root/'games.json',ordered);report=summarize_evaluation(evaluation,cells,ordered)
 report.update(prefix=prefix,evaluation_revision=EVALUATION_REVISION,referee_seconds=REFEREE_SECONDS,manifest_sha256=sha(canonical(plan['manifest']).encode()),allocation_sha256=plan['sha256'],execution_source_sha256=sha(canonical(plan['execution_source']).encode()),references=plan['references'],audit=dict(passed=all(r['audit']['passed'] for r in ordered),actual_games=sum(not r['audit']['virtual'] for r in ordered),virtual_compile_failure_losses=sum(r['audit']['virtual'] for r in ordered),actions=sum(r['audit']['actions'] for r in ordered)),operator_recovery=dict(finalizer_source_sha256=sha(Path(__file__).read_bytes()),authoring_selection_receipt='operator-recoveries/finalizer-authoring-selection.json'))
 selected_ids={item['selected'] for item in selection};cost_by_model={}
 for model in plan['manifest']['models']:
  label=model['label'];part={run:item['recorded_api_usd'] for run,item in costs.items() if item['model_label']==label}
  scored=sum(value for run,value in part.items() if run in selected_ids);total=sum(part.values())
  cost_by_model[label]=dict(scored_completed_calls=scored,excluded_completed_calls=total-scored,all_authoring_completed_calls=total,mean_scored_two_track_pair=scored/plan['manifest']['replicates'])
 report['excluded_attempts']=[attempt for attempt in all_attempts if not attempt['selected']]
 report['all_attempt_api_cost_report']=dict(by_model=cost_by_model,recorded_authoring_total=sum(item['recorded_api_usd'] for item in costs.values()),coverage='lower bound from preserved completed-call receipts; unsaved/replayed or in-flight requests, probes and Modal compute additional; no dollar gate')
 report['peer_ladders']={}
 for track in plan['manifest']['tracks']:
  part=[]
  for row in ordered:
   if row['track']!=track or row['stage']!=(3 if track=='feedback-3r-40m' else 0) or row['status']=='invalid_submission':continue
   job=evaluation['jobs'][row['job_index']];part.append({**row,'a':row['model_label'],'b':job['opponent'].get('model_label',job['opponent']['label'])})
  labels=sorted({r[k] for r in part for k in ('a','b')});report['peer_ladders'][track]=summarize(part,labels,bootstrap=100)
 write_json(root/'report.json',report);state.update(status=report['status'],games_finished=len(rows),audit=report['audit'],errors=[]);save()
 return dict(status=report['status'],games=len(rows),audit=report['audit'])
def retire_authoring_app(identifier):
 import subprocess
 command=[sys.executable,'-m','modal','app','stop',identifier,'--env','staging','--yes']
 response=subprocess.run(command,text=True,capture_output=True)
 output=(response.stdout or '')+(response.stderr or '')
 if response.returncode==0:return dict(app_id=identifier,status='stopped')
 # Modal CLI 1.5.2 uses exit 1 for this confirmed terminal state.
 if response.returncode==1 and output.strip().startswith('App is already stopped.'):
  return dict(app_id=identifier,status='already_stopped')
 raise subprocess.CalledProcessError(response.returncode,command,output=response.stdout,stderr=response.stderr)

@app.local_entrypoint()
def finish(prefix: str, dry_run: bool=False, retire_apps: str=''):
 ready=finalize.remote(prefix,True)
 if dry_run or ready['status']!='ready':print(json.dumps(ready,indent=2));return
 identifiers=[value.strip() for value in retire_apps.split(',') if value.strip()]
 if not identifiers or any(not value.startswith('ap-') for value in identifiers):raise ValueError('Pass the cohort-owned authoring app IDs via --retire-apps; all slots are frozen before stopping them')
 if app.app_id in identifiers:raise ValueError('Do not stop this evaluation app')
 receipts=[retire_authoring_app(identifier) for identifier in identifiers]
 call=finalize.spawn(prefix);print(json.dumps(dict(prefix=prefix,finalizer_call_id=call.object_id,retired_authoring_apps=identifiers,retirement_receipts=receipts)))
