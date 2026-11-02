from pathlib import Path
import sys,json,os,time,subprocess,datetime
root=Path.cwd();sys.path[:0]=[str(root),str(root/'src')]
from scripts.check_skirmish_daytona import credentials
campaign=Path(json.loads((root/'artifacts/harbor-campaigns/latest-full-launch.json').read_text())['directory']);state_path=campaign/'grading-recovery-jobs.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
def save():
 temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
while True:
 if not (campaign/'recovery-qualification-passed.json').exists():raise RuntimeError('Recovery has not passed qualification')
 job_names=[campaign.name+'-local',campaign.name+'-daytona']
 retry=campaign/'infrastructure-retry.json'
 if retry.exists():job_names.append(json.loads(retry.read_text())['replacement_job'])
 accepted_path=campaign/'accepted-results.json'
 accepted_now=json.loads(accepted_path.read_text()).get('accepted',{}) if accepted_path.exists() else {}
 for info in state.values():
  original=str((root/info['original_trial']).resolve())
  if original in accepted_now and not info.get('stopped_after_validated_export'):
   try:
    command=subprocess.check_output(['ps','-p',str(info['pid']),'-o','args='],text=True)
    if info['job_name'] in command:os.kill(info['pid'],2)
   except (ProcessLookupError,subprocess.CalledProcessError):pass
   info['stopped_after_validated_export']=True;info['cleanup_reason']='Complete validated result already preserved; stop redundant grading or large-file export.';save();print(json.dumps(dict(recovery_cleanup=info['job_name'],reason=info['cleanup_reason'])),flush=True)
 for job_name in job_names:
  job=root/'artifacts/harbor-runs'/job_name
  for exception in job.glob('*/exception.txt'):
   trial=exception.parent;key=str(trial.relative_to(root));author_path=trial/'agent/authoring.json'
   accepted_path=campaign/'accepted-results.json'
   accepted=json.loads(accepted_path.read_text()).get('accepted',{}) if accepted_path.exists() else {}
   if str(trial.resolve()) in accepted:continue
   if key in state or not author_path.exists():continue
   author=json.loads(author_path.read_text());backend='local' if author['execution_backend']=='docker' else 'daytona';capture=author['capture'];source=trial/'artifacts/submission/main.rs'
   if not source.exists() or capture.get('error'):continue
   import hashlib
   if hashlib.sha256(source.read_bytes()).hexdigest()!=capture['source_sha256']:raise RuntimeError('Frozen source hash mismatch')
   candidates=[]
   for meta in (campaign/'grading-checkpoints'/backend).glob('*/metadata.json'):
    d=json.loads(meta.read_text())
    if d['source_sha256']==capture['source_sha256'] and (meta.parent/'checkpoint.tar').exists():candidates.append((d['done'],meta.parent))
   for meta in (campaign/'compact-checkpoints'/backend).glob('*/metadata.json'):
    d=json.loads(meta.read_text())
    if d['source_sha256']==capture['source_sha256'] and (meta.parent/'replay.json.gz').exists():candidates.append((d['games'],meta.parent))
   checkpoint=max(candidates,key=lambda x:x[0])[1] if candidates else campaign/'grading-checkpoints'/backend/trial.name
   base=json.loads((campaign/('local.json' if backend=='local' else 'daytona.json')).read_text());config=dict(base);name=campaign.name+'-recover-'+backend+'-'+trial.name
   config.update(job_name=name,n_attempts=1,n_concurrent_trials=1);a=json.loads((trial/'config.json').read_text())['agent'];a['import_path']='grading_recovery:FrozenSourceAgent';a['kwargs']['original_trial']=str(trial);config['agents']=[a]
   config['environment']['import_path']='grading_recovery:RecoveryDockerEnvironment' if backend=='local' else 'grading_recovery:RecoveryDaytonaEnvironment';config['environment'].setdefault('kwargs',{})['recovery_checkpoint']=str(checkpoint);config['environment']['kwargs']['recovery_source_sha256']=capture['source_sha256'];config['verifier']['override_timeout_sec']=43200
   config_path=campaign/(name+'.json');config_path.write_text(json.dumps(config,indent=2)+'\n');auth=credentials();env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'artifacts'),str(root/'src'),str(root)]),DOCKER_CONTEXT='colima-polytopia-skirmish',OPENROUTER_API_KEY=(Path.home()/'.config/polytopia-skirmish/openrouter.key').read_text().strip(),DAYTONA_API_URL=auth['api_url'],DAYTONA_ORGANIZATION_ID=auth['organization_id'],DAYTONA_TARGET='us',LITELLM_LOCAL_MODEL_COST_MAP='True');env['DAYTONA_API_KEY' if auth.get('api_key') else 'DAYTONA_JWT_TOKEN']=auth.get('api_key') or auth['jwt_token']
   with (campaign/(name+'.log')).open('w') as log:p=subprocess.Popen([str(root/'artifacts/harbor-runtime/bin/harbor'),'run','--config',str(config_path)],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   sleep_guard=subprocess.Popen(['/usr/bin/caffeinate','-dimsu','-w',str(p.pid)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
   info=dict(original_trial=key,job_name=name,pid=p.pid,caffeinate_pid=sleep_guard.pid,backend=backend,checkpoint=str(checkpoint),source_sha256=capture['source_sha256'],additional_authoring_seconds=0,additional_model_calls=0,verifier_guard_seconds=43200,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=exception.read_text()[-500:]);state[key]=info;save();print(json.dumps(info),flush=True)
 time.sleep(30)
