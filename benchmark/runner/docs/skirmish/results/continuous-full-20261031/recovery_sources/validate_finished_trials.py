from pathlib import Path
import sys,json,time,hashlib,importlib.util,concurrent.futures,datetime
root=Path.cwd();sys.path.insert(0,str(root/'src'));path=root/'scripts/summarize_skirmish_harbor.py';spec=importlib.util.spec_from_file_location('frozen_native_summary',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
campaign=Path(json.loads((root/'artifacts/harbor-campaigns/latest-full-launch.json').read_text())['directory']);base=campaign/'validated-trials';base.mkdir(exist_ok=True);pending={};seen=set()
def validate(trial):
 key=hashlib.sha256(str(trial).encode()).hexdigest()[:16];view=base/key;view.mkdir(exist_ok=True);link=view/trial.name
 if not link.exists():link.symlink_to(trial.resolve(),target_is_directory=True)
 (view/'result.json').write_text(json.dumps(dict(kind='derived_single_trial_validation_view',native_source_trial=str(trial),n_total_trials=1,stats=dict(n_errored_trials=0,n_running_trials=0)))+'\n')
 summary=module.summarize(view);rows=json.loads((trial/'verifier/games.json').read_text());plan=json.loads((trial/'verifier/evaluation-plan.json').read_text());digests={}
 for row in rows:
  if row['audit'].get('virtual'):continue
  relative=Path(row['trace_relative']).relative_to('games');public=trial/'verifier/traces'/relative;plain=public.with_suffix('') if public.suffix=='.gz' else public;private=plain.with_suffix('.private.jsonl')
  with private.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['private_trace_sha256']
  for p in [public,private]:
   with p.open('rb') as f:digests[str(p.relative_to(trial))]=hashlib.file_digest(f,'sha256').hexdigest()
 schedule=[(j['opponent'],j['map_type'],j['seed'],j['seat'],j['evaluation']) for j in plan['jobs']]
 receipt=dict(passed=True,trial=str(trial),validated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary=summary['trials'][0],schedule_sha256=hashlib.sha256(module.canonical(schedule).encode()).hexdigest(),exported_file_sha256=digests)
 (view/'validated.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(validated_trial=str(trial),games=len(rows),trace_files=len(digests))),flush=True);return key
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 while True:
  for job in (root/'artifacts/harbor-runs').glob(campaign.name+'-*'):
   for result in job.glob('*/result.json'):
    trial=result.parent;key=str(trial)
    if key in seen or key in pending:continue
    try:d=json.loads(result.read_text())
    except (json.JSONDecodeError,OSError):continue
    if d.get('finished_at') is None or d.get('exception_info') is not None or not (trial/'verifier/reward.json').exists():continue
    view_key=hashlib.sha256(str(trial).encode()).hexdigest()[:16]
    if (base/view_key/'validated.json').exists():seen.add(key);continue
    pending[key]=pool.submit(validate,trial)
  for key,future in list(pending.items()):
   if future.done():
    try:future.result();seen.add(key)
    except Exception as e:print(json.dumps(dict(validation_error=key,error=type(e).__name__,detail=str(e)[:300])),flush=True);seen.add(key)
    pending.pop(key)
  time.sleep(20)
