from pathlib import Path
import sys,json,time,datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
root=Path.cwd();sys.path[:0]=[str(root),str(root/'artifacts'),str(root/'src')]
from compact_grading_checkpoint import client,export
D=client();campaign=Path(json.loads((root/'artifacts/harbor-campaigns/latest-full-launch.json').read_text())['directory'])
def one(s):
 trial=(s.labels or {})['harbor.session_id'].split('__verifier__')[0];folder=campaign/'compact-checkpoints/daytona'/trial
 try:
  info=export(s,folder/'replay.json.gz');info.update(backend='daytona',trial=trial,sandbox_id=s.id,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(folder/'metadata.json').write_text(json.dumps(info,indent=2)+'\n');return info
 except Exception as e:return dict(trial=trial,sandbox_id=s.id,error=type(e).__name__)
while True:
 try:
  own={}
  for job in (root/'artifacts/harbor-runs').glob(campaign.name+'-*'):
   for cfg in job.glob('*/config.json'):own[cfg.parent.name.casefold()]=cfg.parent
  accepted_file=campaign/'accepted-results.json';accepted=json.loads(accepted_file.read_text()).get('accepted',{}) if accepted_file.exists() else {}
  S=[]
  for sandbox in D.list(request_timeout=20):
   session=(sandbox.labels or {}).get('harbor.session_id','')
   if '__verifier__' not in session:continue
   trial=own.get(session.split('__verifier__')[0].casefold())
   if trial is None:continue
   original=trial
   for _ in range(10):
    link=original/'agent/grading-recovery.json'
    if not link.exists():break
    original=Path(json.loads(link.read_text())['original_trial'])
   if str(original.resolve()) not in accepted:S.append(sandbox)
  with ThreadPoolExecutor(max_workers=8) as pool:
   for f in as_completed([pool.submit(one,s) for s in S]):print(json.dumps(f.result()),flush=True)
 except Exception as e:print(json.dumps(dict(cycle_error=type(e).__name__)),flush=True)
 time.sleep(120)
