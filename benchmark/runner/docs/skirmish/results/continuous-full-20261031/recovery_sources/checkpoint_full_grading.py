from pathlib import Path
import sys,json,time,datetime,subprocess,shlex,uuid,tarfile,os,threading,shutil
from concurrent.futures import ThreadPoolExecutor,as_completed
root=Path.cwd();sys.path[:0]=[str(root),str(root/'src')]
from scripts.check_skirmish_daytona import credentials
from daytona import Daytona,DaytonaConfig
campaign=Path(json.loads((root/'artifacts/harbor-campaigns/latest-full-launch.json').read_text())['directory']);backups=campaign/'grading-checkpoints';backups.mkdir(exist_ok=True)
D=None
remote='''from pathlib import Path
import json,tarfile,sys,time
base=Path('/results');rows=sorted(base.glob('games/*/summary.json'));start=int(sys.argv[2]) if len(sys.argv)>2 else 0;selected=rows[start:]
if not rows: print(json.dumps(dict(done=0)));raise SystemExit
plan=json.loads((base/'evaluation-plan.json').read_text());capture=json.loads(Path('/verify/cutoffs/stage-0/capture.json').read_text());info=dict(done=len(rows),source_sha256=capture['source_sha256'],created_unix=time.time(),profile=plan['profile'],expected_games=len(plan['jobs']))
size=sum(f.stat().st_size for p in selected for f in p.parent.rglob('*') if f.is_file())
import shutil
if size+512*1024*1024>shutil.disk_usage('/tmp').free:raise RuntimeError('insufficient temporary space for checkpoint')
with tarfile.open(sys.argv[1],'w') as archive:
 for p in selected: archive.add(p.parent,arcname=str(p.parent.relative_to(base)))
print(json.dumps(info))
'''
def cloud(s):
 session=(s.labels or {}).get('harbor.session_id','')
 if '__verifier__' not in session:return None
 trial=session.split('__verifier__')[0];folder=backups/'daytona'/trial;folder.mkdir(parents=True,exist_ok=True);tmp='/tmp/skirmish-checkpoint-'+uuid.uuid4().hex+'.tar'
 try:
  previous=json.loads((folder/'metadata.json').read_text())['done'] if (folder/'metadata.json').exists() and (folder/'checkpoint.tar').exists() else 0
  result=s.process.exec('python -c '+shlex.quote(remote)+' '+shlex.quote(tmp)+' '+str(previous),timeout=120)
  if result.exit_code:raise RuntimeError(result.result[-200:])
  info=json.loads(result.result.strip().splitlines()[-1]);info.update(backend='daytona',trial=trial,sandbox_id=s.id)
  if info['done']:
   cancel=threading.Event();timer=threading.Timer(180,cancel.set);timer.start();delta=folder/'delta.partial'
   try:
    with delta.open('wb') as target:
     for chunk in s.fs.download_file_stream(tmp,timeout=90,cancel_event=cancel):target.write(chunk)
   finally:timer.cancel()
   partial=folder/'checkpoint.partial'
   if previous:
    shutil.copyfile(folder/'checkpoint.tar',partial)
    with tarfile.open(partial,'a') as output,tarfile.open(delta) as incoming:
     for member in incoming:output.addfile(member,incoming.extractfile(member) if member.isfile() else None)
    delta.unlink()
   else:delta.replace(partial)
   partial.replace(folder/'checkpoint.tar');(folder/'metadata.json').write_text(json.dumps(info,indent=2)+'\n')
   s.process.exec('rm -f '+shlex.quote(tmp),timeout=15)
  return info
 except Exception as e:return dict(backend='daytona',trial=trial,error=type(e).__name__,detail=str(e).split('HTTP request headers:')[0][:300])
def local(ident):
 cmd=['docker','--context','colima-polytopia-skirmish'];name=subprocess.check_output(cmd+['inspect','--format','{{.Name}}',ident],text=True).strip().lstrip('/');trial=name.split('__verifier__')[0];folder=backups/'local'/trial;folder.mkdir(parents=True,exist_ok=True);tmp='/tmp/skirmish-checkpoint-'+uuid.uuid4().hex+'.tar'
 try:
  result=subprocess.run(cmd+['exec',ident,'python','-c',remote,tmp],capture_output=True,text=True,timeout=120,check=True);info=json.loads(result.stdout.strip().splitlines()[-1]);info.update(backend='local',trial=trial,container_id=ident)
  if info['done']:
   partial=folder/'checkpoint.partial';subprocess.run(cmd+['cp',ident+':'+tmp,str(partial)],capture_output=True,check=True,timeout=120);partial.replace(folder/'checkpoint.tar');(folder/'metadata.json').write_text(json.dumps(info,indent=2)+'\n');subprocess.run(cmd+['exec',ident,'rm','-f',tmp],capture_output=True,timeout=15)
  return info
 except Exception as e:return dict(backend='local',trial=trial,error=type(e).__name__,detail=str(e).split('HTTP request headers:')[0][:300])
if __name__=='__main__':
 D=Daytona(DaytonaConfig(**credentials(),otel_enabled=False,use_deprecated_polling=True))
 while True:
  try:
   sandboxes=list(D.list(request_timeout=20));ids=subprocess.check_output(['docker','--context','colima-polytopia-skirmish','ps','-q','--filter','label=com.docker.compose.service=main'],text=True).split();results=[]
   with ThreadPoolExecutor(max_workers=4) as pool:
    jobs=[pool.submit(local,i) for i in ids]+[pool.submit(cloud,s) for s in sandboxes if '__verifier__' in (s.labels or {}).get('harbor.session_id','')]
    for f in as_completed(jobs):
     info=f.result()
     if info:results.append(info);print(json.dumps(info),flush=True)
   (backups/'latest.json').write_text(json.dumps(dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=results),indent=2)+'\n')
  except Exception as e:print(json.dumps(dict(cycle_error=type(e).__name__,detail=str(e).split('HTTP request headers:')[0][:300])),flush=True)
  time.sleep(300)
