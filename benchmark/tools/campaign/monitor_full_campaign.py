from pathlib import Path
import json,time,datetime,os,subprocess
root=Path.cwd();campaign=Path(json.loads((root/'artifacts/harbor-campaigns/latest-full-launch.json').read_text())['directory'])
last=None
while True:
 names=[campaign.name+'-local',campaign.name+'-daytona']
 f=campaign/'infrastructure-retry.json'
 if f.exists():names.append(json.loads(f.read_text())['replacement_job'])
 report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':{}}
 for name in names:
  j=root/'artifacts/harbor-runs'/name;f=j/'result.json'
  d=json.loads(f.read_text()) if f.exists() else {}
  report['jobs'][name]={'finished_at':d.get('finished_at'),'stats':{k:v for k,v in d.get('stats',{}).items() if k in ['n_completed_trials','n_errored_trials','n_running_trials','n_pending_trials','n_cancelled_trials']},'coding_complete':len(list(j.glob('*/agent/authoring.json'))),'reward_files':len(list(j.glob('*/verifier/reward.json'))),'errors':{str(e.relative_to(j)):e.read_text()[-700:] for e in j.glob('*/exception.txt')}}
 temp=campaign/'monitor-status.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(campaign/'monitor-status.json')
 stable=json.dumps(report['jobs'],sort_keys=True)
 if stable!=last:
  print(json.dumps(report),flush=True);last=stable
 if all(j['finished_at'] for j in report['jobs'].values()):break
 time.sleep(30)
