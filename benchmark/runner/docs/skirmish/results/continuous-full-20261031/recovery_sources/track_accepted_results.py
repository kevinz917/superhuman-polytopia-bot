from pathlib import Path
import sys,json,datetime,time,hashlib,shutil
from collections import Counter
root=Path.cwd();sys.path.insert(0,str(root/'src'))
from polytopia_bot.skirmish.common import canonical
from polytopia_bot.skirmish.continuous import evaluation_jobs
campaign=Path(json.loads((root/'artifacts/harbor-campaigns/latest-full-launch.json').read_text())['directory']);registry=campaign/'accepted-results.json';accepted=json.loads(registry.read_text())['accepted'] if registry.exists() else {};archive=campaign/'accepted-receipts';archive.mkdir(exist_ok=True)
pkg=root/'artifacts/harbor-packages/split-local-release-v6-20261031';manifest=json.loads((pkg/'manifest.json').read_text());maps=json.loads((pkg/'maps.json').read_text());jobs=evaluation_jobs(manifest,maps);expected=hashlib.sha256(canonical([(j['opponent'],j['map_type'],j['seed'],j['seat'],j['evaluation']) for j in jobs]).encode()).hexdigest()
def origin(trial):
 trial=Path(trial)
 for _ in range(10):
  f=trial/'agent/grading-recovery.json'
  if not f.exists():return trial
  trial=Path(json.loads(f.read_text())['original_trial'])
 raise ValueError('recovery provenance depth')
while True:
 receipts=[]
 for kind,directory in [('full_export','validated-trials'),('lossless_replay','validated-compact-trials')]:
  for f in (campaign/directory).glob('*/validated.json'):
   try:d=json.loads(f.read_text())
   except (json.JSONDecodeError,OSError):continue
   if not d.get('passed'):continue
   original=origin(d.get('original_trial',d.get('trial'))).resolve();receipts.append((d['validated_at'],str(original),kind,f,d))
 for timestamp,key,kind,f,d in sorted(receipts):
  if key in accepted:continue
  assert d['schedule_sha256']==expected,'held-out schedule mismatch'
  author=json.loads((Path(key)/'agent/authoring.json').read_text());assert author['timing']['coding_seconds']<=2400
  ident=hashlib.sha256(key.encode()).hexdigest()[:16];destination=archive/(ident+'.json');shutil.copyfile(f,destination)
  accepted[key]=dict(receipt=str(destination),validation_kind=kind,model=d['summary']['model'],graded_trial=d.get('graded_trial',d.get('trial')),accepted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_source_sha256=author['capture']['source_sha256'])
 counts=dict(Counter(v['model'] for v in accepted.values()));report=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_rule='First complete validated native grading export for each frozen authored submission; no reward-based selection.',validated_evaluations=len(accepted),required_evaluations=60,final_outcomes=sum(json.loads(Path(v['receipt']).read_text())['summary']['games'] for v in accepted.values()),models=counts,validation_kinds=dict(Counter(v['validation_kind'] for v in accepted.values())),accepted=accepted)
 assert all(n<=5 for n in counts.values()),'duplicate authored slot included'
 temp=registry.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(registry)
 if len(accepted)==60:
  assert len(counts)==12 and set(counts.values())=={5};print('All 60 authored submissions have validated results.',flush=True);break
 time.sleep(20)
