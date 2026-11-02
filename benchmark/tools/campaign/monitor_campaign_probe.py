from pathlib import Path
import json,subprocess,concurrent.futures,datetime
p=Path('artifacts/harbor-campaigns/continuous-split-full-20261031T075011Z');a=json.loads((p/'accepted-results.json').read_text());r=json.loads((p/'grading-recovery-jobs.json').read_text());out={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted':a['validated_evaluations'],'games':a['final_outcomes'],'active_recoveries':[],'errors':[]}
for key,v in r.items():
 if v.get('stopped_after_validated_export'):continue
 proc=subprocess.run(['ps','-p',str(v['pid']),'-o','args='],capture_output=True,text=True).stdout
 d=Path('artifacts/harbor-runs')/v['job_name'];trials=list(d.glob('task__*'));name=trials[0].name if len(trials)==1 else None;m=p/'compact-checkpoints/daytona'/str(name)/'metadata.json'
 out['active_recoveries'].append({'origin':key.split('/')[-1],'running':v['job_name'] in proc,'games':json.loads(m.read_text()).get('games') if m.exists() else None})
 for e in d.glob('*/exception.txt'):out['errors'].append({'job':v['job_name'],'detail':e.read_text()[-300:]})
names=subprocess.check_output(['docker','--context','colima-polytopia-skirmish','ps','--format','{{.Names}}'],text=True).splitlines();names=[n for n in names if '__verifier__trial-main-' in n]
def check(n):
 r=subprocess.run(['docker','--context','colima-polytopia-skirmish','exec',n,'python','-c',"from pathlib import Path;print(len(list(Path('/results/games').glob('*/summary.json'))))"],capture_output=True,text=True,timeout=20);return [n.split('__verifier__')[0],r.stdout.strip()]
out['local_progress']=list(concurrent.futures.ThreadPoolExecutor(4).map(check,names));(p/'foreground-probes.jsonl').open('a').write(json.dumps(out)+'\n');print(json.dumps(dict(at=out['at'],accepted=out['accepted'],local=[x[1] for x in out['local_progress']],cloud=[x['games'] for x in out['active_recoveries']],warnings=out['errors']+[x for x in out['active_recoveries'] if not x['running']])))
