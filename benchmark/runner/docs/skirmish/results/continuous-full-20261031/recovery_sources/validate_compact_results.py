from pathlib import Path
import sys,json,gzip,hashlib,base64,time,datetime
root=Path.cwd();sys.path.insert(0,str(root/'src'))
from polytopia_bot.skirmish.common import canonical
from polytopia_bot.skirmish.continuous import rewards,SCHEMA,CODING_SECONDS
campaign=Path(json.loads((root/'artifacts/harbor-campaigns/latest-full-launch.json').read_text())['directory']);output=campaign/'validated-compact-trials';output.mkdir(exist_ok=True);checked={}
def original_trial(trial):
 visited=set()
 while (trial/'agent/grading-recovery.json').exists():
  if str(trial) in visited:raise ValueError('cyclic grading recovery provenance')
  visited.add(str(trial));trial=Path(json.loads((trial/'agent/grading-recovery.json').read_text())['original_trial'])
 return trial

def validate(folder):
 data=(folder/'replay.json.gz').read_bytes();digest=hashlib.sha256(data).hexdigest()
 if checked.get(str(folder))==digest:return
 payload=json.loads(gzip.decompress(data));checked[str(folder)]=digest
 if 'final' not in payload:return
 final=payload['final'];rows=final['games.json'];plan=final['evaluation-plan.json'];summary=final['summary.json'];meta=json.loads((folder/'metadata.json').read_text());trial_name=meta['trial'];matches=list((root/'artifacts/harbor-runs').glob(campaign.name+'-*/'+trial_name))
 if len(matches)!=1:raise ValueError('ambiguous native trial provenance')
 trial=matches[0];origin=original_trial(trial);author=json.loads((origin/'agent/authoring.json').read_text());protocol=json.loads((origin/'agent/protocol.json').read_text());source=(origin/'artifacts/submission/main.rs').read_bytes();source_sha=hashlib.sha256(source).hexdigest()
 assert source_sha==author['capture']['source_sha256']==payload['source_sha256']==summary['submission']['source_sha256']
 assert author['coding_budget_seconds']==CODING_SECONDS and author['timing']['coding_seconds']<=CODING_SECONDS and protocol['schema']==SCHEMA
 assert plan['profile']==summary['profile']=='standard' and len(rows)==len(plan['jobs'])==len(payload['games'])==360
 assert {r['job_index'] for r in rows}==set(range(360))
 assert final['submission.json']==summary['submission']
 build=json.loads((root/'artifacts/harbor-packages/split-daytona-release-v6-20261031/build.json').read_text())
 assert payload['referee_source_sha256']==build['source_files']['src/polytopia_bot/skirmish/local_referee.py']
 records={g['summary']['job_index']:g for g in payload['games']};private_count=0
 for row in rows:
  planned=plan['jobs'][row['job_index']];assert row['job_sha256']==planned['job_sha256']==hashlib.sha256(canonical({k:v for k,v in planned.items() if k!='job_sha256'}).encode()).hexdigest()
  assert row['audit']['passed'] and row['result']!='unresolved' and row['evaluation']==planned['evaluation'];record=records[row['job_index']];assert record['summary']==row
  if row['audit'].get('virtual'):continue
  entries=record.get('entries')
  if entries is None:raise ValueError('complete action-entry fields not yet exported')
  assert len(entries)==row['actions']==row['audit']['actions'];chain='0'*64;private_hash=hashlib.sha256()
  for p,v,t,action in entries:
   entry=dict(player=p,version=v,turn=t,action_id=action);chain=hashlib.sha256((chain+canonical(entry)).encode()).hexdigest();private_hash.update((canonical(dict(**entry,chain=chain))+'\n').encode())
  assert private_hash.hexdigest()==row['private_trace_sha256'] and chain==row['action_chain_sha256'];private_count+=1
 reward=rewards(rows,360);reward['evaluation_is_demo']=0;assert reward==final['reward.json']==summary['rewards'] and summary['score']==100*reward['reward']
 wasm=base64.b64decode(final['wasm_base64']) if 'wasm_base64' in final else None
 if summary['submission']['returncode']==0:
  assert wasm and hashlib.sha256(wasm).hexdigest()==summary['submission']['wasm_sha256']
 schedule=[(j['opponent'],j['map_type'],j['seed'],j['seat'],j['evaluation']) for j in plan['jobs']]
 key=hashlib.sha256((str(origin)+'|'+str(trial)+'|'+digest).encode()).hexdigest()[:24];destination=output/key;destination.mkdir(exist_ok=True);(destination/'replay.json.gz').write_bytes(data)
 config=json.loads((origin/'config.json').read_text());model=config['agent']['model_name'];hero=[r for r in rows if r['opponent_label']=='hero-1v1'];practices=list((origin/'artifacts/practice').glob('*/report.json'))
 trial_summary=dict(model=model,trial=origin.name,score=summary['score'],games=360,profile='standard',quality=summary['submission']['submission_quality'],compile_returncode=summary['submission']['returncode'],champion=dict(wins=sum(r['result']=='win' for r in hero),draws=sum(r['result']=='draw' for r in hero),losses=sum(r['result']=='loss' for r in hero)),practice_matches=len(practices),coding_budget_seconds=CODING_SECONDS,timing=author['timing'],usage=author['usage'],inference=protocol['inference'],replay_actions=summary['replay_actions'],rewards=reward)
 receipt=dict(passed=True,original_trial=str(origin),graded_trial=str(trial),validated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=source_sha,summary=trial_summary,schedule_sha256=hashlib.sha256(canonical(schedule).encode()).hexdigest(),compact_blob_sha256=digest,private_trace_hashes_verified=private_count,native_public_observation_audits_passed=True,public_observation_encoding='Lossless native-engine replay; original public SHA256 retained; decoder qualified against 36 complete original traces',raw_public_trace_bytes_rehashed_locally=False,completed_call_cost_usd=author.get('completed_call_cost_usd'))
 (destination/'validated.json').write_text(json.dumps(receipt,indent=2)+'\n');(destination/'native-final-receipts.json').write_text(json.dumps({k:v for k,v in final.items() if k!='wasm_base64'},indent=2)+'\n')
 if wasm:(destination/'bot.wasm').write_bytes(wasm)
 print(json.dumps(dict(validated_compact_trial=str(origin),graded_trial=str(trial),games=360,private_trace_hashes_verified=private_count)),flush=True)

if __name__=='__main__':
 while True:
  for folder in (campaign/'compact-checkpoints').glob('*/*'):
   if not (folder/'metadata.json').exists() or not (folder/'replay.json.gz').exists():continue
   try:validate(folder)
   except (OSError,json.JSONDecodeError):continue
   except Exception as e:print(json.dumps(dict(compact_validation_error=str(folder),error=type(e).__name__,detail=str(e)[:300])),flush=True)
  time.sleep(20)
