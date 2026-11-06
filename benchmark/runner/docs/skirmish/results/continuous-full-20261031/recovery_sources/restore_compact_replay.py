"""Reconstruct byte-identical audited logs from recorded actions, never rerun bots."""
from pathlib import Path
import json,gzip,hashlib,shutil,sys
from dataclasses import asdict
from polytopia_bot.skirmish.common import canonical,sha,write_json
from polytopia_bot.skirmish.game import reset_map
from polytopia_bot.studio.environment import RustEnvironment
from polytopia_bot.skirmish.local_referee import file_sha

def restore(checkpoint,expected_source_sha256):
 payload=json.loads(gzip.decompress(Path(checkpoint).read_bytes()))
 assert payload['schema']=='skirmish-compact-replay-v1' and payload['source_sha256']==expected_source_sha256
 for game in payload['games']:
  row=game['summary'];index=row['job_index'];folder=Path('/results/games')/f'{index:06d}';folder.mkdir(parents=True,exist_ok=True)
  if row['audit'].get('virtual'):
   write_json(folder/'summary.json',row);continue
  relative=Path(row['trace_relative']);assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[0]=='games'
  compressed=Path('/results')/relative;plain=compressed.with_suffix('') if compressed.suffix=='.gz' else compressed;plain.parent.mkdir(parents=True,exist_ok=True);private=plain.with_suffix('.private.jsonl')
  cfg=row['evaluation'];env=RustEnvironment(size=cfg.get('size',14),max_rounds=cfg.get('max_rounds') if cfg.get('exact_rounds') else None,max_actions=cfg.get('max_actions',10000));chain='0'*64
  try:
   env.restore(row['initial_snapshot'].encode()) if row.get('initial_snapshot') else reset_map(env,row['seed'],row['map_type'])
   with plain.open('w') as public_log,private.open('w') as private_log:
    for action in game['actions']:
     obs=env.observe(int(env.state.current_player));entry=dict(player=obs.player,version=obs.state_version,action_id=action,turn=obs.visible_state['turn']);chain=sha((chain+canonical(entry)).encode());private_log.write(canonical(dict(**entry,chain=chain))+'\n')
     if obs.player==row['seat']:public_log.write(canonical(dict(observation=asdict(obs),action_id=action))+'\n')
     env.step(action_id=action,expected_state_version=obs.state_version)
   assert len(game['actions'])==row['actions']==row['audit']['actions']
   assert chain==row['action_chain_sha256'] and file_sha(private)==row['private_trace_sha256'] and file_sha(plain)==row['trace_sha256']
   assert int(env.state.turn)==row['turns']
   if row['status']=='native':assert bool(env.state.terminated) and list(env.games.batch.terminal_rewards()[0])[row['winner']-1]>0
   elif row['status']=='draw':assert row['winner'] is None and row['reason'] in ('engine_truncation','action_limit')
   elif row['status']=='forfeit':assert not bool(env.state.terminated) and row['winner']==3-row['fault_player']
   if compressed!=plain:
    with plain.open('rb') as src,compressed.open('wb') as dst:
     with gzip.GzipFile(fileobj=dst,mode='wb',compresslevel=1,mtime=0) as out:shutil.copyfileobj(src,out)
    plain.unlink()
   write_json(folder/'summary.json',row)
  finally:env.close()
 print(json.dumps(dict(restored_games=len(payload['games']),all_original_public_and_private_hashes_matched=True,additional_model_calls=0)),flush=True)
if __name__=='__main__':restore(sys.argv[1],sys.argv[2])
