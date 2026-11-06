"""Describe the frozen mixed corpus without altering its training protocol."""
import json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from polytopia_bot.learning.training import file_hash

def main(root):
 root=Path(root);p=json.loads((root/'protocol.json').read_text());s=json.loads((root/'selection.json').read_text())
 if file_hash(root/'records.jsonl')!=s['records_sha256'] or file_hash(root/'selection.json')!=p['training']['mixed']['selection_sha256']:raise ValueError('source changed')
 meta={(r['episode_id'],r['decision_index']):r for r in s['selected']}
 maps=Counter();move_maps=Counter();strict_maps=set();phases=Counter();strict_phases=Counter();strict=0
 for line in (root/'records.jsonl').open():
  r=json.loads(line);maps[r['map_hash']]+=1
  if r['selected_command']['kind']!=1:continue
  phase=meta[r['episode_id'],r['decision_index']]['phase'];phases[phase]+=1;move_maps[r['map_hash']]+=1
  best=max(x['score'] for x in r['teacher_scores']);top=[x for x in r['teacher_scores'] if x['score']==best]
  if all(x['command']['kind']==1 for x in top):strict+=1;strict_phases[phase]+=1;strict_maps.add(r['map_hash'])
 out=dict(records_sha256=s['records_sha256'],strict_movement_count=strict,strict_movement_maps=len(strict_maps),movement_phases=dict(phases),strict_movement_phases=dict(strict_phases),
  examples_per_map_histogram=dict(Counter(maps.values())),movement_per_map_histogram=dict(Counter(move_maps.values())))
 (root/'data-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main(sys.argv[1])
