"""Prepare a larger diverse movement-enriched mixed warm-start experiment."""
import gzip,hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.movement_mixed import select_balanced
from polytopia_bot.learning.data import observation_from_dict,example
from polytopia_bot.learning.training import prepare,file_hash
from polytopia_bot.learning.teacher import ScriptedTeacher
from polytopia_bot.learning.interface import canonical,digest
from polytopia_bot.learning.prompt_search import candidate_messages


def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')

def main(run_id):
 root=ROOT/'runs'/run_id;root.mkdir(exist_ok=False)
 parent=json.loads((ROOT/'runs/movement-epoch2-v1/training-run.json').read_text())
 old=json.loads((ROOT/'runs/movement-epoch2-v1/protocol.json').read_text())
 config=parent['config'];manifest=parent['data_manifest'];teacher=ScriptedTeacher()
 assert teacher.teacher_hash==manifest['audit']['teacher_hashes'][0]
 candidate=ROOT/'runs/sft0/main-candidate-sample-v2';replay=ROOT/'runs/sft0/repaired-smoke-sample-v2'
 cm=json.loads((candidate/'sampling.json').read_text());rm=json.loads((replay/'sampling.json').read_text())
 for directory,meta in [(candidate,cm),(replay,rm)]:
  if file_hash(directory/'records.jsonl')!=meta['records_sha256']:raise ValueError('source changed')
 for name,proof in rm['source_provenance'].items():
  report=ROOT/name/'report.json';r=json.loads(report.read_text())
  if file_hash(report)!=proof['report_sha256'] or not r['complete'] or r.get('invalidated'):raise ValueError('unqualified source')
 if any(rm['source_files'].get(k)!=v for k,v in cm['source_files'].items()):raise ValueError('candidate sources differ from repaired corpus')
 moves=select_balanced([r for r in cm['selected'] if r['family']==1],4000,tag='mixed-movement-v1')
 others=[]
 for kind in range(2,11):
  others+=select_balanced([r for r in rm['selected'] if r['family']==kind],650 if kind==3 else 681 if kind==10 else 667,tag='mixed-replay-v1')
 selected=moves+others;wanted={(r['episode_id'],r['decision_index']):r for r in selected}
 if len(wanted)!=10000 or len({r['input_hash'] for r in selected})!=10000:raise ValueError('duplicate selection')
 labels=0;catalogs=Counter();counts=Counter();new_count=0;oldkeys={r['input_hash'] for r in rm['selected']}
 with (root/'records.jsonl').open('x') as out:
  for directory,kindmove in [(candidate,True),(replay,False)]:
   for line in (directory/'records.jsonl').open():
    row=json.loads(line);key=(row['episode_id'],row['decision_index'])
    if (row['selected_command']['kind']==1)!=kindmove or key not in wanted:continue
    meta=wanted.pop(key);obs=observation_from_dict(row['observation']);choice=teacher.select(obs)
    if row['selected_command']!=dict(choice.action.command) or row['teacher_hash']!=choice.teacher_hash or row['teacher_scores']!=choice.scores:raise ValueError('teacher label changed')
    if example(row)[2]!=meta['input_hash'] or row['map_hash']!=meta['map_hash']:raise ValueError('selection input changed')
    out.write(line);labels+=1;counts[str(row['selected_command']['kind'])]+=1
    if kindmove:
     n=len(obs.legal_actions);catalogs[meta['phase']+'|'+('under64' if n<64 else '64to127' if n<128 else '128plus')]+=1
     new_count+=meta['input_hash'] not in oldkeys
    if labels%1000==0:print(json.dumps({'labels_verified':labels}),flush=True)
 if wanted:raise ValueError('selected records missing')
 selection=dict(schema='movement-mixed-selection-v1',movement_count=4000,nonmovement_count=6000,labels_reproduced=labels,
  records_sha256=file_hash(root/'records.jsonl'),source_sampling_sha256={str(d.relative_to(ROOT)):file_hash(d/'sampling.json') for d in [candidate,replay]},
  source_records_sha256={str(d.relative_to(ROOT)):file_hash(d/'records.jsonl') for d in [candidate,replay]},
  selected=selected,maps=len({r['map_hash'] for r in selected}),movement_maps=len({r['map_hash'] for r in moves}),
  movement_episodes=len({r['episode_id'] for r in moves}),new_movement_inputs=new_count,families=dict(counts),movement_phase_catalog=dict(catalogs),
  replay_sources=dict(Counter(r['collection_source'] for r in others)),
  sampling='4000 movement from existing 50k teacher candidate pool; 6000 nonmovement replay balanced across nine families; phase round robin, prefer underrepresented maps, hash tie break; no model prediction selection.')
 write(root/'selection.json',selection)
 print('TOKENIZATION_STARTED',flush=True)
 prepared=prepare(root/'records.jsonl',root/'prepared',config,split_manifest=manifest['split_manifest'],prompt_manifest=manifest['prompt_manifest'])
 if prepared['audit']['unique_records']!=10000:raise ValueError('prepared coverage differs')
 print(json.dumps({'tokenization_complete':True,'max_tokens':prepared['max_tokens']}),flush=True)
 allocation=json.loads((ROOT/'runs/sft0/efficient-positions-allocation-v1.json').read_text());controls={k:[] for k in range(2,11)};positions=[]
 for name in allocation['shard_files']:
  if file_hash(ROOT/name)!=allocation['uploads'][name]:raise ValueError('held-out input changed')
  for line in gzip.open(ROOT/name,'rt'):
   row=json.loads(line)
   if row['teacher_command']['kind']==1:positions.append(row)
   else:controls[row['teacher_command']['kind']].append(row)
 for kind,pool in sorted(controls.items()):
  if len(pool)<100:raise ValueError('control family too small')
  positions+=sorted(pool,key=lambda r:digest(['mixed-controls-v1',r['position_id']]))[:100]
 scoring=[]
 for row in positions:
  if row['board_hash'] not in manifest['split_manifest']['validation']:raise ValueError('heldout map leakage')
  obs=observation_from_dict(row['observation']);choice=teacher.select(obs)
  if dict(choice.action.command)!=row['teacher_command'] or digest(candidate_messages(obs,''))!=row['prompt_digest']:raise ValueError('heldout teacher/prompt changed')
  scoring.append({**{k:row[k] for k in ['position_id','board_hash','episode_id','phase','teacher_command','prompt_digest']},'split':'held_out','source':'teacher_trajectory','scores':[dict(command=s['command'],score=s['score']) for s in choice.scores]})
 positions.sort(key=lambda r:digest(['mixed-eval-order-v1',r['position_id']]))
 for name,rows in [('positions.jsonl.gz',positions),('scoring.jsonl.gz',scoring)]:
  with gzip.open(root/name,'xt') as out:
   for row in rows:out.write(canonical(row)+'\n')
 training={k:v for k,v in old['training'].items() if k!='continuation'}
 # Readiness refers to qualification of the same teacher/environment; the new exhaustive public-label audit is separate.
 training['readiness']={k:v for k,v in training['readiness'].items() if not k.startswith('label_audit')}
 training.update(run_id=run_id+'/training',prepared='/workspace/runs/'+run_id+'/prepared',
  mixed=dict(parent_run='/workspace/runs/movement-epoch2-v1/training/model/run.json',adapter='/workspace/runs/movement-epoch2-v1/training/model/adapter-final',adapter_files=parent['adapter_files'],optimizer_state='fresh',checkpoint_steps=[40,79,118,157],selection_sha256=file_hash(root/'selection.json')))
 upload_files=['selection.json','positions.jsonl.gz','scoring.jsonl.gz','prepared/manifest.json','prepared/tokens.jsonl']
 protocol=dict(schema='movement-enriched-mixed-v1',run_id=run_id,training=training,config=config,prompt_manifest=old['prompt_manifest'],
  positions_sha256=file_hash(root/'positions.jsonl.gz'),scoring_sha256=file_hash(root/'scoring.jsonl.gz'),
  upload_files=['protocol.json',*upload_files],uploads={name:file_hash(root/name) for name in upload_files},
  checkpoint_steps=[0,40,79,118,157],evaluation_count=len(positions),control_ids=[r['position_id'] for r in positions if r['teacher_command']['kind']!=1],
  batch_size=64,evaluation_seconds_limit=1200,training_selection={k:v for k,v in selection.items() if k!='selected'},
  primary='Tie-aware agreement on held-out movement-required positions; nonmovement exact and tie-aware overall and per action family. Paired map-bootstrap changes versus fresh same-order baseline.',
  retention_rule='Flag overall nonmovement tie-aware losses greater than 3 percentage points or any action-family loss greater than 10 points; show uncertainty and exact scores too. Diagnostic flags, not automatic promotion.',
  optimizer='Fresh AdamW and one-pass cosine schedule for the changed mixture; intentionally do not reuse the previous corpus optimizer state. Same LR 1e-4, batch64, prompt, model, and LoRA settings.',
  scope='One pass over 10000 distinct inputs, 4000 movement and 6000 nonmovement replay. No RL, final-test access, full games, or automatic checkpoint promotion.')
 write(root/'protocol.json',protocol)
 print(json.dumps({'prepared':run_id,'selection':protocol['training_selection'],'evaluation_count':len(positions)}),flush=True)

if __name__=='__main__':main(sys.argv[1])
