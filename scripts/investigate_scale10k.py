"""Offline forensic audit of the 40k experiment's 10k checkpoint dip."""
import argparse,gzip,hashlib,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from polytopia_bot.learning.movement_diagnostic import score_choice
from polytopia_bot.learning.movement_mixed import score_nonmovement
from polytopia_bot.learning.efficient_training import EffectiveBatchLengthSampler
from polytopia_bot.learning.scale_training import batch_ranges,BOUNDARIES
from polytopia_bot.learning.interface import digest
from summarize_movement_continuation import paired_change

RUN=ROOT/'runs/movement-scale40k-v1';OUT=RUN/'investigation-10k'
def read(p):return json.loads(Path(p).read_text())
def gz(p):
 with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def save(name,value):
 OUT.mkdir(exist_ok=True);(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def predictions():
 meta=list(gz(RUN/'scoring.jsonl.gz'));positions={r['position_id']:r for r in gz(RUN/'positions.jsonl.gz')}
 reports={str(step):read(RUN/f'report-{step}.json') for step in [0,157,314]}
 allrows={}
 for step,report in reports.items():
  assert report['complete'] and len(report['results'])==1475
  preds={r['position_id']:r for r in report['results']};assert preds.keys()==positions.keys()
  rows=[]
  for m in meta:
   p=preds[m['position_id']]
   for k in ['board_hash','episode_id','teacher_command','phase']:assert p[k]==m[k]
   scorer=score_choice if m['teacher_command']['kind']==1 else score_nonmovement
   rows.append({**{k:v for k,v in m.items() if k!='scores'},**scorer(m['scores'],m['teacher_command'],p['selected_command']),
                'selected_command':p['selected_command'],'first_attempt_valid':p['first_attempt_valid'],'input_tokens':p['input_tokens'],
                'selected_build':p['selected_command'] is not None and p['selected_command']['kind']==6,
                'selected_road':p['selected_command'] is not None and p['selected_command']['kind']==6 and p['selected_command']['parameter']==3})
  allrows[step]={r['position_id']:r for r in rows}
 ids=[r['position_id'] for r in allrows['0'].values() if r['teacher_command']['kind']==1 and r['movement_required_for_top_score']]
 assert len(ids)==458
 a,b,c=([allrows[s][i] for i in ids] for s in ['0','157','314'])
 losses=[i for i in ids if allrows['0'][i]['tie_aware'] and not allrows['157'][i]['tie_aware']]
 gains=[i for i in ids if not allrows['0'][i]['tie_aware'] and allrows['157'][i]['tie_aware']]
 def kind(r):return str(r['selected_command']['kind']) if r['selected_command'] else 'invalid'
 matrix=Counter((kind(x),kind(y)) for x,y in zip(a,b))
 cases=[]
 for name,keys in [('regressed',losses),('improved',gains)]:
  for i in keys:
   rows=[allrows[s][i] for s in ['0','157','314']]
   cases.append(dict(position_id=i,change=name,board_hash=rows[0]['board_hash'],phase=rows[0]['phase'],
      teacher_command=rows[0]['teacher_command'],legal_count=rows[0]['legal_count'],legal_movement_count=rows[0]['legal_movement_count'],
      top_count=rows[0]['top_count'],input_tokens=rows[0]['input_tokens'],
      checkpoints={s:{k:r[k] for k in ['selected_command','tie_aware','selected_movement','selected_score','top_score','score_gap','movement_uses_top_source','movement_reaches_top_target','first_attempt_valid']} for s,r in zip(['0','157','314'],rows)}))
 cohorts={}
 for name,predicate in [('both_move',lambda x,y:x['selected_movement'] and y['selected_movement']),('stopped_moving',lambda x,y:x['selected_movement'] and not y['selected_movement']),('started_moving',lambda x,y:not x['selected_movement'] and y['selected_movement']),('neither_move',lambda x,y:not x['selected_movement'] and not y['selected_movement'])]:
  chosen=[i for i,(x,y) in enumerate(zip(a,b)) if predicate(x,y)]
  cohorts[name]=dict(count=len(chosen),correct_before=sum(a[i]['tie_aware'] for i in chosen),correct_after=sum(b[i]['tie_aware'] for i in chosen),correct_20k=sum(c[i]['tie_aware'] for i in chosen))
 assert sum(x['count'] for x in cohorts.values())==len(ids)
 assert sum(x['correct_after']-x['correct_before'] for x in cohorts.values())==len(gains)-len(losses)
 existing=read(RUN/'summary.json')['checkpoints']
 for step in allrows:
  assert sum(allrows[step][i]['tie_aware'] for i in ids)==existing[step]['groups']['movement_required']['tie_aware_matches']
 baseline_comparison={}
 other=ROOT/'runs/movement-mixed-v1'
 assert read(other/'protocol.json')['positions_sha256']==read(RUN/'protocol.json')['positions_sha256']
 old={r['position_id']:r for r in read(other/'report-0.json')['results']}
 oldstrict=[]
 for m in meta:
  i=m['position_id']
  if i in ids:oldstrict.append({**allrows['0'][i],**score_choice(m['scores'],m['teacher_command'],old[i]['selected_command'])})
 current_byid={r['position_id']:r for r in a}
 baseline_comparison=dict(same_adapter_hashes=read(other/'protocol.json')['training']['mixed']['adapter_files']==read(RUN/'protocol.json')['training']['scale']['adapter_files'],
  same_positions_sha256=True,changed_commands_all=sum(old[i]['selected_command']!=allrows['0'][i]['selected_command'] for i in positions),
  changed_commands_strict=sum(old[i]['selected_command']!=allrows['0'][i]['selected_command'] for i in ids),
  strict_correct_previous=sum(r['tie_aware'] for r in oldstrict),strict_correct_current=sum(r['tie_aware'] for r in a),
  current_minus_previous=paired_change(oldstrict,a),checkpoint10k_minus_previous=paired_change(oldstrict,b))
 phase={p:dict(count=sum(x['phase']==p for x in a),correct=[sum(x['tie_aware'] for x in rr if x['phase']==p) for rr in [a,b,c]],
              losses=sum(allrows['0'][i]['phase']==p for i in losses),gains=sum(allrows['0'][i]['phase']==p for i in gains)) for p in sorted({x['phase'] for x in a})}
 out=dict(primary=paired_change(a,b),movement_selection_change=paired_change(a,b,metric='selected_movement'),
  build_selection_change=paired_change(a,b,metric='selected_build'),road_selection_change=paired_change(a,b,metric='selected_road'),
  correctness_trajectory={s:sum(allrows[s][i]['tie_aware'] for i in ids) for s in allrows},
  action_kind_counts={s:dict(Counter(kind(allrows[s][i]) for i in ids)) for s in allrows},
  build_parameter_counts={s:dict(Counter(allrows[s][i]['selected_command']['parameter'] for i in ids if kind(allrows[s][i])=='6')) for s in allrows},
  transition_matrix={f'{x}->{y}':n for (x,y),n in sorted(matrix.items())},cohorts=cohorts,
  losses_by_10k_family=dict(Counter(kind(allrows['157'][i]) for i in losses)),gains_by_baseline_family=dict(Counter(kind(allrows['0'][i]) for i in gains)),
  loss_maps=len({allrows['0'][i]['board_hash'] for i in losses}),gain_maps=len({allrows['0'][i]['board_hash'] for i in gains}),
  lost16_recovered_at20k=sum(allrows['314'][i]['tie_aware'] for i in losses),gained9_retained_at20k=sum(allrows['314'][i]['tie_aware'] for i in gains),
  phase=phase,baseline_repeat=baseline_comparison,case_count=len(cases),
  first_attempt_invalid_strict={s:sum(not allrows[s][i]['first_attempt_valid'] for i in ids) for s in allrows},
  report_sha256={s:sha(RUN/f'report-{s}.json') for s in allrows})
 save('prediction-audit.json',out);save('changed-decisions.json',cases)
 print(json.dumps(out,indent=2),flush=True)


def training():
 selection=read(RUN/'selection.json');selected=selection['selected'];original={x['input_hash'] for x in selection['original_selected']}
 stats=[];token_hash=hashlib.sha256()
 with gzip.open(RUN/'prepared/tokens.jsonl.gz','rb') as f:
  for line in f:
   token_hash.update(line);r=json.loads(line);stats.append(dict(length=len(r['input_ids']),supervised=sum(t!=-100 for t in r['labels'])))
 assert token_hash.hexdigest()==read(RUN/'prepared/manifest.json')['tokens_sha256']
 print('Token stream verified; reconstructing actual batches.',flush=True)
 for i,row in enumerate(gz(RUN/'records.jsonl.gz')):
  m=selected[i];assert (row['episode_id'],row['decision_index'])==(m['episode_id'],m['decision_index'])
  scores=row['teacher_scores'];best=max(x['score'] for x in scores);top=[x['command'] for x in scores if x['score']==best]
  assert row['selected_command'] in top and row['selected_command']['kind']==m['family']
  stats[i].update(family=m['family'],parameter=row['selected_command']['parameter'],source=m['collection_source'],phase=m['phase'],original=m['input_hash'] in original,
      strict_movement=all(x['kind']==1 for x in top),legal_moves=sum(x['command']['kind']==1 for x in scores),top_count=len(top))
 assert i+1==len(stats)==40000
 order=list(EffectiveBatchLengthSampler([r['length'] for r in stats],seed=17,effective_batch_size=64))
 receipt=read(RUN/'stopped-training-receipt.json');assert digest(order)==receipt['epoch_order_sha256']
 def describe(indices):
  rows=[stats[i] for i in indices];moves=[x for x in rows if x['family']==1]
  return dict(count=len(rows),original_replay=sum(x['original'] for x in rows),families=dict(Counter(x['family'] for x in rows)),sources=dict(Counter(x['source'] for x in rows)),
   movement_count=len(moves),movement_phases=dict(Counter(x['phase'] for x in moves)),movement_sources=dict(Counter(x['source'] for x in moves)),strict_movement=sum(x['strict_movement'] for x in rows),
   movement_supervised_token_share=sum(x['supervised'] for x in moves)/sum(x['supervised'] for x in rows),
   movement_mean_response_tokens=float(np.mean([x['supervised'] for x in moves])),nonmovement_mean_response_tokens=float(np.mean([x['supervised'] for x in rows if x['family']!=1])),
   mean_prompt_and_response_tokens=float(np.mean([x['length'] for x in rows])),movement_mean_legal_moves=float(np.mean([x['legal_moves'] for x in moves])),
   movement_tied_top_fraction=sum(x['top_count']>1 for x in moves)/len(moves),build_parameters=dict(Counter(x['parameter'] for x in rows if x['family']==6)))
 groups={'original_10k':describe([i for i,x in enumerate(stats) if x['original']]),'full_40k':describe(range(40000))}
 for begin,end in [(0,10000),(10000,20000),(20000,30000),(9000,10000),(19000,20000)]:groups[f'examples_{begin}_{end}']=describe(order[begin:end])
 logs=receipt['logs'];ranges=batch_ranges(40000,64,BOUNDARIES);batch_details=[]
 for j,((begin,end),log) in enumerate(zip(ranges,logs)):
  assert log['step']==j+1 and log['examples_seen']==end
  rows=[stats[i] for i in order[begin:end]];nmove=sum(x['family']==1 for x in rows)
  token_total=sum(x['supervised'] for x in rows)
  batch_details.append(dict(step=j+1,examples=end,batch_count=end-begin,movement=nmove,families=dict(Counter(x['family'] for x in rows)),movement_supervised_share=sum(x['supervised'] for x in rows if x['family']==1)/token_total,
    lr_used=0 if j==0 else logs[j-1]['learning_rate'],lr_after=log['learning_rate'],loss=log['loss'],grad_norm=log['grad_norm']))
 schedules={}
 for label,rows in [('0_to_10k',batch_details[:157]),('10k_to_20k',batch_details[157:314]),('last_20_before_10k',batch_details[137:157])]:
  schedules[label]=dict(updates=len(rows),mean_loss=float(np.mean([x['loss'] for x in rows])),mean_grad_norm=float(np.mean([x['grad_norm'] for x in rows])),max_grad_norm=max(x['grad_norm'] for x in rows),clipped_steps=sum(x['grad_norm']>1 for x in rows),
   min_movement_per_batch=min(x['movement'] for x in rows),zero_movement_batches=sum(x['movement']==0 for x in rows),lr_sum=sum(x['lr_used'] for x in rows),last_lr_used=rows[-1]['lr_used'],last_lr_after=rows[-1]['lr_after'],
   lr_weighted_movement_token_share=sum(x['lr_used']*x['movement_supervised_share'] for x in rows)/sum(x['lr_used'] for x in rows))
 # Independent replay of the declared scheduler, with update t using scheduler(t-1).
 peak=receipt['config']['learning_rate'];warmup=math.ceil(628*.03)
 def lr(t,total,warm):return peak*(t/warm if t<warm else .5*(1+math.cos(math.pi*(t-warm)/(total-warm))))
 assert max(abs(row['lr_used']-lr(i,628,warmup)) for i,row in enumerate(batch_details))<1e-12
 old=read(ROOT/'runs/movement-epoch2-v1/training-run.json')['logs']
 old_used=[0]+[x['learning_rate'] for x in old[:-1]]
 schedule_comparison=dict(planned_updates=628,warmup_updates=warmup,original_10k_updates=157,original_10k_warmup=math.ceil(157*.03),
  first157_lr_sum=sum(x['lr_used'] for x in batch_details[:157]),original157_lr_sum=sum(old_used),ratio=sum(x['lr_used'] for x in batch_details[:157])/sum(old_used),
  checkpoint10k_lr_for_next_update=logs[156]['learning_rate'],checkpoint20k_lr_for_next_update=logs[313]['learning_rate'],original10k_final_lr=old[-1]['learning_rate'])
 save('training-audit.json',dict(groups=groups,windows=schedules,schedule_comparison=schedule_comparison,epoch_order_verified=True,tokens_hash_verified=True,all_record_metadata_aligned=True))
 save('batch-exposure.json',batch_details)
 save('training-row-metadata.json',stats)
 print(json.dumps(dict(groups=groups,windows=schedules,schedule_comparison=schedule_comparison),indent=2),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('part',choices=['predictions','training']);args=parser.parse_args()
 (predictions if args.part=='predictions' else training)()
