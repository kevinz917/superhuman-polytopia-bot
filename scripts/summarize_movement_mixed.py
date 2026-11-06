"""Compare fixed held-out movement and retention controls across mixed checkpoints."""
import gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.training import file_hash
from polytopia_bot.learning.movement_diagnostic import score_choice
from polytopia_bot.learning.movement_mixed import score_nonmovement
from summarize_movement_continuation import paired_change
from summarize_movement_probe import stats,movement_reference_comparison


def control_stats(rows):
 n=len(rows)
 return dict(count=n,maps=len({r['board_hash'] for r in rows}),exact_matches=sum(r['exact'] for r in rows),tie_aware_matches=sum(r['tie_aware'] for r in rows),exact=sum(r['exact'] for r in rows)/n,tie_aware=sum(r['tie_aware'] for r in rows)/n,selected_movement=sum(r['selected_movement'] for r in rows)/n)


def main(root):
 root=Path(root);p=json.loads((root/'protocol.json').read_text())
 for name,key in [('positions.jsonl.gz','positions_sha256'),('scoring.jsonl.gz','scoring_sha256')]:
  if file_hash(root/name)!=p[key]:raise ValueError('evaluation inputs changed')
 positions={r['position_id']:r for r in map(json.loads,gzip.open(root/'positions.jsonl.gz','rt'))}
 metadata=list(map(json.loads,gzip.open(root/'scoring.jsonl.gz','rt')))
 if len(positions)!=p['evaluation_count'] or len(metadata)!=len(positions) or {r['position_id'] for r in metadata}!=positions.keys():raise ValueError('evaluation coverage differs')
 train_maps={r['map_hash'] for r in json.loads((root/'selection.json').read_text())['selected']}
 if train_maps&{r['board_hash'] for r in metadata}:raise ValueError('train/heldout map leakage')
 baseline=None;basegroups=None;previous_groups=None;previous_step=None;checkpoints={}
 for step in p['checkpoint_steps']:
  path=root/f'report-{step}.json'
  if not path.exists():continue
  report=json.loads(path.read_text());predictions={r['position_id']:r for r in report['results']}
  if not report['complete'] or report['decisions']!=len(positions) or len(report['results'])!=len(positions) or predictions.keys()!=positions.keys():raise ValueError('incomplete evaluation')
  for key,pred in predictions.items():
   for field in ['board_hash','episode_id','phase','teacher_command']:
    if pred[field]!=positions[key][field]:raise ValueError('prediction inputs changed')
   if baseline and pred['input_tokens']!=baseline[key]['input_tokens']:raise ValueError('prompt token count changed')
  if step==0:baseline=predictions
  if baseline is None:raise ValueError('fresh baseline required first')
  rows=[]
  for meta in metadata:
   pred=predictions[meta['position_id']]
   for field in ['board_hash','episode_id','phase','teacher_command','prompt_digest']:
    if meta[field]!=positions[meta['position_id']][field]:raise ValueError('scoring input changed')
   row={k:v for k,v in meta.items() if k!='scores'}
   scorer=score_choice if meta['teacher_command']['kind']==1 else score_nonmovement
   row.update(scorer(meta['scores'],meta['teacher_command'],pred['selected_command']))
   row.update(input_tokens=pred['input_tokens'],first_attempt_valid=pred['first_attempt_valid'])
   original=baseline[meta['position_id']]['selected_command'];row['baseline_selected_movement']=original is not None and original['kind']==1
   if row['exact']!=pred['teacher_agreement']:raise ValueError('exact metric differs')
   rows.append(row)
  groups=dict(movement=[r for r in rows if r['teacher_command']['kind']==1],nonmovement=[r for r in rows if r['teacher_command']['kind']!=1])
  groups['movement_required']=[r for r in groups['movement'] if r['movement_required_for_top_score']]
  groups.update({f'kind_{kind}':[r for r in groups['nonmovement'] if r['teacher_command']['kind']==kind] for kind in range(2,11)})
  if step==0:basegroups=groups
  changes={k:{metric:paired_change(basegroups[k],group,metric=metric) for metric in ['tie_aware','exact']} for k,group in groups.items()}
  flags=[]
  if changes['nonmovement']['tie_aware']['delta']<-.03:flags.append('overall_nonmovement_loss_gt_3pp')
  flags += [f'{k}_loss_gt_10pp' for k in groups if k.startswith('kind_') and changes[k]['tie_aware']['delta']<-.10]
  strict=groups['movement_required']
  checkpoints[str(step)]=dict(step=step,examples_seen=p.get('checkpoint_examples',{}).get(str(step),min(step*64,10000)),groups={k:(stats(group) if k.startswith('movement') else control_stats(group)) for k,group in groups.items()},changes=changes,retention_flags=flags,
   held_out_minus_uniform_movement=movement_reference_comparison(strict),
   selected_moves_minus_matched_uniform=movement_reference_comparison([r for r in strict if r['selected_movement']]) if any(r['selected_movement'] for r in strict) else None,
   movement_required_baseline_choice_changes={str(flag):paired_change([r for r in basegroups['movement_required'] if r['baseline_selected_movement']==flag],[r for r in strict if r['baseline_selected_movement']==flag]) for flag in [True,False]},
   movement_required_by_phase={phase:stats([r for r in strict if r['phase']==phase]) for phase in sorted({r['phase'] for r in strict})},
   movement_required_by_baseline_choice={str(flag):stats([r for r in strict if r['baseline_selected_movement']==flag]) for flag in [True,False]},
   first_attempt_valid=report['first_attempt_valid'],eventually_valid=report['eventually_valid'],inference_seconds=report['elapsed_seconds'])
  if p['schema']=='original-proportions-scale40k-v1':
   checkpoints[str(step)]['previous_checkpoint']=previous_step
   checkpoints[str(step)]['changes_from_previous']={} if previous_groups is None else {key:{metric:paired_change(previous_groups[key],groups[key],metric=metric) for metric in ['tie_aware','exact']} for key in ['movement_required','nonmovement']}
  previous_groups=groups;previous_step=step
 result=dict(schema=('original-proportions-scale40k-result-v1' if p['schema']=='original-proportions-scale40k-v1' else 'movement-enriched-mixed-result-v1'),complete=len(checkpoints)==len(p['checkpoint_steps']),protocol_sha256=file_hash(root/'protocol.json'),report_sha256={step:file_hash(root/f'report-{step}.json') for step in checkpoints},checkpoints=checkpoints,
  limitations=['Fixed development maps reused across experiments; final test untouched.','Teacher tie-aware agreement measures imitation of a heuristic, not game strength.','Nonmovement controls balanced by action family, not gameplay frequency.','Paired map-bootstrap intervals exploratory and unadjusted for checkpoint selection.','Retention flags are finite-sample diagnostics, not proof that no other skill regressed.'])
 (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({step:dict(movement_required=c['groups']['movement_required']['tie_aware'],nonmovement_tie=c['groups']['nonmovement']['tie_aware'],nonmovement_exact=c['groups']['nonmovement']['exact'],retention_flags=c['retention_flags']) for step,c in checkpoints.items()},indent=2))

if __name__=='__main__':main(sys.argv[1])
