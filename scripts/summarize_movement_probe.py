"""Separate probe memorization, transfer, and non-movement forgetting."""
import gzip,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.movement_diagnostic import score_choice,summarize
from polytopia_bot.learning.movement_probe import stratum
from polytopia_bot.learning.training import file_hash
from summarize_movement_continuation import paired_change


def stats(rows):
    result=summarize(rows)
    if rows:result['uniform_movement_top_score_baseline']=sum(r['top_movement_count']/r['legal_movement_count'] for r in rows)/len(rows)
    moves=[r for r in rows if r['selected_movement']]
    result['uniform_movement_top_score_baseline_on_selected_moves']=sum(r['top_movement_count']/r['legal_movement_count'] for r in moves)/len(moves) if moves else None
    return result


def movement_reference_comparison(rows):
    """Compare teacher agreement to the analytic expectation of a uniform legal move."""
    import numpy as np
    clusters=defaultdict(lambda:[0.,0])
    for r in rows:
        clusters[r['board_hash']][0]+=int(r['tie_aware'])-r['top_movement_count']/r['legal_movement_count']
        clusters[r['board_hash']][1]+=1
    totals=np.array([clusters[k] for k in sorted(clusters)])
    draws=np.random.default_rng(17).integers(0,len(totals),size=(5000,len(totals)))
    samples=totals[draws].sum(axis=1)
    return dict(delta=totals[:,0].sum()/totals[:,1].sum(),interval_95=np.quantile(samples[:,0]/samples[:,1],[.025,.975]).tolist(),
                method='Exploratory map-cluster bootstrap of correctness minus analytic uniform-movement expectation; 5000 draws, seed 17')


def main(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text())
    for name,key in [('positions.jsonl.gz','positions_sha256'),('scoring.jsonl.gz','scoring_sha256')]:
        if file_hash(root/name)!=p[key]:raise ValueError('evaluation inputs changed')
    positions={r['position_id']:r for r in map(json.loads,gzip.open(root/'positions.jsonl.gz','rt'))}
    metadata=list(map(json.loads,gzip.open(root/'scoring.jsonl.gz','rt')))
    if len(metadata)!=714 or len(positions)!=858:raise ValueError('wrong probe allocation')
    baseline=None;baseline_groups=None;baseline_controls=None;checkpoints={}
    for step in p['checkpoint_steps']:
        report=json.loads((root/f'report-{step}.json').read_text())
        predictions={r['position_id']:r for r in report['results']}
        if not report['complete'] or report['decisions']!=858 or len(report['results'])!=858 or predictions.keys()!=positions.keys():
            raise ValueError('incomplete checkpoint evaluation')
        for key,pred in predictions.items():
            for field in ['board_hash','episode_id','phase','teacher_command']:
                if pred[field]!=positions[key][field]:raise ValueError('prediction input changed')
            if baseline and pred['input_tokens']!=baseline[key]['input_tokens']:raise ValueError('prompt token count changed')
        if step==0:baseline=predictions
        rows=[]
        for meta in metadata:
            pred=predictions[meta['position_id']]
            for field in ['board_hash','episode_id','phase','teacher_command','prompt_digest']:
                if meta[field]!=positions[meta['position_id']][field]:raise ValueError('scoring input changed')
            row={k:v for k,v in meta.items() if k!='scores'}
            row.update(score_choice(meta['scores'],meta['teacher_command'],pred['selected_command']))
            row.update(input_tokens=pred['input_tokens'],first_attempt_valid=pred['first_attempt_valid'],catalog_group=stratum(meta)[1])
            original=baseline[meta['position_id']]['selected_command']
            row['baseline_selected_movement']=original is not None and original['kind']==1
            if not row['movement_required_for_top_score'] or row['exact']!=pred['teacher_agreement']:raise ValueError('movement scoring mismatch')
            rows.append(row)
        groups={group:[r for r in rows if r['probe_group']==group] for group in ['probe_train','parent_train_unfocused','held_out']}
        maps={key:{r['board_hash'] for r in group} for key,group in groups.items()}
        if any(maps[a]&maps[b] for a,b in [('probe_train','parent_train_unfocused'),('probe_train','held_out'),('parent_train_unfocused','held_out')]):
            raise ValueError('comparison maps overlap')
        controls=[predictions[key] for key in p['control_ids']]
        if step==0:baseline_groups=groups;baseline_controls=controls
        group_summaries={key:stats(group) for key,group in groups.items()}
        checkpoints[str(step)]=dict(step=step,probe_epochs=p['checkpoint_epochs'][str(step)],groups=group_summaries,
            fit_target_met=group_summaries['probe_train']['tie_aware']>=p['fit_target']['tie_aware'] and group_summaries['probe_train']['exact']>=p['fit_target']['exact'],
            changes_from_baseline={key:paired_change(baseline_groups[key],group) for key,group in groups.items()},
            held_out_minus_uniform_movement=movement_reference_comparison(groups['held_out']),
            by_phase={key:{phase:stats([r for r in group if r['phase']==phase]) for phase in ['opening','exploration','late']} for key,group in groups.items()},
            by_catalog={key:{cat:stats([r for r in group if r['catalog_group']==cat]) for cat in ['under64','64to127','128plus']} for key,group in groups.items()},
            held_out_by_baseline_movement={str(flag):dict(summary=stats([r for r in groups['held_out'] if r['baseline_selected_movement']==flag]),
                change=paired_change([r for r in baseline_groups['held_out'] if r['baseline_selected_movement']==flag],[r for r in groups['held_out'] if r['baseline_selected_movement']==flag])) for flag in [True,False]},
            controls=dict(count=len(controls),exact_matches=sum(r['teacher_agreement'] for r in controls),
                selected_movement=sum(r['selected_command'] is not None and r['selected_command']['kind']==1 for r in controls),
                change_from_baseline=paired_change(baseline_controls,controls,metric='teacher_agreement'),
                by_kind={str(kind):dict(count=sum(r['family']==kind for r in controls),exact_matches=sum(r['family']==kind and r['teacher_agreement'] for r in controls)) for kind in range(2,11)}),
            first_attempt_valid=report['first_attempt_valid'],eventually_valid=report['eventually_valid'],inference_seconds=report['elapsed_seconds'])
    result=dict(schema='movement-only-overfit-result-v1',protocol_sha256=file_hash(root/'protocol.json'),
        report_sha256={str(step):file_hash(root/f'report-{step}.json') for step in p['checkpoint_steps']},checkpoints=checkpoints,
        first_fit_target_step=next((step for step in p['checkpoint_steps'] if checkpoints[str(step)]['fit_target_met']),None),
        limitations=['A deliberately small, stratified overfit sample tests fitting capacity, not a production training recipe.',
                    'The unfocused training comparison was seen during parent training, so it is not held-out generalization.',
                    'Movement-only supervision changes the action-family prior; inspect non-movement controls and conditional movement quality.',
                    'Paired map-cluster intervals are exploratory, unadjusted for multiple checkpoints or inference-runtime variability.',
                    'Agreement with a frozen heuristic teacher is not game strength. No full games, RL, or final-test evaluation.'])
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({step:dict(epochs=c['probe_epochs'],fit=c['fit_target_met'],groups={k:dict(tie=v['tie_aware'],exact=v['exact'],moves=v['selected_movement']) for k,v in c['groups'].items()},controls=c['controls']['exact_matches']/c['controls']['count']) for step,c in checkpoints.items()},indent=2))


if __name__=='__main__':main(sys.argv[1])
