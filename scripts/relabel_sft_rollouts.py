#!/usr/bin/env python3
"""Label public states from audited training-only smoke rollouts.

The unchanged demonstration schema's source='teacher' identifies the labeler.
Separate mandatory provenance sidecars identify model-driven collection and
link every label to its original trace. Actual model transitions are retained;
teacher labels are never substituted for the executed action during replay.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import gzip,hashlib,json,multiprocessing,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.data import RECORD_SCHEMA,observation_from_dict
from polytopia_bot.learning.interface import canonical,digest


def convert(task):
    from polytopia_bot.learning.teacher import ScriptedTeacher
    trace,game,output,selected_indices,first_deviation_index=task
    trace=Path(trace);output=Path(output);teacher=ScriptedTeacher();selected_indices=set(selected_indices)
    episode=game['id'];target=output/(episode+'.jsonl.gz')
    snapshot=trace.parent/'private'/(episode+'.snapshot')
    shutil.copyfile(snapshot,output/'private'/snapshot.name)
    labels=[];transitions=0;unexecuted=0
    with gzip.open(target,'xt',compresslevel=1) as out:
        for original_index,line in enumerate(gzip.open(trace,'rt')):
            row=json.loads(line)
            if 'game_result' in row:continue
            if 'selected_command' not in row:
                # A final failed response has no transition. It remains in the
                # original trace and outcome receipt, but not this transition dataset.
                if not row.get('forfeit'):raise ValueError('unexpected non-transition row')
                unexecuted+=1;continue
            obs=observation_from_dict(row['observation'])
            matches=[a for a in obs.legal_actions if dict(a.command)==row['selected_command']]
            if len(matches)!=1:raise ValueError('executed action must be uniquely legal')
            record=None
            if row['policy']=='model':
                if original_index in selected_indices:
                    if first_deviation_index is None or original_index<first_deviation_index:raise ValueError('label precedes policy deviation')
                    choice=teacher.select(obs);first_deviation=original_index==first_deviation_index
                    if first_deviation and dict(choice.action.command)==row['selected_command']:raise ValueError('first deviation does not reproduce')
                    record=dict(schema=RECORD_SCHEMA,source='teacher',split='train',episode_id=episode,
                        map_hash=game['board_hash'],decision_index=transitions,observation=row['observation'],
                        teacher_hash=choice.teacher_hash,selected_command=dict(choice.action.command),teacher_scores=choice.scores)
                    labels.append(dict(line=transitions,original_line=original_index,
                        collection_source='smoke_on_policy_relabel',first_deviation=first_deviation,
                        public_observation_digest=digest(row['observation']),teacher_command=dict(choice.action.command),
                        executed_model_command=row['selected_command']))
            elif row['policy']!='scripted_opponent':raise ValueError('unknown acting policy')
            out.write(canonical(dict(record=record,observation=None if record else row['observation'],
                selection=dict(action=asdict(matches[0]),policy=row['policy']),after=row['after'],outcome=row['outcome']))+'\n')
            transitions+=1
    if len(labels)!=len(selected_indices):raise ValueError('selected repair states were not all labeled')
    provenance=dict(schema='model-collected-teacher-labels-v1',episode_id=episode,
        collection_source='smoke_on_policy_relabel',label_source='frozen_teacher',teacher_hash=teacher.teacher_hash,
        source_trace=str(trace.relative_to(ROOT)),source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),labels=labels,
        first_deviation_original_line=first_deviation_index,
        omitted_unexecuted_forfeit_states=unexecuted,game_outcome=game,
        note='Labels use only the acting public observation. Prefix transitions execute the original policy action, not the counterfactual teacher label. All outcomes retained.')
    (output/'provenance'/(episode+'.json')).write_text(json.dumps(provenance,indent=2)+'\n')
    result=dict(id=episode,seed=game['seed'],board_hash=game['board_hash'],seat=game['seat'],opponent=game['opponent'],
        decisions=transitions,teacher_decisions=len(labels),teacher_hash=teacher.teacher_hash,
        terminated=game['status']=='native',truncated=game['status']=='truncated',reward=game['reward'],reason=game['reason'],
        collection_source='smoke_on_policy_relabel')
    (output/(episode+'.outcome.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def eligible_states(trace,teacher):
    first=None;eligible=[]
    for index,line in enumerate(gzip.open(trace,'rt')):
        row=json.loads(line)
        if row.get('policy')!='model' or 'selected_command' not in row:continue
        if first is None:
            choice=teacher.select(observation_from_dict(row['observation']))
            if dict(choice.action.command)==row['selected_command']:continue
            first=index
        eligible.append(index)
    return first,eligible


def relabel(source,audit_path,output,workers=4):
    source=ROOT/source;root=ROOT/output
    protocol=json.loads((source/'protocol.json').read_text());report=json.loads((source/'combined/games/report.json').read_text())
    audit=json.loads((ROOT/audit_path).read_text())
    if not protocol.get('training_only') or protocol.get('repair_pass')!=1 or protocol.get('final_test') or protocol.get('max_relabel_states')!=10000:
        raise ValueError('the registered training-only repair pass is required')
    if not report['complete'] or not audit['passed'] or audit['report_sha256']!=hashlib.sha256((source/'combined/games/report.json').read_bytes()).hexdigest():
        raise ValueError('complete, independently replayed training rollouts required')
    manifest=json.loads((ROOT/'runs/sft0/splits-v1/maps.json').read_text())
    train={(m['seed'],m['board_hash']) for m in manifest['maps']['train']}
    if any((g['seed'],g['board_hash']) not in train for g in report['games']):raise ValueError('non-training map in collection')
    for name,sha in protocol['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('frozen source changed')
    root.mkdir(parents=True,exist_ok=False);(root/'private').mkdir();(root/'provenance').mkdir()
    from polytopia_bot.learning.teacher import ScriptedTeacher
    teacher=ScriptedTeacher();pending=[];candidates=[]
    for game in report['games']:
        paths=list(source.glob('*shard*/games/'+game['id']+'.jsonl.gz'))
        if len(paths)!=1:raise ValueError('unique source trace required')
        first,eligible=eligible_states(paths[0],teacher)
        pending.append((str(paths[0]),game,first))
        candidates.extend((digest(['repair-state-cap-v1',game['id'],index]),game['id'],index) for index in eligible)
    chosen={(episode,index) for _,episode,index in sorted(candidates)[:10000]}
    tasks=[(path,game,str(root),[index for episode,index in chosen if episode==game['id']],first)
           for path,game,first in pending]
    (root/'state-selection.json').write_text(json.dumps(dict(eligible_states=len(candidates),selected_states=len(chosen),
        selection='Lowest hashes of fixed repair-state-cap-v1 salt, episode ID and original trace line; at most10000 labels before any merged corpus sampling.',
        selected=sorted(chosen)),indent=2)+'\n')
    games=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(convert,t) for t in tasks]):
            games.append(future.result());(root/'games.json').write_text(json.dumps(games,indent=2)+'\n')
    receipt=dict(schema='relabeled-training-rollouts-v1',complete=len(games)==len(tasks),games=len(games),
        relabel_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        teacher_decisions=sum(g['teacher_decisions'] for g in games),
        eligible_source_states=len(candidates),state_selection_sha256=hashlib.sha256((root/'state-selection.json').read_bytes()).hexdigest(),
        collection_source='smoke_on_policy_relabel',label_source='frozen_teacher',repair_pass=1,
        original_protocol_sha256=hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest(),
        replay_audit=dict(path=str(audit_path),sha256=hashlib.sha256((ROOT/audit_path).read_bytes()).hexdigest()),
        max_main_relabel_states=10000,max_main_relabel_fraction=.20,
        provenance_sidecars='provenance/*.json; retain these source tags in any merged sampling manifest',
        note='Candidate labels only. De-duplicate, sample within quotas and independently audit labels before training. No validation or final-test states.')
    if receipt['teacher_decisions']!=len(chosen) or len(chosen)>10000:raise ValueError('repair label cap/count mismatch')
    (root/'report.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','audit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();relabel(a.source,a.audit,a.output,a.workers)
