#!/usr/bin/env python3
"""Independently replay 100 hash-random sampled labels plus family coverage."""
import argparse
from collections import defaultdict,Counter
from dataclasses import asdict
import gzip,hashlib,json,multiprocessing,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from polytopia_bot.learning.interface import digest


def replay(path, selections):
    from polytopia_bot.learning.adapter import GPUEnvironment
    from polytopia_bot.learning.teacher import ScriptedTeacher
    path=Path(path);episode=selections[0]['episode_id']
    env=GPUEnvironment(14);teacher=ScriptedTeacher()
    obs=env.restore((path.parent/'private'/f'{episode}.snapshot').read_bytes())
    wanted={r['line']:r for r in selections};last=max(wanted);audits=[]
    try:
        for index,line in enumerate(gzip.open(path,'rt')):
            row=json.loads(line)
            if index in wanted:
                record=row['record']
                assert record is not None
                assert asdict(obs)==record['observation'],f'{episode}:{index}: public replay mismatch'
                selected=teacher.select(obs)
                assert dict(selected.action.command)==record['selected_command']
                assert selected.teacher_hash==record['teacher_hash']
                assert selected.scores==record['teacher_scores']
            action=row['selection']['action']
            transition=env.step(action_id=action['action_id'],expected_state_version=obs.state_version)
            if index in wanted:
                assert [asdict(env.observe(p)) for p in (1,2)]==row['after'],f'{episode}:{index}: post-action replay mismatch'
                assert transition.terminated==row['outcome']['terminated']
                assert transition.truncated==row['outcome']['truncated']
                assert {str(k):v for k,v in transition.rewards.items()}==row['outcome']['rewards']
                assert transition.reason==row['outcome']['reason']
                audits.append(dict(**wanted[index],passed=True,selected_command=dict(selected.action.command),
                    reason=selected.scores[0]['reason'],turn=obs.visible_state['turn'],stars=obs.visible_state['stars'],
                    input_digest=digest(asdict(obs)),scores_digest=digest(selected.scores)))
            if index==last:break
            assert not transition.terminated and not transition.truncated
            obs=env.observe(transition.next_player)
        assert len(audits)==len(selections)
        return dict(episode=episode,steps=last+1,labels=audits)
    finally:env.close()


def main(sample,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((Path(sample)/'sampling.json').read_text())
    assert hashlib.sha256((Path(sample)/'records.jsonl').read_bytes()).hexdigest()==metadata['records_sha256']
    rows=sorted(metadata['selected'],key=lambda r:digest(['label-replay-v1',r['input_hash']]))
    selected=rows[:100]
    if len(selected)<100:raise ValueError('at least 100 labels required')
    for family in sorted({r['family'] for r in rows}):
        needed=max(0,3-sum(r['family']==family for r in selected))
        selected.extend([r for r in rows if r['family']==family and r not in selected][:needed])
    grouped=defaultdict(list)
    for row in selected:grouped[row['file']].append(row)
    for path in grouped:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==metadata['source_files'][path]
    (root/'protocol.json').write_text(json.dumps(dict(schema='label-replay-v1',selected=selected,
        sample_records_sha256=metadata['records_sha256'],selection='100 independent hash-random rows plus at least three examples per available action family',
        checks=['Exact public before/after replay','Current frozen teacher action, scores and hash reproduce','Every replayed prefix action executes through frozen engine']),indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(replay,path,rows) for path,rows in grouped.items()]
        for future in as_completed(futures):
            results.append(future.result())
            (root/'episodes.json').write_text(json.dumps(results,indent=2))
    labels=[row for episode in results for row in episode['labels']]
    report=dict(passed=True,labels=len(labels),episodes=len(results),prefix_actions_replayed=sum(e['steps'] for e in results),
        families=dict(Counter(str(r['family']) for r in labels)),
        limitation='Confirms reproducible, legal heuristic labels and public information integrity; does not prove strategic optimality.')
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sample',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.sample,a.output)
