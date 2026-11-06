#!/usr/bin/env python3
"""Audit all recorded prefixes, explicitly retaining budget-incomplete outcomes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip,hashlib,json,multiprocessing,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def replay(task):
    from polytopia_bot.learning.adapter import GPUEnvironment
    from polytopia_bot.learning.interface import render,parse_response,ResponseError
    from polytopia_bot.learning.prompt_search import candidate_messages
    path,expected=task;path=Path(path)
    snapshot=path.parent/'private'/(expected['id']+'.snapshot')
    env=GPUEnvironment(14);obs=env.restore(snapshot.read_bytes())
    decisions=calls=model_decisions=first_valid=0
    terminal=None;forfeit=False;result=None
    try:
        for line in gzip.open(path,'rt'):
            row=json.loads(line)
            if 'game_result' in row:
                result=row['game_result'];continue
            assert not terminal and not forfeit,'trace continued after terminal'
            assert asdict(obs)==row['observation'],'public observation differs on replay'
            if row['policy']=='model':
                model_decisions+=1
                messages=candidate_messages(obs,expected['strategy'])
                prompt=render(obs)
                attempts=row.get('attempts') if row.get('forfeit') else row['diagnostic']
                assert 1<=len(attempts)<=3,'invalid repair count'
                action=None
                for i,attempt in enumerate(attempts):
                    assert attempt['messages']==messages,'model prompt differs from authoritative public observation/repairs'
                    calls+=1
                    try:
                        action=parse_response(attempt['generation']['text'],prompt,current_version=obs.state_version)
                    except ResponseError as exc:
                        assert attempt.get('error')==str(exc),'repair error mismatch'
                        messages=messages+[dict(role='assistant',content=attempt['generation']['text']),
                            dict(role='user',content=f'Invalid response: {exc}. Return one legal action_id JSON object.')]
                    else:
                        assert i==len(attempts)-1,'valid action was unnecessarily repaired'
                        first_valid+=i==0
                if row.get('forfeit'):
                    assert action is None and len(attempts)==3,'false invalid-response forfeit'
                    forfeit=True;continue
                assert action is not None,'model action absent'
                assert dict(action.command)==row['selected_command'],'executed action differs from generated response'
            else:
                assert row['policy']=='scripted_opponent','unknown policy'
                matches=[a for a in obs.legal_actions if dict(a.command)==row['selected_command']]
                assert len(matches)==1,'opponent command not uniquely legal'
                action=matches[0]
            transition=env.step(action_id=action.action_id,expected_state_version=obs.state_version)
            decisions+=1
            assert [asdict(env.observe(p)) for p in (1,2)]==row['after'],'public post-action replay mismatch'
            outcome=row['outcome']
            assert transition.terminated==outcome['terminated'] and transition.truncated==outcome['truncated'],'terminal mismatch'
            assert {str(k):v for k,v in transition.rewards.items()}==outcome['rewards'],'reward mismatch'
            assert transition.reason==outcome['reason'],'terminal reason mismatch'
            if transition.terminated or transition.truncated:terminal=transition
            else:obs=env.observe(transition.next_player)
        assert result==expected,'combined and trace game results differ'
        for key,value in [('decisions',decisions),('model_decisions',model_decisions),('attempts',calls),('first_attempt_valid',first_valid)]:
            assert result[key]==value,'counter mismatch: '+key
        if forfeit:
            assert result['status']=='forfeit' and result['reason']=='invalid_response_after_two_repairs' and result['reward']==-1
        elif terminal:
            assert result['status']==('native' if terminal.terminated else 'truncated')
            assert result['reason']==terminal.reason and result['reward']==terminal.rewards[expected['seat']]
        elif result['status']=='budget_incomplete':
            assert result['reason']=='occupied_time_ceiling' and result['reward']==0.0
        else:
            assert result['status']=='truncated' and result['reason']=='research_action_limit' and decisions==10000
        return dict(id=expected['id'],passed=True,decisions=decisions,model_decisions=model_decisions,
                    trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    finally:env.close()


def audit(source,output,workers=4):
    root=ROOT/source;protocol=json.loads((root/'protocol.json').read_text())
    report=json.loads((root/'combined/games/report.json').read_text())
    if report['complete'] or len(report['games'])!=len(protocol['jobs']) or report['unstarted_jobs']:raise ValueError('all started job prefixes with incomplete outcomes required')
    if not any(g['status']=='budget_incomplete' for g in report['games']):raise ValueError('expected bounded incompletion')
    for name,sha in protocol['sources'].items():
        if name.startswith('src/polytopia_bot/') and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('replay implementation changed: '+name)
    tasks=[]
    for game in report['games']:
        paths=list(root.glob('*shard*/games/'+game['id']+'.jsonl.gz'))
        if len(paths)!=1:raise ValueError('missing or duplicate trace')
        tasks.append((str(paths[0]),game))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(replay,tasks))
    receipt=dict(schema='incomplete-game-prefix-replay-audit-v1',passed=True,evaluation_complete=False,games=len(rows),
        budget_incomplete=sum(g['status']=='budget_incomplete' for g in report['games']),
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        decisions=sum(r['decisions'] for r in rows),model_decisions=sum(r['model_decisions'] for r in rows),
        report_sha256=hashlib.sha256((root/'combined/games/report.json').read_bytes()).hexdigest(),results=rows,
        note='Every logged transition, public before/after state, generated action/repair, observed terminal and counter replayed. Budget-incomplete prefixes are not native outcomes or accepted full games. This audit cannot satisfy the full-game gate.')
    with (ROOT/output).open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps({k:v for k,v in receipt.items() if k!='results'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();audit(a.source,a.output,a.workers)
