"""Meaningful protocol regressions; real engine and isolation run on Modal."""
import copy
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from polytopia_bot.skirmish.protocol import make_map_suite, validate_map_suite, apply_cap_policy
from polytopia_bot.skirmish.cohort import (authoring_cells,make_evaluation_plan,summarize_evaluation,score_rows)
from polytopia_bot.skirmish.common import REFERENCE
from polytopia_bot.skirmish.tournament import fit_elo,summarize


def manifest():
    p=dict(reasoning=dict(effort='high'),max_tokens=32768,provider=dict(only=['fixture'],allow_fallbacks=False,require_parameters=True))
    return dict(schema='polytopia-mini-cohort-v2',tracks=['independent-40m','feedback-3r-40m'],replicates=3,
        maps_per_family=5,models=[dict(label=label,model='fixture/'+label,parameters=p) for label in ('a','b')],
        references=[dict(label='weak',kind='frozen',run_id='old-weak'),dict(label='medium',kind='frozen',run_id='old-medium'),dict(label=REFERENCE,kind='expert')])


def test_schedule_repeats_all_builds_stages_seats_and_never_selects_best():
    m=manifest();cells=authoring_cells('test',m,'image');runs={}
    for c in cells:
        runs[c['run_id']]=dict(status='completed',stages=[dict(stage=s,returncode=0,source_sha256=f'source-{s}',wasm_sha256=f'wasm-{s}',submission_quality='custom_submission') for s in range(4 if c['track']=='feedback-3r-40m' else 1)])
    # Last revision may be broken; an older working stage cannot be substituted.
    c=next(c for c in cells if c['model_label']=='a' and c['track']=='feedback-3r-40m')
    runs[c['run_id']]['status']='failed_submission';runs[c['run_id']]['stages'][-1].update(returncode=1,submission_quality='compile_failure')
    maps=make_map_suite(sample=lambda population,n:list(range(1_000_000_000,1_000_000_000+n)))
    plan=make_evaluation_plan(cells,runs,maps,m['references'])
    assert plan['primary_games']==2*3*5*3*15*2
    assert len(plan['invalid_stages'])==1
    rows=[]
    for job in plan['jobs']:
        rows.append(dict(job_index=job['index'],job_sha256=job['job_sha256'],phase=job['phase'],
            model_label=job['candidate']['model_label'],track=job['candidate']['track'],
            replicate=job['candidate']['replicate'],stage=job['candidate']['stage'],
            submission_quality=job['candidate']['submission_quality'],opponent_label=job['opponent']['label'],
            map_type=job['map_type'],seed=job['seed'],result='loss' if job['invalid_submission'] else 'win'))
    report=summarize_evaluation(plan,cells,rows)
    model=next(r for r in report['ranking']['feedback-3r-40m'] if r['model_label']=='a')
    assert model['attempt_scores']==[0,100,100] and model['score']==pytest.approx(200/3)
    with pytest.raises(ValueError,match='incomplete|duplicated'):summarize_evaluation(plan,cells,rows[:-1])
    assert all(job['candidate']['model_label']!=job['opponent'].get('model_label') for job in plan['jobs'] if job['phase']=='peer')
    assert {job['seed'] for job in plan['jobs'] if job['phase']=='heldout'}=={m['seed'] for m in maps}


def test_caps_are_explicit_draws_but_infrastructure_timeouts_are_not():
    for reason in ('engine_truncation','action_limit'):
        row=apply_cap_policy(dict(status='unresolved',result='unresolved',reason=reason,winner=None),'engine_draw')
        assert row['status']==row['result']=='draw'
    row=apply_cap_policy(dict(status='unresolved',result='unresolved',reason='game_timeout',winner=None),'engine_draw')
    assert row['status']=='unresolved'
    assert score_rows([dict(result='win'),dict(result='unresolved')])['score'] is None
    assert score_rows([dict(result='win'),dict(result='draw')])['score']==75


def test_draw_elo_and_counts_are_symmetric_and_do_not_erase_unresolved():
    rows=[dict(a='a',b=REFERENCE,seed=1_000_000_000,seat=1,status='draw',winner=None,result='draw')]
    assert abs(fit_elo(rows,['a',REFERENCE])['a'])<1e-8
    report=summarize(rows,['a',REFERENCE],bootstrap=0)
    assert report['draws']==1 and report['unresolved']==0
    assert report['ladder'][0]['draws']==1


def test_suite_balances_families_and_excludes_public_and_practice_maps():
    maps=make_map_suite();assert len(validate_map_suite(maps))==30
    with pytest.raises(ValueError):validate_map_suite(maps[:-1])
    bad=copy.deepcopy(maps);bad[0]['seed']=500000
    with pytest.raises(ValueError):validate_map_suite(bad)
    bad=copy.deepcopy(maps);bad[0]['seed']=bad[1]['seed']
    with pytest.raises(ValueError):validate_map_suite(bad)


def test_metadata_timing_accounts_for_interrupted_calls_without_commands_or_content(tmp_path):
    from polytopia_bot.skirmish.telemetry import summarize_timing
    p=tmp_path/'events.jsonl'
    p.write_text('\n'.join(json.dumps(e) for e in [dict(event='api_start',id='a',unix=10),dict(event='api_end',id='a',unix=12),dict(event='tool_start',id='t',unix=12),dict(event='tool_end',id='t',unix=13),dict(event='api_start',id='b',unix=13)]))
    report=summarize_timing(p,cutoff=20)
    assert report['api_seconds']==9 and report['tool_seconds']==1 and report['interrupted_api_calls']==1


def test_deleted_source_is_a_failed_submission_not_a_retry_or_rollback(tmp_path):
    from polytopia_bot.skirmish.mini import freeze
    sandbox=Mock();sandbox.filesystem.read_text.return_value=json.dumps(dict(error='FileNotFoundError'))
    receipt=freeze(sandbox,tmp_path/'stage',Mock(),cutoff='/verify/cutoffs/stage-0')
    assert receipt['returncode']==-1 and receipt['submission_quality']=='compile_failure'
    assert (tmp_path/'stage/main.rs').read_bytes()==b''
    assert all(call.args[0]!='cargo' for call in sandbox.exec.call_args_list)


def test_debrief_uses_only_public_fields_and_does_not_claim_income_or_kills(tmp_path):
    from polytopia_bot.skirmish.diagnostics import public_summary
    obs=dict(player=1,visible_state=dict(turn=1,stars=5,technology=[1],tiles=[dict(known=False)],
        cities=[dict(owner=1,production=3),dict(owner=2)],units=[dict(owner=1),dict(owner=2)]),
        legal_actions=[dict(action_id='attack',command=dict(kind=2))])
    p=tmp_path/'trace.jsonl';p.write_text(json.dumps(dict(observation=obs,action_id='attack'))+'\n')
    summary=public_summary(p)
    assert summary['last']['own_city_production']==3 and summary['last']['own_units']==1
    assert summary['action_counts']=={'2':1} and 'income' not in summary['last'] and 'kills' not in summary['last']


@pytest.mark.skipif(not os.environ.get('MODAL_TASK_ID'),reason='sandbox dependencies live on Modal')
def test_minisweagent_zero_limits_disable_dollars_and_steps():
    import inspect
    from minisweagent.agents.default import DefaultAgent
    source=inspect.getsource(DefaultAgent)
    assert 'cost_limit' in source and 'step_limit' in source
    from polytopia_bot.skirmish.mini_worker import openrouter_model,ModalBashEnvironment
    agent=DefaultAgent(openrouter_model(model_name='fixture'),ModalBashEnvironment(Mock(),0),system_template='fixture',instance_template='{{ task }}',step_limit=0,cost_limit=0)
    assert agent.config.step_limit==agent.config.cost_limit==0


@pytest.mark.skipif(not os.environ.get('MODAL_TASK_ID'),reason='agent dependencies live on Modal')
def test_large_commands_are_transferred_as_files_not_process_arguments():
    import time
    from polytopia_bot.skirmish.mini_worker import ModalBashEnvironment
    sandbox=Mock();sandbox.exec.return_value.stdout.read.return_value=json.dumps(dict(output='ok',returncode=0))
    command='#'+('x'*300000)+'\nprintf ok'
    result=ModalBashEnvironment(sandbox,time.time()+120).execute(dict(command=command))
    payload,path=sandbox.filesystem.write_text.call_args.args
    assert json.loads(payload)[0]==command
    argv=sandbox.exec.call_args.args
    assert path in argv and sum(len(a) for a in argv)<65536
    assert command not in argv and result['returncode']==0


def test_protocol_guard_rejects_mixed_source_and_budget_before_hidden_maps():
    from polytopia_bot.skirmish.cohort import validate_run_protocol
    from polytopia_bot.skirmish.protocol import SCHEMA,EVALUATION
    c=authoring_cells('protocol-test',manifest(),'image')[0]
    p=dict(schema=SCHEMA,model=c['model'],provider='openrouter',harness='mini-swe-agent',
        harness_version='2.4.6',track=c['track'],source_files={'engine':'fixed'},coding_image_id='image',
        parameters=c['config']['parameters'],evaluation=EVALUATION,smoke=False,stage_seconds=2400,revision_rounds=0)
    assert validate_run_protocol(c,p,{'engine':'fixed'},'image')
    p['stage_seconds']=1200
    with pytest.raises(ValueError,match='stage_seconds'):validate_run_protocol(c,p,{'engine':'fixed'},'image')
    p['stage_seconds']=2400
    with pytest.raises(ValueError,match='source_files'):validate_run_protocol(c,p,{'engine':'changed'},'image')
