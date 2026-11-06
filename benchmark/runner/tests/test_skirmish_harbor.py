"""Continuous contract and Harbor integration invariants, without model requests."""
import copy
import json
from pathlib import Path
import tomllib
import pytest
from polytopia_bot.skirmish.continuous import CODING_SECONDS, evaluation_jobs, rewards


def manifest():
    return json.loads(Path('configs/skirmish/continuous-v1.json').read_text())


def maps(count):
    return [dict(map_type=f,seed=1_000_000_001+i*100+j)
            for i,f in enumerate(('drylands','lakes','continents')) for j in range(count)]


def test_one_continuous_budget_and_one_harbor_agent_phase():
    from harbor.models.task.config import TaskConfig
    task = TaskConfig.model_validate(tomllib.loads(Path('harbor/tasks/polytopia-skirmish/task.toml').read_text()))
    assert not task.steps
    assert task.metadata['coding_budget_seconds'] == CODING_SECONDS == 2400
    assert task.agent.timeout_sec > CODING_SECONDS  # outer guard, not authoring allowance
    assert task.verifier.environment_mode.value == 'separate'
    assert task.environment.network_mode.value == task.verifier.environment.network_mode.value == 'no-network'
    assert [str(a.source) for a in task.artifacts] == ['/verify/cutoffs/stage-0','/bot/practice']


def test_every_final_map_opponent_and_seat_counts_and_demo_is_explicit():
    m = manifest()
    assert len(evaluation_jobs(m,maps(10))) == 360
    assert len(evaluation_jobs(m,maps(1),demo=True)) == 36
    with pytest.raises(ValueError, match='unbalanced'):
        evaluation_jobs(m,maps(1))
    bad = maps(1);bad[0]['seed'] = 410000
    with pytest.raises(ValueError, match='partition'):
        evaluation_jobs(m,bad,demo=True)
    m['coding_seconds'] = 600
    with pytest.raises(ValueError, match='2400'):
        evaluation_jobs(m,maps(1),demo=True)


def test_failures_stay_in_denominator_but_infrastructure_never_becomes_zero():
    rows = [dict(job_index=i,audit={'passed':True},result=r,status=s,opponent_label='hero-1v1')
            for i,(r,s) in enumerate([('win','native'),('draw','draw'),('loss','forfeit'),('loss','invalid_submission')])]
    result = rewards(rows,4)
    assert result['reward'] == .375 and result['compilation_loss_rate'] == .25
    assert result['runtime_fault_rate'] == .25
    with pytest.raises(ValueError, match='incomplete'):
        rewards(rows[:3],4)
    rows[0]['result'] = 'unresolved'
    with pytest.raises(RuntimeError, match='infrastructure'):
        rewards(rows,4)


def test_trace_inspection_honors_offset_and_bounds(tmp_path):
    from polytopia_bot.skirmish.practice import inspect_trace
    path = tmp_path/'trace.jsonl'
    path.write_text('\n'.join(json.dumps({'index':i}) for i in range(5))+'\n')
    assert inspect_trace(path,2,2) == [{'index':2},{'index':3}]
    with pytest.raises(ValueError):inspect_trace(path,0,1000)


def test_public_practice_never_resigns_or_discards_army():
    from polytopia_bot.environment import LegalAction,Observation
    from polytopia_bot.skirmish.practice import PublicPracticeBot
    actions = [LegalAction(str(i),dict(kind=k,source=0,target=1,parameter=p))
               for i,(k,p) in enumerate([(11,0),(3,8),(10,0),(9,0)])]
    observation = Observation(1,0,dict(board_size=14,cities=[],tiles=[],units=[]),{},actions)
    assert PublicPracticeBot().select(observation) == '3'
