"""Counterfactual labels must never overwrite the model's real transitions."""
from dataclasses import asdict
import gzip,importlib.util,json
from pathlib import Path
import pytest
from polytopia_bot.learning.data import fixture_observation
from polytopia_bot.learning.teacher import ScriptedTeacher


def load_script(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/relabel_sft_rollouts.py'
    spec=importlib.util.spec_from_file_location('relabel_cli',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path);return module


def test_teacher_labels_preserve_executed_model_action_and_provenance(tmp_path,monkeypatch):
    module=load_script(tmp_path,monkeypatch)
    obs=fixture_observation();teacher=ScriptedTeacher();preferred=dict(teacher.select(obs).action.command)
    alternative=next(a for a in obs.legal_actions if dict(a.command)!=preferred)
    trace=tmp_path/'source/games/relabel-test.jsonl.gz';trace.parent.mkdir(parents=True)
    private=trace.parent/'private';private.mkdir();(private/'relabel-test.snapshot').write_bytes(b'fixture snapshot')
    rows=[]
    for policy,command in [('model',preferred),('model',dict(alternative.command)),('scripted_opponent',preferred),('model',preferred)]:
        rows.append(dict(observation=asdict(obs),policy=policy,selected_command=command,
                         after=[dict(marker='actual transition')],outcome=dict(terminated=False,truncated=False)))
    rows.append(dict(observation=asdict(obs),policy='model',forfeit=True))
    with gzip.open(trace,'wt') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    output=tmp_path/'converted';output.mkdir();(output/'private').mkdir();(output/'provenance').mkdir()
    game=dict(id='relabel-test',seed=1,board_hash='fixture',seat=1,opponent='weak',status='forfeit',reward=-1,reason='invalid_response_after_two_repairs')
    first,eligible=module.eligible_states(trace,teacher)
    assert first==1 and eligible==[1,3]
    result=module.convert((str(trace),game,str(output),eligible,first))
    converted=[json.loads(line) for line in gzip.open(output/'relabel-test.jsonl.gz','rt')]
    assert result['teacher_decisions']==2 and len(converted)==4
    assert converted[0]['record'] is None and converted[2]['record'] is None
    assert converted[1]['record']['selected_command']==preferred
    assert converted[1]['selection']['action']['command']==dict(alternative.command)
    assert converted[1]['after']==rows[1]['after']
    assert converted[3]['record']['selected_command']==preferred
    provenance=json.loads((output/'provenance/relabel-test.json').read_text())
    assert provenance['omitted_unexecuted_forfeit_states']==1
    assert [r['first_deviation'] for r in provenance['labels']]==[True,False]
    assert all(r['collection_source']=='smoke_on_policy_relabel' for r in provenance['labels'])
    assert (output/'private/relabel-test.snapshot').read_bytes()==b'fixture snapshot'


def test_validation_games_cannot_become_training_labels(tmp_path,monkeypatch):
    module=load_script(tmp_path,monkeypatch)
    source=tmp_path/'source';(source/'combined/games').mkdir(parents=True)
    (source/'protocol.json').write_text(json.dumps(dict(training_only=False,repair_pass=1,final_test=False)))
    (source/'combined/games/report.json').write_text('{}');(tmp_path/'audit.json').write_text('{}')
    with pytest.raises(ValueError,match='training-only'):
        module.relabel(Path('source'),Path('audit.json'),Path('forbidden'))
    assert not (tmp_path/'forbidden').exists()
