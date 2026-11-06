from copy import deepcopy
import pytest
from polytopia_bot.learning.smoke_gate import position_comparison, game_comparison


def positions(correct, valid=5000):
    rows=[dict(position_id=str(i),teacher_command={'kind':1},board_hash='b',episode_id='e',family=1,input_tokens=100,
               teacher_agreement=i<correct,first_attempt_valid=i<valid) for i in range(5000)]
    return dict(complete=True,allocated=5000,results=rows)


def test_paired_positions_reject_changed_inputs_and_require_actual_learning():
    base=positions(1000);smoke=positions(1250,4750)
    assert position_comparison(base,smoke)['learning_passed']
    assert position_comparison(base,smoke)['format_passed']
    assert not position_comparison(base,positions(1249))['learning_passed']
    smoke['results'][0]['input_tokens']=101
    with pytest.raises(ValueError,match='inputs differ'):position_comparison(base,smoke)


def game_data():
    jobs=[dict(seed=i//2,board_hash=str(i//2),seat=1+i%2,candidate='original',strategy='',opponent='weak') for i in range(50)]
    report=dict(complete=True,unstarted_jobs=[],elapsed_seconds=100,games=[dict(j,status='native',reward=1,attempts=11,model_decisions=10,first_attempt_valid=9) for j in jobs])
    common=dict(jobs=jobs,config=dict(model_id='pinned',revision='rev',max_length=100,max_new_tokens=32),
                prompt_manifest_hash='p',map_manifest_hash='m',weak_opponent_hash='w',tokenizer_hash='t',chat_template_hash='c')
    return report,dict(common,kind='base',adapter=None),dict(common,kind='smoke',adapter='saved')


def test_incomplete_games_cannot_pass_and_native_losses_count_as_completion():
    base,bp,sp=game_data();smoke=deepcopy(base)
    for row in smoke['games']:row['reward']=-1
    result=game_comparison(base,smoke,bp,sp)
    assert result['completion_passed'] and result['models']['smoke']['wins']==0
    assert result['models']['smoke']['repairs']==50
    smoke['games'][0]['status']='budget_incomplete';smoke['complete']=False
    with pytest.raises(ValueError,match='all 50'):game_comparison(base,smoke,bp,sp)
