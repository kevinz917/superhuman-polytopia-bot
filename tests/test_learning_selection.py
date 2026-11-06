import pytest
from polytopia_bot.learning.selection import paired_interval, select_prompt


def sample(name, reward, status='native'):
    return [dict(candidate=name,board_hash=f'b{i}',seat=s,reward=reward,status=status,
                 model_decisions=10,first_attempt_valid=10) for i in range(10) for s in (1,2)]


def test_paired_selection_uses_native_scores_and_keeps_ambiguous_original():
    base = sample('original',-1)
    assert select_prompt(base+sample('revision',1),['original','revision'])['selected']=='revision'
    assert select_prompt(base+sample('revision',-1),['original','revision'])['selected']=='original'
    assert select_prompt(base+sample('revision',1,'truncated'),['original','revision'])['selected']=='original'


def test_pairs_must_match_and_both_seats_are_required():
    a,b = sample('original',-1),sample('revision',1)
    b[0]['board_hash']='other'
    with pytest.raises(ValueError,match='map/seat'):
        paired_interval(a,b)
    with pytest.raises(ValueError,match='both seats'):
        paired_interval(a[::2],a[::2])
    with pytest.raises(ValueError,match='incomplete'):
        select_prompt(a+sample('revision',1)[:-1],['original','revision'])
