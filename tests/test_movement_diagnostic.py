import pytest
from polytopia_bot.learning.movement_diagnostic import score_choice


def command(kind, source=0, target=1):
    return dict(kind=kind,source=source,target=target,parameter=0)


def test_global_ties_accept_alternative_moves_and_other_families():
    a,b,c=command(1),command(1,target=2),command(2)
    scores=[dict(command=x,score=10) for x in (a,b,c)]
    result=score_choice(scores,a,b)
    assert result['tie_aware'] and not result['exact']
    assert not result['movement_required_for_top_score']
    assert score_choice(scores,a,c)['tie_aware']
    assert not score_choice(scores,a,c)['selected_movement']


def test_legal_but_lower_scored_move_is_not_accepted():
    a,b=command(1),command(1,target=2)
    result=score_choice([dict(command=a,score=10),dict(command=b,score=3)],a,b)
    assert result['selected_movement'] and not result['tie_aware']
    assert result['movement_required_for_top_score']
    assert result['score_gap']==7


def test_invalid_response_is_failure_and_nonoptimal_target_is_rejected():
    a,b=command(1),command(1,target=2)
    scores=[dict(command=a,score=10),dict(command=b,score=3)]
    assert not score_choice(scores,a,None)['tie_aware']
    with pytest.raises(ValueError,match='not teacher-optimal'):
        score_choice(scores,b,a)
