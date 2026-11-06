from polytopia_bot.learning.progress_diagnostics import public_progress


def row(kind,source=0,target=1,known=(0,),owner=1):
    obs=dict(player=1,visible_state=dict(tiles=[dict(tile=i,known=True) for i in known],cities=[dict(tile=0,owner=owner)]))
    return dict(policy='model',observation=obs,after=[obs],selected_command=dict(kind=kind,source=source,target=target))


def test_reversals_and_no_discovery_streak_are_descriptive():
    result=public_progress([row(1),row(10),row(1,1,0),row(10),row(10)],1)
    assert result['movement_actions']==2 and result['consecutive_movement_reversals']==1
    assert result['newly_seen_tiles']==0 and result['longest_end_turn_streak_without_discovery_or_capture']==3


def test_discovery_or_capture_resets_streak_and_other_views_are_ignored():
    rows=[row(10),row(10),row(1,known=(0,1)),row(10,known=(0,1)),row(8,known=(0,1)),row(10,known=(0,1))]
    rows[-1]['after'].append(dict(player=2,visible_state=dict(tiles=[dict(tile=99,known=True)],cities=[])))
    result=public_progress(rows,1)
    assert result['newly_seen_tiles']==1 and result['captures']==1
    assert result['longest_end_turn_streak_without_discovery_or_capture']==2
