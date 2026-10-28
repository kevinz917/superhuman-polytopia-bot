from copy import deepcopy
from types import SimpleNamespace
import json
import pytest
from polytopia_bot.studio.service import Studio, action_label, SCHEMA

@pytest.fixture
def fixture(tmp_path):
    studio=Studio.__new__(Studio)
    import threading
    studio.lock=threading.RLock();studio.storage=tmp_path;studio.replays={}
    (tmp_path/'replays').mkdir()
    studio.bot={'name':'test','checkpoint_sha256':'a'*64}
    view=lambda p:{'player':p,'state_version':1,'visible_state':{'board_size':14,'turn':0,'stars':99 if p==2 else 5,'tiles':[{'tile':i,'known':p==2} for i in range(196)],'units':[],'cities':[],'technology':[]}}
    frame={'index':0,'current_player':2,'terminated':False,'truncated':False,
           'views':{'1':view(1),'2':view(2),'all':view(2)},
           'event':{'player':2,'turn':0,'action':{'label':'Research','command':{'kind':7,'parameter':22}},'metrics':{'value':.7},'source':'checkpoint'}}
    game={'id':'a'*32,'mode':'play','seat':1,'frames':[frame],'seed':99,'title':'test','created':0,'busy':False,'error':None,
          'env':SimpleNamespace(state=SimpleNamespace(terminated=False,truncated=False))}
    return studio,game

@pytest.mark.parametrize('requested',['all','2','actor','1'])
def test_live_view_cannot_reveal_opponent(fixture,requested):
    studio,game=fixture
    result=studio.public_frame(game,view=requested)
    assert result['perspective']=='1'
    assert result['observation']['visible_state']['stars']==5
    assert 'views' not in result['frame']
    assert result['event']['action']['command']=={}
    assert result['frame']['event']['action']['command']=={}
    assert 'metrics' not in result['event']
    assert 'seed' not in json.dumps(result)


def test_live_export_has_no_seed_hidden_views_or_metrics(fixture):
    studio,game=fixture
    ident=studio.save(game)['replay_id'];replay=studio.replays[ident]
    assert replay['available_views']==['1']
    assert 'seed' not in replay['provenance']
    assert list(replay['frames'][0]['views'])==['1']
    assert replay['frames'][0]['event']['action']['command']=={}


def test_spectator_can_view_both_and_saved_frames_do_not_grow(fixture):
    studio,game=fixture;game['mode']='watch'
    assert studio.public_frame(game,view='all')['perspective']=='all'
    ident=studio.save(game)['replay_id']
    game['frames'].append(deepcopy(game['frames'][0]))
    assert len(studio.replays[ident]['frames'])==1


def test_terminal_player_can_review_opponent(fixture):
    studio,game=fixture;game['frames'][-1]['terminated']=True
    result=studio.public_frame(game,view='2')
    assert result['perspective']=='2'
    assert result['event']['action']['command']['kind']==7


def test_native_export_import_roundtrip(fixture):
    studio,game=fixture;game['mode']='watch'
    original=studio.save(game)['replay_id']
    imported=studio.import_trace(json.dumps(studio.replays[original]).encode())['replay_id']
    assert studio.replay_frame(original,0,'all')|{'id':None}==studio.replay_frame(imported,0,'all')|{'id':None}


def test_action_labels_match_frozen_reward_abi():
    expected={1:'City wall',2:'Park',3:'Workshop',4:'Explorer',5:'Border growth',6:'Super unit',7:'Resources',8:'Population growth'}
    for parameter,name in expected.items():assert action_label({'kind':9,'parameter':parameter})==name


def test_import_rejects_unknown_board(fixture):
    studio,game=fixture;game['mode']='watch'
    ident=studio.save(game)['replay_id'];r=deepcopy(studio.replays[ident])
    r['frames'][0]['views']['1']['visible_state']['board_size']=10000
    with pytest.raises(ValueError,match='Unsupported board'):studio.import_trace(json.dumps(r).encode())


def test_visible_opponent_attack_gets_only_safe_animation_cue(fixture):
    studio,game=fixture
    before=deepcopy(game['frames'][0]);after=game['frames'][0]
    for frame in (before,after):
        for tile in (10,11):frame['views']['1']['visible_state']['tiles'][tile]['known']=True
    before['views']['1']['visible_state']['units']=[{'tile':10,'owner':2,'kind':2,'health':10},{'tile':11,'owner':1,'kind':2,'health':10}]
    after['views']['1']['visible_state']['units']=[{'tile':10,'owner':2,'kind':2,'health':7},{'tile':11,'owner':1,'kind':2,'health':5}]
    after['index']=1;after['event']['action']['command']={'kind':2,'source':10,'target':11,'parameter':12345}
    game['frames']=[before,after]
    result=studio.public_frame(game)
    assert result['event']['animation']=={'kind':2,'source':10,'target':11}
    assert result['event']['action']['command']=={}
    assert 'metrics' not in result['event']
    assert result['frame']['event']==result['event']
    ident=studio.save(game)['replay_id']
    assert studio.replays[ident]['frames'][1]['event']['animation']==result['event']['animation']
    # Explored terrain alone is not enough: the attacker must be visible.
    before['views']['1']['visible_state']['units'].pop(0)
    assert 'animation' not in studio.public_frame(game)['event']


def test_no_animation_cue_for_hidden_target_or_hidden_purchase():
    from polytopia_bot.studio.service import visible_animation
    view={'visible_state':{'tiles':[{'known':True},{'known':False}],
          'units':[{'tile':0,'owner':2},{'tile':1,'owner':1}]}}
    for kind in (1,2,7):
        assert visible_animation(view,view,{'player':2,'action':{'command':{'kind':kind,'source':0,'target':1}}}) is None
