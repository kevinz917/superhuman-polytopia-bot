import gzip,json
from dataclasses import replace
from polytopia_bot.environment import Transition
from polytopia_bot.learning.data import fixture_observation
from polytopia_bot.learning.interface import Generation
from polytopia_bot.learning.fast_gameplay import play_one

class Game:
    def __init__(self):self.version=0
    def observe(self,p):return replace(fixture_observation(),state_version=self.version,player=p)
    def reset(self,**kw):self.version=0;return self.observe(1)
    def snapshot(self):return str(self.version).encode()
    def restore(self,data):self.version=int(data);return self.observe(1)
    def step(self,action_id,expected_state_version):
        assert expected_state_version==self.version
        self.version+=1;done=self.version==35
        return Transition(self.observe(1),None if done else 1,{1:1. if done else 0.,2:-1. if done else 0.},done,False,'elimination' if done else None)
    def close(self):pass

class Generator:
    def __init__(self,crash=None):self.n=0;self.crash=crash
    def batch(self,messages):
        self.n+=1
        if self.n==self.crash:raise RuntimeError('simulated interruption')
        return [Generation('{"action_id":"a0"}',[1],[2])]

def job():return dict(id='game',seed=1,board_hash='board',seat=1,candidate='original',strategy='',opponent='weak')

def test_resume_discards_uncommitted_trace_tail_without_duplicate_steps(tmp_path):
    import pytest
    with pytest.raises(RuntimeError,match='interruption'):
        play_one(tmp_path,job(),Generator(crash=20),seconds_limit=100,environment_factory=Game)
    saved=json.loads((tmp_path/'game.resume.json').read_text())
    assert saved['counters']['decisions']==16
    result=play_one(tmp_path,job(),Generator(),seconds_limit=100,environment_factory=Game)
    assert result['status']=='native' and result['decisions']==35 and result['model_decisions']==35
    trace=[json.loads(x) for x in gzip.open(tmp_path/'game.jsonl.gz','rt')]
    assert len(trace)==36
    assert [r['observation']['state_version'] for r in trace[:-1]]==list(range(35))
    assert play_one(tmp_path,job(),Generator(crash=1),seconds_limit=100,environment_factory=Game)==result


def test_invalid_responses_preserve_three_attempt_forfeit(tmp_path):
    class Invalid:
        def batch(self,m):return [Generation('bad',[1],[2])]
    result=play_one(tmp_path,job(),Invalid(),seconds_limit=100,environment_factory=Game)
    assert result['status']=='forfeit' and result['attempts']==3 and result['decisions']==0


def test_early_game_cap_is_truncation_and_counts_actions(tmp_path):
    result=play_one(tmp_path,job(),Generator(),seconds_limit=100,action_limit=10,environment_factory=Game)
    assert result['status']=='truncated' and result['reason']=='research_action_limit'
    assert result['decisions']==10 and sum(result['model_command_counts'].values())==10


def test_continuous_scheduler_routes_concurrent_games_and_refills_workers(tmp_path):
    from collections import deque
    from polytopia_bot.learning.fast_gameplay import play_continuous
    jobs=deque([{**job(),'id':f'g{i}'} for i in range(4)])
    class Engine:
        def __init__(self):self.pending={};self.sequence=0
        def submit(self,m):
            key=str(self.sequence);self.sequence+=1;self.pending[key]=Generation('{"action_id":"a0"}',[1],[2]);return key
        def step(self):
            result=list(self.pending.items());self.pending.clear();return result
    g=Engine()
    result=play_continuous(tmp_path,lambda:jobs.popleft() if jobs else None,g,lanes=2,seconds_limit=100,environment_factory=Game)
    assert result['finished_games']==4 and result['complete']
    assert g.sequence==140 and all(r['decisions']==35 for r in result['games'])
    assert len({r['id'] for r in result['games']})==4
