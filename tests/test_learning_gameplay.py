from dataclasses import replace
import gzip
import json
from polytopia_bot.environment import Transition
from polytopia_bot.learning.data import fixture_observation
from polytopia_bot.learning.gameplay import play_games
from polytopia_bot.learning.interface import Generation


class Game:
    instances = []

    def __init__(self):
        self.version, self.steps, self.closed = 0, [], False
        self.instances.append(self)

    def reset(self, *, seed, settings):
        return self.observe(1)

    def observe(self, player):
        obs = fixture_observation()
        visible = {**obs.visible_state, 'cities': [dict(tile=4, owner=1)]}
        return replace(obs, player=player, state_version=self.version, visible_state=visible)

    def snapshot(self):
        return b'private fake state'

    def step(self, *, action_id, expected_state_version):
        assert expected_state_version == self.version
        self.steps.append(action_id)
        self.version += 1
        done = self.version == 3
        return Transition(self.observe(1 if done else 2), None if done else 2 if self.version == 1 else 1,
                          {1: 1., 2: -1.} if done else {1: 0., 2: 0.}, done, False,
                          'elimination' if done else None)

    def close(self):
        self.closed = True


def jobs(n=1):
    return [dict(id=f'g{i}', seed=i, board_hash=f'board{i}', seat=1,
                 candidate='original', strategy='', opponent='weak') for i in range(n)]


def test_repair_is_bounded_and_no_scripted_move_replaces_model(tmp_path):
    class Invalid:
        def __init__(self):self.messages=[]
        def batch(self, messages):
            self.messages.extend(messages)
            return [Generation('bad', [1], [2]) for _ in messages]
    generator = Invalid()
    result = play_games(tmp_path/'run', jobs(2), generator, environment_factory=Game)
    assert result['complete']
    assert all(g['status']=='forfeit' and g['attempts']==3 and g['decisions']==0 for g in result['games'])
    assert all(not g.steps and g.closed for g in Game.instances[-2:])
    assert [len(m) for m in generator.messages] == [2,2,4,4,6,6]


def test_repaired_choice_executes_once_and_next_decision_has_fresh_context(tmp_path):
    class Repaired:
        def __init__(self):self.messages=[]
        def batch(self, messages):
            self.messages.extend(messages)
            return [Generation('bad' if len(self.messages)==1 else '{"action_id":"a0"}', [1], [2])]
    generator = Repaired()
    result = play_games(tmp_path/'run', jobs(), generator, environment_factory=Game)
    game = result['games'][0]
    assert game['status']=='native' and game['reward']==1.
    assert (game['decisions'],game['model_decisions'],game['attempts'],game['first_attempt_valid'])==(3,2,3,1)
    assert [len(m) for m in generator.messages] == [2,4,2]
    traces = [json.loads(s) for s in gzip.open(tmp_path/'run/g0.jsonl.gz','rt')]
    model = [r for r in traces if r.get('policy')=='model']
    assert len(model)==2 and len(model[0]['diagnostic'])==2
    assert Game.instances[-1].closed


def test_empty_time_budget_is_not_reported_as_game_completion(tmp_path, monkeypatch):
    clock = iter([0.,2.,3.])
    monkeypatch.setattr('polytopia_bot.learning.gameplay.time.monotonic', lambda: next(clock))
    result = play_games(tmp_path/'run', jobs(), object(), seconds_limit=1, environment_factory=Game)
    assert not result['complete'] and result['unstarted_jobs']==['g0']
