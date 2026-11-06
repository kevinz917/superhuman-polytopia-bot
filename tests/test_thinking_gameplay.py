from functools import partial
import pytest
from polytopia_bot.learning.data import fixture_observation
from polytopia_bot.learning.decision_harness import render_decision
from polytopia_bot.learning.interface import ResponseError, Generation
from polytopia_bot.learning.thinking_gameplay import parse_thinking_action
from polytopia_bot.learning.gameplay import play_games
from test_learning_gameplay import Game, jobs


def test_only_final_action_is_parsed_and_unfinished_reasoning_is_not_executed():
    prompt = render_decision(fixture_observation())
    valid = 'Consider {"action_id":"a1"}. </think>\n{"action_id":"a0"}'
    assert parse_thinking_action(valid,prompt,current_version=0,thinking=True)==prompt.catalog['a0']
    for invalid in ['{"action_id":"a0"}', 'Still thinking about {"action_id":"a0"}']:
        with pytest.raises(ResponseError,match='unfinished_thinking'):
            parse_thinking_action(invalid,prompt,current_version=0,thinking=True)
    with pytest.raises(ResponseError,match='invalid_json'):
        parse_thinking_action(valid,prompt,current_version=0,thinking=False)
    with pytest.raises(ResponseError,match='stale_version'):
        parse_thinking_action(valid,prompt,current_version=1,thinking=True)


def test_thinking_game_executes_final_action_without_repairs(tmp_path):
    class G:
        def batch(self,messages):
            assert all('public-action-facts-v1' in m[0]['content'] for m in messages)
            return [Generation('reasoning</think>{"action_id":"a0"}',[1],[2]) for m in messages]
    r = play_games(tmp_path/'games',jobs(),G(),environment_factory=Game,
        prompt_renderer=render_decision,output_parser=partial(parse_thinking_action,thinking=True),max_attempts=1)
    assert r['complete'] and r['games'][0]['status']=='native'
    assert r['games'][0]['attempts']==r['games'][0]['model_decisions']==2


def test_thinking_budget_failure_is_neither_win_nor_game_loss(tmp_path):
    class G:
        def batch(self,m):return [Generation('unfinished reasoning',[1],[2]) for _ in m]
    r = play_games(tmp_path/'games',jobs(),G(),environment_factory=Game,
        prompt_renderer=render_decision,output_parser=partial(parse_thinking_action,thinking=True),max_attempts=1)
    assert not r['complete']
    g = r['games'][0]
    assert g['status']=='budget_incomplete' and g['reason']=='unfinished_thinking'
    assert g['decisions']==0 and g['attempts']==1 and g['reward']==0
