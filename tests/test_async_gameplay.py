import asyncio
import importlib.util
import json
from pathlib import Path

from polytopia_bot.learning.async_gameplay import play_async_games
from polytopia_bot.learning.interface import Generation
from test_learning_gameplay import Game, jobs


def test_independent_games_do_not_wait_for_long_thinking(tmp_path):
    class G:
        pending = {}
        def __init__(self):
            self.other_finished = asyncio.Event()
        async def generate(self, messages, *, job, decision):
            if job['id']=='g0' and decision==1:
                await asyncio.wait_for(self.other_finished.wait(), timeout=2)
                text = 'reasoning '*5000+'</think>{"action_id":"a0"}'
                tokens = list(range(5000))
            else:
                if job['id']=='g1' and decision==2:
                    self.other_finished.set()
                text, tokens = '</think>{"action_id":"a0"}', [1]
            return Generation(text,[1],tokens),dict(finish_reason='stop')
    async def run():
        return await play_async_games(tmp_path/'games',jobs(2),G(),thinking=True,
            environment_factory=Game,opponent_factory=lambda _:Opponent())
    class Opponent:
        def select(self,obs):
            from dataclasses import dataclass
            @dataclass
            class Choice:
                action: object
            return Choice(obs.legal_actions[0])
    r=asyncio.run(run())
    assert r['complete'] and all(x['status']=='native' for x in r['games'])
    assert [g['id'] for g in r['games']]==['g1','g0']
    assert all(g['model_decisions']==2 and g['attempts']==2 for g in r['games'])


def test_only_native_context_limit_is_incomplete(tmp_path):
    class G:
        pending={}
        async def generate(self,*args,**kwargs):
            return Generation('still thinking',[1],[2]),dict(finish_reason='length')
    r=asyncio.run(play_async_games(tmp_path/'games',jobs(),G(),thinking=True,
        environment_factory=Game,opponent_factory=lambda _:object()))
    assert not r['complete']
    assert r['games'][0]['status']=='context_incomplete'
    assert r['games'][0]['decisions']==0 and r['games'][0]['reward']==0


def test_natural_eos_without_valid_action_is_forfeit_not_budget(tmp_path):
    class G:
        pending={}
        async def generate(self,*args,**kwargs):
            return Generation('bad',[1],[2]),dict(finish_reason='stop')
    r=asyncio.run(play_async_games(tmp_path/'games',jobs(),G(),thinking=True,
        environment_factory=Game,opponent_factory=lambda _:object()))
    assert r['games'][0]['status']=='forfeit'


def test_repetition_abort_never_executes_partial_json_or_counts_as_native_loss(tmp_path):
    class G:
        pending={}
        async def generate(self,*args,**kwargs):
            return Generation('{"action_id":"a0"}' + '-1,'*300,[1],[2]),dict(
                finish_reason='repetition_abort',stop_reason='exact_repetition_period_2')
    r=asyncio.run(play_async_games(tmp_path/'games',jobs(),G(),thinking=True,
        environment_factory=Game,opponent_factory=lambda _:object()))
    g=r['games'][0]
    assert g['status']=='generation_incomplete' and g['reward']==0
    assert g['decisions']==0 and not r['complete']


def test_sampling_explicitly_disables_budgets_and_seed_is_order_independent():
    path=Path(__file__).resolve().parents[1]/'scripts/benchmark_thinking_unbounded.py'
    spec=importlib.util.spec_from_file_location('benchmark_unbounded',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=dict(sampling=dict(temperature=1.),seed=17)
    a=module.sampling_options(p,dict(rng_index=3),7)
    b=module.sampling_options(p,dict(rng_index=3),7)
    assert a==b and a['max_tokens'] is None and a['thinking_token_budget'] is None
    assert not a['ignore_eos']
    assert module.sampling_options(p,dict(rng_index=4),7)['seed']!=a['seed']


def test_qwen_engine_rejects_local_execution_before_importing_model_libraries(monkeypatch,tmp_path):
    import pytest
    monkeypatch.delenv('MODAL_TASK_ID', raising=False)
    path=Path(__file__).resolve().parents[1]/'scripts/benchmark_thinking_unbounded.py'
    spec=importlib.util.spec_from_file_location('benchmark_unbounded_guard',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match='must run on Modal'):
        module.Generator({},True,tmp_path)
