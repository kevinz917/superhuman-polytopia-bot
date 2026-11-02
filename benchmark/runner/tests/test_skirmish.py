"""Benchmark boundary and scoring regression checks. Real games execute only on Modal."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import pytest

from polytopia_bot.skirmish.common import Budget, REFERENCE, identifier, paired_schedule, validate_seed
from polytopia_bot.skirmish.tournament import fit_elo, summarize
from polytopia_bot.skirmish.providers import ChatProvider, ResponsesProvider, AnthropicProvider


def row(a='a', b=REFERENCE, seed=700001, seat=1, winner=1, status='native'):
    return dict(a=a, b=b, seed=seed, seat=seat, winner=winner, status=status,
                result='unresolved' if winner is None else 'win' if winner == seat else 'loss')


def test_schedule_has_both_seats_every_pair_map():
    jobs = paired_schedule(['a', 'b', REFERENCE], [700001, 700002])
    assert len(jobs) == 12
    for a, b in [('a', 'b'), ('a', REFERENCE), ('b', REFERENCE)]:
        assert {(j['seed'], j['seat']) for j in jobs if j['a'] == a and j['b'] == b} == {
            (700001, 1), (700001, 2), (700002, 1), (700002, 2)}


def test_duplicate_entrants_and_maps_rejected():
    with pytest.raises(ValueError): paired_schedule(['a', 'a'], [700001])
    with pytest.raises(ValueError): paired_schedule(['a', 'b'], [])
    with pytest.raises(ValueError): paired_schedule(['a', 'b'], [700001, 700001])


@pytest.mark.parametrize('value', ['../a', '/tmp/foo', 'a.b', '', 'a/b', 'a b'])
def test_paths_are_identifiers(value):
    with pytest.raises(ValueError): identifier(value)


def test_partitions_do_not_overlap():
    assert validate_seed(500000, 'practice') == 500000
    assert validate_seed(1_000_000_000, 'final') == 1_000_000_000
    with pytest.raises(ValueError): validate_seed(500000, 'final')
    with pytest.raises(ValueError): validate_seed(700000, 'practice')


def test_budget_validation():
    Budget().validate()
    with pytest.raises(ValueError): Budget(seconds=-1).validate()
    with pytest.raises(ValueError): Budget(fuel=0).validate()


def test_elo_balanced_is_zero_and_all_wins_finite():
    rows = [row(winner=1), row(seat=2, winner=1)]
    assert abs(fit_elo(rows, ['a', REFERENCE])['a']) < 1e-8
    ratings = fit_elo([row()] * 20, ['a', REFERENCE])
    assert 100 < ratings['a'] < 1500 and ratings[REFERENCE] == 0


def test_unresolved_games_do_not_become_draws():
    rows = [row(), row(seed=700002, winner=None, status='unresolved')]
    assert fit_elo(rows, ['a', REFERENCE]) == fit_elo(rows[:1], ['a', REFERENCE])
    report = summarize(rows, ['a', REFERENCE], bootstrap=10)
    assert report['unresolved'] == 1 and report['resolved'] == 1
    assert report['pairs']['a__' + REFERENCE]['unresolved'] == 1


def test_forfeit_counts_as_loss_separately():
    report = summarize([row(winner=2, status='forfeit')], ['a', REFERENCE], bootstrap=0)
    a = next(r for r in report['ladder'] if r['entrant'] == 'a')
    assert a['losses'] == a['forfeits'] == 1 and a['wins'] == 0


def test_disconnected_entrant_has_no_rating():
    assert fit_elo([row()], ['a', REFERENCE, 'b'])['b'] is None


def test_chat_preserves_reasoning_and_tool_ids():
    out = {'choices': [{'message': {'role': 'assistant', 'content': None,
           'reasoning_details': [{'type': 'reasoning.encrypted', 'data': 'opaque'}],
           'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'compile', 'arguments': '{}'}}]}}]}
    p = ChatProvider(dict(base_url='https://example.test/v1', model='exact-model', key_env='TEST_KEY'), 'system', [])
    with patch.dict(os.environ, TEST_KEY='secret'), patch('polytopia_bot.skirmish.providers.request', return_value=out):
        turn = p.next('start', 1)
    assert p.messages[-1]['reasoning_details'] == out['choices'][0]['message']['reasoning_details']
    p.result(turn['calls'][0], {'ok': True})
    assert p.messages[-1]['tool_call_id'] == 'c1'


def test_responses_preserves_output_items_and_call_id():
    out = {'output': [{'type': 'reasoning', 'id': 'r1', 'summary': []},
           {'type': 'function_call', 'call_id': 'c1', 'name': 'compile', 'arguments': '{}'}]}
    p = ResponsesProvider(dict(base_url='https://example.test/v1', model='exact', key_env='TEST_KEY'), 'system', [])
    with patch.dict(os.environ, TEST_KEY='secret'), patch('polytopia_bot.skirmish.providers.request', return_value=out):
        turn = p.next('start', 1)
    p.result(turn['calls'][0], {'ok': True})
    assert p.items[1]['type'] == 'reasoning' and p.items[-1]['call_id'] == 'c1'


def test_anthropic_groups_tool_results():
    out = {'content': [{'type': 'tool_use', 'id': 'c1', 'name': 'compile', 'input': {}},
                       {'type': 'tool_use', 'id': 'c2', 'name': 'compile', 'input': {}}]}
    p = AnthropicProvider(dict(base_url='https://example.test/v1', model='exact', key_env='TEST_KEY'), 'system', [])
    with patch.dict(os.environ, TEST_KEY='secret'), patch('polytopia_bot.skirmish.providers.request', return_value=out):
        turn = p.next('start', 1)
        for call in turn['calls']: p.result(call, {'ok': True})
        p.next(None, 1)
    assert [x['tool_use_id'] for x in p.messages[-2]['content']] == ['c1', 'c2']


@pytest.mark.skipif(not os.environ.get('MODAL_TASK_ID'), reason='WASI execution stays on Modal')
def test_wasi_filesystem_and_environment_are_unavailable(tmp_path):
    import wasmtime
    # Import only WASI, try to read environment and preopen fd 3. Neither has a capability.
    wat = '''(module
      (import "wasi_snapshot_preview1" "environ_sizes_get" (func $env (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_prestat_get" (func $fd (param i32 i32) (result i32)))
      (memory (export "memory") 1)
      (func (export "check") (result i32)
        (drop (call $env (i32.const 0) (i32.const 4)))
        (if (i32.load (i32.const 0)) (then unreachable))
        (call $fd (i32.const 3) (i32.const 8))))'''
    e = wasmtime.Engine()
    s = wasmtime.Store(e)
    s.set_wasi(wasmtime.WasiConfig())
    linker = wasmtime.Linker(e)
    linker.define_wasi()
    module = wasmtime.Module(e, wat)
    inst = linker.instantiate(s, module)
    assert inst.exports(s)['check'](s) != 0


def test_tools_reject_stale_build_and_reference_path_escape(tmp_path):
    from polytopia_bot.skirmish.tools import AuthoringTools
    tools = AuthoringTools(None, tmp_path, Budget(), None, time.monotonic()+100)
    tools.source = 'changed'
    tools.compiled_source = 'old'
    tools.artifact = Path('unused.wasm')
    with pytest.raises(ValueError, match='compile the current source'):
        tools.campaign(1)
    for path in ('/reference/../results/key', '/results/bot.wasm', '/reference/engine-rs/src/../key.rs'):
        with pytest.raises(ValueError): tools.dispatch('read_file', {'path': path})
    with pytest.raises(ValueError): tools.dispatch('review', {'game_id': '../x', 'offset': 0, 'limit': 1})


def test_no_tools_execute_after_deadline(tmp_path):
    from polytopia_bot.skirmish.tools import AuthoringTools
    tools = AuthoringTools(None, tmp_path, Budget(), None, time.monotonic()-1)
    with pytest.raises(TimeoutError): tools.dispatch('write_source', {'source': 'anything'})


@pytest.mark.skipif(not os.environ.get('MODAL_TASK_ID'), reason='WASI execution stays on Modal')
def test_guest_timeout_and_invalid_action_are_forfeits(tmp_path):
    import wasmtime
    from polytopia_bot.environment import Observation, LegalAction
    from polytopia_bot.skirmish.game import WasiBot, BotFault
    infinite = tmp_path/'loop.wasm'
    infinite.write_bytes(wasmtime.wat2wasm('(module (func (export "_start") (loop $x (br $x))))'))
    obs = Observation(1, 0, {'board_size':14}, {}, [LegalAction('1',{})])
    guest = WasiBot(infinite, 100000, .1)
    try:
        with pytest.raises(BotFault, match='fuel_exhausted|decision_timeout'):
            guest.select(obs)
    finally:
        guest.close()


def test_revision_budget_counts_changed_sources_including_broken_edits(tmp_path):
    from polytopia_bot.skirmish.tools import AuthoringTools
    class FS:
        def write_text(self, source, path): pass
    class SB:
        filesystem = FS()
    tools = AuthoringTools(SB(),tmp_path,Budget(constraint='revisions',max_revisions=2),None,time.monotonic()+60)
    tools.source = 'starter'
    tools.write('version1')
    tools.write('version1')
    assert tools.revisions == 1
    tools.write('broken version')
    assert tools.revisions == 2
    with pytest.raises(ValueError,match='revision budget exhausted'): tools.write('version3')
    assert tools.budget_status()['revisions_remaining'] == 0
    assert len((tmp_path/'revisions.jsonl').read_text().splitlines()) == 2


def test_revision_modes_need_explicit_allowance():
    with pytest.raises(ValueError): Budget(constraint='revisions').validate()
    Budget(constraint='combined',max_revisions=8).validate()


def test_study_assignments_are_balanced_randomized_and_reproducible():
    from polytopia_bot.skirmish.conditions import build_study, condition
    spec = dict(study_id='test',assignment_seed=17,replicates=3,framings=['neutral','evaluative'],
                models=[dict(id='m',config=dict(model='exact-model'))],
                budget_profiles=[dict(id='t',budget=dict(seconds=60))])
    plan = build_study(spec)
    assert plan == build_study(spec)
    assert len(plan['assignments']) == 6
    for rep in range(3):
        rows = [r for r in plan['assignments'] if r['replicate']==rep]
        assert {r['framing'] for r in rows} == {'neutral','evaluative'}
        assert rows[0]['config']['budget'] == rows[1]['config']['budget']
    assert condition({'framing':'neutral'})['text'] != condition({'framing':'evaluative'})['text']
    with pytest.raises(ValueError): condition({'framing':'unknown'})


def test_study_summary_reports_attrition_and_uses_runs_not_games():
    from polytopia_bot.skirmish.conditions import summarize_study
    plan={'assignments':[dict(run_id='a',model_label='m',budget_profile='t',framing='neutral',replicate=0),
                         dict(run_id='b',model_label='m',budget_profile='t',framing='evaluative',replicate=0)]}
    completed=[dict(run_id='a',metadata={'api_cost_usd':.01},result={'status':'completed'}),dict(run_id='b')]
    tournaments={'0':{'ladder':[{'entrant':'a','elo':-50}]}}
    report=summarize_study(plan,completed,tournaments)
    assert report['contrasts'][0]['paired_authoring_runs']==0
    assert report['contrasts'][0]['paired_run_bootstrap_ci95'] is None
    assert sum(c['valid_submissions'] for c in report['cells'])==1
    assert sum(c['allocated'] for c in report['cells'])==2


def test_api_request_has_hard_deadline_and_no_secret_in_argv():
    import subprocess
    from polytopia_bot.skirmish.providers import request
    with patch('subprocess.run',side_effect=subprocess.TimeoutExpired(['python'],1)) as mocked:
        with pytest.raises(TimeoutError,match='model request deadline'):
            request('https://example.test','secret-value',{},.1)
    assert all('secret-value' not in str(arg) for arg in mocked.call_args.args)
    assert 'secret-value' in mocked.call_args.kwargs['input']
