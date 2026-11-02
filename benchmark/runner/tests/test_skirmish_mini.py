"""Protocol boundaries for the standard coding harness and controlled revision stages."""
import json
import os
from pathlib import Path
import time
from unittest.mock import Mock

import pytest
from polytopia_bot.skirmish.mini import TRACKS, evaluation_schedule, track_config


def cfg(track='oneshot-20m', **kw):
    return dict(provider='openrouter', track=track, **kw)


def test_official_tracks_have_equal_twenty_minute_coding_allowance():
    first = track_config(cfg())
    revision = track_config(cfg('feedback-3r'))
    assert first == dict(stage_seconds=1200, revision_rounds=0)
    assert revision == dict(stage_seconds=300, revision_rounds=3)
    assert first['stage_seconds'] == revision['stage_seconds']*(revision['revision_rounds']+1)
    with pytest.raises(ValueError): track_config(cfg(stage_seconds=10))
    with pytest.raises(ValueError): track_config(dict(provider='responses',track='oneshot-20m'))


def test_smokes_are_explicitly_distinguishable_and_bounded():
    assert track_config(cfg(smoke=True,smoke_seconds=30))['stage_seconds'] == 30
    with pytest.raises(ValueError): track_config(cfg(smoke=True,smoke_seconds=1200))


def test_three_map_families_both_seats_every_evaluation():
    jobs = evaluation_schedule([500001,500002,500003])
    assert len(jobs)==6
    assert {(j['map_type'],j['seat']) for j in jobs} == {
        (kind,seat) for kind in ('drylands','lakes','continents') for seat in (1,2)}
    with pytest.raises(ValueError): evaluation_schedule([1,1,2])


def test_early_submit_tool_call_is_closed_before_next_stage():
    from polytopia_bot.skirmish.mini_worker import continued_history
    messages=[dict(role='system',content='build a bot'),
              dict(role='assistant',content=None,reasoning_details=[{'opaque':'preserve'}],tool_calls=[
                  dict(id='compile',type='function',function=dict(name='bash',arguments='{}')),
                  dict(id='submit',type='function',function=dict(name='bash',arguments='{}'))]),
              dict(role='tool',tool_call_id='compile',content='compiled'),
              dict(role='exit',content='',extra=dict(exit_status='Submitted'))]
    resumed=continued_history(messages)
    assert [m['tool_call_id'] for m in resumed if m['role']=='tool']==['compile','submit']
    assert resumed[1]['reasoning_details']==[{'opaque':'preserve'}]
    assert messages[-1]['role']=='exit'  # immutable prior trajectory
    assert continued_history(resumed)==resumed  # no duplicate receipts


@pytest.mark.skipif(not os.environ.get('MODAL_TASK_ID'),reason='upstream agent dependencies live on Modal')
@pytest.mark.parametrize('usage', [dict(cost=0), {}, dict(cost=None), dict(cost='unavailable'), None])
def test_billing_metadata_never_rejects_a_model_tool_response(usage):
    from unittest.mock import patch
    from polytopia_bot.skirmish.mini_worker import openrouter_model
    model=openrouter_model(model_name='fixture')
    response=dict(choices=[dict(finish_reason='tool_calls', message=dict(role='assistant',content=None,
        tool_calls=[dict(id='billing-fixture',type='function',function=dict(name='bash',
            arguments=json.dumps(dict(command='printf BILLING_IS_OPTIONAL'))))]))],usage=usage)
    with patch.object(model,'_query',return_value=response):
        result=model.query([model.format_message(role='user',content='Use bash')])
    assert result['extra']['actions'][0]['command']=='printf BILLING_IS_OPTIONAL'
    assert response['usage'] is usage


@pytest.mark.skipif(not os.environ.get('MODAL_TASK_ID'),reason='agent dependencies live on Modal')
def test_output_truncation_feedback_is_visible_to_agent():
    from unittest.mock import patch
    from minisweagent.models.openrouter_model import OpenRouterModel
    from minisweagent.exceptions import FormatError
    from polytopia_bot.skirmish.mini_worker import FORMAT_ERROR_TEMPLATE
    model=OpenRouterModel(model_name='fixture',format_error_template=FORMAT_ERROR_TEMPLATE)
    response=dict(choices=[dict(finish_reason='length',message=dict(role='assistant',content='partial'))],
                  usage=dict(cost=.01))
    with patch.object(model,'_query',return_value=response),pytest.raises(FormatError) as error:
        model.query([])
    assert 'output-token limit' in error.value.messages[0]['content']


def test_rules_book_matches_current_sources():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    # This generator is operator tooling and is not mounted in the runtime image.
    script = root/'scripts/build_skirmish_rules.py'
    if not script.exists(): pytest.skip('operator tooling is outside runtime image')
    spec = importlib.util.spec_from_file_location('rule_book',script)
    mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    text = (root/'skirmish/mini/RULES.md').read_text()
    assert text == mod.build()
    assert 'class GPUEnvironment' not in text
    assert 'engine-rs/src/military.rs' in text
    assert '## src/polytopia_bot/gpu/rules.json' in text


@pytest.mark.skipif(not os.environ.get('MODAL_TASK_ID'),reason='runtime executes on Modal')
def test_expired_stage_never_executes_shell():
    from minisweagent.exceptions import TimeExceeded
    from polytopia_bot.skirmish.mini_worker import ModalBashEnvironment
    sb = Mock()
    env = ModalBashEnvironment(sb,time.time()-1)
    with pytest.raises(TimeExceeded): env.execute({'command':'touch /bot/too-late'})
    sb.exec.assert_not_called()


@pytest.mark.skipif(not os.environ.get('MODAL_TASK_ID'),reason='games execute on Modal')
def test_variant_observations_are_reproducible_and_replay_legal_actions():
    from dataclasses import asdict
    from polytopia_bot.skirmish.common import canonical
    from polytopia_bot.skirmish.game import reset_map
    from polytopia_bot.studio.environment import RustEnvironment
    for kind in ('drylands','lakes','continents'):
        a,b=RustEnvironment(),RustEnvironment()
        try:
            left=reset_map(a,500010,kind);right=reset_map(b,500010,kind)
            for _ in range(8):
                assert canonical(asdict(left))==canonical(asdict(right))
                action=next(x for x in left.legal_actions if x.command['kind']==10)
                for env in (a,b): env.step(action_id=action.action_id,expected_state_version=left.state_version)
                left=a.observe(int(a.state.current_player));right=b.observe(int(b.state.current_player))
        finally: a.close();b.close()
