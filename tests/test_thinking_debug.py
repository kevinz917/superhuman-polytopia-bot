import json

from polytopia_bot.learning.data import fixture_observation
from polytopia_bot.learning.decision_harness import render_decision
from polytopia_bot.learning.thinking_debug import (
    render_sparse_decision, repeated_suffix, restore_dense_body, sparse_messages,
)


def test_sparse_terrain_preserves_entire_public_input_and_action_catalog():
    original = render_decision(fixture_observation())
    repaired = render_sparse_decision(fixture_observation())
    assert restore_dense_body(json.loads(repaired.messages[1]['content'])) == json.loads(original.messages[1]['content'])
    assert repaired.catalog == original.catalog
    assert repaired.public == original.public
    assert 'Terrain is a row-major array' not in repaired.messages[0]['content']
    assert 'do not expand' in repaired.messages[0]['content']
    assert 'One response selects one action' in repaired.messages[0]['content']
    assert sparse_messages(original.messages) != original.messages
    assert isinstance(json.loads(original.messages[1]['content'])['visible']['tiles']['terrain'], list)


def test_repetition_detects_numeric_loop_but_not_short_repeated_tables():
    assert repeated_suffix([4838, 16] * 512) == 2
    assert repeated_suffix([4838, 16] * 511) == 0
    assert repeated_suffix([4838, 16] * 196) == 0  # A whole finite board is not a loop.
    assert repeated_suffix(list(range(1024))) == 0
    assert repeated_suffix([4838, 16] * 511 + [99, 100]) == 0


def test_repetition_catches_prose_periods_beyond_old_numeric_guard():
    paragraph=list(range(71))
    loop=(paragraph*20)[-1024:]
    assert repeated_suffix(loop,max_period=16)==0
    assert repeated_suffix(loop)==71
    assert repeated_suffix(paragraph*8)==0  # Not a full evidence window.
    assert repeated_suffix(loop[:-1]+[-1])==0
    assert repeated_suffix(list(range(256))*4,max_period=256)==0  # Fewer than eight cycles.


def test_named_fields_preserve_nulls_rules_state_and_every_legal_action():
    from polytopia_bot.learning.thinking_readable import readable_messages, restore_sparse_body, records
    sparse = render_sparse_decision(fixture_observation())
    messages = readable_messages(sparse.messages)
    body = json.loads(messages[1]['content'])
    assert restore_sparse_body(body) == json.loads(sparse.messages[1]['content'])
    assert all('tile' in row for row in body['visible']['units']['records'])
    table = dict(columns=['home', 'tile', 'owner'], rows=[[32, 31, 1], [None, 44, 0]])
    assert records(table)['records'][0] == dict(home=32, tile=31, owner=1)
    assert records(records(table), inverse=True) == table
    assert set(body['action_names']) == set(sparse.catalog)
    assert 'a17' not in messages[0]['content']
    assert 'tile plus one' in messages[0]['content']


def test_output_schema_preserves_all_actions_and_forbids_extra_keys():
    from polytopia_bot.learning.thinking_readable import action_schema, readable_messages
    prompt = render_sparse_decision(fixture_observation())
    schema = action_schema(prompt.messages)
    assert schema == action_schema(readable_messages(prompt.messages))
    assert schema['properties']['action_id']['enum'] == list(prompt.catalog)
    assert schema['required'] == ['action_id']
    assert schema['additionalProperties'] is False
