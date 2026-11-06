"""Compact full-state prompt: same catalog/IDs as the readable prompt, lossless state round trip.

Set POLYTOPIA_EXPERT_GAMES to a directory of expert-sft ``*.jsonl.gz`` game files to also
check real corpus decisions (run on Modal; see experiments/modal_compact_prompt.py).
"""
import json
import os
from pathlib import Path

import pytest

from polytopia_bot.environment import LegalAction, Observation
from polytopia_bot.learning.compact_prompt import (SYSTEM, STATIC_RULES, VERSION, action_ids, action_schema, compact_messages,
                                                   parse_compact, public_from_readable, readable_from_public,
                                                   render_compact_decision, render_user, without_rules)
from polytopia_bot.learning.data import fixture_observation
from polytopia_bot.learning.interface import ResponseError, canonical, parse_response
from polytopia_bot.learning.thinking_readable import action_schema as readable_schema, render_readable_decision

FLAGS = ('veteran', 'moved', 'attacked', 'poison', 'frozen', 'boosted', 'invisible', 'detect')


def _tile(i, terrain, owner=0, **kw):
    return dict(tile=i, known=True, terrain=terrain, owner=owner, resource=kw.get('resource', 0),
                structure=kw.get('structure', 0), road=kw.get('road', 0), capital=kw.get('capital', 0),
                structure_level=kw.get('structure_level', 0))


def _own_unit(tile, kind, health=10.0, **kw):
    return dict(tile=tile, owner=1, kind=kind, health=health, kills=kw.pop('kills', 0), home=kw.pop('home', 0),
                passenger=kw.pop('passenger', 0), **{f: kw.pop(f, False) for f in FLAGS})


def rich_observation():
    """5x5 position exercising every compact field, diplomatic raw IDs and the rule fallback."""
    n = 5
    tiles = [{'tile': i, 'known': False} for i in range(n * n)]
    for i, terrain in enumerate([3, 5, 4, 1, 2, 3, 3, 6, 7, 8, 5, 3, 3, 4, 5, 3, 3]):
        tiles[i] = _tile(i, terrain)
    tiles[6] = _tile(6, 3, owner=1, structure=1, structure_level=3, capital=1, road=1)
    tiles[7] = _tile(7, 6, owner=1, resource=1)
    tiles[1] = _tile(1, 5, owner=1, structure=12, structure_level=1)
    tiles[2] = _tile(2, 4, owner=1, structure=21, structure_level=2, resource=5)
    tiles[12] = _tile(12, 3, owner=2, structure=1, structure_level=1, capital=2)
    tiles[13] = _tile(13, 4, owner=2, road=3)
    tiles[16] = _tile(16, 3, structure=1)
    tiles[11] = _tile(11, 3, owner=1, structure=28, structure_level=0)
    visible = dict(board_size=n, turn=9, stars=17, technology=[0, 1, 15, 20],
        tiles=tiles,
        units=[_own_unit(6, 2, home=7, kills=2, moved=True, veteran=True),
               _own_unit(7, 15, 7.5, home=0, passenger=12, boosted=True),  # Polytaur: outside the static glossary
               _own_unit(11, 12, 40, home=7),                              # integer health stays an integer
               dict(tile=12, owner=2, kind=9, health=6.0, veteran=True),
               dict(tile=13, owner=2, kind=2, health=10.0, veteran=False)],
        cities=[dict(tile=6, owner=1, level=3, population=4, production=3, progress=-2, connected=True, riot=False,
                     rewards=[0, 0, 0, 1, 0, 0, 0, 2, 0]),
                dict(tile=12, owner=2), dict(tile=16, owner=0)],
        relations=dict(known_players=[False, True], peace=[False, False], embassies_sent=[False, True],
                       embassies_received=[True, False], peace_offers=[False, False]),
        achievements=dict(monuments=[28], built=[28, 24]))
    memory = dict(schema='polytopia-memory-v1', historical=[], recent_actions=[
        dict(command=dict(kind=1, source=5, target=6, parameter=0), turn=8, revision=40),
        dict(command=dict(kind=7, source=-1, target=-1, parameter=20), turn=9, revision=41),
        dict(command=dict(kind=10, source=-1, target=-1, parameter=0), turn=9, revision=42)])
    commands = [(1, 6, 5, 0), (1, 6, 0, 0), (1, 7, 8, 0), (2, 6, 12, 0), (3, 6, -1, 7), (3, 2, -1, 8),
                (3, 2, -1, 21), (3, 2, 1, 20), (3, 2, 0, 19), (3, -1, 1, 2), (3, -1, 1, 4), (4, 6, -1, 2), (4, 6, -1, 15),
                (5, -1, 7, 0), (6, -1, 1, 13), (7, -1, -1, 23), (8, 6, -1, 0), (9, -1, 6, 8), (9, -1, 6, 3),
                (10, -1, -1, 0), (11, -1, -1, 0)]
    legal = [LegalAction(f'engine-{i}', dict(kind=k, source=s, target=t, parameter=p))
             for i, (k, s, t, p) in enumerate(reversed(commands))]
    return Observation(1, 43, visible, memory, legal)


def readable_legal(prompt):
    return {r['action_id']: dict(kind=r['kind'], source=r['source'], target=r['target'], parameter=r['parameter'])
            for r in json.loads(prompt.messages[1]['content'])['legal_actions']['records']}


def check_equivalent(obs):
    old, new = render_readable_decision(obs), render_compact_decision(obs)
    # Same catalog (aN -> engine action), same public state.
    assert new.catalog == old.catalog and new.public == old.public and new.state_version == old.state_version
    user = new.messages[1]['content']
    assert new.messages[0]['content'] == SYSTEM and user.startswith(VERSION)
    parsed = parse_compact(user)
    # State round trip: everything in the observation, the rule glossary being derived data.
    assert canonical(parsed) == canonical(without_rules(new.public))
    # Every legal action of the readable prompt, under the identical ID, with the identical command.
    old_legal = readable_legal(old)
    new_legal = {e['action_id']: e['command'] for e in parsed['legal_actions']}
    assert new_legal == old_legal == {k: dict(a.command) for k, a in new.catalog.items()}
    assert action_ids(new.messages) == list(old_legal)
    assert action_schema(new.messages)['properties']['action_id']['enum'] == readable_schema(old.messages)['properties']['action_id']['enum']
    for aid, action in new.catalog.items():
        text = json.dumps({'action_id': aid}, separators=(',', ':'))
        assert parse_response(text, new, current_version=obs.state_version) == parse_response(text, old, current_version=obs.state_version) == action
    return new


def test_fixture_with_irregular_shapes_round_trips_through_json_lines():
    new = check_equivalent(fixture_observation())
    user = new.messages[1]['content']
    assert 'tile json:' in user and 'unit json:' in user and 'city json:' in user and 'remembered json:' in user


def test_rich_position_round_trips_and_keeps_catalog():
    new = check_equivalent(rich_observation())
    user = new.messages[1]['content']
    assert 'json' not in user                       # every regular shape uses the compact lines
    assert 'extra rules: units' in user and 'Polytaur' in user
    assert 'ability from #2: ' in user and 'ability from 2,0: ' in user
    assert 'hp7.5' in user and 'hp40i' in user and 'rewards workshop resources resources' in user
    # Demolition names the player's own structure it removes (v2.1), and the annotation parses away.
    assert ' demolish_own LumberHut 1,0' in user and 'destroy' not in user
    assert 'demolish_own permanently removes one of YOUR OWN structures' in SYSTEM


def test_unknown_action_is_rejected_by_parse_and_grammar():
    new = render_compact_decision(rich_observation())
    enum = action_schema(new.messages)['properties']['action_id']['enum']
    assert enum == [f'a{i}' for i in range(len(new.catalog))]
    with pytest.raises(ResponseError):
        parse_response(json.dumps({'action_id': f'a{len(enum)}'}), new, current_version=43)


def test_slow_content_precedes_fast_content():
    user = render_compact_decision(rich_observation()).messages[1]['content']
    order = ['terrain:', 'territory:', 'structures:', 'cities:', 'technology:', 'enemy units:', 'your units:',
             'recent actions:', 'turn 9 stars 17 version 43', 'legal actions:']
    positions = [user.index(s) for s in order]
    assert positions == sorted(positions)
    # A unit move changes nothing before the unit lists.
    obs = rich_observation()
    v = json.loads(json.dumps(obs.visible_state))
    v['units'][0]['tile'] = 5
    v['units'].sort(key=lambda u: u['tile'])
    moved = render_compact_decision(Observation(1, 44, v, obs.remembered_state, obs.legal_actions)).messages[1]['content']
    cut = user.index('enemy units:')
    assert moved[:cut] == user[:cut]


def test_static_glossary_is_in_system_prompt_only():
    assert all(name in SYSTEM for name in ('Warrior 2*', 'Forestry T2 after Hunting', 'LumberHut 3*'))
    assert len(STATIC_RULES['units']) == 15
    user = render_compact_decision(fixture_observation()).messages[1]['content']
    assert 'extra rules' not in user and 'Warrior' not in user.split('legal actions:')[0].replace('unit json', '')


def test_readable_round_trip_from_stored_text():
    obs = rich_observation()
    old = render_readable_decision(obs)
    public = public_from_readable(old.messages[1]['content'])
    assert public == old.public
    assert readable_from_public(public) == old.messages


@pytest.mark.skipif(not os.environ.get('POLYTOPIA_EXPERT_GAMES'), reason='needs expert game files (Modal)')
def test_real_expert_decisions():
    from polytopia_bot.learning.expert_data import read_game
    files = sorted(Path(os.environ['POLYTOPIA_EXPERT_GAMES']).glob('*.jsonl.gz'))[:int(os.environ.get('POLYTOPIA_EXPERT_GAME_LIMIT', '20'))]
    assert files
    for path in files:
        header, examples, _ = read_game(path)
        for ex in examples:
            public = public_from_readable(ex['user'])
            messages = readable_from_public(public)
            assert messages[0]['content'] == header['system'] and messages[1]['content'] == ex['user']
            user = compact_messages(public)[1]['content']
            assert 'json' not in user and 'extra rules' not in user
            parsed = parse_compact(user)
            assert canonical(parsed) == canonical(without_rules(public))
            old = {e['action_id']: e['command'] for e in public['legal_actions']}
            assert {e['action_id']: e['command'] for e in parsed['legal_actions']} == old
            assert ex['label'] in old and action_ids(compact_messages(public)) == list(old)
