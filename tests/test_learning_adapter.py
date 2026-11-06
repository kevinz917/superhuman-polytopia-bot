from copy import deepcopy
from dataclasses import replace
import json
import jax.numpy as jnp
import numpy as np
import pytest
from polytopia_bot.learning.adapter import GPUEnvironment
from polytopia_bot.learning.interface import canonical, render, public_state
from polytopia_bot.learning.memory import PublicMemory
from polytopia_bot.learning.data import fixture_observation


def test_memory_history_actions_caps_and_idempotency():
    memory = PublicMemory()
    obs = fixture_observation()
    initial = memory.observe(obs)
    assert memory.observe(obs) == initial
    memory.executed(obs, obs.legal_actions[0])
    visible = deepcopy(obs.visible_state)
    visible['tiles'][4] = {'tile': 4, 'known': False}
    visible['units'], visible['cities'] = [], []
    later = replace(obs, state_version=1, visible_state=visible)
    remembered = memory.observe(later)
    assert len(remembered.remembered_state['recent_actions']) == 1
    assert any(r['fact']['tile'] == 4 for r in remembered.remembered_state['historical'])
    public_state(remembered)
    assert PublicMemory.restore(memory.snapshot()).observe(later) == remembered
    assert memory.observe(replace(later, player=2)).remembered_state['recent_actions'] == []
    restored = memory.observe(replace(obs, state_version=2))
    assert restored.remembered_state['historical'] == []
    for revision in range(2, 10):
        current = replace(obs, state_version=revision)
        memory.observe(current)
        memory.executed(current, current.legal_actions[0])
    final = memory.observe(replace(obs, state_version=10))
    assert [r['revision'] for r in final.remembered_state['recent_actions']] == list(range(5, 10))
    with pytest.raises(ValueError, match='regressed'):
        memory.observe(obs)


def test_memory_rejects_hidden_payload_and_future_timestamps():
    obs = fixture_observation()
    for fact in ({'tile': 4, 'known': True, 'seed': 123}, {'tile': 4, 'owner': 2, 'kind': 2, 'health': 10, 'home': 5}):
        memory = dict(schema='polytopia-memory-v1', recent_actions=[], historical=[dict(entity='tiles' if 'known' in fact else 'units', fact=fact, revision=0, last_seen_turn=0)])
        with pytest.raises(ValueError):
            public_state(replace(obs, remembered_state=memory))


def test_memory_retains_most_recent_with_stable_cap():
    obs = fixture_observation()
    visible = dict(board_size=11, turn=0, stars=5, technology=[], units=[], cities=[],
        tiles=[dict(tile=i, known=True, terrain=3) for i in range(121)])
    memory = PublicMemory()
    memory.observe(replace(obs, visible_state=visible))
    hidden = {**visible, 'tiles': [dict(tile=i, known=False) for i in range(121)]}
    result = memory.observe(replace(obs, visible_state=hidden, state_version=1))
    assert [r['fact']['tile'] for r in result.remembered_state['historical']] == list(range(32))


@pytest.fixture(scope='module')
def initial():
    env = GPUEnvironment(14)
    env.reset(seed=100001, settings={})
    return env.snapshot()


def test_adapter_hidden_world_invariance_complete_prompt(initial):
    env = GPUEnvironment(14)
    obs = env.restore(initial)
    expected = render(obs).messages
    s = env.state
    hidden = ~np.asarray(s.tiles.explored[0])
    # Enemy bank/technology, RNG and unobserved ordinary terrain are private.
    terrain = s.tiles.terrain.at[np.flatnonzero(hidden)].set(5)
    env.state = s._replace(seed=jnp.uint32(98765), tiles=s.tiles._replace(terrain=terrain),
        players=s.players._replace(stars=s.players.stars.at[1].set(999), tech=s.players.tech.at[1, 10].set(True)))
    env._cached = {}
    assert render(env.observe(1)).messages == expected


def test_adapter_snapshot_replay_stale_rejection_and_seat_memory(initial):
    env = GPUEnvironment(14)
    obs = env.restore(initial)
    end = next(a for a in obs.legal_actions if a.command['kind'] == 10)
    transition = env.step(action_id=end.action_id, expected_state_version=obs.state_version)
    assert transition.next_player == 2
    assert transition.actor_observation.remembered_state['recent_actions'][0]['command']['kind'] == 10
    assert env.observe(2).remembered_state['recent_actions'] == []
    after = env.snapshot()
    with pytest.raises(ValueError, match='stale'):
        env.step(action_id=end.action_id, expected_state_version=obs.state_version)
    env.restore(initial)
    env.step(action_id=end.action_id, expected_state_version=obs.state_version)
    # Compare semantic snapshot contents; NPZ container timestamps need not match.
    first = json.loads(after)
    second = json.loads(env.snapshot())
    assert first['memory'] == second['memory']
    from polytopia_bot.gpu.snapshot import fields, loads
    import base64
    a = loads(base64.b64decode(first['engine']), profile='native')
    b = loads(base64.b64decode(second['engine']), profile='native')
    for (name, x), (_, y) in zip(fields(a), fields(b)):
        np.testing.assert_array_equal(x, y, err_msg=name)
    env.reset(seed=100001, settings={})
    assert env.observe(1).remembered_state['recent_actions'] == []


def test_terminal_resignation_has_no_catalog_or_fallback(initial):
    env = GPUEnvironment(14)
    obs = env.restore(initial)
    resign = next(a for a in obs.legal_actions if a.command['kind'] == 11)
    result = env.step(action_id=resign.action_id, expected_state_version=obs.state_version)
    assert result.terminated and not result.truncated
    assert result.rewards == {1: -1., 2: 1.}
    assert result.actor_observation.legal_actions == []


def test_compact_prompt_preserves_every_field_and_catalog_entry(initial):
    from polytopia_bot.learning.encoding import compact, expand
    env = GPUEnvironment(14)
    for obs in (fixture_observation(), env.restore(initial)):
        prompt = render(obs)
        assert expand(compact(prompt.public)) == prompt.public
        assert expand(json.loads(prompt.messages[1]['content'])) == prompt.public


def test_starvation_progress_is_preserved_in_public_city():
    obs = fixture_observation()
    visible = deepcopy(obs.visible_state)
    visible['cities'][0]['progress'] = -3
    prompt = render(replace(obs, visible_state=visible))
    assert prompt.public['visible']['cities'][0]['progress'] == -3
    from polytopia_bot.learning.encoding import expand
    assert expand(json.loads(prompt.messages[1]['content'])) == prompt.public


def test_new_memory_fields_require_schema_version():
    obs = fixture_observation()
    with pytest.raises(ValueError, match='memory schema'):
        render(replace(obs, remembered_state={'recent_actions': [{'private_seed': 42}]}))


def test_hidden_enemy_relocation_preserves_catalog_and_memory(initial):
    env = GPUEnvironment(14)
    before = env.restore(initial)
    s = env.state
    hidden = ~np.asarray(s.tiles.explored[0])
    enemies = np.flatnonzero((np.asarray(s.units.owner)==2) & hidden)
    assert len(enemies)
    source = int(enemies[0])
    candidates = np.flatnonzero(hidden & (np.asarray(s.units.owner)==0))
    # A remote hidden tile stays outside the own-unit Detect radius.
    target = max(map(int, candidates), key=lambda i: min(max(abs(i%14-u['tile']%14),abs(i//14-u['tile']//14)) for u in before.visible_state['units'] if u['owner']==1))
    values = {}
    for name in s.units._fields:
        array = getattr(s.units,name)
        values[name] = array.at[target].set(array[source]).at[source].set(jnp.zeros((), dtype=array.dtype))
    env.state=s._replace(units=s.units._replace(**values))
    env._cached={}
    assert render(env.observe(1)).messages==render(before).messages


def test_ruins_remain_public_tiles_but_are_not_cities(initial):
    from polytopia_bot.learning.adapter import serialize
    env = GPUEnvironment(14)
    env.restore(initial)
    view = env._observe(env.state, jnp.int32(1))
    capital = int(np.flatnonzero(np.asarray(view['own_city_level']) > 0)[0])
    others = [int(i) for i in np.flatnonzero(np.asarray(view['explored'])) if i != capital]
    neutral, owned, village = others[:3]
    # Exercise the redacted serializer directly: ruins can lie in territory.
    view = {**view,
        'structure': view['structure'].at[neutral].set(2).at[owned].set(2).at[village].set(1),
        'owner': view['owner'].at[neutral].set(0).at[owned].set(1).at[village].set(0)}
    obs = serialize(view, size=14, legal=[])
    cities = {c['tile']: c for c in obs.visible_state['cities']}
    assert neutral not in cities and owned not in cities
    assert cities[capital]['level'] == 1
    assert cities[village] == {'tile': village, 'owner': 0}
    assert obs.visible_state['tiles'][neutral]['structure'] == 2
    assert obs.visible_state['tiles'][owned]['structure'] == 2
