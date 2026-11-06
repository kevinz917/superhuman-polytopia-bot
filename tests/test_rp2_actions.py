"""Action space and observation additions of 2026-10-31 (docs/rules-parity-2-summary.md).

- disembark: a boat on a bridge lands where it stands (recover family, target 1).
- Four-seat diplomacy: the diplomatic family has one block of six entries per other seat, by turn-order offset from
  the mover; peace requests are answered one at a time, the nearest requesting seat first; attacks, embassies,
  broken treaties and cool-downs are per pair.
- Unit facing in the observation (view field unit_direction; features of observation version 2), with networks of
  version 1 (every checkpoint saved before) reading their old features through the same encoder.
"""
import json

import numpy as np
import pytest

from test_real_rules import (ATTACK, END, FIELD, RAFT, SIZE, WARRIOR, WATER, _capital, _clear_units, _export, _load,
                             _state, _step, _strip, _unit)

RECOVER, DIPLOMATIC = 9, 14
BRIDGE = 48
BOMBER, DINGHY, CLOAK = 41, 39, 38
SIZE4 = 18


def _masks(g, size=SIZE):
    from polytopia_bot.rl.rust_engine import split_masks
    return [np.asarray(m)[0] for m in split_masks(g.batch.legal_masks(), size)]


def _decode(g, family, source, target):
    i = lambda v: np.array([v], np.int32)
    return tuple(int(v) for v in np.asarray(g.batch.decode(i(family), i(source), i(target)))[0])


# ---------------------------------------------------------------------------------------------- disembark


def _boat_on_bridge(kind=BOMBER, passenger=WARRIOR, bridge=True, **kw):
    d = _state()
    row = _strip(d, WATER)
    t = row + 4
    if bridge:
        d['tiles.structure'][0, t] = BRIDGE
    _unit(d, t, 1, kind, passenger=passenger, **kw)
    d['current_player'][0] = 1
    return d, t


def test_disembark_on_a_bridge_lands_in_place():
    """disembark / disembark: the passenger takes the boat's tile, health and facing,
    moved and attacked; the boat's part of the score goes. Legal even after the boat moved and attacked."""
    d, t = _boat_on_bridge(health=7., moved=True, attacked=True, direction=5)
    g = _load(d)
    m = _masks(g)
    assert m[RECOVER].shape == (SIZE * SIZE, 2)
    assert m[RECOVER][t, 1] and not m[RECOVER][t, 0]
    assert _decode(g, RECOVER, t, 1) == (3, t, -1, 23)
    assert _decode(g, RECOVER, t, 0) == (3, t, -1, 7)
    score = int(_export(g)['players.score'][0])
    assert _step(g, RECOVER, t, 1) == 0
    e = _export(g)
    assert e['units.kind'][t] == WARRIOR and e['units.passenger'][t] == 0 and e['units.owner'][t] == 1
    assert e['units.moved'][t] and e['units.attacked'][t] and e['units.direction'][t] == 5
    assert e['units.health'][t] == pytest.approx(7.)
    assert int(e['players.score'][0]) == score - 5 * 15          # Bomber's cost
    assert not _masks(g)[RECOVER][t].any()


def test_disembark_needs_a_bridge():
    d, t = _boat_on_bridge(bridge=False)
    g = _load(d)
    assert not _masks(g)[RECOVER][t, 1]
    assert _step(g, RECOVER, t, 1) != 0


def test_dinghy_lands_as_its_cloak():
    """A Dinghy carries its Cloak implicitly (passenger 0 in our state): it lands as the Cloak."""
    d, t = _boat_on_bridge(kind=DINGHY, passenger=0)
    g = _load(d)
    assert _masks(g)[RECOVER][t, 1]
    assert _step(g, RECOVER, t, 1) == 0
    assert _export(g)['units.kind'][t] == CLOAK


def test_disembark_waits_for_a_pending_peace_request():
    d, t = _boat_on_bridge()
    d['players.known'][0, 0, 1] = d['players.known'][0, 1, 0] = True
    d['players.peace_offer'][0, 0, 1] = True
    g = _load(d)
    assert not _masks(g)[RECOVER].any()
    assert _step(g, RECOVER, t, 1) != 0


# ---------------------------------------------------------------------------------------------- diplomacy


def _game4(seed=21):
    from polytopia_bot.rl.rust_engine import RustGames
    g = RustGames(SIZE4, 1, players=4)
    g.batch.reset(np.array([seed], np.int32))
    return g


def _state4(seed=21):
    d = {k: np.array(v) for k, v in _game4(seed).batch.export().items()}
    d['players.known'][0] = True
    return d


def _load4(d):
    g = _game4()
    g.batch.import_({k: np.ascontiguousarray(v) for k, v in d.items()})
    return g


def _block(mover, seat, seats=4):
    return (seat - mover) % seats - 1


@pytest.mark.parametrize('mover', [1, 2, 3, 4])
def test_blocks_address_seats_in_turn_order(mover):
    d = _state4()
    d['current_player'][0] = mover
    d['players.tech'][0, mover - 1, 7] = True                    # Shields: offer peace
    g = _load4(d)
    m = _masks(g, SIZE4)[DIPLOMATIC]
    assert m.shape == (18,)
    for seat in range(1, 5):
        if seat == mover:
            continue
        b = _block(mover, seat)
        assert m[b * 6 + 4], (mover, seat)
        assert _decode(g, DIPLOMATIC, 0, b * 6 + 4) == (3, seat, -1, 21)
    assert m.sum() == 3


def test_offer_reaches_the_chosen_seat_and_is_answered_on_its_turn():
    d = _state4()
    d['current_player'][0] = 1
    d['players.tech'][0, 0, 7] = True
    g = _load4(d)
    assert _step(g, DIPLOMATIC, 0, _block(1, 3) * 6 + 4) == 0
    e = _export(g)
    assert e['players.peace_offer'][2, 0] and not e['players.peace_offer'][1, 0] and not e['players.peace_offer'][3, 0]
    for _ in range(2):                                           # seats 1 and 2 end their turns
        assert _step(g, END, 0, 0) == 0
    assert int(_export(g)['current_player']) == 3
    m = _masks(g, SIZE4)
    b = _block(3, 1)
    assert set(np.flatnonzero(m[DIPLOMATIC])) == {b * 6 + 1, b * 6 + 2}
    assert not m[END] and not m[0].any()
    assert _step(g, DIPLOMATIC, 0, b * 6 + 2) == 0               # accept
    e = _export(g)
    assert e['players.peace'][2, 0] and e['players.peace'][0, 2] and not e['players.peace'][0, 1]
    assert not e['players.peace_offer'][2, 0]


def test_requests_are_answered_nearest_seat_first():
    """check diplomacy stacks one trigger per message; the oldest (the nearest requester in turn order) comes first."""
    d = _state4()
    d['current_player'][0] = 3
    d['players.peace_offer'][0, 2, 0] = d['players.peace_offer'][0, 2, 3] = True
    g = _load4(d)
    assert set(np.flatnonzero(_masks(g, SIZE4)[DIPLOMATIC])) == {1, 2}           # seat 4 (block 0)
    assert _step(g, DIPLOMATIC, 0, 7) != 0                                        # seat 1 must wait
    assert _step(g, DIPLOMATIC, 0, 1) == 0                                        # decline seat 4
    assert set(np.flatnonzero(_masks(g, SIZE4)[DIPLOMATIC])) == {7, 8}           # then seat 1 (block 1)
    assert _step(g, DIPLOMATIC, 0, 8) == 0
    e = _export(g)
    assert e['players.peace'][2, 0] and not e['players.peace'][2, 3]
    assert not e['players.peace_offer'][2].any()
    assert _masks(g, SIZE4)[END]


def test_requests_of_seats_no_longer_in_are_dropped():
    d = _state4()
    d['current_player'][0] = 2
    d['players.peace_offer'][0, 2, 0] = True                    # seat 3 has a request from seat 1
    d['players.killed_turn'][0, 0] = 1
    d['players.out_order'][0, 0] = 1
    g = _load4(d)
    assert _step(g, END, 0, 0) == 0                               # seat 2 ends; seat 3 starts its turn
    assert int(_export(g)['current_player']) == 3
    assert not _export(g)['players.peace_offer'][2, 0]
    assert _masks(g, SIZE4)[END]


def test_breaking_peace_is_per_pair():
    d = _state4()
    d['current_player'][0] = 1
    for q in (1, 2):
        d['players.peace'][0, 0, q] = d['players.peace'][0, q, 0] = True
    g = _load4(d)
    b = _block(1, 3)
    assert _masks(g, SIZE4)[DIPLOMATIC][b * 6]
    assert _step(g, DIPLOMATIC, 0, b * 6) == 0
    e = _export(g)
    assert not e['players.peace'][0, 2] and not e['players.peace'][2, 0]
    assert e['players.peace'][0, 1] and e['players.peace'][1, 0]
    assert e['players.truce_breaker'][0, 2] == 1 and e['players.truce_breaker'][2, 0] == 1
    assert e['players.truce_breaker'][0, 1] == 0
    turn = int(e['turn'])
    assert e['players.peace_broken'][0, 2] == turn and e['players.peace_broken'][2, 0] == turn


def test_cool_down_ends_on_each_sides_own_clock():
    """diplomacy: towards a lower id the cool-down ends at the first turn start after the break's
    turn, towards a higher id one turn later. Player 1 breaks at turn T: player 2's side ends at its turn T+1, player
    1's side at its turn T+2 (player 1 still cannot attack player 2 at T+1)."""
    d = _state()
    d['current_player'][0] = 1
    d['players.known'][0, 0, 1] = d['players.known'][0, 1, 0] = True
    d['players.peace'][0, 0, 1] = d['players.peace'][0, 1, 0] = True
    g = _load(d)
    t0 = int(_export(g)['turn'])
    assert _step(g, DIPLOMATIC, 0, 0) == 0
    seen = []
    for _ in range(4):
        assert _step(g, END, 0, 0) == 0
        e = _export(g)
        seen.append((int(e['current_player']), int(e['turn']) - t0, bool(e['players.truce_breaker'][0, 1]),
                     bool(e['players.truce_breaker'][1, 0])))
    assert seen == [(2, 0, True, True), (1, 1, True, True), (2, 1, True, False), (1, 2, False, False)]


def test_attacks_and_embassies_are_per_pair():
    """modify diplomacy relations from attack(attacker, victim) destroys only the embassies between the two and stamps
    only that pair."""
    d = _state4()
    for k in [k for k in d if k.startswith('units.')]:
        d[k][0] = 0
    y = SIZE4 // 2
    a, b = y * SIZE4 + 8, y * SIZE4 + 9
    for t in (a, b):
        d['tiles.terrain'][0, t], d['tiles.structure'][0, t], d['tiles.owner'][0, t] = FIELD, 0, 0
        d['cities.owner'][0, t] = 0
        d['tiles.explored'][0, :, t] = True
    _unit(d, a, 1, WARRIOR)
    _unit(d, b, 3, WARRIOR)
    d['current_player'][0] = 1
    d['players.embassy'][0, 0, 1] = d['players.embassy'][0, 0, 2] = d['players.embassy'][0, 2, 0] = True
    g = _load4(d)
    assert _step(g, ATTACK, a, b) == 0
    e = _export(g)
    assert not e['players.embassy'][0, 2] and not e['players.embassy'][2, 0] and e['players.embassy'][0, 1]
    assert e['players.hostile_until'][0, 2] > 0 and e['players.hostile_until'][0, 1] == 0


def test_embassy_in_the_chosen_seats_capital():
    d = _state4()
    d['current_player'][0] = 1
    d['players.tech'][0, 0, 38] = True                           # Diplomacy
    d['players.stars'][0, 0] = 20
    g = _load4(d)
    b3 = _block(1, 3)
    capital = int(np.flatnonzero((d['tiles.capital'][0] == 3) & (d['cities.owner'][0] == 3))[0])
    assert _decode(g, DIPLOMATIC, 0, b3 * 6 + 3) == (3, 3, capital, 20)
    assert _step(g, DIPLOMATIC, 0, b3 * 6 + 3) == 0
    e = _export(g)
    assert e['players.embassy'][0, 2] and not e['players.embassy'][0, 1] and int(e['players.stars'][0]) == 16
    assert _step(g, DIPLOMATIC, 0, _block(1, 2) * 6 + 3) == 0     # the second costs 4 + 2
    assert int(_export(g)['players.stars'][0]) == 10


def test_two_seat_games_use_block_zero_only():
    from polytopia_bot.rl.rust_engine import RustGames, split_masks
    rng = np.random.default_rng(3)
    g = RustGames(SIZE, 4)
    g.batch.reset(np.arange(4, dtype=np.int32) + 40)
    from polytopia_bot.rl.actions import shapes
    shp = shapes(SIZE)
    for _ in range(600):
        masks = split_masks(g.batch.legal_masks(), SIZE)
        assert not np.asarray(masks.diplomatic)[:, 6:].any()
        choice = [np.array([1] * 4, np.int32), np.zeros(4, np.int32), np.zeros(4, np.int32)]
        for k in range(4):
            mats = [np.asarray(m[k]).reshape(shp[f]) for f, m in enumerate(masks)]
            fams = [f for f in range(17) if mats[f].any()]
            f = END if END in fams and rng.random() < .2 else int(rng.choice(fams))
            s = int(rng.choice(np.flatnonzero(mats[f].any(1))))
            choice[0][k], choice[1][k], choice[2][k] = f, s, int(rng.choice(np.flatnonzero(mats[f][s])))
        e = g.batch.export()
        done = np.asarray(e['terminated']) | np.asarray(e['truncated'])
        assert ((g.batch.step(*choice) == 0) | done).all()
    d = _state()
    d['players.known'][0] = True
    gg = _load(d)
    for target in (6, 10, 12, 17):                                # blocks 1 and 2 address no seat
        assert _decode(gg, DIPLOMATIC, 0, target)[1] == 0
        assert _step(gg, DIPLOMATIC, 0, target) != 0


def test_studio_decode_matches_the_engine():
    from polytopia_bot.rl.actions import offsets
    from polytopia_bot.rl.search import first
    from polytopia_bot.studio.environment import choice_of, decode_command
    d, t = _boat_on_bridge()
    d['players.known'][0] = True
    d['players.tech'][0, 0, 7] = d['players.tech'][0, 0, 38] = True
    d['players.stars'][0, 0] = 30
    g = _load(d)
    state = first(g.states())
    flat = np.asarray(g.batch.legal_masks())[0]
    assert flat.size == offsets(SIZE)[-1]
    checked = 0
    for index in np.flatnonzero(flat):
        f, s, tt = choice_of(int(index), SIZE)
        assert tuple(decode_command(int(index), state, SIZE).values()) == _decode(g, f, s, tt)
        checked += f in (RECOVER, DIPLOMATIC)
    assert checked >= 3


def test_states_saved_before_load():
    """hostile_until per attacker ([2]) and no peace_broken: still importable."""
    d = _state()
    d['players.hostile_until'] = np.array([[0, 7]], np.int32)
    del d['players.peace_broken']
    g = _load(d)
    e = _export(g)
    assert e['players.hostile_until'].shape == (2, 2) and (e['players.hostile_until'][1] == 7).all()
    assert (e['players.peace_broken'] == int(e['turn'])).all()


# ---------------------------------------------------------------------------------------------- observation


def test_view_carries_unit_facing():
    from polytopia_bot.rl.cpu_engine import unpack_view, view_layout
    d = _state()
    _clear_units(d)
    cap = _capital(d, 1)
    _unit(d, cap, 1, WARRIOR, direction=3)
    hidden = next(t for t in range(SIZE * SIZE) if t != cap and d['units.owner'][0, t] == 0)
    d['tiles.explored'][0, 0, hidden] = False
    g = _load(d)
    v = unpack_view(np.asarray(g.batch.observe())[0], view_layout(SIZE))
    assert 'unit_direction' in v
    assert v['unit_direction'][cap] == 4 and v['unit_direction'][hidden] == -1
    empty = next(t for t in range(SIZE * SIZE) if d['tiles.explored'][0, 0, t] and d['units.owner'][0, t] == 0)
    assert v['unit_direction'][empty] == 0


@pytest.mark.parametrize('players', [2, 4])
def test_version_one_features_are_the_version_two_ones_without_facing(players):
    import jax.numpy as jnp
    from polytopia_bot.rl.cpu_engine import unpack_view, view_layout
    from polytopia_bot.rl.distributed import feature_shapes
    from polytopia_bot.rl.observation import DIRECTION_CHANNELS, encode_view, fit_board, two_seat_channels
    from polytopia_bot.rl.rust_engine import RustGames
    size = SIZE if players == 2 else SIZE4
    g = RustGames(size, 1, players=players)
    g.batch.reset(np.array([5], np.int32))
    v = unpack_view(jnp.asarray(np.asarray(g.batch.observe())[0]), view_layout(size, players))
    new, old = encode_view(v, size, 2), encode_view(v, size, 1)
    newest = encode_view(v, size)   # version 3: fit_board reads the newest features
    assert new.board.shape[-1] == old.board.shape[-1] + DIRECTION_CHANNELS
    assert feature_shapes(players, 2)[0] == new.board.shape[-1] and feature_shapes(players, 1)[0] == old.board.shape[-1]
    np.testing.assert_array_equal(np.asarray(new.global_state), np.asarray(old.global_state))
    np.testing.assert_array_equal(np.asarray(fit_board(newest.board, old.board.shape[-1])), np.asarray(old.board))
    np.testing.assert_array_equal(np.asarray(fit_board(newest.board, new.board.shape[-1])), np.asarray(new.board))
    # the facing sits right after the two-seat channels, so the two-seat part of version 2 is a prefix
    base = two_seat_channels()
    np.testing.assert_array_equal(np.asarray(new.board[..., :base]), np.asarray(old.board[..., :base]))
    assert float(new.board[..., base:base + DIRECTION_CHANNELS].sum()) > 0          # the starting units face
    # several concatenated views (the omniscient critic)
    two = jnp.concatenate((newest.board, newest.board), -1)
    np.testing.assert_array_equal(np.asarray(fit_board(two, 2 * old.board.shape[-1])),
                                  np.asarray(jnp.concatenate((old.board, old.board), -1)))


def _save(path, params, obs_version):
    from polytopia_bot.rl.runner import Config, save_checkpoint
    save_checkpoint(path, params, Config(size=SIZE, width=8, blocks=1, architecture='normalized-v2',
                                         obs_version=obs_version), stage='test')
    if obs_version == 1:   # a checkpoint saved before observation versions existed has no such key
        meta = json.loads(path.with_suffix('.json').read_text())
        del meta['config']['obs_version']
        path.with_suffix('.json').write_text(json.dumps(meta))


def test_legacy_checkpoint_loads_and_plays_next_to_a_new_one(tmp_path):
    """An old-config checkpoint (no obs_version) builds the old input width, loads, ignores the facing channels and
    plays full games (every action legal) against a version-2 network on the new action layout."""
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import Choice
    from polytopia_bot.rl.distributed import _template_params, checkpoint_template, feature_shapes, network_spec
    from polytopia_bot.rl.observation import DIRECTION_CHANNELS, two_seat_channels
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.rl.rust_engine import RustGames
    old_path, new_path = tmp_path / 'old.npz', tmp_path / 'new.npz'
    _save(old_path, _template_params(SIZE, blocks=1, width=8), 1)
    _save(new_path, jax.tree.map(lambda a: a + .01, _template_params(SIZE, blocks=1, width=8, obs_version=2)), 2)
    assert 'obs_version' not in network_spec(old_path) and network_spec(new_path)['obs_version'] == 2
    old = load_checkpoint(old_path, checkpoint_template(old_path))
    new = load_checkpoint(new_path, checkpoint_template(new_path))
    assert old['stem'].shape[2] == feature_shapes(2, 1)[0]
    assert new['stem'].shape[2] == feature_shapes(2, 1)[0] + DIRECTION_CHANNELS
    games = 4
    engine = RustGames(SIZE, games, max_turns=4)
    states, prepared = engine.reset(np.arange(games, dtype=np.int32) + 61)
    # the old network reads none of the facing channels
    x = jax.tree.map(lambda a: a[0], prepared.features)
    base = two_seat_channels()
    scrambled = x._replace(board=x.board.at[..., base:base + DIRECTION_CHANNELS].set(1.))
    np.testing.assert_array_equal(np.asarray(network.trunk(old, x)[2]), np.asarray(network.trunk(old, scrambled)[2]))
    assert not np.array_equal(np.asarray(network.trunk(new, x)[2]), np.asarray(network.trunk(new, scrambled)[2]))
    act = jax.jit(jax.vmap(lambda p, x, m, k: network.sample(p, x, m, k, size=SIZE)[0], in_axes=(None, 0, 0, 0)))
    key, finished = jax.random.key(0), 0
    for step in range(4000):
        key, k1, k2 = jax.random.split(key, 3)
        a = act(old, prepared.features, prepared.masks, jax.random.split(k1, games))
        b = act(new, prepared.features, prepared.masks, jax.random.split(k2, games))
        seat_one = jnp.asarray(np.asarray(states.current_player) == 1)
        choice = Choice(*(jnp.where(seat_one, p, q) for p, q in zip(a, b)))
        tick, status = engine.tick(states, choice, np.arange(games, dtype=np.int32) + 100 + games * step)
        assert not np.any(np.asarray(status)), 'illegal action'
        finished += int(np.asarray(tick.done).sum())
        states, prepared = tick.states, tick.prepared
        if finished >= games:
            break
    assert finished >= games


def test_four_seat_conversion_keeps_the_observation_version():
    import jax
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import _template_params, feature_shapes
    from polytopia_bot.rl.four_player import old_part, widen_to_players
    from polytopia_bot.rl.rust_engine import RustGames
    g = RustGames(SIZE4, 2, players=4)
    _, prepared = g.reset(np.arange(2, dtype=np.int32) + 9)
    for version in (1, 2, 3):
        two = jax.tree.map(lambda a: a + .01, _template_params(SIZE4, blocks=1, width=8, obs_version=version))
        wide = widen_to_players(two, 4, obs_version=version)
        assert wide['stem'].shape[2] == feature_shapes(4, version)[0]
        value = jax.vmap(lambda p, x: network.trunk(p, x)[2], in_axes=(None, 0))
        np.testing.assert_allclose(np.asarray(value(wide, prepared.features)),
                                   np.asarray(value(two, old_part(prepared.features, obs_version=version))), atol=1e-5)
