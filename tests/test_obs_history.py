"""Observation version 3 (2026-11-01): the seat's observation history from engine-rs (history.rs) as view fields,
its encoding, and the compatibility of older networks with the new features.

The history must be a function of the seat's own past views only (no hidden state): the first test recomputes every
history field from the views the seat was given and requires equality."""
import json

import numpy as np
import pytest

SIZE = 11
END = 15
N = SIZE * SIZE


def _rules():
    from pathlib import Path
    import polytopia_bot
    return json.loads((Path(polytopia_bot.__file__).parent / 'gpu/rules.json').read_text())


def _tech_tables():
    """What engine-rs infers from (rules.rs / history.rs): unit kind -> tech, structure -> tech (when exactly one tech
    unlocks it), and each tech's closure over its edges (next lists and requires, as rules.rs builds tech_edges)."""
    t = _rules()
    exists = {te['id'] for te in t['technologies'] if 0 <= te['id'] < 25} | {38, 44}
    edges = {k: set() for k in range(128)}   # predecessor sets: i -> k
    for te in t['technologies']:
        i = te['id']
        if not 0 <= i < 128:
            continue
        for o in list(te['next']) + ([te['requires']] if te['requires'] is not None else []):
            if 0 <= o < 128:
                edges[o].add(i)
    closure = {}
    def close(k, seen):
        seen.add(k)
        for i in edges[k]:
            if i not in seen:
                close(i, seen)
        return seen
    for k in range(128):
        closure[k] = close(k, set())
    unit_tech = {2: 0, 3: 1, 4: 3, 5: 7, 8: 17, 9: 18, 10: 22, 11: 24, 38: 38, 41: 14, 42: 13, 44: 44}
    struct_tech = {}
    for te in t['technologies']:
        for s in te['structures']:
            struct_tech.setdefault(s, set()).add(te['id'])
    struct_tech = {s: next(iter(v)) for s, v in struct_tech.items() if len(v) == 1}
    return unit_tech, struct_tech, closure, exists


def _raw_tech(view, player, unit_tech, struct_tech):
    """The techs one view implies (history.rs fold_tech before the closure)."""
    out = set()
    vis = view['explored'].astype(bool)
    uo, kind, terrain = view['unit_owner'], view['unit_type'], view['terrain']
    enemy_unit = vis & (uo > 0) & (uo != player)
    for t in np.flatnonzero(enemy_unit):
        k = int(kind[t])
        if k in unit_tech:
            out.add(unit_tech[k])
        if k == 37:
            out.add(38)
        if terrain[t] == 4:
            out.add(20)
        if terrain[t] == 1:
            out.add(10)
    owner = view['owner']
    enemy_tile = vis & (owner > 0) & (owner != player)
    for t in np.flatnonzero(enemy_tile):
        s = int(view['structure'][t])
        if s in struct_tech:
            out.add(struct_tech[s])
        if view['road'][t] > 0 and 3 in struct_tech:
            out.add(struct_tech[3])
    return out


def _code(view):
    vis = view['explored'].astype(bool)
    return np.where(vis, view['structure'] * 64 + np.clip(view['structure_level'], 0, 31) * 2 + (view['road'] > 0), -1)


def _enemy_units(view, player):
    uo = view['unit_owner']
    return view['explored'].astype(bool) & (uo > 0) & (uo != player)


def _unpacked(games, player):
    from polytopia_bot.rl.cpu_engine import unpack_view
    packed = np.asarray(games.batch.observe(player))
    return [{k: np.asarray(v) for k, v in unpack_view(packed[g], games.layout).items()} for g in range(packed.shape[0])]


def _random_choice(masks, rng, size=SIZE, end_bias=.2):
    from polytopia_bot.rl.actions import shapes
    shp = shapes(size)
    games = np.asarray(masks[END]).shape[0]
    out = [np.zeros(games, np.int32) for _ in range(3)]
    for g in range(games):
        mats = [np.asarray(m[g]).reshape(shp[f]) for f, m in enumerate(masks)]
        fams = [f for f in range(17) if mats[f].any()]
        if not fams:
            out[0][g] = 1   # finished game: a rejected self-attack
            continue
        f = END if END in fams and rng.random() < end_bias else int(rng.choice(fams))
        s = int(rng.choice(np.flatnonzero(mats[f].any(1))))
        out[0][g], out[1][g], out[2][g] = f, s, int(rng.choice(np.flatnonzero(mats[f][s])))
    return out


def test_view_layout_agrees_with_the_engine():
    from polytopia_bot.rl.cpu_engine import view_layout
    from polytopia_bot.rl.observation import HISTORY_TURNS
    from polytopia_bot.rl.rust_engine import RustGames
    g = RustGames(SIZE, 1)   # the constructor compares the Python layout with kernels::VIEW_FIELDS
    names = [k for k, _ in g.layout]
    assert names == sorted(names)
    assert dict(g.layout)['hist_enemy_unit'] == (HISTORY_TURNS, N) and dict(g.layout)['enemy_tech'] == (128,)
    assert g.view_width == 43 * N + 525
    assert RustGames(SIZE, 1, players=4).view_width == 43 * N + 525 + 6 * 4 - 10
    assert view_layout(SIZE) == g.layout


def test_history_is_a_function_of_the_seats_own_views():
    """Play random games; at the start of every turn of seat p, every history field equals what the engine must
    derive from p's earlier views (the view before each of p's end-turn actions and the view at each of p's turn
    starts), and nothing else; it stays constant through the turn."""
    from polytopia_bot.rl.rust_engine import RustGames, split_masks
    unit_tech, struct_tech, closure, exists = _tech_tables()
    games = 3
    g = RustGames(SIZE, games, max_turns=40)
    g.batch.reset(np.arange(games, dtype=np.int32) + 17)
    rng = np.random.default_rng(3)
    ends = {(k, p): [] for k in range(games) for p in (1, 2)}      # views right before p's end-turn actions
    raw = {(k, p): set() for k in range(games) for p in (1, 2)}    # techs implied by everything p has seen
    checked = starts = 0
    prev_hist = {}
    for tick in range(2500):
        scal = {k: np.asarray(v) for k, v in g.batch.scalars().items()}
        done = scal['terminated'] | scal['truncated']
        if done.all():
            break
        masks = split_masks(g.batch.legal_masks(), SIZE)
        choice = _random_choice(masks, rng)
        for k in range(games):
            if done[k]:
                continue
            p = int(scal['current_player'][k])
            view = _unpacked(g, p)[k]
            hist = (view['hist_enemy_unit'].copy(), view['hist_enemy_built'].copy(), view['hist_enemy_trained'].copy(),
                    view['enemy_tech'].copy())
            key = (k, p, int(scal['turn'][k]))
            if key not in prev_hist:
                # first decision of this turn: check against the derivation
                starts += 1
                raw[(k, p)] |= _raw_tech(view, p, unit_tech, struct_tech)
                prev = ends[(k, p)]
                if prev:
                    last = prev[-1]
                    np.testing.assert_array_equal(hist[0][0], _enemy_units(last, p).astype(np.int32))
                    for j in range(1, 3):
                        if len(prev) > j:
                            np.testing.assert_array_equal(hist[0][j], _enemy_units(prev[-1 - j], p).astype(np.int32))
                        else:
                            assert not hist[0][j].any()
                    code_then, code_now = _code(last), _code(view)
                    owner = view['owner']
                    enemy_tile = view['explored'].astype(bool) & (owner > 0) & (owner != p)
                    built = (code_then >= 0) & enemy_tile & (code_now != code_then)
                    np.testing.assert_array_equal(hist[1], built.astype(np.int32))
                    enemy_city = enemy_tile & (view['structure'] == 1)
                    trained = _enemy_units(view, p) & ~_enemy_units(last, p) & enemy_city
                    np.testing.assert_array_equal(hist[2], trained.astype(np.int32))
                    checked += 1
                else:
                    assert not hist[0].any() and not hist[1].any() and not hist[2].any()
                expect = set()
                for t in raw[(k, p)]:
                    expect |= closure[t]
                expect &= exists
                assert set(np.flatnonzero(hist[3]).tolist()) == expect, (k, p, tick)
                prev_hist[key] = hist
            else:
                for a, b in zip(prev_hist[key], hist):   # constant through the turn
                    np.testing.assert_array_equal(a, b)
            if choice[0][k] == END:
                ends[(k, p)].append(view)
                raw[(k, p)] |= _raw_tech(view, p, unit_tech, struct_tech)
        g.batch.step(*choice)
    assert checked >= 30 and starts >= 40


def test_history_survives_export_import_and_is_optional():
    from polytopia_bot.rl.rust_engine import RustGames, split_masks
    g = RustGames(SIZE, 2, max_turns=30)
    g.batch.reset(np.array([5, 6], np.int32))
    rng = np.random.default_rng(7)
    for _ in range(400):
        g.batch.step(*_random_choice(split_masks(g.batch.legal_masks(), SIZE), rng))
    states = g.states()
    assert states.history is not None and np.asarray(states.history.seen_unit).shape == (2, 2, 3, N)
    assert np.asarray(states.history.enemy_tech).any() and np.asarray(states.history.seen_unit).any()
    view = np.asarray(g.batch.observe())
    h = RustGames(SIZE, 2)
    h.load(states)
    np.testing.assert_array_equal(np.asarray(h.batch.observe()), view)
    # a State without history (older saves, gpu/state.py States) imports with an empty history
    stripped = states._replace(history=None)
    h.load(stripped)
    out = np.asarray(h.batch.observe())
    at = 0
    for name, shape in h.layout:
        n = int(np.prod(shape))
        if name.startswith('hist_') or name == 'enemy_tech':
            assert not out[:, at:at + n].any(), name
        at += n
    assert at == out.shape[1]


def _views_after_play(players=2, size=SIZE, ticks=300, seed=1):
    from polytopia_bot.rl.rust_engine import RustGames, split_masks
    g = RustGames(size, 2, players=players, max_turns=30)
    g.batch.reset(np.array([seed, seed + 1], np.int32))
    rng = np.random.default_rng(seed)
    for _ in range(ticks):
        g.batch.step(*_random_choice(split_masks(g.batch.legal_masks(), size), rng, size=size))
    return g, np.asarray(g.batch.observe())


@pytest.mark.parametrize('players', [2, 4])
def test_older_versions_read_the_new_features_bitwise(players):
    """fit_board of version-3 features gives exactly what encode_view computes for version 1 and 2 networks, for one
    view and for the omniscient critic's concatenated views."""
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl.cpu_engine import unpack_view
    from polytopia_bot.rl.distributed import feature_shapes
    from polytopia_bot.rl.observation import (ENEMY_TECH_ENTRIES, HISTORY_CHANNELS, OBS_VERSION, encode_view, fit_board,
                                              global_index)
    size = SIZE if players == 2 else 14
    g, packed = _views_after_play(players, size, seed=11, ticks=300 if players == 2 else 900)
    assert OBS_VERSION == 3
    for row in packed:
        v = unpack_view(jnp.asarray(row), g.layout)
        new = encode_view(v, size)
        if players == 2:   # four random seats on 14x14 may not have met yet; the bitwise check below is the point
            assert float(new.board[..., -HISTORY_CHANNELS:].sum()) > 0 and float(new.global_state[-ENEMY_TECH_ENTRIES:].sum()) > 0
        for version in (1, 2):
            old = encode_view(v, size, version)
            assert feature_shapes(players, version) == (old.board.shape[-1], old.global_state.shape[-1])
            np.testing.assert_array_equal(np.asarray(fit_board(new.board, old.board.shape[-1])), np.asarray(old.board))
            np.testing.assert_array_equal(np.asarray(jnp.take(new.global_state, global_index(players, version, players), -1)),
                                          np.asarray(old.global_state))
            two = jnp.concatenate((new.board, new.board), -1)
            np.testing.assert_array_equal(np.asarray(fit_board(two, 2 * old.board.shape[-1])),
                                          np.asarray(jnp.concatenate((old.board, old.board), -1)))
    assert feature_shapes(players, 3) == (jax.eval_shape(lambda: encode_view(unpack_view(jnp.asarray(packed[0]), g.layout), size)).board.shape[-1],
                                          feature_shapes(players, 2)[1] + ENEMY_TECH_ENTRIES)


def test_width_decoding_is_unique():
    from polytopia_bot.rl.observation import _decode_widths, global_layout, global_width, group_width
    widths, gwidths = {}, {}
    for groups in range(1, 5):
        for players in (2, 3, 4):
            w, gw = groups * group_width(players), groups * global_width(players)
            assert w not in widths and gw not in gwidths, (groups, players)
            widths[w], gwidths[gw] = (groups, players), (groups, players)
            for version in (1, 2, 3):
                assert _decode_widths(w, groups * group_width(players, version)) == (groups, players, version)
                g, p, v = _decode_widths(gw, groups * global_width(players, version), global_layout)
                assert (g, p) == (groups, players) and global_layout(players, v) == global_layout(players, version)


def test_version_two_network_widens_to_three_with_identical_outputs():
    """widen_inputs on a version-2 policy and omniscient critic: the same functions on version-3 features."""
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import network
    from polytopia_bot.rl.cpu_engine import unpack_view
    from polytopia_bot.rl.distributed import _critic_template, _template_params, feature_shapes
    from polytopia_bot.rl.observation import encode_view
    from polytopia_bot.rl.distributed import omniscient_features
    g, packed = _views_after_play(2, SIZE, seed=21)
    opp = np.asarray(g.batch.observe_opponent())
    b2, g2 = feature_shapes(2, 2)
    b3, g3 = feature_shapes(2, 3)
    assert (b3, g3) == (b2 + 5, g2 + 128)
    key = jax.random.key(5)
    policy2 = jax.tree.map(lambda a: a + .02 * jax.random.normal(key, a.shape), _template_params(SIZE, blocks=1, width=8, obs_version=2))
    critic2 = jax.tree.map(lambda a: a + .02 * jax.random.normal(key, a.shape), _critic_template(SIZE, blocks=1, width=8, obs_version=2))
    policy3 = network.widen_inputs(policy2, b2, b3, g2, g3)
    critic3 = network.widen_inputs(critic2, b2, b3, g2, g3, groups=2)
    assert policy3['stem'].shape[2] == b3 and critic3['stem'].shape[2] == 2 * b3
    assert policy3['global']['w'].shape[0] == 2 * 8 + g3 and critic3['critic_hidden']['w'].shape[0] == 2 * 8 + 2 * g3
    for row, orow in zip(packed, opp):
        own = encode_view(unpack_view(jnp.asarray(row), g.layout), SIZE)
        other = encode_view(unpack_view(jnp.asarray(orow), g.layout), SIZE)
        assert float(own.board[..., -5:].sum()) + float(own.global_state[-128:].sum()) > 0
        for a, b in zip(network.trunk(policy2, own), network.trunk(policy3, own)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5, rtol=0)
        both = omniscient_features(own, other)
        np.testing.assert_allclose(np.asarray(network.trunk(critic2, both)[2]), np.asarray(network.trunk(critic3, both)[2]),
                                   atol=1e-5, rtol=0)
        # the new rows are read: perturbing them changes the version-3 network only after training (zero now)
        scrambled = own._replace(board=own.board.at[..., -5:].set(1.), global_state=own.global_state.at[-128:].set(1.))
        np.testing.assert_allclose(np.asarray(network.trunk(policy3, scrambled)[2]), np.asarray(network.trunk(policy2, own)[2]),
                                   atol=1e-5, rtol=0)


def test_version_three_config_and_checkpoint(tmp_path):
    from polytopia_bot.rl.distributed import _template_params, checkpoint_template, feature_shapes, network_spec
    from polytopia_bot.rl.runner import Config, load_checkpoint, save_checkpoint
    Config(obs_version=3).validate()
    with pytest.raises(ValueError):
        Config(obs_version=4).validate()
    params = _template_params(SIZE, blocks=1, width=8, obs_version=3)
    path = tmp_path / 'v3.npz'
    save_checkpoint(path, params, Config(size=SIZE, width=8, blocks=1, architecture='normalized-v2', obs_version=3), stage='test')
    assert network_spec(path)['obs_version'] == 3
    loaded = load_checkpoint(path, checkpoint_template(path))
    assert loaded['stem'].shape[2] == feature_shapes(2, 3)[0]


def test_public_view_encodes_with_zero_history():
    from polytopia_bot.gpu import kernels
    from polytopia_bot.gpu.state import empty_state
    from polytopia_bot.rl.observation import encode_view
    s = empty_state(SIZE)
    x = encode_view(kernels.observe(s, 1), SIZE)   # the retired JAX observe has no history fields: zeros
    assert not float(x.board[..., -5:].sum()) and not float(x.global_state[-128:].sum())


def test_tick_cost_two_hundred_ticks_of_twenty_four_games():
    """A timing probe (read its duration with pytest --durations): 200 ticks of 24 games on 14x14, random legal play,
    through RustGames.tick (engine step + masks + observe + feature encode), as an actor runs it."""
    from polytopia_bot.rl.rust_engine import RustGames
    g = RustGames(14, 24, max_turns=60)
    rng = np.random.default_rng(1)
    _, prepared = g.reset(np.arange(24, dtype=np.int32) + 300)
    for step in range(200):
        choice = _random_choice(prepared.masks, rng, size=14)
        tick, status = g.tick(None, choice, np.arange(24, dtype=np.int32) + 5000 + 24 * step)
        assert not np.any(np.asarray(status))
        prepared = tick.prepared
