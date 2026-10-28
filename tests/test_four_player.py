"""Four-seat engine-rs games, maps, features and networks (docs/four-player-2026-10-24.md).

There is no reference engine for four seats, so these are invariant tests: random legal games run to their end
without rejected actions, the turn order skips seats that are out, eliminations follow capture, rank rewards
are zero-sum, a state round-trips through export/import, and the widened two-seat networks compute exactly what
their parents compute. Two-seat bitwise equality with the JAX engine stays in tests/test_rust_engine.py.
"""
import numpy as np
import pytest

SIZE = 18
END_FAMILY, CAPTURE_FAMILY, DIPLOMACY_FAMILY, RESIGN_FAMILY = 15, 8, 14, 16


def _games(games=6, **kw):
    from polytopia_bot.rl.rust_engine import RustGames
    return RustGames(SIZE, games, players=4, **kw)


def _random_choices(masks, rng, size=SIZE, end_bias=.15, allow_end=True):
    """A uniformly random legal (family, source, target) per lane, with end-turn boosted so games advance
    (allow_end=False: end-turn only when nothing else is legal)."""
    import jax.numpy as jnp
    from polytopia_bot.rl.actions import Choice, shapes
    shp = shapes(size)
    games = np.asarray(masks.end).shape[0]
    out = [np.zeros(games, np.int32) for _ in range(3)]
    for g in range(games):
        mats = [np.asarray(m[g]).reshape(shp[f]) for f, m in enumerate(masks)]
        legal = [f for f in range(17) if mats[f].any()]
        if not allow_end and len(legal) > 1:
            legal = [f for f in legal if f != END_FAMILY]
        f = END_FAMILY if END_FAMILY in legal and rng.random() < end_bias else int(rng.choice(legal))
        s = int(rng.choice(np.flatnonzero(mats[f].any(1))))
        t = int(rng.choice(np.flatnonzero(mats[f][s])))
        out[0][g], out[1][g], out[2][g] = f, s, t
    return Choice(*(jnp.asarray(a) for a in out))


def test_layout_has_four_capitals_one_per_quadrant():
    g = _games(1)
    half = SIZE // 2
    for seed in range(20):
        caps = np.asarray(g.batch.layout(seed)['capitals'])
        assert len(caps) == 4 and len(set(caps.tolist())) == 4
        quads = {(int(c % SIZE) >= half, int(c // SIZE) >= half) for c in caps}
        assert len(quads) == 4


def test_variant_maps_four_seats():
    from polytopia_bot.maps.stream import MIXES
    from polytopia_bot.maps.variants import generate, MIN_CAPITAL_DISTANCE_4
    d = lambda a, b: max(abs(a % SIZE - b % SIZE), abs(a // SIZE - b // SIZE))
    for kind in MIXES['four-v1']:
        if kind == 'classic':
            continue
        for seed in range(6):
            layout, info = generate(seed, kind, size=SIZE, players=4)
            caps = [int(c) for c in layout.capitals]
            assert len(set(caps)) == 4
            assert min(d(a, b) for i, a in enumerate(caps) for b in caps[i + 1:]) >= MIN_CAPITAL_DISTANCE_4[SIZE]
            for c in caps:   # every capital starts on a village on land
                assert layout.structure[c] == 1 and layout.terrain[c] >= 3
    with pytest.raises(ValueError):
        generate(0, 'ridge', size=SIZE, players=4)


def test_two_seat_variant_maps_unchanged():
    """The four-seat generalization keeps the two-seat generator's output: equal to the pre-change source
    (tests/fixtures/variants_v2_reference.py, bardur-variants-v2 as of origin/main dd9dc34) map for map."""
    import importlib.util
    from pathlib import Path
    from polytopia_bot.maps.variants import generate, PRESETS
    spec = importlib.util.spec_from_file_location('variants_v2_reference',
                                                  Path(__file__).parent / 'fixtures' / 'variants_v2_reference.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    for kind in list(PRESETS) + ['mixed']:
        for size in (11, 14, 18):
            for seed in range(4):
                a, info_a = generate(seed, kind, size=size)
                b, info_b = reference.generate(seed, kind, size=size)
                for x, y in zip(a, b, strict=True):
                    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
                assert info_a == info_b


def test_random_four_seat_games_run_clean():
    rng = np.random.default_rng(0)
    g = _games(6, both_views=True)
    states, prepared = g.reset(np.arange(6, dtype=np.int32) + 900)
    assert 'alive' in dict(g.layout) and dict(g.layout)['known_players'] == (4,)
    assert np.asarray(prepared.opp_view).shape == (6, 3, g.view_width)
    finished = 0
    for tick_index in range(1500):
        masks = prepared.masks
        assert not np.asarray(masks.diplomatic).any(), 'no diplomacy with four seats'
        assert not np.asarray(masks.resign).any(), 'no resigning with four seats'
        current = np.asarray(states.current_player)
        alive = (np.asarray(states.players.killed_turn) == 0) & (np.asarray(states.players.resigned_turn) == 0)
        assert alive[np.arange(6), current - 1].all(), 'the mover is a seat still in'
        tick, status = g.tick(states, _random_choices(masks, rng), np.arange(6, dtype=np.int32) + 1000 + tick_index)
        assert not np.any(np.asarray(status)), f'rejected action at tick {tick_index}'
        done = np.asarray(tick.done)
        for lane in np.nonzero(done)[0]:
            r = np.asarray(tick.rewards)[lane]
            assert abs(r.sum()) < 1e-6, r
            finished += 1
        states, prepared = tick.states, tick.prepared


def _place_attacker(d, lane, victim):
    """Put an idle seat-1 warrior on seat `victim`'s capital (in place of its unit); returns the tile."""
    n = SIZE * SIZE
    capital = int(np.flatnonzero((d['tiles.capital'][lane] == victim) & (d['cities.owner'][lane] == victim))[0])
    for k in ('units.owner', 'units.kind', 'units.home', 'units.kills', 'units.created', 'units.attacks_performed',
              'units.prev_x', 'units.prev_y', 'units.last_attack', 'units.passenger'):
        d[k][lane, capital] = 0
    for k in ('units.moved', 'units.attacked', 'units.veteran', 'units.poison', 'units.frozen', 'units.boosted',
              'units.converted', 'units.invisible'):
        d[k][lane, capital] = False
    d['units.owner'][lane, capital] = 1
    d['units.kind'][lane, capital] = 2
    d['units.health'][lane, capital] = 10.
    d['units.home'][lane, capital] = 0
    return capital


def test_eliminations_turn_order_and_rank_rewards():
    from polytopia_bot.rl.actions import Choice
    import jax.numpy as jnp
    g = _games(1)
    g.reset(np.array([5], np.int32))
    out_order = []
    for victim in (3, 2, 4):
        d = {k: np.array(v) for k, v in g.batch.export().items()}
        assert int(d['current_player'][0]) == 1
        tile = _place_attacker(d, 0, victim)
        g.batch.import_(d)
        status = g.batch.step(np.array([CAPTURE_FAMILY], np.int32), np.array([tile], np.int32), np.array([0], np.int32))
        assert int(status[0]) == 0
        s = g.batch.export()
        assert int(s['players.killed_turn'][0, victim - 1]) > 0
        assert not (np.asarray(s['units.owner'][0]) == victim).any(), 'the eliminated seat loses its units'
        out_order.append(victim)
        assert int(s['players.out_order'][0, victim - 1]) == len(out_order)
        last = victim == 4
        assert bool(s['terminated'][0]) == last
        if not last:
            # end seat 1's turn: the next seat still in moves, skipping the eliminated ones
            g.batch.step(np.array([END_FAMILY], np.int32), np.array([0], np.int32), np.array([0], np.int32))
            s = g.batch.export()
            alive = [k + 1 for k in range(4) if s['players.killed_turn'][0, k] == 0]
            assert int(s['current_player'][0]) == next(k for k in alive if k > 1)
            # let every other seat end its turn until seat 1 moves again
            while int(g.batch.export()['current_player'][0]) != 1:
                g.batch.step(np.array([END_FAMILY], np.int32), np.array([0], np.int32), np.array([0], np.int32))
    rewards = np.asarray(g.batch.terminal_rewards())[0]
    # seat 1 first; then the last out (4) second, 2 third, 3 (first out) last
    np.testing.assert_allclose(rewards, [1., -1 / 3, -1., 1 / 3], atol=1e-6)


def test_round_cap_shares_places_among_survivors():
    g = _games(1, max_turns=1)
    g.reset(np.array([3], np.int32))
    for _ in range(12):
        status = g.batch.step(np.array([END_FAMILY], np.int32), np.array([0], np.int32), np.array([0], np.int32))
        s = g.batch.export()
        if bool(s['truncated'][0]):
            break
    assert bool(s['truncated'][0]) and not bool(s['terminated'][0])
    np.testing.assert_allclose(np.asarray(g.batch.terminal_rewards())[0], [0., 0., 0., 0.], atol=1e-6)


def test_export_import_round_trip_is_exact():
    rng = np.random.default_rng(1)
    a, b = _games(4), _games(4)
    states, prepared = a.reset(np.arange(4, dtype=np.int32) + 50)
    for t in range(150):
        tick, _ = a.tick(states, _random_choices(prepared.masks, rng), np.arange(4, dtype=np.int32) + 60)
        states, prepared = tick.states, tick.prepared
    b.batch.import_(a.batch.export())
    for t in range(150):
        choice = _random_choices(prepared.masks, rng)
        ta, sa = a.tick(states, choice, np.arange(4, dtype=np.int32) + 70 + t)
        tb, sb = b.tick(states, choice, np.arange(4, dtype=np.int32) + 70 + t)
        ea, eb = a.batch.export(), b.batch.export()
        for k in ea:
            np.testing.assert_array_equal(np.asarray(ea[k]), np.asarray(eb[k]), err_msg=k)
        np.testing.assert_array_equal(np.asarray(ta.prepared.view), np.asarray(tb.prepared.view))
        states, prepared = ta.states, ta.prepared


def _old_encode_view(v, size):
    """encode_view before the multi-seat change (two seats only), for the equality check."""
    import jax.numpy as jnp
    from polytopia_bot.rl.observation import Features, STRUCTURES, UNITS, UNIT_FLAGS, CITY_VALUES, RELATIONS
    p = v["player"]
    known = v["explored"]
    def categorical(name, values):
        return (v[name][:, None] == jnp.asarray(values)[None, :]).astype(jnp.float32) * known[:, None]
    def ownership(name):
        a = v[name]
        return jnp.stack((known & (a == 0), known & (a == p), known & (a == 3 - p)), -1)
    ids = jnp.arange(size * size)
    scalars = [known, ids % size / (size - 1), ids // size / (size - 1),
               jnp.maximum(v["road"], 0), jnp.maximum(v["structure_level"], 0) / 10,
               jnp.maximum(v["unit_health"], 0) / 40, jnp.maximum(v["unit_veteran"], 0), v["detect"]]
    scalars += [v["own_unit_" + k] for k in UNIT_FLAGS]
    scalars += [v["own_unit_kills"] / 10, v["own_unit_home"] / (size * size), v["own_unit_passenger"] / 128]
    scalars += [v["own_city_" + k] / 20 for k in CITY_VALUES]
    scalars += [v["own_city_connected"], v["own_city_riot"]]
    board = jnp.concatenate((categorical("terrain", range(8)), categorical("resource", range(8)),
                             categorical("structure", STRUCTURES), categorical("unit_type", UNITS),
                             ownership("owner"), ownership("unit_owner"), ownership("capital"),
                             jnp.stack(scalars, -1), v["own_city_rewards"] / 10), -1).astype(jnp.float32)
    order = jnp.array([p - 1, 2 - p])
    g = jnp.concatenate((jnp.array([v["turn"] / 100, jnp.log1p(v["stars"]) / 8, p - 1]),
                         v["tech"], v["own_monuments"], v["own_built"],
                         *(v[k][order] for k in RELATIONS))).astype(jnp.float32)
    return Features(board.reshape(size, size, -1), g)


def test_two_seat_features_unchanged():
    import jax
    from polytopia_bot.rl.cpu_engine import unpack_view
    from polytopia_bot.rl.observation import encode_view
    from polytopia_bot.rl.rust_engine import RustGames
    rng = np.random.default_rng(2)
    g = RustGames(14, 4)
    states, prepared = g.reset(np.arange(4, dtype=np.int32) + 7)
    new = jax.jit(jax.vmap(lambda v: encode_view(unpack_view(v, g.layout), 14)))
    old = jax.jit(jax.vmap(lambda v: _old_encode_view(unpack_view(v, g.layout), 14)))
    for t in range(200):
        view = np.asarray(prepared.view)
        a, b = new(view), old(view)
        np.testing.assert_array_equal(np.asarray(a.board), np.asarray(b.board))
        np.testing.assert_array_equal(np.asarray(a.global_state), np.asarray(b.global_state))
        tick, _ = g.tick(states, _random_choices(prepared.masks, rng, 14), np.arange(4, dtype=np.int32) + 30 + t)
        states, prepared = tick.states, tick.prepared


def test_widened_networks_compute_the_parent_function():
    import jax
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import _critic_template, _template_params, critic_features
    from polytopia_bot.rl.four_player import convert, old_part
    rng = np.random.default_rng(3)
    parent = _template_params(14)
    parent_critic = _critic_template(14)
    # perturb so zero-initialized heads do not hide a mismatch
    parent = jax.tree.map(lambda a: a + .05 * rng.standard_normal(a.shape).astype(a.dtype), parent)
    parent_critic = jax.tree.map(lambda a: a + .05 * rng.standard_normal(a.shape).astype(a.dtype), parent_critic)
    g = _games(4, both_views=True)
    states, prepared = g.reset(np.arange(4, dtype=np.int32) + 11)
    for t in range(60):
        tick, _ = g.tick(states, _random_choices(prepared.masks, rng), np.arange(4, dtype=np.int32) + 90 + t)
        states, prepared = tick.states, tick.prepared
    for extra, pool in ((0, False), (2, True)):
        policy, critic = convert(parent, parent_critic, size=SIZE, extra_blocks=extra, global_pool=pool, seed=4)
        trunk = jax.vmap(lambda p, x: network.trunk(p, x), in_axes=(None, 0))
        wide = trunk(policy, prepared.features)
        narrow = trunk(parent, old_part(prepared.features))
        np.testing.assert_allclose(np.asarray(wide[1]), np.asarray(narrow[1]), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(wide[0]), np.asarray(narrow[0]), rtol=1e-5, atol=1e-5)
        full = jax.vmap(lambda f, o: critic_features(f, o, g.layout, SIZE))(prepared.features, prepared.opp_view)
        cw = jax.vmap(lambda x: network.trunk(critic, x)[2])(full)
        cn = jax.vmap(lambda x: network.trunk(parent_critic, x)[2])(old_part(full, groups=2))
        np.testing.assert_allclose(np.asarray(cw), np.asarray(cn), rtol=1e-5, atol=1e-5)


def test_segment_collector_four_seats():
    from polytopia_bot.rl.distributed import SegmentCollector
    c = SegmentCollector(1, 8, players=4)
    width = 5
    for t in range(12):
        seat = t % 4 + 1
        c.add(0, seat, dict(view=np.full(width, t, np.int16), opp_view=np.zeros((3, width), np.int16),
                            logp=np.float32(0)))
    final = np.arange(4 * width, dtype=np.int16).reshape(4, width)
    c.finish(0, np.array([1, 1 / 3, -1 / 3, -1], np.float32), False, final)
    rows, segs = c.take()
    assert len(rows) == 12 and len(segs) == 4
    seat2 = segs[1]
    np.testing.assert_array_equal(seat2['boot_view'], final[1])
    np.testing.assert_array_equal(seat2['boot_opp_view'], final[[2, 3, 0]])


def test_grad_accum_matches_one_minibatch(tmp_path):
    """grad_accum splits each minibatch's gradient into micro-batches; the update equals one full-minibatch step."""
    import jax
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import Learner, LearnerConfig, _template_params
    from polytopia_bot.rl.runner import Config, save_checkpoint
    rng = np.random.default_rng(5)
    params = jax.tree.map(lambda a: a + .05 * rng.standard_normal(a.shape).astype(a.dtype), _template_params(14))
    parent = tmp_path / 'parent.npz'
    save_checkpoint(parent, params, Config(architecture='normalized-v2', size=14), stage='test')
    out = []
    for accum in (1, 4):
        learner = Learner(tmp_path / f'run{accum}', LearnerConfig(size=14, batch=64, minibatch=32, grad_accum=accum,
                                                                   omniscient_critic=True), str(parent),
                          log=lambda s: None)
        batch = jax.device_put(learner.synthetic_batch(64), learner.data_sharding)
        (p, c), _, metrics = learner.train_epoch((learner.params, learner.critic), learner.opt, None, None, batch,
                                                 jax.random.key(1), np.float32(1.), np.float32(.005))
        out.append((jax.device_get((p, c)), {k: float(v) for k, v in metrics.items()}))
    # Adam's first step is ~lr * sign(g), so compare the gradient norm (same gradient) and the loss; the params
    # may differ by float rounding only where a gradient entry is ~0
    for k in ('grad_norm', 'loss', 'value_loss'):
        assert np.isclose(out[0][1][k], out[1][1][k], rtol=1e-4, atol=1e-6), (k, out[0][1][k], out[1][1][k])
    for a, b in zip(jax.tree.leaves(out[0][0]), jax.tree.leaves(out[1][0])):
        assert np.mean(np.isclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7)) > .99


OWNER_FIELDS = ('tiles.owner', 'tiles.capital', 'units.owner', 'cities.owner')


def _rotate(d, k=1):
    """Relabel seats s -> s + k (mod 4) in an exported four-seat state: owner ids, per-seat rows and matrices."""
    relabel = lambda a: np.where(a > 0, (a - 1 + k) % 4 + 1, a)
    out = {}
    for name, a in d.items():
        a = np.array(a)
        if name in OWNER_FIELDS or name == 'current_player':
            a = relabel(a)
        elif name == 'tiles.explored' or (name.startswith('players.') and a.ndim >= 2 and a.shape[1] == 4):
            a = np.roll(a, k, axis=1)
            if a.ndim == 3 and a.shape[2] == 4:
                a = np.roll(a, k, axis=2)
            if name in ('players.killer', 'players.truce_breaker'):
                a = relabel(a)
        out[name] = a
    return out


def test_seats_are_symmetric_under_relabeling():
    """Rotating the seat labels of a mid-game state gives the mover the same legal actions and features (except
    its seat index), and the same action leads to the rotated successor state: no seat is special to the engine
    within a turn (turn order itself is not rotated, so end-turn is not compared)."""
    import jax
    from polytopia_bot.rl.cpu_engine import unpack_view
    from polytopia_bot.rl.observation import encode_view
    rng = np.random.default_rng(6)
    a, b = _games(4), _games(4)
    states, prepared = a.reset(np.arange(4, dtype=np.int32) + 70)
    for t in range(1200):   # play into the middle game
        tick, _ = a.tick(states, _random_choices(prepared.masks, rng, end_bias=.05), np.arange(4, dtype=np.int32) + 80)
        states, prepared = tick.states, tick.prepared
    feats = jax.jit(jax.vmap(lambda v: encode_view(unpack_view(v, a.layout), SIZE)))
    checked = 0
    for rotation in (1, 2, 3):
        base = {k: np.array(v) for k, v in a.batch.export().items()}
        base['cities.infiltrated'][:] = 0   # turn*seats+owner stamps do not relabel; the rule is unused here
        a.batch.import_(base)
        b.batch.import_(_rotate(base, rotation))
        for step in range(40):
            ma, mb = a.batch.legal_masks(), b.batch.legal_masks()
            np.testing.assert_array_equal(ma, mb)
            fa, fb = feats(a.batch.observe()), feats(b.batch.observe())
            np.testing.assert_array_equal(np.asarray(fa.board), np.asarray(fb.board))
            ga, gb = np.asarray(fa.global_state), np.asarray(fb.global_state)
            np.testing.assert_array_equal(np.delete(ga, 2, 1), np.delete(gb, 2, 1))
            from polytopia_bot.rl.rust_engine import split_masks
            choice = _random_choices(split_masks(ma, SIZE), rng, end_bias=0., allow_end=False)
            go = np.asarray(choice.family) != END_FAMILY
            if not go.any():
                break
            # a lane with only end-turn left takes the always-rejected self-attack (rl.search.NOOP) and stays put
            choice = [np.where(go, np.asarray(c, np.int32), n).astype(np.int32) for c, n in zip(choice, (1, 0, 0))]
            sa = a.batch.step(*choice)
            sb = b.batch.step(*choice)
            np.testing.assert_array_equal(np.asarray(sa), np.asarray(sb))
            ea, eb = _rotate(a.batch.export(), rotation), b.batch.export()
            for k in ea:
                # turn*seats+seat stamps encode the seat index by construction (infiltration; hostile_until is
                # read only by two-seat diplomacy)
                if k not in ('cities.infiltrated', 'players.hostile_until'):
                    np.testing.assert_array_equal(ea[k], np.asarray(eb[k]), err_msg=f'{k} rotation {rotation} step {step}')
            checked += int(go.sum())
    assert checked >= 10   # lane-steps that moved


def test_arena4_rotations_play_and_ratings():
    from polytopia_bot.rl.arena4 import fit_ratings, pairwise, play, rotations, summary
    from polytopia_bot.rl.distributed import _template_params
    assert rotations(['a', 'b', 'c', 'd'])[1] == ['d', 'a', 'b', 'c']
    policies = {'x': _template_params(SIZE, players=4), 'y': _template_params(SIZE, players=4, extra_blocks=1)}
    records = play(policies, [['x', 'x', 'y', 'y']], size=SIZE, seeds=[384000, 384001], lanes=8, max_turns=2,
                   deterministic=False)
    assert len(records) == 8
    for r in records:
        assert abs(sum(r['rewards'])) < 1e-6 and sorted(r['seat_names']) == ['x', 'x', 'y', 'y']
    out = summary(records)
    assert out['entrants']['x']['games'] == 16 and set(out['ratings']) == {'x', 'y'}
    # ratings: a strictly better entrant gets the higher rating; a symmetric record gives equal ratings
    fake = [dict(seat_names=['a', 'b', 'a', 'b'], rewards=[1., 1 / 3, -1 / 3, -1.], terminated=True, turns=10)] * 3
    r = fit_ratings(fake)
    assert r['a'] > r['b']
    assert pairwise(fake)[('a', 'b')] == [3 * 3., 3 * 4]   # a wins 3 of the 4 a-b pairs per game
    even = [dict(seat_names=['a', 'b', 'c', 'd'], rewards=[0., 0., 0., 0.], terminated=False, turns=150)]
    assert all(abs(v) < 1e-6 for v in fit_ratings(even).values())
