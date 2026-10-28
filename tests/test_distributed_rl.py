"""Actor-side GAE segments, wire format and arena summaries."""
import socket
import threading

import jax.numpy as jnp
import numpy as np

from polytopia_bot.rl import ppo
from polytopia_bot.rl.distributed import SeatTrajectories, episode_outcome, gae, recv_msg, send_msg, stack_rows
from polytopia_bot.rl.cpu_arena import summarize


def _reference(values, bootstrap, reward, terminal, gamma, lam):
    n = len(values)
    rewards = np.zeros(n, np.float32)
    if terminal:
        rewards[-1] = reward
    nxt = np.append(values[1:], bootstrap).astype(np.float32)
    term = np.zeros(n, bool)
    trunc = np.zeros(n, bool)
    term[-1] = terminal
    adv, ret = ppo.advantages(jnp.asarray(rewards)[:, None], jnp.asarray(values)[:, None], jnp.asarray(nxt)[:, None],
                              jnp.ones((n, 1), bool), jnp.asarray(term)[:, None], jnp.asarray(trunc)[:, None],
                              gamma=gamma, lam=lam)
    return np.asarray(adv)[:, 0], np.asarray(ret)[:, 0]


def test_gae_matches_ppo_advantages():
    rng = np.random.default_rng(0)
    for terminal in (True, False):
        values = rng.normal(size=37).astype(np.float32)
        adv, ret = gae(values, .3, -1., terminal, .995, .99)
        ref_adv, ref_ret = _reference(values, .3, -1., terminal, .995, .99)
        np.testing.assert_allclose(adv, ref_adv, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ret, ref_ret, rtol=1e-5, atol=1e-5)


def test_segments_bootstrap_from_next_decision_and_finish_with_outcome():
    traj = SeatTrajectories(lanes=1, segment=4, gamma=1., lam=1.)
    values = [.1, .2, .3, .4, .5, .6]
    for v in values:
        traj.add(0, 1, dict(value=np.float32(v)))
    first = traj.take()
    assert len(first) == 4
    # gamma=lambda=1: advantage = bootstrap - own value, bootstrap is value[4]
    np.testing.assert_allclose([r['advantage'] for r in first], [.5 - v for v in values[:4]], atol=1e-6)
    traj.add(0, 2, dict(value=np.float32(-.2)))
    traj.finish(0, rewards=np.array([1., -1.]), terminated=True, bootstrap_values=np.zeros(2))
    rest = traj.take()
    assert len(rest) == 3
    np.testing.assert_allclose([r['returns'] for r in rest], [1., 1., -1.], atol=1e-6)


def test_truncation_bootstraps_critic_value():
    traj = SeatTrajectories(lanes=1, segment=64, gamma=1., lam=1.)
    traj.add(0, 1, dict(value=np.float32(.1)))
    traj.finish(0, rewards=np.zeros(2), terminated=False, bootstrap_values=np.array([.7, -.7]))
    (row,) = traj.take()
    np.testing.assert_allclose(row['returns'], .7, atol=1e-6)


def test_turn_cap_as_fixed_terminal_reward():
    rewards, ended = episode_outcome(np.zeros(2), terminated=False, truncated=True, cap_reward=0.)
    assert ended and rewards.tolist() == [0., 0.]
    rewards, ended = episode_outcome(np.zeros(2), terminated=False, truncated=True)
    assert not ended  # default: bootstrap the critic
    rewards, ended = episode_outcome(np.array([1., -1.]), terminated=True, truncated=False, cap_reward=0.)
    assert ended and rewards.tolist() == [1., -1.]
    traj = SeatTrajectories(lanes=1, segment=64, gamma=1., lam=1.)
    traj.add(0, 1, dict(value=np.float32(.6)))
    rewards, ended = episode_outcome(np.zeros(2), False, True, cap_reward=0.)
    traj.finish(0, rewards=rewards, terminated=ended, bootstrap_values=np.array([.7, -.7]))
    (row,) = traj.take()
    np.testing.assert_allclose(row['returns'], 0., atol=1e-6)


def test_wire_roundtrip():
    a, b = socket.socketpair()
    payload = dict(batch=stack_rows([dict(view=np.arange(9, dtype=np.int16), logp=np.float32(i)) for i in range(5)]))
    thread = threading.Thread(target=send_msg, args=(a, payload))
    thread.start()
    got = recv_msg(b)
    thread.join()
    np.testing.assert_array_equal(got['batch']['view'], payload['batch']['view'])
    np.testing.assert_array_equal(got['batch']['logp'], np.arange(5, dtype=np.float32))


def test_arena_summary_counts_and_diversity():
    rows = [dict(seed=1, seat=1, reward=1., truncated=False, steps=10, cand_units={2: 3, 8: 1}, opp_units={2: 4},
                 cand_research_early=1, first_non_warrior=9, opp_first_non_warrior=None),
            dict(seed=1, seat=2, reward=-1., truncated=False, steps=10, cand_units={2: 2}, opp_units={2: 4},
                 cand_research_early=0, first_non_warrior=None, opp_first_non_warrior=None)]
    s = summarize(rows)
    assert (s['wins'], s['losses'], s['candidate_non_warrior_games']) == (1, 1, 1)
    assert abs(s['candidate_unit_mix']['Warrior'] - 5 / 6) < 1e-3


def test_turn_clock_discounts_only_at_end_turn():
    rng = np.random.default_rng(1)
    values = rng.normal(size=9).astype(np.float32)
    # every decision ends a turn: identical to the per-decision clock
    a, r = gae(values, .3, 1., True, .97, .9, end_turn=np.ones(9, bool))
    b, q = gae(values, .3, 1., True, .97, .9)
    np.testing.assert_allclose(a, b, atol=1e-6)
    # no end turn in the segment: gamma = lambda = 1, advantage = outcome - value
    a, r = gae(values, .3, 1., True, .97, .9, end_turn=np.zeros(9, bool))
    np.testing.assert_allclose(r, np.ones(9), atol=1e-6)
    # one end turn at step 4: the outcome reaches steps 0..4 discounted by one turn (lambda = 1 inside turns)
    end = np.zeros(9, bool)
    end[4] = True
    a, r = gae(values, .3, 1., True, .97, 1., end_turn=end)
    np.testing.assert_allclose(r[:5], .97, atol=1e-6)
    np.testing.assert_allclose(r[5:], 1., atol=1e-6)


def test_turn_clock_segments_drop_the_flag():
    traj = SeatTrajectories(lanes=1, segment=3, gamma=.9, lam=1., turn_clock=True)
    for v, e in ((.1, False), (.2, True), (.3, False), (.4, False)):
        traj.add(0, 1, dict(value=np.float32(v), end_turn=e))
    rows = traj.take()
    assert len(rows) == 3 and all('end_turn' not in r for r in rows)
    # bootstrap .4 from the 4th decision; rows 0-1 pass one end turn
    np.testing.assert_allclose([r['returns'] for r in rows], [.9 * .4, .9 * .4, .4], atol=1e-6)


def test_upgo_returns_follow_the_trajectory_only_while_it_beats_the_critic():
    from polytopia_bot.rl.distributed import upgo_returns
    values = np.array([.2, .5, .1], np.float32)
    # win: the last action's Q = +1 beats V_2 = .1, so G_1 follows the outcome; at t = 0 the next action's
    # Q_1 = V_2 = .1 is below V_1 = .5, so G_0 bootstraps V_1
    g, follow = upgo_returns(values, 0., 1., True, 1.)
    np.testing.assert_allclose(g, [.5, 1., 1.], atol=1e-6)
    assert follow.tolist() == [False, True, False]
    # loss: no action beats the critic, every step is a one-step bootstrap
    g, follow = upgo_returns(values, 0., -1., True, 1.)
    np.testing.assert_allclose(g, [.5, .1, -1.], atol=1e-6)
    assert follow.tolist() == [False, False, False]
    # per-decision discount: G_t = r_t + gamma * (...), Q_{t+1} = r_{t+1} + gamma * V_{t+2}; the outcome itself
    # lands undiscounted on the last decision (as in gae), so G = [.9 * V_1, .9 * 1, 1]
    g, _ = upgo_returns(values, 0., 1., True, .9)
    np.testing.assert_allclose(g, [.45, .9, 1.], atol=1e-6)
    # turn clock: gamma only after the end-turn decision (the middle one)
    g, _ = upgo_returns(values, 0., 1., True, .9, end_turn=np.array([False, True, False]))
    np.testing.assert_allclose(g, [.5, .9, 1.], atol=1e-6)
    # segment cut / truncation: the last decision bootstraps the critic; a rising value keeps following
    g, follow = upgo_returns(np.array([.1, .2], np.float32), .7, 0., False, 1.)
    np.testing.assert_allclose(g, [.7, .7], atol=1e-6)
    assert follow.tolist() == [True, False]


def test_upgo_rows_are_opt_in_and_leave_gae_rows_unchanged():
    plain = SeatTrajectories(lanes=1, segment=64, gamma=1., lam=1.)
    upgo = SeatTrajectories(lanes=1, segment=64, gamma=1., lam=1., upgo=True)
    for traj in (plain, upgo):
        for v in (.2, .5, .1):
            traj.add(0, 1, dict(value=np.float32(v)))
        traj.finish(0, rewards=np.array([1., -1.]), terminated=True, bootstrap_values=np.zeros(2))
    a, b = plain.take(), upgo.take()
    assert all(set(r) == {'value', 'advantage', 'returns'} for r in a)
    for x, y in zip(a, b):
        assert x['advantage'] == y['advantage'] and x['returns'] == y['returns']
    np.testing.assert_allclose([r['upgo_advantage'] for r in b], [.5 - .2, 1. - .5, 1. - .1], atol=1e-6)
    assert [r['upgo_follow'] for r in b] == [False, True, False]
    stacked = stack_rows(b)
    assert stacked['upgo_advantage'].dtype == np.float32 and stacked['upgo_follow'].dtype == bool


def test_upgo_loss_is_off_by_default_and_adds_the_clipped_surrogate():
    import jax
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import LearnerConfig
    from test_structured_rl import tiny_policy
    assert LearnerConfig().upgo_coef == 0.
    size, x, params, masks = tiny_policy()
    choice, path, logp, _ = jax.vmap(lambda k: network.sample(params, x, masks, k, size=size))(
        jax.random.split(jax.random.key(3), 4))
    features = jax.tree.map(lambda a: jnp.broadcast_to(a, (4,) + a.shape), x)
    batch = dict(features=features, choice=choice, masks=path, logp=logp,
                 advantage=jnp.array([1., -1., .5, -.5]), returns=jnp.zeros(4))
    u = jnp.array([2., -1., 0., 1.])
    base, m0 = ppo.loss(params, batch)
    same, m1 = ppo.loss(params, dict(batch, upgo_advantage=u), upgo_coef=0.)
    assert float(base) == float(same) and m1.keys() == m0.keys()
    with_upgo, m2 = ppo.loss(params, dict(batch, upgo_advantage=u), upgo_coef=.7)
    # behaviour logp equals the current logp: ratio 1, so the surrogate is -mean(u)
    np.testing.assert_allclose(float(m2['upgo_loss']), -float(u.mean()), atol=1e-5)
    np.testing.assert_allclose(float(with_upgo), float(base) + .7 * float(m2['upgo_loss']), rtol=1e-5)


def test_turn_clock_undiscounted_gamma_one():
    """turn_gamma 1.0 (c6-credit arm C): the outcome reaches every decision of a finished game undiminished, lambda
    still traces only at the seat's end turns, and a truncated game (round cap / action limit, outcome 0, terminal
    False) bootstraps the critic exactly as it does under gamma < 1."""
    from polytopia_bot.rl.distributed import LearnerConfig
    LearnerConfig(turn_gamma=1., turn_lambda=.95).validate()
    rng = np.random.default_rng(3)
    values = rng.normal(size=12).astype(np.float32)
    end = np.zeros(12, bool)
    end[[3, 7, 11]] = True
    # win with lambda = 1: every return is the outcome
    _, r = gae(values, .3, 1., True, 1., 1., end_turn=end)
    np.testing.assert_allclose(r, 1., atol=1e-6)
    # win with lambda = .95: finite, bounded by the outcome/values, last turn sees the outcome undiscounted
    a, r = gae(values, .3, 1., True, 1., .95, end_turn=end)
    assert np.isfinite(a).all() and np.isfinite(r).all()
    assert np.abs(r).max() <= max(1., float(np.abs(values).max())) + 1e-6
    np.testing.assert_allclose(r[8:], 1., atol=1e-6)
    # earlier turns: GAE(lambda) mixes the outcome with the next turns' values, 1 - lambda per end turn
    expected_8 = .95 * 1. + .05 * values[8]  # return at index 7: (1 - lam) * V_8 + lam * G_8
    np.testing.assert_allclose(r[7], expected_8, atol=1e-5)
    # truncation: terminal False, outcome 0, critic bootstrap .3 -> the last turn's returns are the bootstrap
    a1, r1 = gae(values, .3, 0., False, 1., .95, end_turn=end)
    a9, r9 = gae(values, .3, 0., False, .99, .95, end_turn=end)
    assert np.isfinite(a1).all() and np.isfinite(r1).all()
    np.testing.assert_allclose(r1[8:], .3, atol=1e-6)
    np.testing.assert_allclose(r9[8:], .99 * .3, atol=1e-6)
    # a long segmented game through SeatTrajectories stays finite and ends on the outcome
    traj = SeatTrajectories(lanes=1, segment=32, gamma=1., lam=.95, turn_clock=True)
    for k in range(200):
        traj.add(0, 1, dict(value=np.float32(rng.normal() * .3), end_turn=bool(k % 6 == 5)))
    traj.finish(0, np.array([1., -1.], np.float32), True, np.zeros(2, np.float32))
    rows = traj.take()
    assert len(rows) == 200
    adv = np.array([r['advantage'] for r in rows])
    ret = np.array([r['returns'] for r in rows])
    assert np.isfinite(adv).all() and np.isfinite(ret).all() and np.abs(ret).max() <= 1.5
    np.testing.assert_allclose(ret[-1], 1., atol=1e-6)


def test_critic_path_loads_a_saved_omniscient_critic(tmp_path):
    """critic_path: the learner's omniscient critic starts as the parent's saved critic, leaf for leaf, instead of
    a fresh template; it needs omniscient_critic; without it the critic is None or the fresh template."""
    import jax
    import pytest
    from polytopia_bot.rl.distributed import LearnerConfig, _critic_template, initial_critic
    from polytopia_bot.rl.runner import Config, save_checkpoint
    with pytest.raises(ValueError):
        LearnerConfig(critic_path='x').validate()
    network = dict(size=14, width=8, blocks=1)
    saved = jax.tree.map(lambda a: a + .5 if jnp.issubdtype(a.dtype, jnp.floating) else a, _critic_template(**network))
    path = tmp_path / 'final-critic.npz'
    save_checkpoint(path, saved, Config(), stage='final')
    cfg = LearnerConfig(omniscient_critic=True, critic_path=str(path))
    cfg.validate()
    logged = []
    loaded = initial_critic(cfg, network, log=logged.append)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(saved), strict=True))
    assert logged and 'critic_loaded' in logged[0]
    assert initial_critic(LearnerConfig(), network) is None
    fresh = initial_critic(LearnerConfig(omniscient_critic=True), network, log=logged.append)
    assert not np.array_equal(jax.tree.leaves(fresh)[0], jax.tree.leaves(saved)[0])


def _collect_both(rng, lanes, segment, ticks, *, turn_clock, shaping, upgo, width=5):
    """Feed the same random decisions to SeatTrajectories (values in rows, GAE on the actor) and to
    SegmentCollector (learner-side); return the actor rows and the learner-side (rows, segments, values by row)."""
    from polytopia_bot.rl.distributed import SegmentCollector
    gamma, lam = (.97, .9)
    actor = SeatTrajectories(lanes, segment, gamma, lam, turn_clock=turn_clock, shaping=shaping, upgo=upgo)
    learner = SegmentCollector(lanes, segment)
    value_of = {}       # view bytes -> value, so the learner side can be scored like the actor side
    counter = [0]

    def fresh_row(seat):
        counter[0] += 1
        view = np.full(width, counter[0], np.int16)
        value = np.float32(rng.normal() * .5)
        value_of[view.tobytes()] = value
        row = dict(view=view, opp_view=-view, logp=np.float32(-1.), version=np.int32(0))
        if turn_clock:
            row['end_turn'] = bool(rng.random() < .25)
        if shaping:
            row['phi'] = np.float32(rng.normal() * .1)
        return row, value

    out_actor, out_rows, out_segs = [], [], []
    for _ in range(ticks):
        for lane in range(lanes):
            seat = int(rng.integers(1, 3))
            row, value = fresh_row(seat)
            actor.add(lane, seat, dict(row, value=value))
            learner.add(lane, seat, dict(row))
            if rng.random() < .03:
                terminated = rng.random() < .7
                rewards = np.array([1., -1.], np.float32) * (1 if rng.random() < .5 else -1)
                final = np.stack([np.full(width, -7 - lane, np.int16), np.full(width, -70 - lane, np.int16)])
                boot = np.array([value_of.setdefault(final[0].tobytes(), np.float32(rng.normal())),
                                 value_of.setdefault(final[1].tobytes(), np.float32(rng.normal()))], np.float32)
                actor.finish(lane, rewards, terminated, boot)
                learner.finish(lane, rewards, terminated, final)
        out_actor.extend(actor.take())
        rows, segs = learner.take()
        offset = len(out_segs)
        for r in rows:
            r['seg'] = r['seg'] + offset
        out_rows.extend(rows)
        out_segs.extend(segs)
    return out_actor, out_rows, out_segs, value_of, (gamma, lam)


def test_learner_segment_gae_matches_actor_gae():
    """Learner-side GAE (SegmentCollector + segment_gae) equals the actor-side numbers (SeatTrajectories + gae)
    when both score rows and boundaries with the same values: per-decision and turn clocks, PBRS shaping, UPGO,
    segment cuts, native endings and truncation bootstraps."""
    from polytopia_bot.rl.distributed import segment_gae, stack_rows
    rng = np.random.default_rng(5)
    for turn_clock, shaping, upgo in ((False, False, False), (True, False, False), (False, True, True), (True, True, True)):
        actor_rows, rows, segs, value_of, (gamma, lam) = _collect_both(rng, 3, 8, 200, turn_clock=turn_clock,
                                                                      shaping=shaping, upgo=upgo)
        assert len(actor_rows) == len(rows) > 100
        data, table = stack_rows(rows), stack_rows(segs)
        assert 'value' not in data and 'advantage' not in data
        values = np.array([value_of[v.tobytes()] for v in data['view']], np.float32)
        boot = np.array([value_of.get(v.tobytes(), 0.) for v in table['boot_view']], np.float32)
        # the boundary of a cut segment is the next decision of the same seat; a native ending ships zeros
        cut = ~table['terminal']
        assert cut.any() and (table['boot_view'][~cut] == 0).all()
        out = segment_gae(values, data['seg'], data['pos'], table['length'], table['terminal'], table['reward'], boot,
                          gamma, lam, end_turn=data['end_turn'] if turn_clock else None,
                          phi=data['phi'] if shaping else None, phi_next=table.get('phi_next'), upgo=upgo)
        ref_adv = np.array([r['advantage'] for r in actor_rows], np.float32)
        ref_ret = np.array([r['returns'] for r in actor_rows], np.float32)
        # same rows in the same order
        assert all(np.array_equal(a['view'], b['view']) for a, b in zip(actor_rows, rows))
        np.testing.assert_allclose(out[0], ref_adv, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[1], ref_ret, rtol=1e-5, atol=1e-6)
        if upgo:
            np.testing.assert_allclose(out[2], [r['upgo_advantage'] for r in actor_rows], rtol=1e-5, atol=1e-6)
            assert out[3].tolist() == [r['upgo_follow'] for r in actor_rows]
        # dropping a segment's oldest rows (stale) leaves the others unchanged
        drop = (data['pos'] == 0) & (data['seg'] % 2 == 0)
        keep = ~drop
        sub = segment_gae(values[keep], data['seg'][keep], data['pos'][keep], table['length'], table['terminal'],
                          table['reward'], boot, gamma, lam, end_turn=data['end_turn'][keep] if turn_clock else None,
                          phi=data['phi'][keep] if shaping else None, phi_next=table.get('phi_next'), upgo=upgo)
        np.testing.assert_allclose(sub[0], ref_adv[keep], rtol=1e-5, atol=1e-6)


def test_segment_collector_boundaries_and_config_defaults():
    from polytopia_bot.rl.distributed import LearnerConfig, SegmentCollector
    cfg = LearnerConfig()
    assert cfg.learner_values and cfg.trusted_step and cfg.build_rows and cfg.pool_per_process
    assert cfg.trusted_check_every == 64
    cfg.validate()
    import pytest
    with pytest.raises(ValueError):
        LearnerConfig(trusted_check_every=-1).validate()
    c = SegmentCollector(lanes=1, segment=3)
    for i in range(5):
        c.add(0, 1, dict(view=np.full(2, i, np.int16), opp_view=np.full(2, 10 + i, np.int16)))
    rows, segs = c.take()
    assert [r['pos'] for r in rows] == [0, 1, 2] and all(r['seg'] == 0 for r in rows)
    assert len(segs) == 1 and segs[0]['length'] == 3 and not segs[0]['terminal']
    np.testing.assert_array_equal(segs[0]['boot_view'], [3, 3])          # the 4th decision bootstraps the cut
    np.testing.assert_array_equal(segs[0]['boot_opp_view'], [13, 13])
    final = np.array([[-1, -1], [-2, -2]], np.int16)
    c.finish(0, np.array([0., 0.]), False, final)                        # truncation: the final state's views
    rows, segs = c.take()
    assert len(rows) == 2 and segs[0]['length'] == 2 and rows[0]['seg'] == 0   # indexes this take's table
    np.testing.assert_array_equal(segs[0]['boot_view'], final[0])
    np.testing.assert_array_equal(segs[0]['boot_opp_view'], final[1])
    c.add(0, 2, dict(view=np.zeros(2, np.int16), opp_view=np.zeros(2, np.int16)))
    c.finish(0, np.array([1., -1.]), True, final)
    rows, segs = c.take()
    assert segs[0]['terminal'] and segs[0]['reward'] == -1. and (segs[0]['boot_view'] == 0).all()


def test_adapt_entropy_coef_moves_toward_target_and_clips():
    from polytopia_bot.rl.distributed import adapt_entropy_coef
    assert adapt_entropy_coef(.01, 2.0, 3.0, .05) > .01          # below target: more bonus
    assert adapt_entropy_coef(.01, 3.5, 3.0, .05) < .01          # above target: less bonus
    assert adapt_entropy_coef(.01, 3.0, 3.0, .05) == .01
    assert adapt_entropy_coef(.049, 0., 10., 1.) == .05           # clipped high
    assert adapt_entropy_coef(2e-4, 10., 0., 1.) == 1e-4          # clipped low


def test_scheduled_learning_rate_decays_over_the_last_fraction():
    from polytopia_bot.rl.distributed import scheduled_learning_rate
    assert scheduled_learning_rate(3e-4, None, .1, .95) == 3e-4    # no end: constant
    assert scheduled_learning_rate(3e-4, 1e-4, .1, 0.) == 3e-4
    assert scheduled_learning_rate(3e-4, 1e-4, .1, .9) == 3e-4     # decay starts at 1 - fraction
    assert abs(scheduled_learning_rate(3e-4, 1e-4, .1, .95) - 2e-4) < 1e-12
    assert scheduled_learning_rate(3e-4, 1e-4, .1, 1.) == 1e-4
    assert scheduled_learning_rate(3e-4, 1e-4, .1, 1.3) == 1e-4    # past the deadline: clamped
    assert scheduled_learning_rate(3e-4, 1e-4, 0., .99) == 3e-4    # zero fraction: a step at the end
