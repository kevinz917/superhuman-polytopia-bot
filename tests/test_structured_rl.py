"""Probability, information-boundary, and credit-assignment regression tests."""
import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from polytopia_bot.gpu import actions as engine_actions, core, kernels
from polytopia_bot.gpu.rules import load_rules
from polytopia_bot.gpu.state import empty_state
from polytopia_bot.learning.adapter import serialize
from polytopia_bot.rl import actions, network, ppo
from polytopia_bot.rl.observation import Features, encode_view, view_from_public
from polytopia_bot.rl.runner import Config, save_checkpoint, load_checkpoint


def fixture(size=3):
    s = empty_state(size)
    s = s._replace(tiles=s.tiles._replace(explored=jnp.ones((2, size * size), bool)))
    return encode_view(kernels.observe(s, s.current_player), size)


@pytest.mark.parametrize('size', [11, 14])
def test_complete_catalog_roundtrip(size):
    for f, (a, b) in enumerate(actions.shapes(size)):
        for src, dst in [(0, 0), (a - 1, b - 1)]:
            choice = actions.Choice(f, src, dst)
            index = int(actions.flatten_choice(choice, size))
            assert actions.unflatten(index, size) == choice
    s = empty_state(size)
    masks = engine_actions.legal_masks(s, load_rules(), size=size)
    assert actions.offsets(size)[-1] == engine_actions.flatten(masks).size
    assert tuple(m.size for m in masks) == tuple(a * b for a, b in actions.shapes(size))


def test_public_encoder_matches_existing_demonstration_interface():
    size = 11
    s = empty_state(size)
    s = s._replace(tiles=s.tiles._replace(explored=s.tiles.explored.at[0, :8].set(True)),
                   players=s.players._replace(stars=jnp.array([7, 99]), tech=s.players.tech.at[0, 6].set(True)))
    view = kernels.observe(s, jnp.int32(1))
    public = dataclasses.asdict(serialize(view, size=size, legal=[]))
    a, b = encode_view(view, size), encode_view(view_from_public(public), size)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_policy_encoder_excludes_hidden_bank_technology_and_seed():
    size = 11
    a = empty_state(size)
    b = a._replace(seed=jnp.uint32(777), players=a.players._replace(stars=jnp.array([0, 900]),
            tech=a.players.tech.at[1].set(True)), units=a.units._replace(owner=a.units.owner.at[-1].set(2)))
    for x, y in zip(encode_view(kernels.observe(a, 1), size), encode_view(kernels.observe(b, 1), size), strict=True):
        np.testing.assert_array_equal(x, y)


def test_gae_same_turn_then_opponent_terminal_loss():
    # Seat 1 acts twice; seat 2 resigns. Exact Monte Carlo with lambda=1.
    r = jnp.array([[0.], [0.], [-1.]])
    v = jnp.zeros_like(r)
    adv, ret = ppo.advantages(r, v, v, jnp.array([[True], [False], [True]]),
                              jnp.array([[False], [False], [True]]), jnp.zeros_like(r, bool), lam=1.)
    np.testing.assert_allclose(adv[:, 0], [1, 1, -1])
    np.testing.assert_array_equal(adv, ret)


def test_truncation_bootstraps_final_position_but_cuts_reset_episode():
    r = jnp.array([[0.], [100.]])
    v = jnp.zeros_like(r)
    adv, _ = ppo.advantages(r, v, jnp.array([[.7], [0.]]), jnp.array([[False], [True]]),
                            jnp.array([[False], [True]]), jnp.array([[True], [False]]), lam=1.)
    np.testing.assert_allclose(adv[:, 0], [-.7, 100])


def test_rollout_boundary_bootstrap_and_terminal_ignore_next_value():
    r = jnp.array([[1., 0.]])
    adv, _ = ppo.advantages(r, jnp.array([[.2, .3]]), jnp.array([[99., .8]]),
        jnp.array([[False, True]]), jnp.array([[True, False]]), jnp.zeros((1, 2), bool), lam=1.)
    np.testing.assert_allclose(adv, [[.8, .5]])


def tiny_policy():
    size = 3
    x = fixture(size)
    params = network.init(jax.random.key(1), x.board.shape[-1], x.global_state.size, size=size, width=8, blocks=1)
    ms = [jnp.zeros(s, bool) for s in actions.shapes(size)]
    ms[0] = ms[0].at[1, 2].set(True).at[1, 4].set(True).at[3, 6].set(True)
    ms[4] = ms[4].at[0, 6].set(True)
    ms[15] = ms[15].at[0, 0].set(True)
    return size, x, params, engine_actions.Masks(*ms)


def test_joint_distribution_normalizes_and_selected_masks_are_exact():
    size, x, params, masks = tiny_policy()
    choices = [actions.Choice(0, 1, 2), actions.Choice(0, 1, 4), actions.Choice(0, 3, 6),
               actions.Choice(4, 0, 6), actions.Choice(15, 0, 0)]
    probabilities = [np.exp(float(network.evaluate(params, x, c, actions.path_masks(masks, c, size))[0])) for c in choices]
    assert sum(probabilities) == pytest.approx(1, abs=1e-6)
    for i in range(8):
        c, path, lp, _ = network.sample(params, x, masks, jax.random.key(i), size=size)
        assert bool(engine_actions.flatten(masks)[actions.flatten_choice(c, size)])
        expected = network.evaluate(params, x, c, path)[0]
        np.testing.assert_allclose(lp, expected, atol=1e-6)
        offline_c, offline_m = actions.from_catalog(np.flatnonzero(engine_actions.flatten(masks)), int(actions.flatten_choice(c, size)), size)
        assert tuple(int(a) for a in c) == offline_c
        for a, b in zip(path, offline_m, strict=True):
            np.testing.assert_array_equal(a, b)


def test_ppo_positive_advantage_increases_action_probability_and_reload(tmp_path):
    size, x, params, masks = tiny_policy()
    c = actions.Choice(4, 0, 6)
    path = actions.path_masks(masks, c, size)
    lp, value, _ = network.evaluate(params, x, c, path)
    batch = dict(features=jax.tree.map(lambda a: a[None], x), choice=jax.tree.map(lambda a: jnp.asarray([a]), c),
                 masks=jax.tree.map(lambda a: a[None], path), logp=jnp.array([lp]),
                 advantage=jnp.ones(1), returns=jnp.array([value]))
    update = ppo.make_update(entropy_coef=0)
    new, opt, metrics = update(params, ppo.adam_init(params), batch)
    assert float(network.evaluate(new, x, c, path)[0]) > float(lp)
    assert all(np.isfinite(v).all() for v in jax.tree.leaves((new, metrics)))
    path = tmp_path / 'model.npz'
    save_checkpoint(path, new, Config(), stage='test', optimizer=opt)
    loaded = load_checkpoint(path, params)
    for a, b in zip(jax.tree.leaves(new), jax.tree.leaves(loaded), strict=True):
        np.testing.assert_array_equal(a, b)


def test_configuration_rejects_partial_minibatches():
    with pytest.raises(ValueError):
        Config(environments=3, horizon=3, minibatch=8).validate()


def test_public_bot_adapter_returns_a_legal_engine_action(tmp_path):
    from polytopia_bot.environment import LegalAction
    from polytopia_bot.rl.policy import StructuredPolicy
    size = 11
    s = empty_state(size)
    view = kernels.observe(s, 1)
    x = encode_view(view, size)
    params = network.init(jax.random.key(3), x.board.shape[-1], x.global_state.size,
                          size=size, width=8, blocks=1)
    config = Config(size=size, width=8, blocks=1)
    path = tmp_path / 'bot.npz'
    save_checkpoint(path, params, config, stage='adapter-test')
    # A forced reward means END_TURN is unavailable: only that legal command
    # may be returned, despite the policy having all 17 heads.
    index = int(actions.flatten_choice(actions.Choice(7, 5, 3), size))
    command = core.action(9, target=5, parameter=3)
    legal = LegalAction(str(index), {k: int(v) for k, v in zip(core.Action._fields, command)})
    obs = serialize(view, size=size, legal=[legal])
    answer = StructuredPolicy(path).select(obs)
    assert answer.action == legal
    assert answer.log_probability == pytest.approx(0)


def test_rollout_and_larger_learner_batch_log_probabilities_match():
    size, x, params, masks = tiny_policy()
    sample = jax.jit(jax.vmap(lambda k: network.sample(params, x, masks, k, size=size)))
    c, paths, old_logp, _ = sample(jax.random.split(jax.random.key(31), 8))
    features = jax.tree.map(lambda a: jnp.broadcast_to(a, (8,) + a.shape), x)
    repeat = lambda tree: jax.tree.map(lambda a: jnp.tile(a, (4,) + (1,) * (a.ndim - 1)), tree)
    evaluate = jax.jit(jax.vmap(network.evaluate, in_axes=(None, 0, 0, 0)))
    lp, _, _ = evaluate(params, repeat(features), repeat(c), repeat(paths))
    np.testing.assert_allclose(lp, jnp.tile(old_logp, 4), atol=1e-5, rtol=0)
