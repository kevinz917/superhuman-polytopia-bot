"""Transformer V3 (rl/transformer_v3.py): shapes, the one-way information contract, and one-encoder equivalence."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from polytopia_bot.rl import network, ppo, transformer_v3
from polytopia_bot.rl.actions import Choice, PathMasks
from polytopia_bot.rl.distributed import _critic_template, _template_params, feature_shapes
from polytopia_bot.rl.observation import OBS_VERSION, Features

SIZE = 14
TILES = SIZE * SIZE
ARCHITECTURES = ('transformer-v3', 'transformer-v3-p2', 'transformer-v3-c0', 'transformer-v3-p2-c2', 'transformer-v3-p2-k3',
                 'transformer-v3-p2-h2', 'transformer-v3-p2-h4', 'transformer-v3-h1-c0')


def _spec(architecture, width=32, blocks=2):
    return dict(size=SIZE, blocks=blocks, width=width, architecture=architecture, obs_version=OBS_VERSION)


def _features(key, board, glob):
    a, b = jax.random.split(key)
    return Features(jax.random.bernoulli(a, .2, (SIZE, SIZE, board)).astype(jnp.float32),
                    jax.random.normal(b, (glob,)))


def _full(own, opp):
    return Features(jnp.concatenate((own.board, opp.board), -1), jnp.concatenate((own.global_state, opp.global_state)))


def _masks(key):
    a, b, c = jax.random.split(key, 3)
    family = jax.random.bernoulli(a, .5, (17,)).at[0].set(True)
    source = jax.random.bernoulli(b, .5, (TILES,)).at[3].set(True)
    target = jax.random.bernoulli(c, .5, (max(TILES, 128),)).at[5].set(True)
    return PathMasks(family, source, target)


def _engine_masks(key):
    """Random engine legal masks (gpu.actions.Masks leaves), every leaf about half legal."""
    from polytopia_bot.gpu.actions import Masks
    from polytopia_bot.rl.actions import leaf_shapes
    keys = jax.random.split(key, len(leaf_shapes(SIZE)))
    return Masks(*(jax.random.bernoulli(k, .5, shape) for k, shape in zip(keys, leaf_shapes(SIZE))))


@pytest.fixture(scope='module')
def shapes():
    return feature_shapes(2, OBS_VERSION)


@pytest.mark.parametrize('architecture', ARCHITECTURES)
def test_templates_and_parse(architecture, shapes):
    params = _template_params(**_spec(architecture))
    critic = _critic_template(**_spec(architecture))
    options = transformer_v3.parse(architecture)
    assert ('v3_patch' in params) == (options['patch'] == 2)
    assert len(critic['blocks']) == options['critic_blocks']
    assert params['v3_stem']['w'].shape == ((3, 3, shapes[0], 32) if options['conv'] else (shapes[0], 32))
    own = _features(jax.random.key(0), *shapes)
    tiles, g, value = network.trunk(params, own)
    assert tiles.shape == (TILES, 32) and g.shape == (32,) and float(value) == 0.


@pytest.mark.parametrize('name', ('transformer-v2', 'transformer-v3-p3', 'transformer-v3-c', 'v3', 'transformer-v3-k3-p2',
                                  'transformer-v3-p2-k3-h2', 'transformer-v3-p2-h0', 'transformer-v3-h2-p2'))
def test_parse_rejects_other_names(name):
    assert not transformer_v3.is_v3(name)
    with pytest.raises(ValueError):
        transformer_v3.parse(name)


def test_patch_layout_round_trips():
    tiles = jnp.arange(TILES * 3, dtype=jnp.float32).reshape(TILES, 3)
    patches = transformer_v3._to_patches(tiles, SIZE, 2)
    assert patches.shape == (49, 12)
    # patch 0 holds tiles (0,0), (0,1), (1,0), (1,1) in that order
    np.testing.assert_array_equal(patches[0].reshape(4, 3), tiles[jnp.array([0, 1, SIZE, SIZE + 1])])
    np.testing.assert_array_equal(transformer_v3._from_patches(patches, SIZE, 2), tiles)


@pytest.mark.parametrize('architecture', ARCHITECTURES)
def test_policy_never_reads_privileged_input(architecture, shapes):
    """DESIGN.md leakage tests 1, 2 and 3: vary only the other seat's view; the policy is unchanged (exactly), its
    derivative w.r.t. the privileged input is zero, and the critic responds."""
    params = jax.tree.map(lambda a: a + .05, _template_params(**_spec(architecture)))   # nonzero relative biases
    critic = _critic_template(**_spec(architecture))
    own = _features(jax.random.key(1), *shapes)
    opp_a, opp_b = _features(jax.random.key(2), *shapes), _features(jax.random.key(3), *shapes)
    masks, choice = _masks(jax.random.key(4)), Choice(jnp.int32(0), jnp.int32(3), jnp.int32(5))
    logp_a, value_a, ent_a = transformer_v3.evaluate(params, critic, _full(own, opp_a), choice, masks)
    logp_b, value_b, ent_b = transformer_v3.evaluate(params, critic, _full(own, opp_b), choice, masks)
    assert float(logp_a) == float(logp_b) and float(ent_a) == float(ent_b)
    assert abs(float(value_a) - float(value_b)) > 1e-6

    def policy_logp(opp_board, opp_glob):
        return transformer_v3.evaluate(params, critic, _full(own, Features(opp_board, opp_glob)), choice, masks)[0]
    grads = jax.grad(policy_logp, argnums=(0, 1))(opp_a.board, opp_a.global_state)
    assert all(float(jnp.abs(g).max()) == 0. for g in grads)
    # the public-only actor path (network.evaluate on the own view) is the learner's policy
    logp_actor, _, ent_actor = network.evaluate(params, own, choice, masks)
    np.testing.assert_allclose(logp_actor, logp_a, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ent_actor, ent_a, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('architecture', ('transformer-v3', 'transformer-v3-p2', 'transformer-v3-p2-k3', 'transformer-v3-p2-h4'))
def test_one_encoder_paths_match_separate_calls(architecture, shapes):
    params = _template_params(**_spec(architecture))
    critic = _critic_template(**_spec(architecture))
    own, opp = _features(jax.random.key(5), *shapes), _features(jax.random.key(6), *shapes)
    full, masks, key = _full(own, opp), _engine_masks(jax.random.key(7)), jax.random.key(8)
    choice, path, logp, value = transformer_v3.sample_and_value(params, critic, full, masks, key, size=SIZE)
    choice_ref, path_ref, logp_ref, _ = network.sample(params, own, masks, key, size=SIZE)
    assert tuple(int(c) for c in choice) == tuple(int(c) for c in choice_ref)
    assert all(bool(jnp.all(a == b)) for a, b in zip(path, path_ref))
    np.testing.assert_allclose(logp, logp_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(value, transformer_v3.value(params, critic, full), rtol=1e-6, atol=1e-6)


def test_value_gradient_trains_the_shared_encoder(shapes):
    params = _template_params(**_spec('transformer-v3'))
    critic = _critic_template(**_spec('transformer-v3'))
    full = _full(_features(jax.random.key(9), *shapes), _features(jax.random.key(10), *shapes))
    grads = jax.grad(lambda p: transformer_v3.value(p, critic, full))(params)
    assert float(jnp.abs(grads['v3_stem']['w']).max()) > 0.
    # the policy-only heads get no value gradient
    assert float(jnp.abs(grads['family']['w']).max()) == 0.


def test_policy_objective_matches_the_inline_ppo_loss(shapes):
    """ppo.loss after extracting policy_objective computes the original expression."""
    params = network.init(jax.random.key(0), shapes[0], shapes[1], size=SIZE, width=32, blocks=1,
                          architecture='normalized-v2')
    rows = 4
    feats = jax.vmap(lambda k: _features(k, *shapes))(jax.random.split(jax.random.key(11), rows))
    masks = jax.vmap(_masks)(jax.random.split(jax.random.key(12), rows))
    choice = Choice(jnp.zeros(rows, jnp.int32), jnp.full(rows, 3, jnp.int32), jnp.full(rows, 5, jnp.int32))
    batch = dict(features=feats, masks=masks, choice=choice, logp=jnp.full(rows, -2.),
                 advantage=jnp.array([1., -1., .5, 2.]), returns=jnp.array([.1, .2, -.3, .4]))
    total, metrics = ppo.loss(params, batch, clip=.2, value_coef=.5, entropy_coef=.01)
    logp, value, entropy = jax.vmap(network.evaluate, in_axes=(None, 0, 0, 0))(params, feats, choice, masks)
    ratio = jnp.exp(logp - batch['logp'])
    adv = batch['advantage']
    expected = (-jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, .8, 1.2) * adv))
                + .5 * .5 * jnp.mean((value - batch['returns']) ** 2)
                - .01 * ppo.joint_entropy(logp, batch['logp']).mean())
    np.testing.assert_allclose(total, expected, rtol=1e-6)
    np.testing.assert_allclose(metrics['loss'], expected, rtol=1e-6)
    np.testing.assert_allclose(metrics['path_entropy'], entropy.mean(), rtol=1e-6)


def test_checkpoint_round_trip(tmp_path, shapes):
    from dataclasses import replace
    from polytopia_bot.rl.distributed import network_spec
    from polytopia_bot.rl.runner import Config, load_checkpoint, save_checkpoint
    spec = _spec('transformer-v3-p2-c2')
    params = jax.tree.map(lambda a: a + 1., _template_params(**spec))
    config = replace(Config(), architecture=spec['architecture'], width=spec['width'], blocks=spec['blocks'],
                     size=SIZE, obs_version=OBS_VERSION)
    config.validate()
    path = tmp_path / 'v3.npz'
    save_checkpoint(path, params, config, stage='test')
    assert network_spec(path) == dict(spec, extra_blocks=0)
    loaded = load_checkpoint(path, _template_params(**network_spec(path)))
    assert all(bool(jnp.all(a == b)) for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(params)))


@pytest.mark.parametrize('cnn_blocks', (1, 4))
def test_hybrid_torso_is_the_hero_cnn(cnn_blocks, shapes):
    """'-h<n>': with the hero CNN's (normalized-v2, n blocks) stem and block kernels copied in, the hybrid's tile
    embedding equals the CNN trunk's tile features: the transformer reads exactly the hero CNN's convolutional output."""
    cnn = network.init(jax.random.key(3), *shapes, size=SIZE, width=32, blocks=cnn_blocks, architecture='normalized-v2')
    hybrid = _template_params(**_spec(f'transformer-v3-p2-h{cnn_blocks}'))
    assert len(hybrid['v3_conv_blocks']) == cnn_blocks
    for got, want in zip(jax.tree_util.tree_leaves(hybrid['v3_conv_blocks']), jax.tree_util.tree_leaves(cnn['blocks'])):
        assert got.shape == want.shape
        # the hero CNN's initialization scale (fan-in 3 x 3 x width), within sampling error
        np.testing.assert_allclose(float(got.std()), float(want.std()), rtol=.05)
    hybrid = dict(hybrid, v3_stem=dict(hybrid['v3_stem'], w=cnn['stem']), v3_conv_blocks=cnn['blocks'])
    own = _features(jax.random.key(4), *shapes)
    want, _, _ = network.trunk(cnn, own)
    _, embedding = transformer_v3._stem(hybrid['v3_stem'], hybrid['v3_patch'], own.board, SIZE, 2,
                                        hybrid['v3_conv_blocks'])
    np.testing.assert_allclose(embedding, want, rtol=1e-5, atol=1e-5)


def test_hybrid_policy_trains_the_conv_torso(shapes):
    params = _template_params(**_spec('transformer-v3-p2-h2'))
    own = _features(jax.random.key(5), *shapes)
    grads = jax.grad(lambda p: network.trunk(p, own)[1].sum())(params)
    for kernel in jax.tree_util.tree_leaves(grads['v3_conv_blocks']):
        assert float(jnp.abs(kernel).max()) > 0.
