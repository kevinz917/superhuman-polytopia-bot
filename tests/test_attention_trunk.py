"""Transformer blocks over the tiles after the convolutions (network.init attention, lookout 2026-11-01): a hybrid
trunk whose heads are the normalized-v2 heads. The checkpoint spec records the block count, every template builder
(policy, critic, Policy wrapper) reproduces the parameter tree, the forward pass is deterministic and finite on real
features, gradients reach every attention parameter, and a saved checkpoint round-trips through network_spec."""
import json

import numpy as np
import pytest

SIZE = 11


def _features(seed=5):
    import jax.numpy as jnp
    from polytopia_bot.rl.cpu_engine import unpack_view
    from polytopia_bot.rl.observation import encode_view
    from polytopia_bot.rl.rust_engine import RustGames, split_masks
    from test_obs_history import _random_choice   # pytest puts tests/ on sys.path
    g = RustGames(SIZE, 2, max_turns=30)
    g.batch.reset(np.array([seed, seed + 1], np.int32))
    rng = np.random.default_rng(seed)
    for _ in range(150):
        g.batch.step(*_random_choice(split_masks(g.batch.legal_masks(), SIZE), rng, size=SIZE))
    own = [encode_view(unpack_view(jnp.asarray(r), g.layout), SIZE) for r in np.asarray(g.batch.observe())]
    opp = [encode_view(unpack_view(jnp.asarray(r), g.layout), SIZE) for r in np.asarray(g.batch.observe_opponent())]
    return own, opp


def test_attention_trunk_shapes_gradients_and_spec(tmp_path):
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import (_critic_template, _template_params, checkpoint_template, network_spec,
                                              omniscient_features)
    from polytopia_bot.rl.runner import Config, load_checkpoint, save_checkpoint
    own, opp = _features()
    width, blocks, attention = 64, 2, 3
    policy = _template_params(SIZE, blocks=blocks, width=width, obs_version=3, attention=attention)
    critic = _critic_template(SIZE, blocks=blocks, width=width, obs_version=3, attention=attention)
    assert len(policy['attention']) == attention and policy['pos'].shape == (SIZE * SIZE, width)
    assert policy['attention'][0]['qkv']['w'].shape == (width, 3 * width)
    assert policy['attention'][0]['mlp_in']['w'].shape == (width, 4 * width)
    # the forward pass: tiles per board cell, a global vector, a finite scalar value; deterministic
    for x, o in zip(own, opp):
        tiles, g, value = network.trunk(policy, x)
        assert tiles.shape == (SIZE * SIZE, width) and g.shape == (width,) and value.shape == ()
        assert bool(jnp.all(jnp.isfinite(tiles))) and bool(jnp.isfinite(value))
        again = network.trunk(policy, x)
        np.testing.assert_array_equal(np.asarray(tiles), np.asarray(again[0]))
        both = omniscient_features(x, o)
        assert bool(jnp.isfinite(network.trunk(critic, both)[2]))
    # attention changes the tiles (it is not the identity at init) and every attention parameter gets a gradient
    plain = dict(policy)
    for key in ('attention', 'pos', 'attention_norm'):
        plain.pop(key)
    assert float(jnp.abs(network.trunk(plain, own[0])[0] - network.trunk(policy, own[0])[0]).sum()) > 0
    grads = jax.grad(lambda p: jnp.sum(network.trunk(p, own[0])[0] ** 2) + network.trunk(p, own[0])[2])(policy)
    for block in grads['attention']:
        for name, layer in block.items():
            for leaf in jax.tree.leaves(layer):
                assert float(jnp.abs(leaf).sum()) > 0, name
    assert float(jnp.abs(grads['pos'][:SIZE * SIZE]).sum()) > 0
    # the heads sample legal actions through the usual path
    from polytopia_bot.rl.actions import Masks
    from polytopia_bot.rl.rust_engine import RustGames, split_masks
    g = RustGames(SIZE, 1, max_turns=30)
    g.batch.reset(np.array([9], np.int32))
    masks = split_masks(g.batch.legal_masks(), SIZE)
    m = Masks(*(jnp.asarray(np.asarray(a)[0]) for a in masks))
    x = own[0]
    choice, path, logp, value = network.sample(policy, x, m, jax.random.key(0), size=SIZE)
    assert bool(jnp.isfinite(logp)) and bool(jnp.isfinite(value))
    # checkpoint spec round trip: a saved attention network reloads into the same tree through network_spec
    out = tmp_path / 'lookout.npz'
    save_checkpoint(out, policy, Config(architecture='normalized-v2', size=SIZE, width=width, blocks=blocks,
                                        obs_version=3, attention=attention), stage='test')
    spec = network_spec(out)
    assert spec['attention'] == attention and spec['obs_version'] == 3
    loaded = load_checkpoint(out, checkpoint_template(out))
    assert jax.tree.structure(loaded) == jax.tree.structure(policy)
    for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(policy)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert 'attention' not in network_spec(_save_plain(tmp_path, plain, width, blocks))


def _save_plain(tmp_path, params, width, blocks):
    from polytopia_bot.rl.runner import Config, save_checkpoint
    out = tmp_path / 'plain.npz'
    save_checkpoint(out, params, Config(architecture='normalized-v2', size=SIZE, width=width, blocks=blocks, obs_version=3),
                    stage='test')
    return out


def test_attention_needs_normalized_and_head_multiple():
    import jax
    from polytopia_bot.rl import network
    with pytest.raises(ValueError):
        network.init(jax.random.key(0), 8, 4, size=SIZE, width=64, blocks=1, architecture='legacy', attention=1)
    with pytest.raises(ValueError):
        network.init(jax.random.key(0), 8, 4, size=SIZE, width=48, blocks=1, architecture='normalized-v2', attention=1)


def test_lookout_matches_hero_parameter_count():
    """lookout (2 conv blocks + 3 attention blocks at width 128) sits within 15% of hero's 4x128 parameter count,
    so the comparison is at matched size."""
    import jax
    from polytopia_bot.rl.distributed import _template_params
    count = lambda p: int(sum(a.size for a in jax.tree.leaves(p)))
    hero = count(_template_params(14, blocks=4, width=128, obs_version=3))
    lookout = count(_template_params(14, blocks=2, width=128, obs_version=3, attention=3))
    print(json.dumps(dict(hero=hero, lookout=lookout)))
    assert abs(lookout - hero) / hero < .15


def test_warmed_learning_rate_ramps_then_holds():
    from polytopia_bot.rl.distributed import warmed_learning_rate
    assert warmed_learning_rate(3e-4, 0, 0) == 3e-4
    assert warmed_learning_rate(3e-4, 0, 200) == pytest.approx(3e-4 / 200)
    assert warmed_learning_rate(3e-4, 99, 200) == pytest.approx(3e-4 / 2)
    assert warmed_learning_rate(3e-4, 199, 200) == 3e-4 and warmed_learning_rate(3e-4, 5000, 200) == 3e-4


def test_attention_remat_matches_plain_values_and_gradients():
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import _template_params
    own, _ = _features(seed=11)
    params = _template_params(SIZE, blocks=1, width=64, obs_version=3, attention=2)
    tiles = network.trunk(params, own[0])[0]   # the conv part, then both attention paths on the same input
    conv = dict(params)
    for key in ('attention', 'pos', 'attention_norm'):
        conv.pop(key)
    base = network.trunk(conv, own[0])[0]
    plain, remat = network.attend(params, base, remat=False), network.attend(params, base, remat=True)
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(remat))
    np.testing.assert_array_equal(np.asarray(tiles), np.asarray(remat))
    loss = lambda p, r: jnp.sum(network.attend(p, base, remat=r) ** 2)
    g_plain, g_remat = jax.grad(loss)(params, False), jax.grad(loss)(params, True)
    for a, b in zip(jax.tree.leaves(g_plain['attention']), jax.tree.leaves(g_remat['attention'])):
        # the recompute sums the backward terms in another order: fp32 noise on heavily cancelling sums
        scale = float(np.abs(np.asarray(a)).max())
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-2 * scale)
