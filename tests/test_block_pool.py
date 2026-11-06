"""Global pooling bias inside the trunk blocks (network.init block_pool, 2026-11-01): KataGo's pooling layer on the
existing residual blocks instead of extra blocks, so every tile's source and target logits see board-wide context at no
extra convolution cost. A network gains it with zero weights (network.add_block_pool) and must compute exactly what it
computed before, with gradients reaching the new layers from the first step; the checkpoint spec records it."""
import json

import numpy as np
import pytest

SIZE = 11


def _features(seed=3):
    import jax.numpy as jnp
    from polytopia_bot.rl.cpu_engine import unpack_view
    from polytopia_bot.rl.observation import encode_view
    from polytopia_bot.rl.rust_engine import RustGames, split_masks
    from test_obs_history import _random_choice   # pytest puts tests/ on sys.path
    g = RustGames(SIZE, 2, max_turns=30)
    g.batch.reset(np.array([seed, seed + 1], np.int32))
    rng = np.random.default_rng(seed)
    for _ in range(200):
        g.batch.step(*_random_choice(split_masks(g.batch.legal_masks(), SIZE), rng, size=SIZE))
    own = [encode_view(unpack_view(jnp.asarray(r), g.layout), SIZE) for r in np.asarray(g.batch.observe())]
    opp = [encode_view(unpack_view(jnp.asarray(r), g.layout), SIZE) for r in np.asarray(g.batch.observe_opponent())]
    return own, opp


def _perturbed(params, key, scale=.05):
    import jax
    return jax.tree.map(lambda a: a + scale * jax.random.normal(key, a.shape), params)


def test_block_pool_starts_as_the_identity_and_is_trainable():
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import _critic_template, _template_params, omniscient_features
    own, opp = _features()
    key = jax.random.key(7)
    policy = _perturbed(_template_params(SIZE, blocks=2, width=8, obs_version=3), key)
    critic = _perturbed(_critic_template(SIZE, blocks=2, width=8, obs_version=3), key)
    pooled, pooled_critic = network.add_block_pool(policy), network.add_block_pool(critic)
    assert all(len(b) == 3 for b in pooled['blocks']) and all(len(b) == 2 for b in policy['blocks'])
    assert pooled['blocks'][0][2]['w'].shape == (16, 8) and not float(jnp.abs(pooled['blocks'][0][2]['w']).sum())
    template = _template_params(SIZE, blocks=2, width=8, obs_version=3, block_pool=True)
    assert jax.tree.structure(template) == jax.tree.structure(pooled)
    for x, o in zip(own, opp):
        for a, b in zip(network.trunk(policy, x), network.trunk(pooled, x)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        both = omniscient_features(x, o)
        np.testing.assert_array_equal(np.asarray(network.trunk(critic, both)[2]), np.asarray(network.trunk(pooled_critic, both)[2]))
    # gradients reach the pooling layers at once (the pooled features are nonzero), and a trained layer changes tiles
    grads = jax.grad(lambda p: jnp.sum(network.trunk(p, own[0])[0]) + network.trunk(p, own[0])[2])(pooled)
    for block in grads['blocks']:
        assert float(jnp.abs(block[2]['w']).sum()) > 0 and float(jnp.abs(block[2]['b']).sum()) > 0
    with pytest.raises(ValueError):
        network.add_block_pool(pooled)
    moved = jax.tree.map(lambda a: a + .1, pooled)
    assert float(jnp.abs(network.trunk(moved, own[0])[0] - network.trunk(jax.tree.map(lambda a: a + .1, policy), own[0])[0]).sum()) > 0


def test_block_pool_spec_and_checkpoint_round_trip(tmp_path):
    from polytopia_bot.rl.distributed import _template_params, checkpoint_template, network_spec
    from polytopia_bot.rl.runner import Config, load_checkpoint, save_checkpoint
    cfg = Config(size=SIZE, width=8, blocks=2, architecture='normalized-v2', obs_version=3, block_pool=True)
    cfg.validate()
    params = _template_params(SIZE, blocks=2, width=8, obs_version=3, block_pool=True)
    path = tmp_path / 'pool.npz'
    save_checkpoint(path, params, cfg, stage='test')
    spec = network_spec(path)
    assert spec['block_pool'] is True and spec['obs_version'] == 3
    loaded = load_checkpoint(path, checkpoint_template(path))
    assert len(loaded['blocks'][0]) == 3
    # an older checkpoint's spec has no block_pool and its template has two-entry blocks
    plain = tmp_path / 'plain.npz'
    save_checkpoint(plain, _template_params(SIZE, blocks=2, width=8, obs_version=3),
                    Config(size=SIZE, width=8, blocks=2, architecture='normalized-v2', obs_version=3), stage='test')
    assert 'block_pool' not in network_spec(plain) and len(checkpoint_template(plain)['blocks'][0]) == 2
    with pytest.raises(Exception):
        load_checkpoint(plain, checkpoint_template(path))


def test_widen_tool_combines_history_and_pooling(tmp_path):
    """widen_obs.py end to end on a tiny fake run: v2 -> v3 inputs plus the pooling layers, Adam state widened with
    seeded second moments, the parent rewritten, and every widened network agrees with its original."""
    import subprocess
    import sys
    from pathlib import Path
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import network, ppo
    from polytopia_bot.rl.distributed import _critic_template, _template_params, checkpoint_template, network_spec
    from polytopia_bot.rl.runner import Config, load_checkpoint, save_checkpoint
    cfg2 = Config(size=SIZE, width=8, blocks=2, architecture='normalized-v2', obs_version=2)
    key = jax.random.key(9)
    policy = _perturbed(_template_params(SIZE, blocks=2, width=8, obs_version=2), key)
    critic = _perturbed(_critic_template(SIZE, blocks=2, width=8, obs_version=2), key)
    run = tmp_path / 'run'
    (run / 'snapshots').mkdir(parents=True)
    step, m, v = ppo.adam_init((policy, critic))
    opt = (jnp.int32(500), jax.tree.map(lambda a: a + .01, m), jax.tree.map(lambda a: a + 1e-6, v))
    save_checkpoint(run / 'latest-critic.npz', critic, cfg2, stage='latest')
    save_checkpoint(run / 'latest.npz', policy, cfg2, stage='latest', optimizer=opt)
    save_checkpoint(run / 'snapshots' / 'update-250.npz', policy, cfg2, stage='snapshot')
    parent_in = tmp_path / 'parent.npz'
    save_checkpoint(parent_in, _template_params(SIZE, blocks=2, width=8, obs_version=2), cfg2, stage='init')
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo / 'experiments/curio/widen_obs.py'), '--run-dir', str(run), '--backup', str(tmp_path / 'bk'),
           '--parent-in', str(parent_in), '--parent-out', str(tmp_path / 'parent3.npz'), '--block-pool', '--check', '4']
    import os
    env = dict(os.environ, PYTHONPATH=str(repo / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=repo, env=env)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert json.loads(res.stdout.strip().splitlines()[-1])['status'] == 'ok'
    for name in ('latest.npz', 'latest-critic.npz', 'snapshots/update-250.npz'):
        spec = network_spec(run / name)
        assert spec['obs_version'] == 3 and spec['block_pool'] is True, (name, spec)
    assert network_spec(tmp_path / 'parent3.npz')['block_pool'] is True
    wide = load_checkpoint(run / 'latest.npz', checkpoint_template(run / 'latest.npz'))
    own, opp = _features(seed=5)
    for x in own:
        for a, b in zip(network.trunk(policy, x), network.trunk(wide, x)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5, rtol=0)
    # Adam state: the (policy, critic) layout of the widened networks; old entries kept, new second moments seeded > 0
    wide_critic = load_checkpoint(run / 'latest-critic.npz', _critic_template(**network_spec(run / 'latest-critic.npz')))
    leaves, structure = jax.tree.flatten(ppo.adam_init((wide, wide_critic)))
    with np.load(run / 'latest.npz', allow_pickle=False) as archive:
        values = [archive[k] for k in sorted((k for k in archive.files if k.startswith('o')), key=lambda k: int(k[1:]))]
    assert [a.shape for a in leaves] == [b.shape for b in values]
    step3, m3, v3 = structure.unflatten([jnp.asarray(a) for a in values])
    assert int(step3) == 500
    for block in v3[0]['blocks']:
        assert float(block[2]['w'].min()) > 0 and float(block[2]['b'].min()) > 0
    for block in m3[0]['blocks']:
        assert not float(jnp.abs(block[2]['w']).sum())
    assert float(v3[0]['stem'][..., -5:, :].min()) > 0 and float(jnp.abs(m3[0]['stem'][..., -5:, :]).sum()) == 0
    np.testing.assert_allclose(np.asarray(v3[0]['blocks'][0][0]), np.asarray(opt[2][0]['blocks'][0][0]))
    assert (tmp_path / 'bk' / 'latest.npz').exists() and (tmp_path / 'bk' / 'snapshots' / 'update-250.npz').exists()


# ---- layout 2 (farsight v2, 2026-11-01): gated, normalized, in the second block of every pair

def _with_gate(params, value):
    import jax.numpy as jnp
    from polytopia_bot.rl import network
    blocks = list(params['blocks'])
    for i in network.block_pool_indices(len(blocks), 2):
        a, b, layer = blocks[i]
        blocks[i] = (a, b, dict(layer, gate=jnp.float32(value)))
    return dict(params, blocks=tuple(blocks))


def test_block_pool_layout_2_is_the_identity_at_a_zero_gate_and_trains_through_the_gate():
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import _critic_template, _template_params, omniscient_features
    own, opp = _features()
    key = jax.random.key(11)
    policy = _perturbed(_template_params(SIZE, blocks=4, width=8, obs_version=3), key)
    critic = _perturbed(_critic_template(SIZE, blocks=4, width=8, obs_version=3), key)
    pooled = network.add_block_pool(policy, layout=2, key=key)
    pooled_critic = network.add_block_pool(critic, layout=2, key=key)
    assert network.block_pool_indices(4, 2) == (1, 3) and network.block_pool_indices(2, 2) == (1,)
    assert network.block_pool_indices(1, 2) == (0,) and network.block_pool_indices(4, 1) == (0, 1, 2, 3)
    assert [len(b) for b in pooled['blocks']] == [2, 3, 2, 3]
    layer = pooled['blocks'][1][2]
    assert layer['w'].shape == (16, 8) and layer['gate'].shape == () and float(layer['gate']) == 0
    assert float(jnp.abs(layer['w']).sum()) > 0   # random dense weights under the zero gate
    template = _template_params(SIZE, blocks=4, width=8, obs_version=3, block_pool=2)
    assert jax.tree.structure(template) == jax.tree.structure(pooled)
    assert template['blocks'][1][2]['w'].shape == (16, 8)   # min(BLOCK_POOL_CHANNELS, width) channels pooled
    for x, o in zip(own, opp):
        for a, b in zip(network.trunk(policy, x), network.trunk(pooled, x)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        both = omniscient_features(x, o)
        np.testing.assert_array_equal(np.asarray(network.trunk(critic, both)[2]),
                                      np.asarray(network.trunk(pooled_critic, both)[2]))
    # collecting the statistics changes nothing; the bias is zero at a zero gate
    stats = []
    tiles = network.trunk(pooled, own[0], pool_stats=stats)[0]
    np.testing.assert_array_equal(np.asarray(tiles), np.asarray(network.trunk(policy, own[0])[0]))
    assert len(stats) == 2
    assert all(float(s['bias_rms']) == 0 and float(s['gate']) == 0 and float(s['conv_rms']) > 0 for s in stats)
    # step 1: the gate gets a gradient, the dense layer none (it is multiplied by the zero gate); once the gate has
    # moved the dense layer trains and the tiles change
    loss = lambda p: jnp.sum(network.trunk(p, own[0])[0]) + network.trunk(p, own[0])[2]
    grads = jax.grad(loss)(pooled)
    for i in (1, 3):
        g = grads['blocks'][i][2]
        assert float(jnp.abs(g['gate'])) > 0
        assert float(jnp.abs(g['w']).sum()) == 0 and float(jnp.abs(g['b']).sum()) == 0
    for i in (0, 2):
        assert len(grads['blocks'][i]) == 2
    opened = _with_gate(pooled, .1)
    grads = jax.grad(loss)(opened)
    for i in (1, 3):
        g = grads['blocks'][i][2]
        assert float(jnp.abs(g['w']).sum()) > 0 and float(jnp.abs(g['b']).sum()) > 0 and float(jnp.abs(g['gate'])) > 0
    assert float(jnp.abs(network.trunk(opened, own[0])[0] - tiles).sum()) > 0
    with pytest.raises(ValueError):
        network.add_block_pool(pooled, layout=2)
    with pytest.raises(ValueError):
        network.add_block_pool(policy, layout=3)
    with pytest.raises(ValueError):
        network.init(jax.random.key(0), 8, 4, size=SIZE, width=8, blocks=2, architecture='legacy', block_pool=2)
    with pytest.raises(ValueError):
        network.init(jax.random.key(0), 8, 4, size=SIZE, width=8, blocks=2, architecture='normalized-v2', block_pool=3)


def test_block_pool_layout_2_bias_is_normalized_and_linear_in_the_gate():
    """With a unit gate over the random dense layer the bias RMS stays within a small factor of the normalized tiles
    it joins, on real positions (farsight's layout 1 reached 3-5x the convolution), and it is linear in the gate."""
    import jax
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import _template_params
    own, _ = _features(seed=4)
    params = _perturbed(_template_params(SIZE, blocks=4, width=32, obs_version=3, block_pool=2), jax.random.key(3))
    one, half = _with_gate(params, 1.), _with_gate(params, .5)
    for x in own:
        s1, s2 = [], []
        network.trunk(one, x, pool_stats=s1)
        network.trunk(half, x, pool_stats=s2)
        assert len(s1) == len(s2) == 2
        for a in s1:
            ratio = float(a['bias_rms']) / float(a['conv_rms'])
            assert .05 < ratio < 3., ratio
        # linear in the gate at the first pooling layer (the second one's input already moved with the first gate)
        np.testing.assert_allclose(float(s2[0]['bias_rms']), .5 * float(s1[0]['bias_rms']), rtol=1e-5)
        assert float(s2[1]['bias_rms']) < float(s1[1]['bias_rms'])


def test_block_pool_layouts_round_trip_and_load_their_own_checkpoints(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    import jax
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import _template_params, checkpoint_template, network_spec
    from polytopia_bot.rl.runner import Config, load_checkpoint, save_checkpoint
    own, _ = _features(seed=6)
    cfg2 = Config(size=SIZE, width=8, blocks=4, architecture='normalized-v2', obs_version=3, block_pool=2)
    cfg2.validate()
    with pytest.raises(ValueError):
        Config(size=SIZE, width=8, blocks=4, architecture='normalized-v2', obs_version=3, block_pool=3).validate()
    v2 = _perturbed(_template_params(SIZE, blocks=4, width=8, obs_version=3, block_pool=2), jax.random.key(5))
    v1 = _perturbed(_template_params(SIZE, blocks=4, width=8, obs_version=3, block_pool=True), jax.random.key(5))
    p2, p1 = tmp_path / 'v2.npz', tmp_path / 'v1.npz'
    save_checkpoint(p2, v2, cfg2, stage='test')
    save_checkpoint(p1, v1, Config(size=SIZE, width=8, blocks=4, architecture='normalized-v2', obs_version=3,
                                   block_pool=True), stage='test')
    assert network_spec(p2)['block_pool'] == 2 and network_spec(p1)['block_pool'] is True
    assert json.loads(p2.with_suffix('.json').read_text())['config']['block_pool'] == 2
    for path, params in ((p2, v2), (p1, v1)):
        loaded = load_checkpoint(path, checkpoint_template(path))
        assert jax.tree.structure(loaded) == jax.tree.structure(params)
        for a, b in zip(network.trunk(params, own[0]), network.trunk(loaded, own[0])):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert [len(b) for b in checkpoint_template(p2)['blocks']] == [2, 3, 2, 3]
    assert [len(b) for b in checkpoint_template(p1)['blocks']] == [3, 3, 3, 3]
    with pytest.raises(Exception):   # the layouts do not load into each other's templates
        load_checkpoint(p1, checkpoint_template(p2))
    with pytest.raises(Exception):
        load_checkpoint(p2, checkpoint_template(p1))
    # layout 1 reports its statistics too (the farsight measurement) and keeps its forward pass
    stats = []
    network.trunk(v1, own[0], pool_stats=stats)
    assert len(stats) == 4 and all(s['gate'] is None and float(s['bias_rms']) > 0 for s in stats)
    # random_parent.py --block-pool builds a layout-2 parent
    repo = Path(__file__).resolve().parents[1]
    import os
    env = dict(os.environ, PYTHONPATH=str(repo / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
    res = subprocess.run([sys.executable, str(repo / 'experiments/curio/random_parent.py'), '--reference', str(p2), '--out',
                          str(tmp_path / 'parent.npz'), '--block-pool', '--seed', '3'], capture_output=True, text=True,
                         cwd=repo, env=env)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    row = json.loads(res.stdout.strip().splitlines()[-1])
    assert row['spec']['block_pool'] == 2 and row['spec']['obs_version'] == 3
    parent = load_checkpoint(tmp_path / 'parent.npz', checkpoint_template(tmp_path / 'parent.npz'))
    assert [len(b) for b in parent['blocks']] == [2, 3, 2, 3] and float(parent['blocks'][1][2]['gate']) == 0
