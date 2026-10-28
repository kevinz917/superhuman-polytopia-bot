"""18x18 boards: starts, round limit, grown network, checkpoints (and the rules gate), arena stats."""
import json

import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from polytopia_bot.maps.stream import MapSource
from polytopia_bot.ruleset import LEGACY_DIGESTS, RULESET_ID, compatible, round_limit
from polytopia_bot.rl import network
from polytopia_bot.rl.cpu_arena import summarize, tech_summary
from polytopia_bot.rl.distributed import _template_params, checkpoint_template, network_spec


_ENGINES = {}


def _position(seed, size, kind='classic', steps=0):
    """A position after `steps` random legal actions on a `kind` map (engine-rs, one game)."""
    from polytopia_bot.rl.actions import Choice, shapes
    from polytopia_bot.rl.rust_engine import RustGames
    if size not in _ENGINES:
        _ENGINES[size] = RustGames(size, 1, map_batch=True)
    games = _ENGINES[size]
    source = MapSource(kind, size)
    states, prepared = games.reset(source.batch([seed])[0])
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        mats = [np.asarray(m[0]).reshape(shp) for m, shp in zip(prepared.masks, shapes(size))]
        legal = [f for f in range(16) if mats[f].any()]   # no resign
        f = int(rng.choice(legal))
        src = int(rng.choice(np.flatnonzero(mats[f].any(1))))
        choice = Choice(*(jnp.asarray([v], jnp.int32) for v in (f, src, int(rng.choice(np.flatnonzero(mats[f][src]))))))
        tick, status = games.tick(states, choice, source.batch([seed])[0])
        assert not np.any(np.asarray(status))
        states, prepared = tick.states, tick.prepared
    return jax.tree.map(lambda a: jnp.asarray(a[0]), (prepared.features, prepared.masks))


def test_round_limits_and_large_reset():
    from polytopia_bot.rl.rust_engine import RustGames
    assert (round_limit(11), round_limit(14), round_limit(18)) == (100, 100, 150)
    games = RustGames(18, 1)
    states, prepared = games.reset(np.array([370001], np.int32))
    s = jax.tree.map(lambda a: np.asarray(a)[0], states)
    assert s.tiles.terrain.shape == (324,) and int(s.reference_max_turns) == 150
    assert (np.asarray(s.tiles.structure) == 2).sum() == 9   # ruins, reference Large count
    caps = np.flatnonzero(np.asarray(s.tiles.capital) > 0)
    assert len(caps) == 2
    x, y = caps % 18, caps // 18
    assert max(abs(x[0] - x[1]), abs(y[0] - y[1])) >= 5
    m = prepared.masks
    assert bool(np.asarray(m.end)[0]) and np.asarray(m.move)[0].any()


def test_checkpoint_rules_gate():
    """New checkpoints carry the ruleset; the research-rules digests of older ones still load (as references)."""
    assert compatible(dict(ruleset=RULESET_ID, simulator_sha256='0' * 64))
    assert all(compatible(dict(ruleset='bardur-1v1-v2', simulator_sha256=d)) for d in LEGACY_DIGESTS)
    assert not compatible(dict(ruleset='bardur-1v1-v2', simulator_sha256='0' * 64))


def _random_params(size=14, extra_blocks=0, seed=0):
    template = _template_params(size, extra_blocks=extra_blocks)
    keys = jax.random.split(jax.random.key(seed), len(jax.tree.leaves(template)))
    leaves = [jax.random.normal(k, a.shape) * .2 for k, a in zip(keys, jax.tree.leaves(template))]
    return jax.tree.unflatten(jax.tree.structure(template), leaves)


def _old_target_logits(params, tiles, g, family, source):
    """target_logits before the size-independent parameter keys (rows read straight from the embedding)."""
    pre = network.dense(jnp.concatenate((tiles[source], g)), params["query"]) + params["family_embedding"][family]
    query = network.layer_norm(pre)
    spatial = jnp.pad(tiles, ((0, params["parameter_embedding"].shape[1] - tiles.shape[0]), (0, 0)))
    keys = jnp.where(family < 2, spatial, params["parameter_embedding"][family])
    return jnp.matmul(keys, query, precision=jax.lax.Precision.HIGHEST) / jnp.sqrt(tiles.shape[-1])


def test_new_target_keys_keep_every_legal_logprob_at_native_size():
    params = _random_params(14)
    x, m = _position(300123, 14, steps=40)
    tiles, g, _ = network.trunk(params, x)
    from polytopia_bot.rl.actions import family_mask, source_mask, target_mask
    fm = np.asarray(family_mask(m))
    checked = 0
    for f in np.flatnonzero(fm):
        sm = source_mask(m, int(f), 14)
        s = int(np.flatnonzero(np.asarray(sm))[0])
        tm = target_mask(m, int(f), s, 14)
        new = network.log_distribution(network.target_logits(params, tiles, g, f, s), tm)
        old = network.log_distribution(_old_target_logits(params, tiles, g, f, s), tm)
        np.testing.assert_array_equal(np.asarray(new)[np.asarray(tm)], np.asarray(old)[np.asarray(tm)])
        checked += 1
    assert checked >= 3


@pytest.mark.parametrize('size,kind', [(14, 'classic'), (18, 'classic'), (18, 'archipelago')])
def test_grown_network_is_exactly_the_parent(size, kind):
    parent = _random_params(14, seed=1)
    grown = network.grow(parent, jax.random.key(5), size=18, extra_blocks=2)
    assert grown['parameter_embedding'].shape == (17, 324, 64) and len(grown['extra_blocks']) == 2
    assert jax.tree.structure(grown) == jax.tree.structure(_template_params(18, extra_blocks=2))
    x, m = _position(370000 + size, size, kind, steps=30)
    for a, b in zip(network.trunk(parent, x), network.trunk(grown, x)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for det in (True, False):
        out = [network.sample(p, x, m, jax.random.key(7), size=size, deterministic=det) for p in (parent, grown)]
        for a, b in zip(jax.tree.leaves(out[0]), jax.tree.leaves(out[1])):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # The new blocks train: their gradient is nonzero once the zero kernel moves.
    moved = dict(grown, extra_blocks=tuple((a, b + .01) for a, b in grown['extra_blocks']))
    assert not np.array_equal(np.asarray(network.trunk(moved, x)[2]), np.asarray(network.trunk(grown, x)[2]))


def test_grown_checkpoint_round_trip(tmp_path):
    from polytopia_bot.rl.runner import Config, load_checkpoint, save_checkpoint
    grown = network.grow(_random_params(14), jax.random.key(0), size=18, extra_blocks=2)
    path = tmp_path / 'grown.npz'
    save_checkpoint(path, grown, Config(architecture='normalized-v2', size=18, extra_blocks=2), stage='grown')
    assert network_spec(path) == dict(size=18, blocks=2, extra_blocks=2, width=64, architecture='normalized-v2')
    loaded = load_checkpoint(path, checkpoint_template(path))
    for a, b in zip(jax.tree.leaves(grown), jax.tree.leaves(loaded)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['ruleset'] == RULESET_ID


def test_engine_uses_the_size_round_limit():
    from polytopia_bot.rl.rust_engine import RustGames
    games = RustGames(18, 2, map_batch=True)
    states, _ = games.reset(MapSource('proposed-v1', 18).batch([370010, 370011])[0])
    np.testing.assert_array_equal(np.asarray(states.reference_max_turns), [150, 150])
    capped = RustGames(18, 2, map_batch=False, max_turns=120)
    states, _ = capped.reset(np.array([370012, 370013]))
    np.testing.assert_array_equal(np.asarray(states.reference_max_turns), [120, 120])


def test_tech_summary_counts_by_turn_and_tier():
    rows = [dict(seed=1, seat=1, reward=1., truncated=False, steps=10, turn=25, cand_units={2: 1}, opp_units={},
                 cand_research_early=0, first_non_warrior=None, opp_first_non_warrior=None,
                 cand_tech=[[3, 1], [12, 4], [22, 3]], opp_tech=[[5, 10]]),
            dict(seed=2, seat=2, reward=-1., truncated=True, steps=10, turn=150, cand_units={}, opp_units={},
                 cand_research_early=0, first_non_warrior=None, opp_first_non_warrior=None,
                 cand_tech=[], opp_tech=[])]
    t = tech_summary(rows, ('cand_tech',))
    assert t['final_mean'] == 3.5 and t['tier3_mean'] == .5 and t['share_with_tier3'] == .5
    assert t['first_tier3_turn_median'] == 22
    assert t['by_turn']['10'] == dict(games=2, mean_techs=2.5)
    assert t['by_turn']['30'] == dict(games=1, mean_techs=2.)
    s = summarize(rows)
    assert s['candidate_tech']['samples'] == 2 and s['by_map']['classic']['both_tech']['samples'] == 4


def test_critic_warmup_changes_only_critic_parameters(tmp_path):
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    from polytopia_bot.rl.runner import Config, save_checkpoint
    parent = tmp_path / 'parent.npz'
    save_checkpoint(parent, network.grow(_random_params(14), jax.random.key(0), size=18, extra_blocks=1),
                    Config(architecture='normalized-v2', size=18, extra_blocks=1), stage='grown')
    learner = Learner(tmp_path / 'run', LearnerConfig(size=18, batch=64, minibatch=32, critic_warmup=5), str(parent),
                      log=lambda s: None)
    before = jax.device_get(learner.params)
    batch = jax.device_put(learner.synthetic_batch(64), learner.data_sharding)
    for weight in (0., 1.):
        (params, _), _, _ = learner.train_epoch((learner.params, learner.critic), learner.opt, learner.reference,
                                                learner.anchor, batch, jax.random.key(1), np.float32(weight), np.float32(.005))
        params = jax.device_get(params)
        same = {k: all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before[k]), jax.tree.leaves(params[k])))
                for k in before}
        assert not same['critic_hidden'] and not same['value']
        if weight == 0.:   # warm-up: everything the policy reads is bitwise unchanged
            assert all(same[k] for k in before if k not in ('critic_hidden', 'value'))
        else:              # (the synthetic batch only picks family 0, so parameter_embedding gets no gradient)
            assert not same['stem'] and not same['family'] and not same['blocks']


def test_kl_anchor_is_zero_at_the_parent_and_trains(tmp_path):
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    from polytopia_bot.rl.runner import Config, save_checkpoint
    parent = tmp_path / 'parent.npz'
    save_checkpoint(parent, network.grow(_random_params(14), jax.random.key(0), size=18, extra_blocks=1),
                    Config(architecture='normalized-v2', size=18, extra_blocks=1), stage='grown')
    learner = Learner(tmp_path / 'run', LearnerConfig(size=18, batch=64, minibatch=32, anchor_coef=.1,
                                                      turn_gamma=.97, turn_lambda=.92), str(parent), log=lambda s: None)
    batch = jax.device_put(learner.synthetic_batch(64), learner.data_sharding)
    _, _, metrics = learner.train_epoch((learner.params, learner.critic), learner.opt, learner.reference, learner.anchor,
                                        batch, jax.random.key(1), np.float32(1.), np.float32(.005))
    assert 'anchor_kl' in metrics and np.isfinite(float(metrics['anchor_kl']))
    assert float(metrics['anchor_kl']) < 1e-3   # first minibatch starts at the anchor; one epoch moves it little


def _parent(tmp_path, size=14):
    from polytopia_bot.rl.runner import Config, save_checkpoint
    parent = tmp_path / 'parent.npz'
    save_checkpoint(parent, _random_params(size), Config(architecture='normalized-v2', size=size), stage='p')
    return str(parent)


def test_omniscient_critic_trains_separately_from_the_policy(tmp_path):
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    learner = Learner(tmp_path / 'run', LearnerConfig(size=14, batch=64, minibatch=32, omniscient_critic=True),
                      _parent(tmp_path), log=lambda s: None)
    assert learner.critic is not None and learner.critic['stem'].shape[2] == 2 * learner.params['stem'].shape[2]
    before = jax.device_get((learner.params, learner.critic))
    batch = jax.device_put(learner.synthetic_batch(64), learner.data_sharding)
    assert 'opp_view' in learner.synthetic_batch(64)
    (params, critic), _, metrics = learner.train_epoch((learner.params, learner.critic), learner.opt, None, None, batch,
                                                       jax.random.key(1), np.float32(1.), np.float32(.005))
    params, critic = jax.device_get((params, critic))
    assert not np.array_equal(critic['stem'], before[1]['stem'])          # critic learned
    assert not np.array_equal(params['stem'], before[0]['stem'])          # policy learned
    for k in ('value', 'critic_hidden'):                                   # the policy's own value head is idle
        assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before[0][k]), jax.tree.leaves(params[k])))
    assert np.isfinite(float(metrics['value_loss']))
    learner.version = 3
    learner.save('latest')
    assert (tmp_path / 'run' / 'latest-critic.npz').exists()


def test_advantage_filter_keeps_the_largest_and_ema_is_deployed(tmp_path):
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    from polytopia_bot.rl.runner import load_checkpoint
    # Actor-side advantages (learner_values=False): the rows carry 'advantage' and the inbox message is
    # (batch, segments) with no segment table (Stage A1 changed the wire format; learner-side GAE is tested in
    # test_distributed_rl.py).
    cfg = LearnerConfig(size=14, batch=128, minibatch=32, advantage_keep=.25, ema_decay=.5, learner_values=False)
    learner = Learner(tmp_path / 'run', cfg, _parent(tmp_path), log=lambda s: None)
    rows = learner.synthetic_batch(128)
    rows['version'] = np.zeros(128, np.int32)
    rows['advantage'] = np.linspace(-1., 1., 128).astype(np.float32)
    learner.inbox.put((rows, None))
    learner.deadline = float('inf')
    data, info = learner._assemble()   # Stage A1: {host, views, keep}; _device_batch builds the training batch
    host = data['host']
    assert info['kept'] == 32 and len(host['logp']) == 32
    assert np.abs(host['advantage']).min() >= np.sort(np.abs(rows['advantage'] - rows['advantage'].mean()))[-32] / rows['advantage'].std() - 1e-5
    batch = learner._device_batch(data)
    (learner.params, learner.critic), learner.opt, _ = learner.train_epoch(
        (learner.params, learner.critic), learner.opt, None, None, batch, jax.random.key(1), np.float32(1.), np.float32(.005))
    d = cfg.ema_decay
    learner.ema = jax.tree.map(lambda e, p: d * e + (1. - d) * p, learner.ema, learner.params)
    learner.save('latest')
    deployed = load_checkpoint(tmp_path / 'run' / 'latest.npz', checkpoint_template(tmp_path / 'run' / 'latest.npz'))
    raw = load_checkpoint(tmp_path / 'run' / 'latest-raw.npz', checkpoint_template(tmp_path / 'run' / 'latest-raw.npz'))
    assert not np.array_equal(np.asarray(deployed['stem']), np.asarray(raw['stem']))
    np.testing.assert_allclose(np.asarray(deployed['stem']), np.asarray(jax.device_get(learner.ema)['stem']))


def test_potentials_are_zero_sum_and_shaping_telescopes():
    from polytopia_bot.rl.distributed import gae, potentials
    from polytopia_bot.rl.rust_engine import RustGames
    games = RustGames(14, 2, map_batch=True)
    states, _ = games.reset(MapSource('classic', 14).batch([300001, 300002])[0])
    p1 = potentials(states, np.array([1, 1]), 1.)
    p2 = potentials(states, np.array([2, 2]), 1.)
    np.testing.assert_allclose(p1, -p2, atol=1e-6)
    np.testing.assert_allclose(p1, 0., atol=1e-6)          # symmetric starts: level
    s = states._replace(players=states.players._replace(stars=jnp.array([[20, 5], [5, 20]], jnp.int32)))
    q = potentials(s, np.array([1, 1]), .5)
    assert q[0] > 0 > q[1] and abs(q[0]) <= .5 * .3 * 4
    # gamma = lambda = 1, terminal: shaping telescopes to outcome - phi_t
    values = np.zeros(5, np.float32)
    phi = np.array([.1, .3, .2, .4, .05], np.float32)
    _, ret = gae(values, 0., 1., True, 1., 1., phi=phi)
    np.testing.assert_allclose(ret, 1. - phi, atol=1e-6)
    # non-terminal cut: the next state's potential closes the sum
    _, ret = gae(values, .7, 0., False, 1., 1., phi=phi, phi_next=.25)
    np.testing.assert_allclose(ret, .7 + .25 - phi, atol=1e-6)


def test_learner_resumes_from_latest_checkpoint(tmp_path):
    """A replaced learner container continues from latest.npz: weights, Adam state, critic, update count, log,
    the self-play pool from the snapshots and the clock (training_seconds bounds the whole run)."""
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    from polytopia_bot.rl.runner import write_json
    cfg = LearnerConfig(size=14, batch=64, minibatch=32, omniscient_critic=True, historical_every=1, historical_keep=2,
                        snapshot_every=1, checkpoint_every=1, historical_probability=.2)
    parent = _parent(tmp_path)
    first = Learner(tmp_path / 'run', cfg, parent, log=lambda s: None)
    batch = jax.device_put(first.synthetic_batch(64), first.data_sharding)
    for version in (1, 2, 3):
        (first.params, first.critic), first.opt, _ = first.train_epoch(
            (first.params, first.critic), first.opt, None, None, batch, jax.random.key(version), np.float32(1.), np.float32(.005))
        first.version = version
        first.save(f'snapshots/update-{version}')
    first.save('latest')
    write_json(tmp_path / 'run' / 'training.json', [dict(update=u, elapsed=100. * u) for u in (1, 2, 3)])
    logs = []
    again = Learner(tmp_path / 'run', cfg, parent, log=logs.append)
    assert again.version == 3 and again.resume_elapsed == 300. and len(again.resume_history) == 3
    assert any('learner_resumed' in line for line in logs)
    for a, b in zip(jax.tree.leaves(jax.device_get((first.params, first.critic, first.opt))),
                    jax.tree.leaves(jax.device_get((again.params, again.critic, again.opt))), strict=True):
        assert np.array_equal(a, b)
    assert [n for n, _ in again.historical] == ['hist-u2', 'hist-u3']   # last historical_keep snapshots
    # Stage A1: with learner_values (default) actors run no critic, so none is published; without it the critic is.
    assert again.published[0] == 3 and (again.published[2] is not None) == (not cfg.learner_values)
    with_actor_critic = Learner(tmp_path / 'run', dataclasses.replace(cfg, learner_values=False), parent, log=lambda s: None)
    assert with_actor_critic.published[0] == 3 and with_actor_critic.published[2] is not None
    # not a restart: an untouched folder resumes nothing
    fresh = Learner(tmp_path / 'other', cfg, parent, log=lambda s: None)
    assert fresh.version == 0 and fresh.resume_history == []
