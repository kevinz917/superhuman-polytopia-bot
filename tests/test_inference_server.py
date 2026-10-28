"""The in-node inference server returns what the actor's local forward would: same values, legal choices,
path masks consistent with the choice, finite log-probabilities. Runs the server on the CPU platform in fp32.

A served RustGames (packed masks straight from Rust, no features) is stepped in lockstep with an unserved one, for
two seats on 14x14 and four seats on 18x18; its packed rows must equal numpy.packbits of the unserved mask leaves."""
import multiprocessing as mp

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from polytopia_bot.rl import network
from polytopia_bot.rl.actions import Choice, path_masks
from polytopia_bot.rl.distributed import _template_params
from polytopia_bot.rl.inference_server import InferenceClient, serve

pytest.importorskip('polytopia_engine_rs')


def _pack_leaves(masks):
    games = np.asarray(masks[0]).shape[0]
    return np.packbits(np.concatenate([np.asarray(m).reshape(games, -1) for m in masks], 1), axis=1)


@pytest.mark.parametrize('size,players', [(14, 2), (18, 4)])
def test_server_matches_local_forward(size, players):
    from polytopia_bot.rl.rust_engine import RustGames
    games = 4
    seeds = lambda t: np.arange(games, dtype=np.int32) + 200000 + 100 * t
    engine = RustGames(size, games, players=players)
    twin = RustGames(size, games, players=players, served=True)
    ctx = mp.get_context('spawn')
    server_end, client_end = ctx.Pipe()
    server = ctx.Process(target=serve, args=([server_end],),
                         kwargs=dict(platform='cpu', deadline_ms=1., trunk_dtype='float32', precision='HIGHEST'),
                         daemon=True)
    server.start()
    try:
        spec = dict(size=size, blocks=2, extra_blocks=0, width=64, architecture='normalized-v2', players=players)
        params = _template_params(**spec)
        leaves = [np.asarray(a) for a in jax.tree.leaves(params)]
        client = InferenceClient(client_end, size, games, players)
        client.publish('learner', 3, leaves, spec)
        client.publish('learner', 2, leaves, spec)       # older version: ignored, same id
        assert client.ids == {'learner': 0}
        states, prepared = engine.reset(seeds(0))
        served = twin.reset(seeds(0))[1]
        value = jax.jit(jax.vmap(lambda p, x: network.trunk(p, x)[2], in_axes=(None, 0)))
        logp_fn = jax.jit(jax.vmap(lambda p, x, c, m: network.evaluate(p, x, c, m)[0], in_axes=(None, 0, 0, 0)))
        for tick_no in range(6):
            local_value = np.asarray(value(params, prepared.features))
            assert served.features is None
            np.testing.assert_array_equal(served.view, prepared.view)
            np.testing.assert_array_equal(served.masks, _pack_leaves(prepared.masks))
            lanes = np.arange(games) if tick_no % 2 == 0 else np.array([0, 2])
            out = client.sample({'learner': lanes}, served.view, served.masks)
            choice, (fam, src, tgt), logp, values = out['learner']
            np.testing.assert_allclose(values, local_value[lanes], atol=2e-4, rtol=1e-4)
            assert np.all(np.isfinite(logp)) and np.all(logp <= 0)
            masks_np = jax.tree.map(lambda a: np.asarray(a), prepared.masks)
            paths = []
            for r, lane in enumerate(lanes):
                one = jax.tree.map(lambda a: jnp.asarray(a[lane]), masks_np)
                c = Choice(jnp.int32(choice[0][r]), jnp.int32(choice[1][r]), jnp.int32(choice[2][r]))
                path = path_masks(one, c, size)
                paths.append(path)
                # the choice is legal along its path and the packed path masks are exactly the local ones
                assert bool(path.family[c.family]) and bool(path.source[c.source]) and bool(path.target[c.target])
                np.testing.assert_array_equal(fam[r], np.packbits(np.asarray(path.family)))
                np.testing.assert_array_equal(src[r], np.packbits(np.asarray(path.source)))
                np.testing.assert_array_equal(tgt[r], np.packbits(np.asarray(path.target)))
            # the learner's fp32 logp of the served choice equals the served logp (same fp32 path here)
            feats = jax.tree.map(lambda a: a[lanes], prepared.features)
            stacked = jax.tree.map(lambda *a: jnp.stack(a), *paths)
            chosen = Choice(*(jnp.asarray(choice[j]) for j in range(3)))
            np.testing.assert_allclose(np.asarray(logp_fn(params, feats, chosen, stacked)), logp, atol=1e-4)
            c_np = [np.zeros(games, np.int32) for _ in range(3)]
            for j in range(3):
                c_np[j][lanes] = choice[j]
            full = Choice(*(jnp.asarray(a) for a in c_np))
            tick, status = engine.tick(states, full, seeds(tick_no + 1))
            assert not np.any(np.asarray(status)[lanes])
            twin_tick, twin_status = twin.tick(None, full, seeds(tick_no + 1))
            np.testing.assert_array_equal(twin_status, status)
            served = twin_tick.prepared
            states, prepared = tick.states, tick.prepared
        client.close()
    finally:
        server.join(timeout=60)
        if server.is_alive():
            server.terminate()
    assert server.exitcode == 0


def test_mixed_parameter_sets_and_two_clients_gather_the_right_rows():
    """One request per client mixing two parameter sets (the server's np.take gather into its batch buffers), two
    clients batched together over several calls (the buffer ring wraps): every lane's value is its own network's."""
    from polytopia_bot.rl.rust_engine import RustGames
    size, players, games = 14, 2, 6
    ctx = mp.get_context('spawn')
    pipes = [ctx.Pipe() for _ in range(2)]
    server = ctx.Process(target=serve, args=([s for s, _ in pipes],),
                         kwargs=dict(platform='cpu', deadline_ms=50., trunk_dtype='float32', precision='HIGHEST'),
                         daemon=True)
    server.start()
    try:
        spec = dict(size=size, blocks=2, extra_blocks=0, width=64, architecture='normalized-v2', players=players)
        base = _template_params(**spec)
        rng = np.random.default_rng(1)
        nets = {'learner': base, 'pool': jax.tree.map(lambda a: a + rng.normal(0, .05, a.shape).astype(a.dtype), base)}
        value = jax.jit(jax.vmap(lambda p, x: network.trunk(p, x)[2], in_axes=(None, 0)))
        clients, engines = [], []
        for k, (_, end) in enumerate(pipes):
            client = InferenceClient(end, size, games, players)
            for name, p in nets.items():
                client.publish(name, 1, [np.asarray(a) for a in jax.tree.leaves(p)], spec)
            clients.append(client)
            engines.append((RustGames(size, games, players=players), RustGames(size, games, players=players, served=True)))
        import threading
        for call in range(4):
            results, groups, asks = [None, None], [], []
            for k in range(2):
                seeds = np.arange(games, dtype=np.int32) + 300000 + 100 * call + 10 * k
                prepared = engines[k][0].reset(seeds)[1]
                served = engines[k][1].reset(seeds)[1]
                split = 1 + (call + k) % (games - 1)
                g = {'learner': np.arange(split), 'pool': np.arange(split, games)}
                groups.append((g, prepared))
                asks.append(threading.Thread(target=lambda k=k, g=g, s=served: results.__setitem__(
                    k, clients[k].sample(g, s.view, s.masks))))
            for t in asks:   # both requests reach the server within its deadline: one batch, two clients
                t.start()
            for t in asks:
                t.join()
            for k, (g, prepared) in enumerate(groups):
                for name, lanes in g.items():
                    want = np.asarray(value(nets[name], prepared.features))[lanes]
                    np.testing.assert_allclose(results[k][name][3], want, atol=2e-4, rtol=1e-4)
        for client in clients:
            client.close()
    finally:
        server.join(timeout=60)
        if server.is_alive():
            server.terminate()
    assert server.exitcode == 0
