"""Levers 2 and 3 (docs/levers-egress-2026-10-22.md): the per-container weight relay hands every process the same
weights, pool and versions as the direct path; the background bookkeeping and the append-only training log
resume to the same state and keep the same log."""
import json
import socket
import threading
import time

import jax
import numpy as np
import pytest

from polytopia_bot.rl import distributed, training_log
from polytopia_bot.rl.distributed import (RELAY_WEIGHTS, Learner, LearnerConfig, RelayReader, WeightRelay,
                                          _new_actor_stats, _template_params, _write_atomic, recv_msg, send_msg)


def _parent(tmp_path, width=16, blocks=1):
    from polytopia_bot.rl.runner import Config, save_checkpoint
    template = _template_params(14, width=width, blocks=blocks)
    keys = jax.random.split(jax.random.key(0), len(jax.tree.leaves(template)))
    params = jax.tree.unflatten(jax.tree.structure(template),
                                [jax.random.normal(k, a.shape) * .2 for k, a in zip(keys, jax.tree.leaves(template))])
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent = tmp_path / 'parent.npz'
    save_checkpoint(parent, params, Config(architecture='normalized-v2', size=14, width=width, blocks=blocks), stage='p')
    return str(parent)


def _config(**overrides):
    cfg = dict(size=14, batch=256, minibatch=64, epochs=1, omniscient_critic=True, max_lag=10, publish_every=1,
               historical_probability=.2, historical_every=2, historical_keep=2, assembly_threads=0)
    cfg.update(overrides)
    return LearnerConfig(**cfg)


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Client:
    """The weight/pool bookkeeping of one actor process (_actor_process's sender + main loop), over a real socket."""

    def __init__(self, port, relay_dir=None):
        self.relay = RelayReader(relay_dir, timeout=20) if relay_dir else None
        self.sock = socket.create_connection(('127.0.0.1', port))
        send_msg(self.sock, dict(type='hello', actor=0, have=[], relayed=self.relay is not None))
        hello = recv_msg(self.sock)
        self.arrays = int(hello.get('weights') is not None or bool(hello.get('pool_params')))   # arrays on the wire
        if self.relay is not None:
            hello = self.relay.complete(hello)
        self.version, self.weights, self.critic = hello['version'], hello['weights'], hello['critic_weights']
        self.pool, self.spec = dict(hello['pool_params']), hello['pool_spec']

    def exchange(self):
        send_msg(self.sock, dict(type='data', actor=0, have=sorted(self.pool), batch=None, stats=_new_actor_stats(),
                                 segments=None))
        reply = recv_msg(self.sock)
        self.arrays += int(reply.get('weights') is not None or bool(reply.get('pool_params')))
        if self.relay is not None:
            reply = self.relay.complete(reply, self.version, sorted(self.pool))
        if reply.get('weights') is not None:
            self.version, self.weights, self.critic = reply['version'], reply['weights'], reply.get('critic_weights')
        if reply.get('pool_params') is not None:
            self.spec = reply['pool_spec']
            self.pool.update(reply['pool_params'])
            for name in list(self.pool):
                if name not in {n for n, _ in self.spec}:
                    del self.pool[name]
        return reply

    def state(self):
        as_bytes = lambda leaves: [(a.dtype.str, a.shape, a.tobytes()) for a in leaves]
        return (self.version, as_bytes(self.weights), self.critic is None or as_bytes(self.critic), self.spec,
                {n: as_bytes(l) for n, l in sorted(self.pool.items())})


def test_relayed_processes_see_the_direct_path_weights_pool_and_versions(tmp_path):
    """Each step publishes new weights (every other step a historical pool change); a direct client, two relayed
    clients and the relay then exchange one message each. All clients hold bit-identical weights, critic, pool and
    the same version after every step, and that version is the learner's published one. The relayed clients'
    connections carry no arrays, and the learner counts one relay."""
    learner = Learner(tmp_path / 'run', _config(learner_values=False), _parent(tmp_path), log=lambda s: None)
    learner.port, learner.bind_host = _free_port(), '127.0.0.1'
    learner.serve()
    relay_dir = tmp_path / 'relay'
    relay_dir.mkdir()
    relay = WeightRelay(('127.0.0.1', learner.port), relay_dir, connect_timeout=30).start()
    direct, relayed = Client(learner.port), [Client(learner.port, relay_dir) for _ in range(2)]
    assert direct.state() == relayed[0].state() == relayed[1].state()
    assert direct.critic is not None   # learner_values=False: the omniscient critic's leaves travel and are compared
    rng = np.random.default_rng(0)
    for step in range(1, 7):
        learner.version = step
        noisy = jax.tree.map(lambda a: a + rng.normal(size=a.shape).astype(np.float32) * .01, learner.params)
        learner.params = noisy
        learner._publish(published=(step, learner._leaves(noisy), learner._critic_leaves()))
        if step % 2 == 0:
            learner._publish(historical=(learner.historical + [(f'hist-u{step}', learner._leaves(noisy))])[-2:])
        for c in [direct] + relayed:
            c.exchange()
        want = direct.state()
        assert want[0] == step and {n for n, _ in want[3]} == {n for n, _ in learner.pool_spec()}
        for c in relayed:
            assert c.state() == want, step
    with learner.lock:
        assert learner.relays == 1
    assert direct.arrays >= 7 and all(c.arrays == 0 for c in relayed)   # relayed connections carry no arrays
    assert relay.stats['weight_versions'] >= 6 and relay.stats['pool_versions'] >= 3 and relay.stats['error'] is None
    learner.stop = True
    learner._publish()
    relay.stop()
    learner.server.close()


def test_relay_reader_ignores_another_learners_files(tmp_path):
    """A replaced learner restarts its versions: files from the old session never satisfy the new one."""
    _write_atomic(tmp_path / RELAY_WEIGHTS, ('old', 50, [np.zeros(3)], None))
    reader = RelayReader(tmp_path, timeout=.3)
    with pytest.raises(TimeoutError):
        reader.complete(dict(session='new', version=4))
    _write_atomic(tmp_path / RELAY_WEIGHTS, ('new', 4, [np.ones(3)], None))
    got = reader.complete(dict(session='new', version=4), known_version=2)
    assert got['version'] == 4 and got['weights'][0].sum() == 3
    assert reader.complete(dict(session='new', version=4), known_version=4).get('weights') is None


def _feed(learner, messages, stop):
    i = 0
    while not stop.is_set():
        batch, segs = messages[i % len(messages)]
        try:
            learner.inbox.put((dict(batch), dict(segs)), timeout=.1)
            i += 1
        except Exception:
            pass


def test_background_checkpoints_resume_to_the_same_state_and_log(tmp_path, monkeypatch):
    """A short run with checkpoints every 2 updates on the bookkeeping thread. Then: training.jsonl, the final
    training.json and the returned history hold the same rows; the status callback saw the last row; a learner
    restarted in the folder resumes bitwise the parameters, Adam state and critic that the training thread
    snapshotted at latest's update, with the log up to that update; the same from an older run's training.json
    alone, and from a log that runs past latest.npz with a torn last line."""
    from polytopia_bot.rl.assembly_bench import synthetic_messages
    monkeypatch.setattr(distributed, 'VALUE_CHUNK', 64)
    parent = _parent(tmp_path)
    cfg = _config(checkpoint_every=2, snapshot_every=4, training_seconds=150, historical_every=3, max_lag=10 ** 6)
    statuses, snaps = [], {}
    original = Learner.snapshot

    def snapshot(self):
        snap = original(self)
        snaps[snap['version']] = snap
        return snap
    monkeypatch.setattr(Learner, 'snapshot', snapshot)
    root = tmp_path / 'run'
    learner = Learner(root, cfg, parent, log=lambda s: None, status=statuses.append, port=_free_port())
    width = sum(int(np.prod(s)) for _, s in learner.layout)
    messages = synthetic_messages(np.random.default_rng(1), width, 196, 16, rows=(40, 90), omni=True)
    stop = threading.Event()
    threading.Thread(target=_feed, args=(learner, messages, stop), daemon=True).start()
    try:
        history = learner.run(bind_host='127.0.0.1')
    finally:
        stop.set()
    learner.server.close()
    assert len(history) >= 4, len(history)
    rows = json.loads(json.dumps(history))
    assert training_log.read(root / training_log.JSONL) == rows == training_log.read(root / training_log.JSON)
    assert statuses and statuses[-1] is history[-1]
    assert all('bookkeeping_seconds' in r and 'weight_bytes' in r and 'relays' in r for r in history)
    latest = int(json.loads((root / 'latest.json').read_text())['stage'].rsplit('-', 1)[1])
    assert latest == len(history) // 2 * 2

    def resumed_matches():
        again = Learner(root, cfg, parent, log=lambda s: None)
        assert again.version == latest
        assert again.resume_history == rows[:latest]
        for key, now in (('params', again.params), ('opt', again.opt), ('critic', again.critic)):
            for a, b in zip(jax.tree.leaves(snaps[latest][key]), jax.tree.leaves(now), strict=True):
                assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), key
        assert training_log.read(root / training_log.JSONL) == rows[:latest]
        return again
    resumed_matches()
    (root / training_log.JSONL).unlink()           # an older run: training.json only
    resumed_matches()
    training_log.append(root / training_log.JSONL, rows[latest:latest + 1])
    with open(root / training_log.JSONL, 'a') as f:
        f.write('{"update": 99, "cut sho')       # the container died mid-append
    resumed_matches()


def test_upload_zstd_level_is_lossless_and_the_default():
    """Lever 4: a level-7 upload frame decodes to the same arrays as level 1 (kind 2, zero-copy views)."""
    from polytopia_bot.rl.distributed import decode_body, encode_msg
    assert LearnerConfig().upload_zstd_level == 7
    rng = np.random.default_rng(2)
    msg = dict(type='data', batch=dict(view=rng.integers(-16, 347, (64, 50)).astype(np.int16),
                                       logp=rng.normal(size=64).astype(np.float32)), stats={}, segments=None)
    one, seven = encode_msg(msg, level=1), encode_msg(msg, level=7)
    assert one[8] == seven[8] == 2
    for frame in (one, seven):
        got = decode_body(frame[9:], 2)
        for k, v in msg['batch'].items():
            assert got['batch'][k].tobytes() == v.tobytes() and got['batch'][k].dtype == v.dtype
