"""A6: the streamed batch assembly (DeviceRows, stream_views=True) equals the host path bitwise."""
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from polytopia_bot.rl import distributed
from polytopia_bot.rl.assembly_bench import gate, synthetic_messages, with_options


def _learner(tmp_path, **overrides):
    from polytopia_bot.rl.distributed import Learner, LearnerConfig, _template_params
    from polytopia_bot.rl.runner import Config, save_checkpoint
    template = _template_params(14)
    keys = jax.random.split(jax.random.key(0), len(jax.tree.leaves(template)))
    params = jax.tree.unflatten(jax.tree.structure(template),
                                [jax.random.normal(k, a.shape) * .2 for k, a in zip(keys, jax.tree.leaves(template))])
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent = tmp_path / 'parent.npz'
    save_checkpoint(parent, params, Config(architecture='normalized-v2', size=14), stage='p')
    cfg = dict(size=14, batch=256, minibatch=64, omniscient_critic=True, max_lag=10)
    cfg.update(overrides)
    return Learner(tmp_path / 'run', LearnerConfig(**cfg), str(parent), log=lambda s: None)


def _width(learner):
    return sum(int(np.prod(s)) for _, s in learner.layout)


def _check(report):
    assert report['equal'], report
    assert all(report['fields'].values())


def run_cases(tmp_path):
    """Every option that changes what assembly computes, against one learner per critic setting."""
    rng = np.random.default_rng(3)
    learner = _learner(tmp_path / 'omni')
    width = _width(learner)
    base = synthetic_messages(rng, width, 196, 12, rows=(40, 90), omni=True)
    reports = {}
    reports['base'] = gate(learner, base)
    cfg = learner.config
    learner.config = dataclasses.replace(cfg, turn_gamma=.99, turn_lambda=.95)
    reports['turn_clock'] = gate(learner, with_options(base, rng, turn_clock=True))
    learner.config = dataclasses.replace(cfg, upgo_coef=.5)
    reports['upgo_phi'] = gate(learner, with_options(base, rng, shaping=True))
    learner.config, learner.version = cfg, 12
    reports['stale'] = gate(learner, with_options(base, rng, stale_version=12))
    learner.version = 0
    learner.config = dataclasses.replace(cfg, advantage_keep=.5)
    reports['advantage_keep'] = gate(learner, base)
    learner.config = cfg
    plain = _learner(tmp_path / 'plain', omniscient_critic=False)
    reports['policy_value_head'] = gate(plain, synthetic_messages(rng, width, 196, 12, rows=(40, 90), omni=False))
    return reports


def test_streamed_assembly_matches_host_path(tmp_path, monkeypatch):
    # small value chunks: a batch spans several pages and messages straddle page boundaries
    monkeypatch.setattr(distributed, 'VALUE_CHUNK', 64)
    reports = run_cases(tmp_path)
    for name, report in reports.items():
        _check(report)
    assert reports['stale']['info']['stale_dropped'] > 0
    assert {'view', 'opp_view', 'advantage', 'returns', 'logp', 'target_mask'} <= set(reports['base']['fields'])
    assert 'upgo_advantage' in reports['upgo_phi']['fields'] and 'opp_view' not in reports['policy_value_head']['fields']
    assert reports['base']['info']['messages'] >= 4
    assert reports['advantage_keep']['rows'] == 128


def test_device_rows_layout_and_growth(tmp_path, monkeypatch):
    """Pages fill in row order, grow past the initial size, and position() is where finish() put each row."""
    monkeypatch.setattr(distributed, 'VALUE_CHUNK', 16)
    learner = _learner(tmp_path, batch=64, minibatch=32)
    rows = distributed.DeviceRows(learner, pages=1)
    a = np.arange(100 * 3, dtype=np.int16).reshape(100, 3)
    for lo, hi in ((0, 7), (7, 40), (40, 41), (41, 100)):
        rows.add(dict(view=a[lo:hi]))
    rows._poll(block=True)                      # add() sends a page only once its threaded copies are done
    rows._drain()
    assert rows.pages == 6                      # six full pages sent; the partial 7th goes at finish
    n_pad = 112
    out = np.asarray(rows.finish(n_pad)['view'])
    assert rows.pages == 7
    rows.release()
    assert len(learner.host_pages[((3,), np.dtype(np.int16).str)]) == 7   # pages back in the pool
    again = distributed.DeviceRows(learner, pages=7)                       # reused pages: stale tails are zeroed
    again.add(dict(view=a[:20] + 1))
    second = np.asarray(again.finish(32)['view'])
    np.testing.assert_array_equal(second[:20], a[:20] + 1)
    assert (second[20:] == 0).all()
    assert out.shape == (n_pad, 3)
    np.testing.assert_array_equal(out[rows.position(np.arange(100), n_pad)], a)
    assert (out[100:] == 0).all()


@pytest.mark.skipif(os.environ.get('A6_CHILD') == '1', reason='runs in the two-device child')
def test_streamed_assembly_matches_on_two_devices(tmp_path):
    """Pages alternate between devices; the gathered batch and the values are still the host path's."""
    env = dict(os.environ, A6_CHILD='1', XLA_FLAGS='--xla_force_host_platform_device_count=2', JAX_PLATFORMS='cpu')
    code = ('import sys, pathlib; sys.path.insert(0, "tests"); from polytopia_bot.rl import distributed; '
            'distributed.VALUE_CHUNK = 64; import jax; assert jax.device_count() == 2; '
            'import test_a6_assembly as t; reports = t.run_cases(pathlib.Path(sys.argv[1])); '
            '[t._check(r) for r in reports.values()]; print("ok", len(reports))')
    out = subprocess.run([sys.executable, '-c', code, str(tmp_path)], env=env, capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0 and 'ok 6' in out.stdout, out.stdout[-3000:] + out.stderr[-3000:]


def test_wire_kinds_round_trip_and_views_are_zero_copy():
    """Kind 2 (send_msg's compressed framing) decodes to the same message; its arrays view the decompressed
    bytes (read-only, no per-array copy); kinds 0 and 1 still decode (replies, captured files)."""
    import pickle
    import socket
    import threading
    from polytopia_bot.rl.distributed import _compress, _frame, decode_body, encode_msg, recv_msg, send_msg
    rng = np.random.default_rng(1)
    msg = dict(type='data', batch=dict(view=rng.integers(0, 3, (37, 11)).astype(np.int16),
                                       logp=rng.normal(size=37).astype(np.float32),
                                       choice=rng.integers(0, 9, (37, 3)).astype(np.int32)[:, ::2]),   # non-contiguous
               stats=dict(steps=4), segments=None)
    def same(a, b):
        assert a.keys() == b.keys()
        for k in a['batch']:
            np.testing.assert_array_equal(a['batch'][k], b['batch'][k])
            assert a['batch'][k].dtype == b['batch'][k].dtype
        assert a['stats'] == b['stats'] and a['segments'] is None
    for kind, frame in ((2, encode_msg(msg)), (0, encode_msg(msg, compress=False)),
                        (1, _frame(_compress(pickle.dumps(msg, protocol=5)), 1))):
        assert frame[8] == kind
        got = decode_body(frame[9:], kind)
        same(got, msg)
        if kind == 2:
            assert not got['batch']['view'].flags.writeable and not got['batch']['view'].flags.owndata
        a, b = socket.socketpair()
        thread = threading.Thread(target=a.sendall, args=(frame,))
        thread.start()
        clock = dict(decompress=0., unpickle=0.)
        same(recv_msg(b, clock), msg)
        thread.join()
        a.close(), b.close()
