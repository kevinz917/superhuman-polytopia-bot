"""Coded views (rl/view_codec.py): the wire format round-trips exactly, and a learner fed coded messages builds
the same training batch, bitwise, as the dense streamed path (two seats with and without the omniscient critic,
stale rows), on one device and on four (each device rebuilds its own shard; messages straddle shard edges)."""
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from polytopia_bot.rl import distributed, view_codec
from polytopia_bot.rl.assembly_bench import coded_gate, synthetic_messages, with_options
from test_a6_assembly import _learner, _width


def _realistic(messages, rng):
    """Synthetic views are i.i.d. noise; make consecutive rows of a segment mostly equal, as real ones are."""
    out = []
    for batch, segs in messages:
        b, s = dict(batch), dict(segs)
        for key in [k for k in ('view', 'opp_view') if k in b]:
            v = b[key].copy()
            for r in range(1, len(v)):
                if b['pos'][r] > 0:
                    keep = rng.random(v.shape[1:]) < .97
                    v[r] = np.where(keep, v[r - 1], v[r])
            b[key] = v
        out.append((b, s))
    return out


def test_round_trip_two_and_four_seats():
    rng = np.random.default_rng(0)
    for k in (1, 3):
        for batch, segs in synthetic_messages(rng, 300, 196, 6, rows=(40, 90), omni=True):
            batch, segs = dict(batch), dict(segs)
            if k == 3:
                batch['opp_view'] = rng.integers(-1, 300, (len(batch['logp']), 3, 300)).astype(np.int16)
                segs['boot_opp_view'] = rng.integers(-1, 300, (len(segs['length']), 3, 300)).astype(np.int16)
                segs['boot_opp_view'][np.asarray(segs['terminal'])] = 0
            enc_b, enc_s = view_codec.encode(batch, segs)
            dec_b, dec_s = view_codec.decode_numpy(enc_b, enc_s)
            for key in ('view', 'opp_view'):
                np.testing.assert_array_equal(dec_b[key], batch[key])
            for key in ('boot_view', 'boot_opp_view'):
                np.testing.assert_array_equal(dec_s[key], segs[key])


def test_learner_coded_views_match_dense(tmp_path, monkeypatch):
    monkeypatch.setattr(distributed, 'VALUE_CHUNK', 64)
    _check_coded(tmp_path)


@pytest.mark.skipif(os.environ.get('CODEC_CHILD') == '1', reason='runs in the four-device child')
def test_learner_coded_views_match_dense_on_four_devices(tmp_path):
    """Blocks land on the device owning their rows' (and boundaries') shard; straddling blocks run on both."""
    env = dict(os.environ, CODEC_CHILD='1', XLA_FLAGS='--xla_force_host_platform_device_count=4', JAX_PLATFORMS='cpu')
    code = ('import sys, pathlib; sys.path.insert(0, "tests"); from polytopia_bot.rl import distributed, view_codec; '
            'distributed.VALUE_CHUNK = 64; import jax; assert jax.device_count() == 4; '
            'import test_view_codec as t; t._check_coded(pathlib.Path(sys.argv[1])); print("ok")')
    out = subprocess.run([sys.executable, '-c', code, str(tmp_path)], env=env, capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0 and 'ok' in out.stdout.split(), out.stdout[-3000:] + out.stderr[-3000:]


def _check_coded(tmp_path):
    rng = np.random.default_rng(5)
    learner = _learner(tmp_path / 'omni')
    width = _width(learner)
    base = _realistic(synthetic_messages(rng, width, 196, 12, rows=(40, 90), omni=True), rng)
    reports = dict(base=coded_gate(learner, base))
    learner.config, learner.version = dataclasses.replace(learner.config, max_lag=10), 12
    reports['stale'] = coded_gate(learner, with_options(base, rng, stale_version=12))
    learner.version = 0
    cfg = learner.config
    learner.config = dataclasses.replace(cfg, turn_gamma=.99, turn_lambda=.95)
    reports['turn_clock'] = coded_gate(learner, with_options(base, rng, turn_clock=True))
    learner.config = cfg
    plain = _learner(tmp_path / 'plain', omniscient_critic=False)
    reports['policy_value_head'] = coded_gate(plain, _realistic(
        synthetic_messages(rng, width, 196, 12, rows=(40, 90), omni=False), rng))
    for name, report in reports.items():
        assert report['equal'], (name, report)
    assert reports['stale']['info']['stale_dropped'] > 0
    assert {'view', 'opp_view'} <= set(reports['base']['fields'])
