"""LearnerConfig.value_devices: the learner_values pass on its own devices builds the same training batch, bitwise,
as the pass on the training devices, and the batch lands on the training devices (coded views)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CHILD = r'''
import sys, json, pathlib
sys.path.insert(0, "tests")
import numpy as np
from polytopia_bot.rl import distributed
distributed.VALUE_CHUNK = 64
import jax
from polytopia_bot.rl.assembly_bench import assemble_once, synthetic_messages
from polytopia_bot.rl.view_codec import encode
from test_a6_assembly import _learner, _width
from test_view_codec import _realistic
out, devices, value_devices = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
assert jax.device_count() == devices
learner = _learner(out / "run", coded_views=True, stream_views=False, value_devices=value_devices)
rng = np.random.default_rng(7)
messages = _realistic(synthetic_messages(rng, _width(learner), 196, 12, rows=(40, 90), omni=True), rng)
batch, info = assemble_once(learner, [encode(dict(b), dict(s)) for b, s in messages], False, coded=True)
np.savez(out / "batch.npz", **{k: np.asarray(v) for k, v in batch.items()})
train_ids = sorted(d.id for d in learner.mesh.devices.flat)
on_train = {k: sorted(d.id for d in v.sharding.device_set) == train_ids for k, v in batch.items()}
stats = {k: v for k, v in info.items() if not k.endswith("_seconds")}
(out / "meta.json").write_text(json.dumps(dict(on_train=on_train, info=stats, split=learner.split_values,
                                               value_ids=sorted(d.id for d in learner.value_mesh.devices.flat),
                                               train_ids=train_ids), default=str))
print("ok")
'''


def _run(tmp, devices, value_devices):
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, XLA_FLAGS=f'--xla_force_host_platform_device_count={devices}', JAX_PLATFORMS='cpu')
    done = subprocess.run([sys.executable, '-c', CHILD, str(tmp), str(devices), str(value_devices)], env=env,
                          capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert done.returncode == 0 and 'ok' in done.stdout.split(), done.stdout[-3000:] + done.stderr[-3000:]
    return dict(np.load(tmp / 'batch.npz')), json.loads((tmp / 'meta.json').read_text())


def test_value_devices_batch_matches_and_lands_on_training_devices(tmp_path):
    base, base_meta = _run(tmp_path / 'shared', 2, 0)
    split, split_meta = _run(tmp_path / 'split', 4, 2)
    assert not base_meta['split'] and split_meta['split']
    assert split_meta['train_ids'] == [0, 1] and split_meta['value_ids'] == [2, 3]
    assert all(split_meta['on_train'].values()), split_meta['on_train']
    assert set(base) == set(split)
    for k in base:
        np.testing.assert_array_equal(base[k], split[k], err_msg=k)
    assert base_meta['info'] == split_meta['info']


def test_value_devices_validation():
    from polytopia_bot.rl.distributed import LearnerConfig
    with pytest.raises(ValueError):
        LearnerConfig(value_devices=2, stream_views=True, coded_views=False).validate()
    LearnerConfig(value_devices=2, stream_views=True, coded_views=True).validate()
    LearnerConfig(value_devices=2, stream_views=False).validate()
