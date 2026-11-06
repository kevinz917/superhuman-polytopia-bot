"""Both launch platforms stay usable from main (docs/platforms.md): LearnerConfig's defaults are the Modal (H100)
behaviour, Modal's hero configs set none of the Curio knobs, and the Curio configs carry the values measured on the
RTX 4090 head (2026-10-31) and agree with the cluster layout."""
import json
from pathlib import Path

import pytest

from polytopia_bot.rl.distributed import LearnerConfig

ROOT = Path(__file__).resolve().parents[1]
CURIO_KNOBS = ('send_thread', 'serialize_gpu', 'value_devices', 'unroll_loops', 'codec_shape_caps', 'coded_views',
               'stream_views')


def _config(path):
    raw = json.loads(Path(path).read_text())
    cfg = {k: v for k, v in raw.items() if not k.startswith('_') and k != 'parent'}
    cfg['frozen'] = tuple(tuple(x) for x in cfg.get('frozen', ()))
    return raw, LearnerConfig(**cfg)


def test_defaults_are_the_modal_behaviour():
    c = LearnerConfig()
    assert (c.send_thread, c.serialize_gpu, c.value_devices, c.unroll_loops, c.codec_shape_caps) == (True, False, 0, False, False)
    assert (c.coded_views, c.stream_views) == (False, True)


@pytest.mark.parametrize('name', ['hero-1v1.json', 'hero-1v1-cont1h.json', 'hero-ffa4.json'])
def test_modal_hero_configs_use_the_modal_profile(name):
    path = ROOT / 'experiments' / 'hero' / name
    if not path.exists():
        pytest.skip(f'{name} not in this checkout')
    raw, cfg = _config(path)
    cfg.validate()
    assert not {'value_devices', 'unroll_loops', 'codec_shape_caps', 'serialize_gpu'} & set(raw), name
    assert cfg.value_devices == 0 and not cfg.unroll_loops and not cfg.codec_shape_caps and not cfg.serialize_gpu


@pytest.mark.parametrize('name', ['hero-1v1-scratch.json', 'hero-1v1-curio.json', 'hero-1v1-curio-foghead.json'])
def test_curio_configs_use_the_curio_profile(name):
    raw, cfg = _config(ROOT / 'experiments' / 'curio' / 'configs' / name)
    cfg.validate()
    assert all(k in raw for k in ('send_thread', 'serialize_gpu', 'value_devices', 'codec_shape_caps', 'coded_views',
                                  'stream_views')), name
    assert (cfg.send_thread, cfg.stream_views, cfg.coded_views) == (False, False, True)
    assert cfg.value_devices == 4 and cfg.codec_shape_caps and cfg.serialize_gpu


def test_curio_layout_matches_its_learner_config():
    plan = json.loads((ROOT / 'experiments' / 'curio' / 'configs' / 'cluster.json').read_text())
    _, cfg = _config(ROOT / 'experiments' / 'curio' / 'configs' / 'hero-1v1-scratch.json')
    gpus = plan['learner']['gpus']
    train = len(gpus) - cfg.value_devices
    assert train > 0 and cfg.minibatch % train == 0 and cfg.batch % train == 0
    # value_devices with equal counts mirrors the coded-view decode (no dense cross-mesh hop)
    assert cfg.value_devices == train
    assert {a['gpu'] for a in plan['head_actors']} <= set(gpus[train:]), 'inference servers share the value GPUs'
