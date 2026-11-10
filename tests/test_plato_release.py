"""The shipped checkpoint must work through the public Studio path, not just a toy V3 tree."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('plato_release_cli', ROOT / 'scripts' / 'plato.py')
plato = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plato)


def test_released_policy_and_critic_play_legal_studio_moves():
    result = plato.verify(ROOT / 'weights/plato/final.npz')
    assert result['policy_parameters'] == 3212922
    assert result['critic_parameters'] == 400805
    assert result['legal_decisions'] == 8
    assert result['sha256'] == 'aced9e4ea74527631b63236a0e0697bff37896549d7349083698d672c15e9420'


def test_local_training_table_matches_the_map_generator(tmp_path):
    from polytopia_bot.maps.stream import MapSource
    from polytopia_bot.rl.vector_actor import MapTable
    path = tmp_path / 'maps.npz'
    plato.map_table(path, 200000, 200003)
    table = MapTable(path, lambda: 200001)
    seed, kind, layout = table.get()
    expected_kind, expected = MapSource('classic', 14, players=2).layout(seed)
    assert kind == expected_kind
    for got, key in zip(layout, ('terrain', 'structure', 'resource', 'capitals')):
        np.testing.assert_array_equal(got, getattr(expected, key))
