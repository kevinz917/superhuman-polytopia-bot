import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('merge_evals', Path(__file__).parents[1] / 'scripts/combine_structured_evaluations.py')
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


def source(root, seed, *, checkpoint='same', reward=1):
    root.mkdir()
    protocol = dict(checkpoint_sha256=checkpoint, config={'size': 14}, opponent='balanced',
                    teacher_hash='teacher', deterministic=True, paired_seats=True,
                    sampling_seed=42, source_sha256='evaluator', map_seeds=[seed], requested_games=2)
    rows = []
    for seat in (1, 2):
        trace = root / f'{seat}.trace'
        trace.write_bytes(f'{seed}:{seat}:{reward}'.encode())
        rows.append(dict(seed=seed, seat=seat, terminated=bool(reward), truncated=not reward,
                         reward=reward, evaluation_cutoff=False, max_cities=3,
                         trace=trace.name, trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
    for name, data in [('protocol', protocol), ('summary', dict(complete=True, games=2)), ('games', rows)]:
        (root / (name+'.json')).write_text(json.dumps(data))
    return root


def test_merge_preserves_unresolved_and_verifies_coverage(tmp_path):
    a = source(tmp_path/'a', 300000)
    b = source(tmp_path/'b', 300001, reward=0)
    result = merge.combine([a, b], tmp_path/'out', maps=2)
    assert (result['wins'], result['truncations'], result['native_win_fraction']) == (2, 2, .5)
    assert result['trace_hashes_verified'] == 4


@pytest.mark.parametrize('failure', ['duplicate', 'missing', 'checkpoint', 'trace', 'random_opponent'])
def test_rejects_invalid_merge_before_publishing(tmp_path, failure):
    a = source(tmp_path/'a', 300000)
    b = source(tmp_path/'b', 300001, checkpoint='different' if failure == 'checkpoint' else 'same')
    parts = [a, b]
    if failure == 'duplicate':
        parts = [a, a]
    elif failure == 'missing':
        parts = [a]
    elif failure == 'trace':
        (a/'1.trace').write_bytes(b'changed')
    elif failure == 'random_opponent':
        for root in parts:
            protocol = json.loads((root/'protocol.json').read_text())
            protocol['opponent'] = 'random'
            (root/'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        merge.combine(parts, tmp_path/'out', maps=2)
    assert not (tmp_path/'out').exists()
