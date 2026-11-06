import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('movement_summary', Path(__file__).resolve().parents[1]/'scripts/summarize_movement_continuation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_paired_bootstrap_preserves_within_map_cancellation():
    before = [dict(position_id=str(i), board_hash=str(i//2), tie_aware=bool(i % 2)) for i in range(8)]
    after = [{**row, 'tie_aware': not row['tie_aware']} for row in reversed(before)]
    result = module.paired_change(before, after)
    assert result['delta'] == 0
    assert result['interval_95'] == [0, 0]
    assert result['improved'] == result['regressed'] == 4
    assert result['maps'] == 4


def test_paired_bootstrap_rejects_different_states_or_maps():
    before = [dict(position_id='a', board_hash='map1', tie_aware=False)]
    with pytest.raises(ValueError, match='positions differ'):
        module.paired_change(before, [{**before[0], 'position_id': 'b'}])
    with pytest.raises(ValueError, match='map changed'):
        module.paired_change(before, [{**before[0], 'board_hash': 'map2'}])
