import importlib
from pathlib import Path


def test_uniform_move_reference_preserves_probabilities_and_map_pairs(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module=importlib.import_module('summarize_movement_probe')
    rows=[dict(board_hash=str(board),tie_aware=bool(i),top_movement_count=1 if i==0 else 3,legal_movement_count=4)
          for board in range(3) for i in range(2)]
    result=module.movement_reference_comparison(rows)
    assert result['delta']==0
    assert result['interval_95']==[0,0]
