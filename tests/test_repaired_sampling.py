import importlib.util
from pathlib import Path
import sys
import pytest


def test_end_turn_cap_limits_each_source_without_duplicate_rows(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('repaired_sampler',scripts/'sample_repaired_sft.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    groups={('move',1):[dict(input_hash=f'm-{i:02}') for i in range(8)],
            ('end',10):[dict(input_hash=f'e-{i:02}') for i in range(20)]}
    assert module.capacity(groups)==10
    rows=module.choose(groups,10)
    assert len(rows)==len({r['input_hash'] for r in rows})==10
    assert sum(r['input_hash'].startswith('e') for r in rows)==2
    assert {r['input_hash'] for r in rows if r['input_hash'].startswith('e')}=={'e-00','e-01'}
    with pytest.raises(ValueError,match='end-turn cap'):module.choose(groups,11)
