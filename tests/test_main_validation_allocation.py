"""The main allocator must preserve the full validation set and spending gate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from polytopia_bot.learning.interface import digest


def test_full_validation_only_and_gate_required(tmp_path, monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/allocate_main_validation.py'
    spec=importlib.util.spec_from_file_location('main_validation_cli',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    # Prompt cryptography is covered separately; isolate job selection here.
    monkeypatch.setattr(module,'verify_prompt',lambda prompt,config: '')
    own=tmp_path/'scripts/allocate_main_validation.py';own.parent.mkdir();own.write_bytes(script.read_bytes())
    monkeypatch.setattr(module,'__file__',str(own))
    def save(path,value):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    save('config.json',dict(model_id='pinned',revision='rev',max_length=32768,max_new_tokens=32))
    save('prompt.json',dict(manifest_hash='prompt',tokenizer_hash='tok',chat_template_hash='chat',candidate='original',strategy=''))
    save('gate.json',dict(passed=False))
    with pytest.raises(ValueError,match='spending gate'):
        module.allocate(Path('blocked.json'),'main-base',Path('config.json'),Path('prompt.json'),Path('gate.json'),600)
    assert not (tmp_path/'blocked.json').exists()
    save('gate.json',dict(passed=True))
    maps=dict(size=14,maps=dict(validation=[dict(seed=i,board_hash=f'val-{i}') for i in range(200)],
                                test=[dict(seed=1000+i,board_hash=f'test-{i}') for i in range(500)]))
    maps['manifest_hash']=digest(maps);save('runs/sft0/splits-v1/maps.json',maps)
    q=save('qualification.json',dict(teacher_qualified=True))
    save('runs/sft0/qualified-teacher-v1.json',dict(qualification=dict(teacher_qualified=True),
         qualification_path='qualification.json',qualification_sha256=sha(q),interface_sources={},weak_opponent_hash='weak'))
    save('configs/ruleset-v1.json',dict(source_sha256={}))
    for path in ('training/pyproject.toml','training/uv.lock','experiments/modal_prompt_games.py'):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('test fixture')
    module.allocate(Path('allowed.json'),'main-base',Path('config.json'),Path('prompt.json'),Path('gate.json'),600)
    allocation=json.loads((tmp_path/'allowed.json').read_text())
    assert len(allocation['jobs'])==400
    assert {(j['seed'],j['seat']) for j in allocation['jobs']}=={(i,seat) for i in range(200) for seat in (1,2)}
    assert not allocation['final_test'] and allocation['adapter'] is None
    assert allocation['shards']==4 and allocation['config']['max_length']==32768
    gate_receipt=next(r for r in allocation['prerequisites'] if r['path']=='gate.json')
    assert gate_receipt['sha256']==sha(tmp_path/'gate.json')
