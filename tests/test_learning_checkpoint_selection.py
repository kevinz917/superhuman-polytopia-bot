from copy import deepcopy
import pytest
from polytopia_bot.learning.checkpoint_selection import select_checkpoint


def data():
    jobs=[dict(seed=i,board_hash=str(i),seat=seat,opponent='weak',strategy='') for i in range(200) for seat in (1,2)]
    reports={step:dict(complete=True,unstarted_jobs=[],games=[dict(j,status='native',reward=-1,model_decisions=10,first_attempt_valid=10) for j in jobs]) for step in (196,391,782)}
    return jobs,reports


def test_preregistered_score_error_and_earlier_step_tie_breaks():
    jobs,reports=data()
    assert select_checkpoint(reports,jobs,[196,391,782])['selected_step']==196
    reports[782]['games'][0]['reward']=1
    assert select_checkpoint(reports,jobs,[196,391,782])['selected_step']==782
    reports[391]['games'][0]['reward']=1
    assert select_checkpoint(reports,jobs,[196,391,782])['selected_step']==391
    reports[391]['games'][0]['first_attempt_valid']=9
    assert select_checkpoint(reports,jobs,[196,391,782])['selected_step']==782


def test_reject_incomplete_or_changed_validation_and_keep_truncation_unresolved():
    jobs,reports=data();reports[782]['games'][0].update(status='truncated',reward=0)
    out=select_checkpoint(reports,jobs,[196,391,782]);assert out['summaries'][782]['conservative_score']==0
    reports[782]['games'][0]['board_hash']='wrong'
    with pytest.raises(ValueError,match='different'):select_checkpoint(reports,jobs,[196,391,782])
    jobs,reports=data();reports[196]['games'].pop()
    with pytest.raises(ValueError,match='400'):select_checkpoint(reports,jobs,[196,391,782])


def test_selection_cli_checks_actual_checkpoint_files(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    import json
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/select_sft_main_checkpoint.py'
    spec=importlib.util.spec_from_file_location('checkpoint_cli',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    def save(path,value):
        target=tmp_path/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(value))
    jobs,reports=data()
    config=dict(model_id='pinned',revision='revision',max_length=16384,max_new_tokens=32)
    sources={'src/polytopia_bot/learning/interface.py':'frozen-source'}
    frozen=dict(manifest_hash='prompt',strategy='',sources=sources,tokenizer_hash='tokenizer',chat_template_hash='template')
    save(Path('training/execution.json'),dict(exit_code=0,run=dict(status='completed',stage='main',seed=17,data_manifest=dict(prompt_manifest=frozen))))
    save(Path('training/allocation.json'),dict(run_id='main-seed17',config=config))
    save(Path('training/training-run/model/trainer_state.json'),dict(global_step=782,max_steps=782))
    save(Path('runs/sft0/splits-v1/maps.json'),dict(maps=dict(validation=[dict(seed=i,board_hash=str(i)) for i in range(200)])))
    evaluations=[]
    for step,report in reports.items():
        folder=Path(f'eval-{step}');evaluations.append(folder)
        local=Path(f'training/training-run/model/checkpoint-{step}')
        save(local/'adapter_model.safetensors',dict(test_fixture_step=step))
        hashes={'adapter_model.safetensors':hashlib.sha256((tmp_path/local/'adapter_model.safetensors').read_bytes()).hexdigest()}
        remote=f'/workspace/runs/main-seed17/model/checkpoint-{step}'
        model=dict(config=config,adapter=remote,adapter_files=hashes,tokenizer_hash='tokenizer',chat_template_hash='template')
        save(folder/'protocol.json',dict(adapter=remote,prompt_manifest_hash='prompt',sources=sources,shards=4))
        save(folder/'combined/model.json',model)
        save(folder/'combined/games/report.json',report)
        save(folder/'execution.json',[dict(exit_code=0)]*4)
    module.select(Path('training'),evaluations,Path('selected.json'))
    selected=json.loads((tmp_path/'selected.json').read_text())
    assert selected['selected_step']==196
    assert selected['selected_adapter']['local']=='training/training-run/model/checkpoint-196'
    (tmp_path/'training/training-run/model/checkpoint-391/adapter_model.safetensors').write_text('tampered')
    with pytest.raises(ValueError,match='adapter differs'):
        module.select(Path('training'),evaluations,Path('tampered-selection.json'))
