import json
from copy import deepcopy
import pytest
torch=pytest.importorskip('torch')
from polytopia_bot.learning.continuation import load_trainable_adapter
from polytopia_bot.learning.training import file_hash


def saved_adapter(tmp_path):
    from test_efficient_training import model
    original=model()
    with torch.no_grad():
        for name,p in original.named_parameters():
            if 'lora_B' in name:p.fill_(.03)
    original.save_pretrained(tmp_path)
    saved=json.loads((tmp_path/'adapter_config.json').read_text())
    config=dict(model_id=saved['base_model_name_or_path'],revision=saved['revision'],
                lora_rank=saved['r'],lora_alpha=saved['lora_alpha'],lora_dropout=saved['lora_dropout'],target_modules=saved['target_modules'])
    return original,config,{p.name:file_hash(p) for p in tmp_path.glob('adapter*')}


def test_continuation_preserves_loaded_values_and_only_updates_adapter(tmp_path):
    from test_efficient_training import rows
    from polytopia_bot.learning.efficient_training import response_only_loss
    from polytopia_bot.learning.data import ResponseCollator
    from peft import get_peft_model_state_dict
    original,config,hashes=saved_adapter(tmp_path)
    continued=load_trainable_adapter(deepcopy(original).unload(),tmp_path,config,hashes)
    for key,value in get_peft_model_state_dict(original).items():
        torch.testing.assert_close(value,get_peft_model_state_dict(continued)[key],rtol=0,atol=0)
    frozen={n:p.detach().clone() for n,p in continued.named_parameters() if not p.requires_grad}
    before={n:p.detach().clone() for n,p in continued.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in continued.parameters() if p.requires_grad],lr=.001)
    response_only_loss(continued,ResponseCollator(0)(rows())).backward();optimizer.step()
    assert any(not torch.equal(before[n],p) for n,p in continued.named_parameters() if p.requires_grad)
    assert all(torch.equal(frozen[n],p) for n,p in continued.named_parameters() if not p.requires_grad)


def test_continuation_rejects_changed_adapter(tmp_path):
    original,config,hashes=saved_adapter(tmp_path);hashes['adapter_model.safetensors']='bad'
    with pytest.raises(ValueError,match='adapter changed'):
        load_trainable_adapter(deepcopy(original).unload(),tmp_path,config,hashes)


def test_continuation_rejects_different_model_revision(tmp_path):
    original,config,hashes=saved_adapter(tmp_path);config['revision']='different'
    with pytest.raises(ValueError,match='configuration differs'):
        load_trainable_adapter(deepcopy(original).unload(),tmp_path,config,hashes)
