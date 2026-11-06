"""Load an existing LoRA for further optimization without reinitializing it."""
from pathlib import Path
from .training import file_hash


def load_trainable_adapter(base, path, config, expected_files):
    from peft import PeftConfig,PeftModel,get_peft_model_state_dict
    from safetensors.torch import load_file
    import torch
    path=Path(path)
    if set(expected_files)!={'adapter_config.json','adapter_model.safetensors'}:
        raise ValueError('complete adapter identities required')
    if any(file_hash(path/name)!=sha for name,sha in expected_files.items()):
        raise ValueError('initial adapter changed')
    saved=PeftConfig.from_pretrained(path)
    checks={'base_model_name_or_path':config['model_id'],'revision':config['revision'],
            'r':config['lora_rank'],'lora_alpha':config['lora_alpha'],'lora_dropout':config['lora_dropout']}
    if any(getattr(saved,key)!=value for key,value in checks.items()) or set(saved.target_modules)!=set(config['target_modules']):
        raise ValueError('initial adapter configuration differs')
    model=PeftModel.from_pretrained(base,path,is_trainable=True)
    expected=load_file(path/'adapter_model.safetensors',device='cpu')
    actual=get_peft_model_state_dict(model)
    if expected.keys()!=actual.keys() or any(not torch.equal(expected[k],actual[k].detach().cpu()) for k in expected):
        raise ValueError('loaded adapter values differ from checkpoint')
    if not any(p.requires_grad for p in model.parameters()) or any(p.requires_grad and 'lora_' not in name for name,p in model.named_parameters()):
        raise ValueError('LoRA-only trainability required')
    return model
