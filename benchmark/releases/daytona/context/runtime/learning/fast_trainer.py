"""Single-GPU token-budget SFT with immutable adapter publication."""
import json,math,time
from pathlib import Path
from .training import verified_data,load_tokenizer,lora_config,parameter_hash,frozen_weight_hash,training_response_accuracy,file_hash
from .data import ResponseCollator
from .efficient_training import EffectiveBatchLengthSampler,response_only_loss
from .fast_training import pack,token_batches,packed_response_loss,configure_checkpointing,set_backend,set_lora_input_cast


def train_fast(prepared,output,config,*,seed,stage,readiness,benchmark,checkpoint_every=16):
    import torch
    from transformers import AutoModelForCausalLM,set_seed,get_cosine_schedule_with_warmup
    from peft import get_peft_model,PeftModel
    if not benchmark['parity']['passed'] or benchmark['speedup']<=1:
        raise ValueError('passing faster runtime benchmark required')
    variant=benchmark['selected']
    manifest,data=verified_data(prepared,config,stage=stage,readiness=readiness)
    if manifest['max_tokens']>variant['stress_max_length']:raise ValueError('longest input untested')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    set_seed(seed);started=time.monotonic();tokenizer=load_tokenizer(config)
    from .interface import digest
    if digest(tokenizer.backend_tokenizer.to_str())!=manifest['tokenizer_hash'] or digest(tokenizer.chat_template)!=manifest['chat_template_hash']:
        raise ValueError('prepared tokenizer differs')
    rows=list(data);lengths=[len(r['input_ids']) for r in rows]
    model=AutoModelForCausalLM.from_pretrained(config['model_id'],revision=config['revision'],
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    set_backend(model,variant.get('backend','flash_attention_2' if variant['packed'] else 'sdpa'))
    model.config.use_cache=False;model=get_peft_model(model,lora_config(config));model.train()
    configure_checkpointing(model,variant['checkpointing'])
    set_lora_input_cast(model,not variant.get('skip_lora_cast',False))
    if any(p.requires_grad and 'lora_' not in n for n,p in model.named_parameters()):raise ValueError('base trainability changed')
    params=[p for p in model.parameters() if p.requires_grad]
    collate=ResponseCollator(tokenizer.pad_token_id);effective=config['effective_batch_size']
    steps=math.ceil(len(rows)/effective)*int(config['epochs'])
    if config['epochs']!=int(config['epochs']):raise ValueError('integer epochs required')
    optim=torch.optim.AdamW(params,lr=config['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0)
    scheduler=get_cosine_schedule_with_warmup(optim,math.ceil(steps*config['warmup_ratio']),steps)
    receipt=dict(schema='fast-training-v2',status='training',stage=stage,config=config,seed=seed,
        data_manifest=manifest,variant=variant,hardware=torch.cuda.get_device_name(),planned_steps=steps,
        logs=[],checkpoints=[],trainable_parameters=sum(p.numel() for p in params))
    def save():
        tmp=root/'run.json.tmp';tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(root/'run.json')
    def fit_accuracy(m):
        backend=m.get_base_model().config._attn_implementation
        set_backend(m,'sdpa');set_lora_input_cast(m,True)
        try:return training_response_accuracy(m,rows,tokenizer,max_new_tokens=config['max_new_tokens'])
        finally:set_backend(m,backend);set_lora_input_cast(m,not variant.get('skip_lora_cast',False))
    if stage=='fit':
        receipt['frozen_before']=frozen_weight_hash(model);receipt['adapter_before']=parameter_hash(model,trainable=True)
        receipt['fit_before']=fit_accuracy(model)
    save();step=0;train_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(int(config['epochs'])):
            sampler=EffectiveBatchLengthSampler(lengths,seed=seed,effective_batch_size=effective);sampler.set_epoch(epoch)
            order=list(sampler)
            for begin in range(0,len(order),effective):
                group=[rows[i] for i in order[begin:begin+effective]]
                denom=sum(sum(t!=-100 for t in r['labels'][1:]) for r in group)
                batches=token_batches(group,variant['token_budget']) if 'token_budget' in variant else (group[i:i+4] for i in range(0,len(group),4))
                value=torch.zeros((),device='cuda');before=time.monotonic()
                for chunk in batches:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=packed_response_loss(model,pack(chunk,'cuda'),denom) if variant['packed'] else response_only_loss(model,{k:v.cuda() for k,v in collate(chunk).items()},denom)
                    loss.backward();value+=loss.detach();del loss
                norm=torch.nn.utils.clip_grad_norm_(params,config['max_grad_norm'])
                if not torch.isfinite(norm) or not torch.isfinite(value):raise ValueError('nonfinite optimization')
                optim.step();scheduler.step();optim.zero_grad(set_to_none=True);step+=1
                item=dict(step=step,examples_seen=epoch*len(rows)+min(begin+effective,len(rows)),loss=float(value),grad_norm=float(norm),seconds=time.monotonic()-before,elapsed_training_seconds=time.monotonic()-train_start)
                receipt['logs'].append(item);save();print(json.dumps(item),flush=True)
                if stage!='fit' and (step%checkpoint_every==0 or step==steps):
                    path=root/f'checkpoint-{step}';model.save_pretrained(path);tokenizer.save_pretrained(path)
                    hashes={p.name:file_hash(p) for p in path.glob('adapter*')}
                    cp=dict(step=step,path=str(path),adapter_files=hashes,elapsed_training_seconds=time.monotonic()-train_start)
                    receipt['checkpoints'].append(cp);save()
                    ready=root/f'checkpoint-{step}.ready.json'
                    temporary=ready.with_suffix('.tmp');temporary.write_text(json.dumps(cp));temporary.replace(ready)
        train_seconds=time.monotonic()-train_start
        model.save_pretrained(root/'adapter-final');tokenizer.save_pretrained(root/'adapter-final')
        if stage=='fit':
            receipt['fit_after']=fit_accuracy(model)
            receipt['frozen_after']=frozen_weight_hash(model);receipt['adapter_after']=parameter_hash(model,trainable=True)
            if receipt['frozen_before']!=receipt['frozen_after'] or receipt['adapter_before']==receipt['adapter_after']:raise ValueError('fit parameter integrity failed')
            # A fresh adapter load on the same pinned frozen backbone checks serialization.
            base=model.unload();model=PeftModel.from_pretrained(base,root/'adapter-final')
            reload=fit_accuracy(model)
            receipt['reload_identical']=all(a['response_token_ids']==b['response_token_ids'] for a,b in zip(receipt['fit_after']['results'],reload['results']))
            if not receipt['reload_identical'] or receipt['fit_after']['accuracy']<.95:raise ValueError('fit/reload gate failed')
        receipt.update(status='completed',train_seconds=train_seconds,elapsed_seconds=time.monotonic()-started,
                       global_step=step,max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                       adapter_files={p.name:file_hash(p) for p in (root/'adapter-final').glob('adapter*')})
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc));raise
    finally:save()
    return receipt
