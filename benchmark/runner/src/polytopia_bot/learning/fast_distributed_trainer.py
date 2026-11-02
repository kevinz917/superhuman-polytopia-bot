"""Distributed token-budget SFT with immutable adapter publication."""
import json,math,time
from pathlib import Path
from .training import verified_data,load_tokenizer,lora_config,parameter_hash,frozen_weight_hash,training_response_accuracy,file_hash
from .data import ResponseCollator
from .efficient_training import EffectiveBatchLengthSampler,response_only_loss
from .fast_training import pack,token_batches,packed_response_loss,configure_checkpointing,set_backend,set_lora_input_cast


def train_distributed(prepared,output,config,*,seed,stage,readiness,benchmark,checkpoint_every=16,continuation=None,probe=None,mixed=None,scale=None):
    import torch,os
    import torch.distributed as dist
    from .distributed_training import partition_effective_batch,sum_gradients
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
    from transformers import AutoModelForCausalLM,set_seed,get_cosine_schedule_with_warmup
    from peft import get_peft_model,PeftModel
    if not benchmark['parity']['passed'] or benchmark['speedup']<=1:
        raise ValueError('passing faster runtime benchmark required')
    variant=benchmark['selected']
    manifest,data=verified_data(prepared,config,stage=stage,readiness=readiness)
    if sum(x is not None for x in (continuation,probe,mixed,scale))>1:raise ValueError('choose one warm-start mode')
    initial=continuation or probe or mixed or scale
    if (stage=='scale40k')!=bool(scale):raise ValueError('40k stage and bounded scale protocol must be used together')
    if scale:
        from .scale_training import verify_scale
        verify_scale(prepared,manifest,scale,config,stage)
    if mixed:
        from .movement_mixed import verify_mixed
        verify_mixed(prepared,manifest,mixed,config,stage)
    epoch_offset=0
    if probe:
        from .movement_probe import verify_subset
        parent=verify_subset(prepared,manifest,probe['parent_run'],config)
        if stage!='fit' or config['epochs']!=20 or config['effective_batch_size']!=8 or probe['checkpoint_steps']!=[80,160,320]:
            raise ValueError('bounded movement probe requires 20 passes and fixed checkpoints')
        if probe['adapter_files']!=parent['adapter_files'] or probe['optimizer_state']!='fresh':
            raise ValueError('probe initial adapter/optimizer differs')
    if continuation:
        parent=json.loads(Path(continuation['parent_run']).read_text())
        if parent['status']!='completed' or parent['config']!=config or parent['data_manifest']!=manifest:
            raise ValueError('continuation must use the same completed training data and configuration')
        if continuation['adapter_files']!=parent['adapter_files'] or config['epochs']!=1 or stage!='smoke':
            raise ValueError('one-epoch smoke continuation required')
        epoch_offset=continuation['epoch_offset']
        if epoch_offset!=1 or continuation.get('optimizer_state')!='fresh':raise ValueError('declared second-epoch optimizer restart required')
    needs_stress=manifest['max_tokens']>variant['stress_max_length']
    if needs_stress and not scale:raise ValueError('longest input untested')
    root=Path(output)
    if rank==0:root.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    set_seed(seed);started=time.monotonic();tokenizer=load_tokenizer(config)
    from .interface import digest
    if digest(tokenizer.backend_tokenizer.to_str())!=manifest['tokenizer_hash'] or digest(tokenizer.chat_template)!=manifest['chat_template_hash']:
        raise ValueError('prepared tokenizer differs')
    rows=data if scale else list(data)
    lengths=[len(r['input_ids']) for r in rows]
    model=AutoModelForCausalLM.from_pretrained(config['model_id'],revision=config['revision'],
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    set_backend(model,variant.get('backend','flash_attention_2' if variant['packed'] else 'sdpa'))
    model.config.use_cache=False
    if initial:
        from .continuation import load_trainable_adapter
        model=load_trainable_adapter(model,initial['adapter'],config,initial['adapter_files'])
    else:model=get_peft_model(model,lora_config(config))
    model.train()
    configure_checkpointing(model,variant['checkpointing'])
    set_lora_input_cast(model,not variant.get('skip_lora_cast',False))
    if any(p.requires_grad and 'lora_' not in n for n,p in model.named_parameters()):raise ValueError('base trainability changed')
    params=[p for p in model.parameters() if p.requires_grad]
    collate=ResponseCollator(tokenizer.pad_token_id);effective=config['effective_batch_size']
    from .scale_training import batch_ranges
    ranges=batch_ranges(len(rows),effective,scale['checkpoint_examples'] if scale else [len(rows)])
    steps=len(ranges)*int(config['epochs'])
    if config['epochs']!=int(config['epochs']):raise ValueError('integer epochs required')
    optim=torch.optim.AdamW(params,lr=config['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0)
    scheduler=get_cosine_schedule_with_warmup(optim,math.ceil(steps*config['warmup_ratio']),steps)
    receipt=dict(schema='fast-training-v2',status='training',stage=stage,config=config,seed=seed,
        data_manifest=manifest,variant=variant,hardware=torch.cuda.get_device_name(),world_size=world,planned_steps=steps,
        logs=[],checkpoints=[],trainable_parameters=sum(p.numel() for p in params))
    if initial:
        receipt.update(**({'continuation':continuation} if continuation else {'probe':probe} if probe else {'mixed':mixed} if mixed else {'scale':scale}),parent_run_sha256=file_hash(initial['parent_run']),
                       loaded_adapter_exact=True,optimizer_state='fresh AdamW; same peak LR, restarted 3% warmup/cosine schedule',epoch_offset=epoch_offset)
        if rank==0:
            receipt['frozen_before']=frozen_weight_hash(model)
            receipt['adapter_before']=parameter_hash(model,trainable=True)
    def save():
        if rank!=0:return
        tmp=root/'run.json.tmp';tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(root/'run.json')
    def fit_accuracy(m):
        backend=m.get_base_model().config._attn_implementation
        set_backend(m,'sdpa');set_lora_input_cast(m,True)
        try:return training_response_accuracy(m,rows,tokenizer,max_new_tokens=config['max_new_tokens'])
        finally:set_backend(m,backend);set_lora_input_cast(m,not variant.get('skip_lora_cast',False))
    if needs_stress:
        if not variant['packed']:raise ValueError('scale stress expects qualified packed runtime')
        stress=[rows[i] for i in sorted(range(len(rows)),key=lambda i:lengths[i])[-8:]]
        denominator=sum(sum(t!=-100 for t in r['labels'][1:]) for r in stress)
        torch.cuda.reset_peak_memory_stats()
        for chunk in token_batches(partition_effective_batch(stress,world)[rank],variant['token_budget']):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=packed_response_loss(model,pack(chunk,'cuda'),denominator)
            loss.backward();del loss
        sum_gradients(params)
        norm=torch.nn.utils.clip_grad_norm_(params,float('inf'))
        if not torch.isfinite(norm):raise ValueError('long-input stress produced nonfinite gradients')
        optim.zero_grad(set_to_none=True)
        peak=torch.tensor(torch.cuda.max_memory_allocated(),device='cuda');dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        receipt['corpus_stress']=dict(passed=True,examples=8,max_tokens=max(lengths),peak_bytes=int(peak),optimizer_updates=0)
    if stage=='fit' and not probe and rank==0:
        receipt['frozen_before']=frozen_weight_hash(model);receipt['adapter_before']=parameter_hash(model,trainable=True)
        receipt['fit_before']=fit_accuracy(model)
    save();dist.barrier();step=0;train_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(int(config['epochs'])):
            sampler=EffectiveBatchLengthSampler(lengths,seed=seed,effective_batch_size=effective);sampler.set_epoch(epoch+epoch_offset)
            order=list(sampler)
            if sorted(order)!=list(range(len(rows))):raise ValueError('epoch does not cover every example exactly once')
            if initial:receipt['epoch_order_sha256']=digest(order)
            if probe:receipt.setdefault('epoch_order_hashes',[]).append(digest(order))
            for begin,end in ranges:
                group=[rows[i] for i in order[begin:end]]
                denom=sum(sum(t!=-100 for t in r['labels'][1:]) for r in group)
                local_group=partition_effective_batch(group,world)[rank]
                batches=token_batches(local_group,variant['token_budget']) if 'token_budget' in variant else (local_group[i:i+4] for i in range(0,len(local_group),4))
                value=torch.zeros((),device='cuda');before=time.monotonic()
                for chunk in batches:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=packed_response_loss(model,pack(chunk,'cuda'),denom) if variant['packed'] else response_only_loss(model,{k:v.cuda() for k,v in collate(chunk).items()},denom)
                    loss.backward();value+=loss.detach();del loss
                sum_gradients(params);dist.all_reduce(value,op=dist.ReduceOp.SUM)
                norm=torch.nn.utils.clip_grad_norm_(params,config['max_grad_norm'])
                if not torch.isfinite(norm) or not torch.isfinite(value):raise ValueError('nonfinite optimization')
                optim.step();scheduler.step();optim.zero_grad(set_to_none=True);step+=1
                item=dict(step=step,examples_seen=epoch*len(rows)+end,loss=float(value),grad_norm=float(norm),learning_rate=scheduler.get_last_lr()[0],seconds=time.monotonic()-before,elapsed_training_seconds=time.monotonic()-train_start)
                receipt['logs'].append(item);save()
                if rank==0:print(json.dumps(item),flush=True)
                publish=(step in initial['checkpoint_steps']) if initial else (step%checkpoint_every==0 or step==steps)
                if rank==0 and (stage!='fit' or probe) and publish:
                    path=root/f'checkpoint-{step}';model.save_pretrained(path);tokenizer.save_pretrained(path)
                    hashes={p.name:file_hash(p) for p in path.glob('adapter*')}
                    cp=dict(step=step,examples_seen=epoch*len(rows)+end,path=str(path),adapter_files=hashes,elapsed_training_seconds=time.monotonic()-train_start)
                    receipt['checkpoints'].append(cp);save()
                    ready=root/f'checkpoint-{step}.ready.json'
                    temporary=ready.with_suffix('.tmp');temporary.write_text(json.dumps(cp));temporary.replace(ready)
        train_seconds=time.monotonic()-train_start
        if rank==0:model.save_pretrained(root/'adapter-final');tokenizer.save_pretrained(root/'adapter-final')
        dist.barrier()
        if initial and rank==0:
            receipt['frozen_after']=frozen_weight_hash(model);receipt['adapter_after']=parameter_hash(model,trainable=True)
            if receipt['frozen_before']!=receipt['frozen_after'] or receipt['adapter_before']==receipt['adapter_after']:
                raise ValueError('continuation parameter integrity failed')
            torch.save(dict(optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),seed=seed,
                            completed_epochs=epoch_offset+int(config['epochs']),data_manifest_hash=digest(manifest)),root/'optimizer-final.pt')
            receipt['optimizer_state_sha256']=file_hash(root/'optimizer-final.pt')
        if stage=='fit' and not probe and rank==0:
            receipt['fit_after']=fit_accuracy(model)
            receipt['frozen_after']=frozen_weight_hash(model);receipt['adapter_after']=parameter_hash(model,trainable=True)
            if receipt['frozen_before']!=receipt['frozen_after'] or receipt['adapter_before']==receipt['adapter_after']:raise ValueError('fit parameter integrity failed')
            # A fresh adapter load on the same pinned frozen backbone checks serialization.
            base=model.unload();model=PeftModel.from_pretrained(base,root/'adapter-final')
            reload=fit_accuracy(model)
            receipt['reload_identical']=all(a['response_token_ids']==b['response_token_ids'] for a,b in zip(receipt['fit_after']['results'],reload['results']))
            if not receipt['reload_identical'] or receipt['fit_after']['accuracy']<.95:raise ValueError('fit/reload gate failed')
        dist.barrier()
        receipt.update(status='completed',train_seconds=train_seconds,elapsed_seconds=time.monotonic()-started,
                       global_step=step,max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                       adapter_files={p.name:file_hash(p) for p in (root/'adapter-final').glob('adapter*')})
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc));raise
    finally:save();dist.destroy_process_group()
    return receipt
