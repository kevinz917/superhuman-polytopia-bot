"""Audited SFT runner using the benchmarked response-only loss and batching."""
import importlib.metadata
import json
import os
from pathlib import Path
import time

from .data import ResponseCollator
from .efficient_training import trainer_class
from .interface import digest
from .training import (verified_data, load_tokenizer, lora_config, file_hash,
    checkpoint_callback, training_response_accuracy, frozen_weight_hash, parameter_hash)


def train_efficient(data_path, output, config, *, seed, stage, readiness, benchmark):
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, TrainingArguments, set_seed
    from huggingface_hub import snapshot_download

    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        raise ValueError('single-GPU runtime only')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required')
    profile = benchmark['benchmark']
    selected = profile['selected']
    if benchmark.get('exit_code') != 0 or not profile['parity']['passed'] or selected['status'] != 'passed':
        raise ValueError('completed passing runtime benchmark required')
    if config['microbatch_size'] > selected['microbatch'] or config['effective_batch_size'] % config['microbatch_size']:
        raise ValueError('microbatch outside measured range or effective batch')
    manifest, rows = verified_data(data_path,config,stage=stage,readiness=readiness)
    if manifest['max_tokens'] > selected['stress_max_length']:
        raise ValueError('new longest examples need a memory stress test')
    if stage=='main' and seed not in config['seeds']:
        raise ValueError('unregistered main seed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);set_seed(seed)
    started=time.monotonic()
    receipt=dict(status='initializing',stage=stage,seed=seed,config=config,data_manifest=manifest,
        readiness=readiness,hardware=torch.cuda.get_device_name(),cuda=torch.version.cuda,
        source_hashes={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},
        packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','accelerate')},
        runtime=dict(version='response-only-length-grouped-v1',microbatch=config['microbatch_size'],
            effective_batch=config['effective_batch_size'],benchmark_speedup=profile['measured_speedup'],
            length_order='Random permutation; reorder only inside each effective batch',
            loss='Causal response-token sum / total supervised tokens in effective batch'))
    def save(): (output/'run.json').write_text(json.dumps(receipt,indent=2))
    save()
    try:
        tokenizer=load_tokenizer(config)
        if digest(tokenizer.chat_template)!=manifest['chat_template_hash'] or digest(tokenizer.backend_tokenizer.to_str())!=manifest['tokenizer_hash']:
            raise ValueError('prepared tokenizer mismatch')
        model=AutoModelForCausalLM.from_pretrained(config['model_id'],revision=config['revision'],
            torch_dtype=torch.bfloat16,trust_remote_code=False,attn_implementation='sdpa')
        snapshot=Path(snapshot_download(config['model_id'],revision=config['revision'],local_files_only=True))
        receipt['base_files']={str(p.relative_to(snapshot)):file_hash(p) for p in snapshot.rglob('*') if p.is_file() and p.suffix in ('.json','.safetensors','.bin','.txt','.jinja')}
        model.config.use_cache=False;model=get_peft_model(model,lora_config(config))
        if any(p.requires_grad and 'lora_' not in n for n,p in model.named_parameters()):
            raise ValueError('unexpected trainable base parameters')
        receipt['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        lengths=[len(row['input_ids']) for row in rows]
        args=TrainingArguments(output_dir=str(output),per_device_train_batch_size=config['microbatch_size'],
            gradient_accumulation_steps=config['effective_batch_size']//config['microbatch_size'],
            learning_rate=config['learning_rate'],num_train_epochs=config['epochs'],warmup_ratio=config['warmup_ratio'],
            lr_scheduler_type='cosine',max_grad_norm=config['max_grad_norm'],weight_decay=0,optim='adamw_torch',
            bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
            save_strategy='no',logging_steps=1,logging_nan_inf_filter=False,report_to='none',seed=seed,data_seed=seed,
            remove_unused_columns=False,dataloader_num_workers=0)
        trainer=trainer_class()(model=model,args=args,train_dataset=rows,data_collator=ResponseCollator(tokenizer.pad_token_id),
            length_order=lengths,effective_batch_size=config['effective_batch_size'],callbacks=[checkpoint_callback()])
        if stage=='fit':
            receipt['frozen_weights_before']=frozen_weight_hash(model)
            receipt['adapter_weights_before']=parameter_hash(model,trainable=True)
            receipt['fit_before']=training_response_accuracy(model,rows,tokenizer,max_new_tokens=config['max_new_tokens'])
            save()
        result=trainer.train();trainer.save_model(str(output/'adapter-final'));tokenizer.save_pretrained(output/'adapter-final');trainer.save_state()
        if stage=='fit':
            receipt['fit_after']=training_response_accuracy(model,rows,tokenizer,max_new_tokens=config['max_new_tokens'])
            receipt['frozen_weights_after']=frozen_weight_hash(model);receipt['adapter_weights_after']=parameter_hash(model,trainable=True)
            if receipt['frozen_weights_before']!=receipt['frozen_weights_after'] or receipt['adapter_weights_before']==receipt['adapter_weights_after']:
                raise ValueError('parameter update integrity failed')
        receipt.update(status='completed',metrics=result.metrics,elapsed_seconds=time.monotonic()-started,
            max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
            adapter_files={p.name:file_hash(p) for p in (output/'adapter-final').glob('adapter*') if p.is_file()})
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        raise
    finally:save()
    return receipt
