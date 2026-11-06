"""Versioned padding-free response loss; preserves effective-batch membership."""
import itertools


def token_batches(rows, token_budget):
    if token_budget < 1:
        raise ValueError('positive token budget required')
    batch, size = [], 0
    for row in sorted(rows, key=lambda r: len(r['input_ids']), reverse=True):
        n = len(row['input_ids'])
        if batch and size + n > token_budget:
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += n
    if batch:
        yield batch


def pack(rows, device='cpu'):
    import torch
    if not rows:
        raise ValueError('nonempty batch required')
    lengths = [len(r['input_ids']) for r in rows]
    for r, n in zip(rows, lengths):
        if n < 2 or len(r['labels']) != n or r['attention_mask'] != [1]*n:
            raise ValueError('unpadded aligned input required')
        if r['labels'][0] != -100 or not any(x != -100 for x in r['labels'][1:]):
            raise ValueError('each example requires ignored first token and supervised response')
    flat = lambda k: list(itertools.chain.from_iterable(r[k] for r in rows))
    positions = list(itertools.chain.from_iterable(range(n) for n in lengths))
    cu = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device=device)
    labels = torch.tensor([flat('labels')], dtype=torch.long, device=device)
    # Every new sequence starts with an ignored label, so the previous
    # sequence's final hidden state can never predict across a boundary.
    return dict(input_ids=torch.tensor([flat('input_ids')], dtype=torch.long, device=device),
                labels=labels, position_ids=torch.tensor([positions], device=device),
                cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                max_length_q=max(lengths), max_length_k=max(lengths), segment_lengths=lengths)


def packed_response_loss(model, inputs, denominator=None):
    import torch
    import torch.nn.functional as F
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if base.config.model_type != 'qwen3' or base.config._attn_implementation not in ('flash_attention_2','segmented_sdpa','segmented_cudnn'):
        raise ValueError('validated Qwen3 flash_attention_2 backend required')
    labels = inputs['labels'][:, 1:]
    selected = labels != -100
    hidden = base.model(**{k:v for k,v in inputs.items() if k != 'labels'},
                        attention_mask={'full_attention':None}, use_cache=False, return_dict=True).last_hidden_state
    logits = base.lm_head(hidden[:, :-1][selected]).float()
    denom = selected.sum() if denominator is None else denominator
    return F.cross_entropy(logits, labels[selected], reduction='sum') / denom


def configure_checkpointing(model, enabled):
    if enabled:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        if enabled == 'alternating':
            base=model.get_base_model() if hasattr(model,'get_base_model') else model
            for i,layer in enumerate(base.model.layers):layer.gradient_checkpointing=i%2==0
    else:
        model.gradient_checkpointing_disable()


def segmented_sdpa(module, query, key, value, attention_mask, dropout=0., scaling=None, segment_lengths=None, **kwargs):
    """Unpadded projections/MLP with separate fused SDPA calls per example.

    Keeps the production SDPA numerics while enforcing block-diagonal attention.
    Segment lengths are CPU integers prepared by the collator, not GPU reads.
    """
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if not segment_lengths or sum(segment_lengths)!=query.shape[2] or query.shape[0]!=1:
        raise ValueError('invalid packed attention segments')
    outputs=[];start=0
    for length in segment_lengths:
        stop=start+length
        output,_=sdpa_attention_forward(module,query[:,:,start:stop],key[:,:,start:stop],value[:,:,start:stop],None,
            dropout=dropout,scaling=scaling,is_causal=True)
        outputs.append(output);start=stop
    return torch.cat(outputs,dim=1),None


CUDNN_BUCKET=256


def segmented_cudnn(module, query, key, value, attention_mask, dropout=0., scaling=None, segment_lengths=None, **kwargs):
    """Per-example causal SDPA with each segment zero-padded to a multiple of CUDNN_BUCKET tokens.

    cuDNN builds an execution plan per new shape (~1.3 s each); bucketing bounds the shapes a run sees.
    Padding is exact under causal attention: real queries never attend to the padded tail, and padded
    outputs are dropped. The backend itself is forced by ``attention_context`` around forward and backward.
    """
    import torch
    import torch.nn.functional as F
    if not segment_lengths or sum(segment_lengths)!=query.shape[2] or query.shape[0]!=1:
        raise ValueError('invalid packed attention segments')
    if dropout:
        raise ValueError('attention dropout unsupported')
    gqa=key.shape[1]!=query.shape[1];outputs=[];start=0
    for length in segment_lengths:
        q,k,v=(t[:,:,start:start+length] for t in (query,key,value))
        padded=-(-length//CUDNN_BUCKET)*CUDNN_BUCKET
        if padded>length:q,k,v=(F.pad(t,(0,0,0,padded-length)) for t in (q,k,v))
        else:q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scaling,enable_gqa=gqa)
        outputs.append(o[:,:,:length]);start+=length
    return torch.cat(outputs,dim=2).transpose(1,2).contiguous(),None


def attention_context(backend):
    """Force cuDNN SDPA for segmented_cudnn; wrap each chunk's forward and backward (checkpoint recompute)."""
    import contextlib
    if backend!='segmented_cudnn':return contextlib.nullcontext()
    from torch.nn.attention import sdpa_kernel,SDPBackend
    return sdpa_kernel([SDPBackend.CUDNN_ATTENTION])


def prewarm_cudnn(model, max_length):
    """Build the cuDNN forward+backward plan for every bucket <= max_length once, before the first chunk."""
    import torch
    import torch.nn.functional as F
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    c=base.config;heads,kv,dim=c.num_attention_heads,c.num_key_value_heads,c.head_dim
    buckets=list(range(CUDNN_BUCKET,-(-max_length//CUDNN_BUCKET)*CUDNN_BUCKET+1,CUDNN_BUCKET))
    with attention_context('segmented_cudnn'):
        for m in buckets:
            q=torch.randn(1,heads,m,dim,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            k=torch.randn(1,kv,m,dim,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            F.scaled_dot_product_attention(q,k,k,is_causal=True,enable_gqa=kv!=heads).sum().backward()
    torch.cuda.synchronize()
    return len(buckets)


def chunk_checkpointing(variant, tokens):
    """Checkpointing mode for one packed chunk; 'adaptive' fully checkpoints chunks above adaptive_tokens."""
    mode=variant['checkpointing']
    return (tokens>variant['adaptive_tokens']) if mode=='adaptive' else mode


def apply_liger_swiglu(model):
    """Fused Liger SiLU*mul in every decoder MLP of this model instance only.

    RoPE and RMSNorm stay unfused (they move further from an fp32 reference than the qualified path).
    Liger also rebinds module-level Qwen3 classes; they are restored so later models are unaffected.
    """
    import importlib.metadata
    import transformers.models.qwen3.modeling_qwen3 as mq
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    saved={k:getattr(mq,k) for k in ('Qwen3MLP','Qwen3RMSNorm','apply_rotary_pos_emb')}
    try:
        apply_liger_kernel_to_qwen3(rope=False,rms_norm=False,swiglu=True,cross_entropy=False,
                                    fused_linear_cross_entropy=False,model=base)
    finally:
        for k,x in saved.items():setattr(mq,k,x)
    return importlib.metadata.version('liger-kernel')


def set_backend(model, backend):
    from transformers import AttentionInterface
    AttentionInterface.register('segmented_sdpa',segmented_sdpa)
    AttentionInterface.register('segmented_cudnn',segmented_cudnn)
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    base.set_attn_implementation(backend)


def set_lora_input_cast(model, enabled):
    """Autocast already converts operands; avoid redundant BF16→FP32→BF16 inputs.

    Keep adapter weights/master optimizer state in FP32. Re-enable for generation
    outside autocast, where the adapter's FP32 matmuls need matching input dtype.
    """
    from peft.tuners.lora.layer import LoraLayer
    for layer in model.modules():
        if isinstance(layer,LoraLayer):layer.cast_input_dtype_enabled=enabled
