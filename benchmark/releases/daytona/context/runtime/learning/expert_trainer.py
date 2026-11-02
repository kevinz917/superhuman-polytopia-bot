"""Distributed LoRA SFT on an RL-expert corpus (separately versioned from SFT-0).

Same optimizer, LoRA, effective batch, packing runtime and response-only loss as
``fast_distributed_trainer``. Only the data gate differs: the corpus is the
``polytopia-expert-prepared-v1`` schema produced by ``scripts/prepare_expert_sft.py``
instead of the frozen teacher corpus. Rows stay on disk (offset-indexed).
"""
import hashlib
import json
import math
import time
from pathlib import Path

from .data import JsonlDataset
from .efficient_training import EffectiveBatchLengthSampler
from .fast_training import (pack, token_batches, packed_response_loss, configure_checkpointing, set_backend, set_lora_input_cast,
                            apply_liger_swiglu, attention_context, prewarm_cudnn, chunk_checkpointing)
from .training import load_tokenizer, lora_config, parameter_hash, frozen_weight_hash, file_hash


def verify_expert_corpus(prepared, config):
    from .interface import digest
    prepared = Path(prepared)
    manifest = json.loads((prepared/'manifest.json').read_text())
    if manifest['schema'] != 'polytopia-expert-prepared-v1' or manifest['config_hash'] != digest(config):
        raise ValueError('prepared expert data/config mismatch')
    if manifest['sources'] != ['rl_expert']:
        raise ValueError('expert corpus must contain only RL-expert labels')
    sha = hashlib.sha256()
    count, lengths = 0, []
    with open(prepared/'tokens.jsonl', 'rb') as f:
        for line in f:
            sha.update(line)
            row = json.loads(line)
            ids, labels, mask = row['input_ids'], row['labels'], row['attention_mask']
            if not ids or len(ids) != len(labels) or len(ids) != len(mask) or len(ids) > config['max_length']:
                raise ValueError('invalid prepared lengths')
            supervised = [i for i, x in enumerate(labels) if x != -100]
            if not supervised or supervised[0] == 0 or supervised != list(range(supervised[0], len(ids))):
                raise ValueError('labels must be one response-only suffix')
            if len(supervised) > config['max_new_tokens'] or any(labels[i] != ids[i] for i in supervised) or any(x != 1 for x in mask):
                raise ValueError('invalid prepared mask')
            count += 1; lengths.append(len(ids))
    if sha.hexdigest() != manifest['tokens_sha256'] or count != manifest['records']:
        raise ValueError('prepared tokens changed after preparation')
    return manifest, lengths


TRAINER_STATE = 'trainer_state.pt'      # optimizer, scheduler, per-rank RNG, epoch/offset (next to the adapter)
TRAINER_STATE_META = 'trainer_state.json'


def run_identity(config, seed, manifest, variant, world):
    """What a resumed run must share with the checkpoint it continues."""
    from .interface import digest
    return dict(config_hash=digest(config), seed=seed, world_size=world, variant_hash=digest(variant),
                corpus=dict(tokens_sha256=manifest['tokens_sha256'], config_hash=manifest['config_hash'], records=manifest['records']))


def rng_state():
    import random, numpy as np, torch
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return dict(python=random.getstate(), numpy=(name, keys.tolist(), pos, has_gauss, cached),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def set_rng_state(s):
    import random, numpy as np, torch
    random.setstate(s['python'])
    name, keys, pos, has_gauss, cached = s['numpy']
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(s['torch']); torch.cuda.set_rng_state(s['cuda'])


def resumable_checkpoint(path):
    """Metadata of a checkpoint that carries trainer state, or None (adapter-only / incomplete checkpoint)."""
    path = Path(path)
    meta = path/TRAINER_STATE_META
    if not (path.parent/f'{path.name}.ready.json').exists() or not meta.exists() or not (path/TRAINER_STATE).exists():
        return None
    return json.loads(meta.read_text())


BACKENDS = ('flash_attention_2', 'segmented_sdpa', 'segmented_cudnn')
PARITY_GATES = ('strict', 'revised')


def check_variant(variant):
    if not variant['packed']:
        raise ValueError('expert SFT expects the qualified packed runtime')
    if variant.get('backend', 'flash_attention_2') not in BACKENDS:
        raise ValueError('unqualified attention backend')
    if variant['checkpointing'] not in (False, True, 'alternating', 'adaptive'):
        raise ValueError('unknown checkpointing mode')
    if variant['checkpointing'] == 'adaptive' and not (isinstance(variant.get('adaptive_tokens'), int) and variant['adaptive_tokens'] > 0):
        raise ValueError('adaptive checkpointing needs a positive adaptive_tokens threshold')
    if not (isinstance(variant['token_budget'], int) and variant['token_budget'] > 0):
        raise ValueError('positive token budget required')
    return variant


def check_benchmark(benchmark, parity_gate='strict'):
    """The receipt's selected variant, if it passed the named parity gate and is faster.

    'strict': same-batch bf16 parity against the qualified path (the original gate; every receipt has it).
    'revised': fp32-anchored loss check plus a training-trajectory check against the noise floor
    (docs/sft-trainer-benchmark-2026-10-19.md section 3); only qualification-v1 receipts carry it.
    """
    if parity_gate not in PARITY_GATES:
        raise ValueError(f'parity gate must be one of {PARITY_GATES}')
    passed = (benchmark['parity']['passed'] if parity_gate == 'strict' else
              benchmark.get('gates', {}).get('revised', {}).get('passed') is True)
    if not passed or benchmark['speedup'] <= 1:
        raise ValueError(f'passing faster runtime benchmark required ({parity_gate} parity gate)')
    return check_variant(benchmark['selected'])


def build_model(config, variant, precision='bf16'):
    """Base + LoRA with the variant's backend, Liger SwiGLU, checkpointing, LoRA cast and cuDNN plans.

    Call set_seed first: the LoRA initialization draws from the global RNG. Returns (model, info).
    """
    import torch
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model
    model = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['revision'],
        torch_dtype=torch.float32 if precision == 'fp32' else torch.bfloat16, attn_implementation='sdpa').cuda()
    set_backend(model, variant.get('backend', 'flash_attention_2'))
    model.config.use_cache = False
    model = get_peft_model(model, lora_config(config)); model.train(); info = {}
    if variant.get('liger_swiglu'):
        info['liger_kernel'] = apply_liger_swiglu(model)
    configure_checkpointing(model, chunk_checkpointing(variant, 0))
    set_lora_input_cast(model, not variant.get('skip_lora_cast', False))
    if any(p.requires_grad and 'lora_' not in n for n, p in model.named_parameters()):
        raise ValueError('base trainability changed')
    if variant.get('backend') == 'segmented_cudnn':
        started = time.monotonic()
        info['cudnn_prewarm'] = dict(buckets=prewarm_cudnn(model, config['max_length']), max_length=config['max_length'],
                                     seconds=time.monotonic()-started)
    return model, info


class ChunkStep:
    """Forward + backward of one packed chunk under the variant's attention backend and checkpointing."""

    def __init__(self, model, variant, autocast=True):
        self.model, self.variant, self.autocast = model, variant, autocast
        self.mode = chunk_checkpointing(variant, 0)

    def __call__(self, chunk, denominator):
        import torch
        mode = chunk_checkpointing(self.variant, sum(len(r['input_ids']) for r in chunk))
        if mode != self.mode:
            configure_checkpointing(self.model, mode); self.mode = mode
        with attention_context(self.variant.get('backend')):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.autocast):
                loss = packed_response_loss(self.model, pack(chunk, 'cuda'), denominator)
            loss.backward()
        return loss.detach()


def train_expert(prepared, output, config, *, seed, benchmark, checkpoint_every, resume_from=None, parity_gate='strict'):
    import torch, os
    import torch.distributed as dist
    from .distributed_training import partition_effective_batch, sum_gradients
    from .interface import digest
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK'])); dist.init_process_group('nccl')
    from transformers import set_seed, get_cosine_schedule_with_warmup
    variant = check_benchmark(benchmark, parity_gate)
    manifest, lengths = verify_expert_corpus(prepared, config)
    rows = JsonlDataset(Path(prepared)/'tokens.jsonl')
    if len(rows) != len(lengths):
        raise ValueError('prepared record count mismatch')
    identity = run_identity(config, seed, manifest, variant, world)
    if resume_from is not None:
        # Resume: the checkpoint must come from the same corpus/config/seed/runtime/world size and be intact.
        resume_from = Path(resume_from)
        meta = resumable_checkpoint(resume_from)
        if meta is None:
            raise ValueError(f'{resume_from} is not a complete checkpoint with trainer state')
        if meta['identity'] != identity:
            raise ValueError(f'checkpoint identity differs: {meta["identity"]} != {identity}')
        ready = json.loads((resume_from.parent/f'{resume_from.name}.ready.json').read_text())
        previous = json.loads((resume_from.parent/'run.json').read_text())
        recorded = [c for c in previous['checkpoints'] if c['step'] == meta['step']]
        adapter_files = {p.name: file_hash(p) for p in resume_from.glob('adapter*')}
        if not (adapter_files == meta['adapter_files'] == ready['adapter_files'] == (recorded or [{}])[-1].get('adapter_files')):
            raise ValueError('checkpoint adapter files changed after they were saved')
        if file_hash(resume_from/TRAINER_STATE) != meta['state_sha256']:
            raise ValueError('trainer state changed after it was saved')
        if [x['step'] for x in previous['logs'][:meta['step']]] != list(range(1, meta['step']+1)):
            raise ValueError('previous run.json lacks the logs up to the checkpoint')
    root = Path(output)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    set_seed(seed); started = time.monotonic(); tokenizer = load_tokenizer(config)
    if digest(tokenizer.backend_tokenizer.to_str()) != manifest['tokenizer_hash'] or digest(tokenizer.chat_template) != manifest['chat_template_hash']:
        raise ValueError('prepared tokenizer differs')
    model, runtime = build_model(config, variant); step_chunk = ChunkStep(model, variant)
    if resume_from is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        loaded = set_peft_model_state_dict(model, load_file(resume_from/'adapter_model.safetensors'))
        if loaded.unexpected_keys:
            raise ValueError(f'checkpoint has unexpected adapter keys: {loaded.unexpected_keys[:5]}')
        if parameter_hash(model, trainable=True) != meta['adapter_parameter_hash']:
            raise ValueError('loaded adapter differs from the checkpointed parameters')
    params = [p for p in model.parameters() if p.requires_grad]
    effective = config['effective_batch_size']
    if config['epochs'] != int(config['epochs']):
        raise ValueError('integer epochs required')
    batches_per_epoch = math.ceil(len(rows)/effective)
    steps = batches_per_epoch*int(config['epochs'])
    optim = torch.optim.AdamW(params, lr=config['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optim, math.ceil(steps*config['warmup_ratio']), steps)
    step, start_epoch, start_offset, elapsed_before = 0, 0, 0, 0
    if resume_from is None:
        receipt = dict(schema='expert-training-v1', status='training', config=config, seed=seed, data_manifest=manifest,
            variant=variant, parity_gate=parity_gate, runtime=runtime, hardware=torch.cuda.get_device_name(),
            world_size=world, planned_steps=steps,
            logs=[], checkpoints=[], trainable_parameters=sum(p.numel() for p in params))
        if rank == 0:
            receipt['frozen_before'] = frozen_weight_hash(model); receipt['adapter_before'] = parameter_hash(model, trainable=True)
    else:
        state = torch.load(resume_from/TRAINER_STATE, map_location='cpu', weights_only=True)
        if (state['step'], state['epoch'], state['offset']) != (meta['step'], meta['epoch'], meta['offset']):
            raise ValueError('trainer state and its metadata disagree')
        optim.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        if scheduler.last_epoch != meta['step'] or previous['planned_steps'] != steps or len(state['rng']) != world:
            raise ValueError('restored schedule does not match the checkpoint step')
        step, start_epoch, start_offset = meta['step'], meta['epoch'], meta['offset']
        elapsed_before = meta['elapsed_training_seconds']
        receipt = {k: v for k, v in previous.items() if k not in ('error', 'train_seconds', 'elapsed_seconds', 'global_step',
                   'max_gpu_memory_bytes', 'adapter_files', 'frozen_after', 'adapter_after')}
        receipt.update(status='training', logs=previous['logs'][:step],
                       checkpoints=[c for c in previous['checkpoints'] if c['step'] <= step],
                       epoch_order_hashes=previous.get('epoch_order_hashes', [])[:start_epoch])
        receipt.setdefault('resumes', []).append(dict(
            from_checkpoint=str(resume_from), step=step, epoch=start_epoch, offset=start_offset, hardware=torch.cuda.get_device_name(),
            previous_status=previous['status'], previous_error=previous.get('error'), previous_logged_steps=len(previous['logs']),
            previous_corpus_stress=previous.get('corpus_stress'), parity_gate=parity_gate, runtime=runtime,
            started_unix=time.time()))
        if rank == 0 and frozen_weight_hash(model) != previous['frozen_before']:
            raise ValueError('frozen base weights differ from the run being resumed')

    def save():
        if rank != 0:
            return
        tmp = root/'run.json.tmp'; tmp.write_text(json.dumps(receipt, indent=2)); tmp.replace(root/'run.json')

    # Always stress the longest inputs before the first update (readable prompts are long).
    stress = [rows[i] for i in sorted(range(len(rows)), key=lambda i: lengths[i])[-8:]]
    denominator = sum(sum(t != -100 for t in r['labels'][1:]) for r in stress)
    torch.cuda.reset_peak_memory_stats()
    for chunk in token_batches(partition_effective_batch(stress, world)[rank], variant['token_budget']):
        step_chunk(chunk, denominator)
    sum_gradients(params)
    norm = torch.nn.utils.clip_grad_norm_(params, float('inf'))
    if not torch.isfinite(norm):
        raise ValueError('long-input stress produced nonfinite gradients')
    optim.zero_grad(set_to_none=True)
    peak = torch.tensor(torch.cuda.max_memory_allocated(), device='cuda'); dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    receipt['corpus_stress'] = dict(passed=True, examples=8, max_tokens=max(lengths), peak_bytes=int(peak), optimizer_updates=0)
    save(); dist.barrier(); train_start = time.monotonic()-elapsed_before; torch.cuda.reset_peak_memory_stats()
    if resume_from is not None:
        set_rng_state(state['rng'][rank]); del state
    try:
        for epoch in range(start_epoch, int(config['epochs'])):
            sampler = EffectiveBatchLengthSampler(lengths, seed=seed, effective_batch_size=effective); sampler.set_epoch(epoch)
            order = list(sampler)
            if sorted(order) != list(range(len(rows))):
                raise ValueError('epoch does not cover every example exactly once')
            if resume_from is not None and epoch < len(previous.get('epoch_order_hashes', [])) and \
                    previous['epoch_order_hashes'][epoch] != digest(order):
                raise ValueError('resumed epoch order differs from the checkpointed run')
            receipt.setdefault('epoch_order_hashes', []).append(digest(order))
            for begin in range(start_offset if epoch == start_epoch else 0, len(rows), effective):
                group = [rows[i] for i in order[begin:begin+effective]]
                denom = sum(sum(t != -100 for t in r['labels'][1:]) for r in group)
                local_group = partition_effective_batch(group, world)[rank]
                value = torch.zeros((), device='cuda'); before = time.monotonic()
                for chunk in token_batches(local_group, variant['token_budget']):
                    value += step_chunk(chunk, denom)
                sum_gradients(params); dist.all_reduce(value, op=dist.ReduceOp.SUM)
                norm = torch.nn.utils.clip_grad_norm_(params, config['max_grad_norm'])
                if not torch.isfinite(norm) or not torch.isfinite(value):
                    raise ValueError('nonfinite optimization')
                optim.step(); scheduler.step(); optim.zero_grad(set_to_none=True); step += 1
                item = dict(step=step, examples_seen=epoch*len(rows)+min(begin+effective, len(rows)), loss=float(value),
                            grad_norm=float(norm), learning_rate=scheduler.get_last_lr()[0], seconds=time.monotonic()-before,
                            elapsed_training_seconds=time.monotonic()-train_start)
                receipt['logs'].append(item); save()
                if rank == 0:
                    print(json.dumps(item), flush=True)
                if step % checkpoint_every == 0 or step == steps:
                    rngs = [None]*world; dist.all_gather_object(rngs, rng_state())
                if rank == 0 and (step % checkpoint_every == 0 or step == steps):
                    path = root/f'checkpoint-{step}'; model.save_pretrained(path); tokenizer.save_pretrained(path)
                    cp = dict(step=step, path=str(path), adapter_files={p.name: file_hash(p) for p in path.glob('adapter*')},
                              elapsed_training_seconds=time.monotonic()-train_start)
                    # Trainer state next to the adapter; the .ready marker below is written only after it.
                    nxt = begin+effective
                    position = dict(step=step, epoch=epoch if nxt < len(rows) else epoch+1, offset=nxt if nxt < len(rows) else 0)
                    tmp = path/f'{TRAINER_STATE}.tmp'
                    torch.save(dict(position, optimizer=optim.state_dict(), scheduler=scheduler.state_dict(), rng=rngs), tmp)
                    tmp.replace(path/TRAINER_STATE)
                    meta = dict(position, schema='expert-trainer-state-v1', identity=identity, state_sha256=file_hash(path/TRAINER_STATE),
                                adapter_files=cp['adapter_files'], adapter_parameter_hash=parameter_hash(model, trainable=True),
                                elapsed_training_seconds=cp['elapsed_training_seconds'])
                    tmp = path/f'{TRAINER_STATE_META}.tmp'; tmp.write_text(json.dumps(meta, indent=2)); tmp.replace(path/TRAINER_STATE_META)
                    receipt['checkpoints'].append(cp); save()
                    ready = root/f'checkpoint-{step}.ready.json'
                    tmp = ready.with_suffix('.tmp'); tmp.write_text(json.dumps(cp)); tmp.replace(ready)
        train_seconds = time.monotonic()-train_start
        if rank == 0:
            model.save_pretrained(root/'adapter-final'); tokenizer.save_pretrained(root/'adapter-final')
            receipt['frozen_after'] = frozen_weight_hash(model); receipt['adapter_after'] = parameter_hash(model, trainable=True)
            if receipt['frozen_before'] != receipt['frozen_after'] or receipt['adapter_before'] == receipt['adapter_after']:
                raise ValueError('parameter integrity failed')
        dist.barrier()
        receipt.update(status='completed', train_seconds=train_seconds, elapsed_seconds=time.monotonic()-started,
                       global_step=step, max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                       adapter_files={p.name: file_hash(p) for p in (root/'adapter-final').glob('adapter*')} if rank == 0 else None)
    except Exception as exc:
        receipt.update(status='failed', error=repr(exc)); raise
    finally:
        save(); dist.destroy_process_group()
    return receipt
