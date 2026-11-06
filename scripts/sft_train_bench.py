"""Expert-SFT trainer benchmark: runtime parity and throughput (Modal only; read-only on the corpus).

  explore  one GPU. Every variant against the qualified runtime on identical batches
           (loss, gradient cosine, relative error), per-chunk forward/backward seconds on one
           full effective batch, stress peak memory and a CUDA kernel-category profile.
  dist     torchrun. The expert trainer's step loop (partition_effective_batch, token_batches,
           packed_response_loss, SUM all-reduce, clip, AdamW) on the first effective batches of
           the real epoch-0 order, per config, with per-phase timing and loss-trajectory parity.

Variant keys: name, token_budget, checkpointing (False | True | 'alternating' | 'adaptive'),
adaptive_tokens (chunks above this many tokens are fully checkpointed), attention
('default' | 'cudnn' | 'flash' | 'efficient'), liger (fused RMSNorm/RoPE/SwiGLU kernels).
The training objective is never changed: same rows, same denominator, same response-only loss.
"""
import contextlib, gc, json, os, sys, time
from pathlib import Path
import torch
from polytopia_bot.learning.data import JsonlDataset, ResponseCollator
from polytopia_bot.learning.efficient_training import response_only_loss
from polytopia_bot.learning.fast_training import (pack, token_batches, packed_response_loss, configure_checkpointing,
                                                  set_backend, set_lora_input_cast)
from polytopia_bot.learning.training import lora_config

PARITY = dict(loss=.01, relative_gradient_error=.03, cosine=.999)  # same gates as the original qualification


def supervised(rows):
    return sum(sum(t != -100 for t in r['labels'][1:]) for r in rows)


def epoch_order(n, seed):
    # EffectiveBatchLengthSampler epoch 0: randperm(seed); in-window sorting is irrelevant because
    # partition_effective_batch and token_batches re-sort by length.
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()


ATTN = dict(repeat_kv=False, bucket=0, contiguous=False)


def bench_segmented_sdpa(module, query, key, value, attention_mask, dropout=0., scaling=None, segment_lengths=None, **kwargs):
    """fast_training.segmented_sdpa plus options: explicit repeat_kv (no enable_gqa), zero-padding each
    segment's q/k/v to a multiple of `bucket` (causal: real queries never see the padded tail; padded
    outputs are dropped), forced contiguity. Output layout (b, s, h, d) as transformers expects."""
    import torch.nn.functional as F
    from transformers.integrations.sdpa_attention import repeat_kv
    if not segment_lengths or sum(segment_lengths) != query.shape[2] or query.shape[0] != 1:
        raise ValueError('invalid packed attention segments')
    groups = query.shape[1]//key.shape[1]
    if ATTN['repeat_kv'] and groups > 1:
        key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    gqa = key.shape[1] != query.shape[1]
    outputs, start, b = [], 0, ATTN['bucket']
    for n in segment_lengths:
        q, k, v = (t[:, :, start:start+n] for t in (query, key, value))
        m = -(-n//b)*b if b else n
        if m > n:
            q, k, v = (F.pad(t, (0, 0, 0, m-n)) for t in (q, k, v))
        elif ATTN['contiguous']:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scaling, enable_gqa=gqa)
        outputs.append(o[:, :, :n]); start += n
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None


class Runtime:
    def __init__(self, config, device='cuda'):
        import transformers.models.qwen3.modeling_qwen3 as mq
        self.config = config; self.device = device
        self.originals = {k: getattr(mq, k) for k in ('apply_rotary_pos_emb', 'Qwen3RMSNorm', 'Qwen3MLP')}
        self.model = None; self.initial = None; self.liger = None; self.liger_key = None
        self.gqa_patched = False; self.mode = None; self.prewarm_seconds = None; self.collate = ResponseCollator(151643)
        self.load(None, 'bf16')

    def load(self, liger, precision='bf16'):
        """(Re)build base+LoRA; liger is None or dict(rope, rms_norm, swiglu). Same initial adapter every time."""
        import transformers.models.qwen3.modeling_qwen3 as mq
        from transformers import AutoModelForCausalLM, set_seed
        from peft import get_peft_model
        self.model = None; gc.collect(); torch.cuda.empty_cache()
        for k, x in self.originals.items():
            setattr(mq, k, x)
        set_seed(17)
        model = AutoModelForCausalLM.from_pretrained(self.config['model_id'], revision=self.config['revision'],
                                                     torch_dtype=torch.float32 if precision == 'fp32' else torch.bfloat16,
                                                     attn_implementation='sdpa').to(self.device)
        model.config.use_cache = False
        self.model = get_peft_model(model, lora_config(self.config)); self.model.train()
        set_lora_input_cast(self.model, False)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        if self.initial is None:
            self.initial = [p.detach().clone() for p in self.params]
        if liger:
            import importlib.metadata as md
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3
            apply_liger_kernel_to_qwen3(cross_entropy=False, fused_linear_cross_entropy=False,
                                        model=self.model.get_base_model(), **liger)
            self.liger = md.version('liger-kernel')
        else:
            self.liger = None
        self.precision = precision; self.liger_key = json.dumps([liger, precision], sort_keys=True); self.mode = None
        self.reset()

    def reset(self):
        self.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, x in zip(self.params, self.initial):
                p.copy_(x.to(p.dtype))
        gc.collect(); torch.cuda.empty_cache()

    def apply(self, v):
        liger = v.get('liger') or None
        if liger is True:
            liger = dict(rope=True, rms_norm=True, swiglu=True)
        precision = v.get('precision', 'bf16')
        if json.dumps([liger, precision], sort_keys=True) != self.liger_key:
            self.load(liger, precision)
        custom = any(k in v for k in ('repeat_kv', 'bucket', 'contiguous'))
        set_backend(self.model, 'segmented_sdpa')
        if custom:
            from transformers import AttentionInterface
            AttentionInterface.register('segmented_sdpa', bench_segmented_sdpa)
            ATTN.update(repeat_kv=v.get('repeat_kv', False), bucket=v.get('bucket', 0), contiguous=v.get('contiguous', False))
            self.model.get_base_model().set_attn_implementation('segmented_sdpa')
        self.mode = None; self.set_checkpointing(v['checkpointing'] if v['checkpointing'] != 'adaptive' else False)
        if v.get('prewarm') and v.get('bucket'):
            self.prewarm(v)

    def prewarm(self, v):
        """Build the cuDNN plan for every length bucket up to the corpus maximum once (untimed, recorded)."""
        import torch.nn.functional as F
        cfg = self.model.get_base_model().config; started = time.perf_counter()
        heads, kv, dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        with self.attention(v):
            for m in range(v['bucket'], v['prewarm']+v['bucket'], v['bucket']):
                q = torch.randn(1, heads, m, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
                k = torch.randn(1, heads if ATTN['repeat_kv'] else kv, m, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
                F.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=k.shape[1] != heads).sum().backward()
        torch.cuda.synchronize(); self.prewarm_seconds = time.perf_counter()-started

    def set_checkpointing(self, mode):
        if mode != self.mode:
            configure_checkpointing(self.model, mode); self.mode = mode

    def attention(self, v):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        name = v.get('attention', 'default')
        if name == 'default':
            return contextlib.nullcontext()
        backend = dict(cudnn=SDPBackend.CUDNN_ATTENTION, flash=SDPBackend.FLASH_ATTENTION, math=SDPBackend.MATH,
                       efficient=SDPBackend.EFFICIENT_ATTENTION)[name]
        return sdpa_kernel([backend])

    def ensure_gqa(self, v):
        """cuDNN SDPA may reject enable_gqa; fall back to transformers' repeat_kv (recorded)."""
        if v.get('attention', 'default') == 'default' or self.gqa_patched:
            return
        q = torch.randn(1, 32, 64, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 8, 64, 128, device='cuda', dtype=torch.bfloat16)
        try:
            with self.attention(v):
                torch.nn.functional.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=True)
        except RuntimeError:
            raise ValueError('this SDPA backend rejects enable_gqa; use repeat_kv=True')

    def chunk_loss_backward(self, v, chunk, denom, timings=None):
        """One packed chunk: forward + backward; returns detached loss. timings gets (fwd, bwd) seconds."""
        if v['checkpointing'] == 'adaptive':
            self.set_checkpointing(True if sum(len(r['input_ids']) for r in chunk) > v['adaptive_tokens'] else False)
        inputs = pack(chunk, 'cuda')
        with self.attention(v):
            if timings is not None:
                e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]; e[0].record()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.precision == 'bf16'):
                loss = packed_response_loss(self.model, inputs, denom)
            if timings is not None:
                e[1].record()
            loss.backward()
            if timings is not None:
                e[2].record(); torch.cuda.synchronize()
                timings.append((e[0].elapsed_time(e[1])/1000, e[1].elapsed_time(e[2])/1000))
        return loss.detach()

    def flat_grad(self):
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten()
                          for p in self.params])


def compare(ref_loss, ref_grad, loss, grad):
    # Float64: F.cosine_similarity in float32 multiplies two tiny squared norms and underflows
    # (it reports 1.008 for a gradient against itself on this corpus).
    a, b = ref_grad.double(), grad.double()
    out = dict(loss=loss, loss_delta=abs(loss-ref_loss),
               relative_gradient_error=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a)),
               cosine=float(a.dot(b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))))
    out['passed'] = (out['loss_delta'] < PARITY['loss'] and out['relative_gradient_error'] < PARITY['relative_gradient_error']
                     and out['cosine'] > PARITY['cosine'])
    return out


def kernel_mix(rt, v, chunk):
    """CUDA self-time by kernel category for one forward+backward."""
    from torch.autograd import DeviceType
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        rt.chunk_loss_backward(v, chunk, None); torch.cuda.synchronize()
    rt.model.zero_grad(set_to_none=True)
    cats = {}
    for e in prof.key_averages():
        if e.device_type != DeviceType.CUDA:
            continue
        k = e.key.lower()
        cat = ('attention' if any(s in k for s in ('flash', 'fmha', 'cudnn', 'attention', 'sdpa')) else
               'gemm' if any(s in k for s in ('nvjet', 'gemm', 'cutlass', 'sm90_', 'sm100_', 'xmma', 'ampere_', 'cublas')) else
               'nccl' if 'nccl' in k else 'memcpy' if 'memcpy' in k or 'memset' in k else
               'triton/liger' if 'triton' in k or '_kernel' in k and 'elementwise' not in k and 'reduce' not in k else
               'elementwise/reduce')
        cats[cat] = cats.get(cat, 0.) + e.self_device_time_total/1e6
    total = sum(cats.values())
    return dict(seconds=total, tokens=sum(len(r['input_ids']) for r in chunk),
                share={k: round(x/total, 4) for k, x in sorted(cats.items(), key=lambda kv: -kv[1])})


def explore(p, out):
    data = JsonlDataset(Path(p['prepared'])/'tokens.jsonl')
    order = epoch_order(len(data), p['seed'])
    if p.get('length_range'):
        # Short-prompt proxy (e.g. compact prompt): the first effective batch of epoch-0 rows in the range.
        lo, hi = p['length_range']; rows = []
        for i in order:
            r = data[i]
            if lo <= len(r['input_ids']) <= hi:
                rows.append(r)
                if len(rows) == p['effective']:
                    break
    else:
        rows = [data[i] for i in order[:p['effective']]]               # one effective batch, real epoch-0 order
    by_len = sorted(rows, key=lambda r: len(r['input_ids']))
    probe = [by_len[min(len(by_len)-1, int(q*len(by_len)))] for q in p.get('probe_quantiles', (.05, .25, .45, .65, .85, .97))]
    stress = [] if p.get('length_range') else [data[i] for i in p['stress_indices'][:4]]
    rt = Runtime(p['config']); result = dict(schema='sft-train-bench-explore-v1', hardware=torch.cuda.get_device_name(),
        gpu_memory=torch.cuda.get_device_properties(0).total_memory, torch=torch.__version__,
        probe_lengths=[len(r['input_ids']) for r in probe], batch_tokens=sum(len(r['input_ids']) for r in rows),
        stress_lengths=[len(r['input_ids']) for r in stress], variants=[])
    save = lambda: Path(out).write_text(json.dumps(result, indent=2))
    denom = supervised(probe)
    # Original qualification reference: serial unpadded examples, checkpointed, default SDPA.
    configure_checkpointing(rt.model, True); ref_loss = 0.
    for r in probe:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = response_only_loss(rt.model, {k: t.cuda() for k, t in rt.collate([r]).items()}, denom)
        l.backward(); ref_loss += float(l.detach())
    serial = (ref_loss, rt.flat_grad().cpu()); rt.reset(); qualified = None
    for v in p['variants']:
        entry = dict(v, status='running'); result['variants'].append(entry); save()
        try:
            rt.prewarm_seconds = None; rt.apply(v); rt.ensure_gqa(v); entry['gqa_repeat_kv'] = rt.gqa_patched
            entry['prewarm_seconds'] = rt.prewarm_seconds
            rt.reset(); loss = 0.; entry['probe_chunk_losses'] = []
            for chunk in token_batches(probe, v['token_budget']):
                entry['probe_chunk_losses'].append(float(rt.chunk_loss_backward(v, chunk, denom))); loss += entry['probe_chunk_losses'][-1]
            grad = rt.flat_grad().cpu()
            entry['parity_vs_serial'] = compare(*serial, loss, grad)
            if qualified is None:
                qualified = (loss, grad)
            entry['parity_vs_qualified'] = compare(*qualified, loss, grad)
            rt.model.zero_grad(set_to_none=True)
            if p.get('parity_only'):
                entry['status'] = 'passed' if entry['parity_vs_qualified']['passed'] else 'parity_failed'
                continue
            # Timing: one full effective batch on this GPU (= 4 ranks' work), one warm chunk first.
            chunks = list(token_batches(rows, v['token_budget']))
            rt.chunk_loss_backward(v, chunks[-1], None); rt.model.zero_grad(set_to_none=True)
            # cuDNN builds an execution plan per new shape: a second pass over the same shapes is the
            # steady state (a full run sees each length bucket thousands of times); the first is recorded.
            batch_denom = supervised(rows); cold = None
            for _ in range(2 if v.get('attention', 'default') != 'default' else 1):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); timings = []; start = time.perf_counter()
                for chunk in chunks:
                    rt.chunk_loss_backward(v, chunk, batch_denom, timings)
                rt.model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); seconds = time.perf_counter()-start
                cold = seconds if cold is None else cold
            entry['first_pass_seconds'] = cold
            fw, bw = sum(t[0] for t in timings), sum(t[1] for t in timings)
            entry.update(seconds=seconds, forward_seconds=fw, backward_seconds=bw, chunks=len(chunks),
                         tokens=result['batch_tokens'], tokens_per_second=result['batch_tokens']/seconds,
                         batch_peak_bytes=torch.cuda.max_memory_allocated())
            rt.model.zero_grad(set_to_none=True)
            biggest = max(chunks, key=lambda c: sum(len(r['input_ids']) for r in c))
            entry['kernel_mix'] = kernel_mix(rt, v, min(chunks, key=lambda c: abs(sum(len(r['input_ids']) for r in c)-9300)))
            entry.update(status='passed' if entry['parity_vs_qualified']['passed'] else 'parity_failed',
                         stress_chunk_tokens=max(sum(len(r['input_ids']) for r in biggest), max(result['stress_lengths'] or [0])))
            torch.cuda.reset_peak_memory_stats()
            try:
                for chunk in [biggest, *token_batches(stress, v['token_budget'])]:
                    rt.chunk_loss_backward(v, chunk, None); rt.model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                entry.update(stress_status='passed', stress_peak_bytes=torch.cuda.max_memory_allocated())
            except torch.OutOfMemoryError:
                entry.update(stress_status='oom', stress_peak_bytes=torch.cuda.max_memory_allocated())
        except torch.OutOfMemoryError:
            entry['status'] = 'oom'
        except Exception as exc:  # recorded, not fatal: a variant that does not run is a result
            entry.update(status='error', error=repr(exc)[:800])
        finally:
            rt.model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
            entry['liger_version'] = rt.liger; save(); print('VARIANT', json.dumps(entry), flush=True)
    return result


def dist_main(p, out):
    import torch.distributed as dist
    from polytopia_bot.learning.distributed_training import partition_effective_batch, sum_gradients
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK'])); dist.init_process_group('nccl')
    c = p['config']; effective = c['effective_batch_size']
    data = JsonlDataset(Path(p['prepared'])/'tokens.jsonl')
    order = epoch_order(len(data), p['seed'])
    total_steps = p['warmup_steps']+p['steps']
    groups = [order[k*effective:(k+1)*effective] for k in range(total_steps)]
    stress = [data[i] for i in p['stress_indices'][:8]]
    rt = Runtime(c); results = []
    result = dict(schema='sft-train-bench-dist-v1', hardware=torch.cuda.get_device_name(), world_size=world,
                  gpu_memory=torch.cuda.get_device_properties(0).total_memory, torch=torch.__version__,
                  steps=p['steps'], warmup_steps=p['warmup_steps'], configs=results)

    def agree(flag):
        t = torch.tensor(int(flag), device='cuda'); dist.all_reduce(t, op=dist.ReduceOp.MAX); return bool(t.item())

    for v in p['variants']:
        entry = dict(v, status='running'); results.append(entry); failed = False
        try:
            rt.prewarm_seconds = None; rt.apply(v); rt.ensure_gqa(v); rt.reset(); entry['gqa_repeat_kv'] = rt.gqa_patched
            entry['prewarm_seconds'] = rt.prewarm_seconds
            # Stress: the trainer's 8 longest, then the largest token_budget chunk of the measured batches.
            torch.cuda.reset_peak_memory_stats()
            local_stress = partition_effective_batch(stress, world)[rank]
            # Largest chunk any rank will see in the measured steps, found from lengths alone.
            biggest = max((ch for g in groups for part in partition_effective_batch(
                              [dict(input_ids=range(p['lengths'][i]), i=i) for i in g], world)
                           for ch in token_batches(part, v['token_budget'])),
                          key=lambda ch: sum(len(r['input_ids']) for r in ch))
            biggest = [data[r['i']] for r in biggest]
            for chunk in [*token_batches(local_stress, v['token_budget']), biggest]:
                rt.chunk_loss_backward(v, chunk, None); rt.model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            entry['stress_peak_bytes'] = torch.cuda.max_memory_allocated()
            entry['stress_chunk_tokens'] = max(max(len(r['input_ids']) for r in stress), sum(len(r['input_ids']) for r in biggest))
        except torch.OutOfMemoryError:
            failed = True; entry['status'] = 'oom'
        except Exception as exc:
            failed = True; entry.update(status='error', error=repr(exc)[:800])
        rt.model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        if agree(failed):
            entry['status'] = entry['status'] if failed else 'peer_failed'
            if rank == 0:
                print('CONFIG', json.dumps(entry), flush=True); Path(out).write_text(json.dumps(result, indent=2))
            continue
        rt.reset(); opt = torch.optim.AdamW(rt.params, lr=c['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0)
        torch.cuda.reset_peak_memory_stats(); steps = []; dist.barrier()
        for k, group_ids in enumerate(groups):
            t0 = time.perf_counter()
            group = [data[i] for i in group_ids]
            denom = supervised(group); local = partition_effective_batch(group, world)[rank]
            t1 = time.perf_counter(); timings = []; value = torch.zeros((), device='cuda')
            for chunk in token_batches(local, v['token_budget']):
                value += rt.chunk_loss_backward(v, chunk, denom, timings)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            dist.barrier(); t3 = time.perf_counter()                       # waiting for the slowest rank
            sum_gradients(rt.params); dist.all_reduce(value, op=dist.ReduceOp.SUM); torch.cuda.synchronize(); t4 = time.perf_counter()
            norm = torch.nn.utils.clip_grad_norm_(rt.params, c['max_grad_norm'])
            opt.step(); opt.zero_grad(set_to_none=True); loss = float(value); t5 = time.perf_counter()
            steps.append(dict(step=k, loss=loss, grad_norm=float(norm), seconds=t5-t0, data=t1-t0, compute=t2-t1,
                              forward=sum(t[0] for t in timings), backward=sum(t[1] for t in timings),
                              wait=t3-t2, allreduce=t4-t3, optimizer=t5-t4, chunks=len(timings),
                              local_tokens=sum(len(r['input_ids']) for r in local),
                              tokens=sum(len(r['input_ids']) for r in group)))
        peak = torch.tensor(max(torch.cuda.max_memory_allocated(), entry['stress_peak_bytes']), device='cuda')
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        gathered = [None]*world; dist.all_gather_object(gathered, steps)
        del opt; rt.model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        if rank == 0:
            measured = list(zip(*[g[p['warmup_steps']:] for g in gathered]))
            seconds = sum(s[0]['seconds'] for s in measured); tokens = sum(s[0]['tokens'] for s in measured)
            phase = lambda key, f=max: sum(f([x[key] for x in s]) for s in measured)
            mean = lambda xs: sum(xs)/len(xs)
            entry.update(status='passed', peak_bytes=int(peak), seconds=seconds, tokens=tokens,
                         tokens_per_second=tokens/seconds, seconds_per_step=seconds/len(measured),
                         phases_mean_rank=dict(data=phase('data', mean), forward=phase('forward', mean),
                             backward=phase('backward', mean), compute=phase('compute', mean), wait=phase('wait', mean),
                             allreduce=phase('allreduce', mean), optimizer=phase('optimizer', mean)),
                         slowest_rank_compute=phase('compute'), fastest_rank_compute=phase('compute', min),
                         losses=[s[0]['loss'] for s in measured], grad_norms=[s[0]['grad_norm'] for s in measured],
                         warmup_losses=[s[0]['loss'] for s in zip(*gathered)][:p['warmup_steps']],
                         chunks_per_rank=mean([x['chunks'] for s in measured for x in s]),
                         step_seconds=[s[0]['seconds'] for s in measured])
            print('CONFIG', json.dumps({k: x for k, x in entry.items() if k not in ('losses', 'grad_norms', 'step_seconds')}), flush=True)
            Path(out).write_text(json.dumps(result, indent=2))
        dist.barrier()
    if rank == 0:
        base = next((e for e in results if e['status'] == 'passed'), None)
        for e in results:
            if base and e['status'] == 'passed':
                d = [abs(a-b) for a, b in zip(e['losses'], base['losses'])]
                e['loss_trajectory_vs_first'] = dict(max_abs=max(d), mean_abs=sum(d)/len(d), reference=base['name'])
        Path(out).write_text(json.dumps(result, indent=2))
    dist.destroy_process_group()
    return result


if __name__ == '__main__':
    payload = json.loads(Path(sys.argv[1]).read_text())
    (explore if payload['mode'] == 'explore' else dist_main)(payload, sys.argv[2])
