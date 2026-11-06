"""Qualify expert-trainer runtime variants; writes one benchmark receipt per candidate (torchrun, GPU only).

  torchrun --nproc_per_node=4 scripts/qualify_expert_trainer.py input.json out_dir

Every arm is built by the trainer's own build_model / ChunkStep from the same initial adapter.
1. Probes at the initial adapter, split across ranks:
   strict  6 rows at length quantiles of the first effective batch; every arm vs the reference arm.
   fp32    4 shorter rows; an fp32 reference (fp32 weights, no autocast, math SDPA) vs every arm.
2. Trajectory: the trainer's corpus stress, then warmup+steps optimizer steps on the real epoch-0
   order (partition_effective_batch, token_batches, SUM all-reduce, clip, AdamW at constant lr).
3. Gates per candidate (polytopia_bot.learning.trainer_qualification) and one receipt each.
The training objective is never changed: same rows, same denominator, same response-only loss.
"""
import contextlib, gc, json, os, sys, time
from pathlib import Path
import torch
from polytopia_bot.learning.data import JsonlDataset
from polytopia_bot.learning.distributed_training import partition_effective_batch, sum_gradients
from polytopia_bot.learning.efficient_training import EffectiveBatchLengthSampler
from polytopia_bot.learning.expert_trainer import verify_expert_corpus, build_model, ChunkStep, check_variant
from polytopia_bot.learning.fast_training import token_batches
from polytopia_bot.learning import trainer_qualification as tq

STRICT_QUANTILES = (.05, .25, .45, .65, .85, .97)
FP32_QUANTILES = (.05, .2, .35, .5)
FP32_TRUTH = dict(name='fp32_math_truth', packed=True, backend='segmented_sdpa', token_budget=8192, checkpointing=True)


def supervised(rows):
    return sum(sum(t != -100 for t in r['labels'][1:]) for r in rows)


def quantile_rows(rows, quantiles):
    by_len = sorted(rows, key=lambda r: len(r['input_ids']))
    return [by_len[min(len(by_len)-1, int(q*len(by_len)))] for q in quantiles]


class Arms:
    """Fresh model per arm; every arm starts from the first build's initial adapter.

    Build the bf16 reference first: peft rounds a bf16 model's freshly initialized adapter through bf16,
    so that is the adapter the trainer starts from, and the fp32 reference must start from it too.
    """

    def __init__(self, config, seed):
        self.config, self.seed, self.initial = config, seed, None

    def build(self, variant, precision='bf16'):
        from transformers import set_seed
        gc.collect(); torch.cuda.empty_cache(); set_seed(self.seed)
        model, info = build_model(self.config, variant, precision)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if self.initial is None:
            self.initial = {n: p.detach().clone() for n, p in named}
        elif set(self.initial) != {n for n, _ in named}:
            raise ValueError('adapter layout differs between arms')
        with torch.no_grad():
            for n, p in named:
                p.copy_(self.initial[n].to(p.dtype))
        return model, [p for _, p in named], info


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten() for p in params])


def probe(model, params, variant, rows, autocast=True, attention=None):
    step = ChunkStep(model, variant, autocast); denom = supervised(rows); loss = 0.
    with attention or contextlib.nullcontext():
        for chunk in token_batches(rows, variant['token_budget']):
            loss += float(step(chunk, denom))
    grad = flat_grad(params).cpu(); model.zero_grad(set_to_none=True)
    return loss, grad


def run_probes(task, builder, arms, data_rows):
    """strict: every arm vs arms[0]; fp32: the fp32 truth vs every arm. Returns name -> metrics."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    out = {}; ref = None
    rows = quantile_rows(data_rows, STRICT_QUANTILES if task == 'strict' else FP32_QUANTILES)
    if task == 'fp32':
        model, params, _ = builder.build(FP32_TRUTH, 'fp32')
        ref = probe(model, params, FP32_TRUTH, rows, autocast=False, attention=sdpa_kernel([SDPBackend.MATH]))
        out[FP32_TRUTH['name']] = dict(loss=ref[0]); del model, params
    for v in arms:
        started = time.monotonic()
        try:
            model, params, info = builder.build(v)
            loss, grad = probe(model, params, v, rows)
            ref = ref or (loss, grad)
            out[v['name']] = dict(tq.compare(*ref, loss, grad), runtime=info, seconds=time.monotonic()-started)
            del model, params
        except Exception as exc:  # a variant that does not run is a result; the reference must run
            if ref is None:
                raise
            out[v['name']] = dict(passed=False, error=repr(exc)[:800])
        print('PROBE', task, v['name'], json.dumps({k: x for k, x in out[v['name']].items() if k != 'runtime'}), flush=True)
    return dict(lengths=[len(r['input_ids']) for r in rows], results=out)


def run_trajectory(v, p, builder, data, order, lengths, rank, world):
    import torch.distributed as dist
    c = p['config']; effective = c['effective_batch_size']; entry = dict(name=v['name'], status='running')

    def agree(flag):
        t = torch.tensor(int(flag), device='cuda'); dist.all_reduce(t, op=dist.ReduceOp.MAX); return bool(t.item())

    failed = False
    try:
        model, params, info = builder.build(v); entry['runtime'] = info
        step = ChunkStep(model, v)
        stress = [data[i] for i in sorted(range(len(lengths)), key=lambda i: lengths[i])[-8:]]
        torch.cuda.reset_peak_memory_stats()
        for chunk in token_batches(partition_effective_batch(stress, world)[rank], v['token_budget']):
            step(chunk, supervised(stress))
        model.zero_grad(set_to_none=True); torch.cuda.synchronize()
        entry['stress_peak_bytes'] = torch.cuda.max_memory_allocated()
    except torch.OutOfMemoryError:
        failed = True; entry['status'] = 'oom'
    except Exception as exc:
        failed = True; entry.update(status='error', error=repr(exc)[:800])
    if agree(failed):
        entry['status'] = entry['status'] if failed else 'peer_failed'
        return entry
    opt = torch.optim.AdamW(params, lr=c['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0)
    torch.cuda.reset_peak_memory_stats(); steps = []; dist.barrier()
    for k in range(p['warmup_steps']+p['steps']):
        started = time.perf_counter()
        group = [data[i] for i in order[k*effective:(k+1)*effective]]
        denom = supervised(group); value = torch.zeros((), device='cuda')
        for chunk in token_batches(partition_effective_batch(group, world)[rank], v['token_budget']):
            value += step(chunk, denom)
        sum_gradients(params); dist.all_reduce(value, op=dist.ReduceOp.SUM)
        norm = torch.nn.utils.clip_grad_norm_(params, c['max_grad_norm'])
        if not torch.isfinite(norm) or not torch.isfinite(value):
            raise ValueError(f'{v["name"]}: nonfinite optimization at step {k}')
        opt.step(); opt.zero_grad(set_to_none=True); loss = float(value)
        steps.append(dict(loss=loss, grad_norm=float(norm), seconds=time.perf_counter()-started,
                          tokens=sum(len(r['input_ids']) for r in group)))
    peak = torch.tensor(max(torch.cuda.max_memory_allocated(), entry['stress_peak_bytes']), device='cuda')
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    measured = steps[p['warmup_steps']:]; seconds = sum(s['seconds'] for s in measured); tokens = sum(s['tokens'] for s in measured)
    entry.update(status='passed', peak_bytes=int(peak), losses=[s['loss'] for s in steps], grad_norms=[s['grad_norm'] for s in steps],
                 step_seconds=[s['seconds'] for s in steps], tokens=tokens, seconds=seconds, tokens_per_second=tokens/seconds,
                 seconds_per_step=seconds/len(measured))
    del opt, model, params, step; gc.collect(); torch.cuda.empty_cache()
    return entry


def main(p, out):
    import importlib.metadata as md
    import torch.distributed as dist
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    from datetime import timedelta
    # Ranks 2+ wait in all_gather_object while ranks 0/1 run the probes (several model builds).
    torch.cuda.set_device(int(os.environ['LOCAL_RANK'])); dist.init_process_group('nccl', timeout=timedelta(hours=1))
    out = Path(out); started = time.monotonic()
    arms = [check_variant(dict(v, packed=True)) for v in p['arms']]
    names = [v['name'] for v in arms]
    if len(set(names)) != len(names) or not set(p['candidates']) <= set(names[1:]):
        raise ValueError('arm names must be unique; candidates must be non-reference arms')
    shared = [verify_expert_corpus(p['prepared'], p['config']) if rank == 0 else None]
    dist.broadcast_object_list(shared, src=0); manifest, lengths = shared[0]
    data = JsonlDataset(Path(p['prepared'])/'tokens.jsonl')
    sampler = EffectiveBatchLengthSampler(lengths, seed=p['seed'], effective_batch_size=p['config']['effective_batch_size'])
    sampler.set_epoch(0); order = list(sampler)
    first = [data[i] for i in order[:p['config']['effective_batch_size']]]
    builder = Arms(p['config'], p['seed']); builder.build(arms[0])
    mine = {t: run_probes(t, builder, arms, first) for i, t in enumerate(('strict', 'fp32')) if i % world == rank}
    gathered = [None]*world; dist.all_gather_object(gathered, mine)
    probes = {k: x for g in gathered for k, x in g.items()}
    dist.barrier(); trajectories = {}
    for v in arms:
        trajectories[v['name']] = run_trajectory(v, p, builder, data, order, lengths, rank, world)
        if rank == 0:
            print('ARM', json.dumps({k: x for k, x in trajectories[v['name']].items()
                                     if k not in ('losses', 'grad_norms', 'step_seconds')}), flush=True)
        dist.barrier()
    if rank == 0:
        context = dict(config=p['config'], hardware=torch.cuda.get_device_name(), world_size=world,
            gpu_memory=torch.cuda.get_device_properties(0).total_memory,
            versions={k: md.version(k) for k in ('torch', 'transformers', 'peft', 'liger-kernel')},
            corpus=dict(prepared=p['prepared'], tokens_sha256=manifest['tokens_sha256'], records=manifest['records']),
            seed=p['seed'], learning_rate=p['config']['learning_rate'], trajectory_steps=p['warmup_steps']+p['steps'],
            warmup_steps=p['warmup_steps'], probe_setup=dict(strict_lengths=probes['strict']['lengths'],
            fp32_lengths=probes['fp32']['lengths'], fp32_reference=FP32_TRUTH), created_unix=time.time(), tag=p['tag'],
            config_source=p.get('config_source'))
        # Raw measurements first, so receipts can be re-derived on CPU (--assemble) if assembly fails.
        raw = dict(context=context, arms=arms, candidates=p['candidates'], base=p['base'], probes=probes,
                   trajectories=trajectories, seconds=time.monotonic()-started)
        (out/'raw.json').write_text(json.dumps(raw, indent=2))
        write_receipts(raw, out)
    dist.barrier(); dist.destroy_process_group()


def write_receipts(raw, out):
    summary, receipts = tq.assemble(raw)
    for name, r in receipts.items():
        (Path(out)/f'receipt-{name}.json').write_text(json.dumps(r, indent=2))
    for name, gates in summary['candidates'].items():
        print('CANDIDATE', name, json.dumps({k: g.get('passed') for k, g in gates.items() if isinstance(g, dict)}), flush=True)
    (Path(out)/'qualification.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    if sys.argv[1] == '--assemble':  # CPU: python scripts/qualify_expert_trainer.py --assemble raw.json out_dir
        write_receipts(json.loads(Path(sys.argv[2]).read_text()), sys.argv[3])
    else:
        main(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2])
