"""Replay saved expert-SFT prompts through the evaluation engine; Modal only.

  extract <bench_root>                   real prompts + original outputs from the source arm's traces
  run <bench_root> <config>              one engine config: warm-up, then each concurrency phase
  summarize <bench_root>                 throughput table and output-equivalence against the v1 engine

Every request is a real model decision from expert-sft-v1/expert_sft: same token ids (re-tokenized and
checked), same LoRA adapter (hash-checked), same legal-action grammar, same sampling options and the
same per-decision seed. Only engine scheduling knobs change between configs. Workers replay their
prompts in game order, closed loop, no think time: a saturated GPU, as in a real evaluation.
"""
import asyncio
import gzip
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

RUNS = Path('/workspace/runs')


def load_rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def extract(root):
    """Per game: 33 decisions evenly spaced over its model moves; the first of games 0-7 is warm-up."""
    bench = json.loads((root/'protocol.json').read_text())
    src = RUNS/bench['source']['run_id']/bench['source']['arm']
    p = json.loads((RUNS/bench['source']['run_id']/'protocol.json').read_text())
    jobs = {j['id']: j for j in p['jobs']}
    per_game = bench['prompts_per_game'] + 1
    measured, warmup = [], []
    for gi, trace in enumerate(sorted((src/'games').glob('*.jsonl.gz'))):
        job = jobs[trace.name.removesuffix('.jsonl.gz')]
        decisions = []
        with gzip.open(trace, 'rt') as f:
            for line in f:
                row = json.loads(line)
                if row.get('policy') != 'model':
                    continue
                # Played moves keep the entry under `diagnostic`; failed ones under `attempts`.
                for a in row.get('attempts') or row.get('diagnostic') or []:
                    decisions.append(dict(job=job['id'], rng_index=job['rng_index'], decision=a['timing']['decision'],
                        seed=a['timing']['seed'], messages=a['messages'],
                        prompt_token_ids=a['generation']['prompt_token_ids'],
                        original_tokens=a['generation']['response_token_ids'], original_text=a['generation']['text'],
                        original_first_token_seconds=a['timing']['first_token_seconds'],
                        original_request_seconds=a['timing']['request_seconds']))
        n = len(decisions)
        picks = sorted({round(i*(n-1)/(per_game-1)) for i in range(per_game)})
        chosen = [decisions[i] for i in picks]
        if gi < bench['warmup_games']:
            warmup.append(chosen.pop(0))
        else:
            chosen = chosen[1:]
        measured += chosen
    for name, rows in (('prompts', measured), ('warmup', warmup)):
        with gzip.open(root/f'{name}.jsonl.gz', 'wt') as f:
            for r in rows:
                r['key'] = f"{r['job']}-decision-{r['decision']}"
                f.write(json.dumps(r)+'\n')
    out = dict(measured=len(measured), warmup=len(warmup), games=len({r['job'] for r in measured}),
               prompt_tokens=sum(len(r['prompt_token_ids']) for r in measured))
    (root/'prompts-manifest.json').write_text(json.dumps(out, indent=2)+'\n')
    return out


def adapter_for(p, arm):
    from benchmark_expert_sft import adapter_hashes
    spec = p['arms'][arm]
    run = json.loads((RUNS/spec['training_receipt']).read_text())
    path = RUNS/spec['adapter']
    if adapter_hashes(path) != run['adapter_files']:
        raise ValueError('adapter files changed')
    return str(path), run['adapter_files']


def pct(values, q):
    v = sorted(values)
    return v[min(len(v)-1, int(q*len(v)))] if v else None


def metrics_snapshot():
    try:
        from vllm.v1.metrics.reader import Counter, get_metrics_snapshot
        return {m.name: m.value for m in get_metrics_snapshot() if isinstance(m, Counter)}
    except Exception as exc:  # stats are diagnostic only
        return dict(error=repr(exc)[:200])


async def run(root, config, limit=None, concurrencies=None):
    import os
    if not os.environ.get('MODAL_TASK_ID'):
        raise RuntimeError('Qwen inference must run on Modal')
    import torch
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams
    from vllm.v1.engine.async_llm import AsyncLLM
    from benchmark_expert_sft import engine_kwargs
    from benchmark_thinking_unbounded import sampling_options
    from polytopia_bot.learning.inference import decode_completion
    from polytopia_bot.learning.thinking_readable import action_schema
    bench = json.loads((root/'protocol.json').read_text())
    spec = bench['configs'][config]
    p = json.loads((RUNS/bench['source']['run_id']/'protocol.json').read_text())
    p = dict(p, serving={**p['serving'], **spec.get('serving', {})})
    out = root/'bench'/config
    out.mkdir(parents=True, exist_ok=False)
    prompts, warmup = load_rows(root/'prompts.jsonl.gz'), load_rows(root/'warmup.jsonl.gz')
    if limit:
        prompts = prompts[:limit]
    adapter_path, adapter_files = adapter_for(p, bench['source']['arm'])
    lora = LoRARequest(bench['source']['arm'], 1, adapter_path) if spec.get('adapter', True) else None
    # Stats on in every config (prefix-cache and preemption counters); logging never touches numerics.
    args = engine_kwargs(p, disable_log_stats=False)
    started = time.monotonic()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**args))
    tokenizer = engine.get_tokenizer()
    cc = engine.vllm_config.compilation_config
    if cc.max_cudagraph_capture_size is None or cc.max_cudagraph_capture_size < p['concurrency']:
        raise ValueError('CUDA graph ceiling does not cover the scheduler capacity')
    receipt = dict(config=config, spec=spec, engine_args={k: v for k, v in args.items() if k not in ('model', 'tokenizer')},
        load_seconds=time.monotonic()-started, device=torch.cuda.get_device_name(), adapter=adapter_path if lora else None,
        adapter_files=adapter_files if lora else None,
        packages={k: importlib.metadata.version(k) for k in ['vllm', 'torch', 'transformers', 'xgrammar']},
        max_num_seqs=engine.vllm_config.scheduler_config.max_num_seqs,
        max_num_batched_tokens=engine.vllm_config.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=getattr(cc, 'cudagraph_capture_sizes', None),
        max_cudagraph_capture_size=cc.max_cudagraph_capture_size, cudagraph_mode=str(getattr(cc, 'cudagraph_mode', None)),
        num_gpu_blocks=getattr(engine.vllm_config.cache_config, 'num_gpu_blocks', None),
        block_size=engine.vllm_config.cache_config.block_size,
        prefix_caching=engine.vllm_config.cache_config.enable_prefix_caching)
    (out/'runtime.json').write_text(json.dumps(receipt, indent=2, default=str)+'\n')
    print(json.dumps(dict(event='MODEL_READY', **{k: receipt[k] for k in ('config', 'load_seconds', 'max_num_batched_tokens', 'prefix_caching')})), flush=True)

    async def one(row, tag):
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)
        if ids != row['prompt_token_ids']:
            raise ValueError('re-tokenized prompt differs from the saved trace: '+row['key'])
        options = sampling_options(p, dict(rng_index=row['rng_index']), row['decision'])
        if options['seed'] != row['seed']:
            raise ValueError('seed formula differs from the saved trace: '+row['key'])
        options['structured_outputs'] = StructuredOutputsParams(json=action_schema(row['messages']))
        params = SamplingParams(**options, output_kind=RequestOutputKind.DELTA)
        t0 = time.monotonic()
        tokens, first, final, cached = [], None, None, None
        async for o in engine.generate(dict(prompt_token_ids=ids), params, f'{tag}-{row["key"]}', lora_request=lora):
            if cached is None and getattr(o, 'num_cached_tokens', None) is not None:
                cached = o.num_cached_tokens
            part = o.outputs[0]
            if part.token_ids and first is None:
                first = time.monotonic()-t0
            tokens.extend(part.token_ids)
            if o.finished:
                final = part
        return dict(key=row['key'], input_tokens=len(ids), cached_tokens=cached, output_tokens=len(tokens),
                    tokens=tokens, text=decode_completion(tokenizer, tokens), finish_reason=final.finish_reason,
                    first_token_seconds=first, request_seconds=time.monotonic()-t0, start=t0, end=time.monotonic())

    async def phase(rows, concurrency, tag):
        await engine.reset_prefix_cache()
        before = metrics_snapshot()
        chunks = [rows[i*len(rows)//concurrency:(i+1)*len(rows)//concurrency] for i in range(concurrency)]
        done = []

        async def worker(chunk):
            for row in chunk:
                done.append(await one(row, tag))
            return time.monotonic()
        t0 = time.monotonic()
        ends = await asyncio.gather(*(worker(c) for c in chunks if c))
        wall = time.monotonic()-t0
        after = metrics_snapshot()
        # Steady state: until the first worker runs out of prompts, all `concurrency` games are live.
        steady_end = min(ends)
        steady = [r for r in done if r['end'] <= steady_end]
        tokens = sum(r['input_tokens'] for r in done)
        cached = sum(r['cached_tokens'] or 0 for r in done)
        summary = dict(config=config, concurrency=concurrency, requests=len(done), wall_seconds=wall,
            prompt_tokens=tokens, cached_tokens=cached, computed_prompt_tokens=tokens-cached,
            prompt_tokens_per_second=tokens/wall, computed_tokens_per_second=(tokens-cached)/wall,
            steady_seconds=steady_end-t0, steady_requests=len(steady),
            steady_prompt_tokens_per_second=sum(r['input_tokens'] for r in steady)/(steady_end-t0),
            ttft_median=statistics.median(r['first_token_seconds'] for r in done),
            ttft_p90=pct([r['first_token_seconds'] for r in done], .9),
            request_median=statistics.median(r['request_seconds'] for r in done),
            request_p90=pct([r['request_seconds'] for r in done], .9),
            request_mean=statistics.fmean(r['request_seconds'] for r in done),
            decode_median=statistics.median(r['request_seconds']-r['first_token_seconds'] for r in done),
            finish_reasons=dict(sorted({r['finish_reason']: 0 for r in done}.items())),
            counters={k: after[k]-before.get(k, 0) for k in after if isinstance(after.get(k), (int, float))
                      and any(s in k for s in ('preempt', 'prefix_cache', 'prompt_tokens', 'generation_tokens'))})
        for r in done:
            summary['finish_reasons'][r['finish_reason']] += 1
        with (out/f'{tag}.jsonl').open('w') as f:
            for r in sorted(done, key=lambda r: r['key']):
                f.write(json.dumps({k: v for k, v in r.items() if k not in ('start', 'end')})+'\n')
        print(json.dumps(dict(event='PHASE', **{k: v for k, v in summary.items() if k != 'counters'})), flush=True)
        return summary

    try:
        results = dict(runtime=receipt, warmup=await phase(warmup, min(8, len(warmup)), 'warmup'), phases=[])
        for c in concurrencies or bench['concurrencies']:
            results['phases'].append(await phase(prompts, c, f'c{c}'))
            (out/'summary.json').write_text(json.dumps(results, indent=2, default=str)+'\n')
    finally:
        engine.shutdown()
    return results


def summarize(root):
    """Throughput per config/phase; output identity vs the original game and vs v1 at each concurrency."""
    bench = json.loads((root/'protocol.json').read_text())
    original = {r['key']: r for r in load_rows(root/'prompts.jsonl.gz')}
    outputs, table = {}, []
    for config in bench['configs']:
        path = root/'bench'/config/'summary.json'
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        for ph in s['phases']:
            tag = f"c{ph['concurrency']}"
            rows = {json.loads(l)['key']: json.loads(l) for l in (root/'bench'/config/f'{tag}.jsonl').open()}
            outputs[(config, tag)] = rows
            table.append({k: ph[k] for k in ('config', 'concurrency', 'requests', 'prompt_tokens_per_second',
                'computed_tokens_per_second', 'steady_prompt_tokens_per_second', 'ttft_median', 'ttft_p90',
                'request_median', 'request_p90', 'request_mean', 'decode_median', 'wall_seconds', 'cached_tokens',
                'prompt_tokens', 'counters')})

    def compare(a, b):
        keys = sorted(set(a) & set(b))
        same_tokens = [k for k in keys if a[k]['tokens'] == b[k]['tokens']]
        same_text = [k for k in keys if a[k]['text'] == b[k]['text']]
        return dict(n=len(keys), identical_tokens=len(same_tokens), identical_action=len(same_text),
                    differing=[dict(key=k, a=a[k]['text'], b=b[k]['text']) for k in keys if a[k]['text'] != b[k]['text']][:20])
    orig = {k: dict(tokens=v['original_tokens'], text=v['original_text']) for k, v in original.items()}
    equivalence = []
    for (config, tag), rows in outputs.items():
        row = dict(config=config, phase=tag, vs_original_game=compare(rows, orig))
        for ref in ('c30', 'c60'):
            if ('v1', ref) in outputs and config != 'v1-nolora':
                row[f'vs_v1_{ref}'] = compare(rows, outputs[('v1', ref)])
        equivalence.append(row)
    result = dict(throughput=table, equivalence=equivalence,
                  manifest=json.loads((root/'prompts-manifest.json').read_text()))
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    cmd, root = sys.argv[1], Path(sys.argv[2])
    if cmd == 'extract':
        print(json.dumps(extract(root)))
    elif cmd == 'run':
        limit = int(sys.argv[sys.argv.index('--limit')+1]) if '--limit' in sys.argv else None
        cs = [int(x) for x in sys.argv[sys.argv.index('--concurrency')+1].split(',')] if '--concurrency' in sys.argv else None
        asyncio.run(run(root, sys.argv[3], limit, cs))
    elif cmd == 'summarize':
        print(json.dumps(summarize(root)['throughput'], indent=1))
    else:
        raise SystemExit('usage: extract|run|summarize <bench_root> [config]')
