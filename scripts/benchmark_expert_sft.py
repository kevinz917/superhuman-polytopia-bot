"""One evaluation arm of the expert-SFT comparison (Modal only).

Same harness as qwen3-instruct-2507-30paired-v2: readable named-field prompt,
legal-action JSON grammar from token 0, same sampling and per-decision seeds,
one call per move, no repairs. Every arm runs an identical vLLM engine with LoRA
enabled; the base arm simply sends no adapter.

Opt-in speedups (defaults reproduce expert-sft-v1 exactly):
  serving.max_num_batched_tokens / enable_prefix_caching / gpu_memory_utilization
      engine knobs read from the protocol (see docs/sft-eval-speedup-2026-10-19.md);
  --shard I/N
      play only jobs[I::N] of the arm into <arm>/shards/I-of-N; then
  merge <root> <arm> N
      checks every shard ran the same engine and adapter, then moves the shard outputs into the usual
      <arm>/ layout (runtime.json, games/, streams/, generation-timing.jsonl, report.json).
"""
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_qwen3_instruct import InstructGenerator, preflight  # noqa: E402
from polytopia_bot.learning.async_gameplay import atomic_json, play_async_games  # noqa: E402
from polytopia_bot.learning.interface import digest  # noqa: E402
from polytopia_bot.learning.thinking_readable import render_readable_decision  # noqa: E402


class _LoraEngine:
    """Delegate to AsyncLLM, attaching one fixed LoRA request to every generation."""
    def __init__(self, engine, lora):
        self._engine, self._lora = engine, lora

    def generate(self, prompt, params, request_id):
        return self._engine.generate(prompt, params, request_id, lora_request=self._lora)

    def __getattr__(self, name):
        return getattr(self._engine, name)


def engine_kwargs(p, **overrides):
    """vLLM engine arguments for an evaluation arm; unset serving knobs keep the expert-sft-v1 values."""
    serving = p['serving']
    args = dict(
        model=p['model_id'], revision=p['revision'], tokenizer=p['model_id'],
        tokenizer_revision=p['revision'], dtype='bfloat16',
        max_model_len=p['max_length'], gpu_memory_utilization=serving.get('gpu_memory_utilization', .80),
        max_num_seqs=p['concurrency'], max_num_batched_tokens=serving.get('max_num_batched_tokens', 8192),
        enable_prefix_caching=serving.get('enable_prefix_caching', False), enable_chunked_prefill=True,
        enforce_eager=serving['enforce_eager'],
        seed=p['seed'], disable_log_stats=True, enable_log_requests=False,
        compilation_config=dict(max_cudagraph_capture_size=serving['max_cudagraph_capture_size']),
        structured_outputs_config=dict(backend='xgrammar'),
        enable_lora=True, max_lora_rank=p['lora']['max_rank'], max_loras=1)
    args.update(overrides)
    return args


class ArmGenerator(InstructGenerator):
    def __init__(self, p, root, adapter):
        import os
        if not os.environ.get('MODAL_TASK_ID'):
            raise RuntimeError('Qwen inference must run on Modal')
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.lora.request import LoRARequest
        root.mkdir(parents=True, exist_ok=False)
        (root/'streams').mkdir()
        self.p, self.root, self.thinking, self.pending = p, root, False, {}
        serving = p['serving']
        engine_args = engine_kwargs(p)
        self.engine_args = {k: v for k, v in engine_args.items() if k not in ('model', 'tokenizer')}
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_args))
        ceiling = engine.vllm_config.compilation_config.max_cudagraph_capture_size
        if serving.get('require_full_decode_graph_coverage') and (ceiling is None or ceiling < p['concurrency']):
            engine.shutdown()
            raise ValueError(f'CUDA graph ceiling {ceiling} does not cover scheduler capacity {p["concurrency"]}')
        self.lora = LoRARequest(adapter['name'], 1, adapter['path']) if adapter else None
        self.engine = _LoraEngine(engine, self.lora) if self.lora else engine
        self.tokenizer = engine.get_tokenizer()
        self.log = (root/'generation-timing.jsonl').open('x')


def opponent_factory_for(p):
    """Scripted styles by name; names listed in p['rl_opponents'] load a hash-checked RL checkpoint."""
    rl, cache = p.get('rl_opponents') or {}, {}
    def make(name):
        if name in rl:
            if name not in cache:
                from polytopia_bot.rl.policy import StructuredPolicy
                spec = rl[name]
                if hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest() != spec['sha256']:
                    raise ValueError(f'RL opponent {name} checkpoint changed')
                cache[name] = StructuredPolicy(spec['path'], deterministic=True)
            return cache[name]
        from polytopia_bot.learning.teacher import ScriptedTeacher
        return ScriptedTeacher(name)
    return make


def adapter_hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(path).glob('adapter_*'))}


async def gameplay(root, arm, preflight_only=False, shard=None):
    import jax
    import jax.numpy as jnp
    import torch
    from polytopia_bot.gpu.mapgen import layout
    from polytopia_bot.learning.splits import board_hashes
    p = json.loads((root/'protocol.json').read_text())
    spec = p['arms'][arm]
    # Optional per-arm opponent shard; arms without one play every job (expert-sft-v1 behaviour).
    jobs = [j for j in p['jobs'] if spec.get('opponent') in (None, j['opponent'])]
    if shard is not None:
        index, count = shard
        if not 0 <= index < count:
            raise ValueError('invalid shard')
        # Strided split keeps the map/seat/opponent mix of every shard close to the arm's.
        jobs = jobs[index::count]
    if not jobs:
        raise ValueError('arm has no jobs')
    rules = json.loads(Path('/workspace/configs/ruleset-v1.json').read_text())
    for name, sha in {**rules['source_sha256'], **p['source_hashes']}.items():
        if hashlib.sha256((Path('/workspace')/name).read_bytes()).hexdigest() != sha:
            raise ValueError('source changed: '+name)
    generate = jax.jit(lambda seed: layout(seed, size=14))
    for job in jobs:
        if board_hashes(jax.device_get(generate(jnp.int32(job['seed']))), 14)[0] != job['board_hash']:
            raise ValueError('map changed')
    adapter = None
    if spec['adapter']:
        path = Path('/workspace/runs')/spec['adapter']
        expected = spec['adapter_files']
        if expected is None and spec.get('training_receipt'):
            run = json.loads((Path('/workspace/runs')/spec['training_receipt']).read_text())
            if run['status'] != 'completed':
                raise ValueError('expert training not completed')
            expected = run['adapter_files']
        spec = dict(spec, adapter_files=expected)
        if adapter_hashes(path) != expected:
            raise ValueError('adapter files changed')
        cfg = json.loads((path/'adapter_config.json').read_text())
        if cfg['base_model_name_or_path'] != p['model_id'] or cfg['r'] > p['lora']['max_rank']:
            raise ValueError('adapter does not match base model / rank limit')
        adapter = dict(name=arm, path=str(path))
    out = root/arm if shard is None else root/arm/'shards'/shard_name(*shard)
    started = time.monotonic()
    g = ArmGenerator(p, out, adapter)
    try:
        receipt = dict(arm=arm, model_id=p['model_id'], revision=p['revision'], adapter=spec['adapter'],
            adapter_files=spec['adapter_files'], thinking=False, reasoning_parser=None,
            load_seconds=time.monotonic()-started, max_tokens=None, thinking_token_budget=None,
            max_model_len=g.engine.model_config.max_model_len, device=torch.cuda.get_device_name(),
            tokenizer_hash=digest(g.tokenizer.backend_tokenizer.to_str()), chat_template_hash=digest(g.tokenizer.chat_template),
            packages={k: importlib.metadata.version(k) for k in ['vllm', 'torch', 'transformers', 'jax', 'xgrammar']},
            serving=p['serving'], engine_args=g.engine_args,
            max_num_seqs=g.engine.vllm_config.scheduler_config.max_num_seqs,
            actual_cudagraph_capture_size=g.engine.vllm_config.compilation_config.max_cudagraph_capture_size,
            constrain_actions=True, harness=p['harness'])
        if shard is not None:
            receipt['shard'] = dict(index=shard[0], count=shard[1], jobs=[j['id'] for j in jobs])
        if receipt['max_model_len'] != p['max_length']:
            raise ValueError('native context not available')
        receipt['preflight'] = await preflight(g, jobs[0])
        atomic_json(out/'runtime.json', receipt)
        if preflight_only:
            print(json.dumps(dict(event='PREFLIGHT_ONLY', arm=arm, preflight=receipt['preflight'])), flush=True)
            return
        print(json.dumps(dict(event='MODEL_READY', arm=arm)), flush=True)
        result = await play_async_games(out/'games', jobs, g, thinking=False, prompt_renderer=render_readable_decision,
                                        opponent_factory=opponent_factory_for(p))
        atomic_json(out/'report.json', result)
    finally:
        g.log.close(); g.engine.shutdown()


def shard_name(index, count):
    return f'{index:02d}-of-{count:02d}'


# Receipt fields every shard must share: same engine, model, adapter, template and harness.
SHARED = ['arm', 'model_id', 'revision', 'adapter', 'adapter_files', 'max_model_len', 'device', 'tokenizer_hash',
          'chat_template_hash', 'packages', 'serving', 'engine_args', 'max_num_seqs', 'actual_cudagraph_capture_size',
          'constrain_actions', 'harness']


def merge_shards(root, arm, count):
    """Combine <arm>/shards/*-of-N into the unsharded <arm>/ layout; refuses inconsistent shards."""
    p = json.loads((root/'protocol.json').read_text())
    spec = p['arms'][arm]
    jobs = [j for j in p['jobs'] if spec.get('opponent') in (None, j['opponent'])]
    out, games_out = root/arm, root/arm/'games'
    shards = [out/'shards'/shard_name(i, count) for i in range(count)]
    receipts = [json.loads((d/'runtime.json').read_text()) for d in shards]
    reports = [json.loads((d/'report.json').read_text()) for d in shards]
    for i, r in enumerate(receipts):
        if r.get('shard', {}).get('count') != count or r['shard']['index'] != i:
            raise ValueError(f'shard {i} receipt does not match')
        diff = [k for k in SHARED if r.get(k) != receipts[0].get(k)]
        if diff:
            raise ValueError(f'shard {i} differs from shard 0 in {diff}')
    expected = [j['id'] for i in range(count) for j in jobs[i::count]]
    if sorted(x for r in receipts for x in r['shard']['jobs']) != sorted(expected) or len(set(expected)) != len(jobs):
        raise ValueError('shards do not cover the arm exactly once')
    games_out.mkdir(parents=True, exist_ok=True)
    (games_out/'private').mkdir(exist_ok=True)
    (out/'streams').mkdir(exist_ok=True)
    # Traces, checkpoints and per-decision streams are moved, not copied (one file per decision makes
    # copying slow on a network volume); a re-run skips files an earlier merge already moved.
    keep = ('games.json', 'jobs.json', 'report.json', 'progress.json')
    for d in shards:
        for src_dir, dst_dir in ((d/'games', games_out), (d/'games'/'private', games_out/'private'),
                                 (d/'streams', out/'streams')):
            for src in src_dir.iterdir():
                if src.is_file() and src.name not in keep:
                    shutil.move(src, dst_dir/src.name)
    streams = sum(1 for _ in (out/'streams').iterdir())
    decisions = sum(sum(1 for _ in (d/'generation-timing.jsonl').open()) for d in shards)
    if streams != decisions:
        raise ValueError(f'{streams} stream files for {decisions} timed decisions')
    order = {j['id']: i for i, j in enumerate(jobs)}
    games = sorted((g for r in reports for g in r['games']), key=lambda g: order[g['id']])
    with (out/'generation-timing.jsonl.tmp').open('w') as f:
        for d in shards:
            f.write((d/'generation-timing.jsonl').read_text())
    (out/'generation-timing.jsonl.tmp').replace(out/'generation-timing.jsonl')
    atomic_json(games_out/'jobs.json', jobs)
    atomic_json(games_out/'games.json', games)
    shard_rows = [dict(index=i, directory=str(d.relative_to(root)), games=r['finished_games'],
                       elapsed_seconds=r['elapsed_seconds'], complete=r['complete'],
                       load_seconds=c['load_seconds'], device=c['device']) for i, (d, r, c) in enumerate(zip(shards, reports, receipts))]
    report = dict(schema='async-gameplay-v1', allocated_games=len(jobs), finished_games=len(games), games=games,
                  elapsed_seconds=max(r['elapsed_seconds'] for r in reports),
                  complete=len(games) == len(jobs) and all(r['complete'] for r in reports),
                  sharding=dict(count=count, split='jobs[i::count]', shards=shard_rows))
    atomic_json(games_out/'progress.json', dict(elapsed_seconds=report['elapsed_seconds'], finished=len(games),
                                                 allocated=len(jobs), active=[], generating=[]))
    atomic_json(games_out/'report.json', report)
    runtime = {k: v for k, v in receipts[0].items() if k != 'shard'}
    runtime.update(load_seconds=max(r['load_seconds'] for r in receipts),
                   preflight=[r['preflight'] for r in receipts], sharding=report['sharding'])
    atomic_json(out/'runtime.json', runtime)
    atomic_json(out/'report.json', report)
    return report


def parse_shard(argv):
    if '--shard' not in argv:
        return None
    index, count = argv[argv.index('--shard')+1].split('/')
    return int(index), int(count)


if __name__ == '__main__':
    if sys.argv[1] == 'merge':
        result = merge_shards(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
        print(json.dumps(dict(event='SHARDS_MERGED', arm=sys.argv[3], games=result['finished_games'],
                              complete=result['complete'])), flush=True)
    else:
        asyncio.run(gameplay(Path(sys.argv[1]), sys.argv[2], preflight_only='--preflight-only' in sys.argv,
                             shard=parse_shard(sys.argv)))
