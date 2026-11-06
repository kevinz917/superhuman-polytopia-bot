"""One evaluation arm of the compact-prompt ablation (Modal only).

A copy of the expert-sft-v1 arm runner (``benchmark_expert_sft.py`` at commit 5cde6dc)
with exactly two differences: the prompt is ``render_compact_decision`` (and its
legal-action grammar), and an arm may opt into vLLM prefix caching
(``spec['prefix_caching']``) to measure what the prefix-friendly order buys. Sampling,
per-decision seeds, engine settings, LoRA serving and the no-repair single call are
unchanged.
"""
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_qwen3_instruct as instruct  # noqa: E402
from polytopia_bot.learning.async_gameplay import atomic_json, play_async_games  # noqa: E402
from polytopia_bot.learning.compact_prompt import action_schema, render_compact_decision  # noqa: E402
from polytopia_bot.learning.interface import digest  # noqa: E402

# InstructGenerator.generate builds its grammar with this module-level name, and
# preflight renders with it; point both at the compact prompt.
instruct.action_schema = action_schema
instruct.render_readable_decision = render_compact_decision


class _LoraEngine:
    """Delegate to AsyncLLM, attaching one fixed LoRA request to every generation."""
    def __init__(self, engine, lora):
        self._engine, self._lora = engine, lora

    def generate(self, prompt, params, request_id):
        return self._engine.generate(prompt, params, request_id, lora_request=self._lora)

    def __getattr__(self, name):
        return getattr(self._engine, name)


class ArmGenerator(instruct.InstructGenerator):
    def __init__(self, p, root, adapter, prefix_caching=False):
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
        engine_args = dict(
            model=p['model_id'], revision=p['revision'], tokenizer=p['model_id'],
            tokenizer_revision=p['revision'], dtype='bfloat16',
            max_model_len=p['max_length'], gpu_memory_utilization=.80,
            max_num_seqs=p['concurrency'], max_num_batched_tokens=8192,
            enable_prefix_caching=prefix_caching, enable_chunked_prefill=True,
            enforce_eager=serving['enforce_eager'],
            seed=p['seed'], disable_log_stats=True, enable_log_requests=False,
            compilation_config=dict(max_cudagraph_capture_size=serving['max_cudagraph_capture_size']),
            structured_outputs_config=dict(backend='xgrammar'),
            enable_lora=True, max_lora_rank=p['lora']['max_rank'], max_loras=1)
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


def adapter_hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(path).glob('adapter_*'))}


async def gameplay(root, arm, preflight_only=False):
    import jax
    import jax.numpy as jnp
    import torch
    from polytopia_bot.gpu.mapgen import layout
    from polytopia_bot.learning.splits import board_hashes
    p = json.loads((root/'protocol.json').read_text())
    spec = p['arms'][arm]
    rules = json.loads(Path('/workspace/configs/ruleset-v1.json').read_text())
    for name, sha in {**rules['source_sha256'], **p['source_hashes']}.items():
        if hashlib.sha256((Path('/workspace')/name).read_bytes()).hexdigest() != sha:
            raise ValueError('source changed: '+name)
    generate = jax.jit(lambda seed: layout(seed, size=14))
    for job in p['jobs']:
        if board_hashes(jax.device_get(generate(jnp.int32(job['seed']))), 14)[0] != job['board_hash']:
            raise ValueError('map changed')
    path = Path('/workspace/runs')/spec['adapter']
    expected = spec['adapter_files']
    if expected is None:
        run = json.loads((Path('/workspace/runs')/spec['training_receipt']).read_text())
        if run['status'] != 'completed':
            raise ValueError('training not completed')
        expected = run['adapter_files']
    spec = dict(spec, adapter_files=expected)
    if adapter_hashes(path) != expected:
        raise ValueError('adapter files changed')
    cfg = json.loads((path/'adapter_config.json').read_text())
    if cfg['base_model_name_or_path'] != p['model_id'] or cfg['r'] > p['lora']['max_rank']:
        raise ValueError('adapter does not match base model / rank limit')
    adapter = dict(name=arm, path=str(path))
    out = root/arm
    started = time.monotonic()
    g = ArmGenerator(p, out, adapter, prefix_caching=spec.get('prefix_caching', False))
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
        if receipt['max_model_len'] != p['max_length']:
            raise ValueError('native context not available')
        receipt['preflight'] = await instruct.preflight(g, p['jobs'][0])
        # Compact legal lines end the prompt ("resign: a107"); canonical readable JSON has no ': a'.
        if ': a' not in receipt['preflight']['prompt_tail']:
            raise ValueError('preflight did not use the compact prompt')
        atomic_json(out/'runtime.json', receipt)
        if preflight_only:
            print(json.dumps(dict(event='PREFLIGHT_ONLY', arm=arm, preflight=receipt['preflight'])), flush=True)
            return
        print(json.dumps(dict(event='MODEL_READY', arm=arm)), flush=True)
        result = await play_async_games(out/'games', p['jobs'], g, thinking=False, prompt_renderer=render_compact_decision)
        atomic_json(out/'report.json', result)
    finally:
        g.log.close(); g.engine.shutdown()


if __name__ == '__main__':
    asyncio.run(gameplay(Path(sys.argv[1]), sys.argv[2], preflight_only='--preflight-only' in sys.argv))
