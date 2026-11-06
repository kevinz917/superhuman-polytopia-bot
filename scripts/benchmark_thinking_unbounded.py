"""Unadapted Qwen gameplay: native EOS/context boundary, no output allowance."""
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from polytopia_bot.learning.async_gameplay import atomic_json, play_async_games
from polytopia_bot.learning.inference import decode_completion
from polytopia_bot.learning.interface import Generation, digest


def sampling_options(p, job, decision):
    # Explicit None is essential: omitting max_tokens would default to 16.
    return dict(**p['sampling'], max_tokens=None, thinking_token_budget=None,
                seed=p['seed']+job['rng_index']*100000+decision,
                detokenize=False, ignore_eos=False)


class Generator:
    def __init__(self, p, thinking, root):
        if not os.environ.get('MODAL_TASK_ID'):
            raise RuntimeError('Qwen inference and model downloads must run on Modal')
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        self.p, self.thinking, self.pending = p, thinking, {}
        serving = p.get('serving', {})
        extra = {}
        if not serving.get('enforce_eager', True):
            extra['compilation_config'] = dict(max_cudagraph_capture_size=
                serving.get('max_cudagraph_capture_size',p['concurrency']))
        if p.get('constrain_actions', False):
            extra['structured_outputs_config'] = dict(
                backend='xgrammar', reasoning_parser='qwen3', enable_in_reasoning=False)
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=p['model_id'], revision=p['revision'], tokenizer=p['model_id'],
            tokenizer_revision=p['revision'], dtype='bfloat16',
            max_model_len=p['max_length'], gpu_memory_utilization=.80,
            max_num_seqs=p['concurrency'], max_num_batched_tokens=8192,
            enable_prefix_caching=False, enable_chunked_prefill=True,
            enforce_eager=serving.get('enforce_eager', True),
            seed=p['seed'], disable_log_stats=not serving.get('log_stats',False), enable_log_requests=False,
            gdn_prefill_backend='triton', limit_mm_per_prompt={'image':0,'video':0}, **extra))
        if serving.get('require_full_decode_graph_coverage',False):
            ceiling=self.engine.vllm_config.compilation_config.max_cudagraph_capture_size
            if ceiling < p['concurrency']:
                self.engine.shutdown()
                raise ValueError(f'CUDA graph ceiling {ceiling} does not cover scheduler capacity {p["concurrency"]}')
        self.tokenizer = self.engine.get_tokenizer()
        self.log = (root/'generation-timing.jsonl').open('x')

    async def generate(self, messages, *, job, decision):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=self.thinking)
        if len(ids) >= self.p['max_length']:
            return Generation('__CONTEXT_OVERFLOW__', ids, []), dict(finish_reason='length')
        request_id = f"{job['id']}-decision-{decision}"
        options = sampling_options(self.p, job, decision)
        params = SamplingParams(**options, output_kind=RequestOutputKind.DELTA)
        if params.max_tokens is not None or params.thinking_token_budget is not None:
            raise ValueError('unexpected token budget')
        started = time.monotonic()
        status = dict(id=job['id'], decision=decision, input_tokens=len(ids), output_tokens=0,
                      started_unix=time.time())
        self.pending[request_id] = status
        tokens, final, first_token = [], None, None
        try:
            async for out in self.engine.generate(dict(prompt_token_ids=ids), params, request_id):
                part = out.outputs[0]
                if part.token_ids and first_token is None:
                    first_token = time.monotonic()-started
                tokens.extend(part.token_ids)
                status['output_tokens'] = len(tokens)
                if out.finished:
                    final = part
            if final is None:
                raise RuntimeError('generation ended without completion metadata')
            text = decode_completion(self.tokenizer, tokens)
            timing = dict(id=job['id'], decision=decision, input_tokens=len(ids), output_tokens=len(tokens),
                request_seconds=time.monotonic()-started, first_token_seconds=first_token,
                finish_reason=final.finish_reason, stop_reason=final.stop_reason,
                thinking_closed='</think>' in text, max_tokens=None, thinking_token_budget=None,
                seed=options['seed'])
            self.log.write(json.dumps(timing)+'\n'); self.log.flush()
            print(json.dumps(dict(event='DECISION_FINISHED', mode='on' if self.thinking else 'off',
                                  **timing)), flush=True)
            return Generation(text, ids, tokens), timing
        finally:
            self.pending.pop(request_id, None)


async def main(root, mode):
    import jax
    import jax.numpy as jnp
    import torch
    from polytopia_bot.gpu.mapgen import layout
    from polytopia_bot.learning.splits import board_hashes
    root = Path(root)
    p = json.loads((root/'protocol.json').read_text())
    out = root/mode
    if mode not in ('on','off') or p['max_new_tokens'] is not None:
        raise ValueError('invalid unrestricted protocol')
    rules = json.loads(Path('/workspace/configs/ruleset-v1.json').read_text())
    for name, sha in {**rules['source_sha256'], **p['source_hashes']}.items():
        if hashlib.sha256((Path('/workspace')/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('source changed: '+name)
    generate = jax.jit(lambda seed:layout(seed,size=14))
    for job in p['jobs']:
        if board_hashes(jax.device_get(generate(jnp.int32(job['seed']))),14)[0]!=job['board_hash']:
            raise ValueError('map changed')
    start = time.monotonic()
    g = Generator(p, mode=='on', out)
    receipt = dict(model_id=p['model_id'],revision=p['revision'],adapter=None,thinking=mode=='on',
        load_seconds=time.monotonic()-start,device=torch.cuda.get_device_name(),
        max_model_len=g.engine.model_config.max_model_len,max_tokens=None,thinking_token_budget=None,
        tokenizer_hash=digest(g.tokenizer.backend_tokenizer.to_str()),
        chat_template_hash=digest(g.tokenizer.chat_template),
        packages={k:importlib.metadata.version(k) for k in ['vllm','torch','transformers','jax']})
    if receipt['max_model_len'] != p['max_length']:
        raise ValueError('native context not available')
    atomic_json(out/'runtime.json', receipt)
    print(json.dumps(dict(event='MODEL_READY', mode=mode, **receipt)), flush=True)
    try:
        result = await play_async_games(out/'games',p['jobs'],g,thinking=mode=='on')
        atomic_json(out/'report.json', result)
    finally:
        g.log.close()
        g.engine.shutdown()


if __name__=='__main__':
    asyncio.run(main(*sys.argv[1:]))
