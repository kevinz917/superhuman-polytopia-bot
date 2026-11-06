"""Unadapted Qwen3-4B-Instruct-2507 gameplay on the Qwen3.5 paired protocol; Modal only.

Adapted from benchmark_thinking_unbounded/benchmark_thinking_repaired for a
non-thinking model: no enable_thinking, no reasoning parser (the legal-action
grammar constrains the whole response), no Qwen3.5-only serving flags.
"""
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from benchmark_thinking_unbounded import sampling_options
from polytopia_bot.learning.async_gameplay import atomic_json, play_async_games
from polytopia_bot.learning.inference import decode_completion
from polytopia_bot.learning.interface import Generation, ResponseError, digest
from polytopia_bot.learning.thinking_debug import repeated_suffix
from polytopia_bot.learning.thinking_gameplay import parse_thinking_action
from polytopia_bot.learning.thinking_readable import action_schema, render_readable_decision


class InstructGenerator:
    def __init__(self, p, root):
        if not os.environ.get('MODAL_TASK_ID'):
            raise RuntimeError('Qwen inference and model downloads must run on Modal')
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        root.mkdir(parents=True, exist_ok=False)
        (root/'streams').mkdir()
        self.p, self.root, self.thinking, self.pending = p, root, False, {}
        serving = p['serving']
        engine_args = dict(
            model=p['model_id'], revision=p['revision'], tokenizer=p['model_id'],
            tokenizer_revision=p['revision'], dtype='bfloat16',
            max_model_len=p['max_length'], gpu_memory_utilization=.80,
            max_num_seqs=p['concurrency'], max_num_batched_tokens=8192,
            enable_prefix_caching=False, enable_chunked_prefill=True,
            enforce_eager=serving['enforce_eager'],
            seed=p['seed'], disable_log_stats=not serving.get('log_stats', False), enable_log_requests=False,
            compilation_config=dict(max_cudagraph_capture_size=serving['max_cudagraph_capture_size']),
            # No reasoning parser: Instruct-2507 never emits <think>, so the grammar
            # must apply from the first generated token.
            structured_outputs_config=dict(backend='xgrammar'))
        self.engine_args = {k: v for k, v in engine_args.items() if k not in ('model', 'tokenizer')}
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_args))
        ceiling = self.engine.vllm_config.compilation_config.max_cudagraph_capture_size
        if serving.get('require_full_decode_graph_coverage') and (ceiling is None or ceiling < p['concurrency']):
            self.engine.shutdown()
            raise ValueError(f'CUDA graph ceiling {ceiling} does not cover scheduler capacity {p["concurrency"]}')
        self.tokenizer = self.engine.get_tokenizer()
        self.log = (root/'generation-timing.jsonl').open('x')

    def prompt_ids(self, messages):
        # Deliberately no enable_thinking kwarg: Instruct-2507 is non-thinking only.
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    async def generate(self, messages, *, job, decision, record=True, seed=None):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams
        ids = self.prompt_ids(messages)
        if len(ids) >= self.p['max_length']:
            return Generation('__CONTEXT_OVERFLOW__', ids, []), dict(finish_reason='length')
        request_id = f"{job['id']}-decision-{decision}"
        options = sampling_options(self.p, job, decision)
        if seed is not None:
            options['seed'] = seed
        options['structured_outputs'] = StructuredOutputsParams(json=action_schema(messages))
        params = SamplingParams(**options, output_kind=RequestOutputKind.DELTA)
        if params.max_tokens is not None or params.thinking_token_budget is not None:
            raise ValueError('unexpected token budget')
        started = time.monotonic()
        status = dict(id=job['id'], decision=decision, input_tokens=len(ids), output_tokens=0,
                      started_unix=time.time())
        if record:
            self.pending[request_id] = status
        tokens, first_token, final, abort_reason, last_loop_check = [], None, None, None, 0
        try:
            async for out in self.engine.generate(dict(prompt_token_ids=ids), params, request_id):
                part = out.outputs[0]
                if part.token_ids and first_token is None:
                    first_token = time.monotonic()-started
                tokens.extend(part.token_ids)
                status['output_tokens'] = len(tokens)
                # Same exact-repetition guard as the Qwen3.5 arms (inert for tiny answers).
                if len(tokens)-last_loop_check >= 64:
                    last_loop_check = len(tokens)
                    period = repeated_suffix(tokens)
                    if period:
                        abort_reason = f'exact_repetition_period_{period}'
                        await self.engine.abort(request_id)
                        break
                if out.finished:
                    final = part
            if final is None and abort_reason is None:
                raise RuntimeError('missing completion metadata')
            text = decode_completion(self.tokenizer, tokens)
            timing = dict(id=job['id'], decision=decision, input_tokens=len(ids), output_tokens=len(tokens),
                request_seconds=time.monotonic()-started, first_token_seconds=first_token,
                finish_reason='repetition_abort' if abort_reason else final.finish_reason,
                stop_reason=abort_reason if abort_reason else final.stop_reason,
                thinking_closed='</think>' in text, max_tokens=options['max_tokens'],
                thinking_token_budget=None, seed=options['seed'])
            if record:
                atomic_json(self.root/'streams'/(request_id+'.json'), dict(**timing, text=text,
                    response_token_ids=tokens))
                self.log.write(json.dumps(timing)+'\n'); self.log.flush()
                print(json.dumps(dict(event='DECISION_FINISHED', mode='instruct', **timing)), flush=True)
            return Generation(text, ids, tokens), timing
        finally:
            self.pending.pop(request_id, None)


async def preflight(g, job):
    """Gate before any game: template has no think markers, grammar applies from token 0."""
    from polytopia_bot.learning.adapter import GPUEnvironment
    env = GPUEnvironment(14)
    try:
        obs = env.reset(seed=job['seed'], settings={})
        prompt = render_readable_decision(obs)
    finally:
        env.close()
    ids = g.prompt_ids(prompt.messages)
    rendered = g.tokenizer.decode(ids)
    if '<think>' in rendered or '</think>' in rendered:
        raise ValueError('chat template emitted thinking markers')
    if not rendered.endswith('<|im_start|>assistant\n'):
        raise ValueError('unexpected generation prompt: '+repr(rendered[-80:]))
    rows = []
    for i in range(4):
        # Seeds far outside the game formula; these samples never touch a game.
        generated, timing = await g.generate(prompt.messages, job=dict(id=f'preflight-{i}', rng_index=0),
            decision=0, record=False, seed=990000+i)
        try:
            action = parse_thinking_action(generated.text, prompt, current_version=obs.state_version, thinking=False)
            valid, chosen = True, action.action_id
        except ResponseError as exc:
            valid, chosen = False, str(exc)
        rows.append(dict(text=generated.text, valid=valid, action=chosen, **timing))
    result = dict(prompt_tokens=len(ids), prompt_tail=rendered[-200:], samples=rows,
                  passed=all(r['valid'] and r['finish_reason'] == 'stop' and not r['thinking_closed'] for r in rows))
    if not result['passed']:
        raise ValueError('preflight failed: '+json.dumps(rows))
    return result


async def gameplay(root):
    import jax
    import jax.numpy as jnp
    import torch
    from polytopia_bot.gpu.mapgen import layout
    from polytopia_bot.learning.splits import board_hashes
    p = json.loads((root/'protocol.json').read_text())
    if p['max_new_tokens'] is not None or p['thinking'] is not False:
        raise ValueError('invalid instruct protocol')
    rules = json.loads(Path('/workspace/configs/ruleset-v1.json').read_text())
    for name, sha in {**rules['source_sha256'], **p['source_hashes']}.items():
        if hashlib.sha256((Path('/workspace')/name).read_bytes()).hexdigest() != sha:
            raise ValueError('source changed: '+name)
    generate = jax.jit(lambda seed: layout(seed, size=14))
    for job in p['jobs']:
        if board_hashes(jax.device_get(generate(jnp.int32(job['seed']))), 14)[0] != job['board_hash']:
            raise ValueError('map changed')
    out = root/'instruct'
    started = time.monotonic()
    g = InstructGenerator(p, out)
    try:
        receipt = dict(model_id=p['model_id'], revision=p['revision'], adapter=None, thinking=False,
            enable_thinking_passed=False, reasoning_parser=None,
            load_seconds=time.monotonic()-started, max_tokens=None, thinking_token_budget=None,
            max_model_len=g.engine.model_config.max_model_len,
            device=torch.cuda.get_device_name(), tokenizer_hash=digest(g.tokenizer.backend_tokenizer.to_str()),
            chat_template_hash=digest(g.tokenizer.chat_template),
            packages={k: importlib.metadata.version(k) for k in ['vllm', 'torch', 'transformers', 'jax', 'xgrammar']},
            serving=p['serving'], engine_args=g.engine_args,
            max_num_seqs=g.engine.vllm_config.scheduler_config.max_num_seqs,
            actual_cudagraph_capture_size=g.engine.vllm_config.compilation_config.max_cudagraph_capture_size,
            cudagraph_capture_sizes=getattr(g.engine.vllm_config.compilation_config, 'cudagraph_capture_sizes', None),
            structured_outputs_config=str(getattr(g.engine.vllm_config, 'structured_outputs_config', None)),
            constrain_actions=True, harness=p['harness'], serving_changes=p['serving_changes'])
        if receipt['max_model_len'] != p['max_length']:
            raise ValueError('native context not available')
        receipt['preflight'] = await preflight(g, p['jobs'][0])
        atomic_json(out/'runtime.json', receipt)
        print(json.dumps(dict(event='MODEL_READY', **{k: v for k, v in receipt.items() if k != 'preflight'})), flush=True)
        result = await play_async_games(out/'games', p['jobs'], g, thinking=False,
                                        prompt_renderer=render_readable_decision)
        atomic_json(out/'report.json', result)
    finally:
        g.log.close(); g.engine.shutdown()


if __name__ == '__main__':
    asyncio.run(gameplay(Path(sys.argv[1])))
