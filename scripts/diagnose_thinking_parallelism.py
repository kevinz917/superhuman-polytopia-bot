"""Cloud-only isolation of queueing, decode throughput, and reasoning verbosity.

Fixed-length requests are synthetic speed measurements and never game actions.
Natural requests have no output/thinking cap; their complete text is retained.
"""
import asyncio
import json
import os
from pathlib import Path
import statistics
import sys
import time

from polytopia_bot.learning.async_gameplay import atomic_json
from polytopia_bot.learning.inference import decode_completion
from polytopia_bot.learning.thinking_debug import repeated_suffix
from polytopia_bot.learning.thinking_readable import action_schema


async def main(root, phase):
    if not os.environ.get('MODAL_TASK_ID'):
        raise RuntimeError('inference must run on Modal')
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams
    root = Path(root)
    p = json.loads((root/'protocol.json').read_text())
    samples = json.loads((root/'inputs.json').read_text())
    concurrency = int(phase.split('-')[-1])
    args = dict(model=p['model_id'], revision=p['revision'], tokenizer=p['model_id'],
        tokenizer_revision=p['revision'], dtype='bfloat16', max_model_len=p['max_length'],
        gpu_memory_utilization=.80, max_num_seqs=concurrency, max_num_batched_tokens=8192,
        enable_prefix_caching=False, enable_chunked_prefill=True, enforce_eager=False,
        seed=p['seed'], disable_log_stats=False, enable_log_requests=False,
        gdn_prefill_backend='triton', limit_mm_per_prompt={'image': 0, 'video': 0},
        compilation_config=dict(max_cudagraph_capture_size=concurrency),
        structured_outputs_config=dict(backend='xgrammar', reasoning_parser='qwen3',
                                       enable_in_reasoning=False))
    atomic_json(root/(phase+'-engine.json'), args)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**args))
    tokenizer = engine.get_tokenizer()
    outdir = root/phase
    outdir.mkdir()
    rows = []
    pending = {}

    async def monitor():
        while True:
            await engine.do_log_stats()
            atomic_json(root/(phase+'-progress.json'),dict(pending=pending))
            await asyncio.sleep(10)

    watcher=asyncio.create_task(monitor())

    async def request(messages, name, *, schema=None, fixed=None, sampling=None, seed=17):
        ids = tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=True)
        options = dict(p['sampling'] if sampling is None else sampling)
        options.update(max_tokens=fixed, thinking_token_budget=None, seed=seed,
                       detokenize=False, ignore_eos=fixed is not None,
                       output_kind=RequestOutputKind.DELTA)
        if fixed is not None:
            options['min_tokens'] = fixed
        if schema is not None:
            options['structured_outputs'] = StructuredOutputsParams(json=schema)
        params = SamplingParams(**options)
        started = time.monotonic()
        tokens, first, final, saved, checked, abort = [], None, None, 0, 0, None
        pending[name]=dict(input_tokens=len(ids),output_tokens=0,started_unix=time.time())

        def save():
            atomic_json(outdir/(name+'.json'), dict(name=name, messages=messages,
                input_tokens=len(ids), output_tokens=len(tokens),
                text=decode_completion(tokenizer, tokens), elapsed=time.monotonic()-started,
                response_token_ids=tokens,
                first_token_seconds=first, max_tokens=fixed, thinking_token_budget=None,
                sampling={k:v for k,v in options.items() if k not in ('structured_outputs','output_kind')},
                abort_reason=abort))

        async for out in engine.generate(dict(prompt_token_ids=ids), params, name):
            part = out.outputs[0]
            if part.token_ids and first is None:
                first = time.monotonic()-started
            tokens.extend(part.token_ids)
            pending[name]['output_tokens']=len(tokens)
            if fixed is None and len(tokens)-saved >= 512:
                save(); saved = len(tokens)
            if fixed is None and len(tokens)-checked >= 64:
                checked = len(tokens)
                period = repeated_suffix(tokens)
                if period:
                    abort = f'exact_repetition_period_{period}'
                    await engine.abort(name)
                    break
            if out.finished:
                final = part
        elapsed = time.monotonic()-started
        text = decode_completion(tokenizer, tokens)
        valid = None
        if schema:
            try:
                answer = json.loads(text.split('</think>', 1)[1])
                valid = set(answer)=={'action_id'} and answer['action_id'] in schema['properties']['action_id']['enum']
            except (ValueError, IndexError, TypeError):
                valid = False
        row = dict(name=name, input_tokens=len(ids), output_tokens=len(tokens),
            request_seconds=elapsed, first_token_seconds=first,
            decode_tokens_per_second=(len(tokens)-1)/max(elapsed-(first or 0),1e-9),
            finish_reason=abort or (final.finish_reason if final else 'missing'),
            thinking_closed='</think>' in text, valid=valid,
            max_tokens=fixed, thinking_token_budget=None)
        if fixed is None:
            save()
        pending.pop(name,None)
        return row

    try:
        # Identical 100 prompts/seeds/output lengths, only scheduler capacity differs.
        await asyncio.gather(*(request(s['messages'], 'warm-'+str(i), fixed=128)
                               for i,s in enumerate(samples)))
        started = time.monotonic()
        count=p.get('speed_requests',100)
        length=p.get('speed_output_tokens',2048)
        fixed_rows = await asyncio.gather(*(request(samples[i % len(samples)]['messages'],
            'fixed-'+str(i), fixed=length, seed=17+i) for i in range(count)))
        elapsed = time.monotonic()-started
        atomic_json(root/(phase+'-throughput.json'), dict(rows=fixed_rows, seconds=elapsed,
            output_tokens=sum(r['output_tokens'] for r in fixed_rows),
            tokens_per_second=sum(r['output_tokens'] for r in fixed_rows)/elapsed,
            median_first_token_seconds=statistics.median(r['first_token_seconds'] for r in fixed_rows),
            median_request_seconds=statistics.median(r['request_seconds'] for r in fixed_rows)))
        if not p.get('natural_probes',concurrency==40):
            return
        for i, question in enumerate(['What is 2 + 2?',
            'Choose one action in a game. You are not losing. The only legal choices are '
            'a0 = end turn, a1 = resign. Return a JSON object with action_id.']):
            row=await request([dict(role='user', content=question)],'simple-'+str(i))
            rows.append(row); atomic_json(root/'natural-results.json',rows)
        for profile in ['baseline', 'focused', 'precise_recipe']:
            async def one(i, sample):
                messages = [dict(m) for m in sample['messages']]
                sampling = None
                if profile == 'focused':
                    messages[1]['content'] += ('\n\nChoose the next single legal action now. '
                        'Use only the supplied rules and facts. The catalog is complete and already '
                        'validated; do not search for missing actions. Do not restate the board or '
                        'debate output formatting. Compare useful choices briefly, decide, and '
                        'return the action_id JSON. This decision does not end the turn unless '
                        'the selected action is end_turn.')
                if profile == 'precise_recipe':
                    sampling=dict(p['sampling'],temperature=.6,presence_penalty=0.)
                row=await request(messages, profile+'-'+str(i), schema=action_schema(sample['messages']),
                    sampling=sampling, seed=p['seed']+sample['job']['rng_index']*100000+sample['decision'])
                row.update(profile=profile,source_request=sample['source_request'])
                rows.append(row); atomic_json(root/'natural-results.json',rows)
            await asyncio.gather(*(one(i,s) for i,s in enumerate(samples)))
    finally:
        await engine.do_log_stats()
        watcher.cancel()
        await asyncio.gather(watcher,return_exceptions=True)
        engine.shutdown()


if __name__=='__main__':
    asyncio.run(main(*sys.argv[1:]))
