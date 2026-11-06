"""Observable Qwen thinking probe and separately versioned repaired gameplay."""
import asyncio
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

from benchmark_thinking_unbounded import Generator, sampling_options
from polytopia_bot.learning.async_gameplay import atomic_json, play_async_games
from polytopia_bot.learning.inference import decode_completion
from polytopia_bot.learning.interface import Generation
from polytopia_bot.learning.thinking_debug import (
    render_sparse_decision, repeated_suffix, restore_dense_body, sparse_messages,
)


class ObservableGenerator(Generator):
    def __init__(self, p, thinking, root, *, probe=False):
        root.mkdir(parents=True, exist_ok=False)
        super().__init__(p, thinking, root)
        self.root, self.probe = root, probe
        (root/'streams').mkdir()

    async def generate(self, messages, *, job, decision):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=self.thinking)
        if len(ids) >= self.p['max_length']:
            return Generation('__CONTEXT_OVERFLOW__',ids,[]),dict(finish_reason='length')
        request_id = f"{job['id']}-decision-{decision}"
        options = sampling_options(self.p, job, decision)
        options['max_tokens'] = 16384 if self.probe else None
        if self.p.get('constrain_actions', False):
            from vllm.sampling_params import StructuredOutputsParams
            from polytopia_bot.learning.thinking_readable import action_schema
            options['structured_outputs'] = StructuredOutputsParams(json=action_schema(messages))
        params = SamplingParams(**options, output_kind=RequestOutputKind.DELTA)
        started = time.monotonic()
        status = dict(id=job['id'],decision=decision,input_tokens=len(ids),output_tokens=0,
                      started_unix=time.time())
        self.pending[request_id] = status
        tokens, first_token, final, abort_reason = [], None, None, None
        last_saved, last_loop_check = 0, 0
        stream = self.root/'streams'/(request_id+'.json')

        def save():
            text = decode_completion(self.tokenizer, tokens)
            atomic_json(stream, dict(**status,messages=messages,text=text,
                response_token_ids=tokens,thinking_closed='</think>' in text,
                request_seconds=time.monotonic()-started,abort_reason=abort_reason))

        try:
            async for out in self.engine.generate(dict(prompt_token_ids=ids),params,request_id):
                part = out.outputs[0]
                if part.token_ids and first_token is None:
                    first_token = time.monotonic()-started
                    status['first_token_seconds'] = first_token
                tokens.extend(part.token_ids)
                status['output_tokens'] = len(tokens)
                if len(tokens)-last_saved >= 512:
                    save(); last_saved = len(tokens)
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
            timing = dict(id=job['id'],decision=decision,input_tokens=len(ids),output_tokens=len(tokens),
                request_seconds=time.monotonic()-started,first_token_seconds=first_token,
                finish_reason='repetition_abort' if abort_reason else final.finish_reason,
                stop_reason=abort_reason if abort_reason else final.stop_reason,
                thinking_closed='</think>' in text,max_tokens=options['max_tokens'],
                thinking_token_budget=None,seed=options['seed'])
            save()
            self.log.write(json.dumps(timing)+'\n'); self.log.flush()
            print(json.dumps(dict(event='DECISION_FINISHED',mode='on' if self.thinking else 'off',**timing)),flush=True)
            return Generation(text,ids,tokens),timing
        finally:
            self.pending.pop(request_id,None)


async def probe(root):
    p=json.loads((root/'protocol.json').read_text())
    samples=json.loads((root/'probe-inputs.json').read_text())
    g=ObservableGenerator(p,True,root/'probe',probe=True)
    rows=[]
    try:
        for profile in ['original','sparse','sparse_brief']:
            async def one(sample):
                messages=sample['messages']
                if profile!='original':
                    messages=sparse_messages(messages,brief=profile=='sparse_brief')
                    assert restore_dense_body(json.loads(messages[1]['content'])) == json.loads(sample['messages'][1]['content'])
                job=dict(sample['job'],id=profile+'-'+sample['job']['id'])
                generated,timing=await g.generate(messages,job=job,decision=sample['decision'])
                valid=False
                if timing['finish_reason']=='stop' and generated.text.count('</think>')==1:
                    try:
                        action=json.loads(generated.text.split('</think>',1)[1])
                        valid=set(action)=={'action_id'} and action['action_id'] in {r[0] for r in json.loads(messages[1]['content'])['legal_actions']['rows']}
                    except (ValueError,TypeError):pass
                row=dict(profile=profile,original_request=sample['request_id'],valid=valid,**timing)
                rows.append(row);atomic_json(root/'probe-results.json',rows)
            await asyncio.gather(*(one(s) for s in samples))
    finally:
        g.log.close();g.engine.shutdown()


async def gameplay(root,mode):
    import importlib.metadata
    import jax
    import jax.numpy as jnp
    import torch
    from polytopia_bot.gpu.mapgen import layout
    from polytopia_bot.learning.interface import digest
    from polytopia_bot.learning.splits import board_hashes
    p=json.loads((root/'protocol.json').read_text())
    rules=json.loads(Path('/workspace/configs/ruleset-v1.json').read_text())
    for name,sha in {**rules['source_sha256'],**p['source_hashes']}.items():
        if hashlib.sha256((Path('/workspace')/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('source changed: '+name)
    generate=jax.jit(lambda seed:layout(seed,size=14))
    for job in p['jobs']:
        if board_hashes(jax.device_get(generate(jnp.int32(job['seed']))),14)[0]!=job['board_hash']:
            raise ValueError('map changed')
    g=ObservableGenerator(p,mode=='on',root/mode)
    receipt=dict(model_id=p['model_id'],revision=p['revision'],adapter=None,thinking=mode=='on',
        max_tokens=None,thinking_token_budget=None,max_model_len=g.engine.model_config.max_model_len,
        device=torch.cuda.get_device_name(),tokenizer_hash=digest(g.tokenizer.backend_tokenizer.to_str()),
        chat_template_hash=digest(g.tokenizer.chat_template),
        packages={k:importlib.metadata.version(k) for k in ['vllm','torch','transformers','jax']},
        serving=p.get('serving',dict(enforce_eager=True)),
        max_num_seqs=g.engine.vllm_config.scheduler_config.max_num_seqs,
        actual_cudagraph_capture_size=g.engine.vllm_config.compilation_config.max_cudagraph_capture_size,
        constrain_actions=p.get('constrain_actions',False),harness=p['harness'])
    atomic_json(root/mode/'runtime.json',receipt)
    try:
        renderer=render_sparse_decision
        if p.get('named_fields', False):
            from polytopia_bot.learning.thinking_readable import render_readable_decision
            renderer=render_readable_decision
        result=await play_async_games(root/mode/'games',p['jobs'],g,thinking=mode=='on',prompt_renderer=renderer)
        atomic_json(root/mode/'report.json',result)
    finally:
        g.log.close();g.engine.shutdown()


if __name__=='__main__':
    root=Path(sys.argv[1])
    asyncio.run(probe(root) if sys.argv[2]=='probe' else gameplay(root,sys.argv[2]))
