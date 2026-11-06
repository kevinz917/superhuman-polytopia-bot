"""Unadapted Qwen3.5-4B, one native thinking toggle, real frozen gameplay."""
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from polytopia_bot.learning.decision_harness import render_decision
from polytopia_bot.learning.gameplay import play_games
from polytopia_bot.learning.interface import Generation, digest
from polytopia_bot.learning.inference import decode_completion
from polytopia_bot.learning.thinking_gameplay import parse_thinking_action


class Generator:
    def __init__(self, p, thinking, root):
        from vllm import LLM, SamplingParams
        self.p, self.thinking = p, thinking
        self.engine = LLM(model=p['model_id'], revision=p['revision'],
            tokenizer=p['model_id'], tokenizer_revision=p['revision'], dtype='bfloat16',
            max_model_len=p['max_length'], gpu_memory_utilization=.80,
            max_num_seqs=len(p['jobs']), max_num_batched_tokens=8192,
            enable_prefix_caching=False, enable_chunked_prefill=True, enforce_eager=True,
            seed=p['seed'], disable_log_stats=True, gdn_prefill_backend='triton',
            limit_mm_per_prompt={'image':0,'video':0})
        self.tokenizer = self.engine.get_tokenizer()
        self.sampling = SamplingParams(**p['sampling'], max_tokens=p['max_new_tokens'],
            stop_token_ids=[self.tokenizer.eos_token_id], detokenize=False)
        self.log = (root/'generation-timing.jsonl').open('x')
        self.requests = 0

    def batch(self, messages):
        from vllm import SamplingParams
        ids = [self.tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=self.thinking) for m in messages]
        result = [Generation('__CONTEXT_OVERFLOW__', x, []) for x in ids]
        indices = [i for i,x in enumerate(ids) if len(x)+self.p['max_new_tokens'] <= self.p['max_length']]
        if indices:
            started = time.monotonic()
            params = [SamplingParams(**self.p['sampling'], max_tokens=self.p['max_new_tokens'],
                      seed=self.p['seed']+self.requests+j,
                      stop_token_ids=[self.tokenizer.eos_token_id], detokenize=False) for j in range(len(indices))]
            outputs = self.engine.generate([dict(prompt_token_ids=ids[i]) for i in indices],params,use_tqdm=False)
            elapsed = time.monotonic()-started
            record = dict(batch=self.requests, seconds=elapsed, count=len(indices), outputs=[])
            for i,o in zip(indices,outputs):
                tokens = list(o.outputs[0].token_ids)
                text = decode_completion(self.tokenizer,tokens)
                result[i] = Generation(text,ids[i],tokens)
                m = o.metrics
                record['outputs'].append(dict(input_tokens=len(ids[i]),output_tokens=len(tokens),
                    finish_reason=o.outputs[0].finish_reason,
                    thinking_closed='</think>' in text,
                    request_seconds=None if m is None or m.finished_time is None else m.finished_time-m.arrival_time))
            self.requests += len(indices)
            self.log.write(json.dumps(record)+'\n');self.log.flush()
        return result


def main(root, mode):
    import jax
    import jax.numpy as jnp
    import torch
    from polytopia_bot.gpu.mapgen import layout
    from polytopia_bot.learning.splits import board_hashes
    root = Path(root)
    p = json.loads((root/'protocol.json').read_text())
    out = root/mode
    thinking = mode == 'on'
    if mode not in ('on','off'):
        raise ValueError('unknown mode')
    rules = json.loads(Path('/workspace/configs/ruleset-v1.json').read_text())
    for name,sha in {**rules['source_sha256'], **p['source_hashes']}.items():
        if hashlib.sha256((Path('/workspace')/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('source changed: '+name)
    generate = jax.jit(lambda seed:layout(seed,size=14))
    for job in p['jobs']:
        if board_hashes(jax.device_get(generate(jnp.int32(job['seed']))),14)[0]!=job['board_hash']:
            raise ValueError('map changed')
    start = time.monotonic()
    g = Generator(p, thinking, out)
    receipt = dict(model_id=p['model_id'],revision=p['revision'],adapter=None,thinking=thinking,
        load_seconds=time.monotonic()-start,device=torch.cuda.get_device_name(),
        tokenizer_hash=digest(g.tokenizer.backend_tokenizer.to_str()),
        chat_template_hash=digest(g.tokenizer.chat_template),
        packages={k:importlib.metadata.version(k) for k in ['vllm','torch','transformers','jax']})
    (out/'runtime.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(event='MODEL_READY',mode=mode,load_seconds=receipt['load_seconds'])),flush=True)
    result = play_games(out/'games',p['jobs'],g,batch_size=len(p['jobs']),
        seconds_limit=p['game_seconds_limit'], prompt_renderer=render_decision,
        output_parser=partial(parse_thinking_action,thinking=thinking),max_attempts=1,
        model_action_limit=p['model_decision_limit'])
    g.log.close()
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main(*sys.argv[1:])
