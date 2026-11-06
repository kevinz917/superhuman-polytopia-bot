"""Pinned vLLM evaluation only. There is deliberately no optimizer/training path."""
import gzip
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from polytopia_bot.learning.harness_comparison import evaluate_pairs
from polytopia_bot.learning.interface import Generation, digest
from polytopia_bot.learning.inference import decode_completion


class Generator:
    def __init__(self, spec, config):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        self.config = config
        adapter = spec.get('adapter')
        if adapter:
            for name, sha in spec['adapter_files'].items():
                if hashlib.sha256((Path(adapter)/name).read_bytes()).hexdigest() != sha:
                    raise ValueError('adapter changed')
            saved = json.loads((Path(adapter)/'adapter_config.json').read_text())
            if saved['base_model_name_or_path'] != spec['model_id'] or saved['revision'] != spec['revision']:
                raise ValueError('adapter base/revision mismatch')
        self.engine = LLM(model=spec['model_id'], revision=spec['revision'],
            tokenizer=spec['model_id'], tokenizer_revision=spec['revision'],
            dtype='bfloat16', max_model_len=config['max_length'], gpu_memory_utilization=.80,
            max_num_seqs=config['batch_size'], max_num_batched_tokens=8192,
            enable_prefix_caching=False, enable_chunked_prefill=True, enforce_eager=True,
            enable_lora=bool(adapter), max_lora_rank=32, seed=17, disable_log_stats=True,
            limit_mm_per_prompt={'image': 0, 'video': 0} if spec.get('multimodal') else {})
        self.tokenizer = self.engine.get_tokenizer()
        self.adapter = LoRARequest('frozen', 1, adapter) if adapter else None
        self.sampling = SamplingParams(temperature=0, max_tokens=config['max_new_tokens'],
            stop_token_ids=[self.tokenizer.eos_token_id], detokenize=False)
        self.template_kwargs = {'enable_thinking': False} if spec.get('multimodal') else {}

    def batch(self, messages):
        ids = [self.tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True,
                                                 **self.template_kwargs) for m in messages]
        if any(len(x)+self.config['max_new_tokens'] > self.config['max_length'] for x in ids):
            raise ValueError('context overflow: no truncation permitted')
        outputs = self.engine.generate([dict(prompt_token_ids=x) for x in ids],
            self.sampling, lora_request=self.adapter, use_tqdm=False)
        return [Generation(decode_completion(self.tokenizer, list(o.outputs[0].token_ids)), x,
                           list(o.outputs[0].token_ids)) for x, o in zip(ids, outputs)]


def main(root, key):
    root = Path(root)
    p = json.loads((root/'protocol.json').read_text())
    if hashlib.sha256((root/'positions.jsonl.gz').read_bytes()).hexdigest() != p['positions_sha256']:
        raise ValueError('positions changed')
    spec = p['models'][key]
    rows = list(map(json.loads, gzip.open(root/'positions.jsonl.gz', 'rt')))
    if spec['cohort'] == 'screen':
        screen = set(p['screen_ids'])
        rows = [r for r in rows if r['position_id'] in screen]
    started = time.monotonic()
    g = Generator(spec, p['inference'])
    load_seconds = time.monotonic()-started
    # Warm up both representations equally outside the timed paired batches.
    from polytopia_bot.learning.data import observation_from_dict
    from polytopia_bot.learning.decision_harness import render_decision
    warm = observation_from_dict(rows[0]['observation'])
    g.batch([render_decision(warm, v).messages for v in p['versions']])
    report = evaluate_pairs(root/key/'results', rows, g, p['versions'],
        batch_size=p['inference']['batch_size'], seconds_limit=p['inference']['seconds_limit'])
    runtime = dict(load_seconds=load_seconds, model=spec, config=p['inference'],
        tokenizer_hash=digest(g.tokenizer.backend_tokenizer.to_str()),
        chat_template_hash=digest(g.tokenizer.chat_template), template_kwargs=g.template_kwargs,
        versions={k: importlib.metadata.version(k) for k in ['torch','transformers','vllm','peft']})
    (root/key/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
    if not report['complete']:
        raise RuntimeError('incomplete evaluation')


if __name__ == '__main__':
    main(*sys.argv[1:])
