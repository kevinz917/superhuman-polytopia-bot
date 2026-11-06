"""CPU-only tokenizer/context audit for the frozen inference protocol."""
import gzip
import json
from pathlib import Path
import statistics
import sys
from transformers import AutoTokenizer
from polytopia_bot.learning.data import observation_from_dict
from polytopia_bot.learning.decision_harness import render_decision


def main(root):
    root = Path(root)
    p = json.loads((root/'protocol.json').read_text())
    rows = list(map(json.loads,gzip.open(root/'positions.jsonl.gz','rt')))
    prompts = {r['position_id']:{v:render_decision(observation_from_dict(r['observation']),v).messages
                               for v in p['versions']} for r in rows}
    result = {}
    for key in ['main','qwen35-4b','qwen35-9b']:
        spec = p['models'][key]
        tok = AutoTokenizer.from_pretrained(spec['model_id'],revision=spec['revision'])
        group = rows if key=='main' else [r for r in rows if r['position_id'] in p['screen_ids']]
        result[key] = {}
        for v in p['versions']:
            lengths = [len(tok.apply_chat_template(prompts[r['position_id']][v],tokenize=True,
                       add_generation_prompt=True,**({'enable_thinking':False} if key!='main' else {}))) for r in group]
            if max(lengths)+p['inference']['max_new_tokens'] > p['inference']['max_length']:
                raise ValueError('context overflow')
            result[key][v] = dict(count=len(lengths),mean=statistics.mean(lengths),max=max(lengths),min=min(lengths))
        print(key,result[key],flush=True)
    (root/'token-preflight.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main(sys.argv[1])
