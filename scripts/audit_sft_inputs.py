#!/usr/bin/env python3
"""Audit retained public traces under the current renderer and pinned tokenizer."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from polytopia_bot.learning.data import observation_from_dict, tokenize_example
from polytopia_bot.learning.encoding import VERSION, expand
from polytopia_bot.learning.interface import canonical, digest, render
from polytopia_bot.learning.training import load_tokenizer, read_config


def audit(paths, output):
    config = read_config('configs/sft.json')
    tokenizer = load_tokenizer(config)
    lengths, families, bad, masks, files = [], Counter(), [], [], {}
    for path in paths:
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                obs = observation_from_dict(row['observation'])
                prompt = render(obs)
                assert expand(json.loads(prompt.messages[1]['content'])) == prompt.public
                chosen = row['selection']['action']['command'] if 'selection' in row else row['selected_command']
                ids = [k for k, a in prompt.catalog.items() if dict(a.command) == chosen]
                if len(ids) != 1:
                    raise ValueError('recorded choice is not legal')
                response = canonical(dict(action_id=ids[0]))
                size = len(tokenizer.apply_chat_template(prompt.messages, tokenize=True, add_generation_prompt=True))
                lengths.append(size)
                family = str(chosen['kind'])
                families[family] += 1
                if size + config['max_new_tokens'] > config['max_length']:
                    bad.append(dict(file=str(path), index=index, tokens=size))
                elif len(masks) < 64 or family not in {m['family'] for m in masks}:
                    tokenized = tokenize_example(tokenizer, prompt.messages, response,
                        max_length=config['max_length'], max_new_tokens=config['max_new_tokens'])
                    assert all(label == -100 for label in tokenized['labels'][:size])
                    assert tokenized['labels'][-1] == tokenizer.eos_token_id
                    masks.append(dict(file=str(path), index=index, family=family, input_hash=digest(prompt.messages),
                        supervised=sum(x != -100 for x in tokenized['labels']), response=response))
    report = dict(encoding=VERSION, config=config, tokenizer_hash=digest(tokenizer.backend_tokenizer.to_str()),
        chat_template_hash=digest(tokenizer.chat_template), files=files, decisions=len(lengths),
        prompt_tokens=dict(min=min(lengths), median=float(np.median(lengths)), p95=float(np.percentile(lengths,95)),
            max=max(lengths), mean=float(np.mean(lengths))), overflow_count=len(bad), overflows=bad,
        lossless_roundtrips=len(lengths), action_families=dict(families), mask_audits=masks,
        passed=not bad, note='Mechanical input/mask audit, not manual label quality or a trained model result.')
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','overflows','mask_audits','config')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    paths=[]
    for directory in args.input:
        paths.extend(sorted(Path(directory).glob('*.jsonl')))
        paths.extend(sorted(Path(directory).glob('*.jsonl.gz')))
    audit(paths,args.output)
