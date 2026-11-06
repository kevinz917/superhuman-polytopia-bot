#!/usr/bin/env python3
"""Measure every sampled input before selecting a larger context budget."""
import argparse,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.data import example
from polytopia_bot.learning.freeze import verify_prompt
from polytopia_bot.learning.interface import digest
from polytopia_bot.learning.training import load_tokenizer


def audit(sample,output,config_path,prompt_path):
    source=ROOT/sample;config=json.loads((ROOT/config_path).read_text());prompt=json.loads((ROOT/prompt_path).read_text())
    strategy=verify_prompt(prompt,config);tokenizer=load_tokenizer(config)
    sampling=json.loads((source/'sampling.json').read_text());records=source/'records.jsonl'
    sha=hashlib.sha256(records.read_bytes()).hexdigest()
    if sha!=sampling['records_sha256']:raise ValueError('sample records changed')
    rows=[]
    for index,line in enumerate(records.open()):
        record=json.loads(line);messages,response,key=example(record,strategy=strategy)
        n=len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
        rows.append(dict(index=index,episode_id=record['episode_id'],decision_index=record['decision_index'],input_tokens=n,input_hash=key))
        if len(rows)%1000==0:print(json.dumps(dict(checked=len(rows),maximum_input_tokens=max(r['input_tokens'] for r in rows))),flush=True)
    if len(rows)!=sampling['count'] or hashlib.sha256(records.read_bytes()).hexdigest()!=sha:
        raise ValueError('sample changed during audit')
    lengths=sorted(r['input_tokens'] for r in rows)
    minimum=lengths[-1]+config['max_new_tokens']+256
    report=dict(schema='sft-sample-length-audit-v1',count=len(rows),sample=str(sample),records_sha256=sha,
        config_hash=digest(config),prompt_manifest_hash=prompt['manifest_hash'],
        tokenizer_hash=digest(tokenizer.backend_tokenizer.to_str()),chat_template_hash=digest(tokenizer.chat_template),
        tokens=dict(min=lengths[0],median=statistics.median(lengths),mean=statistics.mean(lengths),p95=lengths[int(.95*len(lengths))],max=lengths[-1]),
        overflow_count=sum(n+config['max_new_tokens']>config['max_length'] for n in lengths),
        minimum_context_with_repair_headroom=minimum,suggested_power_of_two_budget=1<<(minimum-1).bit_length(),
        rows=rows,note='Exhaustive sampled-input length audit. Full response-only masks and token prefix/EOS checks remain required during final preparation; nothing is truncated or dropped.')
    with (ROOT/output).open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sample',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/sft.json'));p.add_argument('--prompt',type=Path,default=Path('runs/sft0/frozen-prompt-v1.json'))
    a=p.parse_args();audit(a.sample,a.output,a.config,a.prompt)
