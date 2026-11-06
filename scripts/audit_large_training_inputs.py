#!/usr/bin/env python3
"""Early context stress check: largest serialized public record per finished episode."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.data import observation_from_dict
from polytopia_bot.learning.interface import render,digest
from polytopia_bot.learning.training import load_tokenizer
from polytopia_bot.learning.freeze import verify_prompt
from polytopia_bot.learning.prompt_search import candidate_messages


def audit(source,output):
    source=ROOT/source;config=json.loads((ROOT/'configs/sft.json').read_text())
    prompt=json.loads((ROOT/'runs/sft0/frozen-prompt-v1.json').read_text())
    strategy=verify_prompt(prompt,config);tokenizer=load_tokenizer(config)
    games=json.loads((source/'games.json').read_text());results=[]
    for game in games:
        path=source/(game['id']+'.jsonl.gz');largest=None;largest_size=-1;decision_count=0
        for index,line in enumerate(gzip.open(path,'rt')):
            row=json.loads(line);record=row['record']
            if record is None:continue
            decision_count+=1
            size=len(json.dumps(record['observation'],separators=(',',':')))
            if size>largest_size:largest=(index,record);largest_size=size
        if largest is None:raise ValueError('episode missing teacher records')
        index,record=largest;obs=observation_from_dict(record['observation'])
        tokens=len(tokenizer.apply_chat_template(candidate_messages(obs,strategy),tokenize=True,add_generation_prompt=True))
        results.append(dict(episode=game['id'],file=str(path.relative_to(ROOT)),index=index,
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),teacher_decisions=decision_count,
            tokens=tokens,overflow=tokens+config['max_new_tokens']>config['max_length'],
            prompt_digest=digest(candidate_messages(obs,strategy))))
    report=dict(schema='large-training-input-stress-v1',checked=len(results),
        max_tokens=max(r['tokens'] for r in results),overflows=sum(r['overflow'] for r in results),results=results,
        note='One maximum-serialized-size public observation per completed episode; this proxy is not an exhaustive maximum-token audit. The final sampled corpus must still pass full token/mask auditing.')
    with (ROOT/output).open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();audit(a.source,a.output)
