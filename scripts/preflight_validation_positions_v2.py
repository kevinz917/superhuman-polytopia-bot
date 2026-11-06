#!/usr/bin/env python3
import argparse,gzip,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.data import observation_from_dict
from polytopia_bot.learning.freeze import verify_prompt
from polytopia_bot.learning.interface import digest
from polytopia_bot.learning.prompt_search import candidate_messages
from polytopia_bot.learning.training import load_tokenizer
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 source=ROOT/a.input;root=ROOT/a.output;root.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((source/'manifest.json').read_text())
 assert manifest['count']==5000 and manifest['schema']=='validation-positions-v1'
 assert hashlib.sha256((source/'positions.jsonl.gz').read_bytes()).hexdigest()==manifest['positions_sha256']
 config=json.loads((ROOT/'runs/sft0/context-v2/config.json').read_text());frozen=json.loads((ROOT/'runs/sft0/context-v2/frozen-prompt.json').read_text())
 strategy=verify_prompt(frozen,config);tokenizer=load_tokenizer(config);rows=[]
 for line in gzip.open(source/'positions.jsonl.gz','rt'):
  row=json.loads(line);messages=candidate_messages(observation_from_dict(row['observation']),strategy)
  n=len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
  if n+config['max_new_tokens']+256>config['max_length']:raise ValueError('context overflow including reserved repair headroom')
  row.update(prompt_digest=digest(messages),prompt_tokens=n);rows.append(row)
 assert len(rows)==5000 and len({r['position_id'] for r in rows})==5000
 rows.sort(key=lambda r:(r['prompt_tokens'],r['position_id']))
 files={}
 for shard in range(4):
  path=root/f'shard-{shard}.jsonl.gz'
  with gzip.open(path,'xt',compresslevel=1) as stream:
   for row in rows[shard::4]:stream.write(json.dumps(row,separators=(',',':'))+'\n')
  files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 lengths=[r['prompt_tokens'] for r in rows]
 report=dict(schema='validation-position-preflight-v1',count=5000,shards=4,source_manifest=manifest,
  prompt_manifest=frozen,config=config,files=files,tokenizer_hash=digest(tokenizer.backend_tokenizer.to_str()),
  chat_template_hash=digest(tokenizer.chat_template),passed=True,repair_headroom_tokens=256,
  tokens=dict(min=min(lengths),median=statistics.median(lengths),mean=statistics.mean(lengths),p95=lengths[int(.95*len(lengths))],max=max(lengths)),
  batch_policy='Sort frozen public prompt lengths then position ID; round-robin into four equal shards; batch eight within each shard. Same order for base and smoke.')
 (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('source_manifest','prompt_manifest','config')},indent=2))
