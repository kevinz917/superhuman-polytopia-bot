#!/usr/bin/env python3
"""Bind audited examples to the measured runtime without retokenizing data."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.interface import digest
from allocate_sft_training import allocate

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(a):
 bench=ROOT/a.benchmark;profile=json.loads((bench/'execution.json').read_text());protocol=json.loads((bench/'protocol.json').read_text())
 if (bench/'invalidation.json').exists() or profile.get('exit_code')!=0:raise ValueError('passing production-precision benchmark required')
 r=profile['benchmark']
 if not r.get('mask_free_causal_sdpa_verified'):raise ValueError('actual attention dispatch verification required')
 if r.get('measured_speedup',0)<=1:raise ValueError('measured training speedup required')
 if r.get('cuda_autocast')!='bfloat16' or not r['parity']['passed'] or r['selected']['status']!='passed':raise ValueError('benchmark did not pass')
 source=ROOT/'src/polytopia_bot/learning/efficient_training.py'
 if sha(source)!=protocol['sources'][str(source.relative_to(ROOT))]:raise ValueError('benchmarked runtime changed')
 old=ROOT/a.source;manifest=json.loads((old/'manifest.json').read_text());config=dict(manifest['config'])
 if config['model_id']!=protocol['config']['model_id'] or config['revision']!=protocol['config']['revision']:raise ValueError('benchmark model mismatch')
 config['microbatch_size']=min(r['selected']['microbatch'],config['effective_batch_size'])
 if config['effective_batch_size']%config['microbatch_size']:raise ValueError('invalid accumulation')
 if manifest['max_tokens']>r['selected']['stress_max_length']:raise ValueError('memory stress insufficient')
 dest=ROOT/a.prepared;dest.mkdir(parents=True,exist_ok=False)
 for p in old.rglob('*'):
  if p.is_file() and p.name!='manifest.json':
   q=dest/p.relative_to(old);q.parent.mkdir(parents=True,exist_ok=True);os.link(p,q)
 manifest.update(config=config,config_hash=digest(config))
 (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (dest/'runtime-rebinding.json').write_text(json.dumps(dict(source=str(a.source),source_manifest_sha256=sha(old/'manifest.json'),tokens_unchanged=sha(old/'tokens.jsonl')==sha(dest/'tokens.jsonl'),note='Only physical microbatch configuration changes; audited examples, response masks, context and effective batch are retained.'),indent=2))
 allocate(a.prepared,str(a.output),stage=a.stage,seed=17,run_id=a.run_id,seconds_limit=a.seconds_limit,gpu='B200',fit_result=a.fit_result,label_audit=a.label_audit)
 path=ROOT/a.output;p=json.loads(path.read_text());p.update(runtime_version='response-only-length-grouped-v1',benchmark=str(a.benchmark/'execution.json'))
 for file in [bench/'execution.json',bench/'protocol.json']:
  p['uploads'][str(file.relative_to(ROOT))]=sha(file);p['gate_receipts'][str(file.relative_to(ROOT))]=sha(file)
 for file in [ROOT/'experiments/modal_sft_train_efficient.py',Path(__file__)]:
  p['sources'][str(file.relative_to(ROOT))]=sha(file)
 path.write_text(json.dumps(p,indent=2))
 archive=path.with_suffix('')
 for name in p['sources']:
  q=archive/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/name).read_bytes())
 print(json.dumps(dict(prepared=str(a.prepared),microbatch=config['microbatch_size'],effective_batch=config['effective_batch_size'],stage=a.stage)))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for key in ('source','prepared','output','benchmark','label-audit'):parser.add_argument('--'+key,type=Path,required=True)
 parser.add_argument('--fit-result',type=Path);parser.add_argument('--stage',choices=['fit','smoke'],required=True)
 parser.add_argument('--run-id',required=True);parser.add_argument('--seconds-limit',type=int,required=True)
 main(parser.parse_args())
