#!/usr/bin/env python3
"""Freeze the shared pipelined base/SFT gameplay protocol after workload parity."""
import argparse,hashlib,json
from pathlib import Path
from allocate_sft_gameplay_v2 import allocate
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(a):
 root=ROOT/'runs/sft0/efficiency-rollout-benchmark-v1';r=json.loads((root/'execution.json').read_text());p=json.loads((root/'protocol.json').read_text());c=r.get('comparison') or {}
 if r.get('exit_code')!=0 or not c.get('exact_inputs_and_generations_match') or not c.get('steps_match') or c.get('speedup',0)<=1:
  raise ValueError('passing measured rollout improvement required')
 for name,digest in p['sources'].items():
  if sha(ROOT/name)!=digest:raise ValueError('benchmarked pipeline source changed')
 allocate(a.output,a.run_id,adapter_result=a.adapter_result,seconds_per_shard=a.seconds_per_shard)
 path=ROOT/a.output;r=json.loads(path.read_text());r.update(runtime_version='pipelined-cpu-lanes-v1',cpu_lanes_per_gpu=2,active_games_max=50,batch_policy='Four fixed GPU shards; each original job list split into two fixed CPU lanes by alternating index; batch up to8 per lane; lane requests queued without merging. All50 games active.',gpu_memory_mib=24576,cloud_coordinator=True)
 r['planning_compute_allowance_usd']=round(4*(a.seconds_per_shard+120)*.00079968+(a.seconds_per_shard+240)*.00001532,2)
 r['prerequisites'] += [dict(path=str(x.relative_to(ROOT)),sha256=sha(x)) for x in [root/'execution.json',root/'protocol.json']]
 for x in [ROOT/'experiments/modal_prompt_games_efficient.py',Path(__file__)]:r['sources'][str(x.relative_to(ROOT))]=sha(x)
 path.write_text(json.dumps(r,indent=2));archive=path.with_suffix('')
 for name in r['sources']:
  out=archive/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((ROOT/name).read_bytes())
 print(json.dumps(dict(runtime=r['runtime_version'],kind=r['kind'],compute_allowance=r['planning_compute_allowance_usd'])))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--run-id',required=True);p.add_argument('--adapter-result',type=Path);p.add_argument('--seconds-per-shard',type=int,default=3600);main(p.parse_args())
