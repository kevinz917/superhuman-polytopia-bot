"""Audit corrective source capacity for a proportional 40k selection."""
import json,multiprocessing,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from sample_sft_parallel import scan

def main():
 root=Path('runs/movement-scale40k-v1');root.mkdir(exist_ok=False)
 original=json.load(open('runs/sft0/repaired-smoke-sample-v2/sampling.json'))
 paths=sorted(k for k in original['source_files'] if '/raw-relabels-v1/' in k)
 rows=[]
 with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
  for i,(name,sha,items) in enumerate(pool.map(scan,paths),1):
   if sha!=original['source_files'][name]:raise ValueError('source changed')
   rows.extend(dict(**r,collection_source='smoke_on_policy_relabel',target=t) for t,r in items)
   if i%10==0:print(json.dumps({'files':i,'rows':len(rows)}),flush=True)
 (root/'corrective-index.json').write_text(json.dumps(rows))
 print(json.dumps({'available':dict(Counter(r['family'] for r in rows))}),flush=True)
if __name__=='__main__':main()
