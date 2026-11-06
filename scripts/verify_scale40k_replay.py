"""Prove all original 10k prepared sequences survive unchanged in the 40k corpus."""
import gzip,hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.training import file_hash
from polytopia_bot.learning.scale_training import verify_scale,BOUNDARIES
from polytopia_bot.learning.efficient_training import EffectiveBatchLengthSampler
from polytopia_bot.learning.interface import digest

def main():
 root=ROOT/'runs/movement-scale40k-v1';old=ROOT/'runs/sft0/prepared-efficient-smoke-v1';parent=json.load(open(ROOT/'runs/movement-epoch2-v1/training-run.json'));m=json.load(open(old/'manifest.json'))
 if m!=parent['data_manifest'] or file_hash(old/'tokens.jsonl')!=m['tokens_sha256']:raise ValueError('original prepared corpus changed')
 expected={}
 with (old/'examples.jsonl').open() as examples,(old/'tokens.jsonl').open('rb') as tokens:
  for a,b in zip(examples,tokens,strict=True):
   e=json.loads(a);key=(e['episode_id'],e['decision_index']);expected[key]=hashlib.sha256(b).hexdigest()
 found=set();total=0;lengths=[];token_hash=hashlib.sha256()
 with gzip.open(root/'prepared/examples.jsonl.gz','rt') as examples,gzip.open(root/'prepared/tokens.jsonl.gz','rb') as tokens:
  for a,b in zip(examples,tokens,strict=True):
   e=json.loads(a);key=(e['episode_id'],e['decision_index']);total+=1
   lengths.append(len(json.loads(b)['input_ids']));token_hash.update(b)
   if key in expected:
    if key in found or hashlib.sha256(b).hexdigest()!=expected[key]:raise ValueError('original tokens changed or repeated')
    found.add(key)
 if len(found)!=10000 or total!=40000:raise ValueError('replay count differs')
 manifest=json.load(open(root/'prepared/manifest.json'));protocol=json.load(open(root/'protocol.json'))
 if token_hash.hexdigest()!=manifest['tokens_sha256']:raise ValueError('compressed token stream changed')
 scale={**protocol['training']['scale'],'parent_run':str(ROOT/'runs/movement-epoch2-v1/training-run.json')}
 verify_scale(root/'prepared',manifest,scale,parent['config'],'scale40k')
 order=list(EffectiveBatchLengthSampler(lengths,seed=17,effective_batch_size=64))
 selected=json.load(open(root/'selection.json'))['selected']
 exposure={str(n):dict(examples=n,movement=sum(selected[i]['family']==1 for i in order[:n]),families=dict(Counter(str(selected[i]['family']) for i in order[:n])),sources=dict(Counter(selected[i]['collection_source'] for i in order[:n]))) for n in BOUNDARIES}
 out=dict(passed=True,unchanged_original_sequences=len(found),total_sequences=total,parent_tokens_sha256=m['tokens_sha256'],prepared_manifest_sha256=file_hash(root/'prepared/manifest.json'),epoch_order_sha256=digest(order),checkpoint_composition=exposure)
 (root/'replay-integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
