"""Keep frozen position inference while moving shard ownership to the cloud."""
import argparse,hashlib,json
from pathlib import Path
from allocate_position_evaluation_v2 import allocate,ROOT

def main(a):
    allocate(a.output,a.run_id,adapter_result=a.adapter_result,seconds_per_shard=a.seconds_per_shard)
    path=ROOT/a.output;p=json.loads(path.read_text());p['cloud_coordinator']=True
    p['planning_compute_allowance_usd']+=round((a.seconds_per_shard+240)*.00001532,3)
    for file in (ROOT/'experiments/modal_position_eval_coordinated.py',Path(__file__)):
        name=str(file.relative_to(ROOT));p['sources'][name]=hashlib.sha256(file.read_bytes()).hexdigest()
        q=path.with_suffix('')/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(file.read_bytes())
    path.write_text(json.dumps(p,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--adapter-result',type=Path,required=True);p.add_argument('--seconds-per-shard',type=int,default=1800)
    main(p.parse_args())
