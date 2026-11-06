"""Versioned local VM benchmark with CLI proxy and isolated offline game workers."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from polytopia_bot.skirmish.local_v6 import cohort

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'configs/skirmish/local-v6-proxy-pilot.json')
    p.add_argument('--results',type=Path,default=ROOT/'artifacts/skirmish-v6')
    p.add_argument('--references',type=Path,default=ROOT/'artifacts/skirmish-references')
    p.add_argument('--concurrency',type=int,default=8)
    p.add_argument('--cpu-slots',type=int,default=4)
    p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args();r=cohort(args.prefix,json.loads(args.manifest.read_text()),args.results,args.references,
        concurrency=args.concurrency,cpu_slots=args.cpu_slots,evaluate_only=args.evaluate_only)
    print(json.dumps({k:r[k] for k in ('status','pilot','games','ranking')},indent=2))
