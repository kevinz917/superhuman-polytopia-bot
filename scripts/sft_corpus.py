#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from polytopia_bot.learning.corpus import collect,sample
if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 c=sub.add_parser('collect');c.add_argument('--output',required=True);c.add_argument('--manifest',required=True)
 c.add_argument('--qualification',required=True);c.add_argument('--map-start',type=int,default=0)
 c.add_argument('--map-count',type=int,default=30);c.add_argument('--workers',type=int,default=4)
 s=sub.add_parser('sample');s.add_argument('--input',nargs='+',required=True);s.add_argument('--output',required=True)
 s.add_argument('--count',type=int,default=5000)
 a=p.parse_args()
 if a.command=='collect':r=collect(a.output,json.loads(Path(a.manifest).read_text()),a.qualification,map_start=a.map_start,map_count=a.map_count,workers=a.workers)
 else:r=sample(a.input,a.output,count=a.count)
 print(json.dumps(r,indent=2))
