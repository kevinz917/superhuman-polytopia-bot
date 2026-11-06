#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from polytopia_bot.learning.prompt_search import prepare_development
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();print(json.dumps(prepare_development(a.output,json.loads(Path(a.manifest).read_text())),indent=2))
