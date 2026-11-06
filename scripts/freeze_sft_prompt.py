#!/usr/bin/env python3
"""Select a gameplay-tested prompt and freeze its rendering/evidence contract."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 s=sub.add_parser('select');s.add_argument('--games',required=True);s.add_argument('--candidates',nargs='+',required=True);s.add_argument('--output',required=True)
 f=sub.add_parser('freeze');f.add_argument('--search-games',required=True);f.add_argument('--screen-report',required=True)
 f.add_argument('--qualification',required=True);f.add_argument('--confirmation-games');f.add_argument('--output',required=True)
 r=sub.add_parser('retain-original');r.add_argument('--screen-report',required=True);r.add_argument('--qualification',required=True);r.add_argument('--output',required=True)
 a=p.parse_args()
 if a.command=='select':
  from polytopia_bot.learning.selection import select_prompt
  report=json.loads(Path(a.games).read_text())
  if not report['complete']:raise ValueError('gameplay allocation incomplete')
  result=select_prompt(report['games'],a.candidates)
  with Path(a.output).open('x') as stream:json.dump(result,stream,indent=2)
 elif a.command=='freeze':
  from polytopia_bot.learning.freeze import freeze
  result=freeze(a.output,search_games=a.search_games,screen_report=a.screen_report,qualification=a.qualification,confirmation_games=a.confirmation_games)
 else:
  from polytopia_bot.learning.freeze import retain_original
  result=retain_original(a.output,screen_report=a.screen_report,qualification=a.qualification)
 print(json.dumps(result,indent=2))
