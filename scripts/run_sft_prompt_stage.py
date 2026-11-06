#!/usr/bin/env python3
"""Continue the bounded prompt screen only after qualification and input audits."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    qualification=ROOT/'runs/sft0/teacher-qualification-v3/report.json'
    positions=ROOT/'runs/sft0/prompt-development-v2/positions.json'
    deadline=time.monotonic()+7200
    while not qualification.exists() or not positions.exists():
        if time.monotonic()>deadline:raise TimeoutError('prerequisite jobs did not finish within two hours; no screen launched')
        time.sleep(10)
    report=json.loads(qualification.read_text())
    if report.get('teacher_qualified') is not True:
        print('Teacher qualification failed; prompt screen and SFT remain blocked.',flush=True)
        return
    code=r'''
import json
from pathlib import Path
from polytopia_bot.learning.prompt_search import STRATEGIES,candidate_messages
from polytopia_bot.learning.data import observation_from_dict
from polytopia_bot.learning.training import load_tokenizer
from polytopia_bot.learning.interface import digest
config=json.loads(Path('configs/sft.json').read_text());tokenizer=load_tokenizer(config)
positions=json.loads(Path('runs/sft0/prompt-development-v2/positions.json').read_text())
audit=[]
for name,strategy in STRATEGIES.items():
 if len(strategy)>2000:raise ValueError('candidate exceeds predeclared instruction length')
 lengths=[len(tokenizer.apply_chat_template(candidate_messages(observation_from_dict(p['observation']),strategy),tokenize=True,add_generation_prompt=True)) for p in positions]
 audit.append(dict(candidate=name,max_tokens=max(lengths),overflow=sum(n+config['max_new_tokens']>config['max_length'] for n in lengths),prompt_hash=digest(strategy)))
result=dict(positions_hash=digest(positions),config=config,candidates=audit,passed=all(r['overflow']==0 for r in audit))
Path('runs/sft0/prompt-development-v2/candidate-preflight.json').write_text(json.dumps(result,indent=2))
if not result['passed']:raise ValueError('candidate context overflow; no model calls launched')
'''
    subprocess.run([str(ROOT/'training/.venv/bin/python'),'-c',code],cwd=ROOT,check=True,
        env={**__import__('os').environ,'PYTHONPATH':str(ROOT/'src')})
    subprocess.run(['modal','run','--env','staging','experiments/modal_prompt_screen.py'],cwd=ROOT,check=True)


if __name__=='__main__':main()
