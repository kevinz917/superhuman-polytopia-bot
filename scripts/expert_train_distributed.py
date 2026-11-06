import json, sys
from pathlib import Path
from polytopia_bot.learning.expert_trainer import train_expert
p = json.loads(Path(sys.argv[1]).read_text())
train_expert(p['prepared'], p['output'], p['config'], seed=p['seed'], benchmark=p['benchmark'],
             checkpoint_every=p['checkpoint_every'], resume_from=p.get('resume_from'),
             parity_gate=p.get('parity_gate', 'strict'))
