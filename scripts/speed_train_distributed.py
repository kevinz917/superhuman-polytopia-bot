import json,sys
from pathlib import Path
from polytopia_bot.learning.fast_distributed_trainer import train_distributed
p=json.loads(Path(sys.argv[1]).read_text())
train_distributed(p['prepared'],p['output'],p['config'],seed=17,stage=p['stage'],readiness=p['readiness'],benchmark=p['benchmark'],continuation=p.get('continuation'),probe=p.get('probe'),mixed=p.get('mixed'),scale=p.get('scale'))
