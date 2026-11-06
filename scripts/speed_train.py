import json,sys
from pathlib import Path
from polytopia_bot.learning.fast_trainer import train_fast
p=json.loads(Path(sys.argv[1]).read_text())
train_fast(p['prepared'],p['output'],p['config'],seed=17,stage=p['stage'],readiness=p['readiness'],benchmark=p['benchmark'])
