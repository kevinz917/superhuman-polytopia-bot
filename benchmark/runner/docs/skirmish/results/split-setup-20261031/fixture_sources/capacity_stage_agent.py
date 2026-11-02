import asyncio,json,time
from pathlib import Path
from polytopia_bot.skirmish.harbor_agent import SkirmishAgent
class CapacityStageAgent(SkirmishAgent):
 def __init__(self,*args,barrier_dir,barrier_count,**kwargs):
  super().__init__(*args,**kwargs);self.barrier_dir=Path(barrier_dir);self.barrier_count=barrier_count
 async def run(self,instruction,environment,context):
  self.barrier_dir.mkdir(parents=True,exist_ok=True)
  (self.barrier_dir/self.logs_dir.parent.name).write_text('ready')
  deadline=time.monotonic()+900
  while len(list(self.barrier_dir.iterdir()))<self.barrier_count:
   if time.monotonic()>deadline:raise TimeoutError('not all capacity sandboxes became ready')
   await asyncio.sleep(1)
  await super().run(instruction,environment,context)
  (self.logs_dir/'capacity-check.json').write_text(json.dumps(dict(passed=True,barrier_count=self.barrier_count,actual_model_calls=0)))
