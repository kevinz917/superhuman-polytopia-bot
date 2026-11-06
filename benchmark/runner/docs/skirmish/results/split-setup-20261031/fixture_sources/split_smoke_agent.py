import asyncio,hashlib
from polytopia_bot.skirmish.harbor_agent import SkirmishAgent
from polytopia_bot.skirmish.common import write_json
class SmokeAgent(SkirmishAgent):
 async def setup(self,environment):
  await super().setup(environment)
  data=bytes(range(256))*8192
  path='/verify/large-transfer-smoke.bin'
  await asyncio.to_thread(self.sandbox.filesystem.write_bytes,data,path)
  actual=await asyncio.to_thread(self.sandbox.filesystem.read_bytes,path)
  assert actual==data
  write_json(self.logs_dir/'filesystem-smoke.json',dict(passed=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
