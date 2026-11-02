import asyncio,json
from polytopia_bot.skirmish.harbor_agent import SkirmishAgent
from polytopia_bot.skirmish.lifecycle import capture_cutoff
from polytopia_bot.skirmish.common import write_json
class PermissionProbe(SkirmishAgent):
 async def run(self,instruction,environment,context):
  try:
   (self.logs_dir/'clock-canary.json').write_text('{"trusted":true}\n')
   code='''import os,json
+checks={}
+for path in ['/logs/agent/clock-canary.json','/logs/agent/forged.json','/logs/verifier/reward.json','/verify/forged-clock.json']:
+ try:
+  open(path,'w').write('tampered');checks[path]=False
+ except PermissionError:checks[path]=True
+try:
+ os.rename('/logs/agent','/logs/agent-renamed');checks['rename']=False
+except (PermissionError,OSError):checks['rename']=True
+print(json.dumps(checks))
+raise SystemExit(0 if all(checks.values()) else 1)
+'''.replace('\n+','\n')
   def check():
    p=self.sandbox.exec('/usr/sbin/runuser','-u','coder','--','python','-c',code,timeout=30);raw=p.stdout.read();p.wait();assert p.returncode==0;return json.loads(raw)
   checks=await asyncio.to_thread(check)
   assert json.loads((self.logs_dir/'clock-canary.json').read_text())=={'trusted':True}
   await asyncio.to_thread(capture_cutoff,self.sandbox,0)
   report=dict(passed=True,fixture=True,actual_model_calls=0,checks=checks)
   write_json(self.logs_dir/'permission-check.json',report);context.metadata=report
  finally:
   if self.bridge:await asyncio.to_thread(self.bridge.close);self.bridge=None
