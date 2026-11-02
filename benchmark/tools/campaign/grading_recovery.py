"""Recover grading only from an original immutable authoring cutoff."""
import asyncio,json,shutil,hashlib,tarfile,tempfile,shlex
from pathlib import Path
from polytopia_bot.skirmish.harbor_agent import SkirmishAgent
from polytopia_bot.skirmish.harbor_environments import SkirmishDockerEnvironment,SkirmishDaytonaEnvironment,slot
class FrozenSourceAgent(SkirmishAgent):
 def __init__(self,*args,original_trial,**kwargs):
  super().__init__(*args,**kwargs);self.original=Path(original_trial)
 async def run(self,instruction,environment,context):
  try:
   author=json.loads((self.original/'agent/authoring.json').read_text());capture=json.loads((self.original/'artifacts/submission/capture.json').read_text());source=(self.original/'artifacts/submission/main.rs').read_bytes()
   assert hashlib.sha256(source).hexdigest()==capture['source_sha256']==author['capture']['source_sha256']
   assert author['coding_budget_seconds']==2400 and author['timing']['coding_seconds']<=2400
   await asyncio.to_thread(self.sandbox.filesystem.write_bytes,source,'/verify/cutoffs/stage-0/main.rs')
   await asyncio.to_thread(self.sandbox.filesystem.write_text,json.dumps(capture),' /verify/cutoffs/stage-0/capture.json'.strip())
   for name in ['authoring.json','protocol.json','trajectory.json']:
    shutil.copyfile(self.original/'agent'/name,self.logs_dir/name)
   receipt=dict(original_trial=str(self.original),source_sha256=capture['source_sha256'],additional_authoring_seconds=0,additional_model_calls=0,reason='Recover verifier infrastructure failure; original frozen submission and authoring receipts retained')
   (self.logs_dir/'grading-recovery.json').write_text(json.dumps(receipt,indent=2)+'\n')
   context.n_input_tokens=author['usage']['input_tokens'];context.n_output_tokens=author['usage']['output_tokens'];context.cost_usd=author.get('completed_call_cost_usd');context.metadata=dict(author,grading_recovery=receipt)
  finally:
   if self.bridge:await asyncio.to_thread(self.bridge.close);self.bridge=None
class RestoreGames:
 def __init__(self,*args,recovery_checkpoint,recovery_source_sha256=None,**kwargs):
  self.recovery_checkpoint=Path(recovery_checkpoint);self.recovery_source_sha256=recovery_source_sha256;super().__init__(*args,**kwargs)
 async def restore_games(self):
  if self.environment_dir.name!='tests':return
  folder=self.recovery_checkpoint;archive=folder/'checkpoint.tar'
  if (folder/'replay.json.gz').exists():
   if not self.recovery_source_sha256:raise ValueError('trusted source hash required for compact replay restore')
   await self.upload_file(folder/'replay.json.gz','/tmp/compact-replay.json.gz')
   await self.upload_file(Path(__file__).with_name('restore_compact_replay.py'),'/tmp/restore_compact_replay.py')
   result=await self.exec('python /tmp/restore_compact_replay.py /tmp/compact-replay.json.gz '+shlex.quote(self.recovery_source_sha256),user='root',timeout_sec=43200)
   if result.return_code:raise RuntimeError('compact replay hash verification failed: '+result.stderr[-500:])
   return
  if not archive.exists():return
  # Copy a single atomic checkpoint generation before inspection/upload.
  with tempfile.TemporaryDirectory(prefix='skirmish-grade-recovery-') as tmp:
   local=Path(tmp)/'checkpoint.tar';shutil.copyfile(archive,local)
   with tarfile.open(local) as tar:
    for member in tar.getmembers():
     path=Path(member.name)
     if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='games' or member.issym() or member.islnk():raise ValueError('invalid grading checkpoint path')
   await self.upload_file(local,'/tmp/grading-checkpoint.tar')
   result=await self.exec('mkdir -p /results && tar -xf /tmp/grading-checkpoint.tar -C /results && rm /tmp/grading-checkpoint.tar',user='root',timeout_sec=120)
   if result.return_code:raise RuntimeError('could not restore grading checkpoint')
class RecoveryDockerEnvironment(RestoreGames,SkirmishDockerEnvironment):
 async def start(self,force_build):
  await super().start(force_build);await self.restore_games()
class RecoveryDaytonaEnvironment(RestoreGames,SkirmishDaytonaEnvironment):
 def release_startup(self):
  super().release_startup()
  lease=getattr(self,'_global_startup',None)
  if lease is not None:
   self._global_startup=None;asyncio.get_running_loop().create_task(lease.__aexit__(None,None,None))
 async def _resolve_start_sandbox_params(self,*args,**kwargs):
  params=await super()._resolve_start_sandbox_params(*args,**kwargs);params.ttl_minutes=900;return params
 async def start(self,force_build):
  lease=slot(Path.home()/'.cache/polytopia-skirmish/daytona-recovery-starts',2)
  await lease.__aenter__();self._global_startup=lease
  try:
   await super().start(force_build);await self.restore_games()
  except BaseException:
   self.release_startup();raise
