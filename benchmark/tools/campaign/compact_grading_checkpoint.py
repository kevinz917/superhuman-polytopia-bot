"""Lossless replay checkpoint: exact game receipts and the recorded action sequence."""
EXPORT_SCRIPT=r'''from pathlib import Path
import json,gzip,base64,hashlib,re
capture=json.loads(Path('/verify/cutoffs/stage-0/capture.json').read_text())
base=Path('/results/games')
if not base.exists():base=Path('/logs/verifier/traces')
games=[]
for summary in sorted(base.glob('*/summary.json')):
 row=json.loads(summary.read_text())
 if row['audit'].get('virtual'):games.append(dict(summary=row,actions=[]));continue
 rel=Path(row['trace_relative']).relative_to('games')
 if rel.suffix=='.gz':rel=rel.with_suffix('')
 private=(base/rel).with_suffix('.private.jsonl')
 raw=private.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=row['private_trace_sha256']:raise ValueError('private trace hash changed')
 matches=re.findall(rb'"action_id":"([^"]*)","chain":"[0-9a-f]{64}","player":([12]),"turn":([0-9]+),"version":([0-9]+)',raw)
 entries=[[int(p),int(v),int(t),a.decode('ascii')] for a,p,t,v in matches]
 actions=[e[3] for e in entries]
 if len(actions)!=row['actions']:raise ValueError('incomplete recorded action sequence')
 games.append(dict(summary=row,actions=actions,entries=entries))
payload=dict(schema='skirmish-compact-replay-v1',source_sha256=capture['source_sha256'],games=games)
logs=Path('/logs/verifier')
if (logs/'reward.json').exists():
 names=['summary.json','games.json','evaluation-plan.json','submission.json','reward.json']
 payload['final']={name:json.loads((logs/name).read_text()) for name in names}
 if (logs/'bot.wasm').exists():payload['final']['wasm_base64']=base64.b64encode((logs/'bot.wasm').read_bytes()).decode()
payload['referee_source_sha256']=hashlib.sha256(Path('/workspace/src/polytopia_bot/skirmish/local_referee.py').read_bytes()).hexdigest()
data=gzip.compress(json.dumps(payload,separators=(',',':')).encode(),compresslevel=1,mtime=0)
print(base64.b64encode(data).decode())'''

def client():
 import os,urllib3
 from urllib.parse import urlsplit
 from scripts.check_skirmish_daytona import credentials
 from daytona import Daytona,DaytonaConfig
 d=Daytona(DaytonaConfig(**credentials(),otel_enabled=False,use_deprecated_polling=True))
 proxy=urlsplit(os.environ.get('HTTPS_PROXY',''))
 if proxy.hostname in ('127.0.0.1','localhost'):
  pool=urllib3.ProxyManager(f'http://{proxy.hostname}:{proxy.port}',maxsize=32)
  d._api_client.rest_client.pool_manager=pool;d._toolbox_api_client.rest_client.pool_manager=pool
 return d

def export(sandbox,destination):
 import shlex,base64,gzip,json,hashlib
 from pathlib import Path
 response=session_command(sandbox,'nice -n 19 python -c '+shlex.quote(EXPORT_SCRIPT),timeout=300)
 if response.exit_code:raise RuntimeError(response.result[-500:])
 data=base64.b64decode(response.result.strip(),validate=True);payload=json.loads(gzip.decompress(data))
 destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);tmp=destination.with_suffix('.partial');tmp.write_bytes(data);tmp.replace(destination)
 return dict(games=len(payload['games']),source_sha256=payload['source_sha256'],bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def session_command(sandbox,command,timeout=300):
 import uuid,time
 from types import SimpleNamespace
 from daytona import SessionExecuteRequest
 session='checkpoint-'+uuid.uuid4().hex;sandbox.process.create_session(session,request_timeout=30)
 response=sandbox.process.execute_session_command(session,SessionExecuteRequest(command=command,run_async=True),timeout=60)
 deadline=time.monotonic()+timeout
 while time.monotonic()<deadline:
  state=sandbox.process.get_session_command(session,response.cmd_id,request_timeout=30)
  if state.exit_code is not None:
   logs=sandbox.process.get_session_command_logs(session,response.cmd_id,request_timeout=60)
   return SimpleNamespace(exit_code=state.exit_code,result=logs.stdout+logs.stderr)
  time.sleep(2)
 raise TimeoutError('bounded checkpoint command deadline')
