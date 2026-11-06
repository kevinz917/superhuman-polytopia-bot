from pathlib import Path
import sys,os,json,subprocess,time,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
root=Path.cwd();sys.path[:0]=[str(root),str(root/'src')]
from scripts.check_skirmish_daytona import credentials
class Proxy(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_POST(self):
  r=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  done=any(m.get('role')=='tool' for m in r['messages']);cmd='echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' if done else 'echo CAPACITY_OK'
  out=dict(id='capacity-fixture',choices=[dict(finish_reason='tool_calls',message=dict(role='assistant',content='',tool_calls=[dict(id='bash-1' if not done else 'bash-2',type='function',function=dict(name='bash',arguments=json.dumps(dict(command=cmd))))]))],usage=dict(prompt_tokens=1,completion_tokens=1))
  data=json.dumps(out).encode();self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
server=ThreadingHTTPServer(('127.0.0.1',0),Proxy);server.request_queue_size=128;threading.Thread(target=server.serve_forever,daemon=True).start()
a=credentials();env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'artifacts'),str(root/'src'),str(root)]),DOCKER_CONTEXT='colima-polytopia-skirmish',CLIPROXY_API_KEY='local-fixture-only',OPENROUTER_API_KEY='local-fixture-only',DAYTONA_API_URL=a['api_url'],DAYTONA_ORGANIZATION_ID=a['organization_id'],DAYTONA_TARGET='us');env['DAYTONA_API_KEY' if a.get('api_key') else 'DAYTONA_JWT_TOKEN']=a.get('api_key') or a['jwt_token'];base=root/'artifacts/harbor-packages/split-20261031'
def run(backend,provider,n):
 name='split-'+backend+'-capacity-v5-20261031';barrier=base/(backend+'-capacity-v5-barrier')
 config=dict(job_name=name,jobs_dir=str(root/'artifacts/harbor-runs'),n_attempts=n,n_concurrent_trials=n,skip_credential_checks=True,environment=dict(import_path='polytopia_bot.skirmish.harbor_environments:'+('SkirmishDockerEnvironment' if backend=='local' else 'SkirmishDaytonaEnvironment'),delete=True),tasks=[dict(path=str(root/('artifacts/harbor-packages/split-local-release-v5-20261031/task' if backend=='local' else 'artifacts/harbor-packages/split-daytona-release-v5-20261031/task')))],agents=[dict(import_path='capacity_stage_agent:CapacityStageAgent',model_name='openai/fixture-capacity',kwargs=dict(provider=provider,execution_backend='docker' if backend=='local' else 'daytona',inference_base_url=f'http://127.0.0.1:{server.server_port}/v1',barrier_dir=str(barrier),barrier_count=n))],verifier=dict(disable=True))
 path=base/(backend+'-capacity-v5.json');path.write_text(json.dumps(config,indent=2)+'\n');started=time.time()
 with (base/(backend+'-capacity-v5.log')).open('w') as log:subprocess.run([str(root/'artifacts/harbor-runtime/bin/harbor'),'run','--config',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 job=root/'artifacts/harbor-runs'/name;out=list(job.glob('*/agent/capacity-check.json'));d=json.loads((job/'result.json').read_text());assert len(out)==n and d['stats']['n_errored_trials']==0,(len(out),d['stats'])
 report=dict(backend=backend,passed=True,infrastructure_trials=n,configured_concurrent_trials=n,simultaneously_ready_sandboxes=len(list(barrier.iterdir())),actual_benchmark_attempts=0,actual_model_calls=0,elapsed_seconds=time.time()-started)
 (base/(backend+'-capacity-v5-report.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
try:
 with ThreadPoolExecutor(max_workers=2) as pool:
  fs=[pool.submit(run,'local','cliproxyapi',25),pool.submit(run,'daytona','openrouter',35)]
  for f in fs:f.result()
finally:server.shutdown()
