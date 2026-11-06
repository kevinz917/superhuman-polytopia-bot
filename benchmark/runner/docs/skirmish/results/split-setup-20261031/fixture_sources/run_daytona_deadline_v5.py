import json,os,sys,threading,subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
root=Path.cwd();sys.path[:0]=[str(root),str(root/'src')]
from scripts.check_skirmish_daytona import credentials
class Proxy(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_POST(self):
  request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  command="printf started > /bot/deadline-command-started; sleep 30; printf LATE_WRITE > /bot/src/main.rs"
  response=dict(id='deadline-fixture',choices=[dict(finish_reason='tool_calls',message=dict(role='assistant',content='',tool_calls=[dict(id='bash-1',type='function',function=dict(name='bash',arguments=json.dumps(dict(command=command))))]))],usage=dict(prompt_tokens=1,completion_tokens=1))
  data=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
server=ThreadingHTTPServer(('127.0.0.1',0),Proxy);threading.Thread(target=server.serve_forever,daemon=True).start()
package=root/'artifacts/harbor-packages/split-daytona-20261031c';name='split-daytona-deadline-v5-20261031'
config=dict(job_name=name,jobs_dir=str(root/'artifacts/harbor-runs'),n_attempts=1,n_concurrent_trials=1,skip_credential_checks=True,environment=dict(import_path='polytopia_bot.skirmish.harbor_environments:SkirmishDaytonaEnvironment',delete=True),tasks=[dict(path=str(package/'task'))],agents=[dict(import_path='daytona_deadline_agent:DeadlineAgent',model_name='openai/fixture-deadline',kwargs=dict(provider='openrouter',execution_backend='daytona',inference_base_url=f'http://127.0.0.1:{server.server_port}/v1'))],verifier=dict(disable=True))
path=root/'artifacts/harbor-packages/split-20261031/daytona-deadline-v5.json';path.write_text(json.dumps(config,indent=2)+'\n');a=credentials();env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'artifacts'),str(root/'src'),str(root)]),OPENROUTER_API_KEY='local-fixture-only',DAYTONA_API_URL=a['api_url'],DAYTONA_ORGANIZATION_ID=a['organization_id'],DAYTONA_TARGET='us');env['DAYTONA_API_KEY' if a.get('api_key') else 'DAYTONA_JWT_TOKEN']=a.get('api_key') or a['jwt_token']
try:subprocess.run([str(root/'artifacts/harbor-runtime/bin/harbor'),'run','--config',str(path)],env=env,check=True)
finally:server.shutdown()
reports=list((root/'artifacts/harbor-runs'/name).glob('*/agent/deadline-check.json'));assert len(reports)==1;print(reports[0].read_text())
