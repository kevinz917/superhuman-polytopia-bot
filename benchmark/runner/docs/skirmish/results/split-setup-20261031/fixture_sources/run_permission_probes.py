from pathlib import Path
import sys,os,json,subprocess,time
from concurrent.futures import ThreadPoolExecutor
root=Path.cwd();sys.path[:0]=[str(root),str(root/'src')]
from scripts.check_skirmish_daytona import credentials
a=credentials();env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'artifacts'),str(root/'src'),str(root)]),DOCKER_CONTEXT='colima-polytopia-skirmish',DAYTONA_API_URL=a['api_url'],DAYTONA_ORGANIZATION_ID=a['organization_id'],DAYTONA_TARGET='us');env['DAYTONA_API_KEY' if a.get('api_key') else 'DAYTONA_JWT_TOKEN']=a.get('api_key') or a['jwt_token'];base=root/'artifacts/harbor-packages/split-20261031'
def run(backend,provider):
 deadline=time.monotonic()+300
 package=root/('artifacts/harbor-packages/split-local-release-v4-20261031' if backend=='local' else 'artifacts/harbor-packages/split-daytona-release-v4-20261031')
 while not (package/'build.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('qualified package not ready')
  time.sleep(.5)
 config=dict(job_name='split-'+backend+'-permission-20261031b',jobs_dir=str(root/'artifacts/harbor-runs'),n_attempts=1,n_concurrent_trials=1,skip_credential_checks=True,environment=dict(import_path='polytopia_bot.skirmish.harbor_environments:'+('SkirmishDockerEnvironment' if backend=='local' else 'SkirmishDaytonaEnvironment'),delete=True),tasks=[dict(path=str(root/('artifacts/harbor-packages/split-local-release-v4-20261031/task' if backend=='local' else 'artifacts/harbor-packages/split-daytona-release-v4-20261031/task')))],agents=[dict(import_path='permission_probe_agent:PermissionProbe',model_name='openai/fixture-permission',kwargs=dict(provider=provider,execution_backend='docker' if backend=='local' else 'daytona'))],verifier=dict(disable=True))
 p=base/(backend+'-permission.json');p.write_text(json.dumps(config,indent=2)+'\n')
 with (base/(backend+'-permission.log')).open('w') as log:subprocess.run([str(root/'artifacts/harbor-runtime/bin/harbor'),'run','--config',str(p)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 out=list((root/'artifacts/harbor-runs'/config['job_name']).glob('*/agent/permission-check.json'));assert len(out)==1;print(backend,out[0].read_text(),flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
 fs=[pool.submit(run,'local','cliproxyapi'),pool.submit(run,'daytona','openrouter')]
 for f in fs:f.result()
