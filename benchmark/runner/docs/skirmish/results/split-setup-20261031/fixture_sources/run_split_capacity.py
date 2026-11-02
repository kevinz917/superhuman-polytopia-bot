from pathlib import Path
import sys,os,json,subprocess,time
from concurrent.futures import ThreadPoolExecutor
root=Path.cwd();sys.path[:0]=[str(root),str(root/'src')]
from scripts.check_skirmish_daytona import credentials
a=credentials();env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'artifacts'),str(root/'src'),str(root)]),DOCKER_CONTEXT='colima-polytopia-skirmish',DAYTONA_API_URL=a['api_url'],DAYTONA_ORGANIZATION_ID=a['organization_id'],DAYTONA_TARGET='us');env['DAYTONA_API_KEY' if a.get('api_key') else 'DAYTONA_JWT_TOKEN']=a.get('api_key') or a['jwt_token'];base=root/'artifacts/harbor-packages/split-20261031'
def run(backend,provider,n):
 config=dict(job_name='split-'+backend+'-capacity-20261031',jobs_dir=str(root/'artifacts/harbor-runs'),n_attempts=n,n_concurrent_trials=n,skip_credential_checks=True,environment=dict(import_path='polytopia_bot.skirmish.harbor_environments:'+('SkirmishDockerEnvironment' if backend=='local' else 'SkirmishDaytonaEnvironment'),delete=True),tasks=[dict(path=str(root/('artifacts/harbor-packages/split-local-release-v4-20261031/task' if backend=='local' else 'artifacts/harbor-packages/split-daytona-release-v4-20261031/task')))],agents=[dict(import_path='permission_probe_agent:PermissionProbe',model_name='openai/fixture-permission',kwargs=dict(provider=provider,execution_backend='docker' if backend=='local' else 'daytona'))],verifier=dict(disable=True))
 path=base/(backend+'-capacity.json');path.write_text(json.dumps(config,indent=2)+'\n');started=time.time()
 with (base/(backend+'-capacity.log')).open('w') as log:subprocess.run([str(root/'artifacts/harbor-runtime/bin/harbor'),'run','--config',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 out=list((root/'artifacts/harbor-runs'/config['job_name']).glob('*/agent/permission-check.json'));d=json.loads((root/'artifacts/harbor-runs'/config['job_name']/'result.json').read_text());assert len(out)==n and d['stats']['n_errored_trials']==0
 report=dict(backend=backend,passed=True,infrastructure_trials=n,configured_concurrent_trials=n,actual_benchmark_attempts=0,actual_model_calls=0,elapsed_seconds=time.time()-started)
 (base/(backend+'-capacity-report.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
 fs=[pool.submit(run,'local','cliproxyapi',25),pool.submit(run,'daytona','openrouter',35)]
 for f in fs:f.result()
