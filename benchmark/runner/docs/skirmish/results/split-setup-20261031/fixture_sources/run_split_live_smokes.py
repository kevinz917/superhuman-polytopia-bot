from pathlib import Path
import sys,json,os,subprocess,shutil
root=Path.cwd();sys.path[:0]=[str(root),str(root/'src')]
from scripts.check_skirmish_daytona import credentials
from concurrent.futures import ThreadPoolExecutor
base=root/'artifacts/harbor-packages/split-20261031'
instruction='''This is an infrastructure integration smoke test, not a benchmark performance attempt. Work in /bot. Keep the existing Rust starter unchanged except append a comment with the words skirmish split smoke. Compile it with cargo build --release --offline --locked --target wasm32-wasip1. Run skirmish practice --map drylands-0 --seat 1 and inspect at least one entry of its trace with skirmish trace. After those checks, submit immediately using a separate bash tool call containing only echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT. Do not spend time improving strategy. Use bash tools and complete these checks now.'''
a=credentials();env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'artifacts'),str(root/'src'),str(root)]),DOCKER_CONTEXT='colima-polytopia-skirmish',DAYTONA_API_URL=a['api_url'],DAYTONA_ORGANIZATION_ID=a['organization_id'],DAYTONA_TARGET='us');env['DAYTONA_API_KEY' if a.get('api_key') else 'DAYTONA_JWT_TOKEN']=a.get('api_key') or a['jwt_token']
def run(backend,provider,model,package):
 task=base/(backend+'-live-smoke-task');shutil.copytree(package/'task',task);(task/'instruction.md').write_text(instruction+'\n')
 config=dict(job_name='split-'+backend+'-live-smoke-20261031',jobs_dir=str(root/'artifacts/harbor-runs'),n_attempts=1,n_concurrent_trials=1,skip_credential_checks=True,environment=dict(import_path='polytopia_bot.skirmish.harbor_environments:'+('SkirmishDockerEnvironment' if backend=='local' else 'SkirmishDaytonaEnvironment'),delete=True),tasks=[dict(path=str(task))],agents=[dict(import_path='split_smoke_agent:SmokeAgent',model_name=model,kwargs=dict(provider=provider,execution_backend='docker' if backend=='local' else 'daytona'))],verifier=dict(env={'SKIRMISH_EVALUATION_PROFILE':'demo'}))
 path=base/(backend+'-live-smoke.json');path.write_text(json.dumps(config,indent=2)+'\n')
 with (base/(backend+'-live-smoke.log')).open('w') as log:subprocess.run([str(root/'artifacts/harbor-runtime/bin/harbor'),'run','--config',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 result=root/'artifacts/harbor-runs'/config['job_name'];d=json.loads((result/'result.json').read_text());assert d['stats']['n_errored_trials']==0
 trial=next(result.glob('*/agent/authoring.json'));author=json.loads(trial.read_text());assert author['end_reason']=='agent_exit';assert len(list((trial.parent.parent/'artifacts/practice').glob('*/report.json')))>=1
 print(json.dumps(dict(backend=backend,model=model,status='passed',trial=str(trial.parent.parent))),flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(run,'local','cliproxyapi','openai/gpt-6-luna',root/'artifacts/harbor-packages/split-local-v1-20261031'),pool.submit(run,'daytona','openrouter','deepseek/deepseek-v4.1-flash-1017',root/'artifacts/harbor-packages/split-daytona-v1-20261031')]
 for f in futures:f.result()
