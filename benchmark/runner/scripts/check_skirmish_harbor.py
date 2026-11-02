"""Deterministic real-VM Harbor integration regression; makes no model API calls."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]


class FixtureProxy(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        responded = any(m.get('role')=='tool' for m in request['messages'])
        if responded:
            command = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
        elif request['model'].rsplit('/',1)[-1]=='fixture-broken':
            command = "printf 'deliberately invalid Rust' > /bot/src/main.rs"
        else:
            command = "skirmish practice --map drylands-0 --seat 1 > /bot/practice-check.json && python -c 'import json,subprocess; r=json.load(open(\"/bot/practice-check.json\")); p=subprocess.run([\"skirmish\",\"trace\",\"--path\",r[\"trace_path\"],\"--limit\",\"1\"],capture_output=True,text=True,check=True); assert json.loads(p.stdout); print(\"PRACTICE_AND_TRACE_OK\")'"
        response = dict(id='fixture-'+str(time.time_ns()), object='chat.completion', model=request['model'],
            choices=[dict(index=0,finish_reason='tool_calls',message=dict(role='assistant',content='',
                tool_calls=[dict(id='tool-'+str(time.time_ns()),type='function',function=dict(name='bash',arguments=json.dumps(dict(command=command))))]))],
            usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2))
        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--backend',choices=['docker','daytona'],default='docker')
    args = p.parse_args()
    package = args.package.resolve()
    server = ThreadingHTTPServer(('127.0.0.1',0),FixtureProxy)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    environment = dict(import_path='polytopia_bot.skirmish.harbor_environments:SkirmishDockerEnvironment',
                       delete=True,cpu_enforcement_policy='limit',memory_enforcement_policy='limit')
    agent_kwargs = dict(provider='cliproxyapi',execution_backend='docker')
    if args.backend == 'daytona':
        environment = dict(import_path='polytopia_bot.skirmish.harbor_environments:SkirmishDaytonaEnvironment',delete=True)
        agent_kwargs = dict(provider='openrouter',execution_backend='daytona',
                            inference_base_url=f'http://127.0.0.1:{server.server_port}/v1')
    config = dict(job_name=args.name,jobs_dir=str(ROOT/'artifacts/harbor-runs'),n_attempts=1,
        n_concurrent_trials=2,skip_credential_checks=True,environment=environment,
        tasks=[dict(path=str(package/'task'))],
        agents=[dict(import_path='polytopia_bot.skirmish.harbor_agent:SkirmishAgent',model_name='openai/'+name,kwargs=agent_kwargs)
                for name in ('fixture-valid','fixture-broken')],
        verifier=dict(env={'SKIRMISH_EVALUATION_PROFILE':'demo'}))
    path = package/f'{args.name}.json'
    path.write_text(json.dumps(config,indent=2)+'\n')
    env = dict(os.environ,DOCKER_CONTEXT='colima-polytopia-skirmish',PYTHONPATH=str(ROOT/'src'),
        CLIPROXY_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',CLIPROXY_API_KEY='local-fixture-only',
        OPENROUTER_API_KEY='local-fixture-only')
    if args.backend == 'daytona':
        from scripts.check_skirmish_daytona import credentials
        auth = credentials()
        env.update(DAYTONA_API_URL=auth['api_url'],DAYTONA_ORGANIZATION_ID=auth['organization_id'],
                   DAYTONA_TARGET='us')
        if auth.get('api_key'):
            env['DAYTONA_API_KEY'] = auth['api_key']
        else:
            env['DAYTONA_JWT_TOKEN'] = auth['jwt_token']
    try:
        subprocess.run([str(Path(sys.executable).with_name('harbor')),'run','--config',str(path)],cwd=ROOT,env=env,check=True)
    finally:
        server.shutdown()
    result = ROOT/'artifacts/harbor-runs'/args.name
    trials = list(result.glob('*/verifier/reward.json'))
    if len(trials)!=2:
        raise RuntimeError('expected two complete fixture trials; inspect Harbor error receipts')
    rewards = [json.loads(path.read_text()) for path in trials]
    assert sorted(r['compilation_loss_rate'] for r in rewards)==[0,1]
    assert all(r['games']==36 and r['evaluation_is_demo']==1 for r in rewards)
    authoring = list(result.glob('*/agent/authoring.json'))
    assert len(authoring)==2
    histories = [p.read_text() for p in result.glob('*/agent/trajectory.json')]
    assert any('PRACTICE_AND_TRACE_OK' in history for history in histories)
    report = dict(passed=True,backend=args.backend,real_harbor_trials=2,actual_model_calls=0,
        practice_and_trace=True,compilation_failure_scores_zero=True,rewards=rewards)
    (result/'integration-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
