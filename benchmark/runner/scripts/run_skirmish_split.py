"""Prepare or launch a pinned local-proxy / Daytona-OpenRouter Harbor campaign."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]
from polytopia_bot.skirmish.common import canonical, sha, write_json
from polytopia_bot.skirmish.inference import endpoint
from polytopia_bot.skirmish.split import job_config, route_models


def catalog():
    route = endpoint(dict(provider='cliproxyapi',model='gpt-6-luna'))
    request = urllib.request.Request(route.base_url+'/models',headers={'Authorization':'Bearer '+route.credential()})
    with urllib.request.urlopen(request,timeout=20) as response:
        return {r['id'] for r in json.load(response)['data']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--local-package',type=Path,required=True)
    p.add_argument('--daytona-package',type=Path,required=True)
    p.add_argument('--roster',type=Path,default=ROOT/'configs/skirmish/local-v1-top12.json')
    p.add_argument('--attempts',type=int,default=5)
    p.add_argument('--local-concurrency',type=int,default=32)
    p.add_argument('--daytona-concurrency',type=int,default=64)
    p.add_argument('--probe',action='store_true',help='Verify actual model/tool continuations before marking ready.')
    p.add_argument('--run',action='store_true',help='Launch both complete Harbor jobs concurrently; requires --probe.')
    p.add_argument('--demo',action='store_true',help='36 final matches; same 40-minute authoring protocol.')
    args = p.parse_args()
    if args.attempts<1 or min(args.local_concurrency,args.daytona_concurrency)<1:
        raise ValueError('positive attempts and concurrency required')
    if args.run and not args.probe:
        raise ValueError('full launch requires live transport probes')
    directory = args.directory.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    available = catalog()
    roster = route_models(json.loads(args.roster.read_text())['models'],available)
    local = [r for r in roster if r['execution_backend']=='docker']
    cloud = [r for r in roster if r['execution_backend']=='daytona']
    if not local or not cloud:
        raise ValueError('split campaign expects local and cloud routes')
    from polytopia_bot.skirmish.continuous import source_files
    for package in [args.local_package,args.daytona_package]:
        receipt = json.loads((package/'build.json').read_text())
        if receipt['source_files'] != source_files():
            raise ValueError('source drift since build; create new versioned packages')
    a = json.loads((args.local_package/'build.json').read_text())
    b = json.loads((args.daytona_package/'build.json').read_text())
    for field in ['manifest_sha256','maps_sha256']:
        if a[field] != b[field]:
            raise ValueError('local and Daytona suites must be identical')
    # Resolve credentials before producing a launch-ready plan. The key itself
    # is neither persisted nor injected into any Harbor environment config.
    endpoint(dict(provider='openrouter',model=cloud[0]['model'])).credential()
    write_json(directory/'routes.json',dict(models=roster,proxy_catalog=sorted(available)))
    jobs = []
    from harbor.models.job.config import JobConfig
    for suffix,package,models,concurrency in [('local',args.local_package,local,args.local_concurrency),
            ('daytona',args.daytona_package,cloud,args.daytona_concurrency)]:
        config = job_config(args.name+'-'+suffix,package,models,args.attempts,concurrency,
                            ROOT/'artifacts/harbor-runs',demo=args.demo)
        JobConfig.model_validate(config)
        path = directory/(suffix+'.json')
        write_json(path,config)
        jobs.append((suffix,path))
    probes = []
    if args.probe:
        from scripts.check_skirmish_proxy import mini_check
        with ThreadPoolExecutor(max_workers=min(12,len(roster))) as pool:
            futures = {pool.submit(mini_check,r['model'],dict(provider=r['provider'],model=r['model'],
                parameters=r['parameters'])):r for r in roster}
            for future in as_completed(futures):
                r = futures[future]
                try:
                    probes.append(dict(label=r['label'],model=r['model'],passed=True,receipt=future.result()))
                except Exception as error:
                    probes.append(dict(label=r['label'],model=r['model'],passed=False,error=type(error).__name__))
                write_json(directory/'probes.json',probes)
                print(json.dumps(probes[-1]),flush=True)
        if not all(p['passed'] for p in probes):
            write_json(directory/'status.json',dict(status='probe_failed',no_full_campaign_started=True))
            raise RuntimeError('selected transport failed; no fallback and no campaign launch')
    plan = dict(status='ready' if args.probe else 'prepared_unprobed',protocol='polytopia-skirmish-continuous-v1',
        attempts_per_model=args.attempts,coding_seconds=2400,profile='demo' if args.demo else 'standard',
        local_models=[r['model'] for r in local],daytona_models=[r['model'] for r in cloud],
        local_trials=len(local)*args.attempts,daytona_trials=len(cloud)*args.attempts,
        final_matches=len(roster)*args.attempts*(36 if args.demo else 360),
        manifest_sha256=a['manifest_sha256'],maps_sha256=a['maps_sha256'],
        routing_sha256=sha(canonical(roster).encode()),no_automatic_paid_fallback=True)
    write_json(directory/'status.json',plan)
    if not args.run:
        print(json.dumps(plan,indent=2))
        return
    from scripts.check_skirmish_daytona import credentials
    auth = credentials()
    env = dict(os.environ,PYTHONPATH=str(ROOT/'src'),DOCKER_CONTEXT='colima-polytopia-skirmish',
        DAYTONA_API_URL=auth['api_url'],DAYTONA_ORGANIZATION_ID=auth['organization_id'],DAYTONA_TARGET='us')
    env['DAYTONA_API_KEY' if auth.get('api_key') else 'DAYTONA_JWT_TOKEN'] = auth.get('api_key') or auth['jwt_token']
    processes = []
    try:
        for suffix,path in jobs:
            log = (directory/(suffix+'.log')).open('w')
            process = subprocess.Popen([str(Path(sys.executable).with_name('harbor')),'run','--config',str(path)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((suffix,process,log))
        plan.update(status='running',pids={s:p.pid for s,p,_ in processes})
        write_json(directory/'status.json',plan)
        codes = {suffix:process.wait() for suffix,process,_ in processes}
        from scripts.summarize_skirmish_harbor import summarize
        for suffix,_,_ in processes:
            if codes[suffix] != 0:
                raise RuntimeError('Harbor route failed: '+suffix)
            summarize(ROOT/'artifacts/harbor-runs'/(args.name+'-'+suffix))
        plan['status']='completed'
        write_json(directory/'status.json',plan)
    except BaseException:
        for _,process,_ in processes:
            if process.poll() is None:
                process.terminate()
        plan['status']='interrupted_or_failed'
        write_json(directory/'status.json',plan)
        raise
    finally:
        for _,_,log in processes:
            log.close()


if __name__ == '__main__':
    main()
