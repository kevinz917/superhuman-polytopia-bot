"""Training-map engineering prefixes, never accepted game/learning outcomes."""
from dataclasses import asdict
import gzip
import json
import multiprocessing
from pathlib import Path
import queue
import time

from .interface import canonical, digest, render, parse_response
from .efficient_gameplay import QueueGenerator


def _prefix_lane(lane, jobs, output, batches, events, responses):
    from .adapter import GPUEnvironment
    from .teacher import WeakBaseline
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    games=[];started=time.monotonic();steps=0
    try:
        for job in jobs:
            env=GPUEnvironment(14);obs=env.reset(seed=job['seed'],settings={})
            games.append(dict(env=env,obs=obs,job=job,done=False,opponent=WeakBaseline()))
        generator=QueueGenerator(lane,events,responses)
        with gzip.open(root/'trace.jsonl.gz','xt',compresslevel=1) as trace:
            def execute(game,action,policy,diagnostic=None):
                nonlocal steps
                obs=game['obs'];t=game['env'].step(action_id=action.action_id,expected_state_version=obs.state_version)
                trace.write(canonical(dict(job=game['job']['id'],observation=asdict(obs),policy=policy,
                    selected_command=dict(action.command),diagnostic=diagnostic,
                    after=[asdict(game['env'].observe(p)) for p in (1,2)],
                    outcome=dict(terminated=t.terminated,truncated=t.truncated,rewards=t.rewards,reason=t.reason)))+'\n')
                steps+=1;game['done']=t.terminated or t.truncated
                game['obs']=t.actor_observation if game['done'] else game['env'].observe(t.next_player)
            for _ in range(batches):
                requests=[]
                for g in games:
                    while not g['done'] and g['obs'].player!=g['job']['seat']:
                        execute(g,g['opponent'].select(g['obs']).action,'scripted_opponent')
                    if g['done']:continue
                    p=render(g['obs']);messages=[dict(m) for m in p.messages]
                    if g['job']['strategy']:messages[0]['content']+='\nStrategic guidance:\n'+g['job']['strategy']
                    requests.append((g,p,messages))
                if not requests:break
                generations=generator.batch([r[2] for r in requests])
                for (g,p,messages),generation in zip(requests,generations):
                    action=parse_response(generation.text,p,current_version=g['obs'].state_version)
                    execute(g,action,'model',dict(messages=messages,generation=asdict(generation)))
        result=dict(lane=lane,steps=steps,elapsed_seconds=time.monotonic()-started,
                    profile_only=True,complete_games_not_required=True)
        (root/'summary.json').write_text(json.dumps(result));events.put(('done',lane,result))
    except BaseException as error:
        events.put(('failed',lane,repr(error)));raise
    finally:
        for game in games:game['env'].close()


def profile_prefixes(output,lane_jobs,generator,*,mode,batches=16,seconds_limit=900):
    if mode not in ('serial','pipeline'):raise ValueError('unknown profile mode')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    context=multiprocessing.get_context('spawn');events=context.Queue();responses=[context.Queue() for _ in lane_jobs]
    processes=[];reports={};records=[];started=time.monotonic()
    def start(lane):
        p=context.Process(target=_prefix_lane,args=(lane,lane_jobs[lane],root/f'lane-{lane}',batches,events,responses[lane]))
        p.start();processes.append(p)
    try:
        if mode=='pipeline':
            for i in range(len(lane_jobs)):start(i)
        else:start(0)
        while len(reports)<len(lane_jobs):
            if time.monotonic()-started>seconds_limit:raise TimeoutError('bounded profile exceeded')
            try:event=events.get(timeout=.1)
            except queue.Empty:
                if any(p.exitcode not in (None,0) for p in processes):raise RuntimeError('profile CPU process failed')
                continue
            kind,lane,*payload=event
            if kind=='failed':raise RuntimeError(payload[0])
            if kind=='done':
                reports[lane]=payload[0]
                if mode=='serial' and len(processes)<len(lane_jobs):start(len(processes))
                continue
            request,messages,queued_at=payload;before=time.monotonic();generations=generator.batch(messages);elapsed=time.monotonic()-before
            records.append(dict(lane=lane,request=request,size=len(messages),messages_hash=digest(messages),
                generation_hash=digest([asdict(g) for g in generations]),model_seconds=elapsed,queue_wait_seconds=before-queued_at))
            responses[lane].put(('result',request,generations))
        for p in processes:p.join(timeout=10)
        if any(p.exitcode!=0 for p in processes):raise RuntimeError('profile process did not exit cleanly')
        result=dict(schema='rollout-prefix-profile-v1',mode=mode,elapsed_seconds=time.monotonic()-started,
            requests=sum(r['size'] for r in records),model_seconds=sum(r['model_seconds'] for r in records),
            steps=sum(r['steps'] for r in reports.values()),batches=records,lanes=reports,
            note='Fixed training-map prefixes and real model calls. Engineering throughput/equivalence only, not full-game results. Includes CPU startup/JIT/logging; no new training labels.')
        (root/'report.json').write_text(json.dumps(result,indent=2));return result
    finally:
        for p in processes:
            if p.is_alive():p.terminate()
            p.join(timeout=5)
        for q in responses+[events]:q.close()
