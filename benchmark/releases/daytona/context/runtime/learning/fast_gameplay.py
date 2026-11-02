"""Resumable per-game CPU workers feeding a continuous inference scheduler."""
import base64,gzip,json,multiprocessing,queue,time
from dataclasses import asdict
from pathlib import Path
from .interface import canonical,render,parse_response,ResponseError
from .prompt_search import candidate_messages
from .efficient_gameplay import QueueGenerator


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value));tmp.replace(path)


def play_one(root,job,generator,*,seconds_limit,action_limit=10000,environment_factory=None,opponent_factory=None):
    from .adapter import GPUEnvironment
    from .teacher import WeakBaseline,ScriptedTeacher
    root=Path(root);root.mkdir(parents=True,exist_ok=True);(root/'private').mkdir(exist_ok=True)
    if Path(job['id']).name!=job['id'] or job['id'] in ('.','..'):raise ValueError('unsafe game ID')
    result_path=root/(job['id']+'.result.json')
    if result_path.exists():return json.loads(result_path.read_text())
    if environment_factory is None:
        from .splits import board_hashes
        from polytopia_bot.gpu.mapgen import layout
        import jax
        import jax.numpy as jnp
        if board_hashes(jax.device_get(layout(jnp.int32(job['seed']),size=14)),14)[0]!=job['board_hash']:
            raise ValueError('reserved map mismatch')
    env=(environment_factory or (lambda:GPUEnvironment(14)))()
    opponent=(opponent_factory or (lambda name:WeakBaseline() if name=='weak' else ScriptedTeacher(name)))(job['opponent'])
    checkpoint=root/(job['id']+'.resume.json');trace_path=root/(job['id']+'.jsonl')
    counters=dict(decisions=0,model_decisions=0,attempts=0,first_attempt_valid=0,max_prompt_tokens=0,
                  model_command_counts={})
    elapsed=0.;started=time.monotonic()
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if saved['job']!=job:raise ValueError('resume job differs')
        obs=env.restore(base64.b64decode(saved['snapshot']));counters=saved['counters'];elapsed=saved['elapsed_seconds']
        with trace_path.open('r+b') as f:f.truncate(saved['trace_offset'])
    else:
        obs=env.reset(seed=job['seed'],settings={})
        (root/'private'/(job['id']+'.snapshot')).write_bytes(env.snapshot())
        trace_path.write_bytes(b'')
    trace=trace_path.open('ab')
    def write(row):trace.write((canonical(row)+'\n').encode());trace.flush()
    def persist():
        atomic_json(checkpoint,dict(job=job,snapshot=base64.b64encode(env.snapshot()).decode(),
            counters=dict(counters),trace_offset=trace.tell(),elapsed_seconds=elapsed+time.monotonic()-started))
    def finish(status,reason,reward=0.):
        row={**job,**counters,'status':status,'reason':reason,'reward':reward,
             'round':obs.visible_state['turn'],'seconds':elapsed+time.monotonic()-started}
        write({'game_result':row});trace.close()
        compressed=root/(job['id']+'.jsonl.gz')
        with trace_path.open('rb') as source,gzip.open(str(compressed)+'.tmp','wb',compresslevel=1) as dest:
            import shutil
            shutil.copyfileobj(source,dest)
        Path(str(compressed)+'.tmp').replace(compressed)
        atomic_json(result_path,row);return row
    persist()
    try:
        while True:
            if elapsed+time.monotonic()-started>=seconds_limit:return finish('budget_incomplete','occupied_time_ceiling')
            if obs.player==job['seat']:
                prompt=render(obs);messages=candidate_messages(obs,job['strategy']);attempts=[]
                counters['model_decisions']+=1
                for attempt in range(3):
                    before=time.monotonic();generation=generator.batch([messages])[0]
                    entry=dict(messages=messages,generation=asdict(generation),batch_size=1,batch_seconds=time.monotonic()-before)
                    counters['attempts']+=1;counters['max_prompt_tokens']=max(counters['max_prompt_tokens'],len(generation.prompt_token_ids))
                    try:action=parse_response(generation.text,prompt,current_version=obs.state_version)
                    except ResponseError as exc:
                        entry['error']=str(exc);attempts.append(entry)
                        messages=messages+[dict(role='assistant',content=generation.text),dict(role='user',content=f'Invalid response: {exc}. Return one legal action_id JSON object.')]
                    else:
                        attempts.append(entry);counters['first_attempt_valid']+=attempt==0;break
                else:
                    write(dict(observation=asdict(obs),policy='model',attempts=attempts,forfeit=True))
                    return finish('forfeit','invalid_response_after_two_repairs',-1.)
                policy='model';diagnostic=attempts
                kind=str(action.command['kind'])
                counts=counters.setdefault('model_command_counts',{})
                counts[kind]=counts.get(kind,0)+1
            else:
                choice=opponent.select(obs);action=choice.action;policy='scripted_opponent';diagnostic=asdict(choice)
            transition=env.step(action_id=action.action_id,expected_state_version=obs.state_version)
            counters['decisions']+=1
            write(dict(observation=asdict(obs),policy=policy,selected_command=dict(action.command),diagnostic=diagnostic,
                after=[asdict(env.observe(p)) for p in (1,2)],outcome=dict(terminated=transition.terminated,truncated=transition.truncated,rewards=transition.rewards,reason=transition.reason)))
            if transition.terminated or transition.truncated:
                obs=transition.actor_observation
                return finish('native' if transition.terminated else 'truncated',transition.reason,transition.rewards[job['seat']])
            obs=env.observe(transition.next_player)
            if counters['decisions']>=action_limit:return finish('truncated','research_action_limit')
            if counters['decisions']%16==0:persist()
    finally:
        if not trace.closed:trace.close()
        env.close()


def _worker(lane,root,tasks,events,responses,seconds_limit,action_limit,environment_factory,opponent_factory):
    import os
    if hasattr(os,'sched_getaffinity'):
        cpus=sorted(os.sched_getaffinity(0))
        width=max(1,min(2,len(cpus)))
        chosen={cpus[(lane*width+j)%len(cpus)] for j in range(width)}
        os.sched_setaffinity(0,chosen)
    generator=QueueGenerator(lane,events,responses)
    try:
        while (job:=tasks.get()) is not None:
            events.put(('done',lane,play_one(root,job,generator,seconds_limit=seconds_limit,action_limit=action_limit,environment_factory=environment_factory,opponent_factory=opponent_factory)))
    except BaseException as exc:
        events.put(('failed',lane,repr(exc)));raise


def play_continuous(root,take_job,generator,*,lanes=8,seconds_limit=4500,action_limit=10000,on_complete=None,heartbeat=None,environment_factory=None,opponent_factory=None):
    """take_job returns a unique job or None; coordinator owns the global queue."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    ctx=multiprocessing.get_context('spawn');events=ctx.Queue();workers=[];tasks=[];responses=[]
    assigned={};routes={};completed=[];started=time.monotonic();engine_seconds=0.;engine_steps=0
    last_heartbeat=started;batch_counts={};submitted=0;tokenization_seconds=0.
    try:
        for lane in range(lanes):
            job=take_job()
            if job is None:break
            q=ctx.Queue();r=ctx.Queue();tasks.append(q);responses.append(r)
            proc=ctx.Process(target=_worker,args=(lane,str(root),q,events,r,seconds_limit,action_limit,environment_factory,opponent_factory));proc.start();workers.append(proc)
            assigned[lane]=job;q.put(job)
        while assigned or routes:
            if time.monotonic()-started>seconds_limit+120:raise TimeoutError('continuous worker ceiling')
            if time.monotonic()-last_heartbeat>30:
                atomic_json(root/'progress.json',dict(assigned=list(assigned.values()),completed=len(completed),requests=submitted,engine_seconds=engine_seconds))
                if heartbeat:heartbeat()
                last_heartbeat=time.monotonic()
            batch=[]
            try:batch.append(events.get(timeout=.001 if routes else .1))
            except queue.Empty:pass
            while True:
                try:batch.append(events.get_nowait())
                except queue.Empty:break
            for kind,lane,*payload in batch:
                if kind=='failed':raise RuntimeError(payload[0])
                if kind=='done':
                    row=payload[0];completed.append(row);assigned.pop(lane)
                    atomic_json(root/'games.json',completed)
                    if on_complete:on_complete(row)
                    job=take_job()
                    tasks[lane].put(job)
                    if job is not None:assigned[lane]=job
                elif kind=='batch':
                    sequence,messages,queued=payload
                    if len(messages)!=1:raise ValueError('one outstanding decision per game required')
                    before=time.monotonic();request=generator.submit(messages[0]);tokenization_seconds+=time.monotonic()-before
                    submitted+=1;routes[request]=(lane,sequence)
                else:raise ValueError('unknown worker event')
            if routes:
                batch_counts[len(routes)]=batch_counts.get(len(routes),0)+1
                before=time.monotonic();outputs=generator.step();engine_seconds+=time.monotonic()-before;engine_steps+=1
                for request,generation in outputs:
                    lane,sequence=routes.pop(request);responses[lane].put(('result',sequence,[generation]))
            if any(p.exitcode not in (None,0) for p in workers):raise RuntimeError('game worker exited')
        for p in workers:p.join(timeout=10)
        report=dict(schema='continuous-gameplay-v2',games=completed,finished_games=len(completed),action_limit=action_limit,
            complete=all(g['status'] in ('native','truncated','forfeit') for g in completed),
            elapsed_seconds=time.monotonic()-started,engine_seconds=engine_seconds,engine_steps=engine_steps,requests=submitted,tokenization_seconds=tokenization_seconds,active_request_counts=batch_counts)
        atomic_json(root/'report.json',report);return report
    finally:
        for p in workers:
            if p.is_alive():p.terminate()
            p.join(timeout=5)
        for q in tasks+responses+[events]:q.close()
