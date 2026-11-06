"""Saved-position evaluation; these decisions never advance a game."""
from dataclasses import asdict
import gzip,json,time
from pathlib import Path
from .data import observation_from_dict
from .interface import parse_response,render,ResponseError,canonical,digest
from .prompt_search import candidate_messages


def evaluate(output,positions,generator,*,strategy='',batch_size=8,seconds_limit=3600):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();results=[];incomplete=[]
    if len({p['position_id'] for p in positions})!=len(positions):raise ValueError('duplicate evaluation positions')
    with gzip.open(root/'attempts.jsonl.gz','xt',compresslevel=1) as log:
        for start in range(0,len(positions),batch_size):
            if time.monotonic()-started>=seconds_limit:
                incomplete.extend(p['position_id'] for p in positions[start:]);break
            pending=[]
            for position in positions[start:start+batch_size]:
                obs=observation_from_dict(position['observation'])
                messages=candidate_messages(obs,strategy)
                if position.get('prompt_digest') and digest(messages)!=position['prompt_digest']:
                    raise ValueError('evaluation prompt differs from frozen preflight')
                pending.append(dict(position=position,prompt=render(obs),messages=messages,attempts=[]))
            for attempt in range(3):
                if time.monotonic()-started>=seconds_limit:
                    incomplete.extend(item['position']['position_id'] for item in pending);break
                retry=[];batch_start=time.monotonic()
                outputs=generator.batch([item['messages'] for item in pending])
                seconds=time.monotonic()-batch_start
                if len(outputs)!=len(pending):raise ValueError('generation batch length mismatch')
                for item,generation in zip(pending,outputs):
                    entry=dict(messages=item['messages'],generation=asdict(generation),batch_seconds=seconds,batch_size=len(pending))
                    action=None
                    try:action=parse_response(generation.text,item['prompt'],current_version=item['prompt'].state_version)
                    except ResponseError as exc:
                        entry['error']=str(exc)
                    item['attempts'].append(entry)
                    if action is None and attempt<2:
                        item['messages']=item['messages']+[dict(role='assistant',content=generation.text),
                            dict(role='user',content='Invalid response: '+entry['error']+'. Return one legal action_id JSON object.')]
                        retry.append(item);continue
                    position=item['position']
                    row=dict(position_id=position['position_id'],board_hash=position.get('board_hash'),
                        episode_id=position.get('episode_id'),phase=position.get('phase'),family=position['teacher_command']['kind'],
                        valid=action is not None,first_attempt_valid=action is not None and attempt==0,
                        teacher_agreement=action is not None and dict(action.command)==position['teacher_command'],
                        selected_command=None if action is None else dict(action.command),teacher_command=position['teacher_command'],
                        attempts=len(item['attempts']),input_tokens=len(item['attempts'][0]['generation']['prompt_token_ids']))
                    log.write(canonical(dict(result=row,attempts=item['attempts']))+'\n');results.append(row)
                pending=retry
                if not pending:break
            if incomplete:
                incomplete.extend(p['position_id'] for p in positions[start+batch_size:]);break
            if len(results)%100<batch_size:print(canonical(dict(positions_complete=len(results),allocated=len(positions))),flush=True)
    report=dict(schema='saved-position-evaluation-v1',allocated=len(positions),decisions=len(results),
        complete=len(results)==len(positions),incomplete=incomplete,results=results,
        first_attempt_valid=sum(r['first_attempt_valid'] for r in results),eventually_valid=sum(r['valid'] for r in results),
        teacher_agreement=sum(r['teacher_agreement'] for r in results),attempts=sum(r['attempts'] for r in results),
        elapsed_seconds=time.monotonic()-started)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
