"""One bounded 700-position prompt screen; gameplay decides any prompt change."""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
# Keep the remote entry module self-contained: Modal mounts this file, not the
# local experiments package. The locked runtime layers match the profile image.
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('uv==0.9.25')
    .add_local_file(ROOT/'training/pyproject.toml','/workspace/training/pyproject.toml',copy=True)
    .add_local_file(ROOT/'training/uv.lock','/workspace/training/uv.lock',copy=True)
    .run_commands('UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple uv sync --project /workspace/training --locked --no-dev')
    .env({'PYTHONPATH':'/workspace/src','TOKENIZERS_PARALLELISM':'false'})
    .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
    .add_local_file(ROOT/'configs/sft.json','/workspace/config.json'))
app=modal.App('polytopia-sft0-prompt-screen',image=image)


@app.function(gpu='A100-80GB',cpu=4,memory=16384,timeout=3000,max_containers=1,retries=0,scaledown_window=2)
def screen(positions,strategies,config):
    import subprocess,time
    start=time.monotonic()
    Path('/tmp/screen-input.json').write_text(json.dumps(dict(positions=positions,strategies=strategies,config=config)))
    code=r'''
from dataclasses import asdict
import json,time
from pathlib import Path
import torch
from polytopia_bot.learning.serving import BatchGenerator
from polytopia_bot.learning.prompt_search import candidate_messages
from polytopia_bot.learning.data import observation_from_dict
from polytopia_bot.learning.interface import parse_response,render,ResponseError,digest
payload=json.loads(Path('/tmp/screen-input.json').read_text())
generator=BatchGenerator(payload['config']);results=[]
for name,strategy in payload['strategies'].items():
 pending=[];rows=[]
 for position in payload['positions']:
  obs=observation_from_dict(position['observation']);p=render(obs)
  pending.append(dict(position=position,prompt=p,messages=candidate_messages(obs,strategy),attempts=[]))
 for attempt in range(3):
  retry=[]
  for start in range(0,len(pending),8):
   chunk=pending[start:start+8]
   torch.cuda.synchronize();began=time.monotonic()
   outputs=generator.batch([item['messages'] for item in chunk]);torch.cuda.synchronize()
   elapsed=time.monotonic()-began
   for item,out in zip(chunk,outputs):
    entry=dict(messages=item['messages'],generation=asdict(out),batch_size=len(chunk),batch_seconds=elapsed)
    try:
     action=parse_response(out.text,item['prompt'],current_version=item['prompt'].state_version)
    except ResponseError as exc:
     entry['error']=str(exc);item['attempts'].append(entry)
     if attempt<2:
      item['messages']=item['messages']+[dict(role='assistant',content=out.text),dict(role='user',content='Invalid response: '+str(exc)+'. Return one legal action_id JSON object.')]
      retry.append(item)
     else:
      rows.append(dict(position_id=item['position']['position_id'],attempts=item['attempts'],valid=False,first_attempt_valid=False,mechanically_correct=False if item['position']['mechanically_acceptable'] is not None else None,teacher_agreement=False))
     continue
    item['attempts'].append(entry)
    acceptable=item['position']['mechanically_acceptable']
    rows.append(dict(position_id=item['position']['position_id'],attempts=item['attempts'],selected_command=dict(action.command),valid=True,first_attempt_valid=attempt==0,mechanically_correct=None if acceptable is None else dict(action.command) in acceptable,teacher_agreement=dict(action.command)==item['position']['teacher_command']))
  pending=retry
  if not pending:break
 summary=dict(candidate=name,decisions=len(rows),first_attempt_valid=sum(r['first_attempt_valid'] for r in rows),eventually_valid=sum(r['valid'] for r in rows),mechanical_count=sum(r['mechanically_correct'] is not None for r in rows),mechanical_correct=sum(r['mechanically_correct'] is True for r in rows),teacher_agreement=sum(r['teacher_agreement'] for r in rows),attempts=sum(len(r['attempts']) for r in rows))
 print(json.dumps(summary),flush=True)
 results.append(dict(summary=summary,rows=rows))
Path('/tmp/screen-result.json').write_text(json.dumps(dict(config=payload['config'],results=results,device=torch.cuda.get_device_name(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),tokenizer_hash=digest(generator.tokenizer.backend_tokenizer.to_str()),chat_template_hash=digest(generator.tokenizer.chat_template))))
'''
    subprocess.run(['/workspace/training/.venv/bin/python','-c',code],check=True,timeout=2900)
    result=json.loads(Path('/tmp/screen-result.json').read_text());result['function_elapsed_seconds']=time.monotonic()-start
    return result


@app.local_entrypoint()
def main(development: str='runs/sft0/prompt-development-v2',qualification: str='runs/sft0/teacher-qualification-v3/report.json',output: str='runs/sft0/prompt-screen-v2'):
    import sys,hashlib
    sys.path.insert(0,str(ROOT/'src'))
    from polytopia_bot.learning.prompt_search import STRATEGIES
    from polytopia_bot.learning.interface import digest
    from modal.config import config as modal_config
    if modal_config.get('environment')!='staging':raise RuntimeError('Pass --env staging')
    readiness=json.loads((ROOT/qualification).read_text())
    if (ROOT/qualification).parent.joinpath('invalidation.json').exists() or (ROOT/development/'invalidation.json').exists():
        raise ValueError('Invalidated qualification or development inputs cannot enter a prompt screen')
    if readiness.get('teacher_qualified') is not True:raise ValueError('Teacher must qualify before prompt optimization')
    qualified=json.loads((ROOT/qualification).parent.joinpath('protocol.json').read_text())['source_sha256']
    development_sources=json.loads((ROOT/development/'source-receipt.json').read_text())['sources']
    for name in ('adapter','interface','encoding','memory','teacher'):
        relative=f'src/polytopia_bot/learning/{name}.py'
        current=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
        if qualified.get(relative)!=current or development_sources.get(relative)!=current:
            raise ValueError(f'{name} differs from qualified teacher/development interface')
    out=ROOT/output;out.mkdir(parents=True,exist_ok=False)
    positions=json.loads((ROOT/development/'positions.json').read_text())
    if len(positions)!=100 or len(STRATEGIES)!=7:raise ValueError('exact registered screen size required')
    config=json.loads((ROOT/'configs/sft.json').read_text())
    protocol=dict(schema='prompt-screen-v1',positions_hash=digest(positions),strategies=STRATEGIES,
        qualification_receipt=qualification,qualification_sha256=hashlib.sha256((ROOT/qualification).read_bytes()).hexdigest(),
        config=config,first_attempt_valid_min=95,eventually_valid_min=99,mechanical_fraction_min=.95,
        ranking='first-attempt validity, then mechanical fraction, then fixed candidate insertion order; teacher agreement diagnostic only',
        proposals=dict(method='One assistant-authored reflective batch from public development traces and the failed ID-format profile',model='gpt-6-astra',reasoning_effort='high',session='01a146f6-f752-79e2-9ba1-df309df6243d',exact_served_revision='not exposed by Codex; alias and exact proposal text retained'),
        gpu_hours_ceiling=1,retry_function_seconds_ceiling=3000,failed_attempt='runs/sft0/prompt-screen-v1/failure.json',planning_compute_allowance_usd=3.5,no_gameplay_winner_selected_here=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    result=screen.remote(positions,STRATEGIES,config)
    # Retain all exact prompts, attempts and token sequences for independent review.
    import gzip
    with gzip.open(out/'results.json.gz','wt',compresslevel=1) as stream:json.dump(result,stream)
    summaries=[r['summary'] for r in result['results']]
    survivors=[s for s in summaries if s['candidate']!='original' and s['first_attempt_valid']>=95 and s['eventually_valid']>=99 and (not s['mechanical_count'] or s['mechanical_correct']/s['mechanical_count']>=.95)]
    order={name:i for i,name in enumerate(STRATEGIES)}
    survivors.sort(key=lambda s:(-s['first_attempt_valid'],-s['mechanical_correct']/max(1,s['mechanical_count']),order[s['candidate']]))
    report=dict(summaries=summaries,selected_for_gameplay=[s['candidate'] for s in survivors[:2]],
        proceed_to_gameplay=len(survivors)>=2,selected_prompt=None,function_elapsed_seconds=result['function_elapsed_seconds'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
