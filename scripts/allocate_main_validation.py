#!/usr/bin/env python3
"""Freeze all 200 validation maps/both seats for base or one main snapshot.

This only writes an allocation. It requires the reviewed smoke/cost gate and
never creates final-test jobs or starts a provider function.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.freeze import inference_config, verify_prompt
from polytopia_bot.learning.interface import digest


def allocate(output,run_id,config_path,prompt_path,gate_path,seconds_limit,training_result=None,step=None):
    if Path(run_id).name!=run_id or run_id in ('.','..'):
        raise ValueError('simple fresh run ID required')
    if not 600<=seconds_limit<=28800:
        raise ValueError('bounded per-shard work limit required')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    prerequisites=[]
    def read(path):
        target=ROOT/path;prerequisites.append(target)
        return json.loads(target.read_text())
    config=read(config_path);prompt=read(prompt_path);verify_prompt(prompt,config)
    gate=read(gate_path)
    if not gate.get('passed'):
        raise ValueError('reviewed learning, integrity, context and spending gate required')
    maps=read(Path('runs/sft0/splits-v1/maps.json'))
    if digest({k:v for k,v in maps.items() if k!='manifest_hash'})!=maps['manifest_hash']:
        raise ValueError('split digest mismatch')
    if maps['size']!=14 or len(maps['maps']['validation'])!=200:
        raise ValueError('all 200 reserved 14x14 validation maps required')
    qualified=read(Path('runs/sft0/qualified-teacher-v1.json'))
    if not qualified['qualification']['teacher_qualified']:
        raise ValueError('qualified teacher required')
    qualification=ROOT/qualified['qualification_path'];prerequisites.append(qualification)
    if sha(qualification)!=qualified['qualification_sha256']:
        raise ValueError('qualification changed')
    for name,expected in qualified['interface_sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('public interface or teacher changed')
    adapter=None;adapter_files=None;kind='base';seed=None
    if training_result is not None:
        execution=read(training_result/'execution.json');run=execution.get('run') or {}
        training=read(training_result/'allocation.json')
        state=read(training_result/'training-run/model/trainer_state.json')
        if execution.get('exit_code')!=0 or run.get('status')!='completed' or run.get('stage')!='main':
            raise ValueError('completed main training required')
        if inference_config(training['config'])!=inference_config(config):
            raise ValueError('main training and evaluation settings differ')
        if run['data_manifest']['prompt_manifest']['manifest_hash']!=prompt['manifest_hash']:
            raise ValueError('main training used a different frozen prompt')
        seed=run['seed'];total=state['global_step']
        if seed not in (17,29) or total!=state['max_steps'] or step not in {math.ceil(total*f) for f in (.25,.5,1)}:
            raise ValueError('registered main seed and 25/50/100-percent snapshot required')
        snapshot=f'checkpoint-{step}'
        local=ROOT/training_result/'training-run/model'/snapshot
        adapter_files={str(p.relative_to(local)):sha(p) for p in local.rglob('*')
                       if p.is_file() and p.name not in ('optimizer.pt','scheduler.pt','rng_state.pth')}
        if not {'adapter_config.json','adapter_model.safetensors'}<=adapter_files.keys():
            raise ValueError('downloaded checkpoint adapter required')
        adapter='/workspace/runs/'+training['run_id']+'/model/'+snapshot;kind='main'
    elif step is not None:
        raise ValueError('base evaluation has no checkpoint step')
    jobs=[dict(id=f'{kind}-{m["seed"]}-{seat}',seed=m['seed'],board_hash=m['board_hash'],seat=seat,
               candidate=prompt['candidate'],strategy=prompt['strategy'],opponent='weak')
          for m in maps['maps']['validation'] for seat in (1,2)]
    files=[*sorted((ROOT/'src/polytopia_bot/learning').glob('*.py')),ROOT/'training/pyproject.toml',
           ROOT/'training/uv.lock',ROOT/'configs/ruleset-v1.json',ROOT/'experiments/modal_prompt_games.py',Path(__file__)]
    rules=json.loads((ROOT/'configs/ruleset-v1.json').read_text())
    for name,expected in rules['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('frozen simulator changed')
        files.append(ROOT/name)
    protocol=dict(schema='sft-gameplay-allocation-v1',stage='main-validation',kind=kind,run_id=run_id,
        seed=seed,checkpoint_step=step,jobs=jobs,config=config,adapter=adapter,adapter_files=adapter_files,
        prompt_manifest_hash=prompt['manifest_hash'],tokenizer_hash=prompt['tokenizer_hash'],
        chat_template_hash=prompt['chat_template_hash'],map_manifest_hash=maps['manifest_hash'],
        weak_opponent_hash=qualified['weak_opponent_hash'],
        prerequisites=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in prerequisites],
        sources={str(p.relative_to(ROOT)):sha(p) for p in files},shards=4,seconds_limit=seconds_limit,
        occupied_gpu_hours_ceiling=4*(seconds_limit+120)/3600,
        planning_compute_allowance_usd=round(4*(seconds_limit+120)*.00078192,2),
        comparison='All 200 reserved validation maps, both seats, frozen weak opponent; same base/main inference settings.',
        final_test=False,apex=False)
    target=ROOT/output
    with target.open('x') as stream:json.dump(protocol,stream,indent=2)
    archive=target.with_suffix('');archive.mkdir(exist_ok=False)
    for name in protocol['sources']:
        path=archive/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    print(json.dumps({k:v for k,v in protocol.items() if k not in ('jobs','config','sources','prerequisites','adapter_files')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','config','prompt','gate'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--seconds-limit',type=int,required=True)
    p.add_argument('--training-result',type=Path);p.add_argument('--step',type=int)
    a=p.parse_args();allocate(a.output,a.run_id,a.config,a.prompt,a.gate,a.seconds_limit,a.training_result,a.step)
