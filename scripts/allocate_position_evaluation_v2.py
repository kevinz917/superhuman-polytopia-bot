#!/usr/bin/env python3
"""Freeze the fresh common-32K base/repaired-smoke position comparison."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.freeze import verify_prompt, inference_config


def allocate(output,run_id,*,adapter_result=None,seconds_per_shard=1800):
    if Path(run_id).name!=run_id or run_id in ('.','..'):raise ValueError('run ID must be a simple new name')
    preflight=ROOT/'runs/sft0/validation-preflight-v2/manifest.json'
    data=json.loads(preflight.read_text())
    if data['count']!=5000 or not data['passed'] or data['shards']!=4:raise ValueError('complete position preflight required')
    verify_prompt(data['prompt_manifest'],data['config'])
    adapter=None;receipt=None;kind='base';adapter_files=None
    if adapter_result:
        result=json.loads((ROOT/adapter_result/'execution.json').read_text())
        receipt=result['run']
        if result.get('exit_code')!=0 or receipt.get('status')!='completed' or receipt.get('stage')!='smoke':
            raise ValueError('only a completed smoke adapter may enter this paired smoke comparison')
        allocation=json.loads((ROOT/adapter_result/'allocation.json').read_text())
        if inference_config(allocation['config'])!=inference_config(data['config']):
            raise ValueError('smoke training and paired evaluation inference configuration differ')
        adapter='/workspace/runs/'+allocation['run_id']+'/model/adapter-final';kind='smoke'
        adapter_root=ROOT/adapter_result/'training-run/model/adapter-final'
        adapter_files={str(p.relative_to(adapter_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in adapter_root.rglob('*') if p.is_file()}
        if any(adapter_files.get(name)!=expected for name,expected in receipt['adapter_files'].items()):
            raise ValueError('downloaded adapter differs from training receipt')
        if 'adapter_model.safetensors' not in adapter_files:raise ValueError('downloaded adapter required')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    files=[*sorted((ROOT/'src/polytopia_bot/learning').glob('*.py')),
           ROOT/'training/pyproject.toml',ROOT/'training/uv.lock',ROOT/'experiments/modal_position_eval.py',Path(__file__)]
    protocol=dict(schema='position-evaluation-allocation-v1',kind=kind,run_id=run_id,adapter=adapter,
        adapter_run_receipt=receipt,adapter_files=adapter_files,config=data['config'],prompt_manifest=data['prompt_manifest'],
        shard_files=sorted(data['files']),uploads=data['files'],preflight_sha256=sha(preflight),
        sources={str(p.relative_to(ROOT)):sha(p) for p in files},seconds_per_shard=seconds_per_shard,
        total_occupied_gpu_hours_ceiling=seconds_per_shard*4/3600,
        planning_compute_allowance_usd=round((seconds_per_shard+120)*4*.00078192,2),
        allocation_note='Four independent A100 inference shards, no optimizer; total cap includes every shard. The 120-second-per-shard cost reserve covers startup/archiving overhead. No final-test positions.',
        batch_policy=data['batch_policy'])
    path=ROOT/output
    with path.open('x') as stream:json.dump(protocol,stream,indent=2)
    archive=path.with_suffix('');archive.mkdir(exist_ok=False)
    for name in protocol['sources']:
        target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    print(json.dumps({k:v for k,v in protocol.items() if k not in ('sources','uploads','config','prompt_manifest','adapter_run_receipt')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--adapter-result',type=Path);p.add_argument('--seconds-per-shard',type=int,default=1800)
    a=p.parse_args();allocate(a.output,a.run_id,adapter_result=a.adapter_result,seconds_per_shard=a.seconds_per_shard)
