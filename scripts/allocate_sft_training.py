#!/usr/bin/env python3
"""Bind a bounded training job to audited data and the frozen prompt."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.interface import digest
from polytopia_bot.learning.training import verified_data
from polytopia_bot.learning.freeze import inference_config


def allocate(prepared,output,*,stage,seed,run_id,seconds_limit,gpu='A100-80GB',fit_result=None,smoke_gate=None,label_audit=None):
    manifest=json.loads((ROOT/prepared/'manifest.json').read_text());config=manifest['config']
    q=ROOT/'runs/sft0/teacher-qualification-v3/report.json'
    labels=ROOT/(label_audit or 'runs/sft0/smoke-label-audit-v1/report.json')
    label_protocol=labels.parent/'protocol.json'
    if not json.loads(q.read_text())['teacher_qualified'] or not json.loads(labels.read_text())['passed']:
        raise ValueError('teacher and label replay gates must pass')
    if stage in ('smoke','main') and json.loads(label_protocol.read_text())['sample_records_sha256']!=manifest['source_sha256']:
        raise ValueError('label replay audit must cover this exact sampled corpus')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    environment=ROOT/'configs/ruleset-v1.json'
    for name,expected in json.loads(environment.read_text())['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('frozen simulator changed')
    readiness=dict(full_game_ready=True,teacher_qualified=True,teacher_hashes=manifest['audit']['teacher_hashes'],
        split_manifest_hash=digest(manifest['split_manifest']),
        environment_receipt=dict(path=str(environment.relative_to(ROOT)),sha256=sha(environment)),
        teacher_receipt=dict(path=str(q.relative_to(ROOT)),sha256=sha(q)),
        label_audit_receipt=dict(path=str(labels.relative_to(ROOT)),sha256=sha(labels)),
        label_audit_protocol_receipt=dict(path=str(label_protocol.relative_to(ROOT)),sha256=sha(label_protocol)))
    verified_data(ROOT/prepared,config,stage=stage,readiness=readiness)
    if stage=='fit' and (config['epochs']!=8 or config['effective_batch_size']!=8 or seed!=17):
        raise ValueError('first fit uses the recorded 8-epoch, effective-batch-8, seed-17 allocation')
    gate_receipts={}
    if stage=='smoke':
        if fit_result is None:raise ValueError('smoke allocation requires the completed fit result')
        fit=json.loads((ROOT/fit_result/'execution.json').read_text())
        r=fit.get('run') or {};before=r.get('fit_before',{}).get('accuracy',0);after=r.get('fit_after',{}).get('accuracy',0)
        if r.get('stage')!='fit' or inference_config(r['config'])!=inference_config(config):
            raise ValueError('fit must use the same pinned model and inference settings')
        if fit.get('exit_code')!=0 or fit.get('reload_exit_code')!=0 or r.get('status')!='completed' or not (after>=.9 or after>=.5 and after-before>=.3):
            raise ValueError('fit accuracy, completion or reload gate failed')
        if r.get('frozen_weights_before')!=r.get('frozen_weights_after') or r.get('adapter_weights_before')==r.get('adapter_weights_after'):
            raise ValueError('fit parameter-update gate failed')
        gate_receipts[str(fit_result/'execution.json')]=sha(ROOT/fit_result/'execution.json')
    if stage=='main':
        if smoke_gate is None or not json.loads((ROOT/smoke_gate).read_text()).get('passed'):
            raise ValueError('main allocation requires completed saved-position, live-game and cost gates')
        gate_receipts[str(smoke_gate)]=sha(ROOT/smoke_gate)
    files=[p for p in (ROOT/prepared).rglob('*') if p.is_file()]
    files.extend([q,labels,label_protocol])
    files.extend(ROOT/r['path'] for r in manifest['prompt_manifest']['evidence'])
    sources=[*sorted((ROOT/'src/polytopia_bot/learning').glob('*.py')),
        ROOT/'training/pyproject.toml',ROOT/'training/uv.lock',environment,
        ROOT/'experiments/modal_sft_train.py',Path(__file__)]
    if gpu not in ('A100-80GB','H100','B200'):raise ValueError('unsupported training GPU')
    rate={'A100-80GB':.000694,'H100':.001097,'B200':.001736}[gpu]+4*.0000131+16*.00000222
    allocation=dict(schema='sft-training-allocation-v1',stage=stage,prepared=str(prepared),run_id=run_id,
        seed=seed,gpu=gpu,config=config,readiness=readiness,seconds_limit=seconds_limit,
        fit_result=None if fit_result is None else str(fit_result),smoke_gate=None if smoke_gate is None else str(smoke_gate),
        planning_compute_allowance_usd=round(seconds_limit*rate,2),
        processed_tokens_per_epoch=manifest['processed_tokens'],planned_processed_tokens=manifest['processed_tokens']*config['epochs'],
        gate_receipts=gate_receipts,sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        uploads={str(p.relative_to(ROOT)):sha(p) for p in files},
        fit_gate=dict(base_unchanged=True,adapter_changed=True,reloaded_greedy_responses_identical=True,
            finite_loss_and_gradients=True,response_accuracy='After >=90%, or after >=50% with at least 30 percentage points gain',
            note='Plumbing/overfit diagnostic only; this is not the smoke or main checkpoint') if stage=='fit' else None,
        budget_note='Occupied function time planning ceiling; startup, image builds and storage charges recorded separately. Main training requires the separate smoke and cost gate.')
    with (ROOT/output).open('x') as stream:json.dump(allocation,stream,indent=2)
    archive=(ROOT/output).with_suffix('');archive.mkdir(exist_ok=False)
    for name in allocation['sources']:
        path=archive/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    print(json.dumps({k:v for k,v in allocation.items() if k not in ('sources','uploads','readiness','config')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['fit','smoke','main'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--seconds-limit',type=int,required=True)
    p.add_argument('--gpu',choices=['A100-80GB','H100','B200'],default='A100-80GB');p.add_argument('--fit-result',type=Path);p.add_argument('--smoke-gate',type=Path)
    p.add_argument('--label-audit',type=Path)
    a=p.parse_args();allocate(a.prepared,a.output,stage=a.stage,seed=a.seed,run_id=a.run_id,seconds_limit=a.seconds_limit,gpu=a.gpu,fit_result=a.fit_result,smoke_gate=a.smoke_gate,label_audit=a.label_audit)
