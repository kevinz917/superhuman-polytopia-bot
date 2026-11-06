#!/usr/bin/env python3
"""Select one main-seed checkpoint after all three validation allocations finish."""
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.checkpoint_selection import select_checkpoint
from polytopia_bot.learning.freeze import inference_config


def select(training_result,evaluations,output):
    evidence=[]
    def read(path):
        p=ROOT/path;value=json.loads(p.read_text())
        evidence.append(dict(path=str(path),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        return value
    execution=read(training_result/'execution.json');run=execution.get('run') or {}
    if execution.get('exit_code')!=0 or run.get('status')!='completed' or run.get('stage')!='main':
        raise ValueError('completed main training run required')
    allocation=read(training_result/'allocation.json')
    state=read(training_result/'training-run/model/trainer_state.json')
    total=state['global_step'];steps={math.ceil(total*f) for f in (.25,.5,1)}
    if total!=state['max_steps'] or run['seed'] not in (17,29):raise ValueError('main run incomplete or unregistered seed')
    maps=read(Path('runs/sft0/splits-v1/maps.json'))['maps']['validation']
    frozen=run['data_manifest']['prompt_manifest']
    expected=[dict(seed=m['seed'],board_hash=m['board_hash'],seat=seat,opponent='weak',strategy=frozen['strategy']) for m in maps for seat in (1,2)]
    reports={};adapters={};common_sources=None
    for folder in evaluations:
        protocol=read(folder/'protocol.json');model=read(folder/'combined/model.json')
        remote=model['adapter'];snapshot_name=Path(remote).name
        if not snapshot_name.startswith('checkpoint-'):raise ValueError('checkpoint-specific model required')
        step=int(snapshot_name.split('-')[1])
        if step not in steps or step in reports or remote!='/workspace/runs/'+allocation['run_id']+'/model/'+snapshot_name:
            raise ValueError('unexpected or duplicated main snapshot')
        if inference_config(model['config'])!=inference_config(allocation['config']):raise ValueError('evaluation budget/model differs from main training')
        if protocol['prompt_manifest_hash']!=frozen['manifest_hash']:raise ValueError('different frozen prompt')
        if protocol.get('adapter')!=remote:raise ValueError('model differs from its allocation')
        for name,sha in frozen['sources'].items():
            if protocol['sources'].get(name)!=sha:raise ValueError('frozen public interface changed')
        for key in ('tokenizer_hash','chat_template_hash'):
            if model[key]!=frozen[key]:raise ValueError('tokenizer changed')
        sources={k:v for k,v in protocol['sources'].items() if k.startswith('src/polytopia_bot/')}
        if common_sources is not None and sources!=common_sources:raise ValueError('implementation changed across checkpoints')
        common_sources=sources
        local=ROOT/training_result/'training-run/model'/snapshot_name
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in local.glob('adapter*') if p.is_file()}
        if 'adapter_model.safetensors' not in hashes or any(model['adapter_files'].get(k)!=v for k,v in hashes.items()):
            raise ValueError('evaluated adapter differs from downloaded checkpoint')
        receipts=read(folder/'execution.json')
        if len(receipts)!=protocol['shards'] or any(r.get('exit_code')!=0 for r in receipts):raise ValueError('evaluation shard failed or missing')
        reports[step]=read(folder/'combined/games/report.json');adapters[step]=dict(remote=remote,local=str(local.relative_to(ROOT)),files=hashes)
    result=dict(schema='main-checkpoint-selection-v1',seed=run['seed'],**select_checkpoint(reports,expected,steps),evidence=evidence)
    result['selected_adapter']=adapters[result['selected_step']]
    with (ROOT/output).open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-result',type=Path,required=True)
    p.add_argument('--evaluations',nargs=3,type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();select(a.training_result,a.evaluations,a.output)
