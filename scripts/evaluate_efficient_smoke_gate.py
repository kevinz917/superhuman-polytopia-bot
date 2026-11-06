#!/usr/bin/env python3
"""Gate the common pipelined base/SFT comparison with its actual resource rate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.smoke_gate import position_comparison,game_comparison


def evaluate(base_positions,smoke_positions,base_games,smoke_games,base_replay,smoke_replay,output):
    evidence=[]
    def read(path):
        path=ROOT/path
        evidence.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return json.loads(path.read_text())
    bp,sp=read(base_positions/'allocation.json'),read(smoke_positions/'allocation.json')
    for key in ('uploads','preflight_sha256','batch_policy'):
        if bp[key]!=sp[key]:raise ValueError('saved-position allocation mismatch: '+key)
    if bp['kind']!='base' or sp['kind']!='smoke':raise ValueError('paired base/smoke allocations required')
    # Added adapter/tokenizer receipt verification does not change inference.
    for name,sha in bp['sources'].items():
        if name.startswith('src/polytopia_bot/learning/') and sp['sources'].get(name)!=sha:
            raise ValueError('saved-position implementation changed: '+name)
    for directory,protocol in ((base_positions,bp),(smoke_positions,sp)):
        execution=read(directory/'execution.json')
        if len(execution)!=4 or any(e.get('exit_code')!=0 for e in execution):
            raise ValueError('all position shards must complete successfully')
        for i in range(4):
            model=read(directory/f'shard-{i}/model.json')
            for key in ('tokenizer_hash','chat_template_hash'):
                if model[key]!=protocol['prompt_manifest'][key]:raise ValueError('tokenizer mismatch')
            if model['adapter']!=protocol['adapter']:raise ValueError('model adapter mismatch')
    positions=position_comparison(read(base_positions/'report.json'),read(smoke_positions/'report.json'))
    bg,sg=read(base_games/'protocol.json'),read(smoke_games/'protocol.json')
    for name,sha in bg['sources'].items():
        if name.startswith('src/polytopia_bot/') and sg['sources'].get(name)!=sha:
            raise ValueError('gameplay implementation changed: '+name)
    for directory,protocol in ((base_games,bg),(smoke_games,sg)):
        execution=read(directory/'execution.json')
        if len(execution)!=4 or any(e.get('exit_code')!=0 for e in execution):
            raise ValueError('all gameplay shards must complete successfully')
        model=read(directory/'combined/model.json')
        for key in ('tokenizer_hash','chat_template_hash'):
            if model[key]!=protocol[key]:raise ValueError('game tokenizer mismatch')
        if model['adapter']!=protocol['adapter']:raise ValueError('game model adapter mismatch')
    if sp['adapter']!=sg['adapter'] or sp['adapter_files']!=sg['adapter_files']:
        raise ValueError('different adapters in saved-position and gameplay checks')
    for key in ('runtime_version','batch_policy','cpu_lanes_per_gpu','shards'):
        if bg.get(key)!=sg.get(key):raise ValueError('pipeline protocol differs: '+key)
    if bg.get('runtime_version')!='pipelined-cpu-lanes-v1':raise ValueError('pipelined protocol required')
    games=game_comparison(read(base_games/'combined/games/report.json'),read(smoke_games/'combined/games/report.json'),bg,sg)
    for name,directory in [('base',base_games),('smoke',smoke_games)]:
        entry=games['models'][name]
        coordinator=read(directory/'coordinator.json')
        cost=entry['occupied_function_seconds']*.00079968+coordinator['elapsed_seconds']*.00001532
        entry.update(estimated_compute_usd=cost,estimated_compute_usd_per_game=cost/50)
    games['projected_full_evaluation_usd']=2400*games['models']['base']['estimated_compute_usd_per_game']+6400*games['models']['smoke']['estimated_compute_usd_per_game']
    for source,audit_path in ((base_games,base_replay),(smoke_games,smoke_replay)):
        replay=read(audit_path)
        expected_hash=hashlib.sha256((ROOT/source/'combined/games/report.json').read_bytes()).hexdigest()
        if not replay.get('passed') or replay['games']!=50 or replay['report_sha256']!=expected_hash:
            raise ValueError('complete matching game replay audit required')
    learning_passed=positions['format_passed'] and positions['learning_passed'] and games['completion_passed'] and games['execution_passed']
    result=dict(schema='sft-smoke-diagnostic-v1',positions=positions,games=games,
        learning_diagnostic_passed=learning_passed,passed=False,
        live_game_replay_passed=True,
        remaining_gates=['Review observation/rule integrity evidence and all failures.',
                        'The common32K context budget is already versioned; retain it in any subsequent main protocol.',
                        'Set a full-protocol spend limit from measured throughput before main training.'],
        evidence=evidence,limitation='A learning diagnostic is not final checkpoint acceptance or spending authorization.')
    with (ROOT/output).open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base-positions','smoke-positions','base-games','smoke-games','base-replay','smoke-replay','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();evaluate(a.base_positions,a.smoke_positions,a.base_games,a.smoke_games,a.base_replay,a.smoke_replay,a.output)
