"""One infrastructure retry for the preempted shard of the existing repair pass.

Reuse the frozen evaluate function, jobs, model, GPU and time ceiling. Preserve
all successful shards and the failed attempt; never resample maps or outcomes.
"""
import hashlib,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from modal_prompt_games import app,evaluate,volume


@app.local_entrypoint()
def recover():
    from modal.config import config as modal_config
    if modal_config.get('environment')!='staging':raise ValueError('staging required')
    out=ROOT/'runs/sft0/relabel-games-v1'
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol['run_id']!='sft0-relabel-v1' or protocol['repair_pass']!=1 or not protocol['training_only']:
        raise ValueError('registered repair collection required')
    for relative,sha in protocol['sources'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=sha:raise ValueError('frozen source changed')
    for receipt in protocol['prerequisites']:
        if hashlib.sha256((ROOT/receipt['path']).read_bytes()).hexdigest()!=receipt['sha256']:raise ValueError('prerequisite changed')
    log=ROOT/'runs/sft0/relabel-games-v1.log'
    if 'Container terminated due to preemption' not in log.read_text():raise ValueError('documented provider preemption required')
    original=(out/'execution.json').read_bytes();results=json.loads(original)
    if len(results)!=3 or any(r.get('exit_code')!=0 for r in results):raise ValueError('exactly three successful prior shards required')
    for i,r in enumerate(results):
        if r['archive']!=f'sft0-relabel-v1-shard{i}.tar' or not r['report']['complete']:
            raise ValueError('prior shard identity/completion mismatch')
        if hashlib.sha256((out/r['archive']).read_bytes()).hexdigest()!=r['archive_sha256']:
            raise ValueError('prior archive changed')
    recovery_path=out/'preemption-recovery-v1.json'
    record=dict(schema='sft-shard-infrastructure-recovery-v1',reason='Modal preempted shard3; automatic restart collided with its preserved directory.',
        original_app_id='ap-m6fJAsgp4KrAaFxoDGItlF',original_shard='sft0-relabel-v1-shard3',
        recovery_shard='sft0-relabel-v1-shard3-recovery1',repair_pass=1,
        same_jobs=protocol['jobs'][3::4],original_execution_sha256=hashlib.sha256(original).hexdigest(),
        original_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        recovery_launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        failed_attempt_preserved='runs/sft0/relabel-preempted-shard3-v1',
        note='Infrastructure retry only. Three successful shards retained; all twelve failed-shard jobs rerun in the original order and batch policy. No outcome-based selection, policy change or second repair search. Failed-attempt cost reported separately.')
    with recovery_path.open('x') as f:json.dump(record,f,indent=2)
    with (out/'execution-before-recovery-v1.json').open('xb') as f:f.write(original)
    result=evaluate.with_options(timeout=protocol['seconds_limit']+120).remote(
        record['recovery_shard'],protocol['jobs'][3::4],protocol['config'],protocol['seconds_limit'],
        protocol['adapter'],protocol['adapter_files'],{k:protocol[k] for k in ('tokenizer_hash','chat_template_hash')})
    (out/'recovery-execution-v1.json').write_text(json.dumps(result,indent=2))
    archive=out/result['archive']
    with archive.open('xb') as f:
        for chunk in volume.read_file(result['archive']):f.write(chunk)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=result['archive_sha256']:raise ValueError('recovery archive hash mismatch')
    with tarfile.open(archive) as tar:tar.extractall(out,filter='data')
    if result.get('exit_code')!=0 or not result.get('report',{}).get('complete'):raise ValueError('recovery did not complete; no further retry')
    results.append(result)
    reports=[r['report'] for r in results];games=[g for r in reports for g in r['games']]
    if len(games)!=50 or {g['id'] for g in games}!={j['id'] for j in protocol['jobs']}:raise ValueError('exactly original fifty jobs required')
    models=[json.loads((out/f'sft0-relabel-v1-shard{i}'/'model.json').read_text()) for i in range(3)]
    models.append(json.loads((out/record['recovery_shard']/'model.json').read_text()))
    for model in models:
        for key in ('config','adapter','adapter_files','tokenizer_hash','chat_template_hash'):
            if model[key]!=models[0][key]:raise ValueError('model interface differs between shards')
        for key in ('tokenizer_hash','chat_template_hash'):
            if model[key]!=protocol[key]:raise ValueError('frozen tokenizer mismatch')
    combined=out/'combined';(combined/'games').mkdir(parents=True,exist_ok=False)
    merged=dict(schema='gameplay-v1',allocated_games=50,finished_games=50,games=games,unstarted_jobs=[],complete=True,
        elapsed_seconds=sum(r['elapsed_seconds'] for r in results),occupied_seconds_with_archive=sum(r['occupied_seconds_with_archive'] for r in results),
        shard_archives=[r['archive'] for r in results],infrastructure_recovery_receipt='preemption-recovery-v1.json',
        cost_note='Successful shard execution only; preserved preempted attempt cost is additional and recorded separately.')
    (out/'execution.json').write_text(json.dumps(results,indent=2))
    (combined/'games/report.json').write_text(json.dumps(merged,indent=2))
    (combined/'model.json').write_text(json.dumps({**models[0],'shard_models':models},indent=2))
    print(json.dumps({k:v for k,v in merged.items() if k!='games'},indent=2))
