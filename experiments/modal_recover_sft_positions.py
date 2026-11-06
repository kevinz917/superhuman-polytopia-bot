"""Recover only the disconnected final shard of repaired smoke positions."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modal_position_eval import app, run_shard, volume


@app.local_entrypoint()
def recover():
    from modal.config import config as modal_config
    if modal_config.get('environment') != 'staging':
        raise ValueError('staging required')
    out = ROOT / 'runs/sft0/repaired-positions-v2'
    protocol = json.loads((out / 'allocation.json').read_text())
    if protocol['run_id'] != 'sft0-repaired-positions-v2':
        raise ValueError('registered allocation required')
    for name, sha in protocol['sources'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError('frozen source changed: ' + name)
    original = (out / 'execution.json').read_bytes()
    results = json.loads(original)
    if len(results) != 3:
        raise ValueError('exactly three completed shards required')
    for i, result in enumerate(results):
        if (result.get('exit_code') != 0 or not result['report']['complete'] or
                result['archive'] != f'sft0-repaired-positions-v2-shard-{i}.tar'):
            raise ValueError('successful original shard identity required')
        with (out / result['archive']).open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != result['archive_sha256']:
                raise ValueError('original archive changed')
    log = ROOT / 'runs/sft0/repaired-positions-v2.log'
    if 'Stopping app - local client disconnected' not in log.read_text():
        raise ValueError('documented infrastructure interruption required')
    with (out / 'execution-before-recovery-v1.json').open('xb') as f:
        f.write(original)
    recovery = {**protocol, 'run_id': 'sft0-repaired-positions-v2-recovery1'}
    with (out / 'recovery-allocation-v1.json').open('x') as f:
        json.dump(recovery, f, indent=2)
    call = run_shard.with_options(timeout=protocol['seconds_per_shard'] + 120).spawn(recovery, 3)
    (out / 'recovery-call-v1.json').write_text(json.dumps({'call_id': call.object_id}))
    result = call.get()
    (out / 'recovery-execution-v1.json').write_text(json.dumps(result, indent=2))
    archive = out / result['archive']
    with archive.open('xb') as f:
        for chunk in volume.read_file(result['archive']):
            f.write(chunk)
    with archive.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != result['archive_sha256']:
            raise ValueError('recovery archive hash mismatch')
    if (out / 'shard-3').exists():
        raise ValueError('never overwrite original evidence')
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter='data')
    if result.get('exit_code') != 0 or not result.get('report', {}).get('complete'):
        raise ValueError('recovery incomplete')
    results.append(result)
    rows = [row for r in results for row in r['report']['results']]
    if len(rows) != 5000 or len({r['position_id'] for r in rows}) != 5000:
        raise ValueError('exactly 5000 unique original positions required')
    report = dict(schema='saved-position-evaluation-v1', allocated=5000,
        decisions=len(rows), complete=True, results=rows,
        first_attempt_valid=sum(r['first_attempt_valid'] for r in rows),
        eventually_valid=sum(r['valid'] for r in rows),
        teacher_agreement=sum(r['teacher_agreement'] for r in rows),
        attempts=sum(r['attempts'] for r in rows),
        occupied_function_seconds=sum(r['elapsed_seconds'] for r in results),
        occupied_seconds_with_archive=sum(r['occupied_seconds_with_archive'] for r in results),
        recovery_receipt='runs/sft0/evaluation-disconnect-recovery-v2.json',
        cost_note='Successful shards only; interrupted attempt cost is separately recorded.')
    (out / 'execution.json').write_text(json.dumps(results, indent=2))
    with (out / 'report.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))
