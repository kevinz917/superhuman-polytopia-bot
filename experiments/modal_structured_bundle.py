"""Copy immutable retained checkpoints into training bundles entirely in Modal.

This verifies file provenance and optimizer contents, not playing strength.
Selection and any champion promotion remain separate evaluation decisions.
"""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('numpy>=2.1')
    .add_local_dir(ROOT/'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_file(ROOT/'scripts/verify_ruleset.py', '/workspace/scripts/verify_ruleset.py')
    .add_local_dir(ROOT/'configs', '/workspace/configs'))
app = modal.App('polytopia-checkpoint-bundles', image=image)
retained = modal.Volume.from_name('polytopia-structured-rl-retained', environment_name='staging')
training = modal.Volume.from_name('polytopia-structured-rl', environment_name='staging')


@app.function(cpu=(1,1), memory=(2048,2048), timeout=3600,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={'/retained': retained.with_mount_options(read_only=True),
                       '/training': training})
def materialize(plan):
    import hashlib
    import re
    import shutil
    import subprocess
    import sys
    import tempfile
    import time
    import numpy as np

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def safe_name(value):
        if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',value):
            raise ValueError('Invalid immutable bundle identity')
        return value

    subprocess.run([sys.executable,'/workspace/scripts/verify_ruleset.py'],
                   cwd='/workspace',check=True)
    safe_name(plan['run_id'])
    items = plan['bundles']
    if not 1 <= len(items) <= 8 or len({x['bundle_id'] for x in items}) != len(items):
        raise ValueError('Expected one to eight distinct bundles')
    receipt_root = Path('/training/bundle-materializations')/plan['run_id']
    receipt_root.mkdir(parents=True,exist_ok=False)
    (receipt_root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    reports = []
    try:
        for item in items:
            run = safe_name(item['training_run_id'])
            stage = item['stage']
            expected = item['sha256']
            name = safe_name(item['bundle_id'])
            if (not re.fullmatch(r'update-[0-9]+',stage)
                    or not re.fullmatch(r'[0-9a-f]{64}',expected)
                    or not name.endswith('-'+expected[:12])):
                raise ValueError('Bundle must identify an exact retained update and hash')
            source = Path('/retained/checkpoints')/run/(stage+'-'+expected[:12])
            destination = Path('/training')/name
            if destination.exists():
                raise FileExistsError('Never overwrite an existing training bundle: '+name)
            protocol = json.loads((source/'protocol.json').read_text())
            metadata = json.loads((source/'latest.json').read_text())
            initial = json.loads((source/'initial.json').read_text())
            if metadata['stage'] != stage or metadata['sha256'] != expected:
                raise ValueError('Retained checkpoint identity mismatch')
            for filename,meta in [('latest.npz',metadata),('initial.npz',initial)]:
                if digest(source/filename) != meta['sha256']:
                    raise ValueError('Checkpoint hash mismatch: '+filename)
                if meta['ruleset'] != 'bardur-1v1-v1' or meta['simulator_sha256'] != protocol['simulator_sha256']:
                    raise ValueError('Frozen simulator provenance mismatch')
                with np.load(source/filename,allow_pickle=False) as weights:
                    if not all(np.isfinite(weights[key]).all() for key in weights.files):
                        raise ValueError('Nonfinite checkpoint tensor')
                    count = sum(weights[key].size for key in weights.files if key.startswith('p'))
                    if count != meta['parameter_elements'] or 'o0' not in weights.files:
                        raise ValueError('Incomplete parameter or optimizer checkpoint')
                    if filename == 'latest.npz':
                        parameters,steps = count,int(weights['o0'])
            if not metadata['config']['no_resign']:
                raise ValueError('Non-resigning policy contract required')
            for relative,expected_source in protocol['source_sha256'].items():
                relative = Path(relative)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Invalid archived source path')
                if digest(source/'source'/relative) != expected_source:
                    raise ValueError('Archived source hash mismatch')
            for relative in ('rl/network.py','rl/observation.py','rl/actions.py','rl/policy.py'):
                if digest(Path('/workspace/src/polytopia_bot')/relative) != protocol['source_sha256'][relative]:
                    raise ValueError('Current inference source differs from checkpoint')
            pending = Path(tempfile.mkdtemp(prefix='.'+name+'-',dir='/training'))
            for original,target in [('latest.npz','final.npz'),('latest.json','final.json'),
                                    ('initial.npz','initial.npz'),('initial.json','initial.json'),
                                    ('protocol.json','protocol.json')]:
                shutil.copyfile(source/original,pending/target)
                if digest(pending/target) != digest(source/original):
                    raise ValueError('Copied artifact readback mismatch')
            shutil.copytree(source/'source',pending/'source')
            for relative,expected_source in protocol['source_sha256'].items():
                if digest(pending/'source'/relative) != expected_source:
                    raise ValueError('Copied source readback mismatch')
            report = dict(bundle_id=name,training_run_id=run,stage=stage,sha256=expected,
                source=str(source),destination=str(destination),parameter_count=parameters,
                optimizer_steps=steps,all_tensors_finite=True,
                source_hashes_verified=len(protocol['source_sha256']),
                current_inference_hashes_verified=4,
                protocol_sha256=digest(pending/'protocol.json'),
                checkpoint_readback_sha256=digest(pending/'final.npz'),
                copied_at_unix=time.time(),promoted_to_champion=False,
                scope='Cloud-only immutable artifact copy; no inference or strength certification')
            (pending/'bundle-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
            pending.rename(destination)
            reports.append(report)
            (receipt_root/'results.json').write_text(json.dumps(reports,indent=2)+'\n')
            training.commit()
        return dict(run_id=plan['run_id'],complete=True,bundles=reports)
    finally:
        training.commit()


@app.local_entrypoint()
def main(plan_path: str):
    import sys
    import time
    from modal.config import config
    if config.get('environment') != 'staging':
        raise RuntimeError('staging only')
    sys.path.insert(0,str(ROOT/'src'))
    from polytopia_bot.rl.budget import reserve,record
    plan = json.loads(Path(plan_path).read_text())
    folder = ROOT/'runs/structured-ppo-v3'
    reserve(folder,dict(run_id=plan['run_id'],resource_type='cpu-checkpoint-verifier',
        gpu_type=None,cpu_cores=1,memory_gib=2,rate_usd_per_hour=.07,
        reserved_usd=2,hard_timeout_seconds=3600,status='reserved',
        modal_app_id=app.app_id,start_unix=time.time(),plan_path=plan_path))
    call = materialize.spawn(plan)
    record(folder,plan['run_id'],status='submitted',function_call_id=call.object_id)
    receipt = dict(run_id=plan['run_id'],modal_app_id=app.app_id,function_call_id=call.object_id)
    (folder/(plan['run_id']+'-dispatch.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
