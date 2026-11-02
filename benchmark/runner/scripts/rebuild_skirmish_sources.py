"""Rebuild frozen Rust sources on Modal; only compact proof receipts leave the volume."""
from pathlib import Path
import hashlib
import json
import re
import modal

app=modal.App('polytopia-skirmish-source-rebuilds')
volume=modal.Volume.from_name('polytopia-skirmish',environment_name='staging')
image=modal.Image.debian_slim(python_version='3.12')


def cohort_root(prefix):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}',prefix):raise ValueError('Invalid cohort ID')
    return Path('/results/mini-v5')/prefix


@app.function(image=image,cpu=1,memory=1024,timeout=240,max_containers=6,volumes={'/results':volume})
def rebuild(prefix: str,run_id: str,stage: int):
    volume.reload();base=cohort_root(prefix);plan=json.loads((base/'authoring-plan.json').read_text())
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}',run_id):raise ValueError('Invalid run ID')
    if not any(run_id==c['run_id'] or run_id.startswith(c['run_id']+'-infra') for c in plan['cells']):raise ValueError('Run is outside this cohort')
    root=Path('/results/mini-runs')/run_id;folder=root/f'stage-{stage}'
    receipt=json.loads((folder/'submission.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    source=(folder/'main.rs').read_bytes();source_sha=hashlib.sha256(source).hexdigest();assert source_sha==receipt['source_sha256']
    assert protocol['coding_image_id']==plan['image_id'];expected=receipt.get('wasm_sha256')
    if expected:assert hashlib.sha256((folder/'bot.wasm').read_bytes()).hexdigest()==expected
    sandbox=modal.Sandbox.create('sleep','infinity',image=modal.Image.from_id(plan['image_id']),cpu=2,memory=2048,block_network=True,timeout=180,workdir='/verify',app=modal.App.lookup('polytopia-skirmish-sandboxes',environment_name='staging',create_if_missing=True))
    try:
        sandbox.filesystem.write_bytes(source,'/verify/src/main.rs')
        process=sandbox.exec('cargo','build','--manifest-path','/verify/Cargo.toml','--release','--offline','--locked','--target','wasm32-wasip1',timeout=120)
        output=process.stdout.read()+process.stderr.read();process.wait()
        wasm=sandbox.filesystem.read_bytes('/verify/target/wasm32-wasip1/release/skirmish-bot.wasm') if process.returncode==0 else None
        actual=hashlib.sha256(wasm).hexdigest() if wasm is not None else None
        valid=wasm is not None and wasm.startswith(b'\0asm') and len(wasm)<=8*1024*1024
        result=dict(run_id=run_id,stage=stage,source_sha256=source_sha,returncode=process.returncode,
            expected_wasm_sha256=expected,rebuilt_wasm_sha256=actual,
            identical=actual==expected if expected else not valid,
            compiler_output_sha256=hashlib.sha256(output.encode()).hexdigest(),
            scope='source-only rebuild in fresh pinned offline sandbox',wasm_bytes=len(wasm) if wasm is not None else None)
    finally:sandbox.terminate()
    base=base/'stage-rebuild-checks';base.mkdir(exist_ok=True);target=base/f'{run_id}-stage-{stage}.json'
    if target.exists():
        history=base/'history';history.mkdir(exist_ok=True);previous=json.loads(target.read_text())
        (history/f'{run_id}-stage-{stage}-{previous["source_sha256"]}.json').write_text(json.dumps(previous,indent=2))
    temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(target);volume.commit();return result


@app.function(image=image,cpu=1,memory=1024,timeout=180,volumes={'/results':volume.with_mount_options(read_only=True)})
def pending(prefix: str,all_attempts: bool=False):
    volume.reload();base=cohort_root(prefix);plan=json.loads((base/'authoring-plan.json').read_text());jobs=[]
    for cell in plan['cells']:
        roots=[Path('/results/mini-runs')/cell['run_id']]+sorted(Path('/results/mini-runs').glob(cell['run_id']+'-infra*'),key=lambda p:int(p.name.rsplit('infra',1)[1]))
        if not all_attempts:
            roots=[p for p in roots if (p/'report.json').exists() and json.loads((p/'report.json').read_text())['status'] in ('completed','failed_submission')]
        for root in roots:
            if not (root/'protocol.json').exists():continue
            for folder in root.glob('stage-*'):
                if not folder.is_dir() or not (folder/'submission.json').exists():continue
                stage=int(folder.name.split('-')[-1]);current=json.loads((folder/'submission.json').read_text())
                path=base/'stage-rebuild-checks'/f'{root.name}-stage-{stage}.json';previous=json.loads(path.read_text()) if path.exists() else {}
                if previous.get('source_sha256')!=current['source_sha256'] or previous.get('expected_wasm_sha256')!=current.get('wasm_sha256'):jobs.append((prefix,root.name,stage))
    return jobs


@app.function(image=image,cpu=1,memory=1024,timeout=60,volumes={'/results':volume.with_mount_options(read_only=True)})
def receipts(prefix: str):
    volume.reload();return [json.loads(p.read_text()) for p in sorted((cohort_root(prefix)/'stage-rebuild-checks').glob('*.json'))]


@app.local_entrypoint()
def verify(prefix: str,output: str,all_attempts: bool=False):
    jobs=pending.remote(prefix,all_attempts);list(rebuild.starmap(jobs));rows=receipts.remote(prefix)
    Path(output).write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(checked=len(rows),matched=sum(r['identical'] for r in rows),new=len(jobs),compiled=sum(r['expected_wasm_sha256'] is not None for r in rows))))
