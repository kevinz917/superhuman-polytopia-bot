"""Separate Harbor verifier. Trusted source rebuild, held-out games and replay audits."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from .common import canonical, sha, write_json
from .continuous import evaluation_jobs, rewards
from .local_eval import evaluate


def main():
    root = Path('/results')
    root.mkdir(exist_ok=True)
    logs = Path('/logs/verifier')
    logs.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path('/suite/manifest.json').read_text())
    suite = json.loads(Path('/suite/maps.json').read_text())
    profile = os.environ.get('SKIRMISH_EVALUATION_PROFILE', 'standard')
    if profile not in ('standard','demo'):
        raise ValueError('unknown evaluation profile')
    if profile == 'demo':
        suite = [next(m for m in suite if m['map_type']==family)
                 for family in ('drylands','lakes','continents')]
    jobs = evaluation_jobs(manifest, suite, demo=profile=='demo')
    capture_dir = Path('/verify/cutoffs/stage-0')
    capture = json.loads((capture_dir/'capture.json').read_text())
    source = (capture_dir/'main.rs').read_bytes() if not capture.get('error') else b''
    if not capture.get('error') and sha(source) != capture['source_sha256']:
        raise ValueError('trusted cutoff source hash mismatch')
    destination = root/'runs/candidate/stage-0'
    destination.mkdir(parents=True)
    (destination/'main.rs').write_bytes(source)
    Path('/verify/src/main.rs').write_bytes(source)
    proc = subprocess.run(['cargo','build','--manifest-path','/verify/Cargo.toml','--release',
        '--offline','--locked','--target','wasm32-wasip1'], capture_output=True, text=True, timeout=90)
    receipt = dict(source_sha256=sha(source), returncode=proc.returncode,
                   compile_output=(proc.stdout+proc.stderr)[-12000:])
    if not proc.returncode:
        wasm = Path('/verify/target/wasm32-wasip1/release/skirmish-bot.wasm').read_bytes()
        if not wasm.startswith(b'\0asm') or len(wasm)>8*1024*1024:
            receipt['returncode'] = -1
        else:
            (destination/'bot.wasm').write_bytes(wasm)
            receipt['wasm_sha256'] = sha(wasm)
    starter = sha(Path('/reference/STARTER.rs').read_bytes())
    from .protocol import submission_quality
    receipt['submission_quality'] = submission_quality(receipt['source_sha256'],starter,receipt['returncode'])
    write_json(destination/'submission.json', receipt)
    for job in jobs:
        job['candidate'].update(wasm_sha256=receipt.get('wasm_sha256'),
                                submission_quality=receipt['submission_quality'])
        if receipt['returncode']:
            job['invalid_submission'] = True
        job.pop('job_sha256')
        job['job_sha256'] = sha(canonical(job).encode())
    write_json(root/'evaluation-plan.json', dict(schema=manifest['schema'], profile=profile, jobs=jobs))
    rows = evaluate(dict(jobs=jobs, references=manifest['references'],
                         evaluation=manifest['evaluation'], referee_seconds=1800,compress_traces=True))
    result = rewards(rows, len(jobs))
    result['evaluation_is_demo'] = int(profile=='demo')
    write_json(logs/'games.json', rows)
    write_json(logs/'submission.json', receipt)
    if (destination/'bot.wasm').exists():
        shutil.copyfile(destination/'bot.wasm', logs/'bot.wasm')
    write_json(logs/'evaluation-plan.json', dict(profile=profile, jobs=jobs))
    write_json(logs/'summary.json', dict(schema=manifest['schema'], profile=profile,
        score=100*result['reward'], rewards=result, submission=receipt,
        replay_actions=sum(r['audit']['actions'] for r in rows),
        uncertainty='One independently authored bot; no build-variance confidence interval.'))
    if (root/'games').exists():
        shutil.move(str(root/'games'), str(logs/'traces'))
    # Commit the reward only after all required grading receipts are durable.
    write_json(logs/'reward.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
