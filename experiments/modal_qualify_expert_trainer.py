"""Expert-trainer runtime qualification receipts (Modal only; reads expert-sft-v1/prepared read-only).

  modal run --env staging -m experiments.modal_qualify_expert_trainer --action tests
  modal run --env staging -m experiments.modal_qualify_expert_trainer --action qualify [--steps 30]

qualify runs scripts/qualify_expert_trainer.py on 4 x B200 with the arms below; the first arm is the
reference (the selected variant of the current receipt in movement-scale40k-v1/protocol.json). Results
land in trainer-qualification/<tag>/ on volume polytopia-sft0-training and in runs/trainer-qualification/<tag>/:
qualification.json (every probe, trajectory and gate) and receipt-<candidate>.json (pass one to
experiments.modal_expert_sft --benchmark-receipt). The default runtime is not changed by this script.
"""
import hashlib
import json
import time
from pathlib import Path
import modal
from experiments.modal_expert_sft import training_image, volume, ROOT

app = modal.App('polytopia-qualify-expert-trainer')
image = training_image.add_local_dir(ROOT/'tests', '/workspace/tests', ignore=['**/__pycache__/**'])
RUNS = Path('/workspace/runs')
PREPARED = 'expert-sft-v1/prepared'
CONFIG_SOURCE = 'expert-sft-v1/protocol.json'
BASE_SOURCE = 'movement-scale40k-v1/protocol.json'
OUT = 'trainer-qualification'
TESTS = ['tests/test_fast_training.py', 'tests/test_trainer_qualification.py', 'tests/test_efficient_training.py']
REFERENCE_KEYS = ('name', 'packed', 'backend', 'token_budget', 'checkpointing', 'skip_lora_cast')
# Noise-floor arm, then the candidates (docs/sft-trainer-benchmark-2026-10-19.md section 5).
ARMS = [dict(name='b16384', backend='segmented_sdpa', token_budget=16384, checkpointing=False, skip_lora_cast=True),
        dict(name='b16384_swiglu', backend='segmented_sdpa', token_budget=16384, checkpointing=False, skip_lora_cast=True,
             liger_swiglu=True),
        dict(name='cudnn_b256_swiglu_16384', backend='segmented_cudnn', token_budget=16384, checkpointing=False,
             skip_lora_cast=True, liger_swiglu=True)]
CANDIDATES = ['b16384_swiglu', 'cudnn_b256_swiglu_16384']


@app.function(image=image, gpu='L4', cpu=4, memory=16384, timeout=1800, retries=0)
def tests():
    import os, subprocess
    proc = subprocess.run(['/workspace/training/.venv/bin/python', '-m', 'pytest', '-q', '-rs', *TESTS], cwd='/workspace',
                          capture_output=True, text=True, env=dict(os.environ, PYTHONPATH='/workspace/src:/workspace/tests'))
    return dict(exit_code=proc.returncode, output=proc.stdout[-12000:]+proc.stderr[-4000:])


@app.function(image=image, gpu='B200:4', cpu=16, memory=131072, timeout=3*3600, retries=0, scaledown_window=2,
              volumes={'/workspace/runs': volume})
def qualify(tag, steps, warmup):
    import subprocess
    volume.reload()
    config_bytes, base_bytes = (RUNS/CONFIG_SOURCE).read_bytes(), (RUNS/BASE_SOURCE).read_bytes()
    base = json.loads(base_bytes)['training']['benchmark']
    reference = {k: base['selected'][k] for k in REFERENCE_KEYS if k in base['selected']}
    root = RUNS/OUT/tag; root.mkdir(parents=True, exist_ok=False)
    payload = dict(tag=tag, prepared=str(RUNS/PREPARED), config=json.loads(config_bytes)['training']['config'], seed=17,
                   arms=[reference, *ARMS], candidates=CANDIDATES, steps=steps, warmup_steps=warmup,
                   base=dict(source=dict(path=BASE_SOURCE, sha256=hashlib.sha256(base_bytes).hexdigest()),
                             speedup=base['speedup'], selected=reference),
                   config_source=dict(path=CONFIG_SOURCE, sha256=hashlib.sha256(config_bytes).hexdigest()))
    (root/'input.json').write_text(json.dumps(payload, indent=2))
    started = time.monotonic()
    proc = subprocess.Popen(['/workspace/training/.venv/bin/python', '-m', 'torch.distributed.run', '--standalone',
                             '--nproc_per_node=4', '/workspace/scripts/qualify_expert_trainer.py', str(root/'input.json'),
                             str(root)], cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit(); time.sleep(20)
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=30)
        volume.commit()
    files = {p.name: p.read_text() for p in root.glob('*.json')}
    return dict(exit_code=proc.returncode, container_seconds=time.monotonic()-started, files=files)


@app.local_entrypoint()
def main(action: str = 'tests', steps: int = 30, warmup: int = 2, tag: str = ''):
    if action == 'tests':
        result = tests.remote()
        print(result['output'])
        if result['exit_code']:
            raise SystemExit(result['exit_code'])
        return
    if action != 'qualify':
        raise ValueError('action must be tests or qualify')
    tag = tag or f'qualify-b200x4-{int(time.time())}'
    result = qualify.remote(tag, steps, warmup)
    local = ROOT/'runs'/OUT/tag; local.mkdir(parents=True, exist_ok=True)
    for name, text in result['files'].items():
        (local/name).write_text(text)
    print(json.dumps(dict(exit_code=result['exit_code'], container_seconds=result['container_seconds'], local=str(local))))
    summary = json.loads(result['files'].get('qualification.json', '{}'))
    for name, gates in summary.get('candidates', {}).items():
        print(name, {k: g.get('passed') for k, g in gates.items() if isinstance(g, dict)})
