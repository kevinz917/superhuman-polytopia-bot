"""Expert-SFT data-scaling curve: intermediate checkpoints of expert-sft-v1 on its 30 dev games. Modal only.

  modal run --detach --env staging -m experiments.modal_expert_scaling --action launch --run-id expert-sft-scaling-v1
  modal run --detach --env staging -m experiments.modal_expert_scaling --action resume --run-id expert-sft-scaling-v1
  modal run --env staging -m experiments.modal_expert_scaling --action summarize --run-id expert-sft-scaling-v1

checkpoint-157/314/471 (~25/50/75% of the 39,626 expert moves, one epoch, cosine LR) play exactly the
30 jobs of expert-sft-v1 (15 validation maps x 2 seats vs balanced) with the same harness, sampling,
per-decision seeds and LoRA-enabled vLLM engine (scripts/benchmark_expert_sft.py, unchanged). The final
adapter (checkpoint-620) and base arms are reused from expert-sft-v1 rather than replayed.

The coordinator is restart-safe: spawned call IDs are persisted in collection.json before anything
waits on them, and a restarted coordinator reattaches to those calls instead of spawning new GPU arms.
"""
import hashlib
import json
from pathlib import Path
import time
import modal
from experiments.modal_thinking_repaired import image, volume, ROOT

app = modal.App('polytopia-expert-scaling')
SOURCE_RUN = 'expert-sft-v1'
STEPS = [157, 314, 471]
FINAL_STEP = 620
RUNS = Path('/workspace/runs')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_protocol(root, protocol_sha):
    if sha(root/'protocol.json') != protocol_sha:
        raise ValueError('protocol changed')
    return json.loads((root/'protocol.json').read_text())


@app.function(image=image, gpu='H100', cpu=8, memory=49152, timeout=8*3600, max_containers=3, retries=0,
              scaledown_window=2, volumes={'/workspace/runs': volume})
def evaluate(run_id, protocol_sha, arm):
    import subprocess
    root = RUNS/run_id
    volume.reload()
    check_protocol(root, protocol_sha)
    if (root/arm/'report.json').exists():
        return dict(arm=arm, resumed=True)
    if (root/arm).exists():
        # A restarted input left a partial arm directory; the benchmark refuses to reuse it, so keep it aside.
        aside = root/f'{arm}.partial-{int(time.time())}'
        (root/arm).rename(aside); volume.commit()
    started = time.monotonic()
    proc = subprocess.Popen(['python', '/workspace/scripts/benchmark_expert_sft.py', str(root), arm], cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit(); time.sleep(10)
        result = dict(arm=arm, exit_code=proc.returncode, worker_seconds=time.monotonic()-started, finished_unix=time.time())
        (root/f'{arm}-execution.json').write_text(json.dumps(result, indent=2)); volume.commit()
        if proc.returncode:
            raise RuntimeError(result)
        return result
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=15)
        volume.commit()


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={'/workspace/runs': volume})
def summarize(run_id, protocol_sha):
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from summarize_expert_scaling import summarize as run
    root = RUNS/run_id
    volume.reload()
    check_protocol(root, protocol_sha)
    result = run(root)
    volume.commit()
    return result


@app.function(image=image, cpu=1, memory=4096, timeout=24*3600, max_containers=1, retries=0,
              volumes={'/workspace/runs': volume})
def coordinate(run_id, protocol_sha):
    import traceback
    root = RUNS/run_id
    volume.reload()
    p = check_protocol(root, protocol_sha)
    state = root/'collection.json'
    if state.exists():
        status = json.loads(state.read_text())
        status.setdefault('restarts', []).append(time.time())
    else:
        status = dict(status='starting', started_unix=time.time(), calls={}, restarts=[])
    def publish(**kw):
        status.update(kw, observed_unix=time.time())
        state.write_text(json.dumps(status, indent=2)+'\n'); volume.commit()
    try:
        calls = {}
        for arm in p['arms']:
            if (root/arm/'report.json').exists():
                continue
            if arm in status['calls']:
                calls[arm] = modal.FunctionCall.from_id(status['calls'][arm])
            else:
                calls[arm] = evaluate.spawn(run_id, protocol_sha, arm)
                status['calls'][arm] = calls[arm].object_id
                publish(status='running')  # persist each ID before waiting on anything
        publish(status='running')
        results = status.setdefault('evaluations', {})
        for arm, c in calls.items():
            try:
                results[arm] = c.get()
            except Exception as exc:
                results[arm] = dict(error=repr(exc)[:500])
            publish(evaluations=results)
        small = summarize.remote(run_id, protocol_sha)
        publish(status='complete', summary=small, finished_unix=time.time())
    except Exception:
        publish(status='failed', error=traceback.format_exc(), finished_unix=time.time())
        raise
    return status


def build_protocol(run_id):
    source = json.loads((ROOT/'runs'/SOURCE_RUN/'protocol.json').read_text())
    receipt = source['arms']['expert_sft']['training_receipt']
    run = json.loads(b''.join(volume.read_file(receipt)))
    if run['status'] != 'completed' or run['global_step'] != FINAL_STEP:
        raise ValueError('source training run not completed as expected')
    final_doc = json.loads((ROOT/'docs/expert-sft/checkpoint-expert-sft-v1.json').read_text())['adapter_files_sha256']
    if run['adapter_files'] != final_doc:
        raise ValueError('final adapter hashes disagree between run.json and docs')
    by_step = {c['step']: c for c in run['checkpoints']}
    if by_step[FINAL_STEP]['adapter_files'] != final_doc:
        raise ValueError('checkpoint-620 is not the final adapter')
    seen = {l['step']: l['examples_seen'] for l in run['logs']}
    model_dir = str(Path(receipt).parent)
    arms = {f'ckpt{s}': dict(adapter=f'{model_dir}/checkpoint-{s}', adapter_files=by_step[s]['adapter_files'],
                             step=s, examples_seen=seen[s], training_receipt=receipt) for s in STEPS}
    # Reuse expert-sft-v1's harness exactly; every source file it depends on is re-hashed at launch and
    # re-checked inside the worker.
    hashes = {rel: sha(ROOT/rel) for rel in source['source_hashes']}
    hashes['scripts/summarize_expert_scaling.py'] = sha(ROOT/'scripts/summarize_expert_scaling.py')
    changed = sorted(rel for rel, h in hashes.items() if source['source_hashes'].get(rel) not in (None, h))
    keep = ['model_id', 'revision', 'expert', 'lora', 'harness', 'named_fields', 'constrain_actions', 'max_length',
            'max_new_tokens', 'thinking', 'seed', 'sampling', 'concurrency', 'serving', 'gpu', 'jobs']
    return dict({k: source[k] for k in keep}, schema='expert-sft-scaling-v1', run_id=run_id, created_unix=time.time(),
        arms=arms, reference=dict(run=SOURCE_RUN, final_arm='expert_sft', final_step=FINAL_STEP,
                                  final_examples_seen=seen[FINAL_STEP], base_arm='base', training_records=run['data_manifest']['records'],
                                  effective_batch_size=run['config']['effective_batch_size']),
        source_hashes=hashes, source_files_changed_since_source_run=changed,
        source_protocol=dict(run_id=SOURCE_RUN, sha256=sha(ROOT/'runs'/SOURCE_RUN/'protocol.json')),
        limitations=['intermediate checkpoints of one cosine-LR run, not separate runs trained to completion on less data',
                     '15 validation maps x 2 seats; one sampling sequence per arm',
                     'final/base arms reused from expert-sft-v1, not replayed'])


@app.local_entrypoint()
def main(action: str = 'launch', run_id: str = ''):
    if not run_id:
        raise ValueError('run_id required')
    local = ROOT/'runs'/run_id
    if action == 'summarize':
        result = summarize.remote(run_id, sha(local/'protocol.json'))
        (local/'summary-small.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=1)[:6000])
        return
    if action == 'resume':
        # Respawn only the coordinator; it reattaches to persisted arm calls and skips finished arms.
        call = coordinate.spawn(run_id, sha(local/'protocol.json'))
        receipt = dict(app_id=app.app_id, call_id=call.object_id, protocol_sha256=sha(local/'protocol.json'),
                       launcher_sha256=sha(Path(__file__)), resumed_unix=time.time())
        (local/f'dispatch-resume-{int(time.time())}.json').write_text(json.dumps(receipt, indent=2)+'\n'); print(json.dumps(receipt))
        return
    if action != 'launch':
        raise ValueError('action must be launch, resume or summarize')
    if local.exists():
        raise ValueError(f'{local} exists; choose a new run id')
    protocol = build_protocol(run_id)
    local.mkdir(parents=True)
    (local/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    protocol_sha = sha(local/'protocol.json')
    with volume.batch_upload() as up:
        up.put_file(local/'protocol.json', f'{run_id}/protocol.json')
    call = coordinate.spawn(run_id, protocol_sha)
    receipt = dict(app_id=app.app_id, call_id=call.object_id, protocol_sha256=protocol_sha, launcher_sha256=sha(Path(__file__)))
    (local/'dispatch.json').write_text(json.dumps(receipt, indent=2)+'\n'); print(json.dumps(receipt))
