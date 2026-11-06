"""Evaluation-engine speed benchmark on real expert-sft-v1 prompts; Modal only.

  modal run --env staging -m experiments.modal_eval_bench --action smoke
  modal run --detach --env staging -m experiments.modal_eval_bench --action launch --run-id eval-speedup-v1
  modal run --env staging -m experiments.modal_eval_bench --action summarize --run-id eval-speedup-v1

One H100 per engine config. Every config replays the same 960 saved prompts (32 per expert-sft-v1
game, evenly spaced over the game) at concurrency 30 and then 60, with the expert-SFT adapter, the
legal-action grammar and the original per-decision seeds. `v1` is the expert-sft-v1 engine and `v1b`
a second copy of it, so v1 vs v1b is the run-to-run noise floor that any new setting is judged against.
"""
import hashlib
import json
from pathlib import Path
import time
import modal
from experiments.modal_thinking_repaired import image, volume, ROOT

app = modal.App('polytopia-eval-bench')
RUNS = Path('/workspace/runs')
SCRIPT = '/workspace/scripts/eval_replay_bench.py'
CONFIGS = {
    'v1': dict(serving={}),
    'v1b': dict(serving={}),
    'pc': dict(serving=dict(enable_prefix_caching=True)),
    'b16k': dict(serving=dict(max_num_batched_tokens=16384)),
    'b16k-pc': dict(serving=dict(max_num_batched_tokens=16384, enable_prefix_caching=True)),
    'b32k-pc': dict(serving=dict(max_num_batched_tokens=32768, enable_prefix_caching=True)),
    'b64k-pc': dict(serving=dict(max_num_batched_tokens=65536, enable_prefix_caching=True)),
    'b32k-pc-gmu90': dict(serving=dict(max_num_batched_tokens=32768, enable_prefix_caching=True,
                                       gpu_memory_utilization=.90)),
    # Throughput only: same engine, no adapter attached (bounds what merging the LoRA could buy).
    'v1-nolora': dict(serving={}, adapter=False),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_script(root, args, commit_every=15):
    import subprocess
    proc = subprocess.Popen(['python', SCRIPT, *args], cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit(); time.sleep(commit_every)
        if proc.returncode:
            raise RuntimeError(f'{args} exited {proc.returncode}')
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=15)
        volume.commit()


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={'/workspace/runs': volume})
def extract(run_id, protocol_sha):
    root = RUNS/run_id
    volume.reload()
    if sha(root/'protocol.json') != protocol_sha:
        raise ValueError('protocol changed')
    if not (root/'prompts-manifest.json').exists():
        run_script(root, ['extract', str(root)])
    return json.loads((root/'prompts-manifest.json').read_text())


@app.function(image=image, gpu='H100', cpu=8, memory=49152, timeout=3*3600, max_containers=10, retries=0,
              scaledown_window=2, volumes={'/workspace/runs': volume})
def bench(run_id, protocol_sha, config, extra=()):
    root = RUNS/run_id
    volume.reload()
    if sha(root/'protocol.json') != protocol_sha:
        raise ValueError('protocol changed')
    if (root/'bench'/config/'summary.json').exists() and not extra:
        return json.loads((root/'bench'/config/'summary.json').read_text())
    run_script(root, ['run', str(root), config, *extra])
    return json.loads((root/'bench'/config/'summary.json').read_text())


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={'/workspace/runs': volume})
def summarize(run_id):
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from eval_replay_bench import summarize as run
    volume.reload()
    result = run(RUNS/run_id)
    volume.commit()
    return result


@app.function(image=image, cpu=1, memory=4096, timeout=6*3600, max_containers=1, retries=0,
              volumes={'/workspace/runs': volume})
def coordinate(run_id, protocol_sha):
    import traceback
    root = RUNS/run_id
    status = dict(status='extracting', started_unix=time.time())
    def publish(**kw):
        status.update(kw, observed_unix=time.time())
        (root/'collection.json').write_text(json.dumps(status, indent=2, default=str)+'\n'); volume.commit()
    try:
        publish(manifest=extract.remote(run_id, protocol_sha))
        p = json.loads((root/'protocol.json').read_text())
        calls = {c: bench.spawn(run_id, protocol_sha, c) for c in p['configs']}
        publish(status='running', calls={k: c.object_id for k, c in calls.items()})
        done = {}
        for c, call in calls.items():
            try:
                s = call.get()
                done[c] = [{k: ph[k] for k in ('concurrency', 'prompt_tokens_per_second', 'ttft_median', 'ttft_p90')}
                           for ph in s['phases']]
            except Exception as exc:
                done[c] = dict(error=repr(exc)[:500])
            publish(results=done)
        summarize.remote(run_id)
        publish(status='complete', finished_unix=time.time())
    except Exception:
        publish(status='failed', error=traceback.format_exc(), finished_unix=time.time())
        raise
    return status


@app.local_entrypoint()
def main(action: str = 'smoke', run_id: str = '', configs: str = ''):
    if action == 'summarize':
        result = summarize.remote(run_id)
        local = ROOT/'runs'/run_id
        (local/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result['throughput'], indent=1)); print(json.dumps(
            [{k: (v if not isinstance(v, dict) else {x: v[x] for x in ('n', 'identical_tokens', 'identical_action')})
              for k, v in e.items()} for e in result['equivalence']], indent=1))
        return
    run_id = run_id or (f'eval-speedup-smoke-{int(time.time())}' if action == 'smoke' else '')
    if not run_id:
        raise ValueError('run_id required')
    local = ROOT/'runs'/run_id
    if local.exists():
        raise ValueError(f'{local} exists; choose a new run id')
    local.mkdir(parents=True)
    chosen = {c: CONFIGS[c] for c in (configs.split(',') if configs else CONFIGS)}
    protocol = dict(schema='eval-speedup-bench-v1', run_id=run_id, created_unix=time.time(),
        source=dict(run_id='expert-sft-v1', arm='expert_sft',
                    protocol_sha256=sha(ROOT/'runs/expert-sft-v1/protocol.json')),
        prompts_per_game=32, warmup_games=8, concurrencies=[30, 60], configs=chosen,
        source_hashes={rel: sha(ROOT/rel) for rel in ['scripts/eval_replay_bench.py', 'scripts/benchmark_expert_sft.py',
                                                      'scripts/benchmark_thinking_unbounded.py']},
        launcher_sha256=sha(Path(__file__)))
    (local/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    protocol_sha = sha(local/'protocol.json')
    with volume.batch_upload() as up:
        up.put_file(local/'protocol.json', f'{run_id}/protocol.json')
    if action == 'smoke':
        print('SMOKE EXTRACT', json.dumps(extract.remote(run_id, protocol_sha)))
        for c in chosen:
            s = bench.remote(run_id, protocol_sha, c, ('--limit', '48', '--concurrency', '8'))
            print('SMOKE', c, json.dumps(s['phases'], default=str)[:1500])
        return
    if action != 'launch':
        raise ValueError('action must be smoke, launch or summarize')
    call = coordinate.spawn(run_id, protocol_sha)
    receipt = dict(app_id=app.app_id, call_id=call.object_id, protocol_sha256=protocol_sha)
    (local/'dispatch.json').write_text(json.dumps(receipt, indent=2)+'\n'); print(json.dumps(receipt))
