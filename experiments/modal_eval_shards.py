"""Split one evaluation arm's games across N H100 workers, then merge into the usual arm layout; Modal only.

  modal run --env staging -m experiments.modal_eval_shards --action smoke
  modal run --detach --env staging -m experiments.modal_eval_shards --action arm --run-id <run> --arm <arm> --shards 4
  modal run --env staging -m experiments.modal_eval_shards --action merge --run-id <run> --arm <arm> --shards 4

Works on any run directory whose protocol scripts/benchmark_expert_sft.py accepts (expert-sft-v1,
expert-sft-heldout-*, expert-h2h-*). Shard i plays jobs[i::N] of the arm with the protocol's engine
settings into <arm>/shards/i-of-N/; `merge` then refuses shards whose engine args, packages, model,
tokenizer, template or adapter hashes differ and writes <arm>/{runtime.json, games/, streams/,
generation-timing.jsonl, report.json}, so the existing summarizers read a sharded arm unchanged.
Per-decision seeds depend only on the job and decision index, so sharding changes batch composition
(and therefore last-bit numerics, as concurrency already does) but no seed, prompt or setting.

A launcher adopts it with `app.include(experiments.modal_eval_shards.app)` and calls
`evaluate_sharded.spawn(run_id, protocol_sha, arm, shards)` instead of its own `evaluate`.
"""
import hashlib
import json
from pathlib import Path
import time
import modal
from experiments.modal_thinking_repaired import image, volume, ROOT

app = modal.App('polytopia-eval-shards')
retained = modal.Volume.from_name('polytopia-structured-rl-retained', environment_name='staging')
RUNS = Path('/workspace/runs')
SCRIPT = '/workspace/scripts/benchmark_expert_sft.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_protocol(root, protocol_sha):
    if sha(root/'protocol.json') != protocol_sha:
        raise ValueError('protocol changed')
    return json.loads((root/'protocol.json').read_text())


def shard_name(index, count):
    return f'{index:02d}-of-{count:02d}'


def run_script(args):
    import subprocess
    started = time.monotonic()
    proc = subprocess.Popen(['python', SCRIPT, *args], cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit(); time.sleep(10)
        return proc.returncode, time.monotonic()-started
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=15)
        volume.commit()


@app.function(image=image, gpu='H100', cpu=8, memory=49152, timeout=12*3600, max_containers=16, retries=0,
              scaledown_window=2, volumes={'/workspace/runs': volume, '/retained': retained.read_only()})
def evaluate_shard(run_id, protocol_sha, arm, index, count):
    root = RUNS/run_id
    volume.reload()
    check_protocol(root, protocol_sha)
    out = root/arm/'shards'/shard_name(index, count)
    if (out/'report.json').exists():
        return dict(arm=arm, shard=index, resumed=True)
    code, seconds = run_script([str(root), arm, '--shard', f'{index}/{count}'])
    result = dict(arm=arm, shard=index, count=count, exit_code=code, worker_seconds=seconds, finished_unix=time.time())
    (root/arm/'shards'/f'{shard_name(index, count)}-execution.json').write_text(json.dumps(result, indent=2))
    volume.commit()
    if code:
        raise RuntimeError(result)
    return result


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={'/workspace/runs': volume})
def merge(run_id, protocol_sha, arm, count):
    root = RUNS/run_id
    volume.reload()
    check_protocol(root, protocol_sha)
    code, _ = run_script(['merge', str(root), arm, str(count)])
    if code:
        raise RuntimeError(f'merge of {arm} exited {code}')
    return json.loads((root/arm/'report.json').read_text())


@app.function(image=image, cpu=1, memory=4096, timeout=13*3600, retries=0, volumes={'/workspace/runs': volume})
def evaluate_sharded(run_id, protocol_sha, arm, shards):
    """Drop-in for a launcher's per-arm `evaluate`: same <arm>-execution.json and <arm>/report.json."""
    root = RUNS/run_id
    volume.reload()
    check_protocol(root, protocol_sha)
    if (root/arm/'report.json').exists():
        return dict(arm=arm, resumed=True)
    started = time.monotonic()
    calls = [evaluate_shard.spawn(run_id, protocol_sha, arm, i, shards) for i in range(shards)]
    results, failed = [], False
    for c in calls:
        try:
            results.append(c.get())
        except Exception as exc:
            results.append(dict(error=repr(exc)[:500])); failed = True
    report = None if failed else merge.remote(run_id, protocol_sha, arm, shards)
    result = dict(arm=arm, exit_code=int(failed), worker_seconds=time.monotonic()-started, finished_unix=time.time(),
                  shards=results, gpu_seconds=sum(r.get('worker_seconds', 0) for r in results),
                  complete=bool(report and report['complete']))
    volume.reload()
    (root/f'{arm}-execution.json').write_text(json.dumps(result, indent=2)); volume.commit()
    if failed:
        raise RuntimeError(result)
    return result


@app.function(image=image, gpu='H100', cpu=8, memory=49152, timeout=12*3600, max_containers=4, retries=0,
              scaledown_window=2, volumes={'/workspace/runs': volume, '/retained': retained.read_only()})
def evaluate_single(run_id, protocol_sha, arm):
    """Unsharded reference (same as the launchers' `evaluate`), used by the smoke test."""
    root = RUNS/run_id
    volume.reload()
    check_protocol(root, protocol_sha)
    code, seconds = run_script([str(root), arm])
    result = dict(arm=arm, exit_code=code, worker_seconds=seconds, finished_unix=time.time())
    (root/f'{arm}-execution.json').write_text(json.dumps(result, indent=2)); volume.commit()
    if code:
        raise RuntimeError(result)
    return result


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={'/workspace/runs': volume})
def compare_layouts(run_id, sharded, single):
    """Same files, same JSON keys, and the held-out summarizer's trace metrics read both arms."""
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from summarize_expert_sft import trace_metrics
    volume.reload()
    root = RUNS/run_id
    def files(arm):
        base = root/arm
        return sorted(str(p.relative_to(base)) for p in base.rglob('*') if p.is_file() and 'shards' not in p.parts)
    def keys(path):
        return sorted(json.loads(path.read_text()))
    a, b = files(sharded), files(single)
    p = json.loads((root/'protocol.json').read_text())
    games = {arm: {g['id']: g for g in json.loads((root/arm/'games'/'games.json').read_text())} for arm in (sharded, single)}
    metrics = {arm: {j: trace_metrics(root/arm/'games'/(j+'.jsonl.gz'), games[arm][j]['seat']) for j in games[arm]}
               for arm in (sharded, single)}
    timing = {arm: sum(1 for _ in (root/arm/'generation-timing.jsonl').open()) for arm in (sharded, single)}
    streams = {arm: len(list((root/arm/'streams').iterdir())) for arm in (sharded, single)}
    decisions = {arm: sum(g['model_decisions'] for g in games[arm].values()) for arm in (sharded, single)}
    return dict(same_file_set=a == b, only_sharded=sorted(set(a)-set(b)), only_single=sorted(set(b)-set(a)),
                runtime_keys_extra=sorted(set(keys(root/sharded/'runtime.json'))-set(keys(root/single/'runtime.json'))),
                runtime_keys_missing=sorted(set(keys(root/single/'runtime.json'))-set(keys(root/sharded/'runtime.json'))),
                report_keys_extra=sorted(set(keys(root/sharded/'report.json'))-set(keys(root/single/'report.json'))),
                game_row_keys_same=all(sorted(games[sharded][j]) == sorted(games[single][j]) for j in games[single] if j in games[sharded]),
                games={arm: {j: {k: g[k] for k in ('status', 'reason', 'reward', 'round', 'model_decisions')} for j, g in games[arm].items()}
                       for arm in games},
                timing_rows=timing, stream_files=streams, model_decisions=decisions,
                trace_metrics=metrics, jobs=[j['id'] for j in p['jobs']])


@app.local_entrypoint()
def main(action: str = 'smoke', run_id: str = '', arm: str = '', shards: int = 2):
    if action == 'arm':
        local = ROOT/'runs'/run_id
        protocol_sha = sha(local/'protocol.json')
        call = evaluate_sharded.spawn(run_id, protocol_sha, arm, shards)
        receipt = dict(app_id=app.app_id, call_id=call.object_id, arm=arm, shards=shards, protocol_sha256=protocol_sha)
        (local/f'{arm}-shards-dispatch.json').write_text(json.dumps(receipt, indent=2)+'\n'); print(json.dumps(receipt))
        return
    if action == 'merge':
        started = time.monotonic()
        report = merge.remote(run_id, sha(ROOT/'runs'/run_id/'protocol.json'), arm, shards)
        print(json.dumps(dict(arm=arm, games=report['finished_games'], complete=report['complete'],
                              merge_seconds=time.monotonic()-started)))
        return
    if action != 'smoke':
        raise ValueError('action must be smoke, arm or merge')
    # Smoke: 4 held-out pressure games (2 maps x 2 seats), once in 2 shards and once unsharded, same engine.
    from experiments.modal_expert_heldout import build_protocol
    run_id = run_id or f'eval-shards-smoke-{int(time.time())}'
    local = ROOT/'runs'/run_id
    local.mkdir(parents=True)
    p = build_protocol(run_id)
    p['jobs'] = [j for j in p['jobs'] if j['opponent'] == 'pressure'][:4]
    spec = p['arms']['expert_sft@pressure']
    p['arms'] = {'expert_sft@pressure': spec, 'expert_sft@pressure-single': spec}
    p['schema'], p['limitations'] = 'eval-shards-smoke-v1', ['layout smoke test: 4 games']
    p['launcher_sha256'] = sha(Path(__file__))
    (local/'protocol.json').write_text(json.dumps(p, indent=2)+'\n')
    protocol_sha = sha(local/'protocol.json')
    with volume.batch_upload() as up:
        up.put_file(local/'protocol.json', f'{run_id}/protocol.json')
    single = evaluate_single.spawn(run_id, protocol_sha, 'expert_sft@pressure-single')
    sharded = evaluate_sharded.remote(run_id, protocol_sha, 'expert_sft@pressure', shards)
    print('SHARDED', json.dumps(sharded, default=str)[:2000])
    print('SINGLE', json.dumps(single.get(), default=str))
    result = compare_layouts.remote(run_id, 'expert_sft@pressure', 'expert_sft@pressure-single')
    (local/'smoke.json').write_text(json.dumps(dict(sharded=sharded, compare=result), indent=2, default=str)+'\n')
    print('COMPARE', json.dumps({k: v for k, v in result.items() if k != 'trace_metrics'}, indent=1, default=str))
