"""Held-out evaluation of expert-SFT vs base on reserved test-split maps; Modal only.

  modal run --env staging -m experiments.modal_expert_heldout --action smoke
  modal run --detach --env staging -m experiments.modal_expert_heldout --action launch --run-id expert-sft-heldout-v1

30 test-split maps (seeds 100741-100770, first in seed order), both seats, against the
scripted balanced, pressure and economy styles: 180 games per arm. Same harness, sampling,
LoRA-enabled engine and per-decision seed formula as expert-sft-v1.
"""
import hashlib
import json
from pathlib import Path
import time
import modal
from experiments.modal_thinking_repaired import image, volume, ROOT

app = modal.App('polytopia-expert-heldout')
retained = modal.Volume.from_name('polytopia-structured-rl-retained', environment_name='staging')
EXPERT = 'checkpoints/low256-league-rivals-v1/update-96-f998d15e220c/latest.npz'
SOURCE_RUN = 'expert-sft-v1'
OPPONENTS = ['balanced', 'pressure', 'economy']
MAPS = 30
RUNS = Path('/workspace/runs')
_cache = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_protocol(root, protocol_sha):
    if sha(root/'protocol.json') != protocol_sha:
        raise ValueError('protocol changed')
    return json.loads((root/'protocol.json').read_text())


@app.function(image=image, gpu='H100', cpu=8, memory=49152, timeout=12*3600, max_containers=6, retries=0,
              scaledown_window=2, volumes={'/workspace/runs': volume})
def evaluate(run_id, protocol_sha, arm, preflight_only=False):
    import subprocess
    root = RUNS/run_id
    volume.reload()
    check_protocol(root, protocol_sha)
    if (root/arm/'report.json').exists():
        return dict(arm=arm, resumed=True)
    started = time.monotonic()
    cmd = ['python', '/workspace/scripts/benchmark_expert_sft.py', str(root), arm] + (['--preflight-only'] if preflight_only else [])
    proc = subprocess.Popen(cmd, cwd='/workspace')
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


@app.function(image=image, cpu=2, memory=8192, timeout=3600, max_containers=60, retries=1,
              volumes={'/workspace/runs': volume, '/retained': retained.read_only()})
def rl_game(run_id, protocol_sha, job):
    """The RL bot (deterministic) plays one held-out game through the public interface."""
    from polytopia_bot.learning.expert_data import play_expert_game
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    out = root/'rl-reference'
    if (out/(job['id']+'.json')).exists():
        return json.loads((out/(job['id']+'.json')).read_text())
    if 'expert' not in _cache:
        if sha('/retained/'+EXPERT) != p['expert']['sha256']:
            raise ValueError('expert checkpoint changed')
        from polytopia_bot.learning.adapter import GPUEnvironment
        from polytopia_bot.rl.policy import StructuredPolicy
        _cache['expert'] = StructuredPolicy('/retained/'+EXPERT, deterministic=True)
        _cache['env'] = GPUEnvironment(14)
    env = _cache['env']
    class Keep:
        def __getattr__(self, name):
            return getattr(env, name)
        def close(self):
            pass
    result = play_expert_game(job, '/retained/'+EXPERT, out/'games', environment_factory=lambda: Keep(),
                              policy_factory=lambda _: _cache['expert'])
    (out/(job['id']+'.json')).write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={'/workspace/runs': volume})
def summarize(run_id, protocol_sha):
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from summarize_expert_heldout import summarize as run
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
    p = check_protocol(root, protocol_sha)
    status = dict(status='starting', started_unix=time.time())
    def publish(**kw):
        status.update(kw, observed_unix=time.time())
        (root/'collection.json').write_text(json.dumps(status, indent=2)+'\n'); volume.commit()
    try:
        calls = {arm: evaluate.spawn(run_id, protocol_sha, arm) for arm in p['arms']}
        publish(status='running', calls={k: c.object_id for k, c in calls.items()})
        rl = list(rl_game.map([run_id]*len(p['jobs']), [protocol_sha]*len(p['jobs']), p['jobs'], return_exceptions=True))
        rl_failures = [dict(job=j['id'], error=repr(r)[:300]) for j, r in zip(p['jobs'], rl) if isinstance(r, Exception)]
        publish(rl_reference=dict(games=len(rl)-len(rl_failures), failures=rl_failures))
        results = {}
        for arm, c in calls.items():
            try:
                results[arm] = c.get()
            except Exception as exc:
                results[arm] = dict(error=repr(exc)[:500])
            publish(evaluations=results)
        small = summarize.remote(run_id, protocol_sha)
        publish(status='complete', summary=dict(models={m: v['all'] for m, v in small['models'].items()},
                                                 paired={k: v['all'] for k, v in small['paired'].items()}),
                finished_unix=time.time())
    except Exception:
        publish(status='failed', error=traceback.format_exc(), finished_unix=time.time())
        raise
    return status


def build_protocol(run_id):
    source = json.loads((ROOT/'runs'/SOURCE_RUN/'protocol.json').read_text())
    maps = json.loads((ROOT/'runs/sft0/splits-v1/maps.json').read_text())
    if maps['manifest_hash'] != source['data']['split_manifest_hash']:
        raise ValueError('split manifest changed')
    test = sorted(maps['maps']['test'], key=lambda m: m['seed'])[:MAPS]
    used = {j['seed'] for j in source['data']['jobs']} | {j['seed'] for j in source['jobs']}
    if used & {m['seed'] for m in test}:
        raise ValueError('held-out maps overlap earlier data or evaluation maps')
    jobs, index = [], 0
    for opponent in OPPONENTS:
        for m in test:
            for seat in (1, 2):
                jobs.append(dict(id=f'test-{m["seed"]}-seat-{seat}-{opponent}', seed=m['seed'], board_hash=m['board_hash'],
                                 seat=seat, candidate='expert-sft-heldout', strategy='', opponent=opponent, rng_index=index))
                index += 1
    expert_adapter = source['arms']['expert_sft']
    arms = {}
    for opponent in OPPONENTS:
        arms[f'base@{opponent}'] = dict(model='base', opponent=opponent, adapter=None, adapter_files=None)
        arms[f'expert_sft@{opponent}'] = dict(model='expert_sft', opponent=opponent, adapter=expert_adapter['adapter'],
                                              adapter_files=None, training_receipt=expert_adapter['training_receipt'])
    # Hash every source file as it is at launch; the workers re-check these inside Modal.
    hashes = {rel: sha(ROOT/rel) for rel in source['source_hashes']}
    for rel in ['scripts/benchmark_expert_sft.py', 'scripts/summarize_expert_heldout.py', 'scripts/summarize_expert_sft.py',
                'src/polytopia_bot/learning/expert_data.py', 'src/polytopia_bot/learning/teacher.py']:
        hashes[rel] = sha(ROOT/rel)
    keep = ['model_id', 'revision', 'expert', 'lora', 'harness', 'named_fields', 'constrain_actions', 'max_length',
            'max_new_tokens', 'thinking', 'seed', 'sampling', 'concurrency', 'serving', 'gpu']
    return dict({k: source[k] for k in keep}, schema='expert-sft-heldout-v1', run_id=run_id, created_unix=time.time(),
        split='test (first 30 maps in seed order, 100741-100770)', split_manifest_hash=maps['manifest_hash'],
        opponents=OPPONENTS, arms=arms, jobs=jobs, source_hashes=hashes,
        source_protocol=dict(run_id=SOURCE_RUN, sha256=sha(ROOT/'runs'/SOURCE_RUN/'protocol.json')),
        limitations=['one sampling sequence', 'scripted opponents only', 'adapter trained on balanced-opponent games'])


@app.local_entrypoint()
def main(action: str = 'smoke', run_id: str = ''):
    if action == 'summarize':
        local = ROOT/'runs'/run_id
        result = summarize.remote(run_id, sha(local/'protocol.json'))
        (local/'summary-small.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=1)[:5000])
        return
    run_id = run_id or (f'expert-sft-heldout-smoke-{int(time.time())}' if action == 'smoke' else '')
    if not run_id:
        raise ValueError('run_id required')
    local = ROOT/'runs'/run_id
    if local.exists():
        raise ValueError(f'{local} exists; choose a new run id')
    local.mkdir(parents=True)
    protocol = build_protocol(run_id)
    (local/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    protocol_sha = sha(local/'protocol.json')
    with volume.batch_upload() as up:
        up.put_file(local/'protocol.json', f'{run_id}/protocol.json')
    if action == 'smoke':
        # RL bot on one pressure and one economy game, plus an expert-SFT preflight on a pressure arm.
        jobs = [j for j in protocol['jobs'] if j['opponent'] in ('pressure', 'economy') and j['seat'] == 1][:1] + \
               [j for j in protocol['jobs'] if j['opponent'] == 'economy' and j['seat'] == 2][:1]
        rl = [rl_game.remote(run_id, protocol_sha, j) for j in jobs]
        print('SMOKE RL', json.dumps([{k: r[k] for k in ('id', 'opponent', 'status', 'reward', 'round')} for r in rl]))
        ev = evaluate.remote(run_id, protocol_sha, 'expert_sft@pressure', preflight_only=True)
        print('SMOKE EVAL', json.dumps(ev))
        (local/'smoke.json').write_text(json.dumps(dict(rl=rl, eval=ev), indent=2)+'\n')
        return
    if action != 'launch':
        raise ValueError('action must be smoke, launch or summarize')
    call = coordinate.spawn(run_id, protocol_sha)
    receipt = dict(app_id=app.app_id, call_id=call.object_id, protocol_sha256=protocol_sha, launcher_sha256=sha(Path(__file__)))
    (local/'dispatch.json').write_text(json.dumps(receipt, indent=2)+'\n'); print(json.dumps(receipt))
