"""Head-to-head: expert-SFT LLM vs the best RL bot (dist-c460div-v1 update 3142); Modal only.

  modal run --detach --env staging -m experiments.modal_expert_h2h --action launch --run-id expert-sft-h2h-v1

5 fresh test-split maps (100771-100775) x both seats = 10 games. Reference: league96
(update 96, the bot expert-SFT imitated) vs the same RL bot on the same 10 games.
"""
import hashlib
import json
from pathlib import Path
import time
import modal
from experiments.modal_thinking_repaired import image, volume, ROOT

app = modal.App('polytopia-expert-h2h')
retained = modal.Volume.from_name('polytopia-structured-rl-retained', environment_name='staging')
BEST = dict(name='rl_u3142', path='/retained/checkpoints/distributed-final-v1/dist-c460div-v1-update-3142/final.npz',
            sha256='04cbb4745a4e957dff88de36218c40fd14f4d40cf3bdf0b80021ee2697f0bfc8',
            run='dist-c460div-v1 update 3142 (delivered 2026-10-19; 117-10-1 vs league96 on final maps)')
LEAGUE96 = dict(path='/retained/checkpoints/low256-league-rivals-v1/update-96-f998d15e220c/latest.npz',
                sha256='f998d15e220c92fe5ad091b73c4fd8952612a69b37fdd607d8b007093b26225d')
SOURCE_RUN = 'expert-sft-v1'
RUNS = Path('/workspace/runs')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_protocol(root, protocol_sha):
    if sha(root/'protocol.json') != protocol_sha:
        raise ValueError('protocol changed')
    return json.loads((root/'protocol.json').read_text())


@app.function(image=image, gpu='H100', cpu=8, memory=49152, timeout=8*3600, max_containers=1, retries=0,
              scaledown_window=2, volumes={'/workspace/runs': volume, '/retained': retained.read_only()})
def evaluate(run_id, protocol_sha, arm):
    import subprocess
    root = RUNS/run_id
    volume.reload()
    check_protocol(root, protocol_sha)
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


@app.function(image=image, cpu=2, memory=8192, timeout=3600, max_containers=10, retries=1,
              volumes={'/workspace/runs': volume, '/retained': retained.read_only()})
def rl_vs_rl(run_id, protocol_sha, job):
    """league96 (in the job's seat) vs the best RL bot (other seat), both deterministic."""
    from polytopia_bot.learning.adapter import GPUEnvironment
    from polytopia_bot.rl.policy import StructuredPolicy
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    for spec in (p['reference']['league96'], p['rl_opponents'][BEST['name']]):
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('RL checkpoint changed')
    me = StructuredPolicy(p['reference']['league96']['path'], deterministic=True)
    best = StructuredPolicy(p['rl_opponents'][BEST['name']]['path'], deterministic=True)
    env = GPUEnvironment(14)
    started = time.monotonic(); decisions = 0
    try:
        obs = env.reset(seed=job['seed'], settings={})
        while True:
            policy = me if obs.player == job['seat'] else best
            transition = env.step(action_id=policy.select(obs).action.action_id, expected_state_version=obs.state_version)
            decisions += 1
            if transition.terminated or transition.truncated:
                result = dict(id=job['id'], seed=job['seed'], seat=job['seat'], decisions=decisions,
                              status='native' if transition.terminated else 'truncated', reason=transition.reason,
                              reward=transition.rewards[job['seat']], round=transition.actor_observation.visible_state['turn'],
                              seconds=time.monotonic()-started)
                break
            obs = env.observe(transition.next_player)
    finally:
        env.close()
    out = root/'league96-reference'; out.mkdir(exist_ok=True)
    (out/(job['id']+'.json')).write_text(json.dumps(result, indent=2)); volume.commit()
    return result


@app.function(image=image, cpu=1, memory=4096, timeout=10*3600, max_containers=1, retries=0,
              volumes={'/workspace/runs': volume})
def coordinate(run_id, protocol_sha):
    import sys, traceback
    sys.path.insert(0, '/workspace/scripts')
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    status = dict(status='starting', started_unix=time.time())
    def publish(**kw):
        status.update(kw, observed_unix=time.time())
        (root/'collection.json').write_text(json.dumps(status, indent=2)+'\n'); volume.commit()
    if (root/'calls.json').exists():
        # A platform restart re-runs this input: never spawn a second evaluation.
        calls = json.loads((root/'calls.json').read_text())
        llm = modal.FunctionCall.from_id(calls['expert_sft'])
    else:
        llm = evaluate.spawn(run_id, protocol_sha, 'expert_sft')
        (root/'calls.json').write_text(json.dumps(dict(expert_sft=llm.object_id), indent=2)); volume.commit()
    try:
        publish(status='running', calls=dict(expert_sft=llm.object_id))
        ref = list(rl_vs_rl.map([run_id]*len(p['jobs']), [protocol_sha]*len(p['jobs']), p['jobs'], return_exceptions=True))
        publish(league96_reference=[r if isinstance(r, dict) else dict(error=repr(r)[:300]) for r in ref])
        status['expert_sft'] = llm.get(); volume.reload()
        from summarize_expert_sft import trace_metrics, win
        games = {g['id']: g for g in json.loads((root/'expert_sft'/'games'/'games.json').read_text())}
        rows = []
        for j in p['jobs']:
            g = games.get(j['id'])
            m = trace_metrics(root/'expert_sft'/'games'/(j['id']+'.jsonl.gz'), j['seat'])
            r = next((x for x in ref if isinstance(x, dict) and x['id'] == j['id']), None)
            rows.append(dict(id=j['id'], seed=j['seed'], seat=j['seat'], expert_sft=dict(status=g and g['status'],
                reason=g and g['reason'], reward=g and g['reward'], round=g and g['round'], model_moves=g and g['model_decisions'],
                peak_cities=m['peak_cities'], combat=m['combat']),
                league96=dict(status=r and r['status'], reward=r and r['reward'], round=r and r['round'])))
        summary = dict(schema='expert-sft-h2h-summary-v1', opponent=p['rl_opponents'][BEST['name']]['run'],
            expert_sft=dict(games=len(rows), wins=sum(win(dict(status=x['expert_sft']['status'], reward=x['expert_sft']['reward'])) for x in rows),
                            losses=sum(x['expert_sft']['status'] == 'native' and (x['expert_sft']['reward'] or 0) < 0 for x in rows),
                            unresolved=sum(x['expert_sft']['status'] != 'native' for x in rows)),
            league96=dict(games=len(rows), wins=sum((x['league96']['reward'] or 0) > 0 and x['league96']['status'] == 'native' for x in rows),
                          losses=sum((x['league96']['reward'] or 0) < 0 for x in rows),
                          unresolved=sum(x['league96']['status'] != 'native' for x in rows)),
            games=rows)
        (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        publish(status='complete', summary={k: summary[k] for k in ('expert_sft', 'league96')}, finished_unix=time.time())
    except Exception:
        publish(status='failed', error=traceback.format_exc(), finished_unix=time.time())
        raise
    return status


def build_protocol(run_id):
    source = json.loads((ROOT/'runs'/SOURCE_RUN/'protocol.json').read_text())
    maps = json.loads((ROOT/'runs/sft0/splits-v1/maps.json').read_text())
    test = sorted(maps['maps']['test'], key=lambda m: m['seed'])[30:35]
    held = json.loads((ROOT/'runs/expert-sft-heldout-v1/protocol.json').read_text())
    used = {j['seed'] for j in source['data']['jobs']} | {j['seed'] for j in source['jobs']} | {j['seed'] for j in held['jobs']}
    if used & {m['seed'] for m in test}:
        raise ValueError('head-to-head maps overlap earlier maps')
    jobs = [dict(id=f'h2h-{m["seed"]}-seat-{seat}', seed=m['seed'], board_hash=m['board_hash'], seat=seat,
                 candidate='expert-sft-h2h', strategy='', opponent=BEST['name'], rng_index=i)
            for i, (m, seat) in enumerate((m, s) for m in test for s in (1, 2))]
    expert = source['arms']['expert_sft']
    hashes = {rel: sha(ROOT/rel) for rel in source['source_hashes']}
    for rel in ['scripts/benchmark_expert_sft.py', 'scripts/summarize_expert_sft.py', 'src/polytopia_bot/rl/policy.py']:
        hashes[rel] = sha(ROOT/rel)
    keep = ['model_id', 'revision', 'lora', 'harness', 'named_fields', 'constrain_actions', 'max_length',
            'max_new_tokens', 'thinking', 'seed', 'sampling', 'concurrency', 'serving', 'gpu']
    return dict({k: source[k] for k in keep}, schema='expert-sft-h2h-v1', run_id=run_id, created_unix=time.time(),
        split='test (maps 31-35 in seed order, 100771-100775)', split_manifest_hash=maps['manifest_hash'],
        arms=dict(expert_sft=dict(model='expert_sft', opponent=BEST['name'], adapter=expert['adapter'], adapter_files=None,
                                  training_receipt=expert['training_receipt'])),
        rl_opponents={BEST['name']: dict(path=BEST['path'], sha256=BEST['sha256'], run=BEST['run'], deterministic=True)},
        reference=dict(league96=dict(LEAGUE96, run='low256-league-rivals-v1 update 96 (the imitated expert)')),
        jobs=jobs, source_hashes=hashes, source_protocol=dict(run_id=SOURCE_RUN, sha256=sha(ROOT/'runs'/SOURCE_RUN/'protocol.json')),
        limitations=['10 games; one sampling sequence for the LLM; RL opponent deterministic'])


@app.local_entrypoint()
def main(action: str = 'launch', run_id: str = ''):
    if action != 'launch' or not run_id:
        raise ValueError('use --action launch --run-id RUN')
    local = ROOT/'runs'/run_id
    if local.exists():
        raise ValueError(f'{local} exists')
    local.mkdir(parents=True)
    (local/'protocol.json').write_text(json.dumps(build_protocol(run_id), indent=2)+'\n')
    protocol_sha = sha(local/'protocol.json')
    with volume.batch_upload() as up:
        up.put_file(local/'protocol.json', f'{run_id}/protocol.json')
    call = coordinate.spawn(run_id, protocol_sha)
    receipt = dict(app_id=app.app_id, call_id=call.object_id, protocol_sha256=protocol_sha, launcher_sha256=sha(Path(__file__)))
    (local/'dispatch.json').write_text(json.dumps(receipt, indent=2)+'\n'); print(json.dumps(receipt))
