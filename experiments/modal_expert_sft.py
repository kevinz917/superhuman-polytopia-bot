"""Expert-SFT experiment: RL-expert data -> LoRA SFT -> same-harness evaluation. Modal only.

  modal run --env staging -m experiments.modal_expert_sft --action smoke
  modal run --detach --env staging -m experiments.modal_expert_sft --action launch --run-id expert-sft-v1
  modal run --detach --env staging -m experiments.modal_expert_sft --action launch --run-id <id> --gpus 8 \
      --benchmark-receipt runs/trainer-qualification/<tag>/receipt-<variant>.json [--parity-gate revised]

Training runs on B200:4 (default) or B200:8 (--gpus 8); the effective batch is unchanged. The runtime
variant comes from a qualification receipt (scripts/qualify_expert_trainer.py); --parity-gate picks which
of its gates must pass (strict by default).

Base model Qwen3-4B-Instruct-2507 (pinned). Expert: structured RL bot
low256-league-rivals-v1 update 96 (deterministic). Data maps: training split
100041-100540 (both seats, vs balanced). Evaluation: the 30 qwen35-thinking-30paired-v1
games (15 validation maps x 2 seats, vs balanced) in the readable+grammar harness.
"""
import hashlib
import json
from pathlib import Path
import time
import modal
from experiments.modal_thinking_repaired import image, volume, ROOT

app = modal.App('polytopia-expert-sft')
retained = modal.Volume.from_name('polytopia-structured-rl-retained', environment_name='staging')
wheel = 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
training_image = (modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
    .add_local_file(ROOT/'training/pyproject.toml', '/workspace/training/pyproject.toml', copy=True)
    .add_local_file(ROOT/'training/uv.lock', '/workspace/training/uv.lock', copy=True)
    .run_commands('UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple uv sync --project /workspace/training --locked --no-dev --extra simulation')
    .run_commands(f'uv pip install --python /workspace/training/.venv/bin/python --no-deps {wheel}',
                  'uv pip install --python /workspace/training/.venv/bin/python einops==0.8.1')
    .env({'PYTHONPATH': '/workspace/src', 'TOKENIZERS_PARALLELISM': 'false', 'HF_HOME': '/workspace/runs/hf-cache', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT/'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'scripts', '/workspace/scripts', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'configs', '/workspace/configs'))
EXPERT = 'checkpoints/low256-league-rivals-v1/update-96-f998d15e220c/latest.npz'
EXPERT_SHA = 'f998d15e220c92fe5ad091b73c4fd8952612a69b37fdd607d8b007093b26225d'
OLD_SFT = 'movement-epoch2-v1/training/model/adapter-final'
INSTRUCT = 'qwen3-instruct-2507-30paired-v2'
TRAIN_BENCHMARK = 'movement-scale40k-v1'
TRAIN_GPUS = (4, 8)
RUNS = Path('/workspace/runs')
_cache = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_protocol(root, protocol_sha):
    if sha(root/'protocol.json') != protocol_sha:
        raise ValueError('protocol changed')
    return json.loads((root/'protocol.json').read_text())


@app.function(image=image, cpu=2, memory=8192, timeout=3600, max_containers=100, retries=1,
              volumes={'/workspace/runs': volume, '/retained': retained.read_only()})
def generate_game(run_id, protocol_sha, job):
    """Play one expert game on a training map; the game file lands in expert-games/."""
    import sys
    from polytopia_bot.learning.expert_data import play_expert_game
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    out = root/'expert-games'
    if (out/(job['id']+'.jsonl.gz')).exists():
        from polytopia_bot.learning.expert_data import read_game
        return read_game(out/(job['id']+'.jsonl.gz'))[2]
    if 'expert' not in _cache:
        if sha('/retained/'+EXPERT) != p['expert']['sha256']:
            raise ValueError('expert checkpoint changed')
        from polytopia_bot.learning.adapter import GPUEnvironment
        from polytopia_bot.rl.policy import StructuredPolicy
        _cache['expert'] = StructuredPolicy('/retained/'+EXPERT, deterministic=True)
        _cache['env'] = GPUEnvironment(14)
    env = _cache['env']
    # One environment per container, reset per game; close() is skipped to keep the compiled runtime.
    class Keep:
        def __getattr__(self, name):
            return getattr(env, name)
        def close(self):
            pass
    result = play_expert_game(job, '/retained/'+EXPERT, out, environment_factory=lambda: Keep(),
                              policy_factory=lambda _: _cache['expert'])
    volume.commit()
    return result


@app.function(image=image, cpu=4, memory=16384, timeout=3600, max_containers=40, retries=1,
              volumes={'/workspace/runs': volume})
def prepare_shard(run_id, protocol_sha, index, files):
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from prepare_expert_sft import shard
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    volume.reload()
    (root/'prepared-shards').mkdir(exist_ok=True)
    out = root/'prepared-shards'/f'shard-{index:03d}.jsonl'
    result = shard([root/'expert-games'/f for f in files], p['training']['config'],
                   p['data']['examples_per_game'], p['data']['sample_seed'], out)
    volume.commit()
    return result


@app.function(image=image, cpu=4, memory=32768, timeout=7200, retries=0, volumes={'/workspace/runs': volume})
def combine_shards(run_id, protocol_sha, shards):
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from prepare_expert_sft import combine
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    volume.reload()
    manifest = combine(shards, root/'prepared', p['training']['config'], expert=p['expert'], data_protocol=p['data'])
    volume.commit()
    return manifest


# 7.9 h for 1.49B tokens on 4 x B200 with the strict-gate runtime (docs/sft-trainer-benchmark-2026-10-19.md).
@app.function(image=training_image, gpu='B200:4', cpu=16, memory=131072, timeout=12*3600, max_containers=1,
              retries=0, scaledown_window=2, volumes={'/workspace/runs': volume})
def train(run_id, protocol_sha):
    import subprocess
    root = RUNS/run_id
    volume.reload()
    p = check_protocol(root, protocol_sha)
    bench = p['training']['benchmark']
    gpus = p['training'].get('gpus', 4)
    found = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True,
                           text=True, check=True).stdout.split('\n')
    if len([n for n in found if n.strip()]) != gpus:
        raise RuntimeError(f'protocol expects {gpus} GPUs, container has {found}')
    payload = dict(prepared=str(root/'prepared'), output=str(root/'training'/'model'), config=p['training']['config'],
                   seed=p['training']['seed'], benchmark=bench, checkpoint_every=p['training']['checkpoint_every'],
                   parity_gate=p['training'].get('parity_gate', 'strict'))
    (root/'training').mkdir(exist_ok=False)
    (root/'training'/'input.json').write_text(json.dumps(payload, indent=2)); volume.commit()
    started = time.monotonic()
    proc = subprocess.Popen(['/workspace/training/.venv/bin/python', '-m', 'torch.distributed.run', '--standalone',
                             f'--nproc_per_node={gpus}', '/workspace/scripts/expert_train_distributed.py',
                             str(root/'training'/'input.json')], cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit(); time.sleep(15)
        result = dict(exit_code=proc.returncode, occupied_seconds=time.monotonic()-started, finished_unix=time.time(), gpus=gpus)
        if (root/'training'/'model'/'run.json').exists():
            run = json.loads((root/'training'/'model'/'run.json').read_text())
            result.update(status=run['status'], train_seconds=run.get('train_seconds'), adapter_files=run.get('adapter_files'))
        (root/'training'/'execution.json').write_text(json.dumps(result, indent=2)); volume.commit()
        if proc.returncode:
            raise RuntimeError(result)
        return result
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=30)
        volume.commit()


def train_function(protocol):
    """train sized to the protocol's GPU count (4 x B200 is the decorator default)."""
    gpus = protocol['training'].get('gpus', 4)
    if gpus not in TRAIN_GPUS:
        raise ValueError(f'training.gpus must be one of {TRAIN_GPUS}')
    return train if gpus == 4 else train.with_options(gpu=f'B200:{gpus}', cpu=4*gpus, memory=32768*gpus)


@app.function(image=image, gpu='H100', cpu=8, memory=49152, timeout=6*3600, max_containers=3, retries=0,
              scaledown_window=2, volumes={'/workspace/runs': volume})
def evaluate(run_id, protocol_sha, arm, preflight_only=False):
    import subprocess
    root = RUNS/run_id
    volume.reload()
    check_protocol(root, protocol_sha)
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


@app.function(image=image, cpu=4, memory=16384, timeout=6*3600, retries=0,
              volumes={'/workspace/runs': volume, '/retained': retained.read_only()})
def rl_reference(run_id, protocol_sha):
    """The RL expert itself on the same 30 evaluation games (deterministic, vs balanced)."""
    import os, sys
    os.environ['POLYTOPIA_CLOUD_EVALUATION'] = '1'
    sys.path.insert(0, '/workspace/scripts')
    from evaluate_structured_bot import run
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    if sha('/retained/'+EXPERT) != p['expert']['sha256']:
        raise ValueError('expert checkpoint changed')
    seeds = sorted({j['seed'] for j in p['jobs']})
    if seeds != list(range(seeds[0], seeds[0]+len(seeds))):
        raise ValueError('evaluation seeds must be consecutive for the RL evaluator')
    started = time.monotonic()
    report = run('/retained/'+EXPERT, str(root/'rl-reference'), maps=len(seeds), seed_base=seeds[0],
                 opponent='balanced', deterministic=True, seconds_limit=6*3600-600)
    (root/'rl_reference-execution.json').write_text(json.dumps(dict(report=report, worker_seconds=time.monotonic()-started), indent=2))
    volume.commit()
    return report


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={'/workspace/runs': volume})
def summarize(run_id, protocol_sha):
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from summarize_expert_sft import summarize as run
    root = RUNS/run_id
    volume.reload()
    check_protocol(root, protocol_sha)
    small = run(root)
    volume.commit()
    return small


@app.function(image=image, cpu=1, memory=4096, timeout=24*3600, max_containers=1, retries=0,
              volumes={'/workspace/runs': volume})
def coordinate(run_id, protocol_sha):
    import traceback
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    status = dict(status='starting', started_unix=time.time(), stages={})
    def publish(**kw):
        status.update(kw, observed_unix=time.time())
        (root/'collection.json').write_text(json.dumps(status, indent=2)+'\n'); volume.commit()
    try:
        # Independent arms start immediately; finished arms are not rerun on resume.
        early = {arm: evaluate.spawn(run_id, protocol_sha, arm) for arm in ('base', 'old_sft')
                 if not (root/arm/'report.json').exists()}
        if not (root/'rl-reference'/'summary.json').exists():
            early['rl_reference'] = rl_reference.spawn(run_id, protocol_sha)
        publish(status='generating', calls={k: c.object_id for k, c in early.items()})
        started = time.monotonic()
        results = list(generate_game.map([run_id]*len(p['data']['jobs']), [protocol_sha]*len(p['data']['jobs']),
                                         p['data']['jobs'], return_exceptions=True))
        failures = [dict(job=j['id'], error=repr(r)[:400]) for j, r in zip(p['data']['jobs'], results) if isinstance(r, Exception)]
        done = [r for r in results if not isinstance(r, Exception)]
        status['stages']['generate'] = dict(games=len(done), failures=failures, seconds=time.monotonic()-started,
            expert_wins=sum(r['reward'] > 0 for r in done), expert_losses=sum(r['reward'] < 0 for r in done),
            truncated=sum(r['status'] == 'truncated' for r in done), expert_decisions=sum(r['expert_decisions'] for r in done),
            median_round=sorted(r['round'] for r in done)[len(done)//2] if done else None)
        publish(status='preparing')
        if len(failures) > p['data']['max_failed_games']:
            raise RuntimeError(f'{len(failures)} expert games failed')
        volume.reload()
        files = sorted(f'{r["id"]}.jsonl.gz' for r in done)
        shards = [files[i::p['data']['prepare_shards']] for i in range(p['data']['prepare_shards'])]
        started = time.monotonic()
        if (root/'prepared'/'manifest.json').exists():
            manifest = json.loads((root/'prepared'/'manifest.json').read_text())
        else:
            shard_results = list(prepare_shard.starmap([(run_id, protocol_sha, i, s) for i, s in enumerate(shards)]))
            manifest = combine_shards.remote(run_id, protocol_sha, shard_results)
        status['stages']['prepare'] = dict(seconds=time.monotonic()-started, **{k: manifest[k] for k in
            ('records', 'duplicates_dropped', 'conflicting_duplicate_labels', 'max_tokens', 'mean_tokens', 'total_tokens', 'families')})
        publish(status='training')
        started = time.monotonic()
        run_file = root/'training'/'model'/'run.json'
        if run_file.exists() and json.loads(run_file.read_text())['status'] == 'completed':
            status['stages']['train'] = dict(resumed=True, **json.loads((root/'training'/'execution.json').read_text()))
        else:
            if (root/'training').exists():
                raise RuntimeError('incomplete training directory exists; inspect before retrying')
            status['stages']['train'] = dict(**train_function(p).remote(run_id, protocol_sha), seconds=time.monotonic()-started)
        publish(status='evaluating_expert')
        expert = None if (root/'expert_sft'/'report.json').exists() else evaluate.spawn(run_id, protocol_sha, 'expert_sft')
        if expert:
            status['calls']['expert_sft'] = expert.object_id; publish()
        evaluations = {k: c.get() for k, c in early.items()}
        if expert:
            evaluations['expert_sft'] = expert.get()
        status['stages']['evaluate'] = evaluations
        publish(status='summarizing')
        small = summarize.remote(run_id, protocol_sha)
        publish(status='complete', summary=small, finished_unix=time.time())
    except Exception:
        publish(status='failed', error=traceback.format_exc(), finished_unix=time.time())
        raise
    return status


def build_protocol(run_id, gpus=4, benchmark_receipt='', parity_gate='strict'):
    import sys
    sys.path.insert(0, str(ROOT/'src'))
    from polytopia_bot.learning.expert_trainer import check_benchmark
    if gpus not in TRAIN_GPUS:
        raise ValueError(f'gpus must be one of {TRAIN_GPUS}')
    instruct = json.loads((ROOT/'runs'/INSTRUCT/'protocol.json').read_text())
    maps = json.loads((ROOT/'runs/sft0/splits-v1/maps.json').read_text())
    train_maps = maps['maps']['train']
    eval_seeds = {j['seed'] for j in instruct['jobs']}
    if eval_seeds & {m['seed'] for m in train_maps}:
        raise ValueError('evaluation maps overlap training maps')
    data_jobs = [dict(id=f'expert-{m["seed"]}-seat-{seat}', seed=m['seed'], board_hash=m['board_hash'], seat=seat,
                      opponent='balanced') for m in train_maps for seat in (1, 2)]
    jobs = [dict(j, candidate='expert-sft-comparison') for j in instruct['jobs']]
    source = json.loads((ROOT/'runs'/TRAIN_BENCHMARK/'protocol.json').read_text())['training']
    # 24,576-token cap: the qualified no-checkpointing variant peaked at 145 GB on a 23,664-token
    # stress batch (expert-sft-smoke-1792384845); longer examples are skipped and counted.
    config = dict(source['config'], epochs=1, max_length=24576)
    benchmark, benchmark_source = source['benchmark'], f'{TRAIN_BENCHMARK}/protocol.json'
    if benchmark_receipt:
        benchmark = json.loads((ROOT/benchmark_receipt).read_text())
        benchmark_source = dict(path=benchmark_receipt, sha256=sha(ROOT/benchmark_receipt))
        if benchmark['config'] != config:
            raise ValueError('qualification receipt was made for a different training config')
    check_benchmark(benchmark, parity_gate)
    old_files = {f.name: sha(f) for f in sorted((ROOT/'runs/movement-epoch2-v1/adapter-final').glob('adapter_*'))}
    if set(old_files) != {'adapter_config.json', 'adapter_model.safetensors'}:
        raise ValueError('old SFT adapter files missing locally')
    used = ['src/polytopia_bot/learning/expert_data.py', 'src/polytopia_bot/learning/expert_trainer.py',
            'src/polytopia_bot/learning/fast_training.py',
            'scripts/prepare_expert_sft.py', 'scripts/expert_train_distributed.py', 'scripts/benchmark_expert_sft.py',
            'scripts/summarize_expert_sft.py', 'scripts/evaluate_structured_bot.py', 'scripts/benchmark_qwen3_instruct.py',
            'src/polytopia_bot/rl/policy.py', 'src/polytopia_bot/rl/network.py', 'src/polytopia_bot/rl/observation.py',
            'src/polytopia_bot/rl/runner.py', 'src/polytopia_bot/rl/actions.py', 'src/polytopia_bot/rl/ppo.py',
            'src/polytopia_bot/rl/data.py']
    hashes = dict(instruct['source_hashes'])
    for rel in list(hashes) + used:
        hashes[rel] = sha(ROOT/rel)
    protocol = dict(schema='expert-sft-comparison-v1', run_id=run_id, created_unix=time.time(),
        model_id=instruct['model_id'], revision=instruct['revision'],
        expert=dict(checkpoint=EXPERT, sha256=EXPERT_SHA, run='low256-league-rivals-v1 update 96', deterministic=True,
                    acceptance='64/64 vs scripted teacher, 64/64 vs random on final partition 400000-400031'),
        data=dict(jobs=data_jobs, split='train (seeds 100041-100540)', split_manifest_hash=maps['manifest_hash'],
                  opponent='balanced', examples_per_game=40, sample_seed=17, prepare_shards=40, max_failed_games=50,
                  label='prompt-local aN of the expert legal action, parse round-trip verified',
                  prompt='render_readable_decision (polytopia-named-fields-v1)'),
        training=dict(config=config, seed=17, checkpoint_every=157, benchmark=benchmark, benchmark_source=benchmark_source,
                      gpus=gpus, parity_gate=parity_gate,
                      recipe='LoRA r32/a64 all proj, lr 1e-4 cosine, 3% warmup, effective batch 64, 1 epoch, response-only loss'),
        arms=dict(base=dict(adapter=None, adapter_files=None),
                  old_sft=dict(adapter=OLD_SFT, adapter_files=old_files),
                  expert_sft=dict(adapter=f'{run_id}/training/model/adapter-final', adapter_files=None,
                                  training_receipt=f'{run_id}/training/model/run.json')),
        lora=dict(max_rank=32),
        harness=instruct['harness'], named_fields=True, constrain_actions=True, max_length=instruct['max_length'],
        max_new_tokens=None, thinking=False, seed=instruct['seed'], sampling=instruct['sampling'],
        concurrency=instruct['concurrency'], serving=instruct['serving'], gpu='H100', jobs=jobs,
        source_hashes=hashes, source_protocol=dict(run_id=INSTRUCT, sha256=sha(ROOT/'runs'/INSTRUCT/'protocol.json')),
        limitations=['15 validation maps x 2 seats; one sampling sequence', 'old SFT was trained on a different prompt',
                     'RL reference uses its own structured policy, not the LLM harness'])
    return protocol


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0,
              volumes={'/workspace/runs': volume, '/retained': retained.read_only()})
def smoke_data(run_id, protocol_sha):
    """Two expert games + one prepare shard in a scratch run; verifies labels and token lengths."""
    import sys
    sys.path.insert(0, '/workspace/scripts')
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    results = [generate_game.local(run_id, protocol_sha, j) for j in p['data']['jobs'][:2]]
    from prepare_expert_sft import shard
    from polytopia_bot.learning.expert_data import read_game
    sh = shard([root/'expert-games'/(r['id']+'.jsonl.gz') for r in results], p['training']['config'], 40, 17,
               root/'smoke-shard.jsonl')
    lengths, families = [], {}
    with open(root/'smoke-shard.jsonl') as f:
        for line in f:
            r = json.loads(line); lengths.append(len(r['input_ids'])); families[r['family']] = families.get(r['family'], 0)+1
    header, examples, _ = read_game(root/'expert-games'/(results[0]['id']+'.jsonl.gz'))
    volume.commit()
    return dict(results=results, shard=sh, tokens=dict(n=len(lengths), max=max(lengths), mean=sum(lengths)/len(lengths)),
                families=families, sample_label=examples[0]['label'], sample_user_chars=len(examples[0]['user']))


@app.function(image=image, cpu=2, memory=8192, timeout=7200, retries=0, volumes={'/workspace/runs': volume})
def smoke_train(run_id, protocol_sha):
    """Combine the smoke shard and train on it with the real 4-GPU trainer (a few updates)."""
    import sys
    sys.path.insert(0, '/workspace/scripts')
    from prepare_expert_sft import combine
    from polytopia_bot.learning.interface import digest
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    volume.reload()
    from polytopia_bot.learning.training import load_tokenizer
    tok = load_tokenizer(p['training']['config'])
    shard = dict(stats={}, tokenizer_hash=digest(tok.backend_tokenizer.to_str()), chat_template_hash=digest(tok.chat_template),
                 path=str(root/'smoke-shard.jsonl'))
    manifest = combine([shard], root/'prepared', p['training']['config'], expert=p['expert'], data_protocol=dict(smoke=True))
    volume.commit()
    result = train_function(p).remote(run_id, protocol_sha)
    volume.reload()
    run = json.loads((root/'training'/'model'/'run.json').read_text())
    return dict(manifest={k: manifest[k] for k in ('records', 'max_tokens', 'mean_tokens', 'total_tokens')}, train=result,
                stress=run.get('corpus_stress'), steps=run.get('global_step'), losses=[l['loss'] for l in run['logs']])


@app.local_entrypoint()
def main(action: str = 'smoke', run_id: str = '', gpus: int = 4, benchmark_receipt: str = '', parity_gate: str = 'strict'):
    if action == 'summarize':
        local = ROOT/'runs'/run_id
        small = summarize.remote(run_id, sha(local/'protocol.json'))
        (local/'summary-small.json').write_text(json.dumps(small, indent=2)+'\n')
        print(json.dumps(small, indent=1)[:4000])
        return
    if action == 'smoke-train':
        local = ROOT/'runs'/run_id
        protocol_sha = sha(local/'protocol.json')
        result = smoke_train.remote(run_id, protocol_sha)
        print('SMOKE TRAIN', json.dumps(result, indent=1))
        (local/'smoke-train.json').write_text(json.dumps(result, indent=2)+'\n')
        return
    if action == 'smoke':
        run_id = run_id or f'expert-sft-smoke-{int(time.time())}'
    if not run_id:
        raise ValueError('run_id required')
    local = ROOT/'runs'/run_id
    if local.exists():
        raise ValueError(f'{local} exists; choose a new run id')
    local.mkdir(parents=True)
    protocol = build_protocol(run_id, gpus, benchmark_receipt, parity_gate)
    (local/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    protocol_sha = sha(local/'protocol.json')
    with volume.batch_upload() as up:
        up.put_file(local/'protocol.json', f'{run_id}/protocol.json')
    if action == 'smoke':
        data = smoke_data.remote(run_id, protocol_sha)
        print('SMOKE DATA', json.dumps(data, indent=1))
        ev = evaluate.remote(run_id, protocol_sha, 'old_sft', preflight_only=True)
        print('SMOKE EVAL', json.dumps(ev))
        (local/'smoke.json').write_text(json.dumps(dict(data=data, eval=ev), indent=2)+'\n')
        return
    if action != 'launch':
        raise ValueError('action must be smoke, smoke-train or launch')
    call = coordinate.spawn(run_id, protocol_sha)
    receipt = dict(app_id=app.app_id, call_id=call.object_id, protocol_sha256=protocol_sha,
                   launcher_sha256=sha(Path(__file__)))
    (local/'dispatch.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
