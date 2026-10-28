"""Duper-style distributed PPO on Modal: CPU actor fleet + multi-GPU learner.

    modal run --env staging experiments/modal_distributed_rl.py::tests
    modal run --detach --env staging experiments/modal_distributed_rl.py::launch --run-id X --parent /retained/... ...
    modal run --env staging experiments/modal_distributed_rl.py::arena --candidate ... --opponent ... --seed-base 302000

Actors reach the learner through a Modal TCP tunnel; the learner publishes
its address in a modal.Dict keyed by run id. Actors exit when the learner stops
or disconnects, so no actor outlives its learner. Evaluation (arena) always uses
the frozen engine on CPU.
"""
from pathlib import Path
import json
import threading
import os
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
# Actor image: the JAX CPU engine plus the Rust actor engine (engine-rs, the default: LearnerConfig.engine='auto'),
# built from source with maturin into an abi3 wheel at image-build time.
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('curl', 'build-essential', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0', 'pytest>=9,<10', 'maturin>=1.7,<2')
    .add_local_dir(ROOT / 'engine-rs', '/workspace/engine-rs', copy=True, ignore=['**/target/**'])
    .run_commands('cd /workspace/engine-rs && PATH=/root/.cargo/bin:$PATH maturin build --release -o /workspace/wheels '
                  '&& pip install /workspace/wheels/*.whl')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'tests', '/workspace/tests', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'scripts', '/workspace/scripts', ignore=['**/__pycache__/**']))   # script tests load these
gpu_image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda12]==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cuda', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-distributed-rl')
results = modal.Volume.from_name('polytopia-structured-rl', create_if_missing=False)
retained = modal.Volume.from_name('polytopia-structured-rl-retained', create_if_missing=False)
registry = modal.Dict.from_name('polytopia-distributed-rl-registry', create_if_missing=True)
VOLUMES = {'/results': results, '/retained': retained}
PORT = 7070
# Transport: a Modal TCP tunnel on the learner. i6pn only links containers on one cloud provider
# (probe 2026-10-19: Azure/AWS -> GCP timed out) and this workspace cannot pin a provider; the tunnel
# measured 80-110 MB/s per connection with four concurrent connections, from any cloud.
GPU_RATE = {'H100': 3.95, 'H200': 4.54, 'B200': 6.25, 'L40S': 1.95, 'L4': 0.80, 'A10G': 1.10}
CPU_CORE_RATE = 0.0472
MEM_GIB_RATE = 0.008


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=5400)
def run_tests(paths: str):
    import subprocess
    out = subprocess.run(['python', '-m', 'pytest', '-x', '-q', '--durations=5', *paths.split(',')], cwd='/workspace',
                         capture_output=True, text=True)
    return out.returncode, out.stdout[-6000:] + out.stderr[-3000:]


@app.function(image=gpu_image, gpu='H100', cpu=16, memory=65536, timeout=43200, volumes=VOLUMES,
              retries=0, scaledown_window=2)
def learner(run_id: str, parent: str, config: dict, env: dict = None):
    import traceback
    for k, v in (env or {}).items():   # e.g. POLYTOPIA_MATMUL_PRECISION; read when rl.network is first imported
        os.environ[k] = str(v)
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    from polytopia_bot.rl.runner import write_json
    import jax
    jax.config.update('jax_compilation_cache_dir', '/results/.jax-compilation-cache')
    root = Path('/results/distributed') / run_id
    started = time.time()
    mine = None   # this container's tunnel, once registered
    try:
        cfg = LearnerConfig(**{**config, 'frozen': tuple(tuple(x) for x in config.get('frozen', ()))})
        def commit():
            results.commit()
        def status(row):
            registry[f'{run_id}/status'] = row
        learner_ = Learner(root, cfg, parent, port=PORT, commit=commit, log=lambda s: print(s, flush=True),
                           status=status)
        with modal.forward(PORT, unencrypted=True) as tunnel:
            host, port = tunnel.tcp_socket
            mine = (host, port)
            registry[run_id] = dict(host=host, port=port, started=started, devices=learner_.devices)
            print(json.dumps(dict(stage='registered', host=host, port=port, devices=learner_.devices)), flush=True)
            history = learner_.run(bind_host='0.0.0.0')
        return dict(updates=len(history), seconds=time.time() - started)
    except Exception:
        root.parent.mkdir(parents=True, exist_ok=True)
        write_json(root.parent / f'{run_id}-error.json', dict(traceback=traceback.format_exc()))
        results.commit()
        raise
    finally:
        try:
            # Mark the run stopped only if the entry is still this learner's (or none registered): a second learner
            # on the same run id must not stop the actors of the one that owns the entry.
            entry = registry.get(run_id, {})
            if mine is None and 'host' not in entry or (entry.get('host'), entry.get('port')) == mine:
                registry[run_id] = dict(entry, stopped=time.time())
        except Exception:
            pass


# GPU actor containers (rl/inference_server.py): the CPU image plus the CUDA plugin; actor processes stay on the
# CPU platform (JAX_PLATFORMS=cpu), the container's inference server switches itself to cuda.
gpu_actor_image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('curl', 'build-essential', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
    .pip_install('jax[cuda12]==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0', 'pytest>=9,<10', 'maturin>=1.7,<2')
    .add_local_dir(ROOT / 'engine-rs', '/workspace/engine-rs', copy=True, ignore=['**/target/**'])
    .run_commands('cd /workspace/engine-rs && PATH=/root/.cargo/bin:$PATH maturin build --release -o /workspace/wheels '
                  '&& pip install /workspace/wheels/*.whl')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**']))


@app.function(image=cpu_image, cpu=32, memory=65536, timeout=43200, retries=0, scaledown_window=2)
def actor(run_id: str, index: int, processes: int, games: int, fast: bool = True, wait_seconds: int = 1800,
          relay: bool = True):
    """processes=0: one pinned process per CPU the container exposes (cpu=16 -> 32). relay: one WeightRelay
    connection per container for weights and pool (lever 2); sample uploads keep a connection per process."""
    return _actor_body(run_id, index, processes, games, fast, wait_seconds, relay)


@app.function(image=gpu_actor_image, gpu='L4', cpu=16, memory=98304, timeout=43200, retries=0, scaledown_window=2)
def actor_gpu(run_id: str, index: int, processes: int, games: int, fast: bool = True, wait_seconds: int = 1800,
              relay: bool = True, serve_options: dict = None):
    """An actor container whose policy forwards run on its GPU (one inference server, rl/inference_server.py).
    processes may exceed the CPUs: a process waiting on its batch costs no CPU (48 per cpu=16 measured)."""
    return _actor_body(run_id, index, processes, games, fast, wait_seconds, relay, inference='cuda',
                       serve_options=serve_options)


def _actor_body(run_id, index, processes, games, fast, wait_seconds, relay, inference=None, serve_options=None):
    from polytopia_bot.rl.distributed import run_actor_container
    deadline = time.time() + wait_seconds
    while True:
        entry = registry.get(run_id)
        if entry and 'stopped' not in entry:
            break
        if (entry and 'stopped' in entry) or time.time() > deadline:
            return dict(index=index, status='learner unavailable')
        time.sleep(5)
    try:
        cpu_model = open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': ')
    except (OSError, IndexError):
        cpu_model = None
    def report(summary):
        registry[f'{run_id}/actor/{index}'] = dict(summary, cpus=os.cpu_count(), cpu_model=cpu_model, at=time.time())
    def should_stop():
        return 'stopped' in (registry.get(run_id) or {})
    summary = run_actor_container((entry['host'], entry['port']), processes, games, first_id=index * 1000, fast=fast,
                                  report=report, should_stop=should_stop, relay=relay, inference=inference,
                                  serve_options=serve_options)
    return dict(index=index, cpus=os.cpu_count(), **summary)


def _spawn_actors(run_id, first, count, actor_cpu, processes, games, hard, memory_gib, actor_gpu, serve_options):
    """CPU actor containers, or (actor_gpu, e.g. 'L4') GPU-inference ones."""
    if actor_gpu:
        fn = actor_gpu_fn_options(actor_gpu, actor_cpu, memory_gib, hard)
        return [fn.spawn(run_id, first + i, processes, games, serve_options=serve_options) for i in range(count)]
    return [actor.with_options(cpu=actor_cpu, memory=1024 * (memory_gib or actor_cpu), timeout=hard).spawn(
        run_id, first + i, processes, games) for i in range(count)]


def actor_gpu_fn_options(gpu, actor_cpu, memory_gib, hard):
    return actor_gpu.with_options(gpu=gpu, cpu=actor_cpu, memory=1024 * (memory_gib or 6 * actor_cpu), timeout=hard)


@app.function(image=cpu_image, cpu=32, memory=65536, timeout=7200, volumes=VOLUMES, retries=0)
def arena_match(name: str, candidate: str, opponent: str, seed_base: int, maps: int, games: int = 4,
                deterministic: bool = True, opponent_deterministic: bool = None, map_mix: str = None, size: int = 14):
    from polytopia_bot.rl.cpu_arena import match
    from polytopia_bot.rl.runner import write_json
    retained.reload()
    results.reload()
    summary = match(candidate, opponent, range(seed_base, seed_base + maps), games=games, deterministic=deterministic,
                    processes=min(2 * maps, 62), opponent_deterministic=opponent_deterministic, maps=map_mix, size=size)
    summary.update(name=name, size=size, candidate=candidate, opponent=opponent, seed_base=seed_base, maps=maps,
                   deterministic=deterministic,
                   opponent_deterministic=deterministic if opponent_deterministic is None else opponent_deterministic,
                   engine='engine-rs', map_mix=map_mix or 'classic-seed')
    out = Path('/results/distributed/arena') / f'{name}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, summary)
    results.commit()
    return {k: v for k, v in summary.items() if k != 'details'}


@app.function(image=gpu_image, gpu='H100', cpu=16, memory=65536, timeout=1800, volumes=VOLUMES, retries=0)
def learner_bench(parent: str, config: dict, epochs: int = 4):
    import tempfile
    config = dict(config)
    os.environ['POLYTOPIA_MATMUL_PRECISION'] = config.pop('precision', 'HIGHEST')  # read by rl.network at import
    for k, v in config.pop('env', {}).items():   # e.g. POLYTOPIA_HEAD_PRECISION, as launch --learner-env-json
        os.environ[k] = str(v)
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    cfg = LearnerConfig(**{**config, 'frozen': tuple(tuple(x) for x in config.get('frozen', ()))})
    learner_ = Learner(Path(tempfile.mkdtemp()) / 'bench', cfg, parent, log=lambda s: print(s, flush=True))
    return dict(devices=learner_.devices, rows=learner_.benchmark(epochs))


@app.local_entrypoint()
def bench_learner(parent: str, config_json: str = '{}', gpus: str = 'H100:1,H100:2', epochs: int = 4):
    calls = {g: learner_bench.with_options(gpu=g).spawn(parent, json.loads(config_json), epochs) for g in gpus.split(',')}
    for g, c in calls.items():
        print(g, json.dumps(c.get()))


@app.function(image=cpu_image, cpu=0.25, memory=1024, timeout=86400, retries=0)
def orchestrate(run_id: str, parent: str, config: dict, actors: int, actor_cpu: int, processes: int, games: int,
                gpu: str, hard: int, learner_cpu: int = 16, memory_gib: int = 0, env: dict = None, actor_gpu: str = '',
                serve_options: dict = None):
    """Detached runs keep only the last triggered function alive, so this one spawns the rest.

    Modal replays a preempted orchestrator input (even with retries=0). launch() clears `{run_id}/calls`, so an
    existing entry means this is a replay of a launch that already spawned its fleet: adopt those calls and wait on
    them instead of spawning a duplicate learner and fleet on the same run id."""
    prior = registry.get(f'{run_id}/calls')
    if prior:
        print(json.dumps(dict(stage='orchestrator_replay_adopted', calls=prior)), flush=True)
        call = modal.FunctionCall.from_id(prior['learner'])
        calls = [modal.FunctionCall.from_id(c) for c in prior['actors']]
    else:
        call = learner.with_options(gpu=gpu, timeout=hard, cpu=learner_cpu).spawn(run_id, parent, config, env)
        calls = _spawn_actors(run_id, 0, actors, actor_cpu, processes, games, hard, memory_gib, actor_gpu, serve_options)
        registry[f'{run_id}/calls'] = dict(learner=call.object_id, actors=[c.object_id for c in calls])
    out = dict(learner=None, actors=[])
    try:
        out['learner'] = call.get()
    except Exception as error:
        out['learner'] = dict(error=repr(error))
    for c in calls:
        try:
            out['actors'].append(c.get(timeout=600))
        except Exception as error:
            out['actors'].append(dict(error=repr(error)))
    registry[f'{run_id}/result'] = out
    return out


@app.function(image=cpu_image, cpu=0.25, memory=1024, timeout=86400, retries=0)
def add_actors(run_id: str, first_index: int, actors: int, actor_cpu: int, processes: int, games: int, hard: int,
               memory_gib: int = 0, actor_gpu: str = '', serve_options: dict = None):
    calls = _spawn_actors(run_id, first_index, actors, actor_cpu, processes, games, hard, memory_gib, actor_gpu,
                          serve_options)
    return [c.get() for c in calls]


@app.function(image=cpu_image, cpu=0.25, memory=1024, timeout=86400, retries=0, volumes=VOLUMES)
def ladder(run_ids: list, opponent: str, interval: int, rounds: int, seed_base: int, maps: int):
    """Every `interval` s: freeze each run's latest checkpoint and play it vs `opponent` on a fresh seed block."""
    import shutil
    from polytopia_bot.rl.runner import write_json
    for r in range(rounds):
        time.sleep(interval)
        results.reload()
        calls = {}
        for run_id in run_ids:
            src = Path('/results/distributed') / run_id / 'latest.npz'
            if not src.exists():
                continue
            meta = json.loads(src.with_suffix('.json').read_text())
            dst = Path('/results/distributed/ladder') / f'{run_id}-{meta["stage"]}.npz'
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            shutil.copy(src.with_suffix('.json'), dst.with_suffix('.json'))
            results.commit()
            calls[run_id] = (meta['stage'], arena_match.spawn(f'ladder-{run_id}-{meta["stage"]}-{seed_base + 100 * r}',
                                                               str(dst), opponent, seed_base + 100 * r, maps))
        rows = []
        for run_id, (stage, call) in calls.items():
            try:
                out = call.get()
                rows.append(dict(run_id=run_id, stage=stage, round=r, seed_base=seed_base + 100 * r,
                                 **{k: out[k] for k in ('wins', 'losses', 'draws_or_truncations', 'win_rate', 'ci95',
                                                        'candidate_unit_mix', 'candidate_non_warrior_games',
                                                        'candidate_unit_types_per_game',
                                                        'candidate_first_non_warrior_turn_median')}))
            except Exception as error:
                rows.append(dict(run_id=run_id, stage=stage, round=r, error=repr(error)))
        history = registry.get('ladder/history', []) + rows
        registry['ladder/history'] = history
        out = Path('/results/distributed/ladder/history.json')
        write_json(out, history)
        results.commit()


LADDER_KEYS = ('wins', 'losses', 'draws_or_truncations', 'truncations', 'win_rate', 'ci95', 'candidate_unit_mix',
               'candidate_non_warrior_games', 'candidate_unit_types_per_game', 'candidate_first_non_warrior_turn_median',
               'by_map', 'map_mix')


@app.function(image=cpu_image, cpu=0.25, memory=1024, timeout=86400, retries=0, volumes=VOLUMES)
def elo_ladder(tag: str, run_ids: list, references: dict, interval: int, rounds: int, seed_base: int, maps: int,
               first_delay: int = None, map_mix: str = None, size: int = 14):
    """Every `interval` s: freeze each run's latest checkpoint and play it against every fixed reference
    (name -> path) on one fresh seed block. Rows go to registry `{tag}/history` for scripts/ladder_elo.py."""
    import shutil
    from polytopia_bot.rl.runner import write_json
    for r in range(rounds):
        time.sleep(interval if first_delay is None or r else first_delay)
        results.reload()
        seeds = seed_base + 100 * r
        calls = []
        for run_id in run_ids:
            src = Path('/results/distributed') / run_id / 'latest.npz'
            if not src.exists():
                continue
            meta = json.loads(src.with_suffix('.json').read_text())
            dst = Path('/results/distributed/ladder') / f'{run_id}-{meta["stage"]}.npz'
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            shutil.copy(src.with_suffix('.json'), dst.with_suffix('.json'))
            results.commit()
            for ref, path in references.items():
                calls.append((run_id, meta['stage'], str(dst), ref, path, arena_match.spawn(
                    f'{tag}-{run_id}-{meta["stage"]}-vs-{ref}-{seeds}', str(dst), path, seeds, maps, map_mix=map_mix,
                    size=size)))
        rows = []
        for run_id, stage, candidate, ref, path, call in calls:
            try:
                out = call.get()
                rows.append(dict(run_id=run_id, stage=stage, candidate=candidate, reference=ref, opponent=path,
                                 round=r, seed_base=seeds, finished=time.time(), **{k: out.get(k) for k in LADDER_KEYS}))
            except Exception as error:
                rows.append(dict(run_id=run_id, stage=stage, reference=ref, round=r, error=repr(error)))
        history = registry.get(f'{tag}/history', []) + rows
        registry[f'{tag}/history'] = history
        write_json(Path('/results/distributed/ladder') / f'{tag}-history.json', history)
        results.commit()


@app.local_entrypoint()
def start_elo_ladder(tag: str, run_ids: str, references_json: str, interval: int = 2700, rounds: int = 12,
                     seed_base: int = 310000, maps: int = 32, first_delay: int = -1, map_mix: str = '', size: int = 14):
    call = elo_ladder.spawn(tag, run_ids.split(','), json.loads(references_json), interval, rounds, seed_base, maps,
                            None if first_delay < 0 else first_delay, map_mix or None, size)
    print(json.dumps(dict(ladder_call=call.object_id, app_id=app.app_id)))


@app.function(image=cpu_image, cpu=0.25, memory=1024, timeout=14400, retries=0, volumes=VOLUMES)
def round_robin(tag: str, players: dict, seed_base: int, maps: int):
    """Every pair of fixed players once (anchors the reference Elo scale)."""
    names = list(players)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    calls = [(a, b, arena_match.spawn(f'{tag}-{a}-vs-{b}-{seed_base}', players[a], players[b], seed_base, maps))
             for a, b in pairs]
    rows = []
    for a, b, call in calls:
        out = call.get()
        rows.append(dict(run_id=a, stage='fixed', candidate=players[a], reference=b, opponent=players[b], round=-1,
                         seed_base=seed_base, finished=time.time(), **{k: out.get(k) for k in LADDER_KEYS}))
    registry[f'{tag}/history'] = registry.get(f'{tag}/history', []) + rows
    return rows


@app.function(image=cpu_image, cpu=0.25, memory=1024, timeout=14400, retries=0, volumes=VOLUMES)
def cross(tag: str, candidates: dict, references: dict, seed_base: int, maps: int):
    """Every candidate (name -> path) against every reference once; rows join the `{tag}/history` ladder."""
    results.reload()
    calls = []
    for c, cpath in candidates.items():
        stage = json.loads(Path(cpath).with_suffix('.json').read_text()).get('stage', 'fixed')
        for ref, rpath in references.items():
            if ref != c:
                calls.append((c, stage, cpath, ref, rpath, arena_match.spawn(
                    f'{tag}-{c}-{stage}-vs-{ref}-{seed_base}', cpath, rpath, seed_base, maps)))
    rows = []
    for c, stage, cpath, ref, rpath, call in calls:
        out = call.get()
        rows.append(dict(run_id=c, stage=stage, candidate=cpath, reference=ref, opponent=rpath, round='final',
                         seed_base=seed_base, finished=time.time(), **{k: out.get(k) for k in LADDER_KEYS}))
    registry[f'{tag}/history'] = registry.get(f'{tag}/history', []) + rows
    return rows


@app.local_entrypoint()
def cross_eval(tag: str, candidates_json: str, references_json: str, seed_base: int = 311000, maps: int = 64):
    call = cross.spawn(tag, json.loads(candidates_json), json.loads(references_json), seed_base, maps)
    print(json.dumps(dict(cross_call=call.object_id, app_id=app.app_id)))


@app.local_entrypoint()
def reference_round_robin(tag: str, players_json: str, seed_base: int = 319000, maps: int = 32):
    for row in round_robin.remote(tag, json.loads(players_json), seed_base, maps):
        print(json.dumps({k: row[k] for k in ('run_id', 'reference', 'wins', 'losses', 'draws_or_truncations')}))


@app.local_entrypoint()
def start_ladder(run_ids: str, opponent: str, interval: int = 2700, rounds: int = 12, seed_base: int = 305000,
                 maps: int = 32):
    call = ladder.spawn(run_ids.split(','), opponent, interval, rounds, seed_base, maps)
    print(json.dumps(dict(ladder_call=call.object_id, app_id=app.app_id)))


# ---- A6 batch assembly (docs/a6-batch-assembly-2026-10-22.md; ledger runs/a6)
A6_MESSAGES = '/results/a6/messages'


@app.function(image=cpu_image, cpu=16, memory=32768, timeout=1800, volumes=VOLUMES, retries=0)
def assembly_capture_fn(parent: str, config: dict, count: int, processes: int, games: int, skip: int = 0,
                        messages_dir: str = A6_MESSAGES):
    """Real actor messages for the assembly bench: Rust actors on this container against a never-trained learner."""
    import shutil
    import tempfile
    from polytopia_bot.rl.assembly_bench import capture
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    retained.reload()
    cfg = LearnerConfig(**{**config, 'frozen': tuple(tuple(x) for x in config.get('frozen', ()))})
    learner_ = Learner(Path(tempfile.mkdtemp()) / 'capture', cfg, parent, port=7071, log=lambda s: print(s, flush=True))
    shutil.rmtree(messages_dir, ignore_errors=True)
    out = capture(learner_, messages_dir, count, processes, games, log=lambda s: print(s, flush=True), skip=skip)
    results.commit()
    return out


@app.local_entrypoint()
def assembly_capture(config_json: str, count: int = 64, processes: int = 16, games: int = 24, skip: int = 0,
                     messages_dir: str = A6_MESSAGES, memory_gib: int = 32):
    cfg = json.loads(config_json)
    out = assembly_capture_fn.with_options(memory=memory_gib * 1024).remote(cfg.pop('parent'), cfg, count, processes,
                                                                              games, skip, messages_dir)
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'} | dict(rows_total=sum(out['rows']),
                     rows_min=min(out['rows']), rows_max=max(out['rows']))), flush=True)


# ---- levers (docs/levers-egress-2026-10-22.md; ledger runs/levers)
LEVER_MESSAGES = '/results/levers/messages-mid'


@app.function(image=cpu_image, cpu=16, memory=32768, timeout=3600, volumes=VOLUMES, retries=0)
def upload_codec_fn(parent: str, config: dict, skip: int, count: int, processes: int, games: int, levels: list):
    """Lever 4: (1) unless count=0, capture `count` real actor messages after skipping `skip` (mid-game positions,
    trained policy `parent`) to LEVER_MESSAGES; (2) zstd levels and lossless reformats on those and on A6's 64
    early-game messages, timed on one pinned core like an actor process."""
    import shutil
    import tempfile
    from polytopia_bot.rl.assembly_bench import capture, codec_bench
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    results.reload()
    retained.reload()
    out = {}
    if count:
        cfg = LearnerConfig(**{**config, 'frozen': tuple(tuple(x) for x in config.get('frozen', ()))})
        learner_ = Learner(Path(tempfile.mkdtemp()) / 'capture', cfg, parent, port=7072, log=lambda s: print(s, flush=True))
        shutil.rmtree(LEVER_MESSAGES, ignore_errors=True)
        info = capture(learner_, LEVER_MESSAGES, count, processes, games, log=lambda s: print(s, flush=True), skip=skip)
        out['capture'] = {k: v for k, v in info.items() if k != 'rows'} | dict(rows_total=sum(info['rows']))
        results.commit()
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
    for name, folder in (('early_a6', A6_MESSAGES), ('mid_game', LEVER_MESSAGES)):
        paths = sorted(str(p) for p in Path(folder).glob('msg-*.bin'))
        if paths:
            out[name] = codec_bench(paths, tuple(levels))
            print(json.dumps(dict(stage='codec', set=name, **out[name])), flush=True)
    out['cpu'] = open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': ')
    return out


@app.local_entrypoint()
def upload_codec(parent: str, config_json: str = '{}', skip: int = 2000, count: int = 64, processes: int = 16,
                 games: int = 24, levels: str = '1,3,6', out: str = 'runs/levers/upload-codec.json',
                 ledger: str = 'runs/levers'):
    _ledger(dict(run_id=f'upload-codec-{int(time.time())}', kind='bench', hourly_usd=round(cpu_hourly(16), 2),
                 reserved_usd=0.4, status='reserved', modal_app_id=app.app_id, start_unix=time.time()), ledger)
    result = upload_codec_fn.remote(parent, json.loads(config_json), skip, count, processes, games,
                                    [int(x) for x in levels.split(',')])
    (ROOT / out).write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))


@app.function(image=gpu_image, gpu='H100', cpu=16, memory=65536, timeout=3600, volumes=VOLUMES, retries=0)
def assembly_bench_fn(parent: str, config: dict, env: dict, seconds: int, variants: list, gate_cases: list,
                      clients: int = 6, connections: int = 16, tag: str = 'bench', profile: bool = False,
                      messages_dir: str = A6_MESSAGES, lag: list = None):
    """Same container: (1) the equality gate on real captured messages, per option case; (2) learner throughput
    for each stream_views variant with unthrottled localhost clients replaying the captured messages.
    lag=[low, high]: live-run conditions (assembly_bench.client_process): the config's max_lag stays on, clients
    restamp versions that far behind and connect relayed, as the hero runs' 912 actor processes did."""
    import dataclasses
    import multiprocessing as mp
    import tempfile
    for k, v in env.items():
        os.environ[k] = str(v)
    import jax
    import numpy as np
    jax.config.update('jax_compilation_cache_dir', '/results/.jax-compilation-cache')
    from polytopia_bot.rl import assembly_bench as ab
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    from polytopia_bot.rl.runner import write_json
    results.reload()
    retained.reload()
    paths = sorted(str(p) for p in Path(messages_dir).glob('msg-*.bin'))
    log = lambda s: print(s, flush=True)
    cfg = LearnerConfig(**{**config, 'frozen': tuple(tuple(x) for x in config.get('frozen', ()))})
    learner_ = Learner(Path(tempfile.mkdtemp()) / 'bench', cfg, parent, log=log)
    out = dict(devices=learner_.devices, messages=len(paths), gate={}, throughput={})
    if gate_cases:
        rng = np.random.default_rng(0)
        messages = [ab.decode(Path(p).read_bytes()) for p in paths]
        for case in gate_cases:
            learner_.config, learner_.version = cfg, 0
            msgs = messages
            if case == 'turn_clock':
                learner_.config = dataclasses.replace(cfg, turn_gamma=.99, turn_lambda=.95)
                msgs = ab.with_options(messages, rng, turn_clock=True)
            elif case == 'upgo_phi':
                learner_.config = dataclasses.replace(cfg, upgo_coef=.5)
                msgs = ab.with_options(messages, rng, shaping=True)
            elif case in ('stale', 'coded_stale'):
                learner_.config, learner_.version = dataclasses.replace(cfg, max_lag=10), 12
                msgs = ab.with_options(messages, rng, stale_version=12)
            # coded*: dense streamed vs coded views (view_codec) on these messages; else host vs streamed
            report = (ab.coded_gate if case.startswith('coded') else ab.gate)(learner_, msgs)
            out['gate'][case] = report
            log(json.dumps(dict(stage='gate', case=case, **{k: v for k, v in report.items() if k != 'info_streamed'})))
        learner_.config, learner_.version = cfg, 0
        del messages, msgs
    ctx = mp.get_context('spawn')
    for k, stream in enumerate(variants):
        learner_.config = dataclasses.replace(cfg, stream_views=bool(stream), training_seconds=seconds,
                                              max_lag=cfg.max_lag if lag else 10 ** 9, checkpoint_every=10 ** 9,
                                              snapshot_every=10 ** 9)
        learner_.port, learner_.stop = 7080 + k, False
        procs = [ctx.Process(target=ab.client_process, args=(learner_.port, paths, connections, 1000 * c, seconds + 300,
                                                            bool(cfg.coded_views), lag), daemon=True) for c in range(clients)]
        for p in procs:
            p.start()
        sampler = None
        if profile:   # samples the learner's threads after a minute of warm-up (compiles excluded)
            sampler = ab.StackSampler()
            threading.Timer(60, sampler.start).start()
        history = learner_.run(bind_host='127.0.0.1')
        learner_.server.close()
        while not learner_.inbox.empty():
            learner_.inbox.get()
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        name = 'stream' if stream else 'concat'
        out['throughput'][f'{k}-{name}'] = dict(summary=ab.summarize(history),
                                                history=[{kk: vv for kk, vv in r.items() if kk != 'actors'} for r in history])
        if sampler is not None:
            out['throughput'][f'{k}-{name}']['profile'] = sampler.report()
        log(json.dumps(dict(stage='throughput', variant=name, **out['throughput'][f'{k}-{name}']['summary'])))
    write_json(Path('/results/a6') / f'{tag}.json', out)
    results.commit()
    return out


@app.local_entrypoint()
def assembly_bench(config_json: str, seconds: int = 240, variants: str = '0,1', gate_cases: str = 'base,stale,turn_clock,upgo_phi',
                   env_json: str = '{}', tag: str = 'bench', out: str = '', gpu: str = 'H100', profile: bool = False,
                   clients: int = 6, connections: int = 16, cpu: int = 16, memory_gib: int = 64,
                   messages_dir: str = A6_MESSAGES, spawn: bool = False, lag: str = ''):
    """--spawn (with modal run --detach): start the bench and return; it survives the laptop's connection and
    writes /results/a6/<tag>.json. --lag low,high: live-run version lag (assembly_bench_fn)."""
    cfg = json.loads(config_json)
    fn = assembly_bench_fn.with_options(gpu=gpu, cpu=cpu, memory=memory_gib * 1024)
    args = (cfg.pop('parent'), cfg, json.loads(env_json), seconds, [int(v) for v in variants.split(',') if v],
            [c for c in gate_cases.split(',') if c])
    kwargs = dict(tag=tag, profile=profile, clients=clients, connections=connections, messages_dir=messages_dir,
                  lag=[int(x) for x in lag.split(',')] if lag else None)
    if spawn:
        print(json.dumps(dict(call=fn.spawn(*args, **kwargs).object_id, app_id=app.app_id, tag=tag)), flush=True)
        return
    result = fn.remote(*args, **kwargs)
    if out:
        (ROOT / out).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / out).write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(dict(gate={k: dict(equal=v['equal'], rows=v['rows'], host=v['host_timings'],
                                        streamed=v['streamed_timings']) for k, v in result['gate'].items()},
                          throughput={k: v['summary'] for k, v in result['throughput'].items()}), indent=1), flush=True)


@app.local_entrypoint()
def tests(paths: str = 'tests/test_real_rules.py,tests/test_distributed_rl.py', parallel: bool = False,
          out: str = ''):
    """--parallel runs one container per test file (comma-separated paths)."""
    groups = paths.split(',') if parallel else [paths]
    pending = [(g, run_tests.spawn(g)) for g in groups]
    failed, lines = 0, []
    while pending:  # report each file as it finishes
        for item in list(pending):
            g, call = item
            try:
                code, text = call.get(timeout=5)
            except (TimeoutError, modal.exception.TimeoutError, ConnectionError):   # a dropped client link: poll again
                continue
            pending.remove(item)
            failed |= code
            lines.append(f'===== {g}: exit {code}\n{text}')
            print(lines[-1], flush=True)
    if out:
        (ROOT / out).write_text('\n'.join(lines) + '\n')
    if failed:
        raise SystemExit(1)


LEDGER = 'runs/distributed-rl-v1'
LEDGER_18 = 'runs/rl-18x18-v1'  # $300 cap (budget_usd); see docs/rl-18x18-2026-10-19.md


def _ledger(row, folder=LEDGER):
    import sys
    sys.path.insert(0, str(ROOT / 'src'))
    from polytopia_bot.rl.budget import reserve
    reserve(ROOT / folder, row)


def cpu_hourly(cores):
    return cores * (CPU_CORE_RATE + MEM_GIB_RATE)


ACTOR_GIB_PER_PROCESS = 1.25   # lever 1 smoke (docs/levers-egress-2026-10-22.md): 32 processes x 24 games in 40 GiB


def actor_memory_gib(actor_cpu, processes, memory_gib=0):
    """Actor container memory: memory_gib if given, else ACTOR_GIB_PER_PROCESS per process, where processes=0 means
    one per exposed CPU (2 per reserved Modal core)."""
    return memory_gib or int(-(-ACTOR_GIB_PER_PROCESS * (processes or 2 * actor_cpu) // 1))


@app.local_entrypoint()
def launch(run_id: str, parent: str, config_json: str = '{}', actors: int = 4, actor_cpu: int = 16,
           processes: int = 0, games: int = 24, gpu: str = 'H100:1', seconds: int = 3600, learner_cpu: int = 16,
           ledger: str = LEDGER, memory_gib: int = 0, learner_env_json: str = '{}', actor_gpu: str = '',
           serve_options_json: str = '{}'):
    """Spawn one learner (gpu like H100:4) and `actors` CPU containers of actor_cpu physical cores.
    actor_gpu (e.g. L4, A10G): GPU-inference actor containers instead (rl/inference_server.py; serve_options_json
    e.g. {"trunk_dtype": "bfloat16"}); then processes may exceed the CPUs (48 per cpu=16 measured).
    processes=0 (default): one pinned actor process per CPU the container exposes (cpu=16 shows 32; lever 1).
    memory_gib=0: actor_memory_gib. --learner-env-json sets environment variables in the learner container
    (e.g. the TF32 precision hook)."""
    config = json.loads(config_json)
    learner_env = json.loads(learner_env_json)
    config['training_seconds'] = seconds
    memory_gib = actor_memory_gib(actor_cpu, processes, memory_gib) if not actor_gpu else (memory_gib or 6 * actor_cpu)
    serve_options = json.loads(serve_options_json)
    hard = seconds + 2400
    gpu_type, _, count = gpu.partition(':')
    count = int(count or 1)
    hourly = GPU_RATE[gpu_type] * count + learner_cpu * CPU_CORE_RATE + 64 * MEM_GIB_RATE + actors * (
        actor_cpu * CPU_CORE_RATE + memory_gib * MEM_GIB_RATE + (GPU_RATE.get(actor_gpu, 1.10) if actor_gpu else 0.))
    _ledger(dict(run_id=run_id, parent=parent, config=config, actors=actors, actor_cpu=actor_cpu,
                 processes=processes or 'all CPUs', games=games, gpu=gpu, learner_cpu=learner_cpu, memory_gib=memory_gib,
                 seconds=seconds,
                 learner_env=learner_env, hourly_usd=round(hourly, 2),
                 reserved_usd=round(hourly * (seconds + 900) / 3600, 2), status='reserved', modal_app_id=app.app_id,
                 start_unix=time.time()), ledger)
    # A fresh launch (or a relaunch that resumes the run folder) starts with no registry state, so orchestrate()
    # spawns a fleet; only a Modal replay of that orchestrator finds `{run_id}/calls` and adopts it.
    for key in (run_id, f'{run_id}/calls', f'{run_id}/result'):
        try:
            registry.pop(key)
        except KeyError:
            pass
    call = orchestrate.spawn(run_id, parent, config, actors, actor_cpu, processes, games, gpu, hard, learner_cpu,
                             memory_gib, learner_env, actor_gpu, serve_options)
    print(json.dumps(dict(run_id=run_id, app_id=app.app_id, orchestrator_call=call.object_id,
                          hourly_usd=round(hourly, 2))), flush=True)


@app.local_entrypoint()
def replay_guard_check(run_id: str = 'replay-guard-check'):
    """Simulate a Modal replay of orchestrate(): seed `{run}/calls` with three short actor calls (the run entry is
    marked stopped, so they return at once) and call orchestrate() as a replay would. It must adopt those calls and
    return their results without spawning a learner (none is registered for this run id)."""
    for key in (run_id, f'{run_id}/calls', f'{run_id}/result'):
        try:
            registry.pop(key)
        except KeyError:
            pass
    registry[run_id] = dict(stopped=time.time())
    small = actor.with_options(cpu=1, memory=1024, timeout=300)
    seeded = [small.spawn(run_id, i, 1, 1, wait_seconds=5) for i in range(3)]
    registry[f'{run_id}/calls'] = dict(learner=seeded[0].object_id, actors=[c.object_id for c in seeded[1:]])
    out = orchestrate.remote(run_id, '/nonexistent.npz', {}, 2, 1, 1, 1, 'H100:1', 300)
    adopted = out['learner'] == dict(index=0, status='learner unavailable') and \
        [a.get('index') for a in out['actors']] == [1, 2]
    print(json.dumps(dict(out=out, adopted=adopted, calls_unchanged=registry.get(f'{run_id}/calls')['learner'] ==
                          seeded[0].object_id)))
    if not adopted:
        raise SystemExit(1)


@app.local_entrypoint()
def more_actors(run_id: str, first_index: int, actors: int = 2, actor_cpu: int = 16, processes: int = 0,
                games: int = 24, hard: int = 36000, ledger: str = LEDGER, memory_gib: int = 0):
    """Attach extra actor containers to a live learner (new detached app). processes and memory_gib as in launch;
    18x18 actor processes reach ~2.5 GiB each."""
    memory_gib = actor_memory_gib(actor_cpu, processes, memory_gib)
    hourly = actors * (actor_cpu * CPU_CORE_RATE + memory_gib * MEM_GIB_RATE)
    _ledger(dict(run_id=f'{run_id}-actors-{first_index}', parent=run_id, actors=actors, actor_cpu=actor_cpu,
                 processes=processes, games=games, hourly_usd=round(hourly, 2), reserved_usd=round(hourly * hard / 3600, 2),
                 status='reserved', modal_app_id=app.app_id, start_unix=time.time()), ledger)
    call = add_actors.spawn(run_id, first_index, actors, actor_cpu, processes, games, hard, memory_gib)
    print(json.dumps(dict(run_id=run_id, app_id=app.app_id, call=call.object_id, hourly_usd=round(hourly, 2))))


@app.local_entrypoint()
def arena(name: str, candidate: str, opponent: str, seed_base: int = 302000, maps: int = 32, games: int = 4,
          deterministic: bool = True, opponent_stochastic: bool = False, map_mix: str = '', size: int = 14):
    """Candidate mode follows --deterministic/--no-deterministic; the opponent is deterministic unless
    --opponent-stochastic. --map-mix plays maps.stream layouts (e.g. proposed-v1, uniform, a type name)."""
    out = arena_match.remote(name, candidate, opponent, seed_base, maps, games, deterministic,
                             not opponent_stochastic, map_mix or None, size)
    print(json.dumps(out, indent=1))



# ----------------------------------------------------------------------------- 18x18 campaign

C3 = '/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz'
MAP_TYPES = ('classic', 'drylands', 'lakes', 'pangea', 'continents', 'archipelago', 'waterworld', 'highlands',
             'ridge', 'jungle')


@app.function(image=cpu_image, cpu=0.25, memory=1024, timeout=14400, retries=0, volumes=VOLUMES)
def per_type_report(tag: str, candidate: str, opponent: str, seed_base: int, maps: int, size: int, sampled: bool,
                    types: list):
    """One arena per map type (seeds seed_base + 100 k for the k-th type), compact rows without game details."""
    calls = [(t, arena_match.spawn(f'{tag}-{t}', candidate, opponent, seed_base + 100 * k, maps,
                                   deterministic=not sampled, opponent_deterministic=not sampled, map_mix=t, size=size))
             for k, t in enumerate(types)]
    rows = {}
    for t, call in calls:
        try:
            rows[t] = call.get()
        except Exception as error:
            rows[t] = dict(error=repr(error))
    out = dict(tag=tag, candidate=candidate, opponent=opponent, seed_base=seed_base, maps_per_type=maps, size=size,
               sampled=sampled, types=rows)
    from polytopia_bot.rl.runner import write_json
    write_json(Path('/results/distributed/arena') / f'{tag}-report.json', out)
    results.commit()
    return out


@app.local_entrypoint()
def tech_report(tag: str, candidate: str = C3, opponent: str = '', seed_base: int = 374000, maps: int = 24,
                size: int = 14, sampled: bool = True, types: str = ','.join(MAP_TYPES), minutes: int = 15,
                ledger: str = LEDGER_18):
    """Tech/unit/game-length report per map type (self-play when no opponent is given), both seats."""
    kinds = types.split(',')
    hourly = len(kinds) * cpu_hourly(32)
    _ledger(dict(run_id=tag, kind='tech-report', candidate=candidate, opponent=opponent or candidate, size=size,
                 seed_base=seed_base, maps=maps, types=kinds, hourly_usd=round(hourly, 2),
                 reserved_usd=round(hourly * minutes / 60, 2), status='reserved', modal_app_id=app.app_id,
                 start_unix=time.time()), ledger)
    out = per_type_report.remote(tag, candidate, opponent or candidate, seed_base, maps, size, sampled, kinds)
    path = ROOT / ledger / f'{tag}.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    for t, r in out['types'].items():
        if 'error' in r:
            print(t, r['error'])
            continue
        tech = r['by_map'][t]['both_tech']
        print(f"{t:12s} W-L-T {r['wins']}-{r['losses']}-{r['truncations']} turns {r['mean_turns']:.0f} "
              f"techs {tech['final_mean']:.1f} tier3 {tech['tier3_mean']:.2f} ({tech['share_with_tier3']:.0%}) "
              f"units {list(r['by_map'][t]['both_unit_mix'].items())[:4]} {r['seconds']:.0f}s")
    print(f'saved {path}')


GROWN = '/retained/checkpoints/distributed-final-v1/c3-u4854-grown18x2/final.npz'


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=1800, volumes=VOLUMES, retries=0)
def grow_checkpoint(parent: str, out: str, size: int, extra_blocks: int, seed: int, positions: int):
    """network.grow(parent) saved with its architecture; verified to sample exactly like the parent
    (same choices, log-probs and values, deterministic and stochastic) on 14x14 and size x size positions."""
    import dataclasses
    import jax
    import jax.numpy as jnp
    import numpy as np
    from polytopia_bot.rl import network
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.rl.distributed import checkpoint_template, network_spec
    from polytopia_bot.rl.runner import Config, load_checkpoint, save_checkpoint, write_json
    from polytopia_bot.maps.stream import MapSource
    from polytopia_bot.rl.data import sha256
    retained.reload()
    params = load_checkpoint(parent, checkpoint_template(parent))
    grown = network.grow(params, jax.random.key(seed), size=size, extra_blocks=extra_blocks)
    meta = json.loads(Path(parent).with_suffix('.json').read_text())
    config = Config(**{**meta['config'], 'size': size, 'extra_blocks': extra_blocks})
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(target, grown, config, stage=f'grown-from-{meta["stage"]}')
    loaded = load_checkpoint(target, checkpoint_template(target))
    checks = []
    for board in sorted({14, size}):
        games = RustGames(board, 8, map_batch=True)
        states, prepared = games.reset(MapSource('uniform', board).batch(range(372000, 372008))[0])
        key = jax.random.key(1)
        f = jax.jit(jax.vmap(lambda p, x, m, k, det: network.sample(p, x, m, k, size=board, deterministic=det),
                             in_axes=(None, 0, 0, 0, None)), static_argnums=4)
        for step in range(positions):
            outs = {}
            for name, p in (('parent', params), ('grown', loaded)):
                outs[name] = [f(p, prepared.features, prepared.masks, jax.random.split(key, 8), det) for det in (True, False)]
            same = all(np.array_equal(np.asarray(a), np.asarray(b))
                       for a, b in zip(jax.tree.leaves(outs['parent']), jax.tree.leaves(outs['grown'])))
            checks.append(dict(board=board, step=step, identical=bool(same)))
            choice = outs['parent'][1][0]
            key, _ = jax.random.split(key)
            tick, status = games.tick(states, choice, MapSource('uniform', board).batch(range(372100 + 8 * step, 372108 + 8 * step))[0])
            assert not np.any(np.asarray(status)), 'illegal action'
            states, prepared = tick.states, tick.prepared
    report = dict(parent=parent, parent_sha256=sha256(parent), out=str(target), out_sha256=sha256(target),
                  spec=network_spec(target), seed=seed, positions=len(checks),
                  identical=all(c['identical'] for c in checks), checks=checks)
    write_json(target.parent / 'grow-report.json', report)
    retained.commit()
    return {k: v for k, v in report.items() if k != 'checks'}


@app.local_entrypoint()
def grow(parent: str = C3, out: str = GROWN, size: int = 18, extra_blocks: int = 2, seed: int = 18,
         positions: int = 40, ledger: str = LEDGER_18):
    _ledger(dict(run_id=f'grow-{Path(out).parent.name}', kind='grow', hourly_usd=round(cpu_hourly(8), 2),
                 reserved_usd=0.25, status='reserved', modal_app_id=app.app_id, start_unix=time.time()), ledger)
    print(json.dumps(grow_checkpoint.remote(parent, out, size, extra_blocks, seed, positions), indent=1))


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=1800, volumes=VOLUMES, retries=0)
def random_init_fn(reference: str, out: str, seed: int, head_scale: float, games: int, steps: int, blocks: int = 0,
                   width: int = 0, players: int = 2, size: int = 0):
    """A random-init parent with `reference`'s architecture (network.init(key(seed)), optionally with the policy
    output weights scaled by head_scale; blocks/width > 0 override the reference's), saved with save_checkpoint, plus how close its policy is to uniform over
    the legal actions: per-head entropy vs log(#legal) on positions it reaches in self-play on proposed-v1 maps
    (four-v1 with players=4). players=4 builds four-seat inputs; size > 0 overrides the reference's board size."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import family_mask, source_mask, target_mask
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.rl.distributed import network_spec, checkpoint_template
    from polytopia_bot.rl.runner import Config, load_checkpoint, save_checkpoint, write_json
    from polytopia_bot.maps.stream import MapSource
    from polytopia_bot.rl.distributed import feature_shapes
    retained.reload()
    results.reload()
    spec = network_spec(reference)
    spec.update({k: v for k, v in dict(blocks=blocks, width=width, size=size).items() if v})
    spec.pop('players', None)
    board_channels, global_features = feature_shapes(players)
    params = network.init(jax.random.key(seed), board_channels, global_features,
                          size=spec['size'], width=spec['width'], blocks=spec['blocks'],
                          architecture=spec['architecture'], extra_blocks=spec['extra_blocks'])
    if head_scale != 1.:
        # family/source logits are linear in these; the target head's spatial keys are trunk tiles, so only the
        # non-spatial embeddings can be scaled there
        for k in ('family', 'source'):
            params[k] = dict(params[k], w=params[k]['w'] * head_scale)
        params['parameter_embedding'] = params['parameter_embedding'] * head_scale
    meta = json.loads(Path(reference).with_suffix('.json').read_text())
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(target, params, Config(**{**meta['config'], 'blocks': spec['blocks'], 'width': spec['width'],
                                              'size': spec['size'], 'players': players}),
                    stage=f'random-init-seed{seed}-scale{head_scale:g}')
    params = load_checkpoint(target, checkpoint_template(target))
    size = spec['size']

    def heads(p, x, m, k):
        tiles, g, _ = network.trunk(p, x)
        choice, path, _, _ = network.sample(p, x, m, k, size=size)
        logits = (network.dense(g, p['family']), network.dense(tiles, p['source'])[:, choice.family],
                  network.target_logits(p, tiles, g, choice.family, choice.source))
        rows = []
        for l, mask in zip(logits, path):
            lp = network.log_distribution(l, mask)
            rows.append(jnp.stack([-jnp.sum(jnp.where(mask, jnp.exp(lp) * lp, 0.)), jnp.log(jnp.sum(mask)),
                                   jnp.max(jnp.where(mask, jnp.exp(lp), 0.))]))
        return choice, jnp.stack(rows)                                 # [head, (entropy, log #legal, max prob)]
    f = jax.jit(jax.vmap(heads, in_axes=(None, 0, 0, 0)))
    engine = RustGames(size, games, players=players, map_batch=True)
    source = MapSource('four-v1' if players == 4 else 'proposed-v1', size, players=players)
    states, prepared = engine.reset(source.batch(range(372000, 372000 + games))[0])
    key, stats = jax.random.key(seed + 1), []
    for step in range(steps):
        key, sub = jax.random.split(key)
        choice, h = f(params, prepared.features, prepared.masks, jax.random.split(sub, games))
        stats.append(np.asarray(h))
        tick, status = engine.tick(states, choice, source.batch(range(373000 + games * step, 373000 + games * (step + 1)))[0])
        assert not np.any(np.asarray(status)), 'illegal action'
        states, prepared = tick.states, tick.prepared
    s = np.stack(stats).reshape(-1, 3, 3)                              # [positions, head, stat]
    report = dict(reference=reference, out=str(target), spec=spec, seed=seed, head_scale=head_scale,
                  positions=len(s), heads={})
    for i, name in enumerate(('family', 'source', 'target')):
        choosing = s[:, i, 1] > 1e-6                                     # positions with >1 legal option
        e, u, mx = (s[choosing, i, j] for j in range(3))
        report['heads'][name] = dict(choice_positions=int(choosing.sum()), entropy=float(e.mean()),
                                     uniform=float(u.mean()), ratio=float(e.sum() / u.sum()),
                                     max_gap=float((u - e).max()), mean_max_prob=float(mx.mean()),
                                     mean_uniform_prob=float(np.exp(-u).mean()))
    total = s[:, :, 0].sum(1)
    report['path_entropy'] = dict(mean=float(total.mean()), uniform=float(s[:, :, 1].sum(1).mean()))
    write_json(target.parent / f'{target.stem}-report.json', report)
    results.commit()
    return report


@app.local_entrypoint()
def random_init(reference: str = '/retained/checkpoints/distributed-final-v1/c5-omni-s1-final/final.npz',
                out: str = '/results/a6-scratch/random-init.npz', seed: int = 20261022, head_scale: float = 1.,
                games: int = 16, steps: int = 150, ledger: str = 'runs/scratch', blocks: int = 0, width: int = 0,
                players: int = 2, size: int = 0):
    _ledger(dict(run_id=f'random-init-{Path(out).stem}-{int(time.time())}', kind='random-init',
                 hourly_usd=round(cpu_hourly(8), 2), reserved_usd=0.25, status='reserved', modal_app_id=app.app_id,
                 start_unix=time.time()), ledger)
    print(json.dumps(random_init_fn.remote(reference, out, seed, head_scale, games, steps, blocks, width, players,
                                           size), indent=1))


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=3600, volumes=VOLUMES, retries=0)
def head_check_fn(policy: str, head: str, references: dict, games: int, seed_base: int, max_ticks: int):
    """A fog-of-war value head trained onto `policy` (critic_warmup past the run's end, no omniscient critic):
    1) which parameters differ between `policy` and `head` (only critic_hidden / value may); 2) how well each value
    head ('head' plus `references`, name -> checkpoint) predicts the acting seat's result along `policy`'s sampled
    self-play games on proposed-v1 maps (decided games only): AUC of value for wins vs losses and sign accuracy,
    overall and by game turn."""
    import jax
    import numpy as np
    from polytopia_bot.rl import network
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.rl.distributed import checkpoint_template
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.maps.stream import MapSource
    retained.reload()
    results.reload()
    load = lambda p: load_checkpoint(p, checkpoint_template(p))
    base, trained = load(policy), load(head)
    changed = {}
    for (path, x), (_, y) in zip(jax.tree_util.tree_flatten_with_path(base)[0],
                                 jax.tree_util.tree_flatten_with_path(trained)[0]):
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            changed[jax.tree_util.keystr(path)] = float(np.abs(np.asarray(x) - np.asarray(y)).max())
    heads = {'head': trained, **{name: load(p) for name, p in references.items()}}
    value = jax.jit(jax.vmap(lambda p, x: network.trunk(p, x)[2], in_axes=(None, 0)))
    act = jax.jit(jax.vmap(lambda x, m, k: network.sample(base, x, m, k, size=14)[0], in_axes=(0, 0, 0)))
    engine, source = RustGames(14, games, map_batch=True), MapSource('proposed-v1', 14)
    states, prepared = engine.reset(source.batch(range(seed_base, seed_base + games))[0])
    lanes, seats, turns, values = [], [], [], {n: [] for n in heads}
    outcome, finished = np.zeros((games, 2), np.float32), np.zeros(games, bool)
    key = jax.random.key(seed_base)
    for t in range(max_ticks):
        live = np.nonzero(~finished)[0]
        if not len(live):
            break
        for n, p in heads.items():
            values[n].append(np.asarray(value(p, prepared.features))[live])
        lanes.append(live)
        seats.append(np.asarray(states.current_player)[live])
        turns.append(np.asarray(states.turn)[live])
        key, sub = jax.random.split(key)
        choice = act(prepared.features, prepared.masks, jax.random.split(sub, games))
        tick, status = engine.tick(states, choice, source.batch(range(seed_base + games, seed_base + 2 * games))[0])
        assert not np.any(np.asarray(status)[live]), 'illegal action'
        done, truncated = np.asarray(tick.done), np.asarray(tick.truncated)
        for i in np.nonzero(done & ~finished)[0]:
            finished[i] = True
            outcome[i] = 0. if truncated[i] else np.asarray(tick.rewards)[i]
        states, prepared = tick.states, tick.prepared
    lanes, seats, turns = np.concatenate(lanes), np.concatenate(seats), np.concatenate(turns)
    result = outcome[lanes, seats - 1]
    keep = finished[lanes] & (result != 0)
    def score(v, mask):
        win, loss = v[mask & (result > 0)], v[mask & (result < 0)]
        if not len(win) or not len(loss):
            return None
        order = np.argsort(np.concatenate([win, loss]), kind='stable')
        ranks = np.empty(len(order))
        ranks[order] = np.arange(1, len(order) + 1)
        auc = (ranks[:len(win)].sum() - len(win) * (len(win) + 1) / 2) / (len(win) * len(loss))
        return dict(auc=round(float(auc), 4), sign_accuracy=round(float(np.mean(np.sign(v[mask]) == result[mask])), 4),
                    mean_win=round(float(win.mean()), 4), mean_loss=round(float(loss.mean()), 4), n=int(mask.sum()))
    phases = dict(all=keep, turns_0_9=keep & (turns < 10), turns_10_19=keep & (turns >= 10) & (turns < 20),
                  turns_20_plus=keep & (turns >= 20))
    return dict(policy=policy, head=head, changed_parameters=changed,
                policy_unchanged=all(('critic_hidden' in k or "'value'" in k) for k in changed),
                games=games, finished=int(finished.sum()), decided=int(np.sum(finished & (outcome[:, 0] != 0))),
                positions=int(keep.sum()),
                heads={n: {ph: score(np.concatenate(v), m) for ph, m in phases.items()} for n, v in values.items()})


@app.local_entrypoint()
def head_check(policy: str, head: str, references_json: str = '{}', games: int = 32, seed_base: int = 397000,
               max_ticks: int = 4000, ledger: str = LEDGER):
    _ledger(dict(run_id=f'head-check-{Path(head).parent.name}-{int(time.time())}', kind='head_check',
                 hourly_usd=round(cpu_hourly(8), 2), reserved_usd=0.5, status='reserved', modal_app_id=app.app_id,
                 start_unix=time.time()), ledger)
    print(json.dumps(head_check_fn.remote(policy, head, json.loads(references_json), games, seed_base, max_ticks),
                     indent=1))


@app.local_entrypoint()
def final_eval(tag: str, candidate: str, grown: str = GROWN, parent14: str = C3, maps: int = 128, type_maps: int = 24,
               minutes: int = 60, ledger: str = LEDGER_18):
    """Untouched seed blocks: candidate vs grown-c3 (18x18 mix and per type), vs c3 (14x14 mix), and the
    candidate's sampled self-play per type (turn cap, techs, units)."""
    hourly = 23 * cpu_hourly(32)  # 3 mix arenas + 20 per-type arenas
    _ledger(dict(run_id=tag, kind='final-eval', candidate=candidate, hourly_usd=round(hourly, 2),
                 reserved_usd=round(hourly * minutes / 60, 2), status='reserved', modal_app_id=app.app_id,
                 start_unix=time.time()), ledger)
    out = final_eval_fn.remote(tag, candidate, grown, parent14, maps, type_maps)
    (ROOT / ledger / f'{tag}-final.json').write_text(json.dumps(out, indent=1) + '\n')
    for k in ('mix18', 'mix18_sampled', 'mix14'):
        r = out[k]
        print(k, {x: r.get(x) for x in ('wins', 'losses', 'draws_or_truncations', 'win_rate', 'ci95', 'mean_turns')})
    print(f'saved {ROOT / ledger / (tag + "-final.json")}')


@app.function(image=cpu_image, cpu=1, memory=2048, timeout=1800, volumes=VOLUMES, retries=0)
def archive_fn(src: str, dst_dir: str, run_id: str):
    import shutil
    results.reload()
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst / 'final.npz')
    shutil.copy(Path(src).with_suffix('.json'), dst / 'final.json')
    critic = Path(src).with_name(Path(src).stem + '-critic.npz')   # an omniscient critic, saved beside the policy
    if critic.exists():
        shutil.copy(critic, dst / 'final-critic.npz')
        shutil.copy(critic.with_suffix('.json'), dst / 'final-critic.json')
    run = Path('/results/distributed') / run_id
    for name in ('protocol.json', 'training.json', 'training.jsonl'):
        if (run / name).exists():
            shutil.copy(run / name, dst / name)
    retained.commit()
    return sorted(p.name for p in dst.iterdir())


@app.local_entrypoint()
def archive(src: str, name: str, run_id: str):
    """Copy a checkpoint (+ its run's protocol/training logs) to /retained/checkpoints/distributed-final-v1/<name>/."""
    print(archive_fn.remote(src, f'/retained/checkpoints/distributed-final-v1/{name}', run_id))


@app.function(image=cpu_image, cpu=4, memory=16384, timeout=1800, volumes=VOLUMES, retries=0)
def logp_check_fn(checkpoint: str, size: int, steps: int):
    """Actor-recorded log-probs vs the learner's recomputation from the transported rows (packed view and
    path-mask bits), as distributed.Learner does; both on CPU. Nonzero gaps mean a transport bug."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import Choice, PathMasks
    from polytopia_bot.rl.cpu_engine import view_features
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.rl.distributed import checkpoint_template
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.maps.stream import MapSource
    retained.reload()
    params = load_checkpoint(checkpoint, checkpoint_template(checkpoint))
    games, source = RustGames(size, 8, map_batch=True), MapSource('proposed-v1', size)
    states, prepared = games.reset(source.batch(range(372500, 372508))[0])
    act = jax.jit(jax.vmap(lambda x, m, k: network.sample(params, x, m, k, size=size), in_axes=(0, 0, 0)))
    n = size * size
    unpack = lambda a, w: jnp.unpackbits(jnp.asarray(a), axis=-1)[..., :w].astype(bool)
    def learner_logp(view, fam, src, tgt, choice):
        x = jax.vmap(lambda v: view_features(v, games.layout, size))(view)
        masks = PathMasks(unpack(fam, 17), unpack(src, n), unpack(tgt, max(n, 128)))
        return jax.vmap(lambda f, m, c: network.evaluate(params, f, c, m)[0])(x, masks, choice)
    learner_logp = jax.jit(learner_logp)
    gaps, feature_gaps = [], []
    key = jax.random.key(0)
    for step in range(steps):
        key, k = jax.random.split(key)
        choice, path, logp, _ = act(prepared.features, prepared.masks, jax.random.split(k, 8))
        bits = [np.packbits(np.asarray(a), axis=-1) for a in path]
        again = learner_logp(prepared.view, *bits, choice)
        gaps.append(float(np.max(np.abs(np.asarray(again) - np.asarray(logp)))))
        x = jax.vmap(lambda v: view_features(v, games.layout, size))(prepared.view)
        feature_gaps.append(max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                                for a, b in zip(jax.tree.leaves(x), jax.tree.leaves(prepared.features))))
        tick, status = games.tick(states, choice, source.batch(range(372600 + 8 * step, 372608 + 8 * step))[0])
        states, prepared = tick.states, tick.prepared
    return dict(size=size, steps=steps, max_logp_gap=max(gaps), mean_logp_gap=float(np.mean(gaps)),
                max_feature_gap=max(feature_gaps))


@app.local_entrypoint()
def logp_check(checkpoint: str = GROWN, sizes: str = '14,18', steps: int = 150, ledger: str = LEDGER_18):
    _ledger(dict(run_id=f'logp-check-{int(time.time())}', kind='check', hourly_usd=round(cpu_hourly(4) * 2, 2),
                 reserved_usd=0.3, status='reserved', modal_app_id=app.app_id, start_unix=time.time()), ledger)
    calls = [logp_check_fn.spawn(checkpoint, int(s), steps) for s in sizes.split(',')]
    for c in calls:
        print(json.dumps(c.get()))


@app.function(image=cpu_image, cpu=0.25, memory=1024, timeout=10800, retries=0, volumes=VOLUMES)
def select_fn(tag: str, candidates: dict, opponent: str, seed_base: int, maps: int, size: int, map_mix: str):
    """Every candidate vs one opponent on the same development maps (paired selection)."""
    calls = {name: arena_match.spawn(f'{tag}-{name}', path, opponent, seed_base, maps, map_mix=map_mix, size=size)
             for name, path in candidates.items()}
    out = {}
    for name, c in calls.items():
        try:
            out[name] = c.get()
        except Exception as error:
            out[name] = dict(error=repr(error))
    return out


@app.local_entrypoint()
def select(tag: str, candidates_json: str, opponent: str = GROWN, seed_base: int = 370700, maps: int = 64,
           size: int = 18, map_mix: str = 'proposed-v1', minutes: int = 45, ledger: str = LEDGER_18):
    candidates = json.loads(candidates_json)
    hourly = len(candidates) * cpu_hourly(32)
    _ledger(dict(run_id=tag, kind='selection', candidates=candidates, seed_base=seed_base, maps=maps,
                 hourly_usd=round(hourly, 2), reserved_usd=round(hourly * minutes / 60, 2), status='reserved',
                 modal_app_id=app.app_id, start_unix=time.time()), ledger)
    out = select_fn.remote(tag, candidates, opponent, seed_base, maps, size, map_mix)
    (ROOT / ledger / f'{tag}.json').write_text(json.dumps(out, indent=1) + '\n')
    for name, r in out.items():
        print(name, {k: r.get(k) for k in ('wins', 'losses', 'truncations', 'win_rate', 'ci95', 'mean_turns', 'error')})


@app.function(image=cpu_image, cpu=4, memory=16384, timeout=3600, volumes=VOLUMES, retries=0)
def action_stats_fn(checkpoint: str, size: int, steps: int, games: int):
    """Legal-action counts per decision and the chosen action families, sampled policy self-play."""
    import jax
    import numpy as np
    from polytopia_bot.gpu.actions import Masks
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import shapes
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.rl.distributed import checkpoint_template
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.maps.stream import MapSource
    retained.reload()
    params = load_checkpoint(checkpoint, checkpoint_template(checkpoint))
    engine, source = RustGames(size, games, map_batch=True), MapSource('proposed-v1', size)
    states, prepared = engine.reset(source.batch(range(372700, 372700 + games))[0])
    act = jax.jit(jax.vmap(lambda x, m, k: network.sample(params, x, m, k, size=size), in_axes=(0, 0, 0)))
    legal, chosen, fam_legal = [], np.zeros(17, int), np.zeros(17)
    key = jax.random.key(0)
    for step in range(steps):
        key, k = jax.random.split(key)
        choice = act(prepared.features, prepared.masks, jax.random.split(k, games))[0]
        m = [np.asarray(a).reshape(games, -1) for a in prepared.masks]
        counts = np.stack([a.sum(1) for a in m], 1)        # [games, 17]
        legal.extend(counts.sum(1).tolist())
        fam_legal += counts.sum(0)
        np.add.at(chosen, np.asarray(choice.family), 1)
        tick, _ = engine.tick(states, choice, source.batch(range(373900 + games * step, 373900 + games * (step + 1)))[0])
        states, prepared = tick.states, tick.prepared
    legal = np.array(legal)
    names = Masks._fields
    return dict(size=size, decisions=len(legal), catalog=int(sum(a * b for a, b in shapes(size))),
                legal_median=float(np.median(legal)), legal_mean=float(legal.mean()), legal_p90=float(np.percentile(legal, 90)),
                legal_max=int(legal.max()), mean_legal_by_family={n: round(float(v) / len(legal), 1) for n, v in zip(names, fam_legal) if v},
                chosen_share={n: round(float(c) / float(chosen.sum()), 3) for n, c in zip(names, chosen) if c})


@app.local_entrypoint()
def action_stats(checkpoint: str = GROWN, sizes: str = '14,18', steps: int = 1500, games: int = 16,
                 ledger: str = LEDGER_18):
    _ledger(dict(run_id=f'action-stats-{int(time.time())}', kind='check', hourly_usd=round(cpu_hourly(4) * 2, 2),
                 reserved_usd=0.5, status='reserved', modal_app_id=app.app_id, start_unix=time.time()), ledger)
    for c in [action_stats_fn.spawn(checkpoint, int(s), steps, games) for s in sizes.split(',')]:
        print(json.dumps(c.get()))


# ----------------------------------------------------------------------------- four seats (docs/four-player-2026-10-24.md)

LEDGER_4 = 'runs/four-player-v1'
CHAMPION = '/results/distributed/scratch-cont-2/final.npz'
FOUR_PARENTS = '/results/four-player/parents'


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=1800, volumes=VOLUMES, retries=0)
def four_convert_fn(parent: str, name: str, size: int, extra_blocks: int, global_pool: bool, seed: int, positions: int):
    """Two-seat checkpoint (+ its <stem>-critic.npz) -> FOUR_PARENTS/<name>/final.npz (+ final-critic.npz): widened
    to four-seat features, grown to size x size with extra_blocks identity blocks. Verified on four-seat positions:
    the policy trunk, the deterministic choices and the critic value equal the parent's on the two-seat part of
    the features."""
    import jax
    import numpy as np
    from polytopia_bot.maps.stream import MapSource
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import Choice
    from polytopia_bot.rl.data import sha256
    from polytopia_bot.rl.distributed import _critic_template, checkpoint_template, critic_features, network_spec
    from polytopia_bot.rl.four_player import convert, old_part
    from polytopia_bot.rl.runner import Config, load_checkpoint, save_checkpoint, write_json
    from polytopia_bot.rl.rust_engine import RustGames
    results.reload()
    retained.reload()
    spec = network_spec(parent)
    policy = load_checkpoint(parent, checkpoint_template(parent))
    critic_path = Path(parent).with_name(Path(parent).stem + '-critic.npz')
    critic = load_checkpoint(critic_path, _critic_template(**spec)) if critic_path.exists() else None
    wide, wide_critic = convert(policy, critic, size=size, extra_blocks=extra_blocks, global_pool=global_pool, seed=seed)
    meta = json.loads(Path(parent).with_suffix('.json').read_text())
    config = Config(**{**meta['config'], 'size': size, 'extra_blocks': extra_blocks, 'global_pool': global_pool,
                       'players': 4})
    out = Path(FOUR_PARENTS) / name
    out.mkdir(parents=True, exist_ok=True)
    stage = f'four-seat-from-{meta["stage"]}'
    save_checkpoint(out / 'final.npz', wide, config, stage=stage)
    if wide_critic is not None:
        save_checkpoint(out / 'final-critic.npz', wide_critic, config, stage=stage)
    loaded = load_checkpoint(out / 'final.npz', checkpoint_template(out / 'final.npz'))
    loaded_critic = (load_checkpoint(out / 'final-critic.npz', _critic_template(**network_spec(out / 'final.npz')))
                     if wide_critic is not None else None)
    games = RustGames(size, 8, players=4, map_batch=True, both_views=True)
    source = MapSource('four-v1', size, players=4)
    states, prepared = games.reset(source.batch(range(380000, 380008))[0])
    det = jax.jit(jax.vmap(lambda p, x, m, k: network.sample(p, x, m, k, size=size, deterministic=True),
                           in_axes=(None, 0, 0, 0)))
    sto = jax.jit(jax.vmap(lambda p, x, m, k: network.sample(p, x, m, k, size=size), in_axes=(None, 0, 0, 0)))
    cval = jax.jit(jax.vmap(lambda c, x: network.trunk(c, x)[2], in_axes=(None, 0)))
    key = jax.random.key(seed)
    worst, same_choices, checks = dict(policy=0., critic=0.), True, 0
    for step in range(positions):
        keys = jax.random.split(jax.random.fold_in(key, step), 8)
        a = det(loaded, prepared.features, prepared.masks, keys)
        b = det(policy, old_part(prepared.features), prepared.masks, keys)
        same_choices &= all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a[0], b[0]))
        worst['policy'] = max(worst['policy'], float(np.max(np.abs(np.asarray(a[2]) - np.asarray(b[2])))))
        if loaded_critic is not None:
            full = jax.vmap(lambda f, o: critic_features(f, o, games.layout, size))(prepared.features, prepared.opp_view)
            worst['critic'] = max(worst['critic'], float(np.max(np.abs(
                np.asarray(cval(loaded_critic, full)) - np.asarray(cval(critic, old_part(full, groups=2)))))))
        checks += 1
        choice = sto(loaded, prepared.features, prepared.masks, keys)[0]
        tick, status = games.tick(states, choice, source.batch(range(380100 + 8 * step, 380108 + 8 * step))[0])
        assert not np.any(np.asarray(status)), 'illegal action'
        states, prepared = tick.states, tick.prepared
    report = dict(parent=parent, parent_sha256=sha256(parent), critic=str(critic_path) if critic is not None else None,
                  out=str(out / 'final.npz'), out_sha256=sha256(out / 'final.npz'), spec=network_spec(out / 'final.npz'),
                  positions=checks, same_deterministic_choices=bool(same_choices), max_logp_diff=worst['policy'],
                  max_critic_value_diff=worst['critic'], exact=bool(same_choices and worst['policy'] < 1e-4
                                                                   and worst['critic'] < 1e-4))
    write_json(out / 'convert-report.json', report)
    results.commit()
    return report


@app.local_entrypoint()
def four_convert(name: str, parent: str = CHAMPION, size: int = 18, extra_blocks: int = 0, global_pool: bool = False,
                 seed: int = 18, positions: int = 60, ledger: str = LEDGER_4):
    _ledger(dict(run_id=f'convert-{name}', kind='convert', hourly_usd=round(cpu_hourly(8), 2), reserved_usd=0.3,
                 status='reserved', modal_app_id=app.app_id, start_unix=time.time()), ledger)
    print(json.dumps(four_convert_fn.remote(parent, name, size, extra_blocks, global_pool, seed, positions), indent=1))


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=14400, volumes=VOLUMES, retries=1)
def arena4_part(entrants: dict, tables: list, seeds: list, size: int, maps: str, deterministic: bool, lanes: int,
                seed: int, max_turns: int = None):
    """One container's share of an arena4 run: seat-rotated tables on `seeds`; returns the game records."""
    import jax
    from polytopia_bot.rl.arena4 import play
    from polytopia_bot.rl.distributed import checkpoint_template
    from polytopia_bot.rl.runner import load_checkpoint
    jax.config.update('jax_compilation_cache_dir', '/results/.jax-compilation-cache')
    results.reload()
    retained.reload()
    policies = {name: load_checkpoint(path, checkpoint_template(path)) for name, path in entrants.items()}
    return play(policies, tables, size=size, maps=maps, seeds=seeds, lanes=lanes, deterministic=deterministic,
                seed=seed, max_turns=max_turns)


@app.function(image=cpu_image, cpu=1, memory=4096, timeout=600, retries=0)
def arena4_summary(records: list, anchor: str = None):
    from polytopia_bot.rl.arena4 import summary
    return summary(records, anchor=anchor)


@app.local_entrypoint()
def arena4(tag: str, entrants: str, tables: str, seed_base: int = 381000, maps: int = 32, parts: int = 8,
           size: int = 18, map_mix: str = 'four-v1', sampled: bool = False, anchor: str = '', lanes: int = 16,
           minutes: int = 30, max_turns: int = 0, ledger: str = LEDGER_4):
    """Four-seat tables. --entrants 'a=/path/a.npz,b=/path/b.npz'; --tables 'a,a,b,b;a,b,b,b' (each table is
    played on `maps` / len(tables) seeds x 4 seat rotations). Writes <ledger>/<tag>.json (summary + game records)
    and prints the summary (rank rewards, pairwise scores, Bradley-Terry Elo)."""
    named = dict(item.split('=', 1) for item in entrants.split(','))
    table_list = [t.split(',') for t in tables.split(';')]
    for t in table_list:
        if len(t) != 4 or any(n not in named for n in t):
            raise SystemExit(f'bad table {t}')
    hourly = parts * cpu_hourly(2)
    _ledger(dict(run_id=tag, kind='arena4', entrants=named, tables=table_list, seed_base=seed_base, maps=maps,
                 size=size, map_mix=map_mix, sampled=sampled, hourly_usd=round(hourly, 2),
                 reserved_usd=round(hourly * minutes / 60, 2), status='reserved', modal_app_id=app.app_id,
                 start_unix=time.time()), ledger)
    seeds = list(range(seed_base, seed_base + maps))
    started = time.time()
    # table k of the full list goes with seed index i where i % len(tables) == k; keep that pairing per part
    calls = []
    for p in range(parts):
        mine = seeds[p::parts]
        part_tables = [table_list[(p + parts * j) % len(table_list)] for j in range(len(mine))]
        if mine:
            calls.append(arena4_part.spawn(named, part_tables, mine, size, map_mix, not sampled, lanes, 1000 + p,
                                           max_turns or None))
    records = []
    for c in calls:
        records.extend(c.get())
    out = arena4_summary.remote(records, anchor or None)
    out.update(tag=tag, paths=named, tables=table_list, seed_base=seed_base, maps=maps, size=size, map_mix=map_mix,
               sampled=sampled, seconds=round(time.time() - started, 1))
    path = ROOT / ledger / f'{tag}.json'
    path.write_text(json.dumps(dict(out, records=records), indent=1) + '\n')
    print(json.dumps(out, indent=1))
    print(f'saved {path}')


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=3600, volumes=VOLUMES, retries=0)
def four_trace_fn(policy: str, seeds: list, rounds: int, size: int = 18, maps: str = 'four-v1', sampled: bool = True,
                  zero_seat: bool = False):
    """Per-seat economy at the start of each seat's turn (stars, cities, units, techs, score, explored tiles) for
    four-seat games of one policy in every seat: a check that the seats are treated alike."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from polytopia_bot.maps.stream import MapSource
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import Choice
    from polytopia_bot.rl.distributed import checkpoint_template
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.rl.rust_engine import RustGames
    results.reload()
    params = load_checkpoint(policy, checkpoint_template(policy))
    g = RustGames(size, len(seeds), players=4, map_batch=True)
    source = MapSource(maps, size, players=4)
    states, prepared = g.reset(source.batch(seeds)[0])
    act = jax.jit(jax.vmap(lambda x, m, k: network.sample(params, x, m, k, size=size, deterministic=not sampled)))
    key = jax.random.key(0)
    trace = {}   # (round, seat) -> list of per-game rows
    families = {}   # seat -> chosen / legal family counts from round 6 on
    alive = np.ones(len(seeds), bool)
    last = np.zeros(len(seeds), np.int32) - 1
    for _ in range(200000):
        e = g.batch.export()
        cp, turn = np.asarray(e['current_player']), np.asarray(e['turn'])
        for i in range(len(seeds)):
            if alive[i] and cp[i] != last[i]:
                last[i] = cp[i]
                k = cp[i] - 1
                row = dict(stars=int(e['players.stars'][i, k]), score=int(e['players.score'][i, k]),
                           cities=int((e['cities.owner'][i] == cp[i]).sum()), units=int((e['units.owner'][i] == cp[i]).sum()),
                           techs=int(e['players.tech'][i, k].sum()), explored=int(e['tiles.explored'][i, k].sum()),
                           territory=int((e['tiles.owner'][i] == cp[i]).sum()))
                trace.setdefault(f'{int(turn[i])}/{int(cp[i])}', []).append(row)
        if (turn[alive] > rounds).all() if alive.any() else True:
            break
        key, k1 = jax.random.split(key)
        features = prepared.features
        if zero_seat:   # diagnostic: every seat looks like the first mover to the network
            features = features._replace(global_state=jnp.asarray(features.global_state).at[:, 2].set(0.))
        choice = act(features, prepared.masks, jax.random.split(k1, len(seeds)))[0]
        fam = np.asarray(choice[0])
        legal = np.stack([np.asarray(m).reshape(len(seeds), -1).any(1) for m in prepared.masks], 1)   # [G, 17]
        for i in range(len(seeds)):
            if alive[i] and turn[i] >= 6:
                h = families.setdefault(int(cp[i]), dict(chosen=np.zeros(17, int), legal=np.zeros(17, int)))
                h['chosen'][fam[i]] += 1
                h['legal'] += legal[i]
        tick, status = g.tick(states, choice, source.batch(seeds)[0])
        alive &= ~np.asarray(tick.done)
        states, prepared = tick.states, tick.prepared
    out = {k: {f: float(np.mean([r[f] for r in v])) for f in v[0]} | dict(n=len(v)) for k, v in trace.items()}
    out['families'] = {seat: {k: v.tolist() for k, v in h.items()} for seat, h in families.items()}
    return out


@app.local_entrypoint()
def four_trace(policy: str = '/results/four-player/parents/champ-ffa4-2x64/final.npz', games: int = 16, rounds: int = 25,
               seed_base: int = 383000, zero_seat: bool = False, ledger: str = LEDGER_4):
    _ledger(dict(run_id=f'trace-{int(time.time())}', kind='check', hourly_usd=round(cpu_hourly(2), 2), reserved_usd=0.3,
                 status='reserved', modal_app_id=app.app_id, start_unix=time.time()), ledger)
    out = four_trace_fn.remote(policy, list(range(seed_base, seed_base + games)), rounds, zero_seat=zero_seat)
    families = out.pop('families')
    names = ('move attack summon build research harvest forest reward capture recover promote disband heal destroy '
             'diplomacy end resign').split()
    for seat in sorted(families):
        h = families[seat]
        print(f'seat {seat} chosen', dict((n, c) for n, c in zip(names, h['chosen']) if c))
        print(f'seat {seat} legal ', dict((n, c) for n, c in zip(names, h['legal']) if c))
    keys = sorted(out, key=lambda k: (int(k.split('/')[0]), int(k.split('/')[1])))
    for k in keys:
        r = out[k]
        print(k, ' '.join(f'{f}={r[f]:.1f}' for f in ('stars', 'cities', 'units', 'techs', 'territory', 'explored', 'score', 'n')))


FOUR_MONITOR = '/results/four-player/monitor'


def _four_training_window(root, since_update=0):
    """Training-log summary since `since_update`: last update/samples/elapsed, means of the PPO stats and the actors'
    four-seat records (mean rank reward vs each frozen pool opponent, game length, cap rate)."""
    rows = [json.loads(l) for l in (root / 'training.jsonl').read_text().splitlines() if l.strip()] \
        if (root / 'training.jsonl').exists() else []
    if not rows:
        return None
    window = [r for r in rows if r['update'] > since_update] or rows[-1:]
    mean = lambda k: sum(r.get(k, 0) or 0 for r in window) / len(window)
    vs, fin, turns, caps = {}, 0, 0, 0
    for r in window:
        a = r.get('actors', {})
        for name, (total, n) in a.get('vs_reward', {}).items():
            cell = vs.setdefault(name, [0., 0])
            cell[0] += total
            cell[1] += n
        fin += a.get('finished', 0) or 0
        turns += a.get('turns', 0) or 0
        caps += sum(t for _, t in a.get('maps', {}).values())
    last = rows[-1]
    return dict(update=last['update'], samples=last.get('samples'), elapsed_h=round(last['elapsed'] / 3600, 2),
                window_updates=[window[0]['update'], last['update']],
                samples_per_second=round(mean('received_samples_per_second')), entropy=round(mean('entropy'), 3),
                approx_kl=round(mean('approx_kl'), 4), raw_advantage_std=round(mean('raw_advantage_std'), 4),
                value_std=round(mean('value_std'), 3), finished_games=fin, mean_rounds=round(turns / max(fin, 1), 1),
                cap_rate=round(caps / max(fin, 1), 3),
                vs_pool={k: dict(mean_reward=round(t / n, 3), games=n) for k, (t, n) in vs.items() if n})


def _four_status_md(tag, run_id, history, done):
    lines = [f'# {run_id} monitor ({tag})', '',
             f'Updated {time.strftime("%Y-%m-%d %H:%M %Z")}. ' + ('Run finished.' if done else 'Run in progress.'), '',
             '| Round | Stage | Hours | Samples | Samples/s | Entropy | Rounds/game | Cap | Elo vs START | Elo vs CHAMP | '
             'CUR mean reward | Train vs pool |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for h in history:
        t, a = h.get('training') or {}, h.get('arena') or {}
        rat = a.get('ratings', {})
        cur = rat.get('CUR')
        elo = lambda ref: f"{cur - rat[ref]:+.0f}" if cur is not None and ref in rat else '-'
        ent = a.get('entrants', {}).get('CUR', {})
        pool = ', '.join(f"{k} {v['mean_reward']:+.2f} ({v['games']})" for k, v in t.get('vs_pool', {}).items())
        lines.append(f"| {h['round']} | {h.get('stage', '-')} | {t.get('elapsed_h', '-')} | "
                     f"{(t.get('samples') or 0) / 1e6:.0f}M | {t.get('samples_per_second', '-')} | {t.get('entropy', '-')} | "
                     f"{t.get('mean_rounds', '-')} | {t.get('cap_rate', '-')} | {elo('START')} | {elo('CHAMP')} | "
                     f"{ent.get('mean_reward', float('nan')):+.3f} | {pool} |" + (f" error: {h['error'][:120]}" if h.get('error') else ''))
    lines += ['', 'Elo: Bradley-Terry from pairwise results inside each seat-rotated four-seat game (CUR = the frozen '
              'checkpoint of that round, START = the run\'s parent, CHAMP = the converted 1v1 champion). Train vs pool: '
              'actors\' mean rank reward of the learner seat against 3 seats of a frozen pool opponent.']
    return '\n'.join(lines) + '\n'


@app.function(image=cpu_image, cpu=1, memory=4096, timeout=86400, retries=3, volumes=VOLUMES)
def four_monitor(tag: str, run_id: str, references: dict, interval: int, max_rounds: int, seed_base: int, maps: int,
                 parts: int, final_maps: int, first_delay: int = None):
    """Unattended monitor of a four-seat run. Every `interval` s: freeze the run's latest.npz, play seat-rotated
    tables of it (CUR) against the references (name -> path; START and CHAMP), summarize the training log, and
    write registry `{tag}/history`, FOUR_MONITOR/<tag>-history.json and FOUR_MONITOR/<tag>-status.md. Progress
    lives in the registry, so a restarted container (Modal retry/preemption) resumes at the next round. Once the
    learner has stopped it evaluates final.npz on `final_maps` maps and exits. Every arena call has a timeout."""
    import shutil
    from polytopia_bot.rl.arena4 import summary
    from polytopia_bot.rl.runner import write_json
    root = Path('/results/distributed') / run_id
    out = Path(FOUR_MONITOR)
    names = list(references)
    tables = [['CUR'] + [n] * 3 for n in names] + [['CUR', 'CUR'] + names[:2]] if len(names) >= 2 else \
        [['CUR'] + [n] * 3 for n in names]
    def evaluate(path, round_index, n_maps):
        entrants = dict(CUR=str(path), **references)
        seeds = list(range(seed_base + 1000 * round_index, seed_base + 1000 * round_index + n_maps))
        calls = []
        for p in range(parts):
            mine = seeds[p::parts]
            if mine:
                part_tables = [tables[(p + parts * j) % len(tables)] for j in range(len(mine))]
                calls.append(arena4_part.spawn(entrants, part_tables, mine, 18, 'four-v1', False, 16, 7000 + p))
        records, errors = [], []
        for c in calls:
            try:
                records.extend(c.get(timeout=5400))
            except Exception as error:
                errors.append(repr(error)[:300])
        return (summary(records, anchor='START' if 'START' in references else None) if records else None), errors
    if registry.get(f'{tag}/state') is None:   # persist the clock once; a restarted container keeps it
        registry[f'{tag}/state'] = dict(round=0, last_update=0, started=time.time(),
                                        next_due=time.time() + (interval if first_delay is None else first_delay))
    while True:
        state = registry.get(f'{tag}/state')
        history = registry.get(f'{tag}/history', [])
        run = registry.get(run_id, {}) or {}
        stopped = 'stopped' in run or (root / 'final.npz').exists()
        if state['round'] >= max_rounds and not stopped:
            stopped = True   # safety: stop evaluating snapshots after max_rounds even if the run lingers
        if not stopped:
            due = state['next_due']
            if time.time() < due:
                time.sleep(min(60, due - time.time()))
                continue
        results.reload()
        final = stopped and (root / 'final.npz').exists()
        src = root / ('final.npz' if final else 'latest.npz')
        row = dict(round=state['round'], final=final, finished=None)
        try:
            meta = json.loads(src.with_suffix('.json').read_text())
            frozen = out / run_id / f'{meta["stage"]}{"-final" if final else ""}.npz'
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, frozen)
            shutil.copy(src.with_suffix('.json'), frozen.with_suffix('.json'))
            results.commit()
            row.update(stage=meta['stage'], checkpoint=str(frozen),
                       training=_four_training_window(root, state['last_update']))
            arena, errors = evaluate(frozen, state['round'], final_maps if final else maps)
            row.update(arena={k: v for k, v in (arena or {}).items() if k != 'pairwise'} if arena else None,
                       pairwise=(arena or {}).get('pairwise'), arena_errors=errors)
            if row['training']:
                state['last_update'] = row['training']['update']
        except Exception as error:
            row['error'] = repr(error)[:500]
        row['finished'] = time.time()
        history = history + [row]
        state.update(round=state['round'] + 1, next_due=time.time() + interval)
        registry[f'{tag}/history'] = history
        registry[f'{tag}/state'] = state
        write_json(out / f'{tag}-history.json', history)
        (out / f'{tag}-status.md').write_text(_four_status_md(tag, run_id, history, final or stopped))
        results.commit()
        if stopped:
            registry[f'{tag}/state'] = dict(state, done=time.time())
            return history


@app.local_entrypoint()
def start_four_monitor(tag: str, run_id: str, references_json: str, interval: int = 7200, max_rounds: int = 8,
                       seed_base: int = 390000, maps: int = 100, parts: int = 20, final_maps: int = 200,
                       first_delay: int = -1, ledger: str = LEDGER_4):
    """Detach this (modal run --detach) so the monitor outlives the laptop."""
    rounds_cost = (max_rounds + 1) * parts * cpu_hourly(2) * .1   # ~6 min of `parts` 2-core containers per round
    _ledger(dict(run_id=tag, kind='monitor', run=run_id, references=json.loads(references_json), interval=interval,
                 hourly_usd=0, reserved_usd=round(rounds_cost + 1, 2), status='reserved', modal_app_id=app.app_id,
                 start_unix=time.time()), ledger)
    for key in (f'{tag}/state', f'{tag}/history'):
        try:
            registry.pop(key)
        except KeyError:
            pass
    call = four_monitor.spawn(tag, run_id, json.loads(references_json), interval, max_rounds, seed_base, maps, parts,
                              final_maps, None if first_delay < 0 else first_delay)
    print(json.dumps(dict(monitor_call=call.object_id, app_id=app.app_id, tag=tag)))


@app.function(image=cpu_image, cpu=0.25, memory=512, timeout=21600, retries=2)
def stop_run_at(run_id: str, at_unix: float, cancel_learner_after: int = 180):
    """Graceful early stop: at `at_unix` mark the run stopped in the registry (actors exit at their next report;
    four_monitors run their final tables on latest.npz), then cancel the learner call, which would otherwise idle
    until its training_seconds deadline. latest.npz (every checkpoint_every updates) is the run's last state."""
    while time.time() < at_unix:
        time.sleep(min(30, at_unix - time.time()))
    entry = dict(registry.get(run_id) or {})
    entry.setdefault('stopped', time.time())
    registry[run_id] = entry
    time.sleep(cancel_learner_after)
    calls = registry.get(f'{run_id}/calls') or {}
    if calls.get('learner'):
        try:
            modal.FunctionCall.from_id(calls['learner']).cancel(terminate_containers=True)
        except Exception as error:
            return dict(stopped=entry['stopped'], cancel_error=repr(error))
    return dict(stopped=entry['stopped'], learner_cancelled=calls.get('learner'))


def _keeper_missing(reports, indexes, now, stale, born, grace):
    """Actor indexes past their boot grace (`born[i]` + grace) whose last registry report is older than `stale` s
    (or absent)."""
    return [i for i in indexes if now >= born[i] + grace and (reports.get(i) or {}).get('at', 0) < now - stale]


@app.function(image=cpu_image, cpu=0.25, memory=512, timeout=86400, retries=2)
def fleet_keeper(run_id: str, desired: int, until_unix: float, actor_cpu: int, processes: int, games: int,
                 memory_gib: int, actor_gpu: str = '', serve_options: dict = None, max_replacements: int = 4,
                 interval: int = 300, stale: int = 300, grace: int = 900, dry_run: bool = False):
    """Unattended actor fleet upkeep: every `interval` s, any of the run's actor indexes silent for `stale` s
    (actors report every 20 s) is replaced by a new container at a fresh index, at most `max_replacements` in total.
    Nothing is replaced while no actor is live (the learner is down or the run is over), within `grace` s of an
    index's start (the learner registering, or its spawn), after the run is marked stopped, or after `until_unix`. Replacements get a timeout that
    ends at `until_unix` + 20 min. State lives in registry `{run_id}/keeper`, so a restarted container resumes."""
    key = f'{run_id}/keeper'
    while True:
        now = time.time()
        entry = registry.get(run_id)
        if now > until_unix or (entry and 'stopped' in entry):
            state = registry.get(key) or {}
            registry[key] = dict(state, done=now)
            return state
        if not entry:   # learner not registered yet
            time.sleep(30)
            continue
        state = registry.get(key) or dict(indexes=list(range(desired)), next_index=desired, replaced=[],
                                          retired=[], born={i: now for i in range(desired)}, log=[])
        reports = {i: registry.get(f'{run_id}/actor/{i}') for i in state['indexes']}
        missing = _keeper_missing(reports, state['indexes'], now, stale, state['born'], grace)
        live = len(state['indexes']) - len(missing)
        budget = max_replacements - len(state['replaced'])
        spawn = min(len(missing), budget) if live > 0 else 0
        row = dict(at=now, live=live, missing=missing, spawned=[])
        if spawn:
            first = state['next_index']
            hard = int(until_unix + 1200 - now)
            if not dry_run:
                _spawn_actors(run_id, first, spawn, actor_cpu, processes, games, hard, memory_gib, actor_gpu,
                              serve_options)
            row['spawned'] = list(range(first, first + spawn))
            state['retired'] += missing[:spawn]
            state['indexes'] = [i for i in state['indexes'] if i not in missing[:spawn]] + row['spawned']
            state['replaced'] += row['spawned']
            state['next_index'] = first + spawn
            state['born'].update({i: now for i in row['spawned']})
        state['log'] = (state['log'] + [row])[-50:]
        print(json.dumps(dict(stage='keeper', run_id=run_id, **row)), flush=True)
        registry[key] = state
        time.sleep(interval)


@app.local_entrypoint()
def start_fleet_keeper(run_id: str, desired: int, until_unix: float, actor_cpu: int = 16, processes: int = 0,
                       games: int = 24, memory_gib: int = 0, actor_gpu: str = '', serve_options_json: str = '{}',
                       max_replacements: int = 4, ledger: str = LEDGER):
    """Detach this (modal run --detach). Match the launch's actor options."""
    memory_gib = actor_memory_gib(actor_cpu, processes, memory_gib) if not actor_gpu else (memory_gib or 6 * actor_cpu)
    each = actor_cpu * CPU_CORE_RATE + memory_gib * MEM_GIB_RATE + (GPU_RATE.get(actor_gpu, 1.10) if actor_gpu else 0.)
    _ledger(dict(run_id=f'{run_id}-keeper', kind='keeper', parent=run_id, hourly_usd=0,
                 reserved_usd=round(max_replacements * each * max(0, until_unix - time.time() + 1200) / 3600, 2),
                 status='reserved', modal_app_id=app.app_id, start_unix=time.time()), ledger)
    try:
        registry.pop(f'{run_id}/keeper')
    except KeyError:
        pass
    call = fleet_keeper.spawn(run_id, desired, until_unix, actor_cpu, processes, games, memory_gib, actor_gpu,
                              json.loads(serve_options_json), max_replacements)
    print(json.dumps(dict(keeper_call=call.object_id, app_id=app.app_id, run_id=run_id, desired=desired,
                          until=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(until_unix)))))


@app.local_entrypoint()
def keeper_check(run_id: str = 'keeper-check'):
    """Dry-run fleet_keeper on a fake run: actors 0-1 report, 2-4 are silent, max 2 replacements. Expect one round
    replacing 2 and 3 with indexes 5 and 6 (the cap holds: 4 stays missing, and so do 5 and 6, which never
    report), then no more; a fully silent fleet spawns nothing; the keeper exits once the run is marked stopped."""
    for k in (run_id, f'{run_id}/keeper', f'{run_id}-dead', f'{run_id}-dead/keeper'):
        try:
            registry.pop(k)
        except KeyError:
            pass
    registry[run_id] = dict(host='x', port=0)
    now = time.time()
    for i in (0, 1):
        registry[f'{run_id}/actor/{i}'] = dict(at=now + 3600)   # "fresh" for the whole check
    for i in (2, 3, 4):
        registry[f'{run_id}/actor/{i}'] = dict(at=now - 3600)
    call = fleet_keeper.spawn(run_id, 5, now + 600, 1, 1, 1, 1, max_replacements=2, interval=5, stale=300, grace=0,
                              dry_run=True)
    time.sleep(40)
    registry[run_id] = dict(host='x', port=0, stopped=time.time())
    final = call.get(timeout=120)
    rounds = [r for r in final['log'] if r['spawned']]
    ok = (len(rounds) == 1 and rounds[0]['spawned'] == [5, 6] and sorted(final['retired']) == [2, 3]
          and final['replaced'] == [5, 6] and len(final['log']) >= 3
          and all(r['missing'] == [4, 5, 6] for r in final['log'] if not r['spawned']))
    registry[f'{run_id}-dead'] = dict(host='x', port=0)   # nobody live -> nothing spawned
    dead = fleet_keeper.spawn(f'{run_id}-dead', 3, time.time() + 30, 1, 1, 1, 1, max_replacements=2, interval=5,
                              stale=300, grace=0, dry_run=True).get(timeout=120)
    ok &= not any(r['spawned'] for r in dead['log']) and len(dead['log']) >= 2
    print(json.dumps(dict(ok=ok, final=final, dead_rounds=len(dead['log'])), default=str))
    if not ok:
        raise SystemExit(1)


@app.local_entrypoint()
def schedule_stop(run_id: str, at_unix: float, cancel_learner_after: int = 180):
    call = stop_run_at.spawn(run_id, at_unix, cancel_learner_after)
    print(json.dumps(dict(stop_call=call.object_id, app_id=app.app_id, at=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(at_unix)))))


# ----------------------------------------------------------------------------- four-seat Studio replays

def _four_command(f, s, t):
    """(family, source, target) of a four-seat choice -> the engine command (core.rs decode, no diplomacy)."""
    table = {0: (1, s, t, 0), 1: (2, s, t, 0), 2: (4, s, -1, t), 3: (6, -1, s, t), 4: (7, -1, -1, t),
             5: (5, -1, s, 0), 6: (3, -1, s, t + 1), 7: (9, -1, s, t), 8: (8, s, -1, 0), 9: (3, s, -1, 7),
             10: (3, s, -1, 16), 11: (3, s, -1, 8), 12: (3, s, -1, 9), 13: (3, -1, s, 4), 15: (10, -1, -1, 0),
             16: (11, -1, -1, 0)}
    kind, source, target, parameter = table[f]
    return dict(kind=kind, source=int(source), target=int(target), parameter=int(parameter))


@app.function(image=cpu_image, cpu=2, memory=32768, timeout=5400, volumes=VOLUMES, retries=0)
def four_replay_fn(seats: list, seed: int, title: str, bot_name: str, deterministic: bool = True) -> bytes:
    """One four-seat 18x18 game of `seats` ([(name, checkpoint)] x 4) recorded as a Studio replay
    (bardur-replay-v1): every frame holds the god view ('all') and each seat's own fogged view ('1'-'4'), so the
    viewer can switch perspective like a two-seat replay; 'actor' resolves to the seat to move."""
    import gzip
    from importlib.resources import files
    import jax
    import numpy as np
    from polytopia_engine_rs import GameBatch
    from polytopia_bot.gpu import ENGINE_REVISION
    from polytopia_bot.learning.adapter import serialize
    from polytopia_bot.maps.state import stack
    from polytopia_bot.maps.stream import MapSource
    from polytopia_bot.rl import network
    from polytopia_bot.rl.cpu_engine import unpack_view
    from polytopia_bot.rl.data import sha256
    from polytopia_bot.rl.distributed import checkpoint_template
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.studio.service import SCHEMA, action_label, compact
    results.reload()
    size, players = 18, 4
    params = {name: load_checkpoint(path, checkpoint_template(path)) for name, path in seats}
    act = {name: jax.jit(lambda x, m, k, p=p: network.sample(p, x, m, k, size=size, deterministic=deterministic))
           for name, p in params.items()}
    games = RustGames(size, 1, players=players, map_batch=True)
    kind, layout = MapSource('four-v1', size, players=players).layout(seed)
    games.reset(stack([(seed, layout)]))
    rules = files('polytopia_bot.gpu').joinpath('rules.json').read_text()
    render = GameBatch(size, 1, rules, ENGINE_REVISION, None, True, 10_000, players)
    batch, layout4 = games.batch, games.layout
    one = lambda tree: jax.tree.map(lambda a: np.asarray(a)[0], tree)

    def hud(view, e, player):
        ruling, structure, owner = e['tiles.city'][0], e['tiles.structure'][0], e['tiles.owner'][0]
        tiles = view['visible_state']['tiles']
        territory = {str(c['tile']): [t['tile'] for t in tiles if t['known'] and ruling[t['tile']] == c['tile']]
                     for c in view['visible_state']['cities'] if c['owner']}
        scores = e['players.score'][0].tolist()
        view['hud'] = {'score': int(scores[player - 1]), 'territory': territory,
                       'players': [{'player': p, 'score': int(scores[p - 1]),
                                    'cities': int(((structure == 1) & (owner == p)).sum())} for p in range(1, players + 1)]}
        return view

    def view(source_batch, player, version):
        v = unpack_view(np.asarray(source_batch.observe(player))[0], layout4)
        v = {k: np.asarray(a) for k, a in v.items()}
        for k in ('known_players', 'peace', 'embassies_sent', 'embassies_received', 'peace_offers', 'explored'):
            v[k] = v[k].astype(bool)   # the packed view stores flags as 0/1 int16
        v['state_version'] = np.int32(version)
        return compact(serialize(v, size=size, legal=[]))

    def frame(index, event):
        e = {k: np.asarray(a) for k, a in batch.export().items()}
        cp, version = int(e['current_player'][0]), int(e['state_version'][0])
        god = dict(e)
        god['tiles.explored'] = np.ones_like(e['tiles.explored'])
        god['units.invisible'] = np.zeros_like(e['units.invisible'])
        render.import_(god)
        all_view = view(render, cp, version)
        for c in all_view['visible_state']['cities']:
            for name in ('level', 'progress', 'population', 'production'):
                c[name] = int(e[f'cities.{name}'][0][c['tile']])
        seat_views = {str(p): hud(view(batch, p, version), e, p) for p in range(1, players + 1)}
        return {'index': index, 'current_player': cp, 'terminated': bool(e['terminated'][0]),
                'truncated': bool(e['truncated'][0]), 'event': event,
                'views': {'all': hud(all_view, e, cp), **seat_views}}

    frames = [frame(0, None)]
    key = jax.random.key(seed)
    names = [n for n, _ in seats]
    while not (frames[-1]['terminated'] or frames[-1]['truncated']):
        prepared = games._prepare()
        mover, turn = frames[-1]['current_player'], frames[-1]['views']['all']['visible_state']['turn']
        key, k = jax.random.split(key)
        c = act[names[mover - 1]](one(prepared.features), one(prepared.masks), k)[0]
        f, s, t = (int(x) for x in c)
        status = batch.step(*(np.array([x], np.int32) for x in (f, s, t)))
        if int(np.asarray(status)[0]) != 0:
            raise RuntimeError(f'policy action {(f, s, t)} rejected')
        command = _four_command(f, s, t)
        event = {'player': mover, 'turn': turn, 'source': names[mover - 1],
                 'action': {'command': command, 'label': action_label(command)}}
        frames.append(frame(len(frames), event))
    rewards = [float(r) for r in np.asarray(batch.terminal_rewards())[0]]
    result = {'rewards': rewards, 'winner': names[int(np.argmax(rewards))] if frames[-1]['terminated'] else None,
              'turns': frames[-1]['views']['all']['visible_state']['turn'], 'map': kind,
              'seats': {str(q + 1): names[q] for q in range(players)}}
    path0 = seats[0][1]
    replay = {'schema': SCHEMA, 'title': title, 'created': time.time(),
              'bot': {'name': bot_name, 'checkpoint_sha256': sha256(path0), 'mode': 'deterministic' if deterministic else 'sampled',
                      'training': 'Scratch 1v1 champion, converted to four seats, then ~1.2B samples of four-seat PPO self-play',
                      'ruleset': 'bardur-ffa4-v1'},
              'ruleset': 'bardur-ffa4-v1', 'available_views': ['1', '2', '3', '4', 'all', 'actor'], 'frames': frames,
              'complete': True,
              'result': result,
              'provenance': {'seed': seed, 'map': kind, 'source': 'four-player-recorder', 'seats': result['seats'],
                             'checkpoints': {n: p for n, p in seats}}}
    return gzip.compress(json.dumps(replay, separators=(',', ':')).encode(), 6)


@app.local_entrypoint()
def four_replays(out: str = '', ledger: str = 'runs/four-player-big2'):
    """Record a few four-seat games of the ffa4-big2 bot into the :3000 Studio library (main checkout)."""
    import gzip
    import subprocess
    import uuid
    main = Path(subprocess.run(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()).parent
    target = Path(out) if out else main / 'artifacts/studio-d3142/replays'
    bot = '/results/four-player/monitor/ffa4-big2/update-7650.npz'
    champ = '/results/four-player/parents/champ-ffa4-2x64/final.npz'
    B, C = ('FFA4 bot', bot), ('1v1 champion', champ)
    sets = [([B, B, B, B], 397001, 'FFA4 bot x4 (self-play)'),
            ([B, B, B, B], 397004, 'FFA4 bot x4 (self-play)'),
            ([B, C, C, C], 397002, 'FFA4 bot (blue) vs three converted 1v1 champions'),
            ([C, B, C, B], 397003, 'FFA4 bots (red, gold) vs converted 1v1 champions (blue, green)')]
    _ledger(dict(run_id=f'four-replays-{int(time.time())}', kind='replays', hourly_usd=round(len(sets) * cpu_hourly(2), 2),
                 reserved_usd=1.0, status='reserved', modal_app_id=app.app_id, start_unix=time.time()), ledger)
    calls = [(title, seed, four_replay_fn.spawn(seats, seed, f'{title} · map {seed}', 'Bardur · FFA4 (ffa4-big2 u7650)'))
             for seats, seed, title in sets]
    # A re-recording keeps the replay id of the earlier recording of the same map (links and --featured stay
    # valid); the earlier file moves to archived-replays.
    earlier = {}
    for old in target.glob('*.json.gz'):
        try:
            with gzip.open(old, 'rt') as f:
                provenance = json.load(f).get('provenance', {})
        except (ValueError, OSError):
            continue
        if provenance.get('source') == 'four-player-recorder':
            earlier[provenance.get('seed')] = old
    for title, seed, call in calls:
        data = call.get()
        path = earlier.get(seed) or target / f'{uuid.uuid4().hex}.json.gz'
        if path.exists():
            archive = target.parent / 'archived-replays'
            archive.mkdir(exist_ok=True)
            path.replace(archive / path.name)
        path.write_bytes(data)
        print(f'{title} map {seed}: {len(data) / 1e6:.1f} MB -> {path}', flush=True)
