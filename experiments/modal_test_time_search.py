"""Inference-time turn search for the RL bot: search-K vs raw opponents on dev seeds.

    modal run --env staging experiments/modal_test_time_search.py::smoke
    modal run --detach --env staging experiments/modal_test_time_search.py::sweep --tag v1 --spec-json '[...]'
    modal run --env staging experiments/modal_test_time_search.py::fetch --tag v1

Each spec row is one arm: {name, candidate, opponent, lanes, depth, seed_base, maps, ...}. Maps are dev seeds
(300000-399999); every map is played in both seats. Arms are split into chunks of `per_container` games, one
32-core container per chunk with one pinned game process per core. Results go to
/results/test-time-search/<tag>/<arm>.json on the polytopia-structured-rl volume.
"""
from pathlib import Path
import json
import os
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0', 'pytest>=9,<10')
    .apt_install('curl', 'build-essential', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
    .pip_install('maturin>=1.7,<2')
    .add_local_dir(ROOT / 'engine-rs', '/workspace/engine-rs', copy=True, ignore=['**/target/**'])
    .run_commands('cd /workspace/engine-rs && PATH=/root/.cargo/bin:$PATH maturin build --release -o /workspace/wheels '
                  '&& pip install /workspace/wheels/*.whl')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'tests', '/workspace/tests', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-test-time-search')
results = modal.Volume.from_name('polytopia-structured-rl', create_if_missing=False)
retained = modal.Volume.from_name('polytopia-structured-rl-retained', create_if_missing=False)
VOLUMES = {'/results': results, '/retained': retained}
CPU_CORE_RATE = 0.0472    # $/core-hour, as in modal_distributed_rl.py
MEM_GIB_RATE = 0.008      # $/GiB-hour
CORES, MEMORY_MB = 32, 40960   # v1 used 65536
U3142 = '/retained/checkpoints/distributed-final-v1/dist-c460div-v1-update-3142/final.npz'
LEAGUE96 = '/retained/checkpoints/low256-league-rivals-v1/update-96-f998d15e220c/latest.npz'
E3 = '/results/distributed/dist-e340ent-v1/final.npz'   # dist-e340ent-v1 final, a different lineage
C3 = '/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz'   # campaign 3, map-mix self-play
# c3 trained 1 h with an omniscient critic (docs/ablation-v1/readout-2026-10-20.md). Its own value head is the stale
# c3 warm start (unused while the omniscient critic trains); final-critic.npz needs both views and must not search.
OMNI = '/retained/checkpoints/distributed-final-v1/abl-omni-2-final/final.npz'
LEDGER = ROOT / os.environ.get('SEARCH_LEDGER', 'runs/test-time-search-v1/budget.json')
OUT = Path('/results/test-time-search')


def _cost(seconds, cores=CORES, memory_mb=MEMORY_MB):
    return seconds / 3600 * (cores * CPU_CORE_RATE + memory_mb / 1024 * MEM_GIB_RATE)


@app.function(image=cpu_image, cpu=CORES, memory=MEMORY_MB, timeout=4 * 3600, volumes=VOLUMES, retries=0)
def play_chunk(cfg: dict, items: list):
    from polytopia_bot.rl.search_arena import run, summarize
    started = time.monotonic()
    rows = run(cfg, [tuple(i) for i in items])
    return dict(rows=rows, seconds=time.monotonic() - started, summary=summarize(rows) if rows else {})


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=3600, volumes=VOLUMES, retries=0)
def smoke_check(seeds: list, lanes: int):
    """K=1 must reproduce cpu_arena's raw-vs-raw outcomes exactly; then time one K-lane game."""
    from polytopia_bot.rl.cpu_arena import match
    from polytopia_bot.rl.search_arena import run, summarize
    out = {}
    started = time.monotonic()
    ref = match(U3142, LEAGUE96, seeds, games=2, deterministic=True, processes=4)
    out['cpu_arena_u3142_vs_l96'] = {(d['seed'], d['seat']): d['reward'] for d in ref['details']}
    out['cpu_arena_seconds'] = time.monotonic() - started
    started = time.monotonic()
    rows = run(dict(candidate=U3142, opponent=LEAGUE96, lanes=1), [(s, seat) for s in seeds for seat in (1, 2)], 4)
    out['search_k1_vs_l96'] = {(r['seed'], r['seat']): r['reward'] for r in rows}
    out['k1_summary'] = summarize(rows)
    out['k1_seconds'] = time.monotonic() - started
    out['k1_match'] = out['cpu_arena_u3142_vs_l96'] == out['search_k1_vs_l96']
    started = time.monotonic()
    rows = run(dict(candidate=U3142, opponent=U3142, lanes=lanes), [(seeds[0], 1), (seeds[0], 2)], 2)
    out[f'k{lanes}_summary'] = summarize(rows)
    out[f'k{lanes}_rows'] = [{k: v for k, v in r.items() if k != 'trace'} for r in rows]
    out[f'k{lanes}_seconds'] = time.monotonic() - started
    out['cpu_arena_u3142_vs_l96'] = {f'{k[0]}/{k[1]}': v for k, v in out['cpu_arena_u3142_vs_l96'].items()}
    out['search_k1_vs_l96'] = {f'{k[0]}/{k[1]}': v for k, v in out['search_k1_vs_l96'].items()}
    return out


@app.local_entrypoint()
def smoke(seeds: str = '390000,390001', lanes: int = 8):
    out = smoke_check.remote([int(s) for s in seeds.split(',')], lanes)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    (LEDGER.parent / 'smoke.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    print(json.dumps({k: v for k, v in out.items() if not k.endswith(('_rows', '_summary'))}, indent=1, default=str))


@app.function(image=cpu_image, cpu=CORES, memory=MEMORY_MB, timeout=1800, volumes=VOLUMES, retries=0)
def diag_chunk(cfg: dict, items: list, processes: int):
    """Debug a play_chunk failure: one game in-process (Python errors surface), then a pool with memory sampling."""
    import threading
    import traceback
    import multiprocessing as mp
    from polytopia_bot.rl import search_arena
    out, peak, stop = {}, [0], threading.Event()
    def sample():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], int(Path('/sys/fs/cgroup/memory.current').read_text()))
            except Exception:
                pass
            time.sleep(1)
    threading.Thread(target=sample, daemon=True).start()
    out['affinity'] = len(os.sched_getaffinity(0))
    try:
        out['cpu.max'] = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    except Exception as e:
        out['cpu.max'] = repr(e)
    cfg = _resolved(cfg)
    try:
        from polytopia_bot.rl.distributed import _env_setup
        _env_setup()
        q = mp.get_context('spawn').Queue()
        q.put(None)
        t0 = time.monotonic()
        search_arena._init(cfg, q)
        w = search_arena._W
        if 'error' in w:
            raise RuntimeError(w['error'])
        row = search_arena.play_game(w['searcher'], w['raw'], w['reset'], *items[0])
        out['inprocess'] = dict(ok=True, seconds=time.monotonic() - t0, reward=row['reward'], turns=row['turns'])
    except Exception:
        out['inprocess'] = dict(ok=False, error=traceback.format_exc()[-3000:])
    out['peak_inprocess_gib'] = peak[0] / 2**30
    peak[0] = 0
    if processes:
        t0 = time.monotonic()
        try:
            rows = search_arena.run(cfg, [tuple(i) for i in items[:processes]], processes)
            out['pool'] = dict(ok=True, games=len(rows), seconds=time.monotonic() - t0)
        except Exception:
            out['pool'] = dict(ok=False, error=traceback.format_exc()[-3000:])
        out['peak_pool_gib'] = peak[0] / 2**30
    stop.set()
    return out


@app.local_entrypoint()
def diag(arm: str = 'c3-classic-d1', processes: int = 8, spec_json: str = 'runs/test-time-search-c3/spec-c3.json'):
    cfg = next(a for a in json.loads(Path(spec_json).read_text()) if a['name'] == arm)
    items = [(s, seat) for s in range(cfg['seed_base'], cfg['seed_base'] + 32) for seat in (1, 2)]
    out = diag_chunk.remote(cfg, items, processes)
    print(json.dumps(out, indent=1, default=str))


@app.local_entrypoint()
def smoke_arms(spec_json: str, games: int = 2, seed: int = 386000, out: str = 'smoke-arms.json'):
    """Every arm of the spec plays `games` games (seed, seat 1 / seat 2, ...) in its own play_chunk container,
    concurrently: legality, s/turn and the per-arm setup (checkpoints, scorer) before a sweep."""
    spec = json.loads(Path(spec_json).read_text())
    items = [(seed + i // 2, 1 + i % 2) for i in range(games)]
    calls = [(arm['name'], play_chunk.spawn(_resolved(arm), items)) for arm in spec]
    result = {}
    for name, call in calls:
        try:
            part = call.get()
            rows = part['rows']
            result[name] = dict(ok=True, seconds=part['seconds'], cost_usd=_cost(part['seconds']),
                                summary={k: v for k, v in part['summary'].items() if k != 'unit_mix'},
                                rows=[{k: v for k, v in r.items() if k != 'trace'} for r in rows])
        except Exception as e:
            result[name] = dict(ok=False, error=repr(e))
        r = result[name]
        s = r.get('summary', {})
        print(f"{name:26s} ok={r['ok']} games={s.get('games')} W-L-T={s.get('wins')}-{s.get('losses')}-{s.get('truncations')} "
              f"s/turn={s.get('seconds_per_search_turn', 0):.2f} override={s.get('override_rate', 0):.2f} "
              f"turns={s.get('mean_turns', 0):.0f} compile={s.get('compile_seconds', 0):.0f}s "
              f"wall={r.get('seconds', 0):.0f}s cost=${r.get('cost_usd', 0):.2f} {r.get('error', '')[:300]}")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    (LEDGER.parent / out).write_text(json.dumps(result, indent=1, default=str) + '\n')


def _resolve(name):
    return {'u3142': U3142, 'league96': LEAGUE96, 'e3': E3, 'c3': C3, 'omni': OMNI}.get(name, name)


def _resolved(arm):
    cfg = {**arm, 'candidate': _resolve(arm['candidate']), 'opponent': _resolve(arm['opponent'])}
    if arm.get('value'):
        cfg['value'] = _resolve(arm['value'])
    return cfg


@app.function(image=cpu_image, cpu=0.25, memory=1024, timeout=24 * 3600, volumes=VOLUMES, retries=0)
def orchestrate(tag: str, spec: list, per_container: int):
    from polytopia_bot.rl.search_arena import summarize
    from polytopia_bot.rl.runner import write_json
    (OUT / tag).mkdir(parents=True, exist_ok=True)
    calls = []
    for arm in spec:
        cfg = _resolved(arm)
        items = [(s, seat) for s in range(arm['seed_base'], arm['seed_base'] + arm['maps']) for seat in (1, 2)]
        for i in range(0, len(items), per_container):
            calls.append((arm['name'], cfg, play_chunk.spawn(cfg, items[i:i + per_container])))
    by_arm = {}
    for name, cfg, call in calls:
        try:
            part = call.get()
        except Exception as e:  # keep the other arms
            part = dict(rows=[], seconds=0., error=repr(e))
        arm = by_arm.setdefault(name, dict(cfg=cfg, rows=[], container_seconds=0., errors=[]))
        arm['rows'] += part['rows']
        arm['container_seconds'] += part['seconds']
        if 'error' in part:
            arm['errors'].append(part['error'])
        arm['chunks'] = arm.get('chunks', 0) + 1
        total = sum(1 for n, _, _ in calls if n == name)
        write_json(OUT / tag / f'{name}.json', dict(name=name, config=cfg, chunks=f"{arm['chunks']}/{total}",
                   summary=summarize(arm['rows']) if arm['rows'] else {},
                   container_seconds=arm['container_seconds'], cost_usd=_cost(arm['container_seconds']),
                   errors=arm['errors'], rows=arm['rows']))
        results.commit()
    index = {n: dict(summary=summarize(a['rows']) if a['rows'] else {}, cost_usd=_cost(a['container_seconds']),
                     errors=a['errors']) for n, a in by_arm.items()}
    write_json(OUT / tag / 'index.json', index)
    results.commit()
    return index


@app.local_entrypoint()
def sweep(tag: str, spec_json: str, per_container: int = 32):
    spec = json.loads(Path(spec_json).read_text()) if spec_json.endswith('.json') else json.loads(spec_json)
    for arm in spec:
        for side in ('candidate', 'opponent', 'value'):
            if arm.get(side) and not _resolve(arm[side]).startswith('/'):
                raise ValueError(f"unknown checkpoint {arm[side]!r} in arm {arm['name']}")
        lo, hi = arm['seed_base'], arm['seed_base'] + arm['maps']
        if not (300000 <= lo and hi <= 400000):
            raise ValueError('search evaluation must stay on development seeds 300000-399999')
    containers = sum(-(-2 * a['maps'] // per_container) for a in spec)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    ledger.append(dict(tag=tag, launched=time.strftime('%Y-%m-%d %H:%M:%S %Z'), arms=[a['name'] for a in spec],
                       containers=containers, per_container=per_container, cores=CORES, memory_mb=MEMORY_MB))
    LEDGER.write_text(json.dumps(ledger, indent=1) + '\n')
    call = orchestrate.spawn(tag, spec, per_container)
    print(json.dumps(dict(tag=tag, containers=containers, call=call.object_id)))


@app.function(image=cpu_image, cpu=0.25, memory=1024, timeout=600, volumes=VOLUMES)
def read_results(tag: str):
    results.reload()
    root = OUT / tag
    return {p.stem: json.loads(p.read_text()) for p in sorted(root.glob('*.json'))} if root.exists() else {}


@app.local_entrypoint()
def fetch(tag: str, out_dir: str = 'runs/test-time-search-v1'):
    data = read_results.remote(tag)
    target = ROOT / out_dir / tag
    target.mkdir(parents=True, exist_ok=True)
    for name, d in data.items():
        (target / f'{name}.json').write_text(json.dumps(d, indent=1) + '\n')
        if name != 'index':
            s = d.get('summary', {})
            print(f"{name:28s} games={s.get('games')} W-L-T={s.get('wins')}-{s.get('losses')}-{s.get('truncations')} "
                  f"score={s.get('score', 0):.3f} {s.get('score_ci95')} elo={s.get('elo', 0):+.0f} "
                  f"s/turn={s.get('seconds_per_search_turn', 0):.2f} override={s.get('override_rate', 0):.2f} "
                  f"cost=${d.get('cost_usd', 0):.2f} err={len(d.get('errors', []))}")


@app.function(image=cpu_image, cpu=4, memory=16384, timeout=3600)
def run_tests(paths: str):
    import subprocess
    out = subprocess.run(['python', '-m', 'pytest', '-x', '-q', *paths.split(',')], cwd='/workspace',
                         capture_output=True, text=True)
    return out.returncode, out.stdout[-6000:] + out.stderr[-3000:]


@app.local_entrypoint()
def tests(paths: str = 'tests/test_search.py'):
    code, log = run_tests.remote(paths)
    print(log)
    if code:
        raise SystemExit(code)
