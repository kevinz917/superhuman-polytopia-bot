"""Round-robin Elo tournament among frozen checkpoints on the CPU arena (frozen engine, deterministic).

    modal run --detach --env staging experiments/modal_elo_tournament.py --name X \
        --players-json '{"label": "/retained/.../final.npz", ...}' --anchor label

Every pair plays the same development maps with both seats (rl/cpu_arena.py, the arena of
modal_distributed_rl.py); ratings are Bradley-Terry with map-bootstrap intervals (rl/elo.py).
Per-pair records go to /results/distributed/arena/elo-<name>-<a>-vs-<b>.json, the report to
/results/distributed/elo/<name>/ratings.json. Rerunning the same name resumes finished pairs.
"""
from pathlib import Path
import json
import os
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
image = (modal.Image.debian_slim(python_version='3.12')
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
app = modal.App('polytopia-elo-tournament', image=image)
results = modal.Volume.from_name('polytopia-structured-rl', create_if_missing=False)
retained = modal.Volume.from_name('polytopia-structured-rl-retained', create_if_missing=False)
VOLUMES = {'/results': results, '/retained': retained.with_mount_options(read_only=True)}
CPU_CORE_RATE = 0.0472
MEM_GIB_RATE = 0.008
ARENA = Path('/results/distributed/arena')


@app.function(cpu=8, memory=16384, timeout=1800)
def run_tests(paths: str):
    import subprocess
    out = subprocess.run(['python', '-m', 'pytest', '-x', '-q', *paths.split(',')], cwd='/workspace',
                         capture_output=True, text=True)
    return out.returncode, out.stdout[-6000:] + out.stderr[-3000:]


@app.function(cpu=32, memory=65536, timeout=7200, volumes=VOLUMES, retries=1)
def pair_match(name: str, candidate: str, opponent: str, seed_base: int, maps: int):
    from polytopia_bot.rl.cpu_arena import match
    from polytopia_bot.rl.runner import write_json
    summary = match(candidate, opponent, range(seed_base, seed_base + maps), games=4, deterministic=True,
                    processes=min(2 * maps, 62))
    summary.update(name=name, candidate=candidate, opponent=opponent, seed_base=seed_base, maps=maps,
                   deterministic=True, opponent_deterministic=True, engine='frozen (cpu_engine fast=False)')
    write_json(ARENA / f'{name}.json', summary)
    results.commit()
    return {k: v for k, v in summary.items() if k != 'details'}


@app.function(cpu=1, memory=4096, timeout=86400, retries=0, volumes=VOLUMES)
def tournament(name: str, players: dict, anchor: str, seed_base: int, maps: int, concurrency: int):
    import itertools
    from concurrent.futures import ThreadPoolExecutor
    from polytopia_bot.rl.elo import pair_table, ratings
    from polytopia_bot.rl.runner import write_json
    labels = list(players)
    pairs = list(itertools.combinations(labels, 2))
    path = lambda a, b: ARENA / f'elo-{name}-{a}-vs-{b}.json'
    results.reload()
    todo = [(a, b) for a, b in pairs if not path(a, b).exists()]
    failures = []
    with ThreadPoolExecutor(concurrency) as pool:
        futures = [(pair, pool.submit(pair_match.remote, f'elo-{name}-{pair[0]}-vs-{pair[1]}', players[pair[0]],
                                      players[pair[1]], seed_base, maps)) for pair in todo]
        for pair, future in futures:
            try:
                out = future.result()
                print(json.dumps(dict(pair=pair, **{k: out[k] for k in ('wins', 'losses', 'truncations', 'seconds')})),
                      flush=True)
            except Exception as error:
                failures.append(dict(pair=list(pair), error=repr(error)))
    results.reload()
    rows, seconds = [], {}
    for a, b in pairs:
        if not path(a, b).exists():
            continue
        summary = json.loads(path(a, b).read_text())
        seconds[f'{a}__{b}'] = summary['seconds']
        rows += [dict(a=a, b=b, seed=g['seed'], seat=g['seat'], truncated=g['truncated'],
                      score=1. if g['reward'] > 0 else 0. if g['reward'] < 0 else .5) for g in summary['details']]
    report = dict(name=name, players=players, anchor=anchor, seed_base=seed_base, maps=maps, seats=2,
                  deterministic=True, engine='frozen (cpu_engine fast=False)', games=len(rows),
                  pairs_complete=len(seconds), pairs=len(pairs), failures=failures,
                  ratings=ratings(rows, labels, anchor),
                  decisive_only_ratings=ratings(rows, labels, anchor, decisive_only=True),
                  pair_results=pair_table(rows, labels), match_seconds=seconds,
                  method='Bradley-Terry (MM); truncation = half point; 0.5 virtual draws per played pair; '
                         '2000 map-seed bootstrap resamples; Elo = 400 log10(strength) relative to the anchor')
    root = Path('/results/distributed/elo') / name
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'ratings.json', report)
    write_json(root / 'games.json', rows)
    results.commit()
    return {k: v for k, v in report.items() if k != 'pair_results'}


@app.local_entrypoint()
def tests(paths: str = 'tests/test_elo.py'):
    code, text = run_tests.remote(paths)
    print(text)
    if code:
        raise SystemExit(code)


@app.local_entrypoint()
def main(name: str, players_json: str, anchor: str, seed_base: int = 306500, maps: int = 32, concurrency: int = 12,
         reserve_usd: float = 20.):
    import sys
    players = json.loads(players_json)
    if not (306500 <= seed_base and seed_base + maps <= 400000):
        raise ValueError('fresh development maps only (306500 <= seed < 400000)')
    if anchor not in players or len(players) < 2:
        raise ValueError('anchor must be one of at least two players')
    sys.path.insert(0, str(ROOT / 'src'))
    from polytopia_bot.rl.budget import reserve, record
    hourly = concurrency * (32 * CPU_CORE_RATE + 64 * MEM_GIB_RATE)
    folder = ROOT / 'runs/distributed-rl-v1'
    reserve(folder, dict(run_id=f'elo-{name}', kind='round-robin-elo', players=players, anchor=anchor,
                         seed_base=seed_base, maps=maps, concurrency=concurrency, hourly_usd_max=round(hourly, 2),
                         reserved_usd=reserve_usd, status='reserved', modal_app_id=app.app_id, start_unix=time.time()))
    call = tournament.spawn(name, players, anchor, seed_base, maps, concurrency)
    record(folder, f'elo-{name}', status='submitted', function_call_id=call.object_id)
    print(json.dumps(dict(name=name, app_id=app.app_id, call=call.object_id,
                          pairs=len(players) * (len(players) - 1) // 2)), flush=True)
