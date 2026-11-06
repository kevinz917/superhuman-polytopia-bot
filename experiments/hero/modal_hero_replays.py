"""Record Studio replays (bardur-replay-v1) of the hero bot (hero-1v1-cont1h u11650, 4x128) against the scratch
champion on the exact maps of the hero-1v1-cont1h final round (proposed-v1 mix, seeds 510000+), on Modal CPU workers.

    modal run --env staging experiments/hero/modal_hero_replays.py --out /path/to/main/artifacts/studio-d3142/replays

Both seats play their raw deterministic policy (the hero has no fog-of-war value head yet, so no search), as in the
arena. One game per container; nothing is computed on the laptop.
"""
from pathlib import Path
import json
import os
import modal

# parents[2] is the repo root locally; inside the container the file sits at /root/ (ROOT is only used to build).
ROOT = Path(__file__).resolve().parents[min(2, len(Path(__file__).resolve().parents) - 1)]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
# Same recipe as experiments/modal_distributed_rl.py cpu_image: engine-rs built from source into an abi3 wheel.
image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('curl', 'build-essential', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0', 'maturin>=1.7,<2')
    .add_local_dir(ROOT / 'engine-rs', '/workspace/engine-rs', copy=True, ignore=['**/target/**'])
    .run_commands('cd /workspace/engine-rs && PATH=/root/.cargo/bin:$PATH maturin build --release -o /workspace/wheels '
                  '&& pip install /workspace/wheels/*.whl')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-hero-replays')
results = modal.Volume.from_name('polytopia-structured-rl', create_if_missing=False)
retained = modal.Volume.from_name('polytopia-structured-rl-retained', create_if_missing=False)

HERO = '/results/distributed/hero-1v1-cont1h/latest.npz'   # update-11650, the run's last checkpoint
CHAMP = '/retained/checkpoints/distributed-final-v1/scratch-cont-2-final/final.npz'
# (hero seat, map seed, expected map type, final-round result for the hero)
GAMES = [(1, 510019, 'classic', 'W'), (2, 510011, 'highlands', 'W'), (1, 510036, 'jungle', 'W'),
         (1, 510007, 'continents', 'W'), (2, 510013, 'jungle', 'L'), (1, 510051, 'classic', 'L'),
         (2, 510039, 'waterworld', 'L'), (2, 510006, 'pangea', 'L')]


@app.function(image=image, cpu=4, memory=8192, timeout=3600, volumes={'/results': results, '/retained': retained})
def record(seat: int, seed: int, kind: str, expected: str):
    import tempfile
    import numpy as np
    from polytopia_bot.learning.memory import PublicMemory
    from polytopia_bot.maps.stream import MapSource
    from polytopia_bot.studio.service import Studio
    studio = Studio(Path(tempfile.mkdtemp()), HERO, name='Bardur · Hero (hero-1v1-cont1h u11650)', short='Hero',
                    training='Pure PPO self-play from random weights under the real 2.17.3 rules (4x128, omniscient '
                             'critic, ~6.95 B samples); raw policy',
                    opponents={'Scratch champion': CHAMP})
    studio.warmup()
    ident = studio.create(dict(mode='watch', seed=seed, search=False, opponent='Scratch champion',
                               opponent_seat=3 - seat))['game_id']
    g = studio.games[ident]
    # Start on the final round's map (MapSource picks the same layout the arena used for this seed).
    maps, kinds = MapSource('proposed-v1').batch([seed])
    assert kinds[0] == kind, (seed, kinds[0], kind)
    env = g['env']
    i32 = lambda a: np.ascontiguousarray(np.asarray(a), np.int32)
    env.games.batch.reset_layouts(i32(maps.seed), i32(maps.terrain), i32(maps.structure), i32(maps.resource),
                                  i32(maps.capitals))
    env._sync()
    env.memory, env._cached = PublicMemory(), {}
    env.observe(2)
    env.observe(1)
    # Clear first: frame() numbers itself by len(g['frames']), so the opening frame must be index 0.
    g['frames'].clear()
    g['frames'].append(studio.frame(g))
    g['title'] = (f'Hero (blue) vs Scratch champion (red)' if seat == 1 else
                  f'Scratch champion (blue) vs Hero (red)') + f' · {kind} map {seed}'
    while True:
        desc = studio.advance(ident, dict(count=8))
        if desc['terminated'] or desc['truncated']:
            break
    studio.save(g)
    raw = (studio.storage / 'replays' / f'{ident}.json.gz').read_bytes()
    rewards = (g['frames'][-1]['event'] or {}).get('rewards') or {}
    hero_reward = rewards.get(str(seat), rewards.get(seat))
    return dict(id=ident, title=g['title'], frames=len(g['frames']), rewards=rewards, truncated=desc['truncated'],
                hero_result='W' if (hero_reward or 0) > 0 else 'L' if (hero_reward or 0) < 0 else 'D',
                arena_result=expected, data=raw)


@app.local_entrypoint()
def main(out: str = 'artifacts/studio-d3142/replays', only: str = ''):
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    games = [GAMES[int(i)] for i in only.split(',')] if only else GAMES
    for result in record.starmap(games):
        (folder / f"{result['id']}.json.gz").write_bytes(result.pop('data'))
        print(json.dumps(result), flush=True)
