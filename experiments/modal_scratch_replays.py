"""Record Studio replays (bardur-replay-v1) of the from-scratch overnight policy on Modal CPU workers.

    modal run --env staging experiments/modal_scratch_replays.py --out artifacts/studio-d3142/replays

Raw deterministic policies in both seats (no search), one game per container, through the same Studio service the
local interface uses, so the files load directly into the local replay library. Nothing is computed on the laptop.
The play-omni worktree's recorder (experiments/modal_studio_replays.py there) predates this checkpoint's engine.
"""
from pathlib import Path
import json
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax==0.11.2', 'numpy>=2.1')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-scratch-replays')
retained = modal.Volume.from_name('polytopia-structured-rl-retained', create_if_missing=False)
results = modal.Volume.from_name('polytopia-structured-rl', create_if_missing=False)

# 7 h of pure self-play from random weights (docs/scratch-check-2026-10-22.md): 83% vs c3, ~parity with OMNI.
SCRATCH = '/results/distributed/scratch-overnight-1/final.npz'
C3 = '/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz'
OMNI = '/retained/checkpoints/distributed-final-v1/c5-omni-s1-final/final.npz'
NAMES = {SCRATCH: 'Scratch 7h', C3: 'c3 u4854', OMNI: 'OMNI c3'}
# (seat-1 checkpoint, seat-2 checkpoint, map seed[, map type]); development seeds, never final-test maps.
GAMES = [(SCRATCH, OMNI, 321200), (OMNI, SCRATCH, 321201, 'continents'), (SCRATCH, OMNI, 321202, 'archipelago')]


@app.function(image=image, cpu=4, memory=8192, timeout=3600,
              volumes={'/retained': retained, '/results': results.with_mount_options(read_only=True)})
def record(first: str, second: str, seed: int, kind: str = ''):
    import tempfile
    from polytopia_bot.rl.policy import StructuredPolicy
    from polytopia_bot.studio.service import Studio
    studio = Studio(Path(tempfile.mkdtemp()), SCRATCH, name='Bardur · Scratch 7h (scratch-overnight-1, update 8724)',
                    short='Scratch 7h', training='Pure PPO self-play from random weights, 7 h, omniscient critic')
    studio.warmup()
    studio.policies = {1: StructuredPolicy(first, seed=101, deterministic=True),
                       2: StructuredPolicy(second, seed=102, deterministic=True)}
    ident = studio.create(dict(mode='watch', seed=seed))['game_id']
    g = studio.games[ident]
    if kind:
        # Start on a variant map: same engine, only the starting state differs (maps/state.py).
        import jax
        from polytopia_bot.maps.state import initial_state
        from polytopia_bot.maps.stream import MapSource
        from polytopia_bot.learning.memory import PublicMemory
        _, lay = MapSource(kind).layout(seed, kind)
        env = g['env']
        env.state = jax.jit(lambda: initial_state(seed, *(jax.numpy.asarray(a) for a in
                            (lay.terrain, lay.structure, lay.resource, lay.capitals)), size=14))()
        # Drop the classic map's cached observations/legal actions and public memory (as reset() does).
        env.memory, env._cached = PublicMemory(), {}
        env.observe(2)
        env.observe(1)
        # Clear first: frame() numbers itself by len(g['frames']), so the opening frame must be index 0.
        g['frames'].clear()
        g['frames'].append(studio.frame(g))
    g['title'] = f'{NAMES[first]} (blue) vs {NAMES[second]} (red) · {kind or "classic"} map {seed}'
    while True:
        desc = studio.advance(ident, dict(count=8))
        if desc['terminated'] or desc['truncated']:
            break
    studio.save(g)
    raw = (studio.storage / 'replays' / f'{ident}.json.gz').read_bytes()
    last = g['frames'][-1]['event'] or {}
    return dict(id=ident, title=g['title'], frames=len(g['frames']), rewards=last.get('rewards'),
                truncated=desc['truncated'], data=raw)


@app.local_entrypoint()
def main(out: str = 'artifacts/studio-d3142/replays'):
    folder = ROOT / out
    folder.mkdir(parents=True, exist_ok=True)
    for result in record.starmap(GAMES):
        (folder / f"{result['id']}.json.gz").write_bytes(result.pop('data'))
        print(json.dumps(result), flush=True)
