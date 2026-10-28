"""Record Studio replays (bardur-replay-v1) of the champion (scratch-cont-2 + its fog-of-war value head) playing
with leak-free turn search, on Modal CPU workers.

    modal run --env staging experiments/modal_champion_replays.py --out /path/to/main/artifacts/studio-d3142/replays

The champion is the served bot and searches exactly as the Studio and the public site do (service.SEARCH, scored by
its own value head); a rival in the other seat plays its raw policy (Studio rivals never search), and a mirror game
searches in both seats. One game per 16-core container; nothing is computed on the laptop.
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
    # One XLA thread per search lane (docs/search-speed-2026-10-19.md).
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu',
          'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-champion-replays')
retained = modal.Volume.from_name('polytopia-structured-rl-retained', create_if_missing=False)

R = '/retained/checkpoints/distributed-final-v1'
# scratch-cont-2 final with a trained fog-of-war value head (policy byte-identical; docs/scratch-search-2026-10-23.md)
CHAMP = f'{R}/scratch-cont-2-foghead/final.npz'
OMNI = f'{R}/c5-omni-s1-final/final.npz'
C3 = f'{R}/c3-maps-ctl-update-4854/final.npz'
SCRATCH7 = f'{R}/scratch-7h-final/final.npz'
NAMES = {CHAMP: 'Scratch champion', OMNI: 'OMNI c3', C3: 'c3 u4854', SCRATCH7: 'Scratch 7h'}
SEARCH_WORKERS = 16
# (seat-1 checkpoint, seat-2 checkpoint, map seed, map type); development seeds, never final-test maps.
GAMES = [(CHAMP, OMNI, 321300, 'classic'), (OMNI, CHAMP, 321301, 'continents'),
         (CHAMP, C3, 321302, 'pangea'), (SCRATCH7, CHAMP, 321303, 'archipelago'),
         (CHAMP, CHAMP, 321304, 'highlands'), (CHAMP, CHAMP, 321305, 'waterworld')]


@app.function(image=image, cpu=SEARCH_WORKERS, memory=16384, timeout=3600, volumes={'/retained': retained})
def record(first: str, second: str, seed: int, kind: str):
    import tempfile
    from polytopia_bot.studio.service import Studio
    rival = second if first == CHAMP else first
    studio = Studio(Path(tempfile.mkdtemp()), CHAMP, name='Bardur · Scratch champion (scratch-cont-2)',
                    short='Scratch champion', search_workers=SEARCH_WORKERS,
                    training='Pure PPO self-play from random weights (~20 h, omniscient critic), fog-of-war value '
                             'head; leak-free turn search at play time',
                    opponents={NAMES[rival]: rival} if rival != CHAMP else None)
    studio.warmup()
    studio.warm_search()
    ident = studio.create(dict(mode='watch', seed=seed, search=True,
                               **({'opponent': NAMES[rival], 'opponent_seat': 1 if first != CHAMP else 2}
                                  if rival != CHAMP else {})))['game_id']
    g = studio.games[ident]
    if kind != 'classic':
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
    label = lambda c: NAMES[c] + (' + search' if c == CHAMP else '')
    g['title'] = f'{label(first)} (blue) vs {label(second)} (red) · {kind} map {seed}'
    while True:
        desc = studio.advance(ident, dict(count=8))
        if desc['terminated'] or desc['truncated']:
            break
    studio.save(g)
    raw = (studio.storage / 'replays' / f'{ident}.json.gz').read_bytes()
    last = g['frames'][-1]['event'] or {}
    searched = sum(1 for f in g['frames'] if (f.get('event') or {}).get('source') == 'search')
    return dict(id=ident, title=g['title'], frames=len(g['frames']), rewards=last.get('rewards'),
                truncated=desc['truncated'], searched_actions=searched, data=raw)


@app.local_entrypoint()
def main(out: str = 'artifacts/studio-d3142/replays'):
    folder = ROOT / out
    folder.mkdir(parents=True, exist_ok=True)
    for result in record.starmap(GAMES):
        (folder / f"{result['id']}.json.gz").write_bytes(result.pop('data'))
        print(json.dumps(result), flush=True)
