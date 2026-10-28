"""Record studio replays (bardur-replay-v1) of a checkpoint on Modal CPU workers.

    modal run --env staging experiments/modal_studio_replays.py --out artifacts/studio-d3142/replays [--games elite]
    modal run --env staging experiments/modal_studio_replays.py --out artifacts/studio-omni/replays --games omni

Each game runs in its own container through the same Studio service the local
interface uses (frozen engine, deterministic policies), so the files load
directly into the local replay library. Nothing is computed on the laptop.
Games flagged `search` give the OMNI seat(s) leak-free robust turn search
(service.SEARCH) exactly as the local --search studio and the public site do.
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
app = modal.App('polytopia-studio-replays')
retained = modal.Volume.from_name('polytopia-structured-rl-retained', create_if_missing=False)
results = modal.Volume.from_name('polytopia-structured-rl', create_if_missing=False)

FINAL = '/retained/checkpoints/distributed-final-v1/dist-c460div-v1-update-3142/final.npz'
LEAGUE96 = '/retained/checkpoints/low256-league-rivals-v1/update-96-f998d15e220c/latest.npz'
DIVERSE = '/retained/checkpoints/diverse-h100-lr3e5-v1/update-176-9dc575249964/latest.npz'
E3 = '/results/distributed/dist-e340ent-v1/final.npz'
U2860 = '/results/distributed/ladder/dist-c460div-v1-update-2860.npz'
C3 = '/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz'
# c5-omni-s1-final: c3 + 6 h of PPO with an omniscient critic (docs/c5-omni-long-2026-10-21.md), 77% vs c3.
OMNI = '/retained/checkpoints/distributed-final-v1/c5-omni-s1-final/final.npz'
NAMES = {C3: 'c3 u4854', OMNI: 'OMNI c3', FINAL: 'Distributed u3142', LEAGUE96: 'league96', DIVERSE: 'diverse u176',
         E3: 'E3 final', U2860: 'Distributed u2860'}
SEARCH_WORKERS = 16
# (seat-1 checkpoint, seat-2 checkpoint, map seed[, map type[, search]]); development seeds, never final-test maps.
GAME_SETS = {
    # OMNI c3 with turn search: mirror matches, and against raw c3 in both seats, on the trained map mix.
    'omni': [(OMNI, OMNI, 321100, '', True), (OMNI, OMNI, 321101, 'continents', True),
             (OMNI, OMNI, 321102, 'pangea', True), (OMNI, OMNI, 321103, 'archipelago', True),
             (OMNI, C3, 321110, '', True), (C3, OMNI, 321110, '', True),
             (OMNI, C3, 321111, 'highlands', True), (C3, OMNI, 321112, 'lakes', True),
             (OMNI, C3, 321113, 'waterworld', True), (C3, OMNI, 321114, 'jungle', True),
             (OMNI, C3, 321115, 'ridge', True), (C3, OMNI, 321116, 'continents', True)],
    # Latest bot against itself: two classic maps and one of each new map type (maps/variants.py).
    'c3': [(C3, C3, 321000), (C3, C3, 321001)] + [
        (C3, C3, 321010 + k, kind) for k, kind in enumerate(
            ('continents', 'archipelago', 'waterworld', 'lakes', 'pangea', 'ridge', 'highlands', 'jungle'))],
    'baseline': [(FINAL, LEAGUE96, 306400), (LEAGUE96, FINAL, 306401), (FINAL, LEAGUE96, 306402),
                 (FINAL, FINAL, 306403), (FINAL, DIVERSE, 306404), (DIVERSE, FINAL, 306405)],
    # Top of the ladder: u3142 mirror matches, and u3142 against its closest rivals in both seats.
    'elite': [(FINAL, FINAL, 306410), (FINAL, FINAL, 306411), (FINAL, FINAL, 306412),
              (E3, E3, 306413),
              (FINAL, E3, 306414), (E3, FINAL, 306414), (FINAL, E3, 306415), (E3, FINAL, 306415),
              (FINAL, U2860, 306416), (U2860, FINAL, 306416)],
}


@app.function(image=image, cpu=SEARCH_WORKERS, memory=16384, timeout=3600,
              volumes={'/retained': retained, '/results': results.with_mount_options(read_only=True)})
def record(first: str, second: str, seed: int, kind: str = '', search: bool = False):
    import tempfile
    from polytopia_bot.rl.policy import StructuredPolicy
    from polytopia_bot.studio.service import Studio
    if search:
        # The served bot (OMNI) searches; a different checkpoint in the other seat plays its raw policy.
        if OMNI not in (first, second):
            raise ValueError('search games need OMNI in a seat')
        rival = second if first == OMNI else first
        studio = Studio(Path(tempfile.mkdtemp()), OMNI, name='Bardur · OMNI c3 (c5-omni-s1, update 2212)',
                        short='OMNI c3', search_workers=SEARCH_WORKERS,
                        training='Teacher imitation, PPO self-play on a distributed CPU-actor league (c3), '
                                 'then 6 h of PPO with an omniscient critic; leak-free turn search at play time',
                        opponents={NAMES[rival]: rival} if rival != OMNI else None)
        studio.warmup()
        studio.warm_search()
        ident = studio.create(dict(mode='watch', seed=seed, search=True,
                                   **({'opponent': NAMES[rival], 'opponent_seat': 1 if first != OMNI else 2}
                                      if rival != OMNI else {})))['game_id']
    else:
        studio = Studio(Path(tempfile.mkdtemp()), FINAL, name='Bardur · Distributed PPO u3142',
                        short='Distributed u3142', training='Teacher imitation, PPO, distributed CPU-actor self-play league')
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
        _, lay = MapSource(kind).layout(seed, kind)
        from polytopia_bot.learning.memory import PublicMemory
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
    def label(checkpoint):
        return NAMES[checkpoint] + (' + search' if search and checkpoint == OMNI else '')
    g['title'] = f'{label(first)} (blue) vs {label(second)} (red) · {kind or "classic"} map {seed}'
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
def main(out: str = 'artifacts/studio-d3142/replays', games: str = 'baseline'):
    folder = ROOT / out
    folder.mkdir(parents=True, exist_ok=True)
    for result in record.starmap(GAME_SETS[games]):
        (folder / f"{result['id']}.json.gz").write_bytes(result.pop('data'))
        print(json.dumps(result))
