"""Playability of every map type with the current bot (frozen engine, CPU arena, Modal only).

    modal run --env staging experiments/modal_map_validation.py --maps 24
"""
from pathlib import Path
import json
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-map-validation')
retained = modal.Volume.from_name('polytopia-structured-rl-retained', create_if_missing=False)
results = modal.Volume.from_name('polytopia-structured-rl', create_if_missing=False)
U3142 = '/retained/checkpoints/distributed-final-v1/dist-c460div-v1-update-3142/final.npz'
LEAGUE96 = '/retained/checkpoints/low256-league-rivals-v1/update-96-f998d15e220c/latest.npz'


@app.function(image=image, cpu=32, memory=65536, timeout=5400, volumes={'/retained': retained, '/results': results})
def validate(kind: str, matchup: str, seed_base: int, maps: int, candidate: str = U3142, tag: str = ''):
    from polytopia_bot.rl.cpu_arena import match
    from polytopia_bot.rl.runner import write_json
    if matchup == 'selfplay':
        summary = match(candidate, candidate, range(seed_base, seed_base + maps), games=4, deterministic=False,
                        processes=min(2 * maps, 62), maps=kind)
    else:
        summary = match(candidate, LEAGUE96, range(seed_base, seed_base + maps), games=4, deterministic=True,
                        processes=min(2 * maps, 62), maps=kind)
    out = {k: v for k, v in summary.items() if k != 'details'}
    out.update(kind=kind, matchup=matchup, seed_base=seed_base, maps=maps)
    path = Path('/results/maps/validation') / f'{tag}{kind}-{matchup}-{seed_base}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(out, details=summary['details']))
    results.commit()
    return out


@app.local_entrypoint()
def main(maps: int = 24, seed_base: int = 306600, candidate: str = U3142, tag: str = '', matchups: str = 'selfplay,vs-league96'):
    # Mirrors polytopia_bot.maps.stream.TYPES (not importable here: the local client has no JAX).
    TYPES = ('classic', 'drylands', 'lakes', 'pangea', 'continents', 'archipelago', 'waterworld', 'highlands',
             'ridge', 'jungle')
    jobs = [(kind, matchup, seed_base + 100 * k, maps, candidate, tag) for k, kind in enumerate(TYPES) for matchup in matchups.split(',')]
    rows = list(validate.starmap(jobs))
    out = ROOT / f'artifacts/map-variants/validation{"-" + tag if tag else ""}.json'
    out.write_text(json.dumps(rows, indent=2))
    for r in rows:
        m = r['by_map'][r['kind']]
        print(f"{r['kind']:12s} {r['matchup']:12s} W{r['wins']:3d} L{r['losses']:3d} trunc {r['truncations']:3d}/{r['games']}"
              f"  turns {m['mean_turns']:5.1f}  captures {m['mean_captures']:5.1f}  port-games {m['games_with_ports']:3d}"
              f"  seat1 {m['seat1_win_rate']:.2f}")
