"""Claude V3 (session E) copy of experiments/modal_map_table.py writing to its own volume claude-v3-results.

Precomputed map tables for the vector actor (rl/vector_actor.MapTable).

The training map mixes draw seeds uniformly from [map_seed_low, map_seed_high) (100k seeds) and the board is a pure
function of the seed (maps.stream.MapSource), so every board an actor can ever play is generated once here and
stored on the volume; actors then look boards up instead of running the Python variant generators (~60 us of CPU
per training sample at the hero runs' game lengths).

    modal run --env staging experiments/modal_map_table.py::build --mix proposed-v1 --size 14 --players 2
"""
from pathlib import Path
import json
import os
import modal

# locally the repo root; inside a container the module sits at /root and the sources at /workspace
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('curl', 'build-essential', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0', 'maturin>=1.7,<2')
    .add_local_dir(ROOT / 'engine-rs', '/workspace/engine-rs', copy=True, ignore=['**/target/**'])
    .run_commands('cd /workspace/engine-rs && PATH=/root/.cargo/bin:$PATH maturin build --release -o /workspace/wheels '
                  '&& pip install /workspace/wheels/*.whl')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**']))
app = modal.App('claude-v3-map-table')
results = modal.Volume.from_name('claude-v3-results', create_if_missing=True)


@app.function(image=image, cpu=1, memory=2048, timeout=3600, retries=1, volumes={'/results': results})
def chunk(mix: str, size: int, players: int, lo: int, hi: int, parts_dir: str):
    import time
    import numpy as np
    from polytopia_bot.maps.stream import MapSource
    source = MapSource(mix, size, players=players)
    n = size * size
    out = dict(kind=np.zeros(hi - lo, np.int8), terrain=np.zeros((hi - lo, n), np.int16),
               structure=np.zeros((hi - lo, n), np.int16), resource=np.zeros((hi - lo, n), np.int16),
               capitals=np.zeros((hi - lo, players), np.int16))
    t = time.perf_counter()
    for i, seed in enumerate(range(lo, hi)):
        kind, layout = source.layout(seed)
        out['kind'][i] = source.types.index(kind)
        for k in ('terrain', 'structure', 'resource', 'capitals'):
            a = np.asarray(getattr(layout, k))
            if a.min() < -32768 or a.max() > 32767:
                raise ValueError(f'{k} out of int16 range')
            out[k][i] = a
    target = Path('/results') / parts_dir / f'{lo}.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **out)
    results.commit()
    return lo, source.types, (time.perf_counter() - t) / (hi - lo)


@app.function(image=image, cpu=2, memory=8192, timeout=3600, volumes={'/results': results})
def write(path: str, mix: str, size: int, players: int, lo: int, hi: int, starts: list, parts_dir: str, types: list):
    import io
    import shutil
    import numpy as np
    results.reload()
    loaded = [dict(np.load(Path('/results') / parts_dir / f'{s}.npz')) for s in sorted(starts)]
    arrays = {k: np.concatenate([p[k] for p in loaded]) for k in loaded[0]}
    if len(arrays['kind']) != hi - lo:
        raise ValueError('missing seeds')
    # spot check against MapSource
    from polytopia_bot.maps.stream import MapSource
    source = MapSource(mix, size, players=players)
    rng = np.random.default_rng(0)
    for seed in rng.integers(lo, hi, 64):
        kind, layout = source.layout(int(seed))
        i = int(seed) - lo
        assert types[arrays['kind'][i]] == kind
        for k in ('terrain', 'structure', 'resource', 'capitals'):
            assert np.array_equal(np.asarray(getattr(layout, k)), arrays[k][i]), (seed, k)
    target = Path('/results') / path
    target.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, lo=np.int64(lo), hi=np.int64(hi), types=np.array(types), **arrays)
    target.write_bytes(buf.getvalue())
    shutil.rmtree(Path('/results') / parts_dir, ignore_errors=True)
    results.commit()
    return dict(path=str(target), bytes=target.stat().st_size, seeds=hi - lo)


def table_path(mix, size, players, lo, hi):
    return f'maps/table-{mix}-{size}-{players}p-{lo}-{hi}.npz'


@app.local_entrypoint()
def build(mix: str = 'proposed-v1', size: int = 14, players: int = 2, lo: int = 200000, hi: int = 300000,
          per: int = 2000):
    starts = list(range(lo, hi, per))
    parts_dir = f'maps/parts-{mix}-{size}-{players}p'
    speeds, types = [], None
    for start, types, speed in chunk.starmap([(mix, size, players, s, min(hi, s + per), parts_dir) for s in starts]):
        speeds.append(speed)
    print(json.dumps(dict(chunks=len(speeds), seconds_per_map_mean=sum(speeds) / len(speeds))), flush=True)
    print(json.dumps(write.remote(table_path(mix, size, players, lo, hi), mix, size, players, lo, hi, starts, parts_dir,
                                  list(types))))
