"""Render map-variant previews and per-preset statistics on Modal (nothing runs locally).

    modal run --env staging experiments/modal_map_variants.py --out artifacts/map-variants
"""
from pathlib import Path
import io
import json
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('fonts-dejavu-core')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'pillow==11.2.1')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-map-variants')


@app.function(image=image, cpu=4, memory=8192, timeout=1800)
def build(seeds_per_preset: int = 3, stat_seeds: int = 500):
    import numpy as np
    from polytopia_bot.maps.variants import PRESETS, Layout, generate
    from polytopia_bot.maps.render import render, legend, sheet
    out = {}
    def png(img):
        buf = io.BytesIO()
        img.save(buf, 'PNG', optimize=True)
        return buf.getvalue()
    # Frozen engine generator (today's training maps) for comparison.
    from polytopia_bot.gpu import mapgen
    frozen = []
    for seed in (200001, 200002, 200003):
        lay = mapgen.layout(np.int32(seed), size=14)
        lay = Layout(*(np.asarray(a) for a in lay))
        t = lay.terrain
        frozen.append(render(lay, 14, f'Current generator · seed {seed}',
                             f'land {np.mean(t >= 3):.0%}  water {np.mean(t <= 2):.0%}  mountain {np.mean(t == 4):.0%}  forest {np.mean(t == 5):.0%}'))
    out['current.png'] = png(sheet(frozen, 3, legend()))
    stats = {}
    all_first = []
    for name, preset in PRESETS.items():
        imgs = []
        for k in range(seeds_per_preset):
            lay, info = generate(1000 + k, name)
            s = info['stats']
            sub = (f"land {s['land']:.0%}  water {s['water']:.0%}  mountain {s['mountain']:.0%}  forest {s['forest']:.0%}"
                   f"  villages {s['villages']}  landmasses {s['landmasses']}"
                   f"  {'same' if s['capitals_connected'] else 'separate'} landmass")
            img = render(lay, 14, f'{preset.title} · seed {1000 + k}', sub)
            imgs.append(img)
            if k == 0:
                all_first.append(img)
        out[f'{name}.png'] = png(sheet(imgs, 3, legend()))
        rows = [generate(s, name)[1]['stats'] for s in range(5000, 5000 + stat_seeds)]
        stats[name] = dict(title=preset.title, note=preset.note, **{
            k: dict(mean=float(np.mean([r[k] for r in rows])), p10=float(np.percentile([r[k] for r in rows], 10)),
                    p90=float(np.percentile([r[k] for r in rows], 90)))
            for k in ('land', 'water', 'mountain', 'forest', 'villages', 'landmasses', 'capital_distance')},
            capitals_connected=float(np.mean([r['capitals_connected'] for r in rows])))
    # current generator stats for comparison
    rows = []
    for seed in range(5000, 5000 + min(stat_seeds, 100)):
        t = np.asarray(mapgen.layout(np.int32(seed), size=14).terrain)
        rows.append(dict(land=float(np.mean(t >= 3)), water=float(np.mean(t <= 2)),
                         mountain=float(np.mean(t == 4)), forest=float(np.mean(t == 5))))
    stats['current'] = {k: dict(mean=float(np.mean([r[k] for r in rows]))) for k in rows[0]}
    out['overview.png'] = png(sheet(frozen[:1] + all_first, 4, legend()))
    out['stats.json'] = json.dumps(stats, indent=2).encode()
    return out


@app.local_entrypoint()
def main(out: str = 'artifacts/map-variants'):
    folder = ROOT / out
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in build.remote().items():
        (folder / name).write_bytes(data)
        print(name, len(data))
