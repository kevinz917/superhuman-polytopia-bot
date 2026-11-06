"""Prepare identical paired map hashes in the cloud, without model inference."""
import json
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.12').pip_install('jax==0.11.2','numpy>=2.1')
    .env({'PYTHONPATH':'/workspace/src','JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'1'})
    .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'configs','/workspace/configs'))
app=modal.App('polytopia-paired-map-preparation',image=image)


@app.function(cpu=(2,2),memory=2048,timeout=300,retries=0)
def prepare(start,count):
    import hashlib
    import jax
    import jax.numpy as jnp
    from polytopia_bot.gpu.mapgen import layout
    from polytopia_bot.learning.splits import board_hashes
    rules=json.loads(Path('/workspace/configs/ruleset-v1.json').read_text())
    for name,sha in rules['source_sha256'].items():
        if hashlib.sha256((Path('/workspace')/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('frozen simulator source changed')
    generate=jax.jit(lambda s:layout(s,size=14));jobs=[]
    for seed in range(start,start+count):
        sha=board_hashes(jax.device_get(generate(jnp.int32(seed))),14)[0]
        for seat in [1,2]:
            jobs.append(dict(id=f'seed-{seed}-seat-{seat}',seed=seed,board_hash=sha,
                seat=seat,candidate='qwen35-4b-unadapted',strategy='',opponent='balanced',rng_index=len(jobs)))
    return jobs


@app.local_entrypoint()
def main(run_id: str,start: int=100691,count: int=50):
    root=ROOT/'runs'/run_id;root.mkdir(exist_ok=True)
    jobs=prepare.remote(start,count)
    (root/'paired-jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    print(json.dumps(dict(maps=count,games_per_mode=len(jobs))))
