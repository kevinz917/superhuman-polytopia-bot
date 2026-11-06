"""Per-phase actor tick profile on a Modal CPU container (scripts/bench_actor_profile.py).

    modal run --env staging experiments/modal_actor_profile.py --cpu 16 --specs "16x24:classic:0:0,..."
"""
from pathlib import Path
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax==0.11.2', 'numpy>=2.1', 'zstandard==0.23.0')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'scripts', '/workspace/scripts', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-actor-profile')


@app.function(image=cpu_image, cpu=16, memory=32768, timeout=1800, retries=0)
def profile(processes: int, games: int, seconds: int, maps: str, omni: int, opp_share: float):
    import subprocess
    out = subprocess.run(['python', '/workspace/scripts/bench_actor_profile.py', str(processes), str(games),
                          str(seconds), maps, str(omni), str(opp_share)], capture_output=True, text=True,
                         cwd='/workspace')
    cpu = subprocess.run(['bash', '-c', "grep -m1 'model name' /proc/cpuinfo"], capture_output=True, text=True).stdout.strip()
    return (out.stdout.strip().splitlines()[-1] if out.returncode == 0 else out.stderr[-4000:]) + f' | {cpu}'


@app.local_entrypoint()
def main(cpu: int = 16, seconds: int = 60,
         specs: str = '16x24:classic:0:0,16x24:proposed-v1:0:0,16x24:proposed-v1:1:0,16x24:proposed-v1:1:0.2,16x48:proposed-v1:1:0.2'):
    calls = []
    for spec in specs.split(','):
        pg, maps, omni, opp = spec.split(':')
        p, g = map(int, pg.split('x'))
        calls.append(profile.with_options(cpu=cpu, memory=max(32768, 2048 * p)).spawn(p, g, seconds, maps, int(omni), float(opp)))
    for c in calls:
        print(c.get())


@app.function(image=cpu_image, cpu=16, memory=65536, timeout=3600, retries=0)
def profile_many(specs: str, seconds: int):
    """All specs sequentially in ONE container, so arms share the host."""
    import subprocess
    cpu = subprocess.run(['bash', '-c', "grep -m1 'model name' /proc/cpuinfo; nproc; lscpu | grep -i 'thread\\|core\\|socket' | head -4"],
                         capture_output=True, text=True).stdout.strip().replace('\n', ' / ')
    rows = [cpu]
    for spec in specs.split(','):
        pg, maps, omni, opp = spec.split(':')
        p, g = pg.split('x')
        out = subprocess.run(['python', '/workspace/scripts/bench_actor_profile.py', p, g, str(seconds), maps, omni, opp],
                             capture_output=True, text=True, cwd='/workspace')
        rows.append(out.stdout.strip().splitlines()[-1] if out.returncode == 0 else out.stderr[-2000:])
    return rows


@app.local_entrypoint()
def many(seconds: int = 45, containers: int = 2,
         specs: str = '16x24:proposed-v1:1:0.2,16x48:proposed-v1:1:0.2,16x96:proposed-v1:1:0.2,16x24:classic:0:0,16x24:proposed-v1:0:0,16x24:proposed-v1:1:0'):
    calls = [profile_many.spawn(specs, seconds) for _ in range(containers)]
    for i, c in enumerate(calls):
        for row in c.get():
            print(f'container {i}: {row}')
