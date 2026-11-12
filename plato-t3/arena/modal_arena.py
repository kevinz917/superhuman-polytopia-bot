"""Optional: run play.py on a Modal CPU container instead of this machine.

    modal run plato-t3/arena/modal_arena.py --opponent weights/hero-1v1/final.npz --boards 2 --out runs/plato-t3/smoke.json

Takes play.py's options (run from the repository root; --opponent is a path inside this repository or "t3"). The
container gets this directory and weights/, runs the same play.py command, and the result rows are written to --out
here. Nothing is stored on Modal.
"""
from pathlib import Path
import json
import modal

image = modal.Image.from_registry('node:22-bookworm-slim', add_python='3.12').pip_install('numpy==2.2.6')
if modal.is_local():   # inside the container this file is alone at /root; the tree is at /repo
    ROOT = Path(__file__).resolve().parents[2]
    image = (image.add_local_dir(ROOT / 'plato-t3', '/repo/plato-t3', ignore=['**/target/**', '**/__pycache__/**'])
             .add_local_dir(ROOT / 'weights', '/repo/weights'))
app = modal.App('plato-t3-arena')


@app.function(image=image, cpu=2, memory=4096, timeout=6 * 3600)
def play(argv: list):
    import subprocess
    command = ['python3', 'plato-t3/arena/play.py', *argv, '--out', '/tmp/out.json']
    done = subprocess.run(command, cwd='/repo', capture_output=True, text=True)
    rows = json.load(open('/tmp/out.json')) if Path('/tmp/out.json').exists() else []
    return {'command': ' '.join(command), 'returncode': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr[-6000:], 'rows': rows}


@app.local_entrypoint()
def main(opponent: str, out: str, panel: str = 'classic', size: int = 14, boards: int = 2, seed_base: int = 1_900_000_000,
         reps: int = 1, opponent_name: str = '', opponent_decode: str = '', t3_decode: str = 'joint'):
    argv = ['--opponent', opponent, '--panel', panel, '--size', str(size), '--boards', str(boards),
            '--seed-base', str(seed_base), '--reps', str(reps), '--t3-decode', t3_decode]
    if opponent_name:
        argv += ['--opponent-name', opponent_name]
    if opponent_decode:
        argv += ['--opponent-decode', opponent_decode]
    result = play.remote(argv)
    print(result['command'])
    print(result['stdout'], end='')
    if result['returncode'] != 0:
        print(result['stderr'])
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result['rows']))
    print(f'{len(result["rows"])} rows -> {path} (exit {result["returncode"]})')
