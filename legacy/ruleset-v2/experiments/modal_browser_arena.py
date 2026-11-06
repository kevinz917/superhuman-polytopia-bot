"""Strength check of the in-browser bot: champion + engine-wasm turn search vs the raw champion, on Modal.

    (cd web && npm run browser:wasm)   # engine-wasm build into web/public/bot/
    modal run --env staging experiments/modal_browser_arena.py --seeds 200 --out artifacts/browser-parity/arena.jsonl

Each seed is played twice (search in seat 1, then seat 2) by web/tests/local_arena.ts under Node, the same
WebAssembly module and weights the browser loads. One game per 1-CPU container. Staging only.
"""
from pathlib import Path
import json
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
NODE = 'node-v24.11.0-linux-x64'
image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('curl', 'xz-utils')
    .run_commands(f'curl -fsSL https://nodejs.org/dist/v24.11.0/{NODE}.tar.xz | tar -xJ -C /opt')
    .add_local_file(ROOT / 'web/src/local/engine.ts', '/workspace/web/src/local/engine.ts')
    .add_local_file(ROOT / 'web/tests/local_arena.ts', '/workspace/web/tests/local_arena.ts')
    .add_local_file(ROOT / 'web/public/bot/engine_wasm.wasm', '/workspace/bot/engine_wasm.wasm')
    .add_local_file(ROOT / 'web/public/bot/champion.bin', '/workspace/bot/champion.bin'))
app = modal.App('polytopia-browser-arena')


@app.function(image=image, cpu=1, memory=1024, timeout=3600, retries=1)
def game(seed: int, seat: int):
    import subprocess
    out = subprocess.run([f'/opt/{NODE}/bin/node', '/workspace/web/tests/local_arena.ts', '/workspace/bot/engine_wasm.wasm',
                          '/workspace/bot/champion.bin', str(seed), str(seat)], capture_output=True, text=True, cwd='/workspace')
    if out.returncode:
        return {'seed': seed, 'search_seat': seat, 'error': out.stderr[-2000:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


@app.local_entrypoint()
def main(seeds: int = 200, seed0: int = 396500, out: str = 'artifacts/browser-parity/arena.jsonl'):
    pairs = [(seed0 + i, seat) for i in range(seeds) for seat in (1, 2)]
    rows = list(game.starmap(pairs, order_outputs=False))
    path = ROOT / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    ok = [r for r in rows if 'error' not in r]
    wins = sum(r['rewards'][r['search_seat'] - 1] > 0 for r in ok)
    losses = sum(r['rewards'][r['search_seat'] - 1] < 0 for r in ok)
    draws = len(ok) - wins - losses
    secs = sum(r['search_seconds'] for r in ok) / max(1, sum(r['searched_turns'] for r in ok))
    print(f'{len(ok)} games ({len(rows) - len(ok)} errors): search {wins}-{losses}-{draws} '
          f'({(wins + draws / 2) / max(1, len(ok)):.1%}), {secs:.2f} s per searched turn on one core')
