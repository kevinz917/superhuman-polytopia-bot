"""Play T3 (Plato update 96070) against another bot from the published files, headless, on the browser engine.

    python3 plato-t3/arena/play.py --opponent weights/hero-1v1/final.npz --boards 2 --out runs/plato-t3/smoke.json

Needs Node 20+ and, when the opponent is a checkpoint (.npz), Python with NumPy to convert it to the browser
format. Every board is played twice with the seats swapped; both sides play Bardur. The result rows go to --out and
a win/draw/loss table is printed (summarize.py prints the same table from a saved file).

Opponents:
  a checkpoint .npz with its .json next to it (for example weights/hero-1v1/final.npz or weights/plato/final.npz);
  a browser weight file .bin; or "t3" for a T3 mirror match (then the second side samples its moves).

Panels:
  classic  the engine's classic 14x14 dryland board for seeds --seed-base .. (the default)
  real     boards of the published bank (the real game's map generator): --size 11/14/16/18/20, all six map types,
           up to four Bardur-mirror boards per map type and size. A CNN opponent plays 14x14 only.

Decodes: "joint" (the single most likely whole move; how T3 is served), "greedy" (most likely choice head by head;
how the CNN bots are served) or "sampled" (temperature 1). The default is joint for T3, greedy for a CNN opponent
and joint for a transformer opponent.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
T3 = ROOT / 'weights/plato-t3/plato.bin'
RUNTIME = HERE.parent / 'runtime'
MAPS = ['dryland', 'lakes', 'pangea', 'continents', 'archipelago', 'waterworld']
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'tools'))


def _seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) & 0x7fffffff


def games(panel, board, key, a, b, reps):
    """Both seat orders of one board for the pairing a vs b; a and b are (bot, decode)."""
    rows = []
    for rep in range(reps if 'sampled' in (a[1], b[1]) else 1):
        for first, second in ((a, b), (b, a)):
            gid = f'{panel}|{key}|{first[0]}:{first[1]}|{second[0]}:{second[1]}|{rep}'
            rows.append(dict(board, id=gid, panel=panel, board=key, bots=[first[0], second[0]],
                             decodes=[first[1], second[1]], sample_seed=_seed(gid)))
    return rows


def build(panel, size, boards, seed_base, a, b, reps):
    specs = []
    if panel == 'classic':
        for i in range(boards):
            specs += games('classic14', dict(source='dryland', seed=seed_base + i), f'd{seed_base + i}', a, b, reps)
    else:
        for m in MAPS:
            for index in range(min(boards, 4)):
                specs += games(f'real{size}', dict(source='bank', size=size, map=m, index=index), f'{m}{index}', a, b, reps)
    return specs


def magic(path):
    return Path(path).read_bytes()[:4]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--opponent', required=True, help='checkpoint .npz, browser .bin, or "t3"')
    ap.add_argument('--opponent-name', default='', help='label in the results (default: from the path)')
    ap.add_argument('--opponent-decode', choices=['joint', 'greedy', 'sampled'], default='')
    ap.add_argument('--t3-decode', choices=['joint', 'greedy', 'sampled'], default='joint')
    ap.add_argument('--panel', choices=['classic', 'real'], default='classic')
    ap.add_argument('--size', type=int, choices=[11, 14, 16, 18, 20], default=14, help='board side for --panel real')
    ap.add_argument('--boards', type=int, default=2, help='boards (classic) or boards per map type, at most 4 (real)')
    ap.add_argument('--seed-base', type=int, default=1_900_000_000)
    ap.add_argument('--reps', type=int, default=1, help='repeats per seat order when a side samples')
    ap.add_argument('--wasm', default=str(RUNTIME / 'engine_wasm.wasm'), help='engine build to play on')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        assets = Path(tmp) / 'assets'
        assets.mkdir()
        shutil.copy(args.wasm, assets / 'engine_wasm.wasm')
        shutil.copy(T3, assets / 't3.bin')
        for f in RUNTIME.glob('bank-*.json'):
            shutil.copy(f, assets / f.name)
        if args.opponent == 't3':
            name, decode = 't3', args.opponent_decode or 'sampled'
            if decode == args.t3_decode != 'sampled':
                raise SystemExit('a mirror match with the same deterministic decode on both sides plays one game twice; '
                                 'leave --opponent-decode at sampled')
        else:
            source = Path(args.opponent)
            name = args.opponent_name or (source.parent.name if source.stem == 'final' else source.stem)
            if name == 't3':
                raise SystemExit('pick another --opponent-name')
            if source.suffix == '.npz':
                import export_browser_weights
                export_browser_weights.main(source, assets / f'{name}.bin')
            else:
                shutil.copy(source, assets / f'{name}.bin')
            cnn = magic(assets / f'{name}.bin') == b'PTB1'
            decode = args.opponent_decode or ('greedy' if cnn else 'joint')
            if cnn and args.panel == 'real' and args.size != 14:
                raise SystemExit('a CNN opponent plays 14x14 only')
        specs = build(args.panel, args.size, args.boards, args.seed_base, ('t3', args.t3_decode), (name, decode), args.reps)
        (Path(tmp) / 'specs.json').write_text(json.dumps(specs))
        print(f'{len(specs)} games: t3 ({args.t3_decode}) vs {name} ({decode}), panel {args.panel}', flush=True)
        done = subprocess.run(['node', str(HERE / 'arena.mjs'), str(assets), str(Path(tmp) / 'specs.json'), str(Path(tmp) / 'out.json')])
        if done.returncode != 0:
            raise SystemExit(f'arena.mjs exited with {done.returncode}')
        rows = json.loads((Path(tmp) / 'out.json').read_text())
    out.write_text(json.dumps(rows))
    import summarize
    summarize.report(rows)
    print(f'rows -> {out}')
    return 1 if any(r['outcome'] in ('error', 'step_cap') for r in rows) else 0


if __name__ == '__main__':
    sys.exit(main())
