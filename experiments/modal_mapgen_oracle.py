"""Build the corrected-sampling Polyfish layout oracle on Modal and emit map fixtures.

    modal run --env staging experiments/modal_mapgen_oracle.py

Same cases as scripts/generate_map_fixtures.py plus 18x18 (MapSize::Large). The vendor
checkout (pinned a9f90ab5) is copied read-only; nothing is built on the local machine.
Writes tests/fixtures/corrected-drylands.json.gz after checking that the rebuilt oracle
reproduces every existing 11x11/14x14 case.
"""
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (*range(1, 17), 0, -1, 2147483647, -2147483648)
image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('curl', 'build-essential', 'pkg-config', 'libssl-dev')
         .run_commands('curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'))
if modal.is_local():
    # The vendor checkout lives in the main working tree (worktrees share it).
    MAIN = Path(subprocess.run(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()).parent
    VENDOR = next(p for p in (ROOT / 'vendor/Polyfish/polyfish-rs', MAIN / 'vendor/Polyfish/polyfish-rs') if p.is_dir())
    image = (image.add_local_dir(VENDOR, '/w/vendor/Polyfish/polyfish-rs',
                                 ignore=['target/**', 'replays/**', '**/__pycache__/**', '*.npz', '*.safetensors'])
             .add_local_dir(ROOT / 'tools/mapgen-oracle', '/w/tools/mapgen-oracle', ignore=['target/**']))
app = modal.App('polytopia-mapgen-oracle')


@app.function(image=image, cpu=16, memory=32768, timeout=3600)
def oracle(sizes: list):
    env = {'PATH': '/root/.cargo/bin:/usr/bin:/bin', 'CARGO_TARGET_DIR': '/tmp/target', 'HOME': '/root'}
    build = subprocess.run(['cargo', 'build', '--locked', '--manifest-path', '/w/tools/mapgen-oracle/Cargo.toml'],
                           capture_output=True, text=True, env=env)
    if build.returncode:
        raise RuntimeError(build.stderr[-4000:])
    cases = []
    with subprocess.Popen(['/tmp/target/debug/mapgen-oracle'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True) as process:
        for size in sizes:
            for seed in SEEDS:
                process.stdin.write(json.dumps(dict(size=size, seed=seed)) + '\n')
                process.stdin.flush()
                raw = json.loads(process.stdout.readline())
                tiles = [raw['tiles'][str(i)] for i in range(size * size)]
                cases.append(dict(size=size, seed=seed, terrain=[t['type'] for t in tiles],
                    structure=[(raw['structures'].get(str(i)) or {}).get('type', 0) for i in range(size * size)],
                    resource=[(raw['resources'].get(str(i)) or {}).get('type', 0) for i in range(size * size)],
                    capitals=[next(i for i, t in enumerate(tiles) if t.get('capitalOf') == p) for p in (1, 2)]))
        process.stdin.close()
        process.wait()
    return cases


@app.local_entrypoint()
def main():
    import sys
    sys.path.insert(0, str(ROOT / 'src'))
    from polytopia_bot.gpu import ENGINE_REVISION
    path = ROOT / 'tests/fixtures/corrected-drylands.json.gz'
    old = json.loads(gzip.decompress(path.read_bytes()))
    cases = oracle.remote([11, 14, 18])
    key = lambda c: (c['size'], c['seed'])
    previous = {key(c): c for c in old['cases']}
    rebuilt = {key(c): c for c in cases if c['size'] in (11, 14)}
    if rebuilt != previous:
        raise SystemExit('rebuilt oracle does not reproduce the existing 11/14 fixtures; not writing')
    data = dict(old, engine_revision=ENGINE_REVISION,
                source_sha256=hashlib.sha256((VENDOR / 'src/mapgen.rs').read_bytes()).hexdigest(), cases=cases)
    if data['source_sha256'] != old['source_sha256']:
        raise SystemExit('vendor mapgen.rs differs from the fixture source')
    path.write_bytes(gzip.compress(json.dumps(data, separators=(',', ':')).encode(), mtime=0))
    print(f'{len(cases)} layouts ({sum(c["size"] == 18 for c in cases)} at 18x18) saved to {path}')
