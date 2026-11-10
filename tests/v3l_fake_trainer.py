"""A stand-in for flagship_harness.py in the supervisor tests: same CLI and exit codes, no GPU.

Behaviour per harness faults (plus 'complete_after' updates in the trainer args as --updates N):
  nan:N        exit 70 at update `update` while fewer than `times` NaN rows exist in v3l/nan.jsonl
  crash        exit 1 at update `update` while fewer than `times` resumes exist
  leak         exit 71 at update `update`
  illegal      exit 72 at update `update`
  slow         sleep `seconds` per update
Writes GOOD.json at every update (stage update-N) and training-complete.json at the end; honours the stop file.
"""
import json
import os
import sys
import time
from pathlib import Path


def main():
    argv = sys.argv[1:]
    cut = argv.index('--')
    h = json.loads(argv[argv.index('--harness-json') + 1])
    args = argv[cut + 1:]
    updates = int(args[args.index('--updates') + 1]) if '--updates' in args else 5
    rank = int(args[args.index('--process-id') + 1]) if '--process-id' in args else 0
    root = Path(h['run_dir'])
    v3l = root / 'v3l'
    good = root / 'GOOD.json'
    start = json.loads(good.read_text())['update'] if good.exists() else 0

    def rows(name, key=None):
        p = v3l / name
        if not p.exists():
            return []
        out = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return [r for r in out if key is None or r.get('why') == key]

    for u in range(start + 1, updates + 1):
        for f in h['faults']:
            if u < f.get('update', 1):
                continue
            if f['kind'] == 'nan' and len(rows('nan.jsonl')) < f.get('times', 1):
                print(f'nonfinite at {u}', flush=True)
                sys.exit(70)
            if f['kind'] == 'crash' and int(f.get('rank', 0)) == rank and len(rows('resumes.jsonl')) < f.get('times', 1):
                if rank:
                    (root / '.rank_crashed').write_text(str(rank))   # rank 0 then blocks, as in a real collective
                sys.exit(1)
            if f['kind'] == 'leak' and rank == 0:
                sys.exit(71)
            if f['kind'] == 'illegal':
                sys.exit(72)
            if f['kind'] == 'slow':
                time.sleep(f.get('seconds', .1))
        if rank == 0 and (root / '.rank_crashed').exists():
            (root / '.rank_crashed').unlink()      # a real rank 0 would block in the next collective
            time.sleep(3600)
        print(json.dumps(dict(stage='update', update=u, samples=u * 10, elapsed=u, samples_per_second=10., nonok=0)),
              flush=True)
        if rank:
            if h.get('stop_file') and os.path.exists(h['stop_file']):
                sys.exit(0)
            continue
        good.write_text(json.dumps(dict(update=u, stage=f'update-{u}', dir=f'ckpt/update-{u}')))
        sf = h.get('snap_file')
        if sf and os.path.exists(sf):
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'claude_v3l'))
            from policy import PolicyState
            st = PolicyState(root, dict(run_id=h['run_id'], stage=h['stage']))
            done = {r['label'] for r in st.usd_snapshots()}
            for label in Path(sf).read_text().split():
                if label not in done:
                    st.record_usd_snapshot(label, u)
                    st.event('dollar_snapshot', label=label, update=u)
            os.remove(sf)
        if h.get('stop_file') and os.path.exists(h['stop_file']):
            (root / f'paused-u{u}.json').write_text('{}')
            sys.exit(0)
    if rank == 0:
        (root / 'training-complete.json').write_text(json.dumps(dict(updates=updates)))
    sys.exit(0)


if __name__ == '__main__':
    main()
