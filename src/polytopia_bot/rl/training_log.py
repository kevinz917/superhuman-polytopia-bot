"""The distributed learner's per-update training log (no jax import; the analysis scripts use it too).

While a run is live the learner appends rows to `training.jsonl` (one JSON object per line) at each checkpoint,
instead of rewriting the whole `training.json` list, which grew to 27 MB and an O(n^2) checkpoint stall over a 7-h
run (docs/overnight-bottlenecks-2026-10-22.md). A finished run also writes `training.json` once, so readers of
finished runs are unchanged. `read` takes a run folder or either file.
"""
import json
import os
from pathlib import Path

JSONL, JSON = 'training.jsonl', 'training.json'


def append(path, rows):
    """Append rows to a .jsonl file (created if missing); one write call, flushed to the file system."""
    if not rows:
        return
    with open(path, 'a') as f:
        f.write(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
        f.flush()
        os.fsync(f.fileno())


def rewrite(path, rows):
    """Replace a .jsonl file with exactly these rows (a resume truncating to its checkpoint)."""
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    tmp.replace(path)


def read(path):
    """Rows of a run's training log: a folder (its training.jsonl, else training.json), a .jsonl or a .json file.
    A .jsonl whose last line is cut short (a container killed mid-append) loses that line only."""
    path = Path(path)
    if path.is_dir():
        path = path / JSONL if (path / JSONL).exists() else path / JSON
    if not path.exists():
        return []
    if path.suffix != '.jsonl':
        return json.loads(path.read_text())
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows
