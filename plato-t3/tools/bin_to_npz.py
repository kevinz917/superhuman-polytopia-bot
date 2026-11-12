"""Rebuild a checkpoint-format .npz from the browser weight file (the inverse of export_browser_weights.py).

    python3 plato-t3/tools/bin_to_npz.py weights/plato-t3/plato.bin OUT/final.npz

The browser file holds every policy tensor in float32, in checkpoint leaf order, so the result is the policy half of
a training checkpoint: arrays p0, p1, ... plus OUT/final.json with the configuration the exporter reads. What it
cannot restore: the privileged critic, the optimizer state, and the original file's bytes (the manifest's
source_sha256 is the hash of the trainer's own .npz, which is not published; this file hashes differently).
Needs only NumPy.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_bin(path):
    data = Path(path).read_bytes()
    magic, (count,) = data[:4], struct.unpack_from('<I', data, 4)
    if magic not in (b'PTB1', b'PTB3'):
        raise SystemExit(f'not a browser weight file: {magic!r}')
    offset, shapes = 8, []
    for _ in range(count):
        (rank,) = struct.unpack_from('<I', data, offset)
        shapes.append(struct.unpack_from(f'<{rank}I', data, offset + 4))
        offset += 4 + 4 * rank
    tensors = []
    for shape in shapes:
        size = int(np.prod(shape, dtype=np.int64))
        tensors.append(np.frombuffer(data, dtype='<f4', count=size, offset=offset).reshape(shape).copy())
        offset += 4 * size
    if offset != len(data):
        raise SystemExit('trailing bytes after the last tensor')
    return magic, tensors


def main(source, target):
    source, target = Path(source), Path(target)
    manifest = json.loads(source.with_suffix('.json').read_text())
    magic, tensors = read_bin(source)
    if len(tensors) != len(manifest['leaves']) or sum(t.size for t in tensors) != manifest['parameter_elements']:
        raise SystemExit('tensor count or parameter count differs from the manifest')
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **{f'p{i}': t for i, t in enumerate(tensors)})
    meta = {
        'schema': 'structured-ppo-v1',
        'config': {'architecture': manifest['architecture'], 'size': 14, 'width': manifest['width'],
                   'blocks': manifest['blocks'], 'extra_blocks': 0, 'obs_version': manifest['obs_version'],
                   'no_resign': manifest['no_resign'], 'players': 2},
        'stage': manifest['stage'], 'ruleset': manifest['ruleset'],
        'simulator_sha256': manifest['simulator_sha256'], 'parameter_elements': manifest['parameter_elements'],
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'reconstructed_from': {'file': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                               'original_checkpoint_sha256': manifest['source_sha256']},
        'leaves': manifest['leaves'],
    }
    target.with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'{target}: {len(tensors)} tensors, {meta["parameter_elements"]} parameters ({magic.decode()})')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:3])
