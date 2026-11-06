"""Write a structured-PPO checkpoint as the flat float32 file the in-browser bot loads (engine-wasm/src/net.rs).

    .venv/bin/python scripts/export_browser_weights.py weights/champion/final.npz [web/public/bot/champion.bin]

Layout: b'PTB1', u32 tensor count, then per tensor u32 rank, u32 dims..., followed by every tensor's float32
data (little endian) in checkpoint leaf order p0, p1, ... (jax.tree.flatten of the normalized-v2 parameter dict).
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Leaf order of network.init(architecture='normalized-v2', blocks=2) under jax.tree.flatten (sorted dict keys).
LEAVES = ('blocks.0.a', 'blocks.0.b', 'blocks.1.a', 'blocks.1.b', 'critic_hidden.b', 'critic_hidden.w',
          'family.b', 'family.w', 'family_embedding', 'global.b', 'global.w', 'norm.bias', 'norm.scale',
          'parameter_embedding', 'query.b', 'query.w', 'source.b', 'source.w', 'stem', 'value.b', 'value.w')


def main(source, target=Path(__file__).resolve().parents[1] / 'web/public/bot/champion.bin'):
    source, target = Path(source), Path(target)
    meta = json.loads(source.with_suffix('.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != meta['sha256']:
        raise SystemExit('checkpoint SHA-256 mismatch')
    config = meta['config']
    if (config['architecture'], config['blocks'], config['width'], config['size'], config.get('extra_blocks', 0)) != \
            ('normalized-v2', 2, 64, 14, 0):
        raise SystemExit('the browser network supports normalized-v2, 2 blocks x 64, 14x14 only')
    with np.load(source, allow_pickle=False) as archive:
        tensors = [np.ascontiguousarray(archive[f'p{i}'], dtype='<f4') for i in range(len(LEAVES))]
        if f'p{len(LEAVES)}' in archive.files:
            raise SystemExit('unexpected extra parameter leaves')
    if sum(t.size for t in tensors) != meta['parameter_elements']:
        raise SystemExit('parameter count differs from the checkpoint metadata')
    header = b'PTB1' + struct.pack('<I', len(tensors))
    for t in tensors:
        header += struct.pack('<I', t.ndim) + struct.pack(f'<{t.ndim}I', *t.shape)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(header + b''.join(t.tobytes() for t in tensors))
    target.with_suffix('.json').write_text(json.dumps({
        'source_sha256': meta['sha256'], 'stage': meta.get('stage'), 'ruleset': meta.get('ruleset'),
        'simulator_sha256': meta['simulator_sha256'], 'parameter_elements': meta['parameter_elements'],
        'no_resign': config['no_resign'], 'leaves': LEAVES}, indent=1) + '\n')
    print(f'{target}: {target.stat().st_size} bytes, {meta["parameter_elements"]} parameters')


if __name__ == '__main__':
    main(*sys.argv[1:3])
