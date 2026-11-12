"""Write a structured-PPO checkpoint as the flat float32 file the browser engine loads (engine-wasm/src/net.rs).

    python3 plato-t3/tools/export_browser_weights.py CHECKPOINT/final.npz OUT.bin

Needs only NumPy. CHECKPOINT/final.json (the checkpoint's metadata) must sit next to the .npz.

Layout: b'PTB1', u32 tensor count, then per tensor u32 rank, u32 dims..., followed by every tensor's float32
data (little endian) in checkpoint leaf order p0, p1, ... (jax.tree.flatten of the normalized-v2 parameter dict).
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Leaf order of network.init(architecture='normalized-v2', blocks=B) under jax.tree.flatten (sorted dict keys): the
# residual blocks first, then the heads.
HEAD_LEAVES = ('critic_hidden.b', 'critic_hidden.w', 'family.b', 'family.w', 'family_embedding', 'global.b', 'global.w',
               'norm.bias', 'norm.scale', 'parameter_embedding', 'query.b', 'query.w', 'source.b', 'source.w', 'stem',
               'value.b', 'value.w')


def leaves(blocks):
    return tuple(f'blocks.{i}.{k}' for i in range(blocks) for k in 'ab') + HEAD_LEAVES


def main(source, target):
    source, target = Path(source), Path(target)
    meta = json.loads(source.with_suffix('.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != meta['sha256']:
        raise SystemExit('checkpoint SHA-256 mismatch')
    config = meta['config']
    is_v3 = config['architecture'] == 'transformer-v3-p2-h4'
    if is_v3:
        # observation versions 3 (Plato u85270), 4, 5 and 6 (T3; engine-wasm v3.rs::widths); the tree is
        # built for the 14 x 14 base board (POLYTOPIA_V3_BASE_SIZE=14) and plays 11-20 (v3.rs::trunk)
        if (config['size'], config['width'], config['blocks']) != (14,128,6) or config.get('obs_version') not in (3,4,5,6):
            raise SystemExit('unsupported Plato configuration')
        names = ['family.b','family.w','family_embedding','global_token.b','global_token.w','norm.bias','norm.scale',
                 'parameter_embedding','query.b','query.w','source.b','source.w']
        for i in range(6):
            names += [f'transformer.{i}.{k}' for k in ('attention_norm.bias','attention_norm.scale','ff_in.b','ff_in.w',
                'ff_norm.bias','ff_norm.scale','ff_out.b','ff_out.w','out.b','out.w','qkv.b','qkv.w','relative_bias')]
        names += [f'v3_conv_blocks.{i}.{j}' for i in range(4) for j in range(2)]
        names += ['v3_final_norm.bias','v3_final_norm.scale','v3_patch.b','v3_patch.w','v3_stem.b','v3_stem.w',
                  'v3_unpatch.b','v3_unpatch.w']
        LEAVES = tuple(names)
    else:
        if (config['architecture'], config['size'], config.get('extra_blocks',0), config.get('obs_version') or 1) != \
                ('normalized-v2',14,0,1) or config['width'] not in (64,128) or config.get('global_pool') \
                or config.get('block_pool') or config.get('attention'):
            raise SystemExit('unsupported browser CNN configuration')
        LEAVES = leaves(int(config['blocks']))
    with np.load(source, allow_pickle=False) as archive:
        tensors = [np.ascontiguousarray(archive[f'p{i}'], dtype='<f4') for i in range(len(LEAVES))]
        if f'p{len(LEAVES)}' in archive.files:
            raise SystemExit('unexpected extra parameter leaves')
    if sum(t.size for t in tensors) != meta['parameter_elements']:
        raise SystemExit('parameter count differs from the checkpoint metadata')
    if not all(np.isfinite(t).all() for t in tensors):
        raise SystemExit('non-finite checkpoint tensors')
    header = (b'PTB3' if is_v3 else b'PTB1') + struct.pack('<I', len(tensors))
    for t in tensors:
        header += struct.pack('<I', t.ndim) + struct.pack(f'<{t.ndim}I', *t.shape)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(header + b''.join(t.tobytes() for t in tensors))
    target.with_suffix('.json').write_text(json.dumps({
        'source_sha256': meta['sha256'], 'stage': meta.get('stage'), 'ruleset': meta.get('ruleset'),
        'simulator_sha256': meta['simulator_sha256'], 'parameter_elements': meta['parameter_elements'],
        'architecture': config['architecture'], 'obs_version': config.get('obs_version',1),
        'no_resign': config['no_resign'], 'width': config['width'], 'blocks': config['blocks'],
        'leaves': LEAVES}, indent=1) + '\n')
    print(f'{target}: {target.stat().st_size} bytes, {meta["parameter_elements"]} parameters')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:3])
