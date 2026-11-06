"""Trusted map manifest construction; board hashes and seeds never enter prompts."""
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from polytopia_bot.gpu.mapgen import layout, GENERATOR_REVISION
from polytopia_bot.gpu.snapshot import implementation_digest
from .interface import digest

DEFAULT_COUNTS = dict(teacher_dev=20, prompt_search=10, prompt_confirmation=10, train=500, validation=200, test=500)


def board_hashes(board, size):
    channels = [np.asarray(getattr(board, k)).reshape(size, size) for k in ('terrain', 'structure', 'resource')]
    capitals = np.zeros(size * size, np.int32)
    capitals[np.asarray(board.capitals)] = np.arange(1, 3)
    channels.append(capitals.reshape(size, size))
    array = np.stack(channels, axis=-1).astype('<i4')
    raw = hashlib.sha256(array.tobytes()).hexdigest()
    variants = []
    for swapped in (False, True):
        a = array.copy()
        if swapped:
            a[:, :, -1] = np.where(a[:, :, -1] > 0, 3 - a[:, :, -1], 0)
        for rotations in range(4):
            rotated = np.rot90(a, rotations)
            for flipped in (rotated, np.flip(rotated, axis=0)):
                variants.append(hashlib.sha256(flipped.astype('<i4').tobytes()).hexdigest())
    return raw, min(variants)


def reserve(output, *, counts=None, seed_start=100001, size=14):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = DEFAULT_COUNTS if counts is None else counts
    generate = jax.jit(lambda seed: layout(seed, size=size))
    seen, boards, duplicates = set(), {}, []
    seed = seed_start
    for split, count in counts.items():
        boards[split] = []
        while len(boards[split]) < count:
            board = jax.device_get(generate(jnp.int32(seed)))
            raw, symmetric = board_hashes(board, size)
            if symmetric in seen:
                duplicates.append(dict(seed=seed, board_hash=raw, symmetry_hash=symmetric))
            else:
                seen.add(symmetric)
                boards[split].append(dict(seed=seed, board_hash=raw, symmetry_hash=symmetric, seats=[1, 2]))
            seed += 1
        print(f'{split}: {len(boards[split])} reserved', flush=True)
    manifest = dict(schema='polytopia-splits-v1', generator=GENERATOR_REVISION,
        simulator_sha256=implementation_digest(), size=size, counts=counts,
        duplicates_skipped=duplicates, maps=boards)
    manifest['manifest_hash'] = digest(manifest)
    (output / 'maps.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'training-splits.json').write_text(json.dumps({k: [r['board_hash'] for r in v] for k, v in boards.items()}, indent=2) + '\n')
    return manifest
