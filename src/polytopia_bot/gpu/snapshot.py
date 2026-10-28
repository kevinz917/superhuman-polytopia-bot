"""Private, versioned semantic snapshots. Host I/O never enters the GPU loop.

These files contain the hidden world and RNG. They are diagnostics/replay inputs,
not policy observations. No Python pickles or executable objects are accepted.
"""
import hashlib
import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import ENGINE_REVISION
from .state import State, empty_state

SCHEMA = 1


# Earlier engine digests whose games are bitwise those of this implementation on their boards.
# bardur-1v1-v1 (11x11/14x14): v2 only adds the 18x18 board and its round limit; the map fixture
# tests check v1 boards stay bitwise identical.
COMPATIBLE_DIGESTS = {'7b76e65b74afcd8d4732712b12a2f9e001fafc1a5b2899982b971a9fd78cb134': 'bardur-1v1-v1'}


def compatible_digest(digest):
    """True for this implementation's digest or a declared compatible predecessor."""
    return digest == implementation_digest() or digest in COMPATIBLE_DIGESTS


def implementation_digest():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for p in sorted([*root.glob('*.py'), root / 'rules.json']):
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def fields(state):
    for name in State._fields:
        value = getattr(state, name)
        if isinstance(value, tuple):
            for field, array in zip(value._fields, value, strict=True):
                yield f'{name}.{field}', array
        else:
            yield name, value


def dumps(state, *, profile):
    if profile not in ('native', 'polyfish'):
        raise ValueError('unknown rules profile')
    if state.tiles.terrain.ndim != 1:
        raise ValueError('snapshot expects one game; unstack batches first')
    size = int(state.tiles.terrain.size ** .5)
    meta = dict(schema=SCHEMA, size=size, profile=profile, oracle_revision=ENGINE_REVISION,
                implementation_sha256=implementation_digest())
    arrays = {k: np.asarray(v) for k, v in fields(state)}
    arrays['metadata'] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def loads(payload, *, profile):
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        meta = json.loads(archive['metadata'].tobytes())
        expected = dict(schema=SCHEMA, profile=profile, oracle_revision=ENGINE_REVISION)
        if any(meta.get(k) != v for k, v in expected.items()) or not compatible_digest(meta.get('implementation_sha256')):
            raise ValueError('snapshot schema, profile, or implementation differs')
        size = meta.get('size')
        if type(size) is not int or not 2 <= size <= 32:
            raise ValueError('invalid snapshot board size')
        state = empty_state(size)
        template = dict(fields(state))
        if set(archive.files) != {*template, 'metadata'}:
            raise ValueError('snapshot field set differs')
        arrays = {}
        for name, ref in template.items():
            array = archive[name]
            if array.shape != ref.shape or array.dtype != ref.dtype:
                raise ValueError(f'snapshot shape/dtype differs: {name}')
            arrays[name] = jnp.asarray(array)
        values = []
        for name in State._fields:
            value = getattr(state, name)
            values.append(type(value)(*(arrays[f'{name}.{f}'] for f in value._fields))
                          if isinstance(value, tuple) else arrays[name])
        return State(*values)
