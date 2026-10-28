import io
import json
import numpy as np
import pytest
from polytopia_bot.gpu.snapshot import dumps, fields, loads

SIZE = 14


def _played(seed=3, steps=60):
    """An engine-rs game after `steps` random legal actions, as (RustGames, single-game State)."""
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.rl.search import first
    rng = np.random.default_rng(seed)
    g = RustGames(SIZE, 1)
    g.reset(np.array([seed], np.int32))
    for _ in range(steps):
        g.batch.step(*(np.array([v], np.int32) for v in _random(g, rng)))
    return g, first(g.states())


def _random(g, rng):
    from polytopia_bot.rl.actions import shapes
    from polytopia_bot.rl.rust_engine import split_masks
    mats = [np.asarray(m[0]).reshape(shp) for m, shp in zip(split_masks(g.batch.legal_masks(), SIZE), shapes(SIZE))]
    f = int(rng.choice([f for f in range(16) if mats[f].any()]))   # no resign
    s = int(rng.choice(np.flatnonzero(mats[f].any(1))))
    return f, s, int(rng.choice(np.flatnonzero(mats[f][s])))


def board():
    return _played()[1]


def test_private_snapshot_restores_every_field_and_replays_exactly():
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.rl.search import one
    g, s = _played()
    restored = loads(dumps(s, profile='native'), profile='native')
    for (k, a), (_, b) in zip(fields(s), fields(restored), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=k)
    h = RustGames(SIZE, 1)
    h.load(one(restored))
    rng = np.random.default_rng(9)
    for _ in range(40):
        choice = [np.array([v], np.int32) for v in _random(g, rng)]
        np.testing.assert_array_equal(g.batch.step(*choice), h.batch.step(*choice))
        ea, eb = g.batch.export(), h.batch.export()
        for k in ea:
            np.testing.assert_array_equal(np.asarray(ea[k]), np.asarray(eb[k]), err_msg=k)


def test_snapshot_rejects_wrong_profile_and_missing_fields():
    saved=dumps(board(),profile='native')
    with pytest.raises(ValueError,match='profile'):
        loads(saved,profile='polyfish')
    with np.load(io.BytesIO(saved),allow_pickle=False) as original:
        arrays={name:original[name] for name in original.files if name!='seed'}
    buf=io.BytesIO()
    np.savez(buf,**arrays)
    with pytest.raises(ValueError,match='field set'):
        loads(buf.getvalue(),profile='native')


def test_snapshot_rejects_shape_and_implementation_changes():
    saved=dumps(board(),profile='native')
    with np.load(io.BytesIO(saved),allow_pickle=False) as original:
        arrays={name:original[name] for name in original.files}
    arrays['seed']=np.zeros(2,np.uint32)
    buf=io.BytesIO(); np.savez(buf,**arrays)
    with pytest.raises(ValueError,match='shape/dtype'):
        loads(buf.getvalue(),profile='native')
    meta=json.loads(arrays['metadata'].tobytes())
    meta['implementation_sha256']='different'
    arrays['metadata']=np.frombuffer(json.dumps(meta).encode(),np.uint8)
    buf=io.BytesIO(); np.savez(buf,**arrays)
    with pytest.raises(ValueError,match='implementation'):
        loads(buf.getvalue(),profile='native')
