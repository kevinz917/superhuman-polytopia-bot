"""Micro-benchmark of the coded-view decode on one GPU (Curio, 2026-10-31): how long does CodedRows.finish's device
work take on its own, and does accumulating in int32 (native atomics) instead of int16 (emulated) change it?

    CUDA_VISIBLE_DEVICES=4 python experiments/curio/codec_bench.py [--rows 41000] [--repeats 5]

Builds realistic synthetic coded messages (rl.assembly_bench.synthetic_messages, consecutive rows mostly equal),
feeds them to a one-device learner's CodedRows, and times finish() + block_until_ready, for the current kernel and
for an int32-accumulator variant; checks both decode the same views.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=41000)
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cuda')
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import view_codec
    from polytopia_bot.rl.assembly_bench import synthetic_messages
    from polytopia_bot.rl.view_codec import encode, CodedRows
    from test_a6_assembly import _learner, _width
    from test_view_codec import _realistic
    import tempfile
    learner = _learner(Path(tempfile.mkdtemp()) / 'bench', coded_views=True, stream_views=False,
                       batch=8192, minibatch=1024)
    width = _width(learner)
    rng = np.random.default_rng(0)
    msgs = []
    while sum(len(b['logp']) for b, _ in msgs) < args.rows:
        msgs += _realistic(synthetic_messages(rng, width, 196, 20, rows=(300, 512), omni=True, segment=32), rng)
    coded = [encode(dict(b), dict(s)) for b, s in msgs]
    print(dict(width=width, messages=len(coded), rows=sum(len(b['logp']) for b, _ in msgs),
               nnz=sum(len(b['view_coo_row']) for b, _ in coded)), flush=True)

    def run(label):
        times = []
        for r in range(args.repeats):
            rows = CodedRows(learner)
            seg_off, dst = 0, 0
            for (b, s), (bo, so) in zip(coded, msgs):
                b, s = dict(b), dict(s)
                part = {k: b.pop(k) for k in list(b) if k.startswith('view_coo_') or k == 'view_shape'}
                bpart = {k: s.pop(k) for k in list(s) if k.startswith('boot_coo_') or k == 'boot_shape'}
                fresh = np.ones(len(b['logp']), bool)
                rows.add(part, b['seg'], b['pos'], fresh, bpart, s['length'], s['terminal'])
            n = rows.fresh_rows
            n_seg = rows.segs
            pad = lambda x: -(-x // 8192) * 8192
            t = time.monotonic()
            views, boots = rows.finish(n, pad(n), n_seg, pad(n_seg))
            jax.block_until_ready((views, boots))
            times.append(time.monotonic() - t)
        print(label, dict(seconds=[round(x, 3) for x in times], timing=rows.timing), flush=True)
        return np.asarray(views['view']), np.asarray(views.get('opp_view'))

    base = run('int16')
    orig = view_codec._shard_fn

    def shard_fn_int32(slots, length, rows_local, segs_local, two_seats):
        key = ('i32', slots, length, rows_local, segs_local, two_seats)
        if key not in view_codec._SHARD_FNS:
            def build(row_slot, row_pos, coo, bcoo, src, dst, base, bdst, n_streams, width):
                c_row, c_stream, c_col, c_val = coo
                r = c_row.astype(jnp.int32)
                acc = jnp.zeros((slots, length, n_streams, width), jnp.int32).at[
                    row_slot[r], row_pos[r], c_stream.astype(jnp.int32), c_col.astype(jnp.int32)].add(c_val.astype(jnp.int32))
                dense = jnp.cumsum(acc, axis=1, dtype=jnp.int32)
                b_slot, b_stream, b_col, b_val = bcoo
                boot = dense[base[0], base[1]].at[b_slot, b_stream.astype(jnp.int32), b_col.astype(jnp.int32)].add(
                    b_val.astype(jnp.int32)).astype(jnp.int16)
                take = dense[row_slot[src], row_pos[src]].astype(jnp.int16)
                out_view = jnp.zeros((rows_local, width), jnp.int16).at[dst].set(take[:, 0], mode='drop')
                out_bview = jnp.zeros((segs_local, width), jnp.int16).at[bdst].set(boot[:, 0], mode='drop')
                out_opp = out_bopp = None
                if n_streams > 1:
                    opp, bopp = (take[:, 1], boot[:, 1]) if two_seats else (take[:, 1:], boot[:, 1:])
                    out_opp = jnp.zeros((rows_local,) + opp.shape[1:], jnp.int16).at[dst].set(opp, mode='drop')
                    out_bopp = jnp.zeros((segs_local,) + bopp.shape[1:], jnp.int16).at[bdst].set(bopp, mode='drop')
                return out_view, out_opp, out_bview, out_bopp
            view_codec._SHARD_FNS[key] = jax.jit(build, static_argnums=(8, 9))
        return view_codec._SHARD_FNS[key]

    view_codec._shard_fn = shard_fn_int32
    alt = run('int32')
    view_codec._shard_fn = orig
    print('equal', bool(np.array_equal(base[0], alt[0]) and (base[1] is None or np.array_equal(base[1], alt[1]))), flush=True)


if __name__ == '__main__':
    main()
