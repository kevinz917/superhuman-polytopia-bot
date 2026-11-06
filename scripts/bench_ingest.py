"""Learner-side ingestion cost per sample: zstd decompress + unpickle of a real actor message,
and batch assembly (np.concatenate of 512-row chunks into a 262,144-sample batch).

Rows are real packed views from a few ticks of engine-rs (realistic entropy). Run on Modal only.
argv: rows_per_message batch
"""
import json
import pickle
import sys
import time


def main(rows_per_message, batch):
    from polytopia_bot.rl.distributed import _env_setup, _template_params, _compress, _decompress, stack_rows
    _env_setup()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from polytopia_bot.rl import network
    from polytopia_bot.rl.rust_engine import RustGames
    size, games = 14, 24
    engine = RustGames(size, games, both_views=True)
    params = _template_params(size)
    act = jax.jit(jax.vmap(lambda p, x, m, k: network.sample(p, x, m, k, size=size), in_axes=(None, 0, 0, 0)))
    rng = np.random.default_rng(0)
    seeds = jnp.asarray(rng.integers(200000, 300000, games).astype(np.int32))
    states, prepared = engine.reset(seeds)
    key = jax.random.key(0)
    rows = []
    n = size * size
    while len(rows) < rows_per_message:
        key, k = jax.random.split(key)
        choice, path, logp, values = act(params, prepared.features, prepared.masks, jax.random.split(k, games))
        view = np.asarray(prepared.view)
        opp = np.asarray(prepared.opp_view)
        fam, src, tgt = (np.packbits(np.asarray(a), axis=-1) for a in path)
        c = [np.asarray(a) for a in choice]
        for i in range(games):
            rows.append(dict(view=view[i], opp_view=opp[i], family_mask=fam[i], source_mask=src[i], target_mask=tgt[i],
                             choice=np.array([c[0][i], c[1][i], c[2][i]], np.int32), logp=np.float32(logp[i]),
                             value=np.float32(values[i]), version=np.int32(0), advantage=np.float32(0.),
                             returns=np.float32(0.)))
        tick, _ = engine.tick(states, choice, seeds)
        states, prepared = tick.states, tick.prepared
    rows = rows[:rows_per_message]
    batch_rows = stack_rows(rows)
    msg = dict(type='data', actor=0, have=[], batch=batch_rows, stats={})
    raw = pickle.dumps(msg, protocol=5)
    body = _compress(raw)
    bytes_per_row = {k: v.nbytes // rows_per_message for k, v in batch_rows.items()}
    def timed(fn, reps):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t) / reps
    t_dec = timed(lambda: _decompress(body), 30)
    t_unp = timed(lambda: pickle.loads(raw), 30)
    t_both = timed(lambda: pickle.loads(_decompress(body)), 30)
    t_comp = timed(lambda: _compress(pickle.dumps(msg, protocol=5)), 10)
    per_row_us = 1e6 * t_both / rows_per_message
    # assembly: chunks of rows_per_message rows -> one batch
    chunks = [batch_rows] * (batch // rows_per_message)
    t0 = time.perf_counter()
    data = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    t_cat = time.perf_counter() - t0
    fresh = np.ones(batch, bool)
    t0 = time.perf_counter()
    _ = {k: v[fresh] for k, v in data.items()}
    t_mask = time.perf_counter() - t0
    total_bytes = sum(v.nbytes for v in data.values())
    return dict(rows=rows_per_message, raw_bytes=len(raw), compressed_bytes=len(body),
                compressed_bytes_per_row=len(body) / rows_per_message, bytes_per_row=bytes_per_row,
                decompress_ms=1e3 * t_dec, unpickle_ms=1e3 * t_unp, both_ms=1e3 * t_both, actor_pack_ms=1e3 * t_comp,
                ingest_us_per_row=per_row_us, ingest_rows_per_s_per_core=1e6 / per_row_us,
                batch=batch, batch_bytes_gib=total_bytes / 2**30, concat_s=t_cat, mask_copy_s=t_mask)


if __name__ == '__main__':
    print(json.dumps(main(int(sys.argv[1]), int(sys.argv[2]))))
