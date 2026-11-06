"""Batched policy inference on a GPU: decode packed int16 views + sample, per batch size.

Sizes an inference server for the actors (views and packed masks in, choices/logp out). Synthetic inputs
(zeros views, all-legal masks); the forward cost does not depend on values. Run on Modal only.
argv: comma-separated batch sizes
"""
import json
import sys
import time


def main(batches):
    from polytopia_bot.rl.distributed import _template_params
    import jax
    import jax.numpy as jnp
    import numpy as np
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import PathMasks
    from polytopia_bot.rl.cpu_engine import view_layout, view_features
    from polytopia_bot.gpu.actions import Masks
    size = 14
    n = size * size
    layout = view_layout(size)
    width = sum(int(np.prod(s)) for _, s in layout)
    params = _template_params(size)
    # Masks as the actor holds them (gpu.actions.Masks tuple); shapes from legal_masks
    shapes = jax.eval_shape(lambda: Masks(jnp.zeros((n, n), bool), jnp.zeros((n, n), bool), jnp.zeros((n, 128), bool),
                                          jnp.zeros((n, 128), bool), jnp.zeros(128, bool), jnp.zeros(n, bool),
                                          jnp.zeros((n, 3), bool), jnp.zeros((n, 9), bool), jnp.zeros(n, bool),
                                          jnp.zeros(n, bool), jnp.zeros(n, bool), jnp.zeros(n, bool), jnp.zeros(n, bool),
                                          jnp.zeros(n, bool), jnp.zeros(6, bool), jnp.zeros((), bool), jnp.zeros((), bool)))
    mask_bits = sum(int(np.prod(s.shape)) for s in shapes)

    def serve(p, views, masks_packed, key):
        masks = jax.tree.map(lambda bits, s: jnp.unpackbits(bits, axis=-1)[..., :int(np.prod(s.shape))].reshape((bits.shape[0],) + s.shape).astype(bool),
                             masks_packed, shapes)
        feats = jax.vmap(lambda v: view_features(v, layout, size))(views)
        keys = jax.random.split(key, views.shape[0])
        choice, path, logp, value = jax.vmap(lambda x, m, k: network.sample(p, x, m, k, size=size))(feats, masks, keys)
        return choice, logp
    served = jax.jit(serve)
    rows = []
    for b in batches:
        views = np.zeros((b, width), np.int16)
        packed = jax.tree.map(lambda s: np.packbits(np.ones((b,) + s.shape, bool).reshape(b, -1), axis=-1), shapes)
        key = jax.random.key(0)
        out = served(params, views, packed, key)
        jax.block_until_ready(out)
        t = time.perf_counter()
        reps = 10
        for i in range(reps):
            key = jax.random.fold_in(key, i)
            out = served(params, jnp.asarray(views), jax.tree.map(jnp.asarray, packed), key)
            jax.block_until_ready(out)
        ms = 1e3 * (time.perf_counter() - t) / reps
        # device_put only
        t = time.perf_counter()
        for _ in range(reps):
            jax.block_until_ready((jax.device_put(views), jax.tree.map(jax.device_put, packed)))
        put_ms = 1e3 * (time.perf_counter() - t) / reps
        rows.append(dict(batch=b, ms_per_call=ms, put_ms=put_ms, samples_per_s=b / (ms / 1e3), us_per_sample=1e3 * ms / b,
                         bytes_per_sample=2 * width + mask_bits // 8))
    return dict(device=str(jax.devices()[0]), view_int16=width, mask_bits=mask_bits, rows=rows)


if __name__ == '__main__':
    print(json.dumps(main([int(x) for x in sys.argv[1].split(',')])))
