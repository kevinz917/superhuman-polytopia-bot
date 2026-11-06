"""How large is a checkpoint's trunk-block pooling bias on real positions? The farsight diagnosis (2026-11-01): in the
layout-1 network the bias added inside every block had an RMS 3-5x that of the convolution it joined, so one
board-wide vector dominated every tile's LayerNorm (network.BLOCK_POOL_LAYOUTS). This probe plays sampled self-play
with the checkpoint, keeps positions every `every` actions, and reports per pooling layer the mean / max of
bias_rms / conv_rms, the mean RMS of each, and the gate (layout 2). It also measures what the pooling carries: the
KL of the family distribution and the value change when the pooling is switched off (layout 1: zero weights; layout
2: zero gates), which a function-preserving conversion to layout 2 would have to absorb (it cannot: it starts from
the switched-off function).

    python experiments/daytona/pool_probe.py CKPT.npz [--games 8] [--steps 1600] [--every 40] [--seed 700000] [--out X.json]

Also the keeper's `hero.py pool-probe RUN` (node.py hero_pool_probe on the hub) for a live run's latest.npz.
"""
import argparse
import json
import os
import time

import numpy as np


def switched_off(params):
    """The same network with its trunk-block pooling off: layout 1 zero weights and bias, layout 2 zero gates."""
    import jax.numpy as jnp
    blocks = []
    for block in params['blocks']:
        if len(block) == 3:
            layer = block[2]
            off = dict(layer, gate=jnp.zeros_like(layer['gate'])) if 'gate' in layer else \
                dict(layer, w=jnp.zeros_like(layer['w']), b=jnp.zeros_like(layer['b']))
            block = (block[0], block[1], off)
        blocks.append(block)
    return dict(params, blocks=tuple(blocks))


def probe(ckpt, games=8, steps=1600, every=40, seed=700000, size=14, no_peace=True):
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import family_mask
    from polytopia_bot.rl.distributed import checkpoint_template, network_spec
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.maps.stream import MapSource
    started = time.time()
    params = load_checkpoint(ckpt, checkpoint_template(ckpt))
    spec = network_spec(ckpt)
    layout = int(spec.get('block_pool', 0) or 0)
    pooled = [i for i, b in enumerate(params['blocks']) if len(b) == 3]
    off = switched_off(params)
    engine = RustGames(size, games, map_batch=True, max_turns=200, no_peace=no_peace)
    source = MapSource('proposed-v1', size)
    states, prepared = engine.reset(source.batch(range(seed, seed + games))[0])
    act = jax.jit(jax.vmap(lambda x, m, k: network.sample(params, x, m, k, size=size)[0], in_axes=(0, 0, 0)))

    def measure(x, masks):
        stats = []
        tiles, g, value = network.trunk(params, x, pool_stats=stats)
        tiles2, g2, value2 = network.trunk(off, x)
        fm = family_mask(masks)
        lp, lp2 = network.log_distribution(network.dense(g, params['family']), fm), \
            network.log_distribution(network.dense(g2, params['family']), fm)
        kl = jnp.sum(jnp.where(fm, jnp.exp(lp) * (lp - lp2), 0.))
        rows = [jnp.stack([s['conv_rms'], s['bias_rms'], s['gate'] if s['gate'] is not None else jnp.float32(np.nan)])
                for s in stats]
        return jnp.stack(rows) if rows else jnp.zeros((0, 3)), kl, value - value2, \
            jnp.sqrt(jnp.mean((tiles - tiles2) ** 2)) / (jnp.sqrt(jnp.mean(tiles2 ** 2)) + 1e-8)
    measure = jax.jit(jax.vmap(measure))
    done = np.zeros(games, bool)
    key = jax.random.key(seed)
    stats, kls, dvs, dtiles, kept = [], [], [], [], 0
    for step in range(steps):
        live = ~done
        if not live.any():
            break
        if step % every == 0:
            rows, kl, dv, dt = measure(prepared.features, prepared.masks)
            for i in np.nonzero(live)[0]:
                stats.append(np.asarray(rows[i]))
                kls.append(float(kl[i]))
                dvs.append(float(dv[i]))
                dtiles.append(float(dt[i]))
                kept += 1
        key, sub = jax.random.split(key)
        choice = act(prepared.features, prepared.masks, jax.random.split(sub, games))
        tick, _ = engine.tick(states, choice, source.batch(range(seed + games, seed + 2 * games))[0])
        done |= np.asarray(tick.done)
        states, prepared = tick.states, tick.prepared
    stats = np.stack(stats) if stats else np.zeros((0, len(pooled), 3))
    layers = []
    for j, block in enumerate(pooled):
        conv, bias, gate = stats[:, j, 0], stats[:, j, 1], stats[:, j, 2]
        ratio = bias / np.maximum(conv, 1e-8)
        layers.append(dict(block=block, conv_rms=round(float(conv.mean()), 4), bias_rms=round(float(bias.mean()), 4),
                           ratio_mean=round(float(ratio.mean()), 4), ratio_max=round(float(ratio.max()), 4),
                           gate=None if np.isnan(gate).all() else round(float(gate.mean()), 5)))
    return dict(checkpoint=str(ckpt), stage=json.loads(open(str(ckpt)[:-4] + '.json').read()).get('stage'),
                block_pool=layout, positions=kept, games=games, layers=layers,
                switched_off=dict(family_kl_mean=round(float(np.mean(kls)), 5), family_kl_max=round(float(np.max(kls)), 5),
                                  value_diff_rms=round(float(np.sqrt(np.mean(np.square(dvs)))), 5),
                                  tiles_rel_diff_mean=round(float(np.mean(dtiles)), 5)) if kept else None,
                wall_s=round(time.time() - started, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--games', type=int, default=8)
    ap.add_argument('--steps', type=int, default=1600)
    ap.add_argument('--every', type=int, default=40)
    ap.add_argument('--seed', type=int, default=700000)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    row = probe(args.ckpt, games=args.games, steps=args.steps, every=args.every, seed=args.seed)
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(row, indent=1))
    print(json.dumps(row))


if __name__ == '__main__':
    main()
