"""Subprocess side of experiments/modal_precision_gate.py (never run locally).

rl.network reads POLYTOPIA_MATMUL_PRECISION / POLYTOPIA_HEAD_PRECISION at import, so every precision mode
runs in a fresh interpreter whose environment the caller sets:

    python precision_gate_worker.py rows  CHECKPOINT SIZE OUT.npz PROCESSES GAMES TICKS STRIDE SEED
    python precision_gate_worker.py gate  CHECKPOINT SIZE ROWS.npz OUT.json
    python precision_gate_worker.py bench PARENT CONFIG_JSON EPOCHS OUT.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np


def f16_params(params):
    """The weights an actor runs at version 0: Learner._leaves sends float16, actors widen back to float32."""
    import jax
    import jax.numpy as jnp
    return jax.tree.map(lambda a: jnp.asarray(np.asarray(a).astype(np.float16), jnp.float32)
                        if jnp.issubdtype(a.dtype, jnp.floating) else a, params)


def rows_process(checkpoint, size, games, ticks, stride, seed, core, out):
    """One actor-like process: sampled self-play with the version-0 actor weights, every `stride`-th tick
    recorded in the actor's transport format (int16 view, packed path masks, choice, actor logp)."""
    os.sched_setaffinity(0, {core})
    from polytopia_bot.rl.distributed import _env_setup, checkpoint_template
    _env_setup()
    import jax
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import Choice
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.maps.stream import MapSource
    from polytopia_bot.maps.state import stack as stack_maps
    params = f16_params(load_checkpoint(checkpoint, checkpoint_template(checkpoint)))
    source = MapSource('proposed-v1', size)
    engine = RustGames(size, games, map_batch=True)
    rng = np.random.default_rng(seed)
    lane_maps = [None] * games
    def refill(lanes):
        for i in lanes:
            s = int(rng.integers(200000, 300000))
            lane_maps[i] = (s, source.layout(s)[1])
    refill(range(games))
    states, prepared = engine.reset(stack_maps(lane_maps))
    refill(range(games))
    act = jax.jit(jax.vmap(lambda p, x, m, k: network.sample(p, x, m, k, size=size), in_axes=(None, 0, 0, 0)))
    key = jax.random.key(seed)
    rows = {k: [] for k in ('view', 'family_mask', 'source_mask', 'target_mask', 'choice', 'logp')}
    ended = 0
    for t in range(ticks):
        key, k = jax.random.split(key)
        choice, path, logp, _ = act(params, prepared.features, prepared.masks, jax.random.split(k, games))
        if t % stride == 0:
            rows['view'].append(np.asarray(prepared.view).astype(np.int16))
            for name, a in zip(('family_mask', 'source_mask', 'target_mask'), path):
                rows[name].append(np.packbits(np.asarray(a), axis=-1))
            rows['choice'].append(np.stack([np.asarray(c, np.int32) for c in choice], -1))
            rows['logp'].append(np.asarray(logp, np.float32))
        tick, status = engine.tick(states, Choice(*choice), stack_maps(lane_maps))
        if np.any(np.asarray(status) != 0):
            raise RuntimeError('illegal policy action')
        done = np.nonzero(np.asarray(tick.done))[0]
        ended += len(done)
        refill(done)
        states, prepared = tick.states, tick.prepared
    out.put(dict({k: np.concatenate(v) for k, v in rows.items()}, games_ended=ended))


def make_rows(checkpoint, size, path, processes, games, ticks, stride, seed):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    cores = sorted(os.sched_getaffinity(0))
    t0 = time.monotonic()
    ps = [ctx.Process(target=rows_process, args=(checkpoint, size, games, ticks, stride, seed * 1000 + i,
                                                  cores[i % len(cores)], q)) for i in range(processes)]
    for p in ps:
        p.start()
    parts = [q.get() for _ in ps]
    for p in ps:
        p.join()
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != 'games_ended'}
    np.savez(path, **data)
    return dict(rows=len(data['logp']), games_ended=sum(p['games_ended'] for p in parts), seconds=time.monotonic() - t0,
                checkpoint=checkpoint, size=size, mean_actor_logp=float(data['logp'].mean()))


def log_ratio_stats(log_r):
    log_r = np.asarray(log_r, np.float64)
    r = np.exp(log_r)
    a = np.abs(log_r)
    return dict(mean_ratio=float(r.mean()), mean_k3=float(np.mean(r - log_r - 1)), max_abs_log_r=float(a.max()),
                p999_abs_log_r=float(np.quantile(a, .999)), mean_abs_log_r=float(a.mean()),
                frac_outside_clip=float(np.mean(np.abs(r - 1) > .2)), frac_abs_log_r_gt_1e3=float(np.mean(a > 1e-3)))


def gate(checkpoint, size, rows_path, out):
    """Learner-side recomputation of the actor rows, as Learner._build_trainer.to_batch does, in 4096-row
    minibatches on the GPU under whatever precision the environment sets."""
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import Choice, PathMasks
    from polytopia_bot.rl.cpu_engine import view_features, view_layout
    from polytopia_bot.rl.distributed import checkpoint_template
    from polytopia_bot.rl.runner import load_checkpoint
    fp32 = load_checkpoint(checkpoint, checkpoint_template(checkpoint))
    actor_weights = f16_params(fp32)
    layout, n = view_layout(size), size * size
    unpack = lambda a, w: jnp.unpackbits(a, axis=-1)[..., :w].astype(bool)

    @jax.jit
    def learner_logp(params, view, fam, src, tgt, choice):
        x = jax.vmap(lambda v: view_features(v, layout, size))(view)
        masks = PathMasks(unpack(fam, 17), unpack(src, n), unpack(tgt, max(n, 128)))
        c = Choice(choice[:, 0], choice[:, 1], choice[:, 2])
        return jax.vmap(lambda f, c_, m: network.evaluate(params, f, c_, m)[0])(x, c, masks)

    with np.load(rows_path) as z:
        data = {k: z[k] for k in z.files}
    total, mb = len(data['logp']), 4096
    def recompute(params):
        out_lp = []
        for i in range(0, total, mb):
            chunk = {k: v[i:i + mb] for k, v in data.items()}
            pad = mb - len(chunk['logp'])
            if pad:  # keep one compiled shape; padded rows are dropped
                chunk = {k: np.concatenate([v, np.repeat(v[:1], pad, 0)]) for k, v in chunk.items()}
            lp = learner_logp(params, *(jnp.asarray(chunk[k]) for k in
                                        ('view', 'family_mask', 'source_mask', 'target_mask', 'choice')))
            out_lp.append(np.asarray(lp)[:mb - pad])
        return np.concatenate(out_lp)
    t0 = time.monotonic()
    lp = recompute(actor_weights)
    seconds = time.monotonic() - t0
    result = dict(checkpoint=checkpoint, size=size, rows=total, device=str(jax.devices()[0]),
                  precision=network.PRECISION.name, head_precision=network.HEAD_PRECISION.name,
                  seconds=seconds, **log_ratio_stats(lp - data['logp']),
                  # Context, not the gate: the learner's own fp32 weights vs the float16 copy the actor ran.
                  vs_fp32_weights=log_ratio_stats(recompute(fp32) - data['logp']))
    result['passes'] = result['mean_k3'] < 1e-4 and result['max_abs_log_r'] < .02
    with open(out, 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result))


def bench(parent, config, epochs, out):
    import tempfile
    from pathlib import Path
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    import jax
    learner = Learner(Path(tempfile.mkdtemp()) / 'bench', LearnerConfig(**config), parent,
                      log=lambda s: print(s, flush=True))
    rows = learner.benchmark(epochs)
    steady = rows[1:] or rows
    result = dict(parent=parent, config=config, precision=network.PRECISION.name,
                  head_precision=network.HEAD_PRECISION.name, devices=learner.devices,
                  device=str(jax.devices()[0]), rows=rows,
                  # one PPO epoch = batch samples; sample-epochs/s per GPU, compile epoch excluded
                  sample_epochs_per_s=float(np.median([r['samples_per_second'] for r in steady])) / learner.devices)
    with open(out, 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'rows':
        print(json.dumps(make_rows(args[0], int(args[1]), args[2], *map(int, args[3:8]))))
    elif cmd == 'gate':
        gate(args[0], int(args[1]), args[2], args[3])
    elif cmd == 'bench':
        bench(args[0], json.loads(args[1]), int(args[2]), args[3])
    else:
        raise SystemExit(f'unknown command {cmd}')
