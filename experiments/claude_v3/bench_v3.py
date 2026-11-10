"""Model-only GPU cost of one architecture arm: the GPU-resident trainer's learner step and rollout call, no engine.

    python bench_v3.py '{"name": "v3-6x128", "architecture": "transformer-v3", "width": 128, "blocks": 6}'

Learner: one 16,384-row PPO minibatch as gpu_resident.train_step runs it (grad_accum micro-batches in a lax.scan,
value_and_grad of the trainer's loss: PPO surrogate + entropy + .5 x the critic's .5 MSE, one ppo.adam_step over
(params, critic)). Rollout: one set's fused policy sample + critic value on `rollout_rows` rows (V2: policy trunk +
separate critic trunk; V3: transformer_v3.sample_and_value, one public encoder pass). Inputs are random features of
the real widths (the trainer's per-row view_features encode is identical for every arm and is excluded here; the
end-to-end runs include it). Prints `RESULT {json}`: median seconds per minibatch / rollout call, rows/s, parameter
counts and the device's peak memory.
"""
import json
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


def _features(key, rows, board, glob, size):
    from polytopia_bot.rl.observation import Features
    a, b = jax.random.split(key)
    return Features(jax.random.bernoulli(a, .2, (rows, size, size, board)).astype(jnp.float32),
                    jax.random.normal(b, (rows, glob)))


def _engine_masks(key, rows, size):
    from polytopia_bot.gpu.actions import Masks
    from polytopia_bot.rl.actions import leaf_shapes
    shapes = leaf_shapes(size)
    keys = jax.random.split(key, len(shapes))
    return Masks(*(jax.random.bernoulli(k, .5, (rows,) + s) for k, s in zip(keys, shapes)))


def _path_masks(key, rows, size):
    from polytopia_bot.rl.actions import PathMasks
    n = size * size
    a, b, c = jax.random.split(key, 3)
    return PathMasks(jax.random.bernoulli(a, .5, (rows, 17)).at[:, 0].set(True),
                     jax.random.bernoulli(b, .5, (rows, n)).at[:, 3].set(True),
                     jax.random.bernoulli(c, .5, (rows, max(n, 128))).at[:, 5].set(True))


def _timed(fn, repeats):
    out = fn()
    jax.block_until_ready(out)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t)
    return times, out


def run(arm: dict) -> dict:
    from polytopia_bot.rl import network, ppo, transformer_v3
    from polytopia_bot.rl.actions import Choice
    from polytopia_bot.rl.distributed import _critic_template, _template_params, feature_shapes
    from polytopia_bot.rl.observation import OBS_VERSION, Features
    size = int(arm.get('size', 14))
    spec = dict(size=size, blocks=int(arm['blocks']), width=int(arm['width']), architecture=arm['architecture'],
                obs_version=OBS_VERSION)
    v3 = transformer_v3.is_v3(spec['architecture'])
    board, glob = feature_shapes(2, OBS_VERSION)
    params, critic = _template_params(**spec), _critic_template(**spec)
    n_params = int(sum(x.size for x in jax.tree.leaves(params)))
    n_critic = int(sum(x.size for x in jax.tree.leaves(critic)))
    minibatch, micro = int(arm.get('minibatch', 16384)), int(arm.get('micro', 1024))
    accum = minibatch // micro
    key = jax.random.key(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    own = _features(k1, minibatch, board, glob, size)
    opp = _features(k2, minibatch, board, glob, size)
    full = Features(jnp.concatenate((own.board, opp.board), -1), jnp.concatenate((own.global_state, opp.global_state), -1))
    choice = Choice(jnp.zeros(minibatch, jnp.int32), jnp.full(minibatch, 3, jnp.int32), jnp.full(minibatch, 5, jnp.int32))
    data = dict(features=own, full_features=full, masks=_path_masks(k3, minibatch, size), choice=choice,
                logp=jnp.full(minibatch, -3.), advantage=jax.random.normal(k4, (minibatch,)),
                returns=jax.random.normal(k5, (minibatch,)) * .3)

    def loss(trainable, mb):
        p, c = trainable
        if v3:
            logp, v, ent = jax.vmap(transformer_v3.evaluate, in_axes=(None, None, 0, 0, 0))(
                p, c, mb['full_features'], mb['choice'], mb['masks'])
            total, _ = ppo.policy_objective(logp, ent, mb, clip=.2, entropy_coef=.01)
        else:
            total, _ = ppo.loss(p, mb, clip=.2, entropy_coef=.01, value_coef=0.)
            v = jax.vmap(lambda x: network.trunk(c, x)[2])(mb['full_features'])
        return total + .5 * (.5 * jnp.mean((v - mb['returns']) ** 2))

    @jax.jit
    def train_step(trainable, opt, dev):
        def part(acc, j):
            mb = jax.tree.map(lambda a: jax.lax.dynamic_slice_in_dim(a, j * micro, micro), dev)
            return jax.tree.map(jnp.add, acc, jax.grad(loss)(trainable, mb)), None
        grads, _ = jax.lax.scan(part, jax.tree.map(jnp.zeros_like, trainable), jnp.arange(accum))
        grads = jax.tree.map(lambda g: g / accum, grads)
        trainable, opt, norm = ppo.adam_step(trainable, grads, opt, learning_rate=3e-4)
        return trainable, opt, norm

    trainable = (params, critic)
    opt = ppo.adam_init(trainable)
    state = dict(t=trainable, o=opt)

    def learner_once():
        state['t'], state['o'], norm = train_step(state['t'], state['o'], data)
        return norm
    t_compile = time.perf_counter()
    learner_times, norm = _timed(learner_once, int(arm.get('repeats', 6)))
    compile_and_run = time.perf_counter() - t_compile - sum(learner_times)

    rows = int(arm.get('rollout_rows', 1024))
    r_own = jax.tree.map(lambda a: a[:rows], own)
    r_full = jax.tree.map(lambda a: a[:rows], full)
    r_masks = _engine_masks(jax.random.key(9), rows, size)

    @jax.jit
    def rollout(p, c, key):
        keys = jax.random.split(key, rows)
        if v3:
            choice, _, logp, value = jax.vmap(
                lambda x, m, k: transformer_v3.sample_and_value(p, c, x, m, k, size=size))(r_full, r_masks, keys)
        else:
            choice, _, logp, _ = jax.vmap(lambda x, m, k: network.sample(p, x, m, k, size=size))(r_own, r_masks, keys)
            value = jax.vmap(lambda x: network.trunk(c, x)[2])(r_full)
        return choice, logp, value
    p, c = state['t']
    rollout_times, _ = _timed(lambda: rollout(p, c, jax.random.key(3)), int(arm.get('repeats', 6)) * 3)
    stats = jax.devices()[0].memory_stats() or {}
    learner_s = statistics.median(learner_times)
    rollout_s = statistics.median(rollout_times)
    return dict(name=arm['name'], spec=spec, params=n_params, critic_params=n_critic, minibatch=minibatch, micro=micro,
                learner_s_per_minibatch=learner_s, learner_rows_per_s=minibatch / learner_s,
                rollout_s_per_call=rollout_s, rollout_rows=rows, rollout_rows_per_s=rows / rollout_s,
                # per trained row the GPU-resident trainer runs one rollout call row and one learner row (epochs 1)
                model_s_per_row=learner_s / minibatch + rollout_s / rows,
                learner_times=learner_times, rollout_times=rollout_times, first_call_s=compile_and_run,
                grad_norm=float(norm), finite=bool(np.isfinite(float(norm))),
                peak_gib=round(stats.get('peak_bytes_in_use', 0) / 2 ** 30, 2))


if __name__ == '__main__':
    result = run(json.loads(sys.argv[1]))
    print('RESULT ' + json.dumps(result), flush=True)
