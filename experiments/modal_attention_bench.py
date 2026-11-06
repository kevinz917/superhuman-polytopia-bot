"""Learner-step cost of the lookout attention trunk (policy + omniscient critic, forward + backward, one GPU) under
the kernel/remat variants of rl.network, on random obs-v3 features: the 2026-11-01 smoke measured 6.9 s per 262k
batch on 4x RTX PRO 6000 (0.43 s per 4096-row device minibatch) against 1.86 s for the 4x128 conv hero.

  modal run --env staging experiments/modal_attention_bench.py --gpu H100 --rows 4096
"""
import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('jax[cuda12]==0.11.2', 'numpy>=2.1')
    .env({'PYTHONPATH': '/workspace/src', 'JAX_PLATFORMS': 'cuda', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
          'POLYTOPIA_MATMUL_PRECISION': 'DEFAULT', 'POLYTOPIA_HEAD_PRECISION': 'HIGHEST', 'POLYTOPIA_TRUNK_DTYPE': 'bfloat16'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-attention-bench', image=image)


def _bench(rows, variants, nets):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import _critic_template, _template_params, feature_shapes
    from polytopia_bot.rl.observation import Features
    board, glob = feature_shapes(2, 3)
    rng = np.random.default_rng(0)
    out = []
    for name, spec in nets.items():
        policy, critic = _template_params(14, **spec), _critic_template(14, **spec)
        x = Features(board=jnp.asarray(rng.random((rows, 14, 14, board), np.float32)),
                     global_state=jnp.asarray(rng.random((rows, glob), np.float32)))
        xc = Features(board=jnp.asarray(rng.random((rows, 14, 14, 2 * board), np.float32)),
                      global_state=jnp.asarray(rng.random((rows, 2 * glob), np.float32)))
        for impl, remat, *rest in variants:
            skip, mode, head, sdpa_impl = (rest + ['', 'grad', '32', ''])[:4]
            os.environ['POLYTOPIA_ATTENTION_IMPL'], os.environ['POLYTOPIA_ATTENTION_REMAT'] = impl, remat
            os.environ['POLYTOPIA_ATTENTION_BENCH_SKIP'], os.environ['POLYTOPIA_ATTENTION_HEAD'] = skip, head
            os.environ['POLYTOPIA_SDPA_IMPL'] = sdpa_impl
            def loss(params, x, xc):
                p, c = params
                tiles, g, v = jax.vmap(lambda a: network.trunk(p, a))(x)
                if mode == 'policy':
                    return jnp.mean(tiles ** 2) + jnp.mean(g ** 2) + jnp.mean(v ** 2)
                vc = jax.vmap(lambda a: network.trunk(c, a)[2])(xc)
                return jnp.mean(tiles ** 2) + jnp.mean(g ** 2) + jnp.mean(v ** 2) + jnp.mean(vc ** 2)
            if mode == 'forward':
                step = jax.jit(lambda params, x, xc: (loss(params, x, xc), params))
            else:
                step = jax.jit(jax.value_and_grad(loss))
            try:
                t0 = time.time(); val, grads = step((policy, critic), x, xc); jax.block_until_ready(grads); compile_s = time.time() - t0
                times = []
                for _ in range(8):
                    t0 = time.time(); val, grads = step((policy, critic), x, xc); jax.block_until_ready(grads); times.append(time.time() - t0)
                row = dict(net=name, impl=impl, remat=remat, skip=skip, mode=mode, head=head, sdpa=sdpa_impl, rows=rows, compile_s=round(compile_s, 1),
                           step_ms=round(1000 * float(np.median(times)), 1), loss=float(val))
                # forward agreement with the reference kernel (einsum, fp32 softmax): same bf16 operands
                if impl != 'einsum':
                    os.environ['POLYTOPIA_ATTENTION_IMPL'] = 'einsum'
                    ref = jax.jit(lambda p, a: jax.vmap(lambda b: network.trunk(p, b)[2])(a))(policy, x)
                    os.environ['POLYTOPIA_ATTENTION_IMPL'] = impl
                    got = jax.jit(lambda p, a: jax.vmap(lambda b: network.trunk(p, b)[2])(a))(policy, x)
                    row['value_max_abs_diff'] = float(jnp.max(jnp.abs(ref - got)))
                    row['value_scale'] = float(jnp.max(jnp.abs(ref)))
            except Exception as error:
                row = dict(net=name, impl=impl, remat=remat, skip=skip, mode=mode, head=head, sdpa=sdpa_impl, rows=rows, error=repr(error)[:300])
            print(json.dumps(row), flush=True)
            out.append(row)
    return out


@app.function(gpu='H100', cpu=8, memory=32768, timeout=3600)
def bench_h100(rows: int):
    return _run(rows)


@app.function(gpu='L40S', cpu=8, memory=32768, timeout=3600)
def bench_l40s(rows: int):
    return _run(rows)


def _run(rows):
    nets = {'lookout-2c3a': dict(blocks=2, width=128, obs_version=3, attention=3)}
    # (impl, remat, skip, mode, head dim, sdpa implementation); policy only
    variants = [('einsum', '1', '', 'policy'), ('sdpa', '0', '', 'policy'),
                ('sdpa', '0', '', 'policy', '32', 'cudnn'), ('sdpa', '0', '', 'policy', '32', 'xla'),
                ('sdpa', '0', '', 'policy', '64', ''), ('sdpa', '0', '', 'policy', '64', 'cudnn'),
                ('sdpa', '0', '', 'policy', '128', ''), ('einsum', '1', '', 'policy', '64'),
                ('sdpa', '0', '', 'forward'), ('sdpa', '0', '', 'forward', '64')]
    return _bench(rows, variants, nets)


@app.local_entrypoint()
def main(gpu: str = 'H100', rows: int = 4096):
    fn = {'H100': bench_h100, 'L40S': bench_l40s}[gpu]
    for row in fn.remote(rows):
        print(json.dumps(row))
