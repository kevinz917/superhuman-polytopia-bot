"""Bare multi-GPU probe for the Curio head: does a shard_map all-reduce loop (two alternating jitted programs, as the
learner's training step and value pass) run for N iterations on the visible GPUs, or hang like the learner did?

    CUDA_VISIBLE_DEVICES=4,5,6,7 python nccl_probe.py [iterations=300] [rows_per_device=4096]
Prints a line every 25 iterations with the wall time; a hang shows as silence (watch nvidia-smi: ranks spinning).
"""
import os
import sys
import time

os.environ.setdefault('JAX_PLATFORMS', 'cuda')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
devices = jax.devices()
print(dict(devices=[str(d) for d in devices], jax=jax.__version__, nccl_env={k: v for k, v in os.environ.items() if k.startswith(('NCCL', 'XLA_FLAGS'))}), flush=True)
mesh = Mesh(np.array(devices), ('data',))
D = len(devices)


def step(w, x):
    def local(w, x):
        y = jnp.tanh(x @ w)
        g = jnp.mean(y[..., None] * x[:, None, :], axis=0).T
        g = jax.lax.pmean(g, 'data')
        return w - 1e-3 * g, jnp.mean(y)
    return jax.shard_map(local, mesh=mesh, in_specs=(P(), P('data')), out_specs=(P(), P()), check_vma=False)(w, x)


def values(w, x):
    return jax.shard_map(lambda w, x: jnp.tanh(x @ w).sum(-1), mesh=mesh, in_specs=(P(), P('data')), out_specs=P('data'),
                         check_vma=False)(w, x)


step_j, values_j = jax.jit(step), jax.jit(values)
w = jax.device_put(jnp.zeros((512, 512), jnp.float32) + 1e-3, jax.sharding.NamedSharding(mesh, P()))
t0 = time.time()
for i in range(n):
    x = jax.device_put(np.random.default_rng(i).standard_normal((rows * D, 512), np.float32), jax.sharding.NamedSharding(mesh, P('data')))
    v = values_j(w, x)
    w, m = step_j(w, x)
    if i % 25 == 0 or i == n - 1:
        print(dict(i=i, mean=float(m), vsum=float(v.sum()), seconds=round(time.time() - t0, 1)), flush=True)
print('PROBE_OK', flush=True)
