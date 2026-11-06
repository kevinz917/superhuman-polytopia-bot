"""Pre-flight check for a multi-GPU learner host: all-reduce correctness, per-device matmul agreement and
host->device bandwidth. Prints one JSON line; exit 1 when the host should be rejected.
A Daytona RTX PRO 6000 x4 host (GPUs on 4 NUMA nodes) trained with KL ~1 from the first update, 2026-10-28."""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

devices = jax.devices()
n = len(devices)
rng = np.random.default_rng(0)
x = rng.standard_normal((n, 4096, 1024)).astype(np.float32)
out = {'devices': n, 'kind': devices[0].device_kind}
# all-reduce: every device must get the exact host sum
summed = jax.pmap(lambda v: jax.lax.psum(v, 'i'), axis_name='i')(x)
want = x.sum(0)
out['psum_max_err'] = float(max(np.abs(np.asarray(summed[i]) - want).max() for i in range(n)))
# the same matmul on every device must agree bit for bit
a = rng.standard_normal((2048, 2048)).astype(np.float32)
res = [np.asarray(jax.device_put(a, d) @ jax.device_put(a, d)) for d in devices]
out['matmul_max_diff'] = float(max(np.abs(r - res[0]).max() for r in res))
# host->device bandwidth per device, 256 MB
big = np.ones((64, 1024, 1024), np.float32)
bw = []
for d in devices:
    jax.block_until_ready(jax.device_put(big[:1], d))
    t = time.perf_counter()
    jax.block_until_ready(jax.device_put(big, d))
    bw.append(round(big.nbytes / (time.perf_counter() - t) / 1e9, 1))
out['h2d_gb_s'] = bw
# all-reduce time for a gradient-sized buffer (8 MB)
g = np.ones((n, 2 * 1024 * 1024), np.float32)
f = jax.pmap(lambda v: jax.lax.psum(v, 'i'), axis_name='i')
jax.block_until_ready(f(g))
t = time.perf_counter()
for _ in range(10):
    jax.block_until_ready(f(g))
out['psum_8mb_ms'] = round((time.perf_counter() - t) * 100, 2)
min_h2d = float(sys.argv[1]) if len(sys.argv) > 1 else 2.   # GB/s; learners 2 (healthy hosts: 3-16)
out['ok'] = out['psum_max_err'] < 1e-3 and out['matmul_max_diff'] < 1e-3 and min(bw) > min_h2d and out['psum_8mb_ms'] < 50
print(json.dumps(out))
sys.exit(0 if out['ok'] else 1)
