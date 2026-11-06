"""Learner node on a Verda GPU instance: the real distributed.Learner, no Modal.

    python node_learner.py bench [epochs] [minibatch]   # Learner.benchmark on the visible GPUs (synthetic batch, seeded)
    python node_learner.py run                          # serve actors on 0.0.0.0:7070 until config.training_seconds

`run` keeps /workspace/run between starts: under systemd (polytopia-learner.service, Restart=on-failure) a crashed
learner comes back and resumes latest.npz with its Adam state, log and pool (Learner._resume). The parent is
ckpt/final.npz in the bundle: a continuation's parent, or a random init for a from-scratch run.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

PILOT = Path('/workspace/pilot')
os.environ.setdefault('POLYTOPIA_MATMUL_PRECISION', 'DEFAULT')   # read by rl.network at import (scratch-cont-2 env)
os.environ.setdefault('POLYTOPIA_HEAD_PRECISION', 'HIGHEST')

from polytopia_bot.rl.distributed import Learner, LearnerConfig  # noqa: E402
import jax  # noqa: E402

jax.config.update('jax_compilation_cache_dir', '/workspace/jax-cache')
config = json.loads((PILOT / 'config.json').read_text())
if sys.argv[1] == 'bench' and len(sys.argv) > 3:
    config['minibatch'] = int(sys.argv[3])
cfg = LearnerConfig(**{**config, 'frozen': tuple(tuple(x) for x in config.get('frozen', ()))})
parent = str(PILOT / 'ckpt' / 'final.npz')
log = lambda s: print(s, flush=True)

if sys.argv[1] == 'bench':
    learner = Learner(Path(tempfile.mkdtemp()) / 'bench', cfg, parent, log=log)
    rows = learner.benchmark(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    print(json.dumps(dict(stage='bench_result', devices=learner.devices, minibatch=cfg.minibatch, rows=rows)),
          flush=True)
else:
    Learner(Path('/workspace/run'), cfg, parent, port=7070, log=log).run(bind_host='0.0.0.0')
