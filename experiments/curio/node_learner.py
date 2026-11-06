"""Learner node on the Curio head: the real distributed.Learner on a chosen set of the head's GPUs, no Modal.

    python node_learner.py run --root ROOT --run RUN --config CFG.json --parent PARENT.npz [--critic CRITIC.npz]
                               [--hours H] [--gpus 4,5,6,7] [--cores 0-31] [--bind 0.0.0.0]
    python node_learner.py bench --root ROOT --config CFG.json --parent PARENT.npz [--gpus 4,5,6,7] [--epochs 4]
                                 [--minibatch 16384]

`run` keeps ROOT/runs/RUN between starts: a restarted learner finds latest.npz there and resumes with its Adam
state, log and historical pool (Learner._resume), on the same training clock. The first start of a run uses --parent
(+ --critic for the parent's omniscient critic, LearnerConfig.critic_path) and --hours sets training_seconds.
A STOP file in the run directory ends training gracefully (final.npz is written). On exit the node writes
learner-done.json or learner-error.txt next to the run, which the keeper reads.

Environment (CUDA_VISIBLE_DEVICES, JAX_* and POLYTOPIA_* precision/dtype) is applied here, before any JAX import, so
the command line is the whole story.
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'bench'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--run', default='bench')
    ap.add_argument('--config', required=True)
    ap.add_argument('--parent', required=True)
    ap.add_argument('--critic', default='')
    ap.add_argument('--hours', type=float, default=None)
    ap.add_argument('--gpus', default='')
    ap.add_argument('--cores', default='')
    ap.add_argument('--bind', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=7070)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--minibatch', type=int, default=0)
    return ap.parse_args()


def main():
    args = parse()
    root = Path(args.root)
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ.setdefault('JAX_PLATFORMS', 'cuda')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.95')
    os.environ.setdefault('POLYTOPIA_MATMUL_PRECISION', 'DEFAULT')   # read by rl.network at import
    os.environ.setdefault('POLYTOPIA_HEAD_PRECISION', 'HIGHEST')
    os.environ.setdefault('POLYTOPIA_TRUNK_DTYPE', 'bfloat16')
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(root / 'jax-cache'))
    if args.cores:
        from common import parse_cores
        os.sched_setaffinity(0, parse_cores(args.cores))
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)   # `kill -USR1 <pid>` dumps every thread to the log

    import jax
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    jax.config.update('jax_compilation_cache_dir', os.environ['JAX_COMPILATION_CACHE_DIR'])
    log = lambda s: print(s, flush=True)

    config = {k: v for k, v in json.loads(Path(args.config).read_text()).items() if not k.startswith('_')}
    config['frozen'] = tuple(tuple(x) for x in config.get('frozen', ()))
    if args.critic:
        config['critic_path'] = args.critic
    if args.hours is not None:
        config['training_seconds'] = int(args.hours * 3600)
    if args.minibatch:
        config['minibatch'] = args.minibatch
    cfg = LearnerConfig(**config)
    log(json.dumps(dict(stage='node_start', mode=args.mode, run=args.run, devices=[str(d) for d in jax.devices()],
                        visible=os.environ.get('CUDA_VISIBLE_DEVICES'), parent=args.parent, critic=args.critic,
                        training_seconds=cfg.training_seconds, minibatch=cfg.minibatch, pid=os.getpid())))

    if args.mode == 'bench':
        import tempfile
        learner = Learner(Path(tempfile.mkdtemp()) / 'bench', cfg, args.parent, log=log)
        rows = learner.benchmark(args.epochs)
        log(json.dumps(dict(stage='bench_result', devices=learner.devices, minibatch=cfg.minibatch, rows=rows)))
        return

    run_dir = root / 'runs' / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ('learner-done.json', 'learner-error.txt'):
        (run_dir / stale).unlink(missing_ok=True)
    status_path = run_dir / 'learner-status.json'

    def status(row):
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(row, at=time.time())))
        tmp.replace(status_path)

    learner = Learner(run_dir, cfg, args.parent, port=args.port, log=log, status=status)

    def watch_stop():
        while True:
            if (run_dir / 'STOP').exists():
                log(json.dumps(dict(stage='stop_requested')))
                learner.deadline = time.monotonic()
                return
            time.sleep(15)

    threading.Thread(target=watch_stop, daemon=True).start()
    try:
        history = learner.run(bind_host=args.bind)
    except BaseException as error:   # noqa: BLE001 - the keeper reads the file, then decides
        (run_dir / 'learner-error.txt').write_text(f'{time.time()}\n{"".join(traceback.format_exception(error))}')
        log(json.dumps(dict(stage='learner_error', error=repr(error))))
        os._exit(1)
    (run_dir / 'learner-done.json').write_text(json.dumps(dict(
        at=time.time(), updates=len(history), final_update=learner.version, run=args.run)))
    log(json.dumps(dict(stage='learner_done', updates=len(history), final_update=learner.version)))
    os._exit(0)   # idle copy/bookkeeping threads must not keep the process alive


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
