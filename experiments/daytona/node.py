"""Runs inside a Daytona sandbox (image: experiments/daytona/Dockerfile). Prints one JSON line per result and
appends it to /tmp/polytopia/results.jsonl, so the Mac only polls a file (its network is not on the critical path).

    python node.py host
    python node.py fleet ARMS_JSON        # scripts/bench_actor_fleet.py arms, as modal_inference_server.py::fleet
    python node.py learner SPECS_JSON     # Learner.benchmark on a random-init parent of each architecture
    python node.py smoke SPEC_JSON        # one-box training: real Learner + actor fleet over 127.0.0.1

Daytona sandboxes see every host CPU in their affinity set (48 on a 4-vCPU sandbox) but get a cgroup CPU quota, so
fleet arms and smokes must pass `processes` explicitly: run_actor_container's processes=0 would start one per host CPU.

A learner spec: {name, players, size, blocks, width, extra_blocks, global_pool, config: {LearnerConfig overrides},
env: {...}, epochs}. Parents are random-init (network.init) checkpoints of that architecture: the learner's cost
does not depend on the weights, and nothing has to be uploaded.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT = Path('/tmp/polytopia')
RESULTS = OUT / 'results.jsonl'
LEARNER_ENV = {'POLYTOPIA_MATMUL_PRECISION': 'DEFAULT', 'POLYTOPIA_HEAD_PRECISION': 'HIGHEST',
               'POLYTOPIA_TRUNK_DTYPE': 'bfloat16', 'XLA_PYTHON_CLIENT_MEM_FRACTION': '.95'}


def emit(row):
    OUT.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(row, at=time.time()))
    with RESULTS.open('a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def host():
    sh = lambda c: subprocess.run(['bash', '-c', c], capture_output=True, text=True).stdout.strip()
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    return dict(cpu=sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip(), nproc=sh('nproc'),
                affinity=affinity, cgroup_cpu=sh('cat /sys/fs/cgroup/cpu.max 2>/dev/null'),
                mem_gib=round(int(sh("awk '/MemTotal/{print $2}' /proc/meminfo") or 0) / 2**20, 1),
                cgroup_mem=sh('cat /sys/fs/cgroup/memory.max 2>/dev/null'),
                gpus=sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null'))


class GpuSampler:
    """nvidia-smi every 5 s; mean utilisation over the second half (past compile), peak memory."""
    def __init__(self):
        self.rows, self.stop = [], threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop.wait(5):
            q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True).stdout.strip().splitlines()
            if q:
                self.rows.append([[float(x) for x in line.split(',')] for line in q])

    def summary(self):
        self.stop.set()
        tail = self.rows[len(self.rows) // 2:]
        if not tail:
            return dict(gpu_util_mean=None, gpu_mem_mib_max=None)
        n = len(tail[0])
        return dict(gpu_util_mean=[round(sum(r[i][0] for r in tail) / len(tail), 1) for i in range(n)],
                    gpu_mem_mib_max=[max(r[i][1] for r in self.rows) for i in range(n)])


def fleet(arms):
    info = host()
    for arm in arms:
        gpu = GpuSampler()
        started = time.time()
        arm = dict(arm)
        k = arm.pop('affinity', 0)   # Daytona: all host CPUs are visible under a 16-CPU quota; k narrows the set
        pin = (lambda: os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:k])) if k else None
        out = subprocess.run([sys.executable, '/workspace/scripts/bench_actor_fleet.py', json.dumps(arm)],
                             capture_output=True, text=True, cwd='/workspace', env=dict(os.environ, JAX_PLATFORMS='cpu'),
                             preexec_fn=pin)
        arm['affinity'] = k
        row = dict(kind='fleet', host=info, arm=arm, wall_s=round(time.time() - started, 1), **gpu.summary())
        if out.returncode == 0:
            row.update(json.loads(out.stdout.strip().splitlines()[-1]))
        else:
            row['error'] = (out.stdout + out.stderr)[-4000:]
        emit(row)


def random_parent(spec, path):
    """A random-init checkpoint of spec's architecture (save_checkpoint writes the current ruleset)."""
    from polytopia_bot.rl.distributed import _template_params, feature_shapes
    from polytopia_bot.rl.runner import Config, save_checkpoint
    from polytopia_bot.rl.observation import OBS_VERSION
    net = dict(size=spec['size'], blocks=spec['blocks'], extra_blocks=spec.get('extra_blocks', 0), width=spec['width'],
               architecture='normalized-v2', players=spec.get('players', 2), global_pool=spec.get('global_pool', False),
               block_pool=spec.get('block_pool', False), attention=spec.get('attention', 0),
               obs_version=spec.get('obs_version', OBS_VERSION))
    import jax
    params = _template_params(**{**net, 'size': net['size']}) if spec.get('seed', 0) == 0 else None
    if params is None:   # a seeded draw (the template is key 0)
        from polytopia_bot.rl import network
        board, glob = feature_shapes(net['players'], net['obs_version'])
        params = network.init(jax.random.key(spec['seed']), board, glob, **{k: v for k, v in net.items()
                                                                              if k not in ('players', 'obs_version')})
    import shutil
    import tempfile
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:   # the S3 mount under /vol takes no seeks/renames: write locally, copy
        local = Path(tmp) / Path(path).name
        save_checkpoint(local, params, Config(no_resign=True, **net), stage='daytona-random-init')
        for f in (local, local.with_suffix('.json')):
            shutil.copyfile(f, Path(path).with_suffix(f.suffix))
    n = int(sum(a.size for a in jax.tree.leaves(params)))
    print(json.dumps(dict(stage='random_parent', path=str(path), parameters=n, **net)), flush=True)
    return path


def sweep_parent(spec):
    """spec: {size, blocks, width, players, out}: the sweep arm's random init, written off the volume and copied on
    (experiments/daytona/sweep.py init). An existing parent is kept: the arms' trunks and replicates start from it."""
    import shutil
    out = Path(spec['out'])
    if out.exists():
        emit(dict(kind='sweep_parent', out=str(out), existed=True))
        return
    local = random_parent(spec, Path('/tmp/sweep-parent') / out.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    for f in (local.with_suffix('.json'), local):   # weights last: a reader that sees the .npz finds its .json
        shutil.copyfile(f, out.parent / f.name)
    emit(dict(kind='sweep_parent', out=str(out), bytes=out.stat().st_size))


def sweep_fork(spec):
    """spec: {trunk, fork, update}: the cooldown fork's run folder (distributed.fork_run), built off the volume (its log
    rewrite renames, which mountpoint-s3 cannot) and copied onto it, latest.npz last."""
    import shutil
    from polytopia_bot.rl.distributed import fork_run
    remote = VOL / 'results/distributed'
    if (remote / spec['fork'] / 'latest.npz').exists():
        emit(dict(kind='sweep_fork', fork=spec['fork'], existed=True))
        return
    local = Path('/tmp/sweep-fork') / spec['fork']
    shutil.rmtree(local, ignore_errors=True)
    fork_run(remote / spec['trunk'], local, spec['update'])
    files = sorted((f for f in local.rglob('*') if f.is_file()), key=lambda f: f.name == 'latest.npz')
    for f in files:
        dst = remote / spec['fork'] / f.relative_to(local)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst)
    emit(dict(kind='sweep_fork', fork=spec['fork'], update=spec['update'], files=len(files)))


def _learner_config(spec):
    from polytopia_bot.rl.distributed import LearnerConfig
    config = dict(spec.get('config', {}), size=spec['size'], players=spec.get('players', 2))
    return LearnerConfig(**{**config, 'frozen': tuple(tuple(x) for x in config.get('frozen', ()))})


def learner_one(spec):
    """One architecture, in its own process (the precision environment is read when rl.network is imported)."""
    import tempfile
    from polytopia_bot.rl.distributed import Learner
    parent = random_parent(spec, OUT / 'parents' / f"{spec['name']}.npz")
    learner = Learner(Path(tempfile.mkdtemp()) / 'bench', _learner_config(spec), str(parent), log=lambda s: None)
    rows = learner.benchmark(spec.get('epochs', 4))
    best = max(r['samples_per_second'] for r in rows[1:]) if len(rows) > 1 else None
    return dict(devices=learner.devices, rows=rows, samples_per_second=best)


def learner(specs):
    info = host()
    for spec in specs:
        gpu = GpuSampler()
        env = dict(os.environ, JAX_PLATFORMS='cuda', **LEARNER_ENV, **{k: str(v) for k, v in spec.get('env', {}).items()})
        out = subprocess.run([sys.executable, __file__, '_learner_one', json.dumps(spec)], capture_output=True,
                             text=True, env=env)
        row = dict(kind='learner', host=info, spec=spec, **gpu.summary())
        if out.returncode == 0:
            row.update(json.loads(out.stdout.strip().splitlines()[-1]))
        else:
            row['error'] = (out.stdout + out.stderr)[-4000:]
        emit(row)


def smoke(spec):
    """Real training in one sandbox: the Learner process on spec['learner_gpus'] serving 127.0.0.1:7070, and
    run_actor_container in this process (inference 'cuda' on spec['inference_gpu'], or None: CPU forwards).
    Reports the learner's training.jsonl rows; the run folder stays in /tmp/polytopia/smoke."""
    root = OUT / 'smoke' / spec['name']
    os.environ['JAX_PLATFORMS'] = 'cpu'   # this process and the actors; the learner and the server pick cuda
    parent = random_parent(spec, OUT / 'parents' / f"{spec['name']}.npz")
    config = dict(spec.get('config', {}), training_seconds=spec['seconds'])
    learner_env = dict(os.environ, JAX_PLATFORMS='cuda', CUDA_VISIBLE_DEVICES=str(spec.get('learner_gpus', '0')),
                       **LEARNER_ENV, **{k: str(v) for k, v in spec.get('env', {}).items()})
    log = (OUT / f"smoke-{spec['name']}-learner.log").open('w')
    proc = subprocess.Popen([sys.executable, __file__, '_learner_run', json.dumps(dict(spec, config=config)),
                             str(root), str(parent)], stdout=log, stderr=subprocess.STDOUT, env=learner_env)
    deadline = time.time() + 900
    while not (root / 'protocol.json').exists():   # written by Learner.run once it serves
        if proc.poll() is not None or time.time() > deadline:
            emit(dict(kind='smoke', name=spec['name'], error='learner did not start',
                      log=(OUT / f"smoke-{spec['name']}-learner.log").read_text()[-4000:]))
            return
        time.sleep(2)
    os.environ['JAX_PLATFORMS'] = 'cpu'   # actor processes; the inference server picks its own platform
    if spec.get('inference_gpu') is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(spec['inference_gpu'])
    from polytopia_bot.rl.distributed import run_actor_container
    started = time.time()
    summary = run_actor_container(('127.0.0.1', 7070), spec.get('processes', 0), spec.get('games', 24),
                                  report=lambda d: None, should_stop=lambda: proc.poll() is not None,
                                  inference='cuda' if spec.get('inference_gpu') is not None else None,
                                  serve_options=spec.get('serve_options'))
    proc.wait(timeout=600)
    rows = [json.loads(line) for line in (root / 'training.jsonl').read_text().splitlines()] \
        if (root / 'training.jsonl').exists() else []
    emit(dict(kind='smoke', name=spec['name'], host=host(), wall_s=round(time.time() - started, 1),
              learner_exit=proc.returncode, updates=len(rows), last=rows[-3:] if rows else None,
              actors={k: v for k, v in summary.items() if k != 'errors'}, actor_errors=summary.get('errors', [])[-5:],
              learner_log_tail=(OUT / f"smoke-{spec['name']}-learner.log").read_text()[-3000:]))


# ---- hero runs (experiments/daytona/hero.py). The volume (/vol, mountpoint-s3) takes new files, whole-file
# overwrites and deletes, but no appends or renames: the learner's run folder lives on local disk and is copied
# to /vol/results/distributed/<run> after every checkpoint. /results and /retained are symlinks into /vol, so
# Modal-style paths in configs and references work unchanged.

VOL = Path('/vol')
PORT = 17070   # 7070 is taken by the Daytona daemon in every sandbox
STOP = Path('/workspace/STOP')   # the controller touches it (plus /vol/runs/<run>/STOP) for an early stop
# copied in this order, so a reader (or a resume) never sees latest.npz ahead of the log it needs
SYNC_ORDER = ('training.jsonl', 'protocol.json', 'snapshots', 'latest-critic.json', 'latest-critic.npz',
              'latest.npz', 'latest.json', 'training.json', 'final-critic.json', 'final-critic.npz', 'final.npz',
              'final.json', 'training-complete.json', 'learner-status.json')


def vol_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class VolumeSync:
    """Copies changed files of a local run folder to the volume; wake() after a checkpoint, or every `every` s."""
    def __init__(self, local, remote, every=300):
        self.local, self.remote, self.every = Path(local), Path(remote), every
        self.seen, self.event, self.lock = {}, threading.Event(), threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def wake(self):
        self.event.set()

    def _loop(self):
        while True:
            self.event.wait(self.every)
            self.event.clear()
            try:
                self.sync()
            except Exception as error:
                print(json.dumps(dict(stage='sync_error', error=repr(error))), flush=True)

    def sync(self):
        import shutil
        with self.lock:
            started, copied = time.time(), 0
            for name in SYNC_ORDER:
                src = self.local / name
                files = sorted(src.iterdir()) if src.is_dir() else [src] if src.exists() else []
                for f in files:
                    st = f.stat()
                    key = (st.st_size, st.st_mtime_ns)
                    if self.seen.get(f) == key:
                        continue
                    dst = self.remote / f.relative_to(self.local)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, dst)
                    self.seen[f] = key
                    copied += 1
            print(json.dumps(dict(stage='synced', files=copied, seconds=round(time.time() - started, 1))), flush=True)


CACHE = Path('/workspace/jax-cache')   # JAX_COMPILATION_CACHE_DIR of every hero node (hero.py sets it)


def compile_cache(every=600):
    """Persistent XLA cache per GPU model on the volume: seed the local cache from it, then copy new entries back
    every `every` s, so a replacement sandbox (spot reclaim) skips most of its compile (an inference server spent
    544 s compiling on 2 pinned cores, 2026-10-28). Entries are write-once files, which the S3 mount allows."""
    import shutil
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True,
                         text=True).stdout.strip().splitlines()
    remote = VOL / 'jax-cache' / ((gpu[0] if gpu else 'cpu').replace(' ', '-') + f'-x{len(gpu)}')
    CACHE.mkdir(parents=True, exist_ok=True)
    if remote.exists():
        for f in remote.iterdir():
            if not (CACHE / f.name).exists():
                shutil.copyfile(f, CACHE / f.name)

    def push():
        while True:
            time.sleep(every)
            try:
                remote.mkdir(parents=True, exist_ok=True)
                have = {f.name for f in remote.iterdir()}
                for f in CACHE.iterdir():
                    if f.is_file() and f.name not in have and time.time() - f.stat().st_mtime > 30:
                        shutil.copyfile(f, remote / f.name)
            except Exception as error:
                print(json.dumps(dict(stage='cache_push_error', error=repr(error))), flush=True)
    threading.Thread(target=push, daemon=True).start()


def hero_learner(spec):
    """spec: {run, parent, config (LearnerConfig fields, training_seconds = the run's total), env}. Resumes from the
    run folder on the volume when it has latest.npz (Learner._resume: weights, Adam, critic, update count, log,
    historical pool, clock); otherwise starts fresh from parent."""
    import shutil
    from polytopia_bot.rl.distributed import Learner, LearnerConfig
    run = spec['run']
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)   # the keeper dumps a stalled learner's threads (stderr = log)
    compile_cache()
    local, remote = Path('/workspace/run') / run, VOL / 'results/distributed' / run
    if not local.exists() and (remote / 'latest.npz').exists():
        shutil.copytree(remote, local)   # Learner sees an existing folder with latest.npz -> resume
    sync = VolumeSync(local, remote)
    cfg = spec['config']
    parent = spec['parent']
    if not (local / 'latest.npz').exists() and spec.get('fallback_parent'):   # a from-scratch run before its first checkpoint
        parent = spec['fallback_parent']
        print(json.dumps(dict(stage='fresh_start', parent=parent)), flush=True)
    learner = Learner(local, LearnerConfig(**{**cfg, 'frozen': tuple(tuple(x) for x in cfg.get('frozen', ()))}),
                      parent, port=PORT, commit=sync.wake, log=lambda m: print(m, flush=True),
                      status=lambda row: vol_write(local / 'learner-status.json', json.dumps(dict(row, at=time.time()))))
    stop_files = (STOP, VOL / 'runs' / run / 'STOP')

    def watch_stop():   # an early stop: the learner leaves its loop at the deadline and writes final.npz
        while True:     # keeps pulling the deadline in (run() sets it when it starts)
            if any(p.exists() for p in stop_files):
                print(json.dumps(dict(stage='stop_requested')), flush=True)
                learner.deadline = time.monotonic()
            time.sleep(15)
    threading.Thread(target=watch_stop, daemon=True).start()
    import traceback
    try:
        learner.run(bind_host='127.0.0.1')   # actors arrive through the SSH gateway's forward to localhost
    except BaseException:
        sync.sync()   # the keeper sees the process gone and resumes on a new sandbox
        vol_write(VOL / 'runs' / run / f'learner-error-{int(time.time())}.txt', traceback.format_exc())
        print(traceback.format_exc(), flush=True)
        os._exit(1)   # learner threads would keep a dead run's process alive
    sync.sync()
    vol_write(VOL / 'runs' / run / 'learner-done.json', json.dumps(dict(at=time.time(), host=host())))
    os._exit(0)


def hero_actor(spec):
    """spec: {run, index, processes, games, serve_options, until_unix}. One SSH local forward 127.0.0.1:7070 ->
    the learner sandbox (token in /workspace/learner-token, rewritten by the controller when the learner is
    replaced), and the GPU-inference actor fleet of run_actor_container on it. Exits once stopped (os._exit:
    leftover threads); the keeper replaces a sandbox whose actor process is gone."""
    from polytopia_bot.rl.distributed import run_actor_container
    compile_cache()
    if spec.get('affinity'):   # all host CPUs are visible under the sandbox's CPU quota: pin to as many as it pays for
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:spec['affinity']])
    token_file = Path('/workspace/learner-token')
    stop_files = (STOP, VOL / 'runs' / spec['run'] / 'STOP')

    def stopping():
        return time.time() > spec['until_unix'] or any(p.exists() for p in stop_files)

    def probe():
        """A hello through the tunnel. A long-lived Daytona SSH-gateway session can stop opening new channels while it
        stays up (2026-10-28: every 4p actor connection got 'peer closed' after ~50k samples; a fresh session worked)."""
        import socket
        from polytopia_bot.rl.distributed import recv_msg, send_msg
        try:
            with socket.create_connection(('127.0.0.1', PORT), timeout=20) as s:
                s.settimeout(30)
                send_msg(s, dict(type='hello', actor=-1, have=[], relayed=True))
                return 'version' in recv_msg(s)
        except Exception:
            return False

    def tunnel():
        while not stopping():
            token = token_file.read_text().strip() if token_file.exists() else ''
            if token:   # a tunnel left by an earlier node process would keep the port and serve a dead session
                subprocess.run(['pkill', '-f', f'ssh -N .*-L {PORT}:'], capture_output=True)
                ssh = subprocess.Popen(['ssh', '-N', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
                                        '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
                                        '-o', 'ExitOnForwardFailure=yes', '-o', 'LogLevel=ERROR',
                                        '-L', f'{PORT}:127.0.0.1:{PORT}', f'{token}@ssh.app.daytona.io'])
                time.sleep(10)
                failures = 0
                while ssh.poll() is None and not stopping():   # restart the session after two failed probes
                    failures = 0 if probe() else failures + 1
                    if failures >= 2:
                        print(json.dumps(dict(stage='tunnel_stale', at=time.time())), flush=True)
                        ssh.kill()
                        break
                    time.sleep(30)
                ssh.wait()
                print(json.dumps(dict(stage='tunnel_down', at=time.time())), flush=True)
            time.sleep(5)
    threading.Thread(target=tunnel, daemon=True).start()
    report_path = OUT / f"actor-{spec['index']}.json"

    def report(summary):
        vol_write(report_path, json.dumps(dict(summary, at=time.time())))
    while not stopping():   # run_actor_container returns when the learner says stop or every process gave up
        summary = run_actor_container(('127.0.0.1', PORT), spec['processes'], spec['games'],
                                      first_id=spec['index'] * 1000, report=report, should_stop=stopping,
                                      resolve_address=lambda: None if stopping() else ('127.0.0.1', PORT),
                                      inference='cuda', serve_options=spec.get('serve_options'))
        print(json.dumps(dict(stage='actor_container_returned', **{k: v for k, v in summary.items() if k != 'errors'},
                              errors=summary.get('errors', [])[-3:]), default=str), flush=True)
        time.sleep(30)   # the learner may be restarting on a new sandbox: try again until stopped
    os._exit(0)


def hero_arena(spec):
    """1v1 ladder pair (cpu_arena.match, as modal_distributed_rl.arena_match); writes spec['out'] on the volume."""
    from polytopia_bot.rl.cpu_arena import match
    started = time.time()
    summary = match(spec['candidate'], spec['opponent'], range(spec['seed_base'], spec['seed_base'] + spec['maps']),
                    games=spec.get('games', 4), deterministic=True, processes=spec.get('processes', 4),
                    maps=spec.get('map_mix'), size=spec.get('size', 14))
    summary.update({k: spec[k] for k in ('name', 'candidate', 'opponent', 'seed_base', 'maps')}, engine='engine-rs',
                   map_mix=spec.get('map_mix'), wall_s=round(time.time() - started, 1))
    vol_write(spec['out'], json.dumps(summary))


def hero_arena4(spec):
    """One part of a four-seat table (arena4.play, as modal_distributed_rl.arena4_part); records to spec['out']."""
    from polytopia_bot.rl.arena4 import play
    from polytopia_bot.rl.distributed import checkpoint_template
    from polytopia_bot.rl.runner import load_checkpoint
    policies = {name: load_checkpoint(path, checkpoint_template(path)) for name, path in spec['entrants'].items()}
    records = play(policies, spec['tables'], size=18, maps='four-v1', seeds=spec['seeds'], lanes=spec.get('lanes', 16),
                   deterministic=False, seed=spec['seed'])
    vol_write(spec['out'], json.dumps(records, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))


def hero_pool_probe(spec):
    """Trunk-block pooling statistics of a checkpoint on self-play positions (pool_probe.probe); writes spec['out']."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pool_probe import probe
    row = probe(spec['checkpoint'], games=spec.get('games', 8), steps=spec.get('steps', 1600), every=spec.get('every', 40),
                seed=spec.get('seed', 700000))
    if spec.get('out'):
        vol_write(spec['out'], json.dumps(row, indent=1))
    print(json.dumps(row), flush=True)


def hero_summary4(spec):
    """Merge four-seat part records (spec['parts']) into arena4.summary, anchored at START; writes spec['out']."""
    from polytopia_bot.rl.arena4 import summary
    records = [r for p in spec['parts'] if Path(p).exists() for r in json.loads(Path(p).read_text())]
    out = summary(records, anchor=spec.get('anchor')) if records else None
    vol_write(spec['out'], json.dumps(dict(summary=out, games=len(records)), default=str))


if __name__ == '__main__':
    cmd = sys.argv[1]
    hero = dict(hero_learner=hero_learner, hero_actor=hero_actor, hero_arena=hero_arena, hero_arena4=hero_arena4,
                hero_summary4=hero_summary4, hero_random_parent=lambda spec: random_parent(spec, spec['out']),
                hero_pool_probe=hero_pool_probe,
                sweep_parent=sweep_parent, sweep_fork=sweep_fork)
    if cmd in hero:
        hero[cmd](json.loads(Path(sys.argv[2]).read_text() if sys.argv[2].endswith('.json') else sys.argv[2]))
    elif cmd == 'host':
        emit(dict(kind='host', **host()))
    elif cmd == 'fleet':
        fleet(json.loads(sys.argv[2]))
    elif cmd == 'learner':
        learner(json.loads(sys.argv[2]))
    elif cmd == '_learner_one':
        print(json.dumps(learner_one(json.loads(sys.argv[2]))))
    elif cmd == '_learner_run':
        from polytopia_bot.rl.distributed import Learner
        s = json.loads(sys.argv[2])
        Learner(Path(sys.argv[3]), _learner_config(s), sys.argv[4], port=7070,
                log=lambda m: print(m, flush=True)).run(bind_host='127.0.0.1')
    elif cmd == 'smoke':
        smoke(json.loads(sys.argv[2]))
    else:
        raise SystemExit(f'unknown command {cmd}')
