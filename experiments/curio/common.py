"""Shared pieces of the Curio launcher: the cluster plan, hosts reached over SSH hops, remote background processes.

Every host is addressed by a list of SSH hops. From the Mac the head is ['curio-gpu'] and a worker is
['curio-gpu', 'user@10.0.0.2'] (the Mac's key is not on the workers; the head's is). From the head (where the keeper
runs) the head is [] (local) and a worker is ['user@10.0.0.2']. Commands are plain `sh -c` strings; each hop quotes
the whole command once more.
"""
import json
import shlex
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 7070
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', '-o', 'LogLevel=ERROR']
# Learner process: the GPU platform, bf16 trunk with fp32 heads, the memory fraction hero-1v1 ran with on H100s.
LEARNER_ENV = dict(JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.95',
                   POLYTOPIA_MATMUL_PRECISION='DEFAULT', POLYTOPIA_HEAD_PRECISION='HIGHEST',
                   POLYTOPIA_TRUNK_DTYPE='bfloat16',
                   # No CUDA-graph command buffers: on the head (driver 550 / CUDA 12.4 with jax 0.11.2's pip CUDA runtime)
                   # jit_epoch died with "Failed to add memset node to a CUDA graph: CUDA_ERROR_INVALID_VALUE" (2026-10-31).
                   XLA_FLAGS='--xla_gpu_enable_command_buffer=',
                   # NCCL keeps its default transport: the early hang was DeviceRows streaming, not NCCL, and forcing
                   # sockets cost ~2x on the training step (trials t1 vs t2, 2026-10-31).
                   NCCL_DEBUG='WARN')
# Actor node: the node itself and its engine processes stay on the CPU platform; a GPU inference server picks cuda
# for itself (inference_server.serve sets JAX_PLATFORMS in its own process). CUDA_VISIBLE_DEVICES is added per node.
ACTOR_ENV = dict(JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false')


def load_plan(path=None):
    return json.loads(Path(path or HERE / 'configs' / 'cluster.json').read_text())


def parse_cores(spec):
    """'0-31' or '0-31,40,41' -> sorted list of ints (inclusive ranges)."""
    cores = set()
    for part in str(spec).split(','):
        a, _, b = part.partition('-')
        cores.update(range(int(a), int(b or a) + 1))
    return sorted(cores)


class Host:
    """A machine and how to reach it: name, SSH hops, its polytopia root."""

    def __init__(self, name, hops, root):
        self.name, self.hops, self.root = name, list(hops), root.rstrip('/')

    def __repr__(self):
        return f'Host({self.name}, {" -> ".join(self.hops) or "local"}, {self.root})'

    def wrap(self, command):
        for hop in reversed(self.hops):
            command = ' '.join(SSH + [hop, shlex.quote(command)])
        return command

    def run(self, command, check=True, timeout=600, capture=True):
        """Run `command` with sh on the host; returns CompletedProcess with text output."""
        return subprocess.run(self.wrap(command), shell=True, check=check, timeout=timeout, text=True,
                              capture_output=capture)

    def out(self, command, timeout=600, default=None):
        try:
            return self.run(command, timeout=timeout).stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return default

    def exists(self, path):
        return self.out(f'test -e {shlex.quote(path)} && echo yes') == 'yes'

    def read(self, path, default=None):
        return self.out(f'cat {shlex.quote(path)}', default=default)

    def write(self, path, text):
        """Write a small text file atomically (stdin through nested ssh is awkward; the text rides in the command)."""
        self.run(f'mkdir -p {shlex.quote(str(Path(path).parent))} && printf %s {shlex.quote(text)} > '
                 f'{shlex.quote(path)}.tmp && mv -f {shlex.quote(path)}.tmp {shlex.quote(path)}')

    def pid_alive(self, pidfile):
        return self.out(f'[ -f {shlex.quote(pidfile)} ] && kill -0 $(cat {shlex.quote(pidfile)}) 2>/dev/null && '
                        f'echo alive') == 'alive'

    def start(self, name, command, env, cwd, log, pidfile):
        """Start `command` detached (setsid + nohup) unless the pidfile names a live process. Returns 'started' or
        'already'. The pid recorded is the python process itself (setsid execs in place for a non-leader)."""
        exports = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in env.items())
        script = (f'mkdir -p {shlex.quote(str(Path(log).parent))} {shlex.quote(str(Path(pidfile).parent))}; '
                  f'if [ -f {shlex.quote(pidfile)} ] && kill -0 $(cat {shlex.quote(pidfile)}) 2>/dev/null; '
                  f'then echo already; else cd {shlex.quote(cwd)} || exit 1; ulimit -n 262144 2>/dev/null; setsid nohup env {exports} {command} '
                  f'>> {shlex.quote(log)} 2>&1 < /dev/null & echo $! > {shlex.quote(pidfile)}; echo started; fi')
        # ulimit: a learner holds one socket per actor process (1,300+ with the CPU servers; the default soft limit
        # is 1,024 for non-interactive ssh, which crashed it). `cd X && cmd &` would background the whole list in a subshell that keeps the ssh pipe open and records the
        # subshell's pid; the `;` keeps `&` on the setsid command alone.
        return self.out(script, default='failed')

    def stop(self, pidfile, pattern=None, grace=10):
        """TERM the pidfile's process group, then KILL it; optionally pkill -9 a pattern (spawned actor workers and
        inference servers outlive their node, which stacked fleets on Daytona until the box OOMed)."""
        q = shlex.quote(pidfile)
        self.run(f'if [ -f {q} ]; then p=$(cat {q}); kill -TERM -- -$p 2>/dev/null || kill -TERM $p 2>/dev/null; '
                 f'sleep {grace}; kill -KILL -- -$p 2>/dev/null || kill -KILL $p 2>/dev/null; rm -f {q}; fi; true',
                 check=False, timeout=grace + 60)
        if pattern:
            self.run(f'pkill -9 -f {shlex.quote(pattern)}; true', check=False)


def hosts_from(plan, from_head=False):
    """{'head': Host, 'w83': Host, ...} as reached from the Mac (default) or from the head itself."""
    head = plan['head']
    hosts = {'head': Host('head', [] if from_head else [head['ssh']], head['root'])}
    for w in plan['workers']:
        hosts[w['name']] = Host(w['name'], ([] if from_head else [head['ssh']]) + [w['ssh']], w['root'])
    return hosts


def paths(root):
    """Layout under a host's root (same on the head and the workers)."""
    r = root.rstrip('/')
    return dict(root=r, venv=f'{r}/venv', python=f'{r}/venv/bin/python', code=f'{r}/code/current',
                src=f'{r}/code/current/src', curio=f'{r}/code/current/experiments/curio', wheels=f'{r}/wheels',
                ckpt=f'{r}/ckpt', runs=f'{r}/runs', logs=f'{r}/logs', pids=f'{r}/pids', cache=f'{r}/jax-cache',
                ladder=f'{r}/ladder', arena=f'{r}/arena')


def learner_command(plan, run, config, parent, critic='', hours=None, gpus=None, cores=None, resume=False):
    """The learner's python command line on the head (relative to the code dir)."""
    p = paths(plan['head']['root'])
    gpus = gpus if gpus is not None else plan['learner']['gpus']
    cores = cores or plan['learner']['cores']
    args = [f'{p["python"]} experiments/curio/node_learner.py run', '--root', p['root'], '--run', run,
            '--config', config, '--parent', parent, '--gpus', ','.join(map(str, gpus)), '--cores', str(cores)]
    if critic:
        args += ['--critic', critic]
    if hours is not None:
        args += ['--hours', str(hours)]
    return ' '.join(shlex.quote(a) if i else a for i, a in enumerate(args))


def actor_command(plan, host_root, index, learner_ip, processes, games, cores, inference='cpu', until=None,
                  stop_file=None, serve_options=None):
    p = paths(host_root)
    args = [f'{p["python"]} experiments/curio/node_actor.py', '--host', learner_ip, '--port', str(plan['port']),
            '--index', str(index), '--processes', str(processes), '--games', str(games), '--cores', str(cores),
            '--inference', inference, '--report', f'{p["logs"]}/actor-{index}.json']
    if until:
        args += ['--until-unix', str(int(until))]
    if stop_file:
        args += ['--stop-file', stop_file]
    if serve_options:
        args += ['--serve-options-json', json.dumps(serve_options)]
    return ' '.join(shlex.quote(a) if i else a for i, a in enumerate(args))


def now():
    return time.time()
