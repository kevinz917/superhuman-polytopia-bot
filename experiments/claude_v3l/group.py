"""Session L: multi-node process groups for the tiered-policy supervisor (16 GPUs = 2 x 8xH100, Modal clustered).

The supervisor runs on node 0 only. Every other node runs `run_agent`, which follows "generations" the supervisor
publishes on the control store (modal.Dict in the cloud, JsonStore in tests):

  <run>/grp/gen      {gen, action: run|stop|exit, launch_id, nodes: {"1": [cmd, ...]}, env, good_sha, t}
  <run>/grp/node<n>  {gen, state: idle|running|exited|refused, rcs, t}       (written by node n's agent)

One generation = one start of all ranks (a fresh trainer after a NaN rollback or a crash resume is a new generation).
Before publishing a generation the supervisor repairs the run directory and COMMITS the volume, then publishes the
sha256 of GOOD latest.npz; an agent reloads its volume view and refuses to start its ranks until its latest.npz has
that sha (a stale view would make its ranks resume from an older checkpoint than rank 0: silently inconsistent
replicas). Any rank failing on any node stops the whole group (the supervisor kills its own ranks and publishes
action stop); the tiered policy then decides between rollback, resume and pause exactly as for one node.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

LOST_RC = -99       # a node that stopped reporting (container lost): its ranks count as failed
REFUSED_RC = -98    # an agent whose checkpoint view never matched the published sha


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run_agent(store, key, node, *, reload=lambda: None, run_dir=None, cwd=None, poll_s=1.0, report_s=5.0,
              log=print, idle_exit_s=None):
    """Node `node`'s agent loop. Returns when the supervisor publishes action exit (or after idle_exit_s with no
    generation at all, for tests)."""
    cur, procs, state, rcs = None, [], 'idle', None
    last_report, started = 0., time.time()

    def report(force=False):
        nonlocal last_report
        if force or time.time() - last_report >= report_s:
            store.put(f'{key}/grp/node{node}', dict(gen=cur, state=state, rcs=rcs, t=time.time()))
            last_report = time.time()

    def kill():
        for p in procs:
            if p.poll() is None:
                p.kill()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass

    report(True)
    while True:
        g = store.get(f'{key}/grp/gen')
        if g and float(g.get('t') or 0) < started - 5 and g.get('gen') != cur:
            g = None        # a generation (or exit) of an earlier attempt/launch: wait for this supervisor's own
        if g is None and idle_exit_s and time.time() - started > idle_exit_s:
            return 'idle_timeout'
        if g and g.get('action') == 'exit':
            kill()
            if procs:
                state, rcs = 'exited', [p.returncode for p in procs]
            report(True)
            log(f'[v3l-agent {node}] exit')
            return 'exit'
        if g and g.get('action') == 'run' and g.get('gen') != cur:
            kill()
            cur, procs, rcs = g['gen'], [], None
            cmds = g['nodes'].get(str(node), [])
            ok = True
            if g.get('good_sha') and run_dir:
                ok = False
                for _ in range(60):          # the supervisor committed before publishing; reload until we see it
                    reload()
                    p = Path(run_dir) / 'latest.npz'
                    if p.exists() and sha256(p) == g['good_sha']:
                        ok = True
                        break
                    time.sleep(2)
            if not ok:
                state, rcs = 'refused', [REFUSED_RC] * len(cmds)
                log(f'[v3l-agent {node}] gen {cur}: checkpoint view never matched {g["good_sha"][:12]}; refused')
                report(True)
                continue
            env = dict(os.environ, **{k: str(v) for k, v in (g.get('env') or {}).items()})
            for i, cmd in enumerate(cmds):
                e = dict(env, CUDA_VISIBLE_DEVICES=str(i))
                logf = open(f'/tmp/v3l-node{node}-gen{cur}-r{i}.log', 'w')
                procs.append(subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True, env=e,
                                              cwd=cwd))
            state = 'running'
            log(f'[v3l-agent {node}] gen {cur}: started {len(procs)} ranks')
            report(True)
        elif g and g.get('gen') == cur and g.get('action') == 'stop' and state == 'running':
            kill()
        if state == 'running' and procs and all(p.poll() is not None for p in procs):
            state, rcs = 'exited', [p.returncode for p in procs]
            log(f'[v3l-agent {node}] gen {cur}: ranks exited {rcs}')
            report(True)
        report()
        time.sleep(poll_s)


def tail_of(node, gen, n_ranks, chars=1200):
    out = []
    for i in range(n_ranks):
        p = Path(f'/tmp/v3l-node{node}-gen{gen}-r{i}.log')
        if p.exists():
            out.append(f'--- node {node} rank {i} ---\n' + p.read_text(errors='ignore')[-chars:])
    return '\n'.join(out)


def main():
    """Local agent for tests: group.py STORE_JSON RUN_KEY NODE [RUN_DIR]"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from supervisor import JsonStore
    run_dir = sys.argv[4] if len(sys.argv) > 4 else None
    print(run_agent(JsonStore(sys.argv[1]), sys.argv[2], int(sys.argv[3]), run_dir=run_dir, poll_s=.1, report_s=.5,
                    idle_exit_s=120), flush=True)


if __name__ == '__main__':
    main()
