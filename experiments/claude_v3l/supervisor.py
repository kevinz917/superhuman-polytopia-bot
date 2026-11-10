"""Session L stage supervisor: runs inside the stage worker container (or locally in tests) and applies the tiered
error policy (policy.py, LAUNCH-PLAN.md section 2) around the trainer subprocess (flagship_harness.py).

    sup = Supervisor(manifest, launch_id, run_dir, store, commit)
    outcome = sup.run()      # {'state': 'paused'|'complete', 'reason': ...}; raises only on SIGTERM (let Modal retry)

`store` is the control channel shared with the launcher and the monitor (modal.Dict in the cloud, JsonStore locally):
  <run_id>/pause      set by the launcher/monitor -> graceful checkpoint and stop
  <run_id>/heartbeat  written every `heartbeat_s` by this supervisor
  <run_id>/events     the last 300 events (the monitor relays new ids)
  <run_id>/status     the final state of this worker
`commit` persists the run volume (modal.Volume.commit) and is called after every new good checkpoint and event.
"""
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from policy import PolicyState, classify_exit, now, _read, write_json_atomic  # noqa: E402
from group import LOST_RC, REFUSED_RC, tail_of  # noqa: E402


class JsonStore:
    """A file-backed stand-in for modal.Dict (tests and local dry runs)."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def _load(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, key, default=None):
        with self.lock:
            return self._load().get(key, default)

    def put(self, key, value):
        with self.lock:
            d = self._load()
            d[key] = value
            write_json_atomic(self.path, d)

    def pop(self, key, default=None):
        with self.lock:
            d = self._load()
            v = d.pop(key, default)
            write_json_atomic(self.path, d)
            return v


def combine_rcs(rcs):
    """One trainer exit code from the ranks' codes. NaN on any rank wins (the pmean'd metrics make it all-or-none);
    then rank 0's own code (it alone decides stop reasons: stop file, leakage, global illegal rate); then the first
    rank that failed on its own (codes of ranks the supervisor killed, -9, do not count)."""
    from policy import EXIT_NAN
    if EXIT_NAN in rcs:
        return EXIT_NAN
    if rcs[0] == 0:
        # rank 0 exits 0 only after its final checkpoint/log writes; ranks > 0 can then abort (-6) in
        # jax.distributed's shutdown barrier once rank 0's coordination service is gone (Q3c 11:09 PT). Not a crash.
        return 0
    if rcs[0] not in (0, -9, None):
        return rcs[0]
    own = [rc for rc in rcs[1:] if rc not in (0, -9, None)]
    if own:
        return own[0]
    if rcs[0] == -9 and len(rcs) > 1:
        return -9
    return rcs[0]


class NodeLost(Exception):
    """A remote node stopped reporting (its container is gone): the clustered input must fail so Modal restarts the
    whole cluster; the next attempt counts a container_lost resume."""


class SigTerm(Exception):
    """The container is being stopped (preemption / app stop): no end record, so the next attempt counts a resume."""


class Supervisor:
    def __init__(self, manifest, launch_id, run_dir, store, commit=lambda: None, *, poll_s=None, heartbeat_s=None,
                 kill_grace_s=None, python=sys.executable, coord_host='localhost', prepare_gen=None):
        self.m, self.launch_id, self.run_dir = manifest, launch_id, Path(run_dir)
        self.store, self.commit = store, commit
        self.state = PolicyState(self.run_dir, manifest)
        self.key = manifest['run_id']
        self.poll_s = float(poll_s if poll_s is not None else manifest.get('poll_s', 30))
        self.heartbeat_s = float(heartbeat_s if heartbeat_s is not None else manifest.get('heartbeat_s', 30))
        self.kill_grace_s = float(kill_grace_s if kill_grace_s is not None else manifest.get('kill_grace_s', 600))
        self.python = python
        # several nodes (Modal clustered, 16 GPUs = 2 x 8): this supervisor runs on node 0; group.run_agent on the others
        self.nodes = int(manifest.get('nodes', 1))
        self.coord_host = coord_host
        self.prepare_gen = prepare_gen or (lambda: None)
        self.gen = 0
        self.procs = []
        self.attempt = None
        self.proc = None
        self.stop_file = None
        self.snap_file = None
        self.stop_reason = None
        self.sigterm = False
        self.last_update = {}
        self._lock = threading.Lock()

    # ---- notifications
    def emit(self, kind, **fields):
        row = self.state.event(kind, attempt=self.attempt, **fields)
        self._push_events()
        return row

    def _push_events(self):
        try:
            self.store.put(f'{self.key}/events', self.state.events()[-300:])
        except Exception as exc:      # the control channel is best-effort; the volume copy is authoritative
            print(f'[v3l] store events failed: {exc!r}', flush=True)
        try:
            self.commit()
        except Exception as exc:
            print(f'[v3l] commit failed: {exc!r}', flush=True)

    # ---- stop requests to the running trainer
    def request_stop(self, reason):
        with self._lock:
            if self.stop_reason is None:
                self.stop_reason = reason
                print(f'[v3l] stop requested: {reason}', flush=True)
            if self.stop_file is not None:
                Path(self.stop_file).write_text(self.stop_reason)

    def _on_sigterm(self, signum, frame):
        self.sigterm = True
        self.request_stop('sigterm')

    # ---- the watcher thread: heartbeat, pause flag, budget line, volume commits on new checkpoints
    def _watch(self, done):
        self._last_hb, self._good_mtime = 0., None
        while not done.wait(self.poll_s):
            try:
                self._watch_once()
            except Exception as exc:      # never let one bad iteration end pause / budget enforcement
                print(f'[v3l] watcher iteration failed: {exc!r}', flush=True)

    def _watch_once(self):
        try:
            if self.store.get(f'{self.key}/pause'):
                self.request_stop('pause_request')
        except Exception as exc:
            print(f'[v3l] store read failed: {exc!r}', flush=True)
        left = self.state.seconds_left(self.attempt)
        if left <= 0:
            self.request_stop('stage_budget')
        self._dollar_snapshots()
        if now() - self.started > float(self.m.get('timeout_s', 86400)) - 900:
            self.request_stop('container_timeout_margin')
        good = self.run_dir / 'GOOD.json'
        if good.exists() and good.stat().st_mtime != self._good_mtime:
            self._good_mtime = good.stat().st_mtime
            try:
                self.commit()
            except Exception as exc:
                print(f'[v3l] commit failed: {exc!r}', flush=True)
        if now() - self._last_hb >= self.heartbeat_s:
            self._last_hb = now()
            hb = dict(attempt=self.attempt, stage=self.state.stage, launch_id=self.launch_id,
                      spent_usd=round(self.state.stage_spent_usd(self.attempt), 4),
                      budget_usd=self.state.stage_budget_usd(), seconds_left=round(left, 1),
                      stop_reason=self.stop_reason, **self.last_update)
            self.state.heartbeat(self.attempt, **{k: v for k, v in hb.items() if k not in ('attempt', 'stage')})
            try:
                self.store.put(f'{self.key}/heartbeat', dict(hb, t=now()))
            except Exception as exc:
                print(f'[v3l] store heartbeat failed: {exc!r}', flush=True)
            self._push_events()

    # ---- snapshots at matched cumulative campaign dollars (flagship vs CNN control crossover, Jack ~08:55 PT)
    def cumulative_usd(self):
        return float(self.m.get('campaign_spent_before_usd', 0.)) + self.state.stage_spent_usd(self.attempt)

    def _dollar_snapshots(self):
        marks = [float(x) for x in self.m.get('snapshot_at_usd', [])]
        if not marks or self.snap_file is None:
            return
        done = {r['label'] for r in self.state.usd_snapshots()}
        cum = self.cumulative_usd()
        due = [f'usd-{x:g}' for x in marks if x <= cum and f'usd-{x:g}' not in done]
        if due and not os.path.exists(self.snap_file):
            Path(self.snap_file).write_text('\n'.join(due))

    # ---- one trainer process
    def _trainer_cmd(self):
        harness = dict(run_id=self.key, stage=self.state.stage, run_dir=str(self.run_dir), stop_file=self.stop_file,
                       snap_file=self.snap_file, cumulative_usd_at_start=round(self.cumulative_usd(), 4),
                       faults=self.m.get('faults', []), attempt=self.attempt,
                       leak_check=self.m.get('leak_check', True),
                       illegal_pause_fraction=self.m.get('illegal_pause_fraction', 1e-4),
                       illegal_min_decisions=self.m.get('illegal_min_decisions', 10_000_000),
                       illegal_abs_pause=self.m.get('illegal_abs_pause', 1000),
                       ckpt_keep=self.m.get('ckpt_keep', 3), manifest_sha256=self.m.get('manifest_sha256', ''),
                       lr_restart=self.m.get('lr_restart'), illegal_check_every=self.m.get('illegal_check_every', 10),
                       stage4=self.m.get('stage4'))
        script = self.m.get('harness_script', str(HERE / 'flagship_harness.py'))
        procs = int(self.m.get('processes', 1))
        if procs == 1:
            return [[self.python, script, '--harness-json', json.dumps(harness), '--', *self.m['trainer_args']]]
        # one trainer process per GPU (session A's GIL-free layout): rank r drives CUDA ordinal r; jax.distributed joins
        # them; rank 0 is the only writer (checkpoints, GOOD.json, events, snapshots)
        port = random.randint(20000, 40000)
        cmds = []
        for r in range(procs):
            hr = dict(harness, rank=r, processes=procs)
            cmds.append([self.python, script, '--harness-json', json.dumps(hr), '--', *self.m['trainer_args'],
                         '--devices', '1', '--num-processes', str(procs), '--process-id', str(r),
                         '--coordinator', f'{self.coord_host}:{port}'])
        return cmds

    def _run_trainer(self):
        self.stop_file = str(Path(self.m.get('stop_dir', '/tmp')) / f'v3l-stop-{self.key}-{self.attempt}-{time.time_ns()}')
        self.snap_file = self.stop_file.replace('v3l-stop-', 'v3l-snap-')
        if self.stop_reason:
            Path(self.stop_file).write_text(self.stop_reason)
        all_cmds = self._trainer_cmd()
        per = len(all_cmds) // self.nodes
        cmds, remote = all_cmds[:per], {n: all_cmds[n * per:(n + 1) * per] for n in range(1, self.nodes)}
        print(f'[v3l] trainer x{len(all_cmds)} ({self.nodes} node(s)):', ' '.join(cmds[0][:4]), '...', flush=True)
        env = dict(os.environ, **{k: str(v) for k, v in self.m.get('env', {}).items()})
        if remote:
            # repair + commit the run dir, then publish the generation with GOOD's sha (group.py docstring)
            good_sha = self.prepare_gen()
            self.gen += 1
            self.gen_started = time.time()
            self.store.put(f'{self.key}/grp/gen', dict(gen=self.gen, action='run', launch_id=self.launch_id,
                                                        attempt=self.attempt, nodes={str(n): c for n, c in remote.items()},
                                                        env=self.m.get('env', {}), good_sha=good_sha, t=now()))
        procs, logs = [], []
        for r, cmd in enumerate(cmds):
            e = dict(env)
            if len(cmds) > 1:
                e['CUDA_VISIBLE_DEVICES'] = str(r)
            if r == 0:
                out = subprocess.PIPE
            else:
                path = Path(self.m.get('stop_dir', '/tmp')) / f'v3l-rank{r}-{self.attempt}-{time.time_ns()}.log'
                logs.append(path)
                out = open(path, 'w')
            procs.append(subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, text=True, env=e,
                                          cwd=self.m.get('cwd') or None))
        self.proc = procs[0]
        self.procs = procs
        tail = []

        def read0():
            for line in procs[0].stdout:
                print(line, end='', flush=True)
                tail.append(line)
                del tail[:-200]
                if '"stage": "update"' in line:
                    try:
                        row = json.loads(line)
                        self.last_update = {k: row.get(k) for k in ('update', 'samples', 'elapsed', 'samples_per_second',
                                                                     'nonok', 'finished_total', 'learning_rate')}
                    except json.JSONDecodeError:
                        pass
        reader = threading.Thread(target=read0, daemon=True)
        reader.start()
        # wait: the first exit decides; a non-zero exit (or rank 0 ending) gives the others a grace period, then kill:
        # a rank that died mid-collective leaves the rest blocked forever. Remote nodes report through the store.
        first = None
        remote_rcs = {n: None for n in remote}
        last_poll = 0.
        stop_sent = False

        def stop_remote():
            nonlocal stop_sent
            if remote and not stop_sent:
                self.store.put(f'{self.key}/grp/gen', dict(self.store.get(f'{self.key}/grp/gen') or {}, action='stop',
                                                            t=now()))
                stop_sent = True

        while True:
            if remote and time.time() - last_poll > 2:
                last_poll = time.time()
                for n in remote:
                    if remote_rcs[n] is not None:
                        continue
                    rec = self.store.get(f'{self.key}/grp/node{n}') or {}
                    if rec.get('gen') == self.gen and rec.get('state') in ('exited', 'refused'):
                        remote_rcs[n] = list(rec.get('rcs') or [LOST_RC] * per)
                    elif time.time() - self.gen_started > float(self.m.get('node_start_grace_s', 600)) and \
                            time.time() - float(rec.get('t') or 0) > float(self.m.get('node_stale_s', 90)):
                        remote_rcs[n] = [LOST_RC] * per      # stopped reporting: container lost
            rcs = [p.poll() for p in procs]
            local_done = all(rc is not None for rc in rcs)
            remote_done = all(v is not None for v in remote_rcs.values())
            if local_done and remote_done:
                break
            any_exit = any(rc is not None for rc in rcs) or any(v is not None for v in remote_rcs.values())
            if first is None and any_exit:
                first = time.time()
            if first is not None:
                bad = any(rc not in (None, 0) for rc in rcs) or any(
                    v is not None and any(x != 0 for x in v) for v in remote_rcs.values())
                if time.time() - first > (float(self.m.get('rank_grace_s', 20)) if bad else 180.):
                    for p in procs:
                        if p.poll() is None:
                            p.kill()
                    stop_remote()
                    if local_done and time.time() - first > (float(self.m.get('rank_grace_s', 20)) if bad else 180.) + 60:
                        for n in remote:                 # the agent never confirmed the stop: count it as lost
                            if remote_rcs[n] is None:
                                remote_rcs[n] = [LOST_RC] * per
            time.sleep(.5)
        reader.join(timeout=10)
        self.proc = None
        rcs = [p.returncode for p in procs]
        for r, path in enumerate(logs, start=1):
            if rcs[r] not in (0,):
                try:
                    tail.append(f'--- rank {r} rc {rcs[r]} tail ---\n' + path.read_text()[-1500:])
                except OSError:
                    pass
        for n in sorted(remote):
            rcs += remote_rcs[n]
            if any(x != 0 for x in remote_rcs[n]):
                tail.append(f'--- node {n} rcs {remote_rcs[n]} ---')
        if remote and any(x == REFUSED_RC for x in rcs):
            # a node refused to start (stale checkpoint view): the generation never had all ranks, whatever rank 0 says
            self.emit('node_refused', gen=self.gen, rcs=rcs)
            return REFUSED_RC, ''.join(tail)
        if remote and any(x == LOST_RC for x in rcs):
            self.emit('node_lost', gen=self.gen, rcs=rcs)
            raise NodeLost(f'node(s) stopped reporting in generation {self.gen}; the clustered input must restart')
        return combine_rcs(rcs), ''.join(tail)

    # ---- the stage
    def _pause(self, reason, **extra):
        self.state.set_status('paused', reason, attempt=self.attempt, launch_id=self.launch_id, **extra)
        self.emit('paused', reason=reason, **extra)
        self.state.end_attempt(self.attempt, f'paused:{reason}')
        out = dict(state='paused', reason=reason, **extra)
        self.store.put(f'{self.key}/status', dict(out, t=now()))
        self.commit()
        return out

    def run(self):
        self.started = now()
        old = self.state.status()
        self.attempt, lost = self.state.begin_attempt(self.launch_id)
        try:
            signal.signal(signal.SIGTERM, self._on_sigterm)
        except ValueError:            # not the main thread (some test runners)
            pass
        if old and old.get('state') in ('paused', 'complete') and old.get('launch_id') == self.launch_id:
            # the same launch already finished: a duplicate start must not train
            self.state.end_attempt(self.attempt, 'duplicate_start')
            return dict(state=old['state'], reason='duplicate_start_of_finished_launch')
        self.state.set_status('running', launch_id=self.launch_id, attempt=self.attempt)
        try:       # the control channel's status must not keep a previous launch's final state (Q3c 11:01 PT)
            self.store.put(f'{self.key}/status', dict(state='running', launch_id=self.launch_id, attempt=self.attempt,
                                                      t=now()))
        except Exception as exc:
            print(f'[v3l] store status failed: {exc!r}', flush=True)
        self.emit('stage_start', launch_id=self.launch_id, lost_previous=lost,
                  spent_usd=round(self.state.stage_spent_usd(self.attempt), 4), budget_usd=self.state.stage_budget_usd())
        if lost:
            n = self.state.count_resume('container_lost')
            self.emit('container_lost_resume', resumes=n)
            if not self.state.resume_allowed():
                return self._pause('resume_limit', resumes=n)
        expected = self.m.get('source_sha')
        actual = os.environ.get('V3L_SOURCE_SHA')
        if expected and actual and expected != actual:
            return self._pause('config_mismatch', expected=expected, actual=actual)
        if self.store.get(f'{self.key}/pause'):
            return self._pause('pause_request', note='pause flag set before start')
        done = threading.Event()
        watcher = threading.Thread(target=self._watch, args=(done,), daemon=True, name='v3l-watch')
        watcher.start()
        try:
            while True:
                if self.state.seconds_left(self.attempt) <= 0:
                    return self._pause('stage_budget', spent_usd=round(self.state.stage_spent_usd(self.attempt), 4))
                rc, tail = self._run_trainer()
                if self.sigterm:
                    raise SigTerm('container stop signal; next attempt resumes from latest.npz')
                complete = (self.run_dir / 'training-complete.json').exists() and not self.stop_reason
                stop_file_reason = self.stop_reason
                action, reason = classify_exit(rc, stop_file_reason, complete)
                upd = self.last_update.get('update')
                if action == 'complete':
                    self.state.set_status('complete', reason, attempt=self.attempt, launch_id=self.launch_id)
                    self.emit('complete', update=upd)
                    self.state.end_attempt(self.attempt, 'complete')
                    self.store.put(f'{self.key}/status', dict(state='complete', reason=reason, t=now()))
                    self.commit()
                    return dict(state='complete', reason=reason)
                if action == 'pause':
                    return self._pause(reason, update=upd, rc=rc, tail=tail[-1500:] if rc else '')
                if action == 'rollback':
                    recent = self.state.record_nan(upd)
                    if self.state.nan_should_pause(recent):
                        return self._pause('nan_repeat', nan_in_window=recent, update=upd)
                    self.emit('nan_rollback', update=upd, nan_in_window=recent, rolled_back_to=self._good_stage())
                    continue
                # resume after an unclassified trainer crash inside a live container
                n = self.state.count_resume(reason)
                self.emit('trainer_crash_resume', rc=rc, resumes=n, update=upd, tail=tail[-1500:])
                if not self.state.resume_allowed():
                    return self._pause('resume_limit', resumes=n, rc=rc)
        finally:
            done.set()
            if self.nodes > 1:
                try:
                    self.store.put(f'{self.key}/grp/gen', dict(self.store.get(f'{self.key}/grp/gen') or {},
                                                                action='exit', t=now()))
                except Exception as exc:
                    print(f'[v3l] group exit publish failed: {exc!r}', flush=True)
            for p in getattr(self, 'procs', []):
                if p.poll() is None:
                    p.terminate()

    def _good_stage(self):
        good = self.run_dir / 'GOOD.json'
        return json.loads(good.read_text()).get('stage') if good.exists() else None


def main():
    """Local CLI for tests: supervisor.py MANIFEST_JSON LAUNCH_ID RUN_DIR STORE_JSON"""
    manifest = json.loads(Path(sys.argv[1]).read_text())
    out = Supervisor(manifest, sys.argv[2], sys.argv[3], JsonStore(sys.argv[4])).run()
    print('SUPERVISOR ' + json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
