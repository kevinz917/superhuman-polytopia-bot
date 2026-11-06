"""Many model sessions, bounded CPU work, and explicit uncharged scheduler waits."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time

from .common import write_json


@contextmanager
def cpu_slot(directory, count=6, timeout=14400):
    """Cross-process host locks, shared by compilation and referee workers."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    files = [(directory/f'{i}.lock').open('a') for i in range(count)]
    start = time.monotonic(); acquired = None
    try:
        while acquired is None:
            for stream in files:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = stream
                    break
                except BlockingIOError:
                    pass
            if acquired is None:
                if time.monotonic()-start >= timeout:
                    raise TimeoutError('local CPU scheduler wait exceeded infrastructure deadline')
                time.sleep(.02)
        yield time.monotonic()-start
    finally:
        if acquired is not None:
            fcntl.flock(acquired, fcntl.LOCK_UN)
        for stream in files:
            stream.close()


class StageClock:
    def __init__(self, path, sandbox):
        self.path, self.sandbox = Path(path), sandbox

    def read(self):
        return json.loads(self.path.read_text())

    def save(self, state):
        write_json(self.path, state)
        self.sandbox.filesystem.write_text(json.dumps(state), '/verify/stage-start.json')

    def pause(self):
        state = self.read()
        state.update(paused_unix=time.time(), pause_lease_unix=time.time()+14400)
        self.save(state)

    def resume(self):
        state = self.read()
        wait = max(0, time.time()-state.pop('paused_unix'))
        state.pop('pause_lease_unix')
        state['deadline_unix'] += wait
        state['scheduler_wait_seconds'] = state.get('scheduler_wait_seconds', 0)+wait
        self.save(state)
        return state['deadline_unix']


def bash_environment(sandbox, deadline, timing, config):
    from .mini_worker import ModalBashEnvironment

    class ScheduledBashEnvironment(ModalBashEnvironment):
        def execute(self, action, **kwargs):
            from minisweagent.exceptions import TimeExceeded
            if time.time() >= self.deadline:
                raise TimeExceeded(dict(role='exit', content='TimeExceeded', extra=dict(exit_status='TimeExceeded')))
            clock = StageClock(config['start_path'], sandbox)
            if timing: timing.event('scheduler_wait_start')
            clock.pause()
            try:
                with cpu_slot(config['cpu_slots_directory'], config.get('cpu_slots', 6)):
                    self.deadline = clock.resume()
                    if timing: timing.event('scheduler_wait_end')
                    try:
                        return super().execute(action, **kwargs)
                    finally:
                        # Background coder jobs cannot consume CPU outside their slot.
                        proc = sandbox.exec('pkill', '-KILL', '-u', 'coder', timeout=10)
                        proc.wait()
            except BaseException:
                state = clock.read()
                if 'paused_unix' in state:
                    self.deadline = clock.resume()
                raise

    return ScheduledBashEnvironment(sandbox, deadline, timing)


WATCHDOG = '''import json,time,os
from pathlib import Path
Path('/verify/watchdog.pid').write_text(str(os.getpid()))
p=Path('/verify/stage-start.json');waited=time.monotonic()
while not p.exists():
 if time.monotonic()-waited>120: raise TimeoutError('no stage start')
 time.sleep(.01)
while True:
 state=json.loads(p.read_text());now=time.time()
 if 'paused_unix' in state:
  if now>=state['pause_lease_unix']: break
 elif now>=state['deadline_unix']: break
 time.sleep(.01)
'''
