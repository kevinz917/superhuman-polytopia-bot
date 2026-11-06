"""Shared persistent WASI transport for evaluation and offline protocol checks."""
from dataclasses import asdict
import json
import os
import selectors
import subprocess
import sys
import time

from .common import canonical


class BotFault(Exception):
    pass


class WasiBot:
    def __init__(self, path, fuel, seconds):
        # No model credentials are inherited by the guest worker process either.
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'PYTHONPATH', 'LANG')}
        self.proc = subprocess.Popen([sys.executable, '-m', 'polytopia_bot.skirmish.wasi_worker', str(path), str(fuel)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, bufsize=0)
        self.seconds = seconds
        self.first = True
        self.buffer = bytearray()
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        # Sending a large observation to a bot that never reads must also have a deadline.
        os.set_blocking(self.proc.stdin.fileno(), False)
        os.set_blocking(self.proc.stdout.fileno(), False)

    def select(self, observation):
        # One-time Wasmtime startup allowance; subsequent decisions use the fixed limit.
        deadline = time.monotonic() + self.seconds + (5 if self.first else 0)
        self.first = False
        deadline = min(deadline,getattr(self,'absolute_deadline',deadline))
        data = memoryview((canonical(asdict(observation)) + '\n').encode())
        import select
        try:
            while data:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([], [self.proc.stdin], [], left)[1]:
                    raise BotFault('decision_timeout')
                n = os.write(self.proc.stdin.fileno(), data)
                data = data[n:]
            while b'\n' not in self.buffer:
                left = deadline - time.monotonic()
                if left <= 0 or not self.selector.select(left):
                    raise BotFault('decision_timeout')
                chunk = os.read(self.proc.stdout.fileno(), 4096)
                if not chunk:
                    try:
                        code = self.proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        code = None
                    raise BotFault('fuel_exhausted' if code == 42 else 'guest_exit')
                self.buffer.extend(chunk)
                if len(self.buffer) > 16_384:
                    raise BotFault('oversized_response')
            line, _, tail = self.buffer.partition(b'\n')
            self.buffer = bytearray(tail)
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != {'action_id'} or not isinstance(value['action_id'], str):
                raise BotFault('malformed_response')
            action_id = value['action_id']
            if action_id not in {a.action_id for a in observation.legal_actions}:
                raise BotFault('illegal_action')
            return action_id
        except (BrokenPipeError,OSError,ValueError,TypeError) as error:
            # A guest may exhaust fuel while the parent is still writing a large
            # observation. Preserve the actual exit classification on that path.
            try:code=self.proc.wait(timeout=.2)
            except subprocess.TimeoutExpired:code=None
            raise BotFault('fuel_exhausted' if code==42 else 'malformed_response_or_guest_exit') from error

    def close(self):
        self.proc.kill()
        self.proc.wait()
        self.selector.close()
        self.proc.stdin.close()
        self.proc.stdout.close()
