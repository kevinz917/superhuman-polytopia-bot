"""Controller-independent cutoff source capture and ready/start handshake."""
import json
from pathlib import Path
import subprocess
import time

from .common import write_json
from .telemetry import summarize_timing

# Runs as root inside the coding sandbox, not as the agent. The coder cannot alter
# /verify. A watchdog captures the source even if the authoring controller is lost.
CAPTURE = '''import hashlib,json,os,subprocess,time,fcntl
from pathlib import Path
subprocess.run(['pkill','-KILL','-u','coder'])
root=Path('/verify/cutoffs')/('stage-'+str(stage));root.mkdir(parents=True,exist_ok=True)
lock=(root/'capture.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
if not (root/'capture.json').exists():
 try:
  source=Path('/bot/src/main.rs').read_bytes();tmp=root/'main.rs.tmp';tmp.write_bytes(source);tmp.replace(root/'main.rs')
  receipt=dict(captured_unix=time.time(),source_sha256=hashlib.sha256(source).hexdigest(),source_bytes=len(source))
 except OSError as error:receipt=dict(captured_unix=time.time(),error=type(error).__name__)
 tmp=root/'capture.json.tmp';tmp.write_text(json.dumps(receipt));tmp.replace(root/'capture.json')
'''


def capture_cutoff(sandbox, stage):
    proc=sandbox.exec('python','-c',f'stage={int(stage)}\n'+CAPTURE,timeout=15)
    proc.wait()
    if proc.returncode:raise RuntimeError('cutoff capture failed')
    return f'/verify/cutoffs/stage-{int(stage)}'


def run_stage_worker(sandbox, root, stage, payload, start_watchdog, stop_coder, cancel_watchdog, volume=None):
    """No work during startup. Start clock only after worker + watchdog are ready."""
    import sys
    root=Path(root)
    ready=root/f'stage-{stage}.ready.json';start=root/f'stage-{stage}.start.json'
    timing=root/f'stage-{stage}.timing.jsonl'
    if ready.exists() or start.exists():raise ValueError('stage already allocated; recover, never silently replay')
    payload.update(ready_path=str(ready),start_path=str(start),timing=str(timing))
    launched=time.time()
    with (root/f'stage-{stage}.worker.log').open('w') as log:
        worker=subprocess.Popen([sys.executable,'-m','polytopia_bot.skirmish.mini_worker'],
            stdin=subprocess.PIPE,stdout=log,stderr=log,text=True)
        try:
            worker.stdin.write(json.dumps(payload));worker.stdin.close();worker.stdin=None
            while not ready.exists():
                if worker.poll() is not None:raise RuntimeError('worker failed before readiness')
                if time.time()-launched>120:raise TimeoutError('worker readiness timeout')
                time.sleep(.02)
            # Arm watchdog before allowing the model to do work. Its own start file
            # is written by the controller below; arming time is not charged.
            start_watchdog(sandbox,None,stage=stage,start_path='/verify/stage-start.json')
            started=time.time();deadline=started+payload['seconds']
            receipt=dict(started_unix=started,deadline_unix=deadline,stage=stage,
                         startup_seconds=started-launched)
            sandbox.filesystem.write_text(json.dumps(receipt),'/verify/stage-start.json')
            write_json(start,receipt)
            if volume:volume.commit()
            try:
                worker.wait(timeout=max(.01,deadline-time.time()))
                reason='agent_exit' if worker.returncode==0 else 'agent_error'
            except subprocess.TimeoutExpired:
                worker.kill();worker.wait();reason='deadline'
            ended=time.time()
            stop_coder(sandbox)
            capture_cutoff(sandbox,stage)
            cancel_watchdog(sandbox)
            elapsed=min(payload['seconds'],max(0,ended-started))
            metrics=summarize_timing(timing,min(ended,deadline))
            metrics.update(startup_seconds=receipt['startup_seconds'],coding_seconds=elapsed,
                           overhead_seconds=max(0,elapsed-metrics['api_seconds']-metrics['tool_seconds']))
            return reason,elapsed,metrics
        finally:
            if worker.poll() is None:worker.kill();worker.wait()
