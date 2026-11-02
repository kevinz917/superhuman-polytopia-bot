"""Local scheduling and credential-free container boundaries, without inference."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from unittest.mock import Mock, patch

import pytest

from polytopia_bot.skirmish.local import local_cells, task
from polytopia_bot.skirmish.local_scheduler import cpu_slot, StageClock
from polytopia_bot.skirmish.docker_sandbox import DockerSandbox


def test_local_roster_includes_fable_and_routes_all_openai_anthropic_to_proxy():
    manifest = json.loads(Path('configs/skirmish/local-v1-top12.json').read_text())
    cells = local_cells('fixture', manifest)
    assert len(cells) == 72
    assert len({c['run_id'] for c in cells}) == 72
    assert sum(c['model_label']=='fable51' for c in cells) == 6
    for cell in cells:
        expected = 'auto' if cell['model'].startswith(('openai/', 'anthropic/')) else 'openrouter'
        assert cell['config']['provider'] == expected


def test_many_sessions_share_bounded_cpu_slots(tmp_path):
    active = 0; peak = 0; lock = threading.Lock(); waits = []
    def work(index):
        nonlocal active, peak
        with cpu_slot(tmp_path, 3) as waited:
            with lock:
                active += 1; peak = max(peak, active); waits.append(waited)
            time.sleep(.025)
            with lock:
                active -= 1
    with ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(work, range(24)))
    assert peak == 3 and max(waits) > .05


def test_cpu_queue_wait_extends_cutoff_but_not_coding_allowance(tmp_path, monkeypatch):
    sandbox = Mock()
    clock = StageClock(tmp_path/'clock.json', sandbox)
    clock.save(dict(started_unix=0, deadline_unix=40, scheduler_wait_seconds=0))
    monkeypatch.setattr('polytopia_bot.skirmish.local_scheduler.time.time', lambda:10)
    clock.pause()
    monkeypatch.setattr('polytopia_bot.skirmish.local_scheduler.time.time', lambda:25)
    assert clock.resume() == 55
    assert clock.read()['scheduler_wait_seconds'] == 15
    assert clock.read()['deadline_unix']-clock.read()['scheduler_wait_seconds'] == 40
    assert 'paused_unix' not in clock.read()
    assert json.loads(sandbox.filesystem.write_text.call_args.args[0]) == clock.read()


def test_coder_container_has_no_network_host_mounts_or_credentials():
    with patch('polytopia_bot.skirmish.docker_sandbox.subprocess.run') as run:
        sandbox = DockerSandbox.create('fixed-image-id')
    args = run.call_args.args[0]
    assert args[args.index('--network')+1] == 'none'
    assert args[args.index('--cpus')+1] == '2'
    assert args[args.index('--memory')+1] == '2g'
    assert '--mount' not in args and '-v' not in args and '-e' not in args and '--privileged' not in args
    assert 'ALL' in args and 'no-new-privileges=true' in args


def test_official_local_controls_preserve_feedback_and_submission_constraints():
    first = task(0,2400,0,'')
    revised = task(1,600,3,'\nFEEDBACK_PATH')
    assert 'No expert-game feedback' in first
    assert 'six complete public traces' in revised and 'FEEDBACK_PATH' in revised
    assert 'there is no rollback' in first
