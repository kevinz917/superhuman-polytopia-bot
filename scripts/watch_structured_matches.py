"""Evaluate newly retained checkpoints against one fixed learned reference.

Reads immutable snapshots on the retained Modal volume. It
never submits, stops, or restarts training. One bounded cloud match runs at a
time. Each pass selects the latest available snapshot per run; older snapshots
remain retained, but this screen is not exhaustive checkpoint evaluation.
"""
import argparse
import os
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from watch_structured_checkpoints import ROOT, FOLDER, safe_name, write_json


def newest_snapshot(root):
    choices = []
    for path in root.glob('update-*'):
        match = re.fullmatch(r'update-(\d+)-[0-9a-f]{12}', path.name)
        if match and path.is_dir() and (path/'latest.npz').is_file() and (path/'latest.json').is_file():
            choices.append((int(match[1]), path))
    return max(choices, default=(0, None), key=lambda item: item[0])[1]


def validate_run_ids(run_ids):
    if (not isinstance(run_ids, list) or not 1 <= len(run_ids) <= 64
            or any(not isinstance(name, str) for name in run_ids)
            or len(set(run_ids)) != len(run_ids)):
        raise ValueError('requires 1-64 unique run IDs')
    for name in run_ids:
        safe_name(name)
    return run_ids


def main(run_ids, reference, *, hours=13., interval=30., maps=4,
         seed_base=300200, watch_name='champion51212', max_matches=0,
         run_list=None):
    if os.environ.get('POLYTOPIA_CLOUD_EVALUATION') != '1':
        raise RuntimeError('Checkpoint comparisons must run in Modal; use experiments/modal_structured_comparisons.py')
    if not 0 < hours <= 13 or not 5 <= interval <= 60 or not 1 <= maps <= 8 or max_matches < 0:
        raise ValueError('invalid bounded match watcher allocation')
    if not 300000 <= seed_base < seed_base + maps <= 400000:
        raise ValueError('development maps only')
    if run_list is not None:
        if run_ids:
            raise ValueError('provide positional run IDs or --run-list, not both')
        run_list = Path(run_list).resolve()
        run_ids = json.loads(run_list.read_text())
    run_ids = validate_run_ids(run_ids)
    safe_name(watch_name)
    reference = Path(reference).resolve()
    meta = json.loads(reference.with_suffix('.json').read_text())
    reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
    if reference_sha != meta['sha256'] or not meta['config']['no_resign']:
        raise ValueError('requires intact non-resigning reference')
    destination = FOLDER/'retained-checkpoints'
    lock = (destination/f'.match-{reference_sha[:12]}-{seed_base}-{maps}.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = destination/f'match-watch-{watch_name}-status.json'
    suffix = f'match-{reference_sha[:12]}-{seed_base}-{maps}maps-dev'
    deadline = time.monotonic() + hours*3600
    active, stream, active_info = None, None, None
    serviced, completed = {}, []
    while time.monotonic() < deadline:
        if run_list is not None:
            try:
                updated = validate_run_ids(json.loads(run_list.read_text()))
                if updated != run_ids:
                    run_ids = updated
                    print(json.dumps(dict(event='run_list_updated', run_ids=run_ids)), flush=True)
            except (OSError, ValueError) as exc:
                # Keep the last valid roster. An edit never kills an in-flight
                # evaluator or resets round-robin service history.
                print(json.dumps(dict(event='run_list_observation_error',
                                      error=type(exc).__name__, detail=str(exc))), flush=True)
        if active is not None and active.poll() is not None:
            summary_path = Path(active_info['output'])/'summary.json'
            result = dict(**active_info, returncode=active.returncode)
            if summary_path.exists():
                result['summary'] = json.loads(summary_path.read_text())
            completed.append(result)
            print(json.dumps(dict(event='match_finished', **result)), flush=True)
            stream.close()
            active, stream, active_info = None, None, None
        candidates = []
        for run_id in run_ids:
            snapshot = newest_snapshot(destination/run_id)
            if snapshot is None:
                continue
            checkpoint_meta = json.loads((snapshot/'latest.json').read_text())
            if checkpoint_meta['sha256'] == reference_sha:
                continue
            output = snapshot/suffix
            # An existing partial output may belong to a still-running process.
            # Never delete it or launch a duplicate evaluation automatically.
            if not output.exists() and not (active_info and active_info['output'] == str(output)):
                candidates.append((serviced.get(run_id, 0.), run_id, snapshot, output))
        if active is None and candidates and (not max_matches or len(completed) < max_matches):
            _, run_id, snapshot, output = min(candidates, key=lambda item: (item[0], item[1]))
            serviced[run_id] = time.time()
            stream = (snapshot/(suffix+'.log')).open('w')
            active = subprocess.Popen([
                str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/evaluate_structured_match.py'),
                '--challenger', str(snapshot/'latest.npz'), '--reference', str(reference),
                '--output', str(output), '--maps', str(maps), '--seed-base', str(seed_base),
                '--seconds-limit', '900'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            active_info = dict(run_id=run_id, checkpoint=str(snapshot), output=str(output),
                               pid=active.pid, started_unix=time.time())
            print(json.dumps(dict(event='match_started', **active_info)), flush=True)
        write_json(status_path, dict(updated_unix=time.time(), reference=str(reference),
            reference_sha256=reference_sha, maps=maps, seed_base=seed_base,
            run_ids=run_ids, run_list=str(run_list) if run_list else None,
            active=active_info, completed=completed,
            available_candidates=[str(item[2]) for item in candidates],
            scope='Latest-snapshot development screen only; full evaluations required for promotion.'))
        if max_matches and len(completed) >= max_matches and active is None:
            break
        time.sleep(interval)
    # A bounded evaluator already in flight may finish after the watcher ends.
    # Its log and output path remain inspectable; never kill it on watcher expiry.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_ids', nargs='*')
    parser.add_argument('--run-list', help='JSON list of run IDs; valid changes apply without restarting')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--hours', type=float, default=13.)
    parser.add_argument('--interval', type=float, default=30.)
    parser.add_argument('--maps', type=int, default=4)
    parser.add_argument('--seed-base', type=int, default=300200)
    parser.add_argument('--watch-name', default='champion51212')
    parser.add_argument('--max-matches', type=int, default=0)
    main(**vars(parser.parse_args()))
