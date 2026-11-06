#!/usr/bin/env python3
"""Describe a complete paired position evaluation without changing its gates."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from polytopia_bot.learning.smoke_gate import position_comparison


def summarize(base, smoke, output):
    paths = [ROOT / base / 'report.json', ROOT / smoke / 'report.json']
    before, after = [json.loads(p.read_text()) for p in paths]
    result = position_comparison(before, after)
    left = {r['position_id']: r for r in before['results']}
    pairs = [(left[r['position_id']], r) for r in after['results']]
    clusters = defaultdict(lambda: [0, 0])
    for a, b in pairs:
        clusters[b['board_hash']][0] += int(b['teacher_agreement']) - int(a['teacher_agreement'])
        clusters[b['board_hash']][1] += 1
    groups = np.array([clusters[k] for k in sorted(clusters)])
    indices = np.random.default_rng(0).integers(0, len(groups), size=(10000, len(groups)))
    resampled = groups[indices].sum(axis=1)
    ci = np.quantile(resampled[:, 0] / resampled[:, 1], [.025, .975]).tolist()
    breakdowns = {}
    for field in ('phase', 'family'):
        grouped = defaultdict(list)
        for a, b in pairs:
            grouped[str(b[field])].append((a, b))
        breakdowns[field] = {}
        for value, rows in sorted(grouped.items()):
            old = sum(a['teacher_agreement'] for a, b in rows) / len(rows)
            new = sum(b['teacher_agreement'] for a, b in rows) / len(rows)
            breakdowns[field][value] = dict(count=len(rows), base_agreement=old, smoke_agreement=new, gain=new-old)
    seconds = after.get('occupied_seconds_with_archive', after['occupied_function_seconds'])
    result.update(decisions=len(pairs), base_first_attempt_valid=before['first_attempt_valid']/len(pairs),
        paired_map_cluster_bootstrap=dict(resamples=10000, seed=0, maps=len(groups), agreement_gain_95pct=ci),
        breakdowns=breakdowns,
        selected_action_families=dict(Counter(str(b['selected_command']['kind']) for a, b in pairs if b['selected_command'])),
        occupied_seconds_with_archive=seconds, estimated_compute_usd=seconds*.00078192,
        evidence=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        summary_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='Validation saved positions only. The map-cluster bootstrap describes this paired evaluation, not full-game competence or final-test acceptance.')
    with (ROOT / output).open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('breakdowns', 'evidence')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'smoke', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    summarize(a.base, a.smoke, a.output)
