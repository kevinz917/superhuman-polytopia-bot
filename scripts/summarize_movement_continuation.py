"""Compare fixed checkpoint predictions with paired map-cluster uncertainty."""
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from polytopia_bot.learning.movement_diagnostic import score_choice, summarize


def paired_change(before, after, metric='tie_aware', iterations=5000):
    import numpy as np
    old = {r['position_id']: r for r in before}
    new = {r['position_id']: r for r in after}
    if len(old) != len(before) or len(new) != len(after) or old.keys() != new.keys() or not old:
        raise ValueError('paired positions differ or are empty')
    clusters = defaultdict(lambda: [0, 0])
    improved = regressed = 0
    for key, row in old.items():
        other = new[key]
        if row['board_hash'] != other['board_hash']:
            raise ValueError('paired map changed')
        delta = int(other[metric]) - int(row[metric])
        clusters[row['board_hash']][0] += delta
        clusters[row['board_hash']][1] += 1
        improved += delta == 1
        regressed += delta == -1
    totals = np.array([clusters[k] for k in sorted(clusters)])
    rng = np.random.default_rng(17)
    draws = rng.integers(0, len(totals), size=(iterations, len(totals)))
    boot = totals[draws].sum(axis=1)
    return dict(delta=(improved-regressed)/len(old), improved=improved, regressed=regressed,
                unchanged=len(old)-improved-regressed, count=len(old), maps=len(totals),
                interval_95=np.quantile(boot[:, 0]/boot[:, 1], [.025, .975]).tolist(),
                method='paired map-cluster percentile bootstrap; exploratory, not adjusted for multiple checkpoints',
                iterations=iterations, seed=17)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups(rows):
    result = dict(train=[r for r in rows if r['split'] == 'train'],
                  held_out=[r for r in rows if r['split'] == 'held_out'])
    result['train_teacher_only'] = [r for r in result['train'] if r['source'] == 'teacher_trajectory']
    result['train_corrections'] = [r for r in result['train'] if r['source'] != 'teacher_trajectory']
    result.update({key+'_movement_required': [r for r in group if r['movement_required_for_top_score']]
                   for key, group in list(result.items())})
    return result


def finish(root):
    root = Path(root)
    protocol = read(root/'protocol.json')
    for name, key in [('positions.jsonl.gz', 'positions_sha256'), ('scoring.jsonl.gz', 'scoring_sha256')]:
        if sha(root/name) != protocol[key]:
            raise ValueError('frozen evaluation input changed')
    positions = {r['position_id']: r for r in map(json.loads, gzip.open(root/'positions.jsonl.gz', 'rt'))}
    metadata = list(map(json.loads, gzip.open(root/'scoring.jsonl.gz', 'rt')))
    if len(positions) != protocol['evaluation_count'] or len(metadata) != 1873:
        raise ValueError('unexpected input counts')
    baseline = None
    baseline_groups = None
    baseline_controls = None
    checkpoints = {}
    for step in protocol['checkpoint_steps']:
        report = read(root/f'report-{step}.json')
        predictions = {r['position_id']: r for r in report['results']}
        if not report['complete'] or report['decisions'] != len(positions) or len(report['results']) != len(positions) or predictions.keys() != positions.keys():
            raise ValueError('incomplete or duplicate checkpoint predictions')
        for key, pred in predictions.items():
            for field in ['board_hash', 'episode_id', 'phase', 'teacher_command']:
                if pred[field] != positions[key][field]:
                    raise ValueError('prediction input differs')
            if baseline is not None and pred['input_tokens'] != baseline[key]['input_tokens']:
                raise ValueError('checkpoint prompt token count changed')
        if step == 0:
            baseline = predictions
        rows = []
        for meta in metadata:
            pred = predictions[meta['position_id']]
            for field in ['board_hash', 'episode_id', 'phase', 'teacher_command', 'prompt_digest']:
                if meta[field] != positions[meta['position_id']][field]:
                    raise ValueError('scoring metadata input differs')
            row = {k: v for k, v in meta.items() if k != 'scores'}
            row.update(score_choice(meta['scores'], meta['teacher_command'], pred['selected_command']))
            row.update(first_attempt_valid=pred['first_attempt_valid'], input_tokens=pred['input_tokens'])
            if row['exact'] != pred['teacher_agreement']:
                raise ValueError('exact agreement mismatch')
            rows.append(row)
        grouped = groups(rows)
        controls = [predictions[key] for key in protocol['control_ids']]
        if step == 0:
            baseline_groups = grouped
            baseline_controls = controls
        phases = sorted({r['phase'] for r in rows})
        controls_by_kind = {}
        for kind in range(2, 11):
            selected = [r for r in controls if r['family'] == kind]
            controls_by_kind[str(kind)] = dict(count=len(selected), exact_matches=sum(r['teacher_agreement'] for r in selected),
                                               first_attempt_valid=sum(r['first_attempt_valid'] for r in selected))
        checkpoints[str(step)] = dict(step=step, additional_examples=min(step*64, 10000),
            additional_epoch_fraction=min(step*64, 10000)/10000,
            groups={key: summarize(group) for key, group in grouped.items()},
            changes_from_baseline={key: paired_change(baseline_groups[key], group) for key, group in grouped.items()},
            by_phase={key: {phase: summarize([r for r in group if r['phase'] == phase]) for phase in phases} for key, group in grouped.items()},
            controls=dict(count=len(controls), exact_matches=sum(r['teacher_agreement'] for r in controls),
                          first_attempt_valid=sum(r['first_attempt_valid'] for r in controls), by_kind=controls_by_kind,
                          change_from_baseline=paired_change(baseline_controls, controls, metric='teacher_agreement')),
            first_attempt_valid=report['first_attempt_valid'], eventually_valid=report['eventually_valid'],
            inference_seconds=report['elapsed_seconds'])
    training_maps = {r['board_hash'] for r in metadata if r['split'] == 'train'}
    held_maps = {r['board_hash'] for r in metadata if r['split'] == 'held_out'}
    if training_maps & held_maps:
        raise ValueError('training and held-out maps overlap')
    best_score = max(c['groups']['held_out_movement_required']['tie_aware'] for c in checkpoints.values())
    result = dict(schema='movement-extra-epoch-result-v1', protocol_sha256=sha(root/'protocol.json'),
        report_sha256={str(step): sha(root/f'report-{step}.json') for step in protocol['checkpoint_steps']},
        primary_metric=protocol['primary_metric'], checkpoints=checkpoints,
        best_held_out_movement_required_step=max(protocol['checkpoint_steps'], key=lambda step: checkpoints[str(step)]['groups']['held_out_movement_required']['tie_aware']),
        tied_best_held_out_movement_required_steps=[step for step in protocol['checkpoint_steps'] if checkpoints[str(step)]['groups']['held_out_movement_required']['tie_aware'] == best_score],
        integrity=dict(identical_position_ids=True, identical_prompt_token_counts=True, disjoint_maps=True,
                       training_maps=len(training_maps), held_out_maps=len(held_maps)),
        limitations=['Adapter continuation uses fresh AdamW and a restarted warmup/cosine schedule because the parent saved no optimizer state.',
                     'Tie-aware agreement measures the frozen heuristic teacher, not game strength.',
                     'Training and held-out positions differ in data source and state difficulty; inspect teacher-only training separately.',
                     'Map-cluster intervals are paired exploratory estimates; no multiplicity correction or final-test access.',
                     'A small non-movement control sample is balanced by action family and is not a full-policy evaluation.'])
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({step: dict(train=c['groups']['train']['tie_aware'], held=c['groups']['held_out']['tie_aware'],
                               strict_train=c['groups']['train_movement_required']['tie_aware'], strict_held=c['groups']['held_out_movement_required']['tie_aware'],
                               controls=c['controls']['exact_matches']/c['controls']['count']) for step, c in checkpoints.items()}, indent=2))


if __name__ == '__main__':
    finish(sys.argv[1])
