"""Paired map-cluster analysis, with matched model screen and retention checks."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

from summarize_movement_continuation import paired_change


def metrics(rows):
    valid = [r for r in rows if r['valid']]
    moves = [r for r in rows if r['selected_movement']]
    return dict(count=len(rows), maps=len({r['board_hash'] for r in rows}),
        correct=sum(r['tie_aware'] for r in rows), exact=sum(r['exact'] for r in rows),
        tie_aware=sum(r['tie_aware'] for r in rows)/len(rows) if rows else None,
        valid=len(valid), moves=len(moves),
        correct_given_move=sum(r['tie_aware'] for r in moves)/len(moves) if moves else None,
        mean_score_gap=statistics.mean(r['score_gap'] for r in valid) if valid else None,
        mean_input_tokens=statistics.mean(r['input_tokens'] for r in rows) if rows else None,
        mean_output_tokens=statistics.mean(r['output_tokens'] for r in rows) if rows else None,
        at_output_limit=sum(r['output_at_limit'] for r in rows),
        selected_families=dict(Counter(str(r['selected_command']['kind']) if r['selected_command'] else 'invalid' for r in rows)),
        selected_roads=sum(r['selected_command'] is not None and r['selected_command']['kind']==6
                           and r['selected_command']['parameter']==3 for r in rows))


def cohorts(rows):
    return dict(all=rows, movement_required=[r for r in rows if r['movement_required']],
                nonmovement=[r for r in rows if r['family'] != 1])


def comparison(before, after):
    old, new = cohorts(before), cohorts(after)
    result = {k: paired_change(old[k], new[k]) for k in old}
    baseline_moves = {r['position_id'] for r in old['movement_required'] if r['selected_movement']}
    both_moves = baseline_moves & {r['position_id'] for r in new['movement_required'] if r['selected_movement']}
    result['fixed_original_movers'] = dict(
        before=metrics([r for r in before if r['position_id'] in baseline_moves]),
        after=metrics([r for r in after if r['position_id'] in baseline_moves]))
    result['both_choose_movement'] = dict(
        before=metrics([r for r in before if r['position_id'] in both_moves]),
        after=metrics([r for r in after if r['position_id'] in both_moves]))
    return result


def summarize(root):
    root = Path(root)
    p = json.loads((root/'protocol.json').read_text())
    screen = set(p['screen_ids'])
    summaries, predictions = {}, {}
    for key, spec in p['models'].items():
        report = json.loads((root/key/'results/report.json').read_text())
        if not report['complete']:
            raise ValueError('incomplete model: '+key)
        predictions[key] = {v: [r for r in report['results'] if r['version']==v] for v in p['versions']}
        n = p['evaluation_count'] if spec['cohort']=='full' else len(screen)
        for v, rows in predictions[key].items():
            if len(rows)!=n or len({r['position_id'] for r in rows})!=n:
                raise ValueError('duplicate or missing rows')
            if spec['cohort']=='screen' and {r['position_id'] for r in rows} != screen:
                raise ValueError('screen differs')
        summaries[key] = dict(cohort=spec['cohort'], arms={},
            execution=json.loads((root/key/'execution.json').read_text()),
            runtime=json.loads((root/key/'runtime.json').read_text()))
        for v, rows in predictions[key].items():
            group = [r for r in rows if r['position_id'] in screen]
            batches = [b for b in report['batches'] if b['version']==v]
            summaries[key]['arms'][v] = dict(
                full={name: metrics(g) for name,g in cohorts(rows).items()},
                screen={name: metrics(g) for name,g in cohorts(group).items()},
                nonmovement_by_family={str(k):metrics([r for r in rows if r['family']==k]) for k in range(2,11)},
                measured_batch_seconds=sum(b['seconds'] for b in batches),
                amortized_seconds_per_decision=sum(b['seconds'] for b in batches)/len(rows),
                median_batch_seconds=statistics.median(b['seconds'] for b in batches))
        v0,v1=p['versions']
        summaries[key]['harness_effect'] = comparison(predictions[key][v0],predictions[key][v1])
    cross = {}
    for key in ['qwen35-4b','qwen35-9b']:
        cross[key] = {}
        for v in p['versions']:
            before = [r for r in predictions['20k'][v] if r['position_id'] in screen]
            cross[key][v] = comparison(before,predictions[key][v])
    result = dict(schema='harness-comparison-result-v1', models=summaries, screen_vs_20k=cross,
        protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
        limitations=p['limitations']+['Timing is batched throughput, not interactive single-position latency.',
            'Map-cluster bootstrap intervals are exploratory and not multiplicity-adjusted.'])
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for key,s in summaries.items():
        print(key)
        for version,a in s['arms'].items():
            print(version,{cohort:{k:(x['correct'],x['count']) for k,x in a[cohort].items()} for cohort in ['full','screen']},
                  'seconds/decision',round(a['amortized_seconds_per_decision'],3))
    return result


if __name__=='__main__':
    summarize(sys.argv[1])
