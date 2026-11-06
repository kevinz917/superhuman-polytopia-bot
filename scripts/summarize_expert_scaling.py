"""Summarize the expert-SFT scaling curve on Modal: checkpoint arms vs expert-sft-v1's final adapter and base."""
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_expert_sft import aggregate, llm_arm, paired, win  # noqa: E402


def extra(arm):
    gs = list(arm.values())
    rounds = [g['round'] for g in gs if g['round'] is not None]
    moves = [g['model_moves'] for g in gs if g['model_moves'] is not None]
    peaks = [g['peak_cities'] for g in gs]
    won = [g for g in gs if win(g)]
    return dict(median_model_moves=statistics.median(moves) if moves else None,
                median_peak_cities=statistics.median(peaks), max_peak_cities=max(peaks),
                median_round_wins=statistics.median(g['round'] for g in won) if won else None,
                win_rate=sum(win(g) for g in gs)/len(gs),
                unfinished=sum(g['status'] != 'native' for g in gs))


def loss_near(logs, step, window=20):
    xs = [l['loss'] for l in logs if step-window < l['step'] <= step]
    return statistics.mean(xs)


def summarize(root):
    root = Path(root)
    p = json.loads((root/'protocol.json').read_text())
    ref = p['reference']
    source = root.parent/ref['run']
    jobs = p['jobs']
    run = json.loads((root.parent/p['arms'][next(iter(p['arms']))]['training_receipt']).read_text())
    lr = {l['step']: l['learning_rate'] for l in run['logs']}
    arms = {'base': llm_arm(source, ref['base_arm'], jobs)}
    points = {'base': dict(step=0, examples_seen=0)}
    for a, spec in p['arms'].items():
        arms[a] = llm_arm(root, a, jobs)
        points[a] = dict(step=spec['step'], examples_seen=spec['examples_seen'])
    arms['final'] = llm_arm(source, ref['final_arm'], jobs)
    points['final'] = dict(step=ref['final_step'], examples_seen=ref['final_examples_seen'])
    for a, pt in points.items():
        if pt['step']:
            pt.update(fraction=pt['examples_seen']/ref['final_examples_seen'], learning_rate=lr[pt['step']],
                      train_loss_last20=loss_near(run['logs'], pt['step']))
    curve = {a: dict(**points[a], **aggregate(arms[a]), **extra(arms[a])) for a in arms}
    order = list(arms)
    pairs = {}
    for a in order:
        if a not in ('base', 'final'):
            pairs[f'{a}_vs_final'] = paired(arms[a], arms['final'], jobs)
            pairs[f'{a}_vs_base'] = paired(arms[a], arms['base'], jobs)
    for a, b in zip(order[1:], order[2:]):
        pairs[f'{b}_vs_{a}'] = paired(arms[b], arms[a], jobs)
    result = dict(schema='expert-sft-scaling-summary-v1', curve=curve, paired=pairs, games=arms,
                  note='final and base arms are the expert-sft-v1 games; unfinished games never count as wins.')
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    small = {k: v for k, v in result.items() if k != 'games'}
    small['per_job_wins'] = {a: {j: win(g) for j, g in v.items()} for a, v in arms.items()}
    (root/'summary-small.json').write_text(json.dumps(small, indent=2)+'\n')
    return small
