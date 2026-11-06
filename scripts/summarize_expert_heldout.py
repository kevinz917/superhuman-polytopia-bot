"""Held-out expert-SFT summary on Modal: per model and opponent, map-clustered paired bootstrap."""
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_expert_sft import trace_metrics, win  # noqa: E402


def collect(root, p):
    out = {}
    for arm, spec in p['arms'].items():
        model = spec['model']
        games = {g['id']: g for g in json.loads((root/arm/'games'/'games.json').read_text())}
        for j in p['jobs']:
            if j['opponent'] != spec['opponent']:
                continue
            g = games.get(j['id'])
            m = trace_metrics(root/arm/'games'/(j['id']+'.jsonl.gz'), j['seat'])
            out.setdefault(model, {})[j['id']] = dict(job=j, status=g['status'] if g else 'missing',
                reason=g['reason'] if g else None, reward=g['reward'] if g else None,
                round=g['round'] if g else None, model_moves=g['model_decisions'] if g else None, **m)
    rl = {}
    for j in p['jobs']:
        path = root/'rl-reference'/(j['id']+'.json')
        r = json.loads(path.read_text()) if path.exists() else None
        rl[j['id']] = dict(job=j, status=r['status'] if r else 'missing', reason=r and r['reason'], reward=r and r['reward'],
                           round=r and r['round'], peak_cities=r and r.get('final_cities'))
    out['rl_expert'] = rl
    return out


def aggregate(games):
    gs = list(games.values())
    agg = dict(games=len(gs), wins=sum(win(g) for g in gs),
               losses=sum(g['status'] == 'native' and (g['reward'] or 0) < 0 for g in gs),
               forfeits=sum(g['status'] == 'forfeit' for g in gs), truncated=sum(g['status'] == 'truncated' for g in gs),
               operational=sum(g['status'] not in ('native', 'forfeit', 'truncated') for g in gs),
               statuses=dict(Counter(f"{g['status']}/{g['reason']}" for g in gs)),
               median_round=statistics.median(g['round'] for g in gs if g['round'] is not None) if gs else None)
    if gs and 'actions' in gs[0]:
        acts, combat, ab = Counter(), Counter(), Counter()
        for g in gs:
            acts.update(g['actions']); combat.update(g['combat']); ab.update(g['abilities'])
        agg.update(mean_peak_cities=statistics.mean(g['peak_cities'] for g in gs), actions=dict(acts.most_common()),
                   combat=dict(combat), abilities=dict(ab))
    return agg


def paired(a, b, jobs):
    maps = sorted({j['seed'] for j in jobs})
    diffs = []
    for m in maps:
        ids = [j['id'] for j in jobs if j['seed'] == m]
        diffs.append(statistics.mean(win(a[i])-win(b[i]) for i in ids))
    rng = random.Random(1701)
    boot = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(20000))
    return dict(a_minus_b=statistics.mean(diffs), map_cluster_bootstrap_95=[boot[499], boot[19499]], maps=len(maps),
                maps_a_better=sum(d > 0 for d in diffs), maps_b_better=sum(d < 0 for d in diffs))


def summarize(root):
    root = Path(root)
    p = json.loads((root/'protocol.json').read_text())
    by = collect(root, p)
    opponents = sorted({j['opponent'] for j in p['jobs']})
    result = dict(schema='expert-sft-heldout-summary-v1', models={}, paired={})
    for model, games in by.items():
        result['models'][model] = dict(all=aggregate(games), **{o: aggregate({k: g for k, g in games.items() if g['job']['opponent'] == o})
                                                             for o in opponents})
    for a, b in [('expert_sft', 'base'), ('rl_expert', 'expert_sft'), ('rl_expert', 'base')]:
        result['paired'][f'{a}_vs_{b}'] = dict(all=paired(by[a], by[b], p['jobs']),
            **{o: paired(by[a], by[b], [j for j in p['jobs'] if j['opponent'] == o]) for o in opponents})
    result['note'] = 'Clusters are maps (all seats and opponents of a map together). Unresolved games are never wins or losses.'
    (root/'summary.json').write_text(json.dumps(dict(result, games=by), indent=2, default=str)+'\n')
    (root/'summary-small.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
