"""Summarize an SFT-pipeline evaluation (run on Modal): per model and opponent, configured
map-clustered paired comparisons, and per-arm inference throughput.

Models are arm names (arms sharded by opponent, ``name@opponent``, are merged back) and
``rl:<player>`` for RL references. Unresolved games are never wins or losses.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_expert_heldout import aggregate, paired  # noqa: E402
from summarize_expert_sft import trace_metrics  # noqa: E402


def arm_throughput(arm_dir):
    """Prompt tokens/s and decisions/min from generation-timing.jsonl and the execution receipt."""
    rows = []
    path = arm_dir/'generation-timing.jsonl'
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    runtime = json.loads((arm_dir/'runtime.json').read_text()) if (arm_dir/'runtime.json').exists() else {}
    execution = arm_dir.parent/f'{arm_dir.name}-execution.json'
    worker = json.loads(execution.read_text()).get('worker_seconds') if execution.exists() else None
    play = (worker - runtime.get('load_seconds', 0)) if worker else None
    prompt = sum(r['input_tokens'] for r in rows)
    return dict(decisions=len(rows), prompt_tokens=prompt, output_tokens=sum(r['output_tokens'] for r in rows),
                worker_seconds=worker, load_seconds=runtime.get('load_seconds'),
                prompt_tokens_per_second=prompt/play if play else None,
                decisions_per_minute=60*len(rows)/play if play else None,
                mean_request_seconds=sum(r['request_seconds'] for r in rows)/len(rows) if rows else None)


def collect(root, p):
    out, throughput = {}, {}
    for arm, spec in p['arms'].items():
        model = spec['model']
        games_file = root/arm/'games'/'games.json'
        games = {g['id']: g for g in json.loads(games_file.read_text())} if games_file.exists() else {}
        throughput[arm] = arm_throughput(root/arm)
        for j in p['jobs']:
            if spec.get('opponent') not in (None, j['opponent']):
                continue
            g = games.get(j['id'])
            trace = root/arm/'games'/(j['id']+'.jsonl.gz')
            m = trace_metrics(trace, j['seat']) if trace.exists() else dict(actions={}, combat={}, abilities={}, peak_cities=None)
            out.setdefault(model, {})[j['id']] = dict(job=j, status=g['status'] if g else 'missing',
                reason=g['reason'] if g else None, reward=g['reward'] if g else None,
                round=g['round'] if g else None, model_moves=g['model_decisions'] if g else None, **m)
    for name in p.get('rl_reference', []):
        rl = {}
        for j in p['jobs']:
            path = root/f'reference-{name}'/(j['id']+'.json')
            r = json.loads(path.read_text()) if path.exists() else None
            rl[j['id']] = dict(job=j, status=r['status'] if r else 'missing', reason=r and r['reason'],
                               reward=r and r['reward'], round=r and r['round'], peak_cities=r and r.get('final_cities'))
        out[f'rl:{name}'] = rl
    return out, throughput


def safe_aggregate(games):
    """aggregate() needs complete traces; with missing games fall back to outcome counts."""
    try:
        return aggregate(games)
    except Exception as exc:  # noqa: BLE001
        gs = list(games.values())
        return dict(games=len(gs), wins=sum(g['status'] == 'native' and (g['reward'] or 0) > 0 for g in gs),
                    losses=sum(g['status'] == 'native' and (g['reward'] or 0) < 0 for g in gs),
                    missing=sum(g['status'] == 'missing' for g in gs), partial=repr(exc)[:200])


def summarize(root):
    root = Path(root)
    p = json.loads((root/'protocol.json').read_text())
    by, throughput = collect(root, p)
    opponents = sorted({j['opponent'] for j in p['jobs']})
    result = dict(schema='sft-pipeline-eval-summary-v1', models={}, paired={}, throughput=throughput)
    for model, games in by.items():
        result['models'][model] = dict(all=safe_aggregate(games), **{o: safe_aggregate({k: g for k, g in games.items()
                                                                          if g['job']['opponent'] == o}) for o in opponents})
    for a, b in p.get('pairs', []):
        if a in by and b in by:
            result['paired'][f'{a}_vs_{b}'] = dict(all=paired(by[a], by[b], p['jobs']),
                **{o: paired(by[a], by[b], [j for j in p['jobs'] if j['opponent'] == o]) for o in opponents})
    result['note'] = 'Clusters are maps (all seats and opponents of a map together). Unresolved games are never wins or losses.'
    (root/'summary.json').write_text(json.dumps(dict(result, games=by), indent=2, default=str)+'\n')
    (root/'summary-small.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    return result
