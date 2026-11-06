"""Summarize the Instruct-2507 baseline and compare per map with the Qwen3.5 arms; runs on Modal.

Reads the Qwen3.5 run read-only. Operational failures (unfinished, context,
infrastructure, generation aborts) are never counted as native losses or draws.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

RESOLVED = ('native', 'forfeit')


def trace_stats(folder, game_id):
    actions, errors, tokens, calls = Counter(), Counter(), 0, 0
    with gzip.open(folder/(game_id+'.jsonl.gz'), 'rt') as source:
        for line in source:
            r = json.loads(line)
            if r.get('policy') != 'model':
                continue
            if 'selected_command' in r:
                actions[r['selected_command']['kind']] += 1
            for a in r.get('diagnostic', r.get('attempts', [])):
                calls += 1; tokens += len(a['generation']['response_token_ids'])
                if 'error' in a:
                    errors[a['error']] += 1
    return dict(action_families=dict(actions), errors=dict(errors), calls=calls, output_tokens=tokens)


def arm(games, timing, trace_dirs, expected, *, seed, elapsed, worker_seconds):
    assert len(games) == len(expected) and {g['id'] for g in games} == expected.keys()
    for g in games:
        assert all(g[k] == expected[g['id']][k] for k in ['seed', 'board_hash', 'seat', 'opponent', 'rng_index'])
    traces = {g['id']: trace_stats(trace_dirs[g['id']], g['id']) for g in games}
    assert sum(t['output_tokens'] for t in traces.values()) == sum(x['output_tokens'] for x in timing)
    assert len({(x['id'], x['decision']) for x in timing}) == len(timing)
    for x in timing:
        assert x['max_tokens'] is None and x['thinking_token_budget'] is None
        assert x['seed'] == seed+expected[x['id']]['rng_index']*100000+x['decision']
    families = Counter()
    for t in traces.values():
        families.update(t['action_families'])
    total = sum(families.values())
    return dict(allocated=len(games), status_counts=dict(Counter(g['status'] for g in games)),
        native_wins=sum(g['status'] == 'native' and g['reward'] > 0 for g in games),
        native_losses=sum(g['status'] == 'native' and g['reward'] < 0 for g in games),
        native_draws=sum(g['status'] == 'native' and g['reward'] == 0 for g in games),
        forfeits=sum(g['status'] == 'forfeit' for g in games),
        operational_failures=sum(g['status'] not in RESOLVED for g in games),
        valid_actions=sum(g['first_attempt_valid'] for g in games),
        model_decisions=sum(g['model_decisions'] for g in games), calls=len(timing),
        output_tokens=sum(x['output_tokens'] for x in timing),
        mean_output_tokens=statistics.mean(x['output_tokens'] for x in timing) if timing else None,
        max_output_tokens=max((x['output_tokens'] for x in timing), default=0),
        median_request_seconds=statistics.median(x['request_seconds'] for x in timing) if timing else None,
        mean_request_seconds=statistics.mean(x['request_seconds'] for x in timing) if timing else None,
        finish_reasons=dict(Counter(x['finish_reason'] for x in timing)),
        end_reasons=dict(Counter(g['reason'] for g in games)),
        median_final_round=statistics.median(g['round'] for g in games if g['round'] is not None),
        action_family_counts=dict(families.most_common()),
        action_family_share={k: v/total for k, v in families.most_common()} if total else {},
        elapsed_seconds=elapsed, gpu_worker_seconds=worker_seconds,
        games=sorted(games, key=lambda g: g['id']), traces=traces)


def win(g):
    return int(g['status'] == 'native' and g['reward'] > 0)


def paired(a, b, jobs, value):
    """Map-clustered mean difference a-b; both seats averaged within a map."""
    by = {n: {g['id']: g for g in s['games']} for n, s in (('a', a), ('b', b))}
    diffs = []
    for seed in sorted({j['seed'] for j in jobs}):
        ids = [j['id'] for j in jobs if j['seed'] == seed]
        diffs.append(statistics.mean(value(by['a'][i])-value(by['b'][i]) for i in ids))
    rng = random.Random(1701)
    boot = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(20000))
    return dict(mean=statistics.mean(diffs), map_cluster_bootstrap_95_percent=[boot[499], boot[19499]],
                maps=len(diffs), maps_a_higher=sum(d > 0 for d in diffs),
                maps_b_higher=sum(d < 0 for d in diffs), maps_tied=sum(d == 0 for d in diffs))


def summarize(root, qwen35):
    root, qwen35 = Path(root), Path(qwen35)
    p = json.loads((root/'protocol.json').read_text())
    source = json.loads((qwen35/'protocol.json').read_text())
    expected = {j['id']: j for j in p['jobs']}
    identity = ['id', 'seed', 'board_hash', 'seat', 'opponent', 'rng_index']
    assert [{k: j[k] for k in identity} for j in p['jobs']] == [{k: j[k] for k in identity} for j in source['jobs']]
    assert p['seed'] == source['seed'] and p['sampling'] == source['sampling']
    result = dict(schema='qwen3-instruct-baseline-summary-v1', arms={},
        protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
        qwen35_protocol_sha256=hashlib.sha256((qwen35/'protocol.json').read_bytes()).hexdigest())

    folder = root/'instruct'
    report = json.loads((folder/'report.json').read_text())
    runtime = json.loads((folder/'runtime.json').read_text())
    execution = json.loads((root/'instruct-execution.json').read_text())
    assert execution['exit_code'] == 0
    assert runtime['revision'] == p['revision'] and runtime['adapter'] is None and runtime['thinking'] is False
    assert runtime['max_tokens'] is None and runtime['max_model_len'] == p['max_length']
    timing = [json.loads(l) for l in (folder/'generation-timing.jsonl').read_text().splitlines()]
    s = arm(report['games'], timing, {i: folder/'games' for i in expected}, expected, seed=p['seed'],
            elapsed=report['elapsed_seconds'], worker_seconds=execution['worker_seconds'])
    s['runtime'] = runtime
    result['arms']['instruct'] = s

    # Qwen3.5 thinking-off: finished shards, read in place.
    shards = sorted(int(d.name) for d in (qwen35/'shards').iterdir() if d.name.isdigit())
    games, timing, dirs, elapsed, worker = [], [], {}, [], 0.
    for shard in shards:
        base = qwen35/'shards'/str(shard)
        ex = json.loads((base/'off-execution.json').read_text())
        assert ex['exit_code'] == 0
        worker += ex['worker_seconds']
        r = json.loads((base/'off'/'report.json').read_text())
        games.extend(r['games']); elapsed.append(r['elapsed_seconds'])
        for g in r['games']:
            dirs[g['id']] = base/'off'/'games'
        timing.extend(json.loads(l) for l in (base/'off'/'generation-timing.jsonl').read_text().splitlines())
    merged = json.loads((qwen35/'off'/'games'/'games.json').read_text())
    assert sorted(merged, key=lambda g: g['id']) == sorted(games, key=lambda g: g['id'])
    s = arm(games, timing, dirs, expected, seed=source['seed'], elapsed=max(elapsed), worker_seconds=worker)
    s['runtime'] = json.loads((qwen35/'shards'/str(shards[0])/'off'/'runtime.json').read_text())
    s['replicas'] = len(shards)
    result['arms']['qwen35_off'] = s

    # Qwen3.5 thinking-on: only if that run has published its final summary.
    on_summary = qwen35/'summary.json'
    if on_summary.exists():
        on = json.loads(on_summary.read_text())['modes']['on']
        on = {k: v for k, v in on.items() if k != 'runtime'}
        assert {g['id'] for g in on['games']} == expected.keys()
        result['arms']['qwen35_on'] = on
    result['qwen35_on_available'] = on_summary.exists()

    jobs = p['jobs']
    comparisons = {}
    for other in [a for a in ('qwen35_off', 'qwen35_on') if a in result['arms']]:
        a, b = result['arms']['instruct'], result['arms'][other]
        resolved = all(g['status'] in RESOLVED for s_ in (a, b) for g in s_['games'])
        survived = all(g['round'] is not None for s_ in (a, b) for g in s_['games'])
        comparisons['instruct_minus_'+other] = dict(
            native_win_difference=paired(a, b, jobs, win) if resolved else None,
            final_round_difference=paired(a, b, jobs, lambda g: g['round']) if resolved and survived else None,
            note='Map-clustered (15 maps, both seats averaged per map); exploratory, one sampling sequence. '
                 'Final round is descriptive survival length, not a win metric.')
    result['comparisons'] = comparisons

    per_map = []
    for seed in sorted({j['seed'] for j in jobs}):
        row = dict(seed=seed)
        for name, s_ in result['arms'].items():
            by = {g['id']: g for g in s_['games']}
            row[name] = {f"seat{j['seat']}": dict(status=by[j['id']]['status'], reward=by[j['id']]['reward'],
                         reason=by[j['id']]['reason'], round=by[j['id']]['round'],
                         model_decisions=by[j['id']]['model_decisions'])
                         for j in jobs if j['seed'] == seed}
        per_map.append(row)
    result['per_map'] = per_map
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    small = dict(result, arms={n: {k: v for k, v in s_.items() if k not in ('games', 'traces', 'runtime')}
                               for n, s_ in result['arms'].items()})
    (root/'summary-small.json').write_text(json.dumps(small, indent=2)+'\n')
    for n, s_ in small['arms'].items():
        print(n, json.dumps({k: v for k, v in s_.items() if k not in ('action_family_share',)}))
    print(json.dumps(comparisons, indent=2))
    return small


if __name__ == '__main__':
    summarize(sys.argv[1], sys.argv[2])
