"""Frozen-bot seat-paired round robin, with a fixed-reference Elo fit."""
from collections import Counter
import json
import math
from pathlib import Path
import random
import secrets
import time

from .common import (Budget, FINAL_END, FINAL_START, REFERENCE, REFERENCE_SHA, SCHEMA, canonical,
                     identifier, paired_schedule, sha, validate_seed, write_json)
from .game import ReferenceBot, WasiBot, play_game


def fit_elo(rows, labels, anchor=REFERENCE, *, iterations=2000):
    """Bradley–Terry MM, 0.5 virtual draw per played pair; unresolved games excluded.

    The weak symmetric prior regularizes separation; disconnected real graphs are
    explicitly reported. Ratings have no interpretation across different rosters.
    """
    n = len(labels)
    at = {name: i for i, name in enumerate(labels)}
    if anchor not in at:
        raise ValueError('anchor missing')
    wins, games = [0.] * n, [[0.] * n for _ in labels]
    edges = set()
    for row in rows:
        if row['winner'] is None and row['status'] != 'draw':
            continue
        i, j = at[row['a']], at[row['b']]
        score = .5 if row['status']=='draw' else float(row['winner'] == row['seat'])
        wins[i] += score
        wins[j] += 1 - score
        games[i][j] += 1
        games[j][i] += 1
        edges.add(tuple(sorted((i, j))))
    for i, j in edges:
        games[i][j] += .5
        games[j][i] += .5
        wins[i] += .25
        wins[j] += .25
    strength = [1.] * n
    for _ in range(iterations):
        new = []
        for i in range(n):
            den = sum(games[i][j] / (strength[i] + strength[j]) for j in range(n) if games[i][j])
            new.append(max(1e-12, wins[i] / den) if den else 1.)
        norm = new[at[anchor]]
        new = [x / norm for x in new]
        change = max(abs(math.log(new[i] / strength[i])) for i in range(n))
        strength = new
        if change < 1e-9:
            break
    connected = {at[anchor]}
    while True:
        larger = connected | {j for i in connected for j in range(n) if games[i][j]}
        if larger == connected:
            break
        connected = larger
    return {name: (400 * math.log10(strength[i]) if i in connected else None) for i, name in enumerate(labels)}


def summarize(rows, labels, anchor=REFERENCE, *, bootstrap=400):
    ratings = fit_elo(rows, labels, anchor)
    seeds = sorted({r['seed'] for r in rows})
    rng = random.Random(0)
    samples = {k: [] for k in labels}
    # Resample whole map blocks: both seats and every pairing stay together.
    groups = {seed: [r for r in rows if r['seed'] == seed] for seed in seeds}
    families = {}
    for seed in seeds:
        families.setdefault(groups[seed][0].get('map_type','classic'), []).append(seed)
    if len(seeds) >= 2:
        for _ in range(bootstrap):
            # Keep map-family weights fixed; both seats and all pairs travel together.
            sampled = [s for block in families.values() for s in rng.choices(block,k=len(block))]
            resampled = [row for seed in sampled for row in groups[seed]]
            for key, value in fit_elo(resampled, labels, anchor, iterations=400).items():
                if value is not None:
                    samples[key].append(value)
    ladder = []
    for name in labels:
        played = [r for r in rows if name in (r['a'], r['b'])]
        won = lambda r: r['winner'] == (r['seat'] if name == r['a'] else 3 - r['seat'])
        values = sorted(samples[name])
        ladder.append(dict(entrant=name, elo=ratings[name], games=len(played),
            wins=sum(r['winner'] is not None and won(r) for r in played),
            losses=sum(r['winner'] is not None and not won(r) for r in played),
            draws=sum(r['status']=='draw' for r in played),
            unresolved=sum(r['winner'] is None and r['status']!='draw' for r in played),
            forfeits=sum(r['status'] == 'forfeit' and not won(r) for r in played),
            elo_ci95=[values[int(.025 * (len(values)-1))], values[int(.975 * (len(values)-1))]] if values else None))
    ladder.sort(key=lambda r: r['elo'] if r['elo'] is not None else -math.inf, reverse=True)
    pairs = {}
    for row in rows:
        key = row['a'] + '__' + row['b']
        counts = pairs.setdefault(key, dict(games=0, wins=0, losses=0, draws=0, unresolved=0, forfeits=0, by_seat={}))
        counts['games'] += 1
        counts[{'win': 'wins', 'loss': 'losses', 'draw':'draws', 'unresolved': 'unresolved'}[row['result']]] += 1
        counts['forfeits'] += row['status'] == 'forfeit'
        seat = counts['by_seat'].setdefault(str(row['seat']), dict(games=0, wins=0, losses=0, draws=0, unresolved=0))
        seat['games'] += 1
        seat[{'win': 'wins', 'loss': 'losses', 'draw':'draws', 'unresolved': 'unresolved'}[row['result']]] += 1
    return dict(games=len(rows), resolved=sum(r['winner'] is not None or r['status']=='draw' for r in rows),
                draws=sum(r['status']=='draw' for r in rows),
                unresolved=sum(r['winner'] is None and r['status']!='draw' for r in rows), ladder=ladder, pairs=pairs,
                map_coverage={kind:len(block) for kind,block in families.items()},
                evidence_warning='single layout in a family; intervals are not reliable' if any(len(b)<2 for b in families.values()) else None,
                method='Bradley-Terry MM; anchor=0; 0.5 virtual draw per played pair; unresolved excluded; '
                       '95% intervals from paired-map bootstrap resamples stratified by family')


def submission(run_id):
    root = Path('/results/runs') / identifier(run_id)
    meta = json.loads((root / 'submission.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    if meta['run_id'] != run_id or sha((root / 'bot.wasm').read_bytes()) != meta['wasm_sha256']:
        raise ValueError('frozen artifact mismatch')
    if sha((root / 'main.rs').read_bytes()) != meta['source_sha256']:
        raise ValueError('frozen source mismatch')
    if sha((root / 'protocol.json').read_bytes()) != meta['protocol_sha256']:
        raise ValueError('frozen protocol mismatch')
    if sha((root / 'transcript.jsonl').read_bytes()) != meta['transcript_sha256']:
        raise ValueError('frozen transcript mismatch')
    return root / 'bot.wasm', meta, protocol


def tournament(tournament_id, run_ids, maps, volume, pair_runner=None):
    if type(maps) is not int or not 1 <= maps <= 128:
        raise ValueError('maps must be 1..128')
    if not run_ids or len(set(run_ids)) != len(run_ids) or REFERENCE in run_ids:
        raise ValueError('need distinct frozen submission run IDs')
    root = Path('/results/tournaments') / identifier(tournament_id)
    entrants = {name: submission(name) for name in run_ids}
    # Authoring treatments/budgets may differ; only the game and runtime track must match.
    protocols = [p for _, _, p in entrants.values()]
    keys = ('ruleset', 'size', 'maps', 'players', 'source_files', 'reference_sha256')
    def signature(p):
        values = {k: p[k] for k in keys}
        values['runtime_limits'] = {k: p['budget'][k] for k in ('move_seconds', 'game_seconds', 'fuel')}
        return canonical(values)
    if any(signature(p) != signature(protocols[0]) for p in protocols[1:]):
        raise ValueError('entrants belong to different game/runtime tracks')
    from .agent import source_receipt
    if protocols[0]['source_files'] != source_receipt():
        raise ValueError('evaluation source differs from the frozen authoring protocol')
    budget = Budget(**protocols[0]['budget']).validate()
    root.mkdir(parents=True, exist_ok=False)
    labels = [REFERENCE, *run_ids]
    # Final allocations are private until all immutable entrants have been checked.
    seeds = sorted(secrets.SystemRandom().sample(range(FINAL_START, FINAL_END + 1), maps))
    jobs = paired_schedule(labels, seeds)
    plan = dict(schema=SCHEMA, tournament_id=tournament_id, entrants={k: v[1] for k, v in entrants.items()},
        reference_sha256=REFERENCE_SHA, seeds=seeds, jobs=jobs, protocol_sha256=sha(signature(protocols[0]).encode()))
    write_json(root / 'plan.json', plan)
    write_json(root / 'status.json', dict(status='running', completed=0, games=len(jobs)))
    volume.commit()
    started = time.monotonic()
    groups = [dict(tournament_id=tournament_id, jobs=[dict(index=i, **jobs[i]) for i in range(at,min(at+8,len(jobs)))],
                   budget=protocols[0]['budget']) for at in range(0,len(jobs),8)]
    rows = []
    outputs = pair_runner(groups) if pair_runner else (play_group(g,volume) for g in groups)
    for part in outputs:
        rows.extend(part)
        write_json(root/'status.json',dict(status='running',completed=len(rows),games=len(jobs)))
        volume.commit()
    rows.sort(key=lambda r:r['game_index'])
    report = dict(schema=SCHEMA, tournament_id=tournament_id, status='completed', maps=maps, players=labels,
                  seconds=time.monotonic() - started, plan_sha256=sha((root / 'plan.json').read_bytes()),
                  authoring_conditions={k: dict(model=v[1]['model'], treatment=v[1].get('treatment'),
                      budget=v[1]['budget'], revisions=v[1].get('revisions'),
                      budget_profile=v[1].get('budget_profile')) for k,v in entrants.items()},
                  **summarize(rows, labels))
    write_json(root / 'games.json', rows)
    write_json(root / 'report.json', report)
    volume.commit()
    return report


def play_group(payload, volume):
    """A bounded group of at most eight games, suitable for a separate Modal worker."""
    jobs = payload['jobs']
    if not 1 <= len(jobs) <= 8:
        raise ValueError('game group size must be 1..8')
    root = Path('/results/tournaments')/identifier(payload['tournament_id'])
    budget = Budget(**payload['budget']).validate()
    labels = {j[k] for j in jobs for k in ('a','b')}
    entrants = {label:submission(label)[0] for label in labels if label != REFERENCE}
    reference = ReferenceBot('/results/references/hero-1v1/final.npz',REFERENCE_SHA) if REFERENCE in labels else None
    def make(label):
        return reference if label == REFERENCE else WasiBot(entrants[label],budget.fuel,budget.move_seconds)
    rows = []
    for job in jobs:
        validate_seed(job['seed'],'final')
        row = play_game(make(job['a']),make(job['b']),job['seed'],job['seat'],budget,
                        root/'games'/f'{job["index"]:06d}.jsonl',phase='final',labels=[job['a'],job['b']])
        row['game_index'] = job['index']
        write_json(root/'games'/f'{job["index"]:06d}.summary.json',row)
        volume.commit()
        rows.append(row)
    return rows
