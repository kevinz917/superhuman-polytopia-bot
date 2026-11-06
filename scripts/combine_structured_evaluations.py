"""Combine disjoint deterministic development evaluations with verified traces.

Reject protocol drift, duplicate games and missing seats before publishing.
Stochastic policies and random opponents reset their sampling streams between
partial runs, so this helper only combines deterministic teacher evaluations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def combine(sources, output, *, seed_base=300000, maps=16):
    if maps < 1 or not 300000 <= seed_base < seed_base + maps <= 400000:
        raise ValueError('development maps only')
    expected = {(s, p) for s in range(seed_base, seed_base + maps) for p in (1, 2)}
    invariant = ('checkpoint_sha256', 'config', 'opponent', 'teacher_hash',
                 'deterministic', 'paired_seats', 'sampling_seed', 'source_sha256')
    first, games, seen = None, [], set()
    for source in map(Path, sources):
        protocol = json.loads((source / 'protocol.json').read_text())
        summary = json.loads((source / 'summary.json').read_text())
        rows = json.loads((source / 'games.json').read_text())
        if (not protocol['deterministic'] or protocol['opponent'] != 'balanced'
                or not protocol['paired_seats'] or not summary['complete']):
            raise ValueError('requires complete deterministic paired evaluations')
        settings = {k: protocol[k] for k in invariant}
        if first is not None and settings != first:
            raise ValueError('evaluation protocol mismatch')
        first = settings
        part_expected = {(s, p) for s in protocol['map_seeds'] for p in (1, 2)}
        if len(rows) != protocol['requested_games'] or len(rows) != summary['games']:
            raise ValueError('source game count mismatch')
        part_seen = set()
        for row in rows:
            key = row['seed'], row['seat']
            if key not in expected or key in seen or key in part_seen:
                raise ValueError('duplicate or out-of-partition game')
            if row['evaluation_cutoff']:
                raise ValueError('incomplete game')
            if bool(row['terminated']) == bool(row['truncated']):
                raise ValueError('inconsistent ending')
            if row['reward'] not in (-1, 0, 1) or (row['truncated'] and row['reward'] != 0):
                raise ValueError('invalid native outcome')
            trace = (source / row['trace']).resolve()
            if not trace.is_relative_to(source.resolve()):
                raise ValueError('trace outside source evaluation')
            if hashlib.sha256(trace.read_bytes()).hexdigest() != row['trace_sha256']:
                raise ValueError('trace hash mismatch')
            games.append(dict(row, trace=str(trace)))
            part_seen.add(key)
        if part_seen != part_expected:
            raise ValueError('source seat/map coverage mismatch')
        seen.update(part_seen)
    if seen != expected:
        raise ValueError('combined map/seat coverage incomplete')
    games.sort(key=lambda x: (x['seed'], x['seat']))
    n = len(games)
    wins = sum(x['terminated'] and x['reward'] > 0 for x in games)
    fraction, z = wins / n, 1.959963984540054
    center = (fraction + z*z/(2*n)) / (1+z*z/n)
    radius = z * math.sqrt(fraction*(1-fraction)/n + z*z/(4*n*n)) / (1+z*z/n)
    report = dict(games=n, requested_games=n, wins=wins,
                  losses=sum(x['reward'] < 0 for x in games),
                  truncations=sum(x['truncated'] for x in games), cutoffs=0,
                  complete=True, native_win_fraction=fraction,
                  native_win_fraction_wilson95=[center-radius, center+radius],
                  mean_max_cities=sum(x['max_cities'] for x in games)/n,
                  checkpoint_sha256=first['checkpoint_sha256'], trace_hashes_verified=n,
                  source_evaluations=[str(Path(s).resolve()) for s in sources],
                  interpretation='Development selection only; unresolved games are not wins or draws.')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    merged_protocol = dict(first, map_seeds=list(range(seed_base, seed_base+maps)), requested_games=n)
    for name, data in [('protocol', merged_protocol), ('summary', report), ('games', games)]:
        (root / (name+'.json')).write_text(json.dumps(data, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed-base', type=int, default=300000)
    parser.add_argument('--maps', type=int, default=16)
    print(json.dumps(combine(**vars(parser.parse_args())), indent=2))
