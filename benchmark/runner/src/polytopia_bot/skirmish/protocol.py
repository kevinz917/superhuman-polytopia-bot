"""Versioned evaluation decisions; legacy results are never silently rescored."""
import secrets

from .common import FINAL_START, FINAL_END, canonical, sha

SCHEMA = 'polytopia-mini-v5.1'
MAP_TYPES = ('drylands', 'lakes', 'continents')
EVALUATION = dict(maps_per_family=5, paired_seats=True, cap_policy='engine_draw',
                  game_seconds=180, max_rounds=100, max_actions=10000,
                  draws='engine_truncation or action_limit only; wall time stays unresolved',
                  source_policy='current frozen source; no rollback')


def make_map_suite(maps_per_family=5, *, sample=None):
    if type(maps_per_family) is not int or not 1 <= maps_per_family <= 128:
        raise ValueError('maps_per_family must be 1..128')
    sample = sample or secrets.SystemRandom().sample
    seeds = sample(range(FINAL_START, FINAL_END + 1), len(MAP_TYPES)*maps_per_family)
    return [dict(map_type=kind, seed=seed) for kind, block in zip(MAP_TYPES,
        (seeds[i*maps_per_family:(i+1)*maps_per_family] for i in range(len(MAP_TYPES)))) for seed in block]


def validate_map_suite(maps, *, hidden=True):
    if not maps or {m['map_type'] for m in maps} != set(MAP_TYPES):
        raise ValueError('every map family is required')
    seeds = [m['seed'] for m in maps]
    if any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('map seeds must be distinct integers')
    counts = [sum(m['map_type']==kind for m in maps) for kind in MAP_TYPES]
    if len(set(counts)) != 1:
        raise ValueError('map families must have equal weight')
    if hidden and any(not FINAL_START <= seed <= FINAL_END for seed in seeds):
        raise ValueError('hidden maps must use the final partition')
    return [dict(map_type=m['map_type'], seed=m['seed'], seat=seat) for m in maps for seat in (1,2)]


def suite_digest(maps):
    validate_map_suite(maps)
    return sha(canonical(maps).encode())


def apply_cap_policy(row, policy):
    if policy not in ('unresolved', 'engine_draw'):
        raise ValueError('unknown cap policy')
    row = dict(row)
    if policy == 'engine_draw' and row['status']=='unresolved' and row.get('reason') in ('engine_truncation','action_limit'):
        row.update(status='draw', result='draw', winner=None)
    return row


def submission_quality(source_sha, starter_sha, returncode):
    if returncode != 0:
        return 'compile_failure'
    return 'unchanged_starter' if source_sha==starter_sha else 'custom_submission'
