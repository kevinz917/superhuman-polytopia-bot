"""Versioned single-call public-state decision aids; no teacher or engine access.

The original renderer and catalog ABI remain unchanged. Facts here are geometry
and public rule arithmetic, never an action ranking or a simulated future.
"""
from collections import defaultdict
from dataclasses import replace
import math

from .encoding import compact
from .interface import canonical, render, rule_table

VERSION = 'public-action-facts-v1'
FAMILIES = {1: 'move', 2: 'attack', 3: 'ability', 4: 'recruit', 5: 'harvest',
            6: 'build', 7: 'research', 8: 'capture', 9: 'city_reward',
            10: 'end_turn', 11: 'resign'}
INSTRUCTIONS = '''Additional encoding: public-action-facts-v1. The complete legal
catalog and action IDs are unchanged. action_groups give each family's inclusive
first/last action IDs. Coordinates are [x,y], zero-based, x right and y down.
location_facts columns describe relevant tiles. settlement_distance is geometric
Chebyshev distance to the nearest currently visible non-owned settlement, null
if none; it is NOT a path length. unknown_neighbors counts unknown adjacent tiles,
NOT guaranteed new reveals. visible_enemy_in_range counts visible enemies whose
static attack range covers the tile, NOT next-turn danger or a combat prediction.
movement_facts show destination minus source changes: negative distance_change
approaches a settlement; positive unknown_neighbors_change approaches a frontier.
Unknown terrain, hidden enemies, future captures and safe paths are not inferred.
spending columns give exact immediate star cost and balance for supported legal
actions only; omission means not computed, not free. Structure population/income
values in the rules are base values, not guaranteed net effects or road value.
Compare useful actions across families, considering expansion, combat and economy;
these facts do not rank actions or require movement. Return only the action JSON.'''


def action_facts(public):
    """Accept the validated PUBLIC schema only; never accept simulator State."""
    v, player = public['visible'], public['player']
    size = v['board_size']
    tiles = {t['tile']: t for t in v['tiles']}
    rules = rule_table()
    units = {r['id']: r for r in rules['units']}
    structures = {r['id']: r for r in rules['structures']}
    technologies = {r['id']: r for r in rules['technologies']}
    objectives = {c['tile'] for c in v['cities'] if c['owner'] != player}
    objectives |= {t['tile'] for t in tiles.values() if t['known']
                   and t.get('structure') in (1, 2) and t.get('owner') != player}
    enemy = [u for u in v['units'] if u['owner'] != player]
    own_cities = sum(c['owner'] == player for c in v['cities'])

    def distance(a, b):
        return max(abs(a % size - b % size), abs(a // size - b // size))

    relevant = {e['tile'] for name in ('units', 'cities') for e in v[name]}
    for a in public['legal_actions']:
        c = a['command']
        if c['kind'] in (1, 2):
            relevant.update((c['source'], c['target']))
    locations = {}
    for tile in sorted(relevant):
        locations[tile] = [tile, tile % size, tile // size,
            min((distance(tile, t) for t in objectives), default=None),
            sum(not t['known'] and distance(tile, t['tile']) == 1 for t in tiles.values()),
            sum(distance(tile, u['tile']) <= units[u['kind']]['range'] for u in enemy)]
    groups, movement, spending = defaultdict(list), [], []
    for action in public['legal_actions']:
        aid, c = action['action_id'], action['command']
        kind, src, dst, param = (c[k] for k in ('kind', 'source', 'target', 'parameter'))
        groups[kind].append(aid)
        if kind == 1:
            before, after = locations[src], locations[dst]
            movement.append([aid, None if before[3] is None else after[3] - before[3],
                             after[4] - before[4], dst in objectives])
        cost = None
        if kind in (1, 2, 8, 9, 10, 11):
            cost = 0
        elif kind == 4:
            cost = units[param]['cost']
        elif kind == 5:
            cost = 2
        elif kind == 6:
            cost = structures[param]['cost']
        elif kind == 7:
            cost = 4 + own_cities * technologies[param]['tier']
            if 22 in v['technology']:
                cost = math.ceil(cost * .66)
        # Do not invent costs for abilities (refunds/diplomacy have special rules).
        if cost is not None and cost >= 0 and kind not in (1, 2, 8, 9, 10, 11):
            spending.append([aid, cost, v['stars'] - cost])
    return dict(version=VERSION,
        action_groups=[dict(family=FAMILIES[k], kind=k, first=ids[0], last=ids[-1], count=len(ids))
                       for k, ids in sorted(groups.items())],
        visible_settlements=sorted(objectives),
        location_facts=dict(columns=['tile','x','y','settlement_distance','unknown_neighbors','visible_enemy_in_range'],
                            rows=list(locations.values())),
        movement_facts=dict(columns=['action_id','distance_change','unknown_neighbors_change','destination_is_settlement'],
                            rows=movement),
        spending=dict(columns=['action_id','star_cost','stars_after'], rows=spending))


def render_decision(obs, version=VERSION):
    prompt = render(obs)
    if version == 'original':
        return prompt
    if version != VERSION:
        raise ValueError('unknown decision harness version')
    encoded = compact(prompt.public)
    encoded['decision_facts'] = action_facts(prompt.public)
    return replace(prompt, messages=[
        dict(role='system', content=prompt.messages[0]['content'] + '\n' + INSTRUCTIONS),
        dict(role='user', content=canonical(encoded))])


def choose_action_once(obs, generate, *, version=VERSION):
    """One call; invalid output fails closed rather than silently choosing a move."""
    from .interface import parse_response
    prompt = render_decision(obs, version)
    generation = generate(prompt.messages)
    return parse_response(generation.text, prompt, current_version=obs.state_version), generation
