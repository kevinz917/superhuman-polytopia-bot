"""Small deterministic debriefs derived ONLY from the candidate's public trace.

Counts describe observations/actions, not hidden outcomes or strategy advice.
"""
from collections import Counter
import json
from pathlib import Path


def public_summary(path):
    actions = Counter()
    checkpoints = {}
    first = last = None
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        obs = record['observation']; v = obs['visible_state']; seat = obs['player']
        own_cities = [c for c in v['cities'] if c['owner']==seat]
        point = dict(turn=v['turn'], stars=v['stars'], own_cities=len(own_cities),
                     own_units=sum(u['owner']==seat for u in v['units']),
                     own_city_production=sum(c.get('production',0) for c in own_cities),
                     technologies=len(v['technology']), explored_tiles=sum(t['known'] for t in v['tiles']))
        first = first or point; last = point
        # One small record per round: the last observed position that round.
        checkpoints[v['turn']] = point
        command = next((a['command'] for a in obs['legal_actions'] if a['action_id']==record['action_id']), None)
        if command is None:
            raise ValueError('trace action missing from its public legal catalog')
        actions[str(command['kind'])] += 1
    return dict(schema='polytopia-public-debrief-v1', first=first, last=last,
                action_counts=dict(sorted(actions.items())), rounds=list(checkpoints.values()),
                definitions=dict(own_city_production='sum of public own-city production; not net income',
                    capture_actions='kind 8 attempts, not a claim about hidden city outcomes',
                    combat_actions='kind 2 attacks, not inferred kills',
                    last='last candidate decision, not omniscient terminal state'))
