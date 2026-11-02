"""Bounded factual memory updated exclusively from a seat's public observations."""
from copy import deepcopy
from dataclasses import replace
from .interface import canonical, command, integer, keys

SCHEMA = 'polytopia-memory-v1'
ACTION_CAP = 5
HISTORY_CAP = 32


def validate_memory(memory, obs):
    if memory.get('schema') != SCHEMA:
        raise ValueError('unsupported memory schema')
    keys(memory, ('schema', 'recent_actions', 'historical'), ('tiles',))
    if not isinstance(memory['recent_actions'], list) or len(memory['recent_actions']) > ACTION_CAP:
        raise ValueError('invalid recent-action memory')
    previous = -1
    for row in memory['recent_actions']:
        keys(row, ('command', 'turn', 'revision'))
        command(row['command'])
        integer(row['turn'], 0, obs.visible_state['turn'])
        integer(row['revision'], previous + 1, obs.state_version - 1)
        previous = row['revision']
    if not isinstance(memory['historical'], list) or len(memory['historical']) > HISTORY_CAP:
        raise ValueError('invalid historical memory')
    seen = set()
    for row in memory['historical']:
        keys(row, ('entity', 'fact', 'last_seen_turn', 'revision'))
        if row['entity'] not in ('tiles', 'units', 'cities'):
            raise ValueError('invalid historical entity')
        integer(row['last_seen_turn'], 0, obs.visible_state['turn'])
        integer(row['revision'], 0, obs.state_version)
        fact = row['fact']
        tile = integer(fact.get('tile'), 0, obs.visible_state['board_size'] ** 2 - 1)
        key = (row['entity'], tile)
        if key in seen:
            raise ValueError('duplicate historical fact')
        seen.add(key)
        # Reuse the allowlist on historical rows; never accept arbitrary payloads.
        from .interface import public_state
        visible = deepcopy(obs.visible_state)
        visible['tiles'] = [{'tile': i, 'known': i == tile} for i in range(visible['board_size'] ** 2)]
        visible['units'], visible['cities'] = [], []
        if row['entity'] == 'tiles':
            visible['tiles'][tile] = fact
        else:
            visible[row['entity']] = [fact]
        public_state(replace(obs, visible_state=visible, remembered_state={}))


class PublicMemory:
    def __init__(self):
        self.seats = {str(p): dict(revision=-1, fingerprint=None, facts={}, actions=[], memory={}) for p in (1, 2)}

    def observe(self, obs):
        seat = self.seats[str(obs.player)]
        fingerprint = canonical(obs.visible_state)
        if obs.state_version < seat['revision']:
            raise ValueError('observation timestamp regressed')
        if obs.state_version == seat['revision']:
            if fingerprint != seat['fingerprint']:
                raise ValueError('public observation changed without a revision')
            return replace(obs, remembered_state=deepcopy(seat['memory']))
        known = {t['tile'] for t in obs.visible_state['tiles'] if t['known']}
        # Current public information supersedes prior records, including absence.
        facts = {k: v for k, v in seat['facts'].items() if v['fact']['tile'] not in known}
        historical = sorted(facts.values(), key=lambda r: (-r['revision'], r['entity'], r['fact']['tile']))[:HISTORY_CAP]
        facts = {f"{r['entity']}:{r['fact']['tile']}": r for r in historical}
        for entity in ('tiles', 'units', 'cities'):
            for fact in obs.visible_state[entity]:
                if entity == 'tiles' and not fact['known']:
                    continue
                facts[f"{entity}:{fact['tile']}"] = dict(entity=entity, fact=deepcopy(fact),
                    last_seen_turn=obs.visible_state['turn'], revision=obs.state_version)
        memory = dict(schema=SCHEMA, recent_actions=deepcopy(seat['actions']), historical=historical)
        seat.update(revision=obs.state_version, fingerprint=fingerprint, facts=facts, memory=memory)
        return replace(obs, remembered_state=deepcopy(memory))

    def executed(self, obs, selected):
        seat = self.seats[str(obs.player)]
        if seat['actions'] and obs.state_version <= seat['actions'][-1]['revision']:
            raise ValueError('executed action timestamp regressed')
        seat['actions'] = (seat['actions'] + [dict(command=dict(selected.command), turn=obs.visible_state['turn'], revision=obs.state_version)])[-ACTION_CAP:]

    def snapshot(self):
        return dict(schema=SCHEMA, seats=deepcopy(self.seats))

    @classmethod
    def restore(cls, payload):
        if payload.get('schema') != SCHEMA or set(payload.get('seats', {})) != {'1', '2'}:
            raise ValueError('invalid memory snapshot')
        result = cls()
        result.seats = deepcopy(payload['seats'])
        return result
