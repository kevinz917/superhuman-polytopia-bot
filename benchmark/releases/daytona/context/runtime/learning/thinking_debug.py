"""Lossless terrain transport and observable generation for the thinking repair."""
from copy import deepcopy
from dataclasses import replace
import json

from .decision_harness import render_decision
from .interface import canonical

VERSION = 'public-action-facts-sparse-terrain-v1'
OLD_TERRAIN = 'Input encoding: polytopia-compact-v3. Terrain is a row-major array;\n-1 means unknown.'
NEW_TERRAIN = ('Input encoding: polytopia-compact-sparse-terrain-v1. Terrain gives '
               'size (number of board tiles), default -1 (unknown), and known '
               '[tile_index,terrain_id] pairs. Unlisted tiles are unknown. '
               'Tile indices are row-major. Read the pairs directly; do not expand '
               'or copy a dense terrain array.')
BRIEF = ('\nDecision procedure: assess the current position and compare at most three '
         'promising listed actions briefly. Do not transcribe input tables or enumerate '
         'unknown tiles. Once you have chosen, finish your reasoning and emit the final '
         'action JSON. Every catalog action is legal now; do not second-guess its legality. '
         'One response selects one action, not an entire turn: further actions may follow '
         'before end_turn. A unit\'s tile is its current position; home is its home-city '
         'reference. Harvesting is a city/economy action and does not require moving a unit '
         'onto that resource. Owner 0 denotes unowned/neutral, not the opposing player.')


def sparse_messages(messages, *, brief=False):
    result = deepcopy(messages)
    if OLD_TERRAIN not in result[0]['content']:
        raise ValueError('unexpected terrain instructions')
    body = json.loads(result[1]['content'])
    terrain = body['visible']['tiles']['terrain']
    if not isinstance(terrain, list) or len(terrain) != body['visible']['board_size'] ** 2:
        raise ValueError('unexpected terrain array')
    body['visible']['tiles']['terrain'] = dict(size=len(terrain), default=-1,
        known=[[i, value] for i, value in enumerate(terrain) if value != -1])
    body['encoding'] = 'polytopia-compact-sparse-terrain-v1'
    result[0]['content'] = result[0]['content'].replace(OLD_TERRAIN, NEW_TERRAIN)
    if brief:
        result[0]['content'] += BRIEF
    result[1]['content'] = canonical(body)
    return result


def restore_dense_body(body):
    result = deepcopy(body)
    terrain = result['visible']['tiles']['terrain']
    values = [terrain['default']] * terrain['size']
    for tile, value in terrain['known']:
        values[tile] = value
    result['visible']['tiles']['terrain'] = values
    result['encoding'] = 'polytopia-compact-v3'
    return result


def render_sparse_decision(obs):
    prompt = render_decision(obs)
    return replace(prompt, messages=sparse_messages(prompt.messages, brief=True))


def repeated_suffix(tokens, *, window=1024, max_period=128):
    """Detect exact numeric or prose loops, requiring at least eight cycles.

    This is a repetition failure detector, not an output/thinking allowance.
    A non-periodic response remains unrestricted regardless of its length.
    """
    if len(tokens) < window:
        return 0
    tail = tokens[-window:]
    for period in range(1, min(max_period, window // 8) + 1):
        if all(tail[i] == tail[i % period] for i in range(period, window)):
            return period
    return 0
