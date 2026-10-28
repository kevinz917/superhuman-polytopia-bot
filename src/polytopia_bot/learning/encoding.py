"""Lossless prompt transport, separate from the validated semantic observation."""
from copy import deepcopy

VERSION = 'polytopia-compact-v3'
INSTRUCTIONS = '''Input encoding: polytopia-compact-v3. Terrain is a row-major array;
-1 means unknown. Sparse tile columns list [tile,value]; unlisted known tiles
have value 0 and unknown tiles have no value. Entity tables contain columns and
rows; null means the field is absent, not a known zero. Rule definitions also
use column tables (null retains its usual value there). The legal_actions table
has columns [action_id,kind,source,target,parameter]. Each row is one legal choice.
Copy its action_id exactly, including the letter a. For example, to select the
row ["a17",1,23,24,0], return exactly {"action_id":"a17"}.
Do not return the kind number, command fields, or the placeholder aN.
The table is the complete catalog. Recent actions are executed facts;
historical facts are stale observations, never claims of current presence.'''


def table(rows):
    columns = sorted({k for row in rows for k in row})
    return dict(columns=columns, rows=[[row.get(k) for k in columns] for row in rows])


def compact(state):
    out = deepcopy(state)
    visible = out['visible']
    tiles = visible.pop('tiles')
    terrain = [t.get('terrain', 0) if t['known'] else -1 for t in tiles]
    # Tile indices are explicit in the semantic schema, not necessarily ordered
    # in callers. public_state sorts them before this transport.
    sparse = {k: [[t['tile'], t[k]] for t in tiles if t['known'] and t.get(k, 0) != 0]
              for k in sorted({k for t in tiles for k in t} - {'tile', 'known', 'terrain'})}
    # Presence masks preserve optional-field absence as distinct from zero.
    presence = {k: [t['tile'] for t in tiles if t['known'] and k not in t]
                for k in ('terrain', *sparse)}
    visible['tiles'] = dict(terrain=terrain, sparse=sparse, absent={k: v for k, v in presence.items() if v})
    for entity in ('units', 'cities'):
        visible[entity] = table(visible[entity])
    out['rules'] = {k: table(rows) for k, rows in out['rules'].items()}
    out['legal_actions'] = dict(columns=['action_id','kind','source','target','parameter'],
        rows=[[entry['action_id'], *(entry['command'][k] for k in ('kind','source','target','parameter'))]
              for entry in out['legal_actions']])
    out['encoding'] = VERSION
    return out


def expand(encoded):
    """Exact inverse for regression tests and retained-data audits."""
    out = deepcopy(encoded)
    if out.pop('encoding') != VERSION:
        raise ValueError('unknown encoding')
    visible = out['visible']
    grid = visible.pop('tiles')
    tiles = []
    for i, terrain in enumerate(grid['terrain']):
        tile = dict(tile=i, known=terrain != -1)
        if tile['known']:
            tile['terrain'] = terrain
            tile.update({k: dict(pairs).get(i, 0) for k, pairs in grid['sparse'].items()})
            for k, absent in grid['absent'].items():
                if i in absent:
                    tile.pop(k, None)
        tiles.append(tile)
    visible['tiles'] = tiles
    for entity in ('units', 'cities'):
        t = visible[entity]
        visible[entity] = [{k: v for k, v in zip(t['columns'], row) if v is not None} for row in t['rows']]
    out['rules'] = {k: [dict(zip(t['columns'], row)) for row in t['rows']] for k, t in out['rules'].items()}
    out['legal_actions'] = [dict(action_id=row[0],command=dict(zip(('kind','source','target','parameter'),row[1:])))
                            for row in out['legal_actions']['rows']]
    return out
