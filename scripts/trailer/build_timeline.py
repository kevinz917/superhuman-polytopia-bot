"""Turn games from gen_games.py into the trailer's timeline: the last few turns of each game, half-turn by half-turn,
with every unit tracked by id through moves, attacks, deaths, training and kind changes (embarking).

Usage: python3 scripts/trailer/build_timeline.py OUT.json GAME.json:HALVES [GAME.json:HALVES ...]

HALVES is how many half-turns (one player's turn each) before the end to show. The state at the window start (tiles, cities, units with ids)
is reconstructed from the whole game; within the window each action lists what it did to which unit ids.
"""

import json
import sys


def track(game, show_halves):
    # Half-turn index of every step: it advances whenever the player to move changes.
    half_of, h, mover = [], -1, None
    for step in game['steps']:
        if step['player'] != mover:
            h, mover = h + 1, step['player']
        half_of.append(h)
    first_half = max(0, h + 1 - show_halves)
    tiles = [list(t) for t in game['tiles']]
    cities = {int(k): v for k, v in game['cities'].items()}
    units = {int(k): v for k, v in game['units'].items()}
    ids = {}
    next_id = 0
    for tile in units:
        ids[tile] = next_id
        next_id += 1
    out = None
    halves = []
    prev_scores = None
    for i, step in enumerate(game['steps']):
        if out is None and half_of[i] >= first_half:
            out = {'size': game['size'], 'winner': game['winner'], 'hero_seat': game['hero_seat'],
                   'scores': prev_scores, 'tiles': [list(t) for t in tiles],
                   'cities': {str(k): v for k, v in cities.items()},
                   'units': [{'id': ids[t], 'tile': t, 'owner': u[0], 'kind': u[1]} for t, u in units.items()]}
        new = {int(k): v for k, v in step['units'].items()}
        cmd = step['command']
        kind, src, tgt = cmd['kind'], cmd.get('source', -1), cmd.get('target', -1)
        act = {'kind': kind, 'value': step.get('value'), 'scores': step.get('scores')}
        moved = None
        if kind in (1, 2) and src in units:
            u = units[src]
            gone = src not in new or new[src][0] != u[0]
            if gone and tgt in new and new[tgt][0] == u[0]:
                moved = (src, tgt)
            act['unit'] = ids[src]
            act['from'] = src
            act['to'] = tgt
        if kind == 8 and src in units:
            act['unit'] = ids.get(src)
            act['at'] = src
        if kind == 4:
            act['at'] = src
        died, born, changed = [], [], []
        new_ids = {}
        if moved:
            new_ids[moved[1]] = ids[moved[0]]
            if new[moved[1]][1] != units[moved[0]][1]:
                changed.append([ids[moved[0]], new[moved[1]][1]])
            act['moved'] = True
        for t, u in units.items():
            if moved and t == moved[0]:
                continue
            if t in new and new[t][0] == u[0] and not (moved and t == moved[1]):
                new_ids[t] = ids[t]
                if new[t][1] != u[1]:
                    changed.append([ids[t], new[t][1]])
            else:
                died.append(ids[t])
        for t, u in new.items():
            if t not in new_ids:
                new_ids[t] = next_id
                born.append({'id': next_id, 'tile': t, 'owner': u[0], 'kind': u[1]})
                next_id += 1
        if kind == 2 and tgt in units and tgt in new and new[tgt][0] == units[tgt][0] and new[tgt][2] < units[tgt][2]:
            act['hit'] = ids[tgt]
        if died:
            act['died'] = died
        if born:
            act['born'] = born
        if changed:
            act['changed'] = changed
        if 'tiles' in step:
            for k, v in step['tiles'].items():
                tiles[int(k)] = v
            act['tiles'] = step['tiles']
        if 'cities' in step:
            cities = {int(k): v for k, v in step['cities'].items()}
            act['cities'] = step['cities']
        units, ids = new, new_ids
        prev_scores = step.get('scores')
        if out is None:
            continue
        if not halves or halves[-1]['player'] != step['player']:
            halves.append({'player': step['player'], 'turn': step['turn'], 'actions': []})
        if kind not in (7, 10) or len(act) > 3:  # research and end turn change nothing on the board
            halves[-1]['actions'].append(act)
    out['halves'] = [h for h in halves if h['actions']]
    return out


if __name__ == '__main__':
    games = []
    for arg in sys.argv[2:]:
        path, turns = arg.rsplit(':', 1)
        g = track(json.load(open(path)), int(turns))
        n = sum(len(h['actions']) for h in g['halves'])
        print(f'{path}: {len(g["halves"])} half-turns, {n} actions, {len(g["units"])} units at start')
        games.append(g)
    json.dump(games, open(sys.argv[1], 'w'), separators=(',', ':'))
