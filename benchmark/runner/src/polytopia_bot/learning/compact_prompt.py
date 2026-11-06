"""Compact full-state decision prompt (polytopia-compact-v2).

Same public observation, same legal catalog and the same prompt-local ``aN`` IDs as
``render_readable_decision``; only the presentation changes:

* every move carries the complete public state (no deltas, no transcript);
* rule constants live in a static system prompt, then slowly changing content
  (terrain, territory, features) comes first and fast-changing content (units,
  stars, recent actions, legal actions) last, so consecutive moves share a long
  token prefix;
* derived helper sections (decision_facts, action_names, per-observation rules)
  are dropped; tiles are written as x,y coordinates.

``parse_compact`` is the exact inverse of the user text (public state minus the
derived rule glossary) and backs the equivalence tests and corpus audits.
"""
from copy import deepcopy
from dataclasses import replace
import json
import math
import re

from .interface import canonical, render, rule_table

# v2.1: ability 4 is shown as demolish_own plus the doomed structure. Plain "destroy" read as an
# attack to the model, which then razed its own economy (compact-prompt-v2, 12/30 games).
VERSION = 'polytopia-compact-v2.1'
KINDS = {1: 'move', 2: 'attack', 3: 'ability', 4: 'recruit', 5: 'harvest', 6: 'build', 7: 'research',
         8: 'capture', 9: 'city_reward', 10: 'end_turn', 11: 'resign'}
ABILITIES = {1: 'burn_forest', 2: 'clear_forest', 3: 'grow_forest', 4: 'demolish_own', 7: 'recover', 8: 'disband',
             9: 'heal', 16: 'promote', 18: 'break_peace', 19: 'peace_response', 20: 'embassy', 21: 'offer_peace',
             22: 'remove_embassy'}
# Diplomatic abilities address players (source) and answers (target), not tiles.
DIPLOMATIC = {18, 19, 20, 21, 22}
REWARDS = {1: 'wall', 2: 'park', 3: 'workshop', 4: 'explorer', 5: 'border_growth', 6: 'superunit',
           7: 'resources', 8: 'population'}
TERRAIN = {1: 'w', 2: 'o', 3: 'f', 4: 'm', 5: 't', 6: 'i', 7: 'b', 8: 'g'}
TERRAIN_NAMES = {1: 'water', 2: 'ocean', 3: 'field', 4: 'mountain', 5: 'forest', 6: 'ice', 7: 'wetland', 8: 'mangrove'}
RESOURCES = {1: 'animal', 2: 'crop', 3: 'fish', 5: 'metal', 6: 'fruit', 7: 'spores', 8: 'starfish', 9: 'aquatic_crop'}
OWN_UNIT = ('attacked', 'boosted', 'detect', 'frozen', 'health', 'home', 'invisible', 'kills', 'kind', 'moved',
            'owner', 'passenger', 'poison', 'tile', 'veteran')
UNIT_FLAGS = ('veteran', 'moved', 'attacked', 'poison', 'frozen', 'boosted', 'invisible', 'detect')
OTHER_UNIT = ('health', 'kind', 'owner', 'tile', 'veteran')
OWN_CITY = ('connected', 'level', 'owner', 'population', 'production', 'progress', 'rewards', 'riot', 'tile')
# Every known tile in the corpus carries exactly these keys; any other shape is sent as JSON.
STD_TILE = {'tile', 'known', 'terrain', 'owner', 'resource', 'structure', 'road', 'capital', 'structure_level'}
RELATIONS = ('known_players', 'peace', 'embassies_sent', 'embassies_received', 'peace_offers')
# Rule IDs present anywhere in the 356,727-decision expert-sft-v1 corpus (census 2026-10-19).
# They form the static glossary; any other ID is appended to that observation's user text.
STATIC_RULES = dict(units=[2, 3, 4, 5, 8, 9, 10, 11, 12, 38, 41, 42, 43, 44, 45],
                    technologies=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 38, 44],
                    structures=[0, 1, 2, 3, 8, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 47, 48, 50])


def _names(kind):
    return {r['id']: r['name'] for r in rule_table()[kind]}


def _num(x):
    return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)


def unit_line(r):
    s = f"{r['name']} {r['cost']}* hp{_num(r['health'])} atk{_num(r['attack'])} def{_num(r['defense'])} mov{r['movement']} rng{r['range']}"
    return s + (' ' + ' '.join(r['skills']) if r['skills'] else '')


def tech_line(r, structures):
    s = f"{r['name']} T{r['tier']}"
    if r['requires'] is not None:
        s += ' after ' + _names('technologies').get(r['requires'], f"#{r['requires']}")
    if r['structures']:
        s += ' builds ' + ' '.join(structures.get(i, f'#{i}') for i in r['structures'])
    return s


def structure_line(r):
    cost = '-' if r['cost'] is None else f"{r['cost']}*"
    return f"{r['name']} {cost} pop{r['population']} stars{r['stars']}"


def glossary(ids):
    table = rule_table()
    snames = _names('structures')
    rows = {k: {r['id']: r for r in table[k]} for k in ('units', 'technologies', 'structures')}
    out = []
    if ids['units']:
        out.append('units (cost* hp attack defense movement range skills): ' + '; '.join(unit_line(rows['units'][i]) for i in ids['units']))
    if ids['technologies']:
        out.append('technologies (tier, prerequisite, unlocked structures): ' + '; '.join(tech_line(rows['technologies'][i], snames) for i in ids['technologies']))
    if ids['structures']:
        out.append('structures (cost*, base population, base stars): ' + '; '.join(structure_line(rows['structures'][i]) for i in ids['structures']))
    return out


SYSTEM = '\n'.join([
    'You play Polytopia (1v1). Each message is the complete public state of your current decision.',
    'Choose exactly one listed legal action and answer only {"action_id":"aN"}.',
    'Tiles are x,y (x = column left to right, y = row top to bottom, from 0). Y = you, E = enemy, N = unowned.',
    'terrain grid: ' + ' '.join(f'{c} {TERRAIN_NAMES[k]}' for k, c in TERRAIN.items()) + ', ? unknown. '
    'territory grid: Y/E owner, . unowned, ? unknown.',
    'Features list tiles by kind; structure Lk = level k (no suffix = level 1). road/capital =v gives a value other than 1.',
    'Units: kind x,y hp, then true flags; your units also give their home city (home - = none), kills kN and cargo.',
    'Cities: owner x,y; yours add level, pop, prod (production), prog (progress), connected/riot and rewards taken.',
    'Legal actions are grouped by kind and acting tile ("from x,y"); each item is aN then its parameter and target. '
    'n* is the star cost. One answer is one action; more actions may follow before end_turn.',
    'demolish_own permanently removes one of YOUR OWN structures (named after it); it never harms the enemy.',
    'Rules:', *glossary(STATIC_RULES),
])


class _Tiles:
    def __init__(self, size):
        self.size = size

    def xy(self, tile):
        return f'{tile % self.size},{tile // self.size}'

    def tile(self, text):
        x, y = text.split(',')
        x, y = int(x), int(y)
        if not (0 <= x < self.size and 0 <= y < self.size):
            raise ValueError('tile outside board')
        return y * self.size + x


def _owner(o, player):
    return 'Y' if o == player else 'N' if o == 0 else 'E' if o == 3 - player else f'P{o}'


def _unowner(s, player):
    return dict(Y=player, N=0, E=3 - player).get(s) if s in ('Y', 'N', 'E') else int(s[1:])


def _health(h):
    if type(h) is float:
        return _num(h)
    return f'{h}i'


def _unhealth(s):
    return int(s[:-1]) if s.endswith('i') else float(s)


def _star_cost(public, c):
    """Immediate star cost for display only (same arithmetic as decision_harness)."""
    table = rule_table()
    k, p = c['kind'], c['parameter']
    if k == 4:
        return {r['id']: r['cost'] for r in table['units']}.get(p)
    if k == 5:
        return 2
    if k == 6:
        return {r['id']: r['cost'] for r in table['structures']}.get(p)
    if k == 7:
        v = public['visible']
        tier = {r['id']: r['tier'] for r in table['technologies']}.get(p)
        if tier is None:
            return None
        cost = 4 + sum(ci['owner'] == public['player'] for ci in v['cities']) * tier
        return math.ceil(cost * .66) if 22 in v['technology'] else cost
    return None


def _field(t, kind, param, which, value):
    """Display one command source/target value; -1 is never displayed."""
    if kind == 3 and param in DIPLOMATIC:
        return f'#{value}'
    return t.xy(value)


def _unfield(t, kind, param, text):
    if text.startswith('#'):
        if not (kind == 3 and param in DIPLOMATIC):
            raise ValueError('raw value outside a diplomatic ability')
        return int(text[1:])
    return t.tile(text)


def _param(kind, param):
    if kind == 3:
        return ABILITIES.get(param, f'ability#{param}')
    if kind == 4:
        return _names('units').get(param, f'unit#{param}')
    if kind == 6:
        return _names('structures').get(param, f'structure#{param}')
    if kind == 7:
        return _names('technologies').get(param, f'tech#{param}')
    if kind == 9:
        return REWARDS.get(param, f'reward#{param}')
    return f'p{param}' if param else None


def _unparam(kind, text):
    if text is None:
        return 0
    inverse = {3: {v: k for k, v in ABILITIES.items()}, 4: {v: k for k, v in _names('units').items()},
               6: {v: k for k, v in _names('structures').items()}, 7: {v: k for k, v in _names('technologies').items()},
               9: {v: k for k, v in REWARDS.items()}}.get(kind)
    if inverse is None:
        if not text.startswith('p'):
            raise ValueError('bad parameter')
        return int(text[1:])
    if '#' in text:
        return int(text.split('#')[1])
    return inverse[text]


def command_text(t, c, public=None):
    """One command, e.g. ``move 3,4>4,5`` (used for recent actions)."""
    k = c['kind']
    s = KINDS[k]
    p = _param(k, c['parameter'])
    if p is not None:
        s += ' ' + p
    if c['source'] != -1:
        s += ' ' + _field(t, k, c['parameter'], 'source', c['source'])
    s += '>' + ('' if c['target'] == -1 else _field(t, k, c['parameter'], 'target', c['target']))
    return s


def parse_command(t, text):
    m = re.fullmatch(r'(\w+)(?: ([\w#]+))?(?: ([#\d,-]+))?>([#\d,-]*)', text)
    if not m:
        raise ValueError('bad command '+text)
    kind = {v: k for k, v in KINDS.items()}[m[1]]
    param = _unparam(kind, m[2])
    src = -1 if m[3] is None else _unfield(t, kind, param, m[3])
    tgt = -1 if m[4] == '' else _unfield(t, kind, param, m[4])
    return dict(kind=kind, source=src, target=tgt, parameter=param)


def legal_lines(t, public):
    groups = []
    for entry in public['legal_actions']:
        c = entry['command']
        key = (c['kind'], c['source'], c['kind'] == 3 and c['parameter'] in DIPLOMATIC)
        if not groups or groups[-1][0] != key:
            groups.append((key, []))
        item = entry['action_id']
        p = _param(c['kind'], c['parameter'])
        if p is not None:
            item += ' ' + p
        cost = _star_cost(public, c)
        if cost is not None:
            item += f' {cost}*'
        if c['kind'] == 3 and c['parameter'] == 4 and c['target'] != -1:
            doomed = public['visible']['tiles'][c['target']].get('structure', 0)
            if doomed:
                item += ' ' + _names('structures').get(doomed, f'structure#{doomed}')
        if c['target'] != -1:
            item += ' ' + _field(t, c['kind'], c['parameter'], 'target', c['target'])
        groups[-1][1].append(item)
    lines = []
    for (kind, source, diplomatic), items in groups:
        head = KINDS[kind]
        if source != -1:
            head += ' from ' + (f'#{source}' if diplomatic else t.xy(source))
        lines.append(head + ': ' + '; '.join(items))
    return lines


def _sparse_lines(t, tiles):
    """Known-tile features other than terrain/owner, grouped by value for compactness."""
    snames, lines = _names('structures'), []
    res = {}
    for tile in tiles:
        if tile['known'] and tile.get('resource', 0):
            res.setdefault(tile['resource'], []).append(t.xy(tile['tile']))
    if res:
        lines.append('resources: ' + '; '.join(f"{RESOURCES.get(k, f'resource#{k}')} {' '.join(v)}" for k, v in sorted(res.items())))
    st = {}
    for tile in tiles:
        if tile['known'] and tile.get('structure', 0):
            level = tile.get('structure_level', 0)
            st.setdefault(tile['structure'], []).append(t.xy(tile['tile']) + ('' if level == 1 else f'L{level}'))
    if st:
        lines.append('structures: ' + '; '.join(f"{snames.get(k, f'structure#{k}')} {' '.join(v)}" for k, v in sorted(st.items())))
    stray = [t.xy(x['tile']) + f"L{x['structure_level']}" for x in tiles
             if x['known'] and x.get('structure_level', 0) and not x.get('structure', 0)]
    if stray:
        lines.append('levels without structure: ' + ' '.join(stray))
    for key in ('road', 'capital'):
        vals = [t.xy(x['tile']) + ('' if x[key] == 1 else f'={x[key]}') for x in tiles if x['known'] and x.get(key, 0)]
        if vals:
            lines.append(f'{key}: ' + ' '.join(vals))
    return lines


def render_user(public):
    v, player = public['visible'], public['player']
    n = v['board_size']
    t = _Tiles(n)
    tiles = v['tiles']
    out = [f'{VERSION} player {player} board {n}x{n}']
    # Exotic shapes never seen in the corpus stay lossless through explicit JSON lines.
    exotic = {x['tile'] for x in tiles if x['known'] and set(x) != STD_TILE}

    def cell(x, value):
        return '?' if not x['known'] else '*' if x['tile'] in exotic else value(x)
    out.append('terrain:')
    for y in range(n):
        row = tiles[y * n:(y + 1) * n]
        out.append(f'y{y} ' + ' '.join(cell(x, lambda x: TERRAIN.get(x['terrain'], f"#{x['terrain']}")) for x in row))
    out.append('territory:')
    for y in range(n):
        row = tiles[y * n:(y + 1) * n]
        out.append(f'y{y} ' + ' '.join(cell(x, lambda x: '.' if x['owner'] == 0 else _owner(x['owner'], player)) for x in row))
    out += _sparse_lines(t, [x for x in tiles if x['tile'] not in exotic])
    for x in tiles:
        if x['tile'] in exotic:
            out.append('tile json: ' + canonical(x))
    extra = {k: sorted(set(ids) - set(STATIC_RULES[k])) for k, ids in
             ((k, [r['id'] for r in public.get('rules', {}).get(k, [])]) for k in STATIC_RULES)}
    if any(extra.values()):
        out += ['extra rules: ' + line for line in glossary(extra)]
    snames = _names('structures')
    if 'achievements' in v:
        a = v['achievements']
        out.append('monuments: ' + (' '.join(snames.get(i, f'#{i}') for i in a['monuments']) or '-')
                   + ' | built: ' + (' '.join(snames.get(i, f'#{i}') for i in a['built']) or '-'))
    if 'relations' in v:
        out.append('relations: ' + ' '.join(f"{k}=" + (','.join(str(i + 1) for i, b in enumerate(v['relations'][k]) if b) or '-')
                                            for k in RELATIONS))
    out.append('cities:')
    for c in sorted(v['cities'], key=lambda c: (c['owner'] != player, c['tile'])):
        keys = set(c)
        if c['owner'] == player and keys == set(OWN_CITY):
            rw = ' '.join(REWARDS.get(i, f'reward#{i}') for i, k in enumerate(c['rewards']) for _ in range(k))
            out.append(f"Y {t.xy(c['tile'])} L{c['level']} pop{c['population']} prod{c['production']} prog{c['progress']}"
                       + (' connected' if c['connected'] else '') + (' riot' if c['riot'] else '') + (f' rewards {rw}' if rw else ''))
        elif keys == {'tile', 'owner'} and c['owner'] != player:
            out.append(f"{_owner(c['owner'], player)} {t.xy(c['tile'])}")
        else:
            out.append('city json: ' + canonical(c))
    out.append('technology: ' + ' '.join(_names('technologies').get(i, f'tech#{i}') for i in v['technology']))
    unames = _names('units')
    enemy = [u for u in v['units'] if u['owner'] != player]
    own = [u for u in v['units'] if u['owner'] == player]
    out.append('enemy units:')
    for u in enemy:
        if set(u) == set(OTHER_UNIT):
            out.append(f"{_owner(u['owner'], player)} {unames.get(u['kind'], f'unit#{u['kind']}')} {t.xy(u['tile'])} hp{_health(u['health'])}"
                       + (' veteran' if u['veteran'] else ''))
        else:
            out.append('unit json: ' + canonical(u))
    out.append('your units:')
    for u in own:
        if set(u) == set(OWN_UNIT):
            s = f"{unames.get(u['kind'], f'unit#{u['kind']}')} {t.xy(u['tile'])} hp{_health(u['health'])}"
            s += ' home ' + ('-' if u['home'] == 0 else t.xy(u['home'] - 1))
            s += ''.join(' ' + f for f in UNIT_FLAGS if u[f])
            if u['kills']:
                s += f" k{u['kills']}"
            if u['passenger']:
                s += ' cargo ' + unames.get(u['passenger'], f"unit#{u['passenger']}")
            out.append(s)
        else:
            out.append('unit json: ' + canonical(u))
    memory = public['remembered']
    std_memory = set(memory) == {'schema', 'recent_actions', 'historical', 'tiles'} and memory['schema'] == 'polytopia-memory-v1'
    if std_memory:
        if memory['tiles'] or memory['historical']:
            out.append('memory json: ' + canonical(dict(tiles=memory['tiles'], historical=memory['historical'])))
        out.append('recent actions: ' + ('; '.join(f"t{r['turn']} r{r['revision']} {command_text(t, r['command'])}"
                                                   for r in memory['recent_actions']) or '-'))
    else:
        out.append('remembered json: ' + canonical(memory))
    out.append(f"turn {v['turn']} stars {v['stars']} version {public['state_version']}")
    out.append('legal actions:')
    out += legal_lines(t, public)
    return '\n'.join(out)


def parse_compact(text):
    """Inverse of ``render_user``: the public state without the derived rule glossary."""
    lines = text.split('\n')
    m = re.fullmatch(rf'{VERSION} player (\d+) board (\d+)x(\d+)', lines[0])
    player, n = int(m[1]), int(m[2])
    t = _Tiles(n)
    inv_terrain = {c: k for k, c in TERRAIN.items()}
    i = 1
    assert lines[i] == 'terrain:'
    tiles = []
    for y in range(n):
        cells = lines[i + 1 + y].split(' ')
        assert cells[0] == f'y{y}' and len(cells) == n + 1
        for x, c in enumerate(cells[1:]):
            tile = dict(tile=y * n + x, known=c != '?')
            if c not in '?*':
                tile['terrain'] = int(c[1:]) if c.startswith('#') else inv_terrain[c]
            tiles.append(tile)
    i += 1 + n
    assert lines[i] == 'territory:'
    for y in range(n):
        cells = lines[i + 1 + y].split(' ')
        for x, c in enumerate(cells[1:]):
            tile = tiles[y * n + x]
            if tile['known'] and c != '*':
                tile.update(owner=0 if c == '.' else _unowner(c, player), resource=0, structure=0, road=0,
                            capital=0, structure_level=0)
    i += 1 + n
    snames = {v: k for k, v in _names('structures').items()}
    inv_res = {v: k for k, v in RESOURCES.items()}
    visible = dict(board_size=n, tiles=tiles)
    units, cities, memory = [], [], None
    unames = {v: k for k, v in _names('units').items()}
    tnames = {v: k for k, v in _names('technologies').items()}
    section = None

    def ident(name, table, prefix):
        return int(name.split('#')[1]) if name.startswith(prefix + '#') else table[name]

    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith('resources: '):
            for group in line[11:].split('; '):
                name, *cells = group.split(' ')
                for c in cells:
                    tiles[t.tile(c)]['resource'] = int(name.split('#')[1]) if name.startswith('resource#') else inv_res[name]
        elif line.startswith('structures: '):
            for group in line[12:].split('; '):
                name, *cells = group.split(' ')
                for c in cells:
                    xy, _, level = c.partition('L')
                    tile = tiles[t.tile(xy)]
                    tile['structure'] = ident(name, snames, 'structure')
                    tile['structure_level'] = int(level) if level else 1
        elif line.startswith('levels without structure: '):
            for c in line[26:].split(' '):
                xy, _, level = c.partition('L')
                tiles[t.tile(xy)]['structure_level'] = int(level)
        elif line.startswith(('road: ', 'capital: ')):
            key, _, rest = line.partition(': ')
            for c in rest.split(' '):
                xy, _, val = c.partition('=')
                tiles[t.tile(xy)][key] = int(val) if val else 1
        elif line.startswith('tile json: '):
            x = json.loads(line[11:])
            tiles[x['tile']] = x
        elif line.startswith('extra rules: '):
            pass
        elif line.startswith('monuments: '):
            mon, built = line[11:].split(' | built: ')
            visible['achievements'] = dict(
                monuments=[] if mon == '-' else [ident(s, snames, '') for s in mon.split(' ')],
                built=[] if built == '-' else [ident(s, snames, '') for s in built.split(' ')])
        elif line.startswith('relations: '):
            rel = {}
            for part in line[11:].split(' '):
                k, _, vals = part.partition('=')
                on = set() if vals == '-' else {int(x) for x in vals.split(',')}
                rel[k] = [p in on for p in (1, 2)]
            visible['relations'] = rel
        elif line in ('cities:', 'enemy units:', 'your units:'):
            section = line
        elif line.startswith('city json: '):
            cities.append(json.loads(line[11:]))
        elif line.startswith('unit json: '):
            units.append(json.loads(line[11:]))
        elif line.startswith('technology: '):
            section = None
            visible['technology'] = [ident(s, tnames, 'tech') for s in line[12:].split(' ') if s]
        elif line.startswith('memory json: '):
            memory = json.loads(line[13:])
        elif line.startswith('recent actions: '):
            rest = line[16:]
            recent = []
            if rest != '-':
                for part in rest.split('; '):
                    tt, rr, cmd = part.split(' ', 2)
                    recent.append(dict(command=parse_command(t, cmd), turn=int(tt[1:]), revision=int(rr[1:])))
            base = memory or dict(tiles=[], historical=[])
            memory = dict(schema='polytopia-memory-v1', recent_actions=recent, historical=base['historical'], tiles=base['tiles'])
        elif line.startswith('remembered json: '):
            memory = json.loads(line[17:])
        elif line.startswith('turn '):
            mm = re.fullmatch(r'turn (\d+) stars (\d+) version (\d+)', line)
            visible.update(turn=int(mm[1]), stars=int(mm[2]))
            version = int(mm[3])
        elif line == 'legal actions:':
            break
        elif section == 'cities:':
            parts = line.split(' ')
            c = dict(owner=_unowner(parts[0], player), tile=t.tile(parts[1]))
            if parts[0] == 'Y':
                c.update(level=int(parts[2][1:]), population=int(parts[3][3:]), production=int(parts[4][4:]),
                         progress=int(parts[5][4:]), connected='connected' in parts[6:], riot='riot' in parts[6:], rewards=[0] * 9)
                if 'rewards' in parts:
                    inv = {v: k for k, v in REWARDS.items()}
                    for r in parts[parts.index('rewards') + 1:]:
                        c['rewards'][int(r.split('#')[1]) if r.startswith('reward#') else inv[r]] += 1
            cities.append(c)
        elif section == 'enemy units:':
            parts = line.split(' ')
            units.append(dict(owner=_unowner(parts[0], player), kind=ident(parts[1], unames, 'unit'), tile=t.tile(parts[2]),
                              health=_unhealth(parts[3][2:]), veteran='veteran' in parts[4:]))
        elif section == 'your units:':
            parts = line.split(' ')
            u = dict(owner=player, kind=ident(parts[0], unames, 'unit'), tile=t.tile(parts[1]), health=_unhealth(parts[2][2:]))
            assert parts[3] == 'home'
            u['home'] = 0 if parts[4] == '-' else t.tile(parts[4]) + 1
            rest = parts[5:]
            for f in UNIT_FLAGS:
                u[f] = f in rest
            u['kills'] = next((int(p[1:]) for p in rest if re.fullmatch(r'k\d+', p)), 0)
            u['passenger'] = ident(rest[rest.index('cargo') + 1], unames, 'unit') if 'cargo' in rest else 0
            units.append(u)
        else:
            raise ValueError('unrecognized line: ' + line)
    legal = []
    for line in lines[i:]:
        head, _, rest = line.partition(': ')
        kind_name, _, src = head.partition(' from ')
        kind = {v: k for k, v in KINDS.items()}[kind_name]
        for item in rest.split('; '):
            parts = item.split(' ')
            aid, parts = parts[0], parts[1:]
            param_text = None
            if parts and not re.fullmatch(r'[#\d,-]+|\d+\*', parts[0]):
                param_text, parts = parts[0], parts[1:]
            param = _unparam(kind, param_text)
            # Drop display-only annotations: star costs and the structure a demolish_own removes.
            parts = [p for p in parts if not p.endswith('*') and not (kind == 3 and param == 4 and re.fullmatch(r'[A-Za-z][\w#]*', p))]
            if len(parts) > 1:
                raise ValueError('unexpected legal item: ' + item)
            target = _unfield(t, kind, param, parts[0]) if parts else -1
            source = _unfield(t, kind, param, src) if src else -1
            legal.append(dict(action_id=aid, command=dict(kind=kind, source=source, target=target, parameter=param)))
    visible['units'] = sorted(units, key=lambda u: u['tile'])
    visible['cities'] = sorted(cities, key=lambda c: c['tile'])
    return dict(schema='polytopia-public-v2', player=player, state_version=version, visible=visible,
                remembered=memory, legal_actions=legal)


def without_rules(public):
    """The observation part of a prompt's public state (rules are a derived glossary)."""
    return deepcopy({k: v for k, v in public.items() if k != 'rules'})


def compact_messages(public):
    return [dict(role='system', content=SYSTEM), dict(role='user', content=render_user(public))]


def render_compact_decision(obs):
    prompt = render(obs)
    return replace(prompt, messages=compact_messages(prompt.public))


def action_ids(messages):
    text = messages[1]['content']
    legal = text[text.index('\nlegal actions:\n') + 16:]
    return re.findall(r'(?:^|: |; )(a\d+)', legal, flags=re.M)


def action_schema(messages):
    """Legal-action JSON grammar for compact (or, unchanged, readable) messages."""
    if not messages[1]['content'].startswith(VERSION):
        from .thinking_readable import action_schema as readable
        return readable(messages)
    ids = action_ids(messages)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('invalid action catalog')
    return dict(type='object', properties={'action_id': dict(type='string', enum=ids)},
                required=['action_id'], additionalProperties=False)


def public_from_readable(user_text):
    """Recover the full public state (with rules and legal catalog) from a stored readable user message."""
    from .encoding import expand
    from .thinking_debug import restore_dense_body
    from .thinking_readable import restore_sparse_body
    body = json.loads(user_text)
    body = restore_dense_body(restore_sparse_body(body))
    body.pop('decision_facts')
    return expand(body)


def readable_from_public(public):
    """Re-render the readable user message from a public state (proves public_from_readable is exact)."""
    from .decision_harness import INSTRUCTIONS as FACTS, action_facts
    from .encoding import INSTRUCTIONS, compact
    from .interface import SYSTEM as BASE
    from .thinking_debug import sparse_messages
    from .thinking_readable import readable_messages
    encoded = compact(public)
    encoded['decision_facts'] = action_facts(public)
    messages = [dict(role='system', content=BASE + '\n' + INSTRUCTIONS + '\n' + FACTS), dict(role='user', content=canonical(encoded))]
    return readable_messages(sparse_messages(messages, brief=True))
