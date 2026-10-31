"""The game's rules on engine-rs: Polytopia 2.17.3, game version 123 (docs/rules-alignment-2026-10-27.md).

Each rule where our earlier research rules differed from the game is checked on one hand-edited position, with
a control position where the rule does not apply. Parity with recorded games (every rule at once, on 27,785
transitions and their legal-move sets) was checked separately; see the doc above.
"""
import numpy as np
import pytest

SIZE = 14
MOVE, ATTACK, SUMMON, BUILD, RESEARCH, HARVEST, FOREST, REWARD, CAPTURE, RECOVER, PROMOTE, DISBAND, HEAL, DESTROY = range(14)
END = 15
WATER, FIELD, MOUNTAIN, FOREST_T = 1, 3, 4, 5
WARRIOR, RIDER, KNIGHT, DEFENDER, ARCHER, RAFT, RAMMERSHIP, CLOAK = 2, 3, 4, 5, 9, 43, 44, 38


def _game(seed=11, players=2, size=SIZE):
    from polytopia_bot.rl.rust_engine import RustGames
    g = RustGames(size, 1, players=players)
    g.batch.reset(np.array([seed], np.int32))
    return g


def _state(seed=11):
    return {k: np.array(v) for k, v in _game(seed).batch.export().items()}


def _load(d):
    g = _game()
    g.batch.import_({k: np.ascontiguousarray(v) for k, v in d.items()})
    return g


def _step(g, family, source, target):
    i = lambda v: np.array([v], np.int32)
    return int(g.batch.step(i(family), i(source), i(target))[0])


def _export(g):
    return {k: np.array(v)[0] for k, v in g.batch.export().items()}


def _legal(d, family, source, target):
    from polytopia_bot.rl.rust_engine import split_masks
    m = np.asarray(split_masks(_load(d).batch.legal_masks(), SIZE)[family])[0]
    return bool(m[source, target] if m.ndim == 2 else (m[source] if m.size == SIZE * SIZE else m[target]))


def _dist(a, b):
    return max(abs(a % SIZE - b % SIZE), abs(a // SIZE - b // SIZE))


def _capital(d, player):
    return int(np.flatnonzero((d['tiles.capital'][0] == player) & (d['cities.owner'][0] == player))[0])


def _ring(c, r=1):
    return [t for t in range(SIZE * SIZE) if _dist(c, t) == r]


def _clear_units(d):
    for k in [k for k in d if k.startswith('units.')]:
        d[k][0] = 0


def _unit(d, tile, owner, kind, health=10., **kw):
    d['units.owner'][0, tile], d['units.kind'][0, tile], d['units.health'][0, tile] = owner, kind, health
    d['units.home'][0, tile] = 0
    for k, v in kw.items():
        d[f'units.{k}'][0, tile] = v


def _own_tiles(d, city):
    return [t for t in np.flatnonzero(d['cities.territory'][0, city]) if t != city]


def _own_tile(d, terrain, city=None, skip=()):
    city = _capital(d, 1) if city is None else city
    for t in _own_tiles(d, city):
        if d['tiles.structure'][0, t] == 0 and d['units.owner'][0, t] == 0 and t not in skip:
            d['tiles.terrain'][0, t], d['tiles.resource'][0, t] = terrain, 0
            return t
    raise AssertionError('no free own tile')


def _strip(d, terrain=FIELD):
    """A clean 3-row band through the middle of the board: given terrain, no structures, units or roads."""
    _clear_units(d)
    y = SIZE // 2
    band = [x + yy * SIZE for yy in (y - 1, y, y + 1) for x in range(SIZE)]
    for t in band:
        d['tiles.terrain'][0, t], d['tiles.structure'][0, t], d['tiles.road'][0, t] = terrain, 0, False
        d['tiles.resource'][0, t], d['tiles.owner'][0, t] = 0, 0
        d['cities.owner'][0, t] = 0
        d['tiles.explored'][0, :, t] = True
    return y * SIZE


def _random_play(players, size, games, ticks, seed):
    """Uniform random legal play; every action must be accepted."""
    from polytopia_bot.rl.actions import shapes
    from polytopia_bot.rl.rust_engine import RustGames, split_masks
    rng = np.random.default_rng(seed)
    g = RustGames(size, games, players=players)
    g.batch.reset(np.arange(games, dtype=np.int32))
    shp = shapes(size)
    for _ in range(ticks):
        masks = split_masks(g.batch.legal_masks(), size)
        if players > 2:
            assert not np.asarray(masks.diplomatic).any()   # four seats: no diplomacy
        choice = [np.array([v] * games, np.int32) for v in (1, 0, 0)]   # finished games: the rejected self-attack
        for k in range(games):
            mats = [np.asarray(m[k]).reshape(shp[f]) for f, m in enumerate(masks)]
            fams = [f for f in range(17) if mats[f].any()]
            if not fams:
                continue
            f = END if END in fams and rng.random() < .15 else int(rng.choice(fams))
            s = int(rng.choice(np.flatnonzero(mats[f].any(1))))
            choice[0][k], choice[1][k], choice[2][k] = f, s, int(rng.choice(np.flatnonzero(mats[f][s])))
        e = g.batch.export()
        done = np.asarray(e['terminated']) | np.asarray(e['truncated'])
        status = g.batch.step(*choice)
        assert ((status == 0) | done).all()


@pytest.mark.parametrize('players,size', [(2, 14), (4, 18)])
def test_random_games_run_clean(players, size):
    _random_play(players, size, 3, 1200, players)


def test_engine_has_one_ruleset():
    from polytopia_bot.rl.rust_engine import RustGames
    with pytest.raises(TypeError):
        RustGames(SIZE, 1, ruleset='bardur-1v1-v2')


def test_away_stars_for_seat_two_meeting_player_one():
    """Meeting reward (version >= 123, two players): meeting player 1 pays 2 more than meeting player 2."""
    gain = {}
    for mover in (1, 2):
        d = _state()
        other = 3 - mover
        d['current_player'][0] = mover
        d['players.known'][0, mover - 1, other - 1] = False
        d['players.score'][0, other - 1] = 0                  # both meetings pay the score-free base
        theirs = _capital(d, other)
        d['tiles.explored'][0, mover - 1, theirs] = True     # their capital's unit comes into view
        _unit(d, theirs, other, WARRIOR)
        d['players.stars'][0, mover - 1] = 100
        tech = next(k for k in (1, 6, 10, 15, 20) if not d['players.tech'][0, mover - 1, k])
        g = _load(d)
        assert _step(g, RESEARCH, 0, tech) == 0              # any action runs the meeting check after it
        e = _export(g)
        assert e['players.known'][mover - 1, other - 1]
        cost = 100 - int(_export(_research_only(d, tech))['players.stars'][mover - 1])
        gain[mover] = int(e['players.stars'][mover - 1]) - (100 - cost)
    assert gain == {1: 1, 2: 3}


def _research_only(d, tech):
    """The same research without anyone to meet (to read its price)."""
    e = {k: v.copy() for k, v in d.items()}
    me = int(e['current_player'][0])
    e['players.known'][0, me - 1, 2 - me] = True          # already met: no gift
    g = _load(e)
    assert _step(g, RESEARCH, 0, tech) == 0
    return g


def _duel(attacker_kind, defender_kind, distance, attacker_health=10., explored=True, **defender):
    d = _state()
    _clear_units(d)
    d['current_player'][0] = 1
    cap = _capital(d, 1)
    land = [t for t in range(SIZE * SIZE) if d['tiles.terrain'][0, t] == FIELD and d['tiles.structure'][0, t] == 0]
    a = next(t for t in land if _dist(t, cap) <= 3)
    b = next(t for t in land if _dist(a, t) == distance)
    _unit(d, a, 1, attacker_kind, attacker_health)
    _unit(d, b, 2, defender_kind, **defender)
    d['tiles.explored'][0, 0, b] = True
    d['tiles.explored'][0, 1, a] = explored
    return d, a, b


@pytest.mark.parametrize('defender,retaliates', [(CLOAK, False), (WARRIOR, True)])
def test_cloak_is_stiff(defender, retaliates):
    d, a, b = _duel(WARRIOR, defender, 1, attacker_health=2.)
    if defender == WARRIOR:
        d['units.health'][0, b] = 10.
    g = _load(d)
    assert _step(g, ATTACK, a, b) == 0
    e = _export(g)
    assert e['units.owner'][b] == 2 and (e['units.health'][a] < 2.) == retaliates


@pytest.mark.parametrize('explored', [True, False])
def test_retaliation_needs_the_attacker_tile_explored(explored):
    """Retaliation: the attacker must be one of the defender's attack options."""
    d, a, b = _duel(ARCHER, ARCHER, 2, explored=explored)
    g = _load(d)
    assert _step(g, ATTACK, a, b) == 0
    assert (float(_export(g)['units.health'][a]) < 10.) == explored


def _income_after_end(d):
    g = _load(d)
    before = int(d['players.stars'][0, 0])
    assert _step(g, END, 0, 0) == 0
    return int(_export(g)['players.stars'][0]) - before


def _market_capital():
    d = _state()
    cap = _capital(d, 1)
    ring = [t for t in _ring(cap) if d['units.owner'][0, t] == 0]
    for t in ring:
        d['tiles.structure'][0, t], d['tiles.owner'][0, t] = 0, 1
    # market - forge - mine in a row inside the capital's 3x3: the forge's level is its adjacent mines (1)
    trio = next((m, f, x) for m in ring for f in ring for x in ring
                if len({m, f, x}) == 3 and _dist(m, f) == 1 and _dist(f, x) == 1 and _dist(m, x) == 2)
    for t, k in zip(trio, (50, 22, 21)):
        d['tiles.structure'][0, t], d['tiles.structure_level'][0, t] = k, 1
        d['cities.territory'][0, cap, t] = True
    d['current_player'][0] = 2
    d['turn'][0] = 3
    d['players.known'][0] = True   # already met: a unit placed on the capital must not pay a first-meeting reward
    return d, cap


def test_sieged_city_still_pays_its_markets():
    """CalculateIncomeFor sums CalculateWork per tile: a sieged city earns nothing, its markets still pay."""
    d, cap = _market_capital()
    free = _income_after_end(d)
    _unit(d, cap, 2, WARRIOR)
    sieged = _income_after_end(d)
    others = [c for c in np.flatnonzero(d['cities.owner'][0] == 1) if c != cap]
    assert not others and 0 < sieged < free


def test_hidden_unit_on_city_is_a_siege():
    d, cap = _market_capital()
    free = _income_after_end(d)
    _unit(d, cap, 2, CLOAK, 5., invisible=True)
    assert _income_after_end(d) < free


def test_no_road_on_villages():
    d = _state()
    d['players.tech'][0, 0, 4] = True                      # Roads
    d['players.stars'][0, 0] = 50
    v = int(np.flatnonzero((d['tiles.structure'][0] == 1) & (d['cities.owner'][0] == 0))[0])
    d['tiles.explored'][0, 0, v] = True
    d['units.owner'][0, v] = 0
    assert not _legal(d, BUILD, v, 3)
    d['tiles.structure'][0, v] = 0                         # the same tile without the village
    assert _legal(d, BUILD, v, 3)


def test_lighthouse_cannot_be_destroyed():
    d = _state()
    t = _own_tile(d, FIELD)
    d['players.tech'][0, 0, 3] = True                      # Chivalry: Destroy
    d['tiles.structure'][0, t] = 47
    assert not _legal(d, DESTROY, t, 0)
    d['tiles.structure'][0, t] = 5                         # a farm on the same tile
    assert _legal(d, DESTROY, t, 0)


@pytest.mark.parametrize('peace', [True, False])
def test_work_next_to_a_peace_partner_unit(peace):
    d = _state()
    t = _own_tile(d, FOREST_T)
    _unit(d, t, 2, WARRIOR)
    d['players.peace'][0, 0, 1] = d['players.peace'][0, 1, 0] = peace
    d['players.tech'][0, 0, 16] = True                     # Forestry: clear forest
    assert _legal(d, FOREST, t, 1) == peace


@pytest.mark.parametrize('tech,legal', [(21, True), (22, False)])
def test_mountain_temple_is_meditations(tech, legal):
    d = _state()
    t = _own_tile(d, MOUNTAIN)
    d['players.tech'][0, 0, 20] = True                     # Climbing
    d['players.tech'][0, 0, tech] = True
    d['players.stars'][0, 0] = 50
    assert _legal(d, BUILD, t, 20) == legal


def test_monuments_not_on_forest():
    d = _state()
    d['players.monuments'][0, 0, 27] = True
    forest = _own_tile(d, FOREST_T)
    field = _own_tile(d, FIELD, skip=(forest,))
    assert not _legal(d, BUILD, forest, 27) and _legal(d, BUILD, field, 27)


def test_promotion_at_sea():
    """Promotion judges a boat by its passenger: a Warrior with 3 kills is promoted on a raft."""
    d = _state()
    _clear_units(d)
    t = _own_tile(d, WATER)
    _unit(d, t, 1, RAFT, kills=3, passenger=WARRIOR)
    g = _load(d)
    assert _step(g, PROMOTE, t, 0) == 0
    e = _export(g)
    assert e['units.veteran'][t] and e['units.health'][t] == 15.


@pytest.mark.parametrize('killer,xp', [(RAMMERSHIP, 0), (WARRIOR, 1)])
def test_no_kill_xp_for_boats(killer, xp):
    """Attacks give no xp to vehicles (Carry) or Static units."""
    d = _state()
    _clear_units(d)
    t = _own_tile(d, WATER if killer == RAMMERSHIP else FIELD)
    target = next(x for x in _ring(t) if d['tiles.terrain'][0, x] == FIELD and d['tiles.structure'][0, x] == 0
                  and d['units.owner'][0, x] == 0)
    _unit(d, t, 1, killer, kills=1, **(dict(passenger=WARRIOR) if killer == RAMMERSHIP else {}))
    _unit(d, target, 2, WARRIOR, 1.)
    d['tiles.explored'][0, 0, target] = True
    g = _load(d)
    assert _step(g, ATTACK, t, target) == 0
    e = _export(g)
    assert e['units.owner'][target] != 2 and e['players.kills'][0] == 1
    at = target if e['units.owner'][target] == 1 else t
    assert int(e['units.kills'][at]) == 1 + xp


@pytest.mark.parametrize('unit,radius', [(CLOAK, 2), (WARRIOR, 1)])
def test_training_sight(unit, radius):
    """Training explores with the unit's sight range: Scout units (Cloak) see 2 tiles."""
    d = _state()
    cap = _capital(d, 1)
    d['tiles.terrain'][0, cap] = FIELD
    d['players.tech'][0, 0, 38] = True                     # Diplomacy: Cloak
    d['players.stars'][0, 0] = 50
    d['units.owner'][0, cap] = 0
    d['tiles.explored'][0, 0] = False
    g = _load(d)
    assert _step(g, SUMMON, cap, unit) == 0
    seen = {int(t) for t in np.flatnonzero(_export(g)['tiles.explored'][0])}
    assert seen == {t for t in range(SIZE * SIZE) if _dist(cap, t) <= radius}


def test_turn_start_explores_nothing():
    d = _state()
    d['current_player'][0] = 2
    d['tiles.explored'][0, 0] = False
    g = _load(d)
    assert _step(g, END, 0, 0) == 0
    assert int(_export(g)['tiles.explored'][0].sum()) == 0


def test_grow_forest_leaves_no_resource():
    d = _state()
    t = _own_tile(d, FIELD)
    d['tiles.resource'][0, t] = 2                          # crop
    d['players.tech'][0, 0, 19] = True                     # Spiritualism: grow forest
    d['players.stars'][0, 0] = 50
    g = _load(d)
    assert _step(g, FOREST, t, 2) == 0                     # forest family target 2: grow
    e = _export(g)
    assert e['tiles.terrain'][t] == FOREST_T and e['tiles.resource'][t] == 0


@pytest.mark.parametrize('unit,legal', [(CLOAK, False), (WARRIOR, True)])
def test_cloak_does_not_end_on_enemy_city(unit, legal):
    d = _state()
    _clear_units(d)
    theirs = _capital(d, 2)
    start = next(t for t in _ring(theirs) if d['tiles.terrain'][0, t] == FIELD)
    _unit(d, start, 1, unit, 5. if unit == CLOAK else 10.)
    d['tiles.explored'][0, 0, [start, theirs]] = True
    assert _legal(d, MOVE, start, theirs) == legal


@pytest.mark.parametrize('road_from', [False, True])
def test_forest_entered_off_road_ends_the_move(road_from):
    """Movement cost: a forest costs the whole move unless the step runs road to road."""
    d = _state()
    row = _strip(d)
    a, f, b = row + 3, row + 4, row + 5
    d['tiles.terrain'][0, f], d['tiles.road'][0, f] = FOREST_T, True       # a forest with a road
    d['tiles.terrain'][0, [f - SIZE, f + SIZE]] = WATER                    # no way around it
    d['tiles.road'][0, a] = road_from
    _unit(d, a, 1, RIDER)                                                  # 2 moves
    d['current_player'][0] = 1
    assert _legal(d, MOVE, a, b) == road_from


@pytest.mark.parametrize('unit', [CLOAK, WARRIOR])
def test_hide_units_pass_enemy_units(unit):
    """Passability: a Hide unit (Cloak) may pass a tile held by another player's unit; others may not."""
    d = _state()
    row = _strip(d)
    a, e, b = row + 3, row + 4, row + 5
    _unit(d, a, 1, unit, 5. if unit == CLOAK else 10.)
    _unit(d, e, 2, WARRIOR)
    d['current_player'][0] = 1
    d['tiles.terrain'][0, [e - SIZE, e + SIZE]] = WATER                    # no way around
    if unit == WARRIOR:
        d['units.kind'][0, a] = RIDER                                      # two moves to try
    assert _legal(d, MOVE, a, b) == (unit == CLOAK) and not _legal(d, MOVE, a, e)


def test_neutral_bridge():
    """Building: a Bridge is Freelance (neutral water like a road); its span comes from the true terrain."""
    d = _state()
    row = _strip(d)
    w = row + 6
    d['tiles.terrain'][0, w] = WATER
    d['players.tech'][0, 0, 4] = True                                      # Roads: bridge
    d['players.tech'][0, 0, 10] = True
    d['players.stars'][0, 0] = 50
    d['current_player'][0] = 1
    assert _legal(d, BUILD, w, 48)
    for t in (w - 1, w + 1, w - SIZE, w + SIZE):                           # water all around: no span
        d['tiles.terrain'][0, t] = WATER
    assert not _legal(d, BUILD, w, 48)


@pytest.mark.parametrize('stars,legal', [(37, False), (38, True)])
def test_literacy_price(stars, legal):
    """Tech price: ceil(price x 0.66666f): a 56-star tech costs 38 with Literacy."""
    d = _state()
    d['players.tech'][0, 0, 22] = True                                     # Philosophy: Literacy
    d['players.stars'][0, 0] = stars
    d['cities.owner'][0][d['cities.owner'][0] == 1] = 0
    d['cities.owner'][0, :26] = 1                                          # 26 cities: tier 2 costs 4 + 52 = 56
    parents = {4: 1, 7: 6, 8: 6, 13: 10, 16: 15, 18: 15, 21: 20, 23: 20}
    for k, parent in parents.items():
        d['players.tech'][0, 0, parent] = True
    tier2 = next(k for k in parents if not d['players.tech'][0, 0, k])
    assert _legal(d, RESEARCH, 0, tier2) == legal


def test_wiped_player_units_are_disbanded_with_refund():
    """Wiping a player out: the last city's capture disbands the loser's units, refunding half their cost."""
    d = _state()
    _clear_units(d)
    theirs = _capital(d, 2)
    d['cities.owner'][0][(d['cities.owner'][0] == 2) & (np.arange(SIZE * SIZE) != theirs)] = 0
    _unit(d, theirs, 1, WARRIOR)
    elsewhere = next(t for t in range(SIZE * SIZE) if _dist(t, theirs) > 3 and d['tiles.terrain'][0, t] == FIELD)
    _unit(d, elsewhere, 2, DEFENDER)                                       # cost 3
    d['current_player'][0] = 1
    before = int(d['players.stars'][0, 1])
    g = _load(d)
    assert _step(g, CAPTURE, theirs, 0) == 0
    e = _export(g)
    assert e['units.owner'][elsewhere] == 0 and int(e['players.stars'][1]) == before + 1


@pytest.mark.parametrize('second', [False, True])
def test_persist_needs_another_target(second):
    """After a kill, a Persist unit may attack again only with a target left in range."""
    d = _state()
    row = _strip(d)
    k, v, x = row + 3, row + 4, row + 5
    _unit(d, k, 1, KNIGHT)
    _unit(d, v, 2, WARRIOR, 1.)
    if second:
        _unit(d, x, 2, WARRIOR)
    d['current_player'][0] = 1
    g = _load(d)
    assert _step(g, ATTACK, k, v) == 0
    e = _export(g)
    at = v if e['units.owner'][v] == 1 else k
    assert bool(e['units.attacked'][at]) == (not second)


@pytest.mark.parametrize('peace', [True, False])
def test_peace_partner_units_can_be_passed(peace):
    """Passability: a unit may pass (never end on) a tile held by a peace partner's unit."""
    d = _state()
    row = _strip(d)
    a, p, b = row + 3, row + 4, row + 5
    _unit(d, a, 1, RIDER)
    _unit(d, p, 2, WARRIOR)
    d['tiles.terrain'][0, [p - SIZE, p + SIZE]] = WATER
    d['players.peace'][0, 0, 1] = d['players.peace'][0, 1, 0] = peace
    d['current_player'][0] = 1
    assert _legal(d, MOVE, a, b) == peace and not _legal(d, MOVE, a, p)


@pytest.mark.parametrize('explored', [False, True])
def test_unseen_enemies_exert_zone_of_control(explored):
    """Movement cost reads the raw neighbours: an enemy stops a move next to it, seen or not."""
    d = _state()
    row = _strip(d)
    a, m, b, e = row + 3, row + 4, row + 5, row + 4 - SIZE
    _unit(d, a, 1, RIDER)
    _unit(d, e, 2, WARRIOR)
    d['tiles.explored'][0, 0, e] = explored
    d['tiles.terrain'][0, m + SIZE] = WATER
    d['current_player'][0] = 1
    assert not _legal(d, MOVE, a, b) and _legal(d, MOVE, a, m)


# --- Explorer walk, ruins reward and the push of an occupant (docs/engine-parity-audit-2026-10-31.md items 2, 5, 8,
# 21). Small reference ports of the game's procedures; transition_parity.py checks the engine against the recorded
# games.
GIANT, SWORDSMAN = 12, 11
GRID = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))     # grid direction SW .. S as (dx, dy)
STEPS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))    # explorer candidates, in order
M32 = 0xFFFFFFFF


def _xxh(seed, *words):
    """The game's seeded hash (XXH32 of up to 3 ints)."""
    h = (seed + 374761393 + 4 * len(words)) & M32
    for w in words:
        h = (h + (w & M32) * 3266489917) & M32
        h = ((h << 17 | h >> 15) & M32) * 668265263 & M32
    h ^= h >> 15
    h = h * 2246822519 & M32
    h ^= h >> 13
    h = h * 3266489917 & M32
    return h ^ h >> 16


def _movable(tech, terrain):
    return terrain in (3, 5, 6, 7) or (terrain, True) in ((1, tech[10]), (2, tech[13]), (4, tech[20]))


def _fog_score(look, k, r):
    cx, cy = k % 11, k // 11
    cells = [look[y * 11 + x] for y in range(max(cy - r, 0), min(cy + r, 10) + 1)
             for x in range(max(cx - r, 0), min(cx + r, 10) + 1)]
    count, light = sum(f >> 3 & 1 for f in cells), any(f & 40 == 40 for f in cells)
    if r == 2 and not count and not light:
        return 9999
    return (40 if light else 73) - 10 * min(count, 4)


def _walk(e, p, source):
    """12 explorer steps over the 11x11 fog-distance map (no bridges): the tiles they newly explore."""
    seed0, terrain, structure, tech = int(e['initial_seed']), e['tiles.terrain'], e['tiles.structure'], e['players.tech'][p]
    explored = set(np.flatnonzero(e['tiles.explored'][p]).tolist())
    start = set(explored)
    fx, fy = source % SIZE, source // SIZE
    seed, path = _xxh(seed0, fx, fy), [source]
    for _ in range(12):
        look = [0] * 121
        for k in range(121):
            x, y = fx - 5 + k % 11, fy - 5 + k // 11
            if 0 <= x < SIZE and 0 <= y < SIZE:
                t, st = y * SIZE + x, structure[y * SIZE + x]
                seen = t in explored
                enter = st == 1 if st in (1, 48) else _movable(tech, terrain[t])
                look[k] = (1 | (16 if st == 48 else 0) | (6 if enter else 0) | (0 if seen else 8) | (32 if st == 47 else 0)
                           | (64 if terrain[t] == 4 and seen else 0))
        dist = [1000 if f & 2 else 9999 for f in look]
        queue = [k for k in range(121) if look[k] & 8]
        for k in queue:
            dist[k] = 0
        for k in range(121):
            if look[k] & 0x4a == 0x42 and _fog_score(look, k, 2) != 9999:
                dist[k] = _fog_score(look, k, 2) + 100
                queue.append(k)
                look[k] |= 128
        at = 0
        while at < len(queue):
            u = queue[at]
            right, left = (1 if u % 11 < 10 else 9999), (-1 if u % 11 else -9999)
            step = dist[u] + 100
            for i in (u + 11, u - 11, u + right, u + left, u + left + 11, u + right + 11, u + left - 11, u + right - 11):
                if 0 <= i < 121 and dist[i] > step and not look[i] & 128 and look[i] & 3 == 3:
                    dist[i] = _fog_score(look, i, 1) + 100 if step == 100 and not look[i] & 8 else step
                    queue.append(i)
            if at == 0x1f2:
                break
            at += 1
        long, penalty, count = len(path) > 8, 1, 0
        for t in path:
            lx, ly = t % SIZE - fx + 5, t // SIZE - fy + 5
            if 0 <= lx < 11 and 0 <= ly < 11:
                dist[ly * 11 + lx] += penalty
            penalty += (count & 1) if long else 1
            count += long
        values = [dist[(dy + 5) * 11 + dx + 5] for dx, dy in STEPS]
        if min(values) > 9998:
            break
        ties = [j for j, v in enumerate(values) if v == min(values)]
        dx, dy = STEPS[ties[_xxh(seed0, seed) % len(ties)]]
        fx, fy = fx + dx, fy + dy
        path.append(fy * SIZE + fx)
        r = 2 if terrain[fy * SIZE + fx] == 4 else 1
        explored |= {y * SIZE + x for y in range(fy - r, fy + r + 1) for x in range(fx - r, fx + r + 1)
                     if 0 <= x < SIZE and 0 <= y < SIZE}
        seed = _xxh(seed0, seed)
    return explored - start


def _ruin_order(seed, tile):
    """The ruins reward's shuffled [resources, population, Swordsman, Battleship, Explorer, free tech]."""
    order, x, y = [1, 2, 8, 6, 3, 4], tile % SIZE, tile // SIZE
    for i in range(5):
        j = _xxh(seed, x, y, i) % (6 - i) + i
        order[i], order[j] = order[j], order[i]
    return order


def _unlockable(tech):
    """The unlockable techs over rules.json `next` (the game's tech-unlock order), breadth-first from Basic."""
    import json
    from polytopia_bot.ruleset import rules_json
    nxt = {t['id']: t['next'] for t in json.loads(rules_json())['technologies'] if t['id'] >= 0}
    tree, queue = [], [0]
    while queue:
        k = queue.pop(0)
        if k not in tree:
            tree.append(k)
            queue += nxt[k]
    out = []
    for k in tree:
        if tech[k]:
            out += [c for c in nxt[k] if not tech[c] and c not in out]
        elif any(tech[c] for c in nxt[k]) and k not in out:
            out.append(k)
    return out


def _explorer_city(seed=11, initial_seed=None):
    """A state whose capital (level 2, no reward taken yet) can take the Explorer reward."""
    d = _state(seed)
    cap = _capital(d, 1)
    d['cities.level'][0, cap] = 2
    d['cities.rewards'][0, cap] = 0
    if initial_seed is not None:
        d['initial_seed'][0] = initial_seed
    return d, cap


@pytest.mark.parametrize('initial_seed', [7, 123456, 4000000000])
def test_explorer_walk_is_the_games(initial_seed):
    """The city's Explorer reward: 12 explorer steps seeded by the hash of (x, y), each to the neighbour with the
    lowest fog distance (11x11 window, mountains scored, backtracking penalty), ties by the step hash."""
    d, cap = _explorer_city(initial_seed=initial_seed)
    d['players.tech'][0, 0, 20] = True                 # Climbing: mountains are walkable
    before = {k: np.array(v)[0] for k, v in d.items()}
    g = _load(d)
    assert _step(g, REWARD, cap, 4) == 0
    e = _export(g)
    new = set(np.flatnonzero(e['tiles.explored'][0] & ~before['tiles.explored'][0]).tolist())
    assert new and new == _walk(before, 0, cap)


def test_explorer_walk_reaches_fog_five_tiles_away():
    """The walk sees fog anywhere in its 11x11 window: with everything explored but a column 5 tiles east, it walks
    there (the old engine's 3-round spread random-walked)."""
    d, cap = _explorer_city()
    cx, cy = cap % SIZE, cap // SIZE
    fog_x = cx + 5 if cx + 5 < SIZE else cx - 5
    d['tiles.explored'][0, 0] = True
    for t in range(SIZE * SIZE):
        if t % SIZE == fog_x:
            d['tiles.explored'][0, 0, t] = False
        if abs(t % SIZE - cx) <= 5 and abs(t // SIZE - cy) <= 2:
            d['tiles.terrain'][0, t], d['tiles.structure'][0, t] = FIELD, 0
    d['tiles.structure'][0, cap] = 1
    g = _load(d)
    assert _step(g, REWARD, cap, 4) == 0
    e = _export(g)
    assert (e['tiles.explored'][0].reshape(SIZE, SIZE)[:, fog_x]).any()


def _ruin_state(order_first, seed=11, tech_override=None):
    """A field ruin next to the capital with an idle warrior on it, everything explored (no Explorer reward), and an
    initial_seed whose shuffle makes `order_first` the first valid reward."""
    d = _state(seed)
    _clear_units(d)
    cap = _capital(d, 1)
    ruin = next(t for t in _ring(cap, 2) if d['tiles.structure'][0, t] == 0 and 0 < t % SIZE < SIZE - 1
                and 0 < t // SIZE < SIZE - 1)
    for t in _ring(ruin) + [ruin]:
        d['tiles.terrain'][0, t] = FIELD
        if t != ruin and d['tiles.structure'][0, t] != 1:
            d['tiles.structure'][0, t] = 0
    d['tiles.structure'][0, ruin] = 2
    d['tiles.explored'][0, 0] = True
    d['players.known'][0, 0, 1] = True        # already met: seeing their city pays no gift
    if tech_override is not None:
        d['players.tech'][0, 0] = tech_override
    tech = d['players.tech'][0, 0]
    valid = {1, 2, 8} | ({4} if _unlockable(tech) else set())
    s = next(s for s in range(1, 100000) if next(r for r in _ruin_order(s, ruin) if r in valid) == order_first)
    d['initial_seed'][0] = s
    _unit(d, ruin, 1, WARRIOR, direction=3)
    d['units.home'][0, ruin] = cap + 1
    return d, ruin, cap


@pytest.mark.parametrize('reward', [1, 2, 4, 8])
def test_ruins_reward_is_the_tile_hash(reward):
    """The ruins reward: a hash-shuffled list of rewards, the first valid one wins; the old per-step seed chain
    (State.seed) plays no part."""
    d, ruin, cap = _ruin_state(reward)
    for chain in (0, 99):
        d['seed'][0] = chain
        before = {k: np.array(v)[0] for k, v in d.items()}
        g = _load(d)
        assert _step(g, CAPTURE, ruin, 0) == 0
        e = _export(g)
        assert e['tiles.structure'][ruin] == 0
        if reward == 1:
            assert e['players.stars'][0] == before['players.stars'][0] + 10
        if reward == 2:
            assert (e['cities.population'][cap], e['cities.level'][cap]) != (before['cities.population'][cap],
                                                                              before['cities.level'][cap])
            assert e['players.stars'][0] == before['players.stars'][0]
        if reward == 4:
            gained = np.flatnonzero(e['players.tech'][0] & ~before['players.tech'][0])
            lst = _unlockable(before['players.tech'][0])
            assert gained.tolist() == [lst[_xxh(int(d['initial_seed'][0]), ruin % SIZE, ruin // SIZE) % len(lst)]]
        if reward == 8:
            assert e['units.kind'][ruin] == SWORDSMAN and e['units.veteran'][ruin] and e['units.health'][ruin] == 20.
            assert e['units.home'][ruin] == 0
            # the examining warrior was pushed along its facing (N = +y)
            assert e['units.kind'][ruin + SIZE] == WARRIOR and e['units.owner'][ruin + SIZE] == 1


@pytest.mark.parametrize('owned_ramming_only', [True, False])
def test_navigation_follows_sailing(owned_ramming_only):
    """Navigation's parent is Sailing (the game's tech unlocks), not Ramming: with every Bardur tech but Sailing and
    Navigation, the free tech is Sailing; with every one but Navigation, it is Navigation."""
    tech = np.zeros(128, bool)
    for k in (*range(25), 38, 44):
        tech[k] = k != 11
    tech[14] = False
    tech[13] = not owned_ramming_only
    assert _unlockable(tech) == ([13] if owned_ramming_only else [14])
    d, ruin, _ = _ruin_state(4, tech_override=tech)
    g = _load(d)
    assert _step(g, CAPTURE, ruin, 0) == 0
    e = _export(g)
    assert bool(e['players.tech'][0, 13]) and bool(e['players.tech'][0, 14]) == (not owned_ramming_only)


def _giant_city(occupant_owner, facing, blocked=()):
    """The capital at level 5 taking the Giant, with a unit on it facing `facing`; all neighbours open fields."""
    d = _state()
    _clear_units(d)
    cap = _capital(d, 1)
    d['cities.level'][0, cap] = 5
    d['cities.rewards'][0, cap] = 0
    d['cities.rewards'][0, cap, 3] = 1
    d['cities.rewards'][0, cap, 7] = 1
    d['cities.rewards'][0, cap, 5] = 1
    for t in _ring(cap):
        d['tiles.terrain'][0, t], d['tiles.structure'][0, t] = FIELD, 0
        d['tiles.explored'][0, :, t] = True
    _unit(d, cap, occupant_owner, WARRIOR, direction=facing)
    for k in blocked:
        dx, dy = GRID[k]
        _unit(d, cap + dx + dy * SIZE, 2, WARRIOR)
    return d, cap


@pytest.mark.parametrize('owner,facing,blocked,expect', [
    (1, 5, (), 5),            # own unit: its facing (E)
    (1, 5, (5,), 4),          # E taken: -1 (NE)
    (1, 5, (5, 4), 6),        # then +1 (SE)
    (1, 5, (5, 4, 6), 3),     # then -2 (N)
    (2, 5, (), 1),            # another player's unit: the opposite (W)
    (1, 0, (0,), 7),          # wraps: SW taken, -1 is S
])
def test_push_follows_the_occupant_facing(owner, facing, blocked, expect):
    """Pushing a unit: deterministic, from the occupant's stored facing (reversed for another player's
    unit), trying offsets 0, -1, +1, -2, +2, -3, +3, -4; a push keeps moved/attacked."""
    d, cap = _giant_city(owner, facing, blocked)
    cx, cy = cap % SIZE, cap // SIZE
    if not all(0 <= cx + dx < SIZE and 0 <= cy + dy < SIZE for dx, dy in GRID):
        pytest.skip('capital on the edge')
    g = _load(d)
    assert _step(g, REWARD, cap, 6) == 0
    e = _export(g)
    dx, dy = GRID[expect]
    to = cap + dx + dy * SIZE
    assert e['units.kind'][cap] == GIANT and e['units.kind'][to] == WARRIOR and e['units.owner'][to] == owner
    assert not e['units.moved'][to] and e['units.direction'][to] == expect


def test_push_needs_an_explored_tile():
    """The push's route only uses tiles the occupant's owner has explored."""
    d, cap = _giant_city(1, 5)
    dx, dy = GRID[5]
    d['tiles.explored'][0, 0, cap + dx + dy * SIZE] = False
    g = _load(d)
    assert _step(g, REWARD, cap, 6) == 0
    e = _export(g)
    dx, dy = GRID[4]
    assert e['units.kind'][cap + dx + dy * SIZE] == WARRIOR


def test_units_face_the_centre_then_their_fights():
    """A new unit faces the map centre; an attack turns attacker and defender toward each other."""
    from polytopia_bot.gpu.importer import spawn_direction
    d = _state()
    row = _strip(d)
    a, b = row + 2, row + 3
    _unit(d, a, 1, ARCHER, direction=0)
    _unit(d, b, 2, DEFENDER, direction=0)
    g = _load(d)
    assert _step(g, ATTACK, a, b) == 0
    e = _export(g)
    assert e['units.direction'][a] == 5 and e['units.direction'][b] == 1          # E and W
    d = _state()
    cap = _capital(d, 1)
    d['units.owner'][0, cap] = 0
    d['players.stars'][0, 0] = 50
    g = _load(d)
    assert _step(g, SUMMON, cap, WARRIOR) == 0
    assert _export(g)['units.direction'][cap] == spawn_direction(SIZE, cap % SIZE, cap // SIZE)
