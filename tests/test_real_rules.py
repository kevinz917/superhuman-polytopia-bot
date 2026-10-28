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
    e['units.owner'][0][e['units.owner'][0] != e['current_player'][0]] = 0
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
