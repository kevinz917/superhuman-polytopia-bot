"""Unit rules of the real game (Polytopia 2.17.3, version 123) on engine-rs: facing, Dagger spawns, reveals
(docs/rules-parity-2-summary.md). Each test fails on the engine before that pass, except the regression checks marked
as such, whose behaviour already matched."""
import numpy as np
import pytest

from test_real_rules import (ATTACK, CAPTURE, CLOAK, DEFENDER, ARCHER, END, FIELD, GRID, MOVE, REWARD, SIZE, WARRIOR,
                             _capital, _clear_units, _export, _giant_city, _load, _ring, _ruin_state, _state, _step,
                             _strip, _unit, _xxh)

DAGGER, PIRATE, MINDBENDER, BOMBER, GIANT, SWORDSMAN = 37, 40, 10, 41, 12, 11
FARM, FRUIT = 5, 6


def _vector_direction(vx, vy):
    """direction from distance vector."""
    best, score = 0, 0
    for d, (x, y) in enumerate(GRID):
        v = (y * vy + x * vx) * (1393 if x * x + y * y == 1 else 985)
        if score < v:
            best, score = d, v
    return best


def _dir_xy(f, t):
    """direction on coordinates (8: the same tile)."""
    dx, dy = t[0] - f[0], t[1] - f[1]
    if max(abs(dx), abs(dy)) > 1:
        return _vector_direction(dx, dy)
    return GRID.index((dx, dy)) if (dx, dy) in GRID else 8


def _dir(a, b, size=SIZE):
    return _dir_xy((a % size, a // size), (b % size, b // size))


def _centre(t, size=SIZE):
    return _vector_direction(size // 2 - t % size, size // 2 - t // size)


def _noise(seed, t, turn):
    """infiltration reward Evaluate's tie noise: Range(0f, 0.1f, random hash.hash(x, y, current turn))."""
    h = _xxh(seed, _xxh(seed, t % SIZE, t // SIZE, turn))
    return np.float32(np.float32(h) * np.float32(0.1)) / np.float32(4294967296.0)


def _infiltration(level=3, defender=DEFENDER, defender_facing=None):
    """Player 1's Cloak next to player 2's capital (level `level`, `defender` on it unless None); all explored."""
    d = _state()
    _clear_units(d)
    d['current_player'][0] = 1
    d['players.known'][0] = True
    d['tiles.explored'][0] = True
    city = _capital(d, 2)
    d['cities.level'][0, city] = level
    d['cities.riot'][0, city] = False
    for t in np.flatnonzero(d['tiles.city'][0] == city):
        if t != city:
            d['tiles.terrain'][0, t], d['tiles.resource'][0, t], d['tiles.structure'][0, t] = FIELD, 0, 0
    cloak = next(t for t in _ring(city) if d['tiles.city'][0, t] == city and d['tiles.structure'][0, t] == 0)
    _unit(d, cloak, 1, CLOAK, 5.)
    if defender is not None:
        _unit(d, city, 2, defender, 15. if defender == DEFENDER else 10.,
              direction=defender_facing if defender_facing is not None else 0)
    return d, city, cloak


def _daggers(e, owner=1):
    return [int(t) for t in np.flatnonzero(((e['units.kind'] == DAGGER) | (e['units.kind'] == PIRATE))
                                           & (e['units.owner'] == owner))]


def test_daggers_face_the_infiltrated_city():
    """train: each spawn
    turns toward the city, not toward the map centre."""
    d, city, cloak = _infiltration()
    g = _load(d)
    assert _step(g, ATTACK, cloak, city) == 0
    e = _export(g)
    spawns = _daggers(e)
    assert len(spawns) == 3
    assert all(e['units.direction'][t] == _dir(t, city) for t in spawns)
    assert any(_dir(t, city) != _centre(t) for t in spawns)       # the old facing would differ


def test_city_unit_surviving_infiltration_faces_the_null_origin():
    """perform attack: the city's unit turns toward
    (-1, -1) (perform attack default,."""
    d, city, cloak = _infiltration()
    want = _dir_xy((city % SIZE, city // SIZE), (-1, -1))
    d['units.direction'][0, city] = (want + 3) % 8
    g = _load(d)
    assert _step(g, ATTACK, cloak, city) == 0
    e = _export(g)
    assert e['units.owner'][city] == 2 and e['units.direction'][city] == want


def test_dagger_spawn_ignores_resources_under_improvements():
    """resource returns nothing on a tile with an improvement: a fruit under a
    farm earns no +0.5. Level 1 with a defender that survives: one Dagger, on the best non-city tile."""
    d, city, cloak = _infiltration(level=1)
    land = [int(t) for t in np.flatnonzero(d['tiles.city'][0] == city) if t != city]
    seed, turn = int(d['initial_seed'][0]), int(d['turn'][0])
    best = max(land, key=lambda t: _noise(seed, t, turn))         # the cloak's tile is free by then
    farm = next(t for t in land if t != best)
    d['tiles.resource'][0, farm], d['tiles.structure'][0, farm] = FRUIT, FARM
    g = _load(d)
    assert _step(g, ATTACK, cloak, city) == 0
    assert _daggers(_export(g)) == [best]                         # the old engine put it on the farm


def test_bump_reveal_turns_the_mover():
    """move onto a hidden enemy: reveal
    turns the mover toward the revealed unit; it stays and keeps its move."""
    d = _state()
    row = _strip(d)
    a, b = row + 2, row + 3
    d['current_player'][0] = 1
    d['players.known'][0] = True
    _unit(d, a, 1, WARRIOR, direction=1)
    _unit(d, b, 2, CLOAK, 5., invisible=True)
    g = _load(d)
    assert _step(g, MOVE, a, b) == 0
    e = _export(g)
    assert e['units.owner'][a] == 1 and not e['units.invisible'][b]
    assert e['units.direction'][a] == 5                          # E, toward b (was W)


@pytest.mark.parametrize('owner,peace,revealed', [(2, False, True), (1, False, False), (2, True, False)])
def test_hidden_unit_pushed_for_a_trained_unit_is_revealed(owner, peace, revealed):
    """train unit on occupied space (version > 105,: an occupant invisible to the
    trainer (another player's, no peace: is invisible) gets a reveal before the push."""
    d, cap = _giant_city(owner, 5)
    cx, cy = cap % SIZE, cap // SIZE
    if not all(0 <= cx + dx < SIZE and 0 <= cy + dy < SIZE for dx, dy in GRID):
        pytest.skip('capital on the edge')
    d['units.invisible'][0, cap] = True
    d['players.peace'][0, 0, 1] = d['players.peace'][0, 1, 0] = peace
    g = _load(d)
    assert _step(g, REWARD, cap, 6) == 0
    e = _export(g)
    to = next(t for t in _ring(cap) if e['units.kind'][t] == WARRIOR and e['units.owner'][t] == owner)
    assert e['units.kind'][cap] == GIANT and bool(e['units.invisible'][to]) == (not revealed)


def test_convert_turns_both_units():
    """attack pushes the 0-damage main attack after the convert): it
    runs first and turns converter and target toward each other."""
    d = _state()
    row = _strip(d)
    a, b = row + 2, row + 3
    d['current_player'][0] = 1
    d['players.known'][0] = True
    _unit(d, a, 1, MINDBENDER, direction=0)
    _unit(d, b, 2, WARRIOR, direction=0)
    g = _load(d)
    assert _step(g, ATTACK, a, b) == 0
    e = _export(g)
    assert e['units.owner'][b] == 1
    assert e['units.direction'][a] == 5 and e['units.direction'][b] == 1     # E and W


@pytest.mark.parametrize('defender,retaliates', [(WARRIOR, False), (ARCHER, True)])
def test_splash_turns_targets_toward_the_attacker(defender, retaliates):
    """Splash attack actions run from the attacker's tile: each splashed unit faces the
    attacker and the attacker ends facing the first one in tile neighbors order (the stack is LIFO), unless a
    retaliation (pushed first, runs after them."""
    d = _state()
    row = _strip(d)
    a, b = row + 2, row + 4
    d['current_player'][0] = 1
    d['players.known'][0] = True
    _unit(d, a, 1, BOMBER, direction=0)
    if defender == ARCHER:
        _unit(d, b, 2, ARCHER, 15., veteran=True, direction=0)      # survives the hit, retaliates for >= 1
    else:
        _unit(d, b, 2, defender, direction=0)
    north, east = b + SIZE, b + 1                                   # neighbour order: (0, 1) before (1, 0)
    _unit(d, north, 2, DEFENDER, 15., direction=0)
    _unit(d, east, 2, DEFENDER, 15., direction=0)
    g = _load(d)
    assert _step(g, ATTACK, a, b) == 0
    e = _export(g)
    assert e['units.owner'][b] == 2 and (e['units.health'][a] < 10.) == retaliates
    assert e['units.direction'][north] == _dir(north, a) and e['units.direction'][east] == _dir(east, a)
    assert e['units.direction'][b] == _dir(b, a)
    assert e['units.direction'][a] == (_dir(a, b) if retaliates else _dir(a, north))


def test_riot_lasts_until_the_owner_ends_a_turn_four_seats():
    """Regression (already matched): improvement effect robbed (1) is removed only by end turn for tiles the
    ending player owns, whatever the number of seats."""
    from polytopia_bot.rl.rust_engine import RustGames
    size = 18
    g = RustGames(size, 1, players=4)
    g.batch.reset(np.array([3], np.int32))
    d = {k: np.array(v) for k, v in g.batch.export().items()}
    for k in [k for k in d if k.startswith('units.')]:
        d[k][0] = 0
    d['current_player'][0] = 1
    d['players.known'][0] = True
    d['tiles.explored'][0] = True
    city = int(np.flatnonzero((d['tiles.capital'][0] == 3) & (d['cities.owner'][0] == 3))[0])
    ring = [t for t in range(size * size) if max(abs(t % size - city % size), abs(t // size - city // size)) == 1
            and d['tiles.structure'][0, t] == 0]
    d['units.owner'][0, ring[0]], d['units.kind'][0, ring[0]], d['units.health'][0, ring[0]] = 1, CLOAK, 5.
    g.batch.import_({k: np.ascontiguousarray(v) for k, v in d.items()})
    i = lambda v: np.array([v], np.int32)
    assert int(g.batch.step(i(ATTACK), i(ring[0]), i(city))[0]) == 0
    riot = []
    for _ in range(4):
        e = {k: np.array(v)[0] for k, v in g.batch.export().items()}
        riot.append((int(e['current_player']), bool(e['cities.riot'][city])))
        assert int(g.batch.step(i(END), i(0), i(0))[0]) == 0
    assert riot == [(1, True), (2, True), (3, True), (4, False)]


def test_ruins_never_give_a_giant():
    """Regression (already matched): ruins reward default's list is [Resources, population growth, Swordsman,
    Battleship (6), Explorer, free tech]; super unit (5) is not in it."""
    d, ruin, _ = _ruin_state(8)
    for seed in range(1, 41):
        d['initial_seed'][0] = seed
        g = _load(d)
        assert _step(g, CAPTURE, ruin, 0) == 0
        e = _export(g)
        assert e['units.kind'][ruin] in (0, WARRIOR, SWORDSMAN) and GIANT not in e['units.kind']
