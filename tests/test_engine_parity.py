"""Rules aligned with the game in the 2026-10-31 parity pass (docs/engine-parity-audit-2026-10-31.md).

Each test puts one rule on a hand-edited position (helpers from test_real_rules), with a control where it makes sense.
"""
import numpy as np
import pytest

from test_real_rules import (ATTACK, BUILD, CLOAK, DISBAND, END, FIELD, MOVE, RESEARCH, REWARD, SIZE, WARRIOR, WATER,
                             RAFT, _capital, _clear_units, _duel, _export, _game, _legal, _load, _own_tile, _ring,
                             _state, _step, _strip, _unit)

DIPLOMATIC = 14
BOMBER, RIDER = 41, 3


def _masks(d):
    from polytopia_bot.rl.rust_engine import split_masks
    return [np.asarray(m)[0] for m in split_masks(_load(d).batch.legal_masks(), SIZE)]


def test_landing_ends_the_turn():
    """Disembarking: the passenger lands moved and attacked, whatever its Dash."""
    d = _state()
    row = _strip(d)
    sea, shore, enemy = row + 3, row + 4, row + 5
    d['tiles.terrain'][0, sea] = WATER
    _unit(d, sea, 1, RAFT, passenger=WARRIOR)
    _unit(d, enemy, 2, WARRIOR)
    d['current_player'][0] = 1
    g = _load(d)
    assert _step(g, MOVE, sea, shore) == 0
    e = _export(g)
    assert e['units.kind'][shore] == WARRIOR and e['units.attacked'][shore] and e['units.moved'][shore]
    m = [np.asarray(x)[0] for x in __import__('polytopia_bot.rl.rust_engine', fromlist=['split_masks']).split_masks(
        g.batch.legal_masks(), SIZE)]
    assert not m[ATTACK][shore].any()


def test_land_move_keeps_dash():
    d = _state()
    row = _strip(d)
    a, b, enemy = row + 3, row + 4, row + 5
    _unit(d, a, 1, WARRIOR)
    _unit(d, enemy, 2, WARRIOR)
    d['current_player'][0] = 1
    g = _load(d)
    assert _step(g, MOVE, a, b) == 0
    assert not _export(g)['units.attacked'][b]


def test_damage_rounds_like_the_game():
    """Battle results truncate the forces in integer units: a veteran Warrior at 1/15 HP deals 1 to a
    veteran Warrior at 5/15 (f32 arithmetic gave 2)."""
    d, a, b = _duel(WARRIOR, WARRIOR, 1, attacker_health=1., health=5.)
    d['units.veteran'][0, a] = d['units.veteran'][0, b] = True
    g = _load(d)
    assert _step(g, ATTACK, a, b) == 0
    assert _export(g)['units.health'][b] == pytest.approx(4.)


def test_disband_refunds_boat_and_passenger():
    """Disbanding: half of the full cost (Bomber 15 + Warrior 2)."""
    d = _state()
    row = _strip(d, WATER)
    t = row + 4
    _unit(d, t, 1, BOMBER, passenger=WARRIOR)
    d['current_player'][0] = 1
    d['players.tech'][0, 0, 2] = True       # Free Spirit: disband
    before = int(d['players.stars'][0, 0])
    g = _load(d)
    assert _step(g, DISBAND, t, 0) == 0
    assert int(_export(g)['players.stars'][0]) == before + 8


def _diplomacy(d):
    d['current_player'][0] = 1
    d['players.known'][0, 0, 1] = d['players.known'][0, 1, 0] = True
    return d


@pytest.mark.parametrize('stars,legal', [(4, True), (3, False)])
def test_embassy_costs_four(stars, legal):
    """Embassy cost: 4 + 2 per embassy already held."""
    d = _diplomacy(_state())
    d['players.tech'][0, 0, 38] = True
    d['players.stars'][0, 0] = stars
    assert bool(_masks(d)[DIPLOMATIC][3]) == legal
    if legal:
        g = _load(d)
        assert _step(g, DIPLOMATIC, 0, 3) == 0
        assert int(_export(g)['players.stars'][0]) == 0


def test_peace_request_must_be_answered():
    """A pending peace request is a command trigger: only accept, decline (and resign) are legal."""
    d = _diplomacy(_state())
    d['players.peace_offer'][0, 0, 1] = True
    m = _masks(d)
    assert m[DIPLOMATIC][1] and m[DIPLOMATIC][2]
    assert not m[END] and not m[MOVE].any() and not m[RESEARCH].any()
    g = _load(d)
    assert _step(g, END, 0, 0) != 0
    assert _step(g, DIPLOMATIC, 0, 2) == 0           # accept
    e = _export(g)
    assert e['players.peace'][0, 1] and not e['players.peace_offer'][0, 1]


def test_attack_keeps_peace_requests():
    """An attack is stamped on the relations and destroys embassies; it does not touch requests."""
    d, a, b = _duel(WARRIOR, WARRIOR, 1)
    _diplomacy(d)
    d['players.peace_offer'][0, 1, 0] = True          # player 2 has a request from player 1
    g = _load(d)
    assert _step(g, ATTACK, a, b) == 0
    assert _export(g)['players.peace_offer'][1, 0]


def test_meeting_by_seeing_a_city():
    """Contact on exploring: an owned city in view meets its owner, no unit needed."""
    d = _state()
    _clear_units(d)
    d['current_player'][0] = 1
    d['players.known'][0, 0, 1] = False
    theirs = _capital(d, 2)
    d['tiles.explored'][0, 0, theirs] = True
    d['players.stars'][0, 0] = 100
    tech = next(k for k in (1, 6, 10, 15, 20) if not d['players.tech'][0, 0, k])
    g = _load(d)
    assert _step(g, RESEARCH, 0, tech) == 0
    assert _export(g)['players.known'][0, 1]


def test_moving_onto_a_hidden_unit_reveals_it_and_stays():
    """Moving onto a hidden unit reveals it; the mover keeps its tile and its move."""
    d = _state()
    row = _strip(d)
    a, b = row + 3, row + 4
    _unit(d, a, 1, WARRIOR)
    _unit(d, b, 2, CLOAK, 5., invisible=True)
    d['current_player'][0] = 1
    g = _load(d)
    assert _step(g, MOVE, a, b) == 0
    e = _export(g)
    assert e['units.owner'][a] == 1 and not e['units.moved'][a]
    assert e['units.owner'][b] == 2 and not e['units.invisible'][b]


def test_move_path_is_the_games_astar():
    """The game's route: a straight two-tile move goes straight (Dijkstra's tie order went diagonally), so the row
    two above the start stays unexplored."""
    d = _state()
    row = _strip(d)
    y = row // SIZE
    a, b = row + 6, row + 4
    for t in range(SIZE * SIZE):
        d['tiles.explored'][0, 0, t] = abs(t // SIZE - y) <= 1
    _unit(d, a, 1, RIDER)
    d['current_player'][0] = 1
    g = _load(d)
    assert _step(g, MOVE, a, b) == 0
    e = _export(g)
    assert not e['tiles.explored'][0, (y - 2) * SIZE + 5]


def test_off_turn_level_up_waits_for_the_owner():
    """A population point levels a city only on its owner's turn; the turn start levels it after income."""
    d = _state()
    theirs = _capital(d, 2)
    d['current_player'][0] = 1
    d['turn'][0] = 3
    d['cities.progress'][0, theirs] = 2               # enough for level 2, raised off-turn
    stars = int(d['players.stars'][0, 1])
    g = _load(d)
    assert _step(g, END, 0, 0) == 0
    e = _export(g)
    assert e['cities.level'][theirs] == 2
    income = int(e['players.stars'][1]) - stars
    assert income == int(d['cities.production'][0, theirs]) + 1       # the level-1 income (capital +1)


def test_four_seat_building_population():
    """Every seat's farms feed its city (seats 3 and 4 used to get nothing)."""
    g = _game(players=4, size=18)
    d = {k: np.array(v) for k, v in g.batch.export().items()}
    seat = 3
    d['current_player'][0] = seat
    cap = int(np.flatnonzero((d['tiles.capital'][0] == seat) & (d['cities.owner'][0] == seat))[0])
    t = next(t for t in np.flatnonzero(d['cities.territory'][0, cap]) if t != cap and d['units.owner'][0, t] == 0)
    d['tiles.terrain'][0, t], d['tiles.resource'][0, t], d['tiles.structure'][0, t] = FIELD, 2, 0
    d['tiles.owner'][0, t] = seat
    d['players.tech'][0, seat - 1, 8] = True          # Farming
    d['players.stars'][0, seat - 1] = 20
    before = int(d['cities.progress'][0, cap]) + 10 * int(d['cities.level'][0, cap])
    g.batch.import_({k: np.ascontiguousarray(v) for k, v in d.items()})
    i = lambda v: np.array([v], np.int32)
    assert int(g.batch.step(i(BUILD), i(t), i(5))[0]) == 0
    e = {k: np.array(v)[0] for k, v in g.batch.export().items()}
    assert int(e['cities.progress'][cap]) + 10 * int(e['cities.level'][cap]) > before
