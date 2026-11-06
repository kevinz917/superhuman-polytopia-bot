"""City connections, port routes and market levels as the real game computes them (docs/rules-parity-2-summary.md).

Each test builds a small position next to player 1's capital on an exported state, applies one action on engine-rs and
checks the outcome the game gives; the comment on each says what the previous engine did instead.
"""
import json

import numpy as np
import pytest

SIZE = 14
MOVE, BUILD = 0, 3
END = 15
WATER, FIELD = 1, 3
WARRIOR = 2
ROADS, FISHING = 4, 10
ROAD, PORT, FORGE, MINE, MARKET = 3, 8, 22, 21, 50


def _game(seed):
    from polytopia_bot.rl.rust_engine import RustGames
    g = RustGames(SIZE, 1)
    g.batch.reset(np.array([seed], np.int32))
    return g


def _state(seed):
    return {k: np.array(v) for k, v in _game(seed).batch.export().items()}


def _load(d, routes=True):
    """Import `d`; routes=False drops tiles.route, so the engine computes the routes of the position on import."""
    g = _game(11)
    g.batch.import_({k: np.ascontiguousarray(v) for k, v in d.items() if routes or k != 'tiles.route'})
    return g


def _step(g, family, source, target):
    i = lambda v: np.array([v], np.int32)
    return int(g.batch.step(i(family), i(source), i(target))[0])


def _export(g):
    return {k: np.array(v)[0] for k, v in g.batch.export().items()}


def _capital(d, player):
    return int(np.flatnonzero((d['tiles.capital'][0] == player) & (d['cities.owner'][0] == player))[0])


class Area:
    """Tiles (k, j) from player 1's capital: k steps along its row away from the near edge, j rows away from the
    near edge (down=True: j rows down, i.e. later in tile order). Every tile in k = 0..width, j = -1..rows except the
    capital is cleared to explored, ownerless field."""

    def __init__(self, width, rows, down=False, dx=None):
        for seed in range(11, 80):
            d = _state(seed)
            cap = _capital(d, 1)
            cx, cy = cap % SIZE, cap // SIZE
            self.dx = dx or (1 if cx <= SIZE // 2 else -1)
            self.dy = 1 if down or cy <= SIZE // 2 else -1
            self.cx, self.cy, self.d, self.cap = cx, cy, d, cap
            xs = [cx + k * self.dx for k in range(width + 1)]
            ys = [cy + j * self.dy for j in range(-1, rows + 1)]
            if not all(0 <= v < SIZE for v in xs + ys):
                continue
            area = [y * SIZE + x for x in xs for y in ys if y * SIZE + x != cap]
            if _capital(d, 2) in area:
                continue
            self.area = area
            for k in [k for k in d if k.startswith('units.')]:
                d[k][0] = 0
            for t in area:
                d['tiles.terrain'][0, t], d['tiles.structure'][0, t], d['tiles.structure_level'][0, t] = FIELD, 0, 0
                d['tiles.road'][0, t], d['tiles.resource'][0, t], d['tiles.owner'][0, t] = False, 0, 0
                d['tiles.capital'][0, t], d['tiles.city'][0, t], d['tiles.route'][0, t] = 0, -1, False
                d['cities.owner'][0, t], d['cities.connected'][0, t] = 0, False
                d['cities.territory'][0, :, t] = False
                d['tiles.explored'][0, :, t] = True
            d['cities.territory'][0, cap, cap] = True
            d['players.stars'][0] = 100
            d['current_player'][0] = 1
            d['turn'][0] = 5
            return
        raise AssertionError('no seed with room next to the capital')

    def t(self, k, j=0):
        return (self.cy + j * self.dy) * SIZE + self.cx + k * self.dx

    def water(self, *tiles, owner=0):
        for t in tiles:
            self.d['tiles.terrain'][0, t], self.d['tiles.owner'][0, t] = WATER, owner

    def port(self, t, owner=1, city=None):
        self.water(t, owner=owner)
        self.d['tiles.structure'][0, t] = PORT
        if city is not None:
            self.rule(city, t)

    def road(self, t, owner=0):
        self.d['tiles.road'][0, t], self.d['tiles.owner'][0, t] = True, owner

    def city(self, t, owner=1, village=False):
        """A level-1 city of `owner` (or a neutral village) on field tile t, ruling only itself."""
        d = self.d
        d['tiles.structure'][0, t], d['tiles.owner'][0, t] = 1, (0 if village else owner)
        if village:
            return
        d['cities.owner'][0, t], d['cities.level'][0, t], d['cities.population'][0, t] = owner, 1, 0
        d['cities.progress'][0, t], d['cities.production'][0, t], d['cities.border'][0, t] = 0, 1, 1
        d['cities.rewards'][0, t] = 0
        d['cities.order'][0, t] = int(d['cities.order'][0][d['cities.owner'][0] == owner].max()) + 1
        self.rule(t, t)

    def rule(self, city, t):
        self.d['tiles.city'][0, t] = city
        self.d['tiles.owner'][0, t] = self.d['cities.owner'][0, city]
        self.d['cities.territory'][0, city, t] = True

    def tech(self, player, *techs):
        for k in techs:
            self.d['players.tech'][0, player - 1, k] = True


# --- 1. City connections: find connected cities over matching transport path -------------

@pytest.mark.parametrize('village', [True, False])
def test_neutral_village_links_a_road_chain(village):
    """is connectable counts every City improvement, a neutral village too: capital - own road - village - new road -
    city connects. The previous engine made only owned cities nodes, so the village broke the chain."""
    a = Area(5, 1)
    a.road(a.t(1), owner=1)
    if village:
        a.city(a.t(2), village=True)
    c2 = a.t(4)
    a.city(c2)
    a.tech(1, ROADS)
    g = _load(a.d)
    assert _step(g, BUILD, a.t(3), ROAD) == 0
    e = _export(g)
    assert bool(e['cities.connected'][c2]) == village
    assert int(e['cities.population'][c2]) == int(village)


@pytest.mark.parametrize('from_own_tile', [False, True])
def test_road_through_another_players_land(from_own_tile):
    """has matching transport path checks ownership only between consecutive tiles, and a step from a neutral tile is
    never refused (try get player(0) fails): own road - neutral road - player 2's road - neutral road - city connects.
    Straight from an own tile into player 2's land (not at peace) is refused. The previous engine refused every tile
    of player 2."""
    a = Area(6, 1)
    a.road(a.t(1), owner=1)
    a.road(a.t(2), owner=2 if from_own_tile else 0)
    a.road(a.t(3), owner=2)
    c2 = a.t(5)
    a.city(c2)
    a.tech(1, ROADS)
    g = _load(a.d)
    assert _step(g, BUILD, a.t(4), ROAD) == 0
    assert bool(_export(g)['cities.connected'][c2]) == (not from_own_tile)


# --- 2. Port routes: update routes / find path between routers, path finder router settings -----------

@pytest.mark.parametrize('far', [False, True])
def test_port_route_pays_5_on_earlier_route_tiles(far):
    """movement cost (no unit, should follow transport paths): 5 onto a tile an earlier path of the same update
    marked has route. P0-P1 is routed first; P1-P2 then needs 6 steps, 2 of them on that route (5+10+10+10+10 < 50
    before the last step), so P2's city connects; one step further it does not. The previous engine allowed at most
    5 steps of 10."""
    a = Area(11, 3, down=True)                          # P2 after P0 and P1 in tile order
    a.port(a.t(1))                                      # next to the capital
    a.water(*[a.t(k) for k in (2, 3, 4, 5)])
    a.port(a.t(6))
    a.water(a.t(4, 1))                                  # the only way down
    shift = int(far)
    a.water(*[a.t(k, 2) for k in range(5, 8 + shift)])
    c2 = a.t(9 + shift, 2)
    a.city(c2)
    p2 = a.t(8 + shift, 2)
    a.rule(c2, p2)
    a.water(p2, owner=1)
    a.tech(1, FISHING)
    g = _load(a.d)
    assert _step(g, BUILD, p2, PORT) == 0
    e = _export(g)
    assert bool(e['tiles.route'][a.t(3)]) and bool(e['tiles.route'][a.t(6)])
    assert bool(e['tiles.route'][a.t(4, 1)]) == (not far)
    assert bool(e['tiles.route'][p2]) == (not far)
    assert bool(e['cities.connected'][c2]) == (not far)


@pytest.mark.parametrize('other_route', [True, False])
def test_route_tiles_are_shared_by_all_players(other_route):
    """has route is one tile flag: player 2's route over neutral water next to player 1's ports joins player 1's two
    networks (the flood steps neutral -> player 2's tile -> neutral -> own port). Player 1's own routers cannot pass
    player 2's water tile. The previous engine computed routes per player from friendly ports only."""
    a = Area(10, 2)
    a.port(a.t(1))
    a.water(a.t(2))
    a.port(a.t(3))
    a.water(a.t(4))                                     # n1
    a.water(a.t(5), owner=2)                            # m
    a.water(a.t(6))                                     # n2
    a.port(a.t(7))
    c2 = a.t(9)
    a.city(c2)
    p3 = a.t(8)
    a.water(p3)
    a.rule(c2, p3)
    if other_route:
        a.port(a.t(3, 1), owner=2)                      # Q0: Q0 - n1 - m - n2 - Q1
        a.port(a.t(7, 1), owner=2)
    a.tech(1, FISHING)
    a.tech(2, FISHING)
    g = _load(a.d)
    assert _step(g, BUILD, p3, PORT) == 0
    e = _export(g)
    assert bool(e['tiles.route'][a.t(5)]) == other_route
    assert bool(e['cities.connected'][c2]) == other_route


@pytest.mark.parametrize('dx', [1, -1])
def test_router_path_crosses_a_city(dx):
    """is tile accessible: a City tile is always enterable, and from land a router may enter only its own
    port. Routes run from the earlier router in tile order: port P0 - water - city - new port P1 is a route when P0
    comes first (the path enters P1 from the city), not when P1 does (from the city it cannot enter plain water).
    The previous engine routed over water only."""
    a = Area(5, 1, dx=dx)
    a.port(a.t(1))
    a.water(a.t(2))
    c2 = a.t(3)
    a.city(c2)
    p1 = a.t(4)
    a.water(p1)
    a.rule(c2, p1)
    a.tech(1, FISHING)
    g = _load(a.d)
    assert _step(g, BUILD, p1, PORT) == 0
    e = _export(g)
    assert bool(e['tiles.route'][c2]) == bool(e['cities.connected'][c2]) == (dx == 1)


def test_exploring_updates_routes_and_connections():
    """explore flags has flagged need for update routes; at the end of the command run deferred updates the
    routes and the mover's connections. A move that reveals the missing water tile connects the city at once; the
    previous engine reconnected only on builds, captures and diplomacy."""
    a = Area(5, 2, dx=1)                                # P0 before P1 in tile order (see the test above)
    a.port(a.t(1))
    hidden = a.t(2)
    a.water(hidden)
    c2 = a.t(3)
    a.city(c2)
    a.port(a.t(4), city=c2)
    a.tech(1, FISHING)
    a.d['tiles.explored'][0, 0, hidden] = False
    w = a.t(2, 2)
    a.d['units.owner'][0, w], a.d['units.kind'][0, w], a.d['units.health'][0, w] = 1, WARRIOR, 10.
    g = _load(a.d, routes=False)                          # routes of the position: none without the hidden tile
    e = _export(g)
    assert not e['tiles.route'].any() and not e['cities.connected'][c2]
    assert _step(g, MOVE, w, a.t(2, 1)) == 0
    e = _export(g)
    assert e['tiles.explored'][0, hidden] and bool(e['cities.connected'][c2])


# --- 3. Market level at turn start ---------------------------------------------------------------------------

def _market(forge_level, mines):
    a = Area(4, 1)
    cap = a.cap
    m, f = a.t(1), a.t(2)
    for t, k in ((m, MARKET), (f, FORGE)):
        a.rule(cap, t)
        a.d['tiles.structure'][0, t] = k
    a.d['tiles.structure_level'][0, f] = forge_level
    for k in range(mines):
        t = a.t(2 + (k + 1), 0) if k == 0 else a.t(2, 1)
        a.rule(cap, t)
        a.d['tiles.structure'][0, t] = MINE
    return a, m, f


def test_market_counts_a_level_0_forge_as_1():
    """adjacency bonus at adds max(level, 1) per Windmill/Forge/Sawmill (add level: `if (level < 2) level = 1`): a
    market next to a forge without mines pays 1 star. The previous engine summed the raw levels (0)."""
    from polytopia_bot.ruleset import rules_json
    stars = next(s['stars'] for s in json.loads(rules_json())['structures'] if s['id'] == MARKET)
    a, m, f = _market(0, 0)
    with_forge = int(_load(a.d).batch.income(1)[0])
    a.d['tiles.structure'][0, f] = 0
    without = int(_load(a.d).batch.income(1)[0])
    assert with_forge - without == stars > 0


def test_turn_start_market_reads_stored_levels():
    """start turn pays calculate work(tile, update improvement level(tile)) in tile order, and the level actions it
    queues run after the loop: the market is paid from the forge's stored level (3) even though the forge, with one
    mine, is level 1; the forge's stored level is corrected afterwards. The previous engine recomputed the forge."""
    from polytopia_bot.ruleset import rules_json
    stars = next(s['stars'] for s in json.loads(rules_json())['structures'] if s['id'] == MARKET)
    a, m, f = _market(3, 1)
    stale = int(_load(a.d).batch.income(1)[0])
    a.d['tiles.structure_level'][0, f] = 1
    fresh = int(_load(a.d).batch.income(1)[0])
    assert stale - fresh == 2 * stars
    a.d['tiles.structure_level'][0, f] = 3
    a.d['current_player'][0] = 2
    g = _load(a.d)
    assert _step(g, END, 0, 0) == 0
    e = _export(g)
    assert int(e['current_player']) == 1 and int(e['tiles.structure_level'][f]) == 1


# --- 4. Road terrain --------------------------------------------------------------------------------------------

def test_road_terrain_is_field_and_forest():
    """game logic v28: roads require field or forest (terrain requirements)."""
    from polytopia_bot.ruleset import rules_json
    road = next(s for s in json.loads(rules_json())['structures'] if s['id'] == ROAD)
    assert road['terrain'] == [FIELD, 5]
