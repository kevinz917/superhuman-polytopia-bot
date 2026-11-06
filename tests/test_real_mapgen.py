"""engine-rs mapgen against the real game's Drylands generator (map generator, Polytopia 2.17.3, game version 123).

Seed-exact: 30 boards the real game generated with chosen seeds (11/14/18 tiles, 2-4 seats) equal generate internal here tile for
tile, with the same inequality. Distribution: the recorded real 14x14 games' turn-0
boards (their attempt seeds come from a crypto RNG and are not recorded) satisfy every generator invariant, were kept
by the 50-attempt fairness loop (low inequality), and match the per-board counts of our generator.
"""
import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parent / 'fixtures/real-mapgen'   # game-generated boards, not distributed
needs_fixtures = pytest.mark.skipif(not FIXTURES.is_dir(), reason='game-derived fixtures: private checkouts only')
FIELD, MOUNTAIN, FOREST = 3, 4, 5
CITY, RUIN, LIGHTHOUSE = 1, 2, 47
GAME, METAL, FRUIT = 1, 5, 6
f32 = np.float32


@lru_cache(None)
def _batch(size, players):
    from importlib.resources import files
    from polytopia_engine_rs import GameBatch
    from polytopia_bot.gpu import ENGINE_REVISION
    rules = files('polytopia_bot.gpu').joinpath('rules.json').read_text()
    return GameBatch(size, 1, rules, ENGINE_REVISION, None, True, 10_000, players)


def _fixture(name):
    return json.loads(gzip.decompress((FIXTURES / name).read_bytes()))


def _layout(size, players, seed, exact=False):
    d = _batch(size, players).layout(seed, exact=exact)
    return {k: (np.asarray(v) if k in ('terrain', 'structure', 'resource', 'capitals') else v) for k, v in d.items()}


def _dist(size, a, b):
    return max(abs(a % size - b % size), abs(a // size - b // size))


def inequality(size, terrain, structure, capitals):
    """inequality in float32 (an independent twin of mapgen::inequality)."""
    n = size * size
    visited, nxt, turfs = [False] * n, [], [[] for _ in capitals]

    def grow(cur, cost, seat):
        cx, cy = cur % size, cur // size
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                x, y = cx + dx, cy + dy
                if (dx or dy) and 0 <= x < size and 0 <= y < size and not visited[y * size + x]:
                    t = y * size + x
                    c = (4 if terrain[t] == MOUNTAIN else 1) + cost
                    e = next((e for e in nxt if e[0] == t), None)
                    if e is None:
                        nxt.append([t, c, seat])
                    elif c < e[1] or (c == e[1] and e[2] > seat):
                        e[1], e[2] = c, seat

    for seat, c in enumerate(capitals):
        visited[c] = True
        turfs[seat].append((c, 0))
        grow(c, 0, seat)
    while nxt:
        t, c, seat = nxt.pop(min(range(len(nxt)), key=lambda i: (nxt[i][1], i)))
        visited[t] = True
        turfs[seat].append((t, c))
        grow(t, c, seat)
    reach = f32(size) * f32(.8) + f32(-3)
    scores = []
    for turf in turfs:
        total = f32(0)
        for t, c in turf:
            f = min(f32(1), max(f32(0), (f32(c) + f32(-3)) / reach))
            if structure[t] == CITY:
                s = f * f32(-90) + f32(100) + f32(.1)
                s = s + f32(20) if c == 4 else s + f32(40) if c == 3 else s
            elif structure[t] == RUIN:
                s = f * f32(-3) + f32(5) + f32(.1)
                s = s + f32(5) if c < 5 else s
            else:
                s = f32(.1)
            if 3 <= terrain[t] <= 6:
                s = s + (f32(1.5) - f)
            total = total + s
        scores.append(total)
    lo = min(scores)
    return max(scores) / (lo if lo > 0 else f32(.001))


def check_board(size, terrain, structure, resource, capitals):
    """The invariants generate internal guarantees on a Drylands Bardur board."""
    n, np_ = size * size, len(capitals)
    terrain, structure, resource = (np.asarray(a) for a in (terrain, structure, resource))
    capitals = [int(c) for c in capitals]
    assert set(np.unique(terrain)) <= {FIELD, MOUNTAIN, FOREST}, 'Drylands has no water'
    corners = {0, size - 1, n - size, n - 1}
    assert set(np.flatnonzero(structure == LIGHTHOUSE)) == corners and all(terrain[c] == FIELD for c in corners)
    # generate player capital positions: one ceil(sqrt(seats))^2 domain each, 2 in from its edges
    k = int(np.ceil(np.sqrt(np_)))
    domain, rem = size // k, size % k
    off = lambda i: min(rem, max(1, min(rem, i) - 1))
    lo, hi = (1, -1) if domain < 5 else (2, -2)
    seen = set()
    for c in capitals:
        x, y = c % size, c // size
        dom = (min(x // domain, k - 1), min(y // domain, k - 1))
        assert dom not in seen
        seen.add(dom)
        for v, d in ((x, dom[0]), (y, dom[1])):
            start = domain * d + off(d)
            assert max(2, start + lo) <= v < min(size - 2, start + hi + domain)
        assert structure[c] == CITY and terrain[c] == FIELD and resource[c] == 0
    cities = [int(c) for c in np.flatnonzero(structure == CITY)]
    villages = [c for c in cities if c not in capitals]
    assert len(villages) <= int(size * .33) ** 2 - np_
    for v in villages:   # add post terrain cities / is valid city location
        x, y = v % size, v // size
        assert min(x, y, size - 1 - x, size - 1 - y) >= 1 and terrain[v] == FIELD and resource[v] == 0
        assert all(_dist(size, v, c) >= 3 for c in cities if c != v)
    ruins = [int(t) for t in np.flatnonzero(structure == RUIN)]
    assert len(ruins) == n // 40 + 1   # add ruins(tiles / 40) places amount + 1
    for r in ruins:
        assert r not in corners
        assert all(_dist(size, r, c) >= 2 for c in cities) and all(_dist(size, r, c) >= 3 for c in capitals)
        assert all(_dist(size, r, o) >= 3 for o in ruins if o != r)
    for t in np.flatnonzero(resource):   # Bardur Drylands: game on forest, metal on mountain, fruit on field
        assert (int(resource[t]), int(terrain[t])) in {(GAME, FOREST), (METAL, MOUNTAIN), (FRUIT, FIELD)}
        assert structure[t] in (0, RUIN) and min(_dist(size, t, c) for c in cities) <= 2
    for c in capitals:   # add starting resources to capital: two game in the capital's ring
        assert sum(resource[t] == GAME for t in range(n) if _dist(size, c, t) == 1) >= 2


def _features(b, size):
    s, t, r = (np.asarray(b[k]) for k in ('structure', 'terrain', 'resource'))
    return dict(villages=(s == CITY).sum() - len(b['capitals']), field=(t == FIELD).sum(), mountain=(t == MOUNTAIN).sum(),
                forest=(t == FOREST).sum(), game=(r == GAME).sum(), metal=(r == METAL).sum(), fruit=(r == FRUIT).sum(),
                inequality=float(inequality(size, t, s, [int(c) for c in b['capitals']])))


@needs_fixtures
def test_real_boards_are_reproduced_seed_exact():
    rows = _fixture('host-maps.json.gz')
    assert len(rows) == 30 and {(r['size'], r['players']) for r in rows} == {(11, 2), (14, 2), (18, 2), (14, 4), (18, 3), (18, 4)}
    for r in rows:
        ours = _layout(r['size'], r['players'], r['seed'], exact=True)
        for key in ('terrain', 'structure', 'resource', 'capitals'):
            np.testing.assert_array_equal(ours[key], r[key], err_msg=f"{r['size']}x{r['players']} seed {r['seed']} {key}")
        assert f32(ours['inequality']) == f32(r['inequality']) and ours['attempt_seed'] == r['seed']
        assert f32(inequality(r['size'], r['terrain'], r['structure'], r['capitals'])) == f32(r['inequality'])
        check_board(r['size'], r['terrain'], r['structure'], r['resource'], r['capitals'])


@needs_fixtures
def test_recorded_real_games_start_on_generator_boards():
    real = _fixture('turn0-worlds.json.gz')
    assert len(real) == 18 and {(r['size'], r['players']) for r in real} == {(14, 2)}
    for r in real:
        check_board(r['size'], r['terrain'], r['structure'], r['resource'], r['capitals'])
    real_f = [_features(r, 14) for r in real]
    ours_f = [_features(_layout(14, 2, seed), 14) for seed in range(200)]
    # the fairness loop kept them: the real boards' inequality is as low as the boards our loop keeps
    assert max(f['inequality'] for f in real_f) < 1.1 and max(f['inequality'] for f in ours_f) < 1.2
    for key in real_f[0]:
        a = np.array([f[key] for f in real_f], float)
        b = np.array([f[key] for f in ours_f], float)
        assert abs(a.mean() - b.mean()) <= 4 * b.std() / np.sqrt(len(a)) + .05, (key, a.mean(), b.mean())


@pytest.mark.parametrize('size,players', [(11, 2), (14, 2), (18, 2), (14, 4), (18, 3), (18, 4)])
def test_generated_boards_keep_the_generator_invariants(size, players):
    for seed in range(12):
        b = _layout(size, players, seed)
        check_board(size, b['terrain'], b['structure'], b['resource'], b['capitals'])
        # Generate keeps an attempt: the board is generate internal(attempt seed), its inequality the loop's best
        exact = _layout(size, players, b['attempt_seed'], exact=True)
        for key in ('terrain', 'structure', 'resource', 'capitals'):
            np.testing.assert_array_equal(b[key], exact[key])
        assert f32(b['inequality']) == f32(exact['inequality'])
        assert f32(inequality(size, b['terrain'], b['structure'], list(b['capitals']))) == f32(b['inequality'])
    assert not np.array_equal(_layout(size, players, 0)['terrain'], _layout(size, players, 1)['terrain'])


@pytest.mark.parametrize('size,players', [(11, 2), (14, 2), (18, 2), (18, 4)])
def test_reset_starts_on_the_generated_board(size, players):
    batch = _batch(size, players)
    for seed in (3, 4):
        batch.reset(np.array([seed], np.int32))
        d = {k: np.array(v)[0] for k, v in batch.export().items()}
        b = _layout(size, players, seed)
        np.testing.assert_array_equal(d['tiles.terrain'], b['terrain'])
        np.testing.assert_array_equal(d['tiles.structure'], b['structure'])
        np.testing.assert_array_equal(d['tiles.resource'], b['resource'])
        for seat, c in enumerate(b['capitals'], 1):
            assert d['tiles.capital'][c] == seat and d['cities.owner'][c] == seat and d['units.owner'][c] == seat
