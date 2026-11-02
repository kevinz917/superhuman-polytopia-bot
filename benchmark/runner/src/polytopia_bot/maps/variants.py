"""Variant Bardur map generator: water, mountains, chokepoints and island maps, for 2 seats (1v1) or 4.

players=4 places one capital per quadrant and generalizes the capital-dependent shapes (continents carve
channels along the boundaries between capitals' regions; shared-landmass presets bridge every capital to the
first). Two-seat maps are bitwise those of bardur-variants-v2; the ridge preset is two-seat only.

Generates the same Layout fields as gpu.mapgen.layout (terrain, structure,
resource, capitals) so a board can initialize the frozen engine unchanged. It
lives outside gpu/ on purpose: the engine digest that checkpoints verify covers
only gpu/. Deterministic per (seed, preset). Host-side NumPy: one map costs well
under a millisecond, so actors generate maps between games.

Terrain ids: 1 shallow water, 2 ocean, 3 field, 4 mountain, 5 forest.
Structures: 1 village, 2 ruin, 47 lighthouse. Resources: 1 animal (forest),
3 fish (shallow water), 5 metal (mountain), 6 fruit (field), 8 starfish.
Distances are Chebyshev, as in the engine.
"""
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np

# v2 adds the 18x18 board (scaling below); 11x11 and 14x14 maps are bitwise those of v1.
GENERATOR_REVISION = 'bardur-variants-v2'
SIZES = (11, 14, 18)
RUINS = {11: 4, 14: 5, 18: 9}          # as gpu.mapgen (reference ruin counts)
MIN_CAPITAL_DISTANCE = {11: 6, 14: 6, 18: 8}
MIN_CAPITAL_DISTANCE_4 = {14: 5, 18: 6}  # four seats: pairwise, one capital per quadrant
# 18x18 scaling (324 tiles, 1.65x the 14x14 area). Noise smoothness is in tiles, so by default
# features keep their size and their count grows with the area (more islands, lakes, clumps).
# Single-landmass shapes scale with the board instead, and lakes grow a little.
LARGE_LAND_SCALE = {'pangea': 18 / 14, 'continents': 18 / 14, 'lakes': (18 / 14) ** .5}
LARGE_RIDGE_PASSES = 1                 # one more pass through the longer wall (3-4 instead of 2-3)
SHALLOW, OCEAN, FIELD, MOUNTAIN, FOREST = 1, 2, 3, 4, 5
VILLAGE, RUIN, LIGHTHOUSE = 1, 2, 47
ANIMAL, FISH, METAL, FRUIT, STARFISH = 1, 3, 5, 6, 8


class Layout(NamedTuple):
    terrain: np.ndarray    # [n] int32
    structure: np.ndarray  # [n] int32
    resource: np.ndarray   # [n] int32
    capitals: np.ndarray   # [players] int32 tile indices


@dataclass(frozen=True)
class Preset:
    name: str
    title: str
    land: tuple = (.9, .95)          # target land fraction range
    mountain: tuple = (.12, .16)     # fraction of free land that is mountain
    forest: tuple = (.3, .4)         # fraction of free non-mountain land that is forest
    land_scale: float = 2.5          # noise smoothness of the coastline (larger = bigger blobs)
    shape: str = 'noise'             # noise | continents | pangea | islands | lakes | ridge
    ridge_passes: tuple = (2, 3)
    village_spacing: int = 3         # minimum Chebyshev distance between villages/capitals
    village_keep: tuple = (.8, 1.)   # fraction of possible villages actually placed
    shared_landmass: bool = False    # force a land route between capitals
    note: str = ''


PRESETS = {p.name: p for p in [
    Preset('drylands', 'Drylands (current-like)', land=(.93, .96), mountain=(.12, .16), forest=(.35, .45),
           land_scale=1.0, note='Almost all land, light mountains. Close to today\'s training maps.'),
    Preset('lakes', 'Lakes', land=(.6, .72), mountain=(.15, .22), forest=(.3, .4), shape='lakes', land_scale=1.6,
           shared_landmass=True, note='One landmass broken up by inland lakes; fishing and short boat hops matter.'),
    Preset('pangea', 'Pangea', land=(.5, .62), mountain=(.2, .28), forest=(.3, .4), shape='pangea', land_scale=2.2,
           shared_landmass=True, note='One central continent ringed by ocean; a land war with long coasts.'),
    Preset('continents', 'Continents', land=(.45, .58), mountain=(.2, .28), forest=(.3, .4), shape='continents',
           land_scale=2.0, note='Each tribe starts on its own continent; reaching the enemy needs ports and boats.'),
    Preset('archipelago', 'Archipelago', land=(.32, .42), mountain=(.15, .22), forest=(.3, .4), shape='islands',
           land_scale=1.1, village_spacing=2, note='Many small islands; expansion is by sea.'),
    Preset('waterworld', 'Water world', land=(.18, .26), mountain=(.1, .16), forest=(.25, .35), shape='islands',
           land_scale=.8, village_spacing=2, village_keep=(.9, 1.), note='Tiny islands in open ocean; navy decides.'),
    Preset('highlands', 'Highlands', land=(.85, .93), mountain=(.32, .42), forest=(.25, .35), land_scale=1.4,
           shared_landmass=True, note='Mountain-heavy land with valleys; slow movement, defensive terrain, metal.'),
    Preset('ridge', 'Ridge with passes', land=(.82, .9), mountain=(.04, .07), forest=(.3, .4), shape='ridge',
           land_scale=1.4, shared_landmass=True, note='A mountain wall between the tribes with 2–3 passes: chokepoints.'),
    Preset('jungle', 'Jungle', land=(.78, .88), mountain=(.1, .15), forest=(.65, .8), land_scale=1.3,
           shared_landmass=True, note='Dense forest: slow movement, ambush cover, lots of animals.'),
]}
# 'mixed' samples a preset uniformly, then its parameters; the proposed training default.
# 'classic' (the frozen engine generator, gpu.mapgen) joins the mix when maps are wired into training.
MIXED = tuple(PRESETS)


def _blur(a, sigma):
    if sigma <= 0:
        return a
    r = max(1, int(3 * sigma))
    k = np.exp(-.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, r, mode='reflect')
    a = np.apply_along_axis(lambda v: np.convolve(v, k, 'valid'), 0, pad)
    return np.apply_along_axis(lambda v: np.convolve(v, k, 'valid'), 1, a)


def _noise(rng, size, scale, octaves=3):
    total, amp = np.zeros((size, size)), 1.
    for o in range(octaves):
        layer = _blur(rng.normal(size=(size, size)), scale / (2 ** o))
        layer = (layer - layer.mean()) / (layer.std() + 1e-9)
        total += amp * layer
        amp *= .5
    return (total - total.mean()) / (total.std() + 1e-9)


def _cheb(size):
    ids = np.arange(size * size)
    x, y = ids % size, ids // size
    return np.maximum(np.abs(x[:, None] - x[None, :]), np.abs(y[:, None] - y[None, :]))


def _components(mask, size):
    """4-connected land components (boats are needed between them)."""
    label = -np.ones(size * size, int)
    k = 0
    for start in np.flatnonzero(mask.ravel()):
        if label[start] >= 0:
            continue
        stack = [start]
        label[start] = k
        while stack:
            i = stack.pop()
            x, y = i % size, i // size
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    j = ny * size + nx
                    if mask.ravel()[j] and label[j] < 0:
                        label[j] = k
                        stack.append(j)
        k += 1
    return label, k


def _line(a, b, size):
    ax, ay, bx, by = a % size, a // size, b % size, b // size
    steps = max(abs(bx - ax), abs(by - ay), 1)
    return [int(round(ay + (by - ay) * t / steps)) * size + int(round(ax + (bx - ax) * t / steps)) for t in range(steps + 1)]


def sample_parameters(rng, preset, size=14):
    pick = lambda r: float(rng.uniform(*r))
    extra = LARGE_RIDGE_PASSES if size == 18 else 0
    return dict(land=pick(preset.land), mountain=pick(preset.mountain), forest=pick(preset.forest),
                village_keep=pick(preset.village_keep),
                passes=int(rng.integers(preset.ridge_passes[0] + extra, preset.ridge_passes[1] + extra + 1)))


def _four_capitals(rng, size, d, edge, x, y):
    """One capital per quadrant, 2+ tiles from the edge and from the quadrant lines, pairwise at least
    MIN_CAPITAL_DISTANCE_4 apart (redrawn until they are; the farthest-spread draw if never)."""
    half = size // 2
    quads = [np.flatnonzero((edge >= 2) & ((x < half) == (q % 2 == 0)) & ((y < half) == (q < 2))
                            & (np.abs(x - (half - .5)) >= 2) & (np.abs(y - (half - .5)) >= 2)) for q in range(4)]
    best, best_gap = None, -1
    for _ in range(200):
        caps = np.array([int(rng.choice(c)) for c in quads])
        gap = min(int(d[a, b]) for i, a in enumerate(caps) for b in caps[i + 1:])
        if gap >= MIN_CAPITAL_DISTANCE_4[size]:
            return caps
        if gap > best_gap:
            best, best_gap = caps, gap
    return best


def generate(seed, preset='mixed', size=14, players=2):
    """Returns (Layout, info). info carries the preset, sampled parameters and map statistics."""
    if size not in SIZES:
        raise ValueError('variant maps support 11, 14 or 18')
    if players not in (2, 4) or (players == 4 and size not in MIN_CAPITAL_DISTANCE_4):
        raise ValueError('variant maps support 2 seats, or 4 seats on 14x14 and 18x18')
    if players == 4 and preset == 'ridge':
        raise ValueError('the ridge preset is two-seat only')
    rng = np.random.default_rng([int(seed), sum(map(ord, preset)), 20261019] + ([players] if players != 2 else []))
    if preset == 'mixed':
        mixed = MIXED if players == 2 else tuple(p for p in MIXED if p != 'ridge')
        preset = mixed[int(rng.integers(len(mixed)))]
    p = PRESETS[preset]
    if size == 18 and p.shape in LARGE_LAND_SCALE:
        p = replace(p, land_scale=p.land_scale * LARGE_LAND_SCALE[p.shape])
    q = sample_parameters(rng, p, size)
    n = size * size
    d = _cheb(size)
    ids = np.arange(n)
    x, y = ids % size, ids // size
    edge = np.minimum(np.minimum(x, size - 1 - x), np.minimum(y, size - 1 - y))

    # --- capitals: opposite sides, at least ~half the board apart, 2 tiles from the edge (four seats: quadrants)
    if players == 4:
        caps = _four_capitals(rng, size, d, edge, x, y)
        c0, c1 = int(caps[0]), int(caps[1])
    else:
        caps = None
    orient = rng.integers(3) if caps is None else 0  # 0 horizontal, 1 vertical, 2 diagonal
    def side(i):
        if orient == 0:
            return (x < size // 2 - 1) if i == 0 else (x > size // 2)
        if orient == 1:
            return (y < size // 2 - 1) if i == 0 else (y > size // 2)
        return (x + y < size - 3) if i == 0 else (x + y > size + 1)
    if caps is None:
        flip = rng.random() < .5
        sides = [side(1 if flip else 0), side(0 if flip else 1)]
        ok0 = np.flatnonzero(sides[0] & (edge >= 2))
        ok1 = np.flatnonzero(sides[1] & (edge >= 2))
        far = d[np.ix_(ok0, ok1)] >= MIN_CAPITAL_DISTANCE[size]
        if far.any():
            a_idx, b_idx = np.nonzero(far)
            k = int(rng.integers(len(a_idx)))
            c0, c1 = int(ok0[a_idx[k]]), int(ok1[b_idx[k]])
        else:  # never expected on the supported boards; take the farthest pair
            a_idx, b_idx = np.unravel_index(np.argmax(d[np.ix_(ok0, ok1)]), (len(ok0), len(ok1)))
            c0, c1 = int(ok0[a_idx]), int(ok1[b_idx])
        caps = np.array([c0, c1])
    near_cap = d[caps].min(0)                      # distance to the nearest capital

    # --- land shape: noise height + preset bias, thresholded to the target land fraction
    h = _noise(rng, size, p.land_scale).ravel()
    centre = ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) ** .5 / (size / 2)
    if p.shape == 'pangea':
        h = h * .6 - 2.2 * centre
    elif p.shape == 'continents':
        near = near_cap / size
        h = h * .7 - 4.0 * near
    elif p.shape == 'islands':
        h = h + _noise(rng, size, .7).ravel() * .8
    elif p.shape == 'lakes':
        h = h * .6 + 1.2 * np.minimum(centre, 1.)  # inland water in pockets, coasts mostly land
        h = -np.abs(h - np.quantile(h, .3))         # lake pockets at a mid height band
    land = h >= np.quantile(h, 1 - q['land'])
    # capitals always start with their 3x3 on land
    for c in caps:
        land[d[c] <= 1] = True
    if p.shape == 'continents':
        # carve a sea channel between the starting regions (no land bridge): where the two nearest capitals are
        # (almost) equally far
        if players == 2:
            between = np.abs(d[c0] - d[c1]) <= 1
        else:
            ranked = np.sort(d[caps], 0)
            between = ranked[1] - ranked[0] <= 1
        land &= ~(between & (near_cap >= 3))
        for c in caps:
            land[d[c] <= 1] = True

    # --- elevation for mountains, separate noise for forests
    elev = _noise(rng, size, 1.2).ravel()
    wood = _noise(rng, size, 1.0).ravel()
    protected = np.zeros(n, bool)
    for c in caps:
        protected |= d[c] <= 1
    terrain = np.full(n, OCEAN, np.int32)
    terrain[land] = FIELD
    free = land & ~protected
    if free.any():
        k = int(round(q['mountain'] * free.sum()))
        if k:
            order = np.flatnonzero(free)[np.argsort(-elev[free])]
            terrain[order[:k]] = MOUNTAIN
    if p.shape == 'ridge':
        # Wall along the perpendicular bisector of the capitals (Euclidean), 1-2 tiles thick.
        ax, ay, bx, by = c0 % size, c0 // size, c1 % size, c1 // size
        e0 = np.hypot(x - ax, y - ay)
        e1 = np.hypot(x - bx, y - by)
        wall = (np.abs(e0 - e1) <= rng.uniform(1.0, 1.6)) & ~protected
        terrain[wall & land] = MOUNTAIN
        terrain[wall & ~land] = MOUNTAIN   # the wall also bridges water so it truly separates
        land |= wall
        # passes: spread along the wall (sorted by position along it), each opened 1 tile wide
        along = np.flatnonzero(wall)
        if len(along):
            tang = (x[along] - (ax + bx) / 2) * -(by - ay) + (y[along] - (ay + by) / 2) * (bx - ax)
            along = along[np.argsort(tang)]
            k = q['passes']
            picks = [along[int((j + .5) * len(along) / k)] for j in range(k)]
            for g in picks:
                gx, gy = g % size, g // size
                # open a one-tile-wide pass straight across the wall's thickness
                ux, uy = (bx - ax), (by - ay)
                norm = max(np.hypot(ux, uy), 1e-9)
                for step in np.arange(-2, 2.5, .5):
                    px, py = int(round(gx + step * ux / norm)), int(round(gy + step * uy / norm))
                    if 0 <= px < size and 0 <= py < size and wall[py * size + px]:
                        terrain[py * size + px] = FIELD
    non_m = land & (terrain != MOUNTAIN) & ~protected
    if non_m.any():
        k = int(round(q['forest'] * non_m.sum()))
        if k:
            order = np.flatnonzero(non_m)[np.argsort(-wood[non_m])]
            terrain[order[:k]] = FOREST
    # shared landmass: if required and capitals are split, cut a land bridge along the direct line
    label, _ = _components(terrain.reshape(size, size) >= FIELD, size)
    for other in caps[1:]:
        if p.shared_landmass and label[c0] != label[other]:
            path = _line(c0, int(other), size)
            for a_, b_ in zip(path, path[1:]):
                # diagonal steps get an orthogonal corner tile so the bridge is 4-connected
                for t in (a_, b_, (a_ // size) * size + b_ % size):
                    if terrain[t] < FIELD:
                        terrain[t] = FIELD
            label, _ = _components(terrain.reshape(size, size) >= FIELD, size)
    # water depth: shallow if orthogonally next to land
    landm = terrain >= FIELD
    grid = landm.reshape(size, size)
    adj = np.zeros_like(grid)
    adj[1:] |= grid[:-1]; adj[:-1] |= grid[1:]; adj[:, 1:] |= grid[:, :-1]; adj[:, :-1] |= grid[:, 1:]
    terrain[~landm & adj.ravel()] = SHALLOW
    terrain[~landm & ~adj.ravel()] = OCEAN

    # --- villages: greedy spaced placement on non-mountain land
    structure = np.zeros(n, np.int32)
    taken = np.zeros(n, bool)
    taken[caps] = True
    structure[caps] = VILLAGE
    for t in rng.permutation(n):
        if terrain[t] in (FIELD, FOREST) and edge[t] >= 1 and not np.any(taken & (d[t] < p.village_spacing)):
            if rng.random() < q['village_keep']:
                taken[t] = True
                structure[t] = VILLAGE
                terrain[t] = FIELD
    villages = structure == VILLAGE

    # --- resources around settlements, by terrain (same families as the frozen generator)
    resource = np.zeros(n, np.int32)
    for c in caps:  # two guaranteed animals next to each capital
        spots = np.flatnonzero((d[c] == 1) & (terrain >= FIELD) & (terrain != MOUNTAIN))
        for t in rng.choice(spots, size=min(2, len(spots)), replace=False):
            terrain[t] = FOREST
            resource[t] = ANIMAL
    near = np.zeros(n, int)
    for v in np.flatnonzero(villages):
        near = np.where(d[v] <= 2, np.minimum(np.where(near == 0, 9, near), np.maximum(d[v], 1)), near)
    for t in np.flatnonzero((near > 0) & (resource == 0) & (structure == 0)):
        r1 = near[t] == 1
        prob, kind = {FIELD: ((.18, .06), FRUIT), FOREST: ((.38, .12), ANIMAL), MOUNTAIN: ((.11, .03), METAL),
                      SHALLOW: ((.5, .5), FISH)}.get(int(terrain[t]), ((0, 0), 0))
        if kind and rng.random() < prob[0 if r1 else 1]:
            resource[t] = kind
    # ruins on land away from villages, starfish at sea, lighthouses on corners
    for _ in range(RUINS[size]):
        spots = np.flatnonzero((terrain >= FIELD) & (structure == 0) & (resource == 0)
                               & ~np.any((d <= 1) & ((structure == RUIN) | villages)[None, :], axis=1))
        if len(spots):
            structure[int(rng.choice(spots))] = RUIN
    sea = np.flatnonzero((terrain <= OCEAN) & (resource == 0) & (structure == 0))
    for t in rng.permutation(sea)[:n // 25]:
        resource[t] = STARFISH
    corners = np.array([0, size - 1, n - size, n - 1])
    structure[corners] = LIGHTHOUSE
    resource[corners] = 0

    label, count = _components(terrain.reshape(size, size) >= FIELD, size)
    info = dict(preset=preset, revision=GENERATOR_REVISION, seed=int(seed), size=size,
                params={k: round(v, 3) if isinstance(v, float) else v for k, v in q.items()},
                stats=dict(land=float((terrain >= FIELD).mean()), water=float((terrain <= OCEAN).mean()),
                           mountain=float((terrain == MOUNTAIN).mean()), forest=float((terrain == FOREST).mean()),
                           villages=int(villages.sum() - len(caps)), landmasses=int(count),
                           capitals_connected=bool(np.all(label[caps] == label[c0])),
                           capital_distance=int(min(d[a, b] for i, a in enumerate(caps) for b in caps[i + 1:]))))
    return Layout(terrain.astype(np.int32), structure, resource, caps.astype(np.int32)), info
