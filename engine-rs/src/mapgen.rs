//! The game's map generator (game version 123) on the Drylands preset for Bardur seats,
//! and the reset built on it. `generate_internal` is one attempt for one attempt seed, tile for tile; `layout` is
//! the full generation: up to 50 attempt seeds, kept by the map's inequality. The game draws attempt seeds
//! from a crypto RNG; here they come from System.Random(seed), so a reset seed fixes the board.

use crate::geom::Geom;
use crate::hash::{hash_range3, hash_value3, DotNetRandom};
use crate::kernels::update_exploration;
use crate::state::State;

/// Ruins per board: the ruin pass with amount tiles / 40 places amount + 1.
pub fn ruins(size: usize) -> Option<i32> {
    match size {
        11 | 14 | 18 => Some((size * size / 40) as i32 + 1),
        _ => None,
    }
}

/// Training-harness round cap (the game's Domination mode has none).
pub fn round_limit(size: usize) -> i32 {
    if size == 18 { 150 } else { 100 }
}

/// A generated board in the game's ids: terrain, structure (improvement ids: 1 city, also on
/// capitals, 2 ruin, 47 lighthouse; a ruin can sit on a resource), resource, capitals per seat.
pub struct Layout {
    pub terrain: Vec<i32>,
    pub structure: Vec<i32>,
    pub resource: Vec<i32>,
    /// One capital tile per seat.
    pub capitals: Vec<usize>,
}

const FIELD: i32 = 3;
const MOUNTAIN: i32 = 4;
const FOREST: i32 = 5;
const ICE: i32 = 6;
const CITY: i32 = 1;
const RUIN: i32 = 2;
const LIGHTHOUSE: i32 = 47;
const WHALE: usize = 4;
/// Generator settings for the Drylands preset: equality iterations, equality limit.
const EQUALITY_ITERATIONS: usize = 50;
const EQUALITY_LIMIT: f32 = 1.01;
/// A continent modifier: a positive default is scaled, otherwise the modifier replaces it.
fn modify(default: f32, modifier: f32) -> f32 {
    if default > 0.0 { default * modifier } else { modifier }
}
/// A Bardur continent's resource modifiers by resource id (continent defaults, the tribe's, Bardur crop 0
/// and the wetness-0 climates' fish/whale 0): game, metal, fruit and aquacrop survive.
fn resource_modifiers() -> [f32; 10] {
    [0.0, 0.5, modify(0.5, 0.0), 0.0, 0.0, 0.8, 0.5, 0.0, 0.0, 0.3]
}
/// Terrain requirements per resource, in the game's resource order.
const RESOURCE_TERRAIN: [&[i32]; 10] = [&[], &[5], &[3], &[1, 2], &[2], &[4], &[3], &[3, 5], &[1, 2], &[1, 2]];

fn is_water(t: i32) -> bool {
    t == 1 || t == 2
}

/// The in-board square of `radius`, diagonals included, x-major (dx outer, dy inner).
fn area(w: usize, c: usize, radius: i32, include_center: bool, out: &mut Vec<usize>) {
    out.clear();
    let (cx, cy, s) = ((c % w) as i32, (c / w) as i32, w as i32);
    for dx in -radius..=radius {
        for dy in -radius..=radius {
            if !include_center && dx == 0 && dy == 0 {
                continue;
            }
            let (x, y) = (cx + dx, cy + dy);
            if x >= 0 && x < s && y >= 0 && y < s {
                out.push((y * s + x) as usize);
            }
        }
    }
}

struct Board {
    w: usize,
    n: usize,
    terrain: Vec<i32>,
    improvement: Vec<i32>,
    resource: Vec<i32>,
    /// climate assigned (at a capital or by the climate pass), and the continent (seat) a tile joined in that pass
    climate: Vec<bool>,
    continent: Vec<i32>,
    caps: Vec<usize>,
    rng: DotNetRandom,
    scratch: Vec<usize>,
}

impl Board {
    fn any_in(&mut self, c: usize, radius: i32, f: impl Fn(&Board, usize) -> bool) -> bool {
        let mut a = std::mem::take(&mut self.scratch);
        area(self.w, c, radius, true, &mut a);
        let hit = a.iter().any(|&t| f(self, t));
        self.scratch = a;
        hit
    }

    /// Capital positions: one ceil(sqrt(players))^2 domain grid; each seat draws a free
    /// domain, then a tile inside it (2 from the board edge) weighted by max(1, domain/2 - distance to centre)^3.
    fn capital_positions(&mut self, np: usize) {
        let (w, n) = (self.w as i32, self.n);
        let k = (np as f64).sqrt().ceil() as i32;
        let (domain, rem) = (w / k, w % k);
        assert!(domain >= 3, "domain too small for an isolated capital per seat");
        let off = |i: i32| rem.min(1i32.max(rem.min(i) - 1));
        let mut domains: Vec<i32> = (0..k * k).collect();
        let mut prob = vec![0i32; n];
        for row in 0..k {
            for col in 0..k {
                let (cx, cy) = (off(col) + col * domain + domain / 2, off(row) + row * domain + domain / 2);
                for (i, p) in prob.iter_mut().enumerate() {
                    let (x, y) = (i as i32 % w, i as i32 / w);
                    let d = (x - cx).abs().max((y - cy).abs());
                    let v = 1i32.max(domain / 2 - d).pow(3);
                    *p = if *p != 0 { (*p).max(v) } else { v };
                }
            }
        }
        let (lo, hi) = if domain < 5 { (1, -1) } else { (2, -2) };
        for _ in 0..np {
            let r = self.rng.next_range(0, domains.len() as i32);
            let d = domains.remove(r as usize);
            let (ix, iy) = (domain * (d % k) + off(d % k), domain * (d / k) + off(d / k));
            let (sx, ex) = (2.max(ix + lo), (w - 2).min(ix + hi + domain));
            let (sy, ey) = (2.max(iy + lo), (w - 2).min(iy + hi + domain));
            let cells = || (sy..ey).flat_map(move |y| (sx..ex).map(move |x| (y * w + x) as usize));
            let total: i32 = cells().map(|i| 1i32.max(prob[i])).sum();
            let v = self.rng.next_range(0, total);
            let mut cum = 0;
            let at = cells().find(|&i| {
                cum += 1i32.max(prob[i]);
                v < cum
            });
            self.caps.push(at.expect("IndexForProbabilityValueInRange"));
        }
    }

    /// Noise terrain at wetness 0: one draw per tile, every tile clears the lowest-noise land
    /// threshold, so the board is all field; then the capitals get their cities and climates.
    fn terrain_from_noise(&mut self) {
        for _ in 0..self.n {
            self.rng.value();
        }
        self.terrain.iter_mut().for_each(|t| *t = FIELD);
        for k in 0..self.caps.len() {
            let c = self.caps[k];
            self.improvement[c] = CITY;
            self.climate[c] = true;
        }
    }

    /// Post-terrain cities: villages on random valid spots (a valid city spot: land, not on the edge, no
    /// improvement within 2), each removing the tiles of its 5x5 square from the spot list.
    fn post_terrain_cities(&mut self, count: i32) {
        let w = self.w;
        let mut spots: Vec<i64> = (0..self.n)
            .filter(|&t| {
                let (x, y) = (t % w, t / w);
                let edge = x.min(w - 1 - x).min(y).min(w - 1 - y);
                !is_water(self.terrain[t]) && edge >= 1
            })
            .map(|t| t as i64)
            .collect::<Vec<_>>();
        spots.retain(|&t| !self.any_in(t as usize, 2, |b, a| b.improvement[a] != 0));
        for _ in 0..count {
            if spots.is_empty() {
                return;
            }
            let t = spots[self.rng.next_range(0, spots.len() as i32) as usize] as usize;
            self.terrain[t] = FIELD;
            self.resource[t] = 0;
            self.improvement[t] = CITY;
            let (x, y, wi) = ((t % w) as i64, (t / w) as i64, w as i64);
            for yy in y - 2..=y + 2 {
                for xx in x - 2..=x + 2 {
                    if let Some(p) = spots.iter().position(|&s| s == yy * wi + xx) {
                        spots.remove(p);
                    }
                }
            }
        }
    }

    /// Climates: one continent per seat grown round-robin from its capital: a random tile of the continent, then
    /// a random climate-less neighbour (land first); a continent whose tile has none drops it (panic budget 10/tile).
    fn climates(&mut self) {
        let (n, np) = (self.n, self.caps.len());
        let mut tiles: Vec<Vec<usize>> = self.caps.iter().map(|&c| vec![c]).collect();
        let (mut assigned, mut panic, mut k) = (np, 10 * n as i32, 0isize);
        if !(panic > 0 && np < n) {
            return;
        }
        let (mut a, mut cand) = (Vec::with_capacity(8), Vec::with_capacity(8));
        loop {
            let cont = &mut tiles[k as usize];
            let mut p = panic;
            if cont.is_empty() {
                p -= 1;
            } else {
                let c = cont[self.rng.next_range(0, cont.len() as i32) as usize];
                area(self.w, c, 1, false, &mut a);
                cand.clear();
                cand.extend(a.iter().copied().filter(|&t| !self.climate[t] && !is_water(self.terrain[t])));
                if cand.is_empty() {
                    cand.extend(a.iter().copied().filter(|&t| !self.climate[t]));
                }
                if cand.is_empty() {
                    let at = cont.iter().position(|&t| t == c).unwrap();
                    cont.remove(at);
                    k -= 1;
                    p -= 1;
                } else {
                    let t = cand[self.rng.next_range(0, cand.len() as i32) as usize];
                    self.climate[t] = true;
                    self.continent[t] = k as i32;
                    if !is_water(self.terrain[t]) {
                        self.terrain[t] = FIELD;
                        self.resource[t] = 0;
                    }
                    cont.push(t);
                    assigned += 1;
                }
            }
            panic = p;
            k += 1;
            if assigned < n && (k as usize) < np && panic > 0 {
                continue;
            }
            if assigned < n {
                k = 0;
                if panic > 0 {
                    continue;
                }
            }
            break;
        }
    }

    /// Terrain (continents patched): per continent a bag of ceil(L * share) field, mountain and forest drawn
    /// without replacement over its free land tiles in tile order (field 0.5, mountain 0.15, forest 0.4 x Bardur 0.8).
    fn terrain(&mut self) {
        let (field, mountain, forest) = (0.5f32, 0.15f32, modify(0.4, 0.8));
        let sum = forest + field + mountain;
        let mut bag = Vec::with_capacity(self.n);
        for k in 0..self.caps.len() as i32 {
            let tiles: Vec<usize> = (0..self.n).filter(|&t| self.continent[t] == k).collect();
            let land = tiles.iter().filter(|&&t| !is_water(self.terrain[t]) && self.improvement[t] == 0).count() as f32;
            bag.clear();
            for (kind, count) in [(FIELD, land * (field / sum)), (MOUNTAIN, (mountain / sum) * land), (FOREST, (forest / sum) * land)] {
                if count > 0.0 {
                    let mut i = 1;
                    loop {
                        bag.push(kind);
                        if i as f32 >= count {
                            break;
                        }
                        i += 1;
                    }
                }
            }
            for t in tiles {
                if !is_water(self.terrain[t]) && self.improvement[t] == 0 {
                    self.resource[t] = 0;
                    if bag.is_empty() {
                        break;
                    }
                    let r = self.rng.next_range(0, bag.len() as i32) as usize;
                    self.terrain[t] = bag.remove(r);
                }
            }
        }
    }

    /// Lighthouses: a lighthouse on each corner of the board, the tile set to field.
    fn lighthouses(&mut self) {
        for c in corners(self.w) {
            self.improvement[c] = LIGHTHOUSE;
            if !is_water(self.terrain[c]) {
                self.terrain[c] = FIELD;
            }
        }
    }

    /// One resource: an xxHash seeded from one draw; every resource type passes on hash(x, y, i) <= its continent
    /// modifier (whales not next to whales or cities) and a terrain match; the hash picks among the candidates.
    fn add_resource(&mut self, t: usize, mods: &[f32; 10]) {
        let seed = (self.rng.value() * 1000.0f32) as i32 as u32;
        if self.improvement[t] != 0 {
            return;
        }
        if self.continent[t] < 0 {
            self.terrain[t] = ICE;
            return;
        }
        let (x, y) = ((t % self.w) as i32, (t / self.w) as i32);
        let mut cand = [0usize; 10];
        let mut found = 0;
        for (i, (&m, reqs)) in mods.iter().zip(RESOURCE_TERRAIN.iter()).enumerate() {
            let h = hash_value3(seed, x, y, i as i32);
            if m == 0.0 || h > m {
                continue;
            }
            if i == WHALE {
                let mut a = std::mem::take(&mut self.scratch);
                area(self.w, t, 1, false, &mut a);
                let blocked = a.iter().any(|&u| self.resource[u] == WHALE as i32 || self.improvement[u] == CITY);
                self.scratch = a;
                if blocked {
                    continue;
                }
            }
            if reqs.is_empty() || reqs.contains(&self.terrain[t]) {
                cand[found] = i;
                found += 1;
            }
        }
        if found > 0 {
            self.resource[t] = cand[hash_range3(seed, 0, found as i32, x, y, mods.len() as i32) as usize] as i32;
        }
    }

    /// Resources (richness 1): per city in tile order, each ring-1 tile on Value() < 1, then per city each free
    /// tile of radius 2 on Value() < 1/3; a tile is offered once.
    fn resources(&mut self) {
        let mods = resource_modifiers();
        let mut visited = vec![false; self.n];
        let mut cities = Vec::new();
        let mut a = Vec::with_capacity(25);
        for c in 0..self.n {
            if self.improvement[c] != CITY {
                continue;
            }
            area(self.w, c, 1, false, &mut a);
            for &t in &a {
                if self.rng.value() < 1.0 && !visited[t] && self.resource[t] == 0 && self.improvement[t] == 0 {
                    self.add_resource(t, &mods);
                }
                visited[t] = true;
            }
            cities.push(c);
        }
        let third = 1.0f32 / 3.0;
        for c in cities {
            area(self.w, c, 2, false, &mut a);
            for &t in &a {
                if self.improvement[t] == 0 && self.rng.value() < third && !visited[t] && self.resource[t] == 0 {
                    self.add_resource(t, &mods);
                }
                visited[t] = true;
            }
        }
    }

    /// The generator's shuffle: for i < len - 1 swap i with Next(i, len).
    fn shuffle(&mut self, list: &mut [usize]) {
        for i in 0..list.len().saturating_sub(1) {
            let j = self.rng.next_range(i as i32, list.len() as i32) as usize;
            list.swap(i, j);
        }
    }

    /// Starting resources at the capital (Bardur: game, game; the minimum count is unused): the shuffled capital ring
    /// counts matching resources first, then puts each missing one on a tile of its terrain, then converts any land.
    fn starting_resources(&mut self, cap: usize) {
        let mut want = vec![1i32, 1];
        let mut ring = Vec::with_capacity(8);
        area(self.w, cap, 1, false, &mut ring);
        self.shuffle(&mut ring);
        let mut i = 0isize;
        while (i as usize) < ring.len() {
            let r = self.resource[ring[i as usize]];
            if r != 0 {
                if let Some(j) = want.iter().position(|&x| x == r) {
                    want.remove(j);
                    ring.remove(i as usize);
                    i -= 1;
                }
            }
            i += 1;
        }
        for pass in 0..3 {
            if want.is_empty() {
                return;
            }
            let mut i = 0isize;
            while (i as usize) < want.len() {
                let reqs = RESOURCE_TERRAIN[want[i as usize] as usize];
                let terrain = reqs[self.rng.next_range(0, reqs.len() as i32) as usize];
                let pick = match pass {
                    0 => ring.iter().position(|&t| self.terrain[t] == terrain),
                    1 => ring.iter().position(|&t| !is_water(self.terrain[t])),
                    _ => if ring.is_empty() { None } else { Some(0) },
                };
                if let Some(j) = pick {
                    let t = ring[j];
                    if pass > 0 {
                        self.terrain[t] = terrain;
                    }
                    self.resource[t] = want[i as usize];
                    want.remove(i as usize);
                    ring.remove(j);
                    i -= 1;
                }
                i += 1;
            }
        }
    }

    /// Ruins: amount + 1 ruins along a shuffled tile list, not on corners, not within 1 of a city, 2 of a ruin or
    /// 2 of a capital.
    fn ruins(&mut self, amount: i32) {
        let mut list: Vec<usize> = (0..self.n).collect();
        self.shuffle(&mut list);
        let corners = corners(self.w);
        let mut placed = 0;
        for t in list {
            if corners.contains(&t)
                || self.any_in(t, 1, |b, a| b.improvement[a] == CITY)
                || self.any_in(t, 2, |b, a| b.improvement[a] == RUIN)
                || self.any_in(t, 2, |b, a| b.improvement[a] == CITY && b.caps.contains(&a)) {
                continue;
            }
            self.improvement[t] = RUIN;
            placed += 1;
            if placed > amount {
                break;
            }
        }
    }
}

fn corners(w: usize) -> [usize; 4] {
    [w * w - 1, 0, w * (w - 1), w - 1]
}

/// One generator attempt on the Drylands preset (wetness 0, no smoothing or suburbs, post-terrain city
/// density 1, richness 1) for `np` Bardur seats: the board the game makes from `seed`, before the starting units.
pub fn generate_internal(seed: i32, size: usize, np: usize) -> Layout {
    let n = size * size;
    let mut b = Board {
        w: size, n, terrain: vec![1; n], improvement: vec![0; n], resource: vec![0; n], climate: vec![false; n],
        continent: vec![-1; n], caps: Vec::with_capacity(np), rng: DotNetRandom::new(seed), scratch: Vec::with_capacity(25),
    };
    b.capital_positions(np);
    b.terrain_from_noise();
    let villages = ((size as f32 * 0.33f32) as i32).pow(2) - np as i32;
    b.post_terrain_cities(villages);
    b.climates();
    b.terrain();
    b.lighthouses();
    b.resources();
    for k in 0..np {
        b.starting_resources(b.caps[k]);
    }
    b.ruins((n / 40) as i32);
    Layout { terrain: b.terrain, structure: b.improvement, resource: b.resource, capitals: b.caps }
}

/// Map inequality: each seat's turf grows from its capital by lowest path cost (mountain 4,
/// other land 1; first-listed minimum pops, ties go to the earlier seat); turf tiles score by city / ruin / land and
/// distance; the result is the highest turf score over the lowest.
pub fn inequality(l: &Layout, size: usize) -> f32 {
    let (n, np) = (size * size, l.capitals.len());
    let mut visited = vec![false; n];
    let mut next: Vec<(usize, i32, usize)> = Vec::new();
    let mut turfs: Vec<Vec<(usize, i32)>> = vec![Vec::new(); np];
    let mut a = Vec::with_capacity(8);
    let mut grow = |cur: usize, cost: i32, seat: usize, visited: &Vec<bool>, next: &mut Vec<(usize, i32, usize)>| {
        area(size, cur, 1, false, &mut a);
        for &t in a.iter() {
            if visited[t] {
                continue;
            }
            let step = if l.terrain[t] == MOUNTAIN {
                4
            } else if !is_water(l.terrain[t]) {
                1
            } else if is_water(l.terrain[cur]) {
                4
            } else if l.structure[cur] != CITY {
                1000
            } else if l.capitals.contains(&cur) {
                8
            } else {
                16
            };
            let c = step + cost;
            match next.iter_mut().find(|e| e.0 == t) {
                Some(e) => {
                    if c < e.1 || (c == e.1 && e.2 > seat) {
                        e.1 = c;
                        e.2 = seat;
                    }
                }
                None => next.push((t, c, seat)),
            }
        }
    };
    for (seat, &c) in l.capitals.iter().enumerate() {
        visited[c] = true;
        turfs[seat].push((c, 0));
        grow(c, 0, seat, &visited, &mut next);
    }
    let mut budget = n;
    while !next.is_empty() && budget > 0 {
        budget -= 1;
        let mut best = 0;
        for (i, e) in next.iter().enumerate() {
            if e.1 < next[best].1 {
                best = i;
            }
        }
        let (t, cost, seat) = next.remove(best);
        visited[t] = true;
        turfs[seat].push((t, cost));
        grow(t, cost, seat, &visited, &mut next);
    }
    let reach = size as f32 * 0.8f32 + -3.0f32;
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for turf in &turfs {
        let mut score = 0.0f32;
        for &(t, cost) in turf {
            let f = 1.0f32.min(0.0f32.max((cost as f32 + -3.0f32) / reach));
            let mut s = if l.structure[t] == CITY {
                let s = f * -90.0f32 + 100.0f32 + 0.1f32;
                if cost == 4 { s + 20.0 } else if cost == 3 { s + 40.0 } else { s }
            } else if l.structure[t] == RUIN {
                let s = f * -3.0f32 + 5.0f32 + 0.1f32;
                if cost < 5 { s + 5.0 } else { s }
            } else {
                0.1f32
            };
            if (3..=6).contains(&l.terrain[t]) {
                s += 1.5f32 - f;
            }
            score += s;
        }
        lo = if lo <= score { lo } else { score };
        if hi < score {
            hi = score;
        }
    }
    hi / if lo > 0.0 { lo } else { 0.001 }
}

/// Full generation: one attempt on each of up to 50 attempt seeds (System.Random(seed).Next() each, standing in
/// for the game's crypto RNG draw below int.MaxValue), keeping the first below the 1.01 equality limit,
/// else the lowest inequality.
pub fn generate(seed: i32, size: usize, np: usize) -> (Layout, i32, f32) {
    let mut seeds = DotNetRandom::new(seed);
    let mut best: Option<(Layout, i32, f32)> = None;
    for _ in 0..EQUALITY_ITERATIONS {
        let s = seeds.next_int();
        let l = generate_internal(s, size, np);
        let q = inequality(&l, size);
        let better = best.as_ref().map_or(true, |b| q < b.2);
        if better {
            best = Some((l, s, q));
        }
        if q < EQUALITY_LIMIT {
            break;
        }
    }
    best.unwrap()
}

pub fn layout(seed: i32, g: &Geom) -> Layout {
    layout_players(seed, g, 2)
}

/// The game's Drylands board for `np` Bardur seats (2..=4), one capital domain each.
pub fn layout_players(seed: i32, g: &Geom, np: usize) -> Layout {
    assert!((2..=4).contains(&np));
    generate(seed, g.size, np).0
}

/// maps/state.py::initial_state, bitwise mapgen.reset after the layout step.
pub fn initial_state(seed: u32, terrain: &[i32], structure: &[i32], resource: &[i32], caps: &[usize], g: &Geom,
                     max_turns: i32) -> State {
    let (size, n) = (g.size, g.n);
    let np = caps.len();
    let mut s = State::empty_players(size, np);
    s.seed = seed;
    s.initial_seed = seed;
    s.reference_max_turns = max_turns;
    s.tiles.terrain.copy_from_slice(terrain);
    s.tiles.structure.copy_from_slice(structure);
    s.tiles.resource.copy_from_slice(resource);
    for t in 0..n {
        // owner = max over seats of (area ? seat+1 : 0)
        let mut owner = 0;
        for k in 0..np {
            if g.d(caps[k], t) <= 1 {
                owner = owner.max(k as i32 + 1);
            }
        }
        s.tiles.owner[t] = owner;
        s.tiles.city[t] = if owner > 0 { caps[(owner - 1) as usize] as i32 } else { -1 };
    }
    for k in 0..np {
        let c = caps[k];
        let owner = k as i32 + 1;
        s.tiles.capital[c] = owner;
        s.cities.owner[c] = owner;
        s.cities.level[c] = 1;
        s.cities.production[c] = 1;
        s.cities.border[c] = 1;
        for t in 0..n {
            s.cities.territory[c * n + t] = g.d(c, t) <= 1;
        }
        s.cities.order[c] = 1;
        s.units.owner[c] = owner;
        s.units.kind[c] = 2;
        s.units.health[c] = 10.0;
        s.units.home[c] = c as i32 + 1;
        s.units.prev_x[c] = (c % size) as i32;
        s.units.prev_y[c] = (c / size) as i32;
        s.units.direction[c] = crate::military::spawn_direction(size, c);
    }
    for k in 0..np {
        s.players.stars[k] = 5;
        s.players.tech[k][0] = true;
        s.players.tech[k][15] = true;
        s.players.score[k] = 390;
    }
    for k in 0..np {
        update_exploration(&mut s, k as i32 + 1, g);
    }
    s
}

pub fn reset(seed: i32, g: &Geom, max_turns: Option<i32>) -> State {
    reset_players(seed, g, max_turns, 2)
}

pub fn reset_players(seed: i32, g: &Geom, max_turns: Option<i32>, np: usize) -> State {
    let board = layout_players(seed, g, np);
    initial_state(seed as u32, &board.terrain, &board.structure, &board.resource, &board.capitals, g,
                  max_turns.unwrap_or(round_limit(g.size)))
}
