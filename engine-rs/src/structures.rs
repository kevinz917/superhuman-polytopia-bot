//! gpu/structures.py: buildings, monuments, temples, road networks, income.

use crate::economy::{add_population, add_population_vector, economic_tiles, in_territory};
use crate::geom::Geom;
use crate::rules::{bits128, Rules};
use crate::state::State;

#[inline]
pub fn temple(kind: i32) -> bool {
    (17..=20).contains(&kind)
}

#[inline]
pub fn score_of(s: &State, r: &Rules, t: usize) -> i32 {
    let k = s.tiles.structure[t];
    if temple(k) { 100 * s.tiles.structure_level[t] } else { r.structures.score[k as usize] }
}

/// Native multiplier levels from adjacent owned compatible improvements.
pub fn production_levels(s: &State, r: &Rules, g: &Geom) -> Vec<i32> {
    let n = s.n;
    let t = &s.tiles;
    let mut levels = vec![0i32; n];
    for i in 0..n {
        let k = t.structure[i] as usize;
        let multiplier = r.structures.any_adjacent[k] && t.structure[i] != 50;
        if multiplier {
            let mut count = 0;
            for &j in &g.adj[i] {
                let j = j as usize;
                if t.owner[i] == t.owner[j] && t.owner[i] > 0 && r.structures.adjacent[k][t.structure[j] as usize] {
                    count += 1;
                }
            }
            levels[i] = count;
        } else {
            levels[i] = t.structure_level[i];
        }
    }
    if !t.structure.contains(&50) {
        return levels;
    }
    let mut out = levels.clone();
    for i in 0..n {
        if t.structure[i] == 50 {
            out[i] = market_level(s, r, g, i, &levels);
        }
    }
    out
}

/// improvement level of a Market at `i` given its neighbours' levels: adjacency bonus at
/// adds, for every same-owner neighbour with a Windmill, Forge or Sawmill, that neighbour's
/// level raised to at least 1 (`if (level < 2) level = 1`;
/// the Market is neither Picky nor Simple), capped at the Market's max level 8.
#[inline]
pub fn market_level(s: &State, r: &Rules, g: &Geom, i: usize, levels: &[i32]) -> i32 {
    let t = &s.tiles;
    let mut sum = 0;
    for &j in &g.adj[i] {
        let j = j as usize;
        if t.owner[i] == t.owner[j] && t.owner[i] > 0 && r.structures.adjacent[50][t.structure[j] as usize] {
            sum += levels[j].max(1);
        }
    }
    sum.min(8)
}

pub fn population_contribution(s: &State, r: &Rules, g: &Geom) -> Vec<i32> {
    contribution_from(s, r, &production_levels(s, r, g))
}

/// population_contribution given production_levels(s).
fn contribution_from(s: &State, r: &Rules, levels: &[i32]) -> Vec<i32> {
    (0..s.n)
        .map(|i| {
            let k = s.tiles.structure[i] as usize;
            r.structures.population[k] * if r.structures.any_adjacent[k] { levels[i] } else { 1 }
        })
        .collect()
}

/// structures.adjust_building_population(before, after): `before_contribution` is population_contribution of the
/// pre-action state.
pub fn adjust_building_population(before_contribution: &[i32], s: &mut State, r: &Rules, g: &Geom) {
    let n = s.n;
    let levels = production_levels(s, r, g);   // s is unchanged until structure_level is set below
    let after = contribution_from(s, r, &levels);
    let mut amount = vec![0i32; n];
    for tile in 0..n {
        let city = s.tiles.city[tile];
        if city >= 0 && (city as usize) < n {
            let city = city as usize;
            if s.cities.owner[city] == s.tiles.owner[tile] {
                amount[city] += after[tile] - before_contribution[tile];
            }
        }
    }
    s.tiles.structure_level = levels;
    if amount.iter().any(|&a| a != 0) {   // all-zero amounts change nothing
        for owner in 1..=s.np as i32 {
            add_population_vector(s, &amount, owner, true);
        }
    }
}

pub fn capital_index(s: &State, owner: i32) -> (usize, bool) {
    let mut best = i32::MAX;
    let mut idx = 0usize;
    let mut exists = false;
    for i in 0..s.n {
        if s.cities.owner[i] == owner && s.tiles.capital[i] == owner {
            exists = true;
            if s.cities.order[i] < best {
                best = s.cities.order[i];
                idx = i;
            }
        }
    }
    (idx, exists)
}

/// current capital coordinates: the player's start tile while they own it, else their first city
/// in tile order; None without cities.
pub fn current_capital(s: &State, owner: i32) -> Option<usize> {
    let (cap, exists) = capital_index(s, owner);
    if exists {
        return Some(cap);
    }
    (0..s.n).find(|&i| s.cities.owner[i] == owner)
}

/// game logic v28 route data of the improvements that open routes (route opener:
/// `routes.Count > 0`): the terrain bitset of `routes`, `range`, and whether it has the Network ability (15). Port
/// (routes ocean, water, wetland; range 5), Outpost (ice; 5), Mycelium (field, forest, wetland, ice; 3; Network) and
/// Bridge (water; 1). Only Port and Bridge exist for Bardur.
#[inline]
pub fn route_data(kind: i32) -> Option<(u16, i32, bool)> {
    const fn bits(t: &[i32]) -> u16 {
        let mut b = 0u16;
        let mut k = 0;
        while k < t.len() {
            b |= 1 << t[k];
            k += 1;
        }
        b
    }
    match kind {
        8 => Some((bits(&[2, 1, 7]), 5, false)),
        33 => Some((bits(&[6]), 5, false)),
        39 => Some((bits(&[3, 5, 7, 6]), 3, true)),
        48 => Some((bits(&[1]), 1, false)),
        _ => None,
    }
}

/// unlocked movements as a terrain bitset: the `movementUnlocks` keys of every known tech (game data v28:
/// tech 0 field/forest/ice/wetland, 10 and 27/40/45 water, 13/25 and 43/46 ocean, 20 mountain), plus Water when
/// Ocean is in without it.
pub fn unlocked_movements(tech: &[bool; 128]) -> u16 {
    const UNLOCKS: [(usize, &[i32]); 10] = [(0, &[3, 5, 6, 7]), (10, &[1]), (13, &[2]), (20, &[4]), (25, &[2]), (27, &[1]),
                                            (40, &[1]), (43, &[2]), (45, &[1]), (46, &[2])];
    let mut out = 0u16;
    for (k, terrain) in UNLOCKS {
        if tech[k] {
            for &t in terrain {
                out |= 1 << t;
            }
        }
    }
    if out & (1 << 2) != 0 {
        out |= 1 << 1;
    }
    out
}

/// connectable: a road (the raw the tile has road), a City improvement (any city or village, whoever owns
/// it) or a route opener.
#[inline]
fn connectable(s: &State, i: usize) -> bool {
    s.tiles.road[i] || s.tiles.structure[i] == 1 || route_data(s.tiles.structure[i]).is_some()
}

/// matching transport path: refused when `other` belongs to a player
/// who is neither `cur`'s owner nor at peace with that owner; ownership is checked only between the two tiles, and a
/// neutral `cur` passes (try get player finds no player, so the peace test is skipped). Otherwise two
/// connectable tiles (is connectable) or two route tiles (has route) link.
#[inline]
fn transport_step(s: &State, cur: usize, other: usize, native: bool) -> bool {
    let t = &s.tiles;
    let (a, b) = (t.owner[cur], t.owner[other]);
    if b != 0 && b != a && a > 0 && !(native && s.players.peace[s.pclip(a)][s.pclip(b)]) {
        return false;
    }
    (connectable(s, cur) && connectable(s, other)) || (t.route[cur] && t.route[other])
}

/// Tiles reached from `owner`'s start tile by find connected cities:
/// a flood over the 8 neighbours (area preallocated radius 1) along has matching transport path, starting only if the
/// player owns their start tile. Returns the reached tiles and the start tile with whether they hold it.
pub fn reachable(s: &State, owner: i32, g: &Geom, native: bool) -> (Vec<bool>, usize, bool) {
    let n = s.n;
    let (cap, exists) = capital_index(s, owner);
    let mut reached = vec![false; n];
    if exists {
        reached[cap] = true;
        let mut stack = vec![cap];
        while let Some(x) = stack.pop() {
            for &y in &g.adj[x] {
                let y = y as usize;
                if !reached[y] && transport_step(s, x, y, native) {
                    reached[y] = true;
                    stack.push(y);
                }
            }
        }
    }
    (reached, cap, exists)
}

/// Reusable buffers of router_path.
struct RouterScratch {
    cost: Vec<i32>,
    came_from: Vec<i32>,
    open: crate::pathfinding::OpenList,
    path: Vec<usize>,
}

/// path(map, origin, dest, max cost = the origin's range, create router settings(player,
/// allowed)), as update routes calls it (version 123). Router settings:
/// no unit, occupied tiles allowed, enemy tiles not, peace partners' tiles allowed (version > 59), port required to
/// enter water from land, transport paths followed. is tile accessible: the tile must
/// be the player's, neutral or a peace partner's, and explored by the player; a Bridge is entered unless its terrain
/// is not allowed and the step crosses its span; a water tile entered from land must be an own or peace partner's
/// port (Embark) and the player able to embark (can player embark); then a City tile or allowed terrain.
/// movement cost, no unit, should follow transport paths): 5 onto a road, a Bridge,
/// a City or a tile already on a route (has route, set by the paths found earlier in the same update), else 10.
/// A* as the unit search (pathfinding::search): priority Distance(tile, dest) + cost, the start at its Manhattan
/// distance, no closed set, expansion while cost < range x 10, at most 1000 pops. Fills `sc.path` (dest to origin).
#[allow(clippy::too_many_arguments)]
fn router_path(s: &State, player: i32, embark: bool, origin: usize, dest: usize, range: i32, allowed: u16, route: &[bool],
               sc: &mut RouterScratch) -> bool {
    let (n, size) = (s.n, s.size as i32);
    let t = &s.tiles;
    let p = (player - 1) as usize;
    let peace = |o: i32| o > 0 && o != player && s.players.peace[p][s.pclip(o)];
    let water = |i: usize| t.terrain[i] == 1 || t.terrain[i] == 2;
    let has = |terrain: i32| (0..16).contains(&terrain) && allowed >> terrain & 1 == 1;
    // is allowed to finish on tile: occupied tiles allowed; a City of a player at peace or in cool-down is not
    let o = t.owner[dest];
    if t.structure[dest] == 1 && o > 0 && o != player && (peace(o) || s.players.truce_breaker[p][s.pclip(o)] != 0) {
        return false;
    }
    let accessible = |tile: usize, from: usize| -> bool {
        let o = t.owner[tile];
        if o != 0 && o != player && !peace(o) {
            return false;
        }
        if !t.explored[p][tile] {
            return false;
        }
        if t.structure[tile] == 48 {
            if !has(t.terrain[tile]) {
                // non allowed bridge direction: not straight across the span
                let (dx, dy) = ((from % s.size) as i32 - (tile % s.size) as i32, (from / s.size) as i32 - (tile / s.size) as i32);
                match bridge_span(s, tile) {
                    1 if dy == 0 => return false,
                    2 if dx == 0 => return false,
                    _ => {}
                }
            }
            return true;
        }
        if water(tile) && !water(from) {
            let friendly = o == player || peace(o);
            if !(t.structure[tile] == 8 && friendly) || !embark {
                return false;
            }
        }
        t.structure[tile] == 1 || has(t.terrain[tile])
    };
    let cost = |tile: usize| if t.road[tile] || t.structure[tile] == 48 || t.structure[tile] == 1 || route[tile] { 5 } else { 10 };
    let (tx, ty) = ((dest % s.size) as i32, (dest / s.size) as i32);
    let dist = |i: usize| {
        let (dx, dy) = (tx - (i % s.size) as i32, ty - (i / s.size) as i32);
        ((dy * dy + dx * dx) as f32).sqrt()
    };
    sc.cost[..n].fill(0);
    sc.came_from[..n].fill(-1);
    sc.open.clear();
    let (sx, sy) = ((origin % s.size) as i32, (origin / s.size) as i32);
    sc.open.push(((tx - sx).abs() + (ty - sy).abs()) as f32, origin);
    let max_cost = range * 10;
    let mut found = false;
    for _ in 0..1000 {
        let Some(at) = sc.open.pop() else { break };
        if at == dest {
            found = true;
            break;
        }
        let here = sc.cost[at];
        if here < max_cost {
            let (ax, ay) = ((at % s.size) as i32, (at / s.size) as i32);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let (x, y) = (ax + dx, ay + dy);
                    if (dx == 0 && dy == 0) || x < 0 || y < 0 || x >= size || y >= size {
                        continue;
                    }
                    let next = (y * size + x) as usize;
                    if !accessible(next, at) {
                        continue;
                    }
                    let c = cost(next) + here;
                    let old = sc.cost[next];
                    if old < 1 || c < old {
                        sc.cost[next] = c;
                        sc.came_from[next] = at as i32;
                        sc.open.push(dist(next) + c as f32, next);
                    }
                }
            }
        }
        if sc.open.is_empty() {
            break;
        }
    }
    sc.path.clear();
    if !found {
        return false;
    }
    // reconstruct path: destination back to the start, both included
    let mut at = dest;
    sc.path.push(at);
    for _ in 0..=n {
        match sc.came_from[at] {
            -1 => return false,
            c => at = c as usize,
        }
        sc.path.push(at);
        if at == origin {
            return true;
        }
    }
    false
}

/// update routes: reset routes clears
/// every tile's has route, then for each player in turn order, over their empire tiles (player empire tiles: tile
/// owner = the player, tile order) whose improvement opens routes, every pair i < j is tried with
/// find path between routers(router i, router j): no route beyond router i's range
/// (Chebyshev); the allowed terrain is each of router i's route terrains that router j's routes and the player's
/// movements (unlocked movements) both contain, or every one when j is a City; a path found marks all its tiles
/// has route. A Network router (Mycelium) also tries each of the player's City tiles. has route is one flag per tile
/// for all players, so later paths (any player) pay 5 on earlier route tiles.
pub fn update_routes(s: &mut State, r: &Rules) {
    let n = s.n;
    let mut route = vec![false; n];
    let mut sc = RouterScratch { cost: vec![0; n], came_from: vec![-1; n], open: Default::default(), path: Vec::with_capacity(n) };
    for player in 1..=s.np as i32 {
        let p = (player - 1) as usize;
        let t = &s.tiles;
        let routers: Vec<usize> = (0..n).filter(|&i| t.owner[i] == player && route_data(t.structure[i]).is_some()).collect();
        if routers.is_empty() {
            continue;
        }
        let cities: Vec<usize> = (0..n).filter(|&i| t.owner[i] == player && t.structure[i] == 1).collect();
        let moves = unlocked_movements(&s.players.tech[p]);
        // can player embark: a known tech unlocks an improvement with the Embark ability (the Port)
        let embark = (0..128).any(|k| s.players.tech[p][k] && r.structures.tech_unlocks[k] >> 8 & 1 == 1);
        for a in 0..routers.len() {
            let origin = routers[a];
            let (from_routes, range, network) = route_data(s.tiles.structure[origin]).unwrap();
            let network_targets: &[usize] = if network { &cities } else { &[] };
            for &dest in routers[a + 1..].iter().chain(network_targets) {
                let (ox, oy, dx, dy) = (origin % s.size, origin / s.size, dest % s.size, dest / s.size);
                if (ox.abs_diff(dx)).max(oy.abs_diff(dy)) as i32 > range {
                    continue;
                }
                let city = s.tiles.structure[dest] == 1;
                let to_routes = route_data(s.tiles.structure[dest]).map_or(0, |d| d.0);
                let allowed = if city { from_routes } else { from_routes & to_routes & moves };
                if allowed == 0 {
                    continue;
                }
                if router_path(s, player, embark, origin, dest, range, allowed, &route, &mut sc) {
                    for &k in &sc.path {
                        route[k] = true;
                    }
                }
            }
        }
    }
    s.tiles.route = route;
}

pub fn connections(s: &mut State, owner: i32, g: &Geom, native: bool) {
    let n = s.n;
    let (reached, cap, exists) = reachable(s, owner, g, native);
    let mut amount = vec![0i32; n];
    let mut total = 0i32;
    if !exists {
        // the player no longer holds their start tile: every connected city of theirs disconnects
        // (disconnect city): -1 to it and -1 to the start tile, whoever holds it now
        for c in 0..n {
            if s.cities.owner[c] == owner && s.cities.connected[c] {
                amount[c] = -1;
                total -= 1;
                s.cities.connected[c] = false;
            }
        }
        if total == 0 {
            return;
        }
        add_population_vector(s, &amount, owner, native);
        if let Some(start) = (0..n).find(|&i| s.tiles.capital[i] == owner && s.cities.owner[i] > 0) {
            let mut v = vec![0i32; n];
            v[start] = total;
            add_population_vector(s, &v, s.cities.owner[start], native);
        }
        return;
    }
    for c in 0..n {
        let own = s.cities.owner[c] == owner && c != cap && exists;
        if own {
            let delta = reached[c] as i32 - s.cities.connected[c] as i32;
            amount[c] = delta;
            total += delta;
            s.cities.connected[c] = reached[c];
        }
    }
    amount[cap] += total;
    let pop_owner = if native { owner } else { s.current_player };
    add_population_vector(s, &amount, pop_owner, native);
}

pub fn monument_conditions(s: &State, owner: i32, r: &Rules, g: &Geom, out: &mut [bool; 128]) {
    let c = &s.cities;
    let (mut connected, mut level5) = (0usize, false);
    for i in 0..s.n {
        if c.owner[i] == owner {
            connected += c.connected[i] as usize;
            level5 |= c.level[i] >= 5;
        }
    }
    let bits = monument_bits(s, owner, r, g, connected, level5);
    for k in 0..128 {
        out[k] = bits >> k & 1 == 1;
    }
}

/// monument_conditions as a bitset, given the owner's connected-city count and whether a city of theirs has level 5.
pub fn monument_bits(s: &State, owner: i32, r: &Rules, g: &Geom, connected: usize, level5: bool) -> u128 {
    let p = &s.players;
    let pi = (owner - 1) as usize;
    let tech = &p.tech[pi];
    let all_tech = bits128(tech) | !r.tech_exists_bits == !0;
    let explorer = g.corners.iter().all(|&t| s.tiles.explored[pi][t]) && s.fog;
    let bit = |k: u32, v: bool| (v as u128) << k;
    // the Network task (Grand Bazaar) is enabled by the first city connection, not by the Roads tech
    bit(23, tech[21] && p.pacifist[pi] >= 5) | bit(24, all_tech) | bit(25, connected >= 5)
        | bit(26, tech[5] && p.stars[pi] >= 100) | bit(27, p.kills[pi] >= 10) | bit(28, level5) | bit(29, explorer)
}

pub fn remember_monuments(s: &mut State, r: &Rules, g: &Geom) {
    // one pass over the cities for every seat's connected counts and level-5 flags
    let c = &s.cities;
    let np = s.np as i32;
    let (mut connected, mut level5) = ([0usize; crate::state::MAXP], [false; crate::state::MAXP]);
    for i in 0..s.n {
        let o = c.owner[i];
        if (1..=np).contains(&o) {
            connected[(o - 1) as usize] += c.connected[i] as usize;
            level5[(o - 1) as usize] |= c.level[i] >= 5;
        }
    }
    for owner in 1..=np {
        let pi = (owner - 1) as usize;
        let earned = monument_bits(s, owner, r, g, connected[pi], level5[pi]);
        let m = &mut s.players.monuments[pi];
        for k in 0..128 {
            m[k] |= earned >> k & 1 == 1;
        }
    }
}

/// Per-call inputs of structures.build_mask shared by every tile (BuildCtx::prepare).
pub struct BuildCtx {
    /// economic_tiles(s), filled by prepare(), and the same tiles as a list
    pub econ: Vec<bool>,
    pub econ_list: Vec<u16>,
    own_cities: Vec<usize>,
    /// structures present in the territory of each own city (bit = structure kind)
    present: Vec<u128>,
    unlocked: u128,
    monuments: u128,
}

const ROAD: u128 = 1 << 3;
const BRIDGE: u128 = 1 << 48;

impl BuildCtx {
    pub fn road_unlocked(&self) -> bool {
        self.unlocked & ROAD != 0
    }

    pub fn new(n: usize) -> BuildCtx {
        BuildCtx { econ: vec![false; n], econ_list: Vec::with_capacity(n), own_cities: Vec::with_capacity(n), present: Vec::with_capacity(n), unlocked: 0,
                   monuments: 0 }
    }

    /// Everything of build_mask that does not depend on the tile, including economic_tiles(s) into `econ` (one
    /// pass over each own city's territory row gives both the economy and the structures present per city).
    pub fn prepare(&mut self, s: &State, r: &Rules, g: &Geom) {
        let n = s.n;
        let (t, c, p) = (&s.tiles, &s.cities, &s.players);
        let cp = s.current_player;
        let pi = s.p();
        let rs = &r.structures;
        self.own_cities.clear();
        self.present.clear();
        let econ = &mut self.econ[..n];
        econ.fill(false);
        let mut tribe_has = 0u128;
        let (mut connected, mut level5) = (0usize, false);
        for city in 0..n {
            if c.owner[city] != cp {
                continue;
            }
            connected += c.connected[city] as usize;
            level5 |= c.level[city] >= 5;
            // the city's structures: those on the tiles it rules (a Limited kind is one per ruling city)
            let mut pr = 0u128;
            let row = &c.territory[city * n..city * n + n];
            for tile in 0..n {
                if row[tile] {
                    econ[tile] = true;
                }
                if t.city[tile] == city as i32 {
                    pr |= 1u128 << t.structure[tile];
                }
            }
            tribe_has |= pr;
            self.own_cities.push(city);
            self.present.push(pr);
        }
        self.econ_list.clear();
        for tile in 0..n {
            // can build: the tile itself must be the player's, whatever the city rows say
            econ[tile] = econ[tile] && crate::economy::unit_allows_work(s, tile) && t.owner[tile] == cp;
            if econ[tile] {
                self.econ_list.push(tile as u16);
            }
        }
        let mut tech_ok = 0u128;
        let mut known = bits128(&p.tech[pi]);
        while known != 0 {
            tech_ok |= rs.tech_unlocks[known.trailing_zeros() as usize];
            known &= known - 1;
        }
        self.unlocked = tech_ok & rs.static_ok & rs.cost_le.le(p.stars[pi]) & !(rs.per_tribe_bits & tribe_has);
        let earned = monument_bits(s, cp, r, g, connected, level5) | bits128(&p.monuments[pi]);
        // improvement ability Unique (the player state built unique improvements): a monument is blocked only by the player
        // having built it, not by one standing in a captured city's territory
        self.monuments = earned & !bits128(&p.built[pi]);
    }
}

/// Buildable structure kinds on `tile` as a bitset: one row of structures.build_mask (native profile), with each
/// of its four terms evaluated for all 128 kinds at once as bitset algebra:
///   (a) unlocked & terrain & empty & adjacency & city room   (b) neutral road   (c) the tile's resource building
///   (d) earned monuments on free land; then the bridge span test.
pub fn build_row(s: &State, r: &Rules, g: &Geom, ctx: &BuildCtx, tile: usize) -> u128 {
    let (t, c, p) = (&s.tiles, &s.cities, &s.players);
    let cp = s.current_player;
    let pi = s.p();
    let rs = &r.structures;
    let terrain = rs.terrain_bits[t.terrain[tile].clamp(0, 15) as usize];
    let b = (if neutral_road(s, r, ctx, tile) { ROAD } else { 0 }) | (if neutral_bridge(s, r, ctx, tile) { BRIDGE } else { 0 });
    if !ctx.econ[tile] {
        return b; // every other term carries econ[tile]
    }
    let free = t.structure[tile] == 0;
    let mut adjacency = rs.no_adjacent_bits;
    for &j in &g.adj[tile] {
        let j = j as usize;
        if t.owner[j] == cp {
            adjacency |= rs.adjacent_rev[t.structure[j] as usize];
        }
    }
    // a per-city (Limited) kind has room iff the city ruling this tile (tile.ruling city coordinates) lacks it;
    // territory rows can overlap, the ruler is one city
    let ruler = t.city[tile];
    let room = match ctx.own_cities.iter().position(|&city| city as i32 == ruler) {
        Some(k) => !rs.per_city_bits | !ctx.present[k],
        None => 0,
    };
    // road is true on every City tile, villages included, so no road goes there
    let road_ok = !t.road[tile] && c.owner[tile] == 0 && t.structure[tile] != 1;
    let empty = (if free { !ROAD } else { 0 }) | (if road_ok { ROAD } else { 0 });
    let a = ctx.unlocked & terrain & empty & adjacency & room;
    let (res_tech, res_build) = match t.resource[tile] { 2 => (8, 5), 5 => (23, 21), 7 => (0, 37), _ => (0, 0) };
    let resource = free && res_build > 0 && p.tech[pi][res_tech] && p.stars[pi] >= rs.cost[res_build];
    let d = if free { terrain & ctx.monuments } else { 0 };
    let mut v = a | b | ((resource as u128) << res_build) | d;
    if v & BRIDGE != 0 && bridge_span(s, tile) == 0 {
        v &= !BRIDGE;
    }
    v
}

/// possible bridge direction (from the true terrain): 1 for a north-south span (land north and
/// south), 2 for east-west (land east and west), 0 for none; north-south wins when both hold.
#[inline]
pub fn bridge_span(s: &State, tile: usize) -> u8 {
    let size = s.size;
    let (x, y) = (tile % size, tile / size);
    let land = |i: usize| s.tiles.terrain[i] != 1 && s.tiles.terrain[i] != 2;
    if y > 0 && y + 1 < size && land(tile - size) && land(tile + size) {
        1
    } else if x > 0 && x + 1 < size && land(tile - 1) && land(tile + 1) {
        2
    } else {
        0
    }
}

/// can build (a Bridge is Freelance like a road): an unlocked bridge on a neutral, explored,
/// structure-less water tile with a span, free of other players' visible units.
#[inline]
pub fn neutral_bridge(s: &State, r: &Rules, ctx: &BuildCtx, tile: usize) -> bool {
    let t = &s.tiles;
    ctx.unlocked & BRIDGE != 0 && t.owner[tile] == 0 && t.explored[s.p()][tile] && t.structure[tile] == 0
        && crate::economy::unit_allows_work(s, tile)
        && r.structures.terrain_bits[t.terrain[tile].clamp(0, 15) as usize] & BRIDGE != 0 && bridge_span(s, tile) != 0
}

/// Term (b) of build_row: an unlocked road on a neutral, explored, road-less tile whose terrain allows roads.
#[inline]
pub fn neutral_road(s: &State, r: &Rules, ctx: &BuildCtx, tile: usize) -> bool {
    let t = &s.tiles;
    ctx.unlocked & ROAD != 0 && t.owner[tile] == 0 && t.explored[s.p()][tile] && !t.road[tile]
        && t.structure[tile] != 1
        && crate::economy::unit_allows_work(s, tile)
        && r.structures.terrain_bits[t.terrain[tile].clamp(0, 15) as usize] & ROAD != 0
}

/// build_mask into out[t*128 + kind] (structures.build_mask, native profile).
pub fn build_mask(s: &State, r: &Rules, g: &Geom, out: &mut [bool]) {
    let mut ctx = BuildCtx::new(s.n);
    ctx.prepare(s, r, g);
    for tile in 0..s.n {
        let bits = build_row(s, r, g, &ctx, tile);
        for kind in 0..128 {
            out[tile * 128 + kind] = bits >> kind & 1 == 1;
        }
    }
}

/// One entry of build_mask (step validation).
pub fn build_legal(s: &State, r: &Rules, g: &Geom, ctx: &mut BuildCtx, tile: usize, kind: usize) -> bool {
    ctx.prepare(s, r, g);
    build_row(s, r, g, ctx, tile) >> kind & 1 == 1
}

pub fn build(s: &mut State, target: usize, kind: usize, r: &Rules, g: &Geom) {
    let before = population_contribution(s, r, g);
    let rs = &r.structures;
    let road = kind == 3;
    let pi = s.p();
    if !road {
        s.tiles.structure[target] = kind as i32;
        s.tiles.structure_level[target] = 1;
        s.tiles.structure_turn[target] = s.turn;
    }
    s.tiles.road[target] |= road;
    s.players.stars[pi] -= rs.cost[kind].max(0);
    // build: a route opener or Bridge adds update routes, then (a road
    // too) update city connections for the builder
    let opener = route_data(kind as i32).is_some();
    if opener {
        update_routes(s, r);
    }
    if kind == 3 || opener {
        connections(s, s.current_player, g, true);
    }
    let score = if temple(kind as i32) { 100 } else { rs.score[kind] };
    let owner = s.tiles.owner[target];
    if owner > 0 {
        let o = s.pclip(owner);
        s.players.score[o] = s.players.score[o].wrapping_add(score);
    }
    s.players.built[pi][kind] |= score > 0;
    adjust_building_population(&before, s, r, g);
}

pub fn destroy_mask(s: &State, out: &mut [bool]) {
    economic_tiles(s, out);
    let econ = out[..s.n].to_vec();
    destroy_mask_with(s, &econ, out);
}

/// Structures Destroy removes (destructable): not cities, ruins or lighthouses.
#[inline]
pub fn destructible(k: i32) -> bool {
    k > 0 && k != 1 && k != 2 && k != 47
}

/// destroy_mask given economic_tiles(s).
pub fn destroy_mask_with(s: &State, econ: &[bool], out: &mut [bool]) {
    let tech3 = s.players.tech[s.p()][3];
    for t in 0..s.n {
        out[t] = econ[t] && destructible(s.tiles.structure[t]) && tech3;
    }
}

pub fn destroy(s: &mut State, target: usize, r: &Rules, g: &Geom) {
    let before = population_contribution(s, r, g);
    let kind = s.tiles.structure[target];
    let score = score_of(s, r, target);
    let owner = s.tiles.owner[target];
    s.tiles.structure[target] = 0;
    s.tiles.structure_level[target] = 0;
    s.tiles.structure_turn[target] = 0;
    s.tiles.road[target] = s.tiles.road[target] && kind != 3 && kind != 48;
    if owner > 0 {
        let o = s.pclip(owner);
        s.players.score[o] = s.players.score[o].wrapping_sub(score);
    }
    adjust_building_population(&before, s, r, g);
    // destroy improvement: a route opener or Bridge adds
    // update routes and update city connections for the destroyer
    let opener = route_data(kind).is_some();
    if opener {
        update_routes(s, r);
    }
    if kind == 3 || opener {
        connections(s, s.current_player, g, true);
    }
}

pub fn start_turn(s: &mut State, owner: i32) {
    let n = s.n;
    let mut gain = 0i32;
    for i in 0..n {
        let t = &mut s.tiles;
        if temple(t.structure[i]) && t.owner[i] == owner {
            let expected = (1 + (s.turn - t.structure_turn[i]).div_euclid(2)).clamp(1, 5);
            let new = expected.max(t.structure_level[i]);
            gain += new - t.structure_level[i];
            t.structure_level[i] = new;
        }
    }
    let o = (owner - 1) as usize;
    s.players.score[o] = s.players.score[o].wrapping_add(gain * 100);
}

/// The improvement level up/level down actions of start turn: update improvement level,
/// for each tile of `owner` in tile order, queues |calculate improvement level - stored|
/// actions computed from the stored levels; the action stack then runs them last queued first, each only while it is
/// still valid against the levels at that moment (valid: stored < calculate improvement level;
/// valid: stored > it), one level each, with the improvement's growth population to its
/// ruling city. Windmills, Forges, Sawmills and Markets (temples level by age in start_turn).
pub fn refresh_levels(s: &mut State, owner: i32, r: &Rules, g: &Geom) {
    let n = s.n;
    let rs = &r.structures;
    let adjacency = |s: &State, i: usize| -> i32 {
        let k = s.tiles.structure[i] as usize;
        if k == 50 {
            return market_level(s, r, g, i, &s.tiles.structure_level);
        }
        // a neighbour without levels (farm, mine, lumber hut) counts 1
        g.adj[i].iter().filter(|&&j| {
            let j = j as usize;
            s.tiles.owner[i] == s.tiles.owner[j] && s.tiles.owner[i] > 0 && rs.adjacent[k][s.tiles.structure[j] as usize]
        }).count() as i32
    };
    let mut queue: Vec<(usize, i32)> = Vec::new();
    for i in 0..n {
        let k = s.tiles.structure[i];
        if s.tiles.owner[i] == owner && rs.any_adjacent[k as usize] && !temple(k) {
            let d = adjacency(&*s, i) - s.tiles.structure_level[i];
            for _ in 0..d.abs() {
                queue.push((i, d.signum()));
            }
        }
    }
    if queue.is_empty() {
        return;
    }
    let mut amount = vec![0i32; n];
    while let Some((i, sign)) = queue.pop() {
        let (now, stored) = (adjacency(&*s, i), s.tiles.structure_level[i]);
        if (sign > 0 && stored < now) || (sign < 0 && stored > now && stored > 0) {
            s.tiles.structure_level[i] += sign;
            let city = s.tiles.city[i];
            if city >= 0 && (city as usize) < n && s.cities.owner[city as usize] == s.tiles.owner[i] {
                amount[city as usize] += sign * rs.population[s.tiles.structure[i] as usize];
            }
        }
    }
    if amount.iter().any(|&a| a != 0) {
        add_population_vector(s, &amount, owner, true);
    }
}

pub fn income(s: &State, owner: i32, r: &Rules, g: &Geom) -> i32 {
    let n = s.n;
    let (t, c, u) = (&s.tiles, &s.cities, &s.units);
    let mut total = 0i32;
    if t.structure.contains(&50) {
        // start turn pays work(tile,
        // update improvement level) for every owned tile, in tile order: each owned market pays on its
        // own, whatever happens to its city (a siege or negative progress stops only the city). The level is
        // calculate improvement level against the neighbours' *stored* levels: the improvement level up/Down actions
        // update improvement level queues run only after the whole loop (refresh_levels).
        total += (0..n)
            .filter(|&i| t.owner[i] == owner && t.structure[i] == 50)
            .map(|i| market_level(s, r, g, i, &t.structure_level) * r.structures.stars[50])
            .sum::<i32>();
    }
    let p = (owner - 1) as usize;
    // income from embassy: an embassy pays 2 (4 at peace) unless the two are at war
    let embassy_pay = |builder: usize, host: usize| -> i32 {
        if s.players.embassy[builder][host] && !crate::diplomacy::at_war(s, builder, host) {
            2 * if s.players.peace[builder][host] { 2 } else { 1 }
        } else {
            0
        }
    };
    for city in 0..n {
        if c.owner[city] != owner {
            // the builder's share: paid from the host's capital while no other player's unit stands on it
            let host = t.capital[city];
            if host > 0 && c.owner[city] == host && !(u.owner[city] != 0 && u.owner[city] != host) {
                total += embassy_pay(p, (host - 1) as usize);
            }
            continue;
        }
        // any unit of another player on the city stops its work, a hidden one included; so does a riot
        let siege = (u.owner[city] != 0 && u.owner[city] != c.owner[city]) || c.riot[city];
        if siege {
            continue;
        }
        let start = t.capital[city] == c.owner[city] && t.capital[city] > 0;
        let mut production = c.production[city] + start as i32;
        if start {
            // the host's share is part of its capital's work (work)
            production += (0..s.np).filter(|&q| q != p).map(|q| embassy_pay(q, p)).sum::<i32>();
        }
        if c.progress[city] < 0 {
            production = (production + c.progress[city]).max(0);
        }
        total += production;
    }
    total
}

pub fn _unused(s: &State) {
    let mut v = vec![false; s.n];
    in_territory(s, &mut v);
    let _ = add_population;
}
