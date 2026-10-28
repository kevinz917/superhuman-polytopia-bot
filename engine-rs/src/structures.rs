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
            let mut sum = 0;
            for &j in &g.adj[i] {
                let j = j as usize;
                if t.owner[i] == t.owner[j] && t.owner[i] > 0 && r.structures.adjacent[50][t.structure[j] as usize] {
                    sum += levels[j];
                }
            }
            out[i] = sum.min(8);
        }
    }
    out
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
        add_population_vector(s, &amount, 1, true);
        add_population_vector(s, &amount, 2, true);
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

pub fn connections(s: &mut State, owner: i32, g: &Geom, native: bool) {
    let n = s.n;
    let (cap, exists) = capital_index(s, owner);
    let t = &s.tiles;
    let po = (owner - 1) as usize;
    let friendly: Vec<bool> = (0..n)
        .map(|i| t.owner[i] == 0 || t.owner[i] == owner || (native && s.players.peace[po][s.pclip(t.owner[i])]))
        .collect();
    let nodes: Vec<bool> = (0..n)
        .map(|i| (t.road[i] || t.structure[i] == 48 || t.structure[i] == 8 || (s.cities.owner[i] > 0 && friendly[i])) && friendly[i])
        .collect();
    let ports: Vec<usize> = (0..n).filter(|&i| t.structure[i] == 8).collect();
    let water: Vec<bool> = (0..n).map(|i| t.terrain[i] == 1 || t.terrain[i] == 2).collect();
    // port_edges[p] = ports reachable from port p through friendly water within the reference's distance rule.
    let mut port_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    for &p in &ports {
        seen.fill(false);
        seen[p] = true;
        stack.clear();
        stack.push(p);
        while let Some(x) = stack.pop() {
            if g.d(p, x) < 4 {
                for &y in &g.adj[x] {
                    let y = y as usize;
                    if !seen[y] && water[y] && friendly[y] {
                        seen[y] = true;
                        stack.push(y);
                    }
                }
            }
        }
        for &q in &ports {
            if seen[q] && g.d(p, q) <= 4 {
                port_edges[p].push(q);
            }
        }
    }
    let mut reached = vec![false; n];
    if exists {
        reached[cap] = true;
        stack.clear();
        stack.push(cap);
        while let Some(x) = stack.pop() {
            for &y in &g.adj[x] {
                let y = y as usize;
                if !reached[y] && nodes[y] {
                    reached[y] = true;
                    stack.push(y);
                }
            }
            for &q in &port_edges[x] {
                if !reached[q] {
                    reached[q] = true;
                    stack.push(q);
                }
            }
        }
    }
    let mut amount = vec![0i32; n];
    let mut total = 0i32;
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
            let mut pr = 0u128;
            let row = &c.territory[city * n..city * n + n];
            for tile in 0..n {
                if row[tile] {
                    pr |= 1u128 << t.structure[tile];
                    econ[tile] = true;
                }
            }
            tribe_has |= pr;
            self.own_cities.push(city);
            self.present.push(pr);
        }
        self.econ_list.clear();
        for tile in 0..n {
            // the tile itself must be the player's, whatever the city rows say
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
        // ImprovementAbility Unique (PlayerState.builtUniqueImprovements): a monument is blocked only by the player
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
    // a per-city kind has room iff some own city holding this tile lacks it
    let mut holders = 0usize;
    let mut present_all = !0u128;
    for (k, &city) in ctx.own_cities.iter().enumerate() {
        if c.territory[city * s.n + tile] {
            holders += 1;
            present_all &= ctx.present[k];
        }
    }
    let room = if holders > 0 { !rs.per_city_bits | !present_all } else { 0 };
    // every City tile counts as having a road, villages included, so no road goes there
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

/// Bridge direction (from the true terrain): 1 for a north-south span (land north and
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

/// A Bridge is Freelance like a road: an unlocked bridge on a neutral, explored,
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
    if kind == 3 || kind == 8 || kind == 48 {
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

/// Structures Destroy removes: not cities, ruins or lighthouses.
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
    if kind == 3 || kind == 8 || kind == 48 {
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

pub fn income(s: &State, owner: i32, r: &Rules, g: &Geom) -> i32 {
    let n = s.n;
    let (t, c, u) = (&s.tiles, &s.cities, &s.units);
    // the territory bonus is the markets' value; without a market it is zero
    let markets = t.structure.contains(&50);
    let market_value: Vec<i32> = if markets {
        let levels = production_levels(s, r, g);
        (0..n).map(|i| if t.structure[i] == 50 { levels[i] * r.structures.stars[50] } else { 0 }).collect()
    } else {
        Vec::new()
    };
    let mut total = 0i32;
    if markets {
        // income sums the work of every tile: each owned
        // market pays on its own, whatever happens to its city (a siege or negative progress stops only the city)
        total += (0..n).filter(|&i| t.owner[i] == owner).map(|i| market_value[i]).sum::<i32>();
    }
    for city in 0..n {
        if c.owner[city] != owner {
            continue;
        }
        // any unit of another player on the city stops its work, a hidden one included
        let siege = (u.owner[city] != 0 && u.owner[city] != c.owner[city]) || c.riot[city];
        if siege {
            continue;
        }
        let mut production = c.production[city] + (t.capital[city] == c.owner[city] && t.capital[city] > 0) as i32;
        if c.progress[city] < 0 {
            production = (production + c.progress[city]).max(0);
        }
        total += production;
    }
    let p = (owner - 1) as usize;
    let np = s.np;
    let embassy_count = (0..np).map(|k| s.players.embassy[p][k] as i32).sum::<i32>()
        + (0..np).map(|k| s.players.embassy[k][p] as i32).sum::<i32>();
    // embassies exist only in two-seat games
    total + embassy_count * if np == 2 && s.players.peace[p][1 - p] { 4 } else { 2 }
}

pub fn _unused(s: &State) {
    let mut v = vec![false; s.n];
    in_territory(s, &mut v);
    let _ = add_population;
}
