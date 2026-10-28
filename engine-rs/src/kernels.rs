//! gpu/kernels.py: combat formula, unit health, research, movement reachability, exploration and the observation.

use crate::geom::Geom;
use crate::rules::{Rules, CARRY, CREEP, FLOAT, HIDE, INFILTRATE};
use crate::state::State;

pub const INF: i32 = 1_000_000;

/// Reference f32 damage formula with Rust's half-away-from-zero rounding; (attack, defense, attack*0.5).
#[inline]
pub fn combat(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    let divisor = if total > 0.0 { total } else { 1.0 };
    let attack = ((if total > 0.0 { (af / divisor) * aa * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    let defense = ((if dh > attack && total > 0.0 { (df / divisor) * dd * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    (attack, defense, attack * 0.5)
}

#[inline]
pub fn max_health(s: &State, r: &Rules, i: usize) -> f32 {
    let u = &s.units;
    let kind = if u.passenger[i] > 0 { u.passenger[i] } else { u.kind[i] };
    r.unit_health[(kind as usize).min(127)] + (u.veteran[i] as i32 * 5) as f32
}

pub fn research_cost(s: &State, r: &Rules, k: usize) -> i32 {
    let cp = s.current_player;
    let cities = s.cities.owner.iter().filter(|&&o| o == cp).count() as i32;
    research_cost_with(s, r, k, cities)
}

/// research_cost given the mover's city count.
#[inline]
pub fn research_cost_with(s: &State, r: &Rules, k: usize, cities: i32) -> i32 {
    let cost = 4 + cities * r.tech_tier[k];
    // tech price: Literacy multiplies by 0.66666f
    if s.players.tech[s.p()][22] { ((cost as f32) * 0.66666f32).ceil() as i32 } else { cost }
}

pub fn research_mask(s: &State, r: &Rules, out: &mut [bool]) {
    let active = !s.pending_reward() && !s.terminated && !s.truncated;
    let bits = if active { research_bits(s, r) } else { 0 };
    for k in 0..128 {
        out[k] = bits >> k & 1 == 1;
    }
}

/// research_mask without its `active` gate, as a bitset: techs reachable from a discovered one, not discovered,
/// existing and affordable (one city count, predecessor bitsets).
pub fn research_bits(s: &State, r: &Rules) -> u128 {
    let p = s.p();
    let discovered = &s.players.tech[p];
    let stars = s.players.stars[p];
    let known = crate::rules::bits128(discovered);
    let mut cand = 0u128;
    let mut rest = known;
    while rest != 0 {
        cand |= r.tech_succ[rest.trailing_zeros() as usize];
        rest &= rest - 1;
    }
    cand &= !known & r.tech_exists_bits;
    if cand == 0 {
        return 0;
    }
    let cp = s.current_player;
    let cities = s.cities.owner.iter().filter(|&&o| o == cp).count() as i32;
    let mut out = 0u128;
    while cand != 0 {
        let k = cand.trailing_zeros() as usize;
        cand &= cand - 1;
        out |= ((research_cost_with(s, r, k, cities) <= stars) as u128) << k;
    }
    out
}

pub struct MovementData {
    pub passable: Vec<bool>,
    pub terminal: Vec<bool>,
    pub road: Vec<bool>,
    pub enemy: Vec<bool>,
    pub budget: i32,
}

impl MovementData {
    pub fn new(n: usize) -> MovementData {
        MovementData { passable: vec![false; n], terminal: vec![false; n], road: vec![false; n], enemy: vec![false; n], budget: 0 }
    }
}

pub fn movement_data(s: &State, source: usize, r: &Rules, g: &Geom) -> MovementData {
    let mut md = MovementData::new(s.n);
    movement_data_into(s, source, r, g, &mut md);
    md
}

/// kernels.movement_data into reusable buffers.
pub fn movement_data_into(s: &State, source: usize, r: &Rules, g: &Geom, md: &mut MovementData) {
    let n = s.n;
    let (t, u) = (&s.tiles, &s.units);
    let owner = u.owner[source];
    let p = s.pclip(owner);
    let tech = &s.players.tech[p];
    let kind = u.kind[source] as usize;
    let aquatic = r.skills[kind][FLOAT];
    let creep = r.skills[kind][CREEP];
    let (passable, enemy, road, terminal) = (&mut md.passable, &mut md.enemy, &mut md.road, &mut md.terminal);
    for i in 0..n {
        let bridge = t.structure[i] == 48;
        let water = t.terrain[i] == 1 || t.terrain[i] == 2;
        let terrain_ok = ((t.terrain[i] != 1 || tech[10]) && (t.terrain[i] != 2 || tech[13] || tech[25]) && (t.terrain[i] != 4 || tech[20])) || bridge;
        let own_port = t.structure[i] == 8 && t.owner[i] == owner;
        let land_access = !water || bridge || own_port;
        let hidden_enemy = u.invisible[i] && u.owner[i] != owner;
        passable[i] = t.explored[p][i] && terrain_ok && (aquatic || land_access) && (u.owner[i] == 0 || u.owner[i] == owner || hidden_enemy);
        let peaceful = s.players.peace[p][s.pclip(u.owner[i])];
        enemy[i] = u.owner[i] != 0 && u.owner[i] != owner && !u.invisible[i] && !peaceful;
        let friendly_road = t.owner[i] == 0 || t.owner[i] == owner || s.players.peace[p][s.pclip(t.owner[i])];
        road[i] = (t.road[i] || t.structure[i] == 1 || s.cities.owner[i] != 0 || bridge) && friendly_road;
    }
    let no_zoc = creep || r.skills[kind][INFILTRATE] || u.invisible[source];
    for i in 0..n {
        let zoc = !no_zoc && g.adj[i].iter().any(|&j| enemy[j as usize]);
        let water = t.terrain[i] == 1 || t.terrain[i] == 2;
        let own_port = t.structure[i] == 8 && t.owner[i] == owner;
        terminal[i] = zoc || t.terrain[i] == 4 || (t.terrain[i] == 5 && !road[i] && !creep) || (own_port && !aquatic) || (r.skills[kind][CARRY] && !water);
    }
    md.budget = (r.unit_movement[kind] + u.boosted[source] as i32) * 2;
}

// Unit-independent movement terms of one owner, one byte per tile (movement_flags).
pub const MV_PASS: u8 = 1; // explored, terrain allowed by tech (or bridge), no blocking unit
pub const MV_LAND: u8 = 2; // land access: not water, or a bridge, or an own port
pub const MV_ROAD: u8 = 4; // friendly road network tile
pub const MV_ZOC: u8 = 8; // next to a visible, non-peaceful, explored enemy unit
pub const MV_MOUNTAIN: u8 = 16;
pub const MV_FOREST: u8 = 32;
pub const MV_OWN_PORT: u8 = 64;
pub const MV_WATER: u8 = 128;

/// kernels.movement_data's per-tile terms for units of `owner`, which only the unit's skills (float, creep,
/// carry, zoc immunity) and budget specialise: passable = PASS && (float || LAND); terminal = (ZOC && !no_zoc)
/// || MOUNTAIN || (FOREST && !ROAD && !creep) || (OWN_PORT && !float) || (carry && !WATER).
pub fn movement_flags(s: &State, owner: i32, g: &Geom, flags: &mut [u8]) {
    let n = s.n;
    let (t, u) = (&s.tiles, &s.units);
    let p = s.pclip(owner);
    let tech = &s.players.tech[p];
    let peace = &s.players.peace[p];
    let explored = &t.explored[p];
    for i in 0..n {
        let (terrain, structure) = (t.terrain[i], t.structure[i]);
        let bridge = structure == 48;
        let water = terrain == 1 || terrain == 2;
        let terrain_ok = ((terrain != 1 || tech[10]) && (terrain != 2 || tech[13] || tech[25]) && (terrain != 4 || tech[20])) || bridge;
        let own_port = structure == 8 && t.owner[i] == owner;
        let uo = u.owner[i];
        let hidden_enemy = u.invisible[i] && uo != owner;
        let pass = explored[i] && terrain_ok && (uo == 0 || uo == owner || hidden_enemy);
        let friendly_road = t.owner[i] == 0 || t.owner[i] == owner || peace[s.pclip(t.owner[i])];
        let road = (t.road[i] || structure == 1 || s.cities.owner[i] != 0 || bridge) && friendly_road;
        flags[i] = (pass as u8) * MV_PASS | ((!water || bridge || own_port) as u8) * MV_LAND | (road as u8) * MV_ROAD
            | ((terrain == 4) as u8) * MV_MOUNTAIN | ((terrain == 5) as u8) * MV_FOREST | (own_port as u8) * MV_OWN_PORT
            | (water as u8) * MV_WATER;
    }
    for i in 0..n {
        let uo = u.owner[i];
        // movement cost reads the raw neighbours: unseen enemies exert zone of control too
        let enemy = uo != 0 && uo != owner && !u.invisible[i] && !peace[s.pclip(uo)];
        if enemy {
            for &j in &g.adj[i] {
                flags[j as usize] |= MV_ZOC;
            }
        }
    }
}

/// A Hide unit may pass a tile held by another player's unit (it never
/// ends there). True when tile `i` stops `owner`'s units only because of such a unit: explored, terrain allowed.
#[inline]
pub fn hide_passable(s: &State, owner: i32, i: usize) -> bool {
    let (t, u) = (&s.tiles, &s.units);
    let tech = &s.players.tech[s.pclip(owner)];
    let terrain = t.terrain[i];
    let terrain_ok = ((terrain != 1 || tech[10]) && (terrain != 2 || tech[13] || tech[25]) && (terrain != 4 || tech[20]))
        || t.structure[i] == 48;
    u.owner[i] != 0 && u.owner[i] != owner && t.explored[s.pclip(owner)][i] && terrain_ok
}

/// Any unit may pass a tile held by a peace partner's unit.
#[inline]
pub fn peace_unit(s: &State, owner: i32, i: usize) -> bool {
    let o = s.units.owner[i];
    o != 0 && o != owner && s.players.peace[s.pclip(owner)][s.pclip(o)]
}

/// Entering a forest ends the move unless the step runs road to road. `cand` is the step's cost so far.
#[inline]
pub fn forest_step(creep: bool, road_edge: bool, dest_flags: u8, cand: i32, budget: i32) -> i32 {
    if !creep && !road_edge && dest_flags & MV_FOREST != 0 { cand.max(budget) } else { cand }
}

/// Without water movement (Fishing), a bridge cannot be entered across its
/// span: not straight from the east or west onto a north-south bridge, nor straight from north or south onto an
/// east-west one (diagonal steps are allowed).
#[inline]
pub fn bridge_blocks(s: &State, owner: i32, from: usize, to: usize) -> bool {
    if s.tiles.structure[to] != 48 || s.players.tech[s.pclip(owner)][10] {
        return false;
    }
    let size = s.size;
    let (dx, dy) = ((to % size) as i32 - (from % size) as i32, (to / size) as i32 - (from / size) as i32);
    match crate::structures::bridge_span(s, to) {
        1 => dy == 0,
        2 => dx == 0,
        _ => false,
    }
}

/// Reusable buffers for movement_targets.
pub struct MoveScratch {
    pub flags: Vec<u8>,
    pub dist: Vec<i32>,
    stamp: Vec<u32>,
    epoch: u32,
    frontier: Vec<(u16, i32)>,
    next: Vec<u16>,
    touched: Vec<u16>,
}

impl MoveScratch {
    pub fn new(n: usize) -> MoveScratch {
        MoveScratch { flags: vec![0; n], dist: vec![INF; n], stamp: vec![0; n], epoch: 0, frontier: Vec::with_capacity(n),
                      next: Vec::with_capacity(n), touched: Vec::with_capacity(n) }
    }
}

/// Destination tiles of the unit on `source` (kernels.movement_targets without its `active` gate), given
/// `ms.flags` = movement_flags of the unit's owner. kernels.movement_targets runs 8 synchronous relaxation sweeps
/// over every tile; this runs the same 8 synchronous rounds (Jacobi: every relaxation in round k reads the
/// distances of round k-1) over the frontier only, i.e. the tiles whose distance changed in the previous round,
/// which is exact: a tile that did not change contributes nothing new. Calls `emit(tile)` per destination.
pub fn reach(s: &State, source: usize, r: &Rules, g: &Geom, ms: &mut MoveScratch, mut emit: impl FnMut(usize)) {
    let u = &s.units;
    let owner = u.owner[source];
    let kind = u.kind[source] as usize;
    let sk = &r.skills[kind];
    let (aquatic, creep, carry) = (sk[FLOAT], sk[CREEP], sk[CARRY]);
    let no_zoc = creep || sk[INFILTRATE] || u.invisible[source];
    let budget = (r.unit_movement[kind] + u.boosted[source] as i32) * 2;
    let hide = sk[HIDE];
    // terminal: any of these bits set (plus FOREST without ROAD for non-creepers, handled below)
    let term_bits = (if no_zoc { 0 } else { MV_ZOC }) | MV_MOUNTAIN | (if aquatic { 0 } else { MV_OWN_PORT });
    let pass_need = if aquatic { MV_PASS } else { MV_PASS | MV_LAND };
    let MoveScratch { flags, dist, stamp, epoch, frontier, next, touched } = ms;
    let terminal = |f: u8| f & term_bits != 0 || (f & (MV_FOREST | MV_ROAD) == MV_FOREST && !creep) || (carry && f & MV_WATER == 0);
    touched.clear();
    frontier.clear();
    dist[source] = 0;
    touched.push(source as u16);
    frontier.push((source as u16, 0));
    for _ in 0..8 {
        if frontier.is_empty() {
            break;
        }
        *epoch = epoch.wrapping_add(1);
        if *epoch == 0 {
            stamp.fill(0);
            *epoch = 1;
        }
        next.clear();
        for &(j, dj) in frontier.iter() {
            let j = j as usize;
            let fj = flags[j];
            if dj >= budget || (j != source && terminal(fj)) {
                continue;
            }
            let road_j = fj & MV_ROAD != 0 && !creep;
            for &i in &g.adj[j] {
                let i = i as usize;
                let fi = flags[i];
                if fi & pass_need != pass_need && i != source
                    && !((hide || peace_unit(s, owner, i)) && (aquatic || fi & MV_LAND != 0) && hide_passable(s, owner, i)) {
                    continue;
                }
                if bridge_blocks(s, owner, j, i) {
                    continue;
                }
                let road_edge = road_j && fi & MV_ROAD != 0;
                let cand = forest_step(creep, road_edge, fi, dj + if road_edge { 1 } else { 2 }, budget);
                if cand < dist[i] {
                    if dist[i] == INF {
                        touched.push(i as u16);
                    }
                    dist[i] = cand;
                    if stamp[i] != *epoch {
                        stamp[i] = *epoch;
                        next.push(i as u16);
                    }
                }
            }
        }
        frontier.clear();
        frontier.extend(next.iter().map(|&i| (i, dist[i as usize])));
    }
    // an Infiltrate unit never ends on an opponent's city, and no
    // unit ends on the city of a player it has peace or a cool-down with
    let po = s.pclip(owner);
    let finish_ok = |i: usize| {
        let co = s.cities.owner[i];
        co == 0 || co == owner
            || (!sk[INFILTRATE] && !s.players.peace[po][s.pclip(co)] && s.players.truce_breaker[po][s.pclip(co)] == 0)
    };
    for &i in touched.iter() {
        let i = i as usize;
        dist[i] = INF;
        let empty = u.owner[i] == 0 || (u.invisible[i] && u.owner[i] != owner);
        if empty && i != source && finish_ok(i) {
            emit(i);
        }
    }
}

/// Destination set of the unit on `source` (kernels.movement_targets).
pub fn movement_targets(s: &State, source: usize, r: &Rules, g: &Geom, ms: &mut MoveScratch, out: &mut [bool]) {
    let n = s.n;
    let u = &s.units;
    let kind = u.kind[source] as usize;
    out[..n].fill(false);
    let active = u.owner[source] == s.current_player && !u.moved[source] && r.unit_supported[kind] && !s.pending_reward()
        && !s.terminated && !s.truncated;
    if !active {
        return;
    }
    movement_flags(s, u.owner[source], g, &mut ms.flags);
    reach(s, source, r, g, ms, |i| out[i] = true);
}

pub fn update_exploration(s: &mut State, player: i32, g: &Geom) {
    let n = s.n;
    let p = (player - 1) as usize;
    let mut explored = vec![false; n];
    for t in 0..n {
        explored[t] = s.tiles.explored[p][t] || !s.fog;
    }
    for c in 0..n {
        if s.cities.owner[c] == player {
            let range = if s.tiles.capital[c] > 0 { 2 } else { s.cities.border[c] };
            g.for_ball(c, range, |t| {
                let corner = g.corners.contains(&t);
                if !corner || s.cities.owner[t] == player {
                    explored[t] = true;
                }
            });
        }
    }
    for i in 0..n {
        if s.units.owner[i] == player {
            let k = s.units.kind[i];
            let range = if s.tiles.terrain[i] == 4 || k == 42 || k == 38 || k == 39 { 2 } else { 1 };
            g.for_ball(i, range, |t| explored[t] = true);
        }
    }
    let mut gain = 0;
    for t in 0..n {
        if explored[t] && !s.tiles.explored[p][t] {
            gain += 5;
        }
        s.tiles.explored[p][t] = explored[t];
    }
    s.players.score[p] = s.players.score[p].wrapping_add(gain);
}

/// Packed public view: kernels.observe(state, player) fields in sorted-key order, int16 (cpu_engine.pack_view).
/// Games with more than two seats add "alive" (first in sorted order) and give the five relation fields one entry
/// per seat (view_fields / view_width).
pub const VIEW_FIELDS: [&str; 40] = [
    "capital", "detect", "embassies_received", "embassies_sent", "explored", "known_players", "own_built",
    "own_city_connected", "own_city_level", "own_city_population", "own_city_production", "own_city_progress",
    "own_city_rewards", "own_city_riot", "own_monuments", "own_unit_attacked", "own_unit_boosted", "own_unit_frozen",
    "own_unit_home", "own_unit_invisible", "own_unit_kills", "own_unit_moved", "own_unit_passenger", "own_unit_poison",
    "owner", "peace", "peace_offers", "player", "resource", "road", "stars", "structure", "structure_level", "tech",
    "terrain", "turn", "unit_health", "unit_owner", "unit_type", "unit_veteran",
];

/// Field names of the packed view of a game with `np` seats, in packing order.
pub fn view_fields(np: usize) -> Vec<&'static str> {
    let mut f = VIEW_FIELDS.to_vec();
    if np > 2 {
        f.insert(0, "alive");
    }
    f
}

pub fn view_width(n: usize) -> usize {
    37 * n + 397
}

/// Packed view width for `np` seats: relations take np entries each, plus "alive" when np > 2.
pub fn view_width_np(n: usize, np: usize) -> usize {
    view_width(n) + if np > 2 { 6 * np - 10 } else { 0 }
}

/// Largest board the observation's stack buffers hold (18x18 is the largest map size).
const MAX_TILES: usize = 512;

#[inline(always)]
fn put_sel(dst: &mut [i16], mask: &[bool], v: &[i32], other: i16) {
    for ((d, &m), &x) in dst.iter_mut().zip(mask).zip(v) {
        *d = if m { x as i16 } else { other };
    }
}

#[inline(always)]
fn put_bool(dst: &mut [i16], v: &[bool]) {
    for (d, &x) in dst.iter_mut().zip(v) {
        *d = x as i16;
    }
}

/// visible ? (unit visible ? v : 0) : -1
#[inline(always)]
fn put_unit(dst: &mut [i16], vis: &[bool], uvis: &[bool], v: &[i32]) {
    for (((d, &a), &b), &x) in dst.iter_mut().zip(vis).zip(uvis).zip(v) {
        *d = if a { if b { x as i16 } else { 0 } } else { -1 };
    }
}

/// kernels.observe packed in VIEW_FIELDS order: one tight loop per field over precomputed visibility and
/// ownership masks. The same code is compiled twice and picked at run time: with AVX2 (the i32 -> i16 selects
/// vectorise) when the CPU has it, else for the baseline target. Integer-only, so both give identical bits.
pub fn observe(s: &State, player: i32, g: &Geom, out: &mut [i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2 (checked just above).
            return unsafe { observe_avx2(s, player, g, out) };
        }
    }
    observe_impl(s, player, g, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn observe_avx2(s: &State, player: i32, g: &Geom, out: &mut [i16]) {
    observe_impl(s, player, g, out)
}

#[inline(always)]
fn observe_impl(s: &State, player: i32, g: &Geom, out: &mut [i16]) {
    let n = s.n;
    let np = s.np;
    assert!(n <= MAX_TILES && out.len() >= view_width_np(n, np));
    let p = (player - 1) as usize;
    let (t, u, c, pl) = (&s.tiles, &s.units, &s.cities, &s.players);
    let tech = &pl.tech[p];
    let (mut vis, mut uvis, mut unit) = ([false; MAX_TILES], [false; MAX_TILES], [false; MAX_TILES]);
    // the seat's cities and units: their fields are zero everywhere else
    let (mut cities, mut units) = ([0u16; MAX_TILES], [0u16; MAX_TILES]);
    let (mut nc, mut nu) = (0usize, 0usize);
    for i in 0..n {
        let v = t.explored[p][i] || !s.fog;
        vis[i] = v;
        unit[i] = u.owner[i] == player;
        uvis[i] = v && (unit[i] || !u.invisible[i]);
        cities[nc] = i as u16;
        nc += (c.owner[i] == player) as usize;
        units[nu] = i as u16;
        nu += unit[i] as usize;
    }
    let (vis, uvis, unit, cities, units) = (&vis[..n], &uvis[..n], &unit[..n], &cities[..nc], &units[..nu]);
    let mut at = 0usize;
    macro_rules! field {
        ($len:expr) => {{
            let f = &mut out[at..at + $len];
            at += $len;
            f
        }};
    }
    macro_rules! own {
        ($list:expr, $v:expr) => {{
            let f = field!(n);
            f.fill(0);
            for &i in $list {
                f[i as usize] = $v[i as usize] as i16;
            }
        }};
    }
    macro_rules! small {
        ($($v:expr),*) => {{ $( out[at] = $v as i16; at += 1; )* }};
    }
    if np > 2 {
        for k in 0..np {
            small!(s.alive(k));
        }
    }
    put_sel(field!(n), vis, &t.capital, -1);
    let detect = field!(n);
    detect.fill(0);
    for i in 0..n {
        if unit[i] {
            detect[i] = g.adj[i].iter().any(|&j| {
                let j = j as usize;
                u.owner[j] != 0 && u.owner[j] != player && u.invisible[j]
            }) as i16;
        }
    }
    for k in 0..np {
        small!(pl.embassy[k][p]);
    }
    for k in 0..np {
        small!(pl.embassy[p][k]);
    }
    put_bool(field!(n), vis);
    for k in 0..np {
        small!(pl.known[p][k]);
    }
    put_bool(field!(128), &pl.built[p]);
    own!(cities, c.connected);
    own!(cities, c.level);
    own!(cities, c.population);
    own!(cities, c.production);
    own!(cities, c.progress);
    let rewards = field!(n * 9);
    rewards.fill(0);
    for &i in cities {
        let i = i as usize;
        for k in 0..9 {
            rewards[i * 9 + k] = c.rewards[i * 9 + k] as i16;
        }
    }
    own!(cities, c.riot);
    put_bool(field!(128), &pl.monuments[p]);
    own!(units, u.attacked);
    own!(units, u.boosted);
    own!(units, u.frozen);
    own!(units, u.home);
    own!(units, u.invisible);
    own!(units, u.kills);
    own!(units, u.moved);
    own!(units, u.passenger);
    own!(units, u.poison);
    put_sel(field!(n), vis, &t.owner, -1);
    for k in 0..np {
        small!(pl.peace[p][k]);
    }
    for k in 0..np {
        small!(pl.peace_offer[p][k]);
    }
    small!(player);
    let hide2 = !(tech[6] || tech[8] || tech[9]);
    let hide5 = !(tech[20] || tech[23] || tech[24]);
    for ((d, &m), &r) in field!(n).iter_mut().zip(vis).zip(&t.resource) {
        let hidden = (r == 2 && hide2) || (r == 5 && hide5);
        *d = if m { if hidden { 0 } else { r as i16 } } else { -1 };
    }
    for ((d, &m), &r) in field!(n).iter_mut().zip(vis).zip(&t.road) {
        *d = if m { r as i16 } else { -1 };
    }
    small!(pl.stars[p]);
    put_sel(field!(n), vis, &t.structure, -1);
    put_sel(field!(n), vis, &t.structure_level, -1);
    put_bool(field!(128), tech);
    put_sel(field!(n), vis, &t.terrain, -1);
    small!(s.turn);
    // unit_health is float32 in the view; the pack casts it to int16 (toward zero).
    for (((d, &a), &b), &h) in field!(n).iter_mut().zip(vis).zip(uvis).zip(&u.health) {
        let v: f32 = if a { if b { h } else { 0.0 } } else { -1.0 };
        *d = v as i16;
    }
    put_unit(field!(n), vis, uvis, &u.owner);
    put_unit(field!(n), vis, uvis, &u.kind);
    for (((d, &a), &b), &x) in field!(n).iter_mut().zip(vis).zip(uvis).zip(&u.veteran) {
        *d = if a { if b { x as i16 } else { 0 } } else { -1 };
    }
    debug_assert_eq!(at, view_width_np(n, np));
}
