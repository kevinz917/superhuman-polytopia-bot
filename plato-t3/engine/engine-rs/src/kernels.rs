//! gpu/kernels.py: combat formula, unit health, research, movement reachability, exploration and the observation.

use crate::geom::Geom;
use crate::rules::{Rules, CARRY, CREEP, FLOAT, HIDE, INFILTRATE};
use crate::state::State;

pub const INF: i32 = 1_000_000;

/// get battle results in the game's integer units (health in tenths, attack and defence in hundredths,
/// defence bonus in tenths): forces are truncated, then the damage is rounded half-up to whole HP. Inputs are our
/// f32 HP/stat values and the defender's bonus (poison already halves it, as get defence bonus's `>> 1`); returns
/// (attack damage, retaliation, floored half attack damage) in HP; retaliation is 0 when the attack kills.
#[inline]
pub fn combat(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32, f32) {
    let tenths = |x: f32| (x * 10.0).round() as i64;
    let round10 = |v: i64| if v % 10 < 10 - v % 10 { v / 10 * 10 } else { v / 10 * 10 + 10 };
    let attack = (aa * 100.0).round() as i64;
    let bonus = ((db * 10.0) + 0.001).floor() as i64;
    let defence = tenths(dd) * bonus;
    let (ah, am, dh, dm) = (tenths(ah), tenths(am).max(1), tenths(dh), tenths(dm).max(1));
    let a_force = ah * attack * 100 / am;
    let d_force = defence * dh * 100 / dm;
    let total = a_force + d_force;
    if total <= 0 {
        return (0.0, 0.0, 0.0);
    }
    let damage = round10(attack * a_force * 450 / (total * 1000));
    let retaliation = if damage < dh && bonus > 0 { round10(d_force * defence * 4500 / (total * 1000 * bonus)) } else { 0 };
    (damage as f32 / 10.0, retaliation as f32 / 10.0, (damage / 20 * 10) as f32 / 10.0)
}

#[inline]
pub fn max_health(s: &State, r: &Rules, i: usize) -> f32 {
    let u = &s.units;
    let kind = if u.passenger[i] > 0 { u.passenger[i] } else { u.kind[i] };
    r.unit_health[(kind as usize).min(127)] + (u.veteran[i] as i32 * 5) as f32
}

/// get movement: movement, + 1 Boosted, + 1 Swift or Bubble (full-game engine
/// Aquarion, Cymanti; not both); (Cymanti) Poisoned takes one off unless that would leave 0 (a 1-move unit keeps 1).
#[inline]
pub fn movement(s: &State, r: &Rules, i: usize) -> i32 {
    let u = &s.units;
    let m = r.unit_movement[u.kind[i].clamp(0, 127) as usize] + u.boosted[i] as i32 + (u.bubble[i] || u.swift[i]) as i32;
    if u.poison[i] && m > 1 { m - 1 } else { m }
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
    // get tech price: Literacy multiplies by 0.66666f
    if crate::tribes::literacy(&s.players.tech[s.p()]) { ((cost as f32) * 0.66666f32).ceil() as i32 } else { cost }
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
    let succ = r.tech_succ_of(s.players.tribe[p]);
    while rest != 0 {
        cand |= succ[rest.trailing_zeros() as usize];
        rest &= rest - 1;
    }
    cand &= !known & r.tech_exists_of(s.players.tribe[p]);
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
    let embarks = crate::tribes::can_embark(s.players.tribe[p], tech);
    let (passable, enemy, road, terminal) = (&mut md.passable, &mut md.enemy, &mut md.road, &mut md.terminal);
    for i in 0..n {
        let bridge = t.structure[i] == 48;
        let water = t.terrain[i] == 1 || t.terrain[i] == 2;
        // Cymanti: an Algae water tile is wetland, which Basic opens to every unit (is tile accessible's water branch)
        let algae = t.algae[i];
        let terrain_ok = terrain_ok(tech, t.terrain[i]) || bridge || algae;
        // a friendly Port, and can player embark (Fishing unlocks the Port, the only Embark improvement)
        // will unit embark needs water: a Port on Ice (frozen) is plain land (Polaris)
        let own_port = t.structure[i] == 8 && (t.terrain[i] <= 2 || t.flooded[i]) && embarks && (t.owner[i] == owner || (t.owner[i] > 0 && s.players.peace[p][s.pclip(t.owner[i])]));
        // a chain unit never takes a Port into water (is tile accessible, Cymanti)
        let land_access = !water || bridge || (own_port && u.leader[source] == 0 && u.follower[source] == 0) || algae;
        let hidden_enemy = s.hidden_from(i, owner);
        passable[i] = t.explored[p][i] && terrain_ok && (aquatic || land_access) && (u.owner[i] == 0 || u.owner[i] == owner || hidden_enemy);
        let peaceful = s.players.peace[p][s.pclip(u.owner[i])];
        // get movement cost: a unit with a leader exerts no zone of control (Cymanti)
        enemy[i] = u.owner[i] != 0 && u.owner[i] != owner && !u.invisible[i] && !peaceful && u.leader[i] == 0;
        let friendly_road = t.owner[i] == 0 || t.owner[i] == owner || s.players.peace[p][s.pclip(t.owner[i])];
        road[i] = (t.road[i] || t.structure[i] == 1 || s.cities.owner[i] != 0 || bridge) && friendly_road;
    }
    // get movement cost: no zone of control for Sneak, Hide or an Invisible unit
    let no_zoc = r.skills[kind][HIDE] || r.sneak[kind] || u.invisible[source];
    for i in 0..n {
        let zoc = !no_zoc && g.adj[i].iter().any(|&j| enemy[j as usize]);
        let water = t.terrain[i] == 1 || t.terrain[i] == 2;
        // a friendly Port, and can player embark (Fishing unlocks the Port, the only Embark improvement)
        // will unit embark needs water: a Port on Ice (frozen) is plain land (Polaris)
        let own_port = t.structure[i] == 8 && (t.terrain[i] <= 2 || t.flooded[i]) && embarks && (t.owner[i] == owner || (t.owner[i] > 0 && s.players.peace[p][s.pclip(t.owner[i])]));
        // Algae: 1000 for a land-bound unit without Creep
        // (after the road term, like Forest: a Bridge with Algae on the road network costs 5)
        let algae_stop = t.algae[i] && !aquatic && !creep && !road[i];
        terminal[i] = zoc || t.terrain[i] == 4 || (t.terrain[i] == 5 && !road[i] && !creep) || (own_port && !aquatic) || (r.skills[kind][CARRY] && !water)
            || algae_stop;
    }
    md.budget = movement(s, r, source) * 2;
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

/// get unlocked movements' Water: Fishing (10) or, for Aquarion, aqua base (45) (structures:
/// unlocked_movements; the other tribes' water techs 27/40 are in no ported tree).
#[inline]
pub fn water_move(tech: &[bool; 128]) -> bool {
    tech[10] || tech[45]
}

/// get unlocked movements' Ocean: Sailing (13), Free Diving (25) or, for Aquarion, marine life (46).
#[inline]
pub fn ocean_move(tech: &[bool; 128]) -> bool {
    tech[13] || tech[25] || tech[46]
}

/// get allowed terrain of a unit without Swim or movement terrain, as a test of one terrain: water and ocean by the
/// movements above, mountain with Climbing (20); field, forest, ice and wetland come with Basic.
#[inline]
pub fn terrain_ok(tech: &[bool; 128], terrain: i32) -> bool {
    (terrain != 1 || water_move(tech)) && (terrain != 2 || ocean_move(tech)) && (terrain != 4 || tech[20])
}

/// Aquarion: the movement terms of a Swim unit that the owner's movement_flags cannot hold
/// (water and wetland terms).
/// get allowed terrain appends Water for every Swim unit, whatever its owner's techs; a unit with movement terrain (the
/// Water ability without Carry: Jelly, Shark, Aquapult) keeps only water, ocean (when unlocked) and wetland, so it
/// enters land only where the tile is Flooded (matches any terrain's Wetland key) or a City. Units without Swim get the
/// plain owner terms (`on` false), so the regular tribes run the code they ran before.
///
/// Elyrion: `fly`. is tile accessible returns true for a Fly unit right after the explored
/// and occupied-tile tests, before the Bridge, port and terrain tests; get movement cost
/// charges it 10 on every tile once zone of control (1000) is past, before Skate, Glide and the road step
/// it never embarks.
#[derive(Clone, Copy)]
pub struct Swim {
    pub on: bool,
    pub fly: bool,
    /// movement terrain water/ocean/wetland (Rules.water)
    pub bound: bool,
    /// is water bound: Swim and (Carry or movement terrain); a Flooded tile costs it 1000
    pub water_bound: bool,
    /// the owner has no water movement of its own: Water comes only from Swim (a converted MermaidWarrior)
    pub extra_water: bool,
    /// Cymanti: a land-bound unit (no Swim, no Fly) without Creep: an Algae tile ends its move (`ends`)
    pub algae_stop: bool,
}

impl Swim {
    #[inline]
    pub fn of(s: &State, r: &Rules, kind: usize, owner: i32) -> Swim {
        let on = r.skills[kind][FLOAT];
        let bound = on && r.water[kind];
        Swim { on, fly: r.fly[kind], bound, water_bound: bound || (on && r.skills[kind][CARRY]),
               extra_water: on && !water_move(&s.players.tech[s.pclip(owner)]),
               algae_stop: !on && !r.fly[kind] && !r.skills[kind][CREEP] }
    }
    /// The unit may enter tile `i` given the owner's flags `fi`: Fly (explored, no blocking unit), Swim (`pass`), else
    /// MV_PASS and, for a land unit, MV_LAND.
    #[inline]
    pub fn enters(&self, s: &State, owner: i32, i: usize, fi: u8) -> bool {
        if self.fly {
            let uo = s.units.owner[i];
            return s.tiles.explored[s.pclip(owner)][i] && (uo == 0 || uo == owner || s.hidden_from(i, owner));
        }
        if self.on { self.pass(s, owner, i, fi) } else { fi & (MV_PASS | MV_LAND) == MV_PASS | MV_LAND }
    }
    /// is tile accessible's terrain test (get allowed terrain, matches any terrain, a City always) for this unit of an owner
    /// with `tech`, without the Bridge case.
    #[inline]
    pub fn terrain_ok(&self, s: &State, tech: &[bool; 128], i: usize) -> bool {
        let terrain = s.tiles.terrain[i];
        if self.fly {
            return true;
        }
        if !self.on {
            return terrain_ok(tech, terrain);
        }
        match terrain {
            1 => true,
            // an Algae ocean tile is wetland, in every Swim unit's allowed terrain (Basic, or movement terrain)
            2 => ocean_move(tech) || s.tiles.algae[i],
            _ if self.bound => s.tiles.flooded[i] || s.tiles.structure[i] == 1,
            _ => terrain != 4 || tech[20],
        }
    }
    /// The unit may enter tile `i` given the owner's flags `fi` (movement_flags' MV_PASS: explored, the owner's terrain,
    /// no blocking unit): MV_PASS corrected by the Swim terrain terms above.
    #[inline]
    pub fn pass(&self, s: &State, owner: i32, i: usize, fi: u8) -> bool {
        if fi & MV_PASS != 0 {
            return !self.bound || fi & MV_WATER != 0 || s.tiles.flooded[i] || s.tiles.structure[i] == 1
                || s.tiles.structure[i] == 48;
        }
        // MV_PASS refused the tile: only its terrain can be the Swim unit's (Water without the owner's water movement)
        self.extra_water && s.tiles.terrain[i] == 1 && s.tiles.explored[s.pclip(owner)][i] && {
            let uo = s.units.owner[i];
            uo == 0 || uo == owner || s.hidden_from(i, owner)
        }
    }
    /// get movement cost's Swim terms, after Skate and Glide (a gliding owner's unit pays 9 on Ice first): 1000 on land
    /// that is not Flooded, and 1000 on a Flooded tile for a water-bound unit; (Cymanti)
    /// for a land-bound unit without Creep, 1000 on Algae.
    #[inline]
    pub fn ends(&self, s: &State, i: usize, glide: bool) -> bool {
        if !self.on {
            return self.algae_stop && s.tiles.algae[i];
        }
        let (terrain, flooded) = (s.tiles.terrain[i], s.tiles.flooded[i]);
        !(glide && terrain == ICE) && ((terrain > 2 && !flooded) || (flooded && self.water_bound))
    }
}

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
    let embarks = crate::tribes::can_embark(s.players.tribe[p], tech);
    for i in 0..n {
        let (terrain, structure) = (t.terrain[i], t.structure[i]);
        let bridge = structure == 48;
        let water = terrain == 1 || terrain == 2;
        // Cymanti: Algae water is wetland, open to every unit (Basic) without a Port
        let algae = t.algae[i];
        let terrain_ok = terrain_ok(tech, terrain) || bridge || algae;
        // is tile accessible / will unit embark: a peace partner's port serves like an own one; the
        // port stands on water or a wetland (Flooded) tile (a Port on plain Ice is land, Polaris); can player embark
        let own_port = structure == 8 && (terrain <= 2 || t.flooded[i]) && embarks && (t.owner[i] == owner || (t.owner[i] > 0 && peace[s.pclip(t.owner[i])]));
        let uo = u.owner[i];
        let hidden_enemy = s.hidden_from(i, owner);
        let pass = explored[i] && terrain_ok && (uo == 0 || uo == owner || hidden_enemy);
        let friendly_road = t.owner[i] == 0 || t.owner[i] == owner || peace[s.pclip(t.owner[i])];
        let road = (t.road[i] || structure == 1 || s.cities.owner[i] != 0 || bridge) && friendly_road;
        flags[i] = (pass as u8) * MV_PASS | ((!water || bridge || own_port || algae) as u8) * MV_LAND | (road as u8) * MV_ROAD
            | ((terrain == 4) as u8) * MV_MOUNTAIN | ((terrain == 5) as u8) * MV_FOREST | (own_port as u8) * MV_OWN_PORT
            | (water as u8) * MV_WATER;
    }
    for i in 0..n {
        let uo = u.owner[i];
        // get movement cost reads the raw neighbours: unseen enemies exert zone of control too
        let enemy = uo != 0 && uo != owner && !u.invisible[i] && !peace[s.pclip(uo)] && u.leader[i] == 0;
        if enemy {
            for &j in &g.adj[i] {
                flags[j as usize] |= MV_ZOC;
            }
        }
    }
}

/// is tile accessible: a Hide unit may pass a tile held by another player's unit (it never
/// ends there). True when tile `i` stops `owner`'s units only because of such a unit: explored, terrain allowed.
#[inline]
pub fn hide_passable(s: &State, owner: i32, sw: &Swim, i: usize) -> bool {
    let (t, u) = (&s.tiles, &s.units);
    let tech = &s.players.tech[s.pclip(owner)];
    let terrain_ok = sw.terrain_ok(s, tech, i) || t.structure[i] == 48;
    u.owner[i] != 0 && u.owner[i] != owner && t.explored[s.pclip(owner)][i] && terrain_ok
}

/// is tile accessible: any unit may pass a tile held by a peace partner's unit.
#[inline]
pub fn peace_unit(s: &State, owner: i32, i: usize) -> bool {
    let o = s.units.owner[i];
    o != 0 && o != owner && s.players.peace[s.pclip(owner)][s.pclip(o)]
}

/// get movement cost: entering a forest ends the move unless the step runs road to road. `cand` is the step's cost so far.
pub const ICE: i32 = 6;

/// get movement cost's step in the game's tenths for the relaxation kernels (reach, search_ref), after the
/// terminal tests (ZOC, embark): a Skate unit pays 10 on Ice and ends its move anywhere else (1000: the cost is
/// raised to the budget, like a forest); a Glide owner's unit pays 9 on Ice; a road step costs
/// 5, never onto Ice and never for a skater. `road_edge` = both tiles friendly road, not creep, not aquatic.
#[inline]
pub fn step_tenths(road_edge: bool, dest_terrain: i32, skate: bool, glide: bool, cand: i32, budget: i32) -> i32 {
    let ice = dest_terrain == ICE;
    if skate {
        return if ice { cand + 10 } else { (cand + 10).max(budget) };
    }
    if glide && ice {
        return cand + 9;
    }
    cand + if road_edge && !ice { 5 } else { 10 }
}

/// The owner has the Glide player ability (Polarism, tech 32).
#[inline]
pub fn glides(s: &State, owner: i32) -> bool {
    s.players.tech[s.pclip(owner)][32]
}

#[inline]
pub fn forest_step(creep: bool, road_edge: bool, dest_flags: u8, cand: i32, budget: i32) -> i32 {
    if !creep && !road_edge && dest_flags & MV_FOREST != 0 { cand.max(budget) } else { cand }
}

/// is tile accessible: without water movement (Fishing, aqua base), a bridge cannot be entered across its
/// span: not straight from the east or west onto a north-south bridge, nor straight from north or south onto an
/// east-west one (diagonal steps are allowed).
#[inline]
pub fn bridge_blocks(s: &State, owner: i32, from: usize, to: usize) -> bool {
    // the span rule sits in is tile accessible's water branch: a frozen bridge (Ice, land) has none (Polaris)
    if s.tiles.structure[to] != 48 || water_move(&s.players.tech[s.pclip(owner)]) || s.tiles.terrain[to] == ICE {
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
    let no_zoc = sk[HIDE] || r.sneak[kind] || u.invisible[source];
    // move command validity is get path for unit with the unit's own movement: get move options' chain rules (no
    // moves for a led unit, a slower follower capping its leader) are the UI's and the AI's only
    let budget = movement(s, r, source) * 10;
    // is tile accessible (version > 119; Cymanti): a unit with a follower or a leader
    // never takes a Port into water
    let no_port = u.follower[source] != 0 || u.leader[source] != 0;
    let (skate, glide) = (r.skate[kind], glides(s, owner));
    let sw = Swim::of(s, r, kind, owner);
    let fly = sw.fly;
    let hide = sk[HIDE];
    // terminal: any of these bits set (plus FOREST without ROAD for non-creepers, handled below); a flyer stops only
    // in zone of control
    let zoc_bit = if no_zoc { 0 } else { MV_ZOC };
    let term_bits = if fly { zoc_bit } else { zoc_bit | MV_MOUNTAIN | (if aquatic { 0 } else { MV_OWN_PORT }) };
    let MoveScratch { flags, dist, stamp, epoch, frontier, next, touched } = ms;
    let terminal = |f: u8| f & term_bits != 0 || (!fly && ((f & (MV_FOREST | MV_ROAD) == MV_FOREST && !creep) || (carry && f & MV_WATER == 0)));
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
            // get movement cost: no road speed for Swim units (boats) or Creep units
            let road_j = fj & MV_ROAD != 0 && !creep && !aquatic && !skate;
            for &i in &g.adj[j] {
                let i = i as usize;
                let fi = flags[i];
                if no_port && !aquatic && !fly && fi & MV_OWN_PORT != 0 && !s.tiles.algae[i] {
                    continue;
                }
                if !sw.enters(s, owner, i, fi) && i != source
                    && !((hide || peace_unit(s, owner, i)) && (aquatic || fly || fi & MV_LAND != 0) && hide_passable(s, owner, &sw, i)) {
                    continue;
                }
                // a Swim unit always has Water and a flyer skips the Bridge test: no bridge span rule
                if !aquatic && !fly && bridge_blocks(s, owner, j, i) {
                    continue;
                }
                let cand = if fly {
                    dj + 10
                } else {
                    let road_edge = road_j && fi & MV_ROAD != 0;
                    let road_edge = road_edge && s.tiles.terrain[i] != ICE;
                    let cand = forest_step(creep, road_edge, fi, step_tenths(road_edge, s.tiles.terrain[i], skate, glide, dj, budget), budget);
                    // get movement cost returns the road's 5 before the Algae 1000 (a Bridge with Algae)
                    if sw.ends(s, i, glide) && !road_edge { cand.max(budget) } else { cand }
                };
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
    // is allowed to finish on tile: an Infiltrate unit never ends on an opponent's city, and no
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
        let empty = u.owner[i] == 0 || s.hidden_from(i, owner);
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
/// "unit_direction" (2026-10-31) is the facing of each visible unit plus one (0: no visible unit, -1: unexplored);
/// observation.encode_view gives it to networks of observation version 2 only.
/// "enemy_tech" [128] and "hist_enemy_built" [n], "hist_enemy_trained" [n], "hist_enemy_unit" [HIST_TURNS, n]
/// (2026-11-01) are the seat's observation history (history.rs), for networks of observation version 3.
/// Observation version 6 (special tribes, 2026-11-09; section 2): "converts" [1] (the
/// seat's Convert-task count), "own_structure_age" [n] (min(turn - structure_turn, 4) on the seat's own Sanctuary or
/// Fungi), "own_unit_double_ready" [n], the tile effects "tile_algae", "tile_decomposing", "tile_flooded",
/// "tile_polar" [n] (explored ? effect : -1), and for each own visible unit "unit_age" (min(turn - created, 3); encode_view
/// keeps it for Grow units only), "unit_bubble", "unit_follower", "unit_leader" (has one), "unit_swift" [n]
/// (explored ? (unit visible ? v : 0) : -1).
pub const VIEW_FIELDS: [&str; 58] = [
    "capital", "converts", "detect", "embassies_received", "embassies_sent", "enemy_tech", "explored",
    "hist_enemy_built", "hist_enemy_trained", "hist_enemy_unit", "known_players", "own_built",
    "own_city_connected", "own_city_level", "own_city_population", "own_city_production", "own_city_progress",
    "own_city_rewards", "own_city_riot", "own_monuments", "own_structure_age", "own_unit_attacked", "own_unit_boosted",
    "own_unit_double_ready", "own_unit_frozen", "own_unit_home", "own_unit_invisible", "own_unit_kills",
    "own_unit_moved", "own_unit_passenger", "own_unit_poison", "owner", "peace", "peace_offers", "player", "resource",
    "road", "stars", "structure", "structure_level", "tech", "terrain", "tile_algae", "tile_decomposing",
    "tile_flooded", "tile_polar", "tribe", "turn", "unit_age", "unit_bubble", "unit_direction", "unit_follower",
    "unit_health", "unit_leader", "unit_owner", "unit_swift", "unit_type", "unit_veteran",
];

/// Structures whose age the view shows on the seat's own tiles (observation version 6): Sanctuary (spawn cadence)
/// and Fungi (level cadence).
pub const AGED_STRUCTURES: [i32; 2] = [32, 37];

/// Field names of the packed view of a game with `np` seats, in packing order.
pub fn view_fields(np: usize) -> Vec<&'static str> {
    let mut f = VIEW_FIELDS.to_vec();
    if np > 2 {
        f.insert(0, "alive");
    }
    f
}

pub fn view_width(n: usize) -> usize {
    // + 2: "tribe" [np]; + 11 n + 1: observation version 6 (special tribes)
    (51 + crate::state::HIST_TURNS) * n + 397 + 128 + 2 + 1
}

/// Packed view width for `np` seats: relations take np entries each, plus "alive" when np > 2.
pub fn view_width_np(n: usize, np: usize) -> usize {
    view_width(n) + if np > 2 { 7 * np - 12 } else { 0 }
}

/// Largest board the observation's stack buffers hold (18x18 is the largest map size).
const MAX_TILES: usize = 1024;   // 900 tiles at 30x30 (was 512)

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

/// visible ? (unit visible ? v as 0/1 : 0) : -1 (observation version 6's unit flags)
#[inline(always)]
fn put_unit_bool(dst: &mut [i16], vis: &[bool], uvis: &[bool], v: &[bool]) {
    for (((d, &a), &b), &x) in dst.iter_mut().zip(vis).zip(uvis).zip(v) {
        *d = if a { if b { x as i16 } else { 0 } } else { -1 };
    }
}

/// Own visible chain bookkeeping only. A foreign link can point to an unrendered partner.
#[inline(always)]
fn put_own_unit_link(dst: &mut [i16], vis: &[bool], uvis: &[bool], v: &[i32], owners: &[i32], viewer: i32) {
    for ((((d, &a), &b), &x), &owner) in dst.iter_mut().zip(vis).zip(uvis).zip(v).zip(owners) {
        *d = if a { if b && owner == viewer { (x > 0) as i16 } else { 0 } } else { -1 };
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
        uvis[i] = v && (unit[i] || !s.hidden_from(i, player));
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
    small!(pl.converts[p]);
    let detect = field!(n);
    detect.fill(0);
    for i in 0..n {
        if unit[i] {
            detect[i] = g.adj[i].iter().any(|&j| {
                let j = j as usize;
                s.hidden_from(j, player)
            }) as i16;
        }
    }
    for k in 0..np {
        small!(pl.embassy[k][p]);
    }
    for k in 0..np {
        small!(pl.embassy[p][k]);
    }
    let hist = &s.history[p];
    for (k, d) in field!(128).iter_mut().enumerate() {
        *d = (hist.enemy_tech >> k & 1) as i16;
    }
    put_bool(field!(n), vis);
    put_bool(field!(n), &hist.built);
    put_bool(field!(n), &hist.trained);
    for k in 0..crate::state::HIST_TURNS {
        put_bool(field!(n), &hist.seen_unit[k]);
    }
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
    // own_structure_age: private, the seat's own Sanctuary / Fungi only
    for (i, d) in field!(n).iter_mut().enumerate() {
        let aged = t.owner[i] == player && AGED_STRUCTURES.contains(&t.structure[i]);
        *d = if aged { (s.turn - t.structure_turn[i]).clamp(0, 4) as i16 } else { 0 };
    }
    own!(units, u.attacked);
    own!(units, u.boosted);
    own!(units, u.double_ready);
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
    // is resource visible to player: starfish only once Star Fishing (Navigation) is unlockable;
    // crop never for Cymanti (farm -> fungi) and Spores only for Cymanti (its Fungi)
    let tribe = pl.tribe[p];
    let cymanti = tribe == crate::tribes::CYMANTI;
    let hide2 = cymanti || !(tech[6] || tech[8] || tech[9]);
    let hide5 = !(tech[20] || tech[23] || tech[24]);
    let hide8 = !(tech[crate::tribes::starfishing_parent(tribe)] || tech[crate::tribes::starfishing_tech(tribe)]);
    for ((d, &m), &r) in field!(n).iter_mut().zip(vis).zip(&t.resource) {
        // aqua crop (9): its Aquafarm needs Aquaculture (tech 39), which Bardur's tree never unlocks, so the game never
        // shows it (is resource visible to player; the real game shows the same)
        let hidden = (r == 2 && hide2) || (r == 5 && hide5) || (r == 7 && !cymanti) || (r == 8 && hide8) || (r == 9 && !tech[39]);
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
    for v in [&t.algae, &t.decomposing, &t.flooded, &t.polar] {
        for ((d, &m), &x) in field!(n).iter_mut().zip(vis).zip(v.iter()) {
            *d = if m { x as i16 } else { -1 };
        }
    }
    // "tribe" [np]: each seat's tribe type in seat order, the viewer's own always, another seat's once
    // the viewer has met it (meet action: its units and cities show their tribe), else 0
    for k in 0..np {
        small!(if k == p || pl.known[p][k] { pl.tribe[k] } else { 0 });
    }
    small!(s.turn);
    // unit_age is own bookkeeping; a visible foreign unit does not reveal its creation turn.
    for (i, d) in field!(n).iter_mut().enumerate() {
        *d = if vis[i] { if uvis[i] && u.owner[i] == player { (s.turn - u.created[i]).clamp(0, 3) as i16 } else { 0 } } else { -1 };
    }
    put_unit_bool(field!(n), vis, uvis, &u.bubble);
    // unit state direction of every visible unit, + 1 (the sprite's facing; try push unit starts from it)
    for (i, d) in field!(n).iter_mut().enumerate() {
        *d = if vis[i] { if uvis[i] && u.owner[i] != 0 { (u.direction[i] + 1) as i16 } else { 0 } } else { -1 };
    }
    put_own_unit_link(field!(n), vis, uvis, &u.follower, &u.owner, player);
    // unit_health is float32 in the view; the pack casts it to int16 (toward zero).
    for (((d, &a), &b), &h) in field!(n).iter_mut().zip(vis).zip(uvis).zip(&u.health) {
        let v: f32 = if a { if b { h } else { 0.0 } } else { -1.0 };
        *d = v as i16;
    }
    put_own_unit_link(field!(n), vis, uvis, &u.leader, &u.owner, player);
    put_unit(field!(n), vis, uvis, &u.owner);
    put_unit_bool(field!(n), vis, uvis, &u.swift);
    put_unit(field!(n), vis, uvis, &u.kind);
    for (((d, &a), &b), &x) in field!(n).iter_mut().zip(vis).zip(uvis).zip(&u.veteran) {
        *d = if a { if b { x as i16 } else { 0 } } else { -1 };
    }
    debug_assert_eq!(at, view_width_np(n, np));
}
