use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{self, BufRead, Write};

const INF: i32 = 1_000_000;

#[derive(Clone, Debug)]
struct Cmd {
    action_id: String,
    kind: i32,
    source: i32,
    target: i32,
    param: i32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Tile {
    known: bool,
    terrain: i32,
    owner: i32,
    structure: i32,
    resource: i32,
    road: bool,
    capital: i32,
    structure_level: i32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Unit {
    tile: usize,
    owner: i32,
    kind: i32,
    health: f32,
    veteran: bool,
    moved: bool,
    attacked: bool,
    poison: bool,
    frozen: bool,
    boosted: bool,
    invisible: bool,
    kills: i32,
    home: i32,
    passenger: i32,
    detect: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct City {
    tile: usize,
    owner: i32,
    level: i32,
    population: i32,
    progress: i32,
    production: i32,
    connected: bool,
    riot: bool,
    rewards: [i32; 9],
}

struct PersistentMemory {
    initialized: bool,
    size: usize,
    n: usize,
    me: i32,
    opp: i32,
    my_capital: Option<usize>,
    opp_capital_guess: usize,
    known_opp_cities: HashMap<usize, i32>, // tile -> last seen owner
    prev_owner: Vec<i32>,
    tile_city: Vec<i32>, // for our owned tiles, which city owns it (-1 if unknown)
    last_claim_city: Option<usize>,
    unit_prev_tile: HashMap<usize, usize>, // cur_tile -> prev_tile this turn
    last_turn: i32,
}

impl PersistentMemory {
    fn new() -> Self {
        Self {
            initialized: false,
            size: 14,
            n: 196,
            me: 1,
            opp: 2,
            my_capital: None,
            opp_capital_guess: 0,
            known_opp_cities: HashMap::new(),
            prev_owner: Vec::new(),
            tile_city: Vec::new(),
            last_claim_city: None,
            unit_prev_tile: HashMap::new(),
            last_turn: -1,
        }
    }
}

#[derive(Clone, Copy)]
struct UnitStats {
    cost: i32,
    health: f32,
    attack: f32,
    defense: f32,
    movement: i32,
    range: i32,
    dash: bool,
    fortify: bool,
    persist: bool,
    escape: bool,
    stiff: bool,
    splash: bool,
    float: bool,
    carry: bool,
    creep: bool,
    infiltrate: bool,
    surprise: bool,
}

fn unit_stats(kind: i32) -> UnitStats {
    let mut s = UnitStats {
        cost: 2, health: 10.0, attack: 0.0, defense: 1.0, movement: 1, range: 1,
        dash: false, fortify: false, persist: false, escape: false, stiff: false,
        splash: false, float: false, carry: false, creep: false, infiltrate: false, surprise: false,
    };
    match kind {
        2 => { s.cost = 2; s.health = 10.0; s.attack = 2.0; s.defense = 2.0; s.movement = 1; s.range = 1; s.dash = true; s.fortify = true; } // Warrior
        3 => { s.cost = 3; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 2; s.range = 1; s.dash = true; s.escape = true; s.fortify = true; } // Rider
        4 => { s.cost = 8; s.health = 10.0; s.attack = 3.5; s.defense = 1.0; s.movement = 3; s.range = 1; s.dash = true; s.fortify = true; s.persist = true; } // Knight
        5 => { s.cost = 3; s.health = 15.0; s.attack = 1.0; s.defense = 3.0; s.movement = 1; s.range = 1; s.fortify = true; } // Defender
        8 => { s.cost = 8; s.health = 10.0; s.attack = 4.0; s.defense = 0.0; s.movement = 1; s.range = 3; s.stiff = true; } // Catapult
        9 => { s.cost = 3; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 1; s.range = 2; s.dash = true; s.fortify = true; } // Archer
        10 => { s.cost = 5; s.health = 10.0; s.attack = 0.0; s.defense = 1.0; s.movement = 1; s.range = 1; s.stiff = true; } // MindBender
        11 => { s.cost = 5; s.health = 15.0; s.attack = 3.0; s.defense = 3.0; s.movement = 1; s.range = 1; s.dash = true; } // Swordsman
        12 => { s.cost = 10; s.health = 40.0; s.attack = 5.0; s.defense = 4.0; s.movement = 1; s.range = 1; } // Giant
        37 => { s.cost = 2; s.health = 10.0; s.attack = 2.0; s.defense = 2.0; s.movement = 1; s.range = 1; s.dash = true; s.surprise = true; } // Dagger
        38 => { s.cost = 8; s.health = 5.0; s.attack = 0.0; s.defense = 0.5; s.movement = 2; s.range = 1; s.dash = true; s.creep = true; s.infiltrate = true; s.stiff = true; } // Cloak
        39 => { s.cost = 2; s.health = 5.0; s.attack = 0.0; s.defense = 0.5; s.movement = 2; s.range = 1; s.dash = true; s.float = true; s.carry = true; s.infiltrate = true; s.stiff = true; } // Dinghy
        40 => { s.cost = 2; s.health = 10.0; s.attack = 2.0; s.defense = 2.0; s.movement = 2; s.range = 1; s.dash = true; s.float = true; s.carry = true; s.surprise = true; } // Pirate
        41 => { s.cost = 15; s.health = 10.0; s.attack = 3.0; s.defense = 2.0; s.movement = 2; s.range = 3; s.float = true; s.carry = true; s.splash = true; s.stiff = true; } // Bomber
        42 => { s.cost = 5; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 3; s.range = 2; s.dash = true; s.float = true; s.carry = true; } // Scoutship
        43 => { s.cost = 0; s.health = 10.0; s.attack = 0.0; s.defense = 1.0; s.movement = 2; s.range = 2; s.float = true; s.carry = true; s.stiff = true; } // Raft
        44 => { s.cost = 5; s.health = 10.0; s.attack = 3.0; s.defense = 3.0; s.movement = 3; s.range = 1; s.dash = true; s.float = true; s.carry = true; } // Rammership
        45 => { s.cost = 10; s.health = 40.0; s.attack = 4.0; s.defense = 4.0; s.movement = 2; s.range = 1; s.float = true; s.carry = true; s.stiff = true; } // Juggernaut
        _ => {}
    }
    s
}

fn tech_tier(tech: i32) -> i32 {
    match tech {
        1 | 6 | 10 | 15 | 20 => 1,
        2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2,
        3 | 5 | 9 | 12 | 14 | 17 | 19 | 22 | 24 | 38 => 3,
        _ => 1,
    }
}

fn max_hp(u: &Unit) -> f32 {
    let base_kind = if u.passenger > 0 { u.passenger } else { u.kind };
    let base = unit_stats(base_kind).health;
    let calc = base + if u.veteran { 5.0 } else { 0.0 };
    calc.max(u.health)
}

fn unit_star_value(u: &Unit) -> f32 {
    let st = unit_stats(u.kind);
    let pass_cost = if u.passenger > 0 { unit_stats(u.passenger).cost as f32 } else { 0.0 };
    let base = match u.kind {
        12 | 45 => 18.0, // Giant / Juggernaut
        _ => st.cost as f32 + pass_cost + 1.5,
    };
    base + if u.veteran { 2.0 } else { 0.0 }
}

fn combat_calc(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    let divisor = if total > 0.0 { total } else { 1.0 };
    let attack = ((if total > 0.0 { (af / divisor) * aa * 4.5 } else { 0.0 }) + 0.500001).floor();
    let defense = ((if dh > attack && total > 0.0 { (df / divisor) * dd * 4.5 } else { 0.0 }) + 0.500001).floor();
    (attack, defense, (attack * 0.5).floor())
}

#[inline]
fn cheb(size: usize, a: usize, b: usize) -> i32 {
    let (ax, ay) = ((a % size) as i32, (a / size) as i32);
    let (bx, by) = ((b % size) as i32, (b / size) as i32);
    (ax - bx).abs().max((ay - by).abs())
}

fn neighbors(size: usize, t: usize) -> Vec<usize> {
    let s = size as i32;
    let (x, y) = ((t % size) as i32, (t / size) as i32);
    let mut out = Vec::with_capacity(8);
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && nx < s && ny >= 0 && ny < s {
                out.push((ny * s + nx) as usize);
            }
        }
    }
    out
}
struct ObsState {
    me: i32,
    opp: i32,
    size: usize,
    n: usize,
    turn: i32,
    stars: i32,
    tech: [bool; 128],
    tiles: Vec<Tile>,
    units: Vec<Unit>,
    unit_at: Vec<Option<usize>>, // tile -> index in units
    cities: Vec<City>,
    city_at: Vec<Option<usize>>, // tile -> index in cities
    legal: Vec<Cmd>,
}

impl ObsState {
    fn from_value(obs: &Value, mem: &mut PersistentMemory) -> Self {
        let me = obs["player"].as_i64().unwrap_or(1) as i32;
        let opp = 3 - me;
        let v = &obs["visible_state"];
        let size = v["board_size"].as_u64().unwrap_or(14) as usize;
        let n = size * size;
        let turn = v["turn"].as_i64().unwrap_or(0) as i32;
        let stars = v["stars"].as_i64().unwrap_or(0) as i32;

        let mut tech = [false; 128];
        if let Some(arr) = v["technology"].as_array() {
            for t in arr {
                if let Some(id) = t.as_i64() {
                    if id >= 0 && (id as usize) < 128 {
                        tech[id as usize] = true;
                    }
                }
            }
        }

        let mut tiles = vec![Tile::default(); n];
        if let Some(arr) = v["tiles"].as_array() {
            for row in arr {
                let idx = row["tile"].as_u64().unwrap_or(0) as usize;
                if idx < n {
                    let known = row["known"].as_bool().unwrap_or(false);
                    tiles[idx].known = known;
                    if known {
                        tiles[idx].terrain = row["terrain"].as_i64().unwrap_or(0) as i32;
                        tiles[idx].owner = row["owner"].as_i64().unwrap_or(0) as i32;
                        tiles[idx].structure = row["structure"].as_i64().unwrap_or(0) as i32;
                        tiles[idx].resource = row["resource"].as_i64().unwrap_or(0) as i32;
                        tiles[idx].road = row["road"].as_i64().unwrap_or(0) != 0;
                        tiles[idx].capital = row["capital"].as_i64().unwrap_or(0) as i32;
                        tiles[idx].structure_level = row["structure_level"].as_i64().unwrap_or(0) as i32;
                    }
                }
            }
        }

        let mut units = Vec::new();
        let mut unit_at = vec![None; n];
        if let Some(arr) = v["units"].as_array() {
            for row in arr {
                let tile = row["tile"].as_u64().unwrap_or(0) as usize;
                let owner = row["owner"].as_i64().unwrap_or(0) as i32;
                let u = Unit {
                    tile,
                    owner,
                    kind: row["kind"].as_i64().unwrap_or(0) as i32,
                    health: row["health"].as_f64().unwrap_or(10.0) as f32,
                    veteran: row["veteran"].as_bool().unwrap_or(false),
                    moved: row["moved"].as_bool().unwrap_or(false),
                    attacked: row["attacked"].as_bool().unwrap_or(false),
                    poison: row["poison"].as_bool().unwrap_or(false),
                    frozen: row["frozen"].as_bool().unwrap_or(false),
                    boosted: row["boosted"].as_bool().unwrap_or(false),
                    invisible: row["invisible"].as_bool().unwrap_or(false),
                    kills: row["kills"].as_i64().unwrap_or(0) as i32,
                    home: row["home"].as_i64().unwrap_or(0) as i32,
                    passenger: row["passenger"].as_i64().unwrap_or(0) as i32,
                    detect: row["detect"].as_bool().unwrap_or(false),
                };
                if tile < n {
                    unit_at[tile] = Some(units.len());
                    units.push(u);
                }
            }
        }

        let mut cities = Vec::new();
        let mut city_at = vec![None; n];
        if let Some(arr) = v["cities"].as_array() {
            for row in arr {
                let tile = row["tile"].as_u64().unwrap_or(0) as usize;
                let owner = row["owner"].as_i64().unwrap_or(0) as i32;
                let mut rewards = [0i32; 9];
                if let Some(rarr) = row["rewards"].as_array() {
                    for (k, v) in rarr.iter().enumerate().take(9) {
                        rewards[k] = v.as_i64().unwrap_or(0) as i32;
                    }
                }
                let c = City {
                    tile,
                    owner,
                    level: row["level"].as_i64().unwrap_or(1) as i32,
                    population: row["population"].as_i64().unwrap_or(0) as i32,
                    progress: row["progress"].as_i64().unwrap_or(0) as i32,
                    production: row["production"].as_i64().unwrap_or(1) as i32,
                    connected: row["connected"].as_bool().unwrap_or(false),
                    riot: row["riot"].as_bool().unwrap_or(false),
                    rewards,
                };
                if tile < n {
                    city_at[tile] = Some(cities.len());
                    cities.push(c);
                }
            }
        }

        let mut legal = Vec::new();
        if let Some(arr) = obs["legal_actions"].as_array() {
            for a in arr {
                let action_id = a["action_id"].as_str().unwrap_or("").to_owned();
                let c = &a["command"];
                legal.push(Cmd {
                    action_id,
                    kind: c["kind"].as_i64().unwrap_or(0) as i32,
                    source: c["source"].as_i64().unwrap_or(-1) as i32,
                    target: c["target"].as_i64().unwrap_or(-1) as i32,
                    param: c["parameter"].as_i64().unwrap_or(0) as i32,
                });
            }
        }

        // Update persistent memory
        if !mem.initialized || mem.n != n {
            mem.initialized = true;
            mem.size = size;
            mem.n = n;
            mem.me = me;
            mem.opp = opp;
            mem.prev_owner = vec![0; n];
            mem.tile_city = vec![-1; n];
            for c in &cities {
                if c.owner == me {
                    mem.my_capital = Some(c.tile);
                    let (cx, cy) = (c.tile % size, c.tile / size);
                    let (ox, oy) = ((size - 1).saturating_sub(cx), (size - 1).saturating_sub(cy));
                    mem.opp_capital_guess = oy * size + ox;
                }
            }
        }
        if turn != mem.last_turn {
            mem.last_turn = turn;
            mem.unit_prev_tile.clear();
        }
        for (i, t) in tiles.iter().enumerate() {
            if t.known {
                if t.capital == opp {
                    mem.opp_capital_guess = i;
                }
                if t.structure == 1 {
                    if t.owner == opp {
                        mem.known_opp_cities.insert(i, opp);
                    } else if t.owner == me {
                        mem.known_opp_cities.remove(&i);
                    }
                }
                if t.owner == me && mem.prev_owner[i] != me {
                    if let Some(cc) = mem.last_claim_city {
                        mem.tile_city[i] = cc as i32;
                    }
                }
                if t.owner != me {
                    mem.tile_city[i] = -1;
                }
                mem.prev_owner[i] = t.owner;
            }
        }
        mem.last_claim_city = None;
        // Fallback attribution for any owned tile with unknown city
        for i in 0..n {
            if tiles[i].known && tiles[i].owner == me {
                let cur = mem.tile_city[i];
                let valid_cur = cur >= 0 && city_at.get(cur as usize).copied().flatten().map(|ci| cities[ci].owner == me).unwrap_or(false);
                if !valid_cur {
                    let mut best_c = -1;
                    let mut best_d = 99;
                    for c in &cities {
                        if c.owner == me {
                            let rad = if c.rewards[5] > 0 { 2 } else { 1 };
                            let d = cheb(size, c.tile, i);
                            if d <= rad && d < best_d {
                                best_d = d;
                                best_c = c.tile as i32;
                            }
                        }
                    }
                    if best_c < 0 {
                        for c in &cities {
                            if c.owner == me {
                                let d = cheb(size, c.tile, i);
                                if d <= 2 && d < best_d {
                                    best_d = d;
                                    best_c = c.tile as i32;
                                }
                            }
                        }
                    }
                    mem.tile_city[i] = best_c;
                }
            }
        }

        Self {
            me, opp, size, n, turn, stars, tech, tiles, units, unit_at, cities, city_at, legal,
        }
    }

    fn my_cities(&self) -> Vec<&City> {
        self.cities.iter().filter(|c| c.owner == self.me).collect()
    }

    fn city_unit_count(&self, city_tile: usize) -> i32 {
        let home_id = city_tile as i32 + 1;
        self.units.iter().filter(|u| u.owner == self.me && u.home == home_id).count() as i32
    }

    fn defender_bonus(&self, tile: usize, u: &Unit) -> f32 {
        let t = &self.tiles[tile];
        let st = unit_stats(u.kind);
        let bonus = if t.structure == 1 && t.owner == u.owner && st.fortify {
            if let Some(ci) = self.city_at[tile] {
                if self.cities[ci].owner == self.me && self.cities[ci].rewards[1] > 0 {
                    4.0
                } else {
                    1.5
                }
            } else {
                1.5
            }
        } else if t.terrain == 1 || t.terrain == 2 {
            if u.owner == self.me && self.tech[12] { 1.5 } else { 1.0 }
        } else if t.terrain == 5 {
            if u.owner == self.me && self.tech[18] { 1.5 } else if u.owner != self.me { 1.5 } else { 1.0 }
        } else if t.terrain == 4 {
            if u.owner == self.me && self.tech[20] { 1.5 } else if u.owner != self.me { 1.5 } else { 1.0 }
        } else {
            1.0
        };
        bonus * if u.poison { 0.5 } else { 1.0 }
    }

    fn sim_attack(&self, src: usize, dst: usize) -> (f32, f32, bool, bool) {
        let Some(si) = self.unit_at[src] else { return (0.0, 0.0, false, false); };
        let Some(di) = self.unit_at[dst] else { return (0.0, 0.0, false, false); };
        let att = &self.units[si];
        let def = &self.units[di];
        let ast = unit_stats(att.kind);
        let dst_st = unit_stats(def.kind);
        let hp_s = max_hp(att);
        let hp_t = max_hp(def);
        let dd = dst_st.defense * if def.poison { 0.7 } else { 1.0 };
        let bonus = self.defender_bonus(dst, def);
        let (dmg, ret, _) = combat_calc(ast.attack, att.health, hp_s, dd, def.health, hp_t, bonus);
        let def_dead = def.health - dmg <= 0.0;
        let can_ret = !ast.surprise && !dst_st.stiff && cheb(self.size, src, dst) <= dst_st.range;
        let actual_ret = if def_dead || !can_ret { 0.0 } else { ret };
        let att_dead = att.health - actual_ret <= 0.0;
        (dmg, actual_ret, def_dead, att_dead)
    }

    fn max_enemy_threat_at(&self, tile: usize, ignore_enemy_at: Option<usize>) -> f32 {
        let mut total_dmg = 0.0;
        for u in &self.units {
            if u.owner == self.opp && Some(u.tile) != ignore_enemy_at {
                let st = unit_stats(u.kind);
                if st.attack <= 0.0 {
                    continue;
                }
                let reach = if st.dash { st.movement + st.range } else { st.range };
                let d = cheb(self.size, u.tile, tile);
                if d <= reach {
                    let af = st.attack * (u.health / max_hp(u));
                    let df = 2.0;
                    let dmg = ((af / (af + df)) * st.attack * 4.5 + 0.5).floor();
                    total_dmg += dmg;
                }
            }
        }
        total_dmg
    }
}
impl ObsState {
    fn compute_bfs_distances(&self, targets: &[usize], aquatic: bool, allow_unexplored: bool) -> Vec<i32> {
        let mut dist = vec![INF; self.n];
        let mut q = VecDeque::new();
        for &t in targets {
            if t < self.n {
                dist[t] = 0;
                q.push_back(t);
            }
        }
        while let Some(u) = q.pop_front() {
            let du = dist[u];
            for v in neighbors(self.size, u) {
                let tv = &self.tiles[v];
                let passable = if !tv.known {
                    allow_unexplored
                } else if aquatic {
                    tv.terrain == 1 || tv.terrain == 2 || (tv.structure == 1 && tv.owner != self.me)
                } else {
                    let water = tv.terrain == 1 || tv.terrain == 2;
                    let bridge = tv.structure == 48;
                    let own_port = tv.structure == 8 && tv.owner == self.me;
                    let mtn_ok = tv.terrain != 4 || self.tech[20] || self.turn >= 4;
                    (!water && mtn_ok) || bridge || own_port
                };
                if passable && dist[v] > du + 1 {
                    dist[v] = du + 1;
                    q.push_back(v);
                }
            }
        }
        dist
    }

    fn choose_action(&self, mem: &mut PersistentMemory) -> Cmd {
        // 1. Mandatory City Rewards (kind == 9)
        if self.legal.iter().any(|c| c.kind == 9) {
            let mut best: Option<&Cmd> = None;
            let mut best_score = -INF;
            for c in &self.legal {
                if c.kind != 9 {
                    continue;
                }
                let city_tile = c.target as usize;
                let s = match c.param {
                    // Level 2: Workshop (3) vs Explorer (4)
                    3 => 1000,
                    4 => {
                        // Explorer only if non-capital early on with lots of fog nearby, Actually Workshop is +1 star/turn forever!
                        // Wait: Explorer can be good on a central city if turn <= 4, but Workshop is almost always better in 200-round games!
                        800
                    }
                    // Level 3: Resources +5 stars (7) vs City Wall (1)
                    7 => 1000,
                    1 => {
                        let threatened = self.units.iter().any(|u| u.owner == self.opp && cheb(self.size, u.tile, city_tile) <= 2);
                        if threatened && self.stars >= 8 { 1100 } else { 500 }
                    }
                    // Level 4: Population +3 (8) vs Border Growth (5)
                    8 => {
                        let ci = self.city_at[city_tile].unwrap_or(0);
                        let prog = self.cities.get(ci).map(|ct| ct.progress).unwrap_or(0);
                        // Level 4 -> 5 needs 5 pop. +3 pop gets us to prog + 3!
                        if prog + 3 >= 5 {
                            1200 // Instant Giant!
                        } else {
                            // Check how many neutral resources/forests are in ring 2
                            let mut ring2_val = 0;
                            for t in 0..self.n {
                                if cheb(self.size, city_tile, t) == 2 && self.tiles[t].known && self.tiles[t].owner == 0 {
                                    if self.tiles[t].resource > 0 || self.tiles[t].terrain == 5 {
                                        ring2_val += 1;
                                    }
                                }
                            }
                            if ring2_val >= 4 && prog == 0 { 900 } else { 1050 }
                        }
                    }
                    5 => 1000,
                    // Level 5+: Giant (6) vs Park (2)
                    6 => 2000,
                    2 => 500,
                    _ => 100,
                };
                if s > best_score {
                    best_score = s;
                    best = Some(c);
                }
            }
            if let Some(b) = best {
                if b.param == 5 {
                    mem.last_claim_city = Some(b.target as usize);
                }
                return b.clone();
            }
        }

        // 2. Free Promotions! (kind == 3, param == 16) - costs NO action and heals to full +5 max HP!
        if let Some(c) = self.legal.iter().find(|c| c.kind == 3 && c.param == 16) {
            return c.clone();
        }

        // 3. Free Monuments! (kind == 6, param in 23..=29) - costs 0 stars and adds +3 pop!
        {
            let mut best_mon: Option<&Cmd> = None;
            let mut best_mon_score = -INF;
            for c in &self.legal {
                if c.kind == 6 && (23..=29).contains(&c.param) {
                    let t = c.target as usize;
                    let city_id = mem.tile_city[t];
                    let mut sc = 500;
                    if city_id >= 0 {
                        if let Some(ci) = self.city_at[city_id as usize] {
                            let ct = &self.cities[ci];
                            let needed = (ct.level + 1) - ct.progress;
                            if needed <= 3 {
                                sc += 1000; // Levels up the city for free!
                            } else {
                                sc += 300 - needed * 20;
                            }
                        }
                    }
                    // Prefer placing monument on tile without resource or sawmill/forge/market potential
                    if self.tiles[t].terrain == 1 {
                        sc += 50; // Putting monument on shallow water saves land tiles!
                    }
                    if sc > best_mon_score {
                        best_mon_score = sc;
                        best_mon = Some(c);
                    }
                }
            }
            if let Some(m) = best_mon {
                return m.clone();
            }
        }

        // 4. Ruin & Starfish captures (kind == 8 on structure == 2 or resource == 8)
        // Always do these before spending stars or researching because they can grant free techs or stars!
        for c in &self.legal {
            if c.kind == 8 {
                let src = c.source as usize;
                if self.tiles[src].structure == 2 || self.tiles[src].resource == 8 {
                    return c.clone();
                }
            }
        }

        // 5. Check if we have a pending village/city capture (kind == 8 on structure == 1).
        // BEFORE capturing a village (which increases our city count and therefore all tech costs by +tier),
        // check if there is a high-priority tech we want to research right now!
        let has_city_capture = self.legal.iter().any(|c| c.kind == 8 && self.tiles[c.source as usize].structure == 1);
        if has_city_capture {
            if let Some(tech_cmd) = self.pick_research(true) {
                return tech_cmd;
            }
            // Otherwise, capture the city/village now!
            // Prefer enemy city capture first, then neutral village
            let mut best_cap: Option<&Cmd> = None;
            let mut best_cap_score = -INF;
            for c in &self.legal {
                if c.kind == 8 && self.tiles[c.source as usize].structure == 1 {
                    let src = c.source as usize;
                    let sc = if self.tiles[src].owner == self.opp { 2000 } else { 1000 };
                    if sc > best_cap_score {
                        best_cap_score = sc;
                        best_cap = Some(c);
                    }
                }
            }
            if let Some(bc) = best_cap {
                mem.last_claim_city = Some(bc.source as usize);
                return bc.clone();
            }
        }

        // 6. Combat & Tactical Unit Actions (Attack + Move + Heal)
        // Wait: should we execute obviously winning attacks and unit moves BEFORE spending stars on economy/summons?
        // Let's think:
        // - Moving a unit OFF our own city frees the city tile so we can summon a unit there OR spawn a Giant there!
        // - Moving a unit into fog explores tiles (revealing resources/villages/lighthouses/enemies, and meeting opponent gives +3/+5 stars!).
        // - Attacking enemy units can kill an enemy sieging our city (freeing the city and allowing safe economy/summons).
        // Wait: what if we build a Road or Port first so a unit can move further or embark?
        // Let's check if building a Port or Road right now enables an idle unit to embark or reach an enemy/village!
        if let Some(infra_cmd) = self.pick_urgent_infrastructure(mem) {
            return infra_cmd;
        }

        // 7. Execute high-value attacks (kills, ranged/no-retaliation attacks, siege breaks, good trades)
        if let Some(att_cmd) = self.pick_attack() {
            return att_cmd;
        }

        // 8. Move units that are currently standing on our own cities (so they don't block summoning or Giant pushes!),
        // or units that can step onto a capturable village/ruin/starfish, or explore/advance!
        if let Some(mv_cmd) = self.pick_move(mem) {
            return mv_cmd;
        }

        // 9. Second pass at attacks (in case a Dash unit just moved into attack range!)
        if let Some(att_cmd) = self.pick_attack() {
            return att_cmd;
        }

        // 10. MindBender Heal Others (kind == 3, param == 9)
        if let Some(c) = self.legal.iter().find(|c| c.kind == 3 && c.param == 9) {
            return c.clone();
        }

        // 11. Economy & City Level-Up Planner + Research + Summoning!
        if let Some(econ_or_summon) = self.pick_economy_research_summon(mem) {
            return econ_or_summon;
        }

        // 12. After economy/research/summoning (which might have unlocked Climbing/Fishing/Sailing/Roads or built a Port/Road),
        // check if any units can now move or attack!
        if let Some(mv_cmd) = self.pick_move(mem) {
            return mv_cmd;
        }
        if let Some(att_cmd) = self.pick_attack() {
            return att_cmd;
        }

        // 13. End Turn (kind == 10)
        if let Some(end) = self.legal.iter().find(|c| c.kind == 10) {
            return end.clone();
        }
        self.legal.iter().find(|c| c.kind != 11).unwrap_or(&self.legal[0]).clone()
    }
}
impl ObsState {
    fn map_water_stats(&self) -> (usize, usize, bool) {
        let mut known_cnt = 0;
        let mut water_cnt = 0;
        let mut has_coastal_city = false;
        for i in 0..self.n {
            if self.tiles[i].known {
                known_cnt += 1;
                if self.tiles[i].terrain == 1 || self.tiles[i].terrain == 2 {
                    water_cnt += 1;
                }
                if self.tiles[i].owner == self.me && self.tiles[i].terrain == 1 {
                    has_coastal_city = true;
                }
            }
        }
        (known_cnt, water_cnt, has_coastal_city)
    }

    fn pick_research(&self, before_capture_only: bool) -> Option<Cmd> {
        // Don't research on turn 0 before leveling up capital
        if self.turn == 0 {
            return None;
        }
        let my_city_cnt = self.my_cities().len() as i32;
        let (known_cnt, water_cnt, has_coastal_tile) = self.map_water_stats();
        let water_ratio = if known_cnt > 0 { water_cnt as f32 / known_cnt as f32 } else { 0.0 };

        // Count harvestable / buildable resources in our territory (and nearby ring 2)
        let mut own_forests = 0;
        let mut own_fruits = 0;
        let mut own_crops = 0;
        let mut own_fish = 0;
        let mut own_metals = 0;
        let mut own_mountains = 0;
        let mut own_ports = 0;
        let mut own_lumber_huts = 0;
        for i in 0..self.n {
            if self.tiles[i].known && self.tiles[i].owner == self.me {
                if self.tiles[i].structure == 0 {
                    if self.tiles[i].terrain == 5 { own_forests += 1; }
                    if self.tiles[i].resource == 6 { own_fruits += 1; }
                    if self.tiles[i].resource == 2 { own_crops += 1; }
                    if self.tiles[i].resource == 3 { own_fish += 1; }
                    if self.tiles[i].resource == 5 { own_metals += 1; }
                    if self.tiles[i].terrain == 4 { own_mountains += 1; }
                }
                if self.tiles[i].structure == 8 { own_ports += 1; }
                if self.tiles[i].structure == 12 { own_lumber_huts += 1; }
            }
        }
        let has_rafts = self.units.iter().any(|u| u.owner == self.me && u.kind == 43);
        let has_enemy_ships = self.units.iter().any(|u| u.owner == self.opp && unit_stats(u.kind).float);
        let has_enemy_giants = self.units.iter().any(|u| u.owner == self.opp && (u.kind == 12 || u.kind == 45));

        let mut best: Option<&Cmd> = None;
        let mut best_score = -INF;

        for c in &self.legal {
            if c.kind != 7 {
                continue;
            }
            let tech_id = c.param;
            let tier = tech_tier(tech_id);
            let cost = 4 + my_city_cnt * tier;
            let stars_after = self.stars - cost;

            let mut sc = match tech_id {
                16 => { // Forestry (Lumber Hut + Chop) - Bardur bread & butter!
                    let mut v = 350 + own_forests * 65;
                    if stars_after >= 3 && own_forests >= 1 { v += 150; }
                    v
                }
                1 => { // Riding (Rider) - essential mobility & expansion!
                    let mut v = 420;
                    if self.turn <= 5 { v += 150; }
                    if water_ratio < 0.35 { v += 80; }
                    v
                }
                6 => { // Organization (Fruit)
                    let mut v = 220 + own_fruits * 90;
                    if stars_after >= 2 && own_fruits >= 1 { v += 120; }
                    if own_fruits == 0 { v -= 150; }
                    v
                }
                10 => { // Fishing (Fish + Port)
                    let mut v = 200 + own_fish * 95;
                    if has_coastal_tile && (water_ratio >= 0.18 || self.turn >= 5) {
                        v += 220;
                        if water_ratio >= 0.30 { v += 200; }
                    }
                    if stars_after >= 2 && own_fish >= 1 { v += 100; }
                    if !has_coastal_tile && own_fish == 0 { v -= 250; }
                    v
                }
                4 => { // Roads
                    let mut v = 260 + my_city_cnt * 65;
                    if self.tech[1] && my_city_cnt >= 2 && stars_after >= 3 { v += 120; }
                    v
                }
                17 => { // Mathematics (Sawmill + Catapult)
                    let mut v = 280 + own_lumber_huts * 60 + own_forests * 25;
                    if has_enemy_giants { v += 180; }
                    if stars_after >= 5 && own_lumber_huts >= 2 { v += 120; }
                    v
                }
                18 => { // Archery (Archer + Forest Defense)
                    let mut v = 270 + my_city_cnt * 20;
                    if self.units.iter().any(|u| u.owner == self.opp) { v += 100; }
                    v
                }
                13 => { // Sailing (Scoutship + Ocean movement)
                    let mut v = 180;
                    if own_ports > 0 || has_rafts { v += 350; }
                    if water_ratio >= 0.25 && has_coastal_tile { v += 200; }
                    v
                }
                44 => { // Ramming (Rammership)
                    let mut v = 180;
                    if own_ports > 0 || has_rafts || has_enemy_ships { v += 360; }
                    if water_ratio >= 0.25 && has_coastal_tile { v += 180; }
                    v
                }
                14 => { // Navigation (Bomber + Starfish)
                    let mut v = 160;
                    if own_ports > 0 || has_rafts { v += 280; }
                    if has_enemy_ships { v += 150; }
                    v
                }
                20 => { // Climbing
                    let mut v = 180 + own_mountains * 35;
                    if self.turn >= 6 && own_mountains >= 2 { v += 90; }
                    v
                }
                23 => { // Mining
                    let mut v = 200 + own_metals * 110;
                    if own_metals == 0 { v -= 150; }
                    if stars_after >= 5 && own_metals >= 1 { v += 120; }
                    v
                }
                24 => { // Smithery (Swordsman + Forge)
                    320 + own_metals * 50
                }
                8 => { // Farming
                    let mut v = 200 + own_crops * 100;
                    if own_crops == 0 { v -= 150; }
                    if stars_after >= 5 && own_crops >= 1 { v += 120; }
                    v
                }
                2 => { // Free Spirit
                    if self.tech[4] && my_city_cnt >= 3 && self.stars >= cost + 8 { 240 } else { 110 }
                }
                3 => { // Chivalry (Knight)
                    420
                }
                7 => { // Strategy (Defender)
                    if has_coastal_tile && own_ports > 0 { 260 } else { 210 }
                }
                5 => { // Trade (Market)
                    if own_lumber_huts >= 3 && my_city_cnt >= 3 { 280 } else { 140 }
                }
                12 => { // Aquatism (Water Defense)
                    if has_enemy_ships && own_ports > 0 { 280 } else { 140 }
                }
                _ => 100,
            };

            if before_capture_only {
                // Only research before capturing a village if it's genuinely useful (score >= 320)
                // or we have plenty of stars left!
                sc += tier * 40; // Bonus for saving `tier` stars by researching before capture!
                if sc < 360 {
                    continue;
                }
            } else {
                // In normal flow, don't spend our last stars on a tech if we can't use it this turn
                // unless we have surplus stars or it unlocks a unit/vision we want next turn
                let unlocks_unit = matches!(tech_id, 1 | 3 | 7 | 13 | 17 | 18 | 24 | 44);
                let immediate_use = (tech_id == 16 && stars_after >= 3 && own_forests > 0)
                    || (tech_id == 6 && stars_after >= 2 && own_fruits > 0)
                    || (tech_id == 10 && ((stars_after >= 2 && own_fish > 0) || (stars_after >= 7 && has_coastal_tile)))
                    || (tech_id == 23 && stars_after >= 5 && own_metals > 0)
                    || (tech_id == 8 && stars_after >= 5 && own_crops > 0)
                    || (tech_id == 4 && stars_after >= 3 && my_city_cnt >= 2)
                    || (tech_id == 17 && stars_after >= 5 && own_lumber_huts >= 1)
                    || ((tech_id == 13 || tech_id == 44) && stars_after >= 5 && has_rafts)
                    || (unlocks_unit && stars_after >= 3);
                if !immediate_use && self.stars < cost + 4 && sc < 480 {
                    continue;
                }
            }

            if sc > best_score {
                best_score = sc;
                best = Some(c);
            }
        }
        best.cloned()
    }

    fn pick_urgent_infrastructure(&self, mem: &PersistentMemory) -> Option<Cmd> {
        // Check if building a Port (param == 8) adjacent to an idle land unit lets us embark this turn
        // when we need naval expansion!
        let (known_cnt, water_cnt, _) = self.map_water_stats();
        let water_ratio = if known_cnt > 0 { water_cnt as f32 / known_cnt as f32 } else { 0.0 };
        if water_ratio >= 0.20 {
            let own_ports = self.tiles.iter().filter(|t| t.known && t.owner == self.me && t.structure == 8).count();
            if own_ports == 0 || self.stars >= 12 {
                for c in &self.legal {
                    if c.kind == 6 && c.param == 8 {
                        let pt = c.target as usize;
                        // Is there an unmoved land unit adjacent to this shallow water tile?
                        let adj_idle_unit = self.units.iter().any(|u| {
                            u.owner == self.me && !u.moved && !unit_stats(u.kind).float && cheb(self.size, u.tile, pt) <= 1
                        });
                        if adj_idle_unit {
                            // Also check if this port helps level up its city or gives access to water
                            let _ = mem;
                            return Some(c.clone());
                        }
                    }
                }
            }
        }
        None
    }

    fn pick_attack(&self) -> Option<Cmd> {
        let mut best: Option<&Cmd> = None;
        let mut best_score: f32 = 0.0; // Must be > 0 to attack!

        for c in &self.legal {
            if c.kind != 2 {
                continue;
            }
            let src = c.source as usize;
            let dst = c.target as usize;
            let Some(si) = self.unit_at[src] else { continue; };
            let att = &self.units[si];
            let ast = unit_stats(att.kind);

            // Cloak Infiltrate is always huge value!
            if ast.infiltrate {
                return Some(c.clone());
            }
            // MindBender Convert is always huge value!
            if att.kind == 10 {
                return Some(c.clone());
            }

            let Some(di) = self.unit_at[dst] else { continue; };
            let def = &self.units[di];
            let (dmg, ret, def_dead, att_dead) = self.sim_attack(src, dst);

            let att_val = unit_star_value(att);
            let def_val = unit_star_value(def);
            let hp_s = max_hp(att);
            let hp_t = max_hp(def);

            let mut sc = 0.0;
            // Value of damage dealt vs retaliation taken
            sc += (dmg / hp_t) * def_val * 12.0;
            sc -= (ret / hp_s) * att_val * 10.0;

            if def_dead {
                sc += def_val * 15.0 + 40.0;
                // Persist chain bonus for Knights!
                if ast.persist {
                    sc += 80.0;
                }
                // Check if melee unit advances onto dst
                if ast.range < 2 && !att_dead {
                    // Advancing onto an enemy city or village!
                    if self.tiles[dst].structure == 1 && self.tiles[dst].owner != self.me {
                        sc += 120.0;
                    }
                    // Check if advancing steps OFF our own city that was under threat
                    if self.tiles[src].structure == 1 && self.tiles[src].owner == self.me {
                        let enemy_near_city = self.units.iter().any(|u| {
                            u.owner == self.opp && u.tile != dst && cheb(self.size, u.tile, src) <= 2
                        });
                        if enemy_near_city {
                            sc -= 90.0; // Don't abandon our own city to an adjacent enemy!
                        }
                    }
                    // Check threat at dst if we advance
                    let threat = self.max_enemy_threat_at(dst, Some(dst));
                    if threat >= att.health && att_val > def_val + 2.0 {
                        sc -= (att_val - def_val) * 8.0;
                    }
                }
            } else if att_dead {
                sc -= att_val * 20.0 + 50.0;
            } else {
                // Neither died: check if another of our units can finish off `def` this turn!
                let rem_hp = def.health - dmg;
                let mut finisher_exists = false;
                for u2 in &self.units {
                    if u2.owner == self.me && u2.tile != src && !u2.attacked {
                        let st2 = unit_stats(u2.kind);
                        let reach2 = if !u2.moved && st2.dash { st2.movement + st2.range } else { st2.range };
                        if cheb(self.size, u2.tile, dst) <= reach2 && st2.attack > 0.0 {
                            finisher_exists = true;
                            break;
                        }
                    }
                }
                if finisher_exists && rem_hp <= 6.0 {
                    sc += 35.0; // Good setup for focus-fire kill!
                    // Prefer ranged or high-HP units to soften up before melee finisher!
                    if ast.range >= 2 {
                        sc += 25.0;
                    }
                }
                if ret == 0.0 {
                    sc += 25.0; // Free chip damage (ranged or stiff defender)!
                }
                // Rider Escape bonus: if Rider hasn't moved or gets Escape after attack
                if ast.escape {
                    sc += 15.0;
                }
            }

            // Bonus for attacking an enemy unit that is sieging our city!
            if self.tiles[dst].structure == 1 && self.tiles[dst].owner == self.me {
                sc += 100.0;
            }
            // Bonus for Splash (Bomber) hitting adjacent enemies
            if ast.splash {
                for u2 in &self.units {
                    if u2.owner == self.opp && cheb(self.size, dst, u2.tile) == 1 {
                        sc += 35.0;
                    }
                }
            }

            if sc > best_score {
                best_score = sc;
                best = Some(c);
            }
        }
        best.cloned()
    }
}
impl ObsState {
    fn pick_move(&self, mem: &mut PersistentMemory) -> Option<Cmd> {
        // Precompute target sets for BFS
        let mut land_targets = Vec::new();
        let mut water_targets = Vec::new();
        let mut fog_targets = Vec::new();

        for i in 0..self.n {
            let t = &self.tiles[i];
            if !t.known {
                fog_targets.push(i);
                continue;
            }
            // Unoccupied neutral village or enemy city
            let occupied_by_me = self.unit_at[i].map(|ui| self.units[ui].owner == self.me).unwrap_or(false);
            if t.structure == 1 && t.owner != self.me && !occupied_by_me {
                land_targets.push(i);
                water_targets.push(i);
            }
            if t.structure == 2 && !occupied_by_me {
                land_targets.push(i);
                if t.terrain <= 2 {
                    water_targets.push(i);
                }
            }
            if t.resource == 8 && self.tech[14] && !occupied_by_me {
                water_targets.push(i);
            }
            if t.structure == 8 && t.owner == self.me && !occupied_by_me {
                // Own empty port is a potential embark target for land units if water map
                let (known_cnt, water_cnt, _) = self.map_water_stats();
                if known_cnt > 0 && (water_cnt as f32 / known_cnt as f32) >= 0.22 {
                    land_targets.push(i);
                }
            }
        }
        for (&opp_city, _) in &mem.known_opp_cities {
            if !land_targets.contains(&opp_city) {
                land_targets.push(opp_city);
                water_targets.push(opp_city);
            }
        }
        for u in &self.units {
            if u.owner == self.opp {
                land_targets.push(u.tile);
                water_targets.push(u.tile);
            }
        }
        if land_targets.is_empty() {
            land_targets.extend_from_slice(&fog_targets);
            land_targets.push(mem.opp_capital_guess);
        }
        if water_targets.is_empty() {
            water_targets.extend_from_slice(&fog_targets);
            water_targets.push(mem.opp_capital_guess);
        }

        let bfs_land = self.compute_bfs_distances(&land_targets, false, true);
        let bfs_water = self.compute_bfs_distances(&water_targets, true, true);
        let bfs_fog = self.compute_bfs_distances(&fog_targets, false, true);

        let mut best: Option<&Cmd> = None;
        let mut best_score: f32 = 0.0; // Only take moves with positive score!

        for c in &self.legal {
            if c.kind != 1 {
                continue;
            }
            let src = c.source as usize;
            let dst = c.target as usize;
            let Some(si) = self.unit_at[src] else { continue; };
            let u = &self.units[si];
            let st = unit_stats(u.kind);

            let mut sc = 0.0;

            // 1. Check if unit is currently sitting on an enemy city / neutral village / ruin / starfish!
            // NEVER move off a capturable tile unless we already attacked this turn (e.g. Rider Escape) AND the tile is NOT a city/village!
            let on_capturable = (self.tiles[src].structure == 1 && self.tiles[src].owner != self.me)
                || self.tiles[src].structure == 2
                || (self.tiles[src].resource == 8 && self.tech[14]);
            if on_capturable {
                if self.tiles[src].structure == 1 && self.tiles[src].owner != self.me {
                    continue; // Hold the village/city to capture next turn!
                }
                if !u.attacked {
                    continue;
                }
            }

            // 2. Check if unit is standing on our own city!
            if self.tiles[src].structure == 1 && self.tiles[src].owner == self.me {
                // Only stay on our own city if an enemy can capture it next turn and we have Fortify!
                let enemy_can_reach_city = self.units.iter().any(|eu| {
                    eu.owner == self.opp && cheb(self.size, eu.tile, src) <= unit_stats(eu.kind).movement
                });
                if enemy_can_reach_city && u.health >= 5.0 && st.fortify {
                    sc -= 60.0;
                } else {
                    // Vacating our own city so we can train a unit or spawn a Giant is high priority!
                    sc += 45.0;
                }
            }

            // 3. Stepping directly onto a capturable village / enemy city / ruin / starfish!
            let dst_t = &self.tiles[dst];
            if dst_t.structure == 1 && dst_t.owner != self.me {
                if dst_t.owner == self.opp {
                    sc += 500.0; // Sieging an enemy city!
                    if dst_t.capital == self.opp {
                        sc += 200.0; // Enemy capital!
                    }
                    // Prefer high-HP / Fortify units to hold the siege
                    sc += u.health * 3.0 + if st.fortify { 25.0 } else { 0.0 };
                } else {
                    sc += 420.0; // Neutral village!
                }
                // Check if an enemy next to dst will immediately kill our unit before we can capture
                let threat = self.max_enemy_threat_at(dst, None);
                if threat >= u.health {
                    sc -= 180.0;
                }
            } else if dst_t.structure == 2 {
                sc += 340.0; // Ruin!
            } else if dst_t.resource == 8 && self.tech[14] && dst_t.structure == 0 {
                sc += 310.0; // Starfish (+8 stars next turn)!
            }

            // 4. Embarking / Disembarking rules!
            let embarking = dst_t.structure == 8 && dst_t.owner == self.me && !st.float;
            let disembarking = st.carry && dst_t.terrain > 2 && dst_t.structure != 8;
            if embarking {
                // Don't embark expensive Knights/Catapults if Warriors/Riders/Defenders/Giants are available
                if u.kind == 8 || u.kind == 4 {
                    sc -= 120.0;
                } else if u.kind == 12 {
                    sc += 90.0; // Giant -> Juggernaut is awesome on water maps!
                } else {
                    sc += 40.0;
                }
            }
            if disembarking {
                // Never disembark a Bomber (15 stars) or Rammership/Scoutship unless stepping directly onto a village/city!
                if u.kind == 41 {
                    if !(dst_t.structure == 1 && dst_t.owner != self.me) {
                        continue;
                    }
                    sc -= 200.0;
                } else if u.kind == 42 || u.kind == 44 {
                    if !(dst_t.structure == 1 && dst_t.owner != self.me) && dst_t.structure != 2 {
                        sc -= 60.0;
                    }
                } else if u.kind == 43 {
                    // Raft disembarking onto new land is great!
                    sc += 40.0;
                }
            }

            // 5. Dash attack opportunity from `dst`!
            if !u.attacked && st.dash && !embarking && !disembarking {
                let mut best_dash_att = 0.0f32;
                for eu in &self.units {
                    if eu.owner == self.opp && cheb(self.size, dst, eu.tile) <= st.range {
                        let ast = st;
                        let dst_st = unit_stats(eu.kind);
                        let hp_s = max_hp(u);
                        let hp_t = max_hp(eu);
                        let dd = dst_st.defense * if eu.poison { 0.7 } else { 1.0 };
                        let bonus = self.defender_bonus(eu.tile, eu);
                        let (dmg, ret, _) = combat_calc(ast.attack, u.health, hp_s, dd, eu.health, hp_t, bonus);
                        let def_dead = eu.health - dmg <= 0.0;
                        let can_ret = !ast.surprise && !dst_st.stiff && cheb(self.size, dst, eu.tile) <= dst_st.range;
                        let actual_ret = if def_dead || !can_ret { 0.0 } else { ret };
                        let mut att_sc = (dmg / hp_t) * unit_star_value(eu) * 12.0 - (actual_ret / hp_s) * unit_star_value(u) * 10.0;
                        if def_dead {
                            att_sc += unit_star_value(eu) * 15.0 + 40.0;
                        }
                        if eu.tile < self.n && self.tiles[eu.tile].structure == 1 && self.tiles[eu.tile].owner == self.me {
                            att_sc += 90.0; // Break siege on our city!
                        }
                        if att_sc > best_dash_att {
                            best_dash_att = att_sc;
                        }
                    }
                }
                if best_dash_att > 0.0 {
                    sc += best_dash_att + 30.0;
                }
            }

            // 6. Fog reveal value!
            let sight = if dst_t.terrain == 4 || u.kind == 42 || u.kind == 38 || u.kind == 39 { 2 } else { 1 };
            let mut new_fog = 0;
            for f in 0..self.n {
                if !self.tiles[f].known && cheb(self.size, dst, f) <= sight {
                    new_fog += 1;
                    // Corner lighthouse bonus!
                    let (fx, fy) = (f % self.size, f / self.size);
                    if (fx == 0 || fx == self.size - 1) && (fy == 0 || fy == self.size - 1) {
                        new_fog += 3;
                    }
                }
            }
            sc += new_fog as f32 * 14.0;

            // 7. Distance progress toward targets!
            let d_map = if st.float || embarking { &bfs_water } else { &bfs_land };
            let d_src = d_map[src].min(30);
            let d_dst = d_map[dst].min(30);
            sc += (d_src - d_dst) as f32 * 18.0;

            if new_fog == 0 && d_src == d_dst {
                // Fallback to fog BFS
                let f_src = bfs_fog[src].min(30);
                let f_dst = bfs_fog[dst].min(30);
                sc += (f_src - f_dst) as f32 * 8.0;
            }

            // 8. Positioning for ranged units (Catapults, Archers, Bombers) and low-HP retreat!
            let threat_src = self.max_enemy_threat_at(src, None);
            let threat_dst = self.max_enemy_threat_at(dst, None);
            if u.attacked && st.escape {
                // Rider Escape after attacking: retreat out of threat or onto defensive terrain!
                sc += (threat_src - threat_dst) * 12.0;
            } else if u.health <= 4.0 && threat_src > 0.0 {
                // Wounded unit retreating to safety
                sc += (threat_src - threat_dst) * 10.0;
                if dst_t.owner == self.me {
                    sc += 15.0; // Friendly territory heals +4!
                }
            } else if st.range >= 2 {
                // Ranged unit (Archer/Catapult/Bomber/Scoutship): want to be at range 2..=st.range of enemies, NOT adjacent!
                let min_enemy_d = self.units.iter().filter(|eu| eu.owner == self.opp).map(|eu| cheb(self.size, dst, eu.tile)).min().unwrap_or(99);
                if min_enemy_d == 1 && u.kind == 8 {
                    sc -= 80.0; // Don't walk Catapult adjacent to enemy!
                } else if (2..=st.range).contains(&min_enemy_d) {
                    sc += 35.0;
                }
            }

            // Terrain defense bonus on destination if enemy nearby
            if threat_dst > 0.0 {
                if (dst_t.terrain == 5 && self.tech[18]) || (dst_t.terrain == 4 && self.tech[20]) {
                    sc += 12.0;
                }
            }

            // Avoid oscillating back to previous tile
            if mem.unit_prev_tile.get(&src) == Some(&dst) && !u.attacked {
                sc -= 25.0;
            }

            if sc > best_score {
                best_score = sc;
                best = Some(c);
            }
        }

        if let Some(b) = best {
            mem.unit_prev_tile.insert(b.target as usize, b.source as usize);
            return Some(b.clone());
        }
        None
    }
}
impl ObsState {
    fn pick_economy_research_summon(&self, mem: &PersistentMemory) -> Option<Cmd> {
        // Turn 0 special check: harvest the 2 animals around our capital to reach Level 2 + Workshop!
        if self.turn == 0 {
            for c in &self.legal {
                if c.kind == 5 {
                    return Some(c.clone());
                }
            }
        }

        // First: check if any of our cities can be leveled up THIS TURN with exact or near-exact population!
        // Let's collect all legal population-adding actions per city:
        // - Harvest (kind == 5): cost 2, pop +1
        // - Lumber Hut (kind == 6, param == 12): cost 3, pop +1 (plus +1 if city already has a Sawmill adjacent!)
        // - Farm (kind == 6, param == 5): cost 5, pop +2 (plus +1 if adjacent Windmill!)
        // - Mine (kind == 6, param == 21): cost 5, pop +2 (plus +2 if adjacent Forge!)
        // - Port (kind == 6, param == 8): cost 7, pop +1 (plus connection pop if within 4 water tiles of connected port!)
        // - Sawmill (kind == 6, param == 13): cost 5, pop = count of adjacent own Lumber Huts
        // - Windmill (kind == 6, param == 6): cost 5, pop = count of adjacent own Farms
        // - Forge (kind == 6, param == 22): cost 5, pop = 2 * count of adjacent own Mines
        struct PopItem<'a> {
            cmd: &'a Cmd,
            city_tile: usize,
            cost: i32,
            pop: i32,
            extra_bonus: i32,
        }

        let mut items: Vec<PopItem> = Vec::new();
        for c in &self.legal {
            if c.kind == 5 {
                let t = c.target as usize;
                let cid = mem.tile_city[t];
                if cid >= 0 {
                    items.push(PopItem { cmd: c, city_tile: cid as usize, cost: 2, pop: 1, extra_bonus: 10 });
                }
            } else if c.kind == 6 {
                let t = c.target as usize;
                let cid = mem.tile_city[t];
                if cid < 0 {
                    continue;
                }
                let city_tile = cid as usize;
                match c.param {
                    12 => {
                        // Lumber Hut: check if adjacent to an own Sawmill!
                        let adj_sawmill = neighbors(self.size, t).iter().any(|&nb| {
                            self.tiles[nb].owner == self.me && self.tiles[nb].structure == 13
                        });
                        // Also check if `t` is the ONLY flat field/forest spot for a future Sawmill/Forge, wait: Lumber Hut is on forest (5), Sawmill is on field (3)!
                        // Wait: Forge can be on forest (5), Market is on field (3). So Lumber Hut on forest never blocks Sawmill or Market!
                        let pop = 1 + if adj_sawmill { 1 } else { 0 };
                        items.push(PopItem { cmd: c, city_tile, cost: 3, pop, extra_bonus: 8 });
                    }
                    5 => {
                        let adj_wm = neighbors(self.size, t).iter().any(|&nb| {
                            self.tiles[nb].owner == self.me && self.tiles[nb].structure == 6
                        });
                        let pop = 2 + if adj_wm { 1 } else { 0 };
                        items.push(PopItem { cmd: c, city_tile, cost: 5, pop, extra_bonus: 6 });
                    }
                    21 => {
                        let adj_forge = neighbors(self.size, t).iter().any(|&nb| {
                            self.tiles[nb].owner == self.me && self.tiles[nb].structure == 22
                        });
                        let pop = 2 + if adj_forge { 2 } else { 0 };
                        items.push(PopItem { cmd: c, city_tile, cost: 5, pop, extra_bonus: 8 });
                    }
                    13 => {
                        // Sawmill: count adjacent own Lumber Huts
                        let adj_lh = neighbors(self.size, t).iter().filter(|&&nb| {
                            self.tiles[nb].owner == self.me && self.tiles[nb].structure == 12
                        }).count() as i32;
                        if adj_lh >= 1 {
                            items.push(PopItem { cmd: c, city_tile, cost: 5, pop: adj_lh, extra_bonus: adj_lh * 12 });
                        }
                    }
                    6 => {
                        let adj_fm = neighbors(self.size, t).iter().filter(|&&nb| {
                            self.tiles[nb].owner == self.me && self.tiles[nb].structure == 5
                        }).count() as i32;
                        if adj_fm >= 1 {
                            items.push(PopItem { cmd: c, city_tile, cost: 5, pop: adj_fm, extra_bonus: adj_fm * 10 });
                        }
                    }
                    22 => {
                        let adj_mn = neighbors(self.size, t).iter().filter(|&&nb| {
                            self.tiles[nb].owner == self.me && self.tiles[nb].structure == 21
                        }).count() as i32;
                        if adj_mn >= 1 {
                            items.push(PopItem { cmd: c, city_tile, cost: 5, pop: adj_mn * 2, extra_bonus: adj_mn * 15 });
                        }
                    }
                    8 => {
                        // Port: cost 7, pop 1 (+ connection if applicable)
                        let (known_cnt, water_cnt, _) = self.map_water_stats();
                        let water_ratio = if known_cnt > 0 { water_cnt as f32 / known_cnt as f32 } else { 0.0 };
                        let bonus = if water_ratio >= 0.20 { 35 } else { -10 };
                        items.push(PopItem { cmd: c, city_tile, cost: 7, pop: 1, extra_bonus: bonus });
                    }
                    _ => {}
                }
            }
        }

        // Check if any city can level up THIS TURN using available PopItems!
        let mut best_levelup_cmd: Option<&Cmd> = None;
        let mut best_levelup_roi = -INF;

        for city in self.my_cities() {
            // Don't level up a city that is currently sieged by an enemy unit unless we are spawning a Giant (level 4 -> 5+)!
            let sieged = self.unit_at[city.tile].map(|ui| self.units[ui].owner == self.opp).unwrap_or(false);
            if sieged && city.level < 4 {
                continue;
            }
            let needed = (city.level + 1) - city.progress;
            if needed <= 0 {
                continue;
            }
            let mut city_items: Vec<&PopItem> = items.iter().filter(|it| it.city_tile == city.tile && it.pop > 0).collect();
            // Sort by best cost-per-pop (lowest cost*10 / pop, then highest extra_bonus)
            city_items.sort_by_key(|it| (it.cost * 10 / it.pop.max(1), -it.extra_bonus));

            let mut acc_pop = 0;
            let mut acc_cost = 0;
            let mut first_cmd: Option<&Cmd> = None;
            for it in &city_items {
                if acc_pop >= needed {
                    break;
                }
                if acc_cost + it.cost <= self.stars {
                    acc_pop += it.pop;
                    acc_cost += it.cost;
                    if first_cmd.is_none() {
                        first_cmd = Some(it.cmd);
                    }
                }
            }
            if acc_pop >= needed {
                if let Some(fcmd) = first_cmd {
                    // Reward value of leveling up `city` from `city.level` to `city.level + 1`:
                    let reward_val = match city.level {
                        1 => 140, // Level 2: Workshop (+2 stars/turn total!)
                        2 => 160, // Level 3: +5 stars immediately + +1 star/turn! (Actually net cost is acc_cost - 5!)
                        3 => 150, // Level 4: +3 pop (sets up Giant!) or Border Growth + +1 star/turn!
                        _ => 220, // Level 5+: GIANT!
                    };
                    let roi = reward_val - acc_cost * 12;
                    if roi > best_levelup_roi {
                        best_levelup_roi = roi;
                        best_levelup_cmd = Some(fcmd);
                    }
                }
            }
        }

        if let Some(lcmd) = best_levelup_cmd {
            if best_levelup_roi >= 0 {
                return Some(lcmd.clone());
            }
        }

        // Check if building a Market (kind == 6, param == 50) gives immediate stars/turn!
        for c in &self.legal {
            if c.kind == 6 && c.param == 50 {
                let t = c.target as usize;
                let mut adj_levels = 0;
                for nb in neighbors(self.size, t) {
                    if self.tiles[nb].owner == self.me && matches!(self.tiles[nb].structure, 6 | 13 | 22) {
                        adj_levels += self.tiles[nb].structure_level.max(1);
                    }
                }
                if adj_levels >= 2 {
                    return Some(c.clone());
                }
            }
        }

        // Check if a 1-tile Road (cost 3) or Bridge (cost 5) immediately connects an unconnected city to our capital!
        if let Some(cap_tile) = mem.my_capital {
            // Find all tiles currently connected to our capital
            let mut conn = vec![false; self.n];
            let mut q = VecDeque::new();
            if self.tiles[cap_tile].owner == self.me {
                conn[cap_tile] = true;
                q.push_back(cap_tile);
            }
            while let Some(u) = q.pop_front() {
                for v in neighbors(self.size, u) {
                    if !conn[v] && self.tiles[v].known && self.tiles[v].owner != self.opp {
                        let is_node = self.tiles[v].road
                            || self.tiles[v].structure == 48
                            || self.tiles[v].structure == 8
                            || (self.tiles[v].structure == 1 && self.tiles[v].owner == self.me);
                        if is_node {
                            conn[v] = true;
                            q.push_back(v);
                        }
                    }
                }
            }
            // If a single Road (`param == 3`) or Bridge (`param == 48`) is adjacent to both a `conn` tile
            // and an unconnected own city, building it costs 3 stars and gives +1 pop to that city AND +1 pop to capital!
            for c in &self.legal {
                if c.kind == 6 && (c.param == 3 || c.param == 48) {
                    let t = c.target as usize;
                    let nbs = neighbors(self.size, t);
                    let touches_conn = nbs.iter().any(|&nb| conn[nb]);
                    let touches_unconn_city = nbs.iter().any(|&nb| {
                        self.city_at[nb].map(|ci| self.cities[ci].owner == self.me && !self.cities[ci].connected && nb != cap_tile).unwrap_or(false)
                    });
                    if touches_conn && touches_unconn_city {
                        return Some(c.clone());
                    }
                }
            }
        }

        // Next: Summon units in threatened cities or frontline cities, or upgrade Rafts!
        // Wait: should we research before summoning, or summon before researching?
        // If we are under threat or have empty city capacity and few units, summon first!
        // Wait: if we have 0 Riding/Archery/Forestry and can research a key tech right now, let's check `pick_research(false)`!
        let my_unit_cnt = self.units.iter().filter(|u| u.owner == self.me).count();
        let opp_unit_cnt = self.units.iter().filter(|u| u.owner == self.opp).count();
        let any_city_threatened = self.my_cities().iter().any(|ct| {
            self.units.iter().any(|eu| eu.owner == self.opp && cheb(self.size, eu.tile, ct.tile) <= 3)
        });

        if !any_city_threatened && my_unit_cnt >= 2 {
            if let Some(rcmd) = self.pick_research(false) {
                return Some(rcmd);
            }
        }

        // Evaluate Summoning & Raft Upgrades (`kind == 4`)
        if let Some(scmd) = self.pick_summon(mem) {
            return Some(scmd);
        }

        // If we didn't summon (or after summoning), check Research (`kind == 7`)
        if let Some(rcmd) = self.pick_research(false) {
            return Some(rcmd);
        }

        // Partial city population progress if we have surplus stars (or if it's high-efficiency like Sawmill/Forge/Harvest)
        if !items.is_empty() {
            let mut best_it: Option<&PopItem> = None;
            let mut best_sc = -INF;
            for it in &items {
                let Some(ci) = self.city_at[it.city_tile] else { continue; };
                let ct = &self.cities[ci];
                let needed = (ct.level + 1) - ct.progress;
                // Only invest partial pop if we have enough stars or if it's a Port on a water map
                let is_port = it.cmd.kind == 6 && it.cmd.param == 8;
                if !is_port && self.stars < it.cost + 3 && needed > it.pop {
                    continue;
                }
                let sc = it.pop * 40 - it.cost * 10 + it.extra_bonus - needed * 8;
                if sc > best_sc {
                    best_sc = sc;
                    best_it = Some(it);
                }
            }
            if let Some(bit) = best_it {
                if best_sc >= 5 {
                    return Some(bit.cmd.clone());
                }
            }
        }

        // Chopping forests (`kind == 3, param == 2`) for +1 star!
        // Only chop a forest if doing so unlocks an immediate purchase (like leveling up a city, researching a tech, or summoning a unit)
        // OR if the city has plenty of forests and we need 1 star!
        if let Some(chop_cmd) = self.pick_chop(mem) {
            return Some(chop_cmd);
        }

        None
    }

    fn pick_summon(&self, mem: &PersistentMemory) -> Option<Cmd> {
        let mut best: Option<&Cmd> = None;
        let mut best_score = 0;

        let (known_cnt, water_cnt, _) = self.map_water_stats();
        let water_ratio = if known_cnt > 0 { water_cnt as f32 / known_cnt as f32 } else { 0.0 };
        let has_enemy_ships = self.units.iter().any(|u| u.owner == self.opp && unit_stats(u.kind).float);

        for c in &self.legal {
            if c.kind != 4 {
                continue;
            }
            let src = c.source as usize;
            let kind = c.param;

            // Is this a Raft upgrade (41=Bomber, 42=Scoutship, 44=Rammership)?
            if matches!(kind, 41 | 42 | 44) {
                let Some(ui) = self.unit_at[src] else { continue; };
                let u = &self.units[ui];
                let enemy_near = self.units.iter().any(|eu| eu.owner == self.opp && cheb(self.size, src, eu.tile) <= 4);
                let mut sc = match kind {
                    42 => { // Scoutship (cost 5, move 3, range 2, sight 2!)
                        let mut v = 180;
                        if !u.moved { v += 80; } // Can move 3 tiles and shoot immediately!
                        if self.turn <= 15 { v += 40; } // Great for naval exploration!
                        v
                    }
                    44 => { // Rammership (cost 5, 3 atk, 3 def, move 3!)
                        let mut v = 200;
                        if !u.moved { v += 80; }
                        if enemy_near || has_enemy_ships { v += 60; }
                        v
                    }
                    41 => { // Bomber (cost 15, 3 atk, range 3, splash!)
                        if enemy_near || self.stars >= 20 { 220 } else { 90 }
                    }
                    _ => 0,
                };
                if u.health <= 5.0 {
                    sc -= 100; // Don't spend stars upgrading a nearly dead Raft!
                }
                if sc > best_score {
                    best_score = sc;
                    best = Some(c);
                }
                continue;
            }

            // Otherwise, summoning a land unit at city `src`
            // Check if leveling up this city to Level 5 (Giant!) is imminent this turn — don't block the city tile!
            let enemy_dist = self.units.iter().filter(|eu| eu.owner == self.opp).map(|eu| cheb(self.size, src, eu.tile)).min().unwrap_or(99);
            let target_dist = cheb(self.size, src, mem.opp_capital_guess);

            let mut sc = match kind {
                4 => { // Knight (cost 8, 3.5 atk, 3 move, Persist!)
                    let low_hp_enemies = self.units.iter().filter(|eu| eu.owner == self.opp && eu.health <= 10.0 && unit_stats(eu.kind).defense <= 2.0).count() as i32;
                    240 + low_hp_enemies * 35
                }
                11 => { // Swordsman (cost 5, 15 HP, 3 atk, 3 def, Dash!)
                    250
                }
                3 => { // Rider (cost 3, 2 atk, 2 move, Dash, Escape!)
                    let mut v = 210;
                    if self.turn <= 12 && water_ratio < 0.40 { v += 40; }
                    if enemy_dist == 1 { v -= 40; } // Fragile when spawning right next to an enemy
                    v
                }
                9 => { // Archer (cost 3, 2 atk, range 2, Dash!)
                    let mut v = 205;
                    if enemy_dist <= 3 { v += 35; }
                    v
                }
                8 => { // Catapult (cost 8, 4 atk, range 3)
                    let mut v = 190;
                    if enemy_dist <= 2 {
                        v -= 150; // Never spawn a 0-defense Catapult next to an enemy!
                    } else if enemy_dist <= 5 {
                        v += 55;
                    }
                    // Don't spam too many Catapults
                    let my_cats = self.units.iter().filter(|u| u.owner == self.me && u.kind == 8).count();
                    if my_cats >= 3 { v -= 60; }
                    v
                }
                5 => { // Defender (cost 3, 15 HP, 1 atk, 3 def, Fortify)
                    let mut v = 160;
                    if enemy_dist <= 2 { v += 80; } // Great for holding a threatened city!
                    // Great passenger for boats!
                    let has_port = neighbors(self.size, src).iter().any(|&nb| self.tiles[nb].owner == self.me && self.tiles[nb].structure == 8);
                    if has_port && water_ratio >= 0.25 { v += 60; }
                    v
                }
                2 => { // Warrior (cost 2, 10 HP, 2 atk, 2 def)
                    let mut v = 175;
                    if self.turn <= 6 { v += 30; }
                    if enemy_dist <= 2 && !self.tech[7] && !self.tech[24] { v += 30; }
                    v
                }
                38 => { // Cloak (cost 8)
                    let my_cloaks = self.units.iter().filter(|u| u.owner == self.me && (u.kind == 38 || u.kind == 39)).count();
                    if my_cloaks == 0 && enemy_dist >= 3 { 195 } else { 80 }
                }
                10 => { // MindBender (cost 5)
                    90
                }
                _ => 100,
            };

            // Frontline cities get priority for summoning!
            sc += (20 - enemy_dist.min(20)) * 4;
            sc += (14 - target_dist.min(14)) * 2;

            if sc > best_score {
                best_score = sc;
                best = Some(c);
            }
        }

        best.cloned()
    }

    fn pick_chop(&self, mem: &PersistentMemory) -> Option<Cmd> {
        // Chop forest (`kind == 3, param == 2`) gives +1 star.
        // Do NOT chop if we have Mathematics (17) or plan to build Lumber Huts there, UNLESS:
        // 1) We have 1 star less than needed to level up a city or buy a unit/tech this turn, OR
        // 2) The tile has an enemy unit nearby and we need emergency stars for a defender!
        for c in &self.legal {
            if c.kind == 3 && c.param == 2 {
                let t = c.target as usize;
                // Never chop a forest that has a wild animal (`resource == 1`)!
                if self.tiles[t].resource == 1 {
                    continue;
                }
                let cid = mem.tile_city[t];
                if cid >= 0 {
                    if let Some(ci) = self.city_at[cid as usize] {
                        let ct = &self.cities[ci];
                        let needed = (ct.level + 1) - ct.progress;
                        // If we have `stars == 1` and there's a harvestable animal/fruit/fish (cost 2) that levels up the city:
                        if self.stars == 1 && needed == 1 {
                            return Some(c.clone());
                        }
                        // If we have `stars == 2` and a Lumber Hut (cost 3) on ANOTHER forest levels up the city:
                        if self.stars == 2 && needed == 1 {
                            return Some(c.clone());
                        }
                    }
                }
            }
        }
        None
    }
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = PersistentMemory::new();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break; };
        if line.trim().is_empty() {
            continue;
        }
        let obs: Value = serde_json::from_str(&line).expect("valid obs json");
        let state = ObsState::from_value(&obs, &mut mem);
        let chosen = state.choose_action(&mut mem);
        writeln!(stdout, "{}", json!({"action_id": chosen.action_id})).unwrap();
        stdout.flush().unwrap();
    }
}
