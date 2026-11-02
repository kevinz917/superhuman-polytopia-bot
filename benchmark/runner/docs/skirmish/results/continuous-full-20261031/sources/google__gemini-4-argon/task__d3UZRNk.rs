use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

const INF: i32 = 1_000_000;

#[derive(Clone, Copy, Debug)]
struct UnitStats {
    cost: i32,
    hp: f32,
    atk: f32,
    def: f32,
    mov: i32,
    rng: i32,
    dash: bool,
    escape: bool,
    persist: bool,
    fortify: bool,
    stiff: bool,
    splash: bool,
    convert: bool,
    infiltrate: bool,
    float: bool,
    carry: bool,
    scout: bool,
    stat: bool,
}

fn unit_stats(kind: i32) -> UnitStats {
    let mut s = UnitStats {
        cost: 0, hp: 10.0, atk: 0.0, def: 1.0, mov: 1, rng: 1,
        dash: false, escape: false, persist: false, fortify: false,
        stiff: false, splash: false, convert: false, infiltrate: false,
        float: false, carry: false, scout: false, stat: false,
    };
    match kind {
        2 => { s.cost = 2; s.hp = 10.0; s.atk = 2.0; s.def = 2.0; s.mov = 1; s.rng = 1; s.dash = true; s.fortify = true; } // Warrior
        3 => { s.cost = 3; s.hp = 10.0; s.atk = 2.0; s.def = 1.0; s.mov = 2; s.rng = 1; s.dash = true; s.escape = true; s.fortify = true; } // Rider
        4 => { s.cost = 8; s.hp = 10.0; s.atk = 3.5; s.def = 1.0; s.mov = 3; s.rng = 1; s.dash = true; s.fortify = true; s.persist = true; } // Knight
        5 => { s.cost = 3; s.hp = 15.0; s.atk = 1.0; s.def = 3.0; s.mov = 1; s.rng = 1; s.fortify = true; } // Defender
        8 => { s.cost = 8; s.hp = 10.0; s.atk = 4.0; s.def = 0.0; s.mov = 1; s.rng = 3; s.stiff = true; } // Catapult
        9 => { s.cost = 3; s.hp = 10.0; s.atk = 2.0; s.def = 1.0; s.mov = 1; s.rng = 2; s.dash = true; s.fortify = true; } // Archer
        10 => { s.cost = 5; s.hp = 10.0; s.atk = 0.0; s.def = 1.0; s.mov = 1; s.rng = 1; s.convert = true; s.stiff = true; } // MindBender
        11 => { s.cost = 5; s.hp = 15.0; s.atk = 3.0; s.def = 3.0; s.mov = 1; s.rng = 1; s.dash = true; } // Swordsman
        12 => { s.cost = 10; s.hp = 40.0; s.atk = 5.0; s.def = 4.0; s.mov = 1; s.rng = 1; s.stat = true; } // Giant
        37 => { s.cost = 2; s.hp = 10.0; s.atk = 2.0; s.def = 2.0; s.mov = 1; s.rng = 1; s.dash = true; s.stat = true; } // Dagger
        38 => { s.cost = 8; s.hp = 5.0; s.atk = 0.0; s.def = 0.5; s.mov = 2; s.rng = 1; s.dash = true; s.infiltrate = true; s.scout = true; s.stat = true; s.stiff = true; } // Cloak
        39 => { s.cost = 2; s.hp = 5.0; s.atk = 0.0; s.def = 0.5; s.mov = 2; s.rng = 1; s.carry = true; s.dash = true; s.float = true; s.infiltrate = true; s.scout = true; s.stat = true; s.stiff = true; } // Dinghy
        40 => { s.cost = 2; s.hp = 10.0; s.atk = 2.0; s.def = 2.0; s.mov = 2; s.rng = 1; s.carry = true; s.dash = true; s.float = true; s.stat = true; } // Pirate
        41 => { s.cost = 15; s.hp = 10.0; s.atk = 3.0; s.def = 2.0; s.mov = 2; s.rng = 3; s.carry = true; s.float = true; s.splash = true; s.stat = true; s.stiff = true; } // Bomber
        42 => { s.cost = 5; s.hp = 10.0; s.atk = 2.0; s.def = 1.0; s.mov = 3; s.rng = 2; s.carry = true; s.dash = true; s.float = true; s.scout = true; s.stat = true; } // Scoutship
        43 => { s.cost = 0; s.hp = 10.0; s.atk = 0.0; s.def = 1.0; s.mov = 2; s.rng = 2; s.carry = true; s.float = true; s.stat = true; s.stiff = true; } // Raft
        44 => { s.cost = 5; s.hp = 10.0; s.atk = 3.0; s.def = 3.0; s.mov = 3; s.rng = 1; s.carry = true; s.dash = true; s.float = true; s.stat = true; } // Rammership
        45 => { s.cost = 10; s.hp = 40.0; s.atk = 4.0; s.def = 4.0; s.mov = 2; s.rng = 1; s.carry = true; s.float = true; s.stat = true; s.stiff = true; } // Juggernaut
        _ => {}
    }
    s
}

fn tech_tier(tech: usize) -> i32 {
    match tech {
        1 | 6 | 10 | 15 | 20 => 1,
        2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2,
        3 | 5 | 9 | 12 | 14 | 17 | 19 | 22 | 24 | 38 => 3,
        _ => 1,
    }
}

fn struct_cost(kind: i32) -> i32 {
    match kind {
        3 => 3, // Road
        5 => 5, // Farm
        6 => 5, // Windmill
        8 => 7, // Port
        12 => 3, // LumberHut
        13 => 5, // Sawmill
        17 => 20, // Temple
        18 => 15, // ForestTemple
        19 => 20, // WaterTemple
        20 => 20, // MountainTemple
        21 => 5, // Mine
        22 => 5, // Forge
        23..=29 => 0, // Monuments
        32 => 5, // Sanctuary
        48 => 5, // Bridge
        50 => 5, // Market
        70 => 5, // Embassy
        _ => 5,
    }
}

#[derive(Clone, Debug)]
struct Tile {
    known: bool,
    terrain: i32,
    owner: i32,
    structure: i32,
    resource: i32,
    road: i32,
    capital: i32,
    structure_level: i32,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
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
    detect: bool,
    kills: i32,
    home: i32,
    passenger: i32,
}

impl Unit {
    fn max_hp(&self) -> f32 {
        let base_kind = if self.passenger > 0 { self.passenger } else { self.kind };
        unit_stats(base_kind).hp + if self.veteran { 5.0 } else { 0.0 }
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct City {
    tile: usize,
    owner: i32,
    level: i32,
    progress: i32,
    population: i32,
    production: i32,
    connected: bool,
    riot: bool,
    rewards: [i32; 9],
}

#[derive(Clone, Debug)]
struct LegalAct {
    id: String,
    kind: i32,
    src: i32,
    dst: i32,
    param: i32,
}

struct State {
    player: i32,
    enemy: i32,
    size: usize,
    n: usize,
    turn: i32,
    stars: i32,
    tech: [bool; 128],
    tiles: Vec<Tile>,
    units: Vec<Unit>,
    unit_at: Vec<Option<usize>>,
    cities: Vec<City>,
    city_at: Vec<Option<usize>>,
    tile_city: Vec<Option<usize>>, // For own tiles, which own city index in `cities` owns it
    legal: Vec<LegalAct>,
}

struct BotMemory {
    last_player: i32,
    last_turn: i32,
    last_version: i32,
    tile_city_map: Vec<Option<usize>>, // tile -> city_tile
    city_capture_order: Vec<usize>,
    turn_moves: Vec<(usize, usize)>,
}

impl BotMemory {
    fn new() -> Self {
        Self {
            last_player: -1,
            last_turn: -1,
            last_version: -1,
            tile_city_map: Vec::new(),
            city_capture_order: Vec::new(),
            turn_moves: Vec::new(),
        }
    }
}

fn cheb(a: usize, b: usize, size: usize) -> i32 {
    let ax = (a % size) as i32;
    let ay = (a / size) as i32;
    let bx = (b % size) as i32;
    let by = (b / size) as i32;
    (ax - bx).abs().max((ay - by).abs())
}

fn combat_calc(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    let divisor = if total > 0.0 { total } else { 1.0 };
    let attack = ((if total > 0.0 { (af / divisor) * aa * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    let defense = ((if dh > attack && total > 0.0 { (df / divisor) * dd * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    (attack, defense, (attack * 0.5).floor())
}

impl State {
    fn from_obs(obs: &Value, mem: &mut BotMemory) -> Self {
        let player = obs["player"].as_i64().unwrap_or(1) as i32;
        let enemy = 3 - player;
        let version = obs["state_version"].as_i64().unwrap_or(0) as i32;
        let v = &obs["visible_state"];
        let size = v["board_size"].as_u64().unwrap_or(14) as usize;
        let n = size * size;
        let turn = v["turn"].as_i64().unwrap_or(0) as i32;
        let stars = v["stars"].as_i64().unwrap_or(0) as i32;

        if player != mem.last_player || turn < mem.last_turn || version < mem.last_version || mem.tile_city_map.len() != n {
            mem.tile_city_map = vec![None; n];
            mem.city_capture_order.clear();
            mem.turn_moves.clear();
        } else if turn != mem.last_turn {
            mem.turn_moves.clear();
        }
        mem.last_player = player;
        mem.last_turn = turn;
        mem.last_version = version;

        let mut tech = [false; 128];
        if let Some(arr) = v["technology"].as_array() {
            for t in arr {
                if let Some(k) = t.as_u64() {
                    if (k as usize) < 128 {
                        tech[k as usize] = true;
                    }
                }
            }
        }

        let mut tiles = vec![
            Tile {
                known: false, terrain: 0, owner: 0, structure: 0,
                resource: 0, road: 0, capital: 0, structure_level: 0
            };
            n
        ];
        if let Some(arr) = v["tiles"].as_array() {
            for t in arr {
                let idx = t["tile"].as_u64().unwrap_or(0) as usize;
                if idx < n {
                    let known = t["known"].as_bool().unwrap_or(false);
                    tiles[idx] = Tile {
                        known,
                        terrain: t["terrain"].as_i64().unwrap_or(0) as i32,
                        owner: t["owner"].as_i64().unwrap_or(0) as i32,
                        structure: t["structure"].as_i64().unwrap_or(0) as i32,
                        resource: t["resource"].as_i64().unwrap_or(0) as i32,
                        road: t["road"].as_i64().unwrap_or(0) as i32,
                        capital: t["capital"].as_i64().unwrap_or(0) as i32,
                        structure_level: t["structure_level"].as_i64().unwrap_or(0) as i32,
                    };
                }
            }
        }

        let mut units = Vec::new();
        let mut unit_at = vec![None; n];
        if let Some(arr) = v["units"].as_array() {
            for u in arr {
                let tile = u["tile"].as_u64().unwrap_or(0) as usize;
                let u_idx = units.len();
                let unit = Unit {
                    tile,
                    owner: u["owner"].as_i64().unwrap_or(0) as i32,
                    kind: u["kind"].as_i64().unwrap_or(0) as i32,
                    health: u["health"].as_f64().unwrap_or(10.0) as f32,
                    veteran: u["veteran"].as_bool().unwrap_or(false),
                    moved: u["moved"].as_bool().unwrap_or(false),
                    attacked: u["attacked"].as_bool().unwrap_or(false),
                    poison: u["poison"].as_bool().unwrap_or(false),
                    frozen: u["frozen"].as_bool().unwrap_or(false),
                    boosted: u["boosted"].as_bool().unwrap_or(false),
                    invisible: u["invisible"].as_bool().unwrap_or(false),
                    detect: u["detect"].as_bool().unwrap_or(false),
                    kills: u["kills"].as_i64().unwrap_or(0) as i32,
                    home: u["home"].as_i64().unwrap_or(0) as i32,
                    passenger: u["passenger"].as_i64().unwrap_or(0) as i32,
                };
                if tile < n {
                    unit_at[tile] = Some(u_idx);
                }
                units.push(unit);
            }
        }

        let mut cities = Vec::new();
        let mut city_at = vec![None; n];
        if let Some(arr) = v["cities"].as_array() {
            for c in arr {
                let tile = c["tile"].as_u64().unwrap_or(0) as usize;
                let owner = c["owner"].as_i64().unwrap_or(0) as i32;
                let mut rewards = [0i32; 9];
                if let Some(rarr) = c["rewards"].as_array() {
                    for (i, val) in rarr.iter().enumerate().take(9) {
                        rewards[i] = val.as_i64().unwrap_or(0) as i32;
                    }
                }
                let level = if owner == player {
                    c["level"].as_i64().unwrap_or(1) as i32
                } else if tile < n && tiles[tile].structure_level > 0 {
                    tiles[tile].structure_level
                } else {
                    1
                };
                let c_idx = cities.len();
                let city = City {
                    tile,
                    owner,
                    level,
                    progress: c["progress"].as_i64().unwrap_or(0) as i32,
                    population: c["population"].as_i64().unwrap_or(0) as i32,
                    production: c["production"].as_i64().unwrap_or(1) as i32,
                    connected: c["connected"].as_bool().unwrap_or(false),
                    riot: c["riot"].as_bool().unwrap_or(false),
                    rewards,
                };
                if tile < n {
                    city_at[tile] = Some(c_idx);
                }
                cities.push(city);
            }
        }

        // Update city capture order and tile_city_map for own cities
        mem.city_capture_order.retain(|&ct| city_at.get(ct).and_then(|&ci| ci).map_or(false, |ci| cities[ci].owner == player));
        for c in &cities {
            if c.owner == player && !mem.city_capture_order.contains(&c.tile) {
                mem.city_capture_order.push(c.tile);
            }
        }
        for t in 0..n {
            if tiles[t].owner != player {
                mem.tile_city_map[t] = None;
            } else if let Some(ct) = mem.tile_city_map[t] {
                if !mem.city_capture_order.contains(&ct) {
                    mem.tile_city_map[t] = None;
                }
            }
        }
        // Assign unassigned own tiles to own cities in capture order (radius 1 first, then radius 2 if border grown)
        for &ct in &mem.city_capture_order {
            if let Some(ci) = city_at[ct] {
                let rad = 1 + cities[ci].rewards[5];
                for t in 0..n {
                    if tiles[t].owner == player && mem.tile_city_map[t].is_none() && cheb(ct, t, size) <= rad {
                        mem.tile_city_map[t] = Some(ct);
                    }
                }
            }
        }
        // Fallback for any remaining owned tiles (e.g. captured enemy city with border growth)
        for t in 0..n {
            if tiles[t].owner == player && mem.tile_city_map[t].is_none() {
                if let Some(&ct) = mem.city_capture_order.iter().min_by_key(|&&ct| cheb(ct, t, size)) {
                    mem.tile_city_map[t] = Some(ct);
                }
            }
        }

        let mut tile_city = vec![None; n];
        for t in 0..n {
            if let Some(ct) = mem.tile_city_map[t] {
                tile_city[t] = city_at[ct];
            }
        }

        let mut legal = Vec::new();
        if let Some(arr) = obs["legal_actions"].as_array() {
            for a in arr {
                let id = a["action_id"].as_str().unwrap_or("").to_owned();
                let cmd = &a["command"];
                legal.push(LegalAct {
                    id,
                    kind: cmd["kind"].as_i64().unwrap_or(0) as i32,
                    src: cmd["source"].as_i64().unwrap_or(-1) as i32,
                    dst: cmd["target"].as_i64().unwrap_or(-1) as i32,
                    param: cmd["parameter"].as_i64().unwrap_or(0) as i32,
                });
            }
        }

        Self {
            player, enemy, size, n, turn, stars, tech,
            tiles, units, unit_at, cities, city_at, tile_city, legal,
        }
    }

    fn def_bonus(&self, tile: usize, u: &Unit) -> f32 {
        let t = &self.tiles[tile];
        let st = unit_stats(u.kind);
        let mut bonus = if t.terrain == 1 || t.terrain == 2 {
            if u.owner == self.player && self.tech[12] { 1.5 } else { 1.0 }
        } else if t.terrain == 5 {
            if u.owner != self.player || self.tech[18] { 1.5 } else { 1.0 }
        } else if t.terrain == 4 {
            if u.owner != self.player || self.tech[20] { 1.5 } else { 1.0 }
        } else if let Some(ci) = self.city_at[tile] {
            let c = &self.cities[ci];
            if c.owner == u.owner && st.fortify {
                if c.owner == self.player {
                    if c.rewards[1] > 0 { 4.0 } else { 1.5 }
                } else {
                    1.5
                }
            } else {
                1.0
            }
        } else {
            1.0
        };
        if u.poison {
            bonus *= 0.5;
        }
        bonus
    }

    fn attack_outcome(&self, src_tile: usize, dst_tile: usize) -> Option<(f32, f32, bool, bool)> {
        let su_idx = self.unit_at[src_tile]?;
        let du_idx = self.unit_at[dst_tile]?;
        let su = &self.units[su_idx];
        let du = &self.units[du_idx];
        let sst = unit_stats(su.kind);
        let dst_st = unit_stats(du.kind);
        let db = self.def_bonus(dst_tile, du);
        let dd = dst_st.def * if du.poison { 0.7 } else { 1.0 };
        let (dmg, ret, _) = combat_calc(sst.atk, su.health, su.max_hp(), dd, du.health, du.max_hp(), db);
        let def_dead = du.health - dmg <= 0.0;
        let dist = cheb(src_tile, dst_tile, self.size);
        let can_ret = !def_dead && ! dst_st.stiff && dist <= dst_st.rng && su.kind != 37 && su.kind != 40;
        let actual_ret = if can_ret { ret } else { 0.0 };
        let atk_dead = su.health - actual_ret <= 0.0;
        Some((dmg, actual_ret, def_dead, atk_dead))
    }

    fn my_capital(&self) -> Option<usize> {
        self.cities.iter().find(|c| c.owner == self.player && self.tiles[c.tile].capital == self.player).map(|c| c.tile)
            .or_else(|| self.cities.iter().find(|c| c.owner == self.player).map(|c| c.tile))
    }

    fn enemy_focus_tiles(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self.cities.iter().filter(|c| c.owner == self.enemy).map(|c| c.tile).collect();
        for t in 0..self.n {
            if self.tiles[t].known && self.tiles[t].capital == self.enemy && !out.contains(&t) {
                out.push(t);
            }
        }
        if out.is_empty() {
            for u in &self.units {
                if u.owner == self.enemy {
                    out.push(u.tile);
                }
            }
        }
        if out.is_empty() {
            for t in 0..self.n {
                if self.tiles[t].known && self.tiles[t].owner == self.enemy {
                    out.push(t);
                }
            }
        }
        if out.is_empty() {
            if let Some(cap) = self.my_capital() {
                let cx = cap % self.size;
                let cy = cap / self.size;
                let rx = (self.size - 1 - cx).clamp(2, self.size - 3);
                let ry = (self.size - 1 - cy).clamp(2, self.size - 3);
                out.push(ry * self.size + rx);
            } else {
                out.push(self.n / 2);
            }
        }
        out
    }
}

impl State {
    // Dijkstra distance from `targets` backwards: edge `v -> u` pays cost of entering `u`!
    fn dijkstra_from(&self, targets: &[usize], aquatic: bool) -> Vec<i32> {
        let mut dist = vec![INF; self.n];
        let mut visited = vec![false; self.n];
        for &t in targets {
            if t < self.n {
                dist[t] = 0;
            }
        }
        for _ in 0..self.n {
            let mut u = None;
            let mut best = INF;
            for i in 0..self.n {
                if !visited[i] && dist[i] < best {
                    best = dist[i];
                    u = Some(i);
                }
            }
            let u = match u {
                Some(x) => x,
                None => break,
            };
            visited[u] = true;
            let tu = &self.tiles[u];
            let ux = (u % self.size) as i32;
            let uy = (u / self.size) as i32;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = ux + dx;
                    let ny = uy + dy;
                    if nx < 0 || nx >= self.size as i32 || ny < 0 || ny >= self.size as i32 {
                        continue;
                    }
                    let v = (ny as usize) * self.size + (nx as usize);
                    if visited[v] {
                        continue;
                    }
                    let tv = &self.tiles[v];
                    // Cost of moving from `v` into `u`
                    let step_cost = if !tu.known {
                        3
                    } else {
                        match tu.terrain {
                            1 => {
                                // Entering shallow water `u` from `v`
                                if aquatic {
                                    2
                                } else if tu.structure == 8 && tu.owner == self.player {
                                    2 // Own port embarks!
                                } else if tu.structure == 48 {
                                    1 // Bridge
                                } else if self.tech[10] {
                                    if tv.known && (tv.terrain <= 2 || (tv.structure == 8 && tv.owner == self.player)) {
                                        2
                                    } else if tv.known && tv.terrain == 1 && tv.owner == self.player && self.stars >= 7 {
                                        8
                                    } else {
                                        50
                                    }
                                } else {
                                    80
                                }
                            }
                            2 => {
                                // Entering ocean `u` from `v`
                                if self.tech[13] {
                                    if aquatic || (tv.known && (tv.terrain <= 2 || (tv.structure == 8 && tv.owner == self.player))) {
                                        2
                                    } else {
                                        50
                                    }
                                } else {
                                    80
                                }
                            }
                            3 => {
                                if tu.road > 0 || tu.structure == 1 { 1 } else { 2 }
                            }
                            4 => {
                                if self.tech[20] { 3 } else { 80 }
                            }
                            5 => {
                                if tu.road > 0 { 1 } else { 3 }
                            }
                            _ => 3,
                        }
                    };
                    if best + step_cost < dist[v] {
                        dist[v] = best + step_cost;
                    }
                }
            }
        }
        dist
    }

    fn land_reachable(&self, start: usize) -> Vec<bool> {
        let mut reach = vec![false; self.n];
        let mut stack = vec![start];
        reach[start] = true;
        while let Some(u) = stack.pop() {
            let ux = (u % self.size) as i32;
            let uy = (u / self.size) as i32;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = ux + dx;
                    let ny = uy + dy;
                    if nx < 0 || nx >= self.size as i32 || ny < 0 || ny >= self.size as i32 { continue; }
                    let v = (ny as usize) * self.size + (nx as usize);
                    if reach[v] { continue; }
                    let tv = &self.tiles[v];
                    let ok = !tv.known || tv.terrain == 3 || tv.terrain == 5
                        || (tv.terrain == 4 && self.tech[20])
                        || tv.structure == 48;
                    if ok {
                        reach[v] = true;
                        stack.push(v);
                    }
                }
            }
        }
        reach
    }

    fn build_pop_gain(&self, tile: usize, param: i32) -> i32 {
        let tx = (tile % self.size) as i32;
        let ty = (tile / self.size) as i32;
        let count_adj = |s_kind: i32| -> i32 {
            let mut c = 0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = tx + dx;
                    let ny = ty + dy;
                    if nx >= 0 && nx < self.size as i32 && ny >= 0 && ny < self.size as i32 {
                        let nt = (ny as usize) * self.size + (nx as usize);
                        if self.tiles[nt].known && self.tiles[nt].owner == self.player && self.tiles[nt].structure == s_kind {
                            c += 1;
                        }
                    }
                }
            }
            c
        };
        match param {
            5 => 2 + count_adj(6),
            6 => count_adj(5),
            8 => 1,
            12 => 1 + count_adj(13),
            13 => count_adj(12),
            21 => 2 + 2 * count_adj(22),
            22 => 2 * count_adj(21),
            23..=29 => 3,
            17..=20 => 1,
            _ => 0,
        }
    }

    fn market_stars(&self, tile: usize) -> i32 {
        let tx = (tile % self.size) as i32;
        let ty = (tile / self.size) as i32;
        let mut sum = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 { continue; }
                let nx = tx + dx;
                let ny = ty + dy;
                if nx >= 0 && nx < self.size as i32 && ny >= 0 && ny < self.size as i32 {
                    let nt = (ny as usize) * self.size + (nx as usize);
                    let st = self.tiles[nt].structure;
                    if self.tiles[nt].known && self.tiles[nt].owner == self.player && (st == 6 || st == 13 || st == 22) {
                        sum += self.tiles[nt].structure_level.max(1);
                    }
                }
            }
        }
        sum.min(8)
    }

    fn connects_city(&self, tile: usize, param: i32) -> i32 {
        if param != 3 && param != 48 && param != 8 {
            return 0;
        }
        let cap = match self.my_capital() {
            Some(c) => c,
            None => return 0,
        };
        let has_unconnected = self.cities.iter().any(|c| c.owner == self.player && c.tile != cap && !c.connected);
        if !has_unconnected {
            return 0;
        }
        let mut nodes = vec![false; self.n];
        for i in 0..self.n {
            let ti = &self.tiles[i];
            let friendly = ti.known && (ti.owner == 0 || ti.owner == self.player);
            let is_city = self.city_at[i].map_or(false, |ci| self.cities[ci].owner == self.player);
            if friendly && (ti.road > 0 || ti.structure == 48 || ti.structure == 8 || is_city || i == tile) {
                nodes[i] = true;
            }
        }
        let mut reached = vec![false; self.n];
        let mut stack = vec![cap];
        reached[cap] = true;
        while let Some(u) = stack.pop() {
            let ux = (u % self.size) as i32;
            let uy = (u / self.size) as i32;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = ux + dx;
                    let ny = uy + dy;
                    if nx >= 0 && nx < self.size as i32 && ny >= 0 && ny < self.size as i32 {
                        let v = (ny as usize) * self.size + (nx as usize);
                        if !reached[v] && nodes[v] {
                            reached[v] = true;
                            stack.push(v);
                        }
                    }
                }
            }
            let u_is_port = self.tiles[u].structure == 8 || (u == tile && param == 8);
            if u_is_port {
                for v in 0..self.n {
                    let v_is_port = self.tiles[v].structure == 8 || (v == tile && param == 8);
                    if v_is_port && !reached[v] && cheb(u, v, self.size) <= 4 {
                        reached[v] = true;
                        stack.push(v);
                    }
                }
            }
        }
        let mut newly_connected = 0;
        for c in &self.cities {
            if c.owner == self.player && c.tile != cap && !c.connected && reached[c.tile] {
                newly_connected += 1;
            }
        }
        newly_connected
    }
}

fn choose_action(obs: &Value, mem: &mut BotMemory) -> String {
    let s = State::from_obs(obs, mem);
    if s.legal.is_empty() {
        return String::new();
    }

    // 0. Mandatory City Reward (kind == 9)
    if s.legal.iter().any(|a| a.kind == 9) {
        let mut best_id = s.legal[0].id.clone();
        let mut best_score = -INF;
        for a in &s.legal {
            if a.kind != 9 { continue; }
            let ct = a.dst.clamp(0, s.n as i32 - 1) as usize;
            let enemy_adj = s.units.iter().any(|u| u.owner == s.enemy && cheb(u.tile, ct, s.size) <= 1);
            let score = match a.param {
                // Slot 1 (L2): ALWAYS Workshop (+1 star/turn)!
                3 => 900,
                4 => 100,
                // Slot 2 (L3): 7 = Resources (+5 stars), 1 = City Wall
                7 => if enemy_adj && s.cities.iter().filter(|c| c.owner == s.player).count() <= 2 { 400 } else { 800 },
                1 => if enemy_adj { 650 } else { 200 },
                // Slot 3 (L4): 5 = Border Growth, 8 = Population (+3)
                8 => {
                    let reaches_l5 = s.city_at[ct].map_or(false, |ci| s.cities[ci].progress + 3 >= s.cities[ci].level + 1);
                    if reaches_l5 || enemy_adj { 900 } else { 650 }
                }
                5 => {
                    let unclaimed_res = (0..s.n).filter(|&t| {
                        cheb(ct, t, s.size) == 2 && s.tiles[t].known && s.tiles[t].owner == 0
                            && (s.tiles[t].resource > 0 || s.tiles[t].terrain == 5)
                    }).count() as i32;
                    600 + unclaimed_res * 70
                }
                // Slot 4+ (L5+): 6 = Giant, 2 = Park
                6 => 1000,
                2 => 100,
                _ => 0,
            };
            if score > best_score {
                best_score = score;
                best_id = a.id.clone();
            }
        }
        return best_id;
    }

    let own_cities: Vec<&City> = s.cities.iter().filter(|c| c.owner == s.player).collect();
    let num_own_cities = own_cities.len() as i32;
    let besieged_cities: Vec<usize> = own_cities.iter()
        .filter(|c| s.unit_at[c.tile].map_or(false, |ui| s.units[ui].owner == s.enemy))
        .map(|c| c.tile)
        .collect();

    let start_tile = s.my_capital().or_else(|| s.units.iter().find(|u| u.owner == s.player).map(|u| u.tile)).unwrap_or(0);
    let land_reach = s.land_reachable(start_tile);
    let focus_tiles = s.enemy_focus_tiles();
    let unexplored_land: usize = (0..s.n).filter(|&t| !s.tiles[t].known && land_reach[t]).count();
    let need_navy = focus_tiles.iter().any(|&ft| !land_reach[ft]) || unexplored_land < 8;

    // Assign neutral villages & accessible ruins to only the single closest friendly unit so our army doesn't all chase 1 back-corner ruin!
    let mut assigned_target: Vec<Option<usize>> = vec![None; s.n]; // unit_tile -> assigned village/ruin tile
    let mut neutral_prizes: Vec<usize> = Vec::new();
    for c in &s.cities {
        if c.owner == 0 {
            let blocked_by_own = s.unit_at[c.tile].map_or(false, |ui| s.units[ui].owner == s.player);
            if !blocked_by_own {
                neutral_prizes.push(c.tile);
            }
        }
    }
    for t in 0..s.n {
        if s.tiles[t].known && s.tiles[t].structure == 2 {
            // Only target mountain ruins if we already have Climbing (tech 20)!
            if s.tiles[t].terrain != 4 || s.tech[20] {
                let blocked_by_own = s.unit_at[t].map_or(false, |ui| s.units[ui].owner == s.player);
                if !blocked_by_own {
                    neutral_prizes.push(t);
                }
            }
        }
        if s.tech[14] && s.tiles[t].known && s.tiles[t].resource == 8 && s.tiles[t].structure == 0 {
            neutral_prizes.push(t);
        }
    }

    for &prize in &neutral_prizes {
        // Find the closest idle friendly unit to this prize
        let best_unit = s.units.iter()
            .filter(|u| u.owner == s.player && !u.moved && assigned_target[u.tile].is_none())
            .min_by_key(|u| cheb(u.tile, prize, s.size));
        if let Some(u) = best_unit {
            assigned_target[u.tile] = Some(prize);
        }
    }

    let enemy_city_targets: Vec<usize> = s.cities.iter()
        .filter(|c| c.owner == s.enemy && !s.unit_at[c.tile].map_or(false, |ui| s.units[ui].owner == s.player))
        .map(|c| c.tile)
        .collect();

    let mut attack_targets = enemy_city_targets.clone();
    if attack_targets.is_empty() {
        for u in &s.units {
            if u.owner == s.enemy {
                attack_targets.push(u.tile);
            }
        }
    }
    if attack_targets.is_empty() {
        attack_targets.extend(&focus_tiles);
    }

    let mut explore_targets: Vec<usize> = (0..s.n).filter(|&t| !s.tiles[t].known).collect();
    if explore_targets.is_empty() {
        explore_targets.extend(&attack_targets);
    }

    let dist_atk_land = s.dijkstra_from(&attack_targets, false);
    let dist_atk_sea = s.dijkstra_from(&attack_targets, true);
    let dist_exp_land = s.dijkstra_from(&explore_targets, false);
    let dist_exp_sea = s.dijkstra_from(&explore_targets, true);

    // City level-up evaluation
    let mut city_can_level = vec![false; s.n];
    let mut city_pop_need = vec![99; s.n];
    for c in &own_cities {
        let need = (c.level + 1) - c.progress;
        city_pop_need[c.tile] = need;
        if need <= 0 {
            city_can_level[c.tile] = true;
            continue;
        }
        let mut pop_opts: Vec<(i32, i32)> = Vec::new();
        for a in &s.legal {
            if a.kind == 5 {
                let t = a.dst as usize;
                if s.tile_city[t].map_or(false, |ci| s.cities[ci].tile == c.tile) {
                    pop_opts.push((2, 1));
                }
            } else if a.kind == 6 {
                let t = a.dst as usize;
                let conn = s.connects_city(t, a.param);
                if conn > 0 {
                    pop_opts.push((struct_cost(a.param), conn));
                } else if s.tile_city[t].map_or(false, |ci| s.cities[ci].tile == c.tile) {
                    let g = s.build_pop_gain(t, a.param);
                    if g > 0 {
                        pop_opts.push((struct_cost(a.param), g));
                    }
                }
            }
        }
        pop_opts.sort_by(|a, b| (a.0 * b.1).cmp(&(b.0 * a.1)));
        let mut rem_need = need;
        let mut total_cost = 0;
        for (cost, gain) in pop_opts {
            if rem_need <= 0 { break; }
            rem_need -= gain;
            total_cost += cost;
        }
        if rem_need <= 0 && total_cost <= s.stars {
            city_can_level[c.tile] = true;
        }
    }

    let any_city_can_level = own_cities.iter().any(|c| city_can_level[c.tile] && c.level <= 4);

    let mut best_action = &s.legal[0];
    let mut best_score: i32 = -INF;

    for a in &s.legal {
        let score = match a.kind {
            11 => -1_000_000,
            10 => 0,

            // Capture (village, enemy city, ruin, starfish)
            8 => {
                let src = a.src as usize;
                let st = s.tiles[src].structure;
                if st == 1 && s.tiles[src].owner == s.enemy {
                    50_000
                } else if st == 1 {
                    // Capturing neutral village increases tech cost, so let high-priority tech (score > 22_000) happen first if desired
                    21_000
                } else if st == 2 {
                    20_000
                } else {
                    19_000
                }
            }

            // Attack / Convert / Infiltrate
            2 => {
                let src = a.src as usize;
                let dst = a.dst as usize;
                let su = &s.units[s.unit_at[src].unwrap()];
                let sst = unit_stats(su.kind);
                if sst.infiltrate {
                    35_000
                } else if sst.convert {
                    let du = &s.units[s.unit_at[dst].unwrap()];
                    let dst_st = unit_stats(du.kind);
                    25_000 + dst_st.cost * 500 + (du.health as i32) * 100
                } else if let Some((dmg, ret, def_dead, atk_dead)) = s.attack_outcome(src, dst) {
                    let du = &s.units[s.unit_at[dst].unwrap()];
                    let dst_st = unit_stats(du.kind);
                    let on_our_city = besieged_cities.contains(&dst);
                    let on_enemy_city = s.city_at[dst].map_or(false, |ci| s.cities[ci].owner == s.enemy);
                    let is_ranged = cheb(src, dst, s.size) > 1 || ret == 0.0;

                    let can_promote_now = !su.veteran && su.kills >= 3 && !sst.stat && su.health < su.max_hp();
                    if can_promote_now {
                        -100
                    } else if def_dead && !atk_dead {
                        let mut sc = 15_000 + dst_st.cost * 300 + (dmg as i32) * 50 - (ret as i32) * 20;
                        if on_our_city { sc += 20_000; }
                        if on_enemy_city { sc += 10_000; }
                        if sst.persist { sc += 5_000; }
                        if on_enemy_city && cheb(src, dst, s.size) == 1 { sc += 2_000; }
                        sc
                    } else if on_our_city {
                        let mut sc = 13_000 + (dmg as i32) * 200 - (ret as i32) * 50;
                        if is_ranged { sc += 3_000; }
                        if atk_dead && dmg < 3.0 { sc = -500; }
                        sc
                    } else if is_ranged && dmg >= 1.0 {
                        10_500 + (dmg as i32) * 150 + dst_st.cost * 100 + if on_enemy_city { 2_000 } else { 0 }
                    } else if !atk_dead {
                        let val_dealt = (dmg / du.max_hp()) * (dst_st.cost.max(2) as f32) * 100.0;
                        let val_taken = (ret / su.max_hp()) * (sst.cost.max(2) as f32) * 100.0;
                        let mut ally_followup = false;
                        for u2 in &s.units {
                            if u2.owner == s.player && u2.tile != src && !u2.attacked {
                                let st2 = unit_stats(u2.kind);
                                if cheb(u2.tile, dst, s.size) <= st2.rng && st2.atk > 0.0 {
                                    ally_followup = true;
                                    break;
                                }
                            }
                        }
                        if on_enemy_city && (val_dealt >= val_taken * 0.65 || ally_followup) {
                            7_000 + (val_dealt - val_taken) as i32
                        } else if val_dealt > val_taken * 1.1 || (ally_followup && dmg >= du.health * 0.45) {
                            5_500 + (val_dealt - val_taken) as i32
                        } else {
                            -500
                        }
                    } else {
                        -2_000
                    }
                } else {
                    -1_000
                }
            }

            // Abilities (kind == 3)
            3 => {
                match a.param {
                    16 => {
                        let src = a.src as usize;
                        let su = &s.units[s.unit_at[src].unwrap()];
                        if su.health < su.max_hp() || su.attacked {
                            28_000
                        } else {
                            1_500
                        }
                    }
                    9 => {
                        let src = a.src as usize;
                        let mut healed = 0;
                        for u in &s.units {
                            if u.owner == s.player && cheb(src, u.tile, s.size) == 1 && (u.health < u.max_hp() || u.poison) {
                                healed += ((u.max_hp() - u.health).min(4.0) as i32) * 100 + if u.poison { 200 } else { 0 };
                            }
                        }
                        if healed > 0 { 4_500 + healed } else { -1_000 }
                    }
                    2 => {
                        let dst = a.dst as usize;
                        if !s.tech[16] && s.tiles[dst].resource == 0 { 200 } else { -2_000 }
                    }
                    19 => if a.dst == 0 { 100 } else { -10_000 },
                    _ => -5_000,
                }
            }

            // Move (kind == 1)
            1 => {
                let src = a.src as usize;
                let dst = a.dst as usize;
                if mem.turn_moves.contains(&(dst, src)) {
                    -10_000
                } else {
                    let su = &s.units[s.unit_at[src].unwrap()];
                    let sst = unit_stats(su.kind);
                    let td = &s.tiles[dst];

                    let standing_on_capturable = (s.tiles[src].structure == 1 && s.tiles[src].owner != s.player)
                        || s.tiles[src].structure == 2;
                    if standing_on_capturable {
                        -25_000
                    } else {
                        // Start at -50 so a pointless move does NOT beat End Turn (0)
                        let mut sc = -50;

                        // Stepping directly onto an enemy city or neutral village or ruin!
                        if td.structure == 1 && td.owner == s.enemy {
                            sc += 12_000 + (su.health as i32) * 50;
                        } else if td.structure == 1 && td.owner == 0 {
                            sc += 11_000;
                        } else if td.structure == 2 {
                            sc += 9_500;
                        } else if s.tech[14] && td.resource == 8 && td.structure == 0 {
                            sc += 9_000;
                        }

                        // Dash attack check
                        if sst.dash && !su.attacked {
                            let mut best_dash_bonus = 0;
                            for eu in &s.units {
                                if eu.owner == s.enemy {
                                    let d = cheb(dst, eu.tile, s.size);
                                    if d <= sst.rng {
                                        let est = unit_stats(eu.kind);
                                        let db = s.def_bonus(eu.tile, eu);
                                        let dd = est.def * if eu.poison { 0.7 } else { 1.0 };
                                        let (dmg, ret, _) = combat_calc(sst.atk, su.health, su.max_hp(), dd, eu.health, eu.max_hp(), db);
                                        let kill = eu.health - dmg <= 0.0;
                                        let on_our_city = besieged_cities.contains(&eu.tile);
                                        let on_enemy_city = s.city_at[eu.tile].map_or(false, |ci| s.cities[ci].owner == s.enemy);
                                        if kill {
                                            let b = 7_500 + est.cost * 250 + if on_our_city { 10_000 } else if on_enemy_city { 4_000 } else { 0 };
                                            best_dash_bonus = best_dash_bonus.max(b);
                                        } else if on_our_city {
                                            let b = 6_000 + (dmg as i32) * 150;
                                            best_dash_bonus = best_dash_bonus.max(b);
                                        } else if d > 1 || ret < dmg {
                                            let b = 2_500 + (dmg as i32) * 80 - (ret as i32) * 40;
                                            best_dash_bonus = best_dash_bonus.max(b);
                                        }
                                    }
                                }
                            }
                            if sst.infiltrate {
                                for c in &s.cities {
                                    if c.owner == s.enemy && cheb(dst, c.tile, s.size) <= 1 {
                                        best_dash_bonus = best_dash_bonus.max(12_000);
                                    }
                                }
                            }
                            sc += best_dash_bonus;
                        }

                        // If this unit is assigned to a specific neutral village/ruin, guide it there strongly!
                        if let Some(prize) = assigned_target[src] {
                            let d_prize = s.dijkstra_from(&[prize], sst.float);
                            let is_embark = td.structure == 8 && td.owner == s.player && !sst.float;
                            let d_dst = if is_embark { s.dijkstra_from(&[prize], true)[dst] } else { d_prize[dst] };
                            if d_prize[src] < INF / 2 && d_dst < INF / 2 {
                                sc += (d_prize[src] - d_dst) * 450;
                            }
                        } else {
                            // Otherwise guide toward enemy cities/units AND unexplored frontier!
                            let is_embark = td.structure == 8 && td.owner == s.player && !sst.float;
                            let d_atk_s = if sst.float { dist_atk_sea[src] } else { dist_atk_land[src] };
                            let d_atk_d = if sst.float || is_embark { dist_atk_sea[dst] } else { dist_atk_land[dst] };
                            let d_exp_s = if sst.float { dist_exp_sea[src] } else { dist_exp_land[src] };
                            let d_exp_d = if sst.float || is_embark { dist_exp_sea[dst] } else { dist_exp_land[dst] };

                            if d_atk_s < INF / 2 && d_atk_d < INF / 2 {
                                sc += (d_atk_s - d_atk_d) * 300;
                            }
                            if d_exp_s < INF / 2 && d_exp_d < INF / 2 {
                                let w = if s.turn < 14 || sst.scout || enemy_city_targets.is_empty() { 260 } else { 80 };
                                sc += (d_exp_s - d_exp_d) * w;
                            }
                            if is_embark {
                                if need_navy || dist_atk_sea[dst] + 2 < dist_atk_land[src] {
                                    sc += 2_500;
                                } else {
                                    sc -= 1_000;
                                }
                            }
                        }

                        let is_disembark = sst.carry && td.terrain >= 3 && td.structure != 8;
                        if is_disembark {
                            let near_prize = s.cities.iter().any(|c| c.owner != s.player && cheb(dst, c.tile, s.size) <= 2)
                                || (0..s.n).any(|t| s.tiles[t].known && s.tiles[t].structure == 2 && cheb(dst, t, s.size) <= 2);
                            if near_prize {
                                sc += 3_500;
                            } else if !land_reach[dst] {
                                sc += 1_800;
                            } else {
                                sc -= 2_000;
                            }
                        }

                        // Fog reveal bonus
                        let rev_rad = if td.terrain == 4 || sst.scout { 2 } else { 1 };
                        let mut fog_revealed = 0;
                        for t in 0..s.n {
                            if !s.tiles[t].known && cheb(dst, t, s.size) <= rev_rad {
                                fog_revealed += 1;
                                let tx = t % s.size;
                                let ty = t / s.size;
                                if (tx == 0 || tx == s.size - 1) && (ty == 0 || ty == s.size - 1) {
                                    fog_revealed += 5;
                                }
                            }
                        }
                        sc += fog_revealed * 110;

                        // Defensive terrain preference near enemies
                        let enemy_nearby = s.units.iter().any(|eu| eu.owner == s.enemy && cheb(dst, eu.tile, s.size) <= 2);
                        if enemy_nearby {
                            if (td.terrain == 5 && s.tech[18]) || (td.terrain == 4 && s.tech[20]) {
                                sc += 150;
                            }
                            if sst.rng >= 2 && sst.def <= 1.0 {
                                let adj_enemy = s.units.iter().any(|eu| eu.owner == s.enemy && cheb(dst, eu.tile, s.size) <= 1);
                                if adj_enemy {
                                    sc -= 1_500;
                                }
                            }
                        }

                        // Moving off our own city tile so we can recruit a unit on it
                        if s.city_at[src].map_or(false, |ci| s.cities[ci].owner == s.player) && s.stars >= 2 {
                            sc += 300;
                        }

                        sc
                    }
                }
            }

            // Harvest Resource (kind == 5, cost 2, +1 pop)
            5 => {
                let dst = a.dst as usize;
                if let Some(ci) = s.tile_city[dst] {
                    let c = &s.cities[ci];
                    let need = city_pop_need[c.tile];
                    if need == 1 {
                        if c.level == 2 {
                            9_800 // +5 stars back immediately + 1 star/turn!
                        } else if c.level >= 4 {
                            9_600 // Spawns a Giant!
                        } else {
                            8_800 // Levels L1 -> L2 (+2 stars/turn with Workshop!)
                        }
                    } else if city_can_level[c.tile] {
                        8_400 - need * 80
                    } else if s.stars >= 12 {
                        2_500 - need * 80
                    } else {
                        -100 // Save stars until we can complete the city level-up!
                    }
                } else {
                    100
                }
            }

            // Build Structure (kind == 6)
            6 => {
                let dst = a.dst as usize;
                let param = a.param;
                match param {
                    23..=29 => {
                        let res_penalty = if s.tiles[dst].resource > 0 { 300 } else { 0 };
                        let forest_penalty = if s.tiles[dst].terrain == 5 { 100 } else { 0 };
                        14_000 - res_penalty - forest_penalty
                    }
                    13 | 6 | 22 => {
                        let gain = s.build_pop_gain(dst, param);
                        if gain <= 0 {
                            -5_000
                        } else if let Some(ci) = s.tile_city[dst] {
                            let c = &s.cities[ci];
                            let need = city_pop_need[c.tile];
                            let levels_now = gain >= need;
                            if levels_now {
                                8_900 + gain * 300 + if c.level >= 4 { 1_000 } else { 0 }
                            } else if city_can_level[c.tile] && gain >= 2 {
                                8_300 + gain * 200
                            } else if gain >= 3 {
                                6_500 + gain * 200
                            } else if gain == 2 && s.stars >= 12 {
                                3_500
                            } else {
                                -500
                            }
                        } else {
                            -1_000
                        }
                    }
                    12 | 5 | 21 => {
                        let gain = s.build_pop_gain(dst, param);
                        let cost = struct_cost(param);
                        if let Some(ci) = s.tile_city[dst] {
                            let c = &s.cities[ci];
                            let need = city_pop_need[c.tile];
                            if gain >= need {
                                8_600 + gain * 300 - cost * 80 + if c.level >= 4 { 1_000 } else { 0 }
                            } else if city_can_level[c.tile] {
                                8_100 + gain * 200 - cost * 80 - need * 50
                            } else if s.stars >= 14 {
                                2_200 + gain * 200 - cost * 80 - need * 50
                            } else {
                                -100
                            }
                        } else {
                            -100
                        }
                    }
                    8 => {
                        let has_port_near = (0..s.n).any(|t| s.tiles[t].known && s.tiles[t].owner == s.player && s.tiles[t].structure == 8 && cheb(dst, t, s.size) <= 2);
                        let water_size = (0..s.n).filter(|&t| s.tiles[t].known && s.tiles[t].terrain <= 2 && cheb(dst, t, s.size) <= 3).count();
                        let conn = s.connects_city(dst, 8);
                        let mut sc = -500;
                        if need_navy && !has_port_near && water_size >= 4 {
                            sc = 7_800;
                        } else if conn > 0 {
                            sc = 7_500 + conn * 500;
                        } else if let Some(ci) = s.tile_city[dst] {
                            let c = &s.cities[ci];
                            if city_pop_need[c.tile] == 1 && water_size >= 4 && s.stars >= 12 {
                                sc = 6_000;
                            }
                        }
                        sc
                    }
                    50 => {
                        let stars_per_turn = s.market_stars(dst);
                        if stars_per_turn >= 2 {
                            7_600 + stars_per_turn * 400
                        } else if stars_per_turn == 1 && s.stars >= 18 {
                            2_500
                        } else {
                            -2_000
                        }
                    }
                    3 | 48 => {
                        let conn = s.connects_city(dst, param);
                        if conn > 0 {
                            8_200 + conn * 600
                        } else if param == 48 && need_navy {
                            1_500
                        } else {
                            -1_000
                        }
                    }
                    _ => -3_000,
                }
            }

            // Summon Unit / Upgrade Boat (kind == 4)
            4 => {
                let src = a.src as usize;
                let param = a.param;
                let is_boat_upgrade = s.unit_at[src].map_or(false, |ui| s.units[ui].kind == 43);
                if is_boat_upgrade {
                    let enemy_in_3 = s.units.iter().any(|eu| eu.owner == s.enemy && cheb(src, eu.tile, s.size) <= 3);
                    let enemy_in_2 = s.units.iter().any(|eu| eu.owner == s.enemy && cheb(src, eu.tile, s.size) <= 2);
                    match param {
                        44 => 7_900 + if enemy_in_3 { 1_500 } else { 0 },
                        42 => 7_500 + if enemy_in_2 { 1_200 } else { 300 },
                        41 => if s.stars >= 20 { 7_700 } else { 4_000 },
                        _ => 1_000,
                    }
                } else {
                    let enemy_near = s.units.iter().any(|eu| eu.owner == s.enemy && cheb(src, eu.tile, s.size) <= 2);
                    let d_atk = dist_atk_land[src].min(dist_atk_sea[src]);
                    let own_unit_count = s.units.iter().filter(|u| u.owner == s.player).count() as i32;
                    // If a city can level up right now and this city is not under immediate attack, level up the city first!
                    let base = if enemy_near {
                        9_100
                    } else if s.turn <= 2 && own_unit_count < 3 {
                        7_200 // Early scouting units on turns 0-2
                    } else if any_city_can_level {
                        3_200 // Let city level-ups (score 8_100+) happen before training more units!
                    } else if own_unit_count < num_own_cities + 2 {
                        6_200 - (d_atk.min(20) * 30)
                    } else {
                        4_500 - (d_atk.min(20) * 40)
                    };
                    let unit_pref = match param {
                        4 => 1_400,  // Knight
                        11 => 1_300, // Swordsman
                        8 => if enemy_near { 600 } else { 1_200 }, // Catapult
                        3 => if !enemy_near { 1_150 } else { 900 }, // Rider
                        9 => 950,    // Archer
                        5 => if enemy_near || need_navy { 1_050 } else { 750 }, // Defender
                        2 => 800,    // Warrior
                        38 => if !enemy_near { 850 } else { 200 }, // Cloak
                        10 => 400,   // MindBender
                        _ => 100,
                    };
                    base + unit_pref
                }
            }

            // Research Tech (kind == 7)
            7 => {
                let tech_id = a.param as usize;
                let forests = (0..s.n).filter(|&t| s.tiles[t].known && s.tiles[t].owner == s.player && s.tiles[t].terrain == 5 && s.tiles[t].structure == 0).count() as i32;
                let fruits = (0..s.n).filter(|&t| s.tiles[t].known && s.tiles[t].owner == s.player && s.tiles[t].resource == 6 && s.tiles[t].structure == 0).count() as i32;
                let fish = (0..s.n).filter(|&t| s.tiles[t].known && s.tiles[t].owner == s.player && s.tiles[t].resource == 3 && s.tiles[t].structure == 0).count() as i32;
                let metals = (0..s.n).filter(|&t| s.tiles[t].known && s.tiles[t].owner == s.player && s.tiles[t].resource == 5 && s.tiles[t].structure == 0).count() as i32;
                let shallow_owned = (0..s.n).filter(|&t| s.tiles[t].known && s.tiles[t].owner == s.player && s.tiles[t].terrain == 1 && s.tiles[t].structure == 0).count() as i32;
                let rafts = s.units.iter().filter(|u| u.owner == s.player && u.kind == 43).count() as i32;
                let ports = (0..s.n).filter(|&t| s.tiles[t].known && s.tiles[t].owner == s.player && s.tiles[t].structure == 8).count() as i32;
                let starfish = (0..s.n).filter(|&t| s.tiles[t].known && s.tiles[t].resource == 8 && s.tiles[t].structure == 0).count() as i32;
                let mountain_ruins = (0..s.n).filter(|&t| s.tiles[t].known && s.tiles[t].terrain == 4 && s.tiles[t].structure == 2).count() as i32;

                let cost = 4 + num_own_cities * tech_tier(tech_id);
                let stars_after = s.stars - cost;
                // Bonus if we are about to capture a neutral village right now (research before capture saves stars!)
                let about_to_capture_village = s.legal.iter().any(|act| act.kind == 8 && s.tiles[act.src as usize].structure == 1 && s.tiles[act.src as usize].owner == 0);
                let pre_cap_bonus = if about_to_capture_village { 15_000 } else { 0 };

                let base_tech = match tech_id {
                    16 => {
                        // Forestry (LumberHut)
                        if forests >= 2 && stars_after >= 3 {
                            7_900 + forests * 200
                        } else if forests >= 1 && stars_after >= 6 {
                            6_600
                        } else {
                            -500
                        }
                    }
                    1 => {
                        // Riding
                        if !need_navy && s.turn >= 1 && (stars_after >= 3 || about_to_capture_village) {
                            7_000
                        } else if stars_after >= 6 {
                            5_000
                        } else {
                            -500
                        }
                    }
                    6 => {
                        // Organization (Fruit)
                        if fruits >= 2 && stars_after >= 2 {
                            7_800 + fruits * 200
                        } else if fruits >= 1 && stars_after >= 4 {
                            6_400
                        } else {
                            -500
                        }
                    }
                    10 => {
                        // Fishing (Fish + Port)
                        if (need_navy && shallow_owned >= 1 && stars_after >= 7) || (fish >= 2 && stars_after >= 2) {
                            8_000 + fish * 200
                        } else if (fish >= 1 && stars_after >= 2) || need_navy {
                            6_800
                        } else {
                            -500
                        }
                    }
                    13 => {
                        // Sailing
                        if rafts >= 1 || ports >= 1 || need_navy {
                            7_900 + rafts * 500
                        } else {
                            -500
                        }
                    }
                    44 => {
                        // Ramming
                        if rafts >= 1 || ports >= 1 {
                            7_800 + rafts * 500
                        } else if need_navy && stars_after >= 5 {
                            6_500
                        } else {
                            -500
                        }
                    }
                    14 => {
                        // Navigation
                        if starfish >= 1 || ((rafts >= 1 || ports >= 1) && stars_after >= 8) {
                            7_600 + starfish * 800
                        } else {
                            -500
                        }
                    }
                    17 => {
                        // Mathematics (Sawmill + Catapult)
                        let huts = (0..s.n).filter(|&t| s.tiles[t].known && s.tiles[t].owner == s.player && s.tiles[t].structure == 12).count() as i32;
                        if (huts >= 2 && stars_after >= 5) || stars_after >= 8 {
                            7_700 + huts * 200
                        } else {
                            -500
                        }
                    }
                    18 => {
                        // Archery
                        if stars_after >= 3 { 5_800 } else { -500 }
                    }
                    4 => {
                        // Roads
                        if num_own_cities >= 2 && stars_after >= 3 {
                            6_900 + num_own_cities * 150
                        } else {
                            -500
                        }
                    }
                    20 => {
                        // Climbing
                        if mountain_ruins >= 1 || (metals >= 1 && stars_after >= 5) || stars_after >= 8 {
                            6_500 + mountain_ruins * 800 + metals * 400
                        } else {
                            -500
                        }
                    }
                    23 => {
                        // Mining
                        if metals >= 1 && stars_after >= 5 {
                            7_500 + metals * 400
                        } else if stars_after >= 10 {
                            5_500
                        } else {
                            -500
                        }
                    }
                    24 => {
                        // Smithery (Swordsman + Forge)
                        if stars_after >= 5 { 7_400 } else { -500 }
                    }
                    2 => {
                        // FreeSpirit -> Chivalry
                        if stars_after >= 8 { 6_000 } else { -500 }
                    }
                    3 => {
                        // Chivalry (Knights!)
                        if stars_after >= 8 { 7_800 } else { 5_000 }
                    }
                    7 => {
                        // Strategy (Defender)
                        if (need_navy || stars_after >= 6) && stars_after >= 3 { 5_600 } else { -500 }
                    }
                    5 => {
                        // Trade (Market!)
                        let sawmills = (0..s.n).filter(|&t| s.tiles[t].known && s.tiles[t].owner == s.player && (s.tiles[t].structure == 13 || s.tiles[t].structure == 22)).count() as i32;
                        if sawmills >= 1 && stars_after >= 5 {
                            7_300 + sawmills * 300
                        } else {
                            -500
                        }
                    }
                    _ => -500,
                };
                if base_tech > 0 { base_tech + pre_cap_bonus } else { base_tech }
            }

            _ => -1_000,
        };

        if score > best_score || (score == best_score && a.id < best_action.id) {
            best_score = score;
            best_action = a;
        }
    }

    if best_action.kind == 1 {
        mem.turn_moves.push((best_action.src as usize, best_action.dst as usize));
    }

    best_action.id.clone()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = BotMemory::new();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            _ => continue,
        };
        let obs: Value = serde_json::from_str(&line).expect("valid json");
        let action_id = choose_action(&obs, &mut mem);
        writeln!(stdout, "{}", json!({"action_id": action_id})).unwrap();
        stdout.flush().unwrap();
    }
}
