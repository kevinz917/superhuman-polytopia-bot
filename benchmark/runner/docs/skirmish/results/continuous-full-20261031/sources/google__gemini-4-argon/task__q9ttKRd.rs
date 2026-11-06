use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

const INF: i32 = 1_000_000;

#[derive(Clone, Copy, Default)]
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

#[derive(Clone, Copy, Default)]
struct Unit {
    tile: usize,
    owner: i32,
    kind: i32,
    health: f32,
    veteran: bool,
    moved: bool,
    attacked: bool,
    poison: bool,
    passenger: i32,
}

#[derive(Clone, Copy, Default)]
struct City {
    tile: usize,
    owner: i32,
    level: i32,
    progress: i32,
    connected: bool,
    rewards: [i32; 9],
}

#[derive(Clone)]
struct Cmd {
    action_id: String,
    kind: i32,
    source: i32,
    target: i32,
    parameter: i32,
}

struct ObsState {
    player: i32,
    enemy: i32,
    state_version: i32,
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
    legal: Vec<Cmd>,
}

struct Memory {
    last_version: i32,
    my_capital: Option<usize>,
    enemy_capital: Option<usize>,
    enemy_tech: [bool; 128],
    unit_prev_tile: Vec<usize>,
}

impl Memory {
    fn new() -> Self {
        Self {
            last_version: -1,
            my_capital: None,
            enemy_capital: None,
            enemy_tech: [false; 128],
            unit_prev_tile: Vec::new(),
        }
    }

    fn update(&mut self, s: &ObsState) {
        if s.state_version < self.last_version || (s.turn == 0 && s.state_version <= 2) {
            self.my_capital = None;
            self.enemy_capital = None;
            self.enemy_tech = [false; 128];
            self.enemy_tech[0] = true;
            self.enemy_tech[15] = true;
            self.unit_prev_tile = vec![usize::MAX; s.n];
        }
        if self.unit_prev_tile.len() != s.n {
            self.unit_prev_tile = vec![usize::MAX; s.n];
        }
        self.last_version = s.state_version;
        self.enemy_tech[0] = true;
        self.enemy_tech[15] = true;

        for i in 0..s.n {
            let t = &s.tiles[i];
            if !t.known {
                continue;
            }
            if t.capital == s.player && self.my_capital.is_none() {
                self.my_capital = Some(i);
            }
            if t.capital == s.enemy {
                self.enemy_capital = Some(i);
            }
            if t.owner == s.enemy {
                match t.structure {
                    3 | 48 => { self.enemy_tech[1] = true; self.enemy_tech[4] = true; }
                    5 => { self.enemy_tech[6] = true; self.enemy_tech[8] = true; }
                    6 => { self.enemy_tech[6] = true; self.enemy_tech[8] = true; self.enemy_tech[9] = true; }
                    8 => { self.enemy_tech[10] = true; }
                    12 => { self.enemy_tech[16] = true; }
                    13 => { self.enemy_tech[16] = true; self.enemy_tech[17] = true; }
                    19 => { self.enemy_tech[10] = true; self.enemy_tech[44] = true; self.enemy_tech[12] = true; }
                    20 => { self.enemy_tech[20] = true; self.enemy_tech[21] = true; }
                    21 => { self.enemy_tech[20] = true; self.enemy_tech[23] = true; }
                    22 => { self.enemy_tech[20] = true; self.enemy_tech[23] = true; self.enemy_tech[24] = true; }
                    50 => { self.enemy_tech[1] = true; self.enemy_tech[4] = true; self.enemy_tech[5] = true; }
                    _ => {}
                }
            }
            if let Some(u_idx) = s.unit_at[i] {
                let u = &s.units[u_idx];
                if u.owner == s.enemy {
                    if t.terrain == 4 {
                        self.enemy_tech[20] = true;
                    }
                    match u.kind {
                        3 => { self.enemy_tech[1] = true; }
                        4 => { self.enemy_tech[1] = true; self.enemy_tech[2] = true; self.enemy_tech[3] = true; }
                        5 => { self.enemy_tech[6] = true; self.enemy_tech[7] = true; }
                        8 => { self.enemy_tech[16] = true; self.enemy_tech[17] = true; }
                        9 => { self.enemy_tech[18] = true; }
                        10 => { self.enemy_tech[20] = true; self.enemy_tech[21] = true; self.enemy_tech[22] = true; }
                        11 => { self.enemy_tech[20] = true; self.enemy_tech[23] = true; self.enemy_tech[24] = true; }
                        38 | 39 => { self.enemy_tech[6] = true; self.enemy_tech[7] = true; self.enemy_tech[38] = true; }
                        41 => { self.enemy_tech[10] = true; self.enemy_tech[13] = true; self.enemy_tech[14] = true; }
                        42 => { self.enemy_tech[10] = true; self.enemy_tech[13] = true; }
                        43 => { self.enemy_tech[10] = true; }
                        44 => { self.enemy_tech[10] = true; self.enemy_tech[44] = true; }
                        _ => {}
                    }
                }
            }
        }
    }

    fn record_action(&mut self, s: &ObsState, action_id: &str) {
        if let Some(cmd) = s.legal.iter().find(|c| c.action_id == action_id) {
            if cmd.kind == 1 && cmd.source >= 0 && cmd.target >= 0 {
                let src = cmd.source as usize;
                let dst = cmd.target as usize;
                if src < s.n && dst < s.n {
                    self.unit_prev_tile[dst] = src;
                }
            }
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
    escape: bool,
    persist: bool,
    fortify: bool,
    stiff: bool,
    float: bool,
    carry: bool,
    splash: bool,
    stomp: bool,
    infiltrate: bool,
    scout: bool,
    surprise: bool,
}

fn unit_stats(kind: i32) -> UnitStats {
    let mut s = UnitStats {
        cost: 2, health: 10.0, attack: 2.0, defense: 2.0, movement: 1, range: 1,
        dash: false, escape: false, persist: false, fortify: false, stiff: false,
        float: false, carry: false, splash: false, stomp: false, infiltrate: false,
        scout: false, surprise: false,
    };
    match kind {
        2 => { s.cost = 2; s.health = 10.0; s.attack = 2.0; s.defense = 2.0; s.movement = 1; s.range = 1; s.dash = true; s.fortify = true; }
        3 => { s.cost = 3; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 2; s.range = 1; s.dash = true; s.escape = true; s.fortify = true; }
        4 => { s.cost = 8; s.health = 10.0; s.attack = 3.5; s.defense = 1.0; s.movement = 3; s.range = 1; s.dash = true; s.persist = true; s.fortify = true; }
        5 => { s.cost = 3; s.health = 15.0; s.attack = 1.0; s.defense = 3.0; s.movement = 1; s.range = 1; s.fortify = true; }
        8 => { s.cost = 8; s.health = 10.0; s.attack = 4.0; s.defense = 0.0; s.movement = 1; s.range = 3; s.stiff = true; }
        9 => { s.cost = 3; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 1; s.range = 2; s.dash = true; s.fortify = true; }
        10 => { s.cost = 5; s.health = 10.0; s.attack = 0.0; s.defense = 1.0; s.movement = 1; s.range = 1; s.stiff = true; }
        11 => { s.cost = 5; s.health = 15.0; s.attack = 3.0; s.defense = 3.0; s.movement = 1; s.range = 1; s.dash = true; }
        12 => { s.cost = 10; s.health = 40.0; s.attack = 5.0; s.defense = 4.0; s.movement = 1; s.range = 1; }
        37 => { s.cost = 2; s.health = 10.0; s.attack = 2.0; s.defense = 2.0; s.movement = 1; s.range = 1; s.dash = true; s.surprise = true; }
        38 => { s.cost = 8; s.health = 5.0; s.attack = 0.0; s.defense = 0.5; s.movement = 2; s.range = 1; s.dash = true; s.infiltrate = true; s.scout = true; s.stiff = true; }
        39 => { s.cost = 2; s.health = 5.0; s.attack = 0.0; s.defense = 0.5; s.movement = 2; s.range = 1; s.carry = true; s.dash = true; s.float = true; s.infiltrate = true; s.scout = true; s.stiff = true; }
        40 => { s.cost = 2; s.health = 10.0; s.attack = 2.0; s.defense = 2.0; s.movement = 2; s.range = 1; s.carry = true; s.dash = true; s.float = true; s.surprise = true; }
        41 => { s.cost = 15; s.health = 10.0; s.attack = 3.0; s.defense = 2.0; s.movement = 2; s.range = 3; s.carry = true; s.float = true; s.splash = true; s.stiff = true; }
        42 => { s.cost = 5; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 3; s.range = 2; s.carry = true; s.dash = true; s.float = true; s.scout = true; }
        43 => { s.cost = 0; s.health = 10.0; s.attack = 0.0; s.defense = 1.0; s.movement = 2; s.range = 2; s.carry = true; s.float = true; s.stiff = true; }
        44 => { s.cost = 5; s.health = 10.0; s.attack = 3.0; s.defense = 3.0; s.movement = 3; s.range = 1; s.carry = true; s.dash = true; s.float = true; }
        45 => { s.cost = 10; s.health = 40.0; s.attack = 4.0; s.defense = 4.0; s.movement = 2; s.range = 1; s.carry = true; s.float = true; s.stiff = true; s.stomp = true; }
        _ => {}
    }
    s
}

fn tech_tier(tech_id: usize) -> i32 {
    match tech_id {
        1 | 6 | 10 | 15 | 20 => 1,
        2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2,
        3 | 5 | 9 | 12 | 14 | 17 | 19 | 22 | 24 | 38 => 3,
        _ => 1,
    }
}

fn tech_cost(tech_id: usize, num_cities: i32, has_philosophy: bool) -> i32 {
    let base = 4 + num_cities * tech_tier(tech_id);
    if has_philosophy {
        ((base as f32) * 0.66666f32).ceil() as i32
    } else {
        base
    }
}

fn unit_max_hp(u: &Unit) -> f32 {
    let base_kind = if u.passenger > 0 { u.passenger } else { u.kind };
    unit_stats(base_kind).health + if u.veteran { 5.0 } else { 0.0 }
}

fn unit_star_value(u: &Unit) -> f32 {
    let st = unit_stats(u.kind);
    let pass_cost = if u.passenger > 0 { unit_stats(u.passenger).cost } else { 0 };
    let base = (st.cost + pass_cost) as f32;
    if u.kind == 12 || u.kind == 45 {
        18.0
    } else {
        base.max(2.0)
    }
}

#[inline]
fn cheb(a: usize, b: usize, size: usize) -> i32 {
    let ax = (a % size) as i32;
    let ay = (a / size) as i32;
    let bx = (b % size) as i32;
    let by = (b / size) as i32;
    (ax - bx).abs().max((ay - by).abs())
}

#[inline]
fn combat_formula(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    let divisor = if total > 0.0 { total } else { 1.0 };
    let attack = ((if total > 0.0 { (af / divisor) * aa * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    let defense = ((if dh > attack && total > 0.0 { (df / divisor) * dd * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    (attack, defense, (attack * 0.5).floor())
}

fn defense_bonus_at(s: &ObsState, mem: &Memory, tile: usize, u: &Unit) -> f32 {
    let t = &s.tiles[tile];
    let st = unit_stats(u.kind);
    let tech = if u.owner == s.player { &s.tech } else { &mem.enemy_tech };
    let mut city_bonus = 1.0;
    if let Some(c_idx) = s.city_at[tile] {
        let c = &s.cities[c_idx];
        if c.owner == u.owner && st.fortify {
            if c.owner == s.player {
                city_bonus = if c.rewards[1] > 0 { 4.0 } else { 1.5 };
            } else {
                city_bonus = 1.5;
            }
        }
    }
    let bonus = if t.terrain == 1 || t.terrain == 2 {
        if tech[12] { 1.5 } else { 1.0 }
    } else if t.terrain == 5 {
        if tech[18] { 1.5 } else { 1.0 }
    } else if t.terrain == 4 {
        if tech[20] { 1.5 } else { 1.0 }
    } else {
        city_bonus
    };
    bonus * if u.poison { 0.5 } else { 1.0 }
}

fn parse_obs(obs: &Value) -> ObsState {
    let player = obs["player"].as_i64().unwrap_or(1) as i32;
    let enemy = 3 - player;
    let state_version = obs["state_version"].as_i64().unwrap_or(0) as i32;
    let v = &obs["visible_state"];
    let size = v["board_size"].as_u64().unwrap_or(14) as usize;
    let n = size * size;
    let turn = v["turn"].as_i64().unwrap_or(0) as i32;
    let stars = v["stars"].as_i64().unwrap_or(0) as i32;

    let mut tech = [false; 128];
    if let Some(arr) = v["technology"].as_array() {
        for x in arr {
            let k = x.as_i64().unwrap_or(-1);
            if (0..128).contains(&k) {
                tech[k as usize] = true;
            }
        }
    }

    let mut tiles = vec![Tile::default(); n];
    if let Some(arr) = v["tiles"].as_array() {
        for item in arr {
            let idx = item["tile"].as_u64().unwrap_or(0) as usize;
            if idx >= n {
                continue;
            }
            let known = item["known"].as_bool().unwrap_or(false);
            if known {
                tiles[idx] = Tile {
                    known: true,
                    terrain: item["terrain"].as_i64().unwrap_or(0) as i32,
                    owner: item["owner"].as_i64().unwrap_or(0) as i32,
                    structure: item["structure"].as_i64().unwrap_or(0) as i32,
                    resource: item["resource"].as_i64().unwrap_or(0) as i32,
                    road: item["road"].as_i64().unwrap_or(0) != 0,
                    capital: item["capital"].as_i64().unwrap_or(0) as i32,
                    structure_level: item["structure_level"].as_i64().unwrap_or(0) as i32,
                };
            }
        }
    }

    let mut units = Vec::new();
    let mut unit_at = vec![None; n];
    if let Some(arr) = v["units"].as_array() {
        for item in arr {
            let tile = item["tile"].as_u64().unwrap_or(0) as usize;
            if tile >= n {
                continue;
            }
            let u = Unit {
                tile,
                owner: item["owner"].as_i64().unwrap_or(0) as i32,
                kind: item["kind"].as_i64().unwrap_or(0) as i32,
                health: item["health"].as_f64().unwrap_or(10.0) as f32,
                veteran: item["veteran"].as_bool().unwrap_or(false),
                moved: item["moved"].as_bool().unwrap_or(false),
                attacked: item["attacked"].as_bool().unwrap_or(false),
                poison: item["poison"].as_bool().unwrap_or(false),
                passenger: item["passenger"].as_i64().unwrap_or(0) as i32,
            };
            unit_at[tile] = Some(units.len());
            units.push(u);
        }
    }

    let mut cities = Vec::new();
    let mut city_at = vec![None; n];
    if let Some(arr) = v["cities"].as_array() {
        for item in arr {
            let tile = item["tile"].as_u64().unwrap_or(0) as usize;
            if tile >= n {
                continue;
            }
            let mut rewards = [0i32; 9];
            if let Some(rarr) = item["rewards"].as_array() {
                for (k, rv) in rarr.iter().enumerate().take(9) {
                    rewards[k] = rv.as_i64().unwrap_or(0) as i32;
                }
            }
            let c = City {
                tile,
                owner: item["owner"].as_i64().unwrap_or(0) as i32,
                level: item["level"].as_i64().unwrap_or(1) as i32,
                progress: item["progress"].as_i64().unwrap_or(0) as i32,
                connected: item["connected"].as_bool().unwrap_or(false),
                rewards,
            };
            city_at[tile] = Some(cities.len());
            cities.push(c);
        }
    }

    let mut legal = Vec::new();
    if let Some(arr) = obs["legal_actions"].as_array() {
        for item in arr {
            let c = &item["command"];
            legal.push(Cmd {
                action_id: item["action_id"].as_str().unwrap_or("").to_owned(),
                kind: c["kind"].as_i64().unwrap_or(0) as i32,
                source: c["source"].as_i64().unwrap_or(-1) as i32,
                target: c["target"].as_i64().unwrap_or(-1) as i32,
                parameter: c["parameter"].as_i64().unwrap_or(0) as i32,
            });
        }
    }

    ObsState {
        player,
        enemy,
        state_version,
        size,
        n,
        turn,
        stars,
        tech,
        tiles,
        units,
        unit_at,
        cities,
        city_at,
        legal,
    }
}

fn neighbors(tile: usize, size: usize) -> Vec<usize> {
    let x = (tile % size) as i32;
    let y = (tile / size) as i32;
    let mut out = Vec::with_capacity(8);
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 {
                out.push((ny as usize) * size + (nx as usize));
            }
        }
    }
    out
}

#[derive(Clone)]
struct CityPlan {
    city_tile: usize,
    level: i32,
    progress: i32,
    need: i32,
    can_level_now: bool,
    min_stars_to_level: i32,
    pop_available_now: i32,
    pop_potential_total: i32,
    has_port: bool,
}

fn analyze_cities(s: &ObsState) -> (Vec<CityPlan>, Vec<i32>) {
    let mut tile_city = vec![-1i32; s.n];
    for i in 0..s.n {
        if !s.tiles[i].known || s.tiles[i].owner != s.player {
            continue;
        }
        let mut best_c = -1i32;
        let mut best_d = 99;
        for c in &s.cities {
            if c.owner == s.player {
                let d = cheb(i, c.tile, s.size);
                if d <= 2 && d < best_d {
                    best_d = d;
                    best_c = c.tile as i32;
                }
            }
        }
        tile_city[i] = best_c;
    }

    let mut plans = Vec::new();
    for c in &s.cities {
        if c.owner != s.player {
            continue;
        }
        let need = (c.level + 1 - c.progress).max(1);
        let mut options: Vec<i32> = Vec::new();
        let mut pop_now = 0;
        let mut pop_total = 0;
        let mut has_port = false;

        for i in 0..s.n {
            if tile_city[i] != c.tile as i32 {
                continue;
            }
            let t = &s.tiles[i];
            if t.structure == 8 {
                has_port = true;
            }
            let blocked_by_enemy = match s.unit_at[i] {
                Some(u_idx) => s.units[u_idx].owner != s.player,
                None => false,
            };
            if blocked_by_enemy || t.structure != 0 {
                continue;
            }
            if (t.resource == 1 && s.tech[15]) || (t.resource == 3 && s.tech[10]) || (t.resource == 6 && s.tech[6]) {
                options.push(2);
                pop_now += 1;
            } else if t.terrain == 5 && s.tech[16] {
                options.push(3);
                pop_now += 1;
            } else if t.resource == 2 && s.tech[8] {
                options.push(2);
                options.push(3);
                pop_now += 2;
            } else if t.resource == 5 && s.tech[23] {
                options.push(2);
                options.push(3);
                pop_now += 2;
            } else if t.terrain == 1 && s.tech[10] {
                options.push(7);
                pop_now += 1;
            }

            if t.resource == 1 || t.resource == 3 || t.resource == 6 || t.terrain == 5 {
                pop_total += 1;
            } else if t.resource == 2 || t.resource == 5 {
                pop_total += 2;
            }
        }
        options.sort_unstable();
        let can_level_now = (options.len() as i32) >= need;
        let min_stars: i32 = if can_level_now {
            options.iter().take(need as usize).sum()
        } else {
            999
        };
        plans.push(CityPlan {
            city_tile: c.tile,
            level: c.level,
            progress: c.progress,
            need,
            can_level_now: can_level_now && min_stars <= s.stars,
            min_stars_to_level: min_stars,
            pop_available_now: pop_now,
            pop_potential_total: pop_total,
            has_port,
        });
    }
    (plans, tile_city)
}

fn compute_land_components(s: &ObsState) -> Vec<i32> {
    let mut comp = vec![-1i32; s.n];
    let mut cid = 0;
    let mut stack = Vec::new();
    for start in 0..s.n {
        if comp[start] >= 0 {
            continue;
        }
        let t = &s.tiles[start];
        let is_land = !t.known || t.terrain >= 3 || t.structure == 48 || t.structure == 8;
        if !is_land {
            continue;
        }
        comp[start] = cid;
        stack.push(start);
        while let Some(u) = stack.pop() {
            for v in neighbors(u, s.size) {
                if comp[v] < 0 {
                    let tv = &s.tiles[v];
                    let v_land = !tv.known || tv.terrain >= 3 || tv.structure == 48 || tv.structure == 8;
                    if v_land {
                        comp[v] = cid;
                        stack.push(v);
                    }
                }
            }
        }
        cid += 1;
    }
    comp
}

fn is_enterable(s: &ObsState, tile: usize, aquatic: bool, allow_water: bool) -> bool {
    let t = &s.tiles[tile];
    if !t.known {
        return true;
    }
    if t.terrain == 4 && !s.tech[20] {
        return false;
    }
    if t.terrain == 1 || t.terrain == 2 {
        if t.structure == 48 || (t.structure == 8 && t.owner == s.player) {
            return true;
        }
        if aquatic || allow_water {
            if t.terrain == 2 && !s.tech[13] && !allow_water {
                return false;
            }
            return true;
        }
        return false;
    }
    true
}

fn bfs_distances(s: &ObsState, sources: &[usize], aquatic: bool, allow_water_via_port: bool) -> Vec<i32> {
    let mut dist = vec![INF; s.n];
    let mut q = std::collections::VecDeque::new();
    for &src in sources {
        if src < s.n && is_enterable(s, src, aquatic, allow_water_via_port) {
            dist[src] = 0;
            q.push_back(src);
        }
    }
    while let Some(u) = q.pop_front() {
        let du = dist[u];
        for v in neighbors(u, s.size) {
            if !is_enterable(s, v, aquatic, allow_water_via_port) {
                continue;
            }
            let t = &s.tiles[v];
            let step_cost = if !t.known {
                2
            } else if t.terrain == 1 || t.terrain == 2 {
                if t.structure == 48 || (t.structure == 8 && t.owner == s.player) {
                    1
                } else {
                    2
                }
            } else if t.terrain == 4 {
                2
            } else if t.road {
                1
            } else if t.terrain == 5 {
                2
            } else {
                1
            };
            if du + step_cost < dist[v] {
                dist[v] = du + step_cost;
                q.push_back(v);
            }
        }
    }
    dist
}

fn choose_action(s: &ObsState, mem: &Memory) -> String {
    if s.legal.is_empty() {
        return String::new();
    }
    if s.legal.len() == 1 {
        return s.legal[0].action_id.clone();
    }

    let my_cities_count = s.cities.iter().filter(|c| c.owner == s.player).count() as i32;
    let enemy_cities: Vec<usize> = s.cities.iter().filter(|c| c.owner == s.enemy).map(|c| c.tile).collect();
    let neutral_villages: Vec<usize> = s.cities.iter().filter(|c| c.owner == 0).map(|c| c.tile).collect();
    let ruins: Vec<usize> = (0..s.n)
        .filter(|&i| s.tiles[i].known && s.tiles[i].structure == 2 && (s.tiles[i].terrain != 4 || s.tech[20]))
        .collect();
    let starfish: Vec<usize> = (0..s.n)
        .filter(|&i| s.tiles[i].known && s.tiles[i].resource == 8 && s.tiles[i].structure == 0)
        .collect();

    // 1. Check mandatory City Rewards (kind == 9)
    if s.legal.iter().any(|a| a.kind == 9) {
        let mut best_id = s.legal[0].action_id.clone();
        let mut best_score = -INF;
        let (plans, _) = analyze_cities(s);
        for a in &s.legal {
            if a.kind != 9 {
                continue;
            }
            let city_tile = a.target.max(0) as usize;
            let enemy_near = s.units.iter().any(|u| u.owner == s.enemy && cheb(u.tile, city_tile, s.size) <= 2);
            let plan = plans.iter().find(|p| p.city_tile == city_tile);
            let sc = match a.parameter {
                3 => 1000, // Workshop (+1 star/turn)
                4 => 200,  // Explorer
                7 => if enemy_near { 400 } else { 1000 }, // +5 stars
                1 => if enemy_near { 900 } else { 300 },  // City Wall
                8 => {
                    let cur_prog = plan.map(|p| p.progress).unwrap_or(0);
                    let pop_avail = plan.map(|p| p.pop_potential_total).unwrap_or(0);
                    if cur_prog + 3 >= 5 || cur_prog + 3 + pop_avail >= 5 {
                        1100
                    } else {
                        500
                    }
                }
                5 => 800,  // Border Growth
                6 => 2000, // Giant!
                2 => 100,  // Park
                _ => 0,
            };
            if sc > best_score {
                best_score = sc;
                best_id = a.action_id.clone();
            }
        }
        return best_id;
    }

    let (city_plans, tile_city) = analyze_cities(s);
    let land_comp = compute_land_components(s);

    let known_tiles = s.tiles.iter().filter(|t| t.known).count().max(1);
    let water_tiles = s.tiles.iter().filter(|t| t.known && (t.terrain == 1 || t.terrain == 2)).count();
    let ocean_tiles = s.tiles.iter().filter(|t| t.known && t.terrain == 2).count();
    let is_water_map = water_tiles * 100 / known_tiles >= 20 || ocean_tiles >= 6;

    let my_cap = mem.my_capital.unwrap_or_else(|| {
        s.cities.iter().find(|c| c.owner == s.player).map(|c| c.tile).unwrap_or(s.n / 2)
    });
    let guessed_enemy_cap = mem.enemy_capital.unwrap_or_else(|| {
        let cx = my_cap % s.size;
        let cy = my_cap / s.size;
        let gx = (s.size - 1 - cx).clamp(2, s.size - 3);
        let gy = (s.size - 1 - cy).clamp(2, s.size - 3);
        gy * s.size + gx
    });

    // Interior unexplored tiles first (villages only spawn on x,y in 1..size-1, and >= 3 from capital)
    let mut fog_targets = Vec::new();
    for i in 0..s.n {
        if !s.tiles[i].known {
            let x = i % s.size;
            let y = i / s.size;
            if x >= 1 && x + 1 < s.size && y >= 1 && y + 1 < s.size && cheb(i, my_cap, s.size) >= 3 {
                fog_targets.push(i);
            }
        }
    }
    if fog_targets.is_empty() {
        for i in 0..s.n {
            if !s.tiles[i].known {
                fog_targets.push(i);
            }
        }
    }

    // Assign each neutral village / ruin to the single closest friendly unit so our units fan out!
    let mut assigned_target_for_unit = vec![None::<usize>; s.n];
    let mut claimable: Vec<usize> = Vec::new();
    for &v in &neutral_villages {
        let occupied_by_me = s.unit_at[v].map(|u| s.units[u].owner == s.player).unwrap_or(false);
        if !occupied_by_me {
            claimable.push(v);
        }
    }
    for &r in &ruins {
        let occupied_by_me = s.unit_at[r].map(|u| s.units[u].owner == s.player).unwrap_or(false);
        if !occupied_by_me {
            claimable.push(r);
        }
    }

    let mut claimed_targets = vec![false; claimable.len()];
    for _ in 0..claimable.len() {
        let mut best_pair = None;
        let mut best_d = INF;
        for (ti, &t_tile) in claimable.iter().enumerate() {
            if claimed_targets[ti] {
                continue;
            }
            let dmap = bfs_distances(s, &[t_tile], false, false);
            for u in &s.units {
                if u.owner != s.player || assigned_target_for_unit[u.tile].is_some() {
                    continue;
                }
                let st = unit_stats(u.kind);
                if st.infiltrate {
                    continue;
                }
                let d = dmap[u.tile] * 2 / st.movement.max(1);
                if d < best_d {
                    best_d = d;
                    best_pair = Some((u.tile, ti));
                }
            }
        }
        if let Some((u_tile, ti)) = best_pair {
            if best_d < INF / 2 {
                claimed_targets[ti] = true;
                assigned_target_for_unit[u_tile] = Some(claimable[ti]);
            }
        } else {
            break;
        }
    }

    // Shared objectives: enemy cities + any unclaimed villages/ruins + enemy units on our side
    let mut shared_objectives: Vec<usize> = enemy_cities.clone();
    for (ti, &t_tile) in claimable.iter().enumerate() {
        if !claimed_targets[ti] {
            shared_objectives.push(t_tile);
        }
    }
    // Also add enemy units near our cities as high-priority targets
    for eu in &s.units {
        if eu.owner == s.enemy {
            let near_my_city = s.cities.iter().any(|c| c.owner == s.player && cheb(c.tile, eu.tile, s.size) <= 3);
            if near_my_city {
                shared_objectives.push(eu.tile);
            }
        }
    }

    let dist_to_shared_land = bfs_distances(s, &shared_objectives, false, false);
    let dist_to_shared_all = bfs_distances(s, &shared_objectives, false, true);
    let dist_to_fog_land = bfs_distances(s, &fog_targets, false, false);
    let dist_to_fog_naval = bfs_distances(s, &fog_targets, true, true);

    let own_ports: Vec<usize> = (0..s.n)
        .filter(|&i| s.tiles[i].known && s.tiles[i].structure == 8 && s.tiles[i].owner == s.player)
        .collect();
    let dist_to_port = bfs_distances(s, &own_ports, false, false);

    let my_comp = land_comp[my_cap];
    let need_naval_expansion = is_water_map && {
        let any_unclaimed_on_my_land = neutral_villages.iter().any(|&v| land_comp[v] == my_comp)
            || enemy_cities.iter().any(|&c| land_comp[c] == my_comp)
            || fog_targets.iter().any(|&f| land_comp[f] == my_comp && cheb(f, my_cap, s.size) <= 6);
        !any_unclaimed_on_my_land
    };

    let has_pending_village_capture = s.legal.iter().any(|a| {
        if a.kind != 8 || a.source < 0 {
            return false;
        }
        let st = s.tiles[a.source as usize].structure;
        st == 1 && !(s.tiles[a.source as usize].owner == s.enemy && enemy_cities.len() == 1)
    });

    let mut enemy_threat = vec![0.0f32; s.n];
    for u in &s.units {
        if u.owner != s.enemy {
            continue;
        }
        let st = unit_stats(u.kind);
        let reach = st.movement + st.range;
        for t in 0..s.n {
            if cheb(u.tile, t, s.size) <= reach {
                enemy_threat[t] += st.attack * (u.health / unit_max_hp(u)).max(0.3);
            }
        }
    }

    let mut my_unit_counts = [0i32; 64];
    let mut total_my_units = 0;
    for u in &s.units {
        if u.owner == s.player {
            total_my_units += 1;
            if (u.kind as usize) < 64 {
                my_unit_counts[u.kind as usize] += 1;
            }
        }
    }

    let mut best_action_idx = 0;
    let mut best_score = -1e9f64;

    for (idx, a) in s.legal.iter().enumerate() {
        let sc = score_cmd(
            a,
            s,
            mem,
            my_cities_count,
            &enemy_cities,
            &starfish,
            &city_plans,
            &tile_city,
            &land_comp,
            my_comp,
            is_water_map,
            need_naval_expansion,
            has_pending_village_capture,
            &assigned_target_for_unit,
            &dist_to_shared_land,
            &dist_to_shared_all,
            &dist_to_fog_land,
            &dist_to_fog_naval,
            &dist_to_port,
            &enemy_threat,
            &my_unit_counts,
            total_my_units,
            my_cap,
            guessed_enemy_cap,
        );
        if sc > best_score {
            best_score = sc;
            best_action_idx = idx;
        }
    }

    s.legal[best_action_idx].action_id.clone()
}

#[allow(clippy::too_many_arguments)]
fn score_cmd(
    a: &Cmd,
    s: &ObsState,
    mem: &Memory,
    my_cities_count: i32,
    enemy_cities: &[usize],
    starfish: &[usize],
    city_plans: &[CityPlan],
    tile_city: &[i32],
    land_comp: &[i32],
    my_comp: i32,
    is_water_map: bool,
    need_naval_expansion: bool,
    has_pending_village_capture: bool,
    assigned_target_for_unit: &[Option<usize>],
    dist_to_shared_land: &[i32],
    dist_to_shared_all: &[i32],
    dist_to_fog_land: &[i32],
    dist_to_fog_naval: &[i32],
    dist_to_port: &[i32],
    enemy_threat: &[f32],
    my_unit_counts: &[i32; 64],
    total_my_units: i32,
    my_cap: usize,
    guessed_enemy_cap: usize,
) -> f64 {
    match a.kind {
        11 => -1e9,
        10 => 0.0,

        // Capture
        8 => {
            let src = a.source.max(0) as usize;
            let t = &s.tiles[src];
            if t.structure == 2 {
                return 9500.0;
            }
            if t.resource == 8 && t.structure == 0 {
                return 9400.0;
            }
            if t.structure == 1 && t.owner == s.enemy && enemy_cities.len() == 1 {
                return 100_000.0;
            }
            if t.owner == s.enemy {
                5100.0
            } else {
                5000.0
            }
        }

        // Special actions
        3 => {
            match a.parameter {
                16 => 9800.0, // Promote
                9 => {
                    let src = a.source.max(0) as usize;
                    let mut healed = 0.0f32;
                    for nb in neighbors(src, s.size) {
                        if let Some(u_idx) = s.unit_at[nb] {
                            let u = &s.units[u_idx];
                            if u.owner == s.player {
                                let missing = (unit_max_hp(u) - u.health).max(0.0);
                                healed += missing.min(4.0);
                            }
                        }
                    }
                    if healed >= 3.0 {
                        2200.0 + (healed as f64) * 50.0
                    } else {
                        -100.0
                    }
                }
                7 => {
                    let src = a.source.max(0) as usize;
                    let Some(u_idx) = s.unit_at[src] else { return -1000.0 };
                    let u = &s.units[u_idx];
                    let mhp = unit_max_hp(u);
                    let missing = mhp - u.health;
                    let friendly = s.tiles[src].owner == s.player;
                    let heal = if friendly { 4.0f32 } else { 2.0f32 }.min(missing);
                    let on_my_city = s.city_at[src].map(|ci| s.cities[ci].owner == s.player).unwrap_or(false);
                    let under_threat = enemy_threat[src] > 0.0;
                    if on_my_city && under_threat && u.health <= mhp * 0.65 {
                        800.0 + (heal as f64) * 40.0
                    } else if u.health <= mhp * 0.45 && enemy_threat[src] <= 3.0 && dist_to_shared_land[src] > 2 {
                        350.0 + (heal as f64) * 30.0
                    } else {
                        -50.0
                    }
                }
                2 => {
                    // Clear Forest (+1 star)
                    let dst = a.target.max(0) as usize;
                    let t = &s.tiles[dst];
                    if t.resource == 1 && s.tech[15] {
                        return -500.0;
                    }
                    let cid = tile_city[dst];
                    if let Some(plan) = city_plans.iter().find(|p| p.city_tile as i32 == cid) {
                        if plan.pop_available_now <= plan.need {
                            return -200.0;
                        }
                    }
                    250.0
                }
                1 => {
                    let dst = a.target.max(0) as usize;
                    if s.tech[8] && s.stars >= 16 {
                        let adj_windmill = neighbors(dst, s.size)
                            .iter()
                            .any(|&nb| s.tiles[nb].owner == s.player && s.tiles[nb].structure == 6);
                        if adj_windmill { 120.0 } else { -200.0 }
                    } else {
                        -500.0
                    }
                }
                19 => if a.target == 0 { 100.0 } else { -5000.0 },
                20 => if s.stars >= 15 && s.turn < 120 { 180.0 } else { -200.0 },
                _ => -500.0,
            }
        }

        // Attack
        2 => {
            let src = a.source.max(0) as usize;
            let dst = a.target.max(0) as usize;
            let Some(au_idx) = s.unit_at[src] else { return -1000.0 };
            let au = &s.units[au_idx];
            let ast = unit_stats(au.kind);

            if ast.infiltrate {
                return 8800.0;
            }
            if au.kind == 10 {
                if let Some(du_idx) = s.unit_at[dst] {
                    let du = &s.units[du_idx];
                    return 8500.0 + (unit_star_value(du) as f64) * 200.0 + (du.health as f64) * 20.0;
                }
                return 8000.0;
            }

            let Some(du_idx) = s.unit_at[dst] else { return -1000.0 };
            let du = &s.units[du_idx];
            let dst_st = unit_stats(du.kind);
            let am = unit_max_hp(au);
            let dm = unit_max_hp(du);
            let db = defense_bonus_at(s, mem, dst, du);
            let dd = dst_st.defense * if du.poison { 0.7 } else { 1.0 };
            let (dmg, retal, _) = combat_formula(ast.attack, au.health, am, dd, du.health, dm, db);
            let dist = cheb(src, dst, s.size);
            let dead = du.health - dmg <= 0.0;
            let retaliates = !dead && !ast.surprise && !dst_st.stiff && dist <= dst_st.range;
            let actual_retal = if retaliates { retal } else { 0.0 };
            let self_dead = au.health - actual_retal <= 0.0;

            let enemy_val = unit_star_value(du);
            let my_val = unit_star_value(au);

            let mut splash_score = 0.0f64;
            if ast.splash {
                for nb in neighbors(dst, s.size) {
                    if let Some(nb_u_idx) = s.unit_at[nb] {
                        let nb_u = &s.units[nb_u_idx];
                        if nb_u.owner == s.enemy {
                            let nb_st = unit_stats(nb_u.kind);
                            let nb_dd = nb_st.defense * if nb_u.poison { 0.7 } else { 1.0 };
                            let nb_db = defense_bonus_at(s, mem, nb, nb_u);
                            let (_, _, sp) = combat_formula(ast.attack, au.health, am, nb_dd, nb_u.health, unit_max_hp(nb_u), nb_db);
                            let nb_val = unit_star_value(nb_u);
                            if nb_u.health - sp <= 0.0 {
                                splash_score += 800.0 + (nb_val as f64) * 120.0;
                            } else {
                                splash_score += (sp as f64) * 35.0;
                            }
                        }
                    }
                }
            }

            let target_on_my_city = s.city_at[dst].map(|ci| s.cities[ci].owner == s.player).unwrap_or(false);
            let target_on_enemy_city = s.city_at[dst].map(|ci| s.cities[ci].owner == s.enemy).unwrap_or(false);
            let target_on_village = s.city_at[dst].map(|ci| s.cities[ci].owner == 0).unwrap_or(false);

            let mut score = 3000.0 + splash_score;

            if actual_retal == 0.0 && !dead {
                score += 1400.0;
            }

            if dead {
                score += 1200.0 + (enemy_val as f64) * 150.0;
                if target_on_my_city {
                    score += 2000.0;
                }
                if target_on_enemy_city {
                    if ast.range < 2 {
                        score += 2200.0 + (au.health as f64) * 30.0;
                    } else {
                        score += 600.0;
                    }
                } else if target_on_village && ast.range < 2 {
                    score += 900.0;
                }
                if ast.persist {
                    score += 1800.0;
                }
            } else {
                let dmg_ratio = (dmg / dm).min(1.0);
                let retal_ratio = (actual_retal / am).min(1.0);
                let value_gained = dmg_ratio * enemy_val;
                let value_lost = retal_ratio * my_val + if self_dead { my_val } else { 0.0 };

                if self_dead && !target_on_my_city {
                    return -400.0;
                }
                if actual_retal > 0.0 && value_lost > value_gained * 1.35 && !target_on_my_city {
                    return -150.0 + (value_gained - value_lost) as f64 * 50.0;
                }

                score += (value_gained * 120.0 - value_lost * 90.0) as f64;
                if target_on_my_city {
                    score += 1200.0 + (dmg as f64) * 60.0;
                }
            }

            if ast.escape && !self_dead {
                score += 250.0;
            }

            score
        }

        // Summon / Train unit or Upgrade Raft
        4 => {
            let src = a.source.max(0) as usize;
            let kind = a.parameter;
            let st = unit_stats(kind);

            if kind == 41 || kind == 42 || kind == 44 {
                let Some(u_idx) = s.unit_at[src] else { return -500.0 };
                let u = &s.units[u_idx];
                let can_act_now = !u.moved && !u.attacked;
                let enemy_nearby = s.units.iter().any(|eu| eu.owner == s.enemy && cheb(src, eu.tile, s.size) <= 4);
                let timing_bonus = if can_act_now { 4500.0 } else if enemy_nearby { 900.0 } else { -50.0 };
                let kind_score = match kind {
                    44 => 350.0 + if u.health >= 15.0 { 250.0 } else { 0.0 },
                    42 => 320.0 + if my_unit_counts[42] == 0 { 200.0 } else { 0.0 },
                    41 => 450.0,
                    _ => 0.0,
                };
                return timing_bonus + kind_score;
            }

            if s.turn == 0 {
                if let Some(plan) = city_plans.first() {
                    if plan.level == 1 && plan.can_level_now {
                        return -500.0;
                    }
                }
            }

            let enemy_near_city = s.units.iter().any(|eu| eu.owner == s.enemy && cheb(src, eu.tile, s.size) <= 2);
            let any_city_can_level_to_giant = city_plans.iter().any(|p| p.level >= 4 && p.can_level_now && p.min_stars_to_level <= s.stars);
            if any_city_can_level_to_giant && !enemy_near_city && s.stars - st.cost < 5 {
                return 100.0;
            }

            // Early game expansion boost: ensure we have enough units exploring & claiming villages!
            let early_army_boost = if total_my_units < (my_cities_count + 1).min(4) {
                700.0
            } else if total_my_units <= my_cities_count + 2 {
                300.0
            } else {
                0.0
            };

            let mut base_score = (if enemy_near_city { 2400.0 } else { 950.0 }) + early_army_boost;

            let d_enemy = dist_to_shared_all[src];
            let city_has_port = city_plans.iter().find(|p| p.city_tile == src).map(|p| p.has_port).unwrap_or(false);

            let unit_pref = match kind {
                4 => 550.0,
                11 => 480.0,
                8 => {
                    if enemy_near_city && my_unit_counts[5] + my_unit_counts[11] + my_unit_counts[2] == 0 {
                        100.0
                    } else if my_unit_counts[8] * 4 <= total_my_units + 1 {
                        460.0
                    } else {
                        220.0
                    }
                }
                5 => {
                    if city_has_port || enemy_near_city {
                        430.0
                    } else if my_unit_counts[5] * 4 <= total_my_units {
                        310.0
                    } else {
                        180.0
                    }
                }
                3 => if !need_naval_expansion { 440.0 } else { 300.0 },
                9 => if my_unit_counts[9] * 3 <= total_my_units + 1 { 390.0 } else { 260.0 },
                38 => if !enemy_cities.is_empty() && my_unit_counts[38] == 0 && d_enemy <= 10 { 440.0 } else { 120.0 },
                10 => if my_unit_counts[10] == 0 && total_my_units >= 5 { 240.0 } else { 80.0 },
                2 => 300.0,
                _ => 200.0,
            };

            let front_bonus = (20 - d_enemy.min(20)) as f64 * 12.0 + if city_has_port && is_water_map { 80.0 } else { 0.0 };
            base_score += unit_pref + front_bonus;
            base_score
        }

        // Move unit
        1 => {
            let src = a.source.max(0) as usize;
            let dst = a.target.max(0) as usize;
            let Some(u_idx) = s.unit_at[src] else { return -1000.0 };
            let u = &s.units[u_idx];
            let st = unit_stats(u.kind);
            let dst_t = &s.tiles[dst];

            let src_t = &s.tiles[src];
            if !u.attacked {
                if src_t.structure == 2 || (src_t.resource == 8 && src_t.structure == 0 && s.tech[14]) {
                    return -5000.0;
                }
                if src_t.structure == 1 && src_t.owner != s.player {
                    return -5000.0;
                }
            }

            let mut score = 1500.0;

            // Bonus for moving OFF our own city so we can train a unit there or spawn a Giant!
            if let Some(ci) = s.city_at[src] {
                let c = &s.cities[ci];
                if c.owner == s.player {
                    score += 380.0;
                }
            }

            // Penalty for stepping ONTO our own unthreatened city (blocks recruitment!)
            if let Some(ci) = s.city_at[dst] {
                let c = &s.cities[ci];
                if c.owner == s.player {
                    let city_threatened = s.units.iter().any(|eu| eu.owner == s.enemy && cheb(eu.tile, dst, s.size) <= 2);
                    if !city_threatened {
                        score -= 700.0;
                    }
                }
            }

            // Anti-oscillation penalty: avoid stepping immediately back onto `unit_prev_tile[src]` unless escaping threat
            if mem.unit_prev_tile[src] == dst && !(u.attacked && st.escape) {
                score -= 350.0;
            }

            // 1. Stepping directly onto an empty Enemy City or Neutral Village or Ruin or Starfish!
            if !st.infiltrate {
                if dst_t.structure == 1 && dst_t.owner == s.enemy {
                    score += 4200.0 + (u.health as f64) * 30.0;
                } else if dst_t.structure == 1 && dst_t.owner == 0 {
                    score += 3800.0;
                }
            }
            if dst_t.structure == 2 {
                score += 3600.0;
            }
            if dst_t.resource == 8 && dst_t.structure == 0 && s.tech[14] {
                score += 3200.0;
            }

            // 2. Dash-Attack Setup
            let embarking = dst_t.structure == 8 && !st.float;
            let disembarking = st.carry && dst_t.terrain >= 3 && !embarking;
            if !u.attacked && st.dash && !embarking && !disembarking {
                if st.infiltrate {
                    for &ec in enemy_cities {
                        if cheb(dst, ec, s.size) <= 1 {
                            score += 4000.0;
                        }
                    }
                } else {
                    let mut best_dash_atk = 0.0f64;
                    for eu in &s.units {
                        if eu.owner != s.enemy {
                            continue;
                        }
                        let d = cheb(dst, eu.tile, s.size);
                        if d >= 1 && d <= st.range {
                            let est = unit_stats(eu.kind);
                            let db = defense_bonus_at(s, mem, eu.tile, eu);
                            let dd = est.defense * if eu.poison { 0.7 } else { 1.0 };
                            let (dmg, retal, _) = combat_formula(st.attack, u.health, unit_max_hp(u), dd, eu.health, unit_max_hp(eu), db);
                            let dead = eu.health - dmg <= 0.0;
                            let retaliates = !dead && !st.surprise && !est.stiff && d <= est.range;
                            let actual_retal = if retaliates { retal } else { 0.0 };
                            let self_dead = u.health - actual_retal <= 0.0;
                            let eu_on_my_city = s.city_at[eu.tile].map(|ci| s.cities[ci].owner == s.player).unwrap_or(false);
                            let eu_on_enemy_city = s.city_at[eu.tile].map(|ci| s.cities[ci].owner == s.enemy).unwrap_or(false);
                            if dead {
                                let mut atk_val = 2200.0 + (unit_star_value(eu) as f64) * 120.0;
                                if eu_on_my_city || eu_on_enemy_city {
                                    atk_val += 1200.0;
                                }
                                if st.persist {
                                    atk_val += 1000.0;
                                }
                                best_dash_atk = best_dash_atk.max(atk_val);
                            } else if !self_dead {
                                let gain = (dmg / unit_max_hp(eu)) * unit_star_value(eu);
                                let loss = (actual_retal / unit_max_hp(u)) * unit_star_value(u);
                                if gain > loss * 0.9 || actual_retal == 0.0 || eu_on_my_city {
                                    let atk_val = 900.0 + ((gain - loss * 0.7) as f64) * 80.0 + if actual_retal == 0.0 { 400.0 } else { 0.0 };
                                    best_dash_atk = best_dash_atk.max(atk_val);
                                }
                            }
                        }
                    }
                    score += best_dash_atk;
                }
            }

            // 3. Juggernaut Stomp damage on move!
            if st.stomp {
                for nb in neighbors(dst, s.size) {
                    if let Some(eu_idx) = s.unit_at[nb] {
                        let eu = &s.units[eu_idx];
                        if eu.owner == s.enemy {
                            score += 900.0;
                        }
                    }
                }
            }

            // 4. Escape unit (Rider) retreating after attacking
            if u.attacked && st.escape {
                let threat_before = enemy_threat[src];
                let threat_after = enemy_threat[dst];
                score += ((threat_before - threat_after) as f64) * 200.0;
            }

            // 5. Embarking onto our Port
            if embarking {
                if u.kind == 12 {
                    if is_water_map {
                        score += 2400.0;
                    }
                } else if need_naval_expansion || (assigned_target_for_unit[src].is_none() && dist_to_shared_land[src] >= INF / 2) {
                    score += 1800.0 + (u.health as f64) * 25.0;
                } else if is_water_map && my_unit_counts[41] + my_unit_counts[42] + my_unit_counts[44] + my_unit_counts[43] < 3 {
                    score += 900.0;
                } else {
                    score -= 400.0;
                }
            }

            // 6. Disembarking from a Boat onto Land
            if disembarking {
                let on_foreign_land = land_comp[dst] != my_comp;
                let dst_has_capture = (dst_t.structure == 1 && dst_t.owner != s.player) || dst_t.structure == 2;
                if dst_has_capture {
                    score += 500.0;
                } else if u.kind == 43 && (on_foreign_land || dist_to_shared_land[dst] < 10) {
                    score += 700.0 - (dist_to_shared_land[dst].min(20) as f64) * 30.0;
                } else if u.kind == 41 || u.kind == 42 || u.kind == 44 || u.kind == 45 {
                    score -= 1800.0;
                } else {
                    score -= 600.0;
                }
            }

            // 7. Exploration: value interior tiles much more than outer edge/corner tiles in the early game!
            let sight = if dst_t.terrain == 4 || st.scout { 2 } else { 1 };
            let mut interior_revealed = 0;
            let mut edge_revealed = 0;
            let dx_center = (dst % s.size) as i32;
            let dy_center = (dst / s.size) as i32;
            for ry in (dy_center - sight).max(0)..=(dy_center + sight).min(s.size as i32 - 1) {
                for rx in (dx_center - sight).max(0)..=(dx_center + sight).min(s.size as i32 - 1) {
                    let rt = (ry as usize) * s.size + (rx as usize);
                    if !s.tiles[rt].known {
                        if rx >= 1 && rx + 1 < s.size as i32 && ry >= 1 && ry + 1 < s.size as i32 {
                            interior_revealed += 1;
                        } else {
                            edge_revealed += 1;
                        }
                    }
                }
            }
            score += (interior_revealed as f64) * 65.0 + (edge_revealed as f64) * 15.0;

            // 8. Distance progress toward assigned target, shared objectives, ports, or fog!
            if st.float {
                let d_src = dist_to_shared_all[src].min(dist_to_fog_naval[src]);
                let d_dst = dist_to_shared_all[dst].min(dist_to_fog_naval[dst]);
                score += ((d_src - d_dst).clamp(-10, 10) as f64) * 90.0;
                let c_src = cheb(src, guessed_enemy_cap, s.size);
                let c_dst = cheb(dst, guessed_enemy_cap, s.size);
                score += ((c_src - c_dst) as f64) * 35.0;
            } else if let Some(target_tile) = assigned_target_for_unit[src] {
                let dmap = bfs_distances(s, &[target_tile], false, false);
                let d_src = dmap[src].min(40);
                let d_dst = dmap[dst].min(40);
                score += ((d_src - d_dst) as f64) * 130.0;
            } else {
                let d_src = dist_to_shared_land[src];
                let d_dst = dist_to_shared_land[dst];
                if d_src < INF / 2 || d_dst < INF / 2 {
                    score += ((d_src - d_dst).clamp(-10, 10) as f64) * 95.0;
                } else if need_naval_expansion && dist_to_port[dst] < INF / 2 {
                    let p_src = dist_to_port[src].min(30);
                    let p_dst = dist_to_port[dst].min(30);
                    score += ((p_src - p_dst) as f64) * 110.0;
                } else {
                    let f_src = dist_to_fog_land[src].min(30);
                    let f_dst = dist_to_fog_land[dst].min(30);
                    score += ((f_src - f_dst) as f64) * 85.0;
                    // Push away from our capital and toward enemy capital!
                    let c_src = cheb(src, guessed_enemy_cap, s.size);
                    let c_dst = cheb(dst, guessed_enemy_cap, s.size);
                    let away_src = cheb(src, my_cap, s.size);
                    let away_dst = cheb(dst, my_cap, s.size);
                    score += ((c_src - c_dst) as f64) * 40.0 + ((away_dst - away_src) as f64) * 25.0;
                }
            }

            if dst_t.terrain == 4 && s.tech[20] || dst_t.terrain == 5 && s.tech[18] {
                score += 25.0;
            }
            if st.range >= 2 && enemy_threat[dst] > 3.0 {
                score -= (enemy_threat[dst] as f64) * (if u.kind == 8 { 90.0 } else { 30.0 });
            }

            score
        }

        // Harvest resource
        5 => {
            let dst = a.target.max(0) as usize;
            let cid = tile_city[dst];
            let Some(plan) = city_plans.iter().find(|p| p.city_tile as i32 == cid) else {
                return 600.0;
            };
            if plan.level >= 4 && plan.need == 1 {
                if let Some(u_idx) = s.unit_at[plan.city_tile] {
                    let u = &s.units[u_idx];
                    if u.owner == s.player && !u.moved {
                        return 800.0;
                    }
                }
            }
            let mut score = 1200.0;
            if plan.can_level_now {
                score += 1800.0 - (plan.need as f64) * 100.0;
                if plan.level == 1 {
                    score += 500.0;
                } else if plan.level == 2 {
                    score += 600.0;
                } else if plan.level >= 4 {
                    score += 900.0;
                }
            } else {
                score += (5 - plan.need.min(5)) as f64 * 60.0;
            }
            score
        }

        // Build structure
        6 => {
            let dst = a.target.max(0) as usize;
            let kind = a.parameter;
            let cid = tile_city[dst];
            let plan = city_plans.iter().find(|p| p.city_tile as i32 == cid);

            if let Some(p) = plan {
                if p.level >= 4 && p.need <= 3 {
                    if let Some(u_idx) = s.unit_at[p.city_tile] {
                        let u = &s.units[u_idx];
                        if u.owner == s.player && !u.moved {
                            return 750.0;
                        }
                    }
                }
            }

            match kind {
                23..=29 => {
                    let t = &s.tiles[dst];
                    let mut score = 3100.0;
                    if let Some(p) = plan {
                        if p.progress + 3 >= p.level + 1 {
                            score += 600.0;
                        }
                    }
                    if t.resource > 0 {
                        score -= 400.0;
                    }
                    if t.terrain == 1 && plan.map(|p| !p.has_port).unwrap_or(false) {
                        score -= 600.0;
                    }
                    score
                }
                13 | 6 | 22 => {
                    let target_adj = if kind == 13 { 12 } else if kind == 6 { 5 } else { 21 };
                    let pop_per = if kind == 22 { 2 } else { 1 };
                    let mut adj_count = 0;
                    for nb in neighbors(dst, s.size) {
                        if s.tiles[nb].owner == s.player && s.tiles[nb].structure == target_adj {
                            adj_count += 1;
                        }
                    }
                    let pop_gain = adj_count * pop_per;
                    if pop_gain <= 0 {
                        return -500.0;
                    }
                    let mut score = 1100.0 + (pop_gain as f64) * 280.0;
                    if let Some(p) = plan {
                        if p.need <= pop_gain {
                            score += 1500.0;
                        } else if p.can_level_now {
                            score += 700.0;
                        }
                    }
                    score
                }
                12 => {
                    let adj_sawmill = neighbors(dst, s.size)
                        .iter()
                        .any(|&nb| s.tiles[nb].owner == s.player && s.tiles[nb].structure == 13);
                    let pop_gain = if adj_sawmill { 2 } else { 1 };
                    let mut score = 1150.0 + if adj_sawmill { 500.0 } else { 0.0 };
                    if let Some(p) = plan {
                        if p.need <= pop_gain {
                            score += 1600.0;
                        } else if p.can_level_now {
                            score += 1100.0;
                        }
                    }
                    score
                }
                5 | 21 => {
                    let mult_struct = if kind == 5 { 6 } else { 22 };
                    let adj_mult = neighbors(dst, s.size)
                        .iter()
                        .any(|&nb| s.tiles[nb].owner == s.player && s.tiles[nb].structure == mult_struct);
                    let pop_gain = 2 + if adj_mult { if kind == 21 { 2 } else { 1 } } else { 0 };
                    let mut score = 1200.0 + (pop_gain as f64) * 180.0;
                    if let Some(p) = plan {
                        if p.need <= pop_gain {
                            score += 1600.0;
                        } else if p.can_level_now {
                            score += 1100.0;
                        }
                    }
                    score
                }
                8 => {
                    let has_any_port = city_plans.iter().any(|p| p.has_port);
                    let city_has_port = plan.map(|p| p.has_port).unwrap_or(false);
                    let mut score = 850.0;
                    if is_water_map && !city_has_port {
                        score += if !has_any_port { 1900.0 } else { 1100.0 };
                    }
                    if need_naval_expansion && !has_any_port {
                        score += 1200.0;
                    }
                    if let Some(p) = plan {
                        if p.need == 1 {
                            score += 1000.0;
                        }
                    }
                    let water_nb = neighbors(dst, s.size)
                        .iter()
                        .filter(|&&nb| s.tiles[nb].terrain == 1 || s.tiles[nb].terrain == 2 || !s.tiles[nb].known)
                        .count();
                    score += (water_nb as f64) * 40.0;
                    score
                }
                50 => {
                    let mut stars_per_turn = 0;
                    for nb in neighbors(dst, s.size) {
                        let st = s.tiles[nb].structure;
                        if s.tiles[nb].owner == s.player && (st == 6 || st == 13 || st == 22) {
                            stars_per_turn += s.tiles[nb].structure_level.max(1);
                        }
                    }
                    if stars_per_turn >= 2 {
                        1800.0 + (stars_per_turn as f64) * 300.0
                    } else if stars_per_turn == 1 {
                        900.0
                    } else {
                        -500.0
                    }
                }
                48 => if is_water_map && s.stars >= 10 { 850.0 } else { -200.0 },
                3 => {
                    let mut adj_unconnected_city = false;
                    let mut adj_network = false;
                    for nb in neighbors(dst, s.size) {
                        if let Some(ci) = s.city_at[nb] {
                            let c = &s.cities[ci];
                            if c.owner == s.player {
                                if c.connected || s.tiles[nb].capital == s.player {
                                    adj_network = true;
                                } else {
                                    adj_unconnected_city = true;
                                }
                            }
                        } else if s.tiles[nb].road && s.tiles[nb].owner == s.player {
                            adj_network = true;
                        }
                    }
                    if adj_unconnected_city && adj_network {
                        2100.0
                    } else {
                        -200.0
                    }
                }
                17..=20 => -5000.0,
                _ => -200.0,
            }
        }

        // Research technology
        7 => {
            let tech_id = a.parameter.max(0) as usize;
            let cost = tech_cost(tech_id, my_cities_count, s.tech[22]);
            let stars_after = s.stars - cost;

            let mut owned_fish = 0;
            let mut owned_fruit = 0;
            let mut owned_crops = 0;
            let mut owned_metal = 0;
            let mut owned_forests = 0;
            let mut owned_lumber_huts = 0;
            let mut owned_farms = 0;
            let mut owned_mines = 0;
            let mut owned_shallow_water = 0;
            let mut has_port = false;
            let mut already_usable_pop = 0;

            for i in 0..s.n {
                let t = &s.tiles[i];
                if !t.known || t.owner != s.player {
                    continue;
                }
                if t.structure == 8 {
                    has_port = true;
                }
                if t.structure == 12 {
                    owned_lumber_huts += 1;
                }
                if t.structure == 5 {
                    owned_farms += 1;
                }
                if t.structure == 21 {
                    owned_mines += 1;
                }
                if t.structure == 0 {
                    match t.resource {
                        1 => if s.tech[15] { already_usable_pop += 1; },
                        2 => {
                            owned_crops += 1;
                            if s.tech[8] { already_usable_pop += 2; }
                        }
                        3 => {
                            owned_fish += 1;
                            if s.tech[10] { already_usable_pop += 1; }
                        }
                        5 => {
                            owned_metal += 1;
                            if s.tech[23] { already_usable_pop += 2; }
                        }
                        6 => {
                            owned_fruit += 1;
                            if s.tech[6] { already_usable_pop += 1; }
                        }
                        _ => {}
                    }
                    if t.terrain == 5 {
                        owned_forests += 1;
                        if s.tech[16] {
                            already_usable_pop += 1;
                        }
                    }
                    if t.terrain == 1 {
                        owned_shallow_water += 1;
                    }
                }
            }

            let mut value = match tech_id {
                16 => 550.0 + (owned_forests as f64) * 140.0,
                17 => 450.0 + (owned_lumber_huts as f64) * 180.0 + if my_cities_count >= 2 { 200.0 } else { 0.0 },
                1 => if !need_naval_expansion { 560.0 } else { 280.0 },
                6 => 260.0 + (owned_fruit as f64) * 180.0 - if already_usable_pop >= 3 { 250.0 } else { 0.0 },
                10 => {
                    let mut v = 250.0 + (owned_fish as f64) * 180.0;
                    if is_water_map && owned_shallow_water > 0 {
                        v += 450.0;
                    }
                    if need_naval_expansion && owned_shallow_water > 0 {
                        v += 600.0;
                    }
                    v
                }
                13 => {
                    let mut v = 180.0;
                    if is_water_map {
                        v += 450.0;
                    }
                    if has_port || my_unit_counts[43] > 0 {
                        v += 550.0;
                    }
                    if need_naval_expansion {
                        v += 500.0;
                    }
                    v
                }
                44 => {
                    if has_port || my_unit_counts[43] > 0 || my_unit_counts[42] > 0 {
                        650.0
                    } else if is_water_map {
                        350.0
                    } else {
                        50.0
                    }
                }
                14 => {
                    let sf_count = starfish.len();
                    let mut v = if has_port || my_unit_counts[43] > 0 || my_unit_counts[42] > 0 { 550.0 } else { 150.0 };
                    v += (sf_count as f64) * 200.0;
                    v
                }
                18 => 420.0 + if total_my_units >= 3 { 150.0 } else { 0.0 },
                4 => if my_cities_count >= 2 { 460.0 } else { 200.0 },
                20 => {
                    let mountain_ruin = (0..s.n).any(|i| s.tiles[i].known && s.tiles[i].structure == 2 && s.tiles[i].terrain == 4);
                    340.0 + if mountain_ruin { 300.0 } else { 0.0 }
                }
                23 => 220.0 + (owned_metal as f64) * 240.0,
                24 => 480.0 + (owned_mines as f64) * 220.0,
                8 => 220.0 + (owned_crops as f64) * 240.0,
                9 => 180.0 + (owned_farms as f64) * 200.0,
                7 => 380.0 + if is_water_map { 150.0 } else { 0.0 },
                2 => if my_cities_count >= 3 && s.stars >= cost + 8 { 360.0 } else { 140.0 },
                3 => 680.0,
                5 => if owned_lumber_huts >= 2 || owned_farms >= 2 || owned_mines >= 2 { 520.0 } else { 150.0 },
                38 => 360.0,
                21 => if my_cities_count <= 2 && s.stars >= cost + 6 { 220.0 } else { 80.0 },
                22 => 450.0,
                12 => if is_water_map && (my_unit_counts[41] + my_unit_counts[42] + my_unit_counts[44] >= 2) { 500.0 } else { 100.0 },
                _ => 50.0,
            };

            if stars_after >= 2 {
                value += 180.0;
            }
            if stars_after >= 5 {
                value += 150.0;
            }

            let blocks_immediate_levelup = city_plans.iter().any(|p| p.can_level_now && stars_after < p.min_stars_to_level);
            if blocks_immediate_levelup && !has_pending_village_capture {
                value -= 600.0;
            }

            if value < 480.0 {
                return -100.0;
            }

            if has_pending_village_capture {
                5300.0 + value
            } else {
                800.0 + value
            }
        }

        _ => -100.0,
    }
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Memory::new();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let obs: Value = serde_json::from_str(&line).expect("valid json");
        let state = parse_obs(&obs);
        mem.update(&state);
        let action_id = choose_action(&state, &mem);
        mem.record_action(&state, &action_id);
        writeln!(stdout, "{}", json!({"action_id": action_id})).unwrap();
        stdout.flush().unwrap();
    }
}
