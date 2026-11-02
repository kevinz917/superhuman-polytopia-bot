use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

const INF: i32 = 1_000_000;

#[derive(Clone, Copy, Debug)]
struct UnitStats {
    cost: i32,
    health: f32,
    attack: f32,
    defense: f32,
    movement: i32,
    range: i32,
    dash: bool,
    fortify: bool,
    escape: bool,
    persist: bool,
    stiff: bool,
    stat: bool,
    float: bool,
    carry: bool,
    scout: bool,
    creep: bool,
    hide: bool,
    infiltrate: bool,
    surprise: bool,
    splash: bool,
    stomp: bool,
}

impl Default for UnitStats {
    fn default() -> Self {
        Self {
            cost: -1,
            health: 1.0,
            attack: -1.0,
            defense: 0.0,
            movement: 1,
            range: 1,
            dash: false,
            fortify: false,
            escape: false,
            persist: false,
            stiff: false,
            stat: false,
            float: false,
            carry: false,
            scout: false,
            creep: false,
            hide: false,
            infiltrate: false,
            surprise: false,
            splash: false,
            stomp: false,
        }
    }
}

fn unit_stats(kind: i32) -> UnitStats {
    let mut s = UnitStats::default();
    match kind {
        2 => { s.cost = 2; s.health = 10.0; s.attack = 2.0; s.defense = 2.0; s.movement = 1; s.range = 1; s.dash = true; s.fortify = true; }
        3 => { s.cost = 3; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 2; s.range = 1; s.dash = true; s.escape = true; s.fortify = true; }
        4 => { s.cost = 8; s.health = 10.0; s.attack = 3.5; s.defense = 1.0; s.movement = 3; s.range = 1; s.dash = true; s.fortify = true; s.persist = true; }
        5 => { s.cost = 3; s.health = 15.0; s.attack = 1.0; s.defense = 3.0; s.movement = 1; s.range = 1; s.fortify = true; }
        8 => { s.cost = 8; s.health = 10.0; s.attack = 4.0; s.defense = 0.0; s.movement = 1; s.range = 3; s.stiff = true; }
        9 => { s.cost = 3; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 1; s.range = 2; s.dash = true; s.fortify = true; }
        10 => { s.cost = 5; s.health = 10.0; s.attack = 0.0; s.defense = 1.0; s.movement = 1; s.range = 1; s.stiff = true; }
        11 => { s.cost = 5; s.health = 15.0; s.attack = 3.0; s.defense = 3.0; s.movement = 1; s.range = 1; s.dash = true; }
        12 => { s.cost = 10; s.health = 40.0; s.attack = 5.0; s.defense = 4.0; s.movement = 1; s.range = 1; s.stat = true; }
        37 => { s.cost = 2; s.health = 10.0; s.attack = 2.0; s.defense = 2.0; s.movement = 1; s.range = 1; s.dash = true; s.stat = true; s.surprise = true; }
        38 => { s.cost = 8; s.health = 5.0; s.attack = 0.0; s.defense = 0.5; s.movement = 2; s.range = 1; s.creep = true; s.dash = true; s.hide = true; s.infiltrate = true; s.scout = true; s.stat = true; s.stiff = true; }
        39 => { s.cost = 2; s.health = 5.0; s.attack = 0.0; s.defense = 0.5; s.movement = 2; s.range = 1; s.carry = true; s.dash = true; s.float = true; s.hide = true; s.infiltrate = true; s.scout = true; s.stat = true; s.stiff = true; }
        40 => { s.cost = 2; s.health = 10.0; s.attack = 2.0; s.defense = 2.0; s.movement = 2; s.range = 1; s.carry = true; s.dash = true; s.float = true; s.stat = true; s.surprise = true; }
        41 => { s.cost = 15; s.health = 10.0; s.attack = 3.0; s.defense = 2.0; s.movement = 2; s.range = 3; s.carry = true; s.float = true; s.splash = true; s.stat = true; s.stiff = true; }
        42 => { s.cost = 5; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 3; s.range = 2; s.carry = true; s.dash = true; s.float = true; s.scout = true; s.stat = true; }
        43 => { s.cost = 0; s.health = 10.0; s.attack = 0.0; s.defense = 1.0; s.movement = 2; s.range = 2; s.carry = true; s.float = true; s.stat = true; s.stiff = true; }
        44 => { s.cost = 5; s.health = 10.0; s.attack = 3.0; s.defense = 3.0; s.movement = 3; s.range = 1; s.carry = true; s.dash = true; s.float = true; s.stat = true; }
        45 => { s.cost = 10; s.health = 40.0; s.attack = 4.0; s.defense = 4.0; s.movement = 2; s.range = 1; s.carry = true; s.float = true; s.stat = true; s.stiff = true; s.stomp = true; }
        _ => { s.cost = 3; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 1; s.range = 1; s.dash = true; }
    }
    s
}

fn tech_tier(tech: i32) -> i32 {
    match tech {
        0 => 0,
        1 | 6 | 10 | 15 | 20 => 1,
        2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2,
        3 | 5 | 9 | 12 | 14 | 17 | 19 | 22 | 24 | 38 => 3,
        _ => 1,
    }
}

fn structure_cost(kind: i32) -> i32 {
    match kind {
        3 => 3,
        5 => 5,
        6 => 5,
        8 => 7,
        12 => 3,
        13 => 5,
        17 => 20,
        18 => 15,
        19 => 20,
        20 => 20,
        21 => 5,
        22 => 5,
        23..=29 => 0,
        48 => 5,
        50 => 5,
        70 => 5,
        _ => 5,
    }
}

#[derive(Clone, Debug, Default)]
struct TileObs {
    known: bool,
    terrain: i32,
    owner: i32,
    resource: i32,
    structure: i32,
    structure_level: i32,
    road: bool,
    capital: i32,
}

#[derive(Clone, Debug, Default)]
struct UnitObs {
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

#[derive(Clone, Debug, Default)]
struct CityObs {
    tile: usize,
    owner: i32,
    level: i32,
    population: i32,
    production: i32,
    progress: i32,
    connected: bool,
    riot: bool,
    rewards: [i32; 9],
}

#[derive(Clone, Debug)]
struct LegalCmd {
    id: String,
    kind: i32,
    source: i32,
    target: i32,
    param: i32,
}

struct Memory {
    last_version: i32,
    last_turn: i32,
    tile_city: Vec<i32>,
    city_order: Vec<i32>,
    next_order: i32,
    my_capital: Option<usize>,
    opp_capital: Option<usize>,
    seen_enemy_cities: Vec<bool>,
}

impl Memory {
    fn new() -> Self {
        Self {
            last_version: -1,
            last_turn: -1,
            tile_city: Vec::new(),
            city_order: Vec::new(),
            next_order: 1,
            my_capital: None,
            opp_capital: None,
            seen_enemy_cities: Vec::new(),
        }
    }

    fn reset(&mut self, n: usize) {
        self.last_version = -1;
        self.last_turn = -1;
        self.tile_city = vec![-1; n];
        self.city_order = vec![0; n];
        self.next_order = 1;
        self.my_capital = None;
        self.opp_capital = None;
        self.seen_enemy_cities = vec![false; n];
    }
}

#[inline]
fn cheb(size: usize, a: usize, b: usize) -> i32 {
    let ax = (a % size) as i32;
    let ay = (a / size) as i32;
    let bx = (b % size) as i32;
    let by = (b / size) as i32;
    (ax - bx).abs().max((ay - by).abs())
}

#[inline]
fn combat_calc(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    let divisor = if total > 0.0 { total } else { 1.0 };
    let attack = ((if total > 0.0 { (af / divisor) * aa * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    let defense = ((if dh > attack && total > 0.0 { (df / divisor) * dd * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    (attack, defense, attack * 0.5)
}

struct ParsedState {
    player: i32,
    opp: i32,
    version: i32,
    size: usize,
    n: usize,
    turn: i32,
    stars: i32,
    tech: [bool; 128],
    tiles: Vec<TileObs>,
    units: Vec<UnitObs>,
    unit_at: Vec<Option<usize>>,
    cities: Vec<CityObs>,
    city_at: Vec<Option<usize>>,
    legal: Vec<LegalCmd>,
}

fn parse_obs(obs: &Value, mem: &mut Memory) -> ParsedState {
    let player = obs["player"].as_i64().unwrap_or(1) as i32;
    let opp = 3 - player;
    let version = obs["state_version"].as_i64().unwrap_or(0) as i32;
    let v = &obs["visible_state"];
    let size = v["board_size"].as_u64().unwrap_or(14) as usize;
    let n = size * size;
    let turn = v["turn"].as_i64().unwrap_or(0) as i32;
    let stars = v["stars"].as_i64().unwrap_or(0) as i32;

    if mem.tile_city.len() != n || version < mem.last_version || turn < mem.last_turn {
        mem.reset(n);
    }
    mem.last_version = version;
    mem.last_turn = turn;

    let mut tech = [false; 128];
    if let Some(arr) = v["technology"].as_array() {
        for x in arr {
            if let Some(id) = x.as_i64() {
                if (0..128).contains(&id) {
                    tech[id as usize] = true;
                }
            }
        }
    }

    let mut tiles = vec![TileObs::default(); n];
    if let Some(arr) = v["tiles"].as_array() {
        for row in arr {
            let idx = row["tile"].as_u64().unwrap_or(0) as usize;
            if idx >= n { continue; }
            let known = row["known"].as_bool().unwrap_or(false);
            let mut t = TileObs { known, ..Default::default() };
            if known {
                t.terrain = row["terrain"].as_i64().unwrap_or(0) as i32;
                t.owner = row["owner"].as_i64().unwrap_or(0) as i32;
                t.resource = row["resource"].as_i64().unwrap_or(0) as i32;
                t.structure = row["structure"].as_i64().unwrap_or(0) as i32;
                t.structure_level = row["structure_level"].as_i64().unwrap_or(0) as i32;
                t.road = row["road"].as_i64().unwrap_or(0) != 0;
                t.capital = row["capital"].as_i64().unwrap_or(0) as i32;
                if t.capital == player {
                    mem.my_capital = Some(idx);
                } else if t.capital == opp {
                    mem.opp_capital = Some(idx);
                }
            }
            tiles[idx] = t;
        }
    }

    if let Some(rem) = obs.get("remembered_state").and_then(|r| r.get("historical")).and_then(|h| h.as_array()) {
        for row in rem {
            if row["entity"].as_str() == Some("tiles") {
                let fact = &row["fact"];
                let idx = fact["tile"].as_u64().unwrap_or(0) as usize;
                if idx < n && !tiles[idx].known {
                    tiles[idx].known = true;
                    tiles[idx].terrain = fact["terrain"].as_i64().unwrap_or(0) as i32;
                    tiles[idx].owner = fact["owner"].as_i64().unwrap_or(0) as i32;
                    tiles[idx].resource = fact["resource"].as_i64().unwrap_or(0) as i32;
                    tiles[idx].structure = fact["structure"].as_i64().unwrap_or(0) as i32;
                }
            }
        }
    }

    let mut units = Vec::new();
    let mut unit_at = vec![None; n];
    if let Some(arr) = v["units"].as_array() {
        for row in arr {
            let tile = row["tile"].as_u64().unwrap_or(0) as usize;
            if tile >= n { continue; }
            let u = UnitObs {
                tile,
                owner: row["owner"].as_i64().unwrap_or(0) as i32,
                kind: row["kind"].as_i64().unwrap_or(0) as i32,
                health: row["health"].as_f64().unwrap_or(10.0) as f32,
                veteran: row["veteran"].as_bool().unwrap_or(false),
                moved: row["moved"].as_bool().unwrap_or(false),
                attacked: row["attacked"].as_bool().unwrap_or(false),
                poison: row["poison"].as_bool().unwrap_or(false),
                frozen: row["frozen"].as_bool().unwrap_or(false),
                boosted: row["boosted"].as_bool().unwrap_or(false),
                invisible: row["invisible"].as_bool().unwrap_or(false),
                detect: row["detect"].as_bool().unwrap_or(false),
                kills: row["kills"].as_i64().unwrap_or(0) as i32,
                home: row["home"].as_i64().unwrap_or(0) as i32,
                passenger: row["passenger"].as_i64().unwrap_or(0) as i32,
            };
            unit_at[tile] = Some(units.len());
            units.push(u);
        }
    }

    let mut cities = Vec::new();
    let mut city_at = vec![None; n];
    if let Some(arr) = v["cities"].as_array() {
        for row in arr {
            let tile = row["tile"].as_u64().unwrap_or(0) as usize;
            if tile >= n { continue; }
            let owner = row["owner"].as_i64().unwrap_or(0) as i32;
            let mut rewards = [0i32; 9];
            if let Some(rarr) = row["rewards"].as_array() {
                for (k, val) in rarr.iter().enumerate().take(9) {
                    rewards[k] = val.as_i64().unwrap_or(0) as i32;
                }
            }
            let c = CityObs {
                tile,
                owner,
                level: row["level"].as_i64().unwrap_or(1) as i32,
                population: row["population"].as_i64().unwrap_or(0) as i32,
                production: row["production"].as_i64().unwrap_or(1) as i32,
                progress: row["progress"].as_i64().unwrap_or(0) as i32,
                connected: row["connected"].as_bool().unwrap_or(false),
                riot: row["riot"].as_bool().unwrap_or(false),
                rewards,
            };
            if owner == opp {
                mem.seen_enemy_cities[tile] = true;
            } else if owner == player {
                mem.seen_enemy_cities[tile] = false;
                if mem.city_order[tile] == 0 {
                    mem.city_order[tile] = mem.next_order;
                    mem.next_order += 1;
                }
            }
            city_at[tile] = Some(cities.len());
            cities.push(c);
        }
    }

    // Update tile_city mapping for our owned tiles
    for t in 0..n {
        if !tiles[t].known || tiles[t].owner != player {
            mem.tile_city[t] = -1;
            continue;
        }
        if let Some(ci) = city_at[t] {
            if cities[ci].owner == player {
                mem.tile_city[t] = t as i32;
                continue;
            }
        }
        let cur = mem.tile_city[t];
        let valid_cur = cur >= 0 && (cur as usize) < n && city_at[cur as usize].map(|ci| cities[ci].owner == player).unwrap_or(false);
        if !valid_cur {
            let mut best_c = -1i32;
            let mut best_key = (999, 999999);
            for c in &cities {
                if c.owner != player { continue; }
                let rad = 1 + if c.rewards[5] > 0 { 1 } else { 0 };
                let d = cheb(size, c.tile, t);
                if d <= rad {
                    let ord = if mem.city_order[c.tile] > 0 { mem.city_order[c.tile] } else { 9999 };
                    let key = (if d <= 1 { 0 } else { 1 }, ord);
                    if key < best_key {
                        best_key = key;
                        best_c = c.tile as i32;
                    }
                }
            }
            if best_c < 0 {
                // Fallback to nearest own city within Chebyshev 2
                let mut min_d = 99;
                for c in &cities {
                    if c.owner == player {
                        let d = cheb(size, c.tile, t);
                        if d < min_d {
                            min_d = d;
                            best_c = c.tile as i32;
                        }
                    }
                }
            }
            mem.tile_city[t] = best_c;
        }
    }

    let mut legal = Vec::new();
    if let Some(arr) = obs["legal_actions"].as_array() {
        for a in arr {
            let id = a["action_id"].as_str().unwrap_or("").to_owned();
            let c = &a["command"];
            legal.push(LegalCmd {
                id,
                kind: c["kind"].as_i64().unwrap_or(0) as i32,
                source: c["source"].as_i64().unwrap_or(-1) as i32,
                target: c["target"].as_i64().unwrap_or(-1) as i32,
                param: c["parameter"].as_i64().unwrap_or(0) as i32,
            });
        }
    }

    ParsedState {
        player,
        opp,
        version,
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

fn neighbors(size: usize, tile: usize) -> Vec<usize> {
    let x = (tile % size) as i32;
    let y = (tile / size) as i32;
    let s = size as i32;
    let mut out = Vec::with_capacity(8);
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 { continue; }
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0 && nx < s && ny >= 0 && ny < s {
                out.push((ny * s + nx) as usize);
            }
        }
    }
    out
}

fn max_hp(u: &UnitObs) -> f32 {
    let base_kind = if u.passenger > 0 { u.passenger } else { u.kind };
    unit_stats(base_kind).health + if u.veteran { 5.0 } else { 0.0 }
}

fn unit_allows_work(ps: &ParsedState, tile: usize) -> bool {
    match ps.unit_at[tile] {
        None => true,
        Some(ui) => ps.units[ui].owner == ps.player,
    }
}

fn defense_bonus_at(ps: &ParsedState, tile: usize, u: &UnitObs) -> f32 {
    let t = &ps.tiles[tile];
    let st = unit_stats(u.kind);
    let is_own_city = ps.city_at[tile].map(|ci| ps.cities[ci].owner == u.owner).unwrap_or(false);
    let city_bonus = if is_own_city && st.fortify {
        if let Some(ci) = ps.city_at[tile] {
            if u.owner == ps.player && ps.cities[ci].rewards[1] > 0 {
                4.0
            } else {
                1.5
            }
        } else {
            1.5
        }
    } else {
        1.0
    };
    let bonus = if t.terrain == 1 || t.terrain == 2 {
        if u.owner == ps.player && ps.tech[12] { 1.5 } else { 1.0 }
    } else if t.terrain == 5 {
        if u.owner == ps.player {
            if ps.tech[18] { 1.5 } else { 1.0 }
        } else {
            1.5 // conservatively assume enemy might have Archery in forest or 1.0
        }
    } else if t.terrain == 4 {
        1.5
    } else {
        city_bonus
    };
    bonus * if u.poison { 0.5 } else { 1.0 }
}

// Compute exact attack damage and retaliation if unit `att` at `att_tile` attacks `def` at `def_tile`
fn eval_attack(ps: &ParsedState, att: &UnitObs, att_tile: usize, def: &UnitObs, def_tile: usize) -> (f32, f32, bool, bool) {
    let ast = unit_stats(att.kind);
    let dst = unit_stats(def.kind);
    let am = max_hp(att);
    let dm = max_hp(def);
    let mut db = defense_bonus_at(ps, def_tile, def);
    // For enemy in forest, check actual 1.0 vs 1.5: if 1.5 still kills, definitely kills!
    let dd = dst.defense * if def.poison { 0.7 } else { 1.0 };
    if ps.tiles[def_tile].terrain == 5 && def.owner != ps.player {
        db = 1.0 * if def.poison { 0.5 } else { 1.0 };
    }
    let (dmg, ret, _) = combat_calc(ast.attack, att.health, am, dd, def.health, dm, db);
    let killed = def.health - dmg <= 0.0;
    let dist = cheb(ps.size, att_tile, def_tile);
    let can_retaliate = !killed && !ast.surprise && !dst.stiff && dist <= dst.range;
    let actual_ret = if can_retaliate { ret } else { 0.0 };
    let self_died = att.health - actual_ret <= 0.0;
    (dmg, actual_ret, killed, self_died)
}


#[derive(Clone, Debug, Default)]
struct CityPlan {
    city_tile: usize,
    needed_pop: i32,
    can_level_now: bool,
    min_cost_to_level: i32,
    gateway_to_water: bool,
    unclaimed_forests: i32,
    unclaimed_fruits: i32,
    unclaimed_fish: i32,
}

fn build_city_plans(ps: &ParsedState, mem: &Memory) -> Vec<CityPlan> {
    let mut plans = Vec::new();
    for c in &ps.cities {
        if c.owner != ps.player { continue; }
        let needed = (c.level + 1) - c.progress;
        let mut gateway = false;
        if c.level < 4 && c.rewards[5] == 0 {
            for t in 0..ps.n {
                if cheb(ps.size, c.tile, t) == 2 && ps.tiles[t].known && ps.tiles[t].terrain == 1 && ps.tiles[t].owner == 0 {
                    gateway = true;
                    break;
                }
            }
        }
        let mut unclaimed_forests = 0;
        let mut unclaimed_fruits = 0;
        let mut unclaimed_fish = 0;
        for t in 0..ps.n {
            if mem.tile_city[t] == c.tile as i32 && ps.tiles[t].known && ps.tiles[t].structure == 0 && unit_allows_work(ps, t) {
                if ps.tiles[t].terrain == 5 { unclaimed_forests += 1; }
                if ps.tiles[t].resource == 6 { unclaimed_fruits += 1; }
                if ps.tiles[t].resource == 3 { unclaimed_fish += 1; }
            }
        }

        let mut actions = Vec::new();
        for cmd in &ps.legal {
            if cmd.kind == 5 {
                let t = cmd.target as usize;
                if mem.tile_city[t] == c.tile as i32 {
                    actions.push((5, t, 0, 2, 1));
                }
            } else if cmd.kind == 6 {
                let t = cmd.target as usize;
                let p = cmd.param;
                if mem.tile_city[t] != c.tile as i32 { continue; }
                let cost = structure_cost(p);
                let pop = match p {
                    12 => {
                        let has_sawmill = neighbors(ps.size, t).iter().any(|&nb| {
                            ps.tiles[nb].known && ps.tiles[nb].owner == ps.player && ps.tiles[nb].structure == 13 && mem.tile_city[nb] == c.tile as i32
                        });
                        1 + if has_sawmill { 1 } else { 0 }
                    }
                    5 => {
                        let has_windmill = neighbors(ps.size, t).iter().any(|&nb| {
                            ps.tiles[nb].known && ps.tiles[nb].owner == ps.player && ps.tiles[nb].structure == 6 && mem.tile_city[nb] == c.tile as i32
                        });
                        2 + if has_windmill { 1 } else { 0 }
                    }
                    21 => {
                        let has_forge = neighbors(ps.size, t).iter().any(|&nb| {
                            ps.tiles[nb].known && ps.tiles[nb].owner == ps.player && ps.tiles[nb].structure == 22 && mem.tile_city[nb] == c.tile as i32
                        });
                        2 + if has_forge { 2 } else { 0 }
                    }
                    13 => {
                        neighbors(ps.size, t).iter().filter(|&&nb| {
                            ps.tiles[nb].known && ps.tiles[nb].owner == ps.player && ps.tiles[nb].structure == 12
                        }).count() as i32
                    }
                    6 => {
                        neighbors(ps.size, t).iter().filter(|&&nb| {
                            ps.tiles[nb].known && ps.tiles[nb].owner == ps.player && ps.tiles[nb].structure == 5
                        }).count() as i32
                    }
                    22 => {
                        2 * (neighbors(ps.size, t).iter().filter(|&&nb| {
                            ps.tiles[nb].known && ps.tiles[nb].owner == ps.player && ps.tiles[nb].structure == 21
                        }).count() as i32)
                    }
                    8 => 1,
                    23..=29 => 3,
                    _ => 0,
                };
                if pop > 0 {
                    actions.push((6, t, p, cost, pop));
                }
            }
        }
        actions.sort_by_key(|a| (a.3 * 10) / a.4.max(1));

        let mut used_tile = vec![false; ps.n];
        let mut used_per_city = [false; 128];
        let mut gained = 0;
        let mut spent = 0;
        for &(_k, t, p, cost, pop) in &actions {
            if gained >= needed { break; }
            if used_tile[t] { continue; }
            if (p == 13 || p == 6 || p == 22 || (23..=29).contains(&p)) && used_per_city[p as usize] {
                continue;
            }
            if spent + cost <= ps.stars {
                used_tile[t] = true;
                if p > 0 && (p as usize) < 128 {
                    used_per_city[p as usize] = true;
                }
                spent += cost;
                gained += pop;
            }
        }
        let can_level_now = needed > 0 && gained >= needed;
        let min_cost_to_level = if can_level_now { spent } else { INF };
        plans.push(CityPlan {
            city_tile: c.tile,
            needed_pop: needed,
            can_level_now,
            min_cost_to_level,
            gateway_to_water: gateway,
            unclaimed_forests,
            unclaimed_fruits,
            unclaimed_fish,
        });
    }
    plans
}

fn land_components(ps: &ParsedState) -> (Vec<i32>, usize) {
    let mut comp = vec![-1i32; ps.n];
    let mut cid = 0i32;
    for start in 0..ps.n {
        let is_land = ps.tiles[start].known && (ps.tiles[start].terrain >= 3 || ps.tiles[start].structure == 48);
        if !is_land || comp[start] >= 0 { continue; }
        let mut stack = vec![start];
        comp[start] = cid;
        while let Some(u) = stack.pop() {
            for v in neighbors(ps.size, u) {
                let v_land = ps.tiles[v].known && (ps.tiles[v].terrain >= 3 || ps.tiles[v].structure == 48);
                if v_land && comp[v] < 0 {
                    comp[v] = cid;
                    stack.push(v);
                }
            }
        }
        cid += 1;
    }
    (comp, cid as usize)
}

// Forward Dijkstra from a single source tile `start` to all known tiles, returning step distance
fn forward_land_dist(ps: &ParsedState, start: usize) -> Vec<i32> {
    let mut dist = vec![INF; ps.n];
    let mut pq: std::collections::BinaryHeap<(i32, usize)> = std::collections::BinaryHeap::new();
    dist[start] = 0;
    pq.push((0, start));
    while let Some((neg_d, u)) = pq.pop() {
        let d = -neg_d;
        if d > dist[u] { continue; }
        let tu = &ps.tiles[u];
        let u_road = tu.known && (tu.road || tu.structure == 1 || tu.structure == 48);
        for v in neighbors(ps.size, u) {
            let tv = &ps.tiles[v];
            if !tv.known { continue; }
            let land_ok = tv.terrain == 3 || tv.terrain == 5 || (tv.terrain == 4 && ps.tech[20]) || tv.structure == 48;
            if !land_ok { continue; }
            let v_road = tv.road || tv.structure == 1 || tv.structure == 48;
            let step = if u_road && v_road {
                1
            } else if tv.terrain == 5 || tv.terrain == 4 {
                3
            } else {
                2
            };
            let nd = d + step;
            if nd < dist[v] {
                dist[v] = nd;
                pq.push((-nd, v));
            }
        }
    }
    dist
}

// Backward Dijkstra from `targets` to all tiles: unit moves FROM `v` INTO `u`!
fn backward_dijkstra(ps: &ParsedState, targets: &[(usize, i32)], allow_water: bool, is_boat: bool) -> Vec<i32> {
    let mut dist = vec![INF; ps.n];
    let mut pq: std::collections::BinaryHeap<(i32, usize)> = std::collections::BinaryHeap::new();
    for &(t, base_cost) in targets {
        if t < ps.n && ps.tiles[t].known && base_cost < dist[t] {
            dist[t] = base_cost;
            pq.push((-base_cost, t));
        }
    }
    while let Some((neg_d, u)) = pq.pop() {
        let d = -neg_d;
        if d > dist[u] { continue; }
        let tu = &ps.tiles[u];
        if !tu.known { continue; }
        for v in neighbors(ps.size, u) {
            let tv = &ps.tiles[v];
            if !tv.known { continue; }
            // Cost of stepping FROM `v` INTO `u`
            let u_is_land = tu.terrain >= 3 || tu.structure == 48;
            let v_is_land = tv.terrain >= 3 || tv.structure == 48;
            let u_own_port = tu.structure == 8 && tu.owner == ps.player;
            let v_own_port = tv.structure == 8 && tv.owner == ps.player;

            let step_cost = if is_boat {
                if !u_is_land {
                    if tu.terrain == 2 && !ps.tech[13] { 15 } else { 1 }
                } else {
                    // Boat disembarking onto land `u` from water `v`
                    if !v_is_land && (tu.terrain != 4 || ps.tech[20]) { 3 } else { INF }
                }
            } else if u_is_land && v_is_land {
                if tu.terrain == 4 && !ps.tech[20] {
                    if allow_water { 16 } else { INF }
                } else if (tu.road || tu.structure == 1 || tu.structure == 48) && (tv.road || tv.structure == 1 || tv.structure == 48) {
                    1
                } else if tu.terrain == 5 || tu.terrain == 4 {
                    3
                } else {
                    2
                }
            } else if !allow_water {
                INF
            } else if v_is_land && !u_is_land {
                // Land unit at `v` stepping onto water `u`: MUST be an owned port (or potential port tile)
                if u_own_port && ps.tech[10] {
                    2
                } else if tu.terrain == 1 && tu.owner == ps.player && tu.structure == 0 {
                    10
                } else {
                    INF
                }
            } else if !v_is_land && !u_is_land {
                // Water `v` to water `u`
                if tu.terrain == 2 && !ps.tech[13] { 10 } else { 2 }
            } else {
                // Water `v` to land `u` (disembark)
                if tu.terrain == 4 && !ps.tech[20] { INF } else { 3 }
            };

            if step_cost >= INF { continue; }
            let nd = d + step_cost;
            if nd < dist[v] {
                dist[v] = nd;
                pq.push((-nd, v));
            }
        }
    }
    dist
}

fn choose_action(ps: &ParsedState, mem: &mut Memory) -> String {
    if ps.legal.is_empty() {
        return String::new();
    }
    if ps.legal.len() == 1 {
        return ps.legal[0].id.clone();
    }

    // Phase 1: Mandatory City Rewards (kind == 9)
    if ps.legal.iter().any(|a| a.kind == 9) {
        let mut best_id = ps.legal[0].id.clone();
        let mut best_score = i64::MIN;
        let my_ports = (0..ps.n).filter(|&i| ps.tiles[i].known && ps.tiles[i].owner == ps.player && ps.tiles[i].structure == 8).count();
        let my_shallow = (0..ps.n).filter(|&i| ps.tiles[i].known && ps.tiles[i].owner == ps.player && ps.tiles[i].terrain == 1).count();
        let known_water = (0..ps.n).filter(|&i| ps.tiles[i].known && ps.tiles[i].terrain <= 2).count();
        for a in &ps.legal {
            if a.kind != 9 { continue; }
            let ctile = a.target as usize;
            let param = a.param;
            let city = ps.city_at[ctile].map(|ci| &ps.cities[ci]);
            let s: i64 = match param {
                3 => 10_000, // Workshop (+1 star/turn)
                4 => 2_000,  // Explorer
                7 => 10_000, // Resources (+5 stars)
                1 => {       // City Wall
                    let threats = ps.units.iter().filter(|u| u.owner == ps.opp && cheb(ps.size, u.tile, ctile) <= 2).count();
                    if threats >= 2 && ps.stars >= 8 { 11_000 } else { 1_000 }
                }
                8 => {       // Population (+3)
                    let mut v = 9_000;
                    if let Some(c) = city {
                        if c.progress + 3 >= c.level + 1 {
                            v += 4_000; // Immediately levels to L5 Giant!
                        }
                    }
                    v
                }
                5 => {       // Border Growth
                    let mut new_shallow = 0;
                    let mut new_res = 0;
                    let mut new_land = 0;
                    for t in 0..ps.n {
                        if cheb(ps.size, ctile, t) == 2 && ps.tiles[t].known && ps.tiles[t].owner == 0 {
                            if ps.tiles[t].terrain == 1 { new_shallow += 1; }
                            if ps.tiles[t].terrain >= 3 { new_land += 1; }
                            if ps.tiles[t].resource > 0 || ps.tiles[t].terrain == 5 { new_res += 1; }
                        }
                    }
                    let mut v = 6_000 + (new_res * 600) + (new_land * 150);
                    if new_shallow > 0 && my_ports == 0 && my_shallow == 0 && (known_water >= 15 || new_shallow >= 2) {
                        v += 20_000; // Critical gateway to sea on continents!
                    } else if new_res >= 5 {
                        v += 5_000;
                    }
                    v
                }
                6 => 20_000, // Giant!
                2 => 1_000,  // Park
                _ => 0,
            };
            if s > best_score {
                best_score = s;
                best_id = a.id.clone();
            }
        }
        return best_id;
    }

    // Phase 1b: Immediate free promotions (kind == 3, param == 16)
    for a in &ps.legal {
        if a.kind == 3 && a.param == 16 {
            return a.id.clone();
        }
    }

    let (land_comp, _) = land_components(ps);
    let my_cities_count = ps.cities.iter().filter(|c| c.owner == ps.player).count() as i32;
    let opp_cities_known: Vec<usize> = (0..ps.n)
        .filter(|&t| {
            (ps.city_at[t].map(|ci| ps.cities[ci].owner == ps.opp).unwrap_or(false))
                || (mem.seen_enemy_cities[t] && (!ps.tiles[t].known || ps.tiles[t].owner != ps.player))
        })
        .collect();
    let neutral_villages: Vec<usize> = (0..ps.n)
        .filter(|&t| ps.tiles[t].known && ps.tiles[t].structure == 1 && ps.tiles[t].owner == 0)
        .collect();
    let ruins: Vec<usize> = (0..ps.n)
        .filter(|&t| ps.tiles[t].known && ps.tiles[t].structure == 2)
        .collect();
    let starfish_tiles: Vec<usize> = (0..ps.n)
        .filter(|&t| ps.tiles[t].known && ps.tiles[t].resource == 8 && ps.tiles[t].structure == 0)
        .collect();

    let known_tiles = (0..ps.n).filter(|&t| ps.tiles[t].known).count();
    let known_water = (0..ps.n).filter(|&t| ps.tiles[i_or(0, 0)].known).count(); // dummy replaced below
    let _ = known_water;
    let known_water = (0..ps.n).filter(|&t| ps.tiles[t].known && ps.tiles[t].terrain <= 2).count();
    let water_heavy = known_water * 4 >= known_tiles.max(1) || known_water >= 25;

    let my_ports: Vec<usize> = (0..ps.n)
        .filter(|&t| ps.tiles[t].known && ps.tiles[t].owner == ps.player && ps.tiles[t].structure == 8)
        .collect();
    let my_shallow_free: Vec<usize> = (0..ps.n)
        .filter(|&t| ps.tiles[t].known && ps.tiles[t].owner == ps.player && ps.tiles[t].terrain == 1 && ps.tiles[t].structure == 0 && unit_allows_work(ps, t))
        .collect();

    // Known passable exploration frontier tiles (tiles `m` that are known, passable, and adjacent to unknown tiles)
    let my_cap = mem.my_capital.unwrap_or(ps.n / 2);
    let est_opp_cap = mem.opp_capital.unwrap_or_else(|| (ps.n - 1) - my_cap);

    let mut land_frontier_tiles: Vec<(usize, i32)> = Vec::new();
    let mut water_frontier_tiles: Vec<(usize, i32)> = Vec::new();
    for m in 0..ps.n {
        if !ps.tiles[m].known { continue; }
        let ter = ps.tiles[m].terrain;
        let land_pass = ter == 3 || ter == 5 || (ter == 4 && ps.tech[20]) || ps.tiles[m].structure == 48;
        let water_pass = ter == 1 || (ter == 2 && ps.tech[13]);
        if !land_pass && !water_pass { continue; }
        let occupied_by_me = ps.unit_at[m].map(|ui| ps.units[ui].owner == ps.player).unwrap_or(false);
        if occupied_by_me { continue; }
        let sight = if ter == 4 { 2 } else { 1 };
        let mut unk = 0i32;
        for u in 0..ps.n {
            if !ps.tiles[u].known && cheb(ps.size, m, u) <= sight {
                let ux = (u % ps.size) as i32;
                let uy = (u / ps.size) as i32;
                let s = ps.size as i32;
                let edge = ux.min(s - 1 - ux).min(uy.min(s - 1 - uy));
                unk += if edge >= 1 { 2 } else { 1 };
            }
        }
        if unk > 0 {
            let d_opp = cheb(ps.size, m, est_opp_cap);
            let base = (12 - unk.min(8)).max(2) + d_opp / 2;
            if land_pass {
                land_frontier_tiles.push((m, base));
            }
            if water_pass || land_pass {
                water_frontier_tiles.push((m, base));
            }
        }
    }

    let mut need_navy_urgently = false;
    let mut cap_comp = -1i32;
    if let Some(cap) = mem.my_capital {
        cap_comp = land_comp[cap];
    } else if let Some(c) = ps.cities.iter().find(|c| c.owner == ps.player) {
        cap_comp = land_comp[c.tile];
    }
    if cap_comp >= 0 {
        let land_villages_on_cap = neutral_villages.iter().filter(|&&t| land_comp[t] == cap_comp).count();
        let enemy_cities_on_cap = opp_cities_known.iter().filter(|&&t| land_comp[t] == cap_comp).count();
        let land_frontier_on_cap = land_frontier_tiles.iter().filter(|&&(t, _)| land_comp[t] == cap_comp).count();
        if enemy_cities_on_cap == 0 && ((land_villages_on_cap == 0 && land_frontier_on_cap <= 1) || (water_heavy && land_villages_on_cap == 0 && known_water >= 30)) {
            need_navy_urgently = true;
        }
    }

    let has_ready_city_capture = ps.legal.iter().any(|a| {
        a.kind == 8 && ps.tiles[a.source as usize].structure == 1
    });

    let city_plans = build_city_plans(ps, mem);

    // Build discrete objectives for unit target assignment so multiple units fan out!
    // Each objective is (tile, base_cost, max_units_assigned)
    let mut objectives: Vec<(usize, i32, usize)> = Vec::new();
    for &t in &opp_cities_known {
        let occ = ps.unit_at[t].map(|ui| ps.units[ui].owner == ps.player).unwrap_or(false);
        objectives.push((t, 0, if occ { 2 } else { 4 }));
    }
    for &t in &neutral_villages {
        let occ = ps.unit_at[t].map(|ui| ps.units[ui].owner == ps.player).unwrap_or(false);
        if !occ {
            objectives.push((t, 1, 1)); // Assign 1 unit per uncaptured village!
        }
    }
    for &t in &ruins {
        let occ = ps.unit_at[t].map(|ui| ps.units[ui].owner == ps.player).unwrap_or(false);
        if !occ {
            objectives.push((t, 3, 1)); // Assign 1 unit per ruin!
        }
    }
    for u in &ps.units {
        if u.owner == ps.opp {
            objectives.push((u.tile, 3, 2));
        }
    }
    // Add well-spaced frontier tiles so units spread out across the frontier
    let mut sorted_frontier = land_frontier_tiles.clone();
    sorted_frontier.sort_by_key(|&(_, c)| c);
    let mut chosen_frontiers: Vec<usize> = Vec::new();
    for &(ft, cost) in &sorted_frontier {
        if chosen_frontiers.iter().all(|&other| cheb(ps.size, ft, other) >= 3) {
            chosen_frontiers.push(ft);
            objectives.push((ft, cost, 1));
        }
    }
    for &(ft, cost) in &sorted_frontier {
        if !chosen_frontiers.contains(&ft) {
            objectives.push((ft, cost + 4, 1));
        }
    }

    // Assign each movable land unit to its best distinct objective using forward_land_dist
    let mut unit_assigned_target: Vec<Option<(usize, i32)>> = vec![None; ps.units.len()];
    let mut obj_counts = vec![0usize; objectives.len()];
    let mut unit_dists: Vec<Vec<i32>> = Vec::with_capacity(ps.units.len());
    for u in &ps.units {
        if u.owner == ps.player && !u.moved && !unit_stats(u.kind).float {
            unit_dists.push(forward_land_dist(ps, u.tile));
        } else {
            unit_dists.push(Vec::new());
        }
    }

    loop {
        let mut best_pair: Option<(i32, usize, usize)> = None;
        for (ui, u) in ps.units.iter().enumerate() {
            if u.owner != ps.player || u.moved || unit_stats(u.kind).float || unit_assigned_target[ui].is_some() {
                continue;
            }
            let dmap = &unit_dists[ui];
            for (oi, &(otile, base_cost, cap)) in objectives.iter().enumerate() {
                if obj_counts[oi] >= cap { continue; }
                let d = dmap[otile];
                if d >= INF / 2 { continue; }
                let total = d + base_cost * 2;
                if best_pair.map(|(bt, _, _)| total < bt).unwrap_or(true) {
                    best_pair = Some((total, ui, oi));
                }
            }
        }
        if let Some((_, ui, oi)) = best_pair {
            unit_assigned_target[ui] = Some((objectives[oi].0, objectives[oi].1));
            obj_counts[oi] += 1;
        } else {
            break;
        }
    }

    // Global fallback distance maps (for unassigned units, amphibious routing, and boats)
    let mut naval_targets: Vec<(usize, i32)> = Vec::new();
    for &t in &opp_cities_known {
        naval_targets.push((t, 0));
    }
    for &t in &neutral_villages {
        let occ = ps.unit_at[t].map(|ui| ps.units[ui].owner == ps.player).unwrap_or(false);
        if !occ { naval_targets.push((t, 1)); }
    }
    for &t in &ruins {
        let occ = ps.unit_at[t].map(|ui| ps.units[ui].owner == ps.player).unwrap_or(false);
        if !occ { naval_targets.push((t, 3)); }
    }
    if ps.tech[14] {
        for &t in &starfish_tiles {
            naval_targets.push((t, 2));
        }
    }
    for u in &ps.units {
        if u.owner == ps.opp {
            naval_targets.push((u.tile, 3));
        }
    }
    for &(ft, cost) in &water_frontier_tiles {
        naval_targets.push((ft, cost));
    }
    if naval_targets.is_empty() && ps.tiles[est_opp_cap].known {
        naval_targets.push((est_opp_cap, 0));
    }

    let dist_land_global = backward_dijkstra(ps, &naval_targets, false, false);
    let dist_amphib = backward_dijkstra(ps, &naval_targets, true, false);
    let dist_boat = backward_dijkstra(ps, &naval_targets, true, true);

    // Count resources & open cities
    let mut own_forests = 0;
    let mut own_fish = 0;
    let mut own_fruit = 0;
    let mut own_crops = 0;
    let mut own_mines = 0;
    let mut own_lumber_huts = 0;
    let mut unconn_cities = 0;
    for c in &ps.cities {
        if c.owner == ps.player && !c.connected && Some(c.tile) != mem.my_capital {
            unconn_cities += 1;
        }
    }
    for t in 0..ps.n {
        if ps.tiles[t].known && ps.tiles[t].owner == ps.player && unit_allows_work(ps, t) {
            let st = ps.tiles[t].structure;
            let res = ps.tiles[t].resource;
            let ter = ps.tiles[t].terrain;
            if st == 0 {
                if ter == 5 { own_forests += 1; }
                if res == 3 { own_fish += 1; }
                if res == 6 { own_fruit += 1; }
                if res == 2 { own_crops += 1; }
                if res == 5 { own_mines += 1; }
            } else if st == 12 {
                own_lumber_huts += 1;
            }
        }
    }

    // Count how many of our cities can currently summon a unit (or have an unmoved unit on them)
    let open_summon_cities = ps.legal.iter().filter(|a| a.kind == 4 && a.param == 2).count();
    let my_unit_count = ps.units.iter().filter(|u| u.owner == ps.player).count();

    let mut reserve_stars = 0;
    if need_navy_urgently {
        if !ps.tech[10] {
            reserve_stars = reserve_stars.max(4 + my_cities_count);
        } else if my_ports.is_empty() && !my_shallow_free.is_empty() {
            reserve_stars = reserve_stars.max(7);
        } else if !my_ports.is_empty() && !ps.tech[13] && !ps.tech[44] {
            reserve_stars = reserve_stars.max(4 + my_cities_count * 2);
        }
    }

    let mut best_id = ps.legal[0].id.clone();
    let mut best_score = i64::MIN;

    for a in &ps.legal {
        let s: i64 = match a.kind {
            11 => -1_000_000_000,
            10 => 0,

            // Capture (kind == 8)
            8 => {
                let src = a.source as usize;
                let st = ps.tiles[src].structure;
                if st == 2 {
                    98_000 // Ruin FIRST (can give free tech or +10 stars!)
                } else if st == 0 {
                    97_000 // Starfish (+8 stars!)
                } else if ps.tiles[src].owner == ps.opp {
                    96_000 // Enemy city!
                } else {
                    80_000 // Neutral village (after pre-capture tech if any!)
                }
            }

            // Research (kind == 7): STRICT purpose-driven tech policy!
            7 => {
                let tid = a.param;
                let cost = 4 + my_cities_count * tech_tier(tid);
                let stars_after = ps.stars - cost;
                let mut val: i64 = -5_000; // Default: DO NOT buy techs unless conditions below hold!

                match tid {
                    1 => { // Riding (Rider, cost 3)
                        // Only buy Riding if we have >= 2 units already AND can afford a Rider (3) or Warrior (2) right after, OR before capturing a village!
                        if my_unit_count >= 2 && (stars_after >= 3 || (has_ready_city_capture && stars_after >= 2)) && (!need_navy_urgently || my_cities_count <= 2) {
                            val = 10_500 + if stars_after >= 3 { 2_000 } else { 0 };
                        }
                    }
                    16 => { // Forestry (Lumber Hut, cost 3)
                        // Buy if a city can level up THIS turn using Lumber Huts, or we have >= 2 forests and enough stars to build a Lumber Hut right now!
                        let can_lvl_with_lh = city_plans.iter().any(|cp| {
                            !cp.can_level_now && cp.unclaimed_forests > 0 && cp.needed_pop <= cp.unclaimed_forests && stars_after >= cp.needed_pop * 3
                        });
                        if can_lvl_with_lh {
                            val = 26_000;
                        } else if own_forests >= 2 && stars_after >= 3 && (my_unit_count >= my_cities_count as usize + 1 || open_summon_cities == 0) {
                            val = 9_500 + (own_forests as i64) * 1_000;
                        } else if city_plans.iter().any(|cp| cp.gateway_to_water && cp.unclaimed_forests > 0) && stars_after >= 3 {
                            val = 18_000;
                        }
                    }
                    10 => { // Fishing (Fish cost 2, Port cost 7)
                        let can_lvl_with_fish = city_plans.iter().any(|cp| {
                            !cp.can_level_now && cp.unclaimed_fish > 0 && cp.needed_pop <= cp.unclaimed_fish && stars_after >= cp.needed_pop * 2
                        });
                        if need_navy_urgently && (!my_shallow_free.is_empty() || own_fish > 0) {
                            val = 44_000;
                        } else if can_lvl_with_fish {
                            val = 25_000;
                        } else if water_heavy && (!my_shallow_free.is_empty() && stars_after >= 7 || own_fish >= 2 && stars_after >= 2) {
                            val = 14_000;
                        }
                    }
                    13 => { // Sailing (Ocean + Scoutship cost 5)
                        let has_raft = ps.units.iter().any(|u| u.owner == ps.player && u.kind == 43);
                        if has_raft && stars_after >= 5 {
                            val = 46_000; // Upgrade Raft to Scoutship & cross ocean immediately!
                        } else if (!my_ports.is_empty() || has_raft) && (need_navy_urgently || water_heavy) {
                            val = 36_000;
                        }
                    }
                    44 => { // Ramming (Rammership cost 5)
                        let has_raft = ps.units.iter().any(|u| u.owner == ps.player && u.kind == 43);
                        if has_raft && stars_after >= 5 && ps.tech[13] {
                            val = 45_000;
                        } else if !my_ports.is_empty() && ps.tech[13] && stars_after >= 5 {
                            val = 24_000;
                        }
                    }
                    14 => { // Navigation (Starfish +8 stars, Bomber cost 15)
                        let can_harvest_starfish = ps.units.iter().any(|u| u.owner == ps.player && !u.moved && !u.attacked && ps.tiles[u.tile].resource == 8 && ps.tiles[u.tile].structure == 0);
                        if can_harvest_starfish && cost <= 10 {
                            val = 48_000; // Immediate +8 stars refund!
                        } else if !my_ports.is_empty() && ps.tech[13] && (starfish_tiles.len() >= 2 || stars_after >= 15) {
                            val = 16_000;
                        }
                    }
                    6 => { // Organization (Fruit cost 2)
                        let can_lvl_with_fruit = city_plans.iter().any(|cp| {
                            !cp.can_level_now && cp.unclaimed_fruits > 0 && cp.needed_pop <= cp.unclaimed_fruits && stars_after >= cp.needed_pop * 2
                        });
                        if can_lvl_with_fruit {
                            val = 25_000;
                        } else if city_plans.iter().any(|cp| cp.gateway_to_water && cp.unclaimed_fruits > 0) && stars_after >= 2 {
                            val = 18_000;
                        } else if own_fruit >= 2 && stars_after >= 4 && open_summon_cities == 0 {
                            val = 8_500;
                        }
                    }
                    4 => { // Roads (Road cost 3, Bridge cost 5)
                        if unconn_cities >= 1 && stars_after >= 3 && (my_unit_count >= my_cities_count as usize + 1 || open_summon_cities == 0) {
                            val = 11_000 + (unconn_cities as i64) * 2_000;
                        } else if water_heavy && stars_after >= 5 && ps.turn >= 8 {
                            val = 7_500;
                        }
                    }
                    17 => { // Mathematics (Sawmill cost 5, Catapult cost 8)
                        if own_lumber_huts >= 2 && stars_after >= 5 {
                            val = 22_000 + (own_lumber_huts as i64) * 2_000;
                        } else if open_summon_cities >= 1 && stars_after >= 8 && ps.turn >= 10 {
                            val = 11_000;
                        }
                    }
                    18 => { // Archery (Archer cost 3, Forest defense)
                        if open_summon_cities >= 1 && stars_after >= 3 && ps.turn >= 6 && !need_navy_urgently {
                            val = 6_500;
                        }
                    }
                    20 => { // Climbing
                        let can_lvl_gateway = city_plans.iter().any(|cp| cp.gateway_to_water && !cp.can_level_now);
                        if (own_mines >= 1 && stars_after >= 10) || (can_lvl_gateway && ps.turn >= 8) {
                            val = 6_000;
                        }
                    }
                    23 => { // Mining (Mine cost 5)
                        if own_mines >= 1 && stars_after >= 5 {
                            val = 12_000 + (own_mines as i64) * 3_000;
                        }
                    }
                    24 => { // Smithery (Swordsman cost 5, Forge cost 5)
                        if open_summon_cities >= 1 && stars_after >= 5 {
                            val = 13_500;
                        }
                    }
                    8 => { // Farming (Farm cost 5)
                        if own_crops >= 1 && stars_after >= 5 && (own_crops >= 2 || city_plans.iter().any(|cp| cp.gateway_to_water)) {
                            val = 12_000 + (own_crops as i64) * 2_000;
                        }
                    }
                    7 => { // Strategy (Defender cost 3)
                        if open_summon_cities >= 1 && stars_after >= 3 && (need_navy_urgently && !my_ports.is_empty() || ps.turn >= 9) {
                            val = 7_500;
                        }
                    }
                    2 => { // Free Spirit (for Chivalry)
                        if !need_navy_urgently && ps.stars >= cost + (4 + my_cities_count * 3) + 8 {
                            val = 14_000; // Have enough stars to go straight to Chivalry + Knight!
                        }
                    }
                    3 => { // Chivalry (Knight cost 8!)
                        if open_summon_cities >= 1 && stars_after >= 8 {
                            val = 24_000;
                        }
                    }
                    5 => { // Trade (Market cost 5)
                        let has_mult = (0..ps.n).any(|t| ps.tiles[t].known && ps.tiles[t].owner == ps.player && (ps.tiles[t].structure == 13 || ps.tiles[t].structure == 6 || ps.tiles[t].structure == 22) && ps.tiles[t].structure_level >= 2);
                        if has_mult && stars_after >= 5 {
                            val = 19_000;
                        }
                    }
                    12 => { // Aquatism
                        if water_heavy && ps.units.iter().filter(|u| u.owner == ps.player && unit_stats(u.kind).float).count() >= 3 && stars_after >= 3 {
                            val = 9_000;
                        }
                    }
                    _ => {}
                }

                // If we have a village capture ready THIS turn and we actually want this tech (`val >= 8_000`), research BEFORE capturing!
                if has_ready_city_capture && val >= 8_000 {
                    val += 75_000;
                }
                val
            }

            // Harvest (kind == 5)
            5 => {
                let t = a.target as usize;
                let ctile = mem.tile_city[t];
                let plan = city_plans.iter().find(|p| p.city_tile as i32 == ctile);
                let mut val: i64 = 2_000;
                if let Some(p) = plan {
                    let city = ps.city_at[p.city_tile].map(|ci| &ps.cities[ci]);
                    let lvl = city.map(|c| c.level).unwrap_or(1);
                    if p.can_level_now {
                        val = 30_000 - (p.needed_pop as i64) * 400;
                        if lvl == 2 && p.min_cost_to_level <= 6 {
                            val = 55_000 - (p.needed_pop as i64) * 400; // L2 -> L3 gives +5 stars immediately!
                        } else if lvl == 1 {
                            val += 5_000;
                        }
                        if p.gateway_to_water { val += 8_000; }
                    } else if p.gateway_to_water {
                        val = 16_000;
                    } else if ps.stars - 2 < reserve_stars || open_summon_cities > 0 {
                        val = -200; // Train units / save for urgent port first if harvest doesn't level up city!
                    }
                }
                val
            }

            // Build (kind == 6)
            6 => {
                let t = a.target as usize;
                let p = a.param;
                let ctile = mem.tile_city[t];
                let plan = city_plans.iter().find(|pl| pl.city_tile as i32 == ctile);
                let cost = structure_cost(p);

                match p {
                    23..=29 => {
                        let has_res = ps.tiles[t].resource > 0 || ps.tiles[t].terrain == 5;
                        let mut v = 60_000 - if has_res { 2_000 } else { 0 };
                        if let Some(pl) = plan {
                            if pl.needed_pop <= 3 { v += 10_000; }
                        }
                        v
                    }
                    8 => {
                        let water_nbs = neighbors(ps.size, t).iter().filter(|&&nb| !ps.tiles[nb].known || ps.tiles[nb].terrain <= 2).count() as i64;
                        let ocean_nbs = neighbors(ps.size, t).iter().filter(|&&nb| ps.tiles[nb].known && ps.tiles[nb].terrain == 2).count() as i64;
                        let mut v = if my_ports.is_empty() && (need_navy_urgently || water_heavy) {
                            45_000
                        } else if my_ports.len() < 3 && (need_navy_urgently || water_heavy) {
                            22_000
                        } else if plan.map(|pl| pl.can_level_now && pl.needed_pop == 1).unwrap_or(false) {
                            16_000
                        } else {
                            1_500
                        };
                        v += water_nbs * 300 + ocean_nbs * 400;
                        v
                    }
                    13 | 6 | 22 => {
                        let adj_target = if p == 13 { 12 } else if p == 6 { 5 } else { 21 };
                        let mult = if p == 22 { 2 } else { 1 };
                        let pop_gain = mult * (neighbors(ps.size, t).iter().filter(|&&nb| {
                            ps.tiles[nb].known && ps.tiles[nb].owner == ps.player && ps.tiles[nb].structure == adj_target
                        }).count() as i32);
                        if pop_gain <= 0 {
                            -5_000
                        } else {
                            let mut v = 5_000 + (pop_gain as i64) * 3_500;
                            if let Some(pl) = plan {
                                let lvl = ps.city_at[pl.city_tile].map(|ci| ps.cities[ci].level).unwrap_or(1);
                                if pl.can_level_now && pop_gain >= pl.needed_pop {
                                    v = if lvl == 2 { 54_000 } else { 32_000 } + (pop_gain as i64) * 1_000;
                                } else if pl.gateway_to_water {
                                    v += 12_000;
                                }
                            }
                            if ps.stars - cost < reserve_stars && v < 25_000 {
                                v = -200;
                            }
                            v
                        }
                    }
                    50 => {
                        let stars_gain: i32 = neighbors(ps.size, t).iter().map(|&nb| {
                            if ps.tiles[nb].known && ps.tiles[nb].owner == ps.player && (ps.tiles[nb].structure == 13 || ps.tiles[nb].structure == 6 || ps.tiles[nb].structure == 22) {
                                ps.tiles[nb].structure_level.max(1)
                            } else {
                                0
                            }
                        }).sum();
                        if stars_gain >= 2 {
                            26_000 + (stars_gain as i64) * 4_000
                        } else if stars_gain == 1 {
                            6_000
                        } else {
                            -5_000
                        }
                    }
                    12 | 5 | 21 => {
                        let pop_gain = if p == 12 { 1 } else { 2 };
                        let mut v = 1_800 + (pop_gain as i64) * 1_000;
                        if let Some(pl) = plan {
                            let city = ps.city_at[pl.city_tile].map(|ci| &ps.cities[ci]);
                            let lvl = city.map(|c| c.level).unwrap_or(1);
                            if pl.can_level_now {
                                v = 29_000 - (pl.needed_pop as i64) * 400 + (pop_gain as i64) * 800;
                                if lvl == 2 && pl.min_cost_to_level <= 6 {
                                    v = 53_000 - (pl.needed_pop as i64) * 400;
                                }
                                if pl.gateway_to_water { v += 8_000; }
                            } else if pl.gateway_to_water {
                                v = 15_000;
                            } else if ps.stars - cost < reserve_stars || open_summon_cities > 0 {
                                v = -200;
                            }
                        }
                        v
                    }
                    3 | 48 => {
                        let mut v = -1_000;
                        if p == 48 {
                            v = if water_heavy || need_navy_urgently { 18_000 } else { 4_000 };
                        } else if let Some(cap) = mem.my_capital {
                            for c in &ps.cities {
                                if c.owner == ps.player && !c.connected && c.tile != cap {
                                    if cheb(ps.size, c.tile, t) == 1 && neighbors(ps.size, t).iter().any(|&nb| {
                                        nb != c.tile && ps.tiles[nb].known && (ps.tiles[nb].road || nb == cap || ps.city_at[nb].map(|ci| ps.cities[ci].owner == ps.player && ps.cities[ci].connected).unwrap_or(false))
                                    }) {
                                        v = 27_000; // Immediate 1-tile road connection (+2 total pop)!
                                        break;
                                    } else if cheb(ps.size, c.tile, t) <= 2 && cheb(ps.size, cap, t) <= 2 && cheb(ps.size, c.tile, cap) <= 3 {
                                        v = v.max(4_500);
                                    }
                                }
                            }
                        }
                        if ps.stars - cost < reserve_stars && v < 20_000 {
                            v = -500;
                        }
                        v
                    }
                    _ => -2_000,
                }
            }

            // Combat Attack / Convert / Infiltrate (kind == 2)
            2 => {
                let src = a.source as usize;
                let dst = a.target as usize;
                if let Some(ui) = ps.unit_at[src] {
                    let att = &ps.units[ui];
                    let ast = unit_stats(att.kind);
                    if ast.infiltrate {
                        72_000
                    } else if att.kind == 10 {
                        let def_cost = ps.unit_at[dst].map(|di| unit_stats(ps.units[di].kind).cost).unwrap_or(3);
                        68_000 + (def_cost as i64) * 3_000
                    } else if let Some(di) = ps.unit_at[dst] {
                        let def = &ps.units[di];
                        let dst_stats = unit_stats(def.kind);
                        let (dmg, ret, killed, self_died) = eval_attack(ps, att, src, def, dst);
                        let def_val = (dst_stats.cost.max(2) as i64) * 1_000 + if def.passenger > 0 { (unit_stats(def.passenger).cost as i64) * 800 } else { 0 };
                        let att_val = (ast.cost.max(2) as i64) * 1_000 + if att.passenger > 0 { (unit_stats(att.passenger).cost as i64) * 800 } else { 0 };
                        let on_my_city = ps.city_at[dst].map(|ci| ps.cities[ci].owner == ps.player).unwrap_or(false);
                        let on_opp_city = ps.city_at[dst].map(|ci| ps.cities[ci].owner == ps.opp).unwrap_or(false);

                        let mut v: i64 = 50_000;
                        if killed {
                            v = 65_000 + def_val;
                            if on_my_city { v += 15_000; }
                            if on_opp_city && ast.range == 1 { v += 12_000; }
                            if ast.persist { v += 15_000; }
                            v -= ((dmg - def.health).max(0.0) as i64) * 50;
                        } else if self_died {
                            v = if on_my_city && dmg >= 5.0 { 5_000 } else { -50_000 };
                        } else if ret == 0.0 {
                            // Zero-retaliation attack: shoot FIRST (70_000 > lethal 65_000) to soften targets!
                            v = 70_000 + (dmg as i64) * 800;
                        } else {
                            let other_attackers = ps.units.iter().filter(|u| {
                                u.owner == ps.player && u.tile != src && !u.attacked && cheb(ps.size, u.tile, dst) <= unit_stats(u.kind).range + if !u.moved && unit_stats(u.kind).dash { unit_stats(u.kind).movement } else { 0 }
                            }).count();
                            let dmg_score = (dmg as i64) * (def_val / 10).max(200);
                            let ret_penalty = (ret as i64) * (att_val / 10).max(200);
                            v += dmg_score - ret_penalty;
                            if on_my_city { v += 12_000; }
                            if other_attackers > 0 && dmg >= def.health * 0.4 {
                                v += 8_000;
                            }
                            if ast.escape {
                                v += 4_000;
                            }
                            if ret > dmg * 1.6 && !on_my_city && other_attackers == 0 {
                                v = -5_000;
                            }
                        }
                        v
                    } else {
                        10_000
                    }
                } else {
                    0
                }
            }

            // Move (kind == 1)
            1 => {
                let src = a.source as usize;
                let dst = a.target as usize;
                if let Some(ui) = ps.unit_at[src] {
                    let u = &ps.units[ui];
                    let ust = unit_stats(u.kind);
                    let src_is_my_city = ps.city_at[src].map(|ci| ps.cities[ci].owner == ps.player).unwrap_or(false);
                    let src_is_opp_city = ps.city_at[src].map(|ci| ps.cities[ci].owner == ps.opp).unwrap_or(false);
                    let src_is_capturable = (ps.tiles[src].structure == 1 && ps.tiles[src].owner != ps.player)
                        || ps.tiles[src].structure == 2
                        || (ps.tiles[src].resource == 8 && ps.tech[14] && ps.tiles[src].structure == 0);

                    if !u.attacked && src_is_capturable {
                        -100_000
                    } else if src_is_opp_city && !ps.units.iter().any(|e| e.owner == ps.opp && cheb(ps.size, e.tile, src) <= 1 && u.health <= 4.0) {
                        -80_000
                    } else {
                        let dst_is_opp_city = ps.city_at[dst].map(|ci| ps.cities[ci].owner == ps.opp).unwrap_or(false)
                            || (mem.seen_enemy_cities[dst] && ps.tiles[dst].owner != ps.player);
                        let dst_is_village = ps.tiles[dst].known && ps.tiles[dst].structure == 1 && ps.tiles[dst].owner == 0;
                        let dst_is_ruin = ps.tiles[dst].known && ps.tiles[dst].structure == 2;
                        let dst_is_starfish = ps.tiles[dst].known && ps.tiles[dst].resource == 8 && ps.tech[14] && ps.tiles[dst].structure == 0;
                        let dst_is_own_port = ps.tiles[dst].known && ps.tiles[dst].structure == 8 && ps.tiles[dst].owner == ps.player;
                        let src_is_own_port = ps.tiles[src].known && ps.tiles[src].structure == 8 && ps.tiles[src].owner == ps.player;

                        let mut v: i64 = 35_000;
                        let mut purposeful = false;

                        if dst_is_opp_city && !ust.infiltrate {
                            v += 28_000;
                            purposeful = true;
                        } else if dst_is_village {
                            v += 25_000;
                            purposeful = true;
                        } else if dst_is_ruin {
                            v += 20_000;
                            purposeful = true;
                        } else if dst_is_starfish {
                            v += 22_000;
                            purposeful = true;
                        }

                        // Dash attack setup
                        if ust.dash && !u.attacked && ust.attack > 0.0 && !(dst_is_own_port && !ust.float) && !(ust.carry && ps.tiles[dst].terrain >= 3) {
                            let mut best_dash_bonus = 0i64;
                            for e in &ps.units {
                                if e.owner == ps.opp && cheb(ps.size, dst, e.tile) <= ust.range {
                                    let (dmg, ret, killed, self_died) = eval_attack(ps, u, dst, e, e.tile);
                                    let on_my_city = ps.city_at[e.tile].map(|ci| ps.cities[ci].owner == ps.player).unwrap_or(false);
                                    if killed {
                                        let b = 26_000 + (unit_stats(e.kind).cost as i64) * 1_000 + if on_my_city { 10_000 } else { 0 };
                                        best_dash_bonus = best_dash_bonus.max(b);
                                    } else if !self_died && (ret == 0.0 || dmg >= ret * 0.8 || on_my_city) {
                                        let b = if ret == 0.0 { 16_000 } else { 9_000 } + (dmg as i64) * 500 + if on_my_city { 8_000 } else { 0 };
                                        best_dash_bonus = best_dash_bonus.max(b);
                                    }
                                }
                            }
                            if best_dash_bonus > 0 {
                                v += best_dash_bonus;
                                purposeful = true;
                            }
                        } else if ust.infiltrate && !u.attacked {
                            if opp_cities_known.iter().any(|&ct| cheb(ps.size, dst, ct) == 1) {
                                v += 28_000;
                                purposeful = true;
                            }
                        }

                        // Rider Escape retreat
                        if u.attacked {
                            let threats_src = ps.units.iter().filter(|e| e.owner == ps.opp && cheb(ps.size, src, e.tile) <= unit_stats(e.kind).range + unit_stats(e.kind).movement).count() as i64;
                            let threats_dst = ps.units.iter().filter(|e| e.owner == ps.opp && cheb(ps.size, dst, e.tile) <= unit_stats(e.kind).range + unit_stats(e.kind).movement).count() as i64;
                            if threats_dst < threats_src {
                                v += 25_000 + (threats_src - threats_dst) * 3_500;
                                purposeful = true;
                            }
                        }

                        // Exploration bonus
                        let sight = if ps.tiles[dst].terrain == 4 || ust.scout { 2 } else { 1 };
                        let mut new_vis = 0i64;
                        for t in 0..ps.n {
                            if !ps.tiles[t].known && cheb(ps.size, dst, t) <= sight {
                                new_vis += 1;
                            }
                        }
                        if new_vis > 0 {
                            v += new_vis * 900;
                            purposeful = true;
                        }

                        // Distance reduction toward assigned objective (or global target)
                        if ust.float {
                            let d_src = dist_boat[src];
                            let d_dst = dist_boat[dst];
                            let diff = (d_src - d_dst).clamp(-20, 20);
                            if diff > 0 { purposeful = true; }
                            v += (diff as i64) * 1_100;
                            if src_is_own_port && diff >= 0 {
                                v += 4_000;
                                purposeful = true;
                            }
                            if ps.tiles[dst].terrain >= 3 {
                                let near_target = dst_is_village || dst_is_opp_city || dst_is_ruin
                                    || neighbors(ps.size, dst).iter().any(|&nb| {
                                        (ps.tiles[nb].known && ps.tiles[nb].structure == 1 && ps.tiles[nb].owner != ps.player)
                                            || ps.city_at[nb].map(|ci| ps.cities[ci].owner == ps.opp).unwrap_or(false)
                                    });
                                if near_target {
                                    v += 12_000;
                                    purposeful = true;
                                } else if u.kind == 42 || u.kind == 41 {
                                    v -= 18_000;
                                } else if land_comp[dst] == cap_comp && cap_comp >= 0 {
                                    v -= 25_000;
                                }
                            }
                        } else if dst_is_own_port {
                            let d_land_src = dist_land_global[src];
                            if d_land_src >= INF / 2 || need_navy_urgently || water_heavy {
                                v += 14_000;
                                if ps.stars >= 5 && (ps.tech[13] || ps.tech[44]) {
                                    v += 5_000;
                                }
                                purposeful = true;
                            } else {
                                v -= 8_000;
                            }
                        } else if let Some((target_tile, _)) = unit_assigned_target[ui] {
                            // Distance toward this unit's specifically assigned target!
                            let d_target = forward_land_dist(ps, target_tile);
                            let d_src = d_target[src];
                            let d_dst = d_target[dst];
                            let diff = (d_src - d_dst).clamp(-20, 20);
                            if diff > 0 { purposeful = true; }
                            v += (diff as i64) * 1_200;
                        } else {
                            let d_src = dist_amphib[src];
                            let d_dst = dist_amphib[dst];
                            let diff = (d_src - d_dst).clamp(-20, 20);
                            if diff > 0 { purposeful = true; }
                            v += (diff as i64) * 1_000;
                        }

                        // If standing on our own city and we have stars to summon a unit, vacating the city is purposeful ONLY if we don't move strictly backward (or if no forward move exists)
                        if src_is_my_city && ps.stars >= 2 {
                            v += 600;
                            purposeful = true;
                        }

                        // Prevent pointless oscillation: if move is not purposeful, score below End Turn!
                        if !purposeful {
                            -2_000
                        } else {
                            v
                        }
                    }
                } else {
                    -10_000
                }
            }

            // Summon / Train / Upgrade Raft (kind == 4)
            4 => {
                let src = a.source as usize;
                let uk = a.param;
                let ust = unit_stats(uk);
                let is_upgrade = uk == 41 || uk == 42 || uk == 44;

                if is_upgrade {
                    let unmoved = ps.unit_at[src].map(|ui| !ps.units[ui].moved).unwrap_or(false);
                    let base: i64 = if unmoved { 62_000 } else { 32_000 }; // Upgrade unmoved Rafts BEFORE moving them!
                    let mut v = base + match uk {
                        41 => 2_000,
                        42 => 1_500,
                        44 => 1_000,
                        _ => 0,
                    };
                    let my_scouts = ps.units.iter().filter(|u| u.owner == ps.player && u.kind == 42).count();
                    let my_rams = ps.units.iter().filter(|u| u.owner == ps.player && u.kind == 44).count();
                    if uk == 44 && my_scouts > my_rams {
                        v += 1_000;
                    }
                    v
                } else {
                    let comp_id = land_comp[src];
                    let my_units_on_comp = ps.units.iter().filter(|u| u.owner == ps.player && land_comp[u.tile] == comp_id).count();
                    let ports_on_comp = my_ports.iter().filter(|&&p| {
                        neighbors(ps.size, p).iter().any(|&nb| land_comp[nb] == comp_id)
                    }).count();
                    let enemies_near_city = ps.units.iter().filter(|u| u.owner == ps.opp && cheb(ps.size, u.tile, src) <= 3).count();
                    let enemies_on_comp = ps.units.iter().filter(|u| u.owner == ps.opp && land_comp[u.tile] == comp_id).count()
                        + opp_cities_known.iter().filter(|&&ct| land_comp[ct] == comp_id).count();
                    let villages_on_comp = neutral_villages.iter().filter(|&&vt| land_comp[vt] == comp_id).count();
                    let frontier_on_comp = land_frontier_tiles.iter().filter(|&&(ft, _)| land_comp[ft] == comp_id).count();

                    let island_saturated = enemies_on_comp == 0
                        && enemies_near_city == 0
                        && my_units_on_comp >= villages_on_comp + (if frontier_on_comp > 0 { 1 } else { 0 }) + ports_on_comp * 2;

                    if island_saturated && ports_on_comp == 0 {
                        -5_000
                    } else if ps.stars - ust.cost < reserve_stars && enemies_near_city == 0 {
                        -1_000
                    } else {
                        // Summon score 12_000..15_000: AFTER moving existing units (35_000+) and completing city level-ups (30_000+), BEFORE non-urgent tech/harvests!
                        let unit_score: i64 = match uk {
                            4 => 15_500, // Knight
                            11 => 15_000, // Swordsman
                            3 => if need_navy_urgently && enemies_on_comp == 0 { 12_500 } else { 14_500 }, // Rider
                            8 => if enemies_near_city >= 2 { 11_000 } else { 14_200 }, // Catapult
                            9 => 13_800, // Archer
                            5 => if need_navy_urgently || enemies_near_city > 0 { 14_000 } else { 13_000 }, // Defender
                            2 => 13_200, // Warrior
                            38 => if opp_cities_known.is_empty() { 9_000 } else { 13_600 }, // Cloak
                            10 => 11_500, // MindBender
                            _ => 11_000,
                        };
                        let d_front = dist_amphib[src].min(30) as i64;
                        unit_score + (enemies_near_city as i64) * 2_000 - d_front * 30
                    }
                }
            }

            // Abilities (kind == 3)
            3 => {
                let src = a.source;
                let dst = a.target;
                match a.param {
                    9 => 40_000, // MindBender Heal Others
                    7 => {
                        if let Some(ui) = ps.unit_at[src as usize] {
                            let u = &ps.units[ui];
                            let missing = max_hp(u) - u.health;
                            let on_opp_city = ps.city_at[u.tile].map(|ci| ps.cities[ci].owner == ps.opp).unwrap_or(false);
                            if on_opp_city && missing >= 2.0 {
                                42_000
                            } else if missing >= 4.0 {
                                1_000
                            } else {
                                -500
                            }
                        } else {
                            -500
                        }
                    }
                    2 => {
                        let t = dst as usize;
                        let adj_lh = neighbors(ps.size, t).iter().filter(|&&nb| {
                            ps.tiles[nb].known && ps.tiles[nb].owner == ps.player && ps.tiles[nb].structure == 12
                        }).count();
                        let ctile = mem.tile_city[t];
                        let has_sawmill = (0..ps.n).any(|i| ps.tiles[i].known && ps.tiles[i].owner == ps.player && ps.tiles[i].structure == 13 && mem.tile_city[i] == ctile);
                        if ps.tech[17] && !has_sawmill && adj_lh >= 3 && ps.stars >= 4 {
                            28_000
                        } else {
                            -10_000
                        }
                    }
                    19 => if dst == 0 { 75_000 } else { -100_000 },
                    _ => -50_000,
                }
            }

            _ => -10_000,
        };

        if s > best_score || (s == best_score && a.id < best_id) {
            best_score = s;
            best_id = a.id.clone();
        }
    }

    best_id
}

#[inline]
fn i_or(a: usize, _b: usize) -> usize { a }

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Memory::new();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            _ => continue,
        };
        let obs: Value = serde_json::from_str(&line).expect("valid json");
        let ps = parse_obs(&obs, &mut mem);
        let chosen = choose_action(&ps, &mut mem);
        writeln!(stdout, "{}", json!({"action_id": chosen})).unwrap();
        stdout.flush().unwrap();
    }
}
