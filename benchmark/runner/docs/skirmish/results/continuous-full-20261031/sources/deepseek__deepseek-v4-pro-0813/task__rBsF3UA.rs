use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{self, BufRead, Write};

const TERRAIN_WATER1: i64 = 1;
const TERRAIN_WATER2: i64 = 2;
const TERRAIN_FIELD: i64 = 3;
const TERRAIN_MOUNTAIN: i64 = 4;
const TERRAIN_FOREST: i64 = 5;

#[derive(Clone, Debug, Default)]
struct Tile {
    known: bool,
    terrain: i64,
    owner: i64,
    structure: i64,
    resource: i64,
    road: bool,
    capital: i64,
    structure_level: i64,
}

#[derive(Clone, Debug)]
struct CityInfo {
    tile: i64,
    owner: i64,
    level: i64,
    population: i64,
    production: i64,
    progress: i64,
    connected: bool,
    riot: bool,
    rewards: Vec<i64>,
}

#[derive(Clone, Debug)]
struct UnitInfo {
    tile: i64,
    owner: i64,
    kind: i64,
    health: f64,
    moved: bool,
    attacked: bool,
}

#[derive(Clone, Debug)]
struct Action {
    id: String,
    kind: i64,
    source: i64,
    target: i64,
    parameter: i64,
}

#[derive(Clone, Debug)]
struct State {
    player: i64,
    size: usize,
    n: usize,
    stars: i64,
    turn: i64,
    tech: HashSet<i64>,
    tiles: Vec<Tile>,
    cities: Vec<CityInfo>,
    units: Vec<UnitInfo>,
    actions: Vec<Action>,
    my_capital: i64,
}

struct Memory {
    size: usize,
    visited: Vec<u32>,
    known: Vec<bool>,
}

impl Memory {
    fn new(size: usize) -> Self {
        Memory { size, visited: vec![0; size * size], known: vec![false; size * size] }
    }
    fn reset(&mut self, size: usize) {
        if self.size != size || self.visited.len() != size * size {
            self.size = size;
            self.visited = vec![0; size * size];
            self.known = vec![false; size * size];
        }
    }
    fn observe(&mut self, st: &State) {
        self.reset(st.size);
        for i in 0..st.n {
            if st.tiles[i].known { self.known[i] = true; }
        }
        for u in &st.units {
            if u.owner == st.player && u.tile >= 0 && (u.tile as usize) < st.n {
                let idx = u.tile as usize;
                self.visited[idx] = self.visited[idx].saturating_add(1);
            }
        }
    }
}

fn as_i64(v: &Value, key: &str, default: i64) -> i64 {
    v.get(key).and_then(|x| x.as_i64()).unwrap_or(default)
}
fn as_bool(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
}
fn as_f64(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
}

fn parse_state(obs: &Value) -> State {
    let vs = &obs["visible_state"];
    let player = as_i64(obs, "player", 1);
    let size = as_i64(vs, "board_size", 14) as usize;
    let n = size * size;
    let mut tiles = vec![Tile::default(); n];
    if let Some(arr) = vs["tiles"].as_array() {
        for v in arr {
            let t = as_i64(v, "tile", -1);
            if t < 0 || t as usize >= n { continue; }
            let i = t as usize;
            let known = as_bool(v, "known", false);
            tiles[i].known = known;
            if known {
                tiles[i].terrain = as_i64(v, "terrain", 0);
                tiles[i].owner = as_i64(v, "owner", 0);
                tiles[i].structure = as_i64(v, "structure", 0);
                tiles[i].resource = as_i64(v, "resource", 0);
                tiles[i].road = as_bool(v, "road", false);
                tiles[i].capital = as_i64(v, "capital", 0);
                tiles[i].structure_level = as_i64(v, "structure_level", 0);
            }
        }
    }
    let mut cities = Vec::new();
    if let Some(arr) = vs["cities"].as_array() {
        for v in arr {
            let mut c = CityInfo {
                tile: as_i64(v, "tile", -1),
                owner: as_i64(v, "owner", 0),
                level: as_i64(v, "level", 1),
                population: as_i64(v, "population", 0),
                production: as_i64(v, "production", 1),
                progress: as_i64(v, "progress", 0),
                connected: as_bool(v, "connected", false),
                riot: as_bool(v, "riot", false),
                rewards: Vec::new(),
            };
            if let Some(r) = v["rewards"].as_array() {
                c.rewards = r.iter().map(|x| x.as_i64().unwrap_or(0)).collect();
            }
            cities.push(c);
        }
    }
    let mut units = Vec::new();
    if let Some(arr) = vs["units"].as_array() {
        for v in arr {
            units.push(UnitInfo {
                tile: as_i64(v, "tile", -1),
                owner: as_i64(v, "owner", 0),
                kind: as_i64(v, "kind", 0),
                health: as_f64(v, "health", 10.0),
                moved: as_bool(v, "moved", false),
                attacked: as_bool(v, "attacked", false),
            });
        }
    }
    let mut actions = Vec::new();
    if let Some(arr) = obs["legal_actions"].as_array() {
        for a in arr {
            let cmd = &a["command"];
            actions.push(Action {
                id: a["action_id"].as_str().unwrap_or("").to_string(),
                kind: as_i64(cmd, "kind", 0),
                source: as_i64(cmd, "source", -1),
                target: as_i64(cmd, "target", -1),
                parameter: as_i64(cmd, "parameter", 0),
            });
        }
    }
    let my_capital = cities.iter().find(|c| c.owner == player).map(|c| c.tile).unwrap_or(-1);
    State { player, size, n, stars: as_i64(vs, "stars", 0), turn: as_i64(vs, "turn", 0),
            tech: vs["technology"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default(),
            tiles, cities, units, actions, my_capital }
}

fn cheb(s: usize, a: i64, b: i64) -> i64 {
    let (ax, ay) = (a % s as i64, a / s as i64);
    let (bx, by) = (b % s as i64, b / s as i64);
    (ax - bx).abs().max((ay - by).abs())
}

fn tile_at(st: &State, t: i64) -> &Tile {
    &st.tiles[t.clamp(0, st.n as i64 - 1) as usize]
}
fn city_at(st: &State, t: i64) -> Option<&CityInfo> {
    st.cities.iter().find(|c| c.tile == t)
}
fn unit_at(st: &State, t: i64) -> Option<&UnitInfo> {
    st.units.iter().find(|u| u.tile == t)
}
fn own_units(st: &State) -> Vec<&UnitInfo> {
    st.units.iter().filter(|u| u.owner == st.player && u.tile >= 0).collect()
}
fn own_cities(st: &State) -> Vec<&CityInfo> {
    st.cities.iter().filter(|c| c.owner == st.player).collect()
}
fn enemy_cities(st: &State) -> Vec<&CityInfo> {
    st.cities.iter().filter(|c| c.owner != 0 && c.owner != st.player).collect()
}
fn enemy_units(st: &State) -> Vec<&UnitInfo> {
    st.units.iter().filter(|u| u.owner != 0 && u.owner != st.player && u.tile >= 0).collect()
}
fn is_water(t: i64) -> bool { t == TERRAIN_WATER1 || t == TERRAIN_WATER2 }

fn count_unknown_around(st: &State, t: i64, radius: i64) -> i64 {
    let s = st.size as i64;
    let (cx, cy) = (t % s, t / s);
    let mut cnt = 0;
    for y in (cy - radius).max(0)..=(cy + radius).min(s - 1) {
        for x in (cx - radius).max(0)..=(cx + radius).min(s - 1) {
            let idx = y * s + x;
            if idx >= 0 && (idx as usize) < st.n && !st.tiles[idx as usize].known { cnt += 1; }
        }
    }
    cnt
}

fn is_frontier(st: &State, t: i64) -> bool {
    if t < 0 || (t as usize) >= st.n || !st.tiles[t as usize].known { return false; }
    let s = st.size as i64;
    let (cx, cy) = (t % s, t / s);
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 { continue; }
            let (nx, ny) = (cx + dx, cy + dy);
            if nx >= 0 && nx < s && ny >= 0 && ny < s {
                let idx = (ny * s + nx) as usize;
                if !st.tiles[idx].known { return true; }
            }
        }
    }
    false
}

fn unit_stats(kind: i64) -> (f64, f64, f64, i64, i64) {
    match kind {
        2 => (2.0, 2.0, 10.0, 1, 1),
        3 => (2.0, 1.0, 10.0, 1, 2),
        4 => (3.5, 1.0, 10.0, 1, 3),
        5 => (1.0, 3.0, 15.0, 1, 1),
        8 => (4.0, 0.0, 10.0, 3, 1),
        9 => (2.0, 1.0, 10.0, 2, 1),
        10 => (0.0, 1.0, 10.0, 1, 1),
        11 => (3.0, 3.0, 15.0, 1, 1),
        12 => (5.0, 4.0, 40.0, 1, 1),
        38 => (0.0, 0.5, 5.0, 1, 2),
        39 => (0.0, 0.5, 5.0, 1, 2),
        40 => (2.0, 2.0, 10.0, 1, 2),
        41 => (3.0, 2.0, 10.0, 3, 2),
        42 => (2.0, 1.0, 10.0, 2, 3),
        43 => (0.0, 1.0, 10.0, 2, 2),
        44 => (3.0, 3.0, 10.0, 1, 3),
        45 => (4.0, 4.0, 40.0, 1, 2),
        _ => (1.0, 1.0, 10.0, 1, 1),
    }
}

fn unit_cost(kind: i64) -> i64 {
    match kind {
        2 => 2, 3 => 3, 4 => 8, 5 => 3, 8 => 8, 9 => 3, 10 => 5, 11 => 5, 12 => 10,
        38 => 8, 39 => 2, 40 => 2, 41 => 15, 42 => 5, 43 => 0, 44 => 5, 45 => 10,
        _ => 3,
    }
}

fn tech_cost(st: &State, tech: i64) -> i64 {
    let tier = match tech {
        1 | 6 | 10 | 20 | 15 => 1,
        2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2,
        3 | 5 | 9 | 14 | 17 | 22 | 24 | 38 => 3,
        _ => 2,
    };
    let cities = own_cities(st).len() as i64;
    let mut cost = 4 + cities * tier;
    if st.tech.contains(&22) {
        cost = ((cost as f64) * 0.66666f64).ceil() as i64;
    }
    cost
}

fn reward_score(_st: &State, a: &Action) -> i64 {
    match a.parameter {
        3 => 100, 4 => 60, 7 => 85, 1 => 75, 5 => 95, 8 => 80, 6 => 100, 2 => 50,
        _ => 0,
    }
}

fn capture_score(st: &State, a: &Action) -> i64 {
    let t = tile_at(st, a.source);
    let mut sc = 0;
    if t.structure == 1 {
        if t.owner != st.player && t.owner != 0 { sc += 2500; } else { sc += 1200; }
        if st.my_capital >= 0 { sc += 50 - cheb(st.size, a.source, st.my_capital); }
    } else if t.structure == 2 { sc += 800; }
    else if t.resource == 8 { sc += 400; }
    sc
}

fn combat_damage(atk: f64, a_hp: f64, a_max: f64, def: f64, d_hp: f64, d_max: f64, def_bonus: f64) -> f64 {
    let af = atk * (a_hp / a_max.max(1.0));
    let df = def * (d_hp / d_max.max(1.0)) * def_bonus;
    let total = af + df;
    if total <= 0.0 { return 0.0; }
    ((af / total) * atk * 4.5 + 0.5).floor()
}

fn attack_score(st: &State, a: &Action) -> i64 {
    let src = a.source; let dst = a.target;
    let attacker = match unit_at(st, src) { Some(u) if u.owner == st.player => u, _ => return -10000 };
    let defender = match unit_at(st, dst) { Some(u) if u.owner != 0 && u.owner != st.player => u, _ => return -10000 };
    let (atk, def, hp, _range, _mv) = unit_stats(attacker.kind);
    let (d_atk, d_def, d_hp, d_range, _) = unit_stats(defender.kind);
    let t = tile_at(st, dst);
    let mut bonus = 1.0;
    if t.terrain == TERRAIN_MOUNTAIN && st.tech.contains(&20) { bonus = 1.5; }
    else if t.terrain == TERRAIN_FOREST && st.tech.contains(&18) { bonus = 1.5; }
    else if is_water(t.terrain) && st.tech.contains(&12) { bonus = 1.5; }
    if city_at(st, dst).is_some() { bonus *= 1.5; }
    let dmg = combat_damage(atk, attacker.health, hp, def, defender.health, d_hp, bonus);
    let kills = dmg >= defender.health;
    let dist = cheb(st.size, src, dst);
    let can_retal = dist <= d_range;
    let retal = if can_retal { combat_damage(d_atk, defender.health, d_hp, d_def, attacker.health, hp, 1.0) } else { 0.0 };
    let attacker_dies = !kills && retal >= attacker.health;
    let mut sc = 0;
    if kills {
        sc += 1000 - (dmg as i64) * 5;
        if attacker_dies { sc -= 800; }
    } else {
        let dr = if can_retal { dmg - retal } else { dmg };
        sc += (dr * 30.0) as i64 - 100;
        if attacker_dies { sc -= 500; }
    }
    sc += (100 - defender.health as i64) * 2;
    if can_retal { sc -= 25; }
    if city_at(st, dst).map(|c| c.owner == st.player).unwrap_or(false) { sc += 200; }
    sc
}

fn target_tiles(st: &State) -> Vec<i64> {
    let ec = enemy_cities(st);
    if !ec.is_empty() { return ec.iter().map(|c| c.tile).collect(); }
    let mut villages = Vec::new(); let mut ruins = Vec::new();
    for i in 0..st.n {
        let t = &st.tiles[i];
        if !t.known { continue; }
        if t.structure == 1 && t.owner == 0 { villages.push(i as i64); }
        if t.structure == 2 { ruins.push(i as i64); }
    }
    if !villages.is_empty() || !ruins.is_empty() {
        villages.extend(ruins);
        return villages;
    }
    let mut frontier = Vec::new();
    for i in 0..st.n {
        if is_frontier(st, i as i64) { frontier.push(i as i64); }
    }
    if !frontier.is_empty() { return frontier; }
    vec![(st.n / 2) as i64]
}

fn min_cheb_to(st: &State, from: i64, targets: &[i64]) -> i64 {
    targets.iter().map(|&t| cheb(st.size, from, t)).min().unwrap_or(i64::MAX)
}

fn move_score(st: &State, mem: &Memory, a: &Action) -> i64 {
    let src = a.source; let dst = a.target;
    if dst < 0 || (dst as usize) >= st.n { return -10000; }
    let t = tile_at(st, dst);
    let mut sc = 60i64;
    let vis = mem.visited.get(dst as usize).copied().unwrap_or(0).min(10) as i64;
    sc -= vis * 14;
    let unk = count_unknown_around(st, dst, 1);
    sc += unk * 9;
    if is_frontier(st, dst) { sc += 8; }
    // objective pull
    let targets = target_tiles(st);
    let ds = min_cheb_to(st, src, &targets);
    let dd = min_cheb_to(st, dst, &targets);
    if ds < i64::MAX/2 && dd < i64::MAX/2 { sc += 24 * (ds - dd); }
    // center pull: most capitals are off-center; go toward the middle then far side.
    let size = st.size as i64;
    let center = size / 2;
    let cdist = |tile: i64| -> i64 {
        let (x, y) = (tile % size, tile / size);
        (x - center).abs().max((y - center).abs())
    };
    sc += 18 * (cdist(src) - cdist(dst));
    // opposite-corner estimate: exploring away from our capital is usually right.
    if st.my_capital >= 0 {
        let cx = st.my_capital % size;
        let cy = st.my_capital / size;
        let opp = (size - 1 - cx) + (size - 1 - cy) * size;
        sc += 8 * (cheb(st.size, src, opp) - cheb(st.size, dst, opp));
    }
    if t.known {
        if t.structure == 1 {
            if t.owner != st.player && t.owner != 0 { sc += 2200; }
            else if t.owner == 0 { sc += 1100; }
        }
        if t.structure == 2 { sc += 850; }
        if t.resource == 8 { sc += 350; }
        if t.terrain == TERRAIN_MOUNTAIN { sc -= 4; }
    }
    sc
}

fn harvest_score(st: &State, a: &Action) -> i64 {
    let t = tile_at(st, a.target);
    if !t.known { return 0; }
    let mut sc = 40;
    match t.resource {
        1 => sc += 40, 3 => sc += 35, 6 => sc += 45, _ => sc -= 50,
    }
    // estimate nearest own city progress
    let mut best_d = i64::MAX; let mut best_city: Option<&CityInfo> = None;
    for c in own_cities(st) {
        let d = cheb(st.size, a.target, c.tile);
        if d < best_d { best_d = d; best_city = Some(c); }
    }
    if let Some(c) = best_city {
        sc += c.level * 8;
        if c.progress + 1 >= c.level + 1 { sc += 80; }
    }
    sc
}

fn build_score(st: &State, a: &Action) -> i64 {
    let t = tile_at(st, a.target);
    if !t.known { return -1000; }
    let mut sc = 10;
    match a.parameter {
        5 => { if t.resource == 2 { sc += 120; } else { sc += 35; } if t.terrain == TERRAIN_FIELD { sc += 15; } }
        12 => { if t.terrain == TERRAIN_FOREST { sc += 100; } else { sc += 15; } }
        21 => { if t.terrain == TERRAIN_MOUNTAIN { sc += 90; } else { sc += 15; } }
        8 => { if is_water(t.terrain) { sc += 120; } else { sc += 20; } }
        3 => { sc += 30; }
        6 | 13 | 22 => { sc += 60; }
        50 => { sc += 35; }
        48 => { sc += 25; }
        _ => { sc += 10; }
    }
    if let Some(c) = city_at(st, a.target) { sc += c.level * 5; }
    sc
}

fn research_score(st: &State, a: &Action) -> i64 {
    let tech = a.parameter;
    let mut sc = 0;
    let water = (0..st.n).filter(|&i| st.tiles[i].known && is_water(st.tiles[i].terrain)).count() as i64;
    let forest = (0..st.n).filter(|&i| st.tiles[i].known && st.tiles[i].terrain == TERRAIN_FOREST).count() as i64;
    let mtn = (0..st.n).filter(|&i| st.tiles[i].known && st.tiles[i].terrain == TERRAIN_MOUNTAIN).count() as i64;
    let has_fruit = (0..st.n).any(|i| st.tiles[i].known && st.tiles[i].resource == 6);
    let has_crop = (0..st.n).any(|i| st.tiles[i].known && st.tiles[i].resource == 2);
    let has_metal = (0..st.n).any(|i| st.tiles[i].known && st.tiles[i].resource == 5);
    let has_enemy = !enemy_units(st).is_empty();
    let city_count = own_cities(st).len() as i64;
    match tech {
        6 => { sc += 80; if has_fruit { sc += 30; } }
        1 => { sc += 68; if city_count >= 3 { sc += 10; } }
        10 => { sc += if water >= 3 { 95 } else { 5 }; }
        20 => { sc += 25; if mtn >= 2 || has_metal { sc += 55; } }
        16 => { sc += 45; if forest >= 3 { sc += 35; } }
        18 => { sc += 40; if has_enemy { sc += 35; } }
        8 => { sc += 55; if has_crop { sc += 40; } }
        23 => { sc += 30; if mtn >= 2 || has_metal { sc += 45; } }
        24 => { sc += 40; if has_enemy { sc += 50; } }
        13 => { sc += if water >= 3 { 85 } else { 10 }; }
        14 => { sc += if water >= 3 { 70 } else { 15 }; }
        44 => { sc += if water >= 3 { 60 } else { 10 }; }
        17 => { sc += 45; if forest >= 3 { sc += 30; } }
        4 => { sc += 30; if city_count >= 3 { sc += 45; } }
        2 => { sc += 20; }
        3 => { if st.tech.contains(&2) { sc += 75; } else { sc += 25; } }
        7 => { sc += 20; }
        9 => { sc += 40; }
        21 => { sc += 25; }
        22 => { sc += 20; }
        5 => { sc += 30; }
        _ => { sc += 10; }
    }
    if matches!(tech, 1 | 6 | 10 | 20) && city_count <= 2 { sc += 15; }
    sc
}

fn summon_score(st: &State, a: &Action) -> i64 {
    let unit = a.parameter;
    let army = own_units(st).len() as i64;
    let has_enemy = !enemy_units(st).is_empty();
    let mut sc = 30;
    match unit {
        2 => { sc += 60; if army < 3 { sc += 40; } }
        3 => { sc += 90; if army < 5 { sc += 20; } }
        5 => { sc += 10; }
        9 => { sc += 80; }
        11 => { sc += if has_enemy && st.tech.contains(&24) { 130 } else { 85 }; }
        4 => { sc += 110; }
        12 => { sc += 120; }
        42 | 44 => { sc += 75; }
        41 => { sc += 70; }
        43 => { sc += 25; }
        _ => { sc += 30; }
    }
    if army >= 9 { sc -= 90; } else if army >= 6 { sc -= 30; }
    if st.stars < unit_cost(unit) + 2 { sc -= 25; }
    sc
}

fn best_action(st: &State) -> Option<(&Action, i64, i64)> {
    let mut best: Option<(&Action, i64, i64)> = None;
    for a in &st.actions {
        let mut cand: Option<(i64, i64)> = None;
        match a.kind {
            9 => cand = Some((0, reward_score(st, a))),
            8 => cand = Some((1, capture_score(st, a))),
            2 => {
                let sc = attack_score(st, a);
                if sc > 0 { cand = Some((2, sc)); }
            }
            3 if a.parameter == 16 => cand = Some((3, 100)),
            5 => {
                if st.stars >= 5 {
                    let sc = harvest_score(st, a);
                    if sc > 40 { cand = Some((5, sc)); }
                }
            }
            4 => {
                let sc = summon_score(st, a);
                if st.stars >= unit_cost(a.parameter) && sc > 40 { cand = Some((7, sc)); }
            }
            7 => {
                let cost = tech_cost(st, a.parameter);
                let sc = research_score(st, a);
                let reserve_ok = st.stars >= cost + 2 || (a.parameter == 6 && st.stars >= cost);
                if reserve_ok && sc > 30 { cand = Some((8, sc)); }
            }
            6 => {
                if st.stars >= 8 {
                    let sc = build_score(st, a);
                    if sc > 60 { cand = Some((9, sc)); }
                }
            }
            3 if a.parameter == 7 => {
                if let Some(u) = unit_at(st, a.source) {
                    if u.health < 7.0 { cand = Some((10, (u.health * 10.0) as i64)); }
                }
            }
            3 if a.parameter == 9 => cand = Some((10, 50)),
            10 => cand = Some((100, 0)),
            _ => {}
        }
        if let Some((prio, score)) = cand {
            if let Some((_, bp, bs)) = best {
                if (prio, score) < (bp, bs) { best = Some((a, prio, score)); }
            } else {
                best = Some((a, prio, score));
            }
        }
    }
    best
}

fn choose_action(st: &State, mem: &Memory) -> String {
    // Early priorities (reward/capture/attack/promote/harvest/summon/research/build).
    if let Some((a, prio, _)) = best_action(st) {
        if prio < 10 {
            return a.id.clone();
        }
    }
    // Normally move before recover/end if there is a useful move.
    let moves: Vec<&Action> = st.actions.iter().filter(|a| a.kind == 1).collect();
    if !moves.is_empty() {
        let best = moves.iter().max_by_key(|a| (move_score(st, mem, a), a.id.clone())).unwrap();
        if move_score(st, mem, best) > 20 {
            return best.id.clone();
        }
    }
    if let Some((a, _, _)) = best_action(st) {
        return a.id.clone();
    }
    st.actions.first().map(|a| a.id.clone()).unwrap_or_default()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Memory::new(14);
    for line in stdin.lock().lines() {
        let obs: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let st = parse_state(&obs);
        mem.observe(&st);
        let action_id = choose_action(&st, &mem);
        writeln!(stdout, "{}", json!({"action_id": action_id})).unwrap();
        stdout.flush().unwrap();
    }
}
