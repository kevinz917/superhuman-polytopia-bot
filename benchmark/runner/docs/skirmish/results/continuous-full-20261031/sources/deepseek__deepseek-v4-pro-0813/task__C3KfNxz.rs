use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct Action {
    id: String,
    kind: i32,
    parameter: i32,
    source: i64,
    target: i64,
}

#[derive(Clone, Debug, Default)]
struct Tile {
    known: bool,
    terrain: i32,
    owner: i32,
    resource: i32,
    structure: i32,
}

#[derive(Clone, Debug)]
struct Unit {
    tile: usize,
    owner: i32,
    kind: i32,
    health: f32,
    max_health: f32,
    moved: bool,
    attacked: bool,
}

#[derive(Clone, Debug)]
struct CityInfo {
    tile: usize,
    owner: i32,
    level: i32,
    population: i32,
    progress: i32,
    production: i32,
    wall: bool,
    rewards_count: i32,
}


thread_local! {
    static MEM: RefCell<Mem> = RefCell::new(Mem { heading: HashMap::new(), last_move: HashMap::new() });
}

struct Mem {
    heading: HashMap<usize, (i32, i32)>,
    last_move: HashMap<usize, usize>,
}

fn vi(v: &Value, key: &str) -> i64 {
    v.get(key).and_then(|x| x.as_i64()).unwrap_or(0)
}
fn vf(v: &Value, key: &str) -> f32 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32
}
fn vb(v: &Value, key: &str) -> bool {
    v.get(key).and_then(|x| x.as_bool()).unwrap_or(false)
}

fn board_size(obs: &Value) -> i32 {
    vi(&obs["visible_state"], "board_size") as i32
}

fn get_actions(obs: &Value) -> Vec<Action> {
    let mut out = Vec::new();
    if let Some(arr) = obs.get("legal_actions").and_then(|x| x.as_array()) {
        for a in arr {
            let c = &a["command"];
            out.push(Action {
                id: a["action_id"].as_str().unwrap_or("").to_string(),
                kind: vi(c, "kind") as i32,
                parameter: vi(c, "parameter") as i32,
                source: vi(c, "source"),
                target: vi(c, "target"),
            });
        }
    }
    out
}

fn parse_tiles(obs: &Value) -> Vec<Tile> {
    let n = board_size(obs) as usize;
    let mut tiles = vec![Tile::default(); n * n];
    if let Some(arr) = obs["visible_state"].get("tiles").and_then(|x| x.as_array()) {
        for t in arr {
            let idx = vi(t, "tile") as usize;
            if idx < tiles.len() {
                let known = vb(t, "known");
                tiles[idx] = Tile {
                    known,
                    terrain: if known { vi(t, "terrain") as i32 } else { -1 },
                    owner: if known { vi(t, "owner") as i32 } else { 0 },
                    resource: if known { vi(t, "resource") as i32 } else { 0 },
                    structure: if known { vi(t, "structure") as i32 } else { 0 },
                };
            }
        }
    }
    tiles
}

fn parse_units(obs: &Value) -> Vec<Unit> {
    let mut out = Vec::new();
    if let Some(arr) = obs["visible_state"].get("units").and_then(|x| x.as_array()) {
        for u in arr {
            let kind = vi(u, "kind") as i32;
            let health = vf(u, "health");
            let max_health = unit_max_health(kind, vb(u, "veteran"));
            out.push(Unit {
                tile: vi(u, "tile") as usize,
                owner: vi(u, "owner") as i32,
                kind,
                health,
                max_health,
                moved: vb(u, "moved"),
                attacked: vb(u, "attacked"),
            });
        }
    }
    out
}

fn parse_cities(obs: &Value) -> Vec<CityInfo> {
    let mut out = Vec::new();
    if let Some(arr) = obs["visible_state"].get("cities").and_then(|x| x.as_array()) {
        for c in arr {
            let rewards = c.get("rewards").and_then(|x| x.as_array());
            let rewards_count = rewards.map(|r| r.iter().filter_map(|x| x.as_i64()).sum::<i64>()).unwrap_or(0) as i32;
            let wall = rewards.map(|r| r.first().and_then(|x| x.as_i64()).unwrap_or(0) > 0).unwrap_or(false);
            out.push(CityInfo {
                tile: vi(c, "tile") as usize,
                owner: vi(c, "owner") as i32,
                level: vi(c, "level") as i32,
                population: vi(c, "population") as i32,
                progress: vi(c, "progress") as i32,
                production: vi(c, "production") as i32,
                wall,
                rewards_count,
            });
        }
    }
    out
}

fn own_techs(obs: &Value) -> Vec<i32> {
    let mut out = Vec::new();
    if let Some(arr) = obs["visible_state"].get("technology").and_then(|x| x.as_array()) {
        for t in arr {
            out.push(t.as_i64().unwrap_or(0) as i32);
        }
    }
    out
}

fn unit_attack(kind: i32) -> f32 {
    match kind {
        2 => 2.0, 3 => 2.0, 4 => 3.5, 5 => 1.0, 8 => 4.0, 9 => 2.0,
        10 => 0.0, 11 => 3.0, 12 => 5.0, 15 => 3.0, 37 => 2.0, 40 => 2.0,
        42 => 2.0, 43 => 0.0, 44 => 3.0, _ => 1.0,
    }
}

fn unit_defense(kind: i32) -> f32 {
    match kind {
        2 => 2.0, 3 => 1.0, 4 => 1.0, 5 => 3.0, 8 => 0.0, 9 => 1.0,
        10 => 1.0, 11 => 3.0, 12 => 4.0, 15 => 1.0, 37 => 2.0, 40 => 2.0,
        42 => 1.0, 43 => 1.0, 44 => 3.0, _ => 1.0,
    }
}

fn unit_max_health(kind: i32, veteran: bool) -> f32 {
    let base = match kind {
        2 | 3 | 4 | 8 | 9 | 10 | 37 | 40 | 42 | 43 | 44 => 10.0,
        5 | 11 | 15 | 18 | 23 | 31 | 59 => 15.0,
        12 | 26 | 45 => 40.0,
        19 | 24 | 29 | 33 => 20.0,
        27 => 30.0,
        _ => 10.0,
    };
    if veteran { base + 5.0 } else { base }
}

fn has_fortify(kind: i32) -> bool {
    matches!(kind, 2 | 3 | 4 | 5 | 9 | 15 | 17)
}

fn is_water(t: i32) -> bool { t == 1 || t == 2 }

fn cheb(a: usize, b: usize, size: i32) -> i32 {
    let (ax, ay) = ((a % size as usize) as i32, (a / size as usize) as i32);
    let (bx, by) = ((b % size as usize) as i32, (b / size as usize) as i32);
    (ax - bx).abs().max((ay - by).abs())
}

fn find_unit_at<'a>(units: &'a [Unit], tile: usize) -> Option<&'a Unit> {
    units.iter().find(|u| u.tile == tile)
}

fn nearest_enemy_dist(units: &[Unit], player: i32, tile: usize, size: i32) -> i32 {
    units.iter().filter(|u| u.owner != player)
        .map(|u| cheb(u.tile, tile, size))
        .min()
        .unwrap_or(999)
}

fn nearest_own_city_dist(cities: &[CityInfo], player: i32, tile: usize, size: i32) -> i32 {
    cities.iter().filter(|c| c.owner == player)
        .map(|c| cheb(c.tile, tile, size))
        .min()
        .unwrap_or(999)
}

fn city_at<'a>(cities: &'a [CityInfo], tile: usize) -> Option<&'a CityInfo> {
    cities.iter().find(|c| c.tile == tile)
}

fn own_units_count(units: &[Unit], player: i32) -> i32 {
    units.iter().filter(|u| u.owner == player).count() as i32
}

fn defense_bonus(tiles: &[Tile], cities: &[CityInfo], tile: usize, kind: i32) -> f32 {
    if tile >= tiles.len() { return 1.0; }
    let t = &tiles[tile];
    if !t.known { return 1.0; }
    if t.terrain == 4 { return 1.5; }
    if t.terrain == 5 { return 1.25; }
    if t.structure == 1 && has_fortify(kind) {
        if let Some(c) = city_at(cities, tile) {
            if c.wall { return 4.0; }
            return 1.5;
        }
    }
    1.0
}

fn combat_damage(att: f32, ahp: f32, amax: f32, def: f32, dhp: f32, dmax: f32, bonus: f32) -> (f32, f32) {
    let af = att * (ahp / amax).max(0.0);
    let df = def * (dhp / dmax).max(0.0) * bonus;
    let denom = af + df;
    if denom <= 0.0 { return (0.0, 0.0); }
    let dmg = ((af / denom) * att * 4.5).round();
    let ret = ((df / denom) * def * 4.5).round();
    (dmg, ret)
}

fn passable_land(tiles: &[Tile], idx: usize) -> bool {
    if idx >= tiles.len() { return false; }
    let t = &tiles[idx];
    if !t.known { return true; }
    !is_water(t.terrain) && t.terrain != 4
}

fn bfs_objectives(tiles: &[Tile], player: i32, size: i32) -> Vec<i32> {
    let n = tiles.len();
    let mut dist = vec![-1i32; n];
    let mut q = std::collections::VecDeque::new();
    for i in 0..n {
        if !tiles[i].known { continue; }
        if (tiles[i].structure == 1 && tiles[i].owner != player) || tiles[i].structure == 2 {
            dist[i] = 0;
            q.push_back(i);
        }
    }
    while let Some(x) = q.pop_front() {
        let d = dist[x];
        let sx = (x % size as usize) as i32;
        let sy = (x / size as usize) as i32;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                let nx = sx + dx;
                let ny = sy + dy;
                if nx < 0 || ny < 0 || nx >= size || ny >= size { continue; }
                let ni = (ny * size + nx) as usize;
                if ni < n && dist[ni] == -1 && passable_land(tiles, ni) {
                    dist[ni] = d + 1;
                    q.push_back(ni);
                }
            }
        }
    }
    dist
}

fn bfs_frontier(tiles: &[Tile], size: i32) -> Vec<i32> {
    let n = tiles.len();
    let mut dist = vec![-1i32; n];
    let mut q = std::collections::VecDeque::new();
    for i in 0..n {
        if tiles[i].known && unknown_neighbors(tiles, i, size) > 0 {
            dist[i] = 0;
            q.push_back(i);
        }
    }
    while let Some(x) = q.pop_front() {
        let d = dist[x];
        let sx = (x % size as usize) as i32;
        let sy = (x / size as usize) as i32;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                let nx = sx + dx;
                let ny = sy + dy;
                if nx < 0 || ny < 0 || nx >= size || ny >= size { continue; }
                let ni = (ny * size + nx) as usize;
                if ni < n && dist[ni] == -1 && passable_land(tiles, ni) {
                    dist[ni] = d + 1;
                    q.push_back(ni);
                }
            }
        }
    }
    dist
}

fn unknown_neighbors(tiles: &[Tile], idx: usize, size: i32) -> i32 {
    let n = tiles.len();
    let sx = (idx % size as usize) as i32;
    let sy = (idx / size as usize) as i32;
    let mut c = 0;
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 { continue; }
            let nx = sx + dx;
            let ny = sy + dy;
            if nx < 0 || ny < 0 || nx >= size || ny >= size { continue; }
            let ni = (ny * size + nx) as usize;
            if ni < n && !tiles[ni].known { c += 1; }
        }
    }
    c
}

fn threat_level(units: &[Unit], cities: &[CityInfo], player: i32, size: i32) -> i32 {
    let mut max_danger = 0;
    for c in cities.iter().filter(|c| c.owner == player) {
        for u in units.iter().filter(|u| u.owner != player) {
            let d = cheb(u.tile, c.tile, size);
            if d <= 1 { max_danger = max_danger.max(3); }
            else if d <= 2 { max_danger = max_danger.max(2); }
            else if d <= 3 { max_danger = max_danger.max(1); }
        }
    }
    // Also danger if an enemy is near any own unit and close to our general position.
    max_danger
}

fn choose_reward(actions: &[Action], cities: &[CityInfo], player: i32) -> Option<String> {
    let mut best: Option<String> = None;
    let mut best_score = -1e9f32;
    let level = cities.iter().find(|c| c.owner == player).map(|c| c.level).unwrap_or(1);
    for a in actions.iter().filter(|a| a.kind == 9) {
        let score = match a.parameter {
            3 => if level <= 2 { 200.0 } else { 50.0 },
            4 => if level <= 2 { 130.0 } else { 30.0 },
            1 => 90.0,
            7 => 160.0,
            5 => 120.0,
            8 => 140.0,
            6 => 180.0,
            2 => 70.0,
            _ => 0.0,
        };
        if score > best_score {
            best_score = score;
            best = Some(a.id.clone());
        }
    }
    best
}

fn research_priority(tech: i32, obs: &Value, tiles: &[Tile]) -> f32 {
    let own = own_techs(obs);
    if own.contains(&tech) { return -200.0; }
    let water_tiles = tiles.iter().filter(|t| t.known && is_water(t.terrain)).count();
    match tech {
        1 => 92.0,
        6 => 82.0,
        10 => if water_tiles > 3 { 95.0 } else { 45.0 },
        20 => 62.0,
        18 => 78.0,
        8 => 72.0,
        4 => 88.0,
        23 => 66.0,
        16 => 54.0,
        24 => 85.0,
        17 => 74.0,
        3 => 80.0,
        2 => 64.0,
        13 => if water_tiles > 3 { 82.0 } else { 50.0 },
        44 => if water_tiles > 3 { 76.0 } else { 40.0 },
        14 => if water_tiles > 3 { 80.0 } else { 45.0 },
        38 => 40.0,
        21 => 30.0,
        22 => 38.0,
        19 => 30.0,
        9 => 48.0,
        5 => 44.0,
        _ => 10.0,
    }
}

fn build_priority(structure: i32) -> f32 {
    match structure {
        3 => 42.0,
        5 => 80.0,
        8 => 76.0,
        12 => 66.0,
        13 => 60.0,
        21 => 86.0,
        22 => 72.0,
        6 => 56.0,
        48 => 50.0,
        50 => 62.0,
        _ => 30.0,
    }
}

fn summon_priority(kind: i32, obs: &Value, player: i32, danger: i32) -> f32 {
    let stars = vi(&obs["visible_state"], "stars") as i32;
    let mut base = match kind {
        3 => 84.0,
        11 => 88.0,
        9 => 74.0,
        2 => if danger > 0 { 70.0 } else { 56.0 },
        5 => if danger > 0 { 62.0 } else { 48.0 },
        8 => 62.0,
        4 => 80.0,
        42 => 68.0,
        44 => 70.0,
        _ => 30.0,
    };
    if danger > 0 {
        base += 18.0;
        // cheap defenders are valuable in an emergency
        if matches!(kind, 2 | 3 | 11) { base += 14.0; }
    }
    if stars < 6 && !matches!(kind, 2 | 3) { base -= 16.0; }
    let own_units = obs["visible_state"].get("units").and_then(|x| x.as_array())
        .map(|arr| arr.iter().filter(|u| vi(u, "owner") == player as i64).count()).unwrap_or(0) as i32;
    if kind == 2 && own_units >= 4 { base -= 18.0; }
    if kind == 3 && own_units >= 6 { base -= 10.0; }
    base
}

fn ability_priority(ability: i32, units: &[Unit], player: i32) -> f32 {
    match ability {
        16 => 95.0,
        7 => {
            if units.iter().any(|u| u.owner == player && u.health < u.max_health - 1.0) { 62.0 } else { 0.0 }
        }
        9 => 42.0,
        1 | 2 | 3 => 20.0,
        4 => 10.0,
        8 => -60.0,
        _ => 5.0,
    }
}

fn attack_score(a: &Action, obs: &Value, tiles: &[Tile], cities: &[CityInfo], units: &[Unit], player: i32, size: i32) -> f32 {
    let src = a.source as usize;
    let dst = a.target as usize;
    if src >= tiles.len() || dst >= tiles.len() { return -2000.0; }
    let att = match find_unit_at(units, src) { Some(u) => u, None => return -2000.0 };
    let def = match find_unit_at(units, dst) { Some(u) => u, None => return -2000.0 };
    if att.owner != player || def.owner == player { return -2000.0; }
    let bonus = defense_bonus(tiles, cities, dst, def.kind);
    let (dmg, ret) = combat_damage(
        unit_attack(att.kind), att.health, att.max_health,
        unit_defense(def.kind), def.health, def.max_health, bonus,
    );
    let kill = dmg >= def.health;
    let we_die = ret >= att.health;
    let nearby_enemies = units.iter().filter(|u| u.owner != player && u.tile != dst)
        .filter(|u| cheb(u.tile, dst, size) <= 2).count();

    let mut score = 300.0 + dmg * 18.0 - ret * 24.0;
    if kill { score += 650.0; }
    if kill && !we_die { score += 500.0; }
    if we_die { score -= 900.0; }
    if kill {
        let city_dist = nearest_own_city_dist(cities, player, dst, size);
        if city_dist <= 2 { score += 250.0; }
    }
    if kill && matches!(att.kind, 3 | 4) && !we_die { score += 150.0; }

    if !kill && nearby_enemies >= 2 { score -= 180.0; }
    if !kill && att.health < def.health { score -= 90.0; }
    score
}

fn mem_heading(src: usize) -> Option<(i32, i32)> {
    MEM.with(|m| m.borrow().heading.get(&src).copied())
}

fn mem_last(src: usize) -> Option<usize> {
    MEM.with(|m| m.borrow().last_move.get(&src).copied())
}

fn remember_move(src: usize, dst: usize, size: i32) {
    MEM.with(|m| {
        let mut mem = m.borrow_mut();
        mem.last_move.insert(src, dst);
        let (sx, sy) = ((src % size as usize) as i32, (src / size as usize) as i32);
        let (dx, dy) = ((dst % size as usize) as i32, (dst / size as usize) as i32);
        let dir = ((dx - sx).signum(), (dy - sy).signum());
        mem.heading.remove(&src);
        mem.heading.insert(dst, dir);
    });
}

fn move_score(a: &Action, obs: &Value, tiles: &[Tile], cities: &[CityInfo], units: &[Unit], player: i32, size: i32, danger: i32, obj_dist: &[i32], frontier_dist: &[i32]) -> f32 {
    let src = a.source as usize;
    let target = a.target as usize;
    let att = match find_unit_at(units, src) { Some(u) => u, None => return -2000.0 };
    if att.owner != player { return -2000.0; }
    if target >= tiles.len() { return -500.0; }

    let mut score = 520.0;
    let t = &tiles[target];

    // Persistent direction and anti-backtracking.
    if let Some((hx, hy)) = mem_heading(src) {
        let (sx, sy) = ((src % size as usize) as i32, (src / size as usize) as i32);
        let (tx, ty) = ((target % size as usize) as i32, (target / size as usize) as i32);
        let dir = ((tx - sx).signum(), (ty - sy).signum());
        if dir == (hx, hy) { score += 130.0; }
        else if dir == (-hx, -hy) { score -= 220.0; }
    }
    if mem_last(src) == Some(target) {
        score -= 320.0;
    }

    let target_city = city_at(cities, target);
    let src_city = city_at(cities, src);

    // Keep a garrison in each owned city after the first scout leaves.
    if src_city.map(|c| c.owner == player).unwrap_or(false)
        && target_city.map(|c| c.owner == player).unwrap_or(false) == false
        && own_units_count(units, player) >= 2
        && danger <= 1
    {
        let city_tile = src_city.unwrap().tile;
        let nearby_own = units.iter().filter(|u| u.owner == player && u.tile != src)
            .filter(|u| cheb(u.tile, city_tile, size) <= 1).count();
        if nearby_own == 0 {
            score -= 420.0;
        }
    }

    if danger > 0 {
        if target_city.map(|c| c.owner == player).unwrap_or(false) {
            if !find_unit_at(units, target).map(|u| u.owner == player).unwrap_or(false) {
                score += 1100.0;
            }
        }
        if src_city.map(|c| c.owner == player).unwrap_or(false) && target_city.map(|c| c.owner == player).unwrap_or(false) == false {
            // Leaving a threatened city is usually bad.
            score -= 700.0;
        }
        // When danger is high, withdraw toward the nearest owned city.
        if danger >= 2 {
            let before = nearest_own_city_dist(cities, player, src, size);
            let after = nearest_own_city_dist(cities, player, target, size);
            if after < before { score += 750.0; }
            else if after > before { score -= 550.0; }
        }
        // Fall back when hurt.
        if att.health < att.max_health * 0.45 {
            let enemy_before = nearest_enemy_dist(units, player, src, size);
            let enemy_after = nearest_enemy_dist(units, player, target, size);
            score += (enemy_after - enemy_before) as f32 * 55.0;
        }
    }

    let explore_scale: f32 = 1.0;

    // Exploration and objectives.
    if t.known {
        if t.structure == 1 && t.owner != player {
            score += 4200.0 * explore_scale;
        } else if t.structure == 2 {
            score += 2800.0 * explore_scale;
        }
        if t.resource != 0 && t.owner == player as i32 { score += 14.0; }
        if find_unit_at(units, target).map(|u| u.owner != player).unwrap_or(false) {
            score -= 4000.0;
        }
    } else {
        score += 130.0;
    }
    score += unknown_neighbors(tiles, target, size) as f32 * 18.0 * explore_scale;
    if !t.known { score += 40.0 * explore_scale; }

    if target < obj_dist.len() && obj_dist[target] >= 0 {
        score += 260.0 * explore_scale / (1.0 + obj_dist[target] as f32);
    }
    if target < frontier_dist.len() && frontier_dist[target] >= 0 {
        let d = frontier_dist[target] as f32;
        score += 230.0 * explore_scale / (1.0 + d);
        if d == 0.0 { score += 40.0 * explore_scale; }
    }

    // If no immediate threat, move toward the nearest visible enemy when stronger.
    if danger <= 1 {
        if let Some(enemy) = units.iter().filter(|u| u.owner != player).min_by_key(|u| cheb(u.tile, target, size)) {
            let d = cheb(enemy.tile, target, size) as f32;
            if att.health >= att.max_health * 0.7 {
                score += 18.0 / (1.0 + d);
            } else {
                score -= 25.0 / (1.0 + d);
            }
        }
    }
    score
}

fn action_score(a: &Action, obs: &Value, tiles: &[Tile], cities: &[CityInfo], units: &[Unit], player: i32, size: i32, danger: i32, obj_dist: &[i32], frontier_dist: &[i32]) -> f32 {
    match a.kind {
        9 => 1_000_000.0 + 10.0 * a.parameter as f32,
        8 => 950_000.0 + a.parameter as f32,
        2 => attack_score(a, obs, tiles, cities, units, player, size),
        7 => {
            let mut score = 1500.0 + research_priority(a.parameter, obs, tiles);
            if danger >= 2 { score -= 1200.0; }
            else if danger == 1 { score -= 500.0; }
            score
        }
        5 => {
            let mut score = 600.0;
            let target = a.target as usize;
            if target < tiles.len() && tiles[target].resource != 0 { score += 45.0; }
            for c in cities.iter().filter(|c| c.owner == player) {
                if target < tiles.len() && tiles[target].owner == player as i32 {
                    if c.progress + 1 >= c.level + 1 { score += 350.0; }
                }
            }
            if danger >= 2 { score -= 700.0; }
            score
        }
        6 => {
            let mut score = 380.0 + build_priority(a.parameter);
            let stars = vi(&obs["visible_state"], "stars") as i32;
            if danger >= 2 { score -= 1000.0; }
            else if danger == 1 { score -= 400.0; }
            if stars > 12 && danger == 0 { score += 120.0; }
            if a.parameter == 3 && stars < 8 { score -= 180.0; } // roads are a luxury early
            score
        }
        4 => {
            let mut score = 620.0 + summon_priority(a.parameter, obs, player, danger);
            if danger >= 2 { score += 200.0; }
            score
        }
        3 => 350.0 + ability_priority(a.parameter, units, player),
        1 => move_score(a, obs, tiles, cities, units, player, size, danger, obj_dist, frontier_dist),
        10 => -120.0,
        11 => -9000.0,
        _ => -600.0,
    }
}

fn choose(obs: &Value) -> String {
    let player = vi(obs, "player") as i32;
    let actions = get_actions(obs);
    if actions.is_empty() { return String::new(); }
    let size = board_size(obs);
    let tiles = parse_tiles(obs);
    let units = parse_units(obs);
    let cities = parse_cities(obs);

    if let Some(id) = choose_reward(&actions, &cities, player) {
        return id;
    }

    let danger = threat_level(&units, &cities, player, size);
    let obj_dist = bfs_objectives(&tiles, player, size);
    let frontier_dist = bfs_frontier(&tiles, size);

    let mut best_id = actions[0].id.clone();
    let mut best_score = -1e12f32;
    for a in &actions {
        let score = action_score(a, obs, &tiles, &cities, &units, player, size, danger, &obj_dist, &frontier_dist);
        if score > best_score {
            best_score = score;
            best_id = a.id.clone();
        }
    }

    if best_score < -1000.0 {
        if let Some(e) = actions.iter().find(|a| a.kind == 10) {
            return e.id.clone();
        }
    }

    if let Some(sel) = actions.iter().find(|a| a.id == best_id) {
        if sel.kind == 1 && sel.source >= 0 && sel.target >= 0 {
            remember_move(sel.source as usize, sel.target as usize, size);
        }
    }

    best_id
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let obs: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let id = choose(&obs);
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
