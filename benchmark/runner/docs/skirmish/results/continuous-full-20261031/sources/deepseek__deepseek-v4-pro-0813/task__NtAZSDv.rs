use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Tile {
    known: bool,
    terrain: i32,
    owner: i32,
    resource: i32,
    structure: i32,
    road: bool,
    capital: i32,
    structure_level: i32,
}

#[derive(Clone, Debug)]
struct Unit {
    tile: usize,
    owner: i32,
    kind: i32,
    health: f32,
    veteran: bool,
    moved: bool,
    attacked: bool,
}

#[derive(Clone, Debug)]
struct City {
    tile: usize,
    owner: i32,
    level: i32,
    population: i32,
    production: i32,
    progress: i32,
    connected: bool,
    riot: bool,
    rewards: Vec<i32>,
}

#[derive(Clone, Debug)]
struct View {
    size: usize,
    turn: i32,
    stars: i32,
    tech: Vec<i32>,
    tiles: Vec<Option<Tile>>,
    units: Vec<Unit>,
    cities: Vec<City>,
}

#[derive(Clone, Debug)]
struct MemTile {
    terrain: i32,
    owner: i32,
    resource: i32,
    structure: i32,
    last_turn: i32,
}

#[derive(Clone, Debug, Default)]
struct MapMem {
    size: usize,
    tiles: Vec<Option<MemTile>>,
    last_known_enemy: Vec<(usize, i32)>, // tile, turn
}

fn get_i(v: &Value, key: &str, default: i32) -> i32 {
    v.get(key).and_then(Value::as_i64).unwrap_or(default as i64) as i32
}
fn get_b(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}
fn get_f(v: &Value, key: &str) -> f32 {
    v.get(key).and_then(Value::as_f64).map(|x| x as f32).unwrap_or(0.0)
}

fn parse_view(obs: &Value) -> View {
    let vs = &obs["visible_state"];
    let size = get_i(vs, "board_size", 14) as usize;
    let n = size * size;
    let turn = get_i(vs, "turn", 0);
    let stars = get_i(vs, "stars", 0);
    let tech = vs["technology"]
        .as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap_or(0) as i32).collect()).unwrap_or_default();
    let mut tiles = vec![None; n];
    if let Some(arr) = vs["tiles"].as_array() {
        for t in arr {
            let idx = get_i(t, "tile", 0) as usize;
            if idx >= n { continue; }
            let known = get_b(t, "known");
            if known {
                tiles[idx] = Some(Tile {
                    known: true,
                    terrain: get_i(t, "terrain", 0),
                    owner: get_i(t, "owner", 0),
                    resource: get_i(t, "resource", 0),
                    structure: get_i(t, "structure", 0),
                    road: get_i(t, "road", 0) != 0 || get_b(t, "road"),
                    capital: get_i(t, "capital", 0),
                    structure_level: get_i(t, "structure_level", 0),
                });
            } else {
                tiles[idx] = None;
            }
        }
    }
    let mut units = Vec::new();
    if let Some(arr) = vs["units"].as_array() {
        for u in arr {
            units.push(Unit {
                tile: get_i(u, "tile", 0) as usize,
                owner: get_i(u, "owner", 0),
                kind: get_i(u, "kind", 0),
                health: get_f(u, "health"),
                veteran: get_b(u, "veteran"),
                moved: get_b(u, "moved"),
                attacked: get_b(u, "attacked"),
            });
        }
    }
    let mut cities = Vec::new();
    if let Some(arr) = vs["cities"].as_array() {
        for c in arr {
            cities.push(City {
                tile: get_i(c, "tile", 0) as usize,
                owner: get_i(c, "owner", 0),
                level: get_i(c, "level", 0),
                population: get_i(c, "population", 0),
                production: get_i(c, "production", 0),
                progress: get_i(c, "progress", 0),
                connected: get_b(c, "connected"),
                riot: get_b(c, "riot"),
                rewards: c["rewards"].as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap_or(0) as i32).collect()).unwrap_or_else(|| vec![0;9]),
            });
        }
    }
    View { size, turn, stars, tech, tiles, units, cities }
}

fn update_mem(view: &View, mem: &mut MapMem, player: i32) {
    if mem.size != view.size || mem.tiles.len() != view.tiles.len() {
        mem.size = view.size;
        mem.tiles = vec![None; view.tiles.len()];
    }
    for (i, t) in view.tiles.iter().enumerate() {
        if let Some(t) = t {
            if mem.tiles[i].is_none() || mem.tiles[i].as_ref().unwrap().last_turn <= view.turn {
                mem.tiles[i] = Some(MemTile {
                    terrain: t.terrain, owner: t.owner, resource: t.resource,
                    structure: t.structure, last_turn: view.turn,
                });
            }
        }
    }
    let mut enemies: Vec<(usize,i32)> = Vec::new();
    for u in &view.units {
        if u.owner != 0 && u.owner != player {
            enemies.push((u.tile, view.turn));
        }
    }
}

fn parse_action(a: &Value) -> (i32, i32, i32, i32, String) {
    let c = &a["command"];
    (
        get_i(c, "kind", -1),
        get_i(c, "source", -1),
        get_i(c, "target", -1),
        get_i(c, "parameter", 0),
        a["action_id"].as_str().unwrap_or("").to_string(),
    )
}

fn tile_at(view: &View, idx: usize) -> Option<&Tile> {
    if idx < view.tiles.len() { view.tiles[idx].as_ref() } else { None }
}

fn unit_at(view: &View, idx: usize) -> Option<&Unit> {
    view.units.iter().find(|u| u.tile == idx)
}

fn city_at(view: &View, idx: usize) -> Option<&City> {
    view.cities.iter().find(|c| c.tile == idx)
}

fn unit_stats(kind: i32) -> (f32, f32, f32, i32) {
    // attack, defense, maxhp, cost
    match kind {
        2 => (2.0, 2.0, 10.0, 2),
        3 => (2.0, 1.0, 10.0, 3),
        4 => (3.5, 1.0, 10.0, 8),
        5 => (1.0, 3.0, 15.0, 3),
        8 => (4.0, 0.0, 10.0, 8),
        9 => (2.0, 1.0, 10.0, 3),
        10 => (0.0, 1.0, 10.0, 5),
        11 => (3.0, 3.0, 15.0, 5),
        12 => (5.0, 4.0, 40.0, 10),
        15 => (3.0, 1.0, 15.0, 2),
        37 => (2.0, 2.0, 10.0, 2),
        38 => (0.0, 0.5, 5.0, 8),
        39 => (0.0, 0.5, 5.0, 2),
        40 => (2.0, 2.0, 10.0, 2),
        41 => (3.0, 2.0, 10.0, 15),
        42 => (2.0, 1.0, 10.0, 5),
        43 => (0.0, 1.0, 10.0, 0),
        44 => (3.0, 3.0, 10.0, 5),
        45 => (4.0, 4.0, 40.0, 10),
        _ => (1.0, 1.0, 10.0, 3),
    }
}

fn unit_range(kind: i32) -> i32 {
    match kind { 8 => 3, 9 => 2, 30 => 2, 32 => 3, 41 => 3, 42 => 2, 43 => 2, _ => 1 }
}

fn is_ranged(kind: i32) -> bool { unit_range(kind) > 1 }

fn cheb(size: usize, a: usize, b: usize) -> i32 {
    let (ax, ay) = ((a % size) as i32, (a / size) as i32);
    let (bx, by) = ((b % size) as i32, (b / size) as i32);
    (ax - bx).abs().max((ay - by).abs())
}

fn defense_bonus(view: &View, idx: usize) -> f32 {
    if let Some(t) = tile_at(view, idx) {
        if let Some(u) = unit_at(view, idx) {
            if let Some(c) = city_at(view, idx) {
                if c.owner == u.owner {
                    // fortified city; approximate city wall if reward 1
                    let wall = c.rewards.get(1).copied().unwrap_or(0) > 0;
                    return if wall { 4.0 } else { 1.5 };
                }
            }
        }
        match t.terrain {
            4 => return 1.5,
            5 => return 1.5,
            1 | 2 => return 1.0,
            _ => {}
        }
    }
    1.0
}

fn combat_damage(att_kind: i32, att_health: f32, att_max: f32, def_kind: i32, def_health: f32, def_max: f32, db: f32) -> (f32, f32) {
    let aa = unit_stats(att_kind).0;
    let dd = unit_stats(def_kind).1;
    let af = aa * (att_health / att_max.max(1.0));
    let df = dd * (def_health / def_max.max(1.0)) * db;
    let total = af + df;
    if total <= 0.0 { return (0.0, 0.0); }
    let attack = ((af / total) * aa * 4.5 + 0.5).floor();
    let defense = if def_health > attack { ((df / total) * dd * 4.5 + 0.5).floor() } else { 0.0 };
    (attack, defense)
}

fn can_retaliate(_att_kind: i32, def_kind: i32, size: usize, source: usize, target: usize) -> bool {
    // Surprise attackers and stiff defenders are approximated; assume retaliate if adjacent/range and defender not stiff.
    !matches!(def_kind, 8 | 38 | 39 | 10 | 43) && cheb(size, source, target) <= unit_range(def_kind).max(1)
}

fn attack_score(view: &View, a_kind: i32, source: i32, target: i32, player: i32) -> Option<i64> {
    if a_kind != 2 { return None; }
    let s = source as usize;
    let t = target as usize;
    let attacker = unit_at(view, s)?;
    if attacker.owner != player { return None; }
    let defender = unit_at(view, t)?;
    if defender.owner == player || defender.owner == 0 { return None; }
    let (aa, _, am, acost) = unit_stats(attacker.kind);
    let (_, dd, dm, dcost) = unit_stats(defender.kind);
    let _ = (aa, dd);
    let db = defense_bonus(view, t);
    let (dmg, ret) = combat_damage(attacker.kind, attacker.health, am, defender.kind, defender.health, dm, db);
    let kill = dmg >= defender.health;
    let att_dies = ret >= attacker.health && can_retaliate(attacker.kind, defender.kind, view.size, s, t);
    let enemy_value = dcost.max(1);
    let my_value = acost.max(1);
    let mut score: i64 = 0;
    if kill {
        score += 9000 + (enemy_value * 25) as i64;
        if att_dies {
            score -= (my_value * 30) as i64;
        } else {
            score += 500;
        }
        // prefer killing a unit sitting on a village/city
        if let Some(tile) = tile_at(view, t) {
            if tile.structure == 1 && tile.owner != player { score += 1500; }
        }
        score += (dmg as i64) * 20;
    } else {
        let favorable = dmg >= ret * 1.1;
        if favorable {
            score += 6000 + (dmg as i64) * 15 - (ret as i64) * 12;
            // don't suicide against much stronger healthy unit
            if !att_dies && attacker.health - ret > 0.0 {
                score += 300;
            }
        } else if is_ranged(attacker.kind) && ret == 0.0 {
            // free ranged chip
            score += 4500 + (dmg as i64) * 12;
        }
    }
    if score > 0 { Some(score) } else { None }
}

fn harvest_score(view: &View, a_kind: i32, target: i32, player: i32) -> Option<i64> {
    if a_kind != 5 { return None; }
    let t = target as usize;
    let tile = tile_at(view, t)?;
    if tile.owner != player { return None; }
    if view.stars < 2 { return None; }
    // Harvesting gives +1 pop to the owning city; very strong early.
    let resource_value = match tile.resource {
        1 => 850, // animal (Hunting already)
        6 => 900, // fruit (Organization) -- but legal only if tech
        3 => 750, // fish
        _ => 500,
    };
    let has_low_city = view.cities.iter().any(|c| c.owner == player && c.level < 4);
    let low_bonus = if has_low_city && view.stars >= 5 { 1000 } else { 0 };
    Some(8000 + resource_value + 200 + low_bonus)
}

fn water_ratio(view: &View) -> f32 {
    let mut known = 0;
    let mut water = 0;
    for t in view.tiles.iter().flatten() {
        known += 1;
        if t.terrain == 1 || t.terrain == 2 { water += 1; }
    }
    if known == 0 { 0.0 } else { water as f32 / known as f32 }
}

fn tech_priority(tech: i32, water_ratio: f32) -> i64 {
    let water = water_ratio > 0.10;
    match tech {
        6 => 105, // organization
        1 => 105, // riding
        10 => if water { 115 } else { 70 }, // fishing
        13 => if water { 110 } else { 45 }, // sailing
        14 => if water { 105 } else { 40 }, // navigation
        44 => if water { 85 } else { 30 },  // ramming
        12 => if water { 70 } else { 30 },  // aquatism
        20 => 95,  // climbing
        23 => 90,  // mining
        16 => 90,  // forestry
        18 => 90,  // archery
        24 => 95,  // smithery
        4 => 80,   // roads
        8 => 82,   // farming
        7 => 75,   // strategy
        2 => 70,   // free spirit
        3 => 60,   // chivalry
        17 => 60,  // mathematics
        5 => 45,   // trade
        9 => 45,   // construction
        21 => 40,
        22 => 40,
        19 => 40,
        _ => 20,
    }
}

fn research_score(view: &View, a_kind: i32, param: i32, _player: i32) -> Option<i64> {
    if a_kind != 7 { return None; }
    let wr = water_ratio(view);
    let p = tech_priority(param, wr);
    // Only research techs we actually want.
    let tech_count = view.tech.len() as i32;
    let allowed = match p {
        105..=i64::MAX => true,
        90..=104 => view.stars >= 8 || view.turn < 25,
        70..=89 => view.stars >= 11 && tech_count <= 10,
        50..=69 => view.stars >= 14 && tech_count <= 9,
        _ => false,
    };
    if !allowed { return None; }
    let mut score = 5200 + p * 35;
    if view.turn <= 8 { score += 300; }
    if view.turn > 22 && p < 90 { score -= 1500; }
    if view.stars >= 12 { score += (view.stars as i64 - 8) * 20; }
    Some(score)
}

fn best_reward_action(view: &View, actions: &[Value]) -> usize {
    // choose best reward by slot. legal reward kinds have parameter.
    let mut best = 0usize;
    let mut best_score: i64 = -1000000;
    for (i,a) in actions.iter().enumerate() {
        let c = &a["command"];
        if get_i(c,"kind",-1) != 9 { continue; }
        let p = get_i(c,"parameter",0);
        let city_idx = get_i(c,"target",-1) as usize;
        let mut sc = match p {
            3 => 1200, // workshop
            4 => 900,  // explorer
            1 => 800,  // wall
            7 => 1100, // resources +5 stars
            5 => 1000, // border growth
            8 => 950,  // population +3
            2 => 700,  // park
            6 => 1300, // giant
            _ => 500,
        };
        if p == 1 {
            if let Some(city) = city_at(view, city_idx) {
                let enemy_adjacent = view.units.iter().any(|u| u.owner != 0 && u.owner != city.owner && cheb(view.size, u.tile, city.tile) <= 1);
                if enemy_adjacent { sc += 500; }
            }
        }
        if p == 5 {
            // border growth stronger when frontier and early/mid
            if view.turn < 20 { sc += 300; }
        }
        if p == 6 {
            // giant stronger when at war
            sc += 200;
        }
        if sc > best_score { best_score = sc; best = i; }
    }
    best
}

fn capture_score(view: &View, a_kind: i32, source: i32, player: i32) -> Option<i64> {
    if a_kind != 8 { return None; }
    let s = source as usize;
    let unit = unit_at(view, s)?;
    if unit.owner != player { return None; }
    let tile = tile_at(view, s)?;
    // village or city: extremely valuable
    if tile.structure == 1 && tile.owner != player {
        let enemy_city = tile.owner != 0;
        return Some(50000 + if enemy_city { 20000 } else { 0 });
    }
    if tile.structure == 2 { return Some(30000); } // ruin
    if tile.resource == 8 { return Some(15000); } // starfish
    Some(10000)
}

fn move_score(view: &View, a_kind: i32, source: i32, target: i32, player: i32, mem: &MapMem) -> Option<i64> {
    if a_kind != 1 { return None; }
    let s = source as usize;
    let t = target as usize;
    let unit = unit_at(view, s)?;
    if unit.owner != player { return None; }
    let tile = tile_at(view, t)?;
    let mut score: i64 = 5000;
    // Capturing or reaching a village/ruin.
    if tile.structure == 1 && tile.owner != player {
        score += 30000;
    } else if tile.structure == 2 {
        score += 22000;
    } else if tile.resource == 8 {
        score += 8000;
    }
    // Explore unseen.
    let radius = if tile.terrain == 4 || unit.kind == 42 || unit.kind == 38 || unit.kind == 39 { 2 } else { 1 };
    let mut unknown = 0;
    let mut village_unknown = 0;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let nx = (t % view.size) as i32 + dx;
            let ny = (t / view.size) as i32 + dy;
            if nx < 0 || ny < 0 || nx >= view.size as i32 || ny >= view.size as i32 { continue; }
            let idx = (ny as usize) * view.size + nx as usize;
            match tile_at(view, idx) {
                None => unknown += 1,
                Some(tt) => {
                    if tt.structure == 1 && tt.owner != player { village_unknown += 1; }
                }
            }
        }
    }
    score += unknown as i64 * 18;
    score += village_unknown as i64 * 500;
    // Move off city so it can train (essential early).
    if let Some(city) = city_at(view, s) {
        if city.owner == player && view.turn <= 8 {
            score += 16000;
        } else if city.owner == player {
            score += 2000;
        }
    }
    // Prefer moving toward remembered villages/ruins if any.
    if let Some(mt) = mem.tiles.get(t).and_then(|x| x.as_ref()) {
        if mt.structure == 1 && mt.owner != player { score += 1000; }
        if mt.structure == 2 { score += 800; }
    }
    // Avoid standing still in place if legal and city can train? Still good sometimes.
    if s == t { score += 100; }
    Some(score)
}

fn summon_score(view: &View, a_kind: i32, source: i32, param: i32, player: i32) -> Option<i64> {
    if a_kind != 4 { return None; }
    let city = city_at(view, source as usize)?;
    if city.owner != player { return None; }
    let (atk, def, hp, cost) = unit_stats(param);
    let _ = (atk, def, hp);
    let mut score = 6000 + cost as i64 * 150;
    let wr = water_ratio(view);
    let enemy_near = view.units.iter().any(|u| u.owner != 0 && u.owner != player && cheb(view.size, u.tile, city.tile) <= 3);
    if enemy_near {
        score += 3500;
        // defenders and ranged units near combat
        if matches!(param, 5 | 9 | 11 | 8) { score += 1500; }
    } else {
        // peacetime expansion: riders and warriors
        if matches!(param, 3 | 2) { score += 1500; }
    }
    if matches!(param, 42 | 44 | 41) {
        score += if wr > 0.10 { 3200 } else { 400 };
    }
    if view.stars - cost < 2 && view.turn < 4 { score -= 1500; }
    // Strong early rush: build Warriors quickly while expanding.
    if view.turn <= 8 && param == 2 { score += 9000; }
    else if view.turn <= 14 && param == 3 { score += 5000; }
    Some(score)
}

fn build_score(view: &View, a_kind: i32, target: i32, param: i32, player: i32) -> Option<i64> {
    if a_kind != 6 { return None; }
    let t = target as usize;
    let tile = tile_at(view, t)?;
    let wr = water_ratio(view);
    let ports = view.tiles.iter().flatten().filter(|x| x.structure == 8 && x.owner == player).count() as i64;
    let mut score = 3000;
    let has_low_city = view.cities.iter().any(|c| c.owner == player && c.level < 4);
    let grow_bonus = if has_low_city && view.stars >= 10 && matches!(param, 5 | 21 | 12) { 3000 } else { 0 };
    match param {
        8 => { // port
            if tile.owner == player && (tile.terrain == 1 || tile.terrain == 2) {
                let port_bonus = if ports < 2 { 2500 } else if ports < 4 { 1200 } else { -2500 };
                score += 2000 + port_bonus + if wr > 0.10 { 800 } else { 0 };
            } else { return None; }
        }
        5 => { // farm
            if tile.owner == player && tile.resource == 6 { score += 2800 + grow_bonus; }
            else { score += 200; }
        }
        21 => { // mine
            if tile.owner == player && tile.resource == 5 { score += 3200 + grow_bonus; }
            else if tile.owner == player && tile.terrain == 4 { score += 700; }
            else { score += 100; }
        }
        12 => { // lumber hut
            if tile.owner == player && tile.terrain == 5 { score += 2000 + grow_bonus; }
            else { score += 100; }
        }
        3 => { // road
            score += 500;
        }
        48 => { // bridge
            score += 500;
        }
        50 => { score += 800; }
        6 => { score += 1000; } // windmill
        13 => { score += 1000; } // sawmill
        22 => { score += 1200; } // forge
        19 | 20 | 18 => { score += 300; }
        _ => { score += 50; }
    }
    Some(score)
}

fn ability_score(view: &View, a_kind: i32, source: i32, _target: i32, param: i32, player: i32) -> Option<i64> {
    if a_kind != 3 { return None; }
    let s = source as usize;
    let unit = unit_at(view, s)?;
    if unit.owner != player { return None; }
    match param {
        16 => return Some(9000), // promote
        7 => {
            if unit.health < 10.0 { return Some(6000); }
            return None;
        }
        9 => return Some(6500), // heal others
        1 | 2 | 3 => return Some(3000 + param as i64 * 200),
        8 => return Some(1000), // disband only rarely; never preferred
        4 => return Some(3000), // destroy
        18..=22 => return Some(0), // diplomacy ignored
        _ => return None,
    }
}

fn end_turn_score(view: &View) -> i64 {
    // We almost always want to use units; only end turn after no useful actions.
    -1000 + view.stars as i64 * 2
}

fn choose_action_index(_obs: &Value, view: &View, mem: &mut MapMem, actions: &[Value], player: i32) -> usize {
    // 1) Mandatory city reward.
    if actions.iter().any(|a| get_i(&a["command"], "kind", -1) == 9) {
        return best_reward_action(view, actions);
    }
    let mut best_idx = 0usize;
    let mut best_score: i64 = i64::MIN;
    for (i, a) in actions.iter().enumerate() {
        let (kind, source, target, param, _id) = parse_action(a);
        let score = match kind {
            8 => capture_score(view, kind, source, player).unwrap_or(i64::MIN + 1),
            2 => attack_score(view, kind, source, target, player).unwrap_or(i64::MIN + 1),
            5 => harvest_score(view, kind, target, player).unwrap_or(i64::MIN + 1),
            7 => research_score(view, kind, param, player).unwrap_or(i64::MIN + 1),
            4 => summon_score(view, kind, source, param, player).unwrap_or(i64::MIN + 1),
            6 => build_score(view, kind, target, param, player).unwrap_or(i64::MIN + 1),
            1 => move_score(view, kind, source, target, player, mem).unwrap_or(i64::MIN + 1),
            3 => ability_score(view, kind, source, target, param, player).unwrap_or(i64::MIN + 1),
            10 => end_turn_score(view),
            _ => i64::MIN + 1,
        };
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = MapMem::default();
    for line in stdin.lock().lines() {
        let obs: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let view = parse_view(&obs);
        let player = get_i(&obs, "player", 0);
        update_mem(&view, &mut mem, player);
        let legal = obs["legal_actions"].as_array().expect("legal_actions");
        if legal.is_empty() {
            continue;
        }
        let idx = choose_action_index(&obs, &view, &mut mem, legal, player);
        let id = legal[idx]["action_id"].as_str().unwrap_or("");
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
