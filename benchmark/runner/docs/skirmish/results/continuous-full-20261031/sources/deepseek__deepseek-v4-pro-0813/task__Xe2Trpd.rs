use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

#[derive(Clone, Debug, Default)]
struct Tile {
    known: bool,
    terrain: i32,
    owner: i32,
    resource: i32,
    structure: i32,
    road: i32,
    capital: i32,
}

#[derive(Clone, Debug)]
struct Action {
    id: String,
    kind: i32,
    source: i32,
    target: i32,
    parameter: i32,
}

#[derive(Clone, Debug)]
struct Unit {
    tile: usize,
    owner: i32,
    kind: i32,
    health: f32,
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
}

fn i32v(v: &Value, key: &str, default: i32) -> i32 {
    v.get(key).and_then(Value::as_i64).map(|x| x as i32).unwrap_or(default)
}

fn f32v(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key).and_then(Value::as_f64).map(|x| x as f32).unwrap_or(default)
}

fn bv(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn parse_action(a: &Value) -> Action {
    let cmd = &a["command"];
    Action {
        id: a["action_id"].as_str().unwrap_or("").to_string(),
        kind: i32v(cmd, "kind", -1),
        source: i32v(cmd, "source", -1),
        target: i32v(cmd, "target", -1),
        parameter: i32v(cmd, "parameter", 0),
    }
}

fn cheb(a: i32, b: i32, size: i32) -> i32 {
    let (ax, ay) = (a % size, a / size);
    let (bx, by) = (b % size, b / size);
    (ax - bx).abs().max((ay - by).abs())
}

fn unit_attack(kind: i32) -> f32 {
    match kind {
        2 => 2.0, 3 => 2.0, 4 => 3.5, 5 => 1.0, 8 => 4.0, 9 => 2.0, 10 => 0.0,
        11 => 3.0, 12 => 5.0, 35 => 3.0, 37 => 2.0, 40 => 2.0, 41 => 3.0,
        42 => 2.0, 44 => 3.0, 45 => 4.0, _ => 0.0,
    }
}
fn unit_defense(kind: i32) -> f32 {
    match kind {
        2 => 2.0, 3 => 1.0, 4 => 1.0, 5 => 3.0, 8 => 0.0, 9 => 1.0, 10 => 1.0,
        11 => 3.0, 12 => 4.0, 35 => 2.0, 37 => 2.0, 40 => 2.0, 41 => 2.0,
        42 => 1.0, 44 => 3.0, 45 => 4.0, _ => 1.0,
    }
}
fn unit_maxhp(kind: i32) -> f32 {
    match kind {
        5 | 11 => 15.0, 12 => 40.0, 45 => 40.0, _ => 10.0,
    }
}
fn unit_range(kind: i32) -> i32 {
    match kind { 8 => 3, 9 => 2, 30 => 2, 41 => 3, 42 => 2, 43 => 2, 24 => 2, 25 => 2, _ => 1 }
}
fn unit_cost(kind: i32) -> i32 {
    match kind {
        2 => 2, 3 => 3, 4 => 8, 5 => 3, 8 => 8, 9 => 3, 10 => 5, 11 => 5, 12 => 10,
        40 => 2, 42 => 5, 44 => 5, 41 => 15, 45 => 10, _ => 99,
    }
}
fn is_stiff(kind: i32) -> bool {
    matches!(kind, 8 | 10 | 22 | 24 | 25 | 34 | 38 | 39 | 41 | 43 | 45 | 60 | 61)
}

fn defense_bonus(tile: &Tile, city_owner: i32, unit_owner: i32) -> f32 {
    if tile.terrain == 4 { 1.5 }
    else if tile.terrain == 5 { 1.5 }
    else if tile.terrain == 1 || tile.terrain == 2 { 1.5 }
    else if city_owner == unit_owner && unit_owner > 0 { 1.5 }
    else { 1.0 }
}

fn combat_damage(att_atk: f32, att_hp: f32, att_max: f32, def_def: f32, def_hp: f32, def_max: f32, def_bonus: f32) -> (f32, f32) {
    let af = att_atk * (att_hp / att_max.max(1.0));
    let df = def_def * (def_hp / def_max.max(1.0)) * def_bonus;
    let total = af + df;
    let attack = if total > 0.0 { ((af / total) * att_atk * 4.5 + 0.000001 + 0.5).floor() } else { 0.0 };
    let defense = if def_hp > attack && total > 0.0 { ((df / total) * def_def * 4.5 + 0.000001 + 0.5).floor() } else { 0.0 };
    (attack, defense)
}


fn select_move(actions: &[Action], tiles: &[Tile], units: &[Unit], my_cities: &[City], threatened_cities: &[usize], size: i32, n: usize, me: i32) -> Option<String> {
    let moves: Vec<&Action> = actions.iter().filter(|a| a.kind == 1).collect();
    if moves.is_empty() { return None; }
    // Baseline-style movement: move toward the nearest unknown tile, neutral/enemy city, or visible enemy.
    let mut targets: Vec<usize> = Vec::new();
    for i in 0..n {
        if !tiles[i].known {
            targets.push(i);
        } else if (tiles[i].structure == 1 && tiles[i].owner != me) || tiles[i].structure == 2 {
            targets.push(i);
        } else if tiles[i].resource == 8 && tiles[i].structure == 0 {
            targets.push(i);
        }
    }
    for u in units.iter().filter(|u| u.owner != me && u.owner > 0) {
        if u.tile < n { targets.push(u.tile); }
    }
    if targets.is_empty() { targets.push(((size / 2) + (size / 2) * size) as usize); }
    let distance = |tile: i32| -> i32 {
        let mut best = i32::MAX;
        for &t in &targets {
            let d = cheb(tile, t as i32, size);
            if d < best { best = d; }
        }
        best
    };
    let mut best: Option<&Action> = None;
    let mut bestv = f32::MIN;
    for a in moves {
        let src = a.source.max(0) as usize;
        let dst = a.target.max(0) as usize;
        if src >= n || dst >= n { continue; }
        let mut score;
        if !threatened_cities.is_empty() {
            let mut closest = i32::MAX;
            for &ct in threatened_cities {
                closest = closest.min(cheb(dst as i32, ct as i32, size));
            }
            score = 900.0 - closest as f32 * 60.0;
            if closest <= 1 { score += 250.0; }
        } else {
            let dsrc = distance(src as i32);
            let ddst = distance(dst as i32);
            score = 30.0 + 15.0 * (dsrc - ddst) as f32;
            if tiles[dst].structure == 2 { score += 120.0; }
            if tiles[dst].structure == 1 && tiles[dst].owner == 0 { score += 150.0; }
            if tiles[dst].structure == 1 && tiles[dst].owner == me { score -= 80.0; }
        }
        if score > bestv { bestv = score; best = Some(a); }
    }
    best.map(|a| a.id.clone())
}

fn choose(obs: &Value) -> String {
    let legal = obs["legal_actions"].as_array().expect("legal_actions");
    if legal.is_empty() { return String::new(); }
    let actions: Vec<Action> = legal.iter().map(parse_action).collect();
    let vis = &obs["visible_state"];
    let size = i32v(vis, "board_size", 14).max(1);
    let n = (size * size) as usize;
    let me = obs["player"].as_i64().unwrap_or(1) as i32;
    let stars = i32v(vis, "stars", 0);
    let turn = i32v(vis, "turn", 0);

    // tiles
    let mut tiles = vec![Tile::default(); n];
    if let Some(arr) = vis["tiles"].as_array() {
        for tv in arr {
            let idx = i32v(tv, "tile", -1);
            if idx < 0 || idx as usize >= n { continue; }
            let t = &mut tiles[idx as usize];
            t.known = bv(tv, "known");
            if t.known {
                t.terrain = i32v(tv, "terrain", 0);
                t.owner = i32v(tv, "owner", 0);
                t.resource = i32v(tv, "resource", 0);
                t.structure = i32v(tv, "structure", 0);
                t.road = i32v(tv, "road", 0);
                t.capital = i32v(tv, "capital", 0);
            }
        }
    }
    // units
    let mut units: Vec<Unit> = Vec::new();
    if let Some(arr) = vis["units"].as_array() {
        for uv in arr {
            units.push(Unit {
                tile: i32v(uv, "tile", -1).max(0) as usize,
                owner: i32v(uv, "owner", 0),
                kind: i32v(uv, "kind", 0),
                health: f32v(uv, "health", 0.0),
                moved: bv(uv, "moved"),
                attacked: bv(uv, "attacked"),
            });
        }
    }
    let my_units: Vec<&Unit> = units.iter().filter(|u| u.owner == me).collect();
    // cities
    let mut my_cities: Vec<City> = Vec::new();
    if let Some(arr) = vis["cities"].as_array() {
        for cv in arr {
            let city = City {
                tile: i32v(cv, "tile", -1).max(0) as usize,
                owner: i32v(cv, "owner", 0),
                level: i32v(cv, "level", 0),
                population: i32v(cv, "population", 0),
                production: i32v(cv, "production", 0),
                progress: i32v(cv, "progress", 0),
                connected: bv(cv, "connected"),
                riot: bv(cv, "riot"),
            };
            if city.owner == me { my_cities.push(city); }
        }
    }
    let has_tech = |t: i32| -> bool {
        vis["technology"].as_array().map(|arr| arr.iter().any(|v| v.as_i64() == Some(t as i64))).unwrap_or(false)
    };

    // Threat detection: visible enemy units near our cities.
    let mut threatened_cities: Vec<usize> = Vec::new();
    for c in &my_cities {
        let mut threat = false;
        for u in units.iter().filter(|u| u.owner != me && u.owner > 0) {
            if cheb(c.tile as i32, u.tile as i32, size) <= 1 {
                threat = true;
                break;
            }
        }
        if threat { threatened_cities.push(c.tile); }
    }

    // helpers to find legal actions by kind
    let kind_actions = |k: i32| -> Vec<&Action> { actions.iter().filter(|a| a.kind == k).collect() };

    // 1. Mandatory city reward
    let rewards = kind_actions(9);
    if !rewards.is_empty() {
        let pref = |p: i32| -> i32 {
            match p {
                3 => 1000, 4 => 700, 7 => 900, 1 => 500, 5 => 850, 8 => 800, 6 => 950, 2 => 400, _ => 0
            }
        };
        let mut best: Option<&Action> = None;
        let mut bestv = -1;
        for a in rewards {
            let v = pref(a.parameter);
            if v > bestv { bestv = v; best = Some(a); }
        }
        return best.unwrap().id.clone();
    }

    // 2. Capture villages/ruins/starfish
    let caps = kind_actions(8);
    if !caps.is_empty() {
        // Prefer villages, then ruins, then starfish
        let mut best: Option<&Action> = None;
        let mut bestv = -1;
        for a in caps {
            let t = &tiles[a.source as usize];
            let v = if t.structure == 1 { 1000 } else if t.structure == 2 { 800 } else { 600 };
            if v > bestv { bestv = v; best = Some(a); }
        }
        return best.unwrap().id.clone();
    }

    // 3. Attack
    let attacks = kind_actions(2);
    if !attacks.is_empty() {
        let mut best: Option<&Action> = None;
        let mut bestv = f32::MIN;
        for a in attacks {
            let src = a.source.max(0) as usize;
            let dst = a.target.max(0) as usize;
            if src >= n || dst >= n { continue; }
            let au = units.iter().find(|u| u.tile == src && u.owner == me);
            let du = units.iter().find(|u| u.tile == dst && u.owner != me);
            if au.is_none() || du.is_none() { continue; }
            let au = au.unwrap(); let du = du.unwrap();
            let a_atk = unit_attack(au.kind);
            let d_def = unit_defense(du.kind);
            let a_max = unit_maxhp(au.kind) + if bv(&vis["units"][0], "veteran") { 5.0 } else { 0.0 }; // rough
            let d_max = unit_maxhp(du.kind);
            let db = defense_bonus(&tiles[dst], me, du.owner);
            let (dmg, retal) = combat_damage(a_atk, au.health, a_max, d_def, du.health, d_max, db);
            let kill = dmg >= du.health;
            let our_hp_after = if kill || is_stiff(du.kind) || cheb(src as i32, dst as i32, size) > unit_range(du.kind) { au.health } else { au.health - retal };
            if our_hp_after <= 0.0 { continue; }
            let enemy_near_city = threatened_cities.iter().any(|&ct| cheb(dst as i32, ct as i32, size) <= 2);
            let mut score = if kill {
                1000.0 + dmg - retal * 0.3 + unit_cost(du.kind) as f32
            } else {
                dmg * 2.0 - retal * 3.0 + if retal == 0.0 { 4.0 } else { 0.0 }
            };
            if enemy_near_city { score += 10.0; }
            if score > bestv { bestv = score; best = Some(a); }
        }
        if let Some(a) = best {
            if bestv >= 0.0 {
                return a.id.clone();
            }
        }
    }

    // 4. Harvest resources
    let harvests = kind_actions(5);
    if !harvests.is_empty() {
        let mut best: Option<&Action> = None;
        let mut bestv = -1;
        for a in harvests {
            let t = a.target.max(0) as usize;
            if t >= n { continue; }
            let res = tiles[t].resource;
            let early = turn <= 8 && my_cities.iter().any(|c| c.level < 3);
            let star_ok = if early { stars >= 2 } else { stars >= 7 };
            let good = res == 1 || res == 6 || res == 3;
            if good && star_ok {
                let v = if res == 1 { 100 } else if res == 6 { 90 } else { 80 };
                if v > bestv { bestv = v; best = Some(a); }
            }
        }
        if let Some(a) = best { return a.id.clone(); }
    }

    // Move toward known villages/ruins or defend before spending stars on non-urgent tech.
    if let Some(id) = select_move(&actions, &tiles, &units, &my_cities, &threatened_cities, size, n, me) {
        return id;
    }

    // 5. Summon units (early expansion / later military)
    let summons = kind_actions(4);
    if !summons.is_empty() {
        let mut best: Option<&Action> = None;
        let mut bestv = -1;
        for a in summons {
            let uk = a.parameter;
            let cost = unit_cost(uk);
            if stars < cost { continue; }
            let mut v = 0;
            // early cheap scouts are very valuable; avoid spending all stars on late tech instead of units
            let my_unit_count = my_units.len() as i32;
            let city_count = my_cities.len() as i32;
            let early = turn <= 6 && city_count <= 1;
            if uk == 2 {
                v = if early && my_unit_count <= 1 { 120 } else if my_unit_count < city_count + 1 { 70 } else { 20 };
            } else if uk == 3 {
                v = if has_tech(1) { 110 } else { 0 };
            } else if uk == 5 {
                v = 55;
            } else if uk == 9 {
                v = 75;
            } else if uk == 11 {
                v = 90;
            } else if uk == 4 {
                v = 100;
            } else if uk == 12 {
                v = 200;
            } else if uk == 42 || uk == 44 || uk == 40 {
                v = 90;
            } else {
                v = 40;
            }
            // Require a small reserve after buying.
            if v > 0 && stars - cost >= (if turn <= 3 { 0 } else { 1 }).max(0) {
                if v > bestv { bestv = v; best = Some(a); }
            }
        }
        if let Some(a) = best { return a.id.clone(); }
    }

    // 6. Research
    let researches = kind_actions(7);
    if !researches.is_empty() {
        // Fixed priority order
        let water_known = tiles.iter().take(n).filter(|t| t.known && (t.terrain == 1 || t.terrain == 2)).count();
        let forest_known = tiles.iter().take(n).filter(|t| t.known && t.terrain == 5).count();
        let order = [
            (6, 100),   // Organization
            (16, if forest_known >= 2 { 100 } else { 65 }), // Forestry: Bardur growth
            (1, if forest_known >= 2 { 85 } else { 95 }),   // Riding
            (10, if water_known >= 3 { 95 } else { 30 }),   // Fishing
            (4, 80),    // Roads
            (13, if water_known >= 3 { 85 } else { 25 }),   // Sailing
            (18, 60),   // Archery
            (20, 35),   // Climbing
            (8, 30),    // Farming (crop is uncommon on these maps)
            (24, 75),   // Smithery
            (17, 65),   // Mathematics
            (7, 50),    // Strategy
            (3, 60),    // Chivalry
            (14, if water_known >= 3 { 70 } else { 20 }),   // Navigation
            (9, 40),    // Construction
            (5, 40),    // Trade
            (23, 35),   // Mining
        ];
        let mut best: Option<&Action> = None;
        let mut bestv = -1;
        for a in researches {
            if let Some(v) = order.iter().find(|(t, _)| *t == a.parameter).map(|(_, v)| *v) {
                if v > bestv { bestv = v; best = Some(a); }
            }
        }
        if let Some(a) = best {
            let cost = 4 + my_cities.len() as i32; // rough: tech tier often 1 or 2; legal actions already check affordability
            let reserve = if turn <= 2 { 0 } else if my_units.len() < 2 { 2 } else { 0 };
            if stars >= cost + reserve {
                return a.id.clone();
            }
        }
    }

    // 7. Build improvements (simple: port if water map and no port, roads later, mines/farms/workshops if unlocked)
    let builds = kind_actions(6);
    if !builds.is_empty() {
        let water_known = tiles.iter().take(n).filter(|t| t.known && (t.terrain == 1 || t.terrain == 2)).count();
        let mut best: Option<&Action> = None;
        let mut bestv = -1;
        for a in builds {
            let p = a.parameter;
            let v = match p {
                8 => if water_known >= 4 { 110 } else { 0 }, // Port
                12 => 120, // Lumber hut: important for Bardur city growth
                21 => 110, // Mine
                5 => 100, // Farm
                22 => 95,  // Forge
                13 => 90,  // Sawmill
                6 => 90,   // Windmill
                50 => 70,  // Market
                3 => 70,   // Road
                48 => 70,  // Bridge
                _ => 0,
            };
            if v > bestv { bestv = v; best = Some(a); }
        }
        if let Some(a) = best {
            let min_stars = if bestv >= 110 { 3 } else if bestv >= 90 { 5 } else { 7 };
            if stars >= min_stars {
                return a.id.clone();
            }
        }
    }

    // 8. Move units (also called earlier when no more urgent economic/combat action)
    if let Some(id) = select_move(&actions, &tiles, &units, &my_cities, &threatened_cities, size, n, me) {
        return id;
    }

    // 9. Promote / heal / recover / disband / forest useful actions before end
    for k in [16, 9, 7] {
        if let Some(a) = kind_actions(k).first() {
            return a.id.clone();
        }
    }

    // 10. End turn
    if let Some(a) = kind_actions(10).first() {
        return a.id.clone();
    }
    // Last resort: first legal
    actions.first().map(|a| a.id.clone()).unwrap_or_default()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let Ok(line) = line else { continue };
        let obs: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = choose(&obs);
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
