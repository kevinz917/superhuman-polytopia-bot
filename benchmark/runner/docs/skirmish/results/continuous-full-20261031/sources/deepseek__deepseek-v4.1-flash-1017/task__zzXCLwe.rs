// Polytopia Bardur 1v1 heuristic bot.
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

#[derive(Clone, Copy, Default)]
struct Tile {
    known: bool,
    terrain: i32,
    owner: i32,
    structure: i32,
    resource: i32,
    capital: i32,
}

#[derive(Clone, Copy, Default)]
struct Unit {
    tile: usize,
    owner: i32,
    kind: i32,
    hp: f64,
    vet: bool,
    moved: bool,
    attacked: bool,
    home: i32,
}

#[derive(Clone, Copy, Default)]
struct City {
    tile: usize,
    owner: i32,
    level: i32,
    production: i32,
    connected: bool,
}

#[derive(Clone, Default)]
struct Act {
    id: String,
    kind: i32,
    src: i32,
    tgt: i32,
    param: i32,
}

struct Obs {
    size: usize,
    n: usize,
    player: i32,
    turn: i32,
    stars: i32,
    tiles: Vec<Tile>,
    units: Vec<Unit>,
    cities: Vec<City>,
    acts: Vec<Act>,
    tech: [bool; 128],
}

fn new_obs(size: usize) -> Obs {
    let n = size * size;
    Obs {
        size,
        n,
        player: 1,
        turn: 0,
        stars: 0,
        tiles: vec![Tile::default(); n],
        units: Vec::new(),
        cities: Vec::new(),
        acts: Vec::new(),
        tech: [false; 128],
    }
}

fn gi(v: &Value, k: &str) -> i32 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(0) as i32
}
fn gb(v: &Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}
fn gf(v: &Value, k: &str) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0)
}

fn parse(oj: &Value) -> Obs {
    let v = match oj.get("visible_state") {
        Some(x) => x,
        None => return new_obs(14),
    };
    let size = v.get("board_size").and_then(|x| x.as_u64()).unwrap_or(14) as usize;
    let mut o = new_obs(size);
    o.player = gi(oj, "player").max(1);
    o.turn = gi(v, "turn");
    o.stars = gi(v, "stars");
    if let Some(ts) = v.get("technology").and_then(|x| x.as_array()) {
        for t in ts {
            if let Some(k) = t.as_i64() {
                if (0..128).contains(&k) {
                    o.tech[k as usize] = true;
                }
            }
        }
    }
    if let Some(ts) = v.get("tiles").and_then(|x| x.as_array()) {
        for t in ts {
            let idx = gi(t, "tile");
            if idx < 0 || idx as usize >= o.n {
                continue;
            }
            let idx = idx as usize;
            if gb(t, "known") {
                o.tiles[idx] = Tile {
                    known: true,
                    terrain: gi(t, "terrain"),
                    owner: gi(t, "owner"),
                    structure: gi(t, "structure"),
                    resource: gi(t, "resource"),
                    capital: gi(t, "capital"),
                };
            } else {
                o.tiles[idx].known = false;
            }
        }
    }
    if let Some(us) = v.get("units").and_then(|x| x.as_array()) {
        for u in us {
            let tile = gi(u, "tile");
            if tile < 0 || tile as usize >= o.n {
                continue;
            }
            o.units.push(Unit {
                tile: tile as usize,
                owner: gi(u, "owner"),
                kind: gi(u, "kind"),
                hp: gf(u, "health"),
                vet: gb(u, "veteran"),
                moved: gb(u, "moved"),
                attacked: gb(u, "attacked"),
                home: gi(u, "home"),
            });
        }
    }
    if let Some(cs) = v.get("cities").and_then(|x| x.as_array()) {
        for c in cs {
            let tile = gi(c, "tile");
            if tile < 0 || tile as usize >= o.n {
                continue;
            }
            o.cities.push(City {
                tile: tile as usize,
                owner: gi(c, "owner"),
                level: gi(c, "level"),
                production: gi(c, "production"),
                connected: gb(c, "connected"),
            });
        }
    }
    if let Some(ls) = oj.get("legal_actions").and_then(|x| x.as_array()) {
        for a in ls {
            let id = match a.get("action_id").and_then(|x| x.as_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let c = match a.get("command") {
                Some(c) => c,
                None => continue,
            };
            o.acts.push(Act {
                id,
                kind: gi(c, "kind"),
                src: gi(c, "source"),
                tgt: gi(c, "target"),
                param: gi(c, "parameter"),
            });
        }
    }
    o
}

// ---------------- unit tables ----------------
fn hp_of(k: i32) -> f64 {
    match k {
        2 => 10.0,
        3 => 10.0,
        4 => 10.0,
        5 => 15.0,
        8 => 10.0,
        9 => 10.0,
        10 => 10.0,
        11 => 15.0,
        12 => 40.0,
        17 => 10.0,
        18 => 15.0,
        19 => 20.0,
        20 => 10.0,
        21 => 10.0,
        22 => 10.0,
        23 => 15.0,
        24 => 20.0,
        25 => 10.0,
        26 => 40.0,
        27 => 30.0,
        28 => 5.0,
        29 => 20.0,
        30 => 5.0,
        31 => 15.0,
        32 => 10.0,
        33 => 20.0,
        34 => 10.0,
        35 => 10.0,
        36 => 10.0,
        37 => 10.0,
        38 => 5.0,
        39 => 5.0,
        40 => 10.0,
        41 => 10.0,
        42 => 10.0,
        43 => 10.0,
        44 => 10.0,
        45 => 40.0,
        _ => 10.0,
    }
}
fn atk_of(k: i32) -> f64 {
    match k {
        2 => 2.0,
        3 => 2.0,
        4 => 3.5,
        5 => 1.0,
        8 => 4.0,
        9 => 2.0,
        10 => 0.0,
        11 => 3.0,
        12 => 5.0,
        17 => -1.0,
        18 => 3.0,
        19 => 4.0,
        20 => 2.0,
        21 => 2.5,
        22 => 0.0,
        23 => 3.0,
        24 => 4.0,
        25 => 1.0,
        26 => 4.0,
        27 => 4.0,
        28 => 3.0,
        29 => 3.5,
        30 => 0.7,
        31 => 1.0,
        32 => 3.0,
        33 => 4.0,
        34 => 2.0,
        35 => 3.0,
        36 => 1.0,
        37 => 2.0,
        38 => 0.0,
        39 => 0.0,
        40 => 2.0,
        41 => 3.0,
        42 => 2.0,
        43 => 0.0,
        44 => 3.0,
        45 => 4.0,
        _ => 0.0,
    }
}
fn def_of(k: i32) -> f64 {
    match k {
        2 => 2.0,
        3 => 1.0,
        4 => 1.0,
        5 => 3.0,
        8 => 0.0,
        9 => 1.0,
        10 => 1.0,
        11 => 3.0,
        12 => 4.0,
        17 => 2.0,
        18 => 3.0,
        19 => 3.0,
        20 => 1.0,
        21 => 1.0,
        22 => 1.0,
        23 => 2.0,
        24 => 3.0,
        25 => 1.0,
        26 => 5.0,
        27 => 3.0,
        28 => 1.0,
        29 => 2.0,
        30 => 1.0,
        31 => 3.0,
        32 => 1.0,
        33 => 3.0,
        34 => 2.0,
        35 => 2.0,
        36 => 1.0,
        37 => 2.0,
        38 => 0.5,
        39 => 0.5,
        40 => 2.0,
        41 => 2.0,
        42 => 1.0,
        43 => 1.0,
        44 => 3.0,
        45 => 4.0,
        _ => 1.0,
    }
}
fn cost_of(k: i32) -> f64 {
    match k {
        2 => 2.0,
        3 => 3.0,
        4 => 8.0,
        5 => 3.0,
        8 => 8.0,
        9 => 3.0,
        10 => 5.0,
        11 => 5.0,
        12 => 10.0,
        37 => 2.0,
        38 => 8.0,
        39 => 2.0,
        40 => 2.0,
        41 => 15.0,
        42 => 5.0,
        43 => 0.0,
        44 => 5.0,
        45 => 10.0,
        _ => 2.0,
    }
}
fn range_of(k: i32) -> i32 {
    match k {
        8 => 3,
        9 => 2,
        4 => 1,
        42 => 2,
        41 => 3,
        43 => 2,
        _ => 1,
    }
}
fn is_military(k: i32) -> bool {
    !(k == 38 || k == 39 || k == 43 || k == 10 || k == 22)
}
fn fortify(k: i32) -> bool {
    matches!(k, 2 | 3 | 4 | 5 | 9 | 20)
}

fn combat(aa: f64, ah: f64, am: f64, dd: f64, dh: f64, dm: f64, db: f64) -> (f64, f64) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    let divisor = if total > 0.0 { total } else { 1.0 };
    let mut attack = 0.0;
    if total > 0.0 {
        attack = (af / divisor) * aa * 4.5;
    }
    attack = (attack + 0.000001 + 0.5).floor();
    let mut retal = 0.0;
    if dh > attack && total > 0.0 {
        retal = ((df / divisor) * dd * 4.5 + 0.000001 + 0.5).floor();
    }
    (attack, retal)
}

fn dist(size: usize, a: usize, b: usize) -> i32 {
    let ax = (a % size) as i32;
    let ay = (a / size) as i32;
    let bx = (b % size) as i32;
    let by = (b / size) as i32;
    (ax - bx).abs().max((ay - by).abs())
}

fn dist_map(size: usize, n: usize, targets: &[usize]) -> Vec<i32> {
    let mut d = vec![999i32; n];
    if targets.is_empty() {
        return d;
    }
    for t in 0..n {
        let tx = (t % size) as i32;
        let ty = (t / size) as i32;
        let mut best = 999i32;
        for &g in targets {
            let gx = (g % size) as i32;
            let gy = (g / size) as i32;
            let dd = (tx - gx).abs().max((ty - gy).abs());
            if dd < best {
                best = dd;
            }
        }
        d[t] = best;
    }
    d
}

struct Ctx {
    // derived target lists
    villages: Vec<usize>,
    enemy_cities: Vec<usize>,
    own_cities: Vec<usize>,
    unknown: Vec<usize>,
    enemy_units: Vec<usize>, // tiles
    threats: Vec<usize>,     // enemy unit tiles near own cities
    unit_at: Vec<i32>,       // index into obs.units or -1
}

fn build_ctx(o: &Obs) -> Ctx {
    let mut c = Ctx {
        villages: Vec::new(),
        enemy_cities: Vec::new(),
        own_cities: Vec::new(),
        unknown: Vec::new(),
        enemy_units: Vec::new(),
        threats: Vec::new(),
        unit_at: vec![-1; o.n],
    };
    for i in 0..o.n {
        let t = o.tiles[i];
        if !t.known {
            c.unknown.push(i);
            continue;
        }
        if t.structure == 1 {
            if t.owner == 0 {
                c.villages.push(i);
            } else if t.owner == o.player {
                c.own_cities.push(i);
            } else if t.owner > 0 {
                c.enemy_cities.push(i);
            }
        }
    }
    for (k, u) in o.units.iter().enumerate() {
        if u.tile < o.n {
            c.unit_at[u.tile] = k as i32;
        }
        if u.owner != o.player && u.owner > 0 {
            c.enemy_units.push(u.tile);
        }
    }
    for &et in &c.enemy_units {
        for &ct in &c.own_cities {
            if dist(o.size, et, ct) <= 3 {
                c.threats.push(et);
                break;
            }
        }
    }
    c
}

fn defense_bonus(o: &Obs, t: usize) -> f64 {
    let tile = o.tiles[t];
    if tile.structure == 1 {
        return 1.5;
    }
    match tile.terrain {
        4 | 5 => 1.5,
        1 | 2 => 1.2,
        _ => 1.0,
    }
}

fn tech_prio(o: &Obs, k: usize, forests: i32, mountains: i32, water: i32, fish: i32, enemy_known: bool) -> f64 {
    let has = |x: usize| o.tech[x];
    match k {
        6 => 120.0,                                        // Organization
        8 => {
            if has(6) {
                105.0
            } else {
                20.0
            }
        } // Farming
        16 => {
            if forests > 1 {
                135.0
            } else {
                75.0
            }
        } // Forestry
        17 => {
            if has(16) {
                120.0
            } else {
                30.0
            }
        } // Mathematics
        1 => 95.0,                                         // Riding
        18 => 100.0,                                       // Archery
        20 => {
            if mountains > 0 {
                85.0
            } else {
                45.0
            }
        } // Climbing
        23 => {
            if has(20) {
                115.0
            } else {
                20.0
            }
        } // Mining
        24 => {
            if has(23) {
                125.0
            } else {
                20.0
            }
        } // Smithery
        9 => {
            if has(8) {
                85.0
            } else {
                20.0
            }
        } // Construction
        4 => {
            if has(1) {
                70.0
            } else {
                20.0
            }
        } // Roads
        7 => {
            if has(6) {
                65.0
            } else {
                20.0
            }
        } // Strategy
        10 => {
            if fish > 0 || water > 4 {
                105.0
            } else {
                30.0
            }
        } // Fishing
        13 => {
            if has(10) && (water > 12 || enemy_known) {
                95.0
            } else {
                15.0
            }
        } // Sailing
        44 => {
            if has(10) && (water > 12 || enemy_known) {
                90.0
            } else {
                15.0
            }
        } // Ramming
        14 => {
            if has(13) && water > 12 {
                55.0
            } else {
                10.0
            }
        } // Navigation
        3 => {
            if has(2) {
                75.0
            } else {
                10.0
            }
        } // Chivalry
        2 => 30.0,                                         // FreeSpirit
        5 => 8.0,                                          // Trade
        12 => 10.0,
        38 => 8.0,
        21 => 8.0,
        22 => 8.0,
        19 => 8.0,
        48 => 20.0,
        _ => 6.0,
    }
}

fn research_prio(o: &Obs, c: &Ctx) -> f64 {
    let mut forests = 0;
    let mut mountains = 0;
    let mut water = 0;
    let mut fish = 0;
    for i in 0..o.n {
        let t = o.tiles[i];
        if !t.known {
            continue;
        }
        if t.terrain == 1 || t.terrain == 2 {
            water += 1;
        }
        if t.owner == o.player {
            match t.terrain {
                5 => forests += 1,
                4 => mountains += 1,
                1 | 2 => {
                    if t.resource == 3 {
                        fish += 1
                    }
                }
                _ => {}
            }
        }
    }
    forests += (c.unknown.len() / 20) as i32;
    let _ = mountains;
    let enemy_known = !c.enemy_cities.is_empty() || !c.enemy_units.is_empty();
    let mut best: f64 = 0.0;
    for a in &o.acts {
        if a.kind == 7 {
            let p = tech_prio(&o, a.param.max(0) as usize, forests, mountains, water, fish, enemy_known);
            best = best.max(p);
        }
    }
    best
}

fn build_value(o: &Obs, kind: i32, t: usize) -> f64 {
    let adj_count = |st: i32| -> i32 {
        let mut n = 0;
        let size = o.size;
        let x = (t % size) as i32;
        let y = (t / size) as i32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= size as i32 || ny >= size as i32 {
                    continue;
                }
                let j = (ny as usize) * size + nx as usize;
                if j < o.n && o.tiles[j].structure == st {
                    n += 1;
                }
            }
        }
        n
    };
    match kind {
        3 => 25.0,  // road
        5 => 85.0,  // farm: +2 pop
        6 => 45.0 + 15.0 * adj_count(5) as f64,   // windmill
        8 => 80.0,  // port
        12 => 75.0, // lumber hut
        13 => 45.0 + 22.0 * adj_count(12) as f64, // sawmill
        17 => 10.0,
        18 => 10.0,
        19 => 10.0,
        20 => 10.0,
        21 => 85.0, // mine
        22 => 45.0 + 15.0 * adj_count(21) as f64,
        48 => 20.0,
        50 => 55.0,
        _ => 15.0,
    }
}

fn recruit_value(k: i32) -> f64 {
    match k {
        2 => 105.0,
        3 => 95.0,
        4 => 140.0,
        5 => 100.0,
        8 => 20.0,
        9 => 125.0,
        10 => 10.0,
        11 => 170.0,
        12 => 220.0,
        42 => 120.0,
        44 => 130.0,
        41 => 90.0,
        43 => 30.0,
        _ => 50.0,
    }
}

fn reward_prio(param: i32, giants: i32) -> f64 {
    match param {
        6 => {
            if giants >= 2 {
                60.0
            } else {
                100.0
            }
        }
        3 => 98.0,
        5 => 92.0,
        7 => 88.0,
        8 => 84.0,
        2 => 70.0,
        4 => 60.0,
        1 => 40.0,
        _ => 30.0,
    }
}



fn choose(oj: &Value, mem: &mut Mem) -> String {
    let o = parse(oj);
    if o.acts.is_empty() {
        return "0".to_string();
    }
    let c = build_ctx(&o);
    if mem.plan.len() < o.n {
        mem.plan.resize(o.n, -1);
    }
    mem.turn = o.turn;

    let own_giants = o.units.iter().filter(|u| u.owner == o.player && u.kind == 12).count() as i32;
    let own_military = o.units.iter().filter(|u| u.owner == o.player && is_military(u.kind)).count() as i32;
    let threats = !c.threats.is_empty();
    let mut own_city_tiles = c.own_cities.clone();
    for ct in &o.cities {
        if ct.owner == o.player && !own_city_tiles.contains(&ct.tile) {
            own_city_tiles.push(ct.tile);
        }
    }
    let own_city_tiles = own_city_tiles;

    // nearest distance from a tile to any enemy unit / enemy city
    let mut enemy_presence: Vec<i32> = vec![99; o.n];
    {
        let mut targets: Vec<usize> = c.enemy_units.clone();
        targets.extend(c.enemy_cities.iter().copied());
        if !targets.is_empty() {
            enemy_presence = dist_map(o.size, o.n, &targets);
        }
    }

    // clear stale/complete plans
    for t in 0..o.n {
        if mem.plan[t] >= 0 {
            let tg = mem.plan[t] as usize;
            let ok = tg < o.n && o.tiles[tg].structure == 1 && o.tiles[tg].owner == 0;
            if !ok {
                mem.plan[t] = -1;
            }
        }
    }

    // plan new targets for military units that have none (unique villages)
    let mut unit_target: Vec<i32> = vec![-1; o.units.len()];
    {
        let mut claimed: Vec<usize> = Vec::new();
        for u in o.units.iter() {
            if u.owner == o.player && u.tile < o.n {
                let pl = mem.plan[u.tile];
                if pl >= 0 {
                    claimed.push(pl as usize);
                }
            }
        }
        for (ui, u) in o.units.iter().enumerate() {
            if u.owner != o.player || u.tile >= o.n {
                continue;
            }
            if !is_military(u.kind) {
                continue;
            }
            if mem.plan[u.tile] >= 0 {
                unit_target[ui] = mem.plan[u.tile];
                continue;
            }
            // choose the best unclaimed village: near us but away from the enemy
            let mut best_cost = 1e9;
            let mut best_v = -1i32;
            for &v in c.villages.iter() {
                if claimed.contains(&v) {
                    continue;
                }
                let d = dist(o.size, u.tile, v) as f64;
                let e = (enemy_presence[v].min(9)) as f64;
                let cost = d * 3.0 + e * 4.0;
                if cost < best_cost {
                    best_cost = cost;
                    best_v = v as i32;
                }
            }
            if best_v >= 0 {
                unit_target[ui] = best_v;
                claimed.push(best_v as usize);
                mem.plan[u.tile] = best_v;
            }
        }
    }
    // ---- per-turn effective targets (do not overwrite persistent plans) ----
    let mut et: Vec<i32> = vec![-1; o.units.len()];
    let mut ew: Vec<f64> = vec![62.0; o.units.len()];
    for (ui, u) in o.units.iter().enumerate() {
        if u.owner == o.player && u.tile < o.n {
            et[ui] = unit_target[ui];
        }
    }
    // damaged units retreat to a friendly city
    for (ui, u) in o.units.iter().enumerate() {
        if u.owner != o.player || !is_military(u.kind) {
            continue;
        }
        let maxhp = hp_of(u.kind) + if u.vet { 5.0 } else { 0.0 };
        if u.hp < maxhp * 0.7 && !own_city_tiles.is_empty() {
            let mut bd = 99;
            let mut bt = -1i32;
            for &ct in &own_city_tiles {
                let d = dist(o.size, u.tile, ct);
                if d < bd {
                    bd = d;
                    bt = ct as i32;
                }
            }
            if bd > 0 {
                et[ui] = bt;
                ew[ui] = 34.0;
            }
        }
    }
    // hold position next to a threatened city (only the single guard of a city)
    if !c.enemy_units.is_empty() {
        for &ct in &own_city_tiles {
            let near_enemy = c.enemy_units.iter().any(|&e| dist(o.size, e, ct) <= 4);
            if !near_enemy {
                continue;
            }
            let mut guards: Vec<usize> = Vec::new();
            for (ui, u) in o.units.iter().enumerate() {
                if u.owner == o.player && is_military(u.kind) && dist(o.size, u.tile, ct) <= 1 {
                    guards.push(ui);
                }
            }
            if guards.len() == 1 {
                let ui = guards[0];
                et[ui] = o.units[ui].tile as i32;
                ew[ui] = 80.0;
            }
        }
        // reinforce cities with no guard at all when an enemy is close
        for &ct in &own_city_tiles {
            let enemy_close = c.enemy_units.iter().any(|&e| dist(o.size, e, ct) <= 5);
            if !enemy_close {
                continue;
            }
            let has_guard = o
                .units
                .iter()
                .any(|u| u.owner == o.player && is_military(u.kind) && dist(o.size, u.tile, ct) <= 1);
            if has_guard {
                continue;
            }
            let mut bd = 99;
            let mut bi: i32 = -1;
            for (ui, u) in o.units.iter().enumerate() {
                if u.owner != o.player || !is_military(u.kind) {
                    continue;
                }
                let d = dist(o.size, u.tile, ct);
                if d < bd {
                    bd = d;
                    bi = ui as i32;
                }
            }
            if bi >= 0 {
                et[bi as usize] = ct as i32;
                ew[bi as usize] = 70.0;
            }
        }
    }
    // threat override: units near an enemy that threatens a city go for it
    if !c.threats.is_empty() {
        for (ui, u) in o.units.iter().enumerate() {
            if u.owner != o.player || !is_military(u.kind) {
                continue;
            }
            let mut bd = 99;
            let mut bt = -1i32;
            for &t in &c.threats {
                let d = dist(o.size, u.tile, t);
                if d < bd {
                    bd = d;
                    bt = t as i32;
                }
            }
            if bd <= 60 {
                et[ui] = bt;
                ew[ui] = 95.0;
            }
        }
    }

    let spare = own_military - own_city_tiles.len() as i32;
    let attack_mode = spare >= 3 || o.turn > 60;
    let need_army = threats || own_military < own_city_tiles.len() as i32 || (o.turn < 6 && own_military < 2);
    let mut free_slots = 0i32;
    for ct in &o.cities {
        if ct.owner == o.player {
            let mut cnt = 0;
            for u in &o.units {
                if u.owner == o.player && u.home == ct.tile as i32 + 1 {
                    cnt += 1;
                }
            }
            free_slots += (ct.level - cnt).max(0);
        }
    }

    let dv = dist_map(o.size, o.n, &c.villages);
    let de = dist_map(o.size, o.n, &c.enemy_cities);
    let mut du = dist_map(o.size, o.n, &c.unknown);
    if !c.unknown.is_empty() && !own_city_tiles.is_empty() {
        let mut w = vec![0i32; o.n];
        for &u in &c.unknown {
            let mut bd = 99;
            for &ct in &own_city_tiles {
                let d = dist(o.size, u, ct);
                if d < bd {
                    bd = d;
                }
            }
            w[u] = if o.turn < 10 { 2 * bd.min(8) } else { bd.min(6) };
        }
        for t in 0..o.n {
            let mut best = 999;
            for &u in &c.unknown {
                let val = dist(o.size, t, u) + w[u];
                if val < best {
                    best = val;
                }
            }
            du[t] = best;
        }
    }
    let dth = dist_map(o.size, o.n, &c.threats);
    let dcity = dist_map(o.size, o.n, &own_city_tiles);

    let mut best_i: i32 = -1;
    let mut best_s: f64 = -1.0e18;

    for (ai, a) in o.acts.iter().enumerate() {
        let score = match a.kind {
            10 => 0.0,
            11 => -1.0e9,
            9 => 1_000_000.0 + reward_prio(a.param, own_giants),
            8 => {
                let t = a.src.max(0) as usize;
                let mut v = 900_000.0;
                if t < o.n {
                    let tile = o.tiles[t];
                    if tile.structure == 1 {
                        if tile.owner > 0 && tile.owner != o.player {
                            v += 5000.0;
                        } else {
                            v += 4000.0;
                        }
                    } else if tile.structure == 2 {
                        v += 3000.0;
                    } else {
                        v += 2000.0;
                    }
                }
                v
            }
            3 => match a.param {
                16 => 800_000.0,
                8 => -1.0e9,
                _ => -1000.0,
            },
            2 => {
                let src = a.src.max(0) as usize;
                let dst = a.tgt.max(0) as usize;
                if src >= o.n || dst >= o.n {
                    -1.0
                } else {
                    let v = attack_value(&o, &c, src, dst);
                    if v > 6.0 {
                        700_000.0 + v * 100.0
                    } else {
                        -100.0
                    }
                }
            }
            5 => 730_000.0 + 100.0,
            7 => {
                let mut f = 0;
                let mut mo = 0;
                let mut wa = 0;
                let mut fi = 0;
                for i in 0..o.n {
                    let t = o.tiles[i];
                    if t.known && t.owner == o.player {
                        match t.terrain {
                            5 => f += 1,
                            4 => mo += 1,
                            1 => {
                                if t.resource == 3 {
                                    fi += 1
                                }
                            }
                            _ => {}
                        }
                    }
                    if t.known && (t.terrain == 1 || t.terrain == 2) {
                        wa += 1;
                    }
                }
                let ek = !c.enemy_cities.is_empty() || !c.enemy_units.is_empty();
                let p = tech_prio(&o, a.param.max(0) as usize, f, mo, wa, fi, ek);
                600_000.0 + p * 10.0
            }
            6 => {
                let t = a.tgt.max(0) as usize;
                let bv = if t < o.n { build_value(&o, a.param, t) } else { 10.0 };
                660_000.0 + bv * 10.0
            }
            4 => {
                let rv = recruit_value(a.param);
                if need_army {
                    810_000.0 + rv * 10.0
                } else if free_slots > 0 && o.stars >= 3 {
                    690_000.0 + rv * 10.0
                } else {
                    300_000.0 + rv * 10.0
                }
            }
            1 => {
                let src = a.src.max(0) as usize;
                let dst = a.tgt.max(0) as usize;
                if src >= o.n || dst >= o.n {
                    -1.0
                } else {
                    move_score(&o, &c, src, dst, &et, &ew, attack_mode, &own_city_tiles, &dv, &de, &du, &dth, &dcity)
                }
            }
            _ => -1.0,
        };
        if score > best_s {
            best_s = score;
            best_i = ai as i32;
        }
    }

    let chosen = if best_i >= 0 {
        o.acts[best_i as usize].clone()
    } else {
        o.acts
            .iter()
            .find(|a| a.kind == 10)
            .cloned()
            .unwrap_or_else(|| o.acts[0].clone())
    };
    // carry the plan along with the unit that moved
    if chosen.kind == 1 {
        let src = chosen.src;
        let dst = chosen.tgt;
        if src >= 0 && dst >= 0 && (src as usize) < o.n && (dst as usize) < o.n {
            mem.plan[dst as usize] = mem.plan[src as usize];
            mem.plan[src as usize] = -1;
        }
    } else if chosen.kind == 8 {
        let src = chosen.src;
        if src >= 0 && (src as usize) < o.n {
            mem.plan[src as usize] = -1;
        }
    }
    chosen.id
}

fn attack_value(o: &Obs, c: &Ctx, src: usize, dst: usize) -> f64 {
    let ai = c.unit_at[src];
    let di = c.unit_at[dst];
    if ai < 0 || di < 0 {
        return -1.0;
    }
    let au = o.units[ai as usize];
    let du = o.units[di as usize];
    if au.owner != o.player || du.owner == o.player {
        return -1.0;
    }
    let aa = atk_of(au.kind);
    if aa <= 0.0 {
        return -1.0;
    }
    let att_max = hp_of(au.kind) + if au.vet { 5.0 } else { 0.0 };
    let def_max = hp_of(du.kind) + if du.vet { 5.0 } else { 0.0 };
    let db = defense_bonus(o, dst);
    let (dmg, retal) = combat(aa, au.hp, att_max, def_of(du.kind), du.hp, def_max, db);
    let def_dies = du.hp - dmg <= 0.0;
    let mut v = 0.0;
    let attack_range = range_of(au.kind);
    if def_dies {
        v += 40.0 + 6.0 * cost_of(du.kind);
        if o.tiles[dst].structure == 1 {
            v += 30.0;
        }
        let post = au.hp;
        let mut adj_enemies = 0;
        for other in &o.units {
            if other.owner != o.player && other.owner > 0 && other.tile != dst {
                if dist(o.size, other.tile, dst) <= 1 {
                    adj_enemies += 1;
                }
            }
        }
        if attack_range < 2 && adj_enemies > 0 && post < 8.0 {
            v -= 20.0 * adj_enemies as f64;
        }
    } else {
        v += dmg * 1.7 - retal * 1.9;
        if au.hp - retal <= 0.0 {
            v -= 500.0 + 20.0 * cost_of(au.kind);
        }
    }
    for &ct in &c.own_cities {
        if dist(o.size, dst, ct) <= 2 {
            v += 22.0;
            break;
        }
    }
    v
}

#[allow(clippy::too_many_arguments)]
fn move_score(
    o: &Obs,
    c: &Ctx,
    src: usize,
    dst: usize,
    et: &[i32],
    ew: &[f64],
    attack_mode: bool,
    own_cities: &[usize],
    dv: &[i32],
    de: &[i32],
    du: &[i32],
    dth: &[i32],
    dcity: &[i32],
) -> f64 {
    let ui = c.unit_at[src];
    let mut gain = 0.0;
    let mut targeted = false;
    if ui >= 0 {
        let t = et[ui as usize];
        if t >= 0 {
            let t = t as usize;
            gain += ew[ui as usize] * (dist(o.size, src, t) - dist(o.size, dst, t)) as f64;
            targeted = true;
        }
    }
    if !targeted {
        if attack_mode && !c.enemy_cities.is_empty() {
            gain += 45.0 * (de[src] - de[dst]) as f64;
        } else if !c.unknown.is_empty() {
            gain += 13.0 * (du[src] - du[dst]) as f64;
        } else if !c.enemy_cities.is_empty() {
            gain += 20.0 * (de[src] - de[dst]) as f64;
        }
    }
    if !c.villages.is_empty() && !targeted {
        gain += 16.0 * (dv[src] - dv[dst]) as f64;
    }

    let mut adj_enemies = 0;
    for e in &o.units {
        if e.owner != o.player && e.owner > 0 && dist(o.size, e.tile, dst) <= 1 {
            adj_enemies += 1;
        }
    }
    let mut support = 0;
    for u in &o.units {
        if u.owner == o.player && u.tile != src && dist(o.size, u.tile, dst) <= 1 {
            support += 1;
        }
    }
    if ui >= 0 {
        let u = o.units[ui as usize];
        let maxhp = hp_of(u.kind) + if u.vet { 5.0 } else { 0.0 };
        if adj_enemies > 0 {
            if support == 0 {
                gain -= 40.0 * adj_enemies as f64;
            }
            if u.hp < maxhp * 0.7 {
                gain -= 90.0 * adj_enemies as f64;
            }
        }
        // avoid stepping right next to a group of enemies with no company
        if support == 0 {
            let mut near2 = 0;
            for e in &o.units {
                if e.owner != o.player && e.owner > 0 && dist(o.size, e.tile, dst) <= 2 {
                    near2 += 1;
                }
            }
            if near2 >= 2 {
                gain -= 25.0 * near2 as f64;
            }
        }
        if u.hp < maxhp * 0.7 && !own_cities.is_empty() {
            gain += 26.0 * (dcity[src] - dcity[dst]) as f64;
        }
        // stay in company: avoid isolated positions when enemies are around
        if !c.enemy_units.is_empty() {
            gain += 8.0 * support as f64;
        } else {
            let mut crowd = 0;
            for v in &o.units {
                if v.owner == o.player && v.tile != src && dist(o.size, v.tile, dst) <= 2 {
                    crowd += 1;
                }
            }
            gain -= 14.0 * crowd as f64;
        }
    }

    if o.tiles[src].structure == 1 && !c.threats.is_empty() {
        if dth[dst] >= dth[src] {
            gain -= 250.0;
        }
    }
    if !c.enemy_units.is_empty() {
        let mut guards = vec![0i32; own_cities.len()];
        for u in &o.units {
            if u.owner == o.player {
                for (k, &ct) in own_cities.iter().enumerate() {
                    if dist(o.size, u.tile, ct) <= 1 {
                        guards[k] += 1;
                    }
                }
            }
        }
        for (k, &ct) in own_cities.iter().enumerate() {
            if guards[k] > 0
                && dist(o.size, src, ct) <= 1
                && dist(o.size, dst, ct) > 1
                && guards[k] <= 1
                && c.enemy_units.iter().any(|&e| dist(o.size, e, ct) <= 6)
            {
                gain -= 160.0;
            }
        }
    }
    100_000.0 + gain * 100.0
}

struct Mem {
    turn: i32,
    plan: Vec<i32>,
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Mem { turn: 0, plan: vec![-1; 256] };
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = choose(&v, &mut mem);
        writeln!(stdout, "{}", json!({ "action_id": id })).unwrap();
        stdout.flush().unwrap();
    }
}
