// Polytopia Bardur 1v1 skirmish bot.
// One JSON observation per line on stdin; reply {"action_id":"..."} and flush.
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

const INF: i32 = 1_000_000;

// ---------------- helpers on JSON ----------------
fn gi(v: &Value, k: &str) -> i64 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(0)
}
fn gb(v: &Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}
fn gf(v: &Value, k: &str) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0)
}

// ---------------- unit tables ----------------
// (health, attack, defense, movement, range, cost)
fn u_stat(kind: i32) -> (f32, f32, f32, i32, i32, i32) {
    match kind {
        2 => (10.0, 2.0, 2.0, 1, 1, 2),
        3 => (10.0, 2.0, 1.0, 2, 1, 3),
        4 => (10.0, 3.5, 1.0, 3, 1, 8),
        5 => (15.0, 1.0, 3.0, 1, 1, 3),
        8 => (10.0, 4.0, 0.0, 1, 3, 8),
        9 => (10.0, 2.0, 1.0, 1, 2, 3),
        10 => (10.0, 0.0, 1.0, 1, 1, 5),
        11 => (15.0, 3.0, 3.0, 1, 1, 5),
        12 => (40.0, 5.0, 4.0, 1, 1, 10),
        37 => (10.0, 2.0, 2.0, 1, 1, 2),
        38 => (5.0, 0.0, 0.5, 2, 1, 8),
        39 => (5.0, 0.0, 0.5, 2, 1, 2),
        40 => (10.0, 2.0, 2.0, 2, 1, 2),
        41 => (10.0, 3.0, 2.0, 2, 3, 15),
        42 => (10.0, 2.0, 1.0, 3, 2, 5),
        43 => (10.0, 0.0, 1.0, 2, 2, 0),
        44 => (10.0, 3.0, 3.0, 3, 1, 5),
        45 => (40.0, 4.0, 4.0, 2, 1, 10),
        _ => (10.0, 1.0, 1.0, 1, 1, 5),
    }
}
fn u_dash(kind: i32) -> bool { matches!(kind, 2 | 3 | 4 | 9 | 11 | 20 | 21 | 28 | 29 | 31 | 35 | 56) }
fn u_stiff(kind: i32) -> bool { matches!(kind, 8 | 10 | 17 | 22 | 39 | 41 | 43 | 60 | 61 | 62) }
fn u_fortify(kind: i32) -> bool { matches!(kind, 2 | 3 | 4 | 5 | 9 | 20 | 25 | 31) }
fn u_surprise(kind: i32) -> bool { matches!(kind, 30 | 37 | 40 | 61 | 62) }
fn u_float(kind: i32) -> bool { matches!(kind, 26 | 35 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 57) }
fn u_carry(kind: i32) -> bool { matches!(kind, 39 | 40 | 41 | 42 | 43 | 44 | 45) }
fn u_scout(kind: i32) -> bool { matches!(kind, 18 | 19 | 24 | 29 | 38 | 39 | 42) }

fn tech_tier(id: i32) -> i32 {
    match id {
        0 => 0,
        1 | 6 | 10 | 20 | 25 | 26 | 28 | 30 | 31 | 32 | 35 | 36 | 37 | 41 | 42 | 48 | 49 => 1,
        2 | 4 | 7 | 8 | 13 | 16 | 21 | 23 | 44 | 121 => 2,
        3 | 5 | 9 | 12 | 14 | 17 | 19 | 22 | 38 => 3,
        _ => 1,
    }
}

// research priority (higher = sooner)
fn tech_prio(id: i32) -> f64 {
    match id {
        1 => 100.0,
        6 => 96.0,
        18 => 92.0,
        10 => 90.0,
        20 => 88.0,
        7 => 84.0,
        16 => 82.0,
        23 => 80.0,
        2 => 78.0,
        3 => 76.0,
        24 => 74.0,
        21 => 70.0,
        22 => 68.0,
        17 => 64.0,
        8 => 60.0,
        9 => 56.0,
        4 => 50.0,
        13 => 46.0,
        44 => 44.0,
        5 => 30.0,
        _ => 10.0,
    }
}

// ---------------- state ----------------
#[derive(Clone)]
struct Unit {
    tile: usize,
    kind: i32,
    hp: f32,
    maxhp: f32,
    moved: bool,
    attacked: bool,
    kills: i32,
    home: i32,
    passenger: i32,
    veteran: bool,
}
struct City {
    tile: usize,
    level: i32,
    production: i32,
    progress: i32,
    population: i32,
    connected: bool,
}
struct Act {
    id: String,
    kind: i32,
    src: i32,
    tgt: i32,
    par: i32,
}
struct St {
    size: usize,
    n: usize,
    me: i32,
    opp: i32,
    turn: i32,
    stars: i32,
    tech: Vec<bool>,
    known: Vec<bool>,
    terrain: Vec<i32>,
    owner: Vec<i32>,
    structure: Vec<i32>,
    resource: Vec<i32>,
    capital: Vec<i32>,
    my_units: Vec<Unit>,
    en_units: Vec<Unit>,
    my_cities: Vec<City>,
    en_cities: Vec<usize>,
    vil: Vec<usize>,
    acts: Vec<Act>,
}

#[derive(Default)]
struct Mem {
    last_move: HashMap<usize, (usize, i32)>,
}

fn parse(obs: &Value) -> St {
    let vs = &obs["visible_state"];
    let size = gi(vs, "board_size") as usize;
    let n = size * size;
    let me = gi(obs, "player") as i32;
    let mut tech = vec![false; 128];
    if let Some(a) = vs["technology"].as_array() {
        for t in a {
            let id = t.as_i64().unwrap_or(-1);
            if id >= 0 && id < 128 {
                tech[id as usize] = true;
            }
        }
    }
    let mut known = vec![false; n];
    let mut terrain = vec![0i32; n];
    let mut owner = vec![0i32; n];
    let mut structure = vec![0i32; n];
    let mut resource = vec![0i32; n];
    let mut capital = vec![0i32; n];
    if let Some(a) = vs["tiles"].as_array() {
        for t in a {
            let i = gi(t, "tile") as usize;
            if i >= n {
                continue;
            }
            if gb(t, "known") {
                known[i] = true;
                terrain[i] = gi(t, "terrain") as i32;
                owner[i] = gi(t, "owner") as i32;
                structure[i] = gi(t, "structure") as i32;
                resource[i] = gi(t, "resource") as i32;
                capital[i] = gi(t, "capital") as i32;
            }
        }
    }
    let mut my_units = Vec::new();
    let mut en_units = Vec::new();
    if let Some(a) = vs["units"].as_array() {
        for u in a {
            let i = gi(u, "tile") as usize;
            if i >= n {
                continue;
            }
            let kind = gi(u, "kind") as i32;
            let hp = gf(u, "health") as f32;
            let (mh, _, _, _, _, _) = u_stat(kind);
            let unit = Unit {
                tile: i,
                kind,
                hp,
                maxhp: mh + if gb(u, "veteran") { 5.0 } else { 0.0 },
                moved: gb(u, "moved"),
                attacked: gb(u, "attacked"),
                kills: gi(u, "kills") as i32,
                home: gi(u, "home") as i32,
                passenger: gi(u, "passenger") as i32,
                veteran: gb(u, "veteran"),
            };
            if gi(u, "owner") as i32 == me {
                my_units.push(unit);
            } else {
                en_units.push(unit);
            }
        }
    }
    let mut my_cities = Vec::new();
    let mut en_cities = Vec::new();
    let mut vil = Vec::new();
    if let Some(a) = vs["cities"].as_array() {
        for c in a {
            let i = gi(c, "tile") as usize;
            if i >= n {
                continue;
            }
            let o = gi(c, "owner") as i32;
            if o == me {
                my_cities.push(City {
                    tile: i,
                    level: gi(c, "level") as i32,
                    production: gi(c, "production") as i32,
                    progress: gi(c, "progress") as i32,
                    population: gi(c, "population") as i32,
                    connected: gb(c, "connected"),
                });
            } else if o == 0 {
                vil.push(i);
            } else {
                en_cities.push(i);
            }
        }
    }
    let mut acts = Vec::new();
    if let Some(a) = obs["legal_actions"].as_array() {
        for x in a {
            let c = &x["command"];
            acts.push(Act {
                id: x["action_id"].as_str().unwrap_or("").to_string(),
                kind: gi(c, "kind") as i32,
                src: gi(c, "source") as i32,
                tgt: gi(c, "target") as i32,
                par: gi(c, "parameter") as i32,
            });
        }
    }
    St {
        size,
        n,
        me,
        opp: 3 - me,
        turn: gi(vs, "turn") as i32,
        stars: gi(vs, "stars") as i32,
        tech,
        known,
        terrain,
        owner,
        structure,
        resource,
        capital,
        my_units,
        en_units,
        my_cities,
        en_cities,
        vil,
        acts,
    }
}

// ---------------- geometry / distances ----------------
fn cheb(st: &St, a: usize, b: usize) -> i32 {
    let x1 = (a % st.size) as i32;
    let y1 = (a / st.size) as i32;
    let x2 = (b % st.size) as i32;
    let y2 = (b / st.size) as i32;
    (x1 - x2).abs().max((y1 - y2).abs())
}
fn land_ok(st: &St, i: usize) -> bool {
    match st.terrain[i] {
        3 | 5 => true,
        4 => st.tech[20],
        1 | 2 => false,
        _ => true,
    }
}
fn water_ok(st: &St, i: usize) -> bool {
    st.terrain[i] == 1 || st.terrain[i] == 2
}
// BFS over known tiles; mode 0 land, 1 water, 2 any. Sources may be unknown tiles.
fn bfs(st: &St, srcs: &[usize], mode: u8) -> Vec<i32> {
    let mut dist = vec![INF; st.n];
    let mut q: Vec<usize> = Vec::with_capacity(st.n);
    for &s in srcs {
        if s < st.n && dist[s] != 0 {
            dist[s] = 0;
            q.push(s);
        }
    }
    let mut head = 0usize;
    while head < q.len() {
        let t = q[head];
        head += 1;
        let d = dist[t] + 1;
        let x = (t % st.size) as i32;
        let y = (t / st.size) as i32;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= st.size as i32 || ny >= st.size as i32 {
                    continue;
                }
                let i = (ny * st.size as i32 + nx) as usize;
                if dist[i] <= d || !st.known[i] {
                    continue;
                }
                let ok = match mode {
                    0 => land_ok(st, i),
                    1 => water_ok(st, i),
                    _ => true,
                };
                if !ok {
                    continue;
                }
                dist[i] = d;
                q.push(i);
            }
        }
    }
    dist
}
fn unknown_near(st: &St, tile: usize) -> i32 {
    let r = if st.terrain[tile] == 4 { 2 } else { 1 };
    let size = st.size as i32;
    let cx = (tile % st.size) as i32;
    let cy = (tile / st.size) as i32;
    let mut c = 0;
    for y in (cy - r).max(0)..=(cy + r).min(size - 1) {
        for x in (cx - r).max(0)..=(cx + r).min(size - 1) {
            if !st.known[(y * size + x) as usize] {
                c += 1;
            }
        }
    }
    c
}
fn find_unit(st: &St, tile: usize) -> Option<&Unit> {
    st.my_units.iter().find(|u| u.tile == tile)
}
fn find_enemy(st: &St, tile: usize) -> Option<&Unit> {
    st.en_units.iter().find(|u| u.tile == tile)
}

// ---------------- combat ----------------
fn eval_attack(st: &St, u: &Unit, e: &Unit, from: usize) -> f64 {
    let (_, aatk, _, _, _, acost) = u_stat(u.kind);
    let (_, _, ddef, _, drange, dcost) = u_stat(e.kind);
    let amax = if u.maxhp > 1.0 { u.maxhp } else { 1.0 };
    let dmax = if e.maxhp > 1.0 { e.maxhp } else { 1.0 };
    let af = aatk * (u.hp / amax);
    let db = if st.structure[e.tile] == 1 && st.owner[e.tile] == st.opp && u_fortify(e.kind) {
        1.5
    } else {
        1.0
    };
    let df = ddef * (e.hp / dmax) * db;
    let total = af + df;
    if total <= 0.0 {
        return 0.0;
    }
    let dmg = ((af / total) * aatk * 4.5 + 0.5).floor();
    let kills = e.hp <= dmg;
    let ret = if !kills && !u_stiff(e.kind) && !u_surprise(u.kind) && cheb(st, from, e.tile) <= drange {
        ((df / total) * ddef * 4.5 + 0.5).floor()
    } else {
        0.0
    };
    if kills {
        if ret >= u.hp {
            50.0 * dcost as f64 - 45.0 * acost as f64 - 40.0
        } else {
            55.0 * dcost as f64 + 55.0 - 0.5 * ret as f64
        }
    } else if ret >= u.hp {
        -150.0
    } else if dmg * 2.0 < e.hp {
        -100.0
    } else {
        6.0 * dmg as f64 - 0.8 * ret as f64 - 38.0
    }
}

// ---------------- context ----------------
struct Ctx {
    d_vil: Vec<i32>,
    d_en: Vec<i32>,
    d_enc: Vec<i32>,
    d_unk: Vec<i32>,
    d_unkw: Vec<i32>,
    attack_mode: bool,
    defend: bool,
    navy: bool,
    sail: bool,
    cap_tile: usize,
    has_cap: bool,
    center: usize,
}

fn reward_val(st: &St, a: &Act) -> f64 {
    let mut v = match a.par {
        8 => 105.0,
        4 => 100.0,
        7 => 92.0,
        6 => 88.0,
        2 => 80.0,
        3 => 74.0,
        5 => 70.0,
        1 => 60.0,
        _ => 50.0,
    };
    let t = a.tgt as usize;
    if t < st.n {
        if a.par == 4 && unknown_near(st, t) > 0 {
            v += 12.0;
        }
        if a.par == 5 && unknown_near(st, t) > 0 {
            v += 8.0;
        }
        if a.par == 6 && st.en_cities.len() > 0 {
            v += 6.0;
        }
    }
    v
}

fn research_ok(st: &St, tech: i32) -> bool {
    let tier = tech_tier(tech);
    let cities = st.my_cities.len() as i32;
    let cost = 4 + cities * tier;
    let reserve = match tier {
        1 => {
            if st.my_units.len() <= 3 {
                0
            } else {
                3
            }
        }
        2 => 4,
        _ => 6,
    };
    if st.stars < cost + reserve {
        return false;
    }
    if tier >= 3 && cities > 4 {
        return false;
    }
    if tier >= 2 && cities > 7 {
        return false;
    }
    true
}

fn count_struct(st: &St, kind: i32) -> i32 {
    st.structure
        .iter()
        .enumerate()
        .filter(|(i, &s)| s == kind && st.owner[*i] == st.me)
        .count() as i32
}

fn water_fraction(st: &St) -> f64 {
    let mut known = 0;
    let mut water = 0;
    for i in 0..st.n {
        if st.known[i] {
            known += 1;
            if water_ok(st, i) {
                water += 1;
            }
        }
    }
    if known == 0 {
        0.0
    } else {
        water as f64 / known as f64
    }
}

fn build_val(st: &St, a: &Act, ctx: &Ctx) -> Option<f64> {
    let tile = a.tgt as usize;
    if tile >= st.n {
        return None;
    }
    let cities = st.my_cities.len() as i32;
    match a.par {
        21 => {
            // Mine: metal underfoot -> +2 population per level
            if st.tech[23] && st.resource[tile] == 5 && count_struct(st, 21) < cities + 1 {
                Some(95.0)
            } else {
                None
            }
        }
        5 => {
            // Farm: crop -> +2 population per level
            if st.tech[8] && st.resource[tile] == 2 && count_struct(st, 5) < cities + 1 {
                Some(95.0)
            } else {
                None
            }
        }
        12 => {
            // Lumber hut: forest -> +1 population per level
            if st.tech[16] && st.terrain[tile] == 5 && count_struct(st, 12) < 2 * cities + 1 {
                Some(72.0)
            } else {
                None
            }
        }
        13 => {
            // Sawmill next to a lumber hut
            if st.tech[16] && count_struct(st, 13) < cities {
                let mut ok = false;
                let size = st.size as i32;
                let x = (tile % st.size) as i32;
                let y = (tile / st.size) as i32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx < 0 || ny < 0 || nx >= size || ny >= size {
                            continue;
                        }
                        let j = (ny * size + nx) as usize;
                        if st.structure[j] == 12 && st.owner[j] == st.me {
                            ok = true;
                        }
                    }
                }
                if ok {
                    Some(64.0)
                } else {
                    None
                }
            } else {
                None
            }
        }
        8 => {
            // Port: shallow water in our territory
            if st.tech[10] && st.terrain[tile] == 1 && count_struct(st, 8) < 2 && ctx.navy {
                Some(80.0)
            } else {
                None
            }
        }
        17 => {
            // Temple
            if st.tech[2] && st.terrain[tile] == 3 && count_struct(st, 17) < cities {
                Some(45.0)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn summon_val(st: &St, a: &Act) -> Option<f64> {
    let kind = a.par;
    let cost = u_stat(kind).5;
    if cost > st.stars {
        return None;
    }
    let city = a.src as usize;
    let mut housed = 0;
    for u in &st.my_units {
        if u.home == city as i32 + 1 {
            housed += 1;
        }
    }
    let level = st
        .my_cities
        .iter()
        .find(|c| c.tile == city)
        .map(|c| c.level)
        .unwrap_or(1);
    let free = level + 1 - housed;
    if free <= 0 {
        return None;
    }
    let n = st.my_units.len() as i32;
    let count_of = |k: i32| st.my_units.iter().filter(|u| u.kind == k).count() as i32;
    let v = match kind {
        9 => 86.0,  // archer
        4 => 100.0, // knight
        3 => 90.0,  // rider
        2 => 80.0,  // warrior
        5 => 68.0,  // defender
        11 => 92.0, // swordsman
        8 => 80.0,  // catapult
        10 => 104.0, // mind bender
        43 => 12.0, // raft (free, but takes a slot)
        _ => 30.0,
    };
    if kind == 10 && count_of(10) >= 2 {
        return None;
    }
    if kind == 8 && count_of(8) >= 2 {
        return None;
    }
    if kind == 5 && count_of(5) * 4 > n {
        return None;
    }
    if kind == 43 {
        return None;
    }
    let mut bonus = free as f64 * 4.0 + (n as f64) * 0.2;
    if st.capital.get(city).copied().unwrap_or(0) == st.me {
        bonus += 4.0;
    }
    Some(v + bonus)
}

// ---------------- movement valuation ----------------
fn move_val(st: &St, ctx: &Ctx, mem: &Mem, a: &Act) -> Option<f64> {
    let src = a.src as usize;
    let tgt = a.tgt as usize;
    if src >= st.n || tgt >= st.n || src == tgt {
        return None;
    }
    let u = find_unit(st, src)?;
    let mut best = f64::MIN;

    // a boat with a passenger: sail outward, land on new shores
    if u.kind == 43 && u.passenger > 0 {
        let a1 = ctx.d_unkw[src];
        let a2 = ctx.d_unkw[tgt];
        if a1 < INF && a2 < INF {
            best = best.max((a1 - a2) as f64 * 12.0 + unknown_near(st, tgt) as f64 * 10.0);
        }
        if st.terrain[tgt] >= 3 {
            best = best.max(280.0 + unknown_near(st, tgt) as f64 * 10.0);
        }
        if let Some((prev, _)) = mem.last_move.get(&src) {
            if *prev == tgt {
                best -= 150.0;
            }
        }
        if best <= 0.0 {
            return None;
        }
        return Some(best);
    }

    // tactical: attack from the destination (dash units may move and attack)
    if !u.attacked && u_dash(u.kind) {
        let arange = u_stat(u.kind).4;
        for e in &st.en_units {
            if st.known[e.tile] && cheb(st, tgt, e.tile) <= arange {
                let v = eval_attack(st, u, e, tgt);
                if v > 0.0 {
                    best = best.max(v + 170.0);
                }
            }
        }
    }

    let expand_w = if ctx.attack_mode { 12.0 } else if ctx.defend { 20.0 } else { 26.0 };
    let explore_w = if ctx.attack_mode { 8.0 } else if ctx.defend { 10.0 } else { 13.0 };
    let attack_w = if ctx.attack_mode { 30.0 } else if ctx.defend { 18.0 } else { 10.0 };

    // objective: expand towards the nearest village
    {
        let a1 = ctx.d_vil[src];
        let a2 = ctx.d_vil[tgt];
        if a1 < INF && a2 < INF {
            let mut e = (a1 - a2) as f64 * expand_w;
            if st.structure[tgt] == 1 && st.owner[tgt] == 0 {
                e += if ctx.attack_mode { 40.0 } else { 40.0 };
            }
            best = best.max(e);
        }
    }
    // objective: explore towards the unknown
    {
        let a1 = ctx.d_unk[src];
        let a2 = ctx.d_unk[tgt];
        if a1 < INF && a2 < INF {
            let e = (a1 - a2) as f64 * explore_w + unknown_near(st, tgt) as f64 * 4.0;
            best = best.max(e);
        }
    }
    // objective: attack
    if !st.en_units.is_empty() {
        let w = attack_w;
        let a1 = ctx.d_en[src];
        let a2 = ctx.d_en[tgt];
        if a1 < INF && a2 < INF {
            best = best.max((a1 - a2) as f64 * w);
        }
        if !st.en_cities.is_empty() {
            let b1 = ctx.d_enc[src];
            let b2 = ctx.d_enc[tgt];
            if b1 < INF && b2 < INF {
                best = best.max((b1 - b2) as f64 * (if ctx.attack_mode { 36.0 } else { 10.0 }));
            }
        }
    }
    // embarking on our own port
    if ctx.sail && st.structure[tgt] == 8 && st.owner[tgt] == st.me && (u.kind == 2 || u.kind == 3) {
        best = best.max(420.0);
    }
    // keep going in the same general direction; never step back where we came from
    if let Some((prev, _)) = mem.last_move.get(&src) {
        let dx1 = (src % st.size) as i32 - (*prev % st.size) as i32;
        let dy1 = (src / st.size) as i32 - (*prev / st.size) as i32;
        let dx2 = (tgt % st.size) as i32 - (src % st.size) as i32;
        let dy2 = (tgt / st.size) as i32 - (src / st.size) as i32;
        best += (dx1 * dx2 + dy1 * dy2) as f64 * 7.0;
        if *prev == tgt {
            best -= 160.0;
        }
    }
    if best <= 0.0 {
        return None;
    }
    Some(best)
}

// ---------------- decision ----------------
fn decide(st: &St, mem: &mut Mem) -> String {
    let my_power: f64 = st
        .my_units
        .iter()
        .map(|u| {
            let (_, a, d, _, _, _) = u_stat(u.kind);
            (a as f64 + 0.6 * d as f64) * (u.hp as f64 / u.maxhp.max(1.0) as f64)
        })
        .sum::<f64>()
        + 1.5 * st.my_cities.len() as f64;
    let en_power: f64 = st
        .en_units
        .iter()
        .map(|u| {
            let (_, a, d, _, _, _) = u_stat(u.kind);
            (a as f64 + 0.6 * d as f64) * (u.hp as f64 / u.maxhp.max(1.0) as f64)
        })
        .sum::<f64>();
    let attack_mode = !st.en_units.is_empty()
        && my_power >= en_power * 1.25
        && st.my_units.len() > st.en_units.len();
    // an enemy raiding our cities: defend
    let mut near_city = 99i32;
    for e in &st.en_units {
        for c in &st.my_cities {
            let d = cheb(st, e.tile, c.tile);
            if d < near_city {
                near_city = d;
            }
        }
    }
    let defend = near_city <= 3;

    let en_tiles: Vec<usize> = st.en_units.iter().map(|u| u.tile).collect();
    let unk: Vec<usize> = (0..st.n).filter(|&i| !st.known[i]).collect();
    let enemy_unknown = st.en_units.is_empty() && st.en_cities.is_empty();
    let ctx = Ctx {
        d_vil: bfs(st, &st.vil, 0),
        d_en: bfs(st, &en_tiles, 0),
        d_enc: bfs(st, &st.en_cities, 0),
        d_unk: bfs(st, &unk, 2),
        d_unkw: bfs(st, &unk, 1),
        attack_mode: attack_mode,
        defend: defend,
        navy: st.tech[10] && enemy_unknown && st.turn > 12 && water_fraction(st) > 0.12,
        sail: st.tech[10] && enemy_unknown && st.turn > 18 && count_struct(st, 8) > 0,
        cap_tile: 0,
        has_cap: false,
        center: ((st.size as i32 - 1) / 2) as usize * st.size + ((st.size as i32 - 1) / 2) as usize,
    };

    let mut best_s = -1e18f64;
    let mut best_i = 0usize;
    for (i, a) in st.acts.iter().enumerate() {
        let mut s = -1e18f64;
        match a.kind {
            9 => s = 9.0e9 + reward_val(st, a),
            8 => {
                let tile = a.src as usize;
                if tile < st.n && st.en_cities.contains(&tile) {
                    let cap = st.capital.get(tile).copied().unwrap_or(0);
                    s = 3.0e9 + if cap == st.opp { 400.0 } else { 0.0 };
                } else if tile < st.n && st.structure[tile] == 2 {
                    s = 1.6e9;
                } else if tile < st.n && st.resource[tile] == 8 {
                    s = 1.7e9;
                } else {
                    s = 2.0e9;
                }
            }
            2 => {
                if let (Some(u), Some(e)) = (find_unit(st, a.src as usize), find_enemy(st, a.tgt as usize)) {
                    let v = eval_attack(st, u, e, a.src as usize);
                    if v > 0.0 {
                        s = 2.2e9 + v;
                    }
                }
            }
            7 => {
                let p = tech_prio(a.par);
                if p > 20.0 && research_ok(st, a.par) {
                    s = 1.55e9 + p;
                }
            }
            5 => s = 1.5e9 + 130.0,
            6 => {
                if let Some(v) = build_val(st, a, &ctx) {
                    s = 1.0e9 + v;
                }
            }
            4 => {
                if let Some(v) = summon_val(st, a) {
                    s = 1.45e9 + v;
                }
            }
            3 => {
                if a.par == 16 {
                    s = 0.9e9 + 60.0;
                }
            }
            1 => {
                if let Some(v) = move_val(st, &ctx, mem, a) {
                    s = 0.5e9 + v;
                }
            }
            10 => s = 0.0,
            _ => {}
        }
        if s >= best_s {
            best_s = s;
            best_i = i;
        }
    }
    if st.acts.is_empty() {
        return String::new();
    }
    let act = &st.acts[best_i];
    if act.kind == 1 {
        mem.last_move
            .insert(act.tgt as usize, (act.src as usize, st.turn));
    }
    act.id.clone()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Mem::default();
    let mut buf = String::new();
    loop {
        buf.clear();
        let read = stdin.lock().read_line(&mut buf);
        match read {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = buf.trim();
        if line.is_empty() {
            continue;
        }
        let id = match serde_json::from_str::<Value>(line) {
            Ok(obs) => {
                let st = parse(&obs);
                decide(&st, &mut mem)
            }
            Err(_) => String::new(),
        };
        writeln!(stdout, "{}", json!({ "action_id": id })).unwrap();
        stdout.flush().unwrap();
    }
}
