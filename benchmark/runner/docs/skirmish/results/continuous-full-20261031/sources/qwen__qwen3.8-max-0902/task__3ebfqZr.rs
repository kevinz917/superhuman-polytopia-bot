// Bardur 1v1 skirmish bot: single-action heuristic policy.
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

const NK: usize = 128;
static UATK: [f64; 48] = [
    0.0, 0.0, 2.0, 2.0, 3.5, 1.0, 0.0, 0.0, 4.0, 2.0, // 0..9  (idx=kind)
    0.0, 3.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 10..19
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 20..29
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, // 30..39
    2.0, 3.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, // 40..47
];
static UDEF: [f64; 48] = [
    0.0, 0.0, 2.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0,
    1.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.5,
    2.0, 2.0, 1.0, 1.0, 3.0, 4.0, 0.0, 0.0,
];
static UHP: [f64; 48] = [
    1.0, 0.0, 10.0, 10.0, 10.0, 15.0, 0.0, 0.0, 10.0, 10.0,
    10.0, 15.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 5.0, 5.0,
    10.0, 10.0, 10.0, 10.0, 10.0, 40.0, 0.0, 0.0,
];
static UCOST: [i32; 48] = [
    0, 0, 2, 3, 8, 3, 0, 0, 8, 3, 5, 5, 10, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8, 2,
    2, 15, 5, 0, 5, 10, 0, 0,
];
static URANGE: [i32; 48] = [
    1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 3, 2, 2, 1, 1, 1, 1,
];

fn atk(k: i32) -> f64 { if k >= 0 && (k as usize) < 48 { UATK[k as usize] } else { 0.0 } }
fn def(k: i32) -> f64 { if k >= 0 && (k as usize) < 48 { UDEF[k as usize] } else { 0.0 } }
fn hp(k: i32) -> f64 { if k >= 0 && (k as usize) < 48 { UHP[k as usize] } else { 1.0 } }
fn cost(k: i32) -> i32 { if k >= 0 && (k as usize) < 48 { UCOST[k as usize] } else { 0 } }
fn rng(k: i32) -> i32 { if k >= 0 && (k as usize) < 48 { URANGE[k as usize] } else { 1 } }

struct Obs {
    size: usize,
    n: usize,
    me: i32,
    turn: i32,
    stars: i32,
    known: Vec<bool>,
    terrain: Vec<i32>,
    owner: Vec<i32>,
    resource: Vec<i32>,
    structure: Vec<i32>,
    road: Vec<bool>,
    u_owner: Vec<i32>,
    u_kind: Vec<i32>,
    u_health: Vec<f64>,
    tech: Vec<bool>,
    my_cities: Vec<usize>,
    my_units: Vec<usize>,
    enemy_cities: Vec<usize>,
    enemy_units: Vec<usize>,
    villages: Vec<usize>,
    water_frac: f64,
}

impl Obs {
    fn parse(v: &Value) -> Obs {
        let vis = &v["visible_state"];
        let size = vis["board_size"].as_i64().unwrap_or(14) as usize;
        let n = size * size;
        let mut o = Obs {
            size, n,
            me: v["player"].as_i64().unwrap_or(1) as i32,
            turn: vis["turn"].as_i64().unwrap_or(0) as i32,
            stars: vis["stars"].as_i64().unwrap_or(0) as i32,
            known: vec![false; n], terrain: vec![0; n], owner: vec![0; n],
            resource: vec![0; n], structure: vec![0; n], road: vec![false; n],
            u_owner: vec![0; n], u_kind: vec![0; n], u_health: vec![0.0; n],
            tech: vec![false; NK], my_cities: Vec::new(), my_units: Vec::new(),
            enemy_cities: Vec::new(), enemy_units: Vec::new(), villages: Vec::new(),
            water_frac: 0.0,
        };
        if let Some(t) = vis["tiles"].as_array() {
            for e in t {
                let i = e["tile"].as_i64().unwrap_or(-1);
                if i < 0 || (i as usize) >= n { continue; }
                let i = i as usize;
                o.known[i] = e["known"].as_bool().unwrap_or(false);
                if !o.known[i] { continue; }
                o.terrain[i] = e["terrain"].as_i64().unwrap_or(0) as i32;
                o.owner[i] = e["owner"].as_i64().unwrap_or(0) as i32;
                o.resource[i] = e["resource"].as_i64().unwrap_or(0) as i32;
                o.structure[i] = e["structure"].as_i64().unwrap_or(0) as i32;
                o.road[i] = e["road"].as_i64().unwrap_or(0) != 0;
            }
        }
        if let Some(t) = vis["units"].as_array() {
            for e in t {
                let i = e["tile"].as_i64().unwrap_or(-1);
                if i < 0 || (i as usize) >= n { continue; }
                let i = i as usize;
                o.u_owner[i] = e["owner"].as_i64().unwrap_or(0) as i32;
                o.u_kind[i] = e["kind"].as_i64().unwrap_or(0) as i32;
                o.u_health[i] = e["health"].as_f64().unwrap_or(0.0);
            }
        }
        if let Some(t) = vis["technology"].as_array() {
            for e in t {
                let k = e.as_i64().unwrap_or(-1);
                if k >= 0 && (k as usize) < NK { o.tech[k as usize] = true; }
            }
        }
        let mut water = 0.0;
        let mut land = 0.0;
        for i in 0..n {
            if !o.known[i] { continue; }
            if o.terrain[i] <= 2 { water += 1.0; } else { land += 1.0; }
            if o.structure[i] == 1 {
                if o.owner[i] == o.me { o.my_cities.push(i); } else {
                    if o.owner[i] > 0 { o.enemy_cities.push(i); } else { o.villages.push(i); }
                }
            }
        }
        o.water_frac = if water + land > 0.0 { water / (water + land) } else { 0.0 };
        for i in 0..n {
            if o.u_owner[i] == 0 { continue; }
            if o.u_owner[i] == o.me { o.my_units.push(i); } else { o.enemy_units.push(i); }
        }
        if let Some(t) = vis["cities"].as_array() {
            for e in t {
                let i = e["tile"].as_i64().unwrap_or(-1);
                if i < 0 || (i as usize) >= n { continue; }
                let i = i as usize;
                let ow = e["owner"].as_i64().unwrap_or(0) as i32;
                if ow == o.me {
                    if !o.my_cities.contains(&i) { o.my_cities.push(i); }
                }
            }
        }
        o
    }

    fn cheb(&self, a: usize, b: usize) -> i32 {
        let s = self.size;
        let dx = (a % s) as i32 - (b % s) as i32;
        let dy = (a / s) as i32 - (b / s) as i32;
        dx.abs().max(dy.abs())
    }
    fn army(&self) -> f64 {
        let mut a = 0.0;
        for &i in &self.my_units { a += atk(self.u_kind[i]) * self.u_health[i] / hp(self.u_kind[i]).max(1.0) + 1.0; }
        a
    }
    fn enemy_army(&self) -> f64 {
        let mut a = 0.0;
        for &i in &self.enemy_units { a += atk(self.u_kind[i]) * self.u_health[i] / hp(self.u_kind[i]).max(1.0) + 1.0; }
        a
    }
    fn def_bonus(&self, t: usize) -> f64 {
        let te = self.terrain[t];
        if self.structure[t] == 1 && self.owner[t] == self.u_owner[t] && self.u_owner[t] > 0 { return 2.5; }
        if te == 5 { return 1.3; }
        if te == 4 { return 1.3; }
        if te <= 2 { return 1.1; }
        1.0
    }
}

fn combat_score(o: &Obs, src: usize, dst: usize) -> f64 {
    let ak = o.u_kind[src];
    let dk = o.u_kind[dst];
    let ah = o.u_health[src].max(0.1);
    let dh = o.u_health[dst].max(0.1);
    let amh = hp(ak).max(1.0);
    let dmh = hp(dk).max(1.0);
    let aa = atk(ak);
    let dd = def(dk);
    if aa <= 0.0 { return -100000.0; }
    let af = aa * ah / amh;
    let df = dd * dh / dmh * o.def_bonus(dst);
    let tot = af + df;
    if tot <= 0.0 { return -100000.0; }
    let dmg = ((af / tot) * aa * 4.5 + 0.5).floor();
    let kill = dmg >= dh;
    let ret = if kill || URANGE[(dk as usize).min(47)] < 2 && o.cheb(src, dst) <= rng(dk) {
        ((df / tot) * dd * 4.5 + 0.5).floor()
    } else { 0.0 };
    let mut s = 0.0;
    if kill {
        s += 400.0 + 60.0 * cost(dk) as f64;
        if o.structure[dst] == 1 && o.owner[dst] != o.me { s += 3000.0; }
    } else {
        s += 90.0 * dmg;
    }
    s -= 110.0 * ret;
    if ret >= ah { s -= 2500.0; } // we would die
    if o.u_health[src] < ah { s -= 10.0; }
    s
}

fn reward_pref(p: i32) -> f64 {
    match p {
        3 => 100.0, // workshop: +1 production
        6 => 95.0,  // giant
        8 => 92.0,  // population
        7 => 88.0,  // resources
        4 => 80.0,  // explorer
        5 => 78.0,  // border growth
        2 => 70.0,  // park
        1 => 40.0,  // wall
        _ => 10.0,
    }
}

fn build_pref(o: &Obs, tile: usize, p: i32) -> f64 {
    let base = match p {
        5 => 200.0,   // farm (+2 pop)
        21 => 200.0,  // mine (+2 pop)
        12 => 170.0,  // lumber hut
        22 => 165.0,  // forge
        6 => 160.0,   // windmill
        13 => 150.0,  // sawmill
        50 => 145.0,  // market (+star/turn)
        8 => if o.water_frac > 0.15 { 190.0 } else { 90.0 }, // port
        3 => 70.0,    // road
        48 => 60.0,   // bridge
        17 => 50.0,
        18 => 48.0,
        19 => 46.0,
        20 => 44.0,
        47 => 40.0,
        _ => 30.0,
    };
    let mut b = base;
    if p == 3 {
        // roads matter mainly to connect cities
        let nc = o.my_cities.len();
        if nc < 2 { b = 20.0; }
        let near_city = o.my_cities.iter().any(|&c| o.cheb(c, tile) <= 4);
        if !near_city { b -= 15.0; }
    }
    b
}

fn summon_pref(o: &Obs, p: i32) -> f64 {
    let strong = o.army() >= o.enemy_army();
    let v = match p {
        12 => 300.0, // giant
        45 => 260.0, // juggernaut
        11 => 240.0, // swordsman
        4 => 235.0,  // knight
        8 => 200.0,  // catapult
        9 => 180.0,  // archer
        44 => 175.0, // rammership
        5 => 165.0,  // defender
        41 => 150.0, // bomber
        42 => 140.0, // scoutship
        3 => 130.0,  // rider
        10 => 120.0, // mind bender
        2 => 90.0,   // warrior
        40 => 70.0,  // pirate
        38 => 60.0,  // cloak
        37 => 55.0,  // dagger
        43 => if o.water_frac > 0.15 { 120.0 } else { 15.0 }, // raft
        39 => 10.0,  // dinghy
        _ => 40.0,
    };
    let mut v = v;
    // prefer cheap units when poor
    if o.stars <= 4 && cost(p) >= 5 { v -= 120.0; }
    if !strong && (p == 8 || p == 4) { v -= 20.0; }
    v
}

fn tech_pref(o: &Obs, p: i32) -> f64 {
    let naval = o.water_frac > 0.18;
    let land_order = [6i32, 8, 20, 23, 1, 18, 4, 24, 5, 7, 9, 16, 17, 3, 22, 21, 10, 13, 44, 14, 2, 19, 12, 25, 37];
    let sea_order = [10i32, 13, 44, 6, 8, 20, 23, 1, 18, 4, 14, 24, 5, 7, 9, 16, 17, 3, 22, 21, 2, 19, 12];
    let order = if naval { &sea_order[..] } else { &land_order[..] };
    for (i, &t) in order.iter().enumerate() {
        if t == p { return 500.0 - 8.0 * i as f64; }
    }
    100.0
}

fn main_policy(obs: &Value) -> String {
    let legal = obs["legal_actions"].as_array().expect("legal_actions");
    if legal.is_empty() { return "0".to_string(); }
    let o = Obs::parse(obs);
    // movement potential field
    let mut pot = vec![1e9f64; o.n];
    let mut targets: Vec<(usize, f64)> = Vec::new();
    let strong = o.army() + 2.0 >= o.enemy_army();
    let ew = if strong { 8.0 } else { 1.5 };
    for &v in &o.villages { targets.push((v, 4.0)); }
    for &c in &o.enemy_cities { targets.push((c, ew)); }
    for i in 0..o.n {
        if o.known[i] && o.structure[i] == 2 { targets.push((i, 3.5)); }
        if o.known[i] && o.structure[i] == 47 && o.owner[i] != o.me { targets.push((i, 2.0)); }
        if !o.known[i] { targets.push((i, 0.0)); }
    }
    if targets.is_empty() { targets.push((o.n / 2, 0.0)); }
    for i in 0..o.n {
        let mut best = 1e9f64;
        for &(t, w) in &targets {
            let d = (o.cheb(i, t) as f64) - w;
            if d < best { best = d; }
        }
        pot[i] = best;
    }
    // how much new territory a unit standing here would see
    let mut rev = vec![0i32; o.n];
    for i in 0..o.n {
        let mut c = 0i32;
        for t in 0..o.n { if !o.known[t] && o.cheb(i, t) <= 1 { c += 1; } }
        rev[i] = c;
    }
    // cities with no friendly unit nearby need a guard urgently
    let mut need_defense: Vec<usize> = Vec::new();
    for &c in &o.my_cities {
        if !o.my_units.iter().any(|&u| o.cheb(u, c) <= 1) { need_defense.push(c); }
    }
    let guard = |t: usize| -> bool { need_defense.iter().any(|&c| o.cheb(c, t) <= 1) };
    let mut best_id = legal[0]["action_id"].as_str().unwrap_or("0").to_string();
    let mut best_score = f64::NEG_INFINITY;
    for a in legal {
        let c = &a["command"];
        let k = c["kind"].as_i64().unwrap_or(0) as i32;
        let src = c["source"].as_i64().unwrap_or(-1);
        let dst = c["target"].as_i64().unwrap_or(-1);
        let p = c["parameter"].as_i64().unwrap_or(0) as i32;
        let si = if src >= 0 { src as usize } else { 0 };
        let di = if dst >= 0 { dst as usize } else { 0 };
        let s: f64 = match k {
            11 => -1e9,
            10 => -1000.0,
            9 => 1_000_000.0 + reward_pref(p),
            8 => {
                let mut v = 900_000.0;
                if di == 0 && si < o.n {
                    if o.owner[si] > 0 && o.owner[si] != o.me { v += 50_000.0; }
                }
                v
            }
            2 => {
                if src < 0 || dst < 0 || si >= o.n || di >= o.n { -1e9 }
                else { 800_000.0 + combat_score(&o, si, di) }
            }
            3 => match p {
                16 => 950_000.0, // promote
                7 => { // recover
                    let h = o.u_health[si.min(o.n - 1)];
                    if h > 0.0 && h < 8.0 { 700_000.0 + (10.0 - h) * 100.0 } else { -50000.0 }
                }
                9 => 690_000.0, // heal others
                2 => 30_000.0,  // burn forest: +1 star
                1 => if o.tech[8] { 29_000.0 } else { -60_000.0 }, // clear forest -> crop
                _ => -50_000.0, // grow forest, destroy, disband, diplomacy
            },
            7 => 500_000.0 + tech_pref(&o, p),
            5 => 400_000.0,
            6 => 300_000.0 + build_pref(&o, di.min(o.n - 1), p),
            4 => {
                let mut v = 200_000.0 + summon_pref(&o, p);
                if si < o.n && guard(si) { v += 680_000.0; }
                v
            }
            1 => {
                if src < 0 || dst < 0 || si >= o.n || di >= o.n { -1e9 }
                else {
                    let mut v = 100_000.0 + 400.0 * (pot[si] - pot[di])
                        + 45.0 * (rev[di] - rev[si]) as f64;
                    if guard(di) { v += 250_000.0; }
                    if o.structure[di] == 1 && o.owner[di] != o.me { v += 3000.0; }
                    if o.structure[di] == 2 { v += 2500.0; }
                    if o.terrain[di] == 5 { v += 40.0; }
                    if o.terrain[di] == 4 { v += 30.0; }
                    // danger: weak unit next to enemies
                    let h = o.u_health[si];
                    if h < 9.0 {
                        let near = o.enemy_units.iter().filter(|&&e| o.cheb(e, di) <= 1).count();
                        v -= 250.0 * near as f64;
                    }
                    v
                }
            }
            _ => -2000.0,
        };
        if s > best_score { best_score = s; best_id = a["action_id"].as_str().unwrap_or("0").to_string(); }
    }
    best_id
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let id = main_policy(&obs);
        let _ = writeln!(stdout, "{}", json!({"action_id": id}));
        let _ = stdout.flush();
    }
}
