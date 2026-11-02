use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

const INF: i32 = 1_000_000;

#[derive(Clone)]
struct U { owner: i64, kind: i64, hp: f64, vet: bool, moved: bool, attacked: bool }

struct G {
    size: usize, n: usize, me: i64, turn: i64, stars: i64,
    terrain: Vec<i64>, owner: Vec<i64>, structure: Vec<i64>, resource: Vec<i64>, known: Vec<bool>,
    capital: Vec<i64>,
    units: Vec<Option<U>>, city_owner: Vec<i64>, city_level: Vec<i64>, tech: Vec<bool>,
}

// hp, atk, def, range, cost, move
fn stats(k: i64) -> (f64, f64, f64, i64, f64) {
    match k {
        2 => (10., 2., 2., 1, 2.), 3 => (10., 2., 1., 1, 3.), 4 => (10., 3.5, 1., 1, 8.), 5 => (15., 1., 3., 1, 3.),
        8 => (10., 4., 0., 3, 8.), 9 => (10., 2., 1., 2, 3.), 10 => (10., 0., 1., 1, 5.), 11 => (15., 3., 3., 1, 5.),
        12 => (40., 5., 4., 1, 10.), 37 => (10., 2., 2., 1, 2.), 38 => (5., 0., 0.5, 1, 8.), 39 => (5., 0., 0.5, 1, 2.),
        40 => (10., 2., 2., 1, 2.), 41 => (10., 3., 2., 3, 15.), 42 => (10., 2., 1., 2, 5.), 43 => (10., 0., 1., 2, 0.),
        44 => (10., 3., 3., 1, 5.), 45 => (40., 4., 4., 1, 10.), _ => (10., 2., 2., 1, 3.),
    }
}
fn stiff(k: i64) -> bool { matches!(k, 8 | 10 | 38 | 39 | 41 | 43 | 45 | 17) }
fn fortify(k: i64) -> bool { matches!(k, 2 | 3 | 4 | 5 | 9) }

impl G {
    fn d(&self, a: usize, b: usize) -> i32 {
        let (ax, ay) = ((a % self.size) as i32, (a / self.size) as i32);
        let (bx, by) = ((b % self.size) as i32, (b / self.size) as i32);
        (ax - bx).abs().max((ay - by).abs())
    }
    fn nbrs(&self, i: usize) -> Vec<usize> {
        let s = self.size as i32;
        let (x, y) = ((i % self.size) as i32, (i / self.size) as i32);
        let mut v = Vec::with_capacity(8);
        for dy in -1..=1 { for dx in -1..=1 {
            if dx == 0 && dy == 0 { continue; }
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && nx < s && ny < s { v.push((ny * s + nx) as usize); }
        }}
        v
    }
    fn is_water(&self, i: usize) -> bool { self.known[i] && (self.terrain[i] == 1 || self.terrain[i] == 2) }
    fn land_ok(&self, i: usize) -> bool {
        !self.known[i] || self.terrain[i] == 3 || self.terrain[i] == 5 || (self.terrain[i] == 4 && self.tech[20])
    }
    fn water_ok(&self, i: usize) -> bool {
        self.known[i] && (self.terrain[i] == 1 && self.tech[10] || (self.terrain[i] == 2 && self.tech[13]))
    }
    fn own_port(&self, i: usize) -> bool { self.structure[i] == 8 && self.owner[i] == self.me }
    // can a unit move from j to i (one step)
    fn step_ok(&self, j: usize, i: usize) -> bool {
        if self.land_ok(i) { return true; }
        if self.water_ok(i) || self.own_port(i) {
            return self.is_water(j) || self.own_port(i);
        }
        false
    }
    // reverse field: dist[j] = steps from j to nearest source
    fn field(&self, init: Vec<i32>, nb: &Vec<Vec<usize>>) -> Vec<i32> {
        let mut dist = init;
        let mut changed = true;
        let mut it = 0;
        while changed && it < 60 {
            changed = false; it += 1;
            for i in 0..self.n {
                if dist[i] >= INF { continue; }
                let di = dist[i] + 1;
                for &j in &nb[i] {
                    if di < dist[j] && self.step_ok(j, i) {
                        // j must itself be somewhere a unit can stand
                        if self.land_ok(j) || self.is_water(j) { dist[j] = di; changed = true; }
                    }
                }
            }
        }
        dist
    }
}

fn parse(obs: &Value) -> G {
    let v = &obs["visible_state"];
    let size = v["board_size"].as_u64().unwrap_or(14) as usize;
    let n = size * size;
    let me = obs["player"].as_i64().unwrap_or(1);
    let mut g = G {
        size, n, me, turn: v["turn"].as_i64().unwrap_or(0), stars: v["stars"].as_i64().unwrap_or(0),
        terrain: vec![-1; n], owner: vec![0; n], structure: vec![0; n], resource: vec![0; n], known: vec![false; n],
        capital: vec![0; n], units: vec![None; n], city_owner: vec![-1; n], city_level: vec![0; n], tech: vec![false; 128],
    };
    if let Some(ts) = v["tiles"].as_array() {
        for t in ts {
            let i = t["tile"].as_u64().unwrap_or(0) as usize;
            if i >= n { continue; }
            if t["known"].as_bool().unwrap_or(false) {
                g.known[i] = true;
                g.terrain[i] = t["terrain"].as_i64().unwrap_or(-1);
                g.owner[i] = t["owner"].as_i64().unwrap_or(0);
                g.structure[i] = t["structure"].as_i64().unwrap_or(0);
                g.resource[i] = t["resource"].as_i64().unwrap_or(0);
                g.capital[i] = t["capital"].as_i64().unwrap_or(0);
            }
        }
    }
    if let Some(us) = v["units"].as_array() {
        for u in us {
            let i = u["tile"].as_u64().unwrap_or(0) as usize;
            if i >= n { continue; }
            g.units[i] = Some(U {
                owner: u["owner"].as_i64().unwrap_or(0), kind: u["kind"].as_i64().unwrap_or(2),
                hp: u["health"].as_f64().unwrap_or(10.), vet: u["veteran"].as_bool().unwrap_or(false),
                moved: u["moved"].as_bool().unwrap_or(false), attacked: u["attacked"].as_bool().unwrap_or(false),
            });
        }
    }
    if let Some(cs) = v["cities"].as_array() {
        for c in cs {
            let i = c["tile"].as_u64().unwrap_or(0) as usize;
            if i >= n { continue; }
            g.city_owner[i] = c["owner"].as_i64().unwrap_or(0);
            g.city_level[i] = c["level"].as_i64().unwrap_or(1);
        }
    }
    if let Some(ts) = v["technology"].as_array() {
        for t in ts { let k = t.as_u64().unwrap_or(0) as usize; if k < 128 { g.tech[k] = true; } }
    }
    g
}

fn maxhp(u: &U) -> f64 { let b = stats(u.kind).0 + if u.vet { 5. } else { 0. }; b.max(u.hp) }

struct Ctx {
    p_att: Vec<f64>, p_non: Vec<f64>,
    threatened: Vec<bool>, // own city tiles with enemies near
    need_naval: bool, my_units: usize, my_cities: usize, army_ok: bool,
    enemy_near_count: Vec<i32>,
    threat: Vec<f64>,
    aversion: f64,
    stall: i64,
    salt: u64,
}

struct Mem { last_turn: i64, best_cities: i64, best_turn: i64, salt: u64 }
static mut MEM: Mem = Mem { last_turn: -1, best_cities: 0, best_turn: 0, salt: 0 };

fn hash(a: u64) -> u64 { let mut x = a.wrapping_mul(0x9E3779B97F4A7C15); x ^= x >> 31; x = x.wrapping_mul(0xBF58476D1CE4E5B9); x ^ (x >> 29) }

fn build_ctx(g: &G) -> Ctx {
    let n = g.n;
    let nb: Vec<Vec<usize>> = (0..n).map(|i| g.nbrs(i)).collect();
    let my_unit_on = |i: usize| g.units[i].as_ref().map_or(false, |u| u.owner == g.me);
    let mut classes: Vec<(f64, Vec<i32>, bool)> = Vec::new(); // value, dist, attack-only
    let mut add = |vals: Vec<(usize, f64)>, att: bool, classes: &mut Vec<(f64, Vec<i32>, bool)>| {
        // group by value
        let mut vs: Vec<f64> = vals.iter().map(|x| x.1).collect();
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap()); vs.dedup();
        for val in vs {
            let mut init = vec![INF; n];
            let mut any = false;
            for &(t, v) in &vals { if v == val { init[t] = 0; any = true; } }
            if any { classes.push((val, g.field(init, &nb), att)); }
        }
    };
    let mut targets = Vec::new();
    let mut unknown = Vec::new();
    let mut enemies = Vec::new();
    for i in 0..n {
        if !g.known[i] { unknown.push((i, 22.0)); continue; }
        if g.structure[i] == 1 && g.city_owner[i] != g.me && !my_unit_on(i) {
            let v = if g.city_owner[i] <= 0 { 100.0 } else if g.capital[i] > 0 { 140.0 } else { 120.0 };
            targets.push((i, v));
        }
        if g.structure[i] == 2 && !my_unit_on(i) { targets.push((i, 60.0)); }
        if let Some(u) = &g.units[i] { if u.owner != g.me { enemies.push((i, 55.0)); } }
    }
    let has_ground = !targets.is_empty() || !unknown.is_empty();
    add(targets, false, &mut classes);
    add(unknown, false, &mut classes);
    add(enemies, true, &mut classes);
    let mut p_att = vec![0.0; n];
    let mut p_non = vec![0.0; n];
    for (val, dist, att) in &classes {
        for i in 0..n {
            if dist[i] < INF {
                let p = val * 0.8f64.powi(dist[i]);
                if p > p_att[i] { p_att[i] = p; }
                if !att && p > p_non[i] { p_non[i] = p; }
            }
        }
    }
    let mut my_units = 0; let mut my_cities = 0;
    let mut threatened = vec![false; n];
    let mut enemy_near_count = vec![0; n];
    for i in 0..n {
        if let Some(u) = &g.units[i] {
            if u.owner == g.me { my_units += 1; } else {
                for j in 0..n { if g.d(i, j) <= 2 { enemy_near_count[j] += 1; } }
            }
        }
    }
    let mut threat = vec![0.0; n];
    for e in 0..n {
        if let Some(u) = &g.units[e] {
            if u.owner == g.me { continue; }
            let st = stats(u.kind);
            let mv = match u.kind { 3 => 2, 4 => 3, 42 | 44 => 3, 40 | 39 | 41 | 45 => 2, 8 | 10 | 12 | 5 => 0, _ => 1 };
            let reach = mv + st.3 as i32;
            let hpf = u.hp / maxhp(u);
            for j in 0..n { if g.d(e, j) <= reach { threat[j] += st.1 * hpf; } }
        }
    }
    for i in 0..n {
        if g.city_owner[i] == g.me { my_cities += 1; if enemy_near_count[i] > 0 { threatened[i] = true; } }
    }
    // naval need: no land/own unit reaches any non-enemy target
    let mut reach = false;
    for i in 0..n {
        if (my_unit_on(i) || g.city_owner[i] == g.me) && p_non[i] > 0.0 { reach = true; }
    }
    let need_naval = has_ground && !reach;
    let desired = 2 + 2 * my_cities + (g.turn as usize) / 12;
    Ctx { p_att, p_non, threatened, need_naval, my_units, my_cities, army_ok: my_units >= desired.min(25), enemy_near_count, threat, aversion: 1.0, stall: 0, salt: 0 }
}

fn tech_score(k: i64, g: &G, c: &Ctx) -> f64 {
    let mut fish = false; let mut fruit = false; let mut metal = false; let mut water = false;
    for i in 0..g.n {
        if g.owner[i] == g.me {
            if g.resource[i] == 3 { fish = true; }
            if g.resource[i] == 6 { fruit = true; }
            if g.terrain[i] == 4 { metal = true; }
            if g.terrain[i] == 1 { water = true; }
        }
    }
    if c.need_naval {
        if k == 10 { return 600.0; }
        if k == 13 { return 450.0; }
    }
    let has_port = (0..g.n).any(|i| g.own_port(i));
    match k {
        1 => 320.0,
        6 => if fruit { 300.0 } else { 70.0 },
        2 => if c.my_cities >= 3 { 200.0 } else { 90.0 },
        3 => if c.my_cities >= 3 && c.my_units >= 4 { 280.0 } else { 80.0 },
        10 => if fish { 250.0 } else if water { 90.0 } else { 20.0 },
        13 => if has_port { 230.0 } else { 30.0 },
        18 => 90.0,
        20 => if metal { 190.0 } else { 50.0 },
        23 => if metal { 170.0 } else { 60.0 },
        24 => 180.0,
        8 => if fruit { 200.0 } else { 60.0 },
        7 => if c.threatened.iter().any(|&t| t) { 330.0 } else { 140.0 },
        16 => 100.0,
        9 => 80.0,
        _ => 30.0,
    }
}

fn attack_score(g: &G, src: usize, dst: usize) -> f64 {
    let (a, d) = match (&g.units[src], &g.units[dst]) { (Some(a), Some(d)) => (a.clone(), d.clone()), _ => return -1.0 };
    let sa = stats(a.kind); let sd = stats(d.kind);
    let amax = maxhp(&a); let dmax = maxhp(&d);
    let mut bonus = 1.0;
    if g.city_owner[dst] == d.owner && fortify(d.kind) { bonus = 1.5; }
    else if g.terrain[dst] == 5 || g.terrain[dst] == 4 { bonus = 1.25; }
    let af = sa.1 * a.hp / amax; let df = sd.2 * d.hp / dmax * bonus;
    let tot = af + df; if tot <= 0.0 { return -1.0; }
    let dmg = (af / tot * sa.1 * 4.5 + 0.5).floor();
    let kill = dmg >= d.hp;
    let retal = if !kill && (g.d(src, dst) as i64) <= sd.3 && !stiff(d.kind) { (df / tot * sd.2 * 4.5 + 0.5).floor() } else { 0.0 };
    let dcost = sd.4.max(2.0); let acost = sa.4.max(2.0);
    let near_my_city = (0..g.n).any(|i| g.city_owner[i] == g.me && g.d(i, dst) <= 2);
    let on_port = g.own_port(src);
    if kill { return 1500.0 + dcost * 30.0 + if near_my_city { 300.0 } else { 0.0 } + if g.city_owner[dst] == g.me { 500.0 } else { 0.0 }; }
    if on_port { return 0.5; }
    if retal >= a.hp { return -1.0; }
    let gain = dmg / dmax * dcost - retal / amax * acost * 1.1;
    if gain <= 0.0 && !near_my_city { return -1.0; }
    500.0 + 100.0 * gain + if near_my_city { 150.0 } else { 0.0 }
}

fn score(g: &G, c: &Ctx, a: &Value) -> f64 {
    let cmd = &a["command"];
    let k = cmd["kind"].as_i64().unwrap_or(0);
    let src = cmd["source"].as_i64().unwrap_or(-1);
    let dst = cmd["target"].as_i64().unwrap_or(-1);
    let p = cmd["parameter"].as_i64().unwrap_or(0);
    let n = g.n as i64;
    let su = if src >= 0 && src < n { src as usize } else { 0 };
    let du = if dst >= 0 && dst < n { dst as usize } else { 0 };
    match k {
        11 => -1e9,
        10 => 0.0,
        9 => 10000.0 + match p {
            3 => 10.0, 4 => 5.0,
            7 => if c.threatened[du] { 5.0 } else { 10.0 }, 1 => if c.threatened[du] { 10.0 } else { 5.0 },
            8 => 10.0, 5 => 5.0, 6 => 10.0, 2 => 5.0, _ => 0.0 },
        8 => 6000.0,
        2 => attack_score(g, su, du),
        3 => match p {
            16 => 7000.0,
            7 => {
                let u = match &g.units[su] { Some(u) => u, None => return -1.0 };
                if u.hp < 0.55 * maxhp(u) && c.enemy_near_count[su] == 0 { 160.0 } else if u.hp < 0.4 * maxhp(u) { 120.0 } else { -1.0 }
            }
            9 => 140.0,
            _ => -1e9,
        },
        5 => 420.0,
        7 => {
            let s = tech_score(p, g, c);
            let any_thr = c.threatened.iter().any(|&t| t);
            let mil = matches!(p, 1 | 2 | 3 | 6 | 7);
            if c.need_naval && matches!(p, 10 | 13) { s }
            else if g.stars >= 30 { s }
            else if any_thr && !mil { -1.0 }
            else if c.army_ok { s } else if mil { s * 0.6 } else { -1.0 }
        }
        4 => {
            // summon
            if matches!(p, 41 | 42 | 44) {
                return if p == 42 && (c.need_naval || true) { 280.0 } else if p == 44 { 300.0 } else { 100.0 };
            }
            let city = su;
            let thr = c.threatened[city];
            let best_known = if g.tech[3] { 4 } else if g.tech[24] { 11 } else if g.tech[1] { 3 } else if g.tech[18] { 9 } else { 2 };
            let adj_enemy = g.nbrs(city).iter().any(|&j| g.units[j].as_ref().map_or(false, |u| u.owner != g.me));
            let pref = if adj_enemy {
                match p { 5 => 80.0, 11 => 75.0, 2 => 60.0, 4 => 50.0, 9 => 40.0, 3 => 30.0, 10 => -500.0, _ => 0.0 }
            } else {
                match p { 4 => 60.0, 11 => 55.0, 3 => 40.0, 9 => 35.0, 5 => if thr { 30.0 } else { 15.0 }, 2 => 10.0, 8 => 20.0, 10 => -500.0, _ => 0.0 }
            };
            if adj_enemy { return 750.0 + pref; }
            if !thr && p != best_known && g.stars < 20 && !(best_known == 4 && p == 11) { return -1.0; }
            let base = if thr { 700.0 } else if !c.army_ok { 300.0 } else if g.stars >= 15 { 200.0 } else { 90.0 };
            let _ = c.my_units;
            base + pref
        }
        6 => {
            match p {
                5 | 21 => if c.army_ok { 290.0 } else { 220.0 },
                12 => 200.0,
                6 | 13 | 22 => 260.0,
                50 => 210.0,
                8 => {
                    let has_port = (0..g.n).any(|i| g.own_port(i));
                    if c.need_naval && !has_port { 650.0 } else if !has_port && c.my_cities >= 2 { 150.0 } else { 60.0 }
                }
                23..=29 => 800.0,
                17 | 18 | 19 | 20 => if g.stars >= 25 { 120.0 } else { -1.0 },
                3 => if g.stars >= 30 { 50.0 } else { -1.0 },
                _ => -1.0,
            }
        }
        1 => {
            let u = match &g.units[su] { Some(u) => u, None => return -1.0 };
            let att = stats(u.kind).1 > 0.0;
            let pf = if att { &c.p_att } else { &c.p_non };
            let mut s = 6.0 * (pf[du] - pf[su]);
            if g.city_owner[su] == g.me && c.threatened[su] { s -= 400.0; }
            // weak units avoid enemies
            if u.hp < 0.5 * maxhp(u) && c.enemy_near_count[du] > 0 { s -= 50.0; }
            let capt = g.structure[du] == 1 && g.city_owner[du] != g.me;
            let fort = (g.city_owner[du] == g.me && fortify(u.kind)) || g.terrain[du] == 5 || g.terrain[du] == 4;
            let dmult = (if fort { 0.5 } else { 1.0 }) * (if capt { 0.3 } else { 1.0 }) * 2.0 / (stats(u.kind).2 + 0.5) * maxhp(u) / u.hp.max(1.0);
            s -= 6.0 * (c.threat[du] - c.threat[su]).max(0.0) * dmult * 3.0 * c.aversion;
            // moving onto own city that is threatened is good
            if g.city_owner[du] == g.me && c.threatened[du] && g.units[du].is_none() { s += 120.0; }
            if c.stall > 8 {
                let noise = (hash(c.salt ^ (su as u64) << 20 ^ (du as u64) << 8) % 1000) as f64 / 1000.0;
                s += noise * (20.0 + c.stall as f64 * 2.0);
                if g.own_port(su) { s += 15.0; }
            }
            if s > 1.0 { s + 1.0 } else { -1.0 }
        }
        _ => -1.0,
    }
}

fn choose(obs: &Value) -> String {
    let legal = match obs["legal_actions"].as_array() { Some(l) if !l.is_empty() => l, _ => return String::new() };
    let g = parse(obs);
    let mut c = build_ctx(&g);
    let progress = c.my_cities as i64 * 10 - (0..g.n).filter(|&i| g.city_owner[i] > 0 && g.city_owner[i] != g.me).count() as i64 * 5;
    unsafe {
        let m = &mut *std::ptr::addr_of_mut!(MEM);
        if g.turn < m.last_turn { m.best_cities = 0; m.best_turn = g.turn; }
        if progress > m.best_cities || g.turn < 20 { m.best_cities = m.best_cities.max(progress); m.best_turn = g.turn; }
        m.last_turn = g.turn;
        m.salt = m.salt.wrapping_add(1);
        c.stall = g.turn - m.best_turn;
        c.salt = m.salt;
    }
    if c.stall > 8 { c.aversion = 0.3; }
    if c.stall > 20 { c.aversion = 0.1; }
    let mut best: Option<(&Value, f64)> = None;
    let mut end: Option<&Value> = None;
    for a in legal {
        let k = a["command"]["kind"].as_i64().unwrap_or(0);
        if k == 10 { end = Some(a); }
        let s = score(&g, &c, a);
        if s.is_finite() && s > -1e8 {
            if best.map_or(true, |(_, bs)| s > bs) { best = Some((a, s)); }
        }
    }
    let pick = match (best, end) {
        (Some((a, s)), Some(e)) => if s > 0.0 { a } else { e },
        (Some((a, _)), None) => a,
        (None, Some(e)) => e,
        (None, None) => legal.iter().find(|a| a["command"]["kind"] != 11).unwrap_or(&legal[0]),
    };
    pick["action_id"].as_str().map(|s| s.to_owned()).unwrap_or_else(|| pick["action_id"].to_string())
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let id = choose(&obs);
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
