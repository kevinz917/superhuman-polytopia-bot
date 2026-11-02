use serde_json::{json, Value};
use std::collections::BinaryHeap;
use std::cmp::Reverse;
use std::io::{self, BufRead, Write};

const INF: i32 = 1_000_000;

#[derive(Clone, Copy, Default)]
struct Unit { owner: i32, kind: i32, hp: f64, vet: bool, moved: bool, attacked: bool, passenger: i32 }

fn stats(kind: i32) -> (f64, f64, f64, i32) {
    // attack, defense, health, range
    match kind {
        2 => (2.0, 2.0, 10.0, 1), 3 => (2.0, 1.0, 10.0, 1), 4 => (3.5, 1.0, 10.0, 1),
        5 => (1.0, 3.0, 15.0, 1), 8 => (4.0, 0.0, 10.0, 3), 9 => (2.0, 1.0, 10.0, 2),
        10 => (0.0, 1.0, 10.0, 1), 11 => (3.0, 3.0, 15.0, 1), 12 => (5.0, 4.0, 40.0, 1),
        37 => (2.0, 2.0, 10.0, 1), 38 => (0.0, 0.5, 5.0, 1), 39 => (0.0, 0.5, 5.0, 1),
        40 => (2.0, 2.0, 10.0, 1), 41 => (3.0, 2.0, 10.0, 3), 42 => (2.0, 1.0, 10.0, 2),
        43 => (0.0, 1.0, 10.0, 2), 44 => (3.0, 3.0, 10.0, 1), 45 => (4.0, 4.0, 40.0, 1),
        _ => (2.0, 2.0, 10.0, 1),
    }
}
fn naval(kind: i32) -> bool { (39..=45).contains(&kind) }
fn stiff(kind: i32) -> bool { matches!(kind, 8 | 10 | 38 | 39 | 41 | 43 | 45) }
fn fortify(kind: i32) -> bool { matches!(kind, 2 | 3 | 4 | 5 | 9) }
fn cost(kind: i32) -> i32 {
    match kind { 2 => 2, 3 => 3, 4 => 8, 5 => 3, 8 => 8, 9 => 3, 10 => 5, 11 => 5, 12 => 10, 42 => 5, 44 => 5, 41 => 15, _ => 3 }
}

struct St {
    size: usize, n: usize, me: i32, stars: i32, tech: Vec<bool>,
    known: Vec<bool>, terrain: Vec<i32>, owner: Vec<i32>, structure: Vec<i32>, resource: Vec<i32>,
    units: Vec<Option<Unit>>, city_owner: Vec<i32>, city_level: Vec<i32>, turn: i32,
}

impl St {
    fn d(&self, a: usize, b: usize) -> i32 {
        let (ax, ay, bx, by) = ((a % self.size) as i32, (a / self.size) as i32, (b % self.size) as i32, (b / self.size) as i32);
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
    fn step_cost(&self, t: usize, nav: bool) -> i32 {
        if !self.known[t] { return 2; }
        let ter = self.terrain[t];
        let water = ter == 1 || ter == 2;
        if nav {
            if water { if ter == 2 && !self.tech[13] { return INF; } return 1; }
            return 2;
        }
        if water {
            if !self.tech[10] { return INF; }
            if ter == 2 && !self.tech[13] { return INF; }
            return 4;
        }
        if ter == 4 && !self.tech[20] { return INF; }
        if ter == 4 || ter == 5 { return 2; }
        1
    }
    fn dist_map(&self, targets: &[usize], nav: bool) -> Vec<i32> {
        let mut dist = vec![INF; self.n];
        let mut h = BinaryHeap::new();
        for &t in targets { dist[t] = 0; h.push(Reverse((0, t))); }
        while let Some(Reverse((c, i))) = h.pop() {
            if c > dist[i] { continue; }
            for j in self.nbrs(i) {
                // cost of moving from j into i (approx: cost of i)
                let sc = self.step_cost(i, nav);
                if sc >= INF { continue; }
                let nc = c + sc;
                if nc < dist[j] { dist[j] = nc; h.push(Reverse((nc, j))); }
            }
        }
        dist
    }
}

fn parse(obs: &Value) -> St {
    let v = &obs["visible_state"];
    let size = v["board_size"].as_u64().unwrap_or(14) as usize;
    let n = size * size;
    let me = obs["player"].as_i64().unwrap_or(1) as i32;
    let mut tech = vec![false; 128];
    if let Some(a) = v["technology"].as_array() { for t in a { let k = t.as_u64().unwrap_or(0) as usize; if k < 128 { tech[k] = true; } } }
    let mut s = St { size, n, me, stars: v["stars"].as_i64().unwrap_or(0) as i32, tech,
        known: vec![false; n], terrain: vec![0; n], owner: vec![0; n], structure: vec![0; n], resource: vec![0; n],
        units: vec![None; n], city_owner: vec![-1; n], city_level: vec![0; n], turn: v["turn"].as_i64().unwrap_or(0) as i32 };
    if let Some(a) = v["tiles"].as_array() {
        for t in a {
            let i = t["tile"].as_u64().unwrap_or(0) as usize;
            if i >= n { continue; }
            if t["known"].as_bool().unwrap_or(false) {
                s.known[i] = true;
                s.terrain[i] = t["terrain"].as_i64().unwrap_or(0) as i32;
                s.owner[i] = t["owner"].as_i64().unwrap_or(0) as i32;
                s.structure[i] = t["structure"].as_i64().unwrap_or(0) as i32;
                s.resource[i] = t["resource"].as_i64().unwrap_or(0) as i32;
            }
        }
    }
    if let Some(a) = v["units"].as_array() {
        for u in a {
            let i = u["tile"].as_u64().unwrap_or(0) as usize;
            if i >= n { continue; }
            s.units[i] = Some(Unit {
                owner: u["owner"].as_i64().unwrap_or(0) as i32, kind: u["kind"].as_i64().unwrap_or(0) as i32,
                hp: u["health"].as_f64().unwrap_or(10.0), vet: u["veteran"].as_bool().unwrap_or(false),
                moved: u["moved"].as_bool().unwrap_or(false), attacked: u["attacked"].as_bool().unwrap_or(false),
                passenger: u["passenger"].as_i64().unwrap_or(0) as i32,
            });
        }
    }
    if let Some(a) = v["cities"].as_array() {
        for c in a {
            let i = c["tile"].as_u64().unwrap_or(0) as usize;
            if i >= n { continue; }
            s.city_owner[i] = c["owner"].as_i64().unwrap_or(0) as i32;
            s.city_level[i] = c["level"].as_i64().unwrap_or(1) as i32;
        }
    }
    s
}

fn maxhp(u: &Unit) -> f64 {
    let k = if u.passenger > 0 { u.passenger } else { u.kind };
    stats(k).2 + if u.vet { 5.0 } else { 0.0 }
}

fn def_bonus(s: &St, t: usize, u: &Unit) -> f64 {
    let ter = s.terrain[t];
    if ter == 5 || ter == 4 { return 1.5; }
    if ter == 1 || ter == 2 { return 1.0; }
    if s.city_owner[t] == u.owner && fortify(u.kind) { return 1.5; }
    1.0
}

fn combat(s: &St, a: &Unit, src: usize, d: &Unit, dst: usize) -> (f64, f64) {
    let (aa, _, _, _) = stats(a.kind);
    let (_, dd, _, dr) = stats(d.kind);
    let af = aa * a.hp / maxhp(a);
    let df = dd * d.hp / maxhp(d) * def_bonus(s, dst, d);
    let tot = af + df;
    if tot <= 0.0 { return (0.0, 0.0); }
    let dmg = (af / tot * aa * 4.5 + 0.5).floor();
    let ret = if d.hp > dmg && !stiff(d.kind) && s.d(src, dst) <= dr { (df / tot * dd * 4.5 + 0.5).floor() } else { 0.0 };
    (dmg, ret)
}

struct Ctx { dcap_l: Vec<i32>, dcap_n: Vec<i32>, druin_l: Vec<i32>, dexp_l: Vec<i32>, dexp_n: Vec<i32>, den_l: Vec<i32>, den_n: Vec<i32>,
    threat: Vec<bool>, enemy_city_close: bool, my_defenders: i32, my_units: i32, my_cities: i32, water_frac: f64, enemy_near_city: bool }

fn build_ctx(s: &St) -> Ctx {
    let me = s.me;
    let mut cap = vec![]; let mut ruin = vec![]; let mut exp = vec![]; let mut en = vec![];
    let mut my_units = 0; let mut my_cities = 0; let mut my_defenders = 0;
    let mut water = 0; let mut known = 0;
    for i in 0..s.n {
        if s.known[i] {
            known += 1;
            if s.terrain[i] == 1 || s.terrain[i] == 2 { water += 1; }
            if s.structure[i] == 1 && s.owner[i] != me { cap.push(i); }
            if s.structure[i] == 2 { ruin.push(i); }
            if s.nbrs(i).iter().any(|&j| !s.known[j]) { exp.push(i); }
        } else { exp.push(i); }
        if let Some(u) = s.units[i] { if u.owner == me { my_units += 1; if u.kind == 5 { my_defenders += 1; } } else { en.push(i); } }
        if s.city_owner[i] == me { my_cities += 1; }
    }
    let mut threat = vec![false; s.n];
    let mut enemy_city_close = false;
    for i in 0..s.n { if s.city_owner[i] == me { for j in 0..s.n { if s.city_owner[j] > 0 && s.city_owner[j] != me && s.d(i, j) <= 6 { enemy_city_close = true; } } } }
    let mut enemy_near_city = false;
    for i in 0..s.n {
        if s.city_owner[i] == me {
            if en.iter().any(|&e| s.d(e, i) <= 3) { threat[i] = true; enemy_near_city = true; }
        }
    }
    Ctx {
        dcap_l: s.dist_map(&cap, false), dcap_n: s.dist_map(&cap, true), druin_l: s.dist_map(&ruin, false),
        dexp_l: s.dist_map(&exp, false), dexp_n: s.dist_map(&exp, true), den_l: s.dist_map(&en, false), den_n: s.dist_map(&en, true),
        threat, enemy_city_close, my_defenders, my_units, my_cities, water_frac: if known > 0 { water as f64 / known as f64 } else { 0.0 }, enemy_near_city,
    }
}

fn gain(m: &[i32], a: usize, b: usize) -> f64 {
    if m[a] >= INF || m[b] >= INF { return 0.0; }
    (m[a] - m[b]) as f64
}

fn tech_priority(s: &St, c: &Ctx) -> Vec<usize> {
    if c.water_frac > 0.3 {
        vec![10, 1, 13, 6, 44, 18, 20, 7, 23, 24, 8, 16, 2, 3, 4, 17, 9, 19, 21, 5, 12, 14, 22]
    } else if c.water_frac > 0.15 {
        vec![1, 6, 10, 18, 20, 13, 7, 44, 23, 24, 8, 16, 2, 3, 4, 17, 9, 19, 21, 5, 12, 14, 22]
    } else {
        let _ = s;
        vec![1, 6, 18, 7, 2, 3, 20, 23, 24, 8, 16, 10, 4, 17, 9, 19, 21, 13, 44, 5, 12, 14, 22]
    }
}

fn score(s: &St, c: &Ctx, cmd: &Value) -> f64 {
    let kind = cmd["kind"].as_i64().unwrap_or(0);
    let src = cmd["source"].as_i64().unwrap_or(-1);
    let tgt = cmd["target"].as_i64().unwrap_or(-1);
    let par = cmd["parameter"].as_i64().unwrap_or(0);
    let me = s.me;
    match kind {
        9 => {
            // city reward
            let t = tgt.max(0) as usize; let thr = t < s.n && c.threat[t];
            match par { 3 => 1000.0, 4 => 900.0, 7 => 1000.0, 1 => if thr { 1100.0 } else { 900.0 }, 8 => 1000.0, 5 => 900.0, 6 => 1000.0, 2 => 900.0, _ => 800.0 }
        }
        8 => {
            let t = src as usize;
            match s.structure[t] { 1 => 950.0, 2 => 850.0, _ => 700.0 }
        }
        3 => {
            match par { 16 => 880.0, 19 => if tgt == 0 { 5.0 } else { -1.0 }, _ => -1.0 }
        }
        2 => {
            let (a, d) = (src as usize, tgt as usize);
            let (Some(au), Some(du)) = (s.units[a], s.units[d]) else { return -1.0 };
            if au.kind == 10 { return 500.0; }
            let (dmg, ret) = combat(s, &au, a, &du, d);
            let val = cost(du.kind) as f64;
            if dmg >= du.hp {
                return 700.0 + val * 5.0 + if s.city_owner[d] >= 0 { 50.0 } else { 0.0 };
            }
            if ret >= au.hp { return -1.0; }
            let on_my_city = s.city_owner[d] == me;
            let net = dmg - ret * 0.8;
            if net > 0.0 || on_my_city { 400.0 + net * 10.0 + if on_my_city { 100.0 } else { 0.0 } } else { -1.0 }
        }
        5 => 520.0,
        7 => {
            let pr = tech_priority(s, c);
            let p = par as usize;
            match pr.iter().position(|&t| t == p) {
                Some(i) => {
                    // early techs high, later ones only if rich
                    let base = 470.0 - i as f64 * 8.0;
                    if i > 6 && s.stars < 12 { 100.0 - i as f64 } else { base }
                }
                None => -1.0,
            }
        }
        4 => {
            let t = src as usize;
            let k = par as i32;
            if k == 42 || k == 44 || k == 41 {
                return if k == 44 { 480.0 } else if k == 42 { 460.0 } else { 300.0 };
            }
            let val = match k { 4 => 60.0, 11 => 50.0, 3 => 42.0, 9 => 38.0, 8 => 30.0, 5 => if c.threat[t] && c.my_defenders <= c.my_cities { 55.0 } else if c.threat[t] { 30.0 } else { 12.0 }, 2 => 25.0, 10 => 10.0, _ => 15.0 };
            let desired = 2 + 2 * c.my_cities + (s.turn / 12).min(8) + if c.enemy_city_close { 2 } else { 0 };
            let threat = if c.threat[t] { 250.0 } else { 0.0 };
            if c.my_units < desired || c.enemy_near_city { 480.0 + val + threat } else { 150.0 + val + threat }
        }
        6 => {
            let k = par;
            let t = tgt as usize;
            let _ = t;
            match k {
                23..=29 => 800.0,
                5 | 21 => 330.0,
                6 | 13 | 22 => 320.0,
                12 => 250.0,
                17 | 18 | 19 | 20 => if s.stars >= 25 { 120.0 } else { -1.0 },
                50 => 200.0,
                8 => {
                    if c.water_frac > 0.15 { 360.0 } else { 50.0 }
                }
                _ => -1.0,
            }
        }
        1 => {
            let (a, b) = (src as usize, tgt as usize);
            let Some(u) = s.units[a] else { return -1.0 };
            let nav = naval(u.kind);
            let (dcap, dexp, den) = if nav { (&c.dcap_n, &c.dexp_n, &c.den_n) } else { (&c.dcap_l, &c.dexp_l, &c.den_l) };
            let mut sc = 0.0;
            sc += 12.0 * gain(dcap, a, b);
            if !nav { sc += 5.0 * gain(&c.druin_l, a, b); }
            sc += 2.5 * gain(dexp, a, b);
            let (att, _, _, _) = stats(u.kind);
            let healthy = u.hp >= 0.6 * maxhp(&u);
            if att >= 2.0 && healthy { sc += 6.0 * gain(den, a, b); }
            if !healthy { sc -= 4.0 * gain(den, a, b); }
            if s.structure[b] == 1 && s.owner[b] != me { sc += 60.0; }
            if s.structure[b] == 2 { sc += 40.0; }
            if c.threat[b] && s.units[b].is_none() { sc += 120.0; }
            if s.city_owner[a] == me {
                if c.threat[a] { sc -= 150.0; } else { sc += 4.0; }
            }
            // naval boarding: move into port when water map and nothing to capture on land
            if !nav && s.structure[b] == 8 {
                if c.dcap_l[a] >= INF || c.dcap_l[a] > 12 { sc += 30.0; } else { sc -= 20.0; }
            }
            // danger: enemies that could reach and hit us next turn
            let mut danger = 0.0;
            let mut mu = u;
            for j in 0..s.n {
                if let Some(e) = s.units[j] {
                    if e.owner != me && e.owner > 0 {
                        let (ea, _, _, er) = stats(e.kind);
                        if ea <= 0.0 { continue; }
                        let reach = match e.kind { 3 => 3, 4 => 4, 42 | 44 => 4, 41 => 5, 8 => 4, 9 => 3, _ => 2 } + if er > 1 { 0 } else { 0 };
                        if s.d(j, b) <= reach {
                            let (dmg, _) = combat(s, &e, j, &mu, b);
                            danger += dmg;
                            // opportunity: we can hit it after moving (dash)
                            if s.d(j, b) == 1 && !u.attacked && matches!(u.kind, 2|3|4|9|11|37|40|42|44) {
                                let (k, _) = combat(s, &u, b, &e, j);
                                if k >= e.hp { sc += 35.0; } else { sc += 5.0; }
                            }
                        }
                    }
                }
            }
            mu.hp = u.hp;
            if danger >= u.hp { danger = 30.0 + 3.0 * cost(u.kind) as f64 + danger; } else { danger *= 0.8; }
            if s.city_owner[b] == me { danger *= 0.5; }
            sc -= danger;
            if sc > 0.5 { 100.0 + sc } else { -1.0 }
        }
        10 => 0.0,
        _ => -1.0,
    }
}

fn choose(obs: &Value) -> Option<String> {
    let legal = obs["legal_actions"].as_array()?;
    if legal.is_empty() { return None; }
    let s = parse(obs);
    let c = build_ctx(&s);
    let mut best: Option<(f64, &Value)> = None;
    for a in legal {
        let sc = score(&s, &c, &a["command"]);
        if best.map_or(true, |(b, _)| sc > b) { best = Some((sc, a)); }
    }
    let (bs, ba) = best?;
    if bs <= 0.0 {
        if let Some(e) = legal.iter().find(|a| a["command"]["kind"] == 10) {
            return e["action_id"].as_str().map(|x| x.to_owned());
        }
    }
    ba["action_id"].as_str().map(|x| x.to_owned()).or_else(|| legal[0]["action_id"].as_str().map(|x| x.to_owned()))
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let id = match choose(&obs) { Some(x) => x, None => continue };
        let _ = writeln!(stdout, "{}", json!({"action_id": id}));
        let _ = stdout.flush();
    }
}
