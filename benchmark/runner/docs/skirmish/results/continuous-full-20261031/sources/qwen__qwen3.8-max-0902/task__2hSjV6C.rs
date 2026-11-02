// Bardur 1v1 skirmish bot: priority-scored atomic actions over the public observation.
use serde_json::Value;
use std::io::{self, BufRead, Write};

const DASH: u32 = 1;
const STIFF: u32 = 2;
const FLOAT: u32 = 4;
const CARRY: u32 = 8;
const FORT: u32 = 16;
const SURP: u32 = 32;
const STAT: u32 = 64;
const SCOUT: u32 = 128;

/// (cost, health, attack, defense, movement, range, flags)
fn ustat(k: i32) -> (i32, f32, f32, f32, i32, i32, u32) {
    match k {
        2 => (2, 10.0, 2.0, 2.0, 1, 1, DASH | FORT),
        3 => (3, 10.0, 2.0, 1.0, 2, 1, DASH | FORT),
        4 => (8, 10.0, 3.5, 1.0, 3, 1, DASH | FORT | 1024),
        5 => (3, 15.0, 1.0, 3.0, 1, 1, FORT),
        8 => (8, 10.0, 4.0, 0.0, 1, 3, STIFF),
        9 => (3, 10.0, 2.0, 1.0, 1, 2, DASH | FORT),
        10 => (5, 10.0, 0.0, 1.0, 1, 1, STIFF | STAT),
        11 => (5, 15.0, 3.0, 3.0, 1, 1, DASH),
        12 => (10, 40.0, 5.0, 4.0, 1, 1, STAT),
        15 => (2, 15.0, 3.0, 1.0, 1, 1, DASH | FORT | STAT),
        37 => (2, 10.0, 2.0, 2.0, 1, 1, DASH | STAT | SURP),
        38 => (8, 5.0, 0.0, 0.5, 2, 1, DASH | STAT | STIFF | SCOUT),
        39 => (2, 5.0, 0.0, 0.5, 2, 1, DASH | CARRY | FLOAT | STAT | STIFF | SCOUT),
        40 => (2, 10.0, 2.0, 2.0, 2, 1, DASH | CARRY | FLOAT | STAT | SURP),
        41 => (15, 10.0, 3.0, 2.0, 2, 3, CARRY | FLOAT | STAT | STIFF),
        42 => (5, 10.0, 2.0, 1.0, 3, 2, DASH | CARRY | FLOAT | SCOUT | STAT),
        43 => (0, 10.0, 0.0, 1.0, 2, 2, CARRY | FLOAT | STAT | STIFF),
        44 => (5, 10.0, 3.0, 3.0, 3, 1, DASH | CARRY | FLOAT | STAT),
        45 => (10, 40.0, 4.0, 4.0, 2, 1, CARRY | FLOAT | STAT | STOMP),
        _ => (1, 10.0, 0.0, 0.0, 1, 1, 0),
    }
}
const STOMP: u32 = 2048;
const PERSIST: u32 = 1024;

fn uscore(k: i32) -> f64 {
    match k {
        12 => 60.0,
        41 => 50.0,
        4 => 44.0,
        8 => 42.0,
        11 => 40.0,
        44 => 40.0,
        9 => 34.0,
        5 => 24.0,
        3 => 22.0,
        42 => 20.0,
        2 => 12.0,
        40 => 12.0,
        10 => 6.0,
        43 => 2.0,
        39 => 3.0,
        38 => 5.0,
        _ => 8.0,
    }
}

fn combat(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let tot = af + df;
    let div = if tot > 0.0 { tot } else { 1.0 };
    let atk = ((if tot > 0.0 { (af / div) * aa * 4.5 } else { 0.0 }) + 1e-6 + 0.5).floor();
    let ret = ((if dh > atk && tot > 0.0 { (df / div) * dd * 4.5 } else { 0.0 }) + 1e-6 + 0.5).floor();
    (atk, ret)
}

struct G {
    size: usize,
    n: usize,
    me: i32,
    turn: i32,
    stars: i32,
    known: Vec<bool>,
    terrain: Vec<i32>,
    owner: Vec<i32>,
    structure: Vec<i32>,
    resource: Vec<i32>,
    road: Vec<bool>,
    capital: Vec<i32>,
    uowner: Vec<i32>,
    ukind: Vec<i32>,
    uhp: Vec<f32>,
    umoved: Vec<bool>,
    uatt: Vec<bool>,
    uveteran: Vec<bool>,
    clevel: Vec<i32>,
    cowner: Vec<i32>,
    cwall: Vec<bool>,
    tech: Vec<bool>,
    // derived
    my_cities: Vec<usize>,
    enemy_cities: Vec<usize>,
    neutral_villages: Vec<usize>,
    ruins: Vec<usize>,
    starfish: Vec<usize>,
    enemy_units: Vec<usize>,
    my_units: Vec<usize>,
    frontier: Vec<usize>,
    goal: Vec<i32>,
    water: f64,
    combat_units: usize,
    war: bool,
}

impl G {
    fn dist(&self, a: usize, b: usize) -> i32 {
        let (ax, ay) = ((a % self.size) as i32, (a / self.size) as i32);
        let (bx, by) = ((b % self.size) as i32, (b / self.size) as i32);
        (ax - bx).abs().max((ay - by).abs())
    }
    fn maxhp(&self, t: usize) -> f32 {
        let s = ustat(self.ukind[t]);
        s.1 + if self.uveteran[t] { 5.0 } else { 0.0 }
    }
    /// defensive terrain/city bonus for the unit on tile t (defender's tech unknown -> pessimistic)
    fn def_bonus(&self, t: usize, mine: bool) -> f32 {
        let k = self.ukind[t] as usize;
        let ter = self.terrain[t];
        let b = if ter == 1 || ter == 2 {
            if mine && self.tech[12] { 1.5 } else { 1.0 }
        } else if ter == 5 {
            if (mine && self.tech[18]) || !mine { 1.5 } else { 1.0 }
        } else if ter == 4 {
            if (mine && self.tech[20]) || !mine { 1.5 } else { 1.0 }
        } else if self.structure[t] == 1 && self.cowner[t] == self.uowner[t] && ustat(self.ukind[t] as i32).6 & FORT != 0 {
            if mine && self.cwall[t] { 4.0 } else { 1.5 }
        } else {
            1.0
        };
        let _ = k;
        b
    }
}

fn parse(obs: &Value) -> G {
    let v = &obs["visible_state"];
    let size = v["board_size"].as_i64().unwrap_or(14) as usize;
    let n = size * size;
    let me = obs["player"].as_i64().unwrap_or(1) as i32;
    let mut g = G {
        size, n, me,
        turn: v["turn"].as_i64().unwrap_or(0) as i32,
        stars: v["stars"].as_i64().unwrap_or(0) as i32,
        known: vec![false; n], terrain: vec![3; n], owner: vec![0; n],
        structure: vec![0; n], resource: vec![0; n], road: vec![false; n], capital: vec![0; n],
        uowner: vec![0; n], ukind: vec![0; n], uhp: vec![0.0; n],
        umoved: vec![false; n], uatt: vec![false; n], uveteran: vec![false; n],
        clevel: vec![0; n], cowner: vec![0; n], cwall: vec![false; n],
        tech: vec![false; 128],
        my_cities: Vec::new(), enemy_cities: Vec::new(), neutral_villages: Vec::new(),
        ruins: Vec::new(), starfish: Vec::new(), enemy_units: Vec::new(), my_units: Vec::new(),
        frontier: Vec::new(), goal: vec![-1; n], water: 0.0, combat_units: 0, war: false,
    };
    if let Some(ts) = v["tiles"].as_array() {
        for t in ts {
            let i = t["tile"].as_i64().unwrap_or(-1);
            if i < 0 || (i as usize) >= n { continue; }
            let i = i as usize;
            g.known[i] = t["known"].as_bool().unwrap_or(false);
            if g.known[i] {
                g.terrain[i] = t["terrain"].as_i64().unwrap_or(3) as i32;
                g.owner[i] = t["owner"].as_i64().unwrap_or(0) as i32;
                g.structure[i] = t["structure"].as_i64().unwrap_or(0) as i32;
                g.resource[i] = t["resource"].as_i64().unwrap_or(0) as i32;
                g.road[i] = t["road"].as_i64().unwrap_or(0) != 0;
                g.capital[i] = t["capital"].as_i64().unwrap_or(0) as i32;
            }
        }
    }
    if let Some(us) = v["units"].as_array() {
        for u in us {
            let i = u["tile"].as_i64().unwrap_or(-1);
            if i < 0 || (i as usize) >= n { continue; }
            let i = i as usize;
            g.uowner[i] = u["owner"].as_i64().unwrap_or(0) as i32;
            g.ukind[i] = u["kind"].as_i64().unwrap_or(0) as i32;
            g.uhp[i] = u["health"].as_f64().unwrap_or(10.0) as f32;
            g.uveteran[i] = u["veteran"].as_bool().unwrap_or(false);
            g.umoved[i] = u["moved"].as_bool().unwrap_or(false);
            g.uatt[i] = u["attacked"].as_bool().unwrap_or(false);
        }
    }
    if let Some(cs) = v["cities"].as_array() {
        for c in cs {
            let i = c["tile"].as_i64().unwrap_or(-1);
            if i < 0 || (i as usize) >= n { continue; }
            let i = i as usize;
            g.cowner[i] = c["owner"].as_i64().unwrap_or(0) as i32;
            g.clevel[i] = c["level"].as_i64().unwrap_or(1) as i32;
            if let Some(rw) = c["rewards"].as_array() {
                if rw.len() == 9 { g.cwall[i] = rw[0].as_i64().unwrap_or(0) > 0; }
            }
        }
    }
    if let Some(tl) = v["technology"].as_array() {
        for t in tl {
            let k = t.as_i64().unwrap_or(-1);
            if k >= 0 && k < 128 { g.tech[k as usize] = true; }
        }
    }
    let mut wk = 0.0;
    let mut kt = 0.0;
    for i in 0..n {
        if g.known[i] {
            kt += 1.0;
            if g.terrain[i] <= 2 { wk += 1.0; }
            if g.structure[i] == 1 {
                if g.cowner[i] == me { g.my_cities.push(i); } else if g.cowner[i] == 0 { g.neutral_villages.push(i); }
                else { g.enemy_cities.push(i); }
            }
            if g.structure[i] == 2 && g.uowner[i] == 0 { g.ruins.push(i); }
            if g.resource[i] == 8 { g.starfish.push(i); }
        }
        if g.uowner[i] != 0 {
            if g.uowner[i] == me {
                g.my_units.push(i);
                let st = ustat(g.ukind[i]);
                if st.2 > 0.0 && st.6 & CARRY == 0 { g.combat_units += 1; }
            } else { g.enemy_units.push(i); }
        }
    }
    g.water = if kt > 0.0 { wk / kt } else { 0.0 };
    g.war = !g.enemy_cities.is_empty() || !g.enemy_units.is_empty();
    // frontier tiles: known tiles adjacent to unknown, or unknown tiles next to known
    for i in 0..n {
        if !g.known[i] { continue; }
        let (x, y) = ((i % size) as i32, (i / size) as i32);
        let mut unk = false;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= size as i32 || ny >= size as i32 { continue; }
                if !g.known[(ny * size as i32 + nx) as usize] { unk = true; }
            }
        }
        if unk && g.uowner[i] == 0 && g.terrain[i] > 2 { g.frontier.push(i); }
    }
    if g.frontier.is_empty() { g.frontier.push(n / 2); }
    g.assign_goals();
    g
}

impl G {
    fn assign_goals(&mut self) {
        let n = self.n;
        let aggression = if self.combat_units >= 3 + self.my_cities.len() * 2 && self.war { 1.0 } else { 0.0 };
        let mut used: Vec<f64> = vec![0.0; n];
        // order units: strongest first so they get combat goals
        let mut units = self.my_units.clone();
        units.sort_by(|&a, &b| {
            let ka = ustat(self.ukind[a]).2 * self.uhp[a];
            let kb = ustat(self.ukind[b]).2 * self.uhp[b];
            kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
        });
        for &u in &units {
            let st = ustat(self.ukind[u]);
            let fighter = st.2 > 0.0 && st.6 & CARRY == 0;
            let mut best: Option<(f64, usize)> = None;
            let mut consider = |g: &G, tile: usize, w: f64, extra: f64, best: &mut Option<(f64, usize)>, used: &Vec<f64>| {
                let d = g.dist(u, tile) as f64;
                let v = d * w + extra + used[tile];
                if best.map_or(true, |(bv, _)| v < bv) { *best = Some((v, tile)); }
            };
            if !self.enemy_cities.is_empty() && fighter {
                for &t in &self.enemy_cities { consider(self, t, 1.0 - 0.35 * aggression, 6.0 * (1.0 - aggression), &mut best, &used); }
            }
            if !self.enemy_units.is_empty() && fighter {
                for &t in &self.enemy_units { consider(self, t, 0.9 - 0.3 * aggression, 4.0 * (1.0 - aggression), &mut best, &used); }
            }
            for &t in &self.neutral_villages { consider(self, t, 1.0, if aggression > 0.0 { 5.0 } else { 0.0 }, &mut best, &used); }
            for &t in &self.ruins { consider(self, t, 1.15, if aggression > 0.0 { 6.0 } else { 1.0 }, &mut best, &used); }
            if st.6 & FLOAT != 0 || self.water > 0.15 {
                for &t in &self.starfish { consider(self, t, 1.3, 3.0, &mut best, &used); }
            }
            for &t in &self.frontier { consider(self, t, 1.6, 8.0 + 12.0 * aggression, &mut best, &used); }
            if let Some((_, t)) = best {
                self.goal[u] = t as i32;
                used[t] += 3.0;
            }
        }
    }
    fn threat_at(&self, t: usize) -> i32 {
        let mut c = 0;
        for &e in &self.enemy_units {
            if self.dist(e, t) <= 1 { c += 1; }
        }
        c
    }
}

fn tech_prio(g: &G, k: i32) -> f64 {
    let w = g.water > 0.18;
    match k {
        8 => 96.0,   // Farming: farms (+2 pop)
        18 => 94.0,  // Archery
        6 => 90.0,   // Organization: fruit
        10 => if w { 95.0 } else { 84.0 },
        23 => 85.0,  // Mining
        20 => 80.0,  // Climbing
        13 => if w { 90.0 } else { 58.0 },
        44 => if w { 88.0 } else { 50.0 },
        24 => 78.0,  // Smithery
        9 => 75.0,   // Construction
        17 => 72.0,  // Mathematics
        16 => 68.0,  // Forestry
        5 => 66.0,   // Trade
        22 => 65.0,  // Philosophy (cheaper research)
        4 => 62.0,   // Roads
        1 => 58.0,   // Riding
        7 => 56.0,   // Strategy
        3 => 54.0,   // Chivalry
        14 => if w { 62.0 } else { 52.0 },
        21 => 48.0,
        19 => 44.0,
        12 => 40.0,
        2 => 30.0,
        38 => 18.0,
        _ => 24.0,
    }
}

fn score(g: &G, c: &Value) -> f64 {
    let kind = c["kind"].as_i64().unwrap_or(0);
    let src = c["source"].as_i64().unwrap_or(-1);
    let dst = c["target"].as_i64().unwrap_or(-1);
    let par = c["parameter"].as_i64().unwrap_or(0);
    let s = src as usize;
    let d = dst as usize;
    let ok = |i: i64| i >= 0 && (i as usize) < g.n;
    match kind {
        11 => -1e9,
        10 => 5.0,
        9 => {
            let base = 100000.0;
            let pref = match par {
                6 => 90.0, // Giant
                8 => 80.0, // +3 population
                5 => 70.0, // border growth
                7 => 60.0, // +5 stars
                3 => 55.0, // workshop
                4 => if g.turn < 25 { 65.0 } else { 30.0 },
                1 => 20.0, // wall
                2 => 15.0, // park
                _ => 10.0,
            };
            base + pref
        }
        8 => {
            if !ok(src) { return 90000.0; }
            if g.structure[s] == 1 {
                if g.cowner[s] != 0 { 99000.0 } else { 95000.0 }
            } else if g.structure[s] == 2 {
                93000.0
            } else {
                80000.0
            }
        }
        2 => {
            if !ok(src) || !ok(dst) { return -1e8; }
            let (ac, ah, aa, _ad, _am, arange, aflag) = ustat(g.ukind[s]);
            let (_dc, dh, _da, dd, _dm, drange, dflag) = ustat(g.ukind[d]);
            let _ = (ac, arange);
            let amh = g.maxhp(s);
            let dmh = g.maxhp(d);
            let db = g.def_bonus(d, false);
            let (dmg, ret) = combat(aa, g.uhp[s], amh, dd, g.uhp[d], dmh, db);
            let dist = g.dist(s, d);
            let can_ret = dflag & STIFF == 0 && dist <= ustat(g.ukind[d]).5 && !(aflag & SURP != 0);
            let mut v = 40000.0;
            v += dmg as f64 * 400.0;
            let enemy_on_city = g.structure[d] == 1 && g.cowner[d] != 0 && g.cowner[d] != g.me;
            if dmg >= g.uhp[d] {
                v += 9000.0 + uscore(g.ukind[d]) * 120.0;
                if enemy_on_city { v += 60000.0; }
            }
            if can_ret {
                v -= ret as f64 * 350.0;
                if g.uhp[s] - ret <= 0.0 {
                    v -= 25000.0;
                    if dmg >= g.uhp[d] { v += 8000.0; }
                }
            }
            if g.uhp[s] <= dmg && dmg < g.uhp[d] { v -= 5000.0; }
            v
        }
        5 => {
            let mut v = 30000.0;
            if g.combat_units < 3 + g.my_cities.len() * 2 { v -= 22000.0; }
            v
        }
        7 => 31000.0 + tech_prio(g, par as i32) * 10.0,
        6 => {
            let (pop, cost) = match par {
                5 => (2.0, 5.0), 21 => (2.0, 5.0), 12 => (1.0, 3.0), 8 => (1.0, 7.0),
                6 | 13 => (1.0, 5.0), 22 => (2.0, 5.0), 50 => (0.0, 5.0),
                17 | 18 | 19 | 20 => (1.0, 20.0),
                3 => (0.0, 3.0), 48 => (0.0, 5.0),
                _ => (3.0, 0.0),
            };
            let mut adj = 0.0;
            if ok(dst) {
                let (x, y) = ((d % g.size) as i32, (d / g.size) as i32);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= g.size as i32 || ny >= g.size as i32 { continue; }
                        let j = (ny * g.size as i32 + nx) as usize;
                        let st = g.structure[j];
                        if g.owner[j] == g.me && ((par == 6 && st == 5) || (par == 13 && st == 12)
                            || (par == 22 && st == 21) || (par == 50 && (st == 6 || st == 13 || st == 22))) { adj += 1.0; }
                    }
                }
            }
            let base = match par {
                5 | 21 => 30000.0,
                22 => 26000.0 + adj * 3000.0,
                6 | 13 => 25000.0 + adj * 3000.0,
                50 => 29000.0 + adj * 1500.0,
                12 => 24000.0,
                8 => if g.water > 0.1 { 23000.0 } else { 16000.0 },
                17 | 18 | 19 | 20 => 12000.0,
                3 => 9000.0,
                48 => 9500.0,
                _ => 34000.0, // monuments: +3 pop, free
            };
            let _ = (pop, cost);
            base
        }
        4 => {
            let need = if g.war { g.my_cities.len() as i32 + 2 } else { 2 };
            let base = if (g.combat_units as i32) < need + 2 { 37000.0 } else { 32500.0 };
            let mut v = base + uscore(par as i32) * 10.0;
            if ok(src) && !g.enemy_cities.is_empty() {
                let e = g.enemy_cities[0];
                v -= g.dist(s, e) as f64 * 4.0;
            }
            if par == 43 || par == 39 { v -= 4000.0; }
            v
        }
        3 => match par {
            16 => 36000.0,                       // promote (free +5 hp)
            7 => {                                  // recover
                if !ok(src) { return 0.0; }
                let mh = g.maxhp(s);
                if g.uhp[s] >= mh { return -10.0; }
                if g.threat_at(s) > 0 || g.enemy_units.iter().any(|&e| g.dist(e, s) <= 2) { 900.0 } else { 11500.0 + (mh - g.uhp[s]) as f64 * 60.0 }
            }
            9 => 12000.0,                           // heal others
            8 | 4 => -1e6,                          // disband / destroy
            1 => -50.0,                             // burn forest
            2 => -60.0,                             // clear forest
            3 => -70.0,                             // grow forest
            19 => {                                 // peace response: accept when clearly losing
                let mine: f32 = g.my_units.iter().map(|&t| ustat(g.ukind[t]).2 * g.uhp[t]).sum();
                let theirs: f32 = g.enemy_units.iter().map(|&t| ustat(g.ukind[t]).2 * g.uhp[t]).sum();
                if c["target"].as_i64().unwrap_or(0) == 1 {
                    if mine + 1.0 < theirs * 0.55 && g.my_cities.len() <= g.enemy_cities.len() { 30500.0 } else { -5000.0 }
                } else {
                    if mine + 1.0 < theirs * 0.55 { -4000.0 } else { 100.0 }
                }
            }
            21 => -3000.0,                          // offer peace
            18 => -1e6,                             // break peace
            _ => -1000.0,
        },
        1 => {
            if !ok(src) || !ok(dst) { return -1e8; }
            let goal = if g.goal[s] >= 0 { g.goal[s] as usize } else { s };
            let d0 = g.dist(s, goal) as f64;
            let d1 = g.dist(d, goal) as f64;
            let mut v = 14.0 * (d0 - d1);
            let st = ustat(g.ukind[s]);
            // standing on a capturable tile
            if g.structure[d] == 1 && g.cowner[d] != g.me { v += 3000.0; }
            if g.structure[d] == 2 { v += 2500.0; }
            if g.resource[d] == 8 { v += 1500.0; }
            // move into attack range
            if st.2 > 0.0 {
                let r = st.5;
                for &e in &g.enemy_units {
                    if g.dist(d, e) <= r && g.dist(s, e) > r { v += 7000.0; }
                }
                for &e in &g.enemy_cities {
                    if g.uowner[e] != 0 && g.dist(d, e) <= r && g.dist(s, e) > r { v += 7000.0; }
                }
            }
            // exploration
            let (x, y) = ((d % g.size) as i32, (d / g.size) as i32);
            let mut unk = 0.0;
            let rad = if g.terrain[d] == 4 || st.6 & SCOUT != 0 { 2 } else { 1 };
            for dy in -rad..=rad {
                for dx in -rad..=rad {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= g.size as i32 || ny >= g.size as i32 { continue; }
                    if !g.known[(ny * g.size as i32 + nx) as usize] { unk += 1.0; }
                }
            }
            v += unk * 6.0;
            if g.terrain[d] == 5 { v += 4.0; }
            if g.terrain[d] == 4 { v += 6.0; }
            if g.road[d] { v += 3.0; }
            if g.owner[d] == g.me { v += 2.0; }
            // danger
            let th = g.threat_at(d);
            if th > 0 {
                let (dc, dh, da, dd2, _, _, _) = ustat(g.ukind[d]);
                let _ = (dc, dh, da, dd2);
                let mypower = st.2 * g.uhp[s];
                let mut ep = 0.0;
                for &e in &g.enemy_units {
                    if g.dist(e, d) <= 1 { ep += ustat(g.ukind[e]).2 * g.uhp[e] * 1.2; }
                }
                if ep > mypower { v -= 6000.0 * th as f64; } else { v -= 100.0 * th as f64; }
            }
            if g.uowner[d] != 0 && g.uowner[d] != g.me { v -= 500.0; }
            v
        }
        _ => -1e8,
    }
}

fn choose(obs: &Value) -> String {
    let legal = obs["legal_actions"].as_array();
    let Some(legal) = legal else { return String::new() };
    if legal.is_empty() { return String::new() };
    let g = parse(obs);
    let mut best: f64 = f64::NEG_INFINITY;
    let mut pick: Option<&str> = None;
    for a in legal {
        let sc = score(&g, &a["command"]);
        if sc > best {
            best = sc;
            pick = a["action_id"].as_str();
        }
    }
    if let Some(p) = pick { return p.to_string(); }
    for a in legal {
        if a["command"]["kind"].as_i64() == Some(10) {
            return a["action_id"].as_str().unwrap_or("0").to_string();
        }
    }
    legal[0]["action_id"].as_str().unwrap_or("0").to_string()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => continue };
        let obs: Value = match serde_json::from_str(&line) { Ok(o) => o, Err(_) => continue };
        let id = choose(&obs);
        let out = format!("{{\"action_id\":\"{}\"}}", id);
        let _ = writeln!(stdout, "{}", out);
        let _ = stdout.flush();
    }
}
