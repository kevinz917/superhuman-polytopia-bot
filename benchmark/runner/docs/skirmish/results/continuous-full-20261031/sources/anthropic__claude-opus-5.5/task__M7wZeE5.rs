use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

// (attack, defense, move, range, hp, cost)
fn ustat(k: i64) -> (f64, f64, i64, i64, f64, i64) {
    match k {
        2 => (2.0, 2.0, 1, 1, 10.0, 2),
        3 => (2.0, 1.0, 2, 1, 10.0, 3),
        4 => (3.5, 1.0, 3, 1, 10.0, 8),
        5 => (1.0, 3.0, 1, 1, 15.0, 3),
        8 => (4.0, 0.0, 1, 3, 10.0, 8),
        9 => (2.0, 1.0, 1, 2, 10.0, 3),
        10 => (0.0, 1.0, 1, 1, 10.0, 5),
        11 => (3.0, 3.0, 1, 1, 15.0, 5),
        12 => (5.0, 4.0, 1, 1, 40.0, 10),
        15 => (3.0, 1.0, 1, 1, 15.0, 2),
        39 => (0.0, 0.5, 2, 1, 5.0, 2),
        40 => (2.0, 2.0, 2, 1, 10.0, 2),
        41 => (3.0, 2.0, 2, 3, 10.0, 15),
        42 => (2.0, 1.0, 3, 2, 10.0, 5),
        43 => (0.0, 1.0, 2, 2, 10.0, 0),
        44 => (3.0, 3.0, 3, 1, 10.0, 5),
        45 => (4.0, 4.0, 2, 1, 40.0, 10),
        _ => (2.0, 2.0, 1, 1, 10.0, 3),
    }
}
fn stiff(k: i64) -> bool { matches!(k, 8 | 10 | 39 | 41 | 43 | 45) }
fn can_fortify(k: i64) -> bool { matches!(k, 2 | 3 | 4 | 5 | 9 | 15) }

#[derive(Clone, Default)]
struct Unit { tile: i64, owner: i64, kind: i64, hp: f64, maxhp: f64, moved: bool, attacked: bool }

struct Ctx {
    n: i64,
    me: i64,
    stars: i64,
    turn: i64,
    tech: Vec<i64>,
    known: Vec<bool>,
    terrain: Vec<i64>,
    owner: Vec<i64>,
    structure: Vec<i64>,
    resource: Vec<i64>,
    units: Vec<Unit>,
    unit_at: Vec<i64>,
    my_cities: Vec<i64>,
    city_level: Vec<i64>,
    targets: Vec<(i64, f64)>, // tile, value
    fields: Vec<(Vec<i32>, f64)>,
    need_navy: bool,
    navy_goal: Vec<i64>,
}
fn naval(k: i64) -> bool { matches!(k, 39 | 40 | 41 | 42 | 43 | 44 | 45) }

struct Mem {
    enemy_cities: Vec<i64>,
    seen_villages: Vec<i64>,
}

impl Ctx {
    fn d(&self, a: i64, b: i64) -> i64 {
        let n = self.n;
        (a % n - b % n).abs().max((a / n - b / n).abs())
    }
    fn has(&self, t: i64) -> bool { self.tech.contains(&t) }
    fn def_bonus(&self, u: &Unit) -> f64 {
        let t = u.tile as usize;
        let ter = self.terrain[t];
        // enemy tech unknown: assume they have common defensive techs moderately
        if ter == 5 || ter == 4 { return 1.5; }
        if ter == 1 || ter == 2 { return 1.0; }
        if self.structure[t] == 1 && self.owner[t] == u.owner && can_fortify(u.kind) { return 1.5; }
        1.0
    }
    fn combat(&self, a: &Unit, d: &Unit) -> (f64, f64) {
        let (aa, _, _, _, _, _) = ustat(a.kind);
        let (_, dd, _, drange, _, _) = ustat(d.kind);
        let af = aa * a.hp / a.maxhp;
        let df = dd * d.hp / d.maxhp * self.def_bonus(d);
        let tot = af + df;
        if tot <= 0.0 { return (0.0, 0.0); }
        let dmg = (af / tot * aa * 4.5).round();
        let mut ret = (df / tot * dd * 4.5).round();
        if d.hp - dmg <= 0.0 || stiff(d.kind) || self.d(a.tile, d.tile) > drange { ret = 0.0; }
        (dmg, ret)
    }
    fn enemy_near(&self, tile: i64, r: i64) -> i64 {
        self.units.iter().filter(|u| u.owner != self.me && self.d(u.tile, tile) <= r).count() as i64
    }
    fn land(&self, i: usize) -> bool { self.known[i] && matches!(self.terrain[i], 3 | 4 | 5 | 6) }
    fn bfs(&self, src: i64) -> Vec<i32> {
        let nn = (self.n * self.n) as usize;
        let mut d = vec![-1i32; nn];
        let mut q = std::collections::VecDeque::new();
        d[src as usize] = 0; q.push_back(src);
        while let Some(t) = q.pop_front() {
            if t != src && !self.land(t as usize) { continue; }
            for dy in -1..=1 { for dx in -1..=1 {
                let x = t % self.n + dx; let y = t / self.n + dy;
                if x < 0 || y < 0 || x >= self.n || y >= self.n { continue; }
                let j = (y * self.n + x) as usize;
                if d[j] >= 0 { continue; }
                // can step onto any land tile; also onto own port (embark)
                if self.land(j) || (self.structure[j] == 8 && self.owner[j] == self.me) {
                    d[j] = d[t as usize] + 1;
                    if self.land(j) { q.push_back(j as i64); }
                }
            }}
        }
        d
    }
    fn potential(&self, tile: i64, kind: i64) -> f64 {
        if !naval(kind) {
            let sp = ustat(kind).2 as f64;
            let mut best = 0.0f64;
            for (f, v) in &self.fields {
                let dd = f[tile as usize];
                if dd < 0 { continue; }
                let p = v / (1.0 + dd as f64 / sp.max(1.0));
                if p > best { best = p; }
            }
            return best;
        }
        let mut best = 0.0f64;
        for &(t, v) in &self.targets {
            let dd = self.d(tile, t) as f64;
            let sp = ustat(kind).2 as f64;
            let p = v / (1.0 + dd / sp.max(1.0));
            if p > best { best = p; }
        }
        best
    }
}

fn parse(obs: &Value, mem: &mut Mem) -> Ctx {
    let v = &obs["visible_state"];
    let n = v["board_size"].as_i64().unwrap_or(14);
    let nn = (n * n) as usize;
    let me = obs["player"].as_i64().unwrap_or(1);
    let mut c = Ctx {
        n, me, stars: v["stars"].as_i64().unwrap_or(0), turn: v["turn"].as_i64().unwrap_or(0),
        tech: v["technology"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default(),
        known: vec![false; nn], terrain: vec![0; nn], owner: vec![0; nn], structure: vec![0; nn], resource: vec![0; nn],
        units: vec![], unit_at: vec![-1; nn], my_cities: vec![], city_level: vec![0; nn], targets: vec![], fields: vec![], need_navy: false, navy_goal: vec![],
    };
    if let Some(ts) = v["tiles"].as_array() {
        for t in ts {
            let i = t["tile"].as_i64().unwrap_or(-1);
            if i < 0 || i as usize >= nn { continue; }
            let i = i as usize;
            if t["known"].as_bool() == Some(true) {
                c.known[i] = true;
                c.terrain[i] = t["terrain"].as_i64().unwrap_or(0);
                c.owner[i] = t["owner"].as_i64().unwrap_or(0);
                c.structure[i] = t["structure"].as_i64().unwrap_or(0);
                c.resource[i] = t["resource"].as_i64().unwrap_or(0);
            }
        }
    }
    if let Some(cs) = v["cities"].as_array() {
        for ct in cs {
            let t = ct["tile"].as_i64().unwrap_or(-1);
            if t < 0 || t as usize >= nn { continue; }
            c.city_level[t as usize] = ct["level"].as_i64().unwrap_or(1);
            if ct["owner"].as_i64() == Some(me) { c.my_cities.push(t); }
        }
    }
    if let Some(us) = v["units"].as_array() {
        for u in us {
            let kind = u["kind"].as_i64().unwrap_or(2);
            let vet = u["veteran"].as_bool().unwrap_or(false);
            let unit = Unit {
                tile: u["tile"].as_i64().unwrap_or(0), owner: u["owner"].as_i64().unwrap_or(0), kind,
                hp: u["health"].as_f64().unwrap_or(10.0), maxhp: ustat(kind).4 + if vet { 5.0 } else { 0.0 },
                moved: u["moved"].as_bool().unwrap_or(false), attacked: u["attacked"].as_bool().unwrap_or(false),
            };
            if (unit.tile as usize) < nn { c.unit_at[unit.tile as usize] = c.units.len() as i64; }
            c.units.push(unit);
        }
    }
    // memory of cities
    for i in 0..nn {
        if c.known[i] && c.structure[i] == 1 {
            let t = i as i64;
            if c.owner[i] != me && c.owner[i] != 0 { if !mem.enemy_cities.contains(&t) { mem.enemy_cities.push(t); } }
            else { mem.enemy_cities.retain(|&x| x != t); }
            if c.owner[i] == 0 { if !mem.seen_villages.contains(&t) { mem.seen_villages.push(t); } }
            else { mem.seen_villages.retain(|&x| x != t); }
        }
    }
    // targets
    let mut targets = vec![];
    for &t in &mem.seen_villages { targets.push((t, 100.0)); }
    for &t in &mem.enemy_cities { targets.push((t, 80.0)); }
    if mem.enemy_cities.is_empty() {
        // guess enemy capital: point mirror of my capital
        if let Some(&cap) = c.my_cities.first() {
            let g = (n - 1 - cap / n) * n + (n - 1 - cap % n);
            if !c.known[g as usize] || c.structure[g as usize] == 1 { targets.push((g, 60.0)); }
        }
    }
    let nunk = c.known.iter().filter(|k| !**k).count();
    for i in 0..nn {
        if !c.known[i] {
            // exploration value: higher when near known tiles (frontier)
            let ii = i as i64;
            let mut frontier = false;
            for dy in -1..=1 { for dx in -1..=1 {
                let x = ii % n + dx; let y = ii / n + dy;
                if x >= 0 && y >= 0 && x < n && y < n && c.known[(y * n + x) as usize] { frontier = true; }
            }}
            if frontier { targets.push((ii, if nunk > 20 { 35.0 } else { 15.0 })); }
        }
    }
    for u in &c.units {
        if u.owner != me && u.owner != 0 {
            // enemy units near my cities are high-value targets
            let near = c.my_cities.iter().any(|&ct| c.d(ct, u.tile) <= 2);
            targets.push((u.tile, if near { 90.0 } else { 40.0 }));
        }
    }
    c.targets = targets;
    // land distance fields from each target (symmetric)
    let mut fields = vec![];
    let mut unreachable = vec![];
    let city_fields: Vec<Vec<i32>> = c.my_cities.iter().map(|&ct| c.bfs(ct)).collect();
    for &(t, v) in &c.targets {
        let f = c.bfs(t);
        if v >= 30.0 && !city_fields.is_empty() && city_fields.iter().all(|cf| cf[t as usize] < 0) && f.iter().enumerate().all(|(i, &x)| x < 0 || !(c.owner[i] == me)) {
            unreachable.push(t);
        }
        fields.push((f, v));
    }
    let reach_hi = c.targets.iter().zip(fields.iter()).any(|(&(t, v), _)| v >= 60.0 && city_fields.iter().any(|cf| cf[t as usize] >= 0));
    let hi_unreach = c.targets.iter().any(|&(t, v)| v >= 60.0 && unreachable.contains(&t));
    if !unreachable.is_empty() && ((c.turn >= 8 && !reach_hi) || (c.turn >= 25 && hi_unreach)) {
        c.need_navy = true;
        let ports: Vec<i64> = (0..nn).filter(|&i| c.structure[i] == 8 && c.owner[i] == me).map(|i| i as i64).collect();
        for p in ports { fields.push((c.bfs(p), 70.0)); }
    }
    c.navy_goal = unreachable;
    c.fields = fields;
    c
}

fn tech_value(c: &Ctx, t: i64) -> f64 {
    let cnt = |res: i64| -> f64 {
        let mut k = 0.0;
        for i in 0..c.known.len() { if c.owner[i] == c.me && c.resource[i] == res { k += 1.0; } }
        k
    };
    let fruit = cnt(6); let fish = cnt(3); let crop = cnt(2); let metal = cnt(5);
    let forest = (0..c.known.len()).filter(|&i| c.owner[i] == c.me && c.terrain[i] == 5).count() as f64;
    let water = (0..c.known.len()).filter(|&i| c.known[i] && (c.terrain[i] == 1 || c.terrain[i] == 2)).count() as f64;
    let ntiles = c.known.iter().filter(|k| **k).count() as f64;
    let water_frac = if ntiles > 0.0 { water / ntiles } else { 0.0 };
    match t {
        1 => 60.0,                     // Riding
        6 => 25.0 + 15.0 * fruit,      // Organization
        10 => 10.0 + 15.0 * fish + 80.0 * water_frac + if c.need_navy { 150.0 } else { 0.0 }, // Fishing
        20 => 25.0,                    // Climbing
        18 => 40.0,                    // Archery
        16 => 10.0 + 4.0 * forest.min(8.0),     // Forestry
        8 => 10.0 + 15.0 * crop,       // Farming
        23 => 10.0 + 15.0 * metal,     // Mining
        24 => if c.turn > 12 { 55.0 } else { 15.0 }, // Smithery
        4 => 20.0,                     // Roads
        2 => 15.0,                     // FreeSpirit
        3 => if c.turn > 10 { 50.0 } else { 15.0 }, // Chivalry
        7 => 25.0,                     // Strategy
        17 => if c.turn > 12 { 45.0 } else { 10.0 }, // Mathematics
        13 => 10.0 + 60.0 * water_frac + if c.need_navy { 60.0 } else { 0.0 }, // Sailing
        44 => 5.0 + 40.0 * water_frac + if c.need_navy { 40.0 } else { 0.0 }, // Ramming
        14 => 20.0 * water_frac,
        9 => 10.0, 19 => 8.0, 21 => 5.0, 22 => 10.0, 5 => 15.0, 12 => 5.0,
        _ => 0.0,
    }
}

fn unit_value(c: &Ctx, k: i64, threat: bool) -> f64 {
    let late = c.turn > 10;
    match k {
        2 => if late { 20.0 } else { 40.0 },
        3 => 55.0,
        4 => 85.0,
        5 => if threat { 60.0 } else { 20.0 },
        8 => if late { 60.0 } else { 10.0 },
        9 => 50.0,
        11 => 80.0,
        12 => 100.0,
        42 => 45.0,
        44 => 55.0,
        43 => 10.0,
        10 => 5.0,
        _ => 10.0,
    }
}

fn v_up(p: i64) -> f64 { match p { 44 => 30.0, 42 => 20.0, 41 => 25.0, _ => 0.0 } }

fn score(c: &Ctx, a: &Value, mem: &Mem) -> f64 {
    let cmd = &a["command"];
    let k = cmd["kind"].as_i64().unwrap_or(-1);
    let src = cmd["source"].as_i64().unwrap_or(-1);
    let dst = cmd["target"].as_i64().unwrap_or(-1);
    let p = cmd["parameter"].as_i64().unwrap_or(0);
    let nn = (c.n * c.n) as i64;
    let my_units = c.units.iter().filter(|u| u.owner == c.me).count() as i64;
    let ncity = c.my_cities.len().max(1) as i64;
    let threat_any = c.my_cities.iter().any(|&ct| c.enemy_near(ct, 3) > 0);
    match k {
        11 => -1e9,
        10 => 0.0,
        9 => 10000.0 + match p { 6 => 95.0, 3 => 90.0, 8 => 85.0, 7 => 80.0, 2 => 70.0, 5 => 60.0, 1 => 50.0, 4 => 40.0, _ => 30.0 },
        8 => 9000.0,
        2 => {
            if src < 0 || dst < 0 || src >= nn || dst >= nn { return 1.0; }
            let ai = c.unit_at[src as usize]; let di = c.unit_at[dst as usize];
            if ai < 0 || di < 0 { return 500.0; }
            let au = &c.units[ai as usize]; let du = &c.units[di as usize];
            let (dmg, ret) = c.combat(au, du);
            let dval = ustat(du.kind).5 as f64 + 2.0;
            let on_my_city = c.structure[dst as usize] == 1 && c.owner[dst as usize] == c.me;
            if dmg >= du.hp {
                5000.0 + dval * 10.0 + if on_my_city { 2000.0 } else { 0.0 }
            } else {
                let aval = ustat(au.kind).5 as f64 + 2.0;
                let my_near = c.units.iter().filter(|u| u.owner == c.me && c.d(u.tile, dst) <= 3).count() as f64;
                let en_near = c.units.iter().filter(|u| u.owner != c.me && c.d(u.tile, dst) <= 3).count() as f64;
                let in_enemy_city = c.structure[dst as usize] == 1 && c.owner[dst as usize] != c.me && c.owner[dst as usize] != 0;
                let ag = if naval(au.kind) { 1.5 } else if my_near >= en_near * 2.0 + 2.0 { 0.25 } else if my_near > en_near { 0.6 } else { 1.0 };
                let mut s = dmg * dval / du.maxhp * 30.0 - ag * ret * aval / au.maxhp * 30.0;
                if ret >= au.hp { s -= ag * aval * 40.0; }
                if on_my_city { s += 300.0; }
                if in_enemy_city && ag < 1.0 { s += 40.0; }
                if s > 0.0 { 3000.0 + s } else if on_my_city && ret < au.hp { 2500.0 } else { -1.0 }
            }
        }
        1 => {
            if src < 0 || dst < 0 || src >= nn || dst >= nn { return -1.0; }
            let ui = c.unit_at[src as usize];
            if ui < 0 { return 5.0; }
            let u = &c.units[ui as usize];
            // defend: unit in my city with enemy close stays
            let src_city = c.structure[src as usize] == 1 && c.owner[src as usize] == c.me;
            if src_city && c.enemy_near(src, 2) > 0 && u.owner == c.me {
                // only leave if no one else would... keep simple: stay
                return -1.0;
            }
            let mut s = (c.potential(dst, u.kind) - c.potential(src, u.kind)) * 20.0;
            let st = c.structure[dst as usize];
            if st == 1 && c.owner[dst as usize] != c.me { s += 400.0; }
            if st == 2 { s += 150.0; } // ruin
            // danger: count adjacent enemies at dst
            let danger = c.units.iter().filter(|e| e.owner != c.me && c.d(e.tile, dst) <= ustat(e.kind).3.max(1) + if ustat(e.kind).2 > 1 {1} else {0}).count() as f64;
            let mine_near = c.units.iter().filter(|e| e.owner == c.me && c.d(e.tile, dst) <= 1).count() as f64;
            if danger > 0.0 && u.hp < u.maxhp * 0.5 { s -= 60.0 * danger; }
            if danger > mine_near + 1.0 { s -= 30.0 * (danger - mine_near); }
            // returning to my city under threat
            let dst_city = c.structure[dst as usize] == 1 && c.owner[dst as usize] == c.me;
            if dst_city && c.enemy_near(dst, 3) > 0 && c.unit_at[dst as usize] < 0 { s += 200.0; }
            if s > 0.5 { 1000.0 + s } else { -1.0 }
        }
        3 => {
            match p {
                16 => 8000.0,             // promote
                7 => {                    // recover
                    let ui = if src >= 0 && src < nn { c.unit_at[src as usize] } else { -1 };
                    if ui >= 0 { let u = &c.units[ui as usize]; if u.hp < u.maxhp * 0.6 { return 900.0; } }
                    -1.0
                }
                9 => 700.0,
                _ => -1.0,
            }
        }
        5 => 700.0 - if threat_any { 300.0 } else { 0.0 },
        6 => {
            let cost = match p { 5 => 5, 6 => 5, 8 => 7, 12 => 3, 13 => 5, 21 => 5, 22 => 5, 50 => 5, 3 => 3, 48 => 5, 17 | 18 | 19 | 20 => 20, 32 => 5, _ => 10 };
            if c.stars < cost { return -1.0; }
            let base = match p {
                5 | 21 => 600.0, 6 | 13 | 22 => 620.0, 12 => 550.0, 50 => 500.0,
                8 => {
                    let ports = (0..c.known.len()).filter(|&i| c.owner[i] == c.me && c.structure[i] == 8).count();
                    if c.need_navy && ports < 2 {
                        let dmin = c.navy_goal.iter().map(|&g| c.d(g, dst)).min().unwrap_or(10) as f64;
                        2000.0 - dmin * 30.0 - ports as f64 * 500.0
                    } else if ports == 0 && c.turn > 8 && c.stars > 15 { 300.0 } else { -1.0 }
                }
                23..=29 => 700.0,
                48 => -1.0,
                17 | 18 | 19 | 20 => if c.stars > 30 { 350.0 } else { -1.0 },
                _ => -1.0,
            };
            if threat_any && base > 0.0 { base - 300.0 } else { base }
        }
        7 => {
            let v = tech_value(c, p);
            if v <= 0.0 { return -1.0; }
            // research costs ~ 4 + ncity*tier; keep a reserve if threatened
            if threat_any && c.stars < 10 { return -1.0; }
            let en_vis = c.units.iter().filter(|u| u.owner != c.me && c.my_cities.iter().any(|&ct| c.d(ct, u.tile) <= 5)).count() as i64;
            if en_vis > 0 && my_units <= en_vis + 1 { return 150.0 + v; }
            if my_units < 2 && c.turn > 2 { return 100.0 + v; }
            300.0 + v * 3.0
        }
        4 => {
            let cost = ustat(p).5;
            if c.stars < cost { return -1.0; }
            let threat = dst >= 0 && dst < nn && c.enemy_near(dst, 4) > 0;
            if matches!(p, 41 | 42 | 44) {
                return if c.need_navy || threat { 1500.0 + v_up(p) } else { 300.0 + v_up(p) };
            }
            let want = 3 + ncity * 2 + c.turn / 6;
            let v = unit_value(c, p, threat);
            if v <= 0.0 { return -1.0; }
            if threat { return 4000.0 + v; }
            if my_units < want { 450.0 + v * 4.0 } else if c.stars > 15 { 200.0 + v } else { -1.0 }
        }
        _ => { let _ = mem; -1.0 }
    }
}

fn choose(obs: &Value, mem: &mut Mem) -> String {
    let legal = match obs["legal_actions"].as_array() { Some(l) if !l.is_empty() => l, _ => return "0".into() };
    let c = parse(obs, mem);
    let mut best = &legal[0];
    let mut bs = f64::NEG_INFINITY;
    for a in legal {
        let s = score(&c, a, mem);
        if s > bs { bs = s; best = a; }
    }
    best["action_id"].as_str().unwrap_or("0").to_owned()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Mem { enemy_cities: vec![], seen_villages: vec![] };
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        writeln!(stdout, "{}", json!({"action_id": choose(&obs, &mut mem)})).unwrap();
        stdout.flush().unwrap();
    }
}
